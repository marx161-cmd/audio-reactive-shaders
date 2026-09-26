#!/usr/bin/env python3
"""Bake the source sky/planet imagery into .npy data textures for
pulsar_orrery.mp.

Why .npy and not the iChannel PNG convention: the image pass already spends
iChannel0 on Buffer A, and this shader needs five simultaneous maps. That is
what manifest.json's "static_textures" field is for (uniform name -> path),
loaded by renderer.py's load_static_data_texture at fixed units from 9 up.

Two things that loader does NOT do, so they are done here instead:

  * No row flip. GL's texture origin is bottom-left, but an equirectangular
    map has its north pole in the FIRST row, so every map is flipped here.
    Skip this and every planet renders upside down.

  * No wrap. The loader sets repeat_x = False, so a longitude that wraps past
    1.0 clamps and leaves a visible seam line down the planet. Each map is
    therefore padded with ONE extra column that duplicates column 0, and the
    shader addresses it as (0.5 + u*W) / (W+1) -- see wrapU() in common.glsl.
    The pad is why the widths below are odd numbers in the shader's defines.

Resolutions are deliberately modest: at desktop framing the red dwarf is
~150px across and a moon is ~10px, and there are no mipmaps here, so an 8K
source would only alias. Run after changing a source image:

    python3 pulsar_orrery_prep.py      (needs numpy + Pillow)
"""

import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "textures")
OUT = os.path.join(REPO, "textures", "pulsar_orrery")


def equirect(src_rel, w, h, mode, out_name):
    path = src_rel if os.path.isabs(src_rel) else os.path.join(SRC, src_rel)
    img = Image.open(path).convert(mode)
    img = img.resize((w, h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = np.flipud(arr)                       # north pole -> last row (GL up)
    arr = np.concatenate([arr, arr[:, :1, :]], axis=1)   # wrap column
    arr = np.ascontiguousarray(arr.squeeze() if arr.shape[2] == 1 else arr)
    out = os.path.join(OUT, out_name)
    np.save(out, arr)
    print(f"{out_name:16} {src_rel:40} -> {arr.shape} {arr.dtype}")


def ring_profile(src_rel, w, out_name):
    """The ring strip is a radial cross-section: inner edge at x=0, outer at
    x=1. Collapse it to a single row so the shader's lookup is 1-D, which is
    what a ring profile actually is."""
    path = os.path.join(SRC, src_rel)
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    prof = arr.mean(axis=0)                                  # average the rows
    idx = np.linspace(0, len(prof) - 1, w)
    prof = np.interp(idx, np.arange(len(prof)), prof)
    prof = np.clip(prof, 0, 255).astype(np.uint8)[None, :]   # shape (1, w)
    out = os.path.join(OUT, out_name)
    np.save(out, np.ascontiguousarray(prof))
    print(f"{out_name:16} {src_rel:40} -> {prof.shape} "
          f"min={prof.min()} max={prof.max()} mean={prof.mean():.1f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    # Red dwarf: real solar photosphere. Highest resolution here because it is
    # the largest body on screen and the only one whose granulation is meant
    # to be legible.
    equirect("8k_sun.jpg", 2048, 1024, "RGB", "dwarf.npy")
    # Chromosphere (SDO/STEREO 304A): spicules and active regions, added as an
    # emission layer over the photosphere rather than replacing it.
    equirect("euvi_aia304_2012_carrington.tif", 1024, 512, "L", "chromo.npy")
    # Gas giants: one Saturn map, hue-rotated and band-stretched per giant in
    # the shader so the three do not read as triplets.
    equirect("8k_saturn.jpg", 1024, 512, "RGB", "giant.npy")
    # Moons + the rocky planet: LROC lunar albedo, per-body UV offset.
    equirect("moon_sphere/iChannel0.png", 1024, 512, "RGB", "rock.npy")
    # Ring density: real Cassini structure, replacing the procedural chirp.
    ring_profile("8k_saturn_ring_alpha.png", 2048, "ringprof.npy")
    # Backdrop: the Hipparcos/Tycho all-sky catalog map — the same equirect
    # skybox the kerr_newman_orbit music-video shader used. Baked from the 16k
    # source rather than the 4820px one because the sky fills the entire frame
    # at roughly 2x magnification, so the extra source detail is real detail.
    # 8192x4096 RGB is ~100MB on the GPU, which is nothing on a 16GB card.
    # (skymap.npy is unused since the 64k sky cubemap; only rebuilt if the source exists.)
    if os.path.isfile(os.path.join(SRC, "starmap16k.png")):
        equirect("starmap16k.png", 8192, 4096, "RGB", "skymap.npy")


if __name__ == "__main__":
    sys.exit(main())
