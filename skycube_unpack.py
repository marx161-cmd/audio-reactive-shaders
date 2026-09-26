#!/usr/bin/env python3
"""Step 1 of the 64k sky cubemap: unpack NASA SVS starmap_2020_64k.exr to a raw
float16 memmap, so step 2 (skycube_bake.py, torch/ROCm) can slice it without
OpenEXR and without holding the 12.9 GB decode in RAM.

Run with SYSTEM python3 (it has OpenEXR 3.5; the torch venvs do not).

Source: 65536 x 32768 RGB half, ZIP scanlines, equatorial. Orientation, measured
against the 150 brightest catalogue stars (all hit; every other convention
missed):  x = ((0.5 - RA/360) mod 1) * W,  row = (90 - Dec)/180 * H.
See scope.md 2026-09-26.
"""
import os, sys, time
import numpy as np, OpenEXR, Imath

SRC = sys.argv[1] if len(sys.argv) > 1 else "starmap_2020_64k.exr"
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "textures/starfield/work/starmap_64k_rgb16.raw")
W, H, BAND = 65536, 32768, 256

f = OpenEXR.InputFile(SRC)
assert f.isComplete(), "EXR is truncated"
HALF = Imath.PixelType(Imath.PixelType.HALF)
out = np.lib.format.open_memmap(DST.replace(".raw", ".npy"), mode="w+", dtype=np.float16, shape=(H, W, 3))
t = time.time()
for y in range(0, H, BAND):
    for i, c in enumerate("RGB"):
        out[y:y + BAND, :, i] = np.frombuffer(f.channel(c, HALF, y, y + BAND - 1), np.float16).reshape(BAND, W)
    if y % 4096 == 0:
        print(f"  rows {y}/{H}  {time.time() - t:.0f}s", flush=True)
out.flush()
print(f"done -> {DST.replace('.raw', '.npy')}  {time.time() - t:.0f}s")
