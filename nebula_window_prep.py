"""
nebula_window_prep.py -- a deep-zoom window bake (nebula_bake_modal.py --mode
wbake) -> a texture in the SAME brightness units as the global one, so the
shader can cross-fade between them without a jump.

Both bakes estimate the same quantity per pixel (orbit points per unit c-area
landing in that pixel), so a window pixel, (2h/RES_w)^2 in area, holds
(pixel-area ratio) times less than a global pixel of equal density. Scale by
that ratio, then divide by the GLOBAL texture's per-layer p99.9 (what
nebula_prep.py normalised the global texture by).

In:  textures/nebulabrot/<name>.npy + .json (window bake), nebula16384.npy + .json
Out: textures/nebulabrot/<name>_rgba.npy (RES,RES,4) f16, alpha 0

    python3 nebula_window_prep.py nebula_w0
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "textures/nebulabrot")


def global_scales():
    # identical to nebula_prep.py: per-layer p99.9 of the lit pixels (strided)
    a = np.load(os.path.join(D, "nebula16384.npy"), mmap_mode="r")
    out = []
    for i in range(a.shape[0]):
        layer = np.asarray(a[i], dtype=np.float32)
        lit = layer[layer > 0]
        out.append(float(np.percentile(lit[:: max(1, lit.size // 20_000_000)], 99.9)))
    return out


def main(name):
    gj = json.load(open(os.path.join(D, "nebula16384.json")))
    wj = json.load(open(os.path.join(D, f"{name}.json")))
    g_pix = 2 * gj["window"]["h"] / gj["res"]
    w_pix = 2 * wj["window"]["h"] / wj["res"]
    area = (g_pix / w_pix) ** 2
    scales = global_scales()
    print(f"window {wj['window']}  res {wj['res']}  pixel-area ratio {area:.1f}  global p99.9 {scales}")
    w = np.load(os.path.join(D, f"{name}.npy"), mmap_mode="r")
    L, h, wd = w.shape
    out = np.zeros((h, wd, 4), dtype=np.float16)
    for i in range(L):
        layer = np.asarray(w[i], dtype=np.float32) * area / scales[i]
        out[..., i] = np.minimum(layer, 60000.0)
        print(f"layer {i}: p50 {np.percentile(layer[::8, ::8], 50):.3f}  p99.9 {np.percentile(layer[::8, ::8], 99.9):.3f}")
    dst = os.path.join(D, f"{name}_rgba.npy")
    np.save(dst, out)
    print(f"-> {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nebula_w0")
