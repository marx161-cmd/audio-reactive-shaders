"""
nebula_prep.py -- the Nebulabrot bake (nebula_bake_modal.py) -> the texture the
nebulabrot shader samples.

In:  textures/nebulabrot/nebula16384.npy  (3,RES,RES) f32, one layer per escape band
Out: textures/nebulabrot/nebula16384_rgba.npy (RES,RES,4) f16, each layer divided
     by its own 99.9th percentile (so gains in the shader start near 1), alpha 0.
     textures/nebulabrot/nebula8192_rgba.npy   the same, 2x2 averaged.
     RGBA, not RGB: a 3-channel 16k texture makes radeonsi allocate one 2 GB
     temporary conversion copy on upload, which fails. Row 0 = bottom (Im = -H),
     same as GL; no flip needed.

    python3 nebula_prep.py [--src ...]
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "textures/nebulabrot/nebula16384.npy"))
    args = ap.parse_args()
    a = np.load(args.src, mmap_mode="r")
    L, h, w = a.shape
    out = np.zeros((h, w, 4), dtype=np.float16)
    for i in range(L):
        layer = np.asarray(a[i], dtype=np.float32)
        lit = layer[layer > 0]
        scale = np.percentile(lit[:: max(1, lit.size // 20_000_000)], 99.9)
        out[..., i] = np.minimum(layer / scale, 60000.0)
        print(f"layer {i}: p99.9 {scale:.4g}  max/p99.9 {layer.max() / scale:.1f}  lit {lit.size / layer.size:.3f}")
        del layer, lit
    dst = args.src.replace(".npy", "_rgba.npy")
    np.save(dst, out)
    print(f"-> {dst}")
    half = np.zeros((h // 2, w // 2, 4), dtype=np.float16)
    for y in range(0, h, 1024):
        blk = out[y:y + 1024, :, :3].astype(np.float32)
        half[y // 2:(y + 1024) // 2, :, :3] = blk.reshape(-1, 2, w // 2, 2, 3).mean((1, 3))
    dst2 = dst.replace(f"{h}", f"{h // 2}")
    np.save(dst2, half)
    print(f"-> {dst2}")


if __name__ == "__main__":
    main()
