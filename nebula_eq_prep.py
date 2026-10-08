"""
nebula_eq_prep.py -- a converged glitter bake (nebula_bake_modal.py --mode
mbake) -> an EQUALISED texture: each band's log density replaced by its rank
among the lit pixels (0..1). Faint filaments and dense cores then spread evenly
over the brightness range, with no fixed white point. Verified 2026-09-27 on the
4096 Metropolis test: equalisation + live local contrast shows the fine
filaments/arcs; a fixed log curve with a p99.5 white point blew it out to white.

In:  textures/nebulabrot/<name>.npy  (3,RES,RES) f32
Out: textures/nebulabrot/<name>_eq_rgba.npy (RES,RES,4) f16 -- rank per band,
     0 where unlit; alpha 0. Rows = Im from -H (row 0), as the other textures.

    python3 nebula_eq_prep.py nebula_glitter_mcmc16384
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "textures/nebulabrot")
NQ = 4096


def main(name):
    a = np.load(os.path.join(D, f"{name}.npy"), mmap_mode="r")
    L, h, w = a.shape
    out = np.zeros((h, w, 4), dtype=np.float16)
    for i in range(L):
        # quantiles of log density over lit pixels (subsampled for the table)
        sub = np.asarray(a[i, ::8, ::8], dtype=np.float32)
        lv = np.log(sub[sub > 0])
        qs = np.quantile(lv, np.linspace(0.0, 1.0, NQ))
        ranks = np.linspace(0.0, 1.0, NQ)
        for y in range(0, h, 1024):
            blk = np.asarray(a[i, y:y + 1024], dtype=np.float32)
            lit = blk > 0
            r = np.zeros_like(blk)
            r[lit] = np.interp(np.log(blk[lit]), qs, ranks)
            out[y:y + 1024, :, i] = r
        print(f"band {i}: lit {float((sub > 0).mean()):.3f}  log-density range {qs[0]:.2f} .. {qs[-1]:.2f}")
    dst = os.path.join(D, f"{name}_eq_rgba.npy")
    np.save(dst, out)
    print(f"-> {dst}")


if __name__ == "__main__":
    main(sys.argv[1])
