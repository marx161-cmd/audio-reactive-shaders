"""lens_inv_bake.py -- invert the Kerr-Newman path bake into a position-indexed lens grid.
scope.md 2026-10-12 "ember_hole.mp LENS DECISION".

The bake (textures/kerr_v7/kn/paths32.npy, (32, H, W, 4): xyz in Rs along each pixel's ray, camera first) answers
"pixel -> path". A point source needs "position -> pixel". Every ray is upsampled (Catmull-Rom through the 32
points, F sub-steps per segment) and each sample is binned into a spherical cell grid about the hole
(ln r, cos theta about the spin axis, phi in the disk frame). Per cell and per SIDE (the source on the same /
the opposite side of the screen centre as the pixel) we keep the mean pixel of the rays whose arc length through
that cell is the shortest (drops the higher-order winding images).

Output: textures/kerr_v7/kn/lens_inv.npy  (2, NR, NT, NP+2, 4) f32 = [side][r][cos th][phi (padded 1 each end)] ->
(mean_x * W, mean_y * W, W, 0) so trilinear filtering stays a proper weighted mean (divide by W in the kernel).
"""
import os
import sys
import time

import numpy as np
import torch

PATHS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures/kerr_v7/kn/paths32.npy")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures/kerr_v7/kn/lens_inv.npy")
NR, NT, NP = 64, 48, 96
R0, R1 = 0.8, 60.0          # Rs
F = 32                      # sub-steps per path segment
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 2     # every STRIDE-th pixel in x and y
TILE_ROWS = 8               # image rows (of the strided grid) per tile
DS_EXTRA = 4.0              # Rs: keep only rays within this arc length of the shortest in a cell
dev = torch.device("cuda")

nd = np.array([0.34, 1.0, 0.13]); nd /= np.linalg.norm(nd)         # spin axis = shader DISK_TILT
e1 = np.cross(nd, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
e2 = np.cross(nd, e1)
az, el = -3.103562, 0.087266
camDir = np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * nd
Fv = -camDir
R3 = np.cross(Fv, nd); R3 /= np.linalg.norm(R3)
Uv = np.cross(R3, Fv)
T = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)
tnd, te1, te2, tR3, tU = T(nd), T(e1), T(e2), T(R3), T(Uv)

P = np.load(PATHS, mmap_mode="r")
H, W = P.shape[1], P.shape[2]
NCELL = NR * NT * NP
LR0, LR1 = float(np.log(R0)), float(np.log(R1))
ts = torch.arange(F, device=dev, dtype=torch.float32) / F        # 0 .. 1-1/F


def tile_samples(y0, y1):
    """All samples of the rays at rows y0:y1 (strided grid): returns flat cell index (side-major), arc length, pixel xy."""
    ys = torch.arange(y0, y1, STRIDE, device=dev)
    xs = torch.arange(0, W, STRIDE, device=dev)
    # (32, ny, nx, 3)
    blk = torch.from_numpy(np.ascontiguousarray(P[:, y0:y1:STRIDE, ::STRIDE, :3])).to(dev)
    ny, nx = blk.shape[1], blk.shape[2]
    n = ny * nx
    p = blk.reshape(32, n, 3)
    pm1 = torch.cat([p[:1], p[:-1]], 0)                               # p[k-1] (clamped)
    pp2 = torch.cat([p[1:], p[-1:]], 0)                               # p[k+2] after shifting
    pp2 = torch.cat([pp2[1:], pp2[-1:]], 0)
    p0, p1 = p[:-1], p[1:]                                            # (31, n, 3)
    a, b = pm1[:-1], pp2[:-1]                                         # neighbours of each segment
    t = ts.view(1, F, 1, 1)
    t2, t3 = t * t, t * t * t
    A, B, C, D = p0.unsqueeze(1), p1.unsqueeze(1), a.unsqueeze(1), b.unsqueeze(1)
    pts = 0.5 * ((2 * A) + (-C + B) * t + (2 * C - 5 * A + 4 * B - D) * t2 + (-C + 3 * A - 3 * B + D) * t3)   # (31, F, n, 3)
    pts = pts.reshape(31 * F, n, 3)
    seg = (pts[1:] - pts[:-1]).norm(dim=-1)
    s = torch.cat([torch.zeros(1, n, device=dev), seg.cumsum(0)], 0)  # arc length (31F, n)
    r = pts.norm(dim=-1).clamp_min(1e-6)
    ir = ((torch.log(r) - LR0) / (LR1 - LR0) * NR).floor().long()
    ct = (pts @ tnd) / r
    it = ((ct + 1.0) * 0.5 * NT).floor().long().clamp(0, NT - 1)
    ph = torch.atan2(pts @ te2, pts @ te1)
    ip = ((ph + np.pi) / (2 * np.pi) * NP).floor().long() % NP
    ok = (ir >= 0) & (ir < NR)
    # screen side: pixel offset from the image centre vs the sample's screen-plane direction
    px = xs.float().view(1, nx).expand(ny, nx).reshape(n)
    py = ys.float().view(ny, 1).expand(ny, nx).reshape(n)
    ox, oy = (px + 0.5 - W * 0.5), (py + 0.5 - H * 0.5)
    sx, sy = pts @ tR3, pts @ tU
    side = ((sx * ox.view(1, n) + sy * oy.view(1, n)) < 0).long()
    cell = ((ir.clamp(0, NR - 1) * NT + it) * NP + ip) + side * NCELL
    return cell[ok], s[ok], px.view(1, n).expand(31 * F, n)[ok], py.view(1, n).expand(31 * F, n)[ok]


def rows_iter():
    step = TILE_ROWS * STRIDE
    for y0 in range(0, H, step):
        yield y0, min(H, y0 + step)


t0 = time.time()
min_s = torch.full((2 * NCELL,), 1e9, device=dev)
for k, (y0, y1) in enumerate(rows_iter()):
    cell, s, _, _ = tile_samples(y0, y1)
    min_s.scatter_reduce_(0, cell, s, reduce="amin")
    if k % 20 == 0:
        print(f"pass1 {y0}/{H}  {time.time() - t0:.0f}s", flush=True)
sx = torch.zeros(2 * NCELL, device=dev, dtype=torch.float64)
sy = torch.zeros(2 * NCELL, device=dev, dtype=torch.float64)
cnt = torch.zeros(2 * NCELL, device=dev, dtype=torch.float64)
for k, (y0, y1) in enumerate(rows_iter()):
    cell, s, px, py = tile_samples(y0, y1)
    keep = s <= min_s[cell] + DS_EXTRA
    cell, px, py = cell[keep], px[keep], py[keep]
    sx.index_add_(0, cell, px.double() + 0.5)
    sy.index_add_(0, cell, py.double() + 0.5)
    cnt.index_add_(0, cell, torch.ones_like(px, dtype=torch.float64))
    if k % 20 == 0:
        print(f"pass2 {y0}/{H}  {time.time() - t0:.0f}s", flush=True)

cnt_c = cnt.cpu().numpy().reshape(2, NR, NT, NP)
mx = (sx.cpu().numpy().reshape(2, NR, NT, NP))
my = (sy.cpu().numpy().reshape(2, NR, NT, NP))
cov = (cnt_c > 0)
print("coverage per side:", cov.mean(axis=(1, 2, 3)))
for ir in range(0, NR, 8):
    rr = np.exp(LR0 + (ir + 0.5) / NR * (LR1 - LR0))
    print(f"  r={rr:5.2f} Rs  cov side0 {cov[0, ir].mean():.2f} side1 {cov[1, ir].mean():.2f}")
# weighted-mean texel: (mean*W, W). W = min(count, 1000)/1000 so a thin cell does not dominate its neighbours.
W_ = np.minimum(cnt_c, 1000.0) / 1000.0
meanx = np.where(cov, mx / np.maximum(cnt_c, 1), 0.0)
meany = np.where(cov, my / np.maximum(cnt_c, 1), 0.0)
out = np.zeros((2, NR, NT, NP + 2, 4), np.float32)
body = np.stack([meanx * W_, meany * W_, W_, np.zeros_like(W_)], -1).astype(np.float32)
out[:, :, :, 1:NP + 1] = body
out[:, :, :, 0] = body[:, :, :, NP - 1]
out[:, :, :, NP + 1] = body[:, :, :, 0]
np.save(OUT, out)
np.save(OUT.replace(".npy", "_count.npy"), cnt_c.astype(np.float32))
print("saved", OUT, out.shape, f"{time.time() - t0:.0f}s")
