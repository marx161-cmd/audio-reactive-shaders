"""Lens-aware cutting (scope.md 2026-10-21; user: "we can't cut everybody we don't see, we need to take into
consideration where it is on the lens" + "if we move them toward spots on the lens that are brighter, they are
instantly gone and we can pop them").

For a pass, sample its path; at each moment ask the LENS where the body appears on screen:
  * within 60 Rs of the hole: the inverse lens grid textures/kerr_v7/kn/lens_inv*.npy (lens_inv_bake.py: position ->
    mean screen pixel of its image, per side = the direct image and the one wrapped round the hole), with the
    position turned back into the bake's frame by the camera azimuth;
  * further out: straight projection (the paths are ~straight there).
A moment is NEEDED if the body has an image on screen that is not SWALLOWED; an image is swallowed when the body is
small (apparent radius < POP_PX) and sits on a bright spot of the camera pocket's brightness map
(director/cache/pocket_lum_*.npy, bake_pocket_maps.py) while the camera is parked in that pocket. The pass then
only exists from its first to its last needed moment: it pops in / out where nobody can see it.
"""
import glob, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BH_RS = 0.048
R0, R1, NR, NT, NP = 0.8, 60.0, 64, 48, 96
LR0, LR1 = math.log(R0), math.log(R1)
CANVAS = 2410
FOCAL = 1.8 * CANVAS / 2.0
POP_PX = 4.0          # apparent radius (px) below which a body on a bright spot is swallowed
LUM_POP = 0.35        # pocket-map luminance counted as "bright"
PARKED = math.radians(3.0)
KEEP_UNSEEN = True    # 2026-10-21 (user, after the drill-beat render: "cutting the bodies behind us wasn't the best idea"):
                      # only the bright-spot SWALLOW pops a body; being off screen no longer cuts it

_cnt = np.load(os.path.join(ROOT, "textures/kerr_v7/kn/lens_inv_count.npy"))          # (2, NR, NT, NP)
_inv = np.load(os.path.join(ROOT, "textures/kerr_v7/kn/lens_inv.npy"))                # (2, NR, NT, NP+2, 4)
_maps = {}
for f in glob.glob(os.path.join(HERE, "cache", "pocket_lum_*.npy")):
    _maps[int(os.path.basename(f)[11:14])] = np.load(f)


def _rot(X, axis, ang):        # rotate rows of X about a unit axis
    c, s = math.cos(ang), math.sin(ang)
    return X * c + np.cross(axis, X) * s + np.outer(X @ axis, axis) * (1 - c)


def needed(P, az, cam_dir_fn, parked_az, size, ND, E1, E2):
    """P (N,3) world positions, az (N,) camera azimuth offsets, parked_az (N,) pocket azimuth or nan.
    -> bool (N,): the body has an image that is not swallowed."""
    N = len(P)
    out = np.zeros(N, bool)
    cams = np.array([cam_dir_fn(a) for a in az])                                   # hole -> camera unit
    CAM_R = 42.657458 * BH_RS
    dist = np.linalg.norm(P - CAM_R * cams, axis=1)
    rpx = size / np.maximum(dist, 1e-6) * FOCAL
    Xs = np.array([_rot(P[i:i + 1] / BH_RS, ND, -az[i])[0] for i in range(N)])     # bake frame, Rs
    r = np.linalg.norm(Xs, axis=1)
    pix = [[] for _ in range(N)]
    near = (r >= R0) & (r < R1)
    if near.any():
        rr = r[near]
        ir = np.clip(((np.log(rr) - LR0) / (LR1 - LR0) * NR).astype(int), 0, NR - 1)
        it = np.clip(((Xs[near] @ ND / rr + 1.0) * 0.5 * NT).astype(int), 0, NT - 1)
        ip = (((np.arctan2(Xs[near] @ E2, Xs[near] @ E1) + math.pi) / (2 * math.pi) * NP).astype(int)) % NP
        idx = np.nonzero(near)[0]
        for side in (0, 1):
            c = _cnt[side, ir, it, ip]
            t = _inv[side, ir, it, ip + 1]
            for k in np.nonzero(c > 0)[0]:
                w = max(t[k, 2], 1e-9)
                pix[idx[k]].append((t[k, 0] / w, t[k, 1] / w))
    # straight projection: beyond the grid, and as a fallback where the grid is sparse (> 8 Rs, no recorded image --
    # the bake sampled every 2nd pixel, so an empty cell out there may be missing data, not "no image")
    far = (r >= R1) | ((r >= 8.0) & np.array([len(q) == 0 for q in pix]))
    for i in np.nonzero(far)[0]:                                                    # straight projection
        F = -cams[i]; Rv = np.cross(F, ND); Rv /= np.linalg.norm(Rv); U = np.cross(Rv, F)
        d = P[i] - CAM_R * cams[i]; z = d @ F
        if z <= 0:
            continue
        x, y = d @ Rv / z * 1.8, d @ U / z * 1.8
        if abs(x) <= 1.0 and abs(y) <= 1.0:
            pix[i].append(((x + 1.0) * 0.5 * CANVAS, (y + 1.0) * 0.5 * CANVAS))
    for i in range(N):
        if not pix[i] and KEEP_UNSEEN:
            out[i] = True; continue                    # no image on screen (behind / beside the camera): KEPT, it still lights
        if not pix[i]:
            continue
        m = None
        if not math.isnan(parked_az[i]) and rpx[i] < POP_PX:
            deg = round(math.degrees(parked_az[i])) % 360
            if _maps:
                m = _maps.get(deg)
                if m is None:
                    m = _maps[min(_maps, key=lambda q: min(abs(q - deg), 360 - abs(q - deg)))]
        if m is None:
            out[i] = True; continue
        H = m.shape[0]
        for (px, py) in pix[i]:
            yy, xx = int(np.clip(py / CANVAS * H, 0, H - 1)), int(np.clip(px / CANVAS * H, 0, H - 1))
            if m[yy, xx] < LUM_POP:
                out[i] = True; break
    return out
