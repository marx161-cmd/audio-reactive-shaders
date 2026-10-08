#!/usr/bin/env python3
"""Gaia DR3 point stars -> 6 x 16384^2 sky cube faces -> mips -> BC6H.

2026-10-03 (scope.md "v12 sky: real catalogue POINT stars, no smooth Milky Way
band"): the open-field half of the hybrid point-star sky. Every star of the
catalogue (tools/gaia_fetch.py: 103,936,997 stars, G < 16.4) is deposited as a
point -- bilinear over the 2x2 texels around its exact position, so the mips
keep its sub-texel place and its total flux. No smooth band: the Milky Way is
whatever the star density makes of it. The lens-magnified band (where the cube
is coarser than the screen) gets live catalogue points later; this cube is not
meant to resolve it.

Same container as skycube_bake.py (encode_level, json layout), so the
renderer's .json cube loader takes it unchanged. Same direction frame as the
NASA cube (shader skyDirUV frame): d = (-cos(dec)cos(ra), sin(dec), -cos(dec)sin(ra)),
verified 2026-10-03: 37/40 G 5-7 Gaia stars land on the NASA map's peaks
(median 0.49 texel).

BRIGHTNESS (my choice, flagged in scope.md -- tune live via SKY64_GAIN):
real flux spans 10^(0.4*18) across G -1.5..16.4, which would leave only a few
thousand visible stars. Compressed instead: flux = F_FAINT * 10^(0.4*K*(G_MAX-G)),
K = 0.37, set so the faintest star (G 16.4) comes out as a dim but visible dot
after the shader's chain (x SKY64_GAIN 2 x SKY_GAIN 3.2, tonemap, GRADE_CONTRAST
1.4, GRADE_BLACK -0.02) when sampled about one mip down (open field ~4
texels/screen px, SKY64_SHARP 0.5), and G ~1 saturates.

    ~/onetrainer-gpu/.venv/bin/python gaia_cube_bake.py
"""
import glob, json, math, os, sys, time
import numpy as np, torch
import skycube_bake as SB

ROOT = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(ROOT, "textures/starfield/gaia/chunks")
# Overridable for A/B variants (2026-10-03: GAIA_G_MAX=14 GAIA_F_FAINT=0.13
# GAIA_OUT=textures/starfield/gaia_cube_g14 = user's "reduce density, boost brightness").
OUT = os.path.join(ROOT, os.environ.get("GAIA_OUT", "textures/starfield/gaia_cube"))
N = SB.N
G_MAX = float(os.environ.get("GAIA_G_MAX", 16.4))
K = 0.37
F_FAINT = float(os.environ.get("GAIA_F_FAINT", 0.065))   # texel-units flux of a G_MAX star (see BRIGHTNESS above)
# 2026-10-03 (user: stars "blinking"): with the mip-0 star read (SKY_LOD0) a star
# deposited into 2x2 texels (~1/4 screen px) is only seen when it sits under a
# pixel's sample point, so drifting stars blink. PSF_SIGMA > 0 paints each star
# as a Gaussian of that sigma (texels; ~1.7 = FWHM ~4 texels ~ 1 screen px),
# scaled so its PEAK texel matches what a 2x2 deposit gave on average (0.5 F),
# so the per-pixel brightness stays about the same.
PSF_SIGMA = float(os.environ.get("GAIA_PSF_SIGMA", 0.0))
# 2026-10-03 (user: "I don't want softened stars ... slightly bigger sharp ones"):
# PSF_DISC > 0 paints each star as a HARD-EDGED disc of that radius (texels; 2 =
# ~4 texels across ~ 1 screen px), every covered texel at 0.5 F (the 2x2
# deposit's average peak), so stars get bigger, stay sharp, and a mip-0 read
# lands on them wherever they sit.
PSF_DISC = float(os.environ.get("GAIA_PSF_DISC", 0.0))
# 2026-10-03 (user: "keep size and just tell the shader to make a pixel light up if
# stars are inside it"): MIP_MAX=1 builds every mip level by keeping the BRIGHTEST
# texel of each 2x2 (per channel) instead of averaging. A footprint-sized read then
# returns the brightest star inside the pixel at full brightness: no blinking as
# stars drift, no averaging-away in the lens-squeezed ring, star size unchanged.
MIP_MAX = os.environ.get("GAIA_MIP_MAX", "0") == "1"
dev = SB.dev


def kelvin_rgb(T):
    """Tanner Helland blackbody fit, normalised so the max channel is 1 (flux carries brightness)."""
    t = T / 100.0
    r = torch.where(t <= 66, torch.ones_like(t), 1.292936 * torch.pow(torch.clamp(t - 60, min=1e-3), -0.1332047592))
    g = torch.where(t <= 66, 0.39008157 * torch.log(t) - 0.63184144,
                    1.129890861 * torch.pow(torch.clamp(t - 60, min=1e-3), -0.0755148492))
    b = torch.where(t >= 66, torch.ones_like(t),
                    torch.where(t <= 19, torch.zeros_like(t),
                                0.543206789 * torch.log(torch.clamp(t - 10, min=1e-3)) - 1.19625408))
    c = torch.clamp(torch.stack([r, g, b], -1), 0, 1)
    return c / c.max(-1, keepdim=True).values.clamp(min=1e-6)


def load_stars():
    files = sorted(glob.glob(os.path.join(CHUNKS, "c*.npy")))
    a = np.concatenate([np.load(f) for f in files])
    a = a[a["g"] < G_MAX]
    ra = np.radians(a["ra"]); de = np.radians(a["dec"])      # on CPU: the live renderer shares the GPU
    d = np.stack([-np.cos(de) * np.cos(ra), np.sin(de), -np.cos(de) * np.sin(ra)], -1).astype(np.float32)
    del ra, de
    d = torch.from_numpy(d).to(dev)
    g = torch.from_numpy(a["g"].astype(np.float32)).to(dev)
    bprp = torch.from_numpy(a["bp_rp"].astype(np.float32)).to(dev)
    bprp = torch.where(torch.isnan(bprp), torch.full_like(bprp, 0.82), bprp)   # missing colour -> solar
    bprp = bprp.clamp(-0.5, 4.0)
    # BP-RP -> Teff (dwarf colour-temperature fit, good enough for a tint)
    teff = 5040.0 / (0.4929 + 0.5092 * bprp - 0.0353 * bprp * bprp).clamp(min=0.12)
    flux = F_FAINT * torch.pow(10.0, 0.4 * K * (G_MAX - g))
    rgb = kelvin_rgb(teff.clamp(1500, 40000)) * flux[:, None]
    return d, rgb, len(a)


def face_st(d):
    """Inverse of SB.face_dirs: face id and (s, t) in [-1, 1] for each direction."""
    x, y, z = d.unbind(-1)
    ax, ay, az = x.abs(), y.abs(), z.abs()
    face = torch.where((ax >= ay) & (ax >= az), torch.where(x > 0, 0, 1),
           torch.where(ay >= az, torch.where(y > 0, 2, 3), torch.where(z > 0, 4, 5))).to(torch.uint8)
    s = torch.empty_like(x); t = torch.empty_like(x)
    for f, (ss, tt, m) in enumerate([(-z, -y, ax), (z, -y, ax), (x, z, ay),
                                     (x, -z, ay), (x, -y, az), (-x, -y, az)]):
        k = face == f
        s[k] = ss[k] / m[k]; t[k] = tt[k] / m[k]
    return face, s, t


def paint_face(cols, rows, rgb):
    """Bilinear point deposit into an (N, N, 3) float32 face (or a Gaussian PSF, see PSF_SIGMA)."""
    img = torch.zeros((N * N, 3), dtype=torch.float32, device=dev)
    if PSF_DISC > 0:
        R = int(math.ceil(PSF_DISC))
        xc = torch.floor(cols).long(); yc = torch.floor(rows).long()
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                xi = xc + dx; yi = yc + dy
                ddx = xi.float() + 0.5 - cols; ddy = yi.float() + 0.5 - rows
                ok = (ddx * ddx + ddy * ddy <= PSF_DISC ** 2) & (xi >= 0) & (xi < N) & (yi >= 0) & (yi < N)
                img.index_add_(0, (yi * N + xi)[ok], rgb[ok] * 0.5)
        return img.view(N, N, 3)
    if PSF_SIGMA > 0:
        R = int(math.ceil(3 * PSF_SIGMA))
        peak_scale = 0.5 * 2 * math.pi * PSF_SIGMA ** 2      # peak texel = 0.5 F
        xc = torch.floor(cols).long(); yc = torch.floor(rows).long()
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                xi = xc + dx; yi = yc + dy
                ddx = xi.float() + 0.5 - cols; ddy = yi.float() + 0.5 - rows
                w = torch.exp(-(ddx * ddx + ddy * ddy) / (2 * PSF_SIGMA ** 2)) / (2 * math.pi * PSF_SIGMA ** 2)
                ok = (xi >= 0) & (xi < N) & (yi >= 0) & (yi < N)
                img.index_add_(0, (yi * N + xi)[ok], (rgb[ok] * (w[ok] * peak_scale)[:, None]))
        return img.view(N, N, 3)
    X = cols - 0.5; Y = rows - 0.5
    x0 = torch.floor(X); y0 = torch.floor(Y)
    fx = (X - x0)[:, None]; fy = (Y - y0)[:, None]
    x0 = x0.long(); y0 = y0.long()
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = (x0 + dx).clamp(0, N - 1); yi = (y0 + dy).clamp(0, N - 1)
        img.index_add_(0, yi * N + xi, rgb * w)
    return img.view(N, N, 3)


def main():
    os.makedirs(OUT, exist_ok=True)
    T = time.time()
    d, rgb, n = load_stars()
    face, s, t = face_st(d)
    del d
    print(f"{n:,} stars loaded, {time.time() - T:.0f}s", flush=True)
    levels = int(math.log2(N)) + 1
    parts = {}
    for f in range(6):
        names = [os.path.join(OUT, f".part_L{L:02d}_F{f}.bin") for L in range(levels)]
        if all(os.path.exists(p) for p in names):      # resume: face finished by an earlier run
            for L, p in enumerate(names):
                parts[(L, f)] = (p, os.path.getsize(p), N >> L)
            print(f"  face {f}: parts on disk, skipped", flush=True)
            continue
        k = face == f
        cols = (s[k] + 1) * 0.5 * N; rows = (t[k] + 1) * 0.5 * N
        cur = paint_face(cols, rows, rgb[k])
        nk = int(k.sum())
        del k, cols, rows
        # 2026-10-03: a GPU fault (bake sharing the GPU with the live renderer) once
        # wrote two EMPTY faces without raising -> black squares in the sky. Refuse
        # to save a face whose 64x64-texel tiles are mostly empty.
        lit = float((cur.view(N // 64, 64, N // 64, 64, 3).amax((1, 3, 4)) > 0).float().mean())
        if lit < 0.5 or nk == 0:
            raise RuntimeError(f"face {f}: only {lit:.0%} of tiles lit ({nk} stars) -- GPU fault? not saving")
        print(f"  face {f}: {nk:,} stars painted, peak {float(cur.max()):.1f}, {time.time() - T:.0f}s", flush=True)
        for L in range(levels):
            nL = N >> L
            if L > 0:
                nxt = torch.empty((nL, nL, 3), dtype=torch.float32, device=dev)
                for r in range(0, nL, 1024):
                    rr = min(1024, nL - r)
                    blk4 = cur[2 * r:2 * (r + rr)].view(rr, 2, nL, 2, 3)
                    nxt[r:r + rr] = blk4.amax((1, 3)) if MIP_MAX else blk4.mean((1, 3))
                cur = nxt
            # to CPU f16 in row bands: a whole-face clamp/half copy OOMed next to the live renderer
            h = np.empty((nL, nL, 3), np.float16)
            for r in range(0, nL, 1024):
                h[r:r + 1024] = cur[r:r + 1024].clamp(max=60000).half().cpu().numpy()
            blk = SB.encode_level(h)
            del h
            p = os.path.join(OUT, f".part_L{L:02d}_F{f}.bin")
            with open(p, "wb") as fh:
                fh.write(blk)
            parts[(L, f)] = (p, len(blk), nL)
        del cur
        torch.cuda.empty_cache()
        print(f"  face {f} done, {time.time() - T:.0f}s", flush=True)

    meta = {"face_size": N, "levels": levels, "format": "BC6H_UF16 (GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT)",
            "faces": "+X -X +Y -Y +Z -Z (GL order)",
            "source": f"Gaia DR3 G<{G_MAX}, {n} stars, point deposit, K={K}, F_FAINT={F_FAINT}",
            "entries": []}
    # renderer.load_bc6h_cubemap hardcodes this file name next to the .json
    binpath = os.path.join(OUT, "sky64k_bc6h.bin")
    off = 0
    with open(binpath, "wb") as out:
        for L in range(levels):
            for f in range(6):
                p, size, nL = parts[(L, f)]
                with open(p, "rb") as fh:
                    out.write(fh.read())
                meta["entries"].append({"level": L, "face": f, "offset": off, "size": size, "w": nL})
                off += size
                os.remove(p)
    with open(os.path.join(OUT, "gaia_cube.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"done: {binpath} {off / 1e9:.2f} GB, {time.time() - T:.0f}s")


if __name__ == "__main__":
    main()
