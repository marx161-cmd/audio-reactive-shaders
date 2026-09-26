#!/usr/bin/env python3
"""Step 2 of the 64k sky cubemap: equirect float16 memmap (skycube_unpack.py)
-> 6 x 16384^2 cube faces -> full mip chain -> BC6H (Intel ISPC, profile basic).

Run with a ROCm torch venv (the resample and mips run on the dGPU):
    <torch venv>/bin/python skycube_bake.py      (ROCm or CUDA torch)

Output (textures/starfield/sky64k/):
    sky64k_bc6h.bin   all BC6H blocks, level-major then face (+X -X +Y -Y +Z -Z)
    sky64k.json       face size, levels, and (offset, size, w) per level/face

WHY: the v8/v9 sky searched the star catalogue per pixel (~9 cell + ~22 star
fetches, x3 under treble for the chromatic split): ~23 ms/frame at 2410^2.
The raster is one filtered fetch; mips and anisotropic filtering do in hardware
what starField() did by hand. 0.0055 deg/texel at face centre is finer than the
catalogue renderer's own PSF floor (~0.012 deg). See scope.md 2026-09-26.

DIRECTION FRAME = the shader's skyDirUV frame: u = atan(z,x)/2pi + 0.5 = RA/360,
dir.y = sin(Dec). The shader samples texture(cube, d) with the SAME d it used
to feed the equirect (after sky rotation and skyWarp).
Source mapping (measured, scope.md): x = ((0.5 - RA/360) mod 1) W,
row = (90 - Dec)/180 H  ==>  X = (-atan2(z,x)/2pi mod 1) W,  Y = acos(y)/pi H.
"""
import json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import bc6h_ispc

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "textures/starfield/work/starmap_64k_rgb16.npy")
OUT = os.path.join(ROOT, "textures/starfield/sky64k")
N = int(os.environ.get("SKYCUBE_N", 16384))
SS = 2                      # 2x2 supersampling per cube texel (box over its footprint)
BAND = 512                  # face rows per GPU band
dev = torch.device("cuda")


def face_dirs(face, s, t):
    """GL cube-map convention (spec table 8.19): direction for face coords s,t in [-1,1],
    t increasing with the row index of the uploaded data."""
    one = torch.ones_like(s)
    return {0: ( one, -t, -s), 1: (-one, -t,  s),
            2: (   s, one,  t), 3: (   s, -one, -t),
            4: (   s,  -t, one), 5: (  -s,  -t, -one)}[face]


def src_xy(x, y, z, W, H):
    r = torch.sqrt(x * x + y * y + z * z)
    X = torch.remainder(-torch.atan2(z, x) / (2 * math.pi), 1.0) * W
    Y = torch.acos(torch.clamp(y / r, -1, 1)) / math.pi * H
    return X, Y


def face_region(face, W, H):
    """Source rows/cols this face touches (+2 px margin), from a dense edge+interior probe."""
    g = torch.linspace(-1, 1, 513, device=dev)
    s, t = torch.meshgrid(g, g, indexing="xy")
    X, Y = src_xy(*face_dirs(face, s, t), W, H)
    y0 = max(int(Y.min()) - 2, 0); y1 = min(int(Y.max()) + 3, H)
    if face in (2, 3):                       # polar caps: every RA
        return y0, y1, 0, W, True
    # equatorial faces span RA +-45 deg about their centre column
    cx = float(src_xy(*face_dirs(face, torch.zeros(1, device=dev), torch.zeros(1, device=dev)), W, H)[0])
    half = W // 8 + 2 * 64
    return y0, y1, int(cx) - half, int(cx) + half, False


def load_region(src, y0, y1, x0, x1):
    W = src.shape[1]
    cols = np.arange(x0, x1) % W
    if x1 - x0 >= W:
        a = np.asarray(src[y0:y1])
    elif cols[0] < cols[-1]:
        a = np.asarray(src[y0:y1, cols[0]:cols[-1] + 1])
    else:                                    # wraps through RA seam
        a = np.concatenate([np.asarray(src[y0:y1, cols[0]:]), np.asarray(src[y0:y1, :cols[-1] + 1])], 1)
    return torch.from_numpy(a).to(dev)       # (h, w, 3) float16


def bilinear(reg, X, Y, wrap):
    """reg (h,w,3) f16 on GPU; X,Y in region pixel units (texel centres at i+0.5)."""
    h, w, _ = reg.shape
    X = X - 0.5; Y = Y - 0.5
    x0 = torch.floor(X); y0 = torch.floor(Y)
    fx = (X - x0)[..., None]; fy = (Y - y0)[..., None]
    x0 = x0.long(); y0 = y0.long()
    x1 = x0 + 1; y1 = y0 + 1
    if wrap:
        x0 = x0 % w; x1 = x1 % w
    else:
        x0 = x0.clamp(0, w - 1); x1 = x1.clamp(0, w - 1)
    y0 = y0.clamp(0, h - 1); y1 = y1.clamp(0, h - 1)
    f = lambda yy, xx: reg[yy, xx].float()
    return ((f(y0, x0) * (1 - fx) + f(y0, x1) * fx) * (1 - fy)
            + (f(y1, x0) * (1 - fx) + f(y1, x1) * fx) * fy)


def bake_face(src, face):
    H, W = src.shape[:2]
    y0, y1, x0, x1, wrap = face_region(face, W, H)
    t0 = time.time()
    reg = load_region(src, y0, y1, x0, x1)
    print(f"  face {face}: source rows {y0}-{y1} cols {x0}-{x1} "
          f"({reg.numel() * 2 / 1e9:.2f} GB, {time.time() - t0:.0f}s load)", flush=True)
    lvl0 = torch.empty((N, N, 3), dtype=torch.float16, device=dev)
    off = [(k + 0.5) / SS - 0.5 for k in range(SS)]
    cols = torch.arange(N, device=dev, dtype=torch.float32)
    for r in range(0, N, BAND):
        rows = torch.arange(r, r + BAND, device=dev, dtype=torch.float32)
        acc = torch.zeros((BAND, N, 3), device=dev)
        for oy in off:
            for ox in off:
                s = (2 * (cols[None, :] + 0.5 + ox) / N - 1).expand(BAND, N)
                t = (2 * (rows[:, None] + 0.5 + oy) / N - 1).expand(BAND, N)
                X, Y = src_xy(*face_dirs(face, s, t), W, H)
                if not wrap:                    # region-local column; the window
                    X = torch.remainder(X - x0, W)   # may straddle the RA seam
                acc += bilinear(reg, X, Y - y0, wrap)
        lvl0[r:r + BAND] = (acc / (SS * SS)).half()
    del reg
    torch.cuda.empty_cache()
    return lvl0


def encode_level(img16):
    """img16: (n, n, 3) float16 CPU array. Levels below 4x4 are edge-padded to one block."""
    n = img16.shape[0]
    if n < 4:
        img16 = np.pad(img16, ((0, 4 - n), (0, 4 - n), (0, 0)), mode="edge")
    out = []
    for r in range(0, img16.shape[0], 1024):
        out.append(bc6h_ispc.encode(np.ascontiguousarray(img16[r:r + 1024]), "basic"))
    return b"".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    src = np.load(SRC, mmap_mode="r")
    H, W = src.shape[:2]
    levels = int(math.log2(N)) + 1
    binpath = os.path.join(OUT, "sky64k_bc6h.bin")
    # Encode level-major needs every face's level L together; keep per-face
    # blocks in separate part files and concatenate at the end.
    parts = {}
    T = time.time()
    for face in range(6):
        lvl = bake_face(src, face)                        # (N,N,3) f16, GPU
        cur = lvl
        for L in range(levels):
            n = N >> L
            if L > 0:
                # box 2x2 from the previous level, in float32, in row bands so the
                # 16384^2 level never needs a float32 copy of its own
                nxt = torch.empty((n, n, 3), dtype=torch.float32, device=dev)
                for r in range(0, n, 1024):
                    rr = min(1024, n - r)
                    nxt[r:r + rr] = cur[2 * r:2 * (r + rr)].float().view(rr, 2, n, 2, 3).mean((1, 3))
                cur = nxt
                if L == 1:
                    del lvl
                    torch.cuda.empty_cache()
            blk = encode_level(cur.half().cpu().numpy())
            p = os.path.join(OUT, f".part_L{L:02d}_F{face}.bin")
            with open(p, "wb") as fh:
                fh.write(blk)
            parts[(L, face)] = (p, len(blk), n)
        del cur
        torch.cuda.empty_cache()
        print(f"  face {face} done, {time.time() - T:.0f}s total", flush=True)

    meta = {"face_size": N, "levels": levels, "format": "BC6H_UF16 (GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT)",
            "faces": "+X -X +Y -Y +Z -Z (GL order)", "entries": []}
    off = 0
    with open(binpath, "wb") as out:
        for L in range(levels):
            for face in range(6):
                p, size, n = parts[(L, face)]
                with open(p, "rb") as fh:
                    out.write(fh.read())
                meta["entries"].append({"level": L, "face": face, "offset": off, "size": size, "w": n})
                off += size
                os.remove(p)
    with open(os.path.join(OUT, "sky64k.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"done: {binpath} {off / 1e9:.2f} GB, {time.time() - T:.0f}s")


if __name__ == "__main__":
    main()
