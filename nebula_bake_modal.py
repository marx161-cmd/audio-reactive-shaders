"""
nebula_bake_modal.py -- Nebulabrot (Buddhabrot) density, on a Modal H100.
scope.md 2026-09-27 "two Modal bakes for new shaders".

For c sampled over the plane, iterate z -> z^2 + c (FP64) until |z| > 2. Orbits
that escape after n steps are traced again and every point z_1..z_n is counted
into the pixel it lands in, into one of three DISJOINT layers by n:
  0: n < 50   1: 50 <= n < 500   2: 500 <= n < 5000
(the classic cumulative Nebulabrot channels are sums of these).

Only Im c >= 0 is sampled; the orbit of conj(c) is the conjugate orbit, so each
point is also counted at its mirrored pixel. Main cardioid / period-2 bulb are
rejected analytically, other interior points bail on a periodicity check.

Importance sampling: a coarse escape-count map (GW x GH cells, JIT jittered
points each) gives each layer a per-cell sampling probability ~ hits + floor;
every splat is weighted by 1/q(c), so the estimate stays unbiased (the floor
keeps q > 0 everywhere). Without it the long layer is almost pure noise.

Time-boxed: batches run until BUDGET_S, alternating between two independent
halves A/B so the noise can be measured (A - B) at the end.

Output on the volume (model-weights:nebula/):
  nebula<RES>.npy  (3,RES,RES) f32 -- density per layer, units: (orbit points
                   per unit c-area) per pixel; rows = Im z from -H to +H
                   (row 0 = bottom), cols = Re z from CX-H to CX+H
  nebula<RES>.json samples, rates, noise estimate per layer

    modal run nebula_bake_modal.py --mode test    # 4096^2, ~90 s
    modal run nebula_bake_modal.py --mode bake
    # deep-zoom window (centred on the Im axis): importance map counts each
    # orbit's points INSIDE the window, so sampling concentrates on those c
    modal run nebula_bake_modal.py --mode wtest --cx 0.5244 --h 0.2
    modal run nebula_bake_modal.py --mode wbake --cx 0.5244 --h 0.2 --minutes 40 --name nebula_w0
"""
from __future__ import annotations

import json
import time

import modal

app = modal.App("nebula-bake")
volume = modal.Volume.from_name("model-weights", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "cupy-cuda12x[ctk]")

CX, H = -0.4, 1.6            # image window: Re [CX-H, CX+H], Im [-H, H]
BANDS = [(1, 50), (50, 500), (500, 5000)]
NMAX = 5000
SX0, SX1, SY0, SY1 = -2.0, 2.0, 0.0, 2.0     # c sampling domain (upper half)
GW, GH, JIT = 2048, 1024, 16
FLOOR = 0.05                 # share of sampling mass spread uniformly over all cells
SHARE = [0.1, 0.25, 0.65]      # share of GPU time per layer

KERNEL = r"""
extern "C" {
__device__ __forceinline__ unsigned long long mix(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
__device__ __forceinline__ double u01(unsigned long long& s) {
    s += 0x9E3779B97F4A7C15ULL;
    return (mix(s) >> 11) * (1.0 / 9007199254740992.0);
}

// escape step n (|z_n| > 2, z_0 = 0), or -1 if it never escapes within nmax
__device__ int escape(double cx, double cy, int nmax) {
    double xq = cx - 0.25, q = xq * xq + cy * cy;
    if (q * (q + xq) <= 0.25 * cy * cy) return -1;
    if ((cx + 1.0) * (cx + 1.0) + cy * cy <= 0.0625) return -1;
    double x = 0.0, y = 0.0, sx = 0.0, sy = 0.0;
    int check = 8, cnt = 0;
    for (int n = 1; n <= nmax; n++) {
        double ny = 2.0 * x * y + cy;
        x = x * x - y * y + cx; y = ny;
        if (x * x + y * y > 4.0) return n;
        if (fabs(x - sx) < 1e-14 && fabs(y - sy) < 1e-14) return -1;
        if (++cnt == check) { cnt = 0; check <<= 1; sx = x; sy = y; }
    }
    return -1;
}

__global__ void coarse(int* hits, int gw, int gh, int jit, double x0, double y0,
                       double cw, double ch, int nmax, int b0, int b1, int b2, unsigned long long seed,
                       int windowed, double wx0, double wy0, double wx1, double wy1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gw * gh) return;
    unsigned long long s = mix(seed ^ (unsigned long long)i * 0x632BE59BD9B4E019ULL);
    int ix = i % gw, iy = i / gw;
    int h0 = 0, h1 = 0, h2 = 0;
    for (int j = 0; j < jit; j++) {
        double cx = x0 + (ix + u01(s)) * cw, cy = y0 + (iy + u01(s)) * ch;
        int n = escape(cx, cy, nmax);
        if (n < b0) continue;                  // also n == -1; b0 = lower cut (glitter: 5000)
        int w = 1;
        if (windowed) {
            // how many of this orbit's points land in the window (mirror included):
            // the sampling weight should follow what the sample CONTRIBUTES there
            w = 0;
            double x = 0.0, y = 0.0;
            for (int k = 1; k <= n; k++) {
                double ny = 2.0 * x * y + cy;
                x = x * x - y * y + cx; y = ny;
                if (x >= wx0 && x < wx1) {
                    if (y >= wy0 && y < wy1) w++;
                    if (-y >= wy0 && -y < wy1) w++;
                }
            }
            if (w > 1000) w = 1000;
        }
        if (n < b1) h0 += w; else if (n < b2) h1 += w; else h2 += w;
    }
    hits[i] = h0; hits[gw * gh + i] = h1; hits[2 * gw * gh + i] = h2;
}

__global__ void splat(float* acc, int res, const double* cdf, const float* wcell, int ncell,
                      int gw, double x0, double y0, double cw, double ch,
                      int lo, int hi, double wx0, double wy0, double inv_pix,
                      int per_thread, unsigned long long seed, unsigned long long* accepted) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long s = mix(seed ^ (unsigned long long)t * 0xD1B54A32D192ED03ULL);
    unsigned long long acc_n = 0;
    for (int k = 0; k < per_thread; k++) {
        double u = u01(s);
        int a = 0, b = ncell - 1;              // first cell with cdf >= u
        while (a < b) { int m = (a + b) >> 1; if (cdf[m] < u) a = m + 1; else b = m; }
        int ix = a % gw, iy = a / gw;
        double cx = x0 + (ix + u01(s)) * cw, cy = y0 + (iy + u01(s)) * ch;
        int n = escape(cx, cy, hi - 1);
        if (n < lo) continue;                  // also n == -1
        acc_n++;
        float w = wcell[a];
        double x = 0.0, y = 0.0;
        for (int j = 1; j <= n; j++) {
            double ny = 2.0 * x * y + cy;
            x = x * x - y * y + cx; y = ny;
            double fc = (x - wx0) * inv_pix, fr = (y - wy0) * inv_pix;
            if (fc < 0.0 || fr < 0.0 || fc >= res || fr >= res) continue;
            int col = (int)fc, row = (int)fr;
            atomicAdd(&acc[(size_t)row * res + col], w);
            atomicAdd(&acc[(size_t)(res - 1 - row) * res + col], w);
        }
    }
    atomicAdd(accepted, acc_n);
}
}
"""


def run(res, budget_s, out_name, window=None, jit=JIT, bands=None, nmax=None, share=None):
    bands = bands or BANDS
    nmax = nmax or NMAX
    share = share or SHARE
    # window = (cx, h): bake only Re [cx-h, cx+h], Im [-h, h] (must be centred on Im 0: mirror)
    wcx, wh = window if window else (CX, H)
    import numpy as np
    import cupy as cp

    mod = cp.RawModule(code=KERNEL, options=("-std=c++14",))
    k_coarse, k_splat = mod.get_function("coarse"), mod.get_function("splat")
    ncell = GW * GH
    cw, ch = (SX1 - SX0) / GW, (SY1 - SY0) / GH
    cell_area = cw * ch

    t0 = time.time()
    hits = cp.zeros(3 * ncell, dtype=cp.int32)
    k_coarse(((ncell + 255) // 256,), (256,),
             (hits, np.int32(GW), np.int32(GH), np.int32(jit), np.float64(SX0), np.float64(SY0),
              np.float64(cw), np.float64(ch), np.int32(nmax), np.int32(bands[0][0]), np.int32(bands[0][1]),
              np.int32(bands[1][1]), np.uint64(12345), np.int32(1 if window else 0),
              np.float64(wcx - wh), np.float64(-wh), np.float64(wcx + wh), np.float64(wh)))
    cp.cuda.Stream.null.synchronize()
    hits = hits.reshape(3, ncell).astype(cp.float64)
    t_map = time.time() - t0

    cdfs, wcells = [], []
    for L in range(3):
        h = hits[L]
        p = h / h.sum() * (1 - FLOOR) + FLOOR / ncell
        cdf = cp.cumsum(p)
        cdf /= cdf[-1]
        cdfs.append(cdf)
        wcells.append((cell_area / p).astype(cp.float32))   # 1 / q(c), q = p / cell_area

    pix = 2 * wh / res
    wx0, wy0 = wcx - wh, -wh
    acc = cp.zeros(res * res, dtype=cp.float32)
    tot = [[cp.zeros(res * res, dtype=cp.float64) for _ in range(3)] for _ in range(2)]
    nsamp = [[0, 0, 0], [0, 0, 0]]
    nacc = [[0, 0, 0], [0, 0, 0]]
    accepted = cp.zeros(1, dtype=cp.uint64)
    threads = 1 << 20
    per_thread = [4, 4, 4]
    nbatch = [0, 0, 0]
    used = [0.0, 0.0, 0.0]
    t_start = time.time()
    batch = 0
    last_print = 0.0
    while time.time() - t_start < budget_s:
        el = time.time() - t_start
        L = min(range(3), key=lambda l: used[l] / share[l])
        half = nbatch[L] & 1
        nbatch[L] += 1
        acc.fill(0)
        accepted.fill(0)
        tb = time.time()
        k_splat((threads // 256,), (256,),
                (acc, np.int32(res), cdfs[L], wcells[L], np.int32(ncell), np.int32(GW),
                 np.float64(SX0), np.float64(SY0), np.float64(cw), np.float64(ch),
                 np.int32(bands[L][0]), np.int32(bands[L][1]),
                 np.float64(wx0), np.float64(wy0), np.float64(1.0 / pix),
                 np.int32(per_thread[L]), np.uint64(0xABCDEF + batch * 7919), accepted))
        tot[half][L] += acc
        cp.cuda.Stream.null.synchronize()
        dt = time.time() - tb
        used[L] += dt
        nsamp[half][L] += threads * per_thread[L]
        nacc[half][L] += int(accepted.get()[0])
        # aim for ~2 s launches
        per_thread[L] = int(min(max(per_thread[L] * 2.0 / max(dt, 1e-3), 1), 1 << 16))
        batch += 1
        if el - last_print > 60:
            last_print = el
            print(json.dumps(dict(t=round(el), batch=batch, used=[round(u) for u in used],
                                  samples=[nsamp[0][l] + nsamp[1][l] for l in range(3)],
                                  accepted=[nacc[0][l] + nacc[1][l] for l in range(3)])), flush=True)

    out = np.empty((3, res, res), dtype=np.float32)
    noise = []
    for L in range(3):
        nA, nB = nsamp[0][L], nsamp[1][L]
        est = (tot[0][L] + tot[1][L]) / (nA + nB)
        a, b = tot[0][L] / nA, tot[1][L] / nB
        # relative noise of the full estimate, over the brightest half of the lit pixels
        lit = est > 0
        thr = cp.percentile(est[lit], 50) if int(lit.sum()) else 0
        m = est > thr
        rel = float(cp.median(cp.abs(a[m] - b[m]) / (a[m] + b[m]))) if int(m.sum()) else float("nan")
        noise.append(dict(rel_noise_median_bright=rel, lit_frac=float(lit.mean()),
                          max=float(est.max()), p999=float(cp.percentile(est[lit], 99.9)) if int(lit.sum()) else 0))
        out[L] = cp.asnumpy(est.reshape(res, res).astype(cp.float32))
    info = dict(res=res, window=dict(cx=wcx, h=wh), coarse_hits=[float(hits[l].sum()) for l in range(3)], bands=bands, nmax=nmax, budget_s=budget_s,
                coarse_map_s=round(t_map, 1), gpu_s=used, batches=batch,
                samples=[nsamp[0][l] + nsamp[1][l] for l in range(3)],
                accepted=[nacc[0][l] + nacc[1][l] for l in range(3)], noise=noise,
                sampling=dict(domain=[SX0, SX1, SY0, SY1], grid=[GW, GH], jit=JIT, floor=FLOOR))
    import os
    os.makedirs("/models/nebula", exist_ok=True)
    np.save(f"/models/nebula/{out_name}.npy", out)
    with open(f"/models/nebula/{out_name}.json", "w") as fh:
        json.dump(info, fh)
    volume.commit()
    return info


@app.function(image=image, gpu="H100", timeout=15 * 60, memory=16384, volumes={"/models": volume})
def test():
    return run(4096, 90, "nebula_test4096")


@app.function(image=image, gpu="H100", timeout=100 * 60, memory=32768, volumes={"/models": volume})
def bake(minutes: float = 60):
    return run(16384, minutes * 60, "nebula16384")


@app.function(image=image, gpu="H100", timeout=15 * 60, memory=16384, volumes={"/models": volume})
def wtest(cx: float, h: float):
    return run(4096, 90, f"nebula_win_test", window=(cx, h), jit=64)


@app.function(image=image, gpu="H100", timeout=100 * 60, memory=32768, volumes={"/models": volume})
def wbake(cx: float, h: float, minutes: float, name: str):
    return run(16384, minutes * 60, name, window=(cx, h), jit=64)



# ---- Metropolis-Hastings sampler for deep-zoom WINDOW glitter ---------------
# Target: pi(c) ~ f(c) = number of the orbit's points inside the window
# (mirror included) for orbits escaping in [lo, nmax). 80% small mutations
# (Gaussian, magnitude log-uniform in [SIG_MIN, SIG_MAX]), 20% independent
# jumps from the window-weighted importance map q (correct MH ratio for those).
# Each state is splatted once when the chain leaves it, with weight stay/f,
# so E[histogram] = true density / Z; Z = E_q[f/q] is estimated from the
# independent proposals. Result is in the same units as the uniform bakes.
MCMC_SRC = r"""
__device__ int cellOf(double cx, double cy, int gw, int gh, double x0, double y0, double cw, double ch) {
    int ix = (int)floor((cx - x0) / cw), iy = (int)floor((cy - y0) / ch);
    if (ix < 0 || iy < 0 || ix >= gw || iy >= gh) return -1;
    return iy * gw + ix;
}
__device__ int contrib(double cx, double cy, int lo, int nmax,
                       double wx0, double wy0, double wx1, double wy1, int* nout) {
    int n = escape(cx, cy, nmax - 1);
    *nout = n;
    if (n < lo) return 0;
    double x = 0.0, y = 0.0; int w = 0;
    for (int k = 1; k <= n; k++) {
        double ny = 2.0 * x * y + cy;
        x = x * x - y * y + cx; y = ny;
        if (x >= wx0 && x < wx1) {
            if (y >= wy0 && y < wy1) w++;
            if (-y >= wy0 && -y < wy1) w++;
        }
    }
    return w;
}
__device__ void splatOrbit(float* acc, int res, double cx, double cy, int n, float wt,
                           int b1, int b2, double wx0, double wy0, double inv_pix) {
    size_t plane = (size_t)res * res * (n < b1 ? 0 : (n < b2 ? 1 : 2));
    double x = 0.0, y = 0.0;
    for (int j = 1; j <= n; j++) {
        double ny = 2.0 * x * y + cy;
        x = x * x - y * y + cx; y = ny;
        double fc = (x - wx0) * inv_pix, fr = (y - wy0) * inv_pix;
        if (fc < 0.0 || fr < 0.0 || fc >= res || fr >= res) continue;
        int col = (int)fc, row = (int)fr;
        atomicAdd(&acc[plane + (size_t)row * res + col], wt);
        atomicAdd(&acc[plane + (size_t)(res - 1 - row) * res + col], wt);
    }
}
__global__ void mcmc(float* acc, int res, double* scx, double* scy, int* sf, int* sn, float* sstay, int* sage,
                     const double* cdf, const float* qcell, int ncell, int gw, int gh,
                     double x0, double y0, double cw, double ch,
                     int lo, int b1, int b2, int nmax, double wx0, double wy0, double wx1, double wy1,
                     double inv_pix, double sig_small, double sig_big, float pbig, float plarge, int steps, int flush,
                     int max_age, int burn, unsigned long long seed, double* zsum, unsigned long long* zcnt) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long s = mix(seed ^ (unsigned long long)t * 0x9E6C63D0676A9A99ULL);
    double cx = scx[t], cy = scy[t]; int f = sf[t], n = sn[t]; float stay = sstay[t]; int age = sage[t];
    if (flush) { if (f > 0 && age >= burn) splatOrbit(acc, res, cx, cy, n, stay / (float)f, b1, b2, wx0, wy0, inv_pix); sf[t] = 0; return; }
    double zs = 0.0; unsigned long long zc = 0;
    for (int tries = 0; f == 0 && tries < 512; tries++) {       // seed the chain from q
        double u = u01(s); int a = 0, b = ncell - 1;
        while (a < b) { int m = (a + b) >> 1; if (cdf[m] < u) a = m + 1; else b = m; }
        cx = x0 + ((a % gw) + u01(s)) * cw; cy = y0 + ((a / gw) + u01(s)) * ch;
        f = contrib(cx, cy, lo, nmax, wx0, wy0, wx1, wy1, &n);
        zs += f * (double)qcell[a]; zc++; stay = 1.0f; age = 0;
    }
    for (int k = 0; k < steps && f > 0; k++) {
        double px, py; bool large = u01(s) < plarge; int cellP = -1;
        if (large) {
            double u = u01(s); int a = 0, b = ncell - 1;
            while (a < b) { int m = (a + b) >> 1; if (cdf[m] < u) a = m + 1; else b = m; }
            px = x0 + ((a % gw) + u01(s)) * cw; py = y0 + ((a / gw) + u01(s)) * ch; cellP = a;
        } else {
            // two-scale Gaussian: mostly small nudges, some big exploratory steps
            double mag = (u01(s) < pbig) ? sig_big : sig_small;
            double u1 = fmax(u01(s), 1e-300), u2 = u01(s);
            double r = sqrt(-2.0 * log(u1)) * mag;
            px = cx + r * cos(6.283185307179586 * u2);
            py = fabs(cy + r * sin(6.283185307179586 * u2));      // conj symmetry: stay in Im >= 0
        }
        int np; int fp = (px < -2.0 || px > 2.0 || py > 2.0) ? 0 : contrib(px, py, lo, nmax, wx0, wy0, wx1, wy1, &np);
        double acc_p;
        if (large) {
            zs += fp * (double)qcell[cellP]; zc++;                // qcell = 1/q = cell_area/p
            int cellC = cellOf(cx, cy, gw, gh, x0, y0, cw, ch);
            double inv_qc = cellC >= 0 ? (double)qcell[cellC] : 1e30;
            // independent proposal: a = f' q(c) / (f q(c')) = f' (1/q(c')) / (f (1/q(c)))
            acc_p = (fp * (double)qcell[cellP]) / (f * inv_qc);
        } else {
            acc_p = (double)fp / (double)f;
        }
        if (fp > 0 && u01(s) < acc_p) {
            if (age >= burn) splatOrbit(acc, res, cx, cy, n, stay / (float)f, b1, b2, wx0, wy0, inv_pix);
            cx = px; cy = py; f = fp; n = np; stay = 1.0f; age++;
            if (age >= max_age) { f = 0; break; }                  // retire the chain; reseeded next launch
        } else {
            stay += 1.0f;
        }
    }
    scx[t] = cx; scy[t] = cy; sf[t] = f; sn[t] = n; sstay[t] = stay; sage[t] = age;
    atomicAdd(zsum, zs); atomicAdd(zcnt, zc);
}
"""


def run_mcmc(res, budget_s, out_name, window, bands, nmax, jit=64, threads=1 << 18,
             sig_small=5e-4, sig_big=0.05, pbig=0.1, plarge=0.1, max_age=5000, burn=64):
    import numpy as np
    import cupy as cp
    src = KERNEL.rstrip()
    assert src.endswith("}")
    mod = cp.RawModule(code=src[:-1] + MCMC_SRC + "\n}\n", options=("-std=c++14",))
    k_coarse, k_mcmc = mod.get_function("coarse"), mod.get_function("mcmc")
    wcx, wh = window
    ncell = GW * GH
    cw, ch = (SX1 - SX0) / GW, (SY1 - SY0) / GH
    wx0, wy0, wx1, wy1 = wcx - wh, -wh, wcx + wh, wh
    t0 = time.time()
    hits = cp.zeros(3 * ncell, dtype=cp.int32)
    k_coarse(((ncell + 255) // 256,), (256,),
             (hits, np.int32(GW), np.int32(GH), np.int32(jit), np.float64(SX0), np.float64(SY0),
              np.float64(cw), np.float64(ch), np.int32(nmax), np.int32(bands[0][0]), np.int32(bands[0][1]),
              np.int32(bands[1][1]), np.uint64(777), np.int32(1),
              np.float64(wx0), np.float64(wy0), np.float64(wx1), np.float64(wy1)))
    cp.cuda.Stream.null.synchronize()
    h = hits.reshape(3, ncell).astype(cp.float64).sum(0)
    p = h / max(float(h.sum()), 1.0) * (1 - FLOOR) + FLOOR / ncell
    cdf = cp.cumsum(p); cdf /= cdf[-1]
    qcell = (cw * ch / p).astype(cp.float32)                       # 1/q(c) per cell
    t_map = time.time() - t0
    scx = cp.zeros(threads, cp.float64); scy = cp.zeros(threads, cp.float64)
    sf = cp.zeros(threads, cp.int32); sn = cp.zeros(threads, cp.int32); sstay = cp.zeros(threads, cp.float32)
    sage = cp.zeros(threads, cp.int32)
    acc = cp.zeros(3 * res * res, cp.float32)
    tot = cp.zeros(3 * res * res, cp.float64)
    zsum = cp.zeros(1, cp.float64); zcnt = cp.zeros(1, cp.uint64)
    inv_pix = res / (2 * wh)
    steps, total_steps, batch = 4, 0, 0
    t_start = time.time(); last = 0.0
    args = lambda st, fl, sd: (acc, np.int32(res), scx, scy, sf, sn, sstay, sage, cdf, qcell, np.int32(ncell),
                               np.int32(GW), np.int32(GH), np.float64(SX0), np.float64(SY0),
                               np.float64(cw), np.float64(ch), np.int32(bands[0][0]), np.int32(bands[0][1]),
                               np.int32(bands[1][1]), np.int32(nmax), np.float64(wx0), np.float64(wy0),
                               np.float64(wx1), np.float64(wy1), np.float64(inv_pix), np.float64(sig_small),
                               np.float64(sig_big), np.float32(pbig), np.float32(plarge), np.int32(st), np.int32(fl),
                               np.int32(max_age), np.int32(burn), np.uint64(sd), zsum, zcnt)
    while time.time() - t_start < budget_s:
        acc.fill(0)
        tb = time.time()
        k_mcmc((threads // 256,), (256,), args(steps, 0, 0x5EED + batch * 104729))
        tot += acc
        cp.cuda.Stream.null.synchronize()
        dt = time.time() - tb
        total_steps += threads * steps
        steps = int(min(max(steps * 2.0 / max(dt, 1e-3), 1), 1 << 14))
        batch += 1
        el = time.time() - t_start
        if el - last > 60:
            last = el
            print(json.dumps(dict(t=round(el), batch=batch, steps=total_steps,
                                  live=int((sf > 0).sum()), z=float(zsum.get()[0] / max(int(zcnt.get()[0]), 1)))), flush=True)
    acc.fill(0)
    k_mcmc((threads // 256,), (256,), args(0, 1, 1))                   # flush the chains' last states
    tot += acc
    cp.cuda.Stream.null.synchronize()
    Z = float(zsum.get()[0]) / max(int(zcnt.get()[0]), 1)
    est = (tot * (Z / max(total_steps, 1))).reshape(3, res, res)
    out = cp.asnumpy(est.astype(cp.float32))
    info = dict(res=res, window=dict(cx=wcx, h=wh), bands=bands, nmax=nmax, sampler="mcmc",
                coarse_map_s=round(t_map, 1), coarse_hits=float(h.sum()), budget_s=budget_s,
                batches=batch, total_steps=total_steps, Z=Z, live_chains=int((sf > 0).sum()),
                sig=[sig_small, sig_big], pbig=pbig, plarge=plarge, max_age=max_age, burn=burn, threads=threads,
                lit_frac=[float((out[l] > 0).mean()) for l in range(3)],
                p999=[float(np.percentile(out[l][out[l] > 0], 99.9)) if (out[l] > 0).any() else 0 for l in range(3)])
    import os
    os.makedirs("/models/nebula", exist_ok=True)
    np.save(f"/models/nebula/{out_name}.npy", out)
    with open(f"/models/nebula/{out_name}.json", "w") as fh:
        json.dump(info, fh)
    volume.commit()
    return info



# ---- ANTI-Buddhabrot: orbits of points that NEVER escape ---------------------
# c uniform over the set's area (upper half; conj mirror), orbit checked for
# ANTI_CHECK steps (escapes -> discarded), then the first ANTI_SPLAT points are
# counted: the spiral in plus the attracting cycle. Channels by where c sits:
# 0 main cardioid (fixed point), 1 period-2 bulb, 2 everything else (the
# smaller bulbs: longer cycles, the intricate threads). Same window as the
# main bake (CX, H), so it lines up. Units: points per unit c-area per pixel.
ANTI_SRC = r"""
__global__ void anti(float* acc, int res, double ax0, double ay0, double aw, double ah,
                     int ncheck, int nsplat, double wx0, double wy0, double inv_pix,
                     int per_thread, unsigned long long seed, unsigned long long* accepted) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long s = mix(seed ^ (unsigned long long)t * 0xA24BAED4963EE407ULL);
    unsigned long long an = 0;
    for (int k = 0; k < per_thread; k++) {
        double cx = ax0 + u01(s) * aw, cy = ay0 + u01(s) * ah;
        double xq = cx - 0.25, q = xq * xq + cy * cy;
        int ch = 2;
        if (q * (q + xq) <= 0.25 * cy * cy) ch = 0;
        else if ((cx + 1.0) * (cx + 1.0) + cy * cy <= 0.0625) ch = 1;
        double x = 0.0, y = 0.0; bool esc = false;
        if (ch == 2) {
            for (int n = 0; n < ncheck; n++) {
                double ny = 2.0 * x * y + cy;
                x = x * x - y * y + cx; y = ny;
                if (x * x + y * y > 4.0) { esc = true; break; }
            }
        }
        if (esc) continue;
        an++;
        size_t plane = (size_t)res * res * ch;
        x = 0.0; y = 0.0;
        for (int j = 1; j <= nsplat; j++) {
            double ny = 2.0 * x * y + cy;
            x = x * x - y * y + cx; y = ny;
            double fc = (x - wx0) * inv_pix, fr = (y - wy0) * inv_pix;
            if (fc < 0.0 || fr < 0.0 || fc >= res || fr >= res) continue;
            int col = (int)fc, row = (int)fr;
            atomicAdd(&acc[plane + (size_t)row * res + col], 1.0f);
            atomicAdd(&acc[plane + (size_t)(res - 1 - row) * res + col], 1.0f);
        }
    }
    atomicAdd(accepted, an);
}
"""
ANTI_CHECK, ANTI_SPLAT = 4000, 2000


def run_anti(res, budget_s, out_name, threads=1 << 20):
    import numpy as np
    import cupy as cp
    src = KERNEL.rstrip()
    mod = cp.RawModule(code=src[:-1] + ANTI_SRC + "\n}\n", options=("-std=c++14",))
    k_anti = mod.get_function("anti")
    ax0, ay0, aw, ah = -2.0, 0.0, 2.5, 1.25          # the set lies in Re [-2, 0.5], |Im| < 1.25
    acc = cp.zeros(3 * res * res, cp.float32)
    tot = cp.zeros(3 * res * res, cp.float64)
    accepted = cp.zeros(1, cp.uint64)
    pix = 2 * H / res
    per_thread, nsamp, batch = 2, 0, 0
    t_start = time.time(); last = 0.0
    while time.time() - t_start < budget_s:
        acc.fill(0)
        tb = time.time()
        k_anti((threads // 256,), (256,),
               (acc, np.int32(res), np.float64(ax0), np.float64(ay0), np.float64(aw), np.float64(ah),
                np.int32(ANTI_CHECK), np.int32(ANTI_SPLAT), np.float64(CX - H), np.float64(-H),
                np.float64(1.0 / pix), np.int32(per_thread), np.uint64(0xA471 + batch * 7919), accepted))
        tot += acc
        cp.cuda.Stream.null.synchronize()
        dt = time.time() - tb
        nsamp += threads * per_thread
        per_thread = int(min(max(per_thread * 2.0 / max(dt, 1e-3), 1), 1 << 12))
        batch += 1
        el = time.time() - t_start
        if el - last > 60:
            last = el
            print(json.dumps(dict(t=round(el), batch=batch, samples=nsamp, accepted=int(accepted.get()[0]))), flush=True)
    est = (tot * (aw * ah / max(nsamp, 1))).reshape(3, res, res)     # per unit c-area, like the other bakes
    out = cp.asnumpy(est.astype(cp.float32))
    info = dict(res=res, window=dict(cx=CX, h=H), sampler="anti", check=ANTI_CHECK, splat=ANTI_SPLAT,
                samples=nsamp, accepted=int(accepted.get()[0]), budget_s=budget_s,
                lit_frac=[float((out[l] > 0).mean()) for l in range(3)],
                p999=[float(np.percentile(out[l][out[l] > 0], 99.9)) if (out[l] > 0).any() else 0 for l in range(3)])
    import os
    os.makedirs("/models/nebula", exist_ok=True)
    np.save(f"/models/nebula/{out_name}.npy", out)
    with open(f"/models/nebula/{out_name}.json", "w") as fh:
        json.dump(info, fh)
    volume.commit()
    return info


# GLITTER layer (2026-09-27): only orbits that take >= 5000 steps to escape --
# they circle near attracting cycles and trace the small rings/filaments; three
# length bands -> three channels, so each can get its own tint. Composited on
# top of the smooth bake in the shader.
GLITTER_BANDS = [(5000, 20000), (20000, 80000), (80000, 300000)]
GLITTER_NMAX = 300000
GLITTER_SHARE = [0.3, 0.35, 0.35]


@app.function(image=image, gpu="H100", timeout=20 * 60, memory=16384, volumes={"/models": volume})
def gtest(seconds: float = 120):
    return run(4096, seconds, "nebula_glitter_test", bands=GLITTER_BANDS, nmax=GLITTER_NMAX,
               share=GLITTER_SHARE, jit=16)


@app.function(image=image, gpu="H100", timeout=100 * 60, memory=32768, volumes={"/models": volume})
def gbake(minutes: float, name: str):
    return run(16384, minutes * 60, name, bands=GLITTER_BANDS, nmax=GLITTER_NMAX,
               share=GLITTER_SHARE, jit=16)


@app.function(image=image, gpu="H100", timeout=20 * 60, memory=16384, volumes={"/models": volume})
def mtest(cx: float, h: float, seconds: float = 120):
    return run_mcmc(4096, seconds, "nebula_mcmc_test", (cx, h), GLITTER_BANDS, GLITTER_NMAX)


@app.function(image=image, gpu="H100", timeout=20 * 60, memory=16384, volumes={"/models": volume})
def mvalidate(seconds: float = 90):
    # known answer: nebula_test4096 (uniform sampler, same bands, same window)
    return run_mcmc(4096, seconds, "nebula_mcmc_validate", (CX, H), BANDS, NMAX)


@app.function(image=image, gpu="H100", timeout=100 * 60, memory=32768, volumes={"/models": volume})
def mbake(cx: float, h: float, minutes: float, name: str):
    return run_mcmc(16384, minutes * 60, name, (cx, h), GLITTER_BANDS, GLITTER_NMAX)


@app.function(image=image, gpu="H100", timeout=15 * 60, memory=16384, volumes={"/models": volume})
def atest(seconds: float = 60):
    return run_anti(4096, seconds, "nebula_anti_test")


@app.function(image=image, gpu="H100", timeout=60 * 60, memory=32768, volumes={"/models": volume})
def abake(minutes: float, name: str):
    return run_anti(16384, minutes * 60, name)


@app.function(image=image, gpu="H100", timeout=20 * 60, memory=16384, volumes={"/models": volume})
def wmtest(cx: float, h: float, seconds: float = 90):
    # deep window, ORDINARY bands (same kind as nebula_w0), Metropolis: tiny windows make uniform/importance sampling hopeless
    return run_mcmc(4096, seconds, "nebula_wm_test", (cx, h), BANDS, NMAX)


@app.function(image=image, gpu="H100", timeout=100 * 60, memory=32768, volumes={"/models": volume})
def wmbake(cx: float, h: float, minutes: float, name: str):
    return run_mcmc(16384, minutes * 60, name, (cx, h), BANDS, NMAX)


@app.local_entrypoint()
def main(mode: str = "test", minutes: float = 60, cx: float = 0.0, h: float = 0.0, name: str = ""):
    if mode == "test":
        r = test.remote()
    elif mode == "bake":
        r = bake.remote(minutes)
    elif mode == "wtest":
        r = wtest.remote(cx, h)
    elif mode == "gtest":
        r = gtest.remote()
    elif mode == "gbake":
        r = gbake.remote(minutes, name)
    elif mode == "atest":
        r = atest.remote()
    elif mode == "abake":
        r = abake.remote(minutes, name)
    elif mode == "wmtest":
        r = wmtest.remote(cx, h)
    elif mode == "wmbake":
        r = wmbake.remote(cx, h, minutes, name)
    elif mode == "mvalidate":
        r = mvalidate.remote()
    elif mode == "mtest":
        r = mtest.remote(cx, h)
    elif mode == "mbake":
        r = mbake.remote(cx, h, minutes, name)
    else:
        r = wbake.remote(cx, h, minutes, name)
    print(json.dumps(r, indent=1))
