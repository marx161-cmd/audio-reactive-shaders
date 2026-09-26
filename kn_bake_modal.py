"""
kn_bake_modal.py -- the full Kerr-Newman bake on a Modal H100.
scope.md 2026-09-21 "FULL REDO: Kerr-Newman (Q = 0.07 M), no unneeded limits".

Physics as kerr_bake.py (validated: Kerr-Newman shadow edge on R = R' = 0 to
1e-5..1e-6; Q -> 0 reproduces the Kerr bake): Carter-separated Mino-time
equations with Delta = r^2 - 2Mr + a^2 + Q^2, RK4 float64, backward from a ZAMO,
constraint projection, unscaled phi' step control, EPS 2e-5.

Removed limits (vs kerr_bake_modal.py): sky escape 5e3 -> 1e6 Rs; capture at
r_h (1 + 1e-4) (or inward within 1%) instead of 1.02 r_h; gas layer |z| < 5 H
(was 2.5 H); 8 disk passages x 4 equal-column shares (was 2); 48- AND 32-point
path sets from the same integration; adaptive supersampling (8x8, then 32x32
where sub-rays still disagree) for the shadow, the sky and the disk passages.

Outputs on the volume (model-weights:kerr/kn/):
  paths48.npy (48,H,W,4) f32, paths32.npy (32,H,W,4) f32
      .xyz world Rs; .w: [0] L/E, [1] captured (central ray), [2..4] sky dir
  disk.npy (24,H,W,4) f16 -- slot k layers 3k..3k+2:
      [r0 p0 r1 p1], [r2 p2 r3 p3], [Sigma, s_centroid, coverage, 0]
  sky.npy (4,H,W,4) f32 -- up to 4 (dir.xyz, weight); 1 - sum(weight) = shadow

    modal run kn_bake_modal.py --mode test
    modal run kn_bake_modal.py --mode bake
"""
from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
OUT_DIR = "/models/kerr/kn"
app = modal.App("kn-bake")
volume = modal.Volume.from_name("model-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "cupy-cuda12x[ctk]")
    .add_local_file(str(HERE / "kerr_bake.py"), "/work/kerr_bake.py")
    .add_local_file(str(HERE / "kn_disk.py"), "/work/kn_disk.py")
)

NSLOT = 8
NQ = 4
HMULT = 5.0            # gas layer |z| < HMULT * H
DISK_HR = 0.1
DISK_R_OUT_RS = 0.35 / 0.048

KERNEL = r"""
#define NSLOT 8
#define NQ 4
extern "C" {
struct Y { double r, th, ph, vr, vt; };

__device__ void deriv(const Y& y, double l, double q, double a, double M, double Q2, Y& d) {
    double s = sin(y.th), c = cos(y.th);
    if (fabs(s) < 1e-9) s = copysign(1e-9, s);
    double Del = y.r * y.r - 2.0 * M * y.r + a * a + Q2;
    double P = y.r * y.r + a * a - a * l;
    double dR = 4.0 * y.r * P - (2.0 * y.r - 2.0 * M) * (q + (l - a) * (l - a));
    double cot = c / s;
    double dT = -2.0 * a * a * c * s + 2.0 * l * l * cot / (s * s);
    d.r = y.vr; d.th = y.vt;
    d.ph = a * P / Del - a + l / (s * s);
    d.vr = 0.5 * dR; d.vt = 0.5 * dT;
}

__device__ void toWorld(const Y& y, double a, const double* B, double* p) {
    double rho = sqrt(y.r * y.r + a * a);
    double x = rho * sin(y.th) * cos(y.ph), yy = rho * sin(y.th) * sin(y.ph), z = y.r * cos(y.th);
    for (int k = 0; k < 3; k++) p[k] = x * B[k] + yy * B[3 + k] + z * B[6 + k];
}

__device__ void emitSlot(float* dout, int n, int i, int slot, const double* qw, const double* qr,
                         const double* qc, const double* qs, double sig, double sw, double ss) {
    size_t base = (size_t)slot * 3 * n;
    for (int k = 0; k < NQ; k++) {
        float rk = qw[k] > 0 ? (float)(qr[k] / qw[k]) : 0.0f;
        float pk = (float)atan2(qs[k], qc[k]);
        size_t o = (base + (size_t)(k / 2) * n + i) * 4 + (k % 2) * 2;
        dout[o] = rk; dout[o + 1] = pk;
    }
    size_t o = (base + 2 * (size_t)n + i) * 4;
    dout[o] = (float)sig; dout[o + 1] = (float)(sw > 0 ? ss / sw : 0.0);
    dout[o + 2] = 1.0f; dout[o + 3] = 0.0f;
}

// emit_paths = 0 for supersampling sub-rays (only capture / sky / disk needed)
__global__ void kn(const double* L, const double* Qc, const double* VR, const double* VT,
                   int n, double r0, double th0, double ph0, double a, double M, double Q2,
                   double eps, double rpath, double rsky, const double* B,
                   int emit_paths, float* out48, float* out32, float* skyo, int* stat,
                   double rin, double rout, double hr, double hmult, float* dout, int* steps_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double l = L[i], q = Qc[i];
    double rh = M + sqrt(fmax(M * M - a * a - Q2, 0.0));
    double S = 0.0, TT = 0.0;
    int status = 0; double sky[3] = {0, 0, 0};
    double slotTot[NSLOT];
    for (int k = 0; k < NSLOT; k++) slotTot[k] = 0.0;
    long long nsteps = 0;
    const int NS1 = 48, NS2 = 32;
    for (int pass = 0; pass < 2; pass++) {
        Y y = {r0, th0, ph0, VR[i], VT[i]};
        double prev[3], pdir[3] = {0, 0, 0}; toWorld(y, a, B, prev);
        double s = 0.0, turn = 0.0, mprev = 0.0; int k1 = 0, k2 = 0, havedir = 0, inpath = 1;
        if (pass == 1 && emit_paths) {
            for (int c = 0; c < 3; c++) {
                out48[((size_t)0 * n + i) * 4 + c] = (float)prev[c];
                out32[((size_t)0 * n + i) * 4 + c] = (float)prev[c];
            }
            k1 = 1; k2 = 1;
        }
        double last[3] = {prev[0], prev[1], prev[2]};
        status = 0;
        int dslot = 0, inslab = 0;
        double sig = 0, sarc = 0, sw = 0, ss = 0;
        double qw[NQ], qr[NQ], qc[NQ], qs[NQ];
        double prevd[3] = {prev[0], prev[1], prev[2]};
        for (long long st = 0; st < 4000000000LL; st++) {
            Y d0, g2, g3, g4, t;
            deriv(y, l, q, a, M, Q2, d0);
            double rate = fabs(d0.r) / y.r + fabs(d0.th) + fabs(d0.ph) + 1e-12;
            double h = -eps / rate;
            t = {y.r + 0.5*h*d0.r, y.th + 0.5*h*d0.th, y.ph + 0.5*h*d0.ph, y.vr + 0.5*h*d0.vr, y.vt + 0.5*h*d0.vt};
            deriv(t, l, q, a, M, Q2, g2);
            t = {y.r + 0.5*h*g2.r, y.th + 0.5*h*g2.th, y.ph + 0.5*h*g2.ph, y.vr + 0.5*h*g2.vr, y.vt + 0.5*h*g2.vt};
            deriv(t, l, q, a, M, Q2, g3);
            t = {y.r + h*g3.r, y.th + h*g3.th, y.ph + h*g3.ph, y.vr + h*g3.vr, y.vt + h*g3.vt};
            deriv(t, l, q, a, M, Q2, g4);
            y.r  += h / 6.0 * (d0.r  + 2*g2.r  + 2*g3.r  + g4.r);
            y.th += h / 6.0 * (d0.th + 2*g2.th + 2*g3.th + g4.th);
            y.ph += h / 6.0 * (d0.ph + 2*g2.ph + 2*g3.ph + g4.ph);
            y.vr += h / 6.0 * (d0.vr + 2*g2.vr + 2*g3.vr + g4.vr);
            y.vt += h / 6.0 * (d0.vt + 2*g2.vt + 2*g3.vt + g4.vt);
            double sn = sin(y.th); if (fabs(sn) < 1e-9) sn = copysign(1e-9, sn);
            double cs = cos(y.th);
            double Rv = pow(y.r*y.r + a*a - a*l, 2) - (y.r*y.r - 2*M*y.r + a*a + Q2) * (q + (l-a)*(l-a));
            double Tv = q + a*a*cs*cs - l*l*(cs/sn)*(cs/sn);
            if (Rv > 0) y.vr = copysign(sqrt(Rv), y.vr);
            if (Tv > 0) y.vt = copysign(sqrt(Tv), y.vt);
            nsteps++;
            double p[3]; toWorld(y, a, B, p);
            // ---- path samples (turn + arc weighted), both sets from one measure
            if (inpath) {
                double dv[3] = {p[0]-prev[0], p[1]-prev[1], p[2]-prev[2]};
                double seg = sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]);
                if (seg > 0) {
                    double dn[3] = {dv[0]/seg, dv[1]/seg, dv[2]/seg};
                    if (havedir) {
                        double dd = dn[0]*pdir[0] + dn[1]*pdir[1] + dn[2]*pdir[2];
                        double cx = dn[1]*pdir[2] - dn[2]*pdir[1];
                        double cy = dn[2]*pdir[0] - dn[0]*pdir[2];
                        double cz = dn[0]*pdir[1] - dn[1]*pdir[0];
                        turn += atan2(sqrt(cx*cx + cy*cy + cz*cz), dd);
                    }
                    for (int c = 0; c < 3; c++) pdir[c] = dn[c];
                    havedir = 1; s += seg;
                }
                if (pass == 1 && emit_paths) {
                    double m = (TT > 1e-3) ? 0.5 * (s / S + turn / TT) : s / S;
                    while (k1 < NS1 && m >= (double)k1 / (NS1 - 1)) {
                        double f = (m > mprev) ? ((double)k1 / (NS1 - 1) - mprev) / (m - mprev) : 1.0;
                        for (int c = 0; c < 3; c++) out48[((size_t)k1 * n + i) * 4 + c] = (float)(prev[c] + f * (p[c] - prev[c]));
                        k1++;
                    }
                    while (k2 < NS2 && m >= (double)k2 / (NS2 - 1)) {
                        double f = (m > mprev) ? ((double)k2 / (NS2 - 1) - mprev) / (m - mprev) : 1.0;
                        for (int c = 0; c < 3; c++) out32[((size_t)k2 * n + i) * 4 + c] = (float)(prev[c] + f * (p[c] - prev[c]));
                        k2++;
                    }
                    mprev = m;
                }
                for (int c = 0; c < 3; c++) last[c] = p[c];
                if (y.r > rpath && -y.vr > 0) inpath = 0;
            }
            // ---- disk passages
            {
                double dv2[3] = {p[0]-prevd[0], p[1]-prevd[1], p[2]-prevd[2]};
                double segd = sqrt(dv2[0]*dv2[0] + dv2[1]*dv2[1] + dv2[2]*dv2[2]);
                sarc += segd;
                for (int c = 0; c < 3; c++) prevd[c] = p[c];
                if (dslot < NSLOT) {
                    double zb = y.r * cs;
                    double H = hr * y.r;
                    int inside = (y.r >= rin && y.r <= rout && fabs(zb) < hmult * H);
                    if (inside) {
                        if (!inslab) {
                            inslab = 1; sig = 0.0; sw = 0; ss = 0;
                            for (int k = 0; k < NQ; k++) { qw[k] = 0; qr[k] = 0; qc[k] = 0; qs[k] = 0; }
                        }
                        double wgt = exp(-zb * zb / (2.0 * H * H)) * segd;
                        if (pass == 1 && slotTot[dslot] > 0) {
                            int k = (int)(NQ * (sig + 0.5 * wgt) / slotTot[dslot]);
                            k = k < 0 ? 0 : (k > NQ - 1 ? NQ - 1 : k);
                            qw[k] += wgt; qr[k] += wgt * y.r;
                            qc[k] += wgt * cos(y.ph); qs[k] += wgt * sin(y.ph);
                            sw += wgt; ss += wgt * sarc;
                        }
                        sig += wgt;
                    } else if (inslab) {
                        if (pass == 0) slotTot[dslot] = sig;
                        else emitSlot(dout, n, i, dslot, qw, qr, qc, qs, sig, sw, ss);
                        dslot++; inslab = 0;
                    }
                }
            }
            // ---- termination
            int inward = (y.vr > 0);   // backward trace: inward <=> r decreasing <=> vr*h < 0
            if (y.r < rh * (1.0 + 1e-4) || (y.r < rh * 1.01 && inward)) { status = 1; break; }
            if (y.r > rsky && -y.vr > 0) {
                double dv[3] = {p[0]-prev[0], p[1]-prev[1], p[2]-prev[2]};
                double nn = sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]);
                for (int c = 0; c < 3; c++) sky[c] = dv[c] / nn;
                status = 2; break;
            }
            for (int c = 0; c < 3; c++) prev[c] = p[c];
        }
        if (pass == 0) {
            S = s; TT = turn;
            if (inslab && dslot < NSLOT) slotTot[dslot] = sig;
        } else {
            if (inslab && dslot < NSLOT) emitSlot(dout, n, i, dslot, qw, qr, qc, qs, sig, sw, ss);
            if (emit_paths) {
                for (; k1 < NS1; k1++) for (int c = 0; c < 3; c++) out48[((size_t)k1 * n + i) * 4 + c] = (float)last[c];
                for (; k2 < NS2; k2++) for (int c = 0; c < 3; c++) out32[((size_t)k2 * n + i) * 4 + c] = (float)last[c];
            }
        }
    }
    if (emit_paths) {
        float* outs[2] = {out48, out32}; int nss[2] = {NS1, NS2};
        for (int o = 0; o < 2; o++) {
            float* ot = outs[o];
            ot[((size_t)0 * n + i) * 4 + 3] = (float)l;
            ot[((size_t)1 * n + i) * 4 + 3] = (status == 2) ? 0.0f : 1.0f;
            for (int c = 0; c < 3; c++) ot[((size_t)(2 + c) * n + i) * 4 + 3] = (float)sky[c];
            for (int kk = 5; kk < nss[o]; kk++) ot[((size_t)kk * n + i) * 4 + 3] = 0.0f;
        }
    }
    for (int c = 0; c < 3; c++) skyo[(size_t)i * 3 + c] = (float)sky[c];
    stat[i] = status;
    steps_out[i] = (int)fmin((double)nsteps, 2e9);
}
}
"""


def _setup():
    import sys
    sys.path.insert(0, "/work")
    import numpy as np
    import cupy as cp
    import kerr_bake as K
    import kn_disk as KD
    mod = cp.RawModule(code=KERNEL, options=("-std=c++14",))
    return np, cp, K, KD, mod.get_function("kn")


def _consts(np, K, KD):
    a = K.A_STAR * K.M
    Q = K.Q_STAR * K.M
    rin = float(KD.isco(a, Q))
    rout = float(np.sqrt(DISK_R_OUT_RS ** 2 - a * a))
    return a, Q, rin, rout


def _run(np, cp, K, KD, fn, view_dirs, emit_paths=True, eps=None):
    """view_dirs (n,3) numpy -> dict of cupy arrays (kept on the GPU)."""
    eps = K.EPS if eps is None else eps
    a, Q, rin, rout = _consts(np, K, KD)
    l, q, r0, th0, ph0, vr, vt = K.kerr_consts(-view_dirs, a)
    nd, e1, e2, *_ = K.frames()
    B = cp.asarray(np.concatenate([e1, e2, nd]).astype(np.float64))
    n = len(l)
    o48 = cp.zeros((48, n, 4) if emit_paths else (1, 1, 4), dtype=cp.float32)
    o32 = cp.zeros((32, n, 4) if emit_paths else (1, 1, 4), dtype=cp.float32)
    sky = cp.zeros((n, 3), dtype=cp.float32)
    stat = cp.zeros(n, dtype=cp.int32)
    dout = cp.zeros((NSLOT * 3, n, 4), dtype=cp.float32)
    steps = cp.zeros(n, dtype=cp.int32)
    args = (cp.asarray(l), cp.asarray(q), cp.asarray(vr), cp.asarray(vt), np.int32(n),
            np.float64(r0), np.float64(th0), np.float64(ph0), np.float64(a), np.float64(K.M),
            np.float64(Q * Q), np.float64(eps), np.float64(K.R_PATH), np.float64(K.R_SKY), B,
            np.int32(1 if emit_paths else 0), o48, o32, sky, stat,
            np.float64(rin), np.float64(rout), np.float64(DISK_HR), np.float64(HMULT), dout, steps)
    fn(((n + 127) // 128,), (128,), args)
    cp.cuda.Stream.null.synchronize()
    return dict(o48=o48, o32=o32, sky=sky, stat=stat, disk=dout, steps=steps, lam=cp.asarray(l))


# ------------------------------------------------------------- aggregation
def _aggregate(cp, S, stat, sky, disk, K4=4, iters=8):
    """Sub-ray results for P pixels x S sub-rays -> per-pixel coverage data.
    stat (P,S), sky (P,S,3), disk (24,P,S,4)."""
    P = stat.shape[0]
    esc = (stat == 2)
    # --- sky: up to K4 weighted clusters of escaped directions (spherical k-means)
    d = sky * esc[..., None]
    cent = cp.zeros((P, K4, 3), dtype=cp.float32)
    m0 = d.sum(1); n0 = cp.linalg.norm(m0, axis=-1, keepdims=True)
    cent[:, 0] = cp.where(n0 > 0, m0 / cp.maximum(n0, 1e-12), 0)
    for k in range(1, K4):   # farthest-point init
        sim = cp.einsum("psc,pkc->psk", d, cent[:, :k]).max(-1)
        sim = cp.where(esc, sim, 9.0)
        j = cp.argmin(sim, axis=1)
        cent[:, k] = d[cp.arange(P), j]
    for _ in range(iters):
        sim = cp.einsum("psc,pkc->psk", d, cent)
        asg = cp.argmax(sim, axis=-1)
        oh = (asg[..., None] == cp.arange(K4)[None, None, :]) & esc[..., None]
        ssum = cp.einsum("psk,psc->pkc", oh.astype(cp.float32), d)
        nn = cp.linalg.norm(ssum, axis=-1, keepdims=True)
        cent = cp.where(nn > 0, ssum / cp.maximum(nn, 1e-12), cent)
    w = oh.sum(1).astype(cp.float32) / S                      # (P,K4)
    skyo = cp.concatenate([cent, w[..., None]], -1)            # (P,K4,4)
    # --- disk slots, matched by order along the ray
    dk = cp.zeros((24, P, 4), dtype=cp.float32)
    for slot in range(NSLOT):
        A, Bq, C = disk[3 * slot], disk[3 * slot + 1], disk[3 * slot + 2]   # (P,S,4)
        has = C[..., 0] > 0
        cnt = has.sum(1).astype(cp.float32)
        cov = cnt / S
        den = cp.maximum(cnt, 1.0)[:, None]
        def mean(x):
            return (x * has).sum(1) / den[:, 0]
        def cmean(ph):
            return cp.arctan2((cp.sin(ph) * has).sum(1), (cp.cos(ph) * has).sum(1))
        dk[3 * slot, :, 0] = mean(A[..., 0]); dk[3 * slot, :, 1] = cmean(A[..., 1])
        dk[3 * slot, :, 2] = mean(A[..., 2]); dk[3 * slot, :, 3] = cmean(A[..., 3])
        dk[3 * slot + 1, :, 0] = mean(Bq[..., 0]); dk[3 * slot + 1, :, 1] = cmean(Bq[..., 1])
        dk[3 * slot + 1, :, 2] = mean(Bq[..., 2]); dk[3 * slot + 1, :, 3] = cmean(Bq[..., 3])
        dk[3 * slot + 2, :, 0] = mean(C[..., 0]); dk[3 * slot + 2, :, 1] = mean(C[..., 1])
        dk[3 * slot + 2, :, 2] = cov
    return skyo, dk


def _subray_dirs(np, K, ys, xs, nsub):
    """Pixel list -> (P*nsub*nsub, 3) view directions on a regular sub-grid."""
    o = (np.arange(nsub) + 0.5) / nsub - 0.5
    oy, ox = np.meshgrid(o, o, indexing="ij")
    YY = (ys[:, None] + oy.ravel()[None, :]).ravel()
    XX = (xs[:, None] + ox.ravel()[None, :]).ravel()
    return K.pixel_dirs(YY, XX)


def _supersample(np, cp, K, KD, fn, ys, xs, nsub, batch_rays=6_000_000, log=print):
    S = nsub * nsub
    P = len(ys)
    skyo = np.zeros((P, 4, 4), np.float32); dk = np.zeros((24, P, 4), np.float32)
    stat_mixed = np.zeros(P, bool)
    per = max(1, batch_rays // S)
    for b0 in range(0, P, per):
        b1 = min(P, b0 + per)
        d = _subray_dirs(np, K, ys[b0:b1], xs[b0:b1], nsub)
        R = _run(np, cp, K, KD, fn, d, emit_paths=False)
        stat = R["stat"].reshape(b1 - b0, S)
        sky = R["sky"].reshape(b1 - b0, S, 3)
        disk = R["disk"].reshape(24, b1 - b0, S, 4)
        s_o, d_o = _aggregate(cp, S, stat, sky, disk)
        skyo[b0:b1] = cp.asnumpy(s_o); dk[:, b0:b1] = cp.asnumpy(d_o)
        e = (stat == 2).mean(1)
        cnt = (disk[2::3, ..., 0] > 0).sum(0)                     # passages per sub-ray
        stat_mixed[b0:b1] = cp.asnumpy(((e > 0) & (e < 1)) | (cnt.max(1) != cnt.min(1)))
        log(f"  supersample {nsub}x{nsub}: pixels {b1}/{P}, steps max {int(R['steps'].max())}")
    return skyo, dk, stat_mixed


def _edges(np, stat, sky, npass, px_rad):
    """Pixels whose 4-neighbours disagree in capture, sky direction (> 1 px) or
    passage count; dilated by one pixel."""
    H, W = stat.shape
    e = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = (slice(0, H - dy), slice(0, W - dx)); b = (slice(dy, H), slice(dx, W))
        diff = (stat[a] != stat[b]) | (npass[a] != npass[b])
        dot = np.clip((sky[a] * sky[b]).sum(-1), -1, 1)
        both = (stat[a] == 2) & (stat[b] == 2)
        diff |= both & (np.arccos(dot) > px_rad)
        e[a] |= diff; e[b] |= diff
    d = e.copy()
    d[1:] |= e[:-1]; d[:-1] |= e[1:]; d[:, 1:] |= e[:, :-1]; d[:, :-1] |= e[:, 1:]
    return d


@app.function(image=image, gpu="H100", timeout=60 * 60, memory=16384, volumes={"/models": volume})
def test():
    """Kernel vs the numpy reference (kerr_bake.py, now Kerr-Newman) + smoke test
    of supersampling on a small patch across the shadow edge."""
    import time
    np, cp, K, KD, fn = _setup()
    rng = np.random.default_rng(1)
    ys = np.concatenate([rng.integers(0, 2410, 100), rng.integers(1000, 1450, 150)]).astype(float)
    xs = np.concatenate([rng.integers(0, 2410, 100), rng.integers(1050, 1450, 150)]).astype(float)
    d = K.pixel_dirs(ys, xs)
    t0 = time.time(); R = _run(np, cp, K, KD, fn, d); tg = time.time() - t0
    a = K.A_STAR * K.M
    l, q, st, hist, sky = K.trace(d, a)
    Hh = np.stack(hist)
    ref = np.stack([K.resample(Hh[:, :, j], a) for j in range(len(xs))], 1)
    o32 = cp.asnumpy(R["o32"]); capg = o32[1, :, 3] > 0.5; capr = st != 2
    pos = np.linalg.norm(o32[:, :, :3] - ref, axis=-1)
    esc = ~capg & ~capr
    sg = cp.asnumpy(R["sky"])
    dsk = np.degrees(np.arctan2(np.linalg.norm(np.cross(sg[esc], sky[esc]), axis=1), (sg[esc] * sky[esc]).sum(1)))
    o48 = cp.asnumpy(R["o48"])
    rep = (f"kernel {tg:.1f}s for {len(xs)} rays (EPS {K.EPS}), numpy ref at the same EPS\n"
           f"capture mismatches {int((capg != capr).sum())}; 32-pt position diff median {np.median(pos):.2e} max {pos.max():.2e} Rs\n"
           f"sky dir diff (deg) median {np.median(dsk):.2e} max {dsk.max():.2e}  (numpy ref only integrates to R_SKY too)\n"
           f"48-pt set: first/last points equal 32-pt set's: {np.abs(o48[0]-o32[0]).max():.1e} / {np.abs(o48[-1]-o32[-1]).max():.1e}\n")
    # supersampling smoke test on a 12x12 patch straddling the left shadow edge
    py, px = np.meshgrid(np.arange(1200, 1212, dtype=float), np.arange(1150, 1162, dtype=float), indexing="ij")
    so, dk, mixed = _supersample(np, cp, K, KD, fn, py.ravel(), px.ravel(), 8, log=lambda *a: None)
    shadow = 1 - so[:, :, 3].sum(1)
    rep += (f"supersample 8x8 on a 12x12 edge patch: shadow coverage range {shadow.min():.3f}..{shadow.max():.3f}, "
            f"fractional pixels {int(((shadow > 0.01) & (shadow < 0.99)).sum())}, mixed flagged {int(mixed.sum())}\n")
    print(rep)
    return rep


@app.function(image=image, gpu="H100", timeout=6 * 60 * 60, memory=65536, volumes={"/models": volume})
def bake():
    import os, time
    np, cp, K, KD, fn = _setup()
    K.EPS = 2e-5          # the converged step (convergence ladder, scope.md); every _run below uses it
    os.makedirs(OUT_DIR, exist_ok=True)
    H, W = K.H, K.W
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.0f}s] {s}", flush=True)
    # ---- base: every pixel centre, ONE launch (a single straggler tail)
    yy, xx = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    # 2026-09-22 re-run: paths48/paths32 already on the volume from the first
    # run -- not recomputed. Base results are SAVED before supersampling so a
    # stop can never lose them again.
    R = _run(np, cp, K, KD, fn, K.pixel_dirs(yy.ravel(), xx.ravel()), emit_paths=False)
    log(f"base done: steps mean {float(R['steps'].mean()):.0f} max {int(R['steps'].max())}")
    stat = cp.asnumpy(R["stat"]).reshape(H, W)
    sky = cp.asnumpy(R["sky"]).reshape(H, W, 3)
    disk = cp.asnumpy(R["disk"]).reshape(24, H, W, 4)
    del R
    npass = (disk[2::3, ..., 0] > 0).sum(0)
    skyA = np.zeros((4, H, W, 4), np.float32)
    skyA[0, ..., :3] = sky; skyA[0, ..., 3] = (stat == 2)
    np.save(f"{OUT_DIR}/sky_base.npy", skyA)
    np.save(f"{OUT_DIR}/disk_base.npy", disk.astype(np.float16))
    volume.commit()
    log("base sky + disk saved (sky_base.npy, disk_base.npy)")
    # ---- supersampling (user 2026-09-22): 4x4 sub-rays where the sky jumps
    # > 4 px between neighbours, the capture edge, or the passage count changes.
    # The first run's 1 px threshold flagged 99% of the image. No 32x32 level.
    px_rad = 2 * np.arctan(1 / 1.8) / W
    edge = _edges(np, stat, sky, npass, 4.0 * px_rad)
    ey, ex = np.nonzero(edge)
    my = ey[:0]
    log(f"edge pixels: {len(ey)} ({100 * edge.mean():.2f}%) -> 4x4 = {16 * len(ey)} sub-rays")
    if edge.mean() > 0.10:
        log("ABORT: more than 10% edge pixels -- base results are saved, not supersampling")
        return "aborted: too many edge pixels"
    so, dk, mixed = _supersample(np, cp, K, KD, fn, ey.astype(float), ex.astype(float), 4, log=log)
    skyA[:, ey, ex] = so.transpose(1, 0, 2)
    disk[:, ey, ex] = dk
    # non-edge pixels: each present passage has coverage 1 (set by the kernel)
    np.save(f"{OUT_DIR}/sky.npy", skyA)
    np.save(f"{OUT_DIR}/disk.npy", disk.astype(np.float16))
    a, Q, rin, rout = _consts(np, K, KD)
    with open(f"{OUT_DIR}/info.txt", "w") as f:
        f.write(f"a*={K.A_STAR} Q*={K.Q_STAR} eps={K.EPS} r_sky={K.R_SKY} rin={rin:.6f} rout={rout:.6f} "
                f"H/r={DISK_HR} |z|<{HMULT}H slots={NSLOT}x{NQ} edge_px={len(ey)} level2_px={len(my)}\n")
    volume.commit()
    log("done")
    return "ok"


@app.local_entrypoint()
def main(mode: str = "test"):
    if mode == "test":
        print(test.remote())
    elif mode == "bake":
        print(bake.remote())
