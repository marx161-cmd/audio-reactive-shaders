#!/usr/bin/env python3
"""
2D (vertically integrated) hydrodynamic simulation of the accretion disk, for
the Kerr lens of schwarz_orrery_v7_opt. Decision record: scope.md 2026-09-21
"gas disk: 2D fluid sim + chord bake".

Physics (units Rs = 1, c = 1, M = 0.5, a = a* M):
  * surface density S, radial velocity u, azimuthal velocity w on a log-polar
    grid x = ln r in [ln r_isco, ln r_out], phi in [0, 2 pi).
  * finite volume, conservative in (S, S u, J = S r w) -- angular momentum is
    conserved exactly by construction (no source term in the J equation).
  * gravity g(r) = -r Omega_K(r)^2 with the KERR prograde Omega_K
    = sqrt(M) / (r^1.5 + a sqrt(M)) (BL r), so circular gas orbits at the
    correct Kerr angular velocity at every radius.
  * locally isothermal pressure P = S cs^2, cs = h r Omega_K, h = H/r = 0.1
    (the user's "medium" thickness).
  * turbulence: stochastic forcing potential (Laughlin, Steinacker & Adams
    2004 style) -- short-lived random modes that the Keplerian shear winds into
    filaments; stands in for the MRI, which 2D hydro cannot produce.
  * Rusanov fluxes, MUSCL/minmod reconstruction, Heun (RK2) time stepping,
    wave-damping zones at both radial edges, free inflow into the hole at r_isco.

Output: frames of the density CONTRAST S / <S>(r), float16, (L, N_r, N_phi) --
row = radial index, column = azimuth -- looped over the last FADE frames so
frame L-1 -> 0 is seamless. Works with numpy (CPU test) or cupy (GPU).
"""
import argparse
import sys
import time

import numpy as np

M = 0.5
A_STAR = 0.997
Q_STAR = 0.07          # 2026-09-21: Kerr-Newman charge (scope.md FULL REDO)
H_R = 0.1
R_OUT = float(np.sqrt((0.35 / 0.048) ** 2 - (A_STAR * M) ** 2))


# Kerr-Newman ISCO / Omega from kn_disk.py (validated: Q -> 0 reproduces the
# Kerr closed forms to 4e-8). Charge moves the ISCO 0.6389 -> 0.5726 Rs.
import kn_disk as _kn
R_IN = float(_kn.isco(A_STAR * M, Q_STAR * M))


def omega_k(r):
    a, Q = A_STAR * M, Q_STAR * M
    xp = np if isinstance(r, (float, np.ndarray)) else __import__("cupy")
    s = xp.sqrt(xp.maximum(M * r - Q * Q, 0.0))
    return s / (r * r + a * s)


class Sim:
    def __init__(self, xp, nr, nphi, seed=1, force_amp=0.02, n_modes=60, r_in=None, r_out=None,
                 h_r=None, flare=0.0):
        self.xp = xp
        # 2026-10-01: grid edges per instance (default = the module's KN ISCO / 7.27 Rs);
        # the live v11 disk runs 2-20 Rs (scope.md 2026-10-01).
        self.r_in = R_IN if r_in is None else float(r_in)
        self.r_out = R_OUT if r_out is None else float(r_out)
        # 2026-10-01: thickness H/r = h_r * (r / r_in)^flare (default: constant H_R)
        self.h_r = H_R if h_r is None else float(h_r)
        self.flare = float(flare)
        self.nr, self.nphi = nr, nphi
        self.x0, self.x1 = np.log(self.r_in), np.log(self.r_out)
        self.dx = (self.x1 - self.x0) / nr
        self.dphi = 2 * np.pi / nphi
        xc = self.x0 + (np.arange(nr) + 0.5) * self.dx
        xf = self.x0 + np.arange(nr + 1) * self.dx
        self.r = xp.asarray(np.exp(xc))[:, None]          # cell centres (nr,1)
        self.rf = xp.asarray(np.exp(xf))[:, None]         # radial faces (nr+1,1)
        self.phi = xp.asarray((np.arange(nphi) + 0.5) * self.dphi)[None, :]
        r = self.r
        self.Om = omega_k(r)
        self.g = -r * self.Om ** 2
        self.cs2 = (self.h_r * (r / self.r_in) ** self.flare * r * self.Om) ** 2
        # initial state: S ~ r^-1/2 with a soft inner taper, circular, slight noise
        rng = np.random.default_rng(seed)
        S = r ** -0.5 * (1 - 0.7 * xp.exp(-(r - self.r_in) / (0.3 * self.r_in)))
        S = S * (1 + 0.01 * xp.asarray(rng.standard_normal((nr, nphi))))
        # pressure-supported equilibrium: w^2 = r(-g) + (r/S) dP/dr
        P = S * self.cs2
        dPdr = xp.gradient(P, axis=0) / (r * self.dx)
        w = xp.sqrt(xp.maximum(r * (-self.g) + r * dPdr / S, 0.25 * r * (-self.g)))
        self.S0 = xp.mean(S, axis=1, keepdims=True).copy()
        self.w0 = xp.mean(w, axis=1, keepdims=True).copy()
        self.U = [S, S * 0.0, S * r * w]                   # S, S u, J
        self.rng = rng
        self.force_amp = force_amp
        self.modes = []
        self.n_modes = n_modes
        self.t = 0.0
        self.nstep = 0
        self.pot = None

    # ---- helpers
    def prim(self, U):
        xp = self.xp
        S = xp.maximum(U[0], 1e-8)
        return S, U[1] / S, U[2] / (S * self.r)

    @staticmethod
    def minmod(a, b):
        return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

    def turb_potential(self):
        """Sum of active random modes (Laughlin et al. 2004 form)."""
        xp = self.xp
        # retire / spawn modes
        self.modes = [m for m in self.modes if self.t < m["t0"] + m["dt"]]
        while len(self.modes) < self.n_modes:
            # 2026-09-21: forcing reaches the inner edge (was 1.2 R_IN -- a cap
            # beside the hottest gas); outer limit stays inside the outer damping zone.
            rk = float(np.exp(self.rng.uniform(np.log(self.r_in), np.log(self.r_out * 0.85))))
            m = int(np.clip(np.round(np.exp(self.rng.uniform(0, np.log(40)))), 1, 40))
            Omk = float(omega_k(rk))
            self.modes.append(dict(rk=rk, sig=0.25 * rk * np.pi / m ** 0.5, m=m,
                                   ph0=self.rng.uniform(0, 2 * np.pi), Om=Omk,
                                   amp=self.rng.standard_normal(),
                                   t0=self.t - self.rng.uniform(0, 1) * 2 * np.pi / (m * Omk) * 0.1,
                                   dt=0.2 * 2 * np.pi / Omk * self.rng.uniform(0.5, 1.5)))
        pot = xp.zeros((self.nr, self.nphi))
        for md in self.modes:
            tl = (self.t - md["t0"]) / md["dt"]
            env = np.sin(np.pi * tl)
            pot = pot + (md["amp"] * env) * xp.exp(-((self.r - md["rk"]) / md["sig"]) ** 2) \
                * xp.cos(md["m"] * self.phi - md["ph0"] - md["Om"] * self.t)
        return self.force_amp * (self.r * self.Om) ** 2 * pot / np.sqrt(self.n_modes)

    def rhs(self, U, pot):
        xp = self.xp
        S, u, w = self.prim(U)
        r, rf, dx, dp = self.r, self.rf, self.dx, self.dphi
        P = S * self.cs2
        cs = xp.sqrt(self.cs2)
        q = [S, u, w]
        # ---------------- radial fluxes (faces between cells, ghost via edge copy)
        def pad_r(a):
            return xp.concatenate([a[:1], a, a[-1:]], axis=0)
        qp = [pad_r(v) for v in q]
        slopes = [self.minmod(v[1:-1] - v[:-2], v[2:] - v[1:-1]) for v in qp]
        # left/right states at interior faces i+1/2 (i = 0..nr-2) + boundary faces
        qL = [v[1:-1] + 0.5 * s for v, s in zip(qp, slopes)]      # right edge of cell i
        qR = [v[1:-1] - 0.5 * s for v, s in zip(qp, slopes)]      # left edge of cell i
        # face k between cell k-1 and k, k = 1..nr-1
        SL, uL, wL = [v[:-1] for v in qL]
        SR, uR, wR = [v[1:] for v in qR]
        rfi = rf[1:-1]
        cs2f = 0.5 * (self.cs2[:-1] + self.cs2[1:])
        def fr(Sx, ux, wx):
            Px = Sx * cs2f
            return [Sx * ux, Sx * ux * ux + Px, Sx * rfi * wx * ux]
        FL, FR = fr(SL, uL, wL), fr(SR, uR, wR)
        UL = [SL, SL * uL, SL * rfi * wL]; UR = [SR, SR * uR, SR * rfi * wR]
        smax = xp.maximum(xp.abs(uL), xp.abs(uR)) + xp.sqrt(cs2f)
        Fin = [0.5 * (a + b) - 0.5 * smax * (d - c) for a, b, c, d in zip(FL, FR, UL, UR)]
        # boundary faces: inner = free outflow into the hole (copy, no inflow back out)
        S0_, u0_, w0_ = S[:1], xp.minimum(u[:1], 0.0), w[:1]
        Fi0 = [S0_ * u0_, S0_ * u0_ * u0_ + S0_ * self.cs2[:1], S0_ * rf[:1] * w0_ * u0_]
        Sn_, un_, wn_ = S[-1:], u[-1:], w[-1:]
        Fin_n = [Sn_ * un_, Sn_ * un_ * un_ + Sn_ * self.cs2[-1:], Sn_ * rf[-1:] * wn_ * un_]
        Fr = [xp.concatenate([a, b, c], axis=0) for a, b, c in zip(Fi0, Fin, Fin_n)]   # (nr+1, nphi)
        # divergence (1/r^2) d/dx (r F)
        div_r = [(rf[1:] * f[1:] - rf[:-1] * f[:-1]) / (r * r * dx) for f in Fr]
        # ---------------- azimuthal fluxes (periodic)
        def roll(a, k):
            return xp.roll(a, k, axis=1)
        sl = [self.minmod(v - roll(v, 1), roll(v, -1) - v) for v in q]
        SLp, uLp, wLp = [v + 0.5 * s for v, s in zip(q, sl)]          # right edge of cell j
        SRp, uRp, wRp = [roll(v - 0.5 * s, -1) for v, s in zip(q, sl)]  # left edge of cell j+1
        def fp(Sx, ux, wx):
            Px = Sx * self.cs2
            return [Sx * wx, Sx * ux * wx, (Sx * wx * wx + Px)]
        GL, GR = fp(SLp, uLp, wLp), fp(SRp, uRp, wRp)
        VL = [SLp, SLp * uLp, SLp * r * wLp]; VR = [SRp, SRp * uRp, SRp * r * wRp]
        smp = xp.maximum(xp.abs(wLp), xp.abs(wRp)) + cs
        # Rusanov dissipation must be in the same form as each flux: mass and
        # radial momentum are divided by r*dphi below, J only by dphi, so J's
        # dissipation carries 1/r. (Missing it over-smoothed J up to ~7x vs S
        # and seeded a growing non-axisymmetric mode at r ~ 3-4 Rs.)
        dis = [smp, smp, smp / r]
        Gf = [0.5 * (a + b) - 0.5 * sv * (d - c) for a, b, c, d, sv in zip(GL, GR, VL, VR, dis)]
        # J-flux in phi is (S w^2 + P) per the conservative angular-momentum form;
        # mass & radial momentum use (1/r) d/dphi, J uses d/dphi (no 1/r).
        div_p = [(Gf[0] - roll(Gf[0], 1)) / (r * dp),
                 (Gf[1] - roll(Gf[1], 1)) / (r * dp),
                 (Gf[2] - roll(Gf[2], 1)) / dp]
        # sources: radial momentum gets (S w^2 + P)/r + S g - S dpot/dr; J gets -S dpot/dphi
        dpot_dr = xp.gradient(pot, axis=0) / (r * dx)
        dpot_dp = (roll(pot, -1) - roll(pot, 1)) / (2 * dp)
        src = [0.0, (S * w * w + P) / r + S * self.g - S * dpot_dr, -S * dpot_dp]
        return [-a - b + c for a, b, c in zip(div_r, div_p, src)]

    def damp(self, U, dt):
        """Relax toward the initial axisymmetric state near both radial edges."""
        xp = self.xp
        r = self.r
        # 2026-09-21: OUTER zone only. The inner damping zone (0.64-0.80 Rs)
        # calmed exactly the hottest gas; the inner boundary is a free fall-in
        # into the hole, which needs no wave damping.
        wo = self.r_out * 0.85
        f = xp.maximum(0.0, (r - wo) / (self.r_out - wo)) ** 2
        rate = f * self.Om / (2 * np.pi) * 10.0 * dt
        targets = [self.S0, 0.0 * self.S0, self.S0 * r * self.w0]
        return [u_ - rate * (u_ - tgt) for u_, tgt in zip(U, targets)]

    def step(self, cfl=0.3):
        xp = self.xp
        S, u, w = self.prim(self.U)
        cs = xp.sqrt(self.cs2)
        dt_r = float(xp.min(self.r * self.dx / (xp.abs(u) + cs)))
        dt_p = float(xp.min(self.r * self.dphi / (xp.abs(w) + cs)))
        dt = cfl * min(dt_r, dt_p)
        # the random modes live ~0.2 local orbits -- recomputing them every 8
        # steps (dt << that) changes nothing physically and saves ~8x the cost
        if self.pot is None or self.nstep % 8 == 0:
            self.pot = self.turb_potential()
        pot = self.pot
        self.nstep += 1
        k1 = self.rhs(self.U, pot)
        U1 = [a + dt * b for a, b in zip(self.U, k1)]
        U1[0] = xp.maximum(U1[0], 1e-8)
        k2 = self.rhs(U1, pot)
        self.U = [a + 0.5 * dt * (b + c) for a, b, c in zip(self.U, k1, k2)]
        self.U[0] = xp.maximum(self.U[0], 1e-8)
        self.U = self.damp(self.U, dt)
        self.t += dt
        return dt

    def contrast(self):
        xp = self.xp
        S = self.U[0]
        return S / xp.mean(S, axis=1, keepdims=True)


def run(xp, nr, nphi, t_spin, t_loop, n_frames, n_fade, out, report_every=2000, force_amp=0.03):
    sim = Sim(xp, nr, nphi, force_amp=force_amp)
    t0 = time.time(); n = 0
    while sim.t < t_spin:
        sim.step(); n += 1
        if n % report_every == 0:
            S, u, w = sim.prim(sim.U)
            print(f"spin-up t={sim.t:8.2f}/{t_spin}  steps {n}  wall {time.time()-t0:6.0f}s  "
                  f"contrast std {float(xp.std(sim.contrast())):.3f}  max|u|/cs {float(xp.max(xp.abs(u)/xp.sqrt(sim.cs2))):.2f}",
                  flush=True)
            if not bool(xp.isfinite(sim.U[0]).all()):
                raise SystemExit("NaN in spin-up")
    total = n_frames + n_fade
    dtf = t_loop / n_frames
    frames = np.zeros((total, nr, nphi), np.float16)
    tnext = sim.t
    for k in range(total):
        while sim.t < tnext:
            sim.step(); n += 1
        c = sim.contrast()
        frames[k] = (c.get() if hasattr(c, "get") else c).astype(np.float16)
        tnext += dtf
        if k % 30 == 0:
            print(f"frame {k}/{total}  t={sim.t:.2f}  wall {time.time()-t0:.0f}s", flush=True)
    # seamless loop: blend the overlap back onto the start
    loop = frames[:n_frames].astype(np.float32)
    for i in range(n_fade):
        # equal-power blend of the FLUCTUATIONS (contrast - 1): keeps mean and
        # variance steady through the seam (a linear blend dips mid-fade)
        x = (i + 0.5) / n_fade
        wa, wb = np.sin(0.5 * np.pi * x), np.cos(0.5 * np.pi * x)
        loop[i] = 1.0 + wa * (frames[i].astype(np.float32) - 1.0) \
                      + wb * (frames[n_frames + i].astype(np.float32) - 1.0)
    np.save(out, loop.astype(np.float16))
    print(f"saved {out} {loop.shape}; sim steps {n}, wall {time.time()-t0:.0f}s")
    return loop


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=64)
    ap.add_argument("--nphi", type=int, default=256)
    ap.add_argument("--t-spin", type=float, default=20.0)
    ap.add_argument("--t-loop", type=float, default=4.0)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--fade", type=int, default=4)
    ap.add_argument("--out", default="disk_sim_test.npy")
    a = ap.parse_args()
    run(np, a.nr, a.nphi, a.t_spin, a.t_loop, a.frames, a.fade, a.out)
