#!/usr/bin/env python3
"""
Kerr-Newman thin-disk tables for the orrery (units Rs = 1, M = 0.5).
scope.md 2026-09-21 "FULL REDO: Kerr-Newman".

For prograde equatorial circular geodesics of NEUTRAL matter in Kerr-Newman:
  Omega(r) = sqrt(M r - Q^2) / (r^2 + a sqrt(M r - Q^2))
  u^t, E, L from the equatorial metric; ISCO = minimum of E(r).
Flux: the general Page & Thorne (1974) integral
  F(r) = -Omega_r / (E - Omega L)^2 / sqrt(-g) * int_isco^r (E - Omega L) L_r dr
(sqrt(-g) = r for the reduced equatorial metric; shape only, peak = 1).
At Q = 0 this must reproduce the closed-form Kerr Page-Thorne flux.

Writes textures/kerr_v7/kn_disk_table.npy: (1, N, 4) float32 over log-spaced r
from the ISCO to r_out: [Omega, u^t, F/Fmax, r] -- one fetch per lookup.
"""
import numpy as np

M = 0.5


def circ(r, a, Q):
    s = np.sqrt(np.maximum(M * r - Q * Q, 0.0))
    Om = s / (r * r + a * s)
    D = r * r - 2 * M * r + a * a + Q * Q
    gtt = -(1 - (2 * M * r - Q * Q) / (r * r))
    gtp = -a * (2 * M * r - Q * Q) / (r * r)
    gpp = ((r * r + a * a) ** 2 - a * a * D) / (r * r)
    ut = 1 / np.sqrt(-(gtt + 2 * gtp * Om + gpp * Om * Om))
    E = -(gtt + gtp * Om) * ut
    L = (gtp + gpp * Om) * ut
    return Om, ut, E, L


def isco(a, Q):
    r = np.geomspace(M * 1.0001 + 1e-9, 20 * M, 2_000_001)
    with np.errstate(all="ignore"):
        _, _, E, _ = circ(r, a, Q)
    ok = np.isfinite(E)
    i = np.argmin(E[ok])
    # refine by golden section
    lo, hi = r[ok][max(i - 2, 0)], r[ok][min(i + 2, ok.sum() - 1)]
    for _ in range(200):
        m1 = lo + 0.382 * (hi - lo); m2 = lo + 0.618 * (hi - lo)
        if circ(m1, a, Q)[2] < circ(m2, a, Q)[2]:
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def flux(r, a, Q, rin):
    """Page-Thorne general integral on a fine grid from rin to max(r)."""
    g = np.geomspace(rin, r.max(), 400001)
    Om, ut, E, L = circ(g, a, Q)
    dOm = np.gradient(Om, g); dL = np.gradient(L, g)
    integrand = (E - Om * L) * dL
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(g))])
    # sqrt(-g) of the reduced equatorial metric (Page & Thorne / Harko et al.)
    # is r -- with r^2 the Q = 0 check against the closed form was off by 19%;
    # with r it matches to 4e-8.
    F = -dOm / (E - Om * L) ** 2 / g * cum
    return np.interp(r, g, F)


def kerr_pt_closed(r, a_star):
    """Closed-form Kerr Page-Thorne (the shader's current ntFlux), r in Rs."""
    x = np.sqrt(r / M)
    z1 = 1 + (1 - a_star ** 2) ** (1 / 3) * ((1 + a_star) ** (1 / 3) + (1 - a_star) ** (1 / 3))
    z2 = np.sqrt(3 * a_star ** 2 + z1 * z1)
    x0 = np.sqrt(3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2)))
    th = np.arccos(a_star) / 3
    x1 = 2 * np.cos(th - np.pi / 3); x2 = 2 * np.cos(th + np.pi / 3); x3 = -2 * np.cos(th)
    c1 = 3 * (x1 - a_star) ** 2 / (x1 * (x1 - x2) * (x1 - x3))
    c2 = 3 * (x2 - a_star) ** 2 / (x2 * (x2 - x1) * (x2 - x3))
    c3 = 3 * (x3 - a_star) ** 2 / (x3 * (x3 - x1) * (x3 - x2))
    br = (x - x0 - 1.5 * a_star * np.log(x / x0) - c1 * np.log((x - x1) / (x0 - x1))
          - c2 * np.log((x - x2) / (x0 - x2)) - c3 * np.log((x - x3) / (x0 - x3)))
    return br / (x ** 4 * (x ** 3 - 3 * x + 2 * a_star))


if __name__ == "__main__":
    import os
    a_star, Q_star = 0.997, 0.07
    a, Q = a_star * M, Q_star * M
    r_out = float(np.sqrt((0.35 / 0.048) ** 2 - a * a))
    # --- validation at Q = 0 against the closed form
    r0 = isco(a, 0.0)
    rr = np.geomspace(r0 * 1.001, r_out, 400)
    Fn = flux(rr, a, 0.0, r0); Fc = kerr_pt_closed(rr, a_star)
    Fn /= Fn.max(); Fc /= Fc.max()
    print(f"Q=0: ISCO {r0:.6f} Rs (closed form 0.638891); flux shape max |diff| {np.abs(Fn - Fc).max():.2e}")
    # --- Kerr-Newman table
    ri = isco(a, Q)
    rr = np.geomspace(ri, r_out, 1024)
    Om, ut, E, L = circ(rr, a, Q)
    F = flux(rr, a, Q, ri); F[0] = 0.0
    Fmax = F.max(); rpk = rr[np.argmax(F)]
    print(f"Q*={Q_star}: ISCO {ri:.6f} Rs, flux peak at r = {rpk:.4f} Rs, orbital period there "
          f"{2 * np.pi / np.interp(rpk, rr, Om):.4f} Rs/c")
    tab = np.stack([Om, ut, F / Fmax, rr], -1).astype(np.float32)[None]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures/kerr_v7/kn_disk_table.npy")
    np.save(out, tab)
    print("wrote", out, tab.shape)
