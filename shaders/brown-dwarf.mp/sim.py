"""brown_dwarf.mp's compute stage (scope.md 2026-10-11 "ONE ring, as a real 2D FLUID").
One broad ring as a thin 2D isothermal viscous fluid on a polar grid (compute_common.glsl),
Keplerian rotation removed exactly per row (phases computed here in double precision), moons
pushing it by gravity on analytic orbits. Fields (rgba32f): Sigma, u_r, v', unused.
Per substep: ADVECT (semi-Lagrangian along the deviations) -> FORCE (moons, pressure, Coriolis/
epicycles, viscosity, continuity). The image pass reads the grid per pixel.
"""

import importlib
import math
import os
import sys

import numpy as np

NR, NT = 1024, 4096
R_IN, R_OUT = 1.42, 2.38
SIM_RATE = 0.5
SUBSTEPS = 2
DT_MAX = 0.05

MOONS = [
    (1.47, 0.0020, 0.000, 0.4, 3.0e-5, 0.040),
    (2.35, 0.0060, 0.000, 4.1, 4.0e-5, 0.048),
]
N_ANALYTIC = len(MOONS)

ROS_N = 6
ROS_A = 1.92
ROS_GM = 8.0e-6
ROS_DRAW = (0.022, 0.017, 0.026, 0.019, 0.024, 0.018)
MUTUAL_BOOST = 250.0
STAB_G = 0.05
STAB_L = 0.02
STAB_E = 0.02
ROS_KICK = 0.05
ROS_SOFT = 0.02

POLAR = [
    (2.03, 165.0, 0.0,              +1.0, 0.034, 1.5e-5),
    (1.98, 197.8, 0.5 * math.pi,    -1.0, 0.028, 1.2e-5),
    (2.10, 169.5, math.pi,          -1.0, 0.040, 2.0e-5),
    (2.07, 192.5, 1.5 * math.pi,    +1.0, 0.030, 1.3e-5),
]
N_POLAR = len(POLAR)

CO_A, CO_D, CO_MU, CO_SPEED = 1.67, 0.065, 1.0e-3, 2.5
CO_PHI0 = (0.0, 50.0, 140.0, 250.0)
CO_SIDE = (-1.0, 1.0, -1.0, 1.0)
CO_DRAW = (0.026, 0.022, 0.030, 0.024)
CO_GM = 1.5e-5
N_CO = len(CO_PHI0)
CO_N = CO_A ** -1.5

N_MOONS = N_ANALYTIC + ROS_N + N_POLAR + N_CO

ROCK_SCREEN, ROCK_DEPTH = (-0.17, -0.40), 7.0
RK = dict(HX=0.06, HA=0.12, HB=0.09, HT=0.035, TX=-0.04, EX=-0.23, HR0=0.030, HR1=0.022, K=0.04, BUMP=0.007)
ROCK_SPIN = 2.2
ROCK_KICK = 0.02
ROCK_BOB = (0.05, 0.04, 0.035)
ROCK_BOB_W = (0.31, 0.29, 0.23)
FOV_, AIM_, ROLL_, D_ = 0.356, 0.80, 0.7854, 6.0

def _rock_home():
    el = 0.65
    ro = D_ * np.array([0.0, math.sin(el), -math.cos(el)])
    aim = AIM_ * np.array([0.0, 0.0, -1.0])
    fw = (aim - ro) / np.linalg.norm(aim - ro)
    rt = np.cross([0.0, 1.0, 0.0], fw); rt /= np.linalg.norm(rt)
    up = np.cross(fw, rt)
    cr, sr = math.cos(ROLL_), math.sin(ROLL_)
    rt, up = cr * rt + sr * up, -sr * rt + cr * up
    v = fw + ROCK_SCREEN[0] * 2 * FOV_ * rt + ROCK_SCREEN[1] * 2 * FOV_ * up
    return ro + ROCK_DEPTH * v / np.linalg.norm(v)

ROCK_HOME = _rock_home()

def _capsule(p, a, b, r):
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r

def racket_sdf(p):
    """= image.glsl racketSDF (shape frame; x handle -> head, z = head normal)."""
    k = RK
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = np.array([k["HA"], k["HB"], k["HT"]])
    q = np.stack([x - k["HX"], y, z], -1)
    head = (np.linalg.norm(q / r, axis=-1) - 1.0) * k["HT"]
    t = np.clip((x - k["TX"]) / (k["EX"] - k["TX"]), 0.0, 1.0)
    hx = np.clip(x, k["EX"], k["TX"])
    handle = np.sqrt((x - hx) ** 2 + y ** 2 + z ** 2) - (k["HR0"] + (k["HR1"] - k["HR0"]) * t)
    hh = np.clip(0.5 + 0.5 * (handle - head) / k["K"], 0.0, 1.0)
    d = handle + (head - handle) * hh - k["K"] * hh * (1.0 - hh)
    bump = (np.sin(37.0 * x + 1.0) * np.sin(41.0 * y + 2.0) * np.sin(33.0 * z + 0.5)
            + 0.5 * np.sin(83.0 * x + 0.3) * np.sin(71.0 * y + 1.1) * np.sin(97.0 * z + 2.2))
    return d + k["BUMP"] * bump

def _racket_mass():
    h = 0.002
    xs, ys, zs = np.arange(-0.27, 0.20, h), np.arange(-0.11, 0.11, h), np.arange(-0.05, 0.05, h)
    P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1)
    m = racket_sdf(P) < 0.0
    q = P[m]
    com = q.mean(0); q = q - com
    I = np.array([(q[:, 1] ** 2 + q[:, 2] ** 2).mean(), (q[:, 0] ** 2 + q[:, 2] ** 2).mean(), (q[:, 0] ** 2 + q[:, 1] ** 2).mean()])
    return com, I

ROCK_COM, ROCK_I = _racket_mass()

def rock_omega_dot(w):
    I1, I2, I3 = ROCK_I
    return np.array([(I2 - I3) / I1 * w[1] * w[2], (I3 - I1) / I2 * w[2] * w[0], (I1 - I2) / I3 * w[0] * w[1]])

def rock_step(s, dt):
    """RK4 on Euler's equations (body frame), then turn the orientation by the body-frame spin."""
    w = s["rw"]
    k1 = rock_omega_dot(w); k2 = rock_omega_dot(w + 0.5 * dt * k1)
    k3 = rock_omega_dot(w + 0.5 * dt * k2); k4 = rock_omega_dot(w + dt * k3)
    w = w + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    th = np.linalg.norm(w) * dt
    if th > 0:
        k = w / np.linalg.norm(w)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        s["rR"] = s["rR"] @ (np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K)
        u, _, vt = np.linalg.svd(s["rR"]); s["rR"] = u @ vt
    s["rw"] = w

def rock_uniforms(api, s):
    az = CAM_ORBIT * s["itime"]
    ca, sa = math.cos(az), math.sin(az)
    T = s["itime"]
    q = ROCK_HOME + np.array([ROCK_BOB[i] * math.sin(ROCK_BOB_W[i] * T + 1.7 * i) for i in range(3)])
    Ry = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
    api.render_uniform("u_rockPos", tuple(float(x) for x in Ry @ q))
    api.render_uniform("u_rockRot", tuple(float(x) for x in (Ry @ s["rR"]).T.flatten()))
    api.render_uniform("u_rockCom", tuple(float(x) for x in ROCK_COM))

def co_acc(p):
    th = (p[:, None] - p[None, :] + math.pi) % (2 * math.pi) - math.pi
    sn = np.abs(np.sin(th / 2)); np.fill_diagonal(sn, 1.0)
    f = -3 * CO_N * CO_N * CO_MU * np.sin(th) * (1 - 1 / (8 * sn ** 3)); np.fill_diagonal(f, 0.0)
    return f.sum(1)

def co_init():
    return np.radians(CO_PHI0), -1.5 * CO_N * np.array(CO_SIDE) * CO_D / CO_A

def co_step(s, h):
    h = h * CO_SPEED
    p, w = s["cp"], s["cw"]
    w = w + 0.5 * h * co_acc(p); p = p + h * w; w = w + 0.5 * h * co_acc(p)
    s["cp"], s["cw"] = p, w
ROW_R = R_IN + (np.arange(NR) + 0.5) * (R_OUT - R_IN) / NR
ROW_OM = ROW_R ** -1.5

ROS_SUM = sum(1.0 / (4.0 * math.sin(math.pi * k / ROS_N)) for k in range(1, ROS_N))
ROS_N_EQ = math.sqrt((1.0 + MUTUAL_BOOST * ROS_GM * ROS_SUM) / ROS_A ** 3)

def ros_slots(t):
    n = ROS_N_EQ
    ang = n * t + np.arange(ROS_N) * 2 * math.pi / ROS_N
    pos = np.stack([ROS_A * np.cos(ang), np.zeros(ROS_N), ROS_A * np.sin(ang)], 1)
    vel = np.stack([-ROS_A * n * np.sin(ang), np.zeros(ROS_N), ROS_A * n * np.cos(ang)], 1)
    return pos, vel

def ros_init():
    rng = np.random.default_rng(7)
    n = ROS_N_EQ
    ang = np.arange(ROS_N) * 2 * math.pi / ROS_N + rng.uniform(-ROS_KICK, ROS_KICK, ROS_N)
    pos = np.stack([ROS_A * np.cos(ang), np.zeros(ROS_N), ROS_A * np.sin(ang)], 1)
    vel = np.stack([-ROS_A * n * np.sin(ang), np.zeros(ROS_N), ROS_A * n * np.cos(ang)], 1)
    return pos, vel

def ros_acc(pos, vel, t):
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    acc = -pos / r ** 3
    d = pos[None, :, :] - pos[:, None, :]
    q = (d ** 2).sum(-1) + ROS_SOFT ** 2
    np.fill_diagonal(q, np.inf)
    acc += (MUTUAL_BOOST * ROS_GM * d / q[..., None] ** 1.5).sum(1)
    sp, _ = ros_slots(t)
    rr = np.linalg.norm(pos[:, [0, 2]], axis=1)
    er = np.stack([pos[:, 0] / rr, np.zeros_like(rr), pos[:, 2] / rr], 1)
    et = np.stack([-er[:, 2], np.zeros_like(rr), er[:, 0]], 1)
    dphi = np.arctan2(pos[:, 2], pos[:, 0]) - np.arctan2(sp[:, 2], sp[:, 0])
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
    a_t = ROS_A * (1.0 + STAB_G * dphi)
    L, L_t = rr * (vel * et).sum(1), np.sqrt(a_t * (1.0 + MUTUAL_BOOST * ROS_GM * ROS_SUM))
    acc += (-STAB_L * (L - L_t) / rr)[:, None] * et
    acc += (-STAB_E * (vel * er).sum(1))[:, None] * er
    acc[:, 1] += -STAB_E * vel[:, 1] - 0.01 * pos[:, 1]
    return acc

def ros_step(s, h):
    p, v, t = s["rp"], s["rv"], s["t"]
    a = ros_acc(p, v, t)
    v = v + 0.5 * h * a
    p = p + h * v
    a = ros_acc(p, v, t + h)
    v = v + 0.5 * h * a
    s["rp"], s["rv"] = p, v

def moon_positions(t, rp=None, itime=0.0, co=None):
    out = np.zeros((2, N_MOONS, 4), dtype="f4")
    for i, (a, e, inc, ph, gm, rad) in enumerate(MOONS):
        M = a ** -1.5 * t + ph
        r = a * (1.0 - e * math.cos(M))
        th = M + 2.0 * e * math.sin(M)
        out[0, i] = (r * math.cos(th), r * inc * math.sin(th - 0.7 * ph), r * math.sin(th), rad)
        out[1, i] = (gm, 0.3 * a * (gm / 3.0) ** (1.0 / 3.0) + 2e-3, 0.0, 0.0)
    if rp is not None:
        for k in range(ROS_N):
            out[0, N_ANALYTIC + k] = (*rp[k], ROS_DRAW[k])
            out[1, N_ANALYTIC + k] = (ROS_GM, 0.3 * ROS_A * (ROS_GM / 3.0) ** (1.0 / 3.0) + 2e-3, 0.0, 0.0)
    cam = CAM_ORBIT * itime - 0.5 * math.pi
    for k, (a, rel, ph, sgn, rad, gm) in enumerate(POLAR):
        node = cam + math.radians(rel)
        u = a ** -1.5 * t + ph
        i = N_ANALYTIC + ROS_N + k
        out[0, i] = (a * math.cos(u) * math.cos(node), sgn * a * math.sin(u), a * math.cos(u) * math.sin(node), rad)
        out[1, i] = (gm, 0.3 * a * (gm / 3.0) ** (1.0 / 3.0) + 2e-3, rad, 0.0)
    if co is not None:
        cp, cw = co
        for k in range(N_CO):
            r = CO_A - cw[k] * CO_A / (1.5 * CO_N)
            lam = CO_N * t + cp[k]
            i = N_ANALYTIC + ROS_N + N_POLAR + k
            out[0, i] = (r * math.cos(lam), 0.0, r * math.sin(lam), CO_DRAW[k])
            out[1, i] = (CO_GM, 0.3 * CO_A * (CO_GM / 3.0) ** (1.0 / 3.0) + 2e-3, 0.0, 0.0)
    return out

def phases(t, dt):
    p = np.zeros((1, NR, 4), dtype="f4")
    p[0, :, 0] = np.mod(ROW_OM * t, 2 * math.pi)
    p[0, :, 1] = np.mod(ROW_OM * (t - dt), 2 * math.pi)
    return p

N_BOLT = 6
N_CAND = 256
N_DECALS = 64
BOLT_MEAN_GAP = 1.6
BOLT_SIZES = ((0.70, 0.04, 0.10), (0.25, 0.10, 0.20), (0.05, 0.25, 0.40))
CAM_EL, CAM_ORBIT = 0.65, 0.006

def _bolt_fire(api, s):
    rng = s["rng"]
    t_est = s["itime"]
    az = CAM_ORBIT * t_est
    cam = np.array([math.cos(CAM_EL) * math.sin(az), math.sin(CAM_EL), -math.cos(CAM_EL) * math.cos(az)])
    c = rng.normal(size=(N_CAND, 3))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    c[c @ cam < 0.35] *= -1.0
    c = c[c @ cam > 0.35]
    cand = np.zeros((1, N_CAND, 4), dtype="f4")
    cand[0, :len(c), :3] = c
    s["cand"].write(cand.tobytes())
    api.run(s["k"]["DARK"], ((N_CAND + 64) // 64, 1, 1), images={0: s["cand"], 1: s["luma"]},
            uniforms={"u_nCand": len(c), "u_churn": float(s.get("churn", 0.0) % 10000.0)})
    out = np.frombuffer(s["luma"].read(), dtype="f4").reshape(N_CAND + 1, 4)
    itime = float(out[len(c), 0])
    k = int(np.argmin(out[:len(c), 3]))
    S = out[k, :3] / np.linalg.norm(out[k, :3])
    tang = rng.normal(size=3); tang -= S * (tang @ S); tang /= np.linalg.norm(tang)
    u = rng.random()
    for share, lo, hi in BOLT_SIZES:
        if u < share:
            break
        u -= share
    arc = rng.uniform(lo, hi)
    E = S * math.cos(arc) + tang * math.sin(arc)
    slot = s["bslot"]; s["bslot"] = (slot + 1) % N_BOLT
    b = s["bolts"]
    b[0, slot] = (*S, itime)
    b[1, slot] = (*E, float(rng.integers(N_DECALS)))
    b[2, slot] = (rng.uniform(0.6, 1.0), 0.0, 0.0, 0.0)
    s["bolt_tex"].write(b.tobytes())
    s["itime"] = itime

DBM_LIB = "../_lib/dbm_grid.comp"
DBM_CFG = {
    "spark": dict(sw=96, sh=32, nsx=6, nsy=4, mode=1, eta=2.5, grow_k=0.6, steps=6, relax=2),
    "tube": dict(sw=384, sh=64, nsx=1, nsy=4, mode=1, eta=2.5, grow_k=0.6, steps=10, relax=2),
}
DBM_CHECK = 8

def dbm_atlas(api, sw, sh, nsx, nsy, mode, eta, grow_k, steps, relax):
    defs = dict(DBM_SW=sw, DBM_SH=sh, DBM_NSX=nsx, DBM_NSY=nsy, DBM_MODE=mode, DBM_R0=2.5, DBM_ETA=eta,
                DBM_GROW_K=grow_k)
    ns = nsx * nsy
    d = dict(ns=ns, steps=steps, relax=relax,
             k={v: api.kernel(DBM_LIB, v, defs) for v in ("DBM_RESET", "DBM_HDRGEN", "DBM_RELAX", "DBM_STEP",
                                                         "DBM_GROW", "DBM_TRACE")},
             cells=api.texture2d((sw * nsx, sh * nsy), "rgba32f", nearest=True),
             hdr=api.texture2d((ns, 3), "rgba32f", nearest=True),
             want=api.texture2d((ns, 1), "rgba32f", nearest=True),
             groups=((sw * nsx + 15) // 16, (sh * nsy + 15) // 16, 1))
    d["hdr"].write(np.full((3, ns, 4), -5.0, dtype="f4").tobytes())
    return d

def dbm_tick(api, s, name, d):
    if s.get("n", 0) % DBM_CHECK == 0:
        h = np.frombuffer(d["hdr"].read(), dtype="f4").reshape(3, d["ns"], 4)
        w = np.frombuffer(d["want"].read(), dtype="f4").reshape(d["ns"], 4)
        s["dbm_idle"][name] = bool(np.all((h[0, :, 0] == w[:, 0]) & (h[0, :, 1] > 0.5) & (h[0, :, 3] > 0.5)))
    if s["dbm_idle"][name]:
        return
    K, im, sm = d["k"], {0: d["cells"], 1: d["hdr"]}, {"u_want": d["want"]}
    api.run(K["DBM_RESET"], d["groups"], images=im, samplers=sm)
    api.run(K["DBM_HDRGEN"], ((d["ns"] + 63) // 64, 1, 1), images=im, samplers=sm)
    for _ in range(d["steps"]):
        for _ in range(d["relax"]):
            for par in (0, 1):
                api.run(K["DBM_RELAX"], d["groups"], images=im, samplers=sm, uniforms={"u_parity": par})
        api.run(K["DBM_STEP"], (d["ns"], 1, 1), images=im, samplers=sm)
        api.run(K["DBM_GROW"], d["groups"], images=im, samplers=sm)
    api.run(K["DBM_TRACE"], ((d["ns"] + 63) // 64, 1, 1), images=im, samplers=sm)

A_R0, A_R1 = 1.40, 2.42
A_VT, A_VR, A_FT, A_FR = 512, 96, 2048, 384
A_PSI_ITERS = 6
A_STIR_RATE = 1.5
A_STIR_AMP = (6.0, 14.0)
A_STIR_R = (0.04, 0.10)
A_VROW = A_R0 + (np.arange(A_VR) + 0.5) * (A_R1 - A_R0) / A_VR
A_FROW = A_R0 + (np.arange(A_FR) + 0.5) * (A_R1 - A_R0) / A_FR

def a_phases(rows, t, h):
    p = np.zeros((1, len(rows), 4), dtype="f4")
    om = rows ** -1.5
    p[0, :, 0] = np.mod(om * t, 2 * math.pi)
    p[0, :, 1] = np.mod(om * (t - h), 2 * math.pi)
    return p

def aurora_setup(api, s):
    a = dict(k={n: api.kernel("aurora.comp", n) for n in ("A_INIT", "A_VORT", "A_PSI", "A_VEL", "A_GAS")},
             v=[api.texture2d((A_VT, A_VR), "rgba32f", nearest=True) for _ in range(2)],
             f=[api.texture2d((A_FT, A_FR), "rgba32f", nearest=True) for _ in range(2)],
             vph=api.texture2d((A_VR, 1), "rgba32f", nearest=True),
             fph=api.texture2d((A_FR, 1), "rgba32f", nearest=True),
             cur=0, t=0.0, gv=((A_VT + 15) // 16, (A_VR + 15) // 16, 1), gf=((A_FT + 15) // 16, (A_FR + 15) // 16, 1))
    a["vph"].write(a_phases(A_VROW, 0.0, 0.0).tobytes())
    a["fph"].write(a_phases(A_FROW, 0.0, 0.0).tobytes())
    api.run(a["k"]["A_INIT"], a["gf"], images={0: a["v"][0], 1: a["f"][0]})
    s["aur"] = a

def aurora_step(api, s, dt):
    a, rng = s["aur"], s["rng"]
    h = dt * SIM_RATE
    a["t"] += h
    a["vph"].write(a_phases(A_VROW, a["t"], h).tobytes())
    a["fph"].write(a_phases(A_FROW, a["t"], h).tobytes())
    inj = []
    for _ in range(rng.poisson(A_STIR_RATE * dt)):
        if len(inj) < 4:
            inj.append((rng.uniform(A_R0 + 0.1, A_R1 - 0.1), rng.uniform(0, 2 * math.pi),
                        float(rng.choice((-1, 1)) * rng.uniform(*A_STIR_AMP)), rng.uniform(*A_STIR_R)))
    u = {"u_h": float(h), "u_dtReal": float(dt), "u_simT": float(a["t"]), "u_nVort": len(inj),
         "u_vort": [tuple(float(x) for x in v) for v in inj] + [(0.0, 0.0, 0.0, 1.0)] * (4 - len(inj))}
    K, cur = a["k"], a["cur"]
    v0, v1, f0, f1 = a["v"][cur], a["v"][1 - cur], a["f"][cur], a["f"][1 - cur]
    ph = {"u_vPhase": a["vph"], "u_fPhase": a["fph"]}
    api.run(K["A_VORT"], a["gv"], images={0: v1}, samplers={"u_vSrc": v0, **ph}, uniforms=u)
    for _ in range(A_PSI_ITERS):
        for par in (0, 1):
            api.run(K["A_PSI"], a["gv"], images={0: v1}, samplers={"u_vSrc": v1, **ph}, uniforms={"u_parity": par})
    api.run(K["A_VEL"], a["gv"], images={0: v1}, samplers={"u_vSrc": v1, **ph})
    api.run(K["A_GAS"], a["gf"], images={0: f1},
            samplers={"u_vSrc": v1, "u_fSrc": f0, "u_moons": s["moons"], "u_notes": s["notes"], **ph}, uniforms=u)
    a["cur"] = 1 - cur

MT_RATE = 2.0
MT_MOON_BOOST = 400.0
MT_CAND = 48
MT_GOOD = 0.6
MT_REFINE_TOP = 6
MT_REFINE_N = 8
MT_REFINE_SIG = 0.008
MT_H = 0.02
MT_T = 9.0
MT_STEPS_PER_FRAME = 20
MT_WHIP = 1.35
MT_HIT_P = 0.25
MT_EVERY_BEATS = 8
MT_EVERY_S = 4.0
MT_MAX_ALIVE = 2
MT_NSLOT = 3
MT_HIST = 48
MT_TAIL_KEEP = 0.7
MT_ROCK_R = 0.12
MT_FLIGHT_H = 0.004
MT_CLOCK_EVERY = 10

def mt_camera(itime):
    el, az = 0.65, CAM_ORBIT * itime
    ro = D_ * np.array([math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)])
    aim = AIM_ * np.array([ro[0], 0.0, ro[2]]) / math.hypot(ro[0], ro[2])
    fw = (aim - ro) / np.linalg.norm(aim - ro)
    rt = np.cross([0.0, 1.0, 0.0], fw); rt /= np.linalg.norm(rt)
    up = np.cross(fw, rt)
    cr, sr = math.cos(ROLL_), math.sin(ROLL_)
    return ro, fw, cr * rt + sr * up, -sr * rt + cr * up

def mt_bodies(moons_now, t_sim, itime, dt_real):
    """Body centres (K, nb, 3), radii (nb), GM (nb) at dt_real (array, K) seconds after (t_sim, itime).
    Moons on circles at their Kepler rate; polar moons exact; the racket turns with the camera; dwarf at 0."""
    dt_real = np.asarray(dt_real, dtype=np.float64)
    p = moons_now[0, :, :3].astype(np.float64)
    rad = moons_now[0, :, 3].astype(np.float64)
    gm = moons_now[1, :, 0].astype(np.float64) * MT_MOON_BOOST
    r = np.maximum(np.hypot(p[:, 0], p[:, 2]), 1e-3)
    d = (r ** -1.5)[None, :] * SIM_RATE * dt_real[:, None]
    c, sn = np.cos(d), np.sin(d)
    P = np.empty((len(dt_real), len(rad), 3))
    P[..., 0] = p[None, :, 0] * c - p[None, :, 2] * sn
    P[..., 1] = p[None, :, 1]
    P[..., 2] = p[None, :, 0] * sn + p[None, :, 2] * c
    t = t_sim + dt_real * SIM_RATE
    cam = CAM_ORBIT * (itime + dt_real) - 0.5 * math.pi
    for k, (a, rel, ph, sgn, _, _) in enumerate(POLAR):
        node, u = cam + math.radians(rel), a ** -1.5 * t + ph
        i = N_ANALYTIC + ROS_N + k
        P[:, i, 0] = a * np.cos(u) * np.cos(node); P[:, i, 1] = sgn * a * np.sin(u); P[:, i, 2] = a * np.cos(u) * np.sin(node)
    az = CAM_ORBIT * (itime + dt_real)
    q = ROCK_HOME
    rock = np.stack([q[0] * np.cos(az) - q[2] * np.sin(az), np.full_like(az, q[1]), q[0] * np.sin(az) + q[2] * np.cos(az)], 1)
    allP = np.concatenate([np.zeros((len(dt_real), 1, 3)), P, rock[:, None, :]], 1)
    R = np.concatenate([[1.0], rad, [MT_ROCK_R]])
    G = np.concatenate([[1.0], gm, [0.0]])
    return allP, R, G

def mt_acc(x, P, G):
    """x (K, 3), P (K, nb, 3) -> gravity (K, 3)."""
    d = P - x[:, None, :]
    q = (d ** 2).sum(-1) + 1e-6
    return (G[None, :, None] * d / q[..., None] ** 1.5).sum(1)

def mt_clock(s):
    c = s.get("mt_clk")
    return c if c is not None else dict(it=s["itime"], bc=0.0, tb=0.5, conf=0.0, src=0.0, drop=-1.0, bar=0.0)

def mt_target_time(s, itime, earliest):
    """The moment the whip should land: the drop (analysed songs), else the first bar downbeat >= earliest."""
    c = mt_clock(s)
    if c["src"] > 0.5 and c["drop"] > 0.0:
        td = c["it"] + c["drop"]
        if earliest <= td <= earliest + 6.0:
            return td
    tb = c["tb"] if 0.2 < c["tb"] < 2.0 else 0.5
    if c["conf"] < 0.2:
        return earliest
    t0 = c["it"] - c["bar"] * 4.0 * tb
    k = math.ceil((earliest - t0) / (4.0 * tb))
    return t0 + k * 4.0 * tb

def mt_visible_targets(itime):
    """Bodies a near miss may be staged at: the 12 ring moons, the polar moons, the racket (index into mt_bodies)."""
    return list(range(1, 1 + N_MOONS)) + [1 + N_MOONS]

def mt_search_start(s, itime):
    """AIM BACKWARDS: stage a near miss (or hit) beside a chosen body at time T_n, run the meteor BACKWARD through
    the slingshot and the dwarf whip out to deep space (gravity is time-reversible), launch it forward from there.
    T_n is set so the whip lands on the target beat (one correction pass, as the whip time depends on T_n)."""
    rng = s["rng"]
    K = MT_CAND
    search_s = 2.0 * MT_T / MT_H / MT_STEPS_PER_FRAME / 45.0 + 0.3
    target = mt_target_time(s, itime, itime + search_s + 4.5)
    P0, R, G = mt_bodies(s["moons_np"], s["t"], itime, np.zeros(1))
    tgt = np.array(mt_visible_targets(itime))
    j = rng.choice(tgt, K)
    hit = rng.random(K) < MT_HIT_P
    gap = np.where(hit, -0.15, rng.uniform(0.25, 0.8, K))
    nd = rng.normal(size=(K, 3)); nd /= np.linalg.norm(nd, axis=1, keepdims=True)
    vinf = rng.uniform(0.7, 1.1, K)
    dirn = rng.normal(size=(K, 3))
    s["mt_search"] = dict(j=j, hit=hit, gap=gap, nd=nd, vinf=vinf, dirn=dirn, R=R, target=target,
                          Tn=np.full(K, target + 1.0), phase=1, itime0=itime, tsim0=s["t"],
                          moons0=s["moons_np"].copy())
    mt_back_init(s)

def mt_back_init(s):
    S = s["mt_search"]
    K = len(S["j"])
    P, R, G = mt_bodies(S["moons0"], S["tsim0"], S["itime0"], S["Tn"] - S["itime0"])
    M = P[np.arange(K), S["j"]]
    N = M + S["nd"] * (R[S["j"]] * (1.0 + S["gap"]))[:, None]
    eo = N / np.linalg.norm(N, axis=1, keepdims=True)
    d = eo + 0.9 * S["dirn"]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    into = -S["nd"]
    d = np.where((S["hit"] & ((d * into).sum(1) < 0.3))[:, None], (d + into) / np.linalg.norm(d + into, axis=1, keepdims=True), d)
    spd = np.sqrt(S["vinf"] ** 2 + 2.0 / np.linalg.norm(N, axis=1))
    S.update(x=N.copy(), w=-d * spd[:, None], tau=0.0, done=np.zeros(K, bool), rmin=np.full(K, 99.0),
             tperi=np.zeros(K), xperi=np.zeros((K, 3)), bad=np.zeros(K, bool), N=N, vN=d * spd[:, None],
             tend=np.zeros(K))

def mt_search_step(s):
    S = s["mt_search"]
    h = MT_H
    K = len(S["j"])
    for _ in range(MT_STEPS_PER_FRAME):
        run = ~S["done"] & ~S["bad"]
        if not run.any() or S["tau"] >= MT_T:
            break
        dtr = S["Tn"] - S["itime0"] - S["tau"] / MT_RATE
        P, R, G = mt_bodies(S["moons0"], S["tsim0"], S["itime0"], dtr)
        x, w = S["x"], S["w"]
        am = run[:, None]
        w += 0.5 * h * mt_acc(x, P, G) * am; x += h * w * am
        w += 0.5 * h * mt_acc(x, P, G) * am
        S["tau"] += h
        r = np.linalg.norm(x, axis=1)
        m = run & (r < S["rmin"])
        S["rmin"] = np.where(m, r, S["rmin"]); S["tperi"] = np.where(m, S["tau"], S["tperi"])
        S["xperi"] = np.where(m[:, None], x, S["xperi"])
        dd = np.linalg.norm(x[:, None, :] - P, axis=-1) - R[None]
        dd[np.arange(K), S["j"]] = np.where(S["tau"] < 0.15, 9.0, dd[np.arange(K), S["j"]])
        S["bad"] |= run & (dd.min(1) < 0.0)
        fin = run & (r > 8.5)
        S["tend"] = np.where(fin, S["tau"], S["tend"])
        S["done"] |= fin
    if not ((S["done"] | S["bad"]).all() or S["tau"] >= MT_T):
        return None
    ok = S["done"] & ~S["bad"] & (S["rmin"] > 1.02) & (S["rmin"] < MT_WHIP)
    ro = mt_camera(S["itime0"])[0]
    camdir = ro / np.linalg.norm(ro)
    ok &= (S["xperi"] * camdir[None]).sum(1) / np.maximum(S["rmin"], 1e-6) > -0.2
    if S["phase"] == 1:
        if not ok.any():
            s["mt_search"] = None
            return None
        keep = np.where(ok)[0]
        for k in ("j", "hit", "gap", "nd", "vinf", "dirn"):
            S[k] = S[k][keep]
        S["Tn"] = S["target"] + S["tperi"][keep] / MT_RATE
        S["phase"] = 2
        mt_back_init(s)
        return None
    s["mt_search"] = None
    if not ok.any():
        return None
    k = int(s["rng"].choice(np.where(ok)[0]))
    return dict(x=S["x"][k].copy(), w=-S["w"][k].copy(), start=float(S["Tn"][k] - S["tend"][k] / MT_RATE),
                hist=[], alive=False, done=False, hit_t=-100.0, hit_p=np.zeros(3), hit_r=0.0,
                plan_clear=float(S["gap"][k]), plan_hit=int(S["j"][k]) if S["hit"][k] else -1,
                target=S["target"], plan_peri=float(S["Tn"][k] - S["tperi"][k] / MT_RATE))

def mt_fly(s, m, itime, dt):
    """Live flight of one meteor over this frame (real gravity, adaptive leapfrog, real crashes)."""
    if not m["alive"]:
        if itime < m["start"]:
            return
        m["alive"] = True
    tau = dt * MT_RATE
    P, R, G = mt_bodies(s["moons_np"], s["t"], s["itime"], np.zeros(1))
    P, x, w = P[0], m["x"], m["w"]
    done = 0.0
    while done < tau:
        d = np.linalg.norm(P - x[None], axis=1) - R
        h = min(MT_FLIGHT_H, max(0.0005, 0.05 * float(d.min()) / max(float(np.linalg.norm(w)), 1e-3)), tau - done)
        a = mt_acc(x[None], P[None], G)[0]
        w = w + 0.5 * h * a; x = x + h * w
        a = mt_acc(x[None], P[None], G)[0]
        w = w + 0.5 * h * a
        done += h
        d = np.linalg.norm(P - x[None], axis=1) - R
        if d.min() < 0.0:
            j = int(d.argmin())
            m["hit_t"], m["hit_r"] = itime, float(R[j])
            m["hit_p"] = P[j] + (x - P[j]) / max(np.linalg.norm(x - P[j]), 1e-6) * R[j]
            m["alive"], m["done"] = False, True
            break
    m["x"], m["w"] = x, w
    if np.linalg.norm(x) > 14.0:
        m["alive"], m["done"] = False, True
    if m["alive"] or m["done"]:
        m["hist"].insert(0, (*x, itime))
        m["hist"] = [p for p in m["hist"] if itime - p[3] < MT_TAIL_KEEP][:MT_HIST]

def meteor_setup(api, s):
    s["k"]["CLOCK"] = api.kernel("sim.comp", "CLOCK")
    s["clk_tex"] = api.texture2d((2, 1), "rgba32f", nearest=True)
    s["met_tex"] = api.texture2d((MT_HIST + 2, MT_NSLOT), "rgba32f", nearest=True)
    z = np.zeros((MT_NSLOT, MT_HIST + 2, 4), dtype="f4"); z[:, MT_HIST, 1] = -100.0
    s["met_tex"].write(z.tobytes())
    s["mt_search"], s["mt_list"], s["mt_clk"], s["mt_next"] = None, [], None, 0.0

def meteor_step(api, s, dt):
    itime = s["itime"]
    if s.get("n", 0) % MT_CLOCK_EVERY == 0:
        api.run(s["k"]["CLOCK"], (1, 1, 1), images={0: s["clk_tex"]})
        c = np.frombuffer(s["clk_tex"].read(), dtype="f4").reshape(2, 4)
        if np.all(np.isfinite(c)):
            s["mt_clk"] = dict(it=float(c[0, 0]), bc=float(c[0, 1]), tb=float(c[0, 2]), conf=float(c[0, 3]),
                               src=float(c[1, 0]), drop=float(c[1, 1]), bar=float(c[1, 2]) % 1.0)
            s["itime"] = itime = float(c[0, 0])
    live = [m for m in s["mt_list"] if not m["done"] or itime - m["hit_t"] < 1.5]
    s["mt_list"] = live[-MT_NSLOT:]
    if s["mt_search"] is not None:
        m = mt_search_step(s)
        if m is not None:
            s["mt_list"].append(m)
            s["mt_list"] = s["mt_list"][-MT_NSLOT:]
            print(f"meteor: planned at {itime:.1f}: start {m['start']:.2f} whip {m['plan_peri']:.2f} (target {m['target']:.2f}) "
                  f"{'HIT body ' + str(m['plan_hit']) if m['plan_hit'] >= 0 else 'miss gap %.2f' % m['plan_clear']}", file=sys.stderr)
        elif s["mt_search"] is None:
            s["mt_next"] = itime + 0.3
            s["mt_fails"] = s.get("mt_fails", 0) + 1
    elif itime >= s["mt_next"] and sum(1 for m in s["mt_list"] if not m["done"]) < MT_MAX_ALIVE:
        c = mt_clock(s)
        gap = MT_EVERY_BEATS * c["tb"] if c["conf"] >= 0.2 and 0.2 < c["tb"] < 2.0 else MT_EVERY_S
        s["mt_next"] = itime + gap
        mt_search_start(s, itime)
    for m in s["mt_list"]:
        if not m["done"]:
            was = (m["alive"], m["done"])
            mt_fly(s, m, itime, dt)
            if (m["alive"], m["done"]) != was:
                print(f"meteor: {'launched' if m['alive'] else ('HIT r %.3f' % m['hit_r'] if m['hit_t'] > 0 else 'left')} at {itime:.2f}"
                      f" (failed searches so far {s.get('mt_fails', 0)})", file=sys.stderr)
    tex = np.zeros((MT_NSLOT, MT_HIST + 2, 4), dtype="f4")
    tex[:, MT_HIST, 1] = -100.0
    for i, m in enumerate(s["mt_list"][:MT_NSLOT]):
        for k, p in enumerate(m["hist"][:MT_HIST]):
            tex[i, k] = p
        tex[i, MT_HIST] = (1.0 if m["alive"] else 0.0, m["hit_t"], len(m["hist"]), float(np.linalg.norm(m["w"])))
        tex[i, MT_HIST + 1] = (*m["hit_p"], m["hit_r"])
    s["met_tex"].write(tex.tobytes())

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_lib"))
import hand
importlib.reload(hand)
BOOP_DV_MOON = 0.30
BOOP_DV_MET = 0.8
BOOP_SPIN = 5.0
BOOP_PICK_ANG = 0.03
BOOP_DWARF_BIAS = 0.01
SPLAT_V = 0.06
SPLAT_R = 0.08
SPLAT_T = 0.15
FREE_GONE_R = 6.0

def free_apply(s, arr):
    """Override the moons array with the free (booped) moons: their own position, or hidden + weightless if gone."""
    for i, f in s.get("free", {}).items():
        if f["gone"]:
            arr[0, i] = (0.0, -60.0, 0.0, 0.0)
            arr[1, i, 0] = 0.0
        else:
            arr[0, i, :3] = f["x"]
    return arr

def free_step(s, h):
    for i, f in s.get("free", {}).items():
        if f["gone"]:
            continue
        x, v = f["x"], f["v"]
        v = v - 0.5 * h * x / np.linalg.norm(x) ** 3
        x = x + h * v
        v = v - 0.5 * h * x / np.linalg.norm(x) ** 3
        f["x"], f["v"] = x, v
        r = float(np.linalg.norm(x))
        if r < 1.0 + f["rad"] or r > FREE_GONE_R:
            f["gone"] = True
            print(f"hand: moon {i} {'crashed into the dwarf' if r < 2.0 else 'flew off'} "
                  f"({sum(1 for g in s['free'].values() if g['gone'])} gone, {len(s['free'])} loose)", file=sys.stderr)

def bd_rock_pos(s):
    az = CAM_ORBIT * s["itime"]
    ca, sa = math.cos(az), math.sin(az)
    T = s["itime"]
    q = ROCK_HOME + np.array([ROCK_BOB[i] * math.sin(ROCK_BOB_W[i] * T + 1.7 * i) for i in range(3)])
    return np.array([q[0] * ca - q[2] * sa, q[1], q[0] * sa + q[2] * ca])

def bd_boop(api, s, xy):
    ro, fw, rt, up = mt_camera(s["itime"])
    ro, rd = hand.ray(xy[0], xy[1], hand.canvas(api), ro, fw, rt, up, FOV_)
    m = s["moons_np"]
    cents = [m[0, i, :3] for i in range(N_MOONS)] + [bd_rock_pos(s), np.zeros(3)]
    radii = [float(m[0, i, 3]) for i in range(N_MOONS)] + [0.2, 1.0]
    mets = [mm for mm in s.get("mt_list", []) if mm["alive"]]
    cents += [mm["x"] for mm in mets]
    radii += [0.03] * len(mets)
    cents = [np.asarray(c, dtype=np.float64) for c in cents]
    C, Rr = np.array(cents), np.array(radii)
    rel = C - ro[None]
    dist = np.linalg.norm(rel, axis=1)
    ang = np.arccos(np.clip((rel @ rd) / np.maximum(dist, 1e-9), -1.0, 1.0)) - np.arcsin(np.clip(Rr / dist, 0.0, 1.0))
    ang[N_MOONS + 1] += BOOP_DWARF_BIAS
    i = int(np.argmin(ang))
    print(f"hand: boop at canvas px ({xy[0]:.0f}, {xy[1]:.0f}) -> nearest body {i} "
          f"{max(ang[i], 0.0) * 3385:.0f} px away", file=sys.stderr)
    if ang[i] > BOOP_PICK_ANG:
        i = None
    if i is None or i == N_MOONS + 1:
        if i == N_MOONS + 1:
            _bolt_fire(api, s)
            print("hand: booped the dwarf -> lightning", file=sys.stderr)
            return
        if abs(rd[1]) > 1e-4:
            tr = -ro[1] / rd[1]
            P = ro + rd * tr
            if tr > 0 and R_IN < math.hypot(P[0], P[2]) < R_OUT:
                s["splat"] = (float(P[0]), float(P[2]), s["itime"])
                print("hand: splash in the ring", file=sys.stderr)
        return
    if i < N_MOONS:
        f = s.setdefault("free", {}).get(i)
        if f is None:
            x = m[0, i, :3].astype(np.float64)
            if N_ANALYTIC + ROS_N <= i < N_ANALYTIC + ROS_N + N_POLAR:
                dtt = 1e-3
                m2 = moon_positions(s["t"] + dtt, s["rp"], s["itime"], (s["cp"], s["cw"]))
                v = (m2[0, i, :3].astype(np.float64) - x) / dtt
            else:
                r = math.hypot(x[0], x[2])
                v = math.sqrt(1.0 / r) * np.array([-x[2], 0.0, x[0]]) / r
            f = s["free"][i] = dict(x=x, v=v, gone=False, rad=float(m[0, i, 3]))
        if not f["gone"]:
            f["v"] = f["v"] + rd * BOOP_DV_MOON
            print(f"hand: booped moon {i} loose ({len(s['free'])} loose)", file=sys.stderr)
    elif i == N_MOONS:
        s["rw"] = s["rw"] + s["rng"].normal(size=3) / math.sqrt(3.0) * BOOP_SPIN
        print("hand: booped the racket (spin kick)", file=sys.stderr)
    else:
        mm = mets[i - N_MOONS - 2]
        mm["w"] = mm["w"] + rd * BOOP_DV_MET
        print("hand: booped a meteor", file=sys.stderr)

def bd_reset(api, s):
    s["rp"], s["rv"] = ros_init()
    s["cp"], s["cw"] = co_init()
    s["free"] = {}
    s["cur"] = 0
    api.run(s["k"]["INIT"], s["groups"], images={0: s["f"][0]})
    s["rR"], s["rw"] = np.eye(3), np.array([ROCK_KICK, ROCK_SPIN, ROCK_KICK * 0.5])
    s["mt_list"], s["mt_search"] = [], None
    a = s["aur"]
    api.run(a["k"]["A_INIT"], a["gf"], images={0: a["v"][0], 1: a["f"][0]})
    a["cur"] = 0
    print("hand: RESET -- everything back in formation", file=sys.stderr)

CHURN_AMP = 0.045

def churn_ease(b):
    f = b - math.floor(b)
    return math.floor(b) + 1.0 - (1.0 - f) ** 4

def churn_step(api, s):
    c = s.get("mt_clk")
    if c is not None and c["conf"] > 0.2 and 0.2 < c["tb"] < 2.0:
        bn = c["bc"] + (s["itime"] - c["it"]) / c["tb"]
        prev = s.get("churn_raw")
        if prev is not None:
            d = bn - prev
            if d < -512.0:
                d += 1024.0
            if 0.0 < d < 2.0:
                bu = s.get("churn_bu", 0.0)
                s["churn"] = s.get("churn", 0.0) + CHURN_AMP * (churn_ease(bu + d) - churn_ease(bu))
                s["churn_bu"] = (bu + d) % 4096.0
        s["churn_raw"] = bn
    api.render_uniform("u_churn", float(s.get("churn", 0.0) % 10000.0))

def setup(api):
    s = api.state
    s["f"] = [api.texture2d((NT, NR), "rgba32f", nearest=True) for _ in range(2)]
    s["moons"] = api.texture2d((N_MOONS, 2), "rgba32f", nearest=True)
    s["phase"] = api.texture2d((NR, 1), "rgba32f", nearest=True)
    s["t"] = 0.0
    s["rp"], s["rv"] = ros_init()
    s["cp"], s["cw"] = co_init()
    s["rR"], s["rw"] = np.eye(3), np.array([ROCK_KICK, ROCK_SPIN, ROCK_KICK * 0.5])
    s["cur"] = 0
    s["k"] = {n: api.kernel("sim.comp", n) for n in ("INIT", "ADVECT", "FORCE", "DARK", "NOTES")}
    s["k"]["WANT"] = api.kernel("sim.comp", "WANT")
    s["notes"] = api.texture2d((13, 3), "rgba32f", nearest=True)
    nt = np.zeros((3, 13, 4), dtype="f4"); nt[0, :, 2] = -100.0
    nt[2, 0] = (1.0, 1.0, 1.0, np.nan)
    nt[2, 1] = (-1.0, 0.0, 0.0, 0.0)
    s["notes"].write(nt.tobytes())
    s["spokes"] = api.texture2d((32, 2), "rgba32f", nearest=True)
    sp = np.zeros((2, 32, 4), dtype="f4"); sp[0, :, 2] = -100.0
    s["spokes"].write(sp.tobytes())
    s["dbm"] = {name: dbm_atlas(api, **cfg) for name, cfg in DBM_CFG.items()}
    s["dbm_idle"] = {name: False for name in DBM_CFG}
    s["cand"] = api.texture2d((N_CAND, 1), "rgba32f", nearest=True)
    s["luma"] = api.texture2d((N_CAND + 1, 1), "rgba32f", nearest=True)
    s["bolts"] = np.zeros((3, N_BOLT, 4), dtype="f4")
    s["bolts"][0, :, 3] = -100.0
    s["bolt_tex"] = api.texture2d((N_BOLT, 3), "rgba32f", nearest=True)
    s["bolt_tex"].write(s["bolts"].tobytes())
    s["bslot"], s["itime"], s["rng"] = 0, 0.0, np.random.default_rng()
    s["next_bolt"] = 2.0
    s["groups"] = ((NT + 15) // 16, (NR + 15) // 16, 1)
    api.run(s["k"]["INIT"], s["groups"], images={0: s["f"][0]})
    s["moons_np"] = moon_positions(0.0, s["rp"], 0.0, (s["cp"], s["cw"]))
    s["free"], s["splat"] = {}, (0.0, 0.0, -100.0)
    s["moons"].write(s["moons_np"].tobytes())
    s["phase"].write(phases(0.0, 0.0).tobytes())
    aurora_setup(api, s)
    meteor_setup(api, s)
    _expose(api, s)

def _expose(api, s):
    api.expose("u_boltSlots", s["bolt_tex"])
    api.expose("u_ring", s["f"][s["cur"]])
    api.expose("u_moons", s["moons"])
    api.expose("u_phase", s["phase"])
    api.expose("u_notes", s["notes"])
    api.expose("u_spokes", s["spokes"])
    if "met_tex" in s:
        api.expose("u_met", s["met_tex"])
    if "aur" in s:
        api.expose("u_aurF", s["aur"]["f"][s["aur"]["cur"]])
        api.expose("u_aurPhase", s["aur"]["fph"])
    for name, d in s["dbm"].items():
        api.expose(f"u_{name}", d["cells"])
        api.expose(f"u_{name}Hdr", d["hdr"])

def step(api, dt):
    s = api.state
    dt = min(max(dt, 1e-4), DT_MAX)
    for kind, xy in hand.events(api, s):
        if kind == "reset":
            bd_reset(api, s)
        else:
            bd_boop(api, s, xy)
    h = dt * SIM_RATE / SUBSTEPS
    profr = s.get("n", 0) % 600 == 5 and getattr(api, "ctx", None) is not None
    if profr:
        qr = api.ctx.query(time=True)
        qr.__enter__()
    for _ in range(SUBSTEPS):
        ros_step(s, h)
        co_step(s, h)
        s["t"] += h
        free_step(s, h)
        s["moons_np"] = free_apply(s, moon_positions(s["t"], s["rp"], s["itime"], (s["cp"], s["cw"])))
        s["moons"].write(s["moons_np"].tobytes())
        s["phase"].write(phases(s["t"], h).tobytes())
        a, b = s["f"][s["cur"]], s["f"][1 - s["cur"]]
        api.run(s["k"]["ADVECT"], s["groups"], images={0: b}, samplers={"u_src": a, "u_phase": s["phase"]},
                uniforms={"u_h": h})
        api.run(s["k"]["FORCE"], s["groups"], images={0: a},
                samplers={"u_src": b, "u_phase": s["phase"], "u_moons": s["moons"]},
                uniforms={"u_h": h, "u_nMoons": N_MOONS,
                          "u_splat": (s["splat"][0], s["splat"][1], SPLAT_V / SPLAT_T / SIM_RATE,
                                      1.0 if s["itime"] - s["splat"][2] < SPLAT_T else 0.0)})
    if profr:
        qr.__exit__(None, None, None)
        print(f"profile (GPU ms): ring sim {qr.elapsed / 1e6:.2f} ({SUBSTEPS} substeps, {N_MOONS} moons)", file=sys.stderr)
    api.render_uniform("u_simT", float(s["t"] % 10000.0))
    api.run(s["k"]["NOTES"], (1, 1, 1), images={0: s["notes"], 1: s["spokes"]}, uniforms={"u_dt": float(dt)})
    api.run(s["k"]["WANT"], (1, 1, 1), images={0: s["dbm"]["spark"]["want"], 1: s["dbm"]["tube"]["want"]},
            samplers={"u_notes": s["notes"]})
    prof = s.get("n", 0) % 600 == 5 and getattr(api, "ctx", None) is not None
    tq = {}
    for name, d in s["dbm"].items():
        if prof:
            q = api.ctx.query(time=True)
            with q:
                dbm_tick(api, s, name, d)
            tq["dbm_" + name] = q.elapsed / 1e6
        else:
            dbm_tick(api, s, name, d)
    if prof:
        q = api.ctx.query(time=True)
        with q:
            aurora_step(api, s, dt)
        tq["aurora"] = q.elapsed / 1e6
        print("profile (GPU ms): " + ", ".join(f"{k} {v:.2f}" for k, v in tq.items())
              + f" | dbm idle {s['dbm_idle']}", file=sys.stderr)
    else:
        aurora_step(api, s, dt)
    meteor_step(api, s, dt)
    churn_step(api, s)
    for _ in range(4):
        rock_step(s, dt / 4)
    rock_uniforms(api, s)
    s["itime"] += dt
    s["next_bolt"] -= dt
    if s["next_bolt"] <= 0.0:
        _bolt_fire(api, s)
        s["next_bolt"] = s["rng"].exponential(BOLT_MEAN_GAP) + 0.15
    s["n"] = s.get("n", 0) + 1
    if s["n"] % 1200 == 600:
        nt = np.frombuffer(s["notes"].read(), dtype="f4").reshape(3, 13, 4)
        print(f"notes: moon shocks {int(nt[1, :12, 0].sum())} (per pc {nt[1, :12, 0].astype(int).tolist()}), "
              f"big strikes {int(nt[1, 12, 0])}, spokes {int(nt[2, 1, 1])}", file=sys.stderr)
        af = np.frombuffer(s["aur"]["f"][s["aur"]["cur"]].read(), dtype="f4").reshape(A_FR, A_FT, 4)
        av = np.frombuffer(s["aur"]["v"][s["aur"]["cur"]].read(), dtype="f4").reshape(A_VR, A_VT, 4)
        print(f"aurora: gas mean {np.nanmean(af[..., 0]):.3f} max {np.nanmax(af[..., 0]):.2f} | green max "
              f"{np.nanmax(af[..., 1]):.3f} red max {np.nanmax(af[..., 2]):.3f} | vort |max| {np.nanmax(np.abs(av[..., 0])):.2f} "
              f"nan v {int(np.isnan(av).sum())} f {int(np.isnan(af).sum())} psi |max| {np.nanmax(np.abs(av[..., 1])):.2e} |u| p99 {np.nanpercentile(np.abs(av[..., 2:4]), 99):.2e} max {np.nanmax(np.abs(av[..., 2:4])):.2e} at row {int(np.unravel_index(np.nanargmax(np.abs(av[..., 2])), av.shape[:2])[0])}", file=sys.stderr)
    if s["n"] % 1800 == 1:
        a = np.frombuffer(s["f"][s["cur"]].read(), dtype="f4").reshape(NR, NT, 4)
        S, ur, vp = a[..., 0], a[..., 1], a[..., 2]
        bad = ~np.isfinite(a).all(axis=-1)
        j = int(np.nanargmax(np.abs(np.nan_to_num(ur)).max(axis=1)))
        print(f"ring t={s['t']:.1f} S mean {np.nanmean(S):.3f} max {np.nanmax(S):.2f} |ur| max {np.nanmax(np.abs(ur)):.2e} "
              f"(r {ROW_R[j]:.3f}) |vp| max {np.nanmax(np.abs(vp)):.2e} nonfinite {int(bad.sum())}", file=sys.stderr)
    _expose(api, s)
