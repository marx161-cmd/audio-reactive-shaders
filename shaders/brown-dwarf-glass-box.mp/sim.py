"""brown_dwarf.mp's compute stage (scope.md 2026-10-11 "ONE ring, as a real 2D FLUID").
One broad ring as a thin 2D isothermal viscous fluid on a polar grid (compute_common.glsl),
Keplerian rotation removed exactly per row (phases computed here in double precision), moons
pushing it by gravity on analytic orbits. Fields (rgba32f): Sigma, u_r, v', unused.
Per substep: ADVECT (semi-Lagrangian along the deviations) -> FORCE (moons, pressure, Coriolis/
epicycles, viscosity, continuity). The image pass reads the grid per pixel.
"""

import importlib
import json
import math
import os
import pickle
import subprocess
import sys
import time

import moderngl
import numpy as np

NR, NT = 1024, 4096
DW_R = 0.1
R_IN, R_OUT = 0.25, 3.0
MET_RS = DW_R + 0.03
SIM_RATE = 0.5
SUBSTEPS = 2
DT_MAX = 0.05

PAD_SPEED = 1.0

def sim_rate():
    """Simulation time units per real second, as the pad's Speed dial has them."""
    return SIM_RATE * PAD_SPEED

MOONS = [
    (0.55, 0.0020, 0.000, 0.4, 3.0e-5, 0.040),
    (2.97, 0.0060, 0.000, 4.1, 4.0e-5, 0.048),
]
N_ANALYTIC = len(MOONS)

ROS_N = 3
ROS_A = 1.9
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
POLAR = [(a, rel + 90.0, ph, sg, rad, gm) for (a, rel, ph, sg, rad, gm) in POLAR][:2]
POLAR = [(an, *p[1:]) for an, p in zip((1.45, 2.1), POLAR)]
N_POLAR = len(POLAR)

CO_A, CO_D, CO_MU, CO_SPEED = 1.1, 0.044, 1.0e-3, 2.5
CO_PHI0 = (0.0, 140.0)
CO_SIDE = (-1.0, 1.0)
CO_DRAW = (0.026, 0.030)
CO_GM = 1.5e-5
N_CO = len(CO_PHI0)
CO_N = CO_A ** -1.5

N_MOONS = N_ANALYTIC + ROS_N + N_POLAR + N_CO

ROCK_SCREEN, ROCK_DEPTH = (-0.05, -0.61), 6.75
_ROCK_OLD = 0
RK = dict(HX=0.06, HA=0.12, HB=0.09, HT=0.035, TX=-0.04, EX=-0.23, HR0=0.030, HR1=0.022, K=0.04, BUMP=0.007)
ROCK_SPIN = 2.2
ROCK_KICK = 0.02
ROCK_BOB = (0.05, 0.04, 0.035)
ROCK_BOB_W = (0.31, 0.29, 0.23)
CAM_FOV = 0.12
CAM_ZOOM = 1.0
FOV_, AIM_, ROLL_, D_ = CAM_FOV, 0.0, 0.7854, (3.2 / CAM_FOV + 3.2) * CAM_ZOOM

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
ROCK_ON = False
if not ROCK_ON:
    ROCK_HOME = np.array([0.0, -80.0, 0.0])

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
T_MOON = 6
TK = dict(TA=0.08, TB=0.036, TC=0.08, TR1=0.0176, TR2=0.016)

def t_sdf(p):
    k = TK
    bar = _capsule(p, np.array([-k["TA"], k["TB"], 0.0]), np.array([k["TA"], k["TB"], 0.0]), k["TR1"])
    stem = _capsule(p, np.array([0.0, k["TB"], 0.0]), np.array([0.0, -k["TC"], 0.0]), k["TR2"])
    return np.minimum(bar, stem)

def _t_mass():
    h = 0.002
    xs, ys, zs = np.arange(-0.11, 0.11, h), np.arange(-0.11, 0.07, h), np.arange(-0.025, 0.025, h)
    P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1)
    q = P[t_sdf(P) < 0.0]
    com = q.mean(0); q = q - com
    I = np.array([(q[:, 1] ** 2 + q[:, 2] ** 2).mean(), (q[:, 0] ** 2 + q[:, 2] ** 2).mean(), (q[:, 0] ** 2 + q[:, 1] ** 2).mean()])
    return com, I

ROCK_COM, ROCK_I = _t_mass()

def rock_w0():
    """The start spin: ROCK_SPIN about the INTERMEDIATE principal axis (the flip axis), small kicks on the other two."""
    w = np.full(3, ROCK_KICK)
    w[int(np.argsort(ROCK_I)[1])] = ROCK_SPIN
    return w

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
        s["rR_n"] = s.get("rR_n", 0) + 1
        if s["rR_n"] % 30 == 0:
            u, _, vt = np.linalg.svd(s["rR"]); s["rR"] = u @ vt
    s["rw"] = w

def rock_uniforms(api, s):
    az = cam_az(s["itime"])
    ca, sa = math.cos(az), math.sin(az)
    T = s["itime"]
    q = ROCK_HOME + np.array([ROCK_BOB[i] * math.sin(ROCK_BOB_W[i] * T + 1.7 * i) for i in range(3)])
    Ry = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
    if s.get("moons_np") is not None:
        q, Ry = s["moons_np"][0, T_MOON, :3].astype(np.float64), np.eye(3)
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

MOON_DRAW_K = 1.7
MOON_DRAW = (0.068, 0.13, 0.064, 0.064, 0.072, 0.056, 0.08, 0.088, 0.088)

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
    cam = cam_az(itime) - 0.5 * math.pi
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
    out[0, :, 3] = MOON_DRAW[:N_MOONS] if len(MOON_DRAW) >= N_MOONS else out[0, :, 3] * MOON_DRAW_K
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
CAM_EL, CAM_ORBIT = 0.5236, 0.0

CAM_LOOK = (0.0, 0.0, 0.0, 1.0)
CAM_MOVE = (0.0, 0.0, 0.0)

def cam_az(t):
    return CAM_ORBIT * t

def _bolt_fire(api, s):
    rng = s["rng"]
    t_est = s["itime"]
    cam = mt_camera(t_est)[0]
    cam = cam / np.linalg.norm(cam)
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

A_R0, A_R1 = 0.23, 2.42
A_VT, A_VR, A_FT, A_FR = 512, 96, 2048, 384
A_PSI_ITERS = 6
A_STIR_RATE = 1.5
A_STIR_AMP = (6.0, 14.0)
A_STIR_R = (0.04, 0.10)
A_BURST_AMP = 34.0
A_BURST_R = 0.20
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

def aurora_step(api, s, dt, burst=False):
    a, rng = s["aur"], s["rng"]
    h = dt * sim_rate()
    a["t"] += h
    a["vph"].write(a_phases(A_VROW, a["t"], h).tobytes())
    a["fph"].write(a_phases(A_FROW, a["t"], h).tobytes())
    inj = []
    if burst:
        ph0 = rng.uniform(0.0, 2.0 * math.pi)
        for k in range(4):
            inj.append((rng.uniform(A_R0 + 0.15, A_R1 - 0.15), ph0 + k * (math.pi / 2),
                        A_BURST_AMP, A_BURST_R))
    for _ in range(rng.poisson(A_STIR_RATE * dt)):
        if len(inj) < 4:
            inj.append((rng.uniform(A_R0 + 0.1, A_R1 - 0.1), rng.uniform(0, 2 * math.pi),
                        float(rng.choice((-1, 1)) * rng.uniform(*A_STIR_AMP)), rng.uniform(*A_STIR_R)))
    inj = inj[:4]
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

MT_AMBIENT = False
MT_RATE = 2.0
MT_MOON_BOOST = 400.0
MT_CAND = 24
MT_GOOD = 0.6
MT_REFINE_TOP = 6
MT_REFINE_N = 8
MT_REFINE_SIG = 0.008
MT_H = 0.02
MT_T = 9.0
MT_STEPS_PER_FRAME = 10
MT_WHIP = 1.35
MT_HIT_P = 0.25
MT_EVERY_BEATS = 8
MT_EVERY_S = 4.0
MT_MAX_ALIVE = 2
MT_NSLOT = 3
MT_HIST = 48
MT_TAIL_KEEP = 0.7
MT_ROCK_R = 0.12
MT_FLIGHT_H = 0.008
MT_CLOCK_EVERY = 10

CAM_GLIDE = 0.8
CAM_DB = None

def _qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])

def _qrot(q, v):
    u, w = np.asarray(q[:3]), q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)

def _qaxis(axis, ang):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    return np.array([*(a * math.sin(0.5 * ang)), math.cos(0.5 * ang)])

def _qslerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
    else:
        th = math.acos(d); q = (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)
    return q / np.linalg.norm(q)

SIDE_WIN = (1.0, 1.0, 0.7, 1.0)

def cam_side(k):
    """Side k of the box's four long faces (0 = front, 1 = right, 2 = back, 3 = left): a turn about the long axis; the
    distance puts that side's WINDOW (SIDE_WIN x BOX_H.x) on the screen edges, the near face being BOX_H.z or .x away."""
    bx, by, bz = BOX_AX
    near = BOX_H[2] if k % 2 == 0 else BOX_H[0]
    return _qaxis(by, -0.5 * math.pi * k), CAM_ZOOM * (SIDE_WIN[k] * BOX_H[0] / FOV_ + near)

def cam_db_step(s, dpad_clicks, recenter):
    """THE dpad TIPS THE CUBE LIKE A DIE (2026-10-19, user: "can we add the last missing 2 sides as cam pos?"): all SIX faces.
    RIGHT / LEFT = a 90 deg turn about the camera's own up (round the sides, as before), UP / DOWN = a 90 deg tip about the
    camera's own right (over the top / under the bottom), Recenter = the front. (Before: 4 sides, UP = the opposite side,
    DOWN = the front -- the opposite side is now two presses.) The distance puts the facing face's window on the screen edges
    (SIDE_WIN: the back's 0.7 hole, every other face flush)."""
    global CAM_DB
    I = np.array([0.0, 0.0, 0.0, 1.0])
    cs = s.setdefault("cam_db", dict(q=I.copy(), qT=I.copy(), d=D_, dT=D_, q0=None, d0=D_, t0=-9.0, settled=False, face="front"))
    bx, by, bz = BOX_AX
    qT = cs["qT"].copy()
    moved = False
    if recenter:
        qT, moved = I.copy(), True
    for btn, ax, ang in (("RIGHT", by, -0.5 * math.pi), ("LEFT", by, 0.5 * math.pi),
                         ("UP", bx, 0.5 * math.pi), ("DOWN", bx, -0.5 * math.pi)):
        for _ in range(int(dpad_clicks.get(btn, 0))):
            qT = _qmul(_qaxis(_qrot(qT, ax), ang), qT); moved = True
    if moved:
        qT = qT / np.linalg.norm(qT)
        fw = _qrot(qT, bz)
        dots = np.array([fw @ bx, fw @ by, fw @ bz])
        i = int(np.argmax(np.abs(dots)))
        face = (("left", "right"), ("bottom", "top"), ("front", "back"))[i][0 if dots[i] > 0 else 1]
        win = SIDE_WIN[2] if face == "back" else 1.0
        dT = CAM_ZOOM * (win * BOX_H[0] / FOV_ + BOX_H[i])
        cs.update(q0=cs["q"].copy(), d0=cs["d"], qT=qT, dT=dT, t0=s["itime"], settled=False, face=face)
        print(f"camera: side {face}", file=sys.stderr)
    if CAM_DB is not None and cs["settled"]:
        return CAM_DB
    a = _smooth01((s["itime"] - cs["t0"]) / CAM_GLIDE) if cs["q0"] is not None else 1.0
    cs["settled"] = a >= 1.0
    cs["q"] = _qslerp(cs["q0"], cs["qT"], a) if cs["q0"] is not None else cs["qT"]
    cs["d"] = cs["d0"] + (cs["dT"] - cs["d0"]) * a
    rt, up, fw = _qrot(cs["q"], bx), _qrot(cs["q"], by), _qrot(cs["q"], bz)
    CAM_DB = (-cs["d"] * fw, rt, up, fw)
    return CAM_DB

def mt_camera(itime, look=None, move=None):
    """The brown dwarf's camera. MUST stay a mirror of this variant's image.glsl mainImage: the
    boop / reticle ray is built from this copy, so any divergence between the two pokes empty
    space while you look straight at a moon.

    PAD FLIGHT (2026-10-15, user: "maybe we can actually add 3d camera flight") is applied
    here as well as in the shader: look = (yaw, pitch, roll) ADDED to the authored rig (added, not
    replacing it -- see cam_az: the azimuth is shared with the sky park, the rock and the polar
    moons' nodes), move = (right, up, forward) offset in the camera's own basis. The authored rig
    IS the zero, so centring the sticks returns the exact shot this shader shipped with."""
    look = CAM_LOOK if look is None else look
    move = CAM_MOVE if move is None else move
    el = CAM_EL
    az = CAM_ORBIT * itime
    base = D_ * np.array([math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)])
    aim = AIM_ * np.array([base[0], 0.0, base[2]]) / max(math.hypot(base[0], base[2]), 1e-9)
    fw = (aim - base) / np.linalg.norm(aim - base)
    rt = np.cross([0.0, 1.0, 0.0], fw); rt /= np.linalg.norm(rt)
    up = np.cross(fw, rt)
    cr, sr = math.cos(ROLL_), math.sin(ROLL_)
    rt, up = cr * rt + sr * up, -sr * rt + cr * up
    if CAM_DB is not None:
        base, rt, up, fw = CAM_DB
    ro = base + move[0] * rt + move[1] * up + move[2] * fw
    q = np.asarray(look, dtype=np.float64)
    def vrot(v):
        return v + 2.0 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)
    basis = (rt, up, fw)
    rt2 = vrot(np.array([1.0, 0.0, 0.0])); up2 = vrot(np.array([0.0, 1.0, 0.0])); fw2 = vrot(np.array([0.0, 0.0, 1.0]))
    fwn = fw2[2] * fw + fw2[0] * rt + fw2[1] * up
    rtn = rt2[2] * fw + rt2[0] * rt + rt2[1] * up
    upn = up2[2] * fw + up2[0] * rt + up2[1] * up
    return ro, fwn, rtn, upn

def mt_bodies(moons_now, t_sim, itime, dt_real):
    """Body centres (K, nb, 3), radii (nb), GM (nb) at dt_real (array, K) seconds after (t_sim, itime).
    Moons on circles at their Kepler rate; polar moons exact; the racket turns with the camera; dwarf at 0."""
    dt_real = np.asarray(dt_real, dtype=np.float64)
    p = moons_now[0, :, :3].astype(np.float64)
    rad = moons_now[0, :, 3].astype(np.float64)
    gm = moons_now[1, :, 0].astype(np.float64) * MT_MOON_BOOST
    r = np.maximum(np.hypot(p[:, 0], p[:, 2]), 1e-3)
    d = (r ** -1.5)[None, :] * sim_rate() * dt_real[:, None]
    c, sn = np.cos(d), np.sin(d)
    P = np.empty((len(dt_real), len(rad), 3))
    P[..., 0] = p[None, :, 0] * c - p[None, :, 2] * sn
    P[..., 1] = p[None, :, 1]
    P[..., 2] = p[None, :, 0] * sn + p[None, :, 2] * c
    t = t_sim + dt_real * sim_rate()
    cam = cam_az(itime + dt_real) - 0.5 * math.pi
    for k, (a, rel, ph, sgn, _, _) in enumerate(POLAR):
        node, u = cam + math.radians(rel), a ** -1.5 * t + ph
        i = N_ANALYTIC + ROS_N + k
        P[:, i, 0] = a * np.cos(u) * np.cos(node); P[:, i, 1] = sgn * a * np.sin(u); P[:, i, 2] = a * np.cos(u) * np.sin(node)
    az = cam_az(itime + dt_real)
    q = ROCK_HOME
    rock = np.stack([q[0] * np.cos(az) - q[2] * np.sin(az), np.full_like(az, q[1]), q[0] * np.sin(az) + q[2] * np.cos(az)], 1)
    allP = np.concatenate([np.zeros((len(dt_real), 1, 3)), P, rock[:, None, :]], 1)
    R = np.concatenate([[DW_R], rad, [MT_ROCK_R]])
    G = np.concatenate([[1.0], gm, [0.0]])
    return allP, R, G

def mt_acc(x, P, G):
    """x (K, 3), P (K, nb, 3) -> gravity (K, 3)."""
    d = P - x[:, None, :]
    q = (d ** 2).sum(-1) + 1e-6
    a = (G[None, :, None] * d / q[..., None] ** 1.5).sum(1) + BOX_G * BOX_AX[2][None, :]
    if HOLE is not None:
        dh = HOLE[0][None, :] - x
        a = a + HOLE[1] * dh / (((dh ** 2).sum(-1) + HOLE_SOFT ** 2) ** 1.5)[:, None]
    if NS2 is not None:
        dn = NS2[0][None, :] - x
        a = a + NS2[1] * dn / (((dn ** 2).sum(-1) + NS2_SOFT ** 2) ** 1.5)[:, None]
    return a

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
    return list(range(1, 1 + N_MOONS)) + ([1 + N_MOONS] if ROCK_ON else [])

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
        fin = run & (np.abs(x @ BOX_AX.T) > BOX_H - MT_WALL_M).any(1)
        S["tend"] = np.where(fin, S["tau"], S["tend"])
        S["done"] |= fin
    if not ((S["done"] | S["bad"]).all() or S["tau"] >= MT_T):
        return None
    ok = S["done"] & ~S["bad"] & (S["rmin"] > 1.02 * DW_R) & (S["rmin"] < MT_WHIP)
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
        q, qw = BOX_AX @ x, BOX_AX @ w
        for k in range(3):
            if abs(q[k]) > BOX_H[k]:
                q[k] = math.copysign(BOX_H[k], q[k])
                if qw[k] * q[k] > 0.0:
                    qw[k] = -BOX_BOUNCE * qw[k]
        x, w = BOX_AX.T @ q, BOX_AX.T @ qw
        d = np.linalg.norm(P - x[None], axis=1) - R
        if HOLE is not None and np.linalg.norm(x - HOLE[0]) < HOLE_R:
            m["alive"], m["done"] = False, True
            break
        if d.min() < 0.0:
            j = int(d.argmin())
            m["hit_t"], m["hit_r"] = itime, float(R[j])
            m["hit_p"] = P[j] + (x - P[j]) / max(np.linalg.norm(x - P[j]), 1e-6) * R[j]
            if 1 <= j <= N_MOONS:
                add_puff(s, P[j])
            m["alive"], m["done"] = False, True
            break
    m["x"], m["w"] = x, w
    if itime - m["start"] > MT_BOX_LIFE:
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
    elif MT_AMBIENT and itime >= s["mt_next"] and sum(1 for m in s["mt_list"] if not m["done"]) < MT_MAX_ALIVE:
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
    al = [m for m in s["mt_list"] if m["alive"]]
    for a in range(len(al)):
        for b in range(a + 1, len(al)):
            ma, mb = al[a], al[b]
            if ma["alive"] and mb["alive"] and np.linalg.norm(ma["x"] - mb["x"]) < MT_COLL_R:
                p = 0.5 * (ma["x"] + mb["x"])
                for m in (ma, mb):
                    m["alive"], m["done"], m["hit_t"], m["hit_p"], m["hit_r"] = False, True, itime, p.copy(), 0.02
                add_puff(s, p, 0.5 * FM_COLL_PUFF)
                print(f"meteor: two meteors COLLIDE at r {math.hypot(p[0], p[2]):.2f} -- debris", file=sys.stderr)
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
FLING_CATCH_ANG = 0.075 * FOV_ / 0.356
FLING_ACC_MOON = 3.0
FLING_ACC_MET = 8.0
SPLAT_V = 0.06
SPLAT_R = 0.08
SPLAT_T = 0.15
DWARF_BOUNCE = 0.6
HEAT_TAU = 3.0
HEAT_MAX = 2.0
HOT_V_MIN = 0.12
CRAWL_T = 0.5
CRAWL_KICK = 0.85

def dwarf_contact(x, v, rad, st, h):
    """The red-hot bounce off the dwarf (free_step AND the rescue predictor: identical). st carries the crawl timer.
    Returns (x, v, impact speed or 0)."""
    r = float(np.linalg.norm(x))
    if r >= DW_R + rad + 2e-3:
        st["crawl"] = 0.0
        return x, v, 0.0
    nrm = x / r
    x = nrm * (DW_R + rad + 1e-4)
    vn = float(v @ nrm)
    imp = 0.0
    if vn < 0.0:
        v = v - (1.0 + DWARF_BOUNCE) * vn * nrm
        imp = -vn
    st["crawl"] = st.get("crawl", 0.0) + h
    if st["crawl"] > CRAWL_T:
        v = v - float(v @ nrm) * nrm + CRAWL_KICK * nrm
        st["crawl"] = 0.0
        imp = max(imp, -1.0)
    return x, v, imp

FREE_GONE_R = 6.0
BOX_H = np.array([3.2, 3.2, 3.2])
BOX_BOUNCE = 0.85
EARTH_R = 3.3
BOX_G = 1.0 / EARTH_R ** 2

def box_axes():
    el = CAM_EL
    base = D_ * np.array([0.0, math.sin(el), -math.cos(el)])
    fw = -base / np.linalg.norm(base)
    rt = np.cross([0.0, 1.0, 0.0], fw); rt /= np.linalg.norm(rt)
    up = np.cross(fw, rt)
    cr, sr = math.cos(ROLL_), math.sin(ROLL_)
    rt, up = cr * rt + sr * up, -sr * rt + cr * up
    return np.array([rt, up, fw])

BOX_AX = box_axes()
RB_FILE = os.path.expanduser("~/.cache/shader-backdrop/dwarf_rebirth")
RB_PAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pad.json")
RB_POLL_S = 0.4
HOLE_GM = 40.0
HOLE_RAMP = 1.5
HOLE_R = 0.22
HOLE_SOFT = 0.05
HOLE_CORNER_IN = 0.45
HOLE_MIN_S, HOLE_MAX_S = 4.0, 14.0
HOLE_FLASH_S = 0.8
RING_STRIP = 0.0
DUST_N = 1024 * 512
DUST_GAIN = 1.0
DUST_SUB = 4
RING_STRIP_OLD = 2.4
RB_RING_GONE = 0.01
HOLE = None
NS2 = None
NS2_GM = 1.0
NS2_SOFT = 0.08
NS2_R0 = 1.6
NS2_T = 4.5
NS2_IN = 1.4
NS2_P0 = 3.2
NS2_WMAX = 2.0 * math.pi * 3.0
NS2_TILT = 0.35
NS_FLARE_S = 0.45
NS_FADE_S = 0.6
MERGE_FLASH_S = 0.6
RB2_DELAY = 0.6
RB2_FLIGHT = 2.8
RB2_SPIT_WIN = 0.3
RB2_T0 = 0.0
RB2_LEAD_SIM = 0.9
RB2_READY_SIM = 0.45
RB2_MASS = (0.6, 0.4, 1.0, 0.25, 1.6)
RB2_H = 0.004
RB2_POLE_LAT = 0.80
RB2_PUFF_RATE = 0.9
RB2_PUFF_T = 2.5
RB2_REGROW = 3.0
FU_MIN = 1.0
FU_READY = 0.2
FU_JIT = (1.0, 0.9, 1.1)
LOOP_A_MIN = 2.0
LOOP_TAUS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
LOOP_H = 0.006
LOOP_BRAKE_MU = (0.08, 2.5)
LOOP_BRAKE_U = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
LOOP_BRAKE_ERR = 0.12
LOOP_BRAKE_FLY = 0.8
RB2_REGROW_T = 20.0
RB2_POLE = True
RB2_POLE_MU = np.geomspace(0.12, 4.0, 48)
RB2_POLE_BETA = (0.5, 0.8, 1.1, 1.35)
RB2_POLE_LAT = 0.80
RS_WAIT = 3.0
RS_BUDGET, RS_WINDOW = 3, 20.0
RS_T1 = (1.2, 2.0, 2.8)
RS_TAU = (0.9, 1.5, 2.2)
FM_FRAC = 0.75
FM_T = (2.0, 3.0, 4.0, 5.0)
FM_LAM = (0.5, 0.7, 1.0, 1.4, 2.0)
FM_U = (0.5, 0.9, 1.4)
FM_PUFF = 3.0
PUFF_T = 2.5
AUTOFILL = False
FM_PROF_S = 2.0
FM_BAND = 0.08
FM_MIN_DEF = 0.12
FM_MOON_DR = 0.10
FM_BUDGET, FM_WINDOW = 8, 20.0
FM_FULL = 1.0
FM_CAP_TAU = 600.0
FM_COLL_PUFF = 4.0
MT_COLL_R = 0.05
RS_READY = 1.0
RB2_DIR = os.path.expanduser("~/.cache/shader-backdrop/rb2")
RB2_TAIL = 24
MT_WALL_M = 0.03
MT_BOX_LIFE = 20.0

def free_apply(s, arr):
    """Override the moons array with the free (booped) moons: their own position, or hidden + weightless if gone."""
    for i, f in s.get("free", {}).items():
        if f["gone"]:
            arr[0, i] = (0.0, -60.0, 0.0, 0.0)
            arr[1, i, 0] = 0.0
        else:
            arr[0, i, :3] = f["x"]
            if "heat" in f:
                arr[1, i, 3] = f["heat"] * math.exp(-(s["itime"] - f["heat_t"]) / HEAT_TAU)
    return arr

def free_step(s, h):
    for i, f in s.get("free", {}).items():
        if f["gone"] or f.get("path"):
            continue
        global BOX_AX
        if BOX_AX is None:
            BOX_AX = box_axes()
        gv = BOX_G * BOX_AX[2]
        x, v = f["x"], f["v"]

        def acc(x):
            a = gv - x / np.linalg.norm(x) ** 3
            if HOLE is not None:
                dh = HOLE[0] - x
                a = a + HOLE[1] * dh / (dh @ dh + HOLE_SOFT ** 2) ** 1.5
            if NS2 is not None:
                dn = NS2[0] - x
                a = a + NS2[1] * dn / (dn @ dn + NS2_SOFT ** 2) ** 1.5
            return a
        v = v + 0.5 * h * acc(x)
        x = x + h * v
        v = v + 0.5 * h * acc(x)
        q, qv = BOX_AX @ x, BOX_AX @ v
        lim = BOX_H - f["rad"]
        for k in range(3):
            if abs(q[k]) > lim[k]:
                q[k] = math.copysign(lim[k], q[k])
                if qv[k] * q[k] > 0.0:
                    qv[k] = -BOX_BOUNCE * qv[k]
        x, v = BOX_AX.T @ q, BOX_AX.T @ qv
        f["x"], f["v"] = x, v
        r = float(np.linalg.norm(x))
        if HOLE is not None and np.linalg.norm(x - HOLE[0]) < HOLE_R + f["rad"]:
            f["gone"] = True
            print(f"rebirth: moon {i} eaten by the hole", file=sys.stderr)
            continue
        x, v, imp = dwarf_contact(x, v, f["rad"], f, h)
        f["x"], f["v"] = x, v
        if imp > HOT_V_MIN:
            s.setdefault("wob", {})[i] = s["itime"]
            f["heat"] = min(HEAT_MAX, f.get("heat", 0.0) * math.exp(-(s["itime"] - f.get("heat_t", s["itime"])) / HEAT_TAU)
                            + min(1.0, imp / 0.5))
            f["heat_t"] = s["itime"]
        elif imp < 0.0:
            print(f"moon {i} was crawling the surface -- the dwarf throws it off", file=sys.stderr)

def bd_rock_pos(s):
    az = cam_az(s["itime"])
    ca, sa = math.cos(az), math.sin(az)
    T = s["itime"]
    q = ROCK_HOME + np.array([ROCK_BOB[i] * math.sin(ROCK_BOB_W[i] * T + 1.7 * i) for i in range(3)])
    return np.array([q[0] * ca - q[2] * sa, q[1], q[0] * sa + q[2] * ca])

def _release_moon(s, i, m):
    """The moon at index i, broken loose from its designed orbit (rosette grip / horseshoe /
    analytic) into FREE flight in the dwarf's real gravity. Returns its free entry, or None if
    it is already gone. Shared by the hand's boop and the pad's reticle so both break a moon
    loose in exactly the same way, with the same starting velocity."""
    s.setdefault("wob", {})[i] = s["itime"]
    f = s.setdefault("free", {}).get(i)
    if f is not None:
        if not f["gone"]:
            f["touch"] = s["itime"]
        return None if f["gone"] else f
    x = m[0, i, :3].astype(np.float64)
    if N_ANALYTIC + ROS_N <= i < N_ANALYTIC + ROS_N + N_POLAR:
        dtt = 1e-3
        m2 = moon_positions(s["t"] + dtt, s["rp"], s["itime"], (s["cp"], s["cw"]))
        v = (m2[0, i, :3].astype(np.float64) - x) / dtt
    else:
        r = math.hypot(x[0], x[2])
        v = math.sqrt(1.0 / r) * np.array([-x[2], 0.0, x[0]]) / r
    return s["free"].setdefault(i, dict(x=x, v=v, gone=False, rad=float(m[0, i, 3]), touch=s["itime"]))

def bd_boop(api, s, xy, look=(0.0, 0.0, 0.0), move=(0.0, 0.0, 0.0)):
    ro, fw, rt, up = mt_camera(s["itime"], look, move)
    ro, rd = hand.ray(xy[0], xy[1], hand.canvas(api), ro, fw, rt, up, FOV_)
    m = s["moons_np"]
    cents = [m[0, i, :3] for i in range(N_MOONS)] + [bd_rock_pos(s), np.zeros(3)]
    radii = [float(m[0, i, 3]) for i in range(N_MOONS)] + [0.2, DW_R]
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
        f = _release_moon(s, i, m)
        if f is not None:
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
    s["rR"], s["rw"] = np.eye(3), rock_w0()
    s["mt_list"], s["mt_search"] = [], None
    a = s["aur"]
    api.run(a["k"]["A_INIT"], a["gf"], images={0: a["v"][0], 1: a["f"][0]})
    a["cur"] = 0
    print("hand: RESET -- everything back in formation", file=sys.stderr)

def pad_state(api):
    """The pad's contribution this frame: (look, move, reticle, alive).

    The renderer integrates the sticks (a HELD deflection is a rate, so letting go parks the
    framing) and hands the SAME numbers to the image pass as uniforms and to us here, so the
    camera we ray against is exactly the camera on screen. reticle = (canvas px x, px y, active),
    the same convention as hand's boop_xy."""
    inp = getattr(api, "inputs", None) or {}
    look = tuple(inp.get("pad_cam_q") or (0.0, 0.0, 0.0, 1.0))
    move = tuple(inp.get("pad_move") or (0.0, 0.0, 0.0))
    ret = tuple(inp.get("pad_boop") or (0.0, 0.0, 0.0))
    return look, move, ret, float(inp.get("pad_alive", 0.0))

def _smooth01(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)

PAD_DB_POLL = 0.05

def pad_db(s):
    """This frame's view of the dwarf_box pad: widgets + the press/dpad counters' increments since last frame."""
    if s["itime"] - s.get("pdb_t", -9.0) >= PAD_DB_POLL:
        s["pdb_t"] = s["itime"]
        doc = None
        try:
            mt = os.stat(RB_PAD).st_mtime_ns
            if mt != s.get("pdb_mt"):
                with open(RB_PAD) as fh:
                    doc = json.load(fh)
                s["pdb_mt"] = mt
        except (OSError, ValueError):
            doc = None
        if doc is not None:
            s["pdb_doc"] = doc
    doc = s.get("pdb_doc") or {}
    w = doc.get("widgets", {})
    der = doc.get("derived", {})
    trig, dp = der.get("trigger", {}), der.get("dpad", {})
    prev = s.get("pdb_prev")
    s["pdb_prev"] = (dict(trig), dict(dp))
    if prev is None:
        return w, {}, {}
    pt = {k: (n if n < prev[0].get(k, 0) else n - prev[0].get(k, 0)) for k, n in trig.items()}
    pd = {k: (n if n < prev[1].get(k, 0) else n - prev[1].get(k, 0)) for k, n in dp.items()}
    return w, {k: v for k, v in pt.items() if v > 0}, {k: v for k, v in pd.items() if v > 0}

def _wsw(w, k):
    return bool((w.get(k) or {}).get("state"))

def rb_poll(s):
    """A REBIRTH request from the trigger file or the pad's "Rebirth" button (polled, cheap)."""
    if s.get("pdb_rebirth"):
        s["pdb_rebirth"] = False
        return True
    if s["itime"] - s.get("rb_poll_t", -9.0) < RB_POLL_S:
        return False
    s["rb_poll_t"] = s["itime"]
    hit = False
    try:
        mt = os.path.getmtime(RB_FILE)
        if s.setdefault("rb_file_mt", mt) != mt:
            s["rb_file_mt"], hit = mt, True
    except OSError:
        pass
    if s.pop("pdb_rebirth", False):
        hit = True
    return hit

def mouse_meteor(api, s, xy, look, move):
    """MOUSE METEOR mode: a meteor from a random glass wall, aimed at the cursor's point at the box's middle depth."""
    ro, fw, rt, up = mt_camera(s["itime"], look, move)
    ro, rd = hand.ray(xy[0], xy[1], hand.canvas(api), ro, fw, rt, up, FOV_)
    bz = BOX_AX[2]
    t = -float(np.dot(ro, bz)) / float(np.dot(rd, bz)) if abs(float(np.dot(rd, bz))) > 1e-6 else 8.0
    T = ro + rd * max(t, 0.5)
    rng = s["rng"]
    k = int(rng.integers(0, 6))
    q = rng.uniform(-1.0, 1.0, 3) * BOX_H
    q[k % 3] = (1.0 if k < 3 else -1.0) * BOX_H[k % 3] * 0.98
    if k % 3 == 2 and k < 3:
        q[2] = BOX_H[2] * 0.98
    S = BOX_AX.T @ q
    w = (T - S) / np.linalg.norm(T - S) * 1.6
    s.setdefault("mt_list", []).append(dict(x=S, w=w, start=s["itime"], hist=[], alive=False, done=False, hit_t=-100.0,
                                             hit_p=np.zeros(3), hit_r=0.0, plan_clear=0.0, plan_hit=-1, target=s["itime"],
                                             plan_peri=s["itime"]))
    s["mt_list"] = s["mt_list"][-MT_NSLOT:]
    print("mouse: meteor at the cursor", file=sys.stderr)

def rb_start(api, s, why, merger=False):
    if s.get("rb") is not None:
        return
    for i in range(N_MOONS):
        _release_moon(s, i, s["moons_np"])
    s["mt_next"] = s["itime"] + 1e9
    if merger:
        rng = s["rng"]
        k = int(rng.integers(0, 4))
        q = rng.uniform(-0.6, 0.6, 3) * BOX_H
        ax = 0 if k < 2 else 1
        q[ax] = (1.0 if k % 2 == 0 else -1.0) * BOX_H[ax] * 0.97
        s["rb"] = dict(ph=0, t0=s["itime"], tl=s["itime"], phi=float(rng.uniform(0.0, 2.0 * math.pi)),
                       entry=BOX_AX.T @ q, pos=np.zeros(3), merger=True)
        print(f"rebirth: MERGER ({why}) -- a second neutron star comes in", file=sys.stderr)
        return
    s["rb"] = dict(ph=0, t0=s["itime"], pos=np.zeros(3), merger=False)
    rb_collapse(api, s, why)

def rb_collapse(api, s, why):
    """The star collapses: the hole grows AT THE CENTRE and the ring becomes dust."""
    rb = s["rb"]
    rb.update(ph=1, t0=s["itime"], pos=np.zeros(3))
    k, dw = s["k"], s["dust"]
    api.run(k["PSEED"], ((DUST_N + 255) // 256, 1, 1), images={0: dw["pos"], 1: dw["vel"]},
            samplers={"u_src": s["f"][s["cur"]], "u_phase": s["phase"]},
            uniforms={"u_np": DUST_N, "u_seed": float(s["rng"].integers(1, 1 << 30))})
    for f in s["f"]:
        api.run(k["RZERO"], s["groups"], images={0: f})
    dw["on"], dw["alive"] = True, DUST_N
    print(f"rebirth: PHASE 1 ({why}) -- the star collapses into a black hole", file=sys.stderr)

def rb_step(api, s):
    global HOLE, NS2
    rb, it = s.get("rb"), s["itime"]
    vis, flash, refill = 0.0, 0.0, 1.0
    ns_vis, ns_flare, ns2 = 1.0, 0.0, (0.0, 0.0, 0.0, 0.0)
    NS2 = None
    if rb is not None and rb["ph"] == 0:
        tau = it - rb["t0"]
        dtr = max(it - rb["tl"], 0.0); rb["tl"] = it
        r = NS2_R0 * max(1.0 - tau / NS2_T, 0.0) ** 0.25
        rb["phi"] += min(2.0 * math.pi / NS2_P0 * (NS2_R0 / max(r, 1e-3)) ** 1.5, NS2_WMAX) * dtr
        ct, st = math.cos(NS2_TILT), math.sin(NS2_TILT)
        sp = r * (math.cos(rb["phi"]) * np.array([1.0, 0.0, 0.0]) + math.sin(rb["phi"]) * np.array([0.0, st, ct]))
        w = _smooth01(tau / NS2_IN)
        pc = rb["entry"] * (1.0 - w) + sp * w
        NS2 = (pc, NS2_GM * w)
        ns2 = (float(pc[0]), float(pc[1]), float(pc[2]), 1.0)
        refill = 0.0
        if r < 2.2 * DW_R or tau >= NS2_T:
            NS2 = None
            rb_collapse(api, s, "the stars touch")
            rb["merge_t"] = it
    if rb is None:
        HOLE = None
        fr = s.get("free", {})
        if len(fr) == N_MOONS and all(f["gone"] for f in fr.values()) and s.get("ring_mean", 1.0) < RB_RING_GONE:
            rb_start(api, s, "auto: every moon and the disk have left the building")
    elif rb["ph"] == 1:
        age = it - rb["t0"]
        vis = _smooth01(age / HOLE_RAMP)
        HOLE = (rb["pos"], HOLE_GM * vis)
        ns_flare = math.sin(math.pi * min(age / NS_FLARE_S, 1.0))
        ns_vis = 1.0 - _smooth01((age - 0.5 * NS_FLARE_S) / NS_FADE_S)
        if rb.get("merge_t") is not None:
            flash = max(flash, 1.0 - _smooth01((it - rb["merge_t"]) / MERGE_FLASH_S))
        refill = 0.0
        fr = s.get("free", {})
        empty = len(fr) == N_MOONS and all(f["gone"] for f in fr.values()) and \
            not any(m["alive"] for m in s.get("mt_list", [])) and s["dust"]["alive"] < 0.02 * DUST_N
        if (empty and age > HOLE_MIN_S) or age > HOLE_MAX_S:
            rb.update(ph=2, t1=it)
            rb2_plan(api, s)
            print(f"rebirth: the hole evaporates after {age:.1f} s ({'box empty' if empty else 'time up'})", file=sys.stderr)
    if rb is not None and rb["ph"] == 2:
        HOLE, refill = None, 0.0
        flash = 1.0 - _smooth01((it - rb["t1"]) / HOLE_FLASH_S)
        if it - rb["t1"] > HOLE_FLASH_S:
            s["dust"]["on"] = False
            rb["ph"] = 3
    if rb is not None and rb["ph"] == 3:
        HOLE, refill = None, 0.0
        if rb.get("hit_done"):
            refill = 1.0 + (RB2_REGROW - 1.0) * max(0.0, 1.0 - (it - rb["t_hit_real"]) / RB2_REGROW_T)
            if it - rb["t_hit_real"] > RB2_REGROW_T and all(fu["done"] for fu in rb.get("fus", [])):
                s["rb"] = None
                s["mt_next"] = it + 2.0
                print("rebirth: done -- the system is reborn", file=sys.stderr)
        rb2_draw(api, s)
    pos = rb["pos"] if rb is not None else np.zeros(3)
    s["rb_ring"] = (float(pos[0]), float(pos[1]), float(pos[2]), RING_STRIP * vis)
    s["rb_refill"] = refill if AUTOFILL else 0.0
    api.render_uniform("u_rbHole", (float(pos[0]), float(pos[1]), float(pos[2]), float(vis)))
    api.render_uniform("u_rbFlash", float(flash))
    api.render_uniform("u_ns2", ns2)
    api.render_uniform("u_nsStar", (float(1.0 - ns_vis), float(ns_flare)))
    dust_step(api, s)

def rail_predict(s, t_target):
    """Every body's state on its rail at sim time t_target (the hidden rails, integrated forward from now)."""
    ss = dict(rp=s["rp"].copy(), rv=s["rv"].copy(), cp=np.array(s["cp"], float).copy(), cw=np.array(s["cw"], float).copy(), t=s["t"])
    h = 0.005
    while ss["t"] < t_target - 1e-9:
        hh = min(h, t_target - ss["t"])
        ros_step(ss, hh); co_step(ss, hh); ss["t"] += hh
    it = s["itime"] + (t_target - s["t"]) / max(sim_rate(), 1e-6)
    P0 = moon_positions(t_target, ss["rp"], it, (ss["cp"], ss["cw"]))
    e = 1e-3
    ros_step(ss, e); co_step(ss, e)
    P1 = moon_positions(t_target + e, ss["rp"], it, (ss["cp"], ss["cw"]))
    return P0[0, :, :3].astype(np.float64), (P1[0, :, :3] - P0[0, :, :3]).astype(np.float64) / e, P0[0, :, 3].astype(np.float64)

def _rb2_acc(x):
    r = np.linalg.norm(x, axis=-1, keepdims=True)
    return -x / r ** 3 + BOX_G * BOX_AX[2]

def _rb2_fly(x, v, T, h=RB2_H):
    """Batch forward leapfrog (dwarf + box gravity, no glass) for time T: end state + the minimum radius on the way."""
    n = max(1, int(math.ceil(T / h))); h = T / n
    rmin = np.linalg.norm(x, axis=-1)
    for _ in range(n):
        v = v + 0.5 * h * _rb2_acc(x); x = x + h * v; v = v + 0.5 * h * _rb2_acc(x)
        rmin = np.minimum(rmin, np.linalg.norm(x, axis=-1))
    return x, v, rmin

_LAUNCH = None
PS_FLY_MIN = 0.4
PS_SHIP_BONUS = 12.0
STAR_SHOTS = False
PS_DEV_K = 3.0
PS_DEV_MAX = 0.8
PS_SHIPS = ("squid", "swarm")
PS_SLING, PS_SLING_R = 1.0, 0.45
PS_APPT_W = 2.0

def ps_field_test(fld, x, ta):
    """Points x (n, 3) at sim times ta (scalar): inside a ship's beam field? -> (ok, ship 0 / 1, distance off its ellipse)."""
    b = fld["b0"] + (ta - fld["t0"]) * fld["kb"]
    spU = float(ps_sp(fld["sgn"], fld["phi0"], b))
    phi = float(ps_beam_angle(spU))
    r = np.linalg.norm(x, axis=1)
    az0 = np.arctan2(x[:, 2], x[:, 0])
    el = np.arcsin(np.clip(x[:, 1] / np.maximum(r, 1e-9), -1.0, 1.0))
    ok = np.zeros(len(x), bool); sid = np.full(len(x), -1); dev = np.zeros(len(x))
    for k in (0, 1):
        ph = phi + math.pi * k
        az = np.angle(np.exp(1j * (az0 - ph)))
        sp = spU + math.pi * k
        th = fld["phil"] + math.atan2(fld["b"] * math.sin(sp), fld["a"] * math.cos(sp))
        ro = math.sqrt((fld["a"] * math.cos(sp)) ** 2 + (fld["b"] * math.sin(sp)) ** 2)
        dv = np.hypot(x[:, 0] - ro * math.cos(th), x[:, 2] - ro * math.sin(th))
        u = x / np.maximum(r, 1e-9)[:, None]
        m = (np.abs(az) < 0.85 * PS_WA) & (np.abs(el) < 0.85 * PS_W) & (r > PS_RMIN + 0.1) & (r < 0.95 * _ps_rmax(u)) \
            & (dv < PS_DEV_MAX) & ~ok
        ok |= m; sid[m] = k; dev[m] = dv[m]
    return ok, sid, dev

def ps_field(s, t_ready):
    """The ships' beam field for a search (None until the pulsar runs)."""
    ch = s.get("chase") or {}
    if not CH_PULSAR or "pbc" not in ch or "ps_tb" not in ch:
        return None
    la = BOX_AX[1]
    return dict(t0=s["t"], b0=ch["pbc"], kb=1.0 / max(sim_rate() * ch["ps_tb"], 1e-6), phi0=ch["ps_phi0"], sgn=ch["ps_sgn"],
                t_ready=t_ready, phil=math.atan2(float(la[2]), float(la[0])), a=PS_OPT[0] * float(_ps_rmax(np.array([la]))[0]),
                b=PS_OPT[1] * R_OUT)

def ps_appoint(s, mets):
    """A committed shot: every ship meteor -> an appointment for its ship (the swarm's go round robin to its 5 ships)."""
    for m in mets:
        if m.get("kind") in PS_SHIPS:
            sh = PS_SHIPS.index(m["kind"])
            k = 0
            if sh == 1:
                k = s["ps_rr"] = (s.get("ps_rr", -1) + 1) % CH_SW_N
            s.setdefault("ps_appt", []).append(dict(t=float(m["ts"][0]), x=np.asarray(m["xs"][0], np.float64).copy(), ship=sh, k=k))

def _launch_kind(pole):
    """The kind of the meteor just traced (record=True call): a ship's, else the star's."""
    sh = getattr(_rb2_back, "ship", None)
    if _LAUNCH is not None and sh is not None and len(sh) and sh[0] >= 0:
        return PS_SHIPS[int(sh[0])]
    return "pole" if pole else "dwarf"

def _rb2_back(x, w, Tmax, rs, record=False, bounce=True, h=RB2_H):
    """Batch BACKWARD flight (w = the reversed velocity) in dwarf + box gravity until each path meets r < rs (its launch
    from the dwarf's surface). BANK SHOTS: off the glass the normal velocity is reflected and divided by BOX_BOUNCE (the
    forward bounce loses it). bounce=False: stop at the first wall instead (a wall spawn). Returns tau (inf = none), the
    end point / reversed velocity, and with record the path (states every step, backward order)."""
    n = len(x)
    tau = np.full(n, np.inf); ex = np.zeros_like(x); ew = np.zeros_like(x)
    if not STAR_SHOTS and _LAUNCH is not None and rs == MET_RS and not bounce:
        _rb2_back.bounces = np.zeros(n, int); _rb2_back.ship, _rb2_back.bonus = np.full(n, -1), np.zeros(n)
        return tau, ex, ew, []
    alive = np.isfinite(w).all(1)
    tt, path = 0.0, []
    bnc = np.zeros(n, int)
    fld = _LAUNCH if (rs == MET_RS and bounce) else None
    ship = np.full(n, -1); bonus = np.zeros(n); rmin = np.full(n, 99.0)
    T_imp = fld["t_ready"] + Tmax if fld is not None else 0.0
    while tt < Tmax and alive.any():
        w = w + 0.5 * h * _rb2_acc(x); x = x + h * w; w = w + 0.5 * h * _rb2_acc(x); tt += h
        q = x @ BOX_AX.T; qw = w @ BOX_AX.T
        out = np.abs(q) > BOX_H
        if bounce:
            if out.any():
                bnc += (alive & out.any(1)).astype(int)
                q = np.where(out, np.sign(q) * BOX_H, q)
                qw = np.where(out & (qw * q > 0.0), -qw / BOX_BOUNCE, qw)
                x, w = q @ BOX_AX, qw @ BOX_AX
            hit = alive & (np.linalg.norm(x, axis=1) < rs)
            if fld is not None:
                rmin = np.where(alive, np.minimum(rmin, np.linalg.norm(x, axis=1)), rmin)
            if fld is not None and tt >= PS_FLY_MIN:
                ok, sid, dev = ps_field_test(fld, x, T_imp - tt)
                sh = alive & ~hit & ok
                ship[sh] = sid[sh]
                bonus[sh] = PS_SHIP_BONUS - PS_DEV_K * dev[sh] + PS_SLING * (rmin[sh] < PS_SLING_R)
                hit = hit | sh
        else:
            hit = alive & out.any(1)
            alive &= ~(np.linalg.norm(x, axis=1) < DW_R)
        if record:
            path.append((tt, x[0].copy()))
        tau[hit] = tt; ex[hit] = x[hit]; ew[hit] = w[hit]
        alive &= ~hit
    if fld is not None and not STAR_SHOTS:
        tau[ship < 0] = np.inf
    _rb2_back.bounces = bnc
    _rb2_back.ship, _rb2_back.bonus = ship, bonus
    return tau, ex, ew, path

def _rb2_path(t_hit, path):
    """A recorded backward run -> a forward playback (times ascending, positions)."""
    ts = np.array([t_hit - p[0] for p in path][::-1] + [t_hit])
    xs = np.array([p[1] for p in path][::-1] + [path[0][1] * 0.0])
    return ts, xs

def _rb2_search(P, V, rad, t_hit, t_ready):
    """PHASE 2's shots, per body (background thread). The body's ARRIVAL velocity is searched on a grid: traced backward
    (bank shots allowed) it must leave the dwarf's surface RB2_FLIGHT (+-RB2_SPIT_WIN) before t_hit -- the spit. For each
    such arrival and mass ratio mu, the meteor's impact velocity is v_m = V + (V - v_arr) / mu (merged = exactly the rail
    velocity); traced backward it must come out of the dwarf near a POLE (|sin lat| > RB2_POLE_LAT); else anywhere on the
    dwarf; else from a wall (the old way). Best: the most polar, mu nearest 0.6, the spit nearest the common moment."""
    TF = RB2_FLIGHT
    Tmax = t_hit - t_ready
    G = np.array(np.meshgrid(np.linspace(0.0, 0.9, 6), np.linspace(-1.5, 1.5, 8), np.linspace(-1.5, 1.5, 8),
                             indexing="ij")).reshape(3, -1).T
    MU = np.geomspace(0.15, 3.0, 12)
    out = []
    for i in range(len(P)):
        er = np.array([P[i][0], 0.0, P[i][2]]); er /= max(np.linalg.norm(er), 1e-9)
        Vn = np.linalg.norm(V[i]); vt = V[i] / max(Vn, 1e-9)
        va = Vn * (G[:, :1] * vt + G[:, 1:2] * er + G[:, 2:3] * np.array([0.0, 1.0, 0.0]))
        rs = DW_R + rad[i] + 0.03
        tb, _, _, _ = _rb2_back(np.repeat(P[i][None], len(va), 0), -va, min(TF + RB2_SPIT_WIN, Tmax), rs)
        okb = np.abs(tb - TF) < RB2_SPIT_WIN
        if not okb.any():
            out.append(None)
            continue
        vas, tbs = va[okb], tb[okb]
        VM = (V[i][None, None] + (V[i][None, None] - vas[:, None]) / MU[None, :, None]).reshape(-1, 3)
        tm, Lm, _, _ = _rb2_back(np.repeat(P[i][None], len(VM), 0), -VM, Tmax, MET_RS)
        lat = np.abs(Lm[:, 1]) / np.maximum(np.linalg.norm(Lm, axis=1), 1e-9)
        cb = np.repeat(np.arange(len(vas)), len(MU)); mus = np.tile(MU, len(vas))
        score = np.where(np.isfinite(tm), lat - 0.15 * np.abs(np.log(mus / 0.6)) - 0.5 * np.abs(tbs[cb] - TF), -np.inf)
        score = np.where(lat > RB2_POLE_LAT, score + 10.0, score)
        kind = "none"
        if np.isfinite(score).any():
            c = int(np.argmax(score)); j = cb[c]; mu = float(mus[c]); vm = VM[c]
            kind = "pole" if lat[c] > RB2_POLE_LAT else "dwarf"
            _, _, _, mp = _rb2_back(P[i][None].copy(), -vm[None], Tmax, MET_RS, record=True)
        else:
            j = int(np.argmin(np.abs(tbs - TF))); mp = None
            for mu in (0.6, 0.4, 1.0, 0.25, 1.6):
                vm = V[i] + (V[i] - vas[j]) / mu
                tw, _, _, pw = _rb2_back(P[i][None].copy(), -vm[None], Tmax, MET_RS, record=True, bounce=False)
                if np.isfinite(tw[0]):
                    kind, mp = "wall", pw
                    break
        _, _, _, bp = _rb2_back(P[i][None].copy(), -vas[j][None], TF + RB2_SPIT_WIN, rs, record=True)
        bt, bx = _rb2_path(t_hit, bp); bx[-1] = P[i]
        met = None
        if mp is not None:
            mt, mx = _rb2_path(t_hit, mp); mx[-1] = P[i]
            met = dict(ts=mt, xs=mx, mu=mu, kind=kind, hist=[], on=False, x=mx[0].copy())
        out.append(dict(bts=bt, bxs=bx, met=met))
    return out

def _rb2_followups(res):
    """The FOLLOW-UP pincers for a phase 2 plan (worker): each body's rail state half an orbit (+n) after the impacts, the
    farm's pincer search; stored as res[i]["fu"] (None = no shot found / not in the ring plane)."""
    ss = dict(t=RB2_T0, itime=0.0)
    ss["rp"], ss["rv"] = ros_init()
    ss["cp"], ss["cw"] = co_init()
    t_hit = RB2_T0 + RB2_LEAD_SIM + RB2_FLIGHT
    rail0 = rail_predict_at(ss, [t_hit])[t_hit][0]
    Ts = {}
    for i in range(len(rail0)):
        if res[i] is None or res[i].get("fu"):
            continue
        per = 2.0 * math.pi * float(np.linalg.norm(rail0[i])) ** 1.5
        n = 0
        while (n + 0.5) * per < FU_MIN:
            n += 1
        Ts[i] = [t_hit + (k + 0.5) * per * f for k in (n, n + 1) for f in FU_JIT]
    if not Ts:
        return res
    rails = rail_predict_at(ss, [T for ts in Ts.values() for T in ts])
    for i, ts in Ts.items():
        cand = {}
        for T in ts:
            q = rails[T][0][i]
            if abs(float(q[1])) < 0.1 and R_IN < math.hypot(float(q[0]), float(q[2])) < R_OUT:
                cand[T] = (rails[T][0][i], rails[T][1][i])
        res[i]["fu"] = None
        for T in sorted(cand):
            res[i]["fu"] = _fm_search({T: cand[T]}, t_hit + FU_READY)
            if res[i]["fu"] is not None:
                break
    return res

def _glass_fly(x, v, T, rad, h=LOOP_H, record=False):
    """Batch forward leapfrog in dwarf + box gravity WITH the glass (reflect, x BOX_BOUNCE): end state, min radius, path."""
    n = max(1, int(math.ceil(T / h))); h = T / n
    lim = BOX_H - rad
    rmin = np.linalg.norm(x, axis=-1)
    path = [(0.0, x[0].copy())] if record else None
    for k in range(n):
        v = v + 0.5 * h * _rb2_acc(x); x = x + h * v; v = v + 0.5 * h * _rb2_acc(x)
        q, qv = x @ BOX_AX.T, v @ BOX_AX.T
        out = np.abs(q) > lim
        if out.any():
            q = np.where(out, np.sign(q) * lim, q)
            qv = np.where(out & (qv * q > 0.0), -BOX_BOUNCE * qv, qv)
            x, v = q @ BOX_AX, qv @ BOX_AX
        rmin = np.minimum(rmin, np.linalg.norm(x, axis=-1))
        if record:
            path.append(((k + 1) * h, x[0].copy()))
    return x, v, rmin, path

def _loop_shoot(X1, v0, P2, tau, rad):
    """Newton shooting X1 -> P2 in tau through the glass. (v at X1, the recorded path, v on arrival) or None."""
    vt = v0.copy()
    for _ in range(16):
        e = 1e-4
        X = np.stack([vt, vt + [e, 0, 0], vt + [0, e, 0], vt + [0, 0, e]])
        xe, _, _, _ = _glass_fly(np.stack([X1] * 4), X, tau, rad)
        err = xe[0] - P2
        if np.linalg.norm(err) < 2e-4:
            break
        J = np.stack([(xe[1] - xe[0]) / e, (xe[2] - xe[0]) / e, (xe[3] - xe[0]) / e], 1)
        try:
            step = np.linalg.solve(J, err)
        except np.linalg.LinAlgError:
            return None
        vt = vt - step / max(1.0, np.linalg.norm(step) / 0.5)
        if not np.isfinite(vt).all() or np.linalg.norm(vt) > 12.0:
            return None
    xa, va, rmin, path = _glass_fly(X1[None].copy(), vt[None].copy(), tau, rad, record=True)
    if np.linalg.norm(xa[0] - P2) > 1e-3 or rmin[0] < DW_R + rad + 0.03:
        return None
    ts = np.array([p[0] for p in path]); xs = np.array([p[1] for p in path]); xs[-1] = P2
    return vt, (ts, xs), va[0]

def _brake_pincer(P, V, v_arr, T, t_ready):
    """THE LOOP's hit 2 as meteors that do the work: from the meteor velocities that really trace back to the dwarf in
    the window (few, out at the ring's edge), the PAIR (or one) whose masses make mu_A u_A + mu_B u_B = V - v_arr
    (u = meteor velocity - V), each in flight >= LOOP_BRAKE_FLY. (mets, fit score) or None."""
    D = V - v_arr
    dn = max(float(np.linalg.norm(D)), 1e-9)
    U = np.concatenate([_fib_dirs(160) * m for m in LOOP_BRAKE_U])
    VM = V[None] + U
    tm, Lm, _, _ = _rb2_back(np.repeat(P[None], len(VM), 0), -VM, T - t_ready, MET_RS)
    ok = np.isfinite(tm) & (tm >= LOOP_BRAKE_FLY)
    if not ok.any():
        return None
    Ur, VMr = U[ok], VM[ok]
    lat = np.abs(Lm[ok, 1]) / np.maximum(np.linalg.norm(Lm[ok], axis=1), 1e-9)
    pole = lat > RB2_POLE_LAT
    best = None
    for a in range(len(Ur)):
        for b in range(a, len(Ur)):
            A = np.stack([Ur[a], Ur[b]], 1) if b > a else Ur[a][:, None]
            mu, *_ = np.linalg.lstsq(A, D, rcond=None)
            if (mu < LOOP_BRAKE_MU[0]).any() or (mu > LOOP_BRAKE_MU[1]).any():
                continue
            err = float(np.linalg.norm(A @ mu - D)) / dn
            if err > LOOP_BRAKE_ERR:
                continue
            sc = (0.5 if b > a else 0.0) + 0.5 * (pole[a] + pole[b]) - 3.0 * err - 0.2 * float(mu.sum()) \
                - (0.3 * abs(math.log(mu[0] / mu[1])) if b > a else 0.0)
            if best is None or sc > best[0]:
                best = (sc, [a, b] if b > a else [a])
    if best is None:
        return None
    mets = []
    for k in best[1]:
        _, _, _, mp = _rb2_back(P[None].copy(), -VMr[k][None], T - t_ready, MET_RS, record=True)
        mt, mx = _rb2_path(T, mp); mx[-1] = P
        mets.append(dict(ts=mt, xs=mx, mu=1.0, kind="pole" if pole[k] else "dwarf", hist=[], on=False, x=mx[0].copy()))
    return mets, best[0]

def _met_shot(P, V, v_arr, t_hit, t_ready):
    """ONE capture meteor at P, t_hit: the body arrives with v_arr, the merge leaves at exactly V. Poles preferred."""
    MU = np.geomspace(0.15, 3.0, 12)
    VM = V[None] + (V[None] - v_arr[None]) / MU[:, None]
    tm, Lm, _, _ = _rb2_back(np.repeat(P[None], len(VM), 0), -VM, t_hit - t_ready, MET_RS)
    lat = np.abs(Lm[:, 1]) / np.maximum(np.linalg.norm(Lm, axis=1), 1e-9)
    sc = np.where(np.isfinite(tm), lat + 10.0 * (lat > RB2_POLE_LAT) - 0.15 * np.abs(np.log(MU / 0.6)), -np.inf)
    if np.isfinite(sc).any():
        c = int(np.argmax(sc)); vm = VM[c]
        kind = "pole" if lat[c] > RB2_POLE_LAT else "dwarf"
        _, _, _, mp = _rb2_back(P[None].copy(), -vm[None], t_hit - t_ready, MET_RS, record=True)
    else:
        kind = None
        for mu in (0.6, 0.4, 1.0, 0.25, 1.6):
            vm = V + (V - v_arr) / mu
            tw, _, _, mp = _rb2_back(P[None].copy(), -vm[None], t_hit - t_ready, MET_RS, record=True, bounce=False)
            if np.isfinite(tw[0]):
                kind = "wall"
                break
        if kind is None:
            return None
    mt, mx = _rb2_path(t_hit, mp); mx[-1] = P
    return dict(ts=mt, xs=mx, mu=1.0, kind=kind, hist=[], on=False, x=mx[0].copy())

def _rb2_plan_all(P, V, rad, t_hit, t_ready):
    """The whole phase 2 plan (worker): THE LOOPS for the outer ring-plane bodies, the spit + hit 1 for everyone, hit 2 of
    each loop, then the follow-up pincers for the rest."""
    ss = dict(t=RB2_T0, itime=0.0)
    ss["rp"], ss["rv"] = ros_init()
    ss["cp"], ss["cw"] = co_init()
    loops = {}
    for i in range(len(P)):
        a = float(np.linalg.norm(P[i]))
        if abs(float(P[i][1])) > 0.05 or a < LOOP_A_MIN:
            continue
        for tau in LOOP_TAUS:
            T2 = t_hit + tau
            r2 = rail_predict_at(ss, [T2])[T2]
            P2, V2 = r2[0][i], r2[1][i]
            nh = np.cross(P2, V2); nh /= max(float(np.linalg.norm(nh)), 1e-9)
            ang = -(math.pi + tau * a ** -1.5)
            X1 = P2 * math.cos(ang) + np.cross(nh, P2) * math.sin(ang) + nh * float(nh @ P2) * (1.0 - math.cos(ang))
            if (np.abs(BOX_AX @ X1) > BOX_H - rad[i]).any():
                continue
            v0 = (P2 - X1) / tau - 0.5 * tau * BOX_G * BOX_AX[2]
            sol = _loop_shoot(X1, v0, P2, tau, float(rad[i]))
            br = _brake_pincer(P2, V2, sol[2], T2, t_hit + FU_READY) if sol is not None else None
            if br is None:
                continue
            dv = float(np.linalg.norm(V2 - sol[2]))
            print(f"loop: moon {i} tau {tau}: arrives {np.linalg.norm(sol[2]):.2f} vs rail {np.linalg.norm(V2):.2f}, brake "
                  f"{dv:.2f} by {[m['kind'] for m in br[0]]}", file=sys.stderr)
            if i not in loops or dv < loops[i][8]:
                loops[i] = (X1, sol[0], P2, V2, T2, sol[1], br[0], tau, dv)
        print(f"loop: moon {i} (a {a:.2f}) -> {'%.1f sim units of flight' % loops[i][7] if i in loops else 'no loop found'}", file=sys.stderr)
    PP, VV = P.copy(), V.copy()
    for i, L in loops.items():
        PP[i], VV[i] = L[0], L[1]
    res = _rb2_search(PP, VV, rad, t_hit, t_ready)
    for i, L in list(loops.items()):
        X1, vt, P2, V2, T2, path, br, _, _ = L
        if res[i] is None:
            res[i] = _rb2_search(P[i:i + 1], V[i:i + 1], rad[i:i + 1], t_hit, t_ready)[0]
            print(f"loop: moon {i} -- no capture shot, back to the single hit", file=sys.stderr)
            continue
        res[i]["fu"] = dict(T=T2, P=P2, mets=br, loop=path, x1=X1)
    return _rb2_followups(res)

def _rb2_inputs():
    """The rebirth's fixed starting point: fresh rails at RB2_T0 -> every body's rail state at t_hit."""
    ss = dict(t=RB2_T0, itime=0.0)
    ss["rp"], ss["rv"] = ros_init()
    ss["cp"], ss["cw"] = co_init()
    t_hit = RB2_T0 + RB2_LEAD_SIM + RB2_FLIGHT
    P, V, rad = rail_predict(ss, t_hit)
    return P, V, rad, t_hit, RB2_T0 + RB2_READY_SIM

def _rb2_cache():
    import hashlib
    with open(os.path.abspath(__file__), "rb") as fh:
        key = hashlib.sha1(fh.read()).hexdigest()[:12]
    return os.path.join(RB2_DIR, f"plan_{key}.pkl")

def rb2_precompute(s):
    """Search the shots in the background (own process) unless this sim.py version's plan is already saved."""
    path = _rb2_cache()
    if os.path.isdir(RB2_DIR):
        for f in os.listdir(RB2_DIR):
            if f.startswith("plan_") and os.path.join(RB2_DIR, f) != path and not f.endswith(".part"):
                os.remove(os.path.join(RB2_DIR, f))
    if os.path.exists(path) or (s.get("rb2_proc") is not None and s["rb2_proc"].poll() is None):
        return
    os.makedirs(RB2_DIR, exist_ok=True)
    P, V, rad, t_hit, t_ready = _rb2_inputs()
    fin = os.path.join(RB2_DIR, "in.npz")
    np.savez(fin, P=P, V=V, rad=rad, t_hit=t_hit, t_ready=t_ready)
    s["rb2_proc"] = subprocess.Popen([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rb2_worker.py"),
                                      fin, path], stdout=subprocess.DEVNULL, stderr=sys.stderr)
    _solo_job("plan", s["rb2_proc"])
    s["rb2_t0"] = time.time()
    print("rebirth: searching the phase 2 shots in the background (saved once per sim.py version)", file=sys.stderr)

def rb2_plan(api, s):
    """Start PHASE 2 at the hole's evaporation: restart the rails (and the orbit clock) at RB2_T0, hidden; the shots come
    from the saved plan (rb2_collect)."""
    rb = s["rb"]
    s["t"] = RB2_T0
    s["rp"], s["rv"] = ros_init()
    s["cp"], s["cw"] = co_init()
    s["free"] = {i: dict(x=np.zeros(3), v=np.zeros(3), gone=True, rad=0.0) for i in range(N_MOONS)}
    s["mt_list"], s["mt_search"] = [], None
    P, V, rad, t_hit, t_ready = _rb2_inputs()
    rb.update(t_hit=t_hit, t_ready=t_ready, P=P, V=V, rad=rad, res=None, flashes=[], hit_done=False,
              spat=np.zeros(len(P), bool), cache=_rb2_cache())
    rb2_precompute(s)

def rb2_collect(rb):
    """The saved plan, once it exists (None while the background search runs)."""
    if rb.get("res") is not None:
        return rb["res"]
    if not os.path.exists(rb["cache"]):
        return None
    with open(rb["cache"], "rb") as fh:
        res = pickle.load(fh)
    kinds = [r["met"]["kind"] if r and r["met"] else ("snap" if r is None else "none") for r in res]
    print(f"rebirth: PHASE 2 from the saved plan -- meteors {kinds}", file=sys.stderr)
    rb["res"] = res
    rb["fus"] = [dict(r["fu"], i=i, done=False) for i, r in enumerate(res) if r and r.get("fu")]
    rb["fu_flashes"] = []
    print(f"rebirth: {len(rb['fus'])} follow-up pincers planned (moons {[f['i'] for f in rb['fus']]})", file=sys.stderr)
    return res

def rail_predict_at(s, times):
    """Rail states (P, V, rad) at several future sim times, one forward pass of the hidden rails."""
    ss = dict(rp=s["rp"].copy(), rv=s["rv"].copy(), cp=np.array(s["cp"], float).copy(), cw=np.array(s["cw"], float).copy(), t=s["t"])
    out = {}
    for tg in sorted(set(times)):
        while ss["t"] < tg - 1e-9:
            hh = min(0.005, tg - ss["t"])
            ros_step(ss, hh); co_step(ss, hh); ss["t"] += hh
        it = s["itime"] + (tg - s["t"]) / max(sim_rate(), 1e-6)
        P0 = moon_positions(tg, ss["rp"], it, (ss["cp"], ss["cw"]))
        s2 = dict(rp=ss["rp"].copy(), rv=ss["rv"].copy(), cp=ss["cp"].copy(), cw=ss["cw"].copy(), t=ss["t"])
        ros_step(s2, 1e-3); co_step(s2, 1e-3)
        P1 = moon_positions(tg + 1e-3, s2["rp"], it, (s2["cp"], s2["cw"]))
        out[tg] = (P0[0, :, :3].astype(np.float64), (P1[0, :, :3] - P0[0, :, :3]).astype(np.float64) / 1e-3)
    return out

def _rs_fwd_free(x, v, T, rad, h=RB2_H, crawl0=0.0):
    """The loose moon's own future (dwarf + box gravity, glass bounces), recorded; tau_crash = when it would hit the dwarf."""
    path, tt = [(0.0, x.copy(), v.copy())], 0.0
    cst = dict(crawl=crawl0)
    while tt < T:
        v = v + 0.5 * h * _rb2_acc(x); x = x + h * v; v = v + 0.5 * h * _rb2_acc(x); tt += h
        q, qv = BOX_AX @ x, BOX_AX @ v
        lim = BOX_H - rad
        for k in range(3):
            if abs(q[k]) > lim[k]:
                q[k] = math.copysign(lim[k], q[k])
                if qv[k] * q[k] > 0.0:
                    qv[k] = -BOX_BOUNCE * qv[k]
        x, v = BOX_AX.T @ q, BOX_AX.T @ qv
        x, v, _ = dwarf_contact(x, v, rad, cst, h)
        path.append((tt, x.copy(), v.copy()))
    return path, np.inf

def _rs_search(x0, v0, rad, t0, T1s, TAUs, rails, t_ready):
    """One rescue: the best (T1, tau) whose transfer path is clean and whose two meteors both come out of the dwarf."""
    MU = np.geomspace(0.15, 3.0, 12)
    fpath, tcr = _rs_fwd_free(x0.copy(), v0.copy(), max(T1s) - t0 + 1e-6, rad)
    ft = np.array([p[0] for p in fpath]) + t0
    best = None
    for T1 in T1s:
        if T1 - t0 >= tcr:
            continue
        k = int(np.searchsorted(ft, T1)); k = min(k, len(fpath) - 1)
        xf, vf = fpath[k][1], fpath[k][2]
        for tau in TAUs:
            T2 = T1 + tau
            P2, V2 = rails[T2]
            vt = (P2 - xf) / tau - 0.5 * tau * _rb2_acc(0.5 * (P2 + xf))
            for _ in range(12):
                e = 1e-4
                X = np.stack([vt, vt + [e, 0, 0], vt + [0, e, 0], vt + [0, 0, e]])
                xe, _, _ = _rb2_fly(np.stack([xf] * 4), X, tau)
                J = np.stack([(xe[1] - xe[0]) / e, (xe[2] - xe[0]) / e, (xe[3] - xe[0]) / e], 1)
                try:
                    vt = vt - np.linalg.solve(J, xe[0] - P2)
                except np.linalg.LinAlgError:
                    break
            xa, va, rmin = _rb2_fly(xf[None].copy(), vt[None].copy(), tau)
            if np.linalg.norm(xa[0] - P2) > 1e-3 or rmin[0] < DW_R + rad:
                continue
            xc, vc, inside = xf.copy(), vt.copy(), True
            for _ in range(int(tau / 0.02)):
                vc = vc + 0.01 * _rb2_acc(xc); xc = xc + 0.02 * vc; vc = vc + 0.01 * _rb2_acc(xc)
                if (np.abs(BOX_AX @ xc) > BOX_H - rad).any():
                    inside = False
                    break
            if not inside:
                continue
            sc, picks = 0.0, []
            for (xi, vin, vout, Ti) in ((xf, vf, vt, T1), (P2, va[0], V2, T2)):
                VM = vout[None] + (vout[None] - vin[None]) / MU[:, None]
                tm, Lm, _, _ = _rb2_back(np.repeat(xi[None], len(MU), 0), -VM, Ti - t_ready, MET_RS)
                lat = np.abs(Lm[:, 1]) / np.maximum(np.linalg.norm(Lm, axis=1), 1e-9)
                score = np.where(np.isfinite(tm), lat - 0.15 * np.abs(np.log(MU / 0.6)) + 10.0 * (lat > RB2_POLE_LAT)
                                 + _rb2_back.bonus, -np.inf)
                c = int(np.argmax(score))
                if not np.isfinite(score[c]):
                    break
                sc += score[c]; picks.append((xi, VM[c], Ti, MU[c], lat[c] > RB2_POLE_LAT))
            if len(picks) == 2 and (best is None or sc > best[0]):
                best = (sc, T1, T2, vt, picks)
    if best is None:
        return None
    _, T1, T2, vt, picks = best
    ts_f = [p for p in fpath if p[0] + t0 <= T1]
    free_ts = np.array([p[0] + t0 for p in ts_f] + [T1]); free_xs = np.array([p[1] for p in ts_f] + [ts_f[-1][1]])
    k = int(np.searchsorted(ft, T1)); k = min(k, len(fpath) - 1); free_xs[-1] = fpath[k][1]
    tp, x, w, tt, h = [(T1, free_xs[-1].copy())], free_xs[-1].copy(), vt.copy(), 0.0, RB2_H
    n = max(1, int(math.ceil((T2 - T1) / h))); h = (T2 - T1) / n
    for _ in range(n):
        w = w + 0.5 * h * _rb2_acc(x); x = x + h * w; w = w + 0.5 * h * _rb2_acc(x); tt += h
        tp.append((T1 + tt, x.copy()))
    tr_ts = np.array([p[0] for p in tp]); tr_xs = np.array([p[1] for p in tp]); tr_xs[-1] = rails[T2][0]
    mets = []
    for (xi, vm, Ti, mu, pole) in picks:
        _, _, _, mp = _rb2_back(xi[None].copy(), -vm[None], Ti - t_ready, MET_RS, record=True)
        mt, mx = _rb2_path(Ti, mp); mx[-1] = xi
        mets.append(dict(ts=mt, xs=mx, mu=float(mu), kind=_launch_kind(pole), hist=[], on=False, x=mx[0].copy()))
    return dict(T1=T1, T2=T2, free=(free_ts, free_xs), transfer=(tr_ts, tr_xs), mets=mets)

SNIPER_DIR = os.path.expanduser("~/.cache/shader-backdrop/sniper")
SNIPER_TOP = 6
SNIPER_EXPLORE = 0.15
SNIPER_BUDGET_S = 2.0
SNIPER_N1 = (2, 12)
SNIPER_N2 = (1, 8)

def sniper_features(x0, v0, P0, V0, beat):
    """The situation: the loose moon (x0, v0) vs its rail slot now (P0, V0), and the beat length (sim units)."""
    r = np.linalg.norm(x0); er = x0 / max(r, 1e-9)
    vr = float(v0 @ er); vy = float(v0[1]); vt = float(np.linalg.norm(v0 - vr * er))
    E = 0.5 * float(v0 @ v0) - 1.0 / r
    L = float(np.linalg.norm(np.cross(x0, v0)))
    rs = np.linalg.norm(P0)
    lag = math.atan2(float(np.cross(x0, P0)[1]), float(x0 @ P0))
    return np.array([r, float(x0[1]), vr, vt, vy, E, L, lag, float(np.linalg.norm(x0 - P0)), rs, beat])

def _sniper_phi(F, C):
    """Design rows: situation F (n, 11) x candidate C (n, 3: dt1, tau, downbeat) -> polynomial features."""
    cross = (F[:, :, None] * C[:, None, :2]).reshape(len(F), -1)
    return np.concatenate([np.ones((len(F), 1)), F, C, cross, C[:, :2] ** 2], 1)

def sniper_model_load():
    try:
        d = np.load(os.path.join(SNIPER_DIR, "model.npz"))
        return {k: d[k] for k in d.files}
    except (OSError, ValueError):
        return None

def sniper_rank(model, f, cands):
    """Expected value of each candidate (P x score); no model yet -> the plain order (nearest first)."""
    C = np.array([[c[3], c[1] - c[0], c[2]] for c in cands])
    if model is None:
        return np.argsort(C[:, 0] + 0.3 * C[:, 1])
    X = (_sniper_phi(np.repeat(f[None], len(C), 0), C) - model["mu"]) / model["sd"]
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ model["ws"], -30, 30)))
    return np.argsort(-(p * np.maximum(X @ model["wq"], 0.05)))

def _rs_eval(fpath, ft, rad, T1, T2, P2, V2, t_ready, down):
    """One candidate: the transfer from the moon's predicted state at T1 to its rail point at T2, and the two meteors out of
    the dwarf by momentum. -> None, or (score, vt, picks, info)."""
    MU = np.geomspace(0.15, 3.0, 12)
    k = int(np.searchsorted(ft, T1)); k = min(k, len(fpath) - 1)
    xf, vf = fpath[k][1], fpath[k][2]
    tau = T2 - T1
    vt = (P2 - xf) / tau - 0.5 * tau * _rb2_acc(0.5 * (P2 + xf))
    for _ in range(12):
        e = 1e-4
        X = np.stack([vt, vt + [e, 0, 0], vt + [0, e, 0], vt + [0, 0, e]])
        xe, _, _ = _rb2_fly(np.stack([xf] * 4), X, tau)
        J = np.stack([(xe[1] - xe[0]) / e, (xe[2] - xe[0]) / e, (xe[3] - xe[0]) / e], 1)
        try:
            vt = vt - np.linalg.solve(J, xe[0] - P2)
        except np.linalg.LinAlgError:
            return None
    xa, va, rmin = _rb2_fly(xf[None].copy(), vt[None].copy(), tau)
    ok = np.isfinite(xa).all() and np.linalg.norm(xa[0] - P2) <= 1e-3 and rmin[0] >= DW_R + rad
    if ok:
        xc, vc = xf.copy(), vt.copy()
        for _ in range(int(tau / 0.02)):
            vc = vc + 0.01 * _rb2_acc(xc); xc = xc + 0.02 * vc; vc = vc + 0.01 * _rb2_acc(xc)
            if (np.abs(BOX_AX @ xc) > BOX_H - rad).any():
                ok = False
                break
    tr = None
    if not ok:
        sol = _loop_shoot(xf, (P2 - xf) / tau - 0.5 * tau * BOX_G * BOX_AX[2], P2, tau, rad)
        if sol is None:
            return None
        vt, tr, va1 = sol
        va = va1[None]
    picks, npol, nb, fl = [], 0, 0, 0.0
    for (xi, vin, vout, Ti) in ((xf, vf, vt, T1), (P2, va[0], V2, T2)):
        VM = vout[None] + (vout[None] - vin[None]) / MU[:, None]
        tm, Lm, _, _ = _rb2_back(np.repeat(xi[None], len(MU), 0), -VM, Ti - t_ready, MET_RS)
        bn = _rb2_back.bounces.copy()
        lat = np.abs(Lm[:, 1]) / np.maximum(np.linalg.norm(Lm, axis=1), 1e-9)
        sc = np.where(np.isfinite(tm), lat - 0.15 * np.abs(np.log(MU / 0.6)) + 10.0 * (lat > RB2_POLE_LAT)
                      + 0.3 * np.minimum(bn, 3) + _rb2_back.bonus, -np.inf)
        c = int(np.argmax(sc))
        if not np.isfinite(sc[c]):
            return None
        picks.append((xi, VM[c], Ti, MU[c], lat[c] > RB2_POLE_LAT))
        npol += int(lat[c] > RB2_POLE_LAT); nb += int(bn[c]); fl += float(tm[c])
    score = 0.5 * npol + 0.25 * min(nb, 4) + 0.4 * min(fl / 2.5, 1.0) + 0.6 * down
    return score, vt, picks, dict(npol=npol, bounces=nb, flight=fl, down=down, tr=tr)

def _rs_build(fpath, ft, t0, T1, T2, vt, picks, P2, t_ready, tr=None):
    ts_f = [p for p in fpath if p[0] + t0 <= T1]
    free_ts = np.array([p[0] + t0 for p in ts_f] + [T1]); free_xs = np.array([p[1] for p in ts_f] + [ts_f[-1][1]])
    k = int(np.searchsorted(ft, T1)); k = min(k, len(fpath) - 1); free_xs[-1] = fpath[k][1]
    tp, x, w, tt, h = [(T1, free_xs[-1].copy())], free_xs[-1].copy(), vt.copy(), 0.0, RB2_H
    n = max(1, int(math.ceil((T2 - T1) / h))); h = (T2 - T1) / n
    if tr is not None:
        tr_ts, tr_xs = tr[0] + T1, tr[1].copy()
    else:
        for _ in range(n):
            w = w + 0.5 * h * _rb2_acc(x); x = x + h * w; w = w + 0.5 * h * _rb2_acc(x); tt += h
            tp.append((T1 + tt, x.copy()))
        tr_ts = np.array([p[0] for p in tp]); tr_xs = np.array([p[1] for p in tp])
    tr_xs[-1] = P2
    mets = []
    for (xi, vm, Ti, mu, pole) in picks:
        _, _, _, mp = _rb2_back(xi[None].copy(), -vm[None], Ti - t_ready, MET_RS, record=True)
        mt, mx = _rb2_path(Ti, mp); mx[-1] = xi
        mets.append(dict(ts=mt, xs=mx, mu=float(mu), kind=_launch_kind(pole), hist=[], on=False, x=mx[0].copy()))
    return dict(T1=T1, T2=T2, free=(free_ts, free_xs), transfer=(tr_ts, tr_xs), mets=mets)

def sniper_search(x0, v0, rad, t0, cands, rails, P0V0, beat, t_ready, model, rng, budget_s=SNIPER_BUDGET_S, all_=False):
    """The SNIPER: candidates (T1, T2, downbeat, dt1) on the beat, evaluated in the model's order (all_ = every one: the
    trainer). Returns (the best shot or None, the training rows [features, candidate, success, score])."""
    import time as _t
    t_start = _t.time()
    f = sniper_features(x0, v0, P0V0[0], P0V0[1], beat)
    Tmax = max(c[0] for c in cands) if cands else t0
    fpath, _ = _rs_fwd_free(x0.copy(), v0.copy(), Tmax - t0 + 1e-6, rad)
    ft = np.array([p[0] for p in fpath]) + t0
    order = list(sniper_rank(model, f, cands))
    if not all_:
        top, rest = order[:SNIPER_TOP], order[SNIPER_TOP:]
        nex = max(1, int(round(SNIPER_EXPLORE * len(order)))) if rest else 0
        ex = list(rng.choice(rest, size=min(nex, len(rest)), replace=False)) if nex else []
        order = top + ex + [i for i in rest if i not in ex]
    rows, best = [], None
    for n, j in enumerate(order):
        T1, T2, down, dt1 = cands[j]
        r = _rs_eval(fpath, ft, rad, T1, T2, rails[T2][0], rails[T2][1], t_ready, down)
        rows.append((f, np.array([dt1, T2 - T1, down]), 0.0 if r is None else 1.0, 0.0 if r is None else r[0]))
        if r is not None and (best is None or r[0] > best[0]):
            best = (r[0], T1, T2, r[1], r[2], r[3])
        if not all_ and n + 1 >= SNIPER_TOP + 1 and best is not None and _t.time() - t_start > 0.3:
            break
        if not all_ and _t.time() - t_start > budget_s:
            break
    if best is None:
        return None, rows
    sc, T1, T2, vt, picks, info = best
    res = _rs_build(fpath, ft, t0, T1, T2, vt, picks, rails[T2][0], t_ready, tr=info.pop("tr", None))
    res["info"] = dict(info, score=sc, evaluated=len(rows), secs=round(_t.time() - t_start, 2))
    return res, rows

def sniper_log(rows, tag):
    if not rows:
        return
    os.makedirs(SNIPER_DIR, exist_ok=True)
    F = np.array([r[0] for r in rows]); C = np.array([r[1] for r in rows])
    Y = np.array([[r[2], r[3]] for r in rows])
    path = os.path.join(SNIPER_DIR, f"{tag}_{int(time.time() * 1000)}_{os.getpid()}.npz")
    np.savez(path + ".part.npz", F=F, C=C, Y=Y)
    os.replace(path + ".part.npz", path)

def sniper_beats(s, t0, n=24):
    """Upcoming beats in SIM time from the music clock: [(t_sim, downbeat)]; a steady 0.5 s grid without a lock."""
    rate = max(sim_rate(), 1e-6)
    c = mt_clock(s)
    lock = c["conf"] >= 0.2 and 0.2 < c["tb"] < 2.0
    tb = c["tb"] if lock else 0.5
    if lock:
        it_next = c["it"] + (math.ceil(c["bc"]) - c["bc"]) * tb
        while it_next < s["itime"]:
            it_next += tb
        m0 = int(round(c["bar"] * 4.0 + (it_next - c["it"]) / tb)) % 4
    else:
        it_next, m0 = s["itime"] + tb, -99
    out = [(s["t"] + (it_next + k * tb - s["itime"]) * rate, (m0 + k) % 4 == 0 if lock else False) for k in range(n)]
    return out, tb * rate

def sniper_cands(beats, t0, t_ready):
    """(T1, T2, downbeat of T2, dt1) pairs: T1 on beat n1 (after the launches can still fit), T2 = n2 beats later."""
    out = []
    for i in range(SNIPER_N1[0], min(SNIPER_N1[1], len(beats))):
        T1 = beats[i][0]
        if T1 < t_ready + 0.15:
            continue
        for d in range(SNIPER_N2[0], SNIPER_N2[1] + 1):
            if i + d >= len(beats):
                break
            T2, dn = beats[i + d]
            if 0.5 <= T2 - T1 <= 3.2:
                out.append((T1, T2, 1.0 if dn else 0.0, T1 - t0))
    return out

def write_moons(s):
    """The moons texture: rows 0-1 = moon_positions / free_apply, row 2 = each moon's last hit (iTime; the jelly wobbles)."""
    w = np.zeros((1, N_MOONS, 4), dtype="f4")
    w[0, :, 0] = -100.0
    for i, tt in s.get("wob", {}).items():
        w[0, i, 0] = tt
    s["moons"].write(np.concatenate([s["moons_np"], w]).astype("f4").tobytes())

def stamp_hit(s, p):
    """A hit near moon p: remember when (the jelly's wobble)."""
    mp = s.get("moons_np")
    if mp is None:
        return
    d = np.linalg.norm(mp[0, :, :3] - np.asarray(p)[None], axis=1) - mp[0, :, 3]
    i = int(np.argmin(d))
    if d[i] < 0.15:
        s.setdefault("wob", {})[i] = s["itime"]

def add_puff(s, p, k=1.0, ride=False):
    """A hit on a moon SHEDS ring material at p for PUFF_T (sim.comp K_FORCE reads the 16 newest). ride: the debris cloud
    goes round WITH the ring at its orbit speed (a collision shot into a hole: it stays in the hole) instead of a moon's hose
    spraying a fixed spot that the ring flows past (which smears it along the orbit)."""
    stamp_hit(s, p)
    s.setdefault("puffs", []).append((math.hypot(p[0], p[2]), math.atan2(p[2], p[0]), RB2_PUFF_RATE * k, float(p[1]),
                                      s["t"] + PUFF_T, s["t"], bool(ride)))

def puff_flush(s):
    pl = [q for q in s.get("puffs", []) if q[4] > s["t"]][-16:]
    s["puffs"] = pl
    riding = any(q[6] for q in pl)
    key = len(pl), (pl[-1][4] if pl else 0.0)
    if key == s.get("puff_key") and not riding:
        return
    s["puff_key"] = key
    pf = np.zeros((1, 16, 4), dtype="f4")
    sg = s.get("ring_sg", 1.0)
    for k, q in enumerate(pl):
        th = q[1] + (sg * q[0] ** -1.5 * (s["t"] - q[5]) if q[6] else 0.0)
        pf[0, k] = (q[0], th, q[2], q[3])
    s["rb_puff_tex"].write(pf.tobytes())

def _fib_dirs(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n); th = math.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)

def _fm_search(rails, t_ready):
    """A PINCER: for a rail point (P, V) at T, two meteors hitting at the same instant whose pushes cancel -- u_B = -lam u_A
    with masses mu_A / mu_B = lam (mu_A u_A + mu_B u_B = 0: the moon keeps exactly V) -- both coming out of the dwarf (poles
    preferred, bank shots). The best over the candidate moments, directions, strengths and mass ratios."""
    U = np.concatenate([_fib_dirs(24) * m for m in FM_U])
    best = None
    for T, (P, V) in rails.items():
        u = U * np.linalg.norm(V)
        LAM = np.array(FM_LAM)
        VA = V[None] + u
        VB = (V[None, None] - LAM[:, None, None] * u[None]).reshape(-1, 3)
        VM = np.concatenate([VA, VB])
        tm, Lm, _, _ = _rb2_back(np.repeat(P[None], len(VM), 0), -VM, T - t_ready, MET_RS)
        lat = np.abs(Lm[:, 1]) / np.maximum(np.linalg.norm(Lm, axis=1), 1e-9)
        sc = np.where(np.isfinite(tm), lat + 10.0 * (lat > RB2_POLE_LAT) + _rb2_back.bonus, -np.inf)
        n = len(U)
        sA, sB = sc[:n], sc[n:].reshape(len(LAM), n)
        both = sA[None] + sB - 0.3 * np.abs(np.log(LAM))[:, None]
        l, c = np.unravel_index(int(np.argmax(both)), both.shape)
        if np.isfinite(both[l, c]) and (best is None or both[l, c] > best[0]):
            jb = n + l * n + c
            best = (both[l, c], T, P, [VM[c], VM[jb]], [lat[c] > RB2_POLE_LAT, lat[jb] > RB2_POLE_LAT])
    if best is None:
        return None
    _, T, P, vms, pole = best
    mets = []
    for vm, pl in zip(vms, pole):
        _, _, _, mp = _rb2_back(P[None].copy(), -vm[None], T - t_ready, MET_RS, record=True)
        mt, mx = _rb2_path(T, mp); mx[-1] = P
        mets.append(dict(ts=mt, xs=mx, mu=1.0, kind=_launch_kind(pl), hist=[], on=False, x=mx[0].copy()))
    return dict(T=T, P=P, mets=mets)

def fm_target(api, s):
    """The band of the ring furthest below its resting mass: (radius, deficit fraction) or None. Reads K_PROF (cheap)."""
    if s["itime"] - s.get("prof_t", -99.0) >= FM_PROF_S or s.get("prof") is None:
        api.run(s["k"]["PROF"], ((NR + 63) // 64, 1, 1), images={0: s["prof_tex"]},
                samplers={"u_field": s["f"][s["cur"]], "u_phase": s["phase"]})
        s["prof"] = np.frombuffer(s["prof_tex"].read(), dtype="f4").reshape(NR, 4).astype(np.float64)
        s["prof_t"] = s["itime"]
    S, S0 = s["prof"][:, 0], s["prof"][:, 1]
    r = R_IN + (np.arange(NR) + 0.5) * (R_OUT - R_IN) / NR
    cap = s.get("prof_cap")
    cap = S0.copy() if cap is None else np.maximum(S, cap * math.exp(-(s["itime"] - s.get("prof_cap_t", s["itime"])) / FM_CAP_TAU))
    s["prof_cap"], s["prof_cap_t"] = cap, s["itime"]
    S0 = np.minimum(S0, cap)
    full = float((S * r).sum()) / max(float((s["prof"][:, 1] * r).sum()), 1e-9)
    nb = max(1, int(round((R_OUT - R_IN) / FM_BAND)))
    edges = np.linspace(R_IN, R_OUT, nb + 1)
    best = None
    for b in range(nb):
        m = (r >= edges[b]) & (r < edges[b + 1])
        m0 = float((S0[m] * r[m]).sum())
        if m0 < 1e-6:
            continue
        frac = max(0.0, m0 - float((np.minimum(S[m], S0[m]) * r[m]).sum())) / m0
        if frac >= FM_MIN_DEF and (best is None or frac * m0 > best[0]):
            best = (frac * m0, 0.5 * (edges[b] + edges[b + 1]), frac)
    if s["itime"] - s.get("prof_log_t", -99.0) > 30.0:
        s["prof_log_t"] = s["itime"]
        fr = [(max(0.0, float(((S0 - np.minimum(S, S0)) * r)[(r >= edges[b]) & (r < edges[b + 1])].sum()))
               / max(float((S0 * r)[(r >= edges[b]) & (r < edges[b + 1])].sum()), 1e-9), 0.5 * (edges[b] + edges[b + 1]))
              for b in range(nb)]
        w = max(fr)
        print(f"farm: ring mass {full:.2f} x rest; thinnest band r {w[1]:.2f} {100 * w[0]:.0f} % below"
              f"{' -- FULL, no farming' if full >= FM_FULL else ''}", file=sys.stderr)
    if full >= FM_FULL:
        return None
    return None if best is None else (best[1], best[2])

def rs_hole_angle(s, rstar):
    """The physical angle of the thinnest sector near rstar (K_PROF z; the row whose thinnest sector is furthest below rest)."""
    pr = s["prof"]
    r = R_IN + (np.arange(NR) + 0.5) * (R_OUT - R_IN) / NR
    m = np.abs(r - rstar) < 0.5 * FM_BAND
    if not m.any():
        return float(s["rng"].uniform(0.0, 2.0 * math.pi))
    j = np.flatnonzero(m)[int(np.argmax((pr[m, 1] - pr[m, 3]) / np.maximum(pr[m, 1], 1e-6)))]
    return float(pr[j, 2])

def fm_spin(s, rail):
    """+1 if the ring-plane moons (and the ring) go round toward +theta (theta = atan2(z, x)), else -1."""
    P, V = rail[0], rail[1]
    ly = [float(P[k, 2] * V[k, 0] - P[k, 0] * V[k, 2]) for k in range(len(P)) if abs(float(P[k, 1])) < 0.05]
    return 1.0 if (sum(ly) if ly else -1.0) < 0.0 else -1.0

CH_ON = True
CH_LONG = 3.55
CH_SIDE = 1.05
CH_BEATS = 16.0
CH_LEAD = 0.25
CH_NT = 48
CH_TRAIL_DT = 1.0 / 40.0
CH_LOCK = 0.6
CH_SW_N = 5
CH_SW_R = 0.09
CH_SW_STEER = 4.0
CH_SW_SEP = 0.05
CH_WELL_BEATS = 32.0
CH_MORPH = 2.0
CH_ROSE = 0.6
CH_SPLIT_EDGE, CH_SPLIT_INNER, CH_SPLIT_SWEEP, CH_SPLIT_WARP = 1.0, 0.20, 0.45, 0.40
CH_EDGE_X = 1.55
CH_HAWK_KICK = 0.25
CH_HAWK_TAU = 0.35
CH_HAWK_SPD = 1.5
CH_HAWK_EDGE = 1.3
CH_HAWK_PUSH = 0.6
CH_DASH_CAP = 0.9

CH_PERI = 0.30

def _ch_kepler(M, apo, sgn, ax):
    """A Kepler ellipse round the star (focus) in box (x, y): the far end at apo along axis ax (0 = x, 1 = y) on side sgn,
    the close pass CH_PERI on the other side; mean anomaly M (0 = the close pass)."""
    a = 0.5 * (apo + CH_PERI); e = (apo - CH_PERI) / (apo + CH_PERI); b = a * math.sqrt(1.0 - e * e)
    E = M
    for _ in range(8):
        E -= (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
    u, v = -a * (math.cos(E) - e) * sgn, b * math.sin(E) * sgn
    return (u, v) if ax == 0 else (v, u)

def chase_head(bc, w, side, rose):
    """The leader at beat count bc: the five pocket wells blended by w (box frame -> world)."""
    sp = 2.0 * math.pi * bc / CH_BEATS
    S, L = CH_SIDE, CH_LONG
    p0 = (S * math.cos(sp), L * math.sin(sp))
    p1 = (0.9 * S * math.sin(2.0 * sp), L * math.sin(sp))
    rr = 1.0 - rose * (0.5 - 0.5 * math.cos(3.0 * sp))
    p2 = (S * math.cos(sp) * rr, L * math.sin(sp) * rr)
    p3 = _ch_kepler(sp, CH_EDGE_X, side[0], 0)
    p4 = _ch_kepler(sp, L, side[1], 1)
    ws = max(float(sum(w)), 1e-6)
    x = (w[0] * p0[0] + w[1] * p1[0] + w[2] * p2[0] + w[3] * p3[0] + w[4] * p4[0]) / ws
    y = (w[0] * p0[1] + w[1] * p1[1] + w[2] * p2[1] + w[3] * p3[1] + w[4] * p4[1]) / ws
    bx, by, bz = BOX_AX
    return bx * x + by * y + bz * (0.07 * math.sin(1.5 * sp))

CH_WEAVE = True
CH_WV_H = 0.16
CH_WV_EDGE = 0.30
CH_WV_AHEAD = 8.0
CH_WV_RES = 32
CH_WV_MAXZ = 1.55
CH_WV_CLEAR = 0.07

RAILS_T = 4000.0
RAILS_DT = 0.02
RAILS_TOL = 0.02

def _rails_cache():
    return _rb2_cache().replace("plan_", "rails_").replace(".pkl", ".npy")

def _rails_bake():
    """(worker) Every moon's rail position from t = 0 every RAILS_DT up to RAILS_T: float32 (n, N_MOONS, 3)."""
    ss = dict(t=0.0, itime=0.0)
    ss["rp"], ss["rv"] = ros_init()
    ss["cp"], ss["cw"] = co_init()
    n = int(RAILS_T / RAILS_DT) + 1
    out = np.zeros((n, N_MOONS, 3), dtype="f4")
    h = 0.005
    for i in range(n):
        tg = i * RAILS_DT
        while ss["t"] < tg - 1e-9:
            hh = min(h, tg - ss["t"])
            ros_step(ss, hh); co_step(ss, hh); ss["t"] += hh
        out[i] = moon_positions(tg, ss["rp"], tg / max(sim_rate(), 1e-6), (ss["cp"], ss["cw"]))[0, :, :3]
    return out

def rails_table(s):
    """The baked rail table once it exists (starts the bake in the background the first time)."""
    if s.get("rails_tab") is not None:
        return s["rails_tab"]
    path = _rails_cache()
    if os.path.exists(path):
        try:
            s["rails_tab"] = np.load(path, mmap_mode="r")
            print(f"rails: the baked table is in ({len(s['rails_tab'])} steps)", file=sys.stderr)
            return s["rails_tab"]
        except Exception:
            pass
    job = s.get("rails_job")
    if job is None or (job.poll() is not None and not os.path.exists(path)):
        if job is not None and s.get("rails_tries", 0) >= 2:
            return None
        s["rails_tries"] = s.get("rails_tries", 0) + 1
        os.makedirs(RB2_DIR, exist_ok=True)
        for f in os.listdir(RB2_DIR):
            if f.startswith("rails_") and os.path.join(RB2_DIR, f) != path:
                os.remove(os.path.join(RB2_DIR, f))
        s["rails_job"] = subprocess.Popen([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rb2_worker.py"),
                                           "-", path, "rails"], stdout=subprocess.DEVNULL, stderr=sys.stderr)
        _solo_job("rails", s["rails_job"])
        print("rails: baking the rail table in the background", file=sys.stderr)
    return None

def _ch_moons_ahead(s, dts):
    """Every moon's position dts sim-time from now: the BAKED RAIL TABLE while the live rails match it, else rotated on its
    current circle round the star; a loose moon stays where it is (a gone one far away)."""
    tab = rails_table(s)
    if tab is not None:
        t0 = s["t"]
        i0 = t0 / RAILS_DT
        if i0 + 1 < len(tab):
            live = s["moons_np"][0][:, :3]
            j = int(i0); f = i0 - j
            now = tab[j] * (1.0 - f) + tab[j + 1] * f
            fr = s.get("free", {})
            onr = [k for k in range(N_MOONS) if k not in fr]
            if onr and float(np.max(np.linalg.norm(now[onr] - live[onr], axis=1))) < RAILS_TOL:
                ti = np.clip((t0 + np.asarray(dts)) / RAILS_DT, 0.0, len(tab) - 1.001)
                jj = ti.astype(int); ff = (ti - jj)[:, None, None]
                out = np.asarray(tab[jj], dtype=np.float64) * (1.0 - ff) + np.asarray(tab[jj + 1], dtype=np.float64) * ff
                for k, fz in fr.items():
                    out[:, k] = fz["x"] if not fz["gone"] else 99.0
                if s.get("rails_say") != "table":
                    s["rails_say"] = "table"
                    print("rails: the weave reads the baked table", file=sys.stderr)
                return out, s["moons_np"][0][:, 3].astype(np.float64)
    if s.get("rails_say") != "circle" and tab is not None:
        s["rails_say"] = "circle"
        print("rails: the live rails left the table (or it ran out) -- circle extrapolation", file=sys.stderr)
    try:
        rl = rail_predict_at(s, [s["t"]])[s["t"]]
        P, V = rl[0], rl[1]
    except Exception:
        P = s["moons_np"][0][:, :3].astype(np.float64); V = np.zeros_like(P)
    rad = s["moons_np"][0][:, 3].astype(np.float64)
    fr = s.get("free", {})
    out = np.repeat(P[None], len(dts), 0)
    for k in range(len(P)):
        if k in fr:
            if not fr[k]["gone"]:
                out[:, k] = fr[k]["x"]
            else:
                out[:, k] = 99.0
            continue
        n = np.cross(P[k], V[k]); r2 = float(P[k] @ P[k])
        if r2 < 1e-9 or float(np.linalg.norm(n)) < 1e-9:
            continue
        w = float(np.linalg.norm(n)) / r2; nh = n / np.linalg.norm(n)
        a = w * dts
        c, sn = np.cos(a)[:, None], np.sin(a)[:, None]
        out[:, k] = P[k] * c + np.cross(nh, P[k])[None] * sn + nh * float(nh @ P[k]) * (1.0 - c)
    return out, rad

def ch_weave_plan(s, ch, p0, tb):
    """The depth offset dz(p) for p in [p0, p0 + CH_WV_AHEAD] (p = the path's beat parameter, the leader's)."""
    bz = BOX_AX[2]
    kz = float(bz[1])
    ps = p0 + np.arange(int(CH_WV_AHEAD * CH_WV_RES) + 1) / CH_WV_RES
    base = np.array([chase_head(q, ch["w"], ch["side"], ch["rose"]) for q in ps])
    hb = base[:, 1]
    rho = np.hypot(base[:, 0], base[:, 2])
    over = rho < R_OUT + 0.05
    M, mr = _ch_moons_ahead(s, (ps - p0) * tb * sim_rate())
    sgn0 = np.where(hb >= 0.0, 1.0, -1.0)
    cands = [q for q in np.arange(math.ceil(p0 - CH_LEAD + 1.0), p0 + CH_WV_AHEAD - 1.0) + CH_LEAD]
    cross = list(ch.get("wv_keep", []))
    for c in cands:
        if any(abs(c - x) < 0.99 for x in cross):
            continue
        j = int(round((c - p0) * CH_WV_RES))
        if j < 0 or j >= len(ps) or not over[j]:
            continue
        dzc = hb[j] / -kz
        if abs(dzc) > CH_WV_MAXZ:
            continue
        X = base[j] + bz * dzc
        if np.hypot(X[0], X[2]) < DW_R + 0.12:
            continue
        if (np.linalg.norm(M[j] - X[None], axis=1) < mr + CH_WV_CLEAR + 0.04).any():
            continue
        cross.append(float(c))
    cross = sorted(x for x in cross if p0 - 1.0 <= x <= p0 + CH_WV_AHEAD)
    nflip = np.zeros(len(ps))
    for c in cross:
        nflip += 0.5 + 0.5 * np.tanh((ps - c) / (0.5 * CH_WV_EDGE))
    side = np.cos(np.pi * nflip)
    dm = np.min(np.linalg.norm(M - base[:, None, :], axis=2) - mr[None], axis=1)
    hmin = np.maximum(CH_WV_H, np.where(dm < 0.15, np.max(mr) + CH_WV_CLEAR, 0.0))
    ht = np.where(np.abs(hb) > hmin, np.abs(hb), hmin) * side * sgn0
    wR = np.clip((R_OUT + 0.35 - rho) / 0.3, 0.0, 1.0)
    dz = np.clip((ht - hb) / kz * wR * (1.0 if CH_WEAVE else 0.0), -CH_WV_MAXZ, CH_WV_MAXZ)
    return ps, dz, cross

def ch_weave(ch, p):
    w = ch.get("wv")
    if w is None:
        return 0.0
    ps, dz = w
    if p < ps[0] or p > ps[-1]:
        return 0.0
    return float(np.interp(p, ps, dz))

CH_PULSAR = True
PS_BEATS = 16.0
PS_W = 0.45
PS_WA = 0.6
PS_RMIN = 0.45
PS_MARGIN = 0.14
PS_AHEAD = 16.0
PS_RES = 12
PS_CAND = 24
PS_PASS_R = 0.12
PS_CLEAR = 0.06
PS_SW_V = 1.8
PS_SW_DASH = 1.8
PS_GAP0 = 1.5
PS_GAP_MIN = 0.12
PS_GAP_RATE = 0.12
PS_SW_DODGE = 0.16
PS_TAG = 0.09
PS_JET = 0.36
PS_JET_PAD = 0.15
PS_INSIDE = 0.85
PS_WIDEN_V = 0.02
PS_SLEW = 1.5
PS_TAG_COOL = 6.0

def ps_basis(phi):
    """The beam frame at spin angle phi: d (the beam), Y (the ring axis), tv (in-plane, across the beam)."""
    c, sn = math.cos(phi), math.sin(phi)
    return np.array([c, 0.0, sn]), np.array([0.0, 1.0, 0.0]), np.array([sn, 0.0, -c])

def _ps_ell():
    """The ideal orbit: (long-axis angle, half-axes a, b)."""
    la = BOX_AX[1]
    return math.atan2(float(la[2]), float(la[0])), PS_OPT[0] * float(_ps_rmax(np.array([la]))[0]), PS_OPT[1] * R_OUT

def ps_sp(sgn, phi0, p):
    """The orbit PARAMETER at beat p: uniform, one turn per PS_BEATS (0 = along the long axis at phi0)."""
    return sgn * 2.0 * math.pi * np.asarray(p, np.float64) / PS_BEATS + phi0 - _ps_ell()[0]

def ps_beam_angle(sp):
    """THE BEAM FOLLOWS THE ORBIT (user: "they still trail and overshoot the beam constantly"): the beam points at the ideal
    orbit's position for parameter sp -- one turn per PS_BEATS still, slow past the far ends, quick through the middle."""
    phil, a, b = _ps_ell()
    return phil + sp + np.angle(np.exp(1j * (np.arctan2(b * np.sin(sp), a * np.cos(sp)) - sp)))

def ps_phi(ch, p):
    return ps_beam_angle(ps_sp(ch["ps_sgn"], ch["ps_phi0"], p))

def _ps_rmax(u):
    """Distance from the star to the glass (minus PS_MARGIN) along unit directions u (..., 3)."""
    q = np.abs(u @ BOX_AX.T)
    return np.min((BOX_H - PS_MARGIN) / np.maximum(q, 1e-6), axis=-1)

def _ps_world(phi, F, E, A):
    """Beam-frame pattern values (radial fraction F, elevation E, lateral A; arrays) at spin angles phi -> world points."""
    c, sn = np.cos(phi), np.sin(phi)
    d = np.stack([c, np.zeros_like(c), sn], -1); tv = np.stack([sn, np.zeros_like(c), -c], -1)
    Y = np.array([0.0, 1.0, 0.0])
    u = np.cos(E)[..., None] * (np.cos(A)[..., None] * d + np.sin(A)[..., None] * tv) + np.sin(E)[..., None] * Y
    r = PS_RMIN + F * np.maximum(_ps_rmax(u) - PS_RMIN, 0.0)
    return u * r[..., None]

PS_KEYS = ("far", "mid", "Efar", "Emid", "kep")
PS_LO = np.array([0.72, 0.30, 0.45, 0.20, 0.8])
PS_HI = np.array([0.97, 0.92, 0.75, 0.55, 1.0])
PS_MID = 0.35
PS_XW = 0.80
PS_NX = 6
PS_OPT = (0.90, 0.65)
PS_OFF_K = 4.0
PS_CLOSE = (0.04, 0.22)
PS_MOON_W = 1.0
PS_LEARN = 0.05
PS_MODEL = os.path.join(RB2_DIR, "ps_model3.json")

def ps_model_new():
    return dict(nx=[1.0, 1.0], mu=list(0.5 * (PS_LO + PS_HI)), sd=list(0.35 * (PS_HI - PS_LO)), n=0)

def ps_long(ctx):
    """0..1 per plan sample: how far the beam reaches now (1 = along the long axis, 0 = across)."""
    return np.cos(ctx["sp"]) ** 2

def ps_windows(ctx):
    """The middle phases inside the plan: lists of the whole beats they contain."""
    ps, lg = ctx["ps"], ps_long(ctx)
    beats = np.arange(math.ceil(ps[0] + 0.5), math.floor(ps[-1]) + 1)
    out, cur = [], []
    for b in beats:
        if np.interp(b, ps, lg) < PS_MID:
            cur.append(float(b))
        elif cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out

def ps_sample(rng, n, ctx, model=None):
    """n candidates: genes (half from the learned model) + 1 or 2 crossings on beats in each middle phase."""
    m = n // 2 if model is not None else 0
    g = rng.uniform(PS_LO, PS_HI, (n, len(PS_LO)))
    nxw = np.array(model["nx"]) if model is not None else np.ones(2)
    if m:
        g[:m] = np.clip(rng.normal(model["mu"], model["sd"], (m, len(PS_LO))), PS_LO, PS_HI)
    X = np.full((n, PS_NX), np.nan)
    hold = ctx["p0"] + 1.5
    for i in range(n):
        xs = []
        for w in ctx["win"]:
            w = [b for b in w if b > hold]
            if not w:
                continue
            k = 1 + int(rng.choice(2, p=nxw / nxw.sum())) if i < m else 1 + int(rng.integers(0, 2))
            if k == 2 and len(w) >= 3:
                xs += [w[0], w[-1]]
            else:
                xs.append(float(rng.choice(w)))
        X[i, :min(len(xs), PS_NX)] = xs[:PS_NX]
    return dict(g=g, X=X, s0=np.full(n, ctx["side0"]))

def ps_mutate(rng, best, n, ctx):
    """n small variations of one candidate: genes jittered, a crossing moved a beat, sometimes one added / dropped."""
    out = {k: np.repeat(v[:1], n, 0) for k, v in best.items()}
    out["g"] = np.clip(out["g"] + rng.normal(0.0, 1.0, out["g"].shape) * 0.08 * (PS_HI - PS_LO), PS_LO, PS_HI)
    hold = ctx["p0"] + 1.5
    allw = [b for w in ctx["win"] for b in w if b > hold]
    wof = {b: k for k, w in enumerate(ctx["win"]) for b in w}
    for i in range(n):
        xs = [x for x in out["X"][i] if np.isfinite(x)]
        r = rng.uniform()
        if xs and r < 0.5:
            j = int(rng.integers(0, len(xs)))
            if xs[j] > hold:
                nb = xs[j] + float(rng.choice([-1.0, 1.0]))
                if nb in allw and all(abs(nb - x) >= 2.0 for k2, x in enumerate(xs) if k2 != j):
                    xs[j] = nb
        elif r < 0.7 and allw:
            nb = float(rng.choice(allw))
            if all(abs(nb - x) >= 2.0 for x in xs) and sum(1 for x in xs if wof.get(x) == wof.get(nb)) < 2:
                xs.append(nb)
        elif r < 0.9 and xs and xs[-1] > hold:
            xs.pop(int(rng.integers(0, len(xs))))
        xs = sorted(xs)[:PS_NX]
        out["X"][i] = np.nan; out["X"][i, :len(xs)] = xs
    return out

def ps_cat(*cs):
    return {k: np.concatenate([c[k] for c in cs]) for k in cs[0]}

def ps_take(c, i):
    return {k: v[i:i + 1].copy() for k, v in c.items()}

def ps_curves(c, ctx):
    """Candidates -> beam-frame curves F, E, A (C, n): far out + high on one side in the far phase, in over the disk in the
    middle (an ELLIPSE, the original loop), the side flipping smoothly at each crossing; A = its lag / lead in the beam."""
    ps, phi = ctx["ps"], ctx["phi"]
    lg = ctx["lg"][None]
    g = c["g"]
    ease = lg
    la = BOX_AX[1]
    a_ = g[:, 0:1] * float(_ps_rmax(np.array([la]))[0]); b_ = g[:, 1:2] * R_OUT
    sp = ctx["sp"][None]
    th = np.arctan2(b_ * np.sin(sp), a_ * np.cos(sp))
    rr = np.sqrt((a_ * np.cos(sp)) ** 2 + (b_ * np.sin(sp)) ** 2)
    A = np.clip(-g[:, 4:5] * np.angle(np.exp(1j * (th - (phi[None] - _ps_ell()[0])))), -PS_WA, PS_WA)
    nflip = np.zeros((len(g), len(ps)))
    X = c["X"]
    for j in range(X.shape[1]):
        xj = X[:, j:j + 1]
        nflip += np.where(np.isfinite(xj), 0.5 + 0.5 * np.tanh((ps[None] - np.nan_to_num(xj)) / (0.5 * PS_XW)), 0.0)
    side = c["s0"][:, None] * np.cos(math.pi * nflip)
    E = side * PS_W * (g[:, 3:4] + (g[:, 2:3] - g[:, 3:4]) * ease)
    c_, sn_ = np.cos(phi)[None], np.sin(phi)[None]
    u = np.stack([np.cos(E) * (np.cos(A) * c_ + np.sin(A) * sn_), np.sin(E), np.cos(E) * (np.cos(A) * sn_ - np.sin(A) * c_)], -1)
    F = np.clip((rr - PS_RMIN) / np.maximum(_ps_rmax(u) - PS_RMIN, 1e-6), 0.0, 0.97)
    return F, E, A

def ps_eval(c, ctx):
    """Score: never a hit; clean crossings with CLOSE CALLS (the dodge); moon passes a bonus; no jump. (score, passes, hits)."""
    ps, p0 = ctx["ps"], ctx["p0"]
    F, E, A = ps_curves(c, ctx)
    phi = np.repeat(ctx["phi"][None], len(F), 0)
    X = _ps_world(phi, F, E, A)
    M, mr = ctx["M"], ctx["mr"]
    dxz = np.hypot(X[:, :, None, 0] - M[None, :, :, 0], X[:, :, None, 2] - M[None, :, :, 2])
    dy = np.abs(X[:, :, None, 1] - M[None, :, :, 1])
    over = (dxz < PS_PASS_R + mr) & (dy > mr + PS_CLEAR)
    passes = (over[:, 1:] & ~over[:, :-1]).sum(axis=(1, 2))
    d3 = np.sqrt(dxz * dxz + dy * dy)
    inside = d3 < mr + PS_CLEAR
    hits = (inside[:, 1:] & ~inside[:, :-1]).sum(axis=(1, 2)) + inside[:, 0].sum(1)
    close = np.zeros(len(F))
    for j in range(c["X"].shape[1]):
        xj = c["X"][:, j]
        ok = np.isfinite(xj)
        if not ok.any():
            continue
        lo = np.clip(np.searchsorted(ps, np.nan_to_num(xj) - PS_XW), 0, len(ps) - 1)
        hi = np.clip(np.searchsorted(ps, np.nan_to_num(xj) + PS_XW), 0, len(ps) - 1)
        for i in np.flatnonzero(ok):
            gap = float((d3[i, lo[i]:hi[i] + 1] - mr[None]).min())
            if PS_CLOSE[0] < gap < PS_CLOSE[1]:
                close[i] += 1.0 - (gap - PS_CLOSE[0]) / (PS_CLOSE[1] - PS_CLOSE[0])
    off = np.abs(np.linalg.norm(X, axis=2) - ctx["r_opt"][None]).mean(1)
    score = 2.0 * close + PS_MOON_W * passes - 20.0 * hits - PS_OFF_K * off
    old = ctx.get("old")
    if old is not None:
        pj = p0 + 1.5
        j = min(int(round((pj - p0) * PS_RES)), len(ps) - 1)
        Xb = _ps_world(np.repeat(np.array([ctx["phi"][j]]), len(F)), F[:, j], E[:, j], A[:, j])
        Xo = _ps_world(np.array([ctx["phi"][j]]), *(np.array([np.interp(pj, old[0], old[k])]) for k in (1, 2, 3)))[0]
        score = score - 4.0 * np.linalg.norm(Xb - Xo, axis=1)
    return score, passes, hits

def ps_ctx(s, ch, p0, tb):
    """Everything a plan search needs (picklable: the worker gets it)."""
    ps = p0 + np.arange(int(PS_AHEAD * PS_RES) + 1) / PS_RES
    M, mr = _ch_moons_ahead(s, (ps - p0) * tb * sim_rate())
    old = ch.get("ps_plan")
    side0 = 1.0
    if old is not None:
        e0 = float(np.interp(p0 + 1.5, old[0], old[2]))
        side0 = 1.0 if e0 >= 0.0 else -1.0
    keep = ch.get("ps_keep")
    if not (isinstance(keep, dict) and "s0" in keep):
        keep = None
    ctx = dict(ps=ps, p0=p0, phi=ps_phi(ch, ps), sp=ps_sp(ch["ps_sgn"], ch["ps_phi0"], ps), M=np.asarray(M, np.float64), mr=np.asarray(mr, np.float64), old=old,
               keep=keep, side0=side0)
    ctx["lg"] = ps_long(ctx)
    ctx["win"] = ps_windows(ctx)
    go = np.array([[PS_OPT[0], PS_OPT[1], 0.7, 0.4, 1.0]])
    Fo, Eo, Ao = ps_curves(dict(g=go, X=np.full((1, PS_NX), np.nan), s0=np.ones(1)), ctx)
    ctx["r_opt"] = np.linalg.norm(_ps_world(ctx["phi"][None], Fo, Eo, Ao)[0], axis=1)
    return ctx

def ps_info(c, b, pa, hi, tried, src, model=None):
    nx = int(np.isfinite(c["X"][b]).sum())
    d = dict(passes=int(pa[b]), hits=int(hi[b]), mode=f"{nx} crossings", cross=nx, tried=tried, src=src)
    if model is not None:
        d["learned"] = model["n"]
    return d

def ps_plan(s, ch, p0, tb):
    """In-thread fallback (no worker answer yet): one batch of PS_CAND."""
    ctx = ps_ctx(s, ch, p0, tb)
    c = ps_sample(s["rng"], PS_CAND, ctx)
    if ctx["keep"] is not None:
        c = ps_cat(ctx["keep"], c)
    sc, pa, hi = ps_eval(c, ctx)
    b = int(np.argmax(sc))
    best = ps_take(c, b)
    F, E, A = ps_curves(best, ctx)
    return ctx["ps"], F[0], E[0], A[0], best, ps_info(c, b, pa, hi, len(sc), "thread")

def ps_worker_loop(req_path, out_path, token):
    """(ps_worker.py) Anytime search: for each request (one beat ahead) keep trying -- half from the learned model, half
    mutations of the best so far -- and write the best whenever it improves; a new request moves on (the model learns from
    the old one's elite first). Exits when the sim's token changes or nothing is asked for 30 s."""
    import json
    rng = np.random.default_rng()
    try:
        with open(PS_MODEL) as fh:
            model = json.load(fh)
    except Exception:
        model = ps_model_new()
    seen, last_ask, saved = None, time.time(), time.time()
    ctx = best = None
    pool, tried = [], 0
    while True:
        try:
            st = os.stat(req_path).st_mtime_ns
        except OSError:
            st = None
        if st is not None and st != seen:
            try:
                with open(req_path, "rb") as fh:
                    req = pickle.load(fh)
            except Exception:
                req = None
            if req is not None:
                if req.get("token") != token:
                    return
                seen, last_ask = st, time.time()
                if pool:
                    pool.sort(key=lambda x: -x[0])
                    el = pool[:max(4, len(pool) // 10)]
                    G = np.array([e[1]["g"][0] for e in el])
                    nx = np.array([e[2] for e in el])
                    h = np.array([np.mean(nx <= 1.0), np.mean(nx > 1.0)])
                    model["nx"] = list((1 - PS_LEARN) * np.array(model["nx"]) / sum(model["nx"]) + PS_LEARN * h + 1e-3)
                    model["mu"] = list((1 - PS_LEARN) * np.array(model["mu"]) + PS_LEARN * G.mean(0))
                    model["sd"] = list(np.maximum((1 - PS_LEARN) * np.array(model["sd"]) + PS_LEARN * G.std(0), 0.05 * (PS_HI - PS_LO)))
                    model["n"] = model.get("n", 0) + 1
                ctx, best, pool, tried = req["ctx"], None, [], 0
        if ctx is None or time.time() - last_ask > 30.0:
            if time.time() - last_ask > 30.0:
                return
            time.sleep(0.01)
            continue
        c = ps_sample(rng, 48, ctx, model)
        if best is not None:
            c = ps_cat(c, ps_mutate(rng, best[1], 48, ctx))
        elif ctx.get("keep") is not None:
            c = ps_cat(ctx["keep"], c)
        sc, pa, hi = ps_eval(c, ctx)
        tried += len(sc)
        k = np.argsort(-sc)[:6]
        nwin = max(1, len([w for w in ctx["win"] if any(b > ctx["p0"] + 1.5 for b in w)]))
        pool.extend((float(sc[i]), ps_take(c, int(i)), float(np.isfinite(c["X"][i]).sum()) / nwin) for i in k)
        b = int(k[0])
        if best is None or sc[b] > best[0]:
            best = (float(sc[b]), ps_take(c, b))
            F, E, A = ps_curves(best[1], ctx)
            out = dict(seq=req["seq"], plan=(ctx["ps"], F[0], E[0], A[0]), keep=best[1], score=best[0],
                       info=ps_info(c, b, pa, hi, tried, "worker", model))
            with open(out_path + ".part", "wb") as fh:
                pickle.dump(out, fh)
            os.replace(out_path + ".part", out_path)
        if time.time() - saved > 30.0:
            saved = time.time()
            with open(PS_MODEL + ".part", "w") as fh:
                json.dump(model, fh)
            os.replace(PS_MODEL + ".part", PS_MODEL)

def ps_at(ch, p):
    """The squid's beam-frame pattern values at p (the plan, interpolated)."""
    pl = ch["ps_plan"]
    return tuple(float(np.interp(p, pl[0], pl[k])) for k in (1, 2, 3))

def ps_beamq(phi, X):
    """World points -> the beam frame at phi (x along the beam, y up, z across)."""
    d, Y, tv = ps_basis(phi)
    return np.stack([X @ d, X @ Y, X @ tv], -1)

def ps_clamp(phi, q, inside=1.0):
    """Keep a beam-frame point inside the cone (x inside of the field's half-angles), out of the glare, off the glass."""
    r = float(np.linalg.norm(q))
    if r < 1e-6:
        return np.array([PS_RMIN, 0.0, 0.0])
    u = q / r
    W, WA = PS_W * inside, PS_WA * inside
    el = max(-W, min(W, math.asin(max(-1.0, min(1.0, float(u[1]))))))
    az = max(-WA, min(WA, math.atan2(float(u[2]), float(u[0]))))
    u = np.array([math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)])
    d, Y, tv = ps_basis(phi)
    uw = u[0] * d + u[1] * Y + u[2] * tv
    r = max(PS_RMIN, min(r, float(_ps_rmax(uw))))
    return u * r

def _solo_job(kind, proc):
    """One background job per kind (user saw the ships skip: every sim.py edit had left its old bakes running): stop the
    previous one (pid file), remember this one."""
    import signal
    pf = os.path.join(RB2_DIR, f"{kind}.pid")
    try:
        with open(pf) as fh:
            old = int(fh.read().strip())
        if old != proc.pid:
            with open(f"/proc/{old}/cmdline", "rb") as fh:
                if b"_worker.py" in fh.read():
                    os.kill(old, signal.SIGTERM)
    except Exception:
        pass
    os.makedirs(RB2_DIR, exist_ok=True)
    with open(pf, "w") as fh:
        fh.write(str(proc.pid))
    return proc

def _ps_paths():
    return os.path.join(RB2_DIR, "ps_req.pkl"), os.path.join(RB2_DIR, "ps_best.pkl")

def ps_worker_ask(s, ch, p0, tb):
    """Post the next request (one turn of moons from the rail table) and keep the worker alive."""
    req_p, out_p = _ps_paths()
    os.makedirs(RB2_DIR, exist_ok=True)
    pr = s.get("ps_proc")
    if pr is None or pr.poll() is not None:
        s["ps_token"] = f"{os.getpid()}_{time.time():.3f}"
        s["ps_proc"] = subprocess.Popen([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ps_worker.py"),
                                         req_p, out_p, s["ps_token"]], stdout=subprocess.DEVNULL, stderr=sys.stderr)
        _solo_job("squid", s["ps_proc"])
        print("pulsar: the squid's planner thinks in the background now", file=sys.stderr)
    ch["ps_seq"] = ch.get("ps_seq", 0) + 1
    with open(req_p + ".part", "wb") as fh:
        pickle.dump(dict(seq=ch["ps_seq"], token=s["ps_token"], ctx=ps_ctx(s, ch, p0, tb)), fh)
    os.replace(req_p + ".part", req_p)

def ps_worker_take(s, ch):
    """The worker's best for the request posted last beat (None if it has none)."""
    if ch.get("ps_seq") is None:
        return None
    try:
        with open(_ps_paths()[1], "rb") as fh:
            out = pickle.load(fh)
    except Exception:
        return None
    return out if out.get("seq") == ch["ps_seq"] else None

def chase_step_pulsar(api, s, dt, c, ok, tb, au, ch):
    """THE PULSAR PROJECTOR's chase step (chase_step's timing burns + section clock already ran)."""
    it, rng = s["itime"], s["rng"]
    if "ps_phi0" not in ch:
        try:
            ch["ps_sgn"] = fm_spin(s, rail_predict_at(s, [s["t"]])[s["t"]])
        except Exception:
            ch["ps_sgn"] = 1.0
        la = BOX_AX[1]
        ch["ps_phi0"] = math.atan2(float(la[2]), float(la[0]))
        ch["ps_sw"] = [dict(q=np.array([PS_RMIN + 0.3, 0.0, 0.0]) + rng.normal(0.0, 0.03, 3), v=np.zeros(3), hist=[],
                            ph=float(rng.uniform(0, 6.28))) for _ in range(CH_SW_N)]
        ch["ps_tag"] = -99.0
    db = dt / tb
    if "pbc" not in ch:
        ch["pbc"] = ch["bc"]
    ch["pbc"] += db
    err = ch["bc"] - ch["pbc"]
    err -= PS_BEATS * round(err / PS_BEATS)
    ch["pbc"] += max(-PS_SLEW * dt, min(PS_SLEW * dt, err))
    bc = ch["pbc"]
    if int(bc) != ch.get("ps_beat") or ch.get("ps_plan") is None:
        ch["ps_beat"] = int(bc)
        t0 = time.perf_counter()
        got = ps_worker_take(s, ch)
        if got is None and (ch.get("ps_plan") is None or ch["ps_plan"][0][-1] < bc + 2.0):
            ps, F, E, A, keep, info = ps_plan(s, ch, bc - 0.5, tb)
            got = dict(plan=(ps, F, E, A), keep=keep, info=info)
        if got is not None:
            ps, F, E, A = got["plan"]
            old = ch.get("ps_plan")
            if old is not None:
                k = np.clip((ps - (bc + 0.5)) / 1.0, 0.0, 1.0)
                F = np.interp(ps, old[0], old[1]) * (1.0 - k) + F * k
                E = np.interp(ps, old[0], old[2]) * (1.0 - k) + E * k
                A = np.interp(ps, old[0], old[3]) * (1.0 - k) + A * k
            ch["ps_plan"], ch["ps_keep"] = (ps, F, E, A), got["keep"]
            info = got["info"]
            if it - ch.get("ps_say", -99.0) > 8.0:
                ch["ps_say"] = it
                print(f"pulsar: squid plan {info['mode']} through the middle phases: {info['passes']} moon "
                      f"passes, {info['hits']} hits in {PS_AHEAD:.0f} beats; {info['tried']} tried by the {info['src']}"
                      f"{' (model updates %d)' % info['learned'] if 'learned' in info else ''} ({1e3 * (time.perf_counter() - t0):.1f} ms)",
                      file=sys.stderr)
        ps_worker_ask(s, ch, int(bc) + 0.5, tb)
    ch["ps_tb"] = tb
    phi = ps_phi(ch, bc)
    d, Y, tv = ps_basis(phi)
    rec = it - ch["last"] >= CH_TRAIL_DT
    tex = np.zeros((1 + CH_SW_N, CH_NT + 2, 4), dtype="f4")
    beat_age = (bc % 1.0) * tb
    flare = math.exp(-beat_age / 0.09) + 0.6 * ch["burn"]
    def squid(p):
        F, E, A = ps_at(ch, p)
        return _ps_world(np.array([ps_phi(ch, p)]), np.array([F]), np.array([E]), np.array([A]))[0]
    kb = 1.0 / max(sim_rate() * tb, 1e-6)
    ap = s.get("ps_appt", [])
    for a in ap:
        a["b"] = bc + (a["t"] - s["t"]) * kb
        a.setdefault("b0", bc)
    ap = [a for a in ap if bc < a["b"] + 3.0 * PS_APPT_W]
    s["ps_appt"] = ap
    def fade(a, p):
        q = min(max((p - a["b0"]) / max(a["b"] - a["b0"], 0.5), 0.0), 1.0)
        return q * q * (3.0 - 2.0 * q) if p < a["b"] else 1.0
    for a in ap:
        if a["ship"] == 0:
            phb = ps_phi(ch, a["b"])
            tg = ps_beamq(phb, a["x"]) - ps_beamq(phb, squid(a["b"]))
            a["dl"] = tg if "dl" not in a else a["dl"] + (tg - a["dl"]) * min(1.0, dt / (0.3 * tb))
    def bent(p, ship, base):
        php = ps_phi(ch, p)
        q = ps_beamq(php, base(p))
        for a in ap:
            if a["ship"] == ship and abs(p - a["b"]) < 3.0 * PS_APPT_W:
                q = q + a["dl"] * math.exp(-((p - a["b"]) / PS_APPT_W) ** 2) * fade(a, p)
        q = ps_clamp(php, q, PS_INSIDE)
        dp, Yp, tvp = ps_basis(php)
        return q[0] * dp + q[1] * Yp + q[2] * tvp
    H = bent(bc, 0, squid)
    Hd = bent(bc + 0.02, 0, squid) - H
    spd = float(np.linalg.norm(H - ch["ps_H"])) / max(dt, 1e-3) if ch.get("ps_H") is not None else 0.0
    if spd > 6.0:
        print(f"pulsar: JUMP {spd:.1f}/s at beat {bc:.2f} (dt {1e3 * dt:.0f} ms, clock err {err:+.2f}, plan "
              f"{ch['ps_plan'][0][0]:.1f}..{ch['ps_plan'][0][-1]:.1f})", file=sys.stderr)
    ch["ps_H"] = H
    _ch_trail(ch["lead"], H, it, rec)
    for j, q in enumerate(ch["lead"]):
        tex[0, j] = q
    tex[0, CH_NT] = (flare, 0.0, len(ch["lead"]), 0.0)
    tex[0, CH_NT + 1] = (*(Hd / max(float(np.linalg.norm(Hd)), 1e-9)), 0.0)
    kick = float(au.get("u_kick", 0.0) or 0.0)
    sw = ch["ps_sw"]
    phS = phi + math.pi
    dS, _, tvS = ps_basis(phS)
    cen = np.mean([b["q"] for b in sw], axis=0)
    if kick > ch["kick"] + CH_HAWK_KICK and ch["hawk"] <= 0.0:
        dd = rng.normal(0.0, 1.0, 3); dd /= max(float(np.linalg.norm(dd)), 1e-9)
        ch["hawk"], ch["hawk_p"] = CH_HAWK_TAU, cen + dd * CH_HAWK_EDGE * CH_SW_R * 1.5
    ch["kick"] = kick
    ch["hawk"] = max(0.0, ch["hawk"] - dt)
    ha = ch["hawk"] / CH_HAWK_TAU
    vmax = PS_SW_V * (1.0 + (PS_SW_DASH - 1.0) * min(ha, 1.0))
    anchor = lambda p: -squid(p)
    qA = ps_beamq(phS, anchor(bc))
    hd = qA - ps_beamq(phS, anchor(bc - 0.05)); hd /= max(float(np.linalg.norm(hd)), 1e-9)
    M = s["moons_np"][0][:, :3].astype(np.float64); mr = s["moons_np"][0][:, 3].astype(np.float64)
    Mq = ps_beamq(phS, M)
    swX = []
    for k, b in enumerate(sw):
        ang = b["ph"] + 2.0 * math.pi * bc / 4.0 + 2.0 * math.pi * k / CH_SW_N
        perp = np.cross(hd, [1.0, 0.0, 0.0]); perp /= max(float(np.linalg.norm(perp)), 1e-9)
        perp2 = np.cross(hd, perp)
        slot = qA + CH_SW_R * (math.cos(ang) * perp + math.sin(ang) * perp2)
        gain = CH_SW_STEER
        for a in ap:
            if a["ship"] == 1 and a["k"] == k and abs(bc - a["b"]) < 3.0 * PS_APPT_W:
                wa = math.exp(-((bc - a["b"]) / PS_APPT_W) ** 2) * fade(a, bc)
                slot = slot * (1.0 - wa) + ps_beamq(phS, a["x"]) * wa
                gain = CH_SW_STEER * (1.0 + 4.0 * wa)
        want = slot - b["q"]
        wl = float(np.linalg.norm(want))
        want = want / max(wl, 1e-9) * min(vmax * (gain / CH_SW_STEER), wl * gain)
        for o in sw:
            if o is not b:
                dv = b["q"] - o["q"]; dl = float(np.linalg.norm(dv))
                if 1e-6 < dl < CH_SW_SEP:
                    want = want + dv / dl * (CH_SW_SEP - dl) * 40.0
        dm = b["q"][None] - Mq
        dl = np.linalg.norm(dm, axis=1)
        for j in np.flatnonzero(dl < mr + PS_SW_DODGE):
            want = want + dm[j] / max(dl[j], 1e-6) * vmax * 2.0 * (1.0 - (dl[j] - mr[j]) / PS_SW_DODGE)
        if ha > 0.0:
            dv = b["q"] - ch["hawk_p"]; dl2 = max(float(np.linalg.norm(dv)), 0.02)
            want = want + dv / dl2 * CH_HAWK_PUSH * ha / dl2
        b["v"] = b["v"] + (want - b["v"]) * min(1.0, gain * dt)
        b["q"] = ps_clamp(phS, b["q"] + b["v"] * dt, PS_INSIDE)
        X = b["q"][0] * dS + b["q"][1] * Y + b["q"][2] * tvS
        Xn = b["q"][0] * ps_basis(phS + 0.02)[0] + b["q"][1] * Y + b["q"][2] * ps_basis(phS + 0.02)[2]
        _ch_trail(b["hist"], X, it, rec)
        swX.append(X)
        for j, q in enumerate(b["hist"]):
            tex[1 + k, j] = q
        vw = (b["v"][0] * dS + b["v"][1] * Y + b["v"][2] * tvS) \
            + (Xn - X) / 0.02 * ch["ps_sgn"] * 2.0 * math.pi / (PS_BEATS * tb)
        vd = vw / max(float(np.linalg.norm(vw)), 1e-9)
        tex[1 + k, CH_NT] = (flare + 0.8 * ha, 0.0, len(b["hist"]), 0.0)
        tex[1 + k, CH_NT + 1] = (*vd, 0.0)
    if rec:
        ch["last"] = it
    s["chase_tex"].write(tex.tobytes())
    api.render_uniform("u_chaseOn", 1.0)
    api.render_uniform("u_chaseNrm", (0.0, 1.0, 0.0))
    rp = float(_ps_rmax(d[None])[0]); rn = float(_ps_rmax(-d[None])[0])
    api.render_uniform("u_pulsar", (float(d[0]), float(d[2]), PS_W, float(flare)))
    api.render_uniform("u_pulsarR", (rp, rn, float(np.linalg.norm(H)), PS_WA))
    def off(P, ax):
        return math.acos(max(-1.0, min(1.0, float(P @ ax) / max(float(np.linalg.norm(P)), 1e-9))))
    vS = float(np.linalg.norm(H - ch["ps_H0"])) / max(dt, 1e-3) if ch.get("ps_H0") is not None else 0.0
    ch["ps_H0"] = H
    vW = float(np.mean([np.linalg.norm(b["v"]) for b in sw]))
    want = (max(PS_JET, off(H, d) + PS_JET_PAD + PS_WIDEN_V * vS),
            max(PS_JET, max(off(X, -d) for X in swX) + PS_JET_PAD + PS_WIDEN_V * vW))
    jw = ch.get("ps_jw", list(want))
    for k in (0, 1):
        tau = 0.1 if want[k] > jw[k] else 0.8
        jw[k] += (want[k] - jw[k]) * min(1.0, dt / tau)
    ch["ps_jw"] = jw
    api.render_uniform("u_pulsarW", (min(jw[0], 1.0), min(jw[1], 1.0)))

def _ch_trail(h, x, it, rec):
    if rec or not h:
        h.insert(0, (*x, it)); del h[CH_NT:]
    else:
        h[0] = (*x, h[0][3])

def chase_step(api, s, dt):
    if not CH_ON:
        api.render_uniform("u_chaseOn", 0.0)
        return
    c = mt_clock(s)
    it = s["itime"]
    ok = c["conf"] >= 0.2 and 0.2 < c["tb"] < 2.0
    tb = c["tb"] if ok else 0.5
    rng = s["rng"]
    au = (getattr(api, "inputs", None) or {}).get("audio") or {}
    ch = s.get("chase")
    if ch is None or "well" not in ch:
        ch = s["chase"] = dict(bc=0.0, last=-9.0, burn=0.0, well=0, w=[1.0, 0.0, 0.0, 0.0, 0.0], side=(1.0, 1.0), well_bc=0.0,
                               ms=au.get("u_ms_change"), rose=0.0, lead=[], kick=0.0, hawk=0.0, hawk_p=np.zeros(3), dash=0.0,
                               sw=[dict(x=np.zeros(3), v=np.zeros(3), hist=[], ph=float(rng.uniform(0, 6.28))) for _ in range(CH_SW_N)])
    db = dt / tb
    ch["bc"] += db
    burn = 0.0
    if ok:
        t0 = c["it"] - c["bar"] * 4.0 * tb
        err = (it - t0) / tb - ch["bc"]
        err -= CH_BEATS * round(err / CH_BEATS)
        if abs(err) > 0.5 * CH_BEATS - 1.0:
            ch["bc"] += err
        else:
            step = max(-CH_LOCK * dt, min(CH_LOCK * dt, err))
            ch["bc"] += step
            burn = abs(step) / max(CH_LOCK * dt, 1e-9)
    ch["burn"] = max(burn, ch["burn"] * math.exp(-dt / 0.3))
    msc = au.get("u_ms_change")
    if (msc is not None and msc != ch["ms"]) or ch["bc"] - ch["well_bc"] > CH_WELL_BEATS:
        ch["ms"], ch["well_bc"] = msc, ch["bc"]
        ch["well"] = int(rng.choice([k for k in range(5) if k != ch["well"]]))
        if ch["well"] in (3, 4):
            ch["side"] = (float(rng.choice([-1.0, 1.0])), float(rng.choice([-1.0, 1.0])))
        print(f"chase: the leader's pocket -> {('loop', 'figure-8', 'rose', 'side pocket', 'end pocket')[ch['well']]}", file=sys.stderr)
    for k in range(5):
        ch["w"][k] += ((1.0 if k == ch["well"] else 0.0) - ch["w"][k]) * (1.0 - math.exp(-db / CH_MORPH))
    dens = au.get("u_ms_cur_a")
    try:
        dens = float(dens[2])
    except (TypeError, IndexError, ValueError):
        dens = 0.5
    ch["rose"] += (CH_ROSE * dens - ch["rose"]) * (1.0 - math.exp(-dt / 1.0))
    if CH_PULSAR:
        return chase_step_pulsar(api, s, dt, c, ok, tb, au, ch)
    api.render_uniform("u_pulsar", (1.0, 0.0, 0.0, 0.0))
    beat_age = (ch["bc"] % 1.0) * tb
    flare = math.exp(-beat_age / 0.09) + 0.6 * ch["burn"]
    if CH_WEAVE and (int(ch["bc"]) != ch.get("wv_beat") or ch.get("wv") is None):
        ch["wv_beat"] = int(ch["bc"])
        pL = ch["bc"] + CH_LEAD
        ch["wv_keep"] = [x for x in ch.get("wv_cross", []) if pL <= x < pL + 1.0]
        ps, dz, cr = ch_weave_plan(s, ch, ch["bc"] - 0.5, tb)
        old = ch.get("wv")
        if old is not None:
            ov = np.interp(ps, old[0], old[1], left=0.0, right=0.0)
            k = np.clip((ps - (pL + 0.5)) / 1.0, 0.0, 1.0)
            dz = ov * (1.0 - k) + dz * k
        ch["wv"], ch["wv_cross"] = (ps, dz), cr
    rec = it - ch["last"] >= CH_TRAIL_DT
    tex = np.zeros((1 + CH_SW_N, CH_NT + 2, 4), dtype="f4")
    bL = ch["bc"] + CH_LEAD
    bz = BOX_AX[2]
    H = chase_head(bL, ch["w"], ch["side"], ch["rose"]) + bz * ch_weave(ch, bL)
    Hd = chase_head(bL + 0.02, ch["w"], ch["side"], ch["rose"]) + bz * ch_weave(ch, bL + 0.02) - H
    _ch_trail(ch["lead"], H, it, rec)
    for j, q in enumerate(ch["lead"]):
        tex[0, j] = q
    tex[0, CH_NT] = (flare, 0.0, len(ch["lead"]), 0.0)
    tex[0, CH_NT + 1] = (*(Hd / max(float(np.linalg.norm(Hd)), 1e-9)), 0.0)
    kick = float(au.get("u_kick", 0.0) or 0.0)
    cen = np.mean([b["x"] for b in ch["sw"]], axis=0)
    if kick > ch["kick"] + CH_HAWK_KICK and ch["hawk"] <= 0.0:
        d = rng.normal(0.0, 1.0, 3); d -= BOX_AX[2] * float(d @ BOX_AX[2]); d /= max(float(np.linalg.norm(d)), 1e-9)
        ch["hawk"], ch["hawk_p"] = CH_HAWK_TAU, cen + d * CH_HAWK_EDGE * CH_SW_R * 1.5
    ch["kick"] = kick
    ch["hawk"] = max(0.0, ch["hawk"] - dt)
    ha = ch["hawk"] / CH_HAWK_TAU
    ch["dash"] = min((ch["dash"] + CH_HAWK_SPD * ha * db) * math.exp(-db / 4.0), CH_DASH_CAP * CH_LEAD)
    ba = ch["bc"] + ch["dash"]
    A = chase_head(ba, ch["w"], ch["side"], ch["rose"]) + bz * ch_weave(ch, ba)
    Av = (chase_head(ba + 0.02, ch["w"], ch["side"], ch["rose"]) + bz * ch_weave(ch, ba + 0.02) - A) / (0.02 * tb) * (1.0 + CH_HAWK_SPD * ha)
    tdir = Av / max(float(np.linalg.norm(Av)), 1e-9)
    side = np.cross(tdir, BOX_AX[2]); side /= max(float(np.linalg.norm(side)), 1e-9)
    nrm = BOX_AX[2]
    for k, b in enumerate(ch["sw"]):
        if not b["hist"]:
            b["x"] = A + rng.normal(0.0, 0.03, 3)
        ang = b["ph"] + 2.0 * math.pi * ch["bc"] / 4.0 + 2.0 * math.pi * k / CH_SW_N
        tgt = A + CH_SW_R * (math.cos(ang) * side + 0.6 * math.sin(ang) * nrm) - tdir * 0.04 * k
        want = Av + (tgt - b["x"]) * CH_SW_STEER
        for o in ch["sw"]:
            if o is not b:
                d = b["x"] - o["x"]; dl = float(np.linalg.norm(d))
                if 1e-6 < dl < CH_SW_SEP:
                    want = want + d / dl * (CH_SW_SEP - dl) * 40.0
        if ha > 0.0:
            d = b["x"] - ch["hawk_p"]; dl = max(float(np.linalg.norm(d)), 0.02)
            want = want + d / dl * CH_HAWK_PUSH * ha / dl
        b["v"] = b["v"] + (want - b["v"]) * min(1.0, CH_SW_STEER * dt)
        b["x"] = b["x"] + b["v"] * dt
        _ch_trail(b["hist"], b["x"], it, rec)
        for j, q in enumerate(b["hist"]):
            tex[1 + k, j] = q
        vd = b["v"] / max(float(np.linalg.norm(b["v"])), 1e-9)
        tex[1 + k, CH_NT] = (flare + 0.8 * ha, 0.0, len(b["hist"]), 0.0)
        tex[1 + k, CH_NT + 1] = (*vd, 0.0)
    if rec:
        ch["last"] = it
    s["chase_tex"].write(tex.tobytes())
    api.render_uniform("u_chaseOn", 1.0)
    api.render_uniform("u_chaseNrm", tuple(float(x) for x in BOX_AX[2]))

def rs_step(api, s):
    """THE RESCUE SERVICE (outside the rebirth): queue, plan (worker), budget-check, commit."""
    if s.get("rb") is not None:
        return
    rs = s.setdefault("rs", dict(job=None, active={}, launches=[], next_try=0.0))
    if s.get("rs_off") and s.get("fm_off"):
        return
    it, rate = s["itime"], max(sim_rate(), 1e-6)
    for i, f in list(s.get("free", {}).items()):
        if f["gone"] and i in rs["active"]:
            rs["active"].pop(i)
    job = rs["job"]
    if job is not None:
        if job["proc"].poll() is None:
            return
        rs["job"] = None
        res = None
        if job["proc"].returncode == 0 and os.path.exists(job["out"]):
            with open(job["out"], "rb") as fh:
                res = pickle.load(fh)
        if job.get("kind") == "farm":
            if res is None:
                rs["next_try"] = it + 3.0
                print("farm: no pincer found, retry later", file=sys.stderr)
                return
            lt = [it + (m["ts"][0] - s["t"]) / rate for m in res["mets"]]
            allt = sorted([x for x in rs["launches"] if x > it - RS_WINDOW] + lt)
            if any(sum(1 for y in allt if x <= y < x + RS_WINDOW) > RS_BUDGET for x in allt):
                rs["next_try"] = it + 1.0
                return
            rs["launches"] = [x for x in rs["launches"] if x > it - RS_WINDOW] + lt
            res["coll"] = job["i"] < 0
            rs["farm"] = res
            ps_appoint(s, res["mets"])
            rstar, frac = rs.get("fm_tgt", (0.0, 0.0))
            what = "COLLISION SHOT" if res["coll"] else f"PINCER on moon {job['i']}"
            print(f"farm: {what} at r {rstar:.2f} (band {100 * frac:.0f} % below rest) in {(res['T'] - s['t']) / rate:.1f} s "
                  f"({'/'.join(m['kind'] for m in res['mets'])})", file=sys.stderr)
            return
        i = job["i"]
        f = s.get("free", {}).get(i)
        if res is None or f is None or f["gone"] or f.get("touch", 0.0) > job["it"]:
            rs["next_try"] = it + 2.0
            if f is not None and res is None:
                f["rs_tries"] = f.get("rs_tries", 0) + 1
            print(f"rescue: moon {i} -- {'no shot found' if res is None else 'it moved / was touched'}, retry later", file=sys.stderr)
            return
        lt = [it + (m["ts"][0] - s["t"]) / rate for m in res["mets"]]
        allt = sorted([x for x in rs["launches"] if x > it - RS_WINDOW] + lt)
        if any(sum(1 for y in allt if x <= y < x + RS_WINDOW) > RS_BUDGET for x in allt):
            rs["next_try"] = it + 2.0
            print(f"rescue: moon {i} -- over budget ({RS_BUDGET} per {RS_WINDOW:.0f} s), retry later", file=sys.stderr)
            return
        rs["launches"] = [x for x in rs["launches"] if x > it - RS_WINDOW] + lt
        res["hit1"] = res["hit2"] = False
        rs["active"][i] = res
        ps_appoint(s, res["mets"])
        f["path"] = True
        inf = res.get("info", {})
        print(f"rescue: moon {i} spoken for -- transfer hit in {(res['T1'] - s['t']) / rate:.1f} s, capture in "
              f"{(res['T2'] - s['t']) / rate:.1f} s ({'/'.join(m['kind'] for m in res['mets'])}; sniper score "
              f"{inf.get('score', 0):.2f}, {inf.get('bounces', 0)} bounces, downbeat {bool(inf.get('down'))}, "
              f"{inf.get('evaluated', 0)} tried in {inf.get('secs', 0)} s)", file=sys.stderr)
        return
    if it < rs["next_try"]:
        return
    cand = [] if s.get("rs_off") else \
        [i for i, f in s.get("free", {}).items() if not f["gone"] and not f.get("path") and i not in rs["active"]
         and it - f.get("touch", -99.0) > RS_WAIT]
    used = sum(1 for x in rs["launches"] if x > it - RS_WINDOW)
    cand.sort(key=lambda idx: s["free"][idx].get("rs_tries", 0))
    if not cand or used > RS_BUDGET - 2:
        if not s.get("fm_off") and rs.get("farm") is None and used <= RS_BUDGET - 2:
            tgt = fm_target(api, s)
            if tgt is None:
                rs["next_try"] = it + 1.0
                return
            rstar, frac = tgt
            Ts = [s["t"] + d for d in FM_T]
            rails = rail_predict_at(s, Ts)
            mp = s["moons_np"][0]
            inband = [k for k in range(N_MOONS) if k not in s.get("free", {}) and abs(float(mp[k, 1])) < 0.1
                      and abs(math.hypot(float(mp[k, 0]), float(mp[k, 2])) - rstar) < FM_MOON_DR]
            if inband:
                k = int(s["rng"].choice(inband))
                tg = {T: (rails[T][0][k], rails[T][1][k]) for T in Ts}
            else:
                k = -1
                sg = fm_spin(s, rails[Ts[0]])
                s["ring_sg"] = sg
                th = rs_hole_angle(s, rstar)
                tg = {}
                for T in Ts:
                    thT = th + sg * rstar ** -1.5 * (T - s["t"])
                    X = rstar * np.array([math.cos(thT), 0.0, math.sin(thT)])
                    tg[T] = (X, sg * rstar ** -0.5 * np.array([-math.sin(thT), 0.0, math.cos(thT)]))
            rs["fm_tgt"] = (rstar, frac)
            rs["job"] = _rs_spawn(dict(rails=tg, t_ready=s["t"] + RS_READY, field=ps_field(s, s["t"] + RS_READY)), "farm", k, it)
        elif cand:
            rs["next_try"] = it + 1.0
        return
    i = cand[0]
    f = s["free"][i]
    t0 = s["t"]
    beats, beat = sniper_beats(s, t0)
    cands = sniper_cands(beats, t0, t0 + RS_READY)
    if not cands:
        rs["next_try"] = it + 1.0
        return
    T2s = sorted(set(c[1] for c in cands))
    rails = rail_predict_at(s, T2s + [t0])
    rs["job"] = _rs_spawn(dict(x0=f["x"].copy(), v0=f["v"].copy(), rad=f["rad"], t0=t0, cands=cands, beat=beat,
                               rails={T: (rails[T][0][i], rails[T][1][i]) for T in T2s},
                               P0V0=(rails[t0][0][i], rails[t0][1][i]), t_ready=t0 + RS_READY,
                               field=ps_field(s, t0 + RS_READY)), "snipe", i, it)

PS_SHOT_N = 8

def ps_shots_draw(api, s):
    """THE SHIPS' SHOTS in their ships' colours (user: "make the shots look more like they're coming from the ships, the same
    colour, sized to fit them -- laser or nukes"): the squid's = NUKES (a fat plasma ball), the swarm's = LASERS (thin bolts).
    u_shots rows: the trail (newest first), col RB2_TAIL = (ship, points, s since launch, -)."""
    rs = s.get("rs") or {}
    mets = [m for r in rs.get("active", {}).values() for m in r["mets"]] + (rs["farm"]["mets"] if rs.get("farm") else [])
    mets = [m for m in mets if m.get("kind") in PS_SHIPS and m["on"] and m["hist"]][:PS_SHOT_N]
    if not mets and not s.get("shot_any"):
        return
    tex = np.zeros((PS_SHOT_N, RB2_TAIL + 1, 4), dtype="f4")
    for i, m in enumerate(mets):
        for k, p in enumerate(m["hist"][:RB2_TAIL]):
            tex[i, k] = (*p, 1.0)
        tex[i, RB2_TAIL] = (PS_SHIPS.index(m["kind"]), len(m["hist"][:RB2_TAIL]), s["t"] - float(m["ts"][0]), 0.0)
    s["shot_tex"].write(tex.tobytes())
    s["shot_any"] = bool(mets)
    api.render_uniform("u_shotN", float(len(mets)))

def ps_fired(s, rs, m):
    """A ship's meteor leaves: a launch flash at the ship."""
    if not m["on"] and not m.get("fired") and m.get("kind") in PS_SHIPS:
        m["fired"] = True
        rs.setdefault("done_flashes", []).append((np.asarray(m["xs"][0], np.float64).copy(), s["itime"]))
        print(f"ships: the {m['kind']} FIRES", file=sys.stderr)

def _rs_spawn(inp, kind, i, it):
    os.makedirs(RB2_DIR, exist_ok=True)
    fin, fout = os.path.join(RB2_DIR, f"{kind}_in.pkl"), os.path.join(RB2_DIR, f"{kind}_out.pkl")
    for q in (fout, fout + ".part"):
        if os.path.exists(q):
            os.remove(q)
    with open(fin, "wb") as fh:
        pickle.dump(inp, fh)
    proc = subprocess.Popen([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rb2_worker.py"),
                             fin, fout, kind], stdout=subprocess.DEVNULL, stderr=sys.stderr)
    return dict(proc=proc, out=fout, i=i, it=it, kind=kind)

def rs_tick(api, s, h):
    """Inside the substep loop: play back each spoken-for moon (its free drift, then the transfer), the hits, the capture;
    and the farm's pincer meteors."""
    rs = s.get("rs")
    if not rs:
        return
    t = s["t"]
    fm = rs.get("farm")
    if fm is not None:
        for m in fm["mets"]:
            if m["ts"][0] <= t <= m["ts"][-1]:
                ps_fired(s, rs, m)
                m["on"] = True
                m["x"], _ = _rb2_at(m["ts"], m["xs"], t)
                m["hist"].insert(0, m["x"].copy()); del m["hist"][RB2_TAIL:]
        if t >= fm["T"]:
            rs.setdefault("done_flashes", []).extend([(fm["P"].copy(), s["itime"])] * 2)
            add_puff(s, fm["P"], FM_COLL_PUFF if fm.get("coll") else FM_PUFF, ride=bool(fm.get("coll")))
            rs["farm"] = None
            print("farm: " + ("the meteors COLLIDE -- debris in the band" if fm.get("coll") else
                              "PINCER hit -- the moon keeps its orbit, the ring gets fed"), file=sys.stderr)
    if not rs["active"]:
        return
    for i, r in list(rs["active"].items()):
        f = s["free"].get(i)
        if f is None or f["gone"]:
            rs["active"].pop(i, None)
            continue
        if t < r["T1"]:
            f["x"], f["v"] = _rb2_at(*r["free"], t)
        elif t < r["T2"]:
            if not r["hit1"]:
                r["hit1"] = True
                r.setdefault("flashes", []).append((f["x"].copy(), s["itime"]))
                add_puff(s, f["x"])
                print(f"rescue: moon {i} TRANSFER hit", file=sys.stderr)
            f["x"], f["v"] = _rb2_at(*r["transfer"], t)
        else:
            r.setdefault("flashes", []).append((r["transfer"][1][-1].copy(), s["itime"]))
            add_puff(s, r["transfer"][1][-1])
            s["free"].pop(i, None)
            rs["active"].pop(i, None)
            rs.setdefault("done_flashes", []).extend(r["flashes"])
            print(f"rescue: moon {i} CAPTURED -- back on its orbit", file=sys.stderr)
            continue
        for m in r["mets"]:
            if t >= m["ts"][0] and t <= m["ts"][-1]:
                ps_fired(s, rs, m)
                m["on"] = True
                m["x"], _ = _rb2_at(m["ts"], m["xs"], t)
                m["hist"].insert(0, m["x"].copy()); del m["hist"][RB2_TAIL:]
            elif t > m["ts"][-1]:
                m["on"] = False

def _rb2_at(ts, xs, t):
    k = int(np.searchsorted(ts, t))
    k = min(max(k, 1), len(ts) - 1)
    a = (t - ts[k - 1]) / max(ts[k] - ts[k - 1], 1e-9)
    return xs[k - 1] + (xs[k] - xs[k - 1]) * min(max(a, 0.0), 1.0), (xs[k] - xs[k - 1]) / max(ts[k] - ts[k - 1], 1e-9)

def rb2_tick(api, s, h):
    """Inside the substep loop (sim time): PLAYBACK of the planned spit / meteor paths (exactly as searched, so every
    meteor meets its body), then the impacts."""
    rb = s.get("rb")
    if rb is not None and rb.get("ph") == 3 and rb["hit_done"]:
        t = s["t"]
        for fu in rb.get("fus", []):
            if fu["done"]:
                continue
            lp = fu.get("loop")
            if lp is not None and t < fu["T"] and fu["i"] in s["free"]:
                f = s["free"][fu["i"]]
                f["x"], f["v"] = _rb2_at(lp[0] + rb["t_hit"], lp[1], t)
            for m in fu["mets"]:
                if m["ts"][0] <= t <= m["ts"][-1]:
                    m["on"] = True
                    m["x"], _ = _rb2_at(m["ts"], m["xs"], t)
                    m["hist"].insert(0, m["x"].copy()); del m["hist"][RB2_TAIL:]
            if t >= fu["T"]:
                fu["done"] = True
                for m in fu["mets"]:
                    m["on"] = False
                if lp is not None:
                    s["free"].pop(fu["i"], None)
                    rb["fu_flashes"].extend([(fu["P"].copy(), s["itime"])] * 2)
                    add_puff(s, fu["P"], FM_PUFF)
                    print(f"rebirth: LOOP capture (brake pincer, {len(fu['mets'])} meteors) -- moon {fu['i']} on its orbit, opposite its first debris", file=sys.stderr)
                else:
                    rb["fu_flashes"].extend([(fu["P"].copy(), s["itime"])] * 2)
                    add_puff(s, fu["P"], FM_PUFF)
                    print(f"rebirth: FOLLOW-UP pincer on moon {fu['i']}", file=sys.stderr)
        return
    if rb is None or rb.get("ph") != 3 or rb["hit_done"]:
        return
    t, res = s["t"], rb2_collect(rb)
    if res is None:
        if t >= rb["t_ready"]:
            print("rebirth: no saved plan yet (still searching) -- instant reset", file=sys.stderr)
            rb["res"] = "fail"
        return
    if res == "fail":
        s["rb"] = None
        bd_reset(api, s)
        s["mt_next"] = s["itime"] + 2.0
        return
    for i, r in enumerate(res):
        if r is None:
            continue
        if t >= r["bts"][0]:
            x, v = _rb2_at(r["bts"], r["bxs"], t)
            if not rb["spat"][i]:
                rb["spat"][i] = True
                if rb["spat"][[k for k, q in enumerate(res) if q is not None]].all():
                    print("rebirth: SPIT -- every body out of the dwarf", file=sys.stderr)
            s["free"][i] = dict(x=x, v=v, gone=False, rad=float(rb["rad"][i]), rebirth=True, path=True)
        m = r["met"]
        if m is not None and t >= m["ts"][0]:
            m["on"] = True
            m["x"], _ = _rb2_at(m["ts"], m["xs"], t)
            m["hist"].insert(0, m["x"].copy()); del m["hist"][RB2_TAIL:]
    if t >= rb["t_hit"]:
        rb["hit_done"], rb["t_hit_real"] = True, s["itime"]
        for i in range(N_MOONS):
            fu = res[i].get("fu") if res[i] else None
            if fu and fu.get("loop") is not None:
                p = fu["x1"]
                lt, lx = fu["loop"]
                s["free"][i] = dict(x=p.copy(), v=(lx[1] - lx[0]) / max(lt[1] - lt[0], 1e-9), gone=False,
                                    rad=float(rb["rad"][i]), rebirth=True, path=True)
            else:
                s["free"].pop(i, None)
                p = rb["P"][i]
            add_puff(s, p)
            rb["flashes"].append((p.copy(), s["itime"]))
        rb["puff_t"] = s["itime"]
        print("rebirth: IMPACTS -- every body back on its orbit", file=sys.stderr)

def rb2_draw(api, s):
    rb = s.get("rb")
    if rb is not None and rb.get("ph") == 3 and rb.get("hit_done") and rb.get("fus") is not None:
        mets = [m for fu in rb["fus"] for m in fu["mets"] if m["on"]]
        rb["fu_flashes"] = [x for x in rb["fu_flashes"] if s["itime"] - x[1] < 0.6]
        fl = [x for x in rb["flashes"] if s["itime"] - x[1] < 0.6] + rb["fu_flashes"]
        if not mets and not fl:
            api.render_uniform("u_cmetOn", 0.0)
            return
        rb = dict(res=[dict(met=m) for m in mets], flashes=fl, hit_done=False)
    if rb is None or rb.get("ph") != 3:
        rs = s.get("rs") or {}
        mets = [m for r in rs.get("active", {}).values() for m in r["mets"]] + (rs["farm"]["mets"] if rs.get("farm") else [])
        mets = [m for m in mets if m.get("kind") not in PS_SHIPS]
        fl = [x for r in rs.get("active", {}).values() for x in r.get("flashes", [])] + rs.get("done_flashes", [])
        rs["done_flashes"] = [x for x in rs.get("done_flashes", []) if s["itime"] - x[1] < 0.6]
        if not mets and not fl:
            api.render_uniform("u_cmetOn", 0.0)
            return
        rb = dict(res=[dict(met=m) for m in mets], flashes=fl, hit_done=False)
    cm = np.zeros((16, RB2_TAIL + 1, 4), dtype="f4")
    res = rb.get("res")
    mets = [r["met"] if r else None for r in res] if isinstance(res, list) else []
    for i, m in enumerate(mets[:16]):
        if m is None or not m["on"] or rb["hit_done"]:
            continue
        for k, p in enumerate(m["hist"][:RB2_TAIL]):
            cm[i, k] = (*p, 1.6 * (1.0 - k / RB2_TAIL) ** 1.5 + 0.05)
    for j, (p, t0) in enumerate(rb["flashes"][:16]):
        age = max(0.0, s["itime"] - t0)
        if age < 0.6:
            cm[j, RB2_TAIL] = (*p, -(0.012 + 0.06 * math.sqrt(age)))
    on = (not rb["hit_done"]) or s["itime"] - rb.get("t_hit_real", -9.0) < 0.7
    api.render_uniform("u_cmetOn", 1.0 if on else 0.0)
    if not on:
        return
    s["cm_tex"].write(cm.tobytes())
    cw, ch = (int(round(v)) for v in hand.canvas(api))
    if s["cmet"].size != (cw, ch):
        s["cmet"] = api.ctx.texture((cw, ch), 1, dtype="u4"); s["cmet"].filter = (moderngl.NEAREST, moderngl.NEAREST)
        api.expose("u_cmet", s["cmet"])
    k = s["k"]
    api.run(k["PCLEAR"], ((cw + 15) // 16, (ch + 15) // 16, 1), images={0: s["cmet"], 1: s["dust"]["cnt"]},
            uniforms={"u_canvas": (cw, ch)})
    ro, fw, rt, up = mt_camera(s["itime"])
    api.run(k["CSPLAT"], ((16 * (RB2_TAIL + 1) + 63) // 64, 1, 1), images={0: s["cm_tex"], 1: s["cmet"]},
            uniforms={"u_canvas": (cw, ch), "u_cmSize": (RB2_TAIL + 1, 16), "u_fov": FOV_,
                      "u_ro": tuple(float(x) for x in ro), "u_fw": tuple(float(x) for x in fw),
                      "u_rt": tuple(float(x) for x in rt), "u_up": tuple(float(x) for x in up)})

def dust_step(api, s):
    """The ring's grains while the hole is out: move (dwarf + hole + box gravity, glass bounces), count, draw."""
    dw = s["dust"]
    api.render_uniform("u_dustOn", 1.0 if dw["on"] else 0.0)
    if not dw["on"]:
        return
    k = s["k"]
    cw, ch = (int(round(v)) for v in hand.canvas(api))
    if dw["img"].size != (cw, ch):
        dw["img"] = api.ctx.texture((cw, ch), 1, dtype="u4"); dw["img"].filter = (moderngl.NEAREST, moderngl.NEAREST)
        api.expose("u_dust", dw["img"])
    api.run(k["PCLEAR"], ((cw + 15) // 16, (ch + 15) // 16, 1), images={0: dw["img"], 1: dw["cnt"]},
            uniforms={"u_canvas": (cw, ch)})
    hp = HOLE[0] if HOLE is not None else np.zeros(3)
    g = BOX_G * BOX_AX[2]
    api.run(k["PMOVE"], ((DUST_N + 255) // 256, 1, 1), images={0: dw["pos"], 1: dw["vel"], 2: dw["cnt"]},
            uniforms={"u_np": DUST_N, "u_nsub": DUST_SUB, "u_h": float(min(s.get("dt", 0.01), DT_MAX) * sim_rate() / DUST_SUB),
                      "u_hole": (float(hp[0]), float(hp[1]), float(hp[2]), float(HOLE[1]) if HOLE is not None else 0.0),
                      "u_holeR": HOLE_R, "u_g": tuple(float(x) for x in g),
                      "u_bx": tuple(float(x) for x in BOX_AX[0]), "u_by": tuple(float(x) for x in BOX_AX[1]),
                      "u_bz": tuple(float(x) for x in BOX_AX[2]), "u_bh": tuple(float(x) for x in BOX_H)})
    ro, fw, rt, up = mt_camera(s["itime"])
    api.run(k["PSPLAT"], ((DUST_N + 255) // 256, 1, 1), images={0: dw["pos"], 1: dw["vel"], 2: dw["img"]},
            uniforms={"u_np": DUST_N, "u_canvas": (cw, ch), "u_fov": FOV_, "u_gain": DUST_GAIN,
                      "u_ro": tuple(float(x) for x in ro), "u_fw": tuple(float(x) for x in fw),
                      "u_rt": tuple(float(x) for x in rt), "u_up": tuple(float(x) for x in up)})
    if s.get("n", 0) % 10 == 0:
        dw["alive"] = int(np.frombuffer(dw["cnt"].read(), dtype="u4")[0])

def pad_forces(api):
    """The pad's non-camera surface: (gates, fire, speed).

    gates = (dwarf, moon) SWITCH states. ON = the shader's own automatic shock system is
    ENGAGED (the house rule); OFF = it is quiet and only a button can fire. fire = (big, moon,
    aurora) BUTTON presses landing on THIS frame (1.0 on that frame, else 0.0) -- a press is
    explicit intent, so it is never gated by a switch. speed = the Speed dial as a multiplier
    of this shader's own rate (1.0 when the pad carries no dial), and it changes nothing else.
    Padded defensively: a shader must not care how long a tuple the renderer hands it."""
    inp = getattr(api, "inputs", None) or {}
    gates = tuple(inp.get("pad_shocks") or ()) + (0.0, 0.0)
    fire = tuple(inp.get("pad_fire") or ()) + (0.0, 0.0, 0.0)
    return gates[:2], fire[:3], float(inp.get("pad_speed", 1.0))

GRAB_FOLLOW_S = 0.08
GRAB_THROW = 1.0

def mouse_grab(api, s, xy, held, dt, look, move):
    """MOUSE GRAB (mode 2; 2026-10-19, user: "mode 2 hold doesn't work"): PRESS catches the moons under the cursor (the
    reticle's catch zone), HOLD drags them with the cursor at their own depth, RELEASE throws them with the cursor's
    velocity."""
    g = s.setdefault("grab", dict(on=False, held=[], pv=None, vel=np.zeros(3)))
    ro, fw, rt, up = mt_camera(s["itime"], look, move)
    ro, rd = hand.ray(xy[0], xy[1], hand.canvas(api), ro, fw, rt, up, FOV_)
    rate = max(sim_rate(), 1e-6)
    if held and not g["on"]:
        m = s["moons_np"]
        C = m[0, :, :3].astype(np.float64); Rr = m[0, :, 3].astype(np.float64)
        rel = C - ro[None]; dist = np.maximum(np.linalg.norm(rel, axis=1), 1e-9)
        ang = np.arccos(np.clip((rel @ rd) / dist, -1.0, 1.0)) - np.arcsin(np.clip(Rr / dist, 0.0, 1.0))
        g["held"] = []
        for k in np.nonzero(ang < FLING_CATCH_ANG)[0]:
            f = _release_moon(s, int(k), m)
            if f is not None:
                g["held"].append((int(k), float(rel[k] @ rd)))
        g["on"], g["pv"], g["vel"] = True, None, np.zeros(3)
        print(f"mouse: grabbed moons {[k for k, _ in g['held']]}", file=sys.stderr)
    elif held and g["on"]:
        P0 = ro + rd * (g["held"][0][1] if g["held"] else 7.0)
        if g["pv"] is not None and dt > 1e-4:
            g["vel"] = 0.7 * g["vel"] + 0.3 * (P0 - g["pv"]) / dt
        g["pv"] = P0
        for k, depth in g["held"]:
            f = s.get("free", {}).get(k)
            if f is None or f["gone"]:
                continue
            P = ro + rd * depth
            f["v"] = (P - f["x"]) / (GRAB_FOLLOW_S * rate)
            f["touch"] = s["itime"]
    elif not held and g["on"]:
        for k, _ in g["held"]:
            f = s.get("free", {}).get(k)
            if f is not None and not f["gone"]:
                f["v"] = g["vel"] * GRAB_THROW / rate
                f["touch"] = s["itime"]
        if g["held"]:
            print(f"mouse: threw moons {[k for k, _ in g['held']]} at {float(np.linalg.norm(g['vel'])):.2f}/s", file=sys.stderr)
        g["on"], g["held"] = False, []

def bd_fling(api, s, ret, dt, look, move):
    """The reticle: everything drifting into it is flung along its own ray.

    A CONTINUOUS catch, not a press: hold the stick and whatever wanders in gets launched, which
    is the whole point -- orbiting moons are hard to hit with a cursor, and this is a trap you
    sweep across them. Application is a velocity impulse per frame, so a body sitting in the
    reticle accelerates hard and clears out, then settles to coast in the dwarf's real gravity."""
    ro, fw, rt, up = mt_camera(s["itime"], look, move)
    ro, rd = hand.ray(ret[0], ret[1], hand.canvas(api), ro, fw, rt, up, FOV_)
    m = s["moons_np"]
    mets = [mm for mm in s.get("mt_list", []) if mm["alive"]]
    C = np.array([m[0, i, :3] for i in range(N_MOONS)] + [mm["x"] for mm in mets], dtype=np.float64)
    Rr = np.array([float(m[0, i, 3]) for i in range(N_MOONS)] + [0.03] * len(mets))
    rel = C - ro[None]
    dist = np.maximum(np.linalg.norm(rel, axis=1), 1e-9)
    ang = np.arccos(np.clip((rel @ rd) / dist, -1.0, 1.0)) - np.arcsin(np.clip(Rr / dist, 0.0, 1.0))
    sd = rd - fw * float(rd @ fw)
    sd = sd / float(np.linalg.norm(sd)) if float(np.linalg.norm(sd)) > 1e-6 else up
    for k in np.nonzero(ang < FLING_CATCH_ANG)[0]:
        if k < N_MOONS:
            was_free = k in s.get("free", {})
            f = _release_moon(s, k, m)
            if f is not None:
                f["v"] = f["v"] + sd * FLING_ACC_MOON * dt
                if not was_free:
                    print(f"pad: reticle caught moon {k} -> flung loose, "
                          f"{len(s['free'])} loose", file=sys.stderr)
        else:
            mm = mets[k - N_MOONS]
            mm["w"] = mm["w"] + sd * FLING_ACC_MET * dt

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
    s["moons"] = api.texture2d((N_MOONS, 3), "rgba32f", nearest=True)
    s["phase"] = api.texture2d((NR, 1), "rgba32f", nearest=True)
    s["t"] = 0.0
    s["rp"], s["rv"] = ros_init()
    s["cp"], s["cw"] = co_init()
    s["rR"], s["rw"] = np.eye(3), rock_w0()
    s["cur"] = 0
    s["k"] = {n: api.kernel("sim.comp", n) for n in ("INIT", "ADVECT", "FORCE", "DARK", "NOTES")}
    s["k"]["WANT"] = api.kernel("sim.comp", "WANT")
    s["k"]["PROF"] = api.kernel("sim.comp", "PROF")
    s["prof_tex"] = api.texture2d((NR, 1), "rgba32f", nearest=True)
    for n in ("RZERO", "PSEED", "PMOVE", "PCLEAR", "PSPLAT", "CSPLAT"):
        s["k"][n] = api.kernel("sim.comp", n)
    dimg = api.ctx.texture((1, 1), 1, dtype="u4"); dimg.filter = (moderngl.NEAREST, moderngl.NEAREST)
    dcnt = api.ctx.texture((1, 1), 1, dtype="u4"); dcnt.filter = (moderngl.NEAREST, moderngl.NEAREST)
    s["dust"] = dict(pos=api.texture2d((1024, DUST_N // 1024), "rgba32f", nearest=True),
                     vel=api.texture2d((1024, DUST_N // 1024), "rgba32f", nearest=True),
                     img=dimg, cnt=dcnt, on=False, alive=0)
    s["dust"]["vel"].write(bytes(DUST_N * 16))
    api.expose("u_dust", dimg)
    rb2_precompute(s)
    s["rb_puff_tex"] = api.texture2d((16, 1), "rgba32f", nearest=True)
    s["rb_puff_tex"].write(bytes(16 * 16))
    s["cm_tex"] = api.texture2d((RB2_TAIL + 1, 16), "rgba32f", nearest=True)
    s["shot_tex"] = api.texture2d((RB2_TAIL + 1, PS_SHOT_N), "rgba32f", nearest=True)
    s["shot_tex"].write(bytes((RB2_TAIL + 1) * PS_SHOT_N * 16))
    api.expose("u_shots", s["shot_tex"])
    s["cmet"] = api.ctx.texture((1, 1), 1, dtype="u4"); s["cmet"].filter = (moderngl.NEAREST, moderngl.NEAREST)
    api.expose("u_cmet", s["cmet"])
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
    s["chase_tex"] = api.texture2d((CH_NT + 2, 1 + CH_SW_N), "rgba32f", nearest=True)
    s["chase_tex"].write(bytes((CH_NT + 2) * (1 + CH_SW_N) * 16))
    api.expose("u_chase", s["chase_tex"])
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
    write_moons(s)
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
    global CAM_LOOK, CAM_MOVE, PAD_SPEED
    s = api.state
    dt = min(max(dt, 1e-4), DT_MAX)
    s["dt"] = dt
    look, move, ret, palive = pad_state(api)
    CAM_LOOK, CAM_MOVE = look, move
    (pad_dsw, pad_msw), pad_fire, PAD_SPEED = pad_forces(api)
    pw, pt, pd = pad_db(s)
    pad_dsw = 0.0 if _wsw(pw, "dwarf_shocks_off") else 1.0
    pad_msw = 0.0 if _wsw(pw, "moon_shocks_off") else 1.0
    if (pad_dsw, pad_msw) != s.get("pad_gates"):
        s["pad_gates"] = (pad_dsw, pad_msw)
        print(f"pad: dwarf shocks {'ON' if pad_dsw > 0.5 else 'off'}, "
              f"moon shocks {'ON' if pad_msw > 0.5 else 'off'}", file=sys.stderr)
        if pad_dsw < 0.5 and s.get("n", 0) == 0:
            print("pad: (no widget touched yet -- ON = the dwarf sparks by itself, "
                  "the buttons fire either way)", file=sys.stderr)
    if abs(PAD_SPEED - s.get("pad_speed", 1.0)) > 1e-6:
        s["pad_speed"] = PAD_SPEED
        print(f"pad: speed x{PAD_SPEED:.2f}", file=sys.stderr)
    if pad_fire[0] > 0.5:
        print("pad: dwarf big shock", file=sys.stderr)
    if pad_fire[1] > 0.5:
        print("pad: moon shock", file=sys.stderr)
    if pad_fire[2] > 0.5:
        print("pad: ring aurora burst", file=sys.stderr)
    if pt.get("black_hole") or pt.get("Rebirth"):
        s["pdb_rebirth"] = True
    s["rs_off"], s["fm_off"] = _wsw(pw, "auto_correct_off"), _wsw(pw, "farm_off")
    if (pw.get("budget") or {}).get("type"):
        global RS_BUDGET
        RS_BUDGET = int(round(max(3.0, min(12.0, float(pw["budget"].get("value", 3.0))))))
    api.render_uniform("u_dbHue", (float((pw.get("ring_hue") or {}).get("value", 0.0)),
                                   float((pw.get("dwarf-hue") or {}).get("value", 0.0))))
    api.render_uniform("u_chaseHue", (float((pw.get("squid_hue") or {}).get("value", 0.0)),
                                      float((pw.get("swarm_hue") or {}).get("value", 0.0))))
    ro_, rt_, up_, fw_ = cam_db_step(s, pd, pt.get("Recenter", 0) > 0 or pt.get("cam_recenter", 0) > 0)
    api.render_uniform("u_dbCamOn", 1.0)
    for nm, vv in (("u_dbRo", ro_), ("u_dbRt", rt_), ("u_dbUp", up_), ("u_dbFw", fw_)):
        api.render_uniform(nm, tuple(float(x) for x in vv))
    minp = tuple((getattr(api, "inputs", None) or {}).get("mouse", (0.0,) * 7))
    if pt.get("Poke") or pt.get("boop"):
        bd_boop(api, s, (minp[1], minp[2]), look, move)
    if int(minp[0]) == 2 or s.get("grab", {}).get("on"):
        mouse_grab(api, s, (minp[1], minp[2]), int(minp[0]) == 2 and minp[3] > 0.5, dt, look, move)
    if int(minp[0]) == 5 and minp[4] != s.get("m_met_seen", minp[4]):
        mouse_meteor(api, s, (minp[5], minp[6]), look, move)
    s["m_met_seen"] = minp[4]
    if rb_poll(s):
        rb_start(api, s, "trigger", merger=True)
    rb_step(api, s)
    rs_step(api, s)
    chase_step(api, s, dt)
    puff_flush(s)
    if s.get("rb") is None:
        rb2_draw(api, s)
    ps_shots_draw(api, s)
    for kind, xy in hand.events(api, s):
        if kind == "reset":
            rb_start(api, s, "hand reset keys", merger=True)
        else:
            bd_boop(api, s, xy, look, move)
    if palive > 0.5 and ret[2] > 0.5:
        bd_fling(api, s, ret, dt, look, move)
    h = dt * sim_rate() / SUBSTEPS
    profr = s.get("n", 0) % 600 == 5 and getattr(api, "ctx", None) is not None
    if profr:
        qr = api.ctx.query(time=True)
        qr.__enter__()
    for _ in range(SUBSTEPS):
        ros_step(s, h)
        co_step(s, h)
        s["t"] += h
        free_step(s, h)
        rb2_tick(api, s, h)
        rs_tick(api, s, h)
        s["moons_np"] = free_apply(s, moon_positions(s["t"], s["rp"], s["itime"], (s["cp"], s["cw"])))
        write_moons(s)
        s["phase"].write(phases(s["t"], h).tobytes())
        a, b = s["f"][s["cur"]], s["f"][1 - s["cur"]]
        api.run(s["k"]["ADVECT"], s["groups"], images={0: b}, samplers={"u_src": a, "u_phase": s["phase"]},
                uniforms={"u_h": h})
        api.run(s["k"]["FORCE"], s["groups"], images={0: a},
                samplers={"u_src": b, "u_phase": s["phase"], "u_moons": s["moons"], "u_rbPuff": s["rb_puff_tex"]},
                uniforms={"u_h": h, "u_nMoons": N_MOONS,
                          "u_rbHole": s.get("rb_ring", (0.0, 0.0, 0.0, 0.0)), "u_rbRefill": s.get("rb_refill", 1.0),
                          "u_splat": (s["splat"][0], s["splat"][1], SPLAT_V / SPLAT_T / SIM_RATE,
                                      1.0 if s["itime"] - s["splat"][2] < SPLAT_T else 0.0)})
    if profr:
        qr.__exit__(None, None, None)
        print(f"profile (GPU ms): ring sim {qr.elapsed / 1e6:.2f} ({SUBSTEPS} substeps, {N_MOONS} moons)", file=sys.stderr)
    api.render_uniform("u_simT", float(s["t"] % 10000.0))
    api.run(s["k"]["NOTES"], (1, 1, 1), images={0: s["notes"], 1: s["spokes"]},
            uniforms={"u_dt": float(dt),
                      "u_pad_moon_gate": float(pad_msw),
                      "u_pad_big_gate": float(pad_dsw),
                      "u_pad_moon_fire": 1.0 if pad_fire[1] > 0.5 else 0.0,
                      "u_pad_big_fire": 1.0 if pad_fire[0] > 0.5 else 0.0})
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
            aurora_step(api, s, dt, burst=pad_fire[2] > 0.5)
        tq["aurora"] = q.elapsed / 1e6
        print("profile (GPU ms): " + ", ".join(f"{k} {v:.2f}" for k, v in tq.items())
              + f" | dbm idle {s['dbm_idle']}", file=sys.stderr)
    else:
        aurora_step(api, s, dt, burst=pad_fire[2] > 0.5)
    meteor_step(api, s, dt)
    churn_step(api, s)
    rock_step(s, dt)
    rock_uniforms(api, s)
    s["itime"] += dt
    s["next_bolt"] -= dt
    if s["next_bolt"] <= 0.0:
        if pad_dsw > 0.5:
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
        s["ring_mean"] = float(np.nanmean(S))
        if s.get("rb") is None:
            s["ring_ref"] = max(s.get("ring_ref", 0.0), s["ring_mean"])
        bad = ~np.isfinite(a).all(axis=-1)
        j = int(np.nanargmax(np.abs(np.nan_to_num(ur)).max(axis=1)))
        print(f"ring t={s['t']:.1f} S mean {np.nanmean(S):.3f} max {np.nanmax(S):.2f} |ur| max {np.nanmax(np.abs(ur)):.2e} "
              f"(r {ROW_R[j]:.3f}) |vp| max {np.nanmax(np.abs(vp)):.2e} nonfinite {int(bad.sum())}", file=sys.stderr)
    _expose(api, s)

PROFILE = False
PROF_EVERY = 600
if PROFILE:
    import functools as _ft
    _PROF = {}

    def _prof_wrap(name):
        f = globals()[name]

        @_ft.wraps(f)
        def w(*a, **k):
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                d = _PROF.setdefault(name, [0.0, 0])
                d[0] += time.perf_counter() - t0
                d[1] += 1
        globals()[name] = w

    for _n in ("pad_state", "pad_forces", "pad_db", "rb_poll", "rb_step", "rs_step", "puff_flush", "rb2_draw", "ros_step",
               "co_step", "free_step", "rb2_tick", "rs_tick", "moon_positions", "free_apply", "write_moons", "phases",
               "dbm_tick", "aurora_step", "meteor_step", "churn_step", "rock_step", "rock_uniforms", "dust_step",
               "cam_db_step", "_expose", "mt_camera", "mt_search_step", "mt_fly", "mt_bodies", "mt_acc", "_bolt_fire"):
        if _n in globals():
            _prof_wrap(_n)
    _step_inner = step

    def step(api, dt):
        t0 = time.perf_counter()
        _step_inner(api, dt)
        d = _PROF.setdefault("STEP (total)", [0.0, 0])
        d[0] += time.perf_counter() - t0
        d[1] += 1
        if d[1] % PROF_EVERY == 0:
            n = d[1]
            top = sorted(_PROF.items(), key=lambda kv: -kv[1][0])[:14]
            print("cpu profile (ms / frame over %d frames): " % n
                  + ", ".join(f"{k} {1000 * v[0] / n:.2f}" + (f" x{v[1] / n:.1f}" if v[1] != n else "") for k, v in top),
                  file=sys.stderr)
            _PROF.clear()
