"""ctrl_ringbox_v2.mp -- REMOTE-ONLY copy of ringbox_v2.mp (scope 2026-10-22): the horseshoe MODES (normal / slingshot / coast / recover) and the 16M ring. The auto half (ringbox_v2.mp) stays pure.

Everything that moves anything is in the LEDGER in scope.md, and nowhere else. In this file:
  - star + moons: ONE REBOUND simulation (IAS15), G = 1, star mass 1, every body pulls on every other body.
  - moon/moon and moon/star contact: MERGE (masses add, momentum conserved, radius from combined volume);
    the impact energy 1/2 mu v_rel^2 becomes HEAT.
  - heat: one number per moon. Sources: merge energy, the star's light (equilibrium ~ r^-1/2, the real law).
    Cooling toward that equilibrium over COOL_TAU (a declared speed-up).
  - walls: a world-fixed cube of half-width BOX_W; a body moving out of it has its outward speed reversed x WALL_E
    (toy rule 1). Velocity only, no position is set.
  - ring: GRAINS on the GPU (ring.comp), each a massless test particle in the gravity of the star + the live moons.
    A grain touching a moon or the star is absorbed (gone). Walls reflect grains too.
  - your hand: boop (a velocity kick at the cursor), grab / throw (mouse mode 2). Reset = a fresh system.
STEP 2 (scope.md 2026-10-21 STEP 2 PLAN): two ships as one team push disturbed moons back toward their home orbits with
lasers and nukes. Every push is ABLATION: the moon loses mass J / V_EJ and that mass becomes ejecta grains (GPU pool).
"""
import math
import os
import sys
import importlib
from collections import deque

import moderngl
import numpy as np

try:
    import rebound
except ImportError:
    rebound = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_lib"))
import hand
importlib.reload(hand)

def _cross(a, b):
    """3-vector cross product without numpy's generic overhead (np.cross was 60 % of a ship job's CPU, 2026-10-23)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])

SIM_RATE = 0.6
DT_MAX = 0.05
STAR_R = 0.22
R_IN, R_OUT = 1.0, 2.4
BOX_W = 3.0 * R_OUT
WALL_E = 0.85
PAIR = ("A", "B")
PAIR_A = 1.45
_mA, _mB = 2.0e-4, 5.0e-4
_DA = PAIR_A * math.sqrt(8.0 * (_mA + _mB) / 3.0 * (1.86 - 1.5))
MOONS = [
    ("A", PAIR_A + _DA * _mB / (_mA + _mB), _mA, 0.065, 0.0, 320.0),
    ("B", PAIR_A - _DA * _mA / (_mA + _mB), _mB, 0.085, 0.0, 140.0),
    ("C", 2.55, 3.0e-5, 0.075, 0.0, 250.0),
]
N_MOONS = len(MOONS)
T_STAR_EQ = 1.15
HEAT_C = 0.02
COOL_TAU = 20.0
STAR_LUM = 1.0
STAR_CYCLE = 180.0
STAR_MODES = ("white/blue", "orange", "red")
FLARE_FIRST = 8.0
FLARE_GAP = 26.0
FLARE_SIM = 4.5
FLARE_COS = math.cos(math.radians(36.0))
FLARE_J = 5.0e-6
FLARE_RING_ACC = 0.05
BOOP_DV = 0.20
GRAB_FOLLOW_S = 0.08
GRAB_THROW = 1.0
SPLASH_R, SPLASH_T, SPLASH_DV = 0.18, 0.3, 0.25
PLOW_R, PLOW_A = 0.12, 0.8
PICK_SLACK = 0.12
CAM_EL = math.radians(28.0)
CAM_AZ = math.radians(0.0)
CAM_FOV = 0.20
CAM_D = R_OUT * 1.08 / CAM_FOV
CAM_MODE = "normal"
PORTRAIT_EL = math.radians(32.0)
PORTRAIT_AZ = math.radians(0.0)
PORTRAIT_ROLL = math.radians(90.0)
PORTRAIT_ZOOM = 1.20
RIDE_BACK, RIDE_UP, RIDE_AHEAD = 0.30, 0.07, 0.6
RIDE_FOV = 0.55
RIDE_TAU = 0.25
_RIDE = dict(ro=None, at=None, dir=None)
RIDE_MODES = {2: "ship0", 3: "ship1", 4: "C"}
FREE_MOVE, FREE_LOOK, FREE_DEAD, FREE_STEP = 2.5, 0.9, 0.10, 0.25
STICK_MOVE, STICK_LOOK = "joystick", "joystick2"
CAM_PRESETS = ("normal", "portrait", "RIDE ship A", "RIDE ship B", "SIT on C")
_FREE = dict(ro=None, yaw=0.0, pitch=0.0)

def _dz(v):
    return 0.0 if abs(v) < FREE_DEAD else float(v)

def free_engage(ro, fw):
    """Start the free camera from the pose the screen shows right now."""
    _FREE["ro"] = np.array(ro, dtype=float)
    _FREE["yaw"] = math.atan2(fw[1], fw[0])
    _FREE["pitch"] = math.asin(max(-1.0, min(1.0, fw[2])))

def free_update(mx, my, lx, ly, climb, dt):
    _FREE["yaw"] -= lx * FREE_LOOK * dt
    _FREE["pitch"] = max(-1.5, min(1.5, _FREE["pitch"] + ly * FREE_LOOK * dt))
    fw, rt, _ = free_basis()
    ro = _FREE["ro"] + (fw * my + rt * mx) * FREE_MOVE * dt + np.array([0.0, 0.0, climb * FREE_STEP])
    _FREE["ro"] = np.clip(ro, -CAM_D * 1.5, CAM_D * 1.5)

def free_basis():
    y, p = _FREE["yaw"], _FREE["pitch"]
    fw = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    rt = _cross(fw, [0.0, 0.0, 1.0]); rt /= max(float(np.linalg.norm(rt)), 1e-9)
    return fw, rt, _cross(rt, fw)

def cam_fov():
    return RIDE_FOV if CAM_MODE in ("ride", "free") else CAM_FOV

def _ease(prev, want, dt, tau=RIDE_TAU):
    if prev is None:
        return np.array(want, dtype=float)
    k = 1.0 - math.exp(-dt / tau)
    return prev + (np.array(want, dtype=float) - prev) * k

def ride_update(w, mode_k, dt):
    """Per frame: aim the ride cam at the chosen body. Returns False when the target is missing (-> normal cam)."""
    nm = RIDE_MODES.get(mode_k)
    if nm is None or w.get("sim") is None or nm not in w["ids"]:
        _RIDE.update(ro=None, at=None, dir=None)
        return False
    if nm in SHIPS and w["ships"][nm].get("dead_until") is not None:
        _RIDE.update(ro=None, at=None, dir=None)
        return False
    p = _pp(w, nm)
    st = w["sim"].particles[0]
    x = np.array(p.xyz); v = np.array(p.vxyz) - np.array(st.vxyz)
    zup = np.array([0.0, 0.0, 1.0])
    if nm in SHIPS:
        sp = float(np.linalg.norm(v))
        want_dir = v / sp if sp > 1e-6 else (_RIDE["dir"] if _RIDE["dir"] is not None else np.array([1.0, 0.0, 0.0]))
        d = _ease(_RIDE["dir"], want_dir, dt); d /= max(float(np.linalg.norm(d)), 1e-9)
        _RIDE["dir"] = d
        ro_want = x - d * RIDE_BACK + zup * RIDE_UP
        at_want = x + d * RIDE_AHEAD
    else:
        rC = float(p.r)
        out = x - np.array(st.xyz); out /= max(float(np.linalg.norm(out)), 1e-9)
        ro_want = x + out * (2.0 * rC) + zup * (2.5 * rC)
        at_want = np.array(st.xyz)
    _RIDE["ro"] = _ease(_RIDE["ro"], ro_want, dt)
    _RIDE["at"] = _ease(_RIDE["at"], at_want, dt)
    return True

GW, GH = 4096, 4096
SPLAT_K = 64.0
SPLAT_K_RING = SPLAT_K * (1024 * 1024) / (GW * GH)
PR_K = R_OUT ** 2 / (4.0 * 90.0 * 60.0 * SIM_RATE)
M_RING0 = 1.5e-3
RING_GRAIN_M = M_RING0 / (GW * GH)
RING_PULL = True
RING_GRID_N, RING_GRID_HALF, RING_GRID_SAMPLE, RING_GRID_EVERY = 48, 3.0, 16, 0.2
RING_EPS2 = (2.0 * RING_GRID_HALF / RING_GRID_N) ** 2

def ring_cells_from_counts(counts):
    """(K, 3) cell centres + (K,) masses for the non-empty cells of a RING_GRID_N^2 count array [iy, ix]."""
    iy, ix = np.nonzero(counts)
    if len(ix) == 0:
        return None
    cell = 2.0 * RING_GRID_HALF / RING_GRID_N
    P = np.stack([-RING_GRID_HALF + (ix + 0.5) * cell, -RING_GRID_HALF + (iy + 0.5) * cell, np.zeros(len(ix))], 1)
    return P, counts[iy, ix].astype(np.float64) * RING_GRID_SAMPLE * RING_GRAIN_M

def ring_acc(w, x):
    """The ring's pull at x (G = 1), from the last read grid; zero until the first read / with RING_PULL off."""
    cells = w.get("ring_cells")
    if not RING_PULL or cells is None:
        return np.zeros(3)
    P, M = cells
    d = P - np.asarray(x, dtype=float)
    r2 = (d * d).sum(1) + RING_EPS2
    return (M / (r2 * np.sqrt(r2))) @ d

def ring_kick(w, h):
    """RING PULLS BACK (ledger): one velocity kick per step on the star + moons (same splitting as THE ENGINE); the
    kinetic energy it adds/removes is booked in E_ext (the ring is an external, time-varying field: no reaction)."""
    sim = w["sim"]
    if not RING_PULL or sim is None or w.get("ring_cells") is None:
        return
    dKE = 0.0
    for nm, c, r, m, _, k in bodies(w):
        p = sim.particles[k]
        dv = ring_acc(w, c) * h
        v0 = np.array(p.vxyz)
        p.vx += float(dv[0]); p.vy += float(dv[1]); p.vz += float(dv[2])
        v1 = v0 + dv
        dKE += 0.5 * m * (float(v1 @ v1) - float(v0 @ v0))
    w["E_ext"] += dKE

CNT_SAMPLE = 256
CNT_EVERY = 1.0
RING_LOW = 0.6
FEED_PAIR = ("A", "B")
FEED_TIMEOUT = 150.0
FEED_COOL = 60.0
DIVE_DR = 0.3
DIVE_F = np.linspace(0.30, 0.75, 10)
DIVE_DVO = (0.0, 0.15, 0.3, 0.45)
DIVE_RBURN = 0.8
DIVE_QMIN = 0.40
DIVE_T = 8.0
DIVE_PLAN_EVERY = 0.5
DIVE_WAIT_MAX = 40.0
DIVE_FLY_MAX = 8.0
DIVE_EFF = 0.9
LOG_S = 10.0
SHIP_M, SHIP_R = 0.0, 0.006
NUKE_M, NUKE_R = 0.0, 0.004
R_PARK = 3.2
R_PARK_STEP = 0.15
PARK_DEAD = 0.01
A_SHIP = 6.0
PILOT_FILE = os.path.expanduser("~/.cache/shader-backdrop/ship_pilot.npz")
PILOT_ON = os.path.expanduser("~/.cache/shader-backdrop/ship_pilot_on")
PILOT_OBS, PILOT_H, PILOT_OUT = 39, 48, 9
PILOT_SHAPES = [(PILOT_OBS, PILOT_H), (PILOT_H,), (PILOT_H, PILOT_H), (PILOT_H,), (PILOT_H, PILOT_OUT), (PILOT_OUT,)]
PILOT_AIM = 0.15
PILOT_N = sum(int(np.prod(sh)) for sh in PILOT_SHAPES)
_PILOT = None
_PILOT_T = {"t": -1e9, "mtime": None}

def pilot_unpack(theta):
    out, k = [], 0
    for sh in PILOT_SHAPES:
        n = int(np.prod(sh))
        out.append(np.asarray(theta[k:k + n], dtype=float).reshape(sh))
        k += n
    return out

def pilot_set(theta):
    """The trainer (and the live poller) install a policy; None = back to the spring."""
    global _PILOT
    _PILOT = None if theta is None else pilot_unpack(theta)

def pilot_poll(now):
    """Live: re-read the policy file when it changes; only while PILOT_ON exists."""
    if now - _PILOT_T["t"] < 2.0:
        return
    _PILOT_T["t"] = now
    try:
        if not os.path.exists(PILOT_ON):
            if _PILOT is not None:
                pilot_set(None); _PILOT_T["mtime"] = None
                print("ringbox: PILOT off (spring)", file=sys.stderr)
            return
        mt = os.path.getmtime(PILOT_FILE)
        if mt != _PILOT_T["mtime"]:
            pilot_set(np.load(PILOT_FILE)["theta"]); _PILOT_T["mtime"] = mt
            print("ringbox: PILOT loaded (learned flight)", file=sys.stderr)
    except (OSError, KeyError, ValueError) as e:
        if _PILOT is not None:
            pilot_set(None)
        print(f"ringbox: PILOT load failed: {e}", file=sys.stderr)

def pilot_obs(w, sn, x, v, S, vS, aS, T, need, gate, a_spring):
    """The ship's view, in the target moon's orbital frame (radial / prograde / normal round the star): everything
    relative, so one policy serves any moon, any side, any phase of the orbit."""
    st = w["sim"].particles[0]
    xs, vs = _xv(st)
    pT = _pp(w, T)
    xT, vT = _xv(pT)
    r = xT - xs
    er = r / max(float(np.linalg.norm(r)), 1e-9)
    ez = _cross(r, vT - vs); ez /= max(float(np.linalg.norm(ez)), 1e-9)
    et = _cross(ez, er)
    R = np.stack([er, et, ez])
    other = [o for o in SHIPS if o != sn and o in w["ids"]]
    xo = (np.array(_pp(w, other[0]).xyz) - x) if other else np.zeros(3)
    g = grav(w, x)
    o = np.concatenate([
        R @ (x - xT) / D_STAND,
        R @ (v - vT),
        R @ (S - x) / D_STAND,
        R @ (vS - v),
        R @ (x - xs) / 2.0,
        R @ (v - vs),
        R @ g / A_SHIP,
        R @ aS / A_SHIP,
        R @ xo / 2.0,
        R @ need / 0.2,
        R @ a_spring / A_SHIP,
        [pT.r / 0.1, float(np.linalg.norm(x - xs)) / 2.0, 1.0 if gate == "eff" else 0.0,
         1.0 if w["ships"][sn]["laser"] else 0.0, float(np.linalg.norm(S - x)) / D_STAND, 1.0],
    ])
    return np.clip(o, -5.0, 5.0), R

def pilot_act(obs):
    """-> dict(a=thrust vector, laser=gate, nuke=gate, aim=correction (moon frame), yscale=yield multiplier)."""
    W1, b1, W2, b2, W3, b3 = _PILOT
    h = np.tanh(obs @ W1 + b1)
    h = np.tanh(h @ W2 + b2)
    o = np.tanh(h @ W3 + b3)
    return dict(a=o[:3] * A_SHIP, laser=float(o[3]), nuke=float(o[4]), aim=o[5:8] * PILOT_AIM,
                yscale=math.exp(0.5 * float(o[8])))

def _gate(rule, g):
    """A net gate overrides a rule only when it is decided (|g| > 0.5); near 0 the rule stands (warm start)."""
    return True if g > 0.5 else False if g < -0.5 else rule

V_CLOSE = 1.2
K_POS, TAU_V = 2.0, 0.25
D_STAND = 0.30
TEAM_ANG = math.radians(25.0)
FIRE_TOL = 0.06
FIRE_COS = math.cos(math.radians(65.0))
FIRE_COS_NOW = math.cos(math.radians(32.0))
AVOID, AVOID_STAR = 0.12, 0.20
STAR_ROUTE = 0.25
STAR_SAFE = 0.03
K_HOME, VR_MAX = 0.5, 0.15
V_EJ = 1.0
F_LASER = 5.0e-5
D_SHAPE = 0.015
SIZE_TOL = 0.002
PAIR_C_LO, PAIR_C_HI = 1.6, 2.3
PAIR_AIM = 0.1
LANE_TOL = 0.03
MODE_NORMAL = "normal"
MODE_SLING = "slingshot"
MODE_RECOVER = "recover"
MODE_STAR = "star_sling"
MODE_FARM = "star_farm"
MODE_ROSETTE = "rosette"
REFILL_ORDER = ("C", "A", "B")
LIFT_M = 7.6e-4
LIFT_R = 0.098
LIFT_R0 = 0.40
STAR_LIFT_J = 2.0e-3
FARM_STATION_R = 0.55
FARM_TIMEOUT = 90.0
ROSETTE_A = PAIR_A
ROSETTE_ANG = 2.0 * math.pi / 3.0
ROSETTE_K_PHASE = 0.38
ROSETTE_K_RAD = 0.55
ROSETTE_KP = 2.20
ROSETTE_KD = 2.60
ROSETTE_DV_LIMIT = 0.15
ROSETTE_PHASE_TOL = 0.06
ROSETTE_R_TOL = 0.06
ROSETTE_TIMEOUT = 320.0
STAR_F = 0.55
STAR_MIN_R = 0.33
STAR_ARM_EVERY = 2.0
STAR_BRAKE_TOL = 0.03
STAR_CATCH_TOL = 0.03
SLING_C_MIN, SLING_C_MAX = 2.55, 2.80
SLING_C_DEFAULT = 2.75
SLING_RECOVER_TIMEOUT = 500.0
MIN_MASS = 0.3
Q_STAR = 0.2
DEMO_OVER = 1.2
DEMO_FRAC = 0.32
EFF_MIN = 0.6
EFF_NUKE = 0.85
EFF_EVERY = 0.2
COAST_DV = 0.004
MIN_BURN = 0.3
NUKE_DV = 0.12
NUKE_V = 0.8
FUSE = 0.04
FUSE_NEAR = 0.16
NUKE_LIFE = 3.0
NUKE_RELOAD = 4.0
NUKE_SHARE = 0.6
J_NUKE_MAX = 1.0e-4
HEAT_FRAC = 0.05
GRAIN_MASS = 2.0e-9
EJ_COS = 0.5
SHIP_REBUILD = 10.0
ROCHE_R = 0.35
TIDAL_SWEEP = 0.5
DANGER_Q = ROCHE_R + 0.15
LS_SUPER = 1.8
EJ_FRAC_V = 0.3
REMNANT_MIN = 0.01 * 2.0e-4
FLASH_SIM = 0.4
EW = 1024
SHIPS = ("ship0", "ship1")
RACE_GATES = [(3.0, 0.0, 0.0), (2.0, 70.0, 0.6), (0.75, 150.0, 0.0), (1.9, 230.0, -0.6), (3.0, 300.0, 0.0)]
RACE_GATE_R = 0.30
RACE_GRID_R = 0.25
RACE_TAU = 0.35
RACE_LEAD_D = 0.6
RACE_PILOTS = (dict(name="late braker", v=2.2, lead=0.2), dict(name="smooth line", v=2.05, lead=0.9))
PAIR_HYST = 0.05
HERD_START = 0.4
HERD_DONE_R, HERD_DONE_V = 0.12, 1.0e-3
HERD_TIMEOUT = 300.0
HERD_COOL = 60.0
HERD_ROSETTE_LOSE = 0.65
HERD_K, HERD_VMAX = 0.012, 0.004
HERD_DV_MIN = 1.0e-3
HERD_SHARE = 0.8
HERD_STATION_R = 0.6
HERD_FIRE_COS = math.cos(math.radians(30.0))
V_EJ_STAR = 3.0
STAR_EJ_MAX = 40000

def camera():
    if CAM_MODE == "free" and _FREE["ro"] is not None:
        fw, rt, up = free_basis()
        return np.array(_FREE["ro"]), fw, rt, up
    if CAM_MODE == "ride" and _RIDE["ro"] is not None:
        ro = np.array(_RIDE["ro"])
        fw = np.array(_RIDE["at"]) - ro; fw /= max(float(np.linalg.norm(fw)), 1e-9)
        rt = _cross(fw, [0.0, 0.0, 1.0])
        if float(np.linalg.norm(rt)) < 1e-6:
            rt = _cross(fw, [0.0, 1.0, 0.0])
        rt /= np.linalg.norm(rt)
        up = _cross(rt, fw)
        return ro, fw, rt, up
    if CAM_MODE == "portrait":
        el, az, dist = PORTRAIT_EL, PORTRAIT_AZ, CAM_D * PORTRAIT_ZOOM
    else:
        el, az, dist = CAM_EL, CAM_AZ, CAM_D
    ce, se = math.cos(el), math.sin(el)
    ca, sa = math.cos(az), math.sin(az)
    ro = np.array([dist * ce * sa, -dist * ce * ca, dist * se])
    fw = -ro / np.linalg.norm(ro)
    rt = _cross(fw, [0.0, 0.0, 1.0]); rt /= np.linalg.norm(rt)
    up = _cross(rt, fw)
    if CAM_MODE == "portrait":
        cr, sr = math.cos(PORTRAIT_ROLL), math.sin(PORTRAIT_ROLL)
        rt, up = cr * rt + sr * up, -sr * rt + cr * up
    return ro, fw, rt, up

def new_world():
    """A fresh system: the star at rest at the centre, three moons on circular two-body orbits."""
    global STAR_LUM
    STAR_LUM = 1.0
    w = dict(sim=None, ids=[], heat={}, n_wall=0, n_merge=0, log=[], E_heat=0.0, E_self=0.0, E_ext=0.0, E_grain=0.0,
             home={m[0]: (PAIR_A if m[0] in PAIR else m[1]) for m in MOONS}, mass0={m[0]: m[2] for m in MOONS}, contacts={}, nukes={}, n_nuke=0,
             ships={s: dict(reload=0.0, laser=False, a=np.zeros(3), beam=None, dead_until=None) for s in SHIPS},
             team=dict(target=None, dir=None, t0=0.0, J=0.0, dm=0.0), emit=[], carry={}, flash=[],
             mode=MODE_NORMAL, mode_c_lo=PAIR_C_LO, mode_c_hi=PAIR_C_HI, mode_target_c=2.75,
             mode_t0=0.0, mode_seen=False, coast=False, retro_mass=None, farm=None, refill_catch=None,
             star=dict(phase="brake", allow=False, last_check=-1e9, plan=None, t0=0.0),
             star_lum=1.0, star_spot=0.0, star_mode=0, flare=None, flare_next=FLARE_FIRST, n_flare=0,
             rosette=None, ring_mass=0.0, n_refill=0, L_ext=np.zeros(3),
             n_det=0, n_miss=0, n_wreck=0, dm_total=0.0)
    w["mass_total0"] = 1.0 + sum(m[2] for m in MOONS)
    w.update(ring_live=GW * GH, ring_frac=1.0, ring_in=0.0, ring_in_star=0.0)
    if rebound is None:
        return w
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.integrator = "ias15"
    sim.collision = "direct"
    sim.add(m=1.0, r=STAR_R)
    for name, a, gm, rad, inc, ph in MOONS:
        i, p = math.radians(inc), math.radians(ph)
        v = math.sqrt((1.0 + gm) / a)
        x = (a * math.cos(p), a * math.sin(p) * math.cos(i), a * math.sin(p) * math.sin(i))
        vv = (-v * math.sin(p), v * math.cos(p) * math.cos(i), v * math.cos(p) * math.sin(i))
        sim.add(m=gm, r=rad, x=x[0], y=x[1], z=x[2], vx=vv[0], vy=vv[1], vz=vv[2])
        w["heat"][name] = t_eq(a)
    for i, sn in enumerate(SHIPS):
        _add_ship(sim, i)
    sim.move_to_com()
    w["ids"] = ["star"] + [m[0] for m in MOONS] + list(SHIPS)
    w["sim"] = sim
    sim.collision_resolve = _merge_resolver(w)
    w["E0"] = energy(w)
    w["L0"] = ang_mom(w)
    return w

def _add_ship(sim, i):
    ph = math.radians(60.0 + 180.0 * i)
    rp = R_PARK + R_PARK_STEP * i
    v = math.sqrt(sim.particles[0].m / rp)
    st = sim.particles[0]
    sim.add(m=SHIP_M, r=SHIP_R, x=st.x + rp * math.cos(ph), y=st.y + rp * math.sin(ph), z=st.z,
            vx=st.vx - v * math.sin(ph), vy=st.vy + v * math.cos(ph), vz=st.vz)

def kind(nm):
    return ("star" if nm == "star" else "ship" if nm.startswith("ship") else "nuke" if nm.startswith("nuke")
            else "moon")

def t_eq(r):
    return T_STAR_EQ * math.sqrt(STAR_R / max(r, STAR_R)) * (STAR_LUM ** 0.25)

def _merge_resolver(w):
    def resolve(simp, c):
        sim = simp.contents
        i, j = int(c.p1), int(c.p2)
        if i >= len(w["ids"]) or j >= len(w["ids"]):
            return 0
        ki, kj = kind(w["ids"][i]), kind(w["ids"][j])
        small_i, small_j = ki in ("ship", "nuke"), kj in ("ship", "nuke")
        if small_i or small_j:
            if small_i and small_j:
                return 0
            s, o = (i, j) if small_i else (j, i)
            ps = sim.particles[s]
            w["contacts"].setdefault(w["ids"][s], (w["ids"][o], np.array(ps.xyz)))
            return 0
        pi, pj = sim.particles[i], sim.particles[j]
        k, d = (i, j) if pi.m >= pj.m else (j, i)
        pk, pd = sim.particles[k], sim.particles[d]
        mk, md = pk.m, pd.m
        M = mk + md
        mu = mk * md / M
        vrel2 = (pk.vx - pd.vx) ** 2 + (pk.vy - pd.vy) ** 2 + (pk.vz - pd.vz) ** 2
        E = 0.5 * mu * vrel2
        dr = math.sqrt((pk.x - pd.x) ** 2 + (pk.y - pd.y) ** 2 + (pk.z - pd.z) ** 2)
        w["E_self"] += sim.G * mk * md / max(dr, 1e-9)
        nk, nd = w["ids"][k], w["ids"][d]
        xc = np.array([(mk * pk.x + md * pd.x) / M, (mk * pk.y + md * pd.y) / M, (mk * pk.z + md * pd.z) / M])
        vc = np.array([(mk * pk.vx + md * pd.vx) / M, (mk * pk.vy + md * pd.vy) / M, (mk * pk.vz + md * pd.vz) / M])
        R = (pk.r ** 3 + pd.r ** 3) ** (1.0 / 3.0)
        m_lr = M
        Q_R = E / M
        if nk != "star":
            if Q_R < LS_SUPER * Q_STAR:
                m_lr = M * (1.0 - Q_R / (2.0 * Q_STAR))
            else:
                m_lr = M * 0.1 * (Q_R / (LS_SUPER * Q_STAR)) ** -1.5
            if m_lr < REMNANT_MIN:
                m_lr = 0.0
        m_ej = M - m_lr
        n_gr = int(m_ej / GRAIN_MASS)
        if n_gr < 1:
            m_lr, m_ej, n_gr = M, 0.0, 0
        KE_ej = 0.0
        if m_ej > 0.0:
            s_ej = EJ_FRAC_V * math.sqrt(vrel2)
            if m_ej * s_ej * s_ej > 0.5 * E:
                s_ej = math.sqrt(0.5 * E / m_ej)
            KE_ej = m_ej * s_ej * s_ej
            phi = 0.0
            for q in range(sim.N):
                if q in (k, d):
                    continue
                po = sim.particles[q]
                phi -= sim.G * po.m / max(math.sqrt((po.x - xc[0]) ** 2 + (po.y - xc[1]) ** 2 + (po.z - xc[2]) ** 2), 1e-9)
            w["E_grain"] += m_ej * (0.5 * float(vc @ vc) + phi) + KE_ej
            r_b = R * (m_lr / M) ** (1.0 / 3.0) if m_lr > 0.0 else R
            w["emit"].append((xc, vc, np.array([0.0, 0.0, 1.0]), r_b, n_gr, -1.0, s_ej))
            w["dm_total"] += m_ej
            w["ring_mass"] += m_ej
            if m_ej > 0.02 * M:
                w["flash"].append((xc, sim.t, 3.0 * J_NUKE_MAX if m_ej > 0.3 * M else 0.5 * J_NUKE_MAX))
                w["flash"] = w["flash"][-4:]
        w["E_heat"] += E - KE_ej
        w["n_merge"] += 1
        rm = w.setdefault("pending_remove", [])
        if m_lr <= 0.0:
            for nm in (nk, nd):
                w["heat"].pop(nm, None)
            rm.extend([nk, nd])
            w["log"].append(f"COLLISION {nd} + {nk} at {math.sqrt(vrel2):.3f}: Q_R {Q_R:.3f} >= {LS_SUPER} x strength "
                            f"{Q_STAR} -> BOTH SHATTERED, {n_gr} grains (fragments ~{s_ej:.2f})")
            return 3
        pk.x, pk.y, pk.z = float(xc[0]), float(xc[1]), float(xc[2])
        pk.vx, pk.vy, pk.vz = float(vc[0]), float(vc[1]), float(vc[2])
        pk.m = m_lr
        pk.r = R * (m_lr / M) ** (1.0 / 3.0)
        if nk != "star":
            tk, td = w["heat"].get(nk, 0.0), w["heat"].get(nd, 0.0)
            w["heat"][nk] = (mk * tk + md * td) / M + (E - KE_ej) / (HEAT_C * m_lr)
        w["heat"].pop(nd, None)
        if nk == "star":
            w["log"].append(f"SWALLOWED: moon {nd} hit the star at {math.sqrt(vrel2):.3f}")
        elif m_ej > 0.0:
            w["log"].append(f"COLLISION {nd} + {nk} at {math.sqrt(vrel2):.3f}: Q_R {Q_R:.4f} (strength {Q_STAR}) -> "
                            f"{nk} keeps {m_lr / M * 100:.1f} % of the pair, {n_gr} grains (~{s_ej:.2f}), heat "
                            f"+{(E - KE_ej) / (HEAT_C * m_lr):.2f}")
        else:
            w["log"].append(f"MERGE {nd} -> {nk} (v_rel {math.sqrt(vrel2):.3f}, heat +{E / (HEAT_C * M):.2f})")
        rm.append(nd)
        return 1 if d == i else 2
    return resolve

def tidal(w, nm):
    """ROCHE (scope 2026-10-21 evening): a moon whose centre is inside ROCHE_R of the star is torn apart. All its mass
    becomes grains on its surface with its own velocity (no added spread: the star's tide shears them by itself)."""
    sim = w["sim"]
    k = w["ids"].index(nm)
    p = sim.particles[k]
    m = p.m
    x, v = _xv(p)
    phi = 0.0
    for q in range(sim.N):
        if q != k:
            po = sim.particles[q]
            phi -= sim.G * po.m / max(float(np.linalg.norm(np.array(po.xyz) - x)), 1e-9)
    w["E_grain"] += m * (0.5 * float(v @ v) + phi)
    n = int(m / GRAIN_MASS)
    w["emit"].append((x, v, np.array([0.0, 0.0, 1.0]), p.r, n, -1.0, 1e-6))
    sim.remove(k)
    w["ids"].remove(nm)
    w["heat"].pop(nm, None)
    w["dm_total"] += m
    w["ring_mass"] += m
    w["flash"].append((x, sim.t, 0.5 * J_NUKE_MAX))
    w["flash"] = w["flash"][-4:]
    st = sim.particles[0]
    w["log"].append(f"TORN APART: {nm} came within {float(np.linalg.norm(x - np.array(st.xyz))):.3f} of the star "
                    f"(Roche limit {ROCHE_R}) -> {n} grains")

def peri(w, nm):
    """Closest approach to the star on the moon's current two-body orbit (inf if it is leaving on an open orbit)."""
    sim = w["sim"]
    st, p = sim.particles[0], _pp(w, nm)
    r = np.array(p.xyz) - np.array(st.xyz)
    u = np.array(p.vxyz) - np.array(st.vxyz)
    mu = st.m + p.m
    rn = float(np.linalg.norm(r))
    hv = _cross(r, u)
    h2 = float(hv @ hv)
    E = 0.5 * float(u @ u) - mu / rn
    if E >= 0.0 and float(r @ u) > 0.0:
        return math.inf
    e = math.sqrt(max(0.0, 1.0 + 2.0 * E * h2 / (mu * mu)))
    return h2 / (mu * (1.0 + e))

def phys_step(w, h):
    """Advance the world by h sim units: gravity + collisions (REBOUND: merge / chip / shatter), then the Roche limit,
    then the walls (toy rule 1)."""
    sim = w["sim"]
    if sim is None or h <= 0.0:
        return
    ring_kick(w, h)
    sim.integrate(sim.t + h)
    for nd in w.pop("pending_remove", []):
        if nd in w["ids"]:
            w["ids"].remove(nd)
    st = sim.particles[0]
    tide = w.setdefault("tide", {})
    torn = []
    for k, nm in enumerate(w["ids"]):
        if kind(nm) != "moon":
            continue
        p = sim.particles[k]
        r2 = (p.x - st.x) ** 2 + (p.y - st.y) ** 2 + (p.z - st.z) ** 2
        if r2 < ROCHE_R * ROCHE_R:
            tide[nm] = tide.get(nm, 0.0) + math.sqrt(st.m / (r2 * math.sqrt(r2))) * h
            if tide[nm] >= TIDAL_SWEEP:
                torn.append(nm)
        else:
            tide.pop(nm, None)
    for nm in torn:
        tide.pop(nm, None)
        tidal(w, nm)
    for k in range(sim.N):
        p = sim.particles[k]
        lim = BOX_W - p.r
        for ax in ("x", "y", "z"):
            q, vax = getattr(p, ax), "v" + ax
            qv = getattr(p, vax)
            if abs(q) > lim and q * qv > 0.0:
                setattr(p, vax, -WALL_E * qv)
                w["E_ext"] += 0.5 * p.m * (WALL_E ** 2 - 1.0) * qv * qv
                w["n_wall"] += 1
                w["log"].append(f"WALL {w['ids'][k]} ({ax}, speed {abs(qv):.3f})")

def cool(w, dt_real):
    """Heat relaxes toward the star-light equilibrium (COOL_TAU real seconds; declared speed-up)."""
    sim = w["sim"]
    if sim is None:
        return
    st = sim.particles[0]
    a = 1.0 - math.exp(-dt_real / COOL_TAU)
    for k in range(1, sim.N):
        p = sim.particles[k]
        r = math.sqrt((p.x - st.x) ** 2 + (p.y - st.y) ** 2 + (p.z - st.z) ** 2)
        nm = w["ids"][k]
        if kind(nm) != "moon":
            continue
        T = w["heat"].get(nm, t_eq(r))
        w["heat"][nm] = T + (t_eq(r) - T) * a

def energy(w):
    sim = w["sim"]
    return sim.energy() if sim is not None else 0.0

def ang_mom(w):
    sim = w["sim"]
    return np.array(sim.angular_momentum()) if sim is not None else np.zeros(3)

def hand_kick(w, k, dv):
    """YOUR HAND (boop): add dv to body k's velocity. Logged."""
    p = w["sim"].particles[k]
    e0 = 0.5 * p.m * (p.vx ** 2 + p.vy ** 2 + p.vz ** 2)
    p.vx += float(dv[0]); p.vy += float(dv[1]); p.vz += float(dv[2])
    w["E_ext"] += 0.5 * p.m * (p.vx ** 2 + p.vy ** 2 + p.vz ** 2) - e0
    w["log"].append(f"HAND boop {w['ids'][k]} dv {float(np.linalg.norm(dv)):.3f}")

def hand_set_vel(w, k, v):
    """YOUR HAND (grab drag / throw): set body k's velocity. Logged by the caller."""
    p = w["sim"].particles[k]
    e0 = 0.5 * p.m * (p.vx ** 2 + p.vy ** 2 + p.vz ** 2)
    p.vx, p.vy, p.vz = float(v[0]), float(v[1]), float(v[2])
    w["E_ext"] += 0.5 * p.m * (p.vx ** 2 + p.vy ** 2 + p.vz ** 2) - e0

def bodies(w):
    """[(id, xyz, radius, mass, heat, particle index)] for the star (first) and the moons; ships / nukes left out."""
    sim = w["sim"]
    out = []
    if sim is None:
        return out
    for k in range(sim.N):
        nm = w["ids"][k]
        if kind(nm) not in ("star", "moon"):
            continue
        p = sim.particles[k]
        out.append((nm, np.array(p.xyz), p.r, p.m, w["heat"].get(nm, 0.0), k))
    return out

def refill_slot(w):
    """The first missing moon slot, or None when A/B/C are all alive. The user can keep the system empty."""
    for nm in REFILL_ORDER:
        if nm not in w["ids"]:
            return nm
    return None

def _lift_phase(w, target=None):
    """Pick the spawn angle. The transfer crosses A/B/C's lanes, so a quick endpoint gap is not enough: dry-run the
    transfer on copies of the sim and keep the lane with the most clearance from the star and the existing moons."""
    sim = w["sim"]
    st = sim.particles[0]
    M0 = max(float(st.m), 1e-9)
    target = 2.55 if target is None else target
    r0 = max(LIFT_R0, st.r + 0.12)
    a_mid = 0.5 * (r0 + target)
    v0 = math.sqrt(M0 * max(2.0 / r0 - 1.0 / a_mid, 0.0))
    t_end = math.pi * math.sqrt(max(a_mid, 1e-6) ** 3 / M0) * 1.12 + 0.2
    moon_idx = [i for i, nm in enumerate(w["ids"]) if kind(nm) == "moon"]
    xs, vs = np.array(st.xyz), np.array(st.vxyz)
    best, best_clear = 0.0, -9.9
    for i in range(24):
        a = 2.0 * math.pi * i / 24.0
        x = xs + r0 * np.array([math.cos(a), math.sin(a), 0.0])
        v = vs + v0 * np.array([-math.sin(a), math.cos(a), 0.0])
        try:
            sim2 = sim.copy()
            sim2.collision = "none"
            sim2.add(m=LIFT_M, r=LIFT_R, x=float(x[0]), y=float(x[1]), z=float(x[2]),
                     vx=float(v[0]), vy=float(v[1]), vz=float(v[2]))
            k = sim2.N - 1
            clear, ok = 9.9, True
            while sim2.t < t_end and ok:
                sim2.integrate(min(sim2.t + 0.06, t_end))
                tp = sim2.particles[k]
                for j in moon_idx:
                    pm = sim2.particles[j]
                    d = math.sqrt((tp.x - pm.x) ** 2 + (tp.y - pm.y) ** 2 + (tp.z - pm.z) ** 2)
                    gap = d - pm.r - LIFT_R
                    clear = min(clear, gap)
                    if gap < 0.03:
                        ok = False
                        break
                d = math.sqrt((tp.x - xs[0]) ** 2 + (tp.y - xs[1]) ** 2 + (tp.z - xs[2]) ** 2)
                if d < st.r + LIFT_R + 0.05:
                    ok = False
        except Exception:
            clear, ok = -9.9, False
        if clear > best_clear:
            best, best_clear = a, clear
        if ok and clear > 0.12:
            return a
    return best

def lift_star(w, slot=None, pos=None):
    """REFILL SHOT: take LIFT_M from the star and put it on the transfer orbit (periapsis LIFT_R0, apocenter = the
    missing slot's home). Momentum is conserved by the star's tiny recoil; the angular momentum the new orbit carries
    is booked in L_ext (a point star has no spin to absorb it). The normal team then catches it at apocenter.
    Energy is booked by ships_step's existing E_ext bracket around the nuke fuse (the blast is the source)."""
    sim = w["sim"]
    if sim is None:
        return None
    slot = refill_slot(w) if slot is None else slot
    if slot is None or slot in w["ids"]:
        return None
    st = sim.particles[0]
    M0 = float(st.m)
    m = min(LIFT_M, 0.25 * M0)
    if m < 0.5 * LIFT_M:
        w["log"].append("REFILL: the star is too light for another lift")
        return None
    target = w["home"].get(slot, 2.55)
    r0 = max(LIFT_R0, st.r + 0.12)
    a_mid = 0.5 * (r0 + target)
    v0 = math.sqrt(M0 * max(2.0 / r0 - 1.0 / a_mid, 0.0))
    phi = _lift_phase(w, target)
    xs, vs = np.array(st.xyz), np.array(st.vxyz)
    x = xs + r0 * np.array([math.cos(phi), math.sin(phi), 0.0])
    v = vs + v0 * np.array([-math.sin(phi), math.cos(phi), 0.0])
    L0 = ang_mom(w)
    vst = (M0 * vs - m * v) / (M0 - m)
    st.m = M0 - m
    st.vx, st.vy, st.vz = float(vst[0]), float(vst[1]), float(vst[2])
    sim.add(m=m, r=LIFT_R, x=float(x[0]), y=float(x[1]), z=float(x[2]),
            vx=float(v[0]), vy=float(v[1]), vz=float(v[2]))
    w["ids"].append(slot)
    w["heat"][slot] = t_eq(r0)
    w["home"][slot] = target
    w["mass0"][slot] = m
    w["n_refill"] += 1
    w["L_ext"] += ang_mom(w) - L0
    q = np.array(st.xyz) if pos is None else np.array(pos, dtype=float)
    w["flash"].append((q, sim.t, STAR_LIFT_J))
    w["flash"] = w["flash"][-4:]
    w["log"].append(f"REFILL: lifted {m:.2e} from the star -> {slot} at r {r0:.2f}, transfer apocenter "
                    f"{target:.2f}, launch speed {v0:.3f}; waiting for the catch")
    return slot

def _pp(w, nm):
    return w["sim"].particles[w["ids"].index(nm)]

def _xv(p):
    return np.array(p.xyz), np.array(p.vxyz)

def grav(w, x, skip=None):
    """Gravity of the star + moons at x (what every body feels; ships use it to plan)."""
    a = ring_acc(w, x)
    for nm, c, r, m, _, _ in bodies(w):
        if nm == skip:
            continue
        d = c - x
        r2 = max(float(d @ d), 1e-9)
        a += m * d / (r2 * math.sqrt(r2))
    return a

def wanted_vel(w, x, a_home, m_self):
    """The home-orbit velocity field: circular speed at the body's distance, plus a capped pull back to radius a_home
    and to the ring plane. The ships push a moon toward this (and fly their own park orbit with it)."""
    st = w["sim"].particles[0]
    xs, vs = _xv(st)
    rel = x - xs
    rho = max(math.hypot(rel[0], rel[1]), 1e-6)
    er = np.array([rel[0] / rho, rel[1] / rho, 0.0])
    et = np.array([-er[1], er[0], 0.0])
    vc = math.sqrt((st.m + m_self) / rho)
    vr = float(np.clip(-K_HOME * (rho - a_home), -VR_MAX, VR_MAX))
    vz = float(np.clip(-K_HOME * rel[2], -VR_MAX, VR_MAX))
    return vs + et * vc + er * vr + np.array([0.0, 0.0, vz])

def _orbit_a(w, nm):
    sim = w["sim"]
    st, p = sim.particles[0], _pp(w, nm)
    r = np.array(p.xyz) - np.array(st.xyz)
    u = np.array(p.vxyz) - np.array(st.vxyz)
    inv_a = 2.0 / max(float(np.linalg.norm(r)), 1e-9) - float(u @ u) / (st.m + p.m)
    return 1.0 / inv_a if inv_a > 1e-9 else 1e9, r

def _hs_f(th):
    return 1.0 / (2.0 * max(math.sin(0.5 * th), 1e-3)) - math.cos(th)

def pair_state(w):
    """(C, theta, da, mean a, masses) of the horseshoe pair, or None if one of them is gone."""
    if not all(n in w["ids"] for n in PAIR):
        return None
    (aA, rA), (aB, rB) = _orbit_a(w, PAIR[0]), _orbit_a(w, PAIR[1])
    mA, mB = _pp(w, PAIR[0]).m, _pp(w, PAIR[1]).m
    mu = (mA + mB) / w["sim"].particles[0].m
    th = math.acos(float(np.clip(rA @ rB / (np.linalg.norm(rA) * np.linalg.norm(rB)), -1.0, 1.0)))
    am = (mA * aA + mB * aB) / (mA + mB)
    da = aA - aB
    return 3.0 / (8.0 * mu) * (da / am) ** 2 + _hs_f(th), th, da, am, mA, mB, mu

def target_a(w, nm):
    """The orbit size moon nm should have right now (see above); a single moon: its home distance."""
    if nm in PAIR:
        ps = pair_state(w)
        if ps is not None:
            C, th, da, am, mA, mB, mu = ps
            am_t = min(max(am, PAIR_A - LANE_TOL), PAIR_A + LANE_TOL)
            lo = w.get("mode_c_lo", PAIR_C_LO)
            hi = w.get("mode_c_hi", PAIR_C_HI)
            C_t = min(max(C, lo + PAIR_AIM), hi - PAIR_AIM)
            da_t = math.copysign(am_t * math.sqrt(max(0.0, 8.0 * mu / 3.0 * (C_t - _hs_f(th)))), da if da != 0.0 else 1.0)
            return am_t + da_t * mB / (mA + mB) if nm == PAIR[0] else am_t - da_t * mA / (mA + mB)
    return w["home"].get(nm, 1.0)

def _pair_sep(w):
    if not all(n in w["ids"] for n in PAIR):
        return 9.9
    a, b = np.array(_pp(w, PAIR[0]).xyz), np.array(_pp(w, PAIR[1]).xyz)
    return float(np.linalg.norm(a - b))

def _mode_set(w, mode, target=None):
    """Change only the pair's TARGET BAND (and the coast flag); the world, bodies and clocks are untouched."""
    old = w.get("mode", MODE_NORMAL)
    f = w.get("farm")
    if old == MODE_FARM and mode != MODE_FARM and f and f.get("nuke"):
        nm = f["nuke"]
        live = nm in w["ids"] or nm in w["nukes"]
        w["nukes"].pop(nm, None)
        if nm in w["ids"]:
            w["sim"].remove(w["ids"].index(nm))
            w["ids"].remove(nm)
        if live:
            w["log"].append(f"REFILL: charge {nm} scuttled with the abort")
    w["coast"] = False
    if mode == MODE_NORMAL:
        w["mode_c_lo"], w["mode_c_hi"] = PAIR_C_LO, PAIR_C_HI
        w["mode_seen"] = False
    elif mode == MODE_SLING:
        c = SLING_C_DEFAULT if target is None else min(SLING_C_MAX, max(SLING_C_MIN, target))
        w["mode_target_c"] = c
        w["mode_c_lo"], w["mode_c_hi"] = c - 0.10, c + 0.10
        w["mode_seen"] = False
    elif mode == MODE_RECOVER:
        w["mode_c_lo"], w["mode_c_hi"] = PAIR_C_LO, PAIR_C_HI
    elif mode == MODE_STAR:
        w["retro_mass"] = None
        w["star"] = dict(phase="brake", allow=False, last_check=-1e9, plan=None,
                         t0=w["sim"].t if w["sim"] is not None else 0.0)
        w["team"].update(target="C", mode="star", cls="star", dir=None,
                         t0=w["sim"].t if w["sim"] is not None else 0.0, J=0.0, dm=0.0, prevS={})
    elif mode == MODE_FARM:
        slot = target if (target is not None and target not in w["ids"]) else refill_slot(w)
        if slot is None:
            w["log"].append("REFILL: all three moons are alive -- no slot to fill")
            return
        if w["sim"] is None:
            return
        w["farm"] = dict(phase="launch", slot=slot, t0=w["sim"].t, phi=_lift_phase(w, w["home"].get(slot, 2.55)),
                         launcher=None, nuke=None)
        w["team"].update(target=None, mode=None, cls=None, dir=None, J=0.0, dm=0.0, prevS={})
    elif mode == MODE_ROSETTE:
        if w["sim"] is None:
            return
        w["rosette"] = dict(phase="drop", t0=w["sim"].t)
        w["team"].update(target=None, mode=None, cls=None, dir=None, J=0.0, dm=0.0, prevS={})
    if mode != MODE_FARM:
        w["farm"] = None
    if mode != MODE_ROSETTE:
        w["rosette"] = None
    w["mode"] = mode
    w["mode_t0"] = w["sim"].t if w["sim"] is not None else 0.0
    if mode != MODE_STAR and w["team"].get("target") is not None:
        w["team"]["target"] = None
        w["team"]["mode"] = None
    w["log"].append(f"MODE {old} -> {mode}" +
                    (f", target C {w['mode_target_c']:.2f}" if mode == MODE_SLING else "")
                    + (", brake C then catch retrograde" if mode == MODE_STAR else "")
                    + (f", manual star lift for {w['farm']['slot']}" if mode == MODE_FARM else "")
                    + (", A lead / B,C at 120/240 deg" if mode == MODE_ROSETTE else ""))

def _mode_update(w):
    """Remote mode state machine. Only the target band changes; the same integrator and ships do the work."""
    if w["sim"] is None:
        return
    mode = w.get("mode", MODE_NORMAL)
    if mode == MODE_STAR:
        _star_update(w)
        return
    if mode == MODE_FARM:
        _farm_update(w)
        return
    if mode == MODE_ROSETTE:
        _rosette_update(w)
        return
    ps = pair_state(w)
    if ps is None:
        if mode != MODE_NORMAL:
            w["log"].append("MODE aborted: a pair moon is gone")
            _mode_set(w, MODE_NORMAL)
        return
    C, th = ps[0], ps[1]
    sep = _pair_sep(w)
    t = w["sim"].t
    if mode == MODE_SLING:
        if not w.get("coast") and sep < 0.90 and th < math.radians(45.0):
            w["coast"] = True
            w["log"].append(f"MODE slingshot: coasting through close pass (sep {sep:.3f}, "
                            f"{math.degrees(th):.0f} deg)")
        elif w.get("coast") and sep > 1.10 and th > math.radians(60.0):
            w["coast"] = False
            w["log"].append("MODE slingshot: pass done, corrections resume")
    elif mode == MODE_RECOVER:
        errs = [size_err(w, n) for n in PAIR if n in w["ids"]]
        if w["mode_c_lo"] <= C <= w["mode_c_hi"] and errs and max(errs) < 3.0 * SIZE_TOL:
            _mode_set(w, MODE_NORMAL)
        elif t - w["mode_t0"] > SLING_RECOVER_TIMEOUT:
            w["log"].append("MODE recover timed out; back to normal target")
            _mode_set(w, MODE_NORMAL)

def _farm_update(w):
    """REFILL state machine: launch (ship 0 holds the circular star station) -> shot (nuke in flight) -> hand the new
    moon to the normal team. There is deliberately no autostart: this mode exists only after the REFILL button."""
    sim = w["sim"]
    f = w.get("farm")
    if sim is None or f is None:
        _mode_set(w, MODE_NORMAL)
        return
    slot, t = f["slot"], sim.t
    if slot in w["ids"]:
        w["log"].append(f"REFILL: {slot} is back -- the normal team takes the catch (lasers only)")
        _mode_set(w, MODE_NORMAL)
        w["refill_catch"] = slot
        w["team"].update(target=slot, mode="fix", cls="shape", dir=None, t0=t, J=0.0, dm=0.0, prevS={})
        return
    if f.get("phase") == "shot" and f.get("nuke") not in w["ids"]:
        w["log"].append("REFILL: charge missed the star -- no moon lifted (press REFILL to try again)")
        _mode_set(w, MODE_NORMAL)
        return
    if t - f["t0"] > FARM_TIMEOUT:
        w["log"].append("REFILL: timed out -- back to normal")
        _mode_set(w, MODE_NORMAL)

def _rosette_update(w):
    """ROSETTE phase machine: C drops into the lane, then B/C are phased around the lead A."""
    sim = w["sim"]
    f = w.get("rosette")
    if sim is None or f is None:
        _mode_set(w, MODE_NORMAL)
        return
    if not all(n in w["ids"] for n in ("A", "B", "C")):
        w["log"].append("ROSETTE aborted: A/B/C must all be present")
        _mode_set(w, MODE_NORMAL)
        return
    st = sim.particles[0]
    ra = np.array(_pp(w, "A").xyz) - np.array(st.xyz)
    rb = np.array(_pp(w, "B").xyz) - np.array(st.xyz)
    rc = np.array(_pp(w, "C").xyz) - np.array(st.xyz)
    ta = math.atan2(ra[1], ra[0])
    rA = float(np.linalg.norm(ra))
    rB, rC = float(np.linalg.norm(rb)), float(np.linalg.norm(rc))
    def wrap(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi
    eB = wrap(math.atan2(rb[1], rb[0]) - (ta + ROSETTE_ANG))
    eC = wrap(math.atan2(rc[1], rc[0]) - (ta + 2.0 * ROSETTE_ANG))
    if f.get("phase") == "drop":
        if abs(rC - ROSETTE_A) < ROSETTE_R_TOL:
            f["phase"] = "phase"
            w["log"].append(f"ROSETTE: C is in the lane (r {rC:.2f}); phasing B/C around A")
    elif f.get("phase") == "phase":
        if (abs(rB - ROSETTE_A) < ROSETTE_R_TOL and abs(rC - ROSETTE_A) < ROSETTE_R_TOL
                and abs(eB) < ROSETTE_PHASE_TOL and abs(eC) < ROSETTE_PHASE_TOL):
            f["phase"] = "lock"
            w["log"].append(f"ROSETTE LOCK: A lead, B {math.degrees(eB):+.1f} deg, "
                            f"C {math.degrees(eC):+.1f} deg from target")
    if f.get("phase") != "lock" and sim.t - f["t0"] > ROSETTE_TIMEOUT:
        w["log"].append("ROSETTE: timed out; back to normal")
        _mode_set(w, MODE_NORMAL)

def _star_kin(w):
    """C's state relative to the star + the local prograde tangent."""
    p = _pp(w, "C")
    st = w["sim"].particles[0]
    r = np.array(p.xyz) - np.array(st.xyz)
    v = np.array(p.vxyz) - np.array(st.vxyz)
    rn = max(float(np.linalg.norm(r)), 1e-9)
    th = np.array([-r[1], r[0], 0.0])
    th /= max(float(np.linalg.norm(th)), 1e-9)
    return p, st, r, v, rn, math.sqrt(st.m / rn), th, float(_cross(r, v)[2])

def _star_need(w):
    """The velocity change C needs in the current STAR_SLING phase (brake / catch / retrograde hold)."""
    p, st, r, v, rn, vc, th, _ = _star_kin(w)
    phase = w.get("star", {}).get("phase", "brake")
    if phase == "brake":
        target = th * (STAR_F * vc)
    else:
        a_t = target_a(w, "C")
        vt = math.sqrt(max(2.0 / rn - 1.0 / a_t, 0.0))
        target = -th * vt
    err = target - v
    n = float(np.linalg.norm(err))
    return err, (err / n if n > 1e-9 else np.zeros(3)), "now"

def _star_path_plan(w, f=None):
    """Dry-run the brake on a COPY of the sim. None if C would hit the star/tide or A/B; else the outbound state.
    `f` is the fraction of circular speed left after the brake (default STAR_F)."""
    sim = w["sim"]
    if sim is None or "C" not in w["ids"]:
        return None
    f = STAR_F if f is None else f
    try:
        sim2 = sim.copy()
    except Exception:
        return None
    sim2.collision = "none"
    k = w["ids"].index("C")
    p2 = sim2.particles[k]
    st2 = sim2.particles[0]
    r = np.array(p2.xyz) - np.array(st2.xyz)
    rn = max(float(np.linalg.norm(r)), 1e-9)
    th = np.array([-r[1], r[0], 0.0]) / max(math.hypot(r[0], r[1]), 1e-9)
    vc = math.sqrt(st2.m / rn)
    p2.vx, p2.vy, p2.vz = (float(st2.vx + th[0] * f * vc), float(st2.vy + th[1] * f * vc),
                            float(st2.vz + th[2] * f * vc))
    t0 = sim2.t
    minr = 9.9
    for _ in range(int(60.0 / 0.01)):
        sim2.integrate(sim2.t + 0.01)
        p2 = sim2.particles[k]
        st = sim2.particles[0]
        rr = np.array(p2.xyz) - np.array(st.xyz)
        vv = np.array(p2.vxyz) - np.array(st.vxyz)
        rn = float(np.linalg.norm(rr))
        minr = min(minr, rn)
        if minr < STAR_MIN_R:
            return None
        for nm in ("A", "B"):
            if nm not in w["ids"]:
                continue
            j = w["ids"].index(nm)
            q = sim2.particles[j]
            d = float(np.linalg.norm(np.array(q.xyz) - np.array(p2.xyz)))
            if d < q.r + p2.r + 0.03:
                return None
        if rn > 2.50 and float(rr @ vv) > 0.0:
            return dict(pos=np.array(p2.xyz), vel=vv, dt=sim2.t - t0, minr=minr)
    return None

def _star_update(w):
    """STAR_SLING phase machine. Runs after the ships have flown/fired; changes only the target/phase."""
    if w["sim"] is None:
        return
    if "C" not in w["ids"]:
        w["log"].append("STAR SLING aborted: C is gone")
        _mode_set(w, MODE_NORMAL)
        return
    p, st, r, v, rn, vc, th, Lz = _star_kin(w)
    star = w.setdefault("star", dict(phase="brake", allow=False, last_check=-1e9, plan=None, t0=w["sim"].t))
    phase = star.get("phase", "brake")
    t = w["sim"].t
    if phase == "brake":
        if t - star.get("last_check", -1e9) >= STAR_ARM_EVERY:
            star["last_check"] = t
            plan = _star_path_plan(w)
            allow = plan is not None
            if allow != star.get("allow"):
                w["log"].append("STAR SLING: plunge path " + ("clear" if allow else "blocked by A/B or the star"))
            star["allow"] = allow
            if plan is not None:
                star["plan"] = plan
        target = th * (STAR_F * vc)
        if float(np.linalg.norm(target - v)) < STAR_BRAKE_TOL:
            star["phase"] = "plunge"
            star["allow"] = False
            w["team"]["target"] = None
            w["team"]["mode"] = None
            w["log"].append(f"STAR SLING: brake done at r {rn:.2f}, C goes in (path check "
                            f"{'clear' if star.get('plan') else 'not stored'})")
    elif phase == "plunge":
        if rn > 1.0 and float(r @ v) > 0.0:
            star["phase"] = "catch"
            star["allow"] = True
            w["team"].update(target="C", mode="star", cls="star", dir=None, t0=t, J=0.0, dm=0.0, prevS={})
            w["log"].append(f"STAR SLING: C outbound at r {rn:.2f}; CATCH phase")
    elif phase == "catch":
        a_t = target_a(w, "C")
        vt = math.sqrt(max(2.0 / rn - 1.0 / a_t, 0.0))
        target = -th * vt
        if Lz < 0.0 and float(np.linalg.norm(target - v)) < STAR_CATCH_TOL:
            w["retro_mass"] = p.m
            star["phase"] = "hold"
            star["allow"] = True
            w["team"].update(target="C", mode="star", cls="star", dir=None, t0=t, J=0.0, dm=0.0, prevS={})
            w["log"].append(f"STAR SLING: C caught retrograde at r {rn:.2f} (Lz {Lz:+.3f}); HOLDING the backwards orbit")

def size_err(w, nm):
    """|a - a_target| / a_target (the pair's dance depends on its distances to a fraction of a percent)."""
    a, _ = _orbit_a(w, nm)
    return abs(a / target_a(w, nm) - 1.0)

def orbit_err(w, x, v, m_self, a_home):
    """How far a body's ORBIT is from home, as a speed: D = v_home * sqrt((da/2a)^2 + (e/2)^2 + tilt^2) (the classic
    small-change costs: changing a by da costs v da/2a, removing eccentricity e ~ v e/2, a tilt i ~ v i). Built from
    the orbit's energy, eccentricity vector and angular momentum around the star -- quantities that stay put while a
    moon coasts, so the team never chases the natural swing of an eccentric orbit."""
    st = w["sim"].particles[0]
    xs, vs = _xv(st)
    mu = st.m + m_self
    r = x - xs
    u = v - vs
    rn = max(float(np.linalg.norm(r)), 1e-9)
    inv_a = 2.0 / rn - float(u @ u) / mu
    term_a = 0.5 * (1.0 - inv_a * a_home)
    hv = _cross(r, u)
    hn = max(float(np.linalg.norm(hv)), 1e-12)
    ev = _cross(u, hv) / mu - r / rn
    term_e = 0.5 * float(np.linalg.norm(ev))
    retro = w.get("retro_mass")
    sign = -1.0 if retro is not None and abs(m_self - retro) < 1e-12 else 1.0
    term_i = math.sqrt(max(0.0, 2.0 * (1.0 - sign * hv[2] / hn)))
    return math.sqrt(mu / a_home) * math.sqrt(term_a * term_a + term_e * term_e + term_i * term_i)

def _err_grad(w, x, v, m, ah):
    """orbit_err and its gradient over the velocity (central differences)."""
    D = orbit_err(w, x, v, m, ah)
    g = np.zeros(3)
    eps = 1e-5
    for k in range(3):
        dvk = np.zeros(3); dvk[k] = eps
        g[k] = (orbit_err(w, x, v + dvk, m, ah) - orbit_err(w, x, v - dvk, m, ah)) / (2.0 * eps)
    return D, g

def moon_dv(w, nm):
    """The push the moon needs, as a vector: length = orbit_err, direction = the steepest way down (minus the gradient
    of orbit_err over the moon's velocity). A small push within 90 deg of it always brings the orbit closer to home."""
    p = _pp(w, nm)
    x, v = _xv(p)
    D, g = _err_grad(w, x, v, p.m, target_a(w, nm))
    gn = float(np.linalg.norm(g))
    return -g / gn * D if gn > 0.0 else np.zeros(3)

def push_eff(w, nm):
    """How much orbit error one unit of push removes HERE (|gradient|), and the best value anywhere on the moon's
    current orbit (sampled around the two-body ellipse, refreshed every EFF_EVERY). Pushing where it pays is what
    real orbit corrections do (burn at the right point of the orbit); the team holds fire elsewhere."""
    p = _pp(w, nm)
    x, v = _xv(p)
    ah = target_a(w, nm)
    _, g = _err_grad(w, x, v, p.m, ah)
    now = float(np.linalg.norm(g))
    cache = w.setdefault("effmax", {})
    t = w["sim"].t
    if nm not in cache or t - cache[nm][0] > EFF_EVERY:
        st = w["sim"].particles[0]
        xs, vs = _xv(st)
        mu = st.m + p.m
        r, u = x - xs, v - vs
        hv = _cross(r, u)
        hn = float(np.linalg.norm(hv))
        ev = _cross(u, hv) / mu - r / max(float(np.linalg.norm(r)), 1e-9)
        e = float(np.linalg.norm(ev))
        best = now
        if hn > 1e-9:
            hh = hv / hn
            P = ev / e if e > 1e-6 else r / float(np.linalg.norm(r))
            Q = _cross(hh, P)
            pl = hn * hn / mu
            for f in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
                den = 1.0 + e * math.cos(f)
                if den < 0.05:
                    continue
                rr = pl / den
                xo = xs + rr * (math.cos(f) * P + math.sin(f) * Q)
                vo = vs + math.sqrt(mu / pl) * (-math.sin(f) * P + (e + math.cos(f)) * Q)
                best = max(best, float(np.linalg.norm(_err_grad(w, xo, vo, p.m, ah)[1])))
        cache[nm] = (t, best)
    return now, max(cache[nm][1], now)

def best_push(w, nm, dirn, smax):
    """Along direction dirn: the speed change (0..smax) that brings the moon's orbit closest to home (a line search
    over orbit_err; how a ship sizes a nuke for the line it actually fires along)."""
    p = _pp(w, nm)
    x, v = _xv(p)
    ah = target_a(w, nm)
    ss = np.linspace(0.0, smax, 41)
    errs = [orbit_err(w, x, v + dirn * sv, p.m, ah) for sv in ss]
    i = int(np.argmin(errs))
    return float(ss[i]), errs[0], errs[i]

def _rel(w, nm):
    """Position / velocity relative to the star."""
    st = w["sim"].particles[0]
    xs, vs = _xv(st)
    x, v = _xv(_pp(w, nm))
    return x - xs, v - vs

def team_need(w, T):
    """The push moon T needs RIGHT NOW (vector), the direction to get ready for next, and how the burn is gated
    ("eff": only where a push pays; "now": burn at once). Orbit shape (size / roundness / tilt) toward its target;
    then a fine size trim (the pair's dance needs its distances to ~0.2 %)."""
    if w.get("mode") == MODE_STAR and T == "C":
        return _star_need(w)
    fd = w.get("feed")
    if fd and fd.get("phase") == "flip" and T == fd.get("imp"):
        return _feed_need(w, T)
    p = _pp(w, T)
    mu = w["sim"].particles[0].m + p.m
    dv = moon_dv(w, T)
    if float(np.linalg.norm(dv)) > D_SHAPE:
        return dv, dv, "eff"
    r, v = _rel(w, T)
    rn = float(np.linalg.norm(r))
    rh = np.array([r[0], r[1], 0.0]) / max(math.hypot(r[0], r[1]), 1e-9)
    th = np.array([-rh[1], rh[0], 0.0])
    if size_err(w, T) >= SIZE_TOL:
        vsz = th * math.sqrt(mu * max(2.0 / rn - 1.0 / target_a(w, T), 0.0)) - v
        return vsz, vsz, "now"
    return np.zeros(3), th, "now"

def _sky_frac(R, d):
    """Fraction of a blast's sky covered by a sphere of radius R at distance d (1/2 at contact)."""
    if d <= R:
        return 0.5
    return 0.5 * (1.0 - math.sqrt(max(0.0, 1.0 - (R / d) ** 2)))

def light_push(w, nm, J, heat=1.0):
    """LASER (user 2026-10-21: "stop the lasers burning down the moons"): the beam pushes with LIGHT (radiation
    pressure). Momentum J onto the moon, no mass lost, no ejecta, a little heat. Real light pressure is far weaker;
    F_LASER is a declared strength. `heat` scales the thermal book only (STAR_SLING uses a gentler value on C)."""
    p = _pp(w, nm)
    Jm = float(np.linalg.norm(J))
    if Jm <= 0.0:
        return 0.0, 0.0
    dv = J / p.m
    p.vx += float(dv[0]); p.vy += float(dv[1]); p.vz += float(dv[2])
    w["heat"][nm] = w["heat"].get(nm, 0.0) + heat * HEAT_FRAC * 0.5 * Jm * V_EJ / (HEAT_C * p.m)
    return Jm, 0.0

def ablate(w, nm, J):
    """THE PUSH RULE (laser and nuke alike): impulse J (vector) on body nm, paid with mass |J| / V_EJ blasted off the
    side facing the source. Moon momentum + J; radius shrinks with mass; HEAT_FRAC of the ejecta energy heats it;
    the lost mass is queued as ejecta grains."""
    p = _pp(w, nm)
    Jm = float(np.linalg.norm(J))
    if Jm <= 0.0:
        return 0.0, 0.0
    dm = Jm / V_EJ
    if dm > 0.9 * p.m:
        dm = 0.9 * p.m
        J = J * (dm * V_EJ / Jm)
        Jm = dm * V_EJ
    m0 = p.m
    m1 = m0 - dm
    dv = J / m1
    p.vx += float(dv[0]); p.vy += float(dv[1]); p.vz += float(dv[2])
    p.m = m1
    p.r = p.r * (m1 / m0) ** (1.0 / 3.0)
    w["heat"][nm] = w["heat"].get(nm, 0.0) + HEAT_FRAC * 0.5 * Jm * V_EJ / (HEAT_C * m1)
    n = dm / GRAIN_MASS + w["carry"].get(nm, 0.0)
    ni = int(n)
    w["carry"][nm] = n - ni
    if ni > 0:
        x, v = _xv(p)
        w["emit"].append((x, v, -J / Jm, p.r, ni, EJ_COS, V_EJ / (0.5 * (1.0 + EJ_COS))))
    w["dm_total"] += dm
    w["ring_mass"] += dm
    return Jm, dm

def shatter(w, nm, J, e_spec):
    """SHATTER RULE: the blast gave body nm more than Q_STAR per unit mass -> it breaks apart. It leaves REBOUND; ALL its
    mass becomes grains around its post-blast velocity, thrown in all directions with the surplus energy."""
    p = _pp(w, nm)
    m = p.m
    x, v = _xv(p)
    v1 = v + J / m
    n = int(m / GRAIN_MASS)
    vf = math.sqrt(2.0 * max(e_spec - Q_STAR, 1e-6))
    Jn = float(np.linalg.norm(J))
    w["emit"].append((x, v1, J / max(Jn, 1e-30), p.r, n, -1.0, vf))
    w["sim"].remove(w["ids"].index(nm))
    w["ids"].remove(nm)
    w["heat"].pop(nm, None)
    w["dm_total"] += m
    w["ring_mass"] += m
    w["n_shatter"] = w.get("n_shatter", 0) + 1
    w["flash"].append((x, w["sim"].t, 3.0 * J_NUKE_MAX))
    w["flash"] = w["flash"][-4:]
    w["log"].append(f"SHATTERED: {nm} ({m / w['mass0'].get(nm, m) * 100:.0f} % of its start mass) took {e_spec:.2f} per "
                    f"unit mass > strength {Q_STAR} -> {n} grains, fragments fly at ~{vf:.2f}")

def detonate(w, nm, pos, why):
    """A nuke goes off at pos: every moon is pushed by yield x its sky fraction, away from the blast (ablation)."""
    nk = w["nukes"].pop(nm, None)
    sim = w["sim"]
    if nm in w["ids"]:
        sim.remove(w["ids"].index(nm))
        w["ids"].remove(nm)
    if nk is None:
        return
    hits = []
    sc = np.array(sim.particles[0].xyz)
    p0 = np.array(pos, dtype=float)
    rr = p0 - sc
    rn = float(np.linalg.norm(rr))
    if rn < sim.particles[0].r + 0.005:
        p0 = sc + rr / max(rn, 1e-9) * (sim.particles[0].r + 0.005)
    for bn, c, r, m, _, _ in bodies(w)[1:]:
        d = c - pos
        dist = float(np.linalg.norm(d))
        if _star_blocks(w, p0, c):
            continue
        J = nk["Jfull"] * _sky_frac(r, dist)
        if J > 1e-12:
            e_spec = 0.5 * J * V_EJ / m
            if e_spec > Q_STAR:
                shatter(w, bn, d / max(dist, 1e-9) * J, e_spec)
                hits.append(f"{bn} SHATTERED")
                continue
            Jm, dm = ablate(w, bn, d / max(dist, 1e-9) * J)
            hits.append(f"{bn} dv {Jm / _pp(w, bn).m:.3f} (mass -{dm / (_pp(w, bn).m + dm) * 100:.1f} %, "
                        f"{dist - r:.3f} off its surface)")
            if bn == w["team"]["target"]:
                w["team"]["J"] += Jm; w["team"]["dm"] += dm
    if nk.get("star_push"):
        st = sim.particles[0]
        dd = np.array(st.xyz) - np.array(pos)
        dist = float(np.linalg.norm(dd))
        if dist <= st.r + FUSE + 0.12:
            Jm, dm = star_push(w, dd / max(dist, 1e-9) * nk["Jfull"] * _sky_frac(st.r, dist))
            hd = w.get("herd")
            if hd:
                hd["J"] += Jm; hd["dm"] += dm
            hits.append(f"STAR PUSH dv {Jm / st.m:.4f} (star mass -{dm / (st.m + dm) * 100:.3f} %)")
        else:
            hits.append("STAR PUSH MISS (blast not at the star)")
    if nk.get("star_lift"):
        st = sim.particles[0]
        dist = float(np.linalg.norm(np.array(pos) - np.array(st.xyz)))
        if dist <= st.r + FUSE + 0.12:
            slot = lift_star(w, nk.get("slot"), pos)
            hits.append(f"STAR LIFT -> {slot}" if slot else "STAR LIFT failed (no free slot)")
        else:
            hits.append("STAR LIFT MISS (blast not at the star)")
    w["flash"].append((np.array(pos, dtype=float), sim.t, nk["Jfull"]))
    w["flash"] = w["flash"][-4:]
    w["n_det"] += 1
    if not hits or why.startswith("timer"):
        w["n_miss"] += 1
    w["log"].append(f"NUKE {nm} {why}: " + ("; ".join(hits) if hits else "nothing near -- MISS"))

def wreck(w, sn, other):
    """TOY RULE 2: a ship that touches a moon / the star is gone; a new one starts on the park orbit later."""
    sim = w["sim"]
    if sn not in w["ids"]:
        return
    p = _pp(w, sn)
    w["flash"].append((np.array(p.xyz), sim.t, 2e-7))
    w["flash"] = w["flash"][-4:]
    sim.remove(w["ids"].index(sn))
    w["ids"].remove(sn)
    s = w["ships"][sn]
    s.update(laser=False, beam=None, a=np.zeros(3), dead_until=sim.t + SHIP_REBUILD)
    w["n_wreck"] += 1
    w["log"].append(f"WRECK {sn} hit {other} (toy rule 2: a new one in {SHIP_REBUILD:.0f} sim units)")

def _spring_accel(w, x, v, S, vS, aS, T):
    """The hand-written flight: fly to the station S (riding with the moon), round the star / the moon if in the way."""
    P, via = _route(w, x, S)
    if not via:
        S, _ = _route_round(x, S, np.array(_pp(w, T).xyz), _pp(w, T).r + AVOID + 0.06)
    if via:
        a = (_clip(K_POS * (P - x), V_CLOSE) - v) / TAU_V
    else:
        a = (vS + _clip(K_POS * (S - x), V_CLOSE) - v) / TAU_V + (aS - grav(w, x))
    return _clip(a, A_SHIP)

def _clip(v, lim):
    n = float(np.linalg.norm(v))
    return v * (lim / n) if n > lim else v

def _avoid(w, x, v=None):
    """Steer clear of every body: a radial push near its surface, and (looking ahead 0.6 sim units along the ship's
    motion relative to the body) a full-thrust sidestep when the straight path would hit it."""
    a = np.zeros(3)
    sim = w["sim"]
    for nm, c, r, m, _, k in bodies(w):
        d = x - c
        dist = float(np.linalg.norm(d))
        mg = AVOID_STAR if nm == "star" else AVOID
        if dist < r + mg:
            a += 2.0 * A_SHIP * (1.0 - max(dist - r, 0.0) / mg) * d / max(dist, 1e-9)
        if v is None or nm == "star":
            continue
        u = v - np.array(sim.particles[k].vxyz)
        uu = float(u @ u)
        if uu < 1e-12:
            continue
        tca = -float(d @ u) / uu
        if 0.0 < tca < 0.6:
            miss = d + u * tca
            mn = float(np.linalg.norm(miss))
            if mn < r + mg:
                side = miss / mn if mn > 1e-6 else _cross(u, [0.0, 0.0, 1.0])
                side /= max(float(np.linalg.norm(side)), 1e-9)
                a += A_SHIP * (side - 0.5 * u / math.sqrt(uu))
    return a

def _star_blocks(w, a, b):
    st = w["sim"].particles[0]
    c = np.array(st.xyz)
    d = b - a
    L = float(np.linalg.norm(d))
    d /= max(L, 1e-9)
    t = float(np.clip((c - a) @ d, 0.0, L))
    q = a + d * t - c
    return float(q @ q) < st.r * st.r

def _route(w, x, S):
    """Where to fly first: the station itself, or (when the straight line passes the star too closely) a waypoint
    beside the star. Returns (point, is_waypoint)."""
    st = w["sim"].particles[0]
    c = np.array(st.xyz)
    d = S - x
    L = float(np.linalg.norm(d))
    if L < 0.5:
        return S, False
    dh = d / L
    t = float(np.clip((c - x) @ dh, 0.0, L))
    q = x + dh * t - c
    Rs = st.r + STAR_ROUTE
    if float(np.linalg.norm(q)) >= Rs or t <= 0.0 or t >= L:
        return S, False
    out = q if float(np.linalg.norm(q)) > 1e-6 else _cross(dh, [0.0, 0.0, 1.0])
    out /= max(float(np.linalg.norm(out)), 1e-9)
    return c + out * Rs * 1.25, True

def _route_round(x, S, c, Rs):
    """Going to the other side of sphere (c, Rs): fly via the point halfway round it (repeated every frame, this
    walks the ship round the moon at a safe height instead of into it)."""
    u1, u2 = x - c, S - c
    n1, n2 = float(np.linalg.norm(u1)), float(np.linalg.norm(u2))
    if n1 < 1e-9 or n2 < 1e-9:
        return S, False
    u1, u2 = u1 / n1, u2 / n2
    if float(u1 @ u2) > 0.5:
        return S, False
    mid = u1 + u2
    if float(np.linalg.norm(mid)) < 0.1:
        mid = _cross(u1, [0.0, 0.0, 1.0])
        if float(np.linalg.norm(mid)) < 0.1:
            mid = _cross(u1, [1.0, 0.0, 0.0])
    mid /= float(np.linalg.norm(mid))
    return c + mid * max(Rs * 1.3, n2), True

def _fall_guard(w, x, v):
    """If the ship's own two-body path around the star dips below the safe height, turn sideways + outward at full
    thrust (the engines are weaker than the star's pull close in, so this has to happen early)."""
    st = w["sim"].particles[0]
    xs, vs = _xv(st)
    r, u = x - xs, v - vs
    rn = float(np.linalg.norm(r))
    if float(r @ u) >= 0.0:
        return None
    mu = st.m
    hv = _cross(r, u)
    h2 = float(hv @ hv)
    E = 0.5 * float(u @ u) - mu / rn
    e = math.sqrt(max(0.0, 1.0 + 2.0 * E * h2 / (mu * mu)))
    q = h2 / (mu * (1.0 + e))
    if q >= st.r + STAR_SAFE:
        return None
    rh = r / rn
    tang = u - rh * float(u @ rh)
    tn = float(np.linalg.norm(tang))
    th = tang / tn if tn > 1e-9 else _cross(rh, [0.0, 0.0, 1.0]) / max(float(np.linalg.norm(_cross(rh, [0.0, 0.0, 1.0]))), 1e-9)
    a = th + 0.5 * rh
    return a / float(np.linalg.norm(a)) * A_SHIP

def _farm_step(w, h):
    """The manual REFILL crew. Ship 0 holds a circular station at FARM_STATION_R and fires one heavy charge straight
    at the star; the other ship waits on its park orbit (the normal team catches the lifted moon after the handoff)."""
    sim = w["sim"]
    f = w.get("farm")
    if sim is None or f is None:
        _mode_set(w, MODE_NORMAL)
        return
    st = sim.particles[0]
    if f.get("launcher") not in w["ids"]:
        alive = [sn for sn in SHIPS if sn in w["ids"]]
        if not alive:
            return
        f["launcher"] = alive[0]
    launcher = f["launcher"]
    phase = f.get("phase", "launch")
    for sn in SHIPS:
        s = w["ships"][sn]
        if sn not in w["ids"]:
            continue
        p = _pp(w, sn)
        x, v = _xv(p)
        if sn == launcher and phase == "launch":
            phi = f.get("phi", 0.0)
            rs = FARM_STATION_R
            om = math.sqrt(max(st.m, 1e-9) / rs ** 3)
            ang = phi + om * (sim.t - f["t0"])
            S = np.array(st.xyz) + rs * np.array([math.cos(ang), math.sin(ang), 0.0])
            vc = math.sqrt(max(st.m, 1e-9) / rs)
            vS = np.array(st.vxyz) + vc * np.array([-math.sin(ang), math.cos(ang), 0.0])
            aS = grav(w, S)
            P, via = _route(w, x, S)
            if via:
                a = (_clip(K_POS * (P - x), V_CLOSE) - v) / TAU_V
            else:
                a = (vS + _clip(K_POS * (S - x), V_CLOSE) - v) / TAU_V + (aS - grav(w, x))
            a = _clip(a + _avoid(w, x, v), A_SHIP)
            fg = _fall_guard(w, x, v)
            if fg is not None:
                a = fg
            p.vx += float(a[0]) * h; p.vy += float(a[1]) * h; p.vz += float(a[2]) * h
            s["a"] = a
            s["laser"], s["beam"] = False, None
            off = float(np.linalg.norm(x - S))
            if off < 2.0 * FIRE_TOL and not any(nk.get("star_lift") for nk in w["nukes"].values()):
                aim = np.array(st.xyz) - x
                an = float(np.linalg.norm(aim))
                if an > 1e-9:
                    aim /= an
                    w["n_nuke"] += 1
                    nm = f"nuke{w['n_nuke']}"
                    vn = np.array(st.vxyz) + aim * NUKE_V
                    xn = x + aim * 2.0 * (SHIP_R + NUKE_R)
                    sim.add(m=NUKE_M, r=NUKE_R, x=float(xn[0]), y=float(xn[1]), z=float(xn[2]),
                            vx=float(vn[0]), vy=float(vn[1]), vz=float(vn[2]))
                    w["ids"].append(nm)
                    w["nukes"][nm] = dict(target="star", Jfull=STAR_LIFT_J, t0=sim.t, prev=None,
                                          fuse=FUSE, near=0.03, star_lift=True, slot=f["slot"])
                    f["phase"], f["nuke"] = "shot", nm
                    s["reload"] = sim.t + NUKE_RELOAD
                    w["log"].append(f"REFILL: heavy charge {nm} launched by {sn} at the star "
                                    f"({off:.2f} off station, yield x{STAR_LIFT_J / J_NUKE_MAX:.0f})")
            continue
        dvp = wanted_vel(w, x, R_PARK + R_PARK_STEP * SHIPS.index(sn), p.m) - v
        a = _clip(dvp / TAU_V, A_SHIP) if float(np.linalg.norm(dvp)) > PARK_DEAD else np.zeros(3)
        a = _clip(a + _avoid(w, x, v), A_SHIP)
        fg = _fall_guard(w, x, v)
        if fg is not None:
            a = fg
        p.vx += float(a[0]) * h; p.vy += float(a[1]) * h; p.vz += float(a[2]) * h
        s["a"] = a
        s["laser"], s["beam"] = False, None

def _rosette_step(w, h):
    """Two ships, one follower each: ship0 holds B at +120 deg from A, ship1 holds C at +240 deg.
    A damped position/velocity spring pulls each follower to its rotating target point; the ship's beam supplies
    that correction (light push: no mass loss). This is a guided formation, not a blind orbital burn."""
    sim = w["sim"]
    st = sim.particles[0]
    sx = np.array(st.xyz)
    sv = np.array(st.vxyz)
    A = _pp(w, "A")
    ra = np.array(A.xyz) - sx
    va = np.array(A.vxyz) - sv
    rA = max(float(np.linalg.norm(ra)), 1e-9)
    omA = float(ra[0] * va[1] - ra[1] * va[0]) / (rA * rA)
    ta = math.atan2(ra[1], ra[0])
    for i, sn in enumerate(SHIPS):
        s = w["ships"][sn]
        if sn not in w["ids"]:
            continue
        T = ("B", "C")[i]
        if T not in w["ids"]:
            s["laser"], s["beam"] = False, None
            continue
        sp = _pp(w, sn)
        x, v = _xv(sp)
        q = _pp(w, T)
        xT, vT = _xv(q)
        ang = ta + ROSETTE_ANG * (1.0 if i == 0 else 2.0)
        xdes = sx + ROSETTE_A * np.array([math.cos(ang), math.sin(ang), 0.0])
        rdes = xdes - sx
        vdes = sv + omA * np.array([-rdes[1], rdes[0], 0.0])
        acc = -ROSETTE_KP * (xT - xdes) - ROSETTE_KD * (vT - vdes)
        dv = _clip(acc * h, ROSETTE_DV_LIMIT)
        needn = float(np.linalg.norm(dv))
        d = dv / needn if needn > 1e-9 else np.array([-math.sin(ang), math.cos(ang), 0.0])
        S = xT - d * D_STAND
        vS = vT
        aS = grav(w, S)
        P, via = _route(w, x, S)
        if via:
            a = (_clip(K_POS * (P - x), V_CLOSE) - v) / TAU_V
        else:
            a = (vS + _clip(K_POS * (S - x), V_CLOSE) - v) / TAU_V + (aS - grav(w, x))
        a = _clip(a + _avoid(w, x, v), A_SHIP)
        fg = _fall_guard(w, x, v)
        if fg is not None:
            a = fg
        sp.vx += float(a[0]) * h; sp.vy += float(a[1]) * h; sp.vz += float(a[2]) * h
        s["a"] = a
        if needn > (0.0006 if s["laser"] else 0.0040):
            light_push(w, T, dv * q.m)
            f = xT - x
            fn = max(float(np.linalg.norm(f)), 1e-9)
            s["beam"] = xT - (f / fn) * q.r
            s["pw"] = 1.0
            fire = True
        else:
            fire = False
        s["laser"] = fire
        if not fire:
            s["beam"] = None

def star_push(w, J):
    """A blast on the STAR (herding): impulse J onto the star, paid with |J| / V_EJ_STAR of its mass, blown off the
    blasted side as ejecta grains (drawn count capped). Same rule as ablate(); the star's radius is left alone."""
    st = w["sim"].particles[0]
    Jm = float(np.linalg.norm(J))
    if Jm <= 0.0:
        return 0.0, 0.0
    dm = Jm / V_EJ_STAR
    m1 = st.m - dm
    dv = J / m1
    st.vx += float(dv[0]); st.vy += float(dv[1]); st.vz += float(dv[2])
    st.m = m1
    n = int(dm / GRAIN_MASS)
    if n > 0:
        x, v = _xv(st)
        w["emit"].append((x, v, -J / Jm, st.r, n, EJ_COS, V_EJ_STAR / (0.5 * (1.0 + EJ_COS))))
    w["dm_total"] += dm
    w["ring_mass"] += dm
    return Jm, dm

def _herd_update(w):
    """Is the star to be herded home this frame? Sets w["herd"]["dv"] (the star velocity change wanted)."""
    sim = w["sim"]
    st = sim.particles[0]
    x, v = np.array(st.xyz), np.array(st.vxyz)
    d, vn = float(np.linalg.norm(x)), float(np.linalg.norm(v))
    if w.get("mode") == MODE_FARM:
        return False
    hd = w.get("herd")
    if hd is None:
        if d <= HERD_START or w.get("herd_cool", -1e9) > sim.t:
            return False
        hd = w["herd"] = dict(t0=sim.t, shots=0, J=0.0, dm=0.0, m0=st.m, dv=np.zeros(3))
        w["log"].append(f"STAR HERD: the star is {d:.2f} from home, drifting {vn:.4f} -- both ships to the star")
    else:
        if d < HERD_DONE_R and vn < HERD_DONE_V:
            w["log"].append(f"STAR HERD done after {sim.t - hd['t0']:.1f} sim units: star {d:.3f} from home, drifting "
                            f"{vn:.5f}; {hd['shots']} charges, star lost {(hd['m0'] - st.m) / hd['m0'] * 100:.3f} %")
            w["herd"] = None
            return False
        if sim.t - hd["t0"] > HERD_TIMEOUT:
            w["log"].append(f"STAR HERD gives up after {HERD_TIMEOUT:.0f} sim units (star {d:.2f} from home)")
            w["herd"] = None
            w["herd_cool"] = sim.t + HERD_COOL
            return False
    for nm in w["ids"]:
        if kind(nm) == "moon" and peri(w, nm) < DANGER_Q:
            return False
    if w.get("mode") == MODE_ROSETTE and d > HERD_ROSETTE_LOSE:
        _mode_set(w, MODE_NORMAL)
        w["herd_restore"] = sim.t
        w["log"].append(f"ROSETTE broken off: the star got {d:.2f} from home -- the team restores the moons first, "
                        f"then herds the star")
        return False
    rs0 = w.get("herd_restore")
    if rs0 is not None:
        if sim.t - rs0 < 5.0 or w["team"]["target"] is not None:
            return False
        w["herd_restore"] = None
    vw = -x / d * min(HERD_K * d, HERD_VMAX) if d > 1e-9 else np.zeros(3)
    hd["dv"] = vw - v
    return True

def _herd_step(w, h):
    """Both ships ride circular station orbits round the star and drop heavy charges on its far side."""
    sim = w["sim"]
    st = sim.particles[0]
    hd = w["herd"]
    xs, vs = _xv(st)
    dv = hd["dv"]
    dvn = float(np.linalg.norm(dv))
    pdir = dv / dvn if dvn > 1e-12 else None
    rs = HERD_STATION_R
    om = math.sqrt(max(st.m, 1e-9) / rs ** 3)
    vc = math.sqrt(max(st.m, 1e-9) / rs)
    for i, sn in enumerate(SHIPS):
        if sn not in w["ids"]:
            continue
        s = w["ships"][sn]
        p = _pp(w, sn)
        x, v = _xv(p)
        ang = math.pi * i + om * (sim.t - hd["t0"])
        S = xs + rs * np.array([math.cos(ang), math.sin(ang), 0.0])
        vS = vs + vc * np.array([-math.sin(ang), math.cos(ang), 0.0])
        P, via = _route(w, x, S)
        if via:
            a = (_clip(K_POS * (P - x), V_CLOSE) - v) / TAU_V
        else:
            a = (vS + _clip(K_POS * (S - x), V_CLOSE) - v) / TAU_V + (grav(w, S) - grav(w, x))
        a = _clip(a + _avoid(w, x, v), A_SHIP)
        fg = _fall_guard(w, x, v)
        if fg is not None:
            a = fg
        p.vx += float(a[0]) * h; p.vy += float(a[1]) * h; p.vz += float(a[2]) * h
        s["a"] = a
        s["laser"], s["beam"] = False, None
        if float(np.linalg.norm(x - S)) <= 2.0 * FIRE_TOL:
            _herd_fire(w, sn)

def _herd_fire(w, sn, max_r=None):
    """Drop one heavy charge on the star's far side if this ship is round there (and reloaded). max_r: fire from
    wherever the ship is (ROSETTE: the ships keep holding the formation and lob charges down from it)."""
    sim = w["sim"]
    st = sim.particles[0]
    hd = w["herd"]
    s = w["ships"][sn]
    if sn not in w["ids"]:
        return
    xs, vs = _xv(st)
    x, v = _xv(_pp(w, sn))
    dv = hd["dv"]
    dvn = float(np.linalg.norm(dv))
    if dvn < HERD_DV_MIN or s.get("reload", 0.0) > sim.t:
        return
    if any(nk.get("star_push") and nk.get("ship") == sn for nk in w["nukes"].values()):
        return
    if max_r is not None and float(np.linalg.norm(x - xs)) > max_r:
        return
    pdir = dv / dvn
    side = (x - xs) / max(float(np.linalg.norm(x - xs)), 1e-9)
    cosang = float(side @ -pdir)
    if cosang < HERD_FIRE_COS:
        return
    Jw = min(st.m * dvn * HERD_SHARE / cosang, STAR_LIFT_J)
    fp = 0.5
    Pb = xs + side * st.r
    aim = Pb - x
    aim /= max(float(np.linalg.norm(aim)), 1e-9)
    vn = vs + aim * NUKE_V
    xn = x + aim * 2.0 * (SHIP_R + NUKE_R)
    w["n_nuke"] += 1
    nm = f"nuke{w['n_nuke']}"
    sim.add(m=NUKE_M, r=NUKE_R, x=float(xn[0]), y=float(xn[1]), z=float(xn[2]),
            vx=float(vn[0]), vy=float(vn[1]), vz=float(vn[2]))
    w["ids"].append(nm)
    w["nukes"][nm] = dict(target="star", Jfull=Jw / fp, t0=sim.t, prev=None, fuse=0.0, near=0.03,
                          star_push=True, ship=sn)
    s["reload"] = sim.t + NUKE_RELOAD
    hd["shots"] += 1
    w["log"].append(f"STAR HERD: {sn} drops heavy charge {nm} on the far side (push {Jw / st.m:.4f} toward home, "
                    f"{cosang:.2f} on line)")

def _inc(w, nm):
    r, v = _rel(w, nm)
    hv = _cross(r, v)
    return math.degrees(math.acos(max(-1.0, min(1.0, hv[2] / max(float(np.linalg.norm(hv)), 1e-12)))))

def _dive_law(xr, vr, mu, f, vc0, amax):
    """BRAKE steering (vectorised): toward the backwards tangential velocity f x vc0 at the current position."""
    rn = np.linalg.norm(xr, axis=-1, keepdims=True)
    th = np.cross(np.array([0.0, 0.0, 1.0]), xr / rn)
    th /= np.maximum(np.linalg.norm(th, axis=-1, keepdims=True), 1e-9)
    vt = -(f * vc0)[..., None] * th
    d = vt - vr
    dn = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.maximum(dn, 1e-12) * np.minimum(amax, dn / 0.01), dn[..., 0]

def _dive_plan(w, imp, tgt):
    """Dry-run every (f, dv_o) dive with the ships' own steering in a two-body world (star only): the hardest hit
    that stays outside DIVE_QMIN, or None. Vectorised: ~milliseconds."""
    st = w["sim"].particles[0]
    mu = float(st.m)
    xi, vi = _rel(w, imp)
    xt, vt0 = _rel(w, tgt)
    amax = 2.0 * F_LASER / _pp(w, imp).m * DIVE_EFF
    hit = (_pp(w, imp).r + _pp(w, tgt).r) * 0.6
    F, D = np.meshgrid(DIVE_F, np.array(DIVE_DVO), indexing="ij")
    F, D = F.ravel(), D.ravel()
    n = F.size
    x = np.repeat(xi[None], n, 0); v = np.repeat(vi[None], n, 0)
    y = np.repeat(xt[None], n, 0); u = np.repeat(vt0[None], n, 0)
    vc0 = np.full(n, math.sqrt(mu / float(np.linalg.norm(xi))))
    braking = np.ones(n, bool); burnt = np.zeros(n); alive = np.ones(n, bool)
    dmin = np.full(n, 9.0); vrel = np.zeros(n); tmin = np.zeros(n); qmin = np.full(n, 9.0)
    dt = 0.01
    def g(p):
        r = np.linalg.norm(p, axis=-1, keepdims=True)
        return -mu * p / r ** 3
    for k in range(int(DIVE_T / dt)):
        a = g(x)
        ab, dn = _dive_law(x, v, mu, F, vc0, amax)
        braking &= dn > 0.02
        a = a + np.where(braking[:, None], ab, 0.0)
        r = np.linalg.norm(x, axis=-1)
        burn = (~braking) & (r < DIVE_RBURN) & (burnt < D)
        vh = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        a = a + np.where(burn[:, None], vh * amax, 0.0)
        burnt += np.where(burn, amax * dt, 0.0)
        v = v + a * dt; x = x + v * dt
        u = u + g(y) * dt; y = y + u * dt
        qmin = np.minimum(qmin, np.linalg.norm(x, axis=-1))
        dd = np.linalg.norm(x - y, axis=-1)
        better = alive & (dd < dmin)
        dmin = np.where(better, dd, dmin)
        vrel = np.where(better, np.linalg.norm(v - u, axis=-1), vrel)
        tmin = np.where(better, (k + 1) * dt, tmin)
        alive &= ~(dd < hit)
    ok = (dmin < hit) & (qmin > DIVE_QMIN)
    if not ok.any():
        return None
    k = int(np.argmax(np.where(ok, vrel, -1.0)))
    return dict(f=float(F[k]), dvo=float(D[k]), vc0=float(vc0[k]), vrel=float(vrel[k]), t_hit=float(tmin[k]),
                q=float(qmin[k]), amax=float(amax))

def _feed_update(w):
    """RING FEED state machine (scope 2026-10-22). Returns the moon the team must push now, or None."""
    sim = w["sim"]
    fd = w.get("feed")
    if fd is None:
        if (not w.get("farm_on", True) or w.get("mode", MODE_NORMAL) != MODE_NORMAL or w.get("herd")
                or w.get("ring_frac", 1.0) >= RING_LOW or w.get("feed_cool", -1e9) > sim.t):
            return None
        fd = w["feed"] = dict(phase="gather", t0=sim.t)
        w["log"].append(f"RING FEED: the ring is down to {w.get('ring_frac', 1.0) * 100:.0f} % -- the team will crash "
                        f"two moons into it")
    if not w.get("farm_on", True):
        w["log"].append("RING FEED stopped (RING FARM OFF)")
        w["feed"] = None
        return None
    if fd["phase"] == "gather":
        missing = [n for n in FEED_PAIR if n not in w["ids"]]
        if w.get("mode", MODE_NORMAL) != MODE_NORMAL or w.get("refill_catch"):
            return None
        if missing:
            _mode_set(w, MODE_FARM, target=missing[0])
            w["log"].append(f"RING FEED: lifting a new {missing[0]} off the star first (auto refill)")
            return None
        imp = min(FEED_PAIR, key=lambda n: _pp(w, n).m)
        tgt = [n for n in FEED_PAIR if n != imp][0]
        fd.update(phase="flip", sub="tilt", imp=imp, tgt=tgt, t1=sim.t, merges0=w["n_merge"])
        w["log"].append(f"RING FEED: tipping {imp}'s orbit over (lasers, sideways only) until it flies backwards -- "
                        f"then a STAR DIVE into {tgt} if the timing works, else head-on")
    if fd["phase"] != "flip":
        return None
    imp, tgt = fd["imp"], fd["tgt"]
    if imp not in w["ids"] or tgt not in w["ids"]:
        if w["n_merge"] > fd.get("merges0", 0):
            how = ("STAR-DIVE" if fd.get("sub") in ("brake", "coast", "burn", "fly") else
                   "MID-FLIP" if fd.get("sub") == "tilt" else "HEAD-ON")
            w["log"].append(f"RING FEED: {how} CRASH after {sim.t - fd['t1']:.1f} sim units -- the debris goes into "
                            f"the ring")
        else:
            gone = imp if imp not in w["ids"] else tgt
            w["log"].append(f"RING FEED: lost {gone} before the crash (star / tide) -- no debris this time")
        w["feed"] = None
        w["feed_cool"] = sim.t + FEED_COOL
        return None
    if sim.t - fd["t1"] > FEED_TIMEOUT:
        w["log"].append(f"RING FEED gives up: no crash after {FEED_TIMEOUT:.0f} sim units")
        w["feed"] = None
        w["feed_cool"] = sim.t + FEED_COOL
        return None
    sub = fd["sub"]
    r, v = _rel(w, imp)
    rn = float(np.linalg.norm(r))
    if sub in ("brake", "coast", "burn", "fly") and peri(w, imp) < ROCHE_R + 0.02:
        fd["sub"] = "aim"
        w["log"].append(f"RING FEED: the dive went too deep ({peri(w, imp):.2f}) -- pulling {imp} out, head-on instead")
        return imp
    if sub == "tilt" and _inc(w, imp) > 165.0:
        fd.update(sub="wait", t_wait=sim.t, next_plan=sim.t + 2.0)
        w["log"].append(f"RING FEED: {imp} flies backwards -- waiting {DIVE_DR} outside {tgt}'s lane for a dive window")
    elif sub == "wait":
        if sim.t - fd["t_wait"] > DIVE_WAIT_MAX:
            fd["sub"] = "aim"
            w["log"].append("RING FEED: no dive window -- head-on instead")
        elif sim.t >= fd["next_plan"]:
            fd["next_plan"] = sim.t + DIVE_PLAN_EVERY
            pl = _dive_plan(w, imp, tgt)
            if pl is not None:
                fd.update(sub="brake", plan=pl, t_dive=sim.t)
                w["log"].append(f"RING FEED: STAR DIVE -- {imp} brakes to {pl['f']:.2f} of orbit speed, dives to "
                                f"{pl['q']:.2f} from the star, burns {pl['dvo']:.2f} there; predicted hit on {tgt} at "
                                f"{pl['vrel']:.2f} in {pl['t_hit']:.1f} sim units")
    elif sub == "brake":
        pl = fd["plan"]
        _, dn = _dive_law(r[None], v[None], 0.0, np.array([pl["f"]]), np.array([pl["vc0"]]), 1.0)
        if float(dn[0]) <= 0.02 or peri(w, imp) <= pl["q"] + 0.02:
            fd["sub"] = "coast"
    elif sub == "coast":
        pl = fd["plan"]
        if rn < DIVE_RBURN and pl["dvo"] > 0.0:
            fd.update(sub="burn", t_burn=sim.t, burn_T=pl["dvo"] / pl["amax"])
        elif sim.t - fd["t_dive"] > DIVE_FLY_MAX or (rn > float(np.linalg.norm(_rel(w, tgt)[0])) + 0.6
                                                     and float(r @ v) > 0.0 and sim.t - fd["t_dive"] > 1.0):
            fd["sub"] = "fly"; fd["t_fly"] = sim.t
    elif sub == "burn":
        if sim.t - fd["t_burn"] > fd["burn_T"] or (rn > DIVE_RBURN and float(r @ v) > 0.0):
            fd.update(sub="fly", t_fly=sim.t)
            w["log"].append(f"RING FEED: burn at the star done ({imp} at {rn:.2f}, speed {float(np.linalg.norm(v)):.2f})")
    elif sub == "fly":
        if sim.t - fd["t_dive"] > DIVE_FLY_MAX:
            fd["sub"] = "aim"
            w["log"].append(f"RING FEED: the dive missed {tgt} -- head-on instead")
    return imp

def _feed_need(w, T):
    """The push for impactor T in each stage of the feed (see _feed_update)."""
    st = w["sim"].particles[0]
    fd = w.get("feed") or {}
    sub = fd.get("sub", "tilt")
    r, v = _rel(w, T)
    rn = float(np.linalg.norm(r))
    rh = r / max(rn, 1e-9)
    th = _cross([0.0, 0.0, 1.0], rh)
    th /= max(float(np.linalg.norm(th)), 1e-9)
    vh = v / max(float(np.linalg.norm(v)), 1e-9)
    if sub in ("wait", "aim") and fd.get("tgt") in w["ids"]:
        a_t = float(np.linalg.norm(_rel(w, fd["tgt"])[0])) + (DIVE_DR if sub == "wait" else 0.0)
        vw = -th * math.sqrt(max(st.m, 1e-9) * max(2.0 / rn - 1.0 / a_t, 0.0))
        need = vw - v
        return need, need, "now"
    if sub == "brake":
        pl = fd["plan"]
        a, dn = _dive_law(r[None], v[None], 0.0, np.array([pl["f"]]), np.array([pl["vc0"]]), 1.0)
        need = a[0] / max(float(np.linalg.norm(a[0])), 1e-12) * float(dn[0])
        return need, need, "now"
    if sub == "burn":
        return vh, vh, "now"
    if sub in ("coast", "fly"):
        return np.zeros(3), vh, "now"
    vc = math.sqrt(max(st.m, 1e-9) / rn)
    dv = -vc * th - v
    dn = float(np.linalg.norm(dv))
    if dn < 0.005:
        return np.zeros(3), th, "now"
    dperp = dv - float(dv @ vh) * vh
    if float(np.linalg.norm(dperp)) < 0.15 * dn:
        hv = _cross(r, v)
        dperp = hv / max(float(np.linalg.norm(hv)), 1e-12) * dn
    d = dperp / max(float(np.linalg.norm(dperp)), 1e-12) + (vc - float(np.linalg.norm(v))) * vh * 4.0 \
        - float(v @ rh) * rh * 2.0
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    return d, d, "now"

def _race_gate(w, k):
    st = w["sim"].particles[0]
    r, deg, z = RACE_GATES[k % len(RACE_GATES)]
    a = math.radians(deg)
    return np.array([st.x + r * math.cos(a), st.y + r * math.sin(a), st.z + z])

def _race_grid(w, i):
    """Side by side on the start line (G0), 0.15 apart radially."""
    g = _race_gate(w, 0)
    st = np.array(w["sim"].particles[0].xyz)
    rh = (g - st) / max(float(np.linalg.norm(g - st)), 1e-9)
    return g + rh * (0.075 if i == 0 else -0.075)

def _race_off(w, why):
    if w.get("race"):
        w["log"].append(f"RACE called off: {why}")
    w["race"] = None

def _race_update(w):
    """The team is free: line up, then count gates, laps and wins (scope 2026-10-22 IDLE RACE)."""
    sim = w["sim"]
    rc = w.get("race")
    if rc is None:
        rc = w["race"] = dict(phase="grid", ships={sn: dict(gate=1, lap=0, t_lap=0.0) for sn in SHIPS},
                              score={sn: 0 for sn in SHIPS}, laps={})
        w["log"].append("RACE: nothing to do -- the ships line up on the grid")
    if rc["phase"] == "grid":
        if all(float(np.linalg.norm(np.array(_pp(w, sn).xyz) - _race_grid(w, i))) < RACE_GRID_R
               for i, sn in enumerate(SHIPS)):
            rc["phase"] = "go"
            for sn in SHIPS:
                rc["ships"][sn].update(gate=1, lap=0, t_lap=sim.t)
            w["log"].append("RACE: lights out!")
        return
    n = len(RACE_GATES)
    for i, sn in enumerate(SHIPS):
        s = rc["ships"][sn]
        x = np.array(_pp(w, sn).xyz)
        dg = float(np.linalg.norm(x - _race_gate(w, s["gate"])))
        prev = s.get("dprev", 9.0)
        s["dprev"] = dg
        if not (prev < RACE_GATE_R and dg > prev):
            continue
        s["dprev"] = 9.0
        if s["gate"] % n == 0:
            s["lap"] += 1
            lt = (sim.t - s["t_lap"]) / SIM_RATE
            s["t_lap"] = sim.t
            L = rc["laps"].setdefault(s["lap"], {})
            L[sn] = lt
            if len(L) == len(SHIPS):
                win = min(L, key=L.get)
                rc["score"][win] += 1
                other = [k for k in SHIPS if k != win][0]
                w["log"].append(f"RACE lap {s['lap']}: {win} ({RACE_PILOTS[SHIPS.index(win)]['name']}) "
                                f"{L[win]:.1f} s vs {L[other]:.1f} s -> wins by {L[other] - L[win]:.1f} s | score "
                                f"ship0 {rc['score']['ship0']} - {rc['score']['ship1']} ship1")
                rc["laps"].pop(s["lap"])
        s["gate"] += 1

def _race_accel(w, i, x, v):
    """Racing steering: same engine, same gravity (felt, not cancelled), only the target changes."""
    rc = w["race"]
    vs = np.array(w["sim"].particles[0].vxyz)
    if rc["phase"] == "grid":
        P, via = _route(w, x, _race_grid(w, i))
        a = (vs + _clip(K_POS * (P - x), V_CLOSE) - v) / TAU_V - grav(w, x)
        return _clip(a, A_SHIP)
    pl = RACE_PILOTS[i]
    s = rc["ships"][SHIPS[i]]
    g, gn = _race_gate(w, s["gate"]), _race_gate(w, s["gate"] + 1)
    aim, _ = _route(w, x, g)
    d = aim - x
    dg = float(np.linalg.norm(d))
    dh = d / max(dg, 1e-9)
    if dg < RACE_LEAD_D:
        u = (gn - g) / max(float(np.linalg.norm(gn - g)), 1e-9)
        dh = dh + u * pl["lead"] * (1.0 - dg / RACE_LEAD_D)
        dh /= max(float(np.linalg.norm(dh)), 1e-9)
    vw = vs + dh * pl["v"]
    return _clip((vw - v) / RACE_TAU, A_SHIP)

def ships_step(w, h, grabbed=()):
    """The team, once per frame after the integrator: contacts, fuses, the job, flight, lasers, nukes."""
    sim = w["sim"]
    if sim is None or h <= 0.0:
        return
    e0 = energy(w)
    f = w.get("flare")
    if f is not None and not f.get("applied"):
        st = sim.particles[0]
        sx = np.array(st.xyz)
        fd = np.asarray(f["dir"], dtype=float)
        for bn, c, r, m, _, _ in bodies(w)[1:]:
            d = c - sx
            dd = float(np.linalg.norm(d))
            if dd <= 1e-9:
                continue
            cang = float((d / dd) @ fd)
            if cang >= FLARE_COS:
                J = FLARE_J * f.get("amp", 1.0) * (r / 0.085) ** 2
                light_push(w, bn, fd * J)
                w["log"].append(f"STAR FLARE: {bn} caught the CME (cos {cang:.2f}, J {J:.1e})")
        f["applied"] = True
    rc = w.get("refill_catch")
    if rc and rc not in w["ids"]:
        w["refill_catch"] = None
        w["log"].append("REFILL: catch target gone")
    for nm, (other, pos) in list(w["contacts"].items()):
        if nm in w["ids"]:
            if kind(nm) == "nuke":
                detonate(w, nm, pos, f"hit {other}")
            else:
                wreck(w, nm, other)
    w["contacts"].clear()
    for nm in list(w["nukes"]):
        nk = w["nukes"][nm]
        if nm not in w["ids"]:
            w["nukes"].pop(nm)
            continue
        x = np.array(_pp(w, nm).xyz)
        tg = nk["target"]
        if tg in w["ids"]:
            pt = _pp(w, tg)
            surf = float(np.linalg.norm(x - np.array(pt.xyz))) - pt.r
            if surf <= nk.get("fuse", FUSE):
                detonate(w, nm, x, f"fuse at {surf:.3f} above {tg}")
                continue
            if nk["prev"] is not None and surf > nk["prev"] and nk["prev"] < nk.get("near", FUSE_NEAR):
                detonate(w, nm, x, f"closest approach {nk['prev']:.3f} above {tg}")
                continue
            nk["prev"] = surf
        if sim.t - nk["t0"] > NUKE_LIFE:
            detonate(w, nm, x, "timer: MISS")
    for i, sn in enumerate(SHIPS):
        s = w["ships"][sn]
        if sn not in w["ids"] and s["dead_until"] is not None and sim.t >= s["dead_until"]:
            _add_ship(sim, i)
            w["ids"].append(sn)
            s["dead_until"] = None
            w["log"].append(f"NEW SHIP {sn} on the park orbit (toy rule 2)")
    if _herd_update(w):
        if w.get("race"):
            _race_off(w, "the star is drifting off")
        if w.get("mode") == MODE_ROSETTE:
            _rosette_step(w, h)
            for sn in SHIPS:
                _herd_fire(w, sn, max_r=2.0)
        else:
            _herd_step(w, h)
        w["E_ext"] += energy(w) - e0
        return
    if w.get("mode") in (MODE_FARM, MODE_ROSETTE) and w.get("race"):
        _race_off(w, f"mode {w.get('mode')}")
    if w.get("mode") == MODE_FARM:
        _farm_step(w, h)
        w["E_ext"] += energy(w) - e0
        return
    if w.get("mode") == MODE_ROSETTE:
        _rosette_step(w, h)
        w["E_ext"] += energy(w) - e0
        return
    tm = w["team"]
    moons = [b[0] for b in bodies(w)[1:] if b[0] not in grabbed and kind(b[0]) == "moon"
             and b[3] >= MIN_MASS * w["mass0"].get(b[0], b[3])]
    errs = {}
    for nm in moons:
        p = _pp(w, nm)
        errs[nm] = (orbit_err(w, *_xv(p), p.m, target_a(w, nm)), size_err(w, nm))
    small = [b[0] for b in bodies(w)[1:] if b[0] not in grabbed and b[3] < MIN_MASS * w["mass0"].get(b[0], b[3])]
    danger = {}
    fdv = w.get("feed") or {}
    for nm in moons:
        q = peri(w, nm)
        if (nm == fdv.get("imp") and fdv.get("sub") in ("brake", "coast", "burn", "fly")
                and q > ROCHE_R + 0.02):
            continue
        if q < DANGER_Q:
            danger[nm] = q

    ps = pair_state(w)
    coast = bool(w.get("coast", False))
    lo = w.get("mode_c_lo", PAIR_C_LO)
    hi = w.get("mode_c_hi", PAIR_C_HI)
    pair_off = ps is not None and not coast and ps[1] > math.radians(60.0) and not (lo <= ps[0] <= hi)
    pair_far = ps is None or ps[1] > math.radians(60.0)
    pair_off_start = pair_off and not (lo - PAIR_HYST <= ps[0] <= hi + PAIR_HYST)

    def needs_fix(nm):
        if coast and nm in PAIR and nm not in danger:
            return False
        size_bad = errs[nm][1] > 3.0 * SIZE_TOL and (nm not in PAIR or pair_far)
        return (nm in danger or errs[nm][0] > 2.0 * D_SHAPE or size_bad
                or (pair_off_start and nm == PAIR[0]))

    star_mode = w.get("mode") == MODE_STAR
    star_phase = w.get("star", {}).get("phase", "")
    if star_mode:
        want = "C" if star_phase in ("brake", "catch", "hold") and "C" in w["ids"] else None
        tm["target"] = want
        tm["mode"] = "star" if want else None
    feeding = False
    if not star_mode and w.get("mode", MODE_NORMAL) == MODE_NORMAL:
        fi = _feed_update(w)
        if fi is not None and not danger and fi not in grabbed:
            feeding = True
            if tm["target"] != fi or tm.get("mode") != "feed":
                tm.update(target=fi, mode="feed", cls="feed", dir=None, t0=sim.t, J=0.0, dm=0.0, prevS={})
        elif tm.get("mode") == "feed":
            tm.update(target=None, mode=None)
    T = tm["target"]
    demo = tm.get("mode") == "demolish"
    if coast and T in PAIR and not demo:
        w["log"].append(f"MODE coast: team holds fire on {T}")
        T = tm["target"] = None
        tm["mode"] = None
    if T is not None and demo and T not in small:
        if T not in w["ids"]:
            w["log"].append(f"TEAM: {T} is gone -- its lane is free for a new moon")
        else:
            w["log"].append(f"TEAM drops the demolition of {T} ({'in your hand' if T in grabbed else 'no longer small'})")
        T = tm["target"] = None
        tm["mode"] = None
    elif T is not None and not demo and T not in errs:
        why = ("in your hand" if T in grabbed else "gone" if T not in w["ids"] else
               f"down to {MIN_MASS * 100:.0f} % of its mass: it will be demolished")
        w["log"].append(f"TEAM drops {T} ({why})")
        T = tm["target"] = None
    if T is not None and not demo and not star_mode:
        if T in danger:
            tm["cls"] = "danger"
        elif tm.get("cls") == "danger":
            tm["cls"] = "shape"
        if [n for n in danger if n != T] and T not in danger:
            w["log"].append(f"TEAM leaves {T}: {min((n for n in danger if n != T), key=danger.get)} is heading for the star")
            T = tm["target"] = None
    if T is None and small and not star_mode:
        T = small[0]
        tm.update(target=T, mode="demolish", dir=None, t0=sim.t, J=0.0, dm=0.0, prevS={})
        w["log"].append(f"TEAM: {T} is down to {_pp(w, T).m / w['mass0'].get(T, 1.0) * 100:.0f} % -- blasting it "
                        f"apart for the ring (its lane then waits for a new moon)")
    demo = tm.get("mode") == "demolish" and T is not None
    if (T is not None and not demo and not star_mode and not feeding and T not in danger and errs[T][0] < D_SHAPE
            and errs[T][1] < SIZE_TOL and not (pair_off and T == PAIR[0])):
        mm = _pp(w, T).m
        w["log"].append(f"TEAM done: {T}'s orbit is back after {sim.t - tm['t0']:.1f} sim units (pushed "
                        f"{tm['J'] / mm:.3f} of speed, {T} lost {tm['dm'] / (mm + tm['dm']) * 100:.1f} % of its mass)")
        if T == w.get("refill_catch"):
            w["refill_catch"] = None
            w["log"].append("REFILL: catch complete")
        T = tm["target"] = None
    if T is None and errs and not star_mode:
        best, cls = None, None
        if danger:
            best, cls = min(danger, key=danger.get), "danger"
        else:
            sh = [n for n in errs if needs_fix(n)]
            if sh:
                best, cls = max(sh, key=lambda n: errs[n][0] + errs[n][1]), "shape"
        if best is not None:
            T = best
            tm.update(target=T, mode="fix", cls=cls, dir=None, t0=sim.t, J=0.0, dm=0.0, prevS={})
            w["log"].append(f"TEAM takes {T} ({cls}{f': its orbit dips to {danger[T]:.2f} from the star' if cls == 'danger' else ''}): "
                            f"orbit off by {errs[T][0]:.3f} (speed units), size {errs[T][1] * 100:.2f} % off its target")
    if T is None and not star_mode and w.get("mode", MODE_NORMAL) == MODE_NORMAL and all(sn in w["ids"] for sn in SHIPS):
        _race_update(w)
    elif w.get("race"):
        _race_off(w, f"the team is needed for {T}" if T is not None else
                  "a ship is down" if not all(sn in w["ids"] for sn in SHIPS) else f"mode {w.get('mode')}")
    stations = {}
    need = np.zeros(3)
    gate = "now"
    if demo:
        vrel = _rel(w, T)[1]
        nxt = vrel / max(float(np.linalg.norm(vrel)), 1e-9)
    elif T is not None:
        need, nxt, gate = team_need(w, T)
    if T is not None:
        nn = float(np.linalg.norm(nxt))
        if nn > 1e-9:
            dh = nxt / nn
            al = 1.0 - math.exp(-h / (0.3 if gate == "eff" else 0.05))
            tm["dir"] = dh if tm["dir"] is None else tm["dir"] * (1.0 - al) + dh * al
            tm["dir"] /= max(float(np.linalg.norm(tm["dir"])), 1e-12)
    if T is not None and tm["dir"] is not None:
        d = tm["dir"]
        u = _cross(d, [0.0, 0.0, 1.0])
        if float(np.linalg.norm(u)) < 0.2:
            u = _cross(d, [1.0, 0.0, 0.0])
        u /= float(np.linalg.norm(u))
        up = _cross(u, d)
        pT = _pp(w, T)
        xT, vT = _xv(pT)
        aT = grav(w, xT, skip=T)
        prevS = tm.setdefault("prevS", {})
        for i, sn in enumerate(SHIPS):
            di = d * math.cos(TEAM_ANG) + up * math.sin(TEAM_ANG) * (1.0 if i == 0 else -1.0)
            rel = -D_STAND * di
            vrel = (rel - prevS[sn]) / h if sn in prevS and h > 0.0 else np.zeros(3)
            prevS[sn] = rel
            stations[sn] = (xT + rel, vT + _clip(vrel, V_CLOSE), aT)
    needn = float(np.linalg.norm(need))
    dnow = need / max(needn, 1e-12)
    effok = nukeok = True
    if T is not None and gate == "eff" and needn > 0.0 and tm.get("cls") != "danger":
        en, emx = push_eff(w, T)
        effok, nukeok = en >= EFF_MIN * emx, en >= EFF_NUKE * emx
    nuke_out = any(nk["target"] == T for nk in w["nukes"].values()) if T is not None else True
    for sn in SHIPS:
        s = w["ships"][sn]
        if sn not in w["ids"]:
            continue
        p = _pp(w, sn)
        x, v = _xv(p)
        pact, pR = None, None
        if sn in stations:
            S, vS, aS = stations[sn]
            a = _spring_accel(w, x, v, S, vS, aS, T)
            if _PILOT is not None:
                obs, pR = pilot_obs(w, sn, x, v, S, vS, aS, T, need, gate, a)
                pact = pilot_act(obs)
                a = _clip(a + pact["a"], A_SHIP)
            s["thrust_sum"] = s.get("thrust_sum", 0.0) + float(np.linalg.norm(a)) * h
        elif w.get("race"):
            a = _race_accel(w, SHIPS.index(sn), x, v)
        else:
            dvp = wanted_vel(w, x, R_PARK + R_PARK_STEP * SHIPS.index(sn), p.m) - v
            a = _clip(dvp / TAU_V, A_SHIP) if float(np.linalg.norm(dvp)) > PARK_DEAD else np.zeros(3)
        a = _clip(a + _avoid(w, x, v), A_SHIP)
        fg = _fall_guard(w, x, v)
        if fg is not None:
            a = fg
        p.vx += float(a[0]) * h; p.vy += float(a[1]) * h; p.vz += float(a[2]) * h
        s["a"] = a
        fire = False
        if T is not None and sn in stations:
            pT = _pp(w, T)
            xT = np.array(pT.xyz)
            off = float(np.linalg.norm(x - stations[sn][0]))
            f = xT - x
            fn = float(np.linalg.norm(f))
            fh = f / max(fn, 1e-9)
            if demo:
                if (not nuke_out and tm.get("reload", 0.0) <= sim.t and off < 2.0 * FIRE_TOL
                        and not _star_blocks(w, x, xT)):
                    vT = np.array(pT.vxyz)
                    tf = fn / NUKE_V
                    aim = xT + (vT - v) * tf - x
                    aim /= max(float(np.linalg.norm(aim)), 1e-9)
                    w["n_nuke"] += 1
                    nm = f"nuke{w['n_nuke']}"
                    vn = v + aim * NUKE_V
                    xn = x + aim * 2.0 * (SHIP_R + NUKE_R)
                    sim.add(m=NUKE_M, r=NUKE_R, x=float(xn[0]), y=float(xn[1]), z=float(xn[2]),
                            vx=float(vn[0]), vy=float(vn[1]), vz=float(vn[2]))
                    w["ids"].append(nm)
                    w["nukes"][nm] = dict(target=T, Jfull=DEMO_OVER * Q_STAR * 2.0 * pT.m / (V_EJ * DEMO_FRAC), t0=sim.t, prev=None, fuse=0.0, near=0.03)
                    tm["reload"] = sim.t + NUKE_RELOAD
                    nuke_out = True
                    w["log"].append(f"NUKE {nm} (demolition, sized to just break it) launched by {sn} at {T}")
                if s["laser"]:
                    w["log"].append(f"LASER {sn} off")
                s["laser"], s["beam"] = False, None
                continue
            aim_ok = (needn > 0.0 and float(fh @ dnow) > (FIRE_COS if gate == "eff" else FIRE_COS_NOW)
                      and not _star_blocks(w, x, xT))
            nuke_due = (needn > NUKE_DV and tm.get("reload", 0.0) <= sim.t and not star_mode
                        and T != w.get("refill_catch") and not feeding)
            hold = nuke_due and off < 2.0 * FIRE_TOL and aim_ok
            on_at = COAST_DV if gate == "eff" else 0.25 * COAST_DV
            lit = needn > (0.5 * on_at if s["laser"] else on_at) and (effok or s["laser"])
            nuke_ok_here = nukeok
            if pact is not None:
                lit = _gate(lit, pact["laser"]) and needn > 0.0
                nuke_ok_here = _gate(nukeok, pact["nuke"])
            star_go = (not star_mode) or w.get("star", {}).get("allow", False)
            fire = aim_ok and lit and fn < 1.6 * D_STAND and not nuke_out and not hold and star_go
            if (s["laser"] and not fire and aim_ok and not nuke_out and fn < 1.6 * D_STAND
                    and needn > 0.25 * COAST_DV and star_go):
                fire = sim.t - s.get("t_on", 0.0) < MIN_BURN
            if fire:
                Jf = min(F_LASER * h, 0.5 * pT.m * needn * max(float(fh @ dnow), 0.0))
                if Jf > 0.0:
                    Jm, dm = light_push(w, T, fh * Jf)
                    tm["J"] += Jm; tm["dm"] += dm
                    needn = max(needn - Jm / pT.m, 0.0)
                s["beam"] = xT - fh * pT.r
                s["pw"] = Jf / (F_LASER * h)
            if nuke_due and nuke_ok_here and not nuke_out and off < 2.0 * FIRE_TOL and aim_ok:
                vT = np.array(pT.vxyz)
                Pb = xT - dnow * (pT.r + FUSE)
                tf = float(np.linalg.norm(Pb - x)) / NUKE_V
                aim = Pb + (vT - v) * tf - x
                aim /= max(float(np.linalg.norm(aim)), 1e-9)
                if pact is not None:
                    aim = aim + pR.T @ pact["aim"]
                    aim /= max(float(np.linalg.norm(aim)), 1e-9)
                fp = _sky_frac(pT.r, pT.r + FUSE)
                dvw = needn * NUKE_SHARE
                Jw = min(pT.m * dvw / (1.0 + dvw / V_EJ), J_NUKE_MAX)
                if pact is not None:
                    Jw = min(Jw * pact["yscale"], J_NUKE_MAX)
                w["n_nuke"] += 1
                nm = f"nuke{w['n_nuke']}"
                vn = v + aim * NUKE_V
                xn = x + aim * 2.0 * (SHIP_R + NUKE_R)
                sim.add(m=NUKE_M, r=NUKE_R, x=float(xn[0]), y=float(xn[1]), z=float(xn[2]),
                        vx=float(vn[0]), vy=float(vn[1]), vz=float(vn[2]))
                w["ids"].append(nm)
                w["nukes"][nm] = dict(target=T, Jfull=Jw / fp, t0=sim.t, prev=None)
                s["reload"] = sim.t + NUKE_RELOAD
                tm["reload"] = sim.t + NUKE_RELOAD
                nuke_out = True
                w["log"].append(f"NUKE {nm} launched by {sn} at {T} ({fn:.2f} away, yield for dv {Jw / pT.m:.3f})")
        if fire and not s["laser"]:
            s["t_on"] = sim.t
        if fire != s["laser"]:
            w["log"].append(f"LASER {sn} {'on' if fire else 'off'}" + (f" -> {T}" if fire else ""))
        s["laser"] = fire
        if not fire:
            s["beam"] = None
    w["E_ext"] += energy(w) - e0

def _start_flare(w, amp=None, direction=None, why="AUTO"):
    """Ignite one flare/CME. `direction` is the outward world direction; None = automatic pseudo-random angle."""
    sim = w["sim"]
    if sim is None:
        return None
    spot = float(w.get("star_spot", 0.0))
    if amp is None:
        amp = 0.65 + 0.35 * spot
    if direction is None:
        a = (sim.t * 2.399 + 0.7) % (2.0 * math.pi)
        d = np.array([math.cos(a), math.sin(a), 0.12])
    else:
        d = np.asarray(direction, dtype=float).copy()
    d /= max(float(np.linalg.norm(d)), 1e-9)
    w["flare"] = dict(t0=sim.t, dir=d, amp=float(amp), applied=False)
    w["flare_next"] = sim.t + FLARE_GAP * (0.8 + 0.4 * (1.0 - spot))
    w["n_flare"] = w.get("n_flare", 0) + 1
    w["log"].append(f"STAR FLARE #{w['n_flare']} amp {float(amp):.2f} ({why} CME cone)")
    return w["flare"]

def _grains_init(n):
    """Grains on circular orbits around the star (massless: their speed is the two-body speed around the star);
    surface density ~ flat between R_IN and R_OUT, a hair of thickness (z ~ 0.004) and eccentricity."""
    rng = np.random.default_rng(1)
    r = np.sqrt(rng.uniform(R_IN ** 2, R_OUT ** 2, n))
    th = rng.uniform(0.0, 2.0 * math.pi, n)
    z = rng.normal(0.0, 0.004, n)
    v = np.sqrt(1.0 / r) * (1.0 + rng.normal(0.0, 0.001, n))
    P = np.zeros((n, 4), dtype="f4")
    V = np.zeros((n, 4), dtype="f4")
    P[:, 0], P[:, 1], P[:, 2], P[:, 3] = r * np.cos(th), r * np.sin(th), z, 1.0
    V[:, 0], V[:, 1] = -v * np.sin(th), v * np.cos(th)
    V[:, 2] = rng.normal(0.0, 0.002, n)
    return P, V

def _reset(api, s, why):
    s["w"] = new_world()
    P, V = _grains_init(GW * GH)
    s["gpos"].write(P.tobytes())
    s["gvel"].write(V.tobytes())
    Z = np.zeros((EW * EW, 4), dtype="f4")
    s["epos"].write(Z.tobytes())
    s["evel"].write(Z.tobytes())
    s["e_off"], s["e_wrapped"], s["e_seed"] = 0, False, 1
    s["cnt"].write(np.zeros(16, dtype=np.uint32).tobytes())
    s["cnt_t"], s["cnt_pending"], s["cnt_names"] = 0.0, False, []
    s["g_off"] = 0
    s["grab"] = dict(on=False, held=[], pv=None, vel=np.zeros(3))
    if s["w"]["sim"] is None:
        print("ringbox: REBOUND NOT INSTALLED -- nothing moves", file=sys.stderr)
    else:
        print(f"ringbox: {why} -- REBOUND {rebound.__version__} IAS15, star + {N_MOONS} moons "
              f"(GM {', '.join('%.0e' % m[2] for m in MOONS)}), {GW * GH} grains, box half-width {BOX_W:.1f}",
              file=sys.stderr)

def setup(api):
    s = api.state
    s["k_clear"] = api.kernel("ring.comp", "CLEAR")
    s["k_grain"] = api.kernel("ring.comp", "GRAIN")
    s["gpos"] = api.texture2d((GW, GH), "rgba32f", nearest=True)
    s["gvel"] = api.texture2d((GW, GH), "rgba32f", nearest=True)
    s["k_spawn"] = api.kernel("ring.comp", "SPAWN")
    s["k_spawnr"] = api.kernel("ring.comp", "SPAWNR")
    s["epos"] = api.texture2d((EW, EW), "rgba32f", nearest=True)
    s["evel"] = api.texture2d((EW, EW), "rgba32f", nearest=True)
    c = api.ctx.texture((16, 1), 1, dtype="u4")
    c.filter = (moderngl.NEAREST, moderngl.NEAREST)
    s["cnt"] = c
    g4 = api.ctx.texture((RING_GRID_N, RING_GRID_N), 1, dtype="u4")
    g4.filter = (moderngl.NEAREST, moderngl.NEAREST)
    g4.write(np.zeros(RING_GRID_N * RING_GRID_N, dtype=np.uint32).tobytes())
    s["grid"], s["grid_pending"], s["grid_t"] = g4, False, 0.0
    cw, ch = hand.canvas(api)
    s["canvas"] = (int(cw), int(ch))
    g = api.ctx.texture(s["canvas"], 1, dtype="u4")
    g.filter = (moderngl.NEAREST, moderngl.NEAREST)
    s["acc"] = g
    api.expose("u_acc", g)
    s["t_log"] = 0.0
    s["itime"] = 0.0
    _reset(api, s, "fresh system")

def _pick(w, ro, rd):
    B = bodies(w)[1:]
    if not B:
        return None
    i, _ = hand.pick(ro, rd, [b[1] for b in B], [b[2] for b in B], PICK_SLACK)
    return None if i is None else B[i][5]

def _ring_point(w, ro, rd):
    """Where the hand's ray meets the ring plane (the star's z), if it does (and not absurdly far)."""
    sz = float(w["sim"].particles[0].z)
    if abs(rd[2]) < 1e-6:
        return None
    t = (sz - ro[2]) / rd[2]
    if t <= 0.0:
        return None
    P = ro + rd * t
    return P if float(np.linalg.norm(P - np.array(w["sim"].particles[0].xyz))) < BOX_W else None

def _grab(api, s, xy, held, dt):
    g = s["grab"]
    w = s["w"]
    ro, fw, rt, up = camera()
    ro, rd = hand.ray(xy[0], xy[1], hand.canvas(api), ro, fw, rt, up, cam_fov())
    rate = SIM_RATE
    if held and not g["on"]:
        k = _pick(w, ro, rd)
        g["held"] = []
        if k is not None:
            p = w["sim"].particles[k]
            g["held"] = [(w["ids"][k], float((np.array(p.xyz) - ro) @ rd))]
            w["log"].append(f"HAND grab {w['ids'][k]}")
        else:
            g["plow"] = _ring_point(w, ro, rd)
            if g["plow"] is not None:
                w["log"].append("HAND plows the ring")
        g["on"], g["pv"], g["vel"] = True, None, np.zeros(3)
    elif held and g["on"]:
        if g.get("plow") is not None:
            g["plow"] = _ring_point(w, ro, rd)
        for nm, depth in g["held"]:
            if nm not in w["ids"]:
                continue
            k = w["ids"].index(nm)
            P = ro + rd * depth
            if g["pv"] is not None and dt > 1e-4:
                g["vel"] = 0.7 * g["vel"] + 0.3 * (P - g["pv"]) / dt
            g["pv"] = P
            hand_set_vel(w, k, (P - np.array(w["sim"].particles[k].xyz)) / (GRAB_FOLLOW_S * rate))
    elif not held and g["on"]:
        for nm, _ in g["held"]:
            if nm in w["ids"]:
                v = g["vel"] * GRAB_THROW / rate
                hand_set_vel(w, w["ids"].index(nm), v)
                w["log"].append(f"HAND throw {nm} at {float(np.linalg.norm(v)):.3f}")
        g["on"], g["held"], g["plow"] = False, [], None

def step(api, dt):
    s = api.state
    dt = min(max(dt, 0.0), DT_MAX)
    s["itime"] += dt
    w = s["w"]
    pilot_poll(s["itime"])
    global CAM_MODE, STAR_LUM
    if w["sim"] is not None:
        tsim = w["sim"].t
        cyc = 0.5 - 0.5 * math.cos(2.0 * math.pi * tsim / STAR_CYCLE)
        spot = max(0.0, cyc - 0.55) / 0.45
        STAR_LUM = 1.0 - 0.12 * spot + 0.02 * (cyc - 0.5)
        w["star_lum"], w["star_spot"] = STAR_LUM, spot
        w["flare"] = None
    inp = getattr(api, "inputs", None) or {}
    shocks = tuple(inp.get("pad_shocks") or ()) + (0.0, 0.0)
    sl = inp.get("pad_sliders") or {}
    want = None
    if sl.get("cam_mode") is not None:
        ck = max(0, min(4, int(round(float(sl["cam_mode"])))))
        if s.get("sl_cam") is not None and ck != s["sl_cam"]:
            want = ck
        s["sl_cam"] = ck
    elif "sl_cam" not in s:
        want = 1 if shocks[1] > 0.5 else 0
    dp = inp.get("pad_dpad") or {}
    turn = int(dp.get("RIGHT", 0)) - int(dp.get("LEFT", 0))
    if turn:
        want = (int(s.get("cam_preset", 0)) + turn) % len(CAM_PRESETS)
    if want is not None:
        s["cam_preset"] = want
        _RIDE.update(ro=None, at=None, dir=None)
        CAM_MODE = "ride" if want >= 2 else ("portrait" if want == 1 else "normal")
        w["log"].append(f"CAM {CAM_PRESETS[want]} ({'dpad' if turn else 'slider'})")
    if CAM_MODE == "ride" and not ride_update(w, int(s.get("cam_preset", 0)), dt):
        CAM_MODE = "normal"
    joy = inp.get("pad_joy") or {}
    mx, my = (_dz(v) for v in (joy.get(STICK_MOVE) or (0.0, 0.0)))
    lx, ly = (_dz(v) for v in (joy.get(STICK_LOOK) or (0.0, 0.0)))
    climb = int(dp.get("UP", 0)) - int(dp.get("DOWN", 0))
    if (mx or my or lx or ly or climb) and CAM_MODE != "free":
        ro0, fw0, _, _ = camera()
        free_engage(ro0, fw0)
        CAM_MODE = "free"
        w["log"].append("CAM FREE (stick) from the current view")
    if CAM_MODE == "free":
        free_update(mx, my, lx, ly, climb, dt)
    farm_on = not float(inp.get("pad_farm_off", 0.0) or 0.0) > 0.5
    if farm_on != s["w"].get("farm_on", True):
        print(f"ringbox: RING FARMING {'ON' if farm_on else 'OFF'} (pad switch)", file=sys.stderr)
    s["w"]["farm_on"] = farm_on
    for kind, xy in hand.events(api, s):
        if kind == "reset":
            _reset(api, s, "YOUR HAND: reset")
            w = s["w"]
        elif kind == "boop" and w["sim"] is not None:
            ro, fw, rt, up = camera()
            ro, rd = hand.ray(xy[0], xy[1], hand.canvas(api), ro, fw, rt, up, cam_fov())
            k = _pick(w, ro, rd)
            if k is not None:
                hand_kick(w, k, rd * BOOP_DV)
            else:
                P = _ring_point(w, ro, rd)
                if P is not None:
                    s["splash"] = (P, w["sim"].t + SPLASH_T)
                    rr = float(np.linalg.norm(P - np.array(w["sim"].particles[0].xyz)))
                    w["log"].append(f"HAND splash in the ring at r {rr:.2f}")
    minp = tuple((getattr(api, "inputs", None) or {}).get("mouse", (0.0,) * 7)) + (0.0,) * 7
    if w["sim"] is not None and (int(minp[0]) == 2 or s["grab"]["on"]):
        _grab(api, s, (minp[1], minp[2]), int(minp[0]) == 2 and minp[3] > 0.5, dt)
    fire = tuple(inp.get("pad_fire") or ()) + (0.0, 0.0, 0.0)
    mode_fire = tuple(inp.get("pad_mode") or ()) + (0.0, 0.0, 0.0)
    if fire[0] > 0.5:
        if w.get("mode", MODE_NORMAL) != MODE_NORMAL:
            w["log"].append("STAR SLING: button ignored -- press NORMAL first")
        elif "C" not in w["ids"]:
            w["log"].append("STAR SLING: C is gone -- nothing to brake")
        else:
            _mode_set(w, MODE_STAR)
    if fire[1] > 0.5:
        if w.get("mode", MODE_NORMAL) != MODE_NORMAL:
            w["log"].append("SLINGSHOT: button ignored -- press NORMAL first")
        else:
            _mode_set(w, MODE_SLING)
    if fire[2] > 0.5 and w.get("mode", MODE_NORMAL) != MODE_NORMAL:
        _mode_set(w, MODE_NORMAL)
    if mode_fire[0] > 0.5:
        m = w.get("mode", MODE_NORMAL)
        if m in (MODE_NORMAL, MODE_SLING):
            _mode_set(w, MODE_RECOVER)
        else:
            w["log"].append("RECOVER: button ignored -- press NORMAL first")
    if mode_fire[1] > 0.5:
        if w.get("mode", MODE_NORMAL) != MODE_NORMAL:
            w["log"].append("ROSETTE: button ignored -- press NORMAL first")
        else:
            _mode_set(w, MODE_ROSETTE)
    scol = sl.get("star_colour")
    if scol is not None:
        k = max(0, min(len(STAR_MODES) - 1, int(round(float(scol)))))
        if k != int(w.get("star_mode", 0)) and s.get("sl_col") != k:
            w["star_mode"] = k
            w["log"].append(f"STAR COLOUR {STAR_MODES[k]} (slider)")
        s["sl_col"] = k
    smode = sl.get("sys_mode")
    if smode is not None:
        k = max(0, min(2, int(round(float(smode)))))
        if s.get("sl_sys") is not None and k != s["sl_sys"]:
            w["sys_req"] = dict(want=(MODE_NORMAL, MODE_SLING, MODE_ROSETTE)[k], left=False)
            w["log"].append(f"MODE slider -> {w['sys_req']['want']}")
        s["sl_sys"] = k
    rq = w.get("sys_req")
    if rq is not None:
        m = w.get("mode", MODE_NORMAL)
        if m == rq["want"]:
            w["sys_req"] = None
        elif m == MODE_FARM:
            pass
        elif m != MODE_NORMAL:
            if not rq["left"]:
                _mode_set(w, MODE_RECOVER if m == MODE_SLING else MODE_NORMAL)
                rq["left"] = True
        elif rq["want"] == MODE_NORMAL:
            w["sys_req"] = None
        elif w["team"]["target"] is None and not w.get("herd") and not w.get("feed"):
            _mode_set(w, rq["want"])
            w["sys_req"] = None
    if float(inp.get("pad_refill", 0.0) or 0.0) > 0.5:
        if w.get("mode", MODE_NORMAL) != MODE_NORMAL:
            w["log"].append("REFILL: button ignored -- abort the current mode first")
        elif w.get("refill_catch"):
            w["log"].append(f"REFILL: finish the {w['refill_catch']} catch first")
        else:
            slot = refill_slot(w)
            if slot is None:
                w["log"].append("REFILL: A/B/C are all alive -- nothing to fill")
            else:
                _mode_set(w, MODE_FARM, target=slot)
    if float(inp.get("pad_star_tint", 0.0) or 0.0) > 0.5:
        w["star_mode"] = (int(w.get("star_mode", 0)) + 1) % len(STAR_MODES)
        w["log"].append(f"STAR COLOUR {STAR_MODES[w['star_mode']]}")
    h = dt * SIM_RATE
    phys_step(w, h)
    ships_step(w, h, grabbed=[nm for nm, _ in s["grab"]["held"]] if s["grab"]["on"] else ())
    _mode_update(w)
    cool(w, dt)
    for line in w["log"]:
        print("ringbox: " + line, file=sys.stderr)
    w["log"].clear()

    B = bodies(w)
    st = B[0][1] if B else np.zeros(3)
    mo = np.zeros((4, 4), dtype="f4")
    mr = np.zeros(4, dtype="f4")
    mh = np.zeros(4, dtype="f4")
    for j, b in enumerate(B[1:5]):
        mo[j, :3], mo[j, 3], mr[j], mh[j] = b[1], b[3], b[2], b[4]
    star_m = B[0][3] if B else 1.0
    star_r = B[0][2] if B else STAR_R
    grid_now = 0
    if s["grid_pending"]:
        counts = np.frombuffer(s["grid"].read(), dtype=np.uint32).reshape(RING_GRID_N, RING_GRID_N)
        w["ring_cells"] = ring_cells_from_counts(counts)
        s["grid"].write(np.zeros(RING_GRID_N * RING_GRID_N, dtype=np.uint32).tobytes())
        s["grid_pending"] = False
    else:
        s["grid_t"] += dt
        if s["grid_t"] >= RING_GRID_EVERY:
            s["grid_t"], grid_now, s["grid_pending"] = 0.0, 1, True
    count_now = 0
    if s["cnt_pending"] and w["sim"] is not None:
        raw = np.frombuffer(s["cnt"].read(), dtype=np.uint32).astype(np.float64)
        s["cnt"].write(np.zeros(16, dtype=np.uint32).tobytes())
        s["cnt_pending"] = False
        w["ring_live"] = raw[0] * CNT_SAMPLE
        w["ring_frac"] = w["ring_live"] / (GW * GH)
        e0 = energy(w)
        dm_star = raw[1] * CNT_SAMPLE * RING_GRAIN_M
        if dm_star > 0.0:
            w["sim"].particles[0].m += dm_star
            w["ring_in_star"] += dm_star
        dm_all = dm_star
        for j, nm in enumerate(s["cnt_names"][:4]):
            dmj = raw[2 + j] * CNT_SAMPLE * RING_GRAIN_M
            if dmj > 0.0 and nm in w["ids"]:
                _pp(w, nm).m += dmj
                dm_all += dmj
        w["ring_in"] += dm_all
        w["E_ext"] += energy(w) - e0
    else:
        s["cnt_t"] += dt
        if s["cnt_t"] >= CNT_EVERY:
            s["cnt_t"], count_now, s["cnt_pending"] = 0.0, 1, True
            s["cnt_names"] = [b[0] for b in B[1:5]]
    hand_u, hand_r = (0.0, 0.0, 0.0, 0.0), SPLASH_R
    spl = s.get("splash")
    if spl is not None and w["sim"] is not None and w["sim"].t < spl[1]:
        hand_u = (float(spl[0][0]), float(spl[0][1]), float(spl[0][2]), SPLASH_DV / SPLASH_T)
    elif s["grab"].get("plow") is not None:
        P = s["grab"]["plow"]
        hand_u, hand_r = (float(P[0]), float(P[1]), float(P[2]), PLOW_A), PLOW_R
    ro, fw, rt, up = camera()
    cw, ch = s["canvas"]
    api.run(s["k_clear"], ((cw + 15) // 16, (ch + 15) // 16, 1), images={0: s["acc"]},
            uniforms={"u_canvas": (cw, ch)})
    NG = GW * GH
    for (ex, ev, ed, eR, en, ecos, escale) in w["emit"]:
        n = int(float(en) * GRAIN_MASS / RING_GRAIN_M)
        if n <= 0:
            continue
        dead = max(1.0 - float(w.get("ring_frac", 1.0)), 0.01)
        scan = min(int(n / dead * 1.3) + 4096, NG)
        api.run(s["k_spawnr"], ((scan + 255) // 256, 1, 1), images={0: s["gpos"], 1: s["gvel"], 3: s["cnt"]},
                uniforms={"u_ec": tuple(map(float, ex)), "u_ev": tuple(map(float, ev)), "u_ed": tuple(map(float, ed)),
                          "u_eR": float(eR), "u_es": float(escale), "u_ecos": float(ecos), "u_en": n,
                          "u_escan": scan, "u_eoff": int(s["g_off"]), "u_epw": GW, "u_eseed": int(s["e_seed"])})
        got = int(np.frombuffer(s["cnt"].read(), dtype=np.uint32)[14])
        s["cnt"].write(np.zeros(1, dtype=np.uint32).tobytes(), viewport=(14, 0, 1, 1))
        placed = min(got, n)
        s["e_seed"] += 1
        s["g_off"] = (s["g_off"] + scan) % NG
        w["ring_placed"] = w.get("ring_placed", 0) + placed
        if placed < n:
            w["dust"] = w.get("dust", 0.0) + (n - placed) * RING_GRAIN_M
            print(f"ringbox: RING FEED: {n - placed} of {n} debris grains found no free ring slot (booked as dust)",
                  file=sys.stderr)
    w["emit"].clear()
    flare = w.get("flare")
    if flare is not None and w["sim"] is not None:
        flare_age = max(0.0, min(1.0, (w["sim"].t - flare["t0"]) / FLARE_SIM))
        flare_amp = float(flare["amp"] * max(0.0, 1.0 - flare_age) ** 0.7)
        flare_dir = tuple(map(float, flare["dir"]))
    else:
        flare_age, flare_amp, flare_dir = 0.0, 0.0, (0.0, 0.0, 1.0)
    gu = {"u_h": float(h), "u_star": (float(st[0]), float(st[1]), float(st[2]), float(star_m)),
          "u_starR": float(star_r),
          "u_m0": tuple(map(float, mo[0])), "u_m1": tuple(map(float, mo[1])),
          "u_m2": tuple(map(float, mo[2])), "u_m3": tuple(map(float, mo[3])),
          "u_mr": tuple(map(float, mr)), "u_boxW": float(BOX_W), "u_wallE": float(WALL_E),
          "u_ro": tuple(map(float, ro)), "u_fw": tuple(map(float, fw)), "u_rt": tuple(map(float, rt)),
          "u_up": tuple(map(float, up)), "u_fov": float(cam_fov()), "u_canvas": (cw, ch),
          "u_gw": GW, "u_splatK": float(SPLAT_K_RING), "u_starLum": float(STAR_LUM),
          "u_flare": flare_amp, "u_flareDir": flare_dir, "u_flareAge": flare_age,
          "u_prK": float(PR_K), "u_countLive": int(count_now), "u_cbase": 0, "u_gridOn": int(grid_now),
          "u_hand": hand_u, "u_handR": float(hand_r),
          "u_starV": tuple(map(float, w["sim"].particles[0].vxyz)) if w["sim"] is not None else (0.0, 0.0, 0.0)}
    api.run(s["k_grain"], (GW // 16, GH // 16, 1), images={0: s["gpos"], 1: s["gvel"], 2: s["acc"], 3: s["cnt"], 4: s["grid"]},
            uniforms=gu)
    api.render_uniform("u_ro", tuple(map(float, ro)))
    api.render_uniform("u_fw", tuple(map(float, fw)))
    api.render_uniform("u_rt", tuple(map(float, rt)))
    api.render_uniform("u_up", tuple(map(float, up)))
    api.render_uniform("u_fov", float(cam_fov()))
    api.render_uniform("u_star", (float(st[0]), float(st[1]), float(st[2]), float(star_r)))
    for j in range(4):
        api.render_uniform(f"u_m{j}", (float(mo[j, 0]), float(mo[j, 1]), float(mo[j, 2]), float(mr[j])))
    api.render_uniform("u_mh", tuple(map(float, mh)))
    api.render_uniform("u_splatK", float(SPLAT_K))
    api.render_uniform("u_starLum", float(STAR_LUM))
    api.render_uniform("u_starMode", int(w.get("star_mode", 0)))
    api.render_uniform("u_flare", float(flare_amp))
    api.render_uniform("u_flareDir", flare_dir)
    api.render_uniform("u_flareAge", float(flare_age))
    for i, sn in enumerate(SHIPS):
        sh = w["ships"][sn]
        alive = w["sim"] is not None and sn in w["ids"]
        sp = np.array(_pp(w, sn).xyz) if alive else np.zeros(3)
        api.render_uniform(f"u_sp{i}", (float(sp[0]), float(sp[1]), float(sp[2]), 1.0 if alive else 0.0))
        fa = sh["a"] / A_SHIP if alive else np.zeros(3)
        api.render_uniform(f"u_sa{i}", (float(fa[0]), float(fa[1]), float(fa[2]), 0.0))
        sv = (np.array(_pp(w, sn).vxyz) - np.array(w["sim"].particles[0].vxyz)) if alive else np.zeros(3)
        thr = float(np.linalg.norm(fa))
        want = fa / thr if thr > 0.08 else sv / max(float(np.linalg.norm(sv)), 1e-9)
        hd = sh.get("hd")
        if hd is None or not alive:
            hd = want
        else:
            k = 1.0 - math.exp(-dt / 0.12)
            hd = hd * (1.0 - k) + want * k
            hd = hd / max(float(np.linalg.norm(hd)), 1e-9)
        sh["hd"] = hd
        api.render_uniform(f"u_shv{i}", (float(hd[0]), float(hd[1]), float(hd[2]), 0.0))
        bm = sh["beam"] if (alive and sh["beam"] is not None) else None
        api.render_uniform(f"u_bm{i}", (0.0, 0.0, 0.0, 0.0) if bm is None else
                           (float(bm[0]), float(bm[1]), float(bm[2]), 1.0))
    nk = [nm for nm in w["nukes"] if nm in w["ids"]][:4]
    for j in range(4):
        if j < len(nk):
            q = np.array(_pp(w, nk[j]).xyz)
            api.render_uniform(f"u_nk{j}", (float(q[0]), float(q[1]), float(q[2]), 1.0))
        else:
            api.render_uniform(f"u_nk{j}", (0.0, 0.0, 0.0, 0.0))
    tnow = w["sim"].t if w["sim"] is not None else 0.0
    fl = w["flash"]
    fs = [0.0] * 4
    for j in range(4):
        if j < len(fl) and tnow - fl[j][1] < FLASH_SIM:
            q = fl[j][0]
            api.render_uniform(f"u_fx{j}", (float(q[0]), float(q[1]), float(q[2]), (tnow - fl[j][1]) / FLASH_SIM))
            fs[j] = min(1.0, math.sqrt(fl[j][2] / J_NUKE_MAX * 0.3) + 0.3)
        else:
            api.render_uniform(f"u_fx{j}", (0.0, 0.0, 0.0, -1.0))
    api.render_uniform("u_fxs", tuple(fs))

    if s["itime"] >= s["t_log"]:
        s["t_log"] = s["itime"] + LOG_S
        if w["sim"] is not None:
            E, L = energy(w), ang_mom(w)
            dE = (E + w["E_heat"] - w["E_self"] - w["E_ext"] + w["E_grain"] - w["E0"]) / abs(w["E0"])
            mass_now = B[0][3] + sum(b[3] for b in B[1:]) + w["ring_mass"] - w.get("ring_in", 0.0)
            dM = (mass_now - w["mass_total0"]) / w["mass_total0"]
            alive = [f"{b[0]} r {math.sqrt(((b[1] - st) ** 2).sum()):.2f} T {b[4]:.2f}" for b in B[1:]]
            if w.get("ring_cells") is not None:
                rp = []
                for b in B[1:]:
                    d2 = float(((b[1] - st) ** 2).sum())
                    rp.append(f"{b[0]} {100.0 * float(np.linalg.norm(ring_acc(w, b[1]))) / (B[0][3] / max(d2, 1e-9)):.2f} %")
                print(f"ringbox: ring pull vs the star's pull: {', '.join(rp)} | grid cells {len(w['ring_cells'][1])} mass {w['ring_cells'][1].sum():.2e}", file=sys.stderr)
            print(f"ringbox: t {w['sim'].t:.1f} | energy error (hand/walls/heat booked) {dE:+.1e} | |L| {np.linalg.norm(L):.5f} "
                  f"(start {np.linalg.norm(w['L0']):.5f}, lift input {np.linalg.norm(w['L_ext']):.5f}) | "
                  f"mass error {dM:+.1e} | star at ({st[0]:+.5f}, {st[1]:+.5f}, {st[2]:+.5f}) lum {w.get('star_lum', 1.0):.3f} "
                  f"spot {w.get('star_spot', 0.0):.2f} mode {STAR_MODES[int(w.get('star_mode', 0))]} | walls {w['n_wall']} merges {w['n_merge']} | "
                  f"{'; '.join(alive)}", file=sys.stderr)
            tm = w["team"]
            fdl = w.get("feed")
            job = (f"herding the star ({w['herd']['shots']} charges)" if w.get("herd") else
                   f"RING FEED {fdl['phase']}" + (f" ({fdl['imp']} -> {fdl['tgt']})" if fdl.get("imp") else "")
                   if fdl else
                   f"on {tm['target']} (size {size_err(w, tm['target']) * 100:.2f} % off)"
                   if tm["target"] in w["ids"] else
                   (f"racing lap {max(r['lap'] for r in w['race']['ships'].values()) + 1} "
                    f"(score {w['race']['score']['ship0']}-{w['race']['score']['ship1']})"
                    if w.get("race") and w["race"]["phase"] == "go" else
                    "lining up for a race" if w.get("race") else "idle"))
            masses = ", ".join(f"{b[0]} {b[3] / w['mass0'].get(b[0], b[3]) * 100:.0f}%" for b in B[1:])
            print(f"ringbox: ships {job} | lasers {sum(w['ships'][n]['laser'] for n in SHIPS)} | nukes {w['n_nuke']} "
                  f"({w['n_miss']} missed) | wrecks {w['n_wreck']} | moon mass {masses} | ring mass "
                  f"{w['ring_mass']:.2e} ({int(w['ring_mass'] / GRAIN_MASS)} grains) | refills {w['n_refill']} | "
                  f"flares {w.get('n_flare', 0)} | RING {w.get('ring_frac', 1.0) * 100:.1f} % "
                  f"(PR drag; {w.get('ring_in_star', 0.0):.2e} fell into the star; debris placed "
                  f"{w.get('ring_placed', 0)}, dust {w.get('dust', 0.0):.1e}; farming "
                  f"{'on' if w.get('farm_on', True) else 'OFF'})", file=sys.stderr)
            ps = pair_state(w)
            if ps is not None:
                print(f"ringbox: horseshoe {PAIR[0]}+{PAIR[1]}: C {ps[0]:.3f} "
                      f"(mode {w.get('mode', MODE_NORMAL)}, target {w.get('mode_c_lo', PAIR_C_LO):.2f}-"
                      f"{w.get('mode_c_hi', PAIR_C_HI):.2f}), "
                      f"{math.degrees(ps[1]):.0f} deg apart, da {ps[2]:+.4f}, mean distance {ps[3]:.4f}", file=sys.stderr)
