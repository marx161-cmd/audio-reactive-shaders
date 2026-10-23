"""v12's live disk sim on the generic compute stage (renderer.py ComputeStage).

2026-10-03 (user: v11's rough disk as the dwarf flare in v12): v11's disk ran on a one-off
renderer hook that was moved out on 2026-10-02 (shaders/schwarz_orrery_v11.mp/
NOTE_disk_compute_hook.md). This is that hook's load/step logic ported verbatim onto the
script API: same kernel (disk_sim.comp = ../../disk_sim_live.comp minus its #version line,
which api.kernel() adds), same 3-way ping-pong, same seed (disk_sim.Sim), same settings as
v11's manifest "compute" block (below).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import disk_sim as ref

NR, NPHI = 128, 512
FORCE_AMP = 0.6
R_IN, R_OUT = 2.0, 20.0
H_R, FLARE_ = 0.06, 0.0
GAMMA, COOL_BETA = 1.6667, 10.0
DT = 0.003
RATE = 8.0
SUBSTEPS_MAX = 120

def setup(api):
    s = api.state
    s["prog"] = api.kernel("disk_sim.comp")
    sim0 = ref.Sim(np, NR, NPHI, seed=1, force_amp=0.0, r_in=R_IN, r_out=R_OUT, h_r=H_R, flare=FLARE_)
    arr = np.zeros((NPHI, NR, 4), dtype="f4")
    arr[..., 0], arr[..., 1], arr[..., 2] = sim0.U[0].T, sim0.U[1].T, sim0.U[2].T
    arr[..., 3] = (sim0.U[0] * sim0.cs2).T
    bufs = []
    for i in range(3):
        t = api.texture2d((NR, NPHI), "rgba32f", data=arr.tobytes() if i == 0 else None)
        t.repeat_x, t.repeat_y = False, True
        bufs.append(t)
    s.update(bufs=bufs, roles=[0, 1, 2], t=sim0.t, x0=sim0.x0, dx=sim0.dx, dphi=sim0.dphi,
             r_in=sim0.r_in, r_out=sim0.r_out, h_r=sim0.h_r, flare=sim0.flare)
    api.set(s["prog"], {"NR": NR, "NPHI": NPHI, "X0": sim0.x0, "DX": sim0.dx, "DPHI": sim0.dphi,
                        "M_": ref.M, "A_": ref.A_STAR * ref.M, "Q_": ref.Q_STAR * ref.M,
                        "H_R": sim0.h_r, "FLARE": sim0.flare, "GAMMA_M1": GAMMA - 1.0,
                        "COOL_BETA": COOL_BETA, "R_IN": sim0.r_in, "R_OUT": sim0.r_out,
                        "FORCE_AMP": FORCE_AMP})
    api.expose("u_disklive", bufs[0])

def step(api, frame_dt):
    s = api.state
    total = RATE * frame_dt
    n = min(SUBSTEPS_MAX, max(1, int(math.ceil(total / DT))))
    dt = min(total / n, DT)
    bufs, roles, prog = s["bufs"], s["roles"], s["prog"]
    g = ((NR + 15) // 16, (NPHI + 15) // 16, 1)
    for _ in range(n):
        cur, scratch, nxt = bufs[roles[0]], bufs[roles[1]], bufs[roles[2]]
        api.run(prog, g, images={0: cur, 1: cur, 2: scratch}, uniforms={"RK_PASS": 0, "DT": dt, "TIME": s["t"]})
        api.run(prog, g, images={0: scratch, 1: cur, 2: nxt}, uniforms={"RK_PASS": 1, "DT": dt, "TIME": s["t"]})
        s["t"] += dt
        roles = [roles[2], roles[0], roles[1]]
    s["roles"] = roles
    api.expose("u_disklive", bufs[roles[0]])
    s["n"] = s.get("n", 0) + 1
    if s["n"] % 900 == 1:
        S = np.frombuffer(bufs[roles[0]].read(), dtype="f4").reshape(NPHI, NR, 4)[..., 0]
        print(f"disk sim: S min {S.min():.3e} max {S.max():.3e} mean {S.mean():.3e} "
              f"finite {bool(np.isfinite(S).all())} t {s['t']:.1f}", file=sys.stderr)
    for name, val in (("u_disk_x0", s["x0"]), ("u_disk_dx", s["dx"]), ("u_disk_dphi", s["dphi"]),
                      ("u_disk_rin", s["r_in"]), ("u_disk_rout", s["r_out"]),
                      ("u_disk_hr0", s["h_r"]), ("u_disk_flare", s["flare"]), ("u_disk_time", s["t"])):
        api.render_uniform(name, float(val))

import json as _json
import threading as _threading
import time as _time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DIRECTOR = os.path.join(_ROOT, "director")
_REPLAY_FLAG = os.path.expanduser("~/.cache/shader-backdrop/replay_song")
_REPLAY_NOW = os.path.expanduser("~/.cache/shader-backdrop/replay_now.txt")
if _DIRECTOR not in sys.path:
    sys.path.insert(0, _DIRECTOR)

def _plan_source():
    env = os.environ.get("ORRERY_PLAN")
    if env:
        return env, None
    if not os.path.exists(_REPLAY_FLAG):
        return None, None
    try:
        npz = open(_REPLAY_NOW).readline().strip()
    except OSError:
        return None, None
    base = os.path.splitext(os.path.basename(npz))[0]
    return os.path.join(_DIRECTOR, "plans", base + ".json"), npz

def _build_plan(npz, out):
    try:
        import importlib
        import orchestrate
        importlib.reload(orchestrate)
        plan = orchestrate.plan_song(npz, 0)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp"
        _json.dump(plan, open(tmp, "w"), indent=1)
        os.replace(tmp, out)
        print(f"director plan built: {os.path.basename(out)} ({len(plan['passes'])} passes)", flush=True)
    except Exception as e:
        print(f"director plan build failed for {npz}: {e}", flush=True)

def _plan_expose(api, arr, cam, n, m):
    s = api.state
    tp = api.texture2d((arr.shape[1], arr.shape[0]), "rgba32f", data=arr.tobytes(), nearest=True)
    tc = api.texture2d((cam.shape[1], 1), "rgba32f", data=cam.tobytes(), nearest=True)
    s["plan_tex"] = (tp, tc)
    api.expose("u_plan", tp)
    api.expose("u_camplan", tc)
    api.render_uniform("u_plan_n", float(n))
    api.render_uniform("u_cam_n", float(m))
    api.render_uniform("u_plan_ts", float(api.state.get("plan_ts", 1.0)))

def _lens_setup(api):
    """2026-10-21: the inverse lens grid as two 3D textures (u_lens0 / u_lens1) for the star flares' lensed positions."""
    s = api.state
    if "lens_tex" in s:
        return
    import numpy as _np
    lens = _np.load(os.path.join(_ROOT, "textures", "kerr_v7", "kn", "lens_inv.npy"))
    s["lens_tex"] = [api.texture3d((lens.shape[3], lens.shape[2], lens.shape[1]), "rgba32f") for _ in range(2)]
    for k in range(2):
        s["lens_tex"][k].write(_np.ascontiguousarray(lens[k], dtype="f4").tobytes())
    api.expose("u_lens0", s["lens_tex"][0])
    api.expose("u_lens1", s["lens_tex"][1])

def _plan_step(api):
    s = api.state
    _lens_setup(api)
    now = _time.monotonic()
    if now - s.get("plan_chk", 0.0) < 0.5:
        return
    s["plan_chk"] = now
    path, npz = _plan_source()
    if path is None:
        if s.get("plan_path") is not None or "plan_tex" not in s:
            import numpy as _np
            _plan_expose(api, _np.zeros((8, 1, 4), "f4"), _np.zeros((1, 1, 4), "f4"), 0, 0)
            s["plan_path"] = None
        return
    if not os.path.exists(path):
        if npz and s.get("plan_building") != path:
            s["plan_building"] = path
            _threading.Thread(target=_build_plan, args=(npz, path), daemon=True).start()
        return
    mt = os.path.getmtime(path)
    if s.get("plan_path") == path and s.get("plan_mt") == mt:
        return
    try:
        import importlib
        import plan_texture
        importlib.reload(plan_texture)
        arr, cam, n, m = plan_texture.load(path, os.path.join(_DIRECTOR, "reservoir.json"))
        s["plan_ts"] = float(_json.load(open(path)).get("time_scale", 1.0))
        _plan_expose(api, arr, cam, n, m)
        s["plan_path"], s["plan_mt"] = path, mt
        print(f"director plan loaded: {os.path.basename(path)} ({n} passes, {m} camera stations)", flush=True)
    except Exception as e:
        print(f"director plan load failed: {e}", flush=True)
        s["plan_path"], s["plan_mt"] = path, mt

_step_disk = step

def step(api, frame_dt):
    _plan_step(api)
    _step_disk(api, frame_dt)
