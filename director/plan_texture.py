"""Pack a director plan (director/plans/<song>.json) into the textures shaders/orrery_kerr.mp reads.
scope.md 2026-10-21 "step 2a".

u_plan     RGBA32F, width = number of passes (>= 1), PLAN_ROWS rows; column j = pass j:
  row 0  (type 0 star / 1 planet / 2 comet, periapsis beat, lap beats, periapsis rp world)
  row 1  (u.xyz periapsis direction, size)
  row 2  (v.xyz in-plane direction of motion, reservoir index)
  row 3  star: (temp_k, saturation, glow, jets)    planet: (texture 0 gas / 1 rock / 2 lava, hue, bands, atmosphere)
         comet: (tail_len, tail_temp_k, brightness, tail_split)
  row 4  star: (corona, flares, shredded, 0)       planet: (tint.rgb, moons)          comet: (tail_count, 0, 0, 0)
  row 5  rings: (ring_count, inner, outer, opacity) in body radii; ring_count 0 = none
  row 6  rings: (ring hue, ring width, tilt deg, moon_size)
  row 7  planet: (atmosphere colour rgb, night-side emissive / aurora)
  row 8  (window start, window end) in beats from periapsis (lens-aware cutting); start >= end = the whole lap
u_camplan  RGBA32F, width = number of camera stations, 1 row: (section start beat, azimuth offset rad, 0, 0)
"""
import json
import os

import numpy as np

PLAN_ROWS = 9
TYPES = {"star": 0, "planet": 1, "comet": 2}
TEXTURES = {"gas_bands": 0, "rock": 1, "lava": 2}


def _ring(k):
    r = k.get("ring")
    n = int(k.get("ring_count", 1 if r else 0))
    if not r or n <= 0:
        return (0, 0, 0, 0), (0, 0, 0)
    return ((n, r.get("inner", 1.5), r.get("outer", 1.8), r.get("opacity", 0.8)),
            (r.get("hue", k.get("ring_hue", 0.0)), r.get("width", 1.0), r.get("tilt_deg", 20.0)))


def pack(plan, reservoir):
    """-> (plan array (PLAN_ROWS, N, 4) f32, camera array (1, M, 4) f32)."""
    idx = {b["id"]: i for i, b in enumerate(reservoir or [])}
    passes = plan["passes"]
    a = np.zeros((PLAN_ROWS, max(len(passes), 1), 4), np.float32)
    for j, p in enumerate(passes):
        # 2026-10-21: plans carry each pass's FULL generated knobs; a reservoir entry (old plans) is only a fallback base
        b = reservoir[idx[p["body"]]] if p["body"] in idx else {"type": p["type"], "knobs": {}}
        k = dict(b.get("knobs", {}))
        k.update(p.get("knobs", {}))
        t = TYPES[p.get("type", b["type"])]
        a[0, j] = (t, p["peri_beat"], p["lap"], p["rp"])
        a[1, j] = (*p["u"], k.get("size", 0.05))
        a[2, j] = (*p["v"], p.get("ecc", 0.0) if p.get("path") == "flyby" else 0.0)   # w > 1 = flyby eccentricity
        rg0, rg1 = _ring(k)
        if t == 0:
            a[3, j] = (k.get("temp_k", 5800), k.get("saturation", 1.0), k.get("glow", 1.0), k.get("jets", 0.0))
            a[4, j] = (k.get("corona", 1.0), k.get("flares", 0.0), 1.0 if k.get("shredded") else 0.0, 0)
        elif t == 1:
            atm_default = {"gas_bands": 0.35, "rock": 0.0, "lava": 0.2}.get(k.get("texture"), 0.0)   # until the orchestrator sets it
            a[3, j] = (TEXTURES.get(k.get("texture"), 0), k.get("hue", 0.0), k.get("bands", 1.0), k.get("atmosphere", atm_default))
            a[4, j] = (*k.get("tint", [1, 1, 1]), k.get("moons", 0))
            a[7, j] = (*k.get("atmosphere_col", [0.55, 0.75, 1.0]), k.get("night_glow", 0.0))
        else:
            a[3, j] = (k.get("tail_len", 1.0), k.get("tail_temp_k", 9000), k.get("brightness", 1.0), k.get("tail_split", 0.0))
            a[4, j] = (k.get("tail_count", 1), 0, 0, 0)
        a[5, j] = rg0
        a[6, j] = (*rg1, k.get("moon_size", 0.0))
        if "win" in p:
            a[8, j] = (p["win"][0], p["win"][1], 0, 0)
    cam = plan.get("camera", [])
    c = np.zeros((1, max(len(cam), 1), 4), np.float32)
    for m, s in enumerate(cam):
        c[0, m] = (s["beat"], s["az"], 0, 0)
    return a, c, len(passes), len(cam)


def load(plan_path, reservoir_path):
    plan = json.load(open(plan_path))
    res = json.load(open(reservoir_path))["bodies"] if os.path.exists(reservoir_path) else None
    if res is None:
        import orchestrate
        res = orchestrate.PRESETS
    return pack(plan, res)
