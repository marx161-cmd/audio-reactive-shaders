#!/usr/bin/env python3
"""Orchestrator (director) v1 for shaders/orrery_kerr.mp -- scope.md 2026-10-20 "ORCHESTRATOR (director) v1".

One song analysis + the body reservoir -> a plan: camera pockets per section and body passes timed to the beats.

    python3 director/orchestrate.py [<analysis.npz> | random] [--seed N]

Output: director/plans/<song>.json and <song>.png (timeline to judge by eye).
Beats come first: drops / section changes get hero passes (pairs on the biggest = dual whip), downbeats and snares
get passes by section energy, fillers add wide inclined passes for motion in the lens periphery. Every pass rides the
shader's canonical orbit (canonPos: periapsis default 17 Rs, apoapsis 200 world, warped far clock), periapsis ON the
target beat, its plane chosen against the camera direction at that beat. Melodic accents = v2.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import song_features  # noqa: E402

SHADER = os.path.join(ROOT, "shaders", "kerr-orrery.mp", "common.glsl")
NEBULA = os.path.join(ROOT, "textures", "starfield", "nebula_photo4.npy")
CACHE = os.path.join(HERE, "cache")
PLANS = os.path.join(HERE, "plans")

# ---- geometry, mirrored from the shader (image.glsl camera, common.glsl canonPos) ----------------------------------
BH_RS = 0.048
CAM_R = 42.657458 * BH_RS
AZ0, EL0 = -3.103562, 0.087266
ND = np.array([0.34, 1.0, 0.13]); ND /= np.linalg.norm(ND)
E1 = np.cross(ND, [0.0, 0.0, 1.0]); E1 /= np.linalg.norm(E1)
E2 = np.cross(ND, E1)
CANON_RP, CANON_RA, CANON_GM, CANON_RWARP = 0.82, 200.0, 0.5, 6.0   # = shader canonPos (2026-10-21: GM 37.2 -> 2.0 -> 0.5 = half speed, same paths)
SWING_PRE, SWING_END = 1.0, 2.0          # shader CAM_SWING_PRE / CAM_SWING_END
CLEAR = 0.35                             # world: minimum distance of any body path from the camera
# ---- SLIDERS (2026-10-21, user: "sliders going from calm to high energy so it adjusts to our music values") ---------
# Every section has an energy e in 0..1 (its rank within the song: 0 = calmest section, 1 = the most energetic).
# Each setting slides from its CALM value to its HIGH value as calm + (high - calm) * e ** curve (curve > 1 = rises late).
SLIDERS = {                 # name:          (calm, high, curve)   2026-10-21 (user: "a bar is a full 4/4? then we need to go higher"): raised
    "downbeat_chance":       (0.70, 1.00, 1.0),   # 2026-10-21 test render: ~30 on screen was too many -> all ~1/3 lower   # chance a downbeat gets a pass
    "beat3_chance":          (0.20, 0.70, 1.0),   # chance beat 3 of a bar gets a pass
    "snare_chance":          (0.10, 0.60, 1.5),   # chance a snare hit gets a pass (kicks in late)
    "fillers_per_8_bars":    (2.00, 6.00, 1.0),   # wide passes through the lens periphery
    "boost":                 (-0.10, 0.25, 1.0),  # bigger / brighter bodies, rings more likely
    "star_share":            (0.15, 0.30, 1.0),   # 2026-10-21: was 0.25 -> 0.45 ("a solid third were pulsars")   # of downbeat / snare passes
    "comet_share":           (0.05, 0.25, 1.5),
    "flyby_share":           (0.50, 0.80, 1.0),   # accents (not hero passes) that FLY BY instead of orbiting
    "flyby_vinf":            (0.35, 0.60, 1.0),   # flyby speed far away, world/beat (faster = shorter on screen)
}
FLYBY_OUT = 40.0   # world: a flyby is dropped (gone for good) beyond this


def slide(name, e):
    lo, hi, p = SLIDERS[name]
    return lo + (hi - lo) * max(0.0, min(1.0, e)) ** p


TIME_SCALE = 1.0   # 2026-10-21 slow-motion clock, back to 1 (user chose the lighter hole instead, same effect). Was 0.5: (user: "make the shader move half as fast without changing gravity -- a second takes 2"):
                   # the bodies' orbit clock runs at this rate around each target beat (periapsis still ON the beat);
                   # a pass is in flight lap / TIME_SCALE beats. The shader reads it from the plan (u_plan_ts).


def nz(v):
    return v / np.linalg.norm(v)


def cam_dir(az):
    """hole -> camera unit vector for azimuth offset az (0 = the home view)."""
    a = AZ0 + az
    return nz(math.cos(EL0) * (math.cos(a) * E1 + math.sin(a) * E2) + math.sin(EL0) * ND)


def min_lap(rp):
    """Shortest lap (beats) that still holds the whole real-Kepler window (periapsis <-> CANON_RWARP), + 2 beats."""
    a = 0.5 * (rp + CANON_RA); e = (CANON_RA - rp) / (CANON_RA + rp); n = math.sqrt(CANON_GM / a ** 3)
    E1 = math.acos(max(-1.0, min(1.0, (1 - CANON_RWARP / a) / e)))
    return 2.0 * (E1 - e * math.sin(E1)) / n + 2.0


def flyby_ecc(rp, vinf):
    return 1.0 + rp * vinf * vinf / CANON_GM


FLYBY_ACC = 60.0   # world/beat^2: beyond CANON_RWARP a flyby keeps its direction and accelerates out (like the orbits' far dash)


def _hyp(u, v, rp, e, t):
    aa = rp / (e - 1.0); n = math.sqrt(CANON_GM / aa ** 3); M = n * t
    F = math.asinh(M / e)
    for _ in range(30):
        F -= (e * math.sinh(F) - F - M) / (e * math.cosh(F) - 1.0)
    return u * (aa * (e - math.cosh(F))) + v * (aa * math.sqrt(e * e - 1.0) * math.sinh(F))


def _t_at(rp, e, R):
    aa = rp / (e - 1.0); n = math.sqrt(CANON_GM / aa ** 3)
    F = math.acosh(max(1.0, (R / aa + 1.0) / e))
    return (e * math.sinh(F) - F) / n


def flyby_pos(u, v, rp, e, t):
    """2026-10-21 (user: "fly-by paths instead of orbits are a thing we can use too"): hyperbolic pass, same hole,
    periapsis rp on the target beat, t beats from periapsis -- REAL within CANON_RWARP, then straight on with
    FLYBY_ACC acceleration (out of sight in about a beat, like the orbits' warped far side); = shader flybyPos."""
    tw = _t_at(rp, e, CANON_RWARP)
    if abs(t) <= tw:
        return _hyp(u, v, rp, e, t)
    sg = 1.0 if t > 0 else -1.0
    P = _hyp(u, v, rp, e, sg * tw)
    V = (_hyp(u, v, rp, e, sg * tw + sg * 1e-3) - P) / 1e-3
    sp = np.linalg.norm(V); tau = abs(t) - tw
    return P + V / sp * (sp * tau + 0.5 * FLYBY_ACC * tau * tau)


def flyby_window(rp, e, R=None):
    """Beats from periapsis until the flyby is FLYBY_OUT away (the in-flight window is twice that)."""
    R = R or FLYBY_OUT
    tw = _t_at(rp, e, CANON_RWARP)
    V = (_hyp(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), rp, e, tw + 1e-3) - _hyp(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), rp, e, tw)) / 1e-3
    sp = float(np.linalg.norm(V)); d = R - CANON_RWARP
    tau = (-sp + math.sqrt(sp * sp + 2 * FLYBY_ACC * d)) / FLYBY_ACC
    return tw + tau


def canon_pos_vec(u, v, rp, lap, t):
    """canon_pos for an array of t (vectorised)."""
    t = np.asarray(t, float)
    a = 0.5 * (rp + CANON_RA); e = (CANON_RA - rp) / (CANON_RA + rp); n = math.sqrt(CANON_GM / a ** 3)
    E1w = math.acos(max(-1.0, min(1.0, (1 - CANON_RWARP / a) / e))); tn = (E1w - e * math.sin(E1w)) / n
    h = 0.5 * lap; t = (t + h) % lap - h
    M = n * t; E = M + e * np.sin(M)
    for _ in range(40):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    if h > tn:
        far = np.abs(t) > tn
        D = h - tn; s = (np.abs(t) - tn) / D
        m0 = n / (1 - e * math.cos(E1w)) * D; m1 = math.pi - E1w
        Ef = ((2 * s**3 - 3 * s**2 + 1) * E1w + (s**3 - 2 * s**2 + s) * m0 + (-2 * s**3 + 3 * s**2) * math.pi
              + (s**3 - s**2) * m1) * np.sign(t)
        E = np.where(far, Ef, E)
    return np.outer(a * (np.cos(E) - e), u) + np.outer(a * math.sqrt(1 - e * e) * np.sin(E), v)


def flyby_pos_vec(u, v, rp, e, t):
    return np.array([flyby_pos(u, v, rp, e, float(x)) for x in np.asarray(t, float)])


def canon_pos(u, v, rp, lap, t):
    """Shader canonPos (with a per-pass periapsis rp): position t beats after periapsis."""
    a = 0.5 * (rp + CANON_RA); e = (CANON_RA - rp) / (CANON_RA + rp)
    n = math.sqrt(CANON_GM / a ** 3)
    E1w = math.acos(max(-1.0, min(1.0, (1 - CANON_RWARP / a) / e)))
    tn = (E1w - e * math.sin(E1w)) / n
    h = 0.5 * lap
    t = (t + h) % lap - h
    if abs(t) <= tn or h <= tn:
        M = n * t; E = M + e * math.sin(M)
        for _ in range(40):
            E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    else:
        D = h - tn; s = (abs(t) - tn) / D
        m0 = n / (1 - e * math.cos(E1w)) * D; m1 = math.pi - E1w
        E = ((2 * s**3 - 3 * s**2 + 1) * E1w + (s**3 - 2 * s**2 + s) * m0
             + (-2 * s**3 + 3 * s**2) * math.pi + (s**3 - s**2) * m1) * math.copysign(1, t)
    return u * (a * (math.cos(E) - e)) + v * (a * math.sqrt(1 - e * e) * math.sin(E))


# ---- scenic profile: nebula content behind the hole per camera azimuth (the shader's SKY_Q sky) ---------------------
def sky_q():
    m = re.search(r"#define SKY_Q mat3\(([^)]*)\)", open(SHADER).read())
    vals = [float(x) for x in m.group(1).split(",")]
    return np.array(vals).reshape(3, 3).T          # GLSL mat3 is column-major


def scenic_profile(step_deg=5):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"scenic_{step_deg}.npy")
    Q = sky_q()
    key = np.round(Q, 6)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True).item()
        if np.array_equal(d["q"], key):
            return d["az"], d["s"]
    neb = np.load(NEBULA, mmap_mode="r")
    lum = np.asarray(neb[::16, ::16, :3], dtype=np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    H, W = lum.shape
    azs = np.radians(np.arange(0, 360, step_deg))
    offs = [(a, b) for a in np.radians([-20, -10, 0, 10, 20]) for b in np.radians([-15, -7, 0, 7, 15])]
    out = []
    for az in azs:
        F = -cam_dir(az); R = nz(np.cross(F, ND)); U = np.cross(R, F)
        acc = 0.0
        for a, b in offs:
            d = Q @ nz(F + math.tan(a) * R + math.tan(b) * U)
            uu = math.atan2(d[2], d[0]) / (2 * math.pi) + 0.5
            vv = 1 - math.acos(max(-1, min(1, d[1]))) / math.pi
            acc += lum[min(int(vv * (H - 1)), H - 1), int(uu * (W - 1)) % W]
        out.append(acc / len(offs))
    s = np.array(out)
    np.save(path, {"q": key, "az": azs, "s": s})
    return azs, s


def pocket_candidates():
    """The camera's possible pockets: 4 well-separated azimuths per class (calm = most scenic third, groove = middle,
    energetic = darkest third of the scenic profile). Shared with director/tools/bake_pocket_maps.py."""
    azs, scen = scenic_profile()
    order = np.argsort(scen)                     # darkest .. most scenic
    def spread(idx, n=4, sep=40):                # n well-separated azimuths from a ranked list
        out = []
        for k in idx:
            a = float(azs[k])
            if all(abs((a - o + math.pi) % (2 * math.pi) - math.pi) >= math.radians(sep) for o in out):
                out.append(a)
            if len(out) == n:
                break
        return out
    third = len(order) // 3
    return {"energetic": spread(order[:third]), "groove": spread(order[third:2 * third]),
            "calm": spread(order[::-1][:third])}


# ---- reservoir ------------------------------------------------------------------------------------------------------
PRESETS = [
    {"id": "pulsar", "type": "star", "role": "hero", "mood": "energetic",
     "knobs": {"temp_k": 12000, "size": 0.0012, "glow": 1.0, "jets": 1.0, "ring": None, "shredded": False}},
    {"id": "orange_dwarf", "type": "star", "role": "hero", "mood": "any",
     "knobs": {"temp_k": 3800, "size": 0.09, "glow": 0.6, "jets": 0.0, "ring": None, "shredded": False}},
    {"id": "victim", "type": "star", "role": "hero", "mood": "energetic",
     "knobs": {"temp_k": 5800, "size": 0.08, "glow": 0.8, "jets": 0.0, "ring": None, "shredded": True}},
    {"id": "pg", "type": "planet", "role": "filler", "mood": "any",
     "knobs": {"size": 0.07, "texture": "gas_bands", "hue": 0.0, "tint": [1, 1, 1], "bands": 1.0, "moons": 4,
               "moon_size": 0.01, "ring": None}},
    {"id": "dga", "type": "planet", "role": "hero", "mood": "calm",
     "knobs": {"size": 0.08, "texture": "gas_bands", "hue": 0.0, "tint": [1, 1, 1], "bands": 1.0, "moons": 2,
               "moon_size": 0.01, "ring": {"inner": 1.5, "outer": 1.8, "opacity": 0.8, "tilt_deg": 20}}},
    {"id": "gg3", "type": "planet", "role": "filler", "mood": "groove",
     "knobs": {"size": 0.075, "texture": "gas_bands", "hue": 0.52, "tint": [0.8, 0.98, 1.25], "bands": 0.7,
               "moons": 0, "moon_size": 0.0, "ring": None}},
    {"id": "lava", "type": "planet", "role": "filler", "mood": "energetic",
     "knobs": {"size": 0.02, "texture": "lava", "hue": 0.0, "tint": [1, 1, 1], "bands": 0.0, "moons": 0,
               "moon_size": 0.0, "ring": None}},
]


def load_reservoir():
    p = os.path.join(HERE, "reservoir.json")
    if os.path.exists(p):
        return json.load(open(p))["bodies"], "reservoir.json"
    return PRESETS, "presets (reservoir.json missing)"


class Pool:
    """GENERATED bodies (2026-10-21, user: "why is it still using the presets"): one star / one planet / one comet
    renderer in the shader, so every pass simply gets its own knob set, drawn here from designed distributions (not
    uniform, so combinations stay tasteful). Hero passes and energetic sections get bigger / brighter bodies and rings
    more often. reservoir.json is no longer used by plans (kept for the swatch tool)."""

    def __init__(self, bodies, rng):
        self.rng, self.n = rng, 0
        self.bodies = []                       # nothing to pick from any more; kept for callers that look

    def take(self, peri, lap, want_type=None, want_role=None, mood=None, exclude=()):
        r = self.rng
        t = want_type or r.choices(["planet", "star", "comet"], [0.5, 0.35, 0.15])[0]
        hero = want_role == "hero"
        boost = (0.25 if hero else 0.0) + (slide("boost", mood) if isinstance(mood, float) else 0.0)   # mood = section energy
        self.n += 1
        return {"id": f"gen_{t}_{self.n}", "type": t, "role": "hero" if hero else "filler", "mood": mood,
                "knobs": self.make(t, boost)}

    def make(self, t, boost):
        r = self.rng
        cl = lambda x, lo, hi: min(hi, max(lo, x))
        if t == "star":
            temp = math.exp(r.triangular(math.log(2800), math.log(30000), math.log(6500)))
            ring = {"inner": r.uniform(1.4, 1.8), "outer": r.uniform(1.9, 2.8), "opacity": r.uniform(0.4, 0.8),
                    "tilt_deg": r.uniform(0, 40)} if r.random() < 0.1 else None
            return {"temp_k": temp, "saturation": r.uniform(1.0, 1.6),
                    "size": cl(math.exp(r.uniform(math.log(0.006), math.log(0.07))) * (1 + boost), 0.001, 0.09),
                    "glow": cl(r.uniform(0.6, 1.5) + boost, 0.0, 2.0),
                    "jets": r.uniform(0.5, 1.2) if r.random() < 0.05 else 0.0,   # 2026-10-21: 20% -> 5% (jets read as pulsars)
                    "corona": r.uniform(0.8, 1.4), "flares": r.uniform(0, 1) if r.random() < 0.3 else 0.0,
                    "ring": ring, "ring_count": r.randint(1, 3) if ring else 0, "shredded": False}
        if t == "planet":
            tex = r.choices(["gas_bands", "rock", "lava"], [0.55, 0.30, 0.15])[0]
            size = {"gas_bands": r.uniform(0.045, 0.08), "rock": r.uniform(0.02, 0.05), "lava": r.uniform(0.02, 0.04)}[tex]
            hue = r.random() if tex == "gas_bands" else r.uniform(-0.05, 0.05) % 1.0
            p_ring = {"gas_bands": 0.35, "rock": 0.10, "lava": 0.05}[tex] + boost * 0.4
            ring = {"inner": r.uniform(1.3, 1.7), "outer": r.uniform(1.9, 2.6), "opacity": r.uniform(0.5, 0.95),
                    "tilt_deg": r.uniform(0, 35), "hue": r.random() * 0.3} if r.random() < p_ring else None
            # atmosphere colour from the planet's own hue (a lighter, cooler version of it)
            base = colorsys_hsv((hue + 0.55) % 1.0, 0.45, 1.0)
            return {"texture": tex, "size": cl(size * (1 + 0.5 * boost), 0.02, 0.08), "hue": hue,
                    "tint": [cl(r.gauss(1.0, 0.1), 0.6, 1.4) for _ in range(3)],
                    "bands": r.uniform(0.4, 1.0),
                    "atmosphere": {"gas_bands": r.uniform(0.2, 0.6), "rock": r.uniform(0.0, 0.2), "lava": r.uniform(0.1, 0.3)}[tex],
                    "atmosphere_col": base,
                    "night_glow": r.uniform(0.3, 0.9) if r.random() < 0.15 else 0.0,
                    "moons": r.choice([0, 1, 2, 3, 4]) if tex == "gas_bands" else r.choice([0, 0, 1]),
                    "moon_size": r.uniform(0.004, 0.012),
                    "ring": ring, "ring_count": r.choices([1, 2, 3], [0.5, 0.3, 0.2])[0] if ring else 0}
        return {"size": r.uniform(0.003, 0.015), "tail_len": r.uniform(0.2, 1.0) * (1 + boost),   # 2026-10-21: was 0.4-2.5
                "tail_temp_k": r.uniform(6000, 16000), "brightness": cl(r.uniform(0.6, 1.5) + boost, 0.2, 2.0),
                "tail_split": r.random(), "tail_count": r.choice([1, 1, 2])}

    def vary(self, b):
        return {}


def colorsys_hsv(h, s, v):
    import colorsys
    return list(colorsys.hsv_to_rgb(h, s, v))


# ---- song -> sections, pockets, targets ------------------------------------------------------------------------------
def energy_class(e):
    return "calm" if e < 0.35 else ("groove" if e < 0.7 else "energetic")


def plan_song(npz, seed=0):
    rng = random.Random(seed)
    song = song_features.Song(npz)
    z = np.load(npz)
    hits = {k: z["hits_" + k].astype(float) for k in ("kick", "snare", "hat")}
    secs = []
    for i, s in enumerate(song.sections):
        e = float(s["score"]["energy"])
        secs.append({"i": i, "start": float(s["start_beat"]), "end": float(s["end_beat"]),
                     "energy": e, "repeat": int(s.get("repeat_id", i))})
    # classes are SONG-RELATIVE (section scores are percentiles within the song): thirds by energy rank
    ranks = np.argsort(np.argsort([x["energy"] for x in secs]))
    for x, r in zip(secs, ranks):
        q = r / max(len(secs) - 1, 1)
        x["cls"] = "calm" if q < 1 / 3 else ("groove" if q < 2 / 3 else "energetic")
        x["e"] = float(q)                                  # the slider position of this section

    # camera pockets: calm -> the most scenic azimuth, energetic -> the cleanest dark one, groove -> in between;
    # a repeated section returns to its pocket; consecutive pockets differ (every section change is a move)
    azs, scen = scenic_profile()
    cands = pocket_candidates()
    nxt = {k: 0 for k in cands}
    # 2026-10-21 (user: "it hops more often than on section change"): the analysis cuts songs into ~4-bar sections, so
    # the camera only moves where the section CHARACTER changes (calm / groove / energetic), and at most every
    # MOVE_GAP beats; sections in between keep the pocket (no camera entry)
    MOVE_GAP = 32.0
    pockets, cam = {}, []
    prev = 0.0
    for s in secs:
        if cam and (s["cls"] == cam[-1]["cls"] or s["start"] - cam[-1]["beat"] < MOVE_GAP):
            continue
        if s["repeat"] in pockets:
            az = pockets[s["repeat"]]
        else:
            lst = cands[s["cls"]]
            az = None
            for j in range(len(lst)):            # next candidate of this class that is a comfortable move
                cand = lst[(nxt[s["cls"]] + j) % len(lst)]
                d = abs((cand - prev + math.pi) % (2 * math.pi) - math.pi)
                if not cam or math.radians(25) <= d <= math.radians(120):
                    az = cand; nxt[s["cls"]] += j + 1; break
            if az is None:
                az = lst[nxt[s["cls"]] % len(lst)]; nxt[s["cls"]] += 1
            pockets[s["repeat"]] = az
        cam.append({"section": s["i"], "beat": s["start"], "az": az, "cls": s["cls"],
                    "scenic": float(np.interp(az % (2 * math.pi), azs, scen))})
        prev = az

    def cam_az_at(b):                     # station at beat b incl. the swing (warning 1 beat ahead, land +2)
        az = cam[0]["az"]
        for k in range(1, len(cam)):
            c = cam[k]["beat"]
            if b >= c - SWING_PRE:
                d = (cam[k]["az"] - cam[k - 1]["az"] + math.pi) % (2 * math.pi) - math.pi
                f = min(1.0, max(0.0, (b - (c - SWING_PRE)) / (SWING_PRE + SWING_END)))
                az = cam[k - 1]["az"] + d * (f * f * (3 - 2 * f))
        return az

    # targets (beat index, kind, weight)
    beat_of = song.beat_index_at
    dbs = [beat_of(t) for t in song.downbeats]
    targets = []
    for s in secs:
        ds = [b for b in dbs if s["start"] <= b < s["end"]]
        e = s["e"]                                       # sliders instead of the three fixed rule sets
        targets += [(b, "downbeat", 1) for b in ds if rng.random() < slide("downbeat_chance", e)]
        targets += [(b + 0.5 * song.bpb, "downbeat", 1) for b in ds
                    if b + 0.5 * song.bpb < s["end"] and rng.random() < slide("beat3_chance", e)]
        sn = [beat_of(t) for t in hits["snare"]]
        targets += [(b, "snare", 1) for b in sn if s["start"] <= b < s["end"] and rng.random() < slide("snare_chance", e)]
    for k in range(1, len(secs)):
        targets.append((secs[k]["start"], "section_change", 3))
    for t in song.drops:
        targets.append((beat_of(t), "drop", 4))
    targets.sort()

    pool = Pool(load_reservoir()[0], rng)
    passes, unfilled = [], 0

    def make_pass(body, b, kind, style, lap, rp, mirror=None, ecc=0.0):
        c = cam_dir(cam_az_at(b))
        for _ in range(24):
            if style == "ring":                # periapsis straight behind the hole: Einstein ring on the beat
                u = -c
            else:                              # whip: periapsis beside the hole, around the line of sight
                ang = rng.uniform(0, 2 * math.pi)
                side = nz(np.cross(c, ND)) * math.cos(ang) + nz(np.cross(c, np.cross(c, ND))) * math.sin(ang)
                u = nz(side * 0.9 - c * 0.45)
            w = nz(np.cross(u, np.array([rng.gauss(0, 1) for _ in range(3)])))
            v = nz(np.cross(w, u))
            if mirror is not None:
                u, v = mirror
            cpos = CAM_R * c
            pos = (lambda t: flyby_pos(u, v, rp, ecc, t)) if ecc > 1.0 else (lambda t: canon_pos(u, v, rp, lap, t))
            dmin = min(np.linalg.norm(pos(t) - cpos) for t in np.linspace(-lap / 2, lap / 2, 80))
            if dmin >= CLEAR:
                return {"body": body["id"], "type": body["type"], "role": body.get("role"), "kind": kind,
                        "knobs": body["knobs"],                       # generated per pass (no presets)
                        "style": style, "peri_beat": round(float(b), 4), "lap": lap, "rp": rp,
                        "u": [round(x, 6) for x in u], "v": [round(x, 6) for x in v],
                        "cam_clear": round(float(dmin), 3),
                        **({"path": "flyby", "ecc": round(ecc, 5)} if ecc > 1.0 else {"path": "orbit"})}
            mirror = None
        return None

    for b, kind, wgt in targets:
        s = next((x for x in secs if x["start"] <= b < x["end"]), secs[-1])
        hero = wgt >= 3
        rp_pass = 0.82 if hero else rng.uniform(0.82, 1.4)
        lap = max(24.0 if hero else 16.0, min_lap(rp_pass))   # the lighter hole's Kepler window is longer
        ecc = 0.0
        if not hero and kind not in ("drop", "section_change") and rng.random() < slide("flyby_share", s["e"]):
            ecc = flyby_ecc(rp_pass, slide("flyby_vinf", s["e"]))   # accent: a one-way flyby
            lap = 2.0 * flyby_window(rp_pass, ecc)
        # 2026-10-21 (user: "we need a higher amount of stars passing by, the normal ones"): type mix per target kind
        st, co = slide("star_share", s["e"]), slide("comet_share", s["e"])
        mix = {"drop": {"star": 1.0}, "section_change": {"star": 0.6, "planet": 0.4},
               "downbeat": {"planet": 1 - st - co, "star": st, "comet": co},
               "snare": {"star": st + 0.1, "planet": 0.9 - st - co, "comet": co}}
        w = mix.get(kind, {"planet": 1.0}); r = rng.random(); want = None
        for tname, pw in w.items():
            r -= pw
            if r <= 0: want = tname; break
        body = pool.take(b, lap, want_type=want, want_role="hero" if hero else "filler", mood=s["e"])
        if body is None:
            unfilled += 1; continue
        p = make_pass(body, b, kind, "ring" if hero or rng.random() < 0.4 else "whip", lap, rp_pass, ecc=ecc)
        if p:
            passes.append(p)
        if kind == "drop":                    # dual whip: a second body, mirrored plane, same beat
            b2 = pool.take(b, lap, want_role="hero", mood=s["e"], exclude=(body["id"],))
            if b2 and p:
                u = np.array(p["u"]); v = np.array(p["v"]); c = cam_dir(cam_az_at(b))
                ref = lambda x: nz(2 * np.dot(x, c) * c - x)     # mirror through the line of sight: other side of the hole
                p2 = make_pass(b2, b, "drop_pair", "whip", lap, 0.82, mirror=(ref(u), ref(v)))
                if p2:
                    passes.append(p2)

    # fillers: wide inclined passes through the lens periphery, more in busier sections
    for s in secs:
        n = slide("fillers_per_8_bars", s["e"])
        span = max(s["end"] - s["start"], 1.0)
        for k in range(int(round(n * span / (8 * song.bpb)))):   # n per 8 bars
            b = s["start"] + rng.uniform(0, span)
            rpf = rng.uniform(1.6, 3.5); lapf = max(20.0, min_lap(rpf)); eccf = 0.0
            if rng.random() < slide("flyby_share", s["e"]):
                eccf = flyby_ecc(rpf, slide("flyby_vinf", s["e"])); lapf = 2.0 * flyby_window(rpf, eccf)
            body = pool.take(b, lapf, want_type="planet", want_role="filler", mood=s["e"])
            if body:
                p = make_pass(body, b, "filler", "whip", lapf, rpf, ecc=eccf)
                if p:
                    passes.append(p)

    # SHREDDED SUNS MASK THE CAMERA TURNS (user 2026-10-21: "make them mask the camera turns, time them nicely on each
    # turn"; never picked as normal passes). The gas flipbook is pre-lensed for the view and turns with the camera, so
    # the star is torn up while the backdrop swings behind it: periapsis TDE_LEAD beats before the section change, so
    # the big disk is full (+3.75) mid-swing (-1..+2). Only real turns (> 10 deg), events >= TDE_GAP beats apart.
    TDE_LEAD, TDE_GAP, TDE_EVERY = 3.0, 12.0, 2      # 2026-10-21 (user): every OTHER turn, not every one
    shredders = [{"id": "shredded_sun", "type": "star", "role": "hero",
                  "knobs": {"temp_k": 5800, "size": 0.08, "glow": 0.8, "shredded": True}}]   # the TDE look is the flipbook
    last = -1e9
    nturn = 0
    for k in range(1, len(cam)):
        turn = abs((cam[k]["az"] - cam[k - 1]["az"] + math.pi) % (2 * math.pi) - math.pi)
        pk = cam[k]["beat"] - TDE_LEAD
        if turn < math.radians(10):
            continue
        nturn += 1
        if not shredders or (nturn - 1) % TDE_EVERY or pk - last < TDE_GAP or pk < 2.5:
            continue
        star = shredders[len([p for p in passes if p["kind"] == "shredded"]) % len(shredders)]
        passes.append({"body": star["id"], "type": "star", "role": "hero", "kind": "shredded", "style": "tde",
                       "knobs": star["knobs"],
                       "peri_beat": round(float(pk), 4), "lap": 24.0, "rp": 0.24,
                       "u": [1.0, 0.0, 0.0], "v": [0.0, 0.0, 1.0], "cam_clear": None})
        last = pk
    # LENS-AWARE CUTTING (lens_cut.py): every pass only exists from its first to its last NEEDED moment -- it has an
    # image somewhere on the lens that isn't swallowed by a bright spot of the parked camera's pocket
    import lens_cut
    def parked_az(b):
        for k in range(1, len(cam)):
            if cam[k]["beat"] - SWING_PRE - 0.5 <= b <= cam[k]["beat"] + SWING_END + 1.0:
                return float("nan")                     # mid-swing: no popping
        st = cam[0]["az"]
        for k in range(1, len(cam)):
            if b >= cam[k]["beat"]:
                st = cam[k]["az"]
        return st
    kept, dropped = [], 0
    for p in passes:
        if p["kind"] == "shredded":
            kept.append(p); continue
        u, v = np.array(p["u"]), np.array(p["v"])
        ts = np.arange(-p["lap"] / 2, p["lap"] / 2 + 1e-6, 0.25)
        P = flyby_pos_vec(u, v, p["rp"], p["ecc"], ts) if p.get("path") == "flyby" else canon_pos_vec(u, v, p["rp"], p["lap"], ts)
        bs = p["peri_beat"] + ts
        need = lens_cut.needed(P, np.array([cam_az_at(b) for b in bs]), cam_dir, np.array([parked_az(b) for b in bs]),
                               p["knobs"].get("size", 0.05), ND, E1, E2)
        if not need.any():
            dropped += 1; continue
        i0, i1 = int(np.argmax(need)), len(need) - 1 - int(np.argmax(need[::-1]))
        p["win"] = [round(float(ts[max(i0 - 1, 0)]), 3), round(float(ts[min(i1 + 1, len(ts) - 1)]), 3)]
        kept.append(p)
    passes = kept
    passes.sort(key=lambda p: p["peri_beat"])
    return {"song": os.path.basename(npz), "duration_s": song.duration, "beats": len(song.beats),
            "sections": secs, "camera": cam, "passes": passes, "unfilled": unfilled,
            "reservoir": load_reservoir()[1], "time_scale": TIME_SCALE, "lens_dropped": dropped}


def draw(plan, png):
    from PIL import Image, ImageDraw
    nb = max(plan["beats"], 1)
    W, H, L = 1800, 420, 60
    im = Image.new("RGB", (W, H), (18, 18, 24)); d = ImageDraw.Draw(im)
    X = lambda b: L + (W - L - 10) * b / nb
    col = {"calm": (40, 70, 120), "groove": (60, 110, 70), "energetic": (140, 60, 50)}
    for s in plan["sections"]:
        d.rectangle([X(s["start"]), 20, X(s["end"]), 70], fill=col[s["cls"]])
        d.text((X(s["start"]) + 3, 24), f"{s['cls'][0]}{s['repeat']}", fill=(230, 230, 230))
    pts = [(X(c["beat"]), 220 - 60 * math.degrees((c["az"] + math.pi) % (2 * math.pi) - math.pi) / 180) for c in plan["camera"]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + [(X(nb), pts[-1][1])]):
        d.line([x0, y0, x1, y0], fill=(240, 220, 120), width=2); d.line([x1, y0, x1, y1], fill=(240, 220, 120), width=1)
    tc = {"star": (255, 230, 150), "planet": (120, 200, 255), "comet": (200, 255, 200)}
    for p in plan["passes"]:
        y = {"drop": 300, "drop_pair": 310, "section_change": 330, "downbeat": 350, "snare": 370, "filler": 395, "shredded": 290}[p["kind"]]
        x = X(p["peri_beat"])
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=tc.get(p["type"], (255, 255, 255)))
    for lab, y in (("sections", 40), ("camera az", 220), ("drop", 300), ("sect.chg", 330), ("downbeat", 350),
                   ("snare", 370), ("filler", 395)):
        d.text((4, y - 6), lab, fill=(180, 180, 180))
    d.text((L, 2), f"{plan['song']}  passes {len(plan['passes'])}  unfilled {plan['unfilled']}  ({plan['reservoir']})",
           fill=(255, 255, 255))
    im.save(png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song", nargs="?", default="random")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    npz = a.song
    if npz == "random":
        npz = random.Random(a.seed).choice(glob.glob(os.path.join(song_features.FEATURES_DIR, "*.npz")))
    plan = plan_song(npz, a.seed)
    os.makedirs(PLANS, exist_ok=True)
    base = os.path.join(PLANS, os.path.splitext(os.path.basename(npz))[0])
    json.dump(plan, open(base + ".json", "w"), indent=1)
    draw(plan, base + ".png")
    kinds = {}
    for p in plan["passes"]:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    inw = lambda p, c: (p["win"][0] <= c - p["peri_beat"] <= p["win"][1]) if "win" in p else abs(c - p["peri_beat"]) <= p["lap"] / 2
    occ = max((sum(1 for p in plan["passes"] if p["kind"] != "shredded" and inw(p, c))
               for c in np.arange(0, plan["beats"], 0.5)), default=0)
    if occ > 128:
        print(f"WARNING: up to {occ} bodies in flight > the shader's 128 slots (PLAN_SLOTS): the extras won't be drawn")
    print(f"{plan['song']}: {len(plan['sections'])} sections, {len(plan['passes'])} passes {kinds}, max in flight {occ}, "
          f"unfilled {plan['unfilled']}, reservoir: {plan['reservoir']}")
    print(base + ".json"); print(base + ".png")


if __name__ == "__main__":
    main()
