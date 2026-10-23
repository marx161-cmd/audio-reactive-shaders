"""THE HAND (scope.md 2026-10-12 "THE HAND"), shared helper for compute-stage sims.

The renderer counts BOOPS (steer key + a tap of Alt_L: one poke at the cursor) and RESETS (steer key + Ctrl_L +
Shift_L) and hands them to every sim.py as api.inputs = {"boop": n, "boop_xy": (px, py), "reset": n,
"canvas": (w, h)} (canvas px, y up = GL fragCoord). The same values reach every pass/kernel as
u_hand = (px, py, boop count, reset count). React when a COUNT changes (never misses a one-frame press):

    ev = hand.events(api, s)            # -> ("boop", (px, py)) / ("reset", None) list, since last frame
    ro, rd = hand.ray(px, py, canvas, ro, fw, rt, up, fov)   # the boop as a world ray (this shader's camera)
    i, t = hand.pick(ro, rd, centres, radii, slack)          # nearest body the ray touches (or None)
"""
import numpy as np

def events(api, s, key="_hand_seen"):
    inp = getattr(api, "inputs", None) or {}
    seen = s.get(key)
    cur = (int(inp.get("boop", 0)), int(inp.get("reset", 0)))
    s[key] = cur
    if seen is None:
        return []
    out = []
    if cur[1] != seen[1]:
        out.append(("reset", None))
    if cur[0] != seen[0]:
        out.append(("boop", tuple(inp.get("boop_xy", (0.0, 0.0)))))
    return out

def canvas(api):
    inp = getattr(api, "inputs", None) or {}
    return tuple(inp.get("canvas", (2410.0, 2410.0)))

def ray(px, py, canvas_wh, ro, fw, rt, up, fov):
    """Canvas px -> world ray, for the usual camera: uv = (frag - res/2) / min(res) * 2 * fov,
    rd = normalize(fw + uv.x rt + uv.y up)."""
    w, h = canvas_wh
    m = min(w, h)
    ux = (px - 0.5 * w) / m * 2.0 * fov
    uy = (py - 0.5 * h) / m * 2.0 * fov
    rd = np.asarray(fw) + ux * np.asarray(rt) + uy * np.asarray(up)
    return np.asarray(ro, dtype=np.float64), rd / np.linalg.norm(rd)

def pick(ro, rd, centres, radii, slack=0.0):
    """Index + distance of the nearest sphere (centre, radius + slack) the ray touches, or (None, inf)."""
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(radii, dtype=np.float64).reshape(-1) + slack
    oc = ro[None] - c
    b = oc @ rd
    q = (oc ** 2).sum(1) - r * r
    disc = b * b - q
    t = np.where(disc >= 0.0, -b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
    t = np.where(t > 0.0, t, np.inf)
    if not np.isfinite(t).any():
        return None, np.inf
    i = int(np.argmin(t))
    return i, float(t[i])
