"""lagoon_worms.mp (2026-10-13, scope.md): lagoon.mp's worms in OPEN WATER, with the fish/swarm
half removed. A calm water box (worms.comp fluid, no ball) stirred by the two worms, and the
~8M glow particles they release -- speed-boosted so a fast dash stays dense instead of thinning
to a thread -- splatted (straight-projected with the submerged camera, no refraction) into an
integer screen accumulator the image pass reads as u_glow.

The two worms move on fireworms.mp's base egg-carton (pickMode state: circle / figure-8 / rose /
starburst, rose pocket, hat pirouette, drop windup, sponge). lagoon's fish-planner / mirror /
threading movement is gone with the fish. Movement v2 (painting) is deliberately NOT here yet.
"""

import sys

import moderngl
import numpy as np

FW_N = 192
FW_H = 1.6
FW_LEVELS = 6
FW_PRE, FW_POST, FW_WALL, FW_COARSE = 2, 2, 2, 16
FW_EPS = 1e-3
FW_PW, FW_PH = 2048, 4096
FW_NP = FW_PW * FW_PH
FW_RATE = 180000.0
FW_DENS = 160000.0
FW_RATE_MAX = 540000.0
FW_VISC_PASSES = 2
STATS_PERIOD_S = 5.0

def _canvas():
    main = sys.modules.get("__main__")
    return int(getattr(main, "CANVAS_W", 2410)), int(getattr(main, "CANVAS_H", 2410))

def _g3(n):
    g = max(1, (n + 7) // 8)
    return (g, g, g)

def _zeroed(api, n, fmt):
    t = api.texture3d((n, n, n), fmt)
    bpp = {"rgba16f": 8, "r32f": 4, "r16f": 2}[fmt]
    t.write(bytes(n * n * n * bpp))
    return t

def setup(api):
    w = {}
    w["vel"] = [_zeroed(api, FW_N, "rgba16f") for _ in range(2)]
    w["vtmp"] = _zeroed(api, FW_N, "rgba16f")
    w["wall"] = _zeroed(api, FW_N, "rgba16f")
    w["sizes"] = [FW_N >> l for l in range(FW_LEVELS)]
    w["q"] = [_zeroed(api, n, "r32f") for n in w["sizes"]]
    w["b"] = [_zeroed(api, n, "r32f") for n in w["sizes"]]
    w["m"] = [_zeroed(api, n, "r16f") for n in w["sizes"]]
    w["k"] = {name: api.kernel("worms.comp", name) for name in
              ("MASK", "ADV_FWD", "ADV_CORR", "FORCE", "DIV", "MARK", "SMOOTH",
               "RESTRICT", "PROLONG", "PROJECT", "WORM", "CLEAR", "PART", "VISC")}
    w["cur"] = 0
    w["marked0"] = False
    w["st"] = api.texture2d((64, 1), "r32f", nearest=True)
    w["st"].write(bytes(64 * 4))
    w["worm"] = api.texture2d((6, 1), "rgba32f", nearest=True)
    w["worm"].write(bytes(6 * 16))
    init = np.zeros((FW_PH, FW_PW, 4), dtype="f4")
    init[..., 3] = -1.0
    w["part"] = api.texture2d((FW_PW, FW_PH), "rgba32f", data=init.tobytes(), nearest=True)
    w["pcol"] = api.texture2d((FW_PW, FW_PH), "r32f", nearest=True)
    w["pcol"].write(bytes(FW_PW * FW_PH * 4))
    cw, ch = _canvas()
    def _acc():
        t = api.ctx.texture((cw, ch), 1, dtype="u4")
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return t
    w["glowA"] = _acc()
    w["glowB"] = _acc()
    w["glowR"] = _acc()
    w["emit"] = 0
    api.state.update(w)
    api.expose("u_glowA", w["glowA"])
    api.expose("u_glowB", w["glowB"])
    api.expose("u_glowR", w["glowR"])
    api.expose("u_worm", w["worm"])

def _sweeps(api, s, lvl, sweeps, wall_only):
    n = s["sizes"][lvl]
    for _ in range(sweeps):
        for parity in (0, 1):
            api.run(s["k"]["SMOOTH"], _g3(n),
                    images={0: s["q"][lvl], 1: s["b"][lvl], 2: s["m"][lvl]},
                    uniforms={"u_n": n, "u_parity": parity, "u_wallOnly": int(wall_only),
                              "u_eps": FW_EPS * 4 ** lvl})

def _smooth(api, s, lvl, sweeps):
    _sweeps(api, s, lvl, FW_WALL, True)
    _sweeps(api, s, lvl, sweeps, False)
    _sweeps(api, s, lvl, FW_WALL, True)

def _vcycle(api, s, lvl=0):
    if lvl > 0 or not s["marked0"]:
        api.run(s["k"]["MARK"], _g3(s["sizes"][lvl]), images={0: s["m"][lvl]}, uniforms={"u_n": s["sizes"][lvl]})
        s["marked0"] = True
    if lvl == FW_LEVELS - 1:
        _smooth(api, s, lvl, FW_COARSE)
        return
    _smooth(api, s, lvl, FW_PRE)
    nc = s["sizes"][lvl + 1]
    api.run(s["k"]["RESTRICT"], _g3(nc),
            images={0: s["q"][lvl], 1: s["b"][lvl], 2: s["m"][lvl],
                    3: s["q"][lvl + 1], 4: s["b"][lvl + 1], 5: s["m"][lvl + 1]},
            uniforms={"u_n": nc, "u_eps": FW_EPS * 4 ** lvl})
    _vcycle(api, s, lvl + 1)
    api.run(s["k"]["PROLONG"], _g3(s["sizes"][lvl]),
            images={0: s["q"][lvl], 1: s["m"][lvl], 2: s["q"][lvl + 1], 3: s["m"][lvl + 1]},
            uniforms={"u_n": s["sizes"][lvl]})
    _smooth(api, s, lvl, FW_POST)

def step(api, dt):
    s = api.state
    k = s["k"]
    dt = min(max(dt, 1e-4), 0.05)
    for prog in k.values():
        api.set(prog, {"u_H": FW_H, "u_dt": dt, "u_n": FW_N})
    c = s["cur"]
    vel0, vel1 = s["vel"][c], s["vel"][1 - c]
    g = _g3(FW_N)

    api.run(k["WORM"], (1, 1, 1), images={0: s["st"], 1: s["worm"]})
    api.run(k["MASK"], g, images={0: s["wall"], 1: s["m"][0], 2: vel0})
    api.run(k["ADV_FWD"], g, images={0: s["vtmp"]}, samplers={"u_src": vel0, "u_vel": vel0})
    api.run(k["ADV_CORR"], g, images={0: vel1, 1: s["m"][0]},
            samplers={"u_src": vel0, "u_fwd": s["vtmp"], "u_vel": vel0})
    api.run(k["FORCE"], g, images={0: vel1}, samplers={"u_worm": s["worm"]})
    for _ in range(FW_VISC_PASSES // 2):
        api.run(k["VISC"], g, images={0: s["vtmp"]}, samplers={"u_src": vel1})
        api.run(k["VISC"], g, images={0: vel1}, samplers={"u_src": s["vtmp"]})
    api.run(k["DIV"], g, images={0: s["b"][0], 1: vel1, 2: s["wall"]})
    s["marked0"] = False
    _vcycle(api, s)
    api.run(k["PROJECT"], g, images={0: vel1, 1: s["q"][0], 2: s["wall"]})
    s["cur"] = 1 - c

    cw, ch = s["canvas"] if "canvas" in s else _canvas()
    s["canvas"] = (cw, ch)
    api.run(k["CLEAR"], ((cw + 15) // 16, (ch + 15) // 16, 1),
            images={0: s["glowA"], 1: s["glowB"], 2: s["glowR"]},
            uniforms={"u_canvas": (cw, ch)})
    wp = np.frombuffer(s["worm"].read(), dtype="f4").reshape(6, 4)
    ns = []
    for i in range(2):
        d = float(np.linalg.norm(wp[i, :3] - wp[2 + i, :3])) if wp[i, 3] > 0.5 and wp[2 + i, 3] > 0.5 else 0.0
        rate = min(max(FW_RATE, FW_DENS * d / max(dt, 1e-4)), FW_RATE_MAX)
        ns.append(max(1, int(round(rate * dt))))
    api.run(k["PART"], ((FW_PW + 15) // 16, (FW_PH + 15) // 16, 1),
            images={0: s["part"], 1: s["glowA"], 2: s["glowB"], 3: s["glowR"], 4: s["pcol"]},
            samplers={"u_vel": vel1, "u_worm": s["worm"]},
            uniforms={"u_pw": FW_PW, "u_ph": FW_PH, "u_np": FW_NP, "u_emitStart": s["emit"],
                      "u_emitN": ns[0], "u_emitN1": ns[1], "u_canvas": (cw, ch)})
    s["emit"] = (s["emit"] + ns[0] + ns[1]) % FW_NP
    api.expose("u_glowA", s["glowA"])
    api.expose("u_glowB", s["glowB"])
    api.expose("u_glowR", s["glowR"])
