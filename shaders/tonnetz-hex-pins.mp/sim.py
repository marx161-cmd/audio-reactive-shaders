"""tonnetz_hexpins.mp's compute stage (renderer.py ComputeStage, scope.md
2026-10-08 "tonnetz_hexpins.mp: GO"). Owns the hex pins: one rgba32f texel
per pin in axial coordinates (height, velocity, surface), ping-ponged.
Kernels are in sim.comp. PIN_S / AX_M are handed to the render pass too.
"""

import math

PIN_S = 0.009
EXT = 1.20
SUBSTEPS = 3
DT_MAX = 0.05
AX_M = math.ceil(EXT / (1.5 * PIN_S)) + math.ceil(EXT / (math.sqrt(3) * PIN_S)) + 2
W = 2 * AX_M + 1
NB = 120
NB_USED = 108

def _groups(n):
    g = max(1, (n + 7) // 8)
    return (g, g, 1)

def setup(api):
    s = api.state
    s["st"] = []
    for _ in range(2):
        t = api.texture2d((W, W), "rgba32f", nearest=True)
        t.write(bytes(W * W * 16))
        s["st"].append(t)
    s["k"] = api.kernel("sim.comp", "STEP")
    s["cur"] = 0
    api.expose("u_pins", s["st"][0])
    s["unbleed"] = api.kernel("sim.comp", "UNBLEED")
    import band_filters
    leak = band_filters.leak_matrix(NB_USED, 12).astype("f4")
    s["leak"] = api.texture2d((25, NB_USED), "r32f", data=leak.tobytes(), nearest=True)
    print(f"tonnetz_hexpins: leak model from filter orders {band_filters.read_orders()}, "
          f"+1 semitone {10 * __import__('math').log10(max(float(leak[60, 13]), 1e-30)):.1f} dB at band 60")
    s["clean"] = api.texture2d((NB, 1), "r32f", nearest=True)
    s["clean"].write(bytes(NB * 4))
    s["unbleed_x"] = api.texture2d((NB, 1), "r32f", nearest=True)
    s["unbleed_x"].write(bytes(NB * 4))
    s["chord"] = api.texture2d((16, 1), "r32f", nearest=True)
    s["chord"].write(bytes(16 * 4))
    s["ring"] = api.texture2d((20, 1), "r32f", nearest=True)
    s["ring"].write(bytes(20 * 4))
    s["fairy"] = api.texture2d((257, 2), "rgba32f", nearest=True)
    s["fairy"].write(bytes(257 * 2 * 16))
    s["mv"] = api.texture2d((8, 1), "r32f", nearest=True)
    s["mv"].write(bytes(8 * 4))
    api.expose("u_chord", s["chord"])
    api.expose("u_clean", s["clean"])

def step(api, dt):
    s = api.state
    dt = min(max(dt, 1e-4), DT_MAX)
    sub = dt / SUBSTEPS
    api.run(s["unbleed"], (1, 1, 1), images={0: s["clean"], 1: s["unbleed_x"], 2: s["chord"], 3: s["ring"], 4: s["fairy"], 5: s["mv"]},
            samplers={"u_leak": s["leak"]}, uniforms={"u_nb": NB_USED})
    api.expose("u_clean", s["clean"])
    A = api.buffer("A")
    for _ in range(SUBSTEPS):
        c = s["cur"]
        api.run(s["k"], _groups(W), images={0: s["st"][1 - c]},
                samplers={"u_prev": s["st"][c], "u_field": A, "u_beatring": s["ring"]},
                uniforms={"u_w": W, "u_axM": AX_M, "u_pinS": PIN_S, "u_dt": sub})
        s["cur"] = 1 - c
    api.expose("u_pins", s["st"][s["cur"]])
    api.expose("u_chord", s["chord"])
    api.expose("u_fairyring", s["ring"])
    api.expose("u_fairyhist", s["fairy"])
    api.render_uniform("u_pinS", PIN_S)
    api.render_uniform("u_axM", float(AX_M))
