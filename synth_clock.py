"""Synthetic motion clock -- host-side fallback when audio is not music
(scope.md 2026-10-13; v2 "synthetic song structure" same day).

Fills the SAME scalar interface the cache path publishes (u_beat_*, u_kick /
u_drum_*, and the full u_ms_* set incl. look-ahead) from a synthetic SONG:
cycles of 256 beats, each a FORM of sections (calm / groove / build / break /
drop) with their own kick patterns, stable section values, a tempo arc and
drops the shaders can see coming. Motion-only: it never touches bands/u_rms.

Stateless by design: everything is a pure function of (count, origin, seed).
The caller (music_state) owns the one continuous clock: it advances the count at
bpm_at(count - origin, seed) / 60 per second, and on each switch INTO synthetic
picks origin = the current count rounded to a beat (so the song starts at a
section start) and a new seed (so every documentary gets a different song).

All tunables are in THE SONG block below.
"""

import math

# ==== THE SONG (the one editable block) =====================================
CYCLE = 256                  # beats per form (every form's sections sum to this)
PULSE_RATE = 5.0             # hit decay per BEAT: 1/e after 0.2 beat, every hit equally tight
GLIDE_BEATS = 16.0           # tempo glide from the previous cycle's base at a cycle start
BREAK_GLIDE = 8.0            # drop tempo -> base tempo over the first beats of a break

# section base values: energy, weight, density, brightness, pulse, voice, tension, impact
# (BUILD ramps from its first row to its second over the section)
BASE = {
    "calm":   (0.20, 0.25, 0.15, 0.55, 1.0, 0.0, 0.10, 0.00),
    "groove": (0.62, 0.45, 0.45, 0.50, 1.0, 0.0, 0.30, 0.20),
    "build":  ((0.50, 0.45, 0.45, 0.55, 1.0, 0.0, 0.40, 0.10),
               (0.90, 0.60, 0.90, 0.75, 1.0, 0.0, 1.00, 0.30)),
    "break":  (0.28, 0.20, 0.20, 0.70, 1.0, 0.0, 0.55, 0.00),
    "drop":   (0.95, 0.55, 0.70, 0.65, 1.0, 0.0, 0.45, 0.85),   # weight 0.55: bar accents full, the rest ~half
}
TYPES = ("calm", "groove", "build", "break", "drop")

# forms: (name, base bpm lo, hi, drop bpm = base + this (or absolute if >= 100), sections)
# every section list sums to CYCLE. A section may carry its own drop bpm: ("drop", beats, bpm).
# "rave": the first drop is fast but under the 150 bpm FRENZY threshold; only the second, short
# drop (164) tips into frenzy -- the climax, not the default (user 2026-10-13: "frenzy too often").
FORMS = [
    ("intro", 112, 120, 10, [("calm", 32), ("groove", 32), ("build", 16), ("drop", 48), ("break", 16),
                             ("groove", 32), ("build", 16), ("drop", 48), ("calm", 16)]),
    ("club", 118, 128, 6, [("groove", 32), ("build", 16), ("drop", 32), ("break", 16), ("groove", 32),
                           ("build", 16), ("drop", 48), ("break", 16), ("groove", 32), ("calm", 16)]),
    ("slowburn", 96, 110, 22, [("calm", 32), ("groove", 48), ("build", 32), ("drop", 64), ("break", 32),
                               ("groove", 48)]),
    ("stutter", 124, 136, 4, [("groove", 16), ("break", 8), ("groove", 16), ("build", 8), ("drop", 32),
                              ("groove", 24), ("break", 16), ("build", 8), ("drop", 32), ("break", 8),
                              ("build", 8), ("drop", 48), ("calm", 32)]),
    ("rave", 140, 146, 148, [("groove", 32), ("build", 16), ("drop", 48), ("break", 16), ("groove", 32),
                             ("build", 16), ("drop", 32, 164), ("break", 16), ("groove", 32), ("calm", 16)]),
]
DENSITY_RAVE = 0.80          # rave drops run denser (FRENZY also needs density)

# patterns: per voice a list of (length_beats, [(hit_beat, amp), ...]) layers, beats from the
# section start. Several variants per section type; a section picks one by hash.
# Calibrated against real music (2026-10-13 captures): the live kick detector passes ~0.5 kick
# EDGES/s (u_kick crossing 0.6 = ember_flock's accent + hawk trigger). So only the ACCENTED hits
# cross 0.6; the rest are GHOST hits at <= 0.55: they still drive the continuous kick push /
# wobble, but don't fire an accent each. (v2 had every hit at ~1.0 -> 2.7-3.7 edges/s = explosions.)
GHOST = 0.50
PATTERNS = {
    "calm": [
        {"kick": [(8, [(0, 0.75)])], "mid": [], "high": [(1, [(0.5, 0.30)])]},
        {"kick": [(8, [(0, 0.75), (5.5, GHOST)])], "mid": [(8, [(4, 0.45)])], "high": [(2, [(1, 0.30)])]},
    ],
    "groove": [
        # four-on-floor, accent on the bar
        {"kick": [(4, [(0, 0.95), (1, GHOST), (2, GHOST), (3, GHOST)])], "mid": [(2, [(1, 0.80)])],
         "high": [(1, [(0.5, 0.50)])]},
        # 4:3 polyrhythm: the 4 accents, the 3 ghosts against it
        {"kick": [(4, [(0, 1.00)]), (3, [(0, GHOST)])], "mid": [(8, [(4, 0.85)])],
         "high": [(1, [(0, 0.40)]), (7, [(0, 0.70)])]},
        # broken beat
        {"kick": [(4, [(0, 1.00), (1.5, 0.55), (2.75, GHOST)])], "mid": [(4, [(1, 0.85), (3, 0.85), (3.5, 0.40)])],
         "high": [(0.5, [(0, 0.35)])]},
        # halftime
        {"kick": [(4, [(0, 1.00), (2.5, GHOST)])], "mid": [(4, [(2, 1.00)])], "high": [(1, [(0, 0.40), (0.5, 0.30)])]},
        # odd 7: two accents per 7
        {"kick": [(7, [(0, 1.00), (2, GHOST), (4, 0.80), (5.5, GHOST)])], "mid": [(7, [(1, 0.70), (3, 0.70), (6, 0.90)])],
         "high": [(0.5, [(0, 0.30)])]},
    ],
    "break": [
        {"kick": [], "mid": [(16, [(0, 0.60)])], "high": [(2, [(1, 0.25)])]},
        {"kick": [(16, [(0, 0.80)])], "mid": [], "high": [(1, [(0.5, 0.20)])]},
    ],
    "drop": [
        # four-on-floor: full accent on 1, half accent on 3, ghosts between
        {"kick": [(4, [(0, 1.00), (1, GHOST), (2, 0.80), (3, GHOST)])], "mid": [(2, [(1, 1.00)])],
         "high": [(0.25, [(0, 0.45)])]},
        # four-on-floor ghosts + a 3-beat accent cycling against the bar
        {"kick": [(4, [(0, 1.00), (1, GHOST), (2, GHOST), (3, GHOST)]), (6, [(3.5, 0.80)])],
         "mid": [(2, [(1, 1.00)])], "high": [(0.5, [(0, 0.50)])]},
        # breakbeat drop
        {"kick": [(2, [(0, 1.00), (0.75, 0.55), (1.5, GHOST)])], "mid": [(2, [(1, 1.00), (1.75, 0.50)])],
         "high": [(0.25, [(0, 0.40)])]},
    ],
}
FILL_EVERY = 8               # groove/drop: a snare fill over the last beat of every 8-beat phrase
# ============================================================================

AXES = ("energy", "weight", "density", "brightness", "pulse", "voice", "tension", "impact")


def _h(*vals):
    """Deterministic hash of ints -> [0, 1)."""
    x = 2166136261
    for v in vals:
        x = ((x ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0x5BD1E995) & 0xFFFFFFFF
    x ^= x >> 15
    return x / 4294967296.0


def _cycle(seed, c):
    """-> (form name, base bpm, drop bpm, [(type, start, length, variant)])."""
    if c == 0:
        fi = 0                                            # every synthetic song opens with the intro
    else:
        fi = 1 + int(_h(seed, c, 1) * (len(FORMS) - 1))
    name, lo, hi, dr, secs = FORMS[fi]
    base = lo + (hi - lo) * _h(seed, c, 2)
    drop = dr if dr >= 100 else base + dr
    # tempo target per section: a drop plays at its own (or the form's) drop bpm, a build runs
    # up to the NEXT drop's, a break glides down from the section before it, the rest = base
    dbpm = [(sec[2] if len(sec) > 2 else drop) if sec[0] == "drop" else None for sec in secs]
    out, s = [], 0
    for i, sec in enumerate(secs):
        typ, n = sec[0], sec[1]
        var = int(_h(seed, c, 100 + i) * len(PATTERNS.get(typ, [None])))
        if typ == "drop":
            tgt = dbpm[i]
        elif typ == "build":
            tgt = next((x for x in dbpm[i + 1:] if x is not None), drop)
        elif typ == "break" and out:
            tgt = out[-1][4] if out[-1][0] in ("drop", "build") else base
        else:
            tgt = base
        out.append((typ, s, n, var, tgt))
        s += n
    return name, base, drop, out


def _locate(pos, seed):
    c = int(math.floor(pos / CYCLE))
    pc = pos - c * CYCLE
    name, base, drop, secs = _cycle(seed, c)
    for i, (typ, s, n, var, tgt) in enumerate(secs):
        if pc < s + n or i == len(secs) - 1:
            return c, pc, name, base, drop, secs, i
    raise AssertionError


def _bpm(pc, base, tgt, typ, s, n):
    into = pc - s
    if typ == "build":
        return base + (tgt - base) * (into / n)
    if typ == "drop":
        return tgt
    if typ == "break":
        t = min(into / BREAK_GLIDE, 1.0)
        return tgt + (base - tgt) * t * t * (3.0 - 2.0 * t)
    return base


def bpm_at(pos, seed=0):
    """Tempo at pos beats into the synthetic song (the rate the caller advances the count at)."""
    c, pc, name, base, drop, secs, i = _locate(pos, seed)
    typ, s, n, var, tgt = secs[i]
    bpm = _bpm(pc, base, tgt, typ, s, n)
    if c > 0 and pc < GLIDE_BEATS:                         # glide in from the previous cycle's base
        prev_base = _cycle(seed, c - 1)[1]
        t = pc / GLIDE_BEATS
        bpm = prev_base + (bpm - prev_base) * t * t * (3.0 - 2.0 * t)
    return bpm


def _voice(layers, into):
    v = 0.0
    for length, hits in layers:
        ph = into % length
        best, amp = None, 0.0
        for hb, a in hits:
            d = (ph - hb) % length
            if best is None or d < best:
                best, amp = d, a
        if best is not None:
            v += amp * math.exp(-PULSE_RATE * best)
    return v


def _roll(into, n, amp0, amp1, accent=True):
    """BUILD roll: quarter notes, then 8ths, then 16ths; the last beat is a silent gap.
    Only the bar starts are accents (rising amp0 -> amp1); the roll in between are ghosts,
    so a build tightens visibly without an accent on every 16th."""
    u = into / n
    if into >= n - 1.0:
        return 0.0
    step = 1.0 if u < 0.5 else (0.5 if u < 0.75 else 0.25)
    bar_hit = accent and (into % 4.0) < step
    amp = (amp0 + (amp1 - amp0) * u) if bar_hit or not accent else GHOST * (0.8 + 0.2 * u)
    return amp * math.exp(-PULSE_RATE * (into % step))


def _base(typ, u, rave):
    b = BASE[typ]
    if typ == "build":
        b = tuple(x + (y - x) * u for x, y in zip(b[0], b[1]))
    if typ == "drop" and rave:
        b = b[:2] + (DENSITY_RAVE,) + b[3:]
    return b


def _clip(v):
    return min(max(v, 0.0), 1.0)


def uniforms_at(count, origin=0.0, seed=0):
    """Pure function -> the full synthetic uniform set (u_beat_*, drums, u_ms_*)."""
    b = float(count)
    pos = max(b - origin, 0.0)
    c, pc, name, base_bpm, drop_bpm, secs, i = _locate(pos, seed)
    typ, s, n, var, tgt = secs[i]
    into = pc - s
    u = into / n
    rave = name == "rave"
    bpm = bpm_at(pos, seed)
    spb = 60.0 / bpm

    # drums
    if typ == "build":
        kick = _roll(into, n, 0.70, 1.00)
        mid = _roll(into, n, 0.40, 1.00, accent=False)
        high = _roll(into, n, 0.20, 0.60, accent=False)
    else:
        pat = PATTERNS[typ][var]
        kick = _voice(pat["kick"], into)
        mid = _voice(pat["mid"], into)
        high = _voice(pat["high"], into)
        if typ in ("groove", "drop") and (into % FILL_EVERY) >= FILL_EVERY - 1.0:
            mid = max(mid, (0.55 + 0.45 * ((into % 1.0))) * math.exp(-PULSE_RATE * (into % 0.25)))
    if typ == "drop":                                       # the drop hit itself
        kick = max(kick, math.exp(-0.8 * into))

    # section values: stable base (sec_*) + small per-beat motion (cur_*)
    sec = _base(typ, 0.5 if typ == "build" else 0.0, rave)
    cur = list(_base(typ, u, rave))
    cur[0] += 0.06 * math.sin(2.0 * math.pi * pos / 8.0)
    cur[1] += 0.10 * math.sin(math.pi * ((pos / 4.0) % 1.0)) - 0.05
    cur[2] += 0.12 * kick
    cur[3] += 0.08 * math.sin(2.0 * math.pi * pos / 5.0)
    if typ == "drop":
        cur[7] = max(cur[7] * 0.7, math.exp(-into / 4.0))  # impact boom at the drop
    cur = [_clip(x) for x in cur]
    sec = [_clip(x) for x in sec]

    # look-ahead (same units as the cache path: next_in in BEATS, drop_in in SECONDS)
    if i + 1 < len(secs):
        ntyp, ns = secs[i + 1][0], secs[i + 1][1]
        nxt_rave = rave
    else:
        nname, _, _, nsecs = _cycle(seed, c + 1)
        ntyp, ns = nsecs[0][0], CYCLE
        nxt_rave = nname == "rave"
    nxt = [_clip(x) for x in _base(ntyp, 0.5 if ntyp == "build" else 0.0, nxt_rave)]
    next_in = ns - pc
    drop_beats = None
    for (t2, s2, n2, v2, b2) in secs[i + 1:]:
        if t2 == "drop":
            drop_beats = s2 - pc
            break
    if drop_beats is None:
        for (t2, s2, n2, v2, b2) in _cycle(seed, c + 1)[3]:
            if t2 == "drop":
                drop_beats = CYCLE + s2 - pc
                break
    drop_in = drop_beats * spb if drop_beats is not None else -1.0

    return {
        "u_beat_count": b,
        "u_beat_phase": b % 1.0,
        "u_beat_conf": 1.0,
        "u_beat_bpm": float(bpm),
        "u_kick": _clip(kick),
        "u_drum_mid": _clip(mid),
        "u_drum_high": _clip(high),
        "u_ms_source": 1.0,                                # a "known song": shaders trust the look-ahead
        "u_ms_cur_a": tuple(cur[:4]), "u_ms_cur_b": tuple(cur[4:]),
        "u_ms_sec_a": tuple(sec[:4]), "u_ms_sec_b": tuple(sec[4:]),
        "u_ms_next_a": tuple(nxt[:4]), "u_ms_next_b": tuple(nxt[4:]),
        "u_ms_change": float(c * 32 + i),
        "u_ms_change_age": into * spb,
        "u_ms_next_in": float(next_in),
        "u_ms_drop_in": float(drop_in),
        "u_ms_bar_phase": (into % 4.0) / 4.0,
        "u_ms_repeat": float(TYPES.index(typ) * 8 + var),
        "u_ms_song_pos": pos * spb,
        # debug readout (not a uniform)
        "synth_section": f"{name}:{typ}{var}",
    }


def _main():
    """Print the first two cycles of a song: one line per 2 beats."""
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    pos, last = 0.0, None
    print(f"{'beat':>6} {'section':>16} {'bpm':>5} {'kick':>4} {'mid':>4} {'nrg':>4} {'den':>4} "
          f"{'next':>5} {'drop_s':>6}")
    while pos < 2 * CYCLE:
        u = uniforms_at(pos, 0.0, seed)
        if u["synth_section"] != last or int(pos) % 8 == 0:
            print(f"{pos:6.1f} {u['synth_section']:>16} {u['u_beat_bpm']:5.1f} {u['u_kick']:4.2f} "
                  f"{u['u_drum_mid']:4.2f} {u['u_ms_cur_a'][0]:4.2f} {u['u_ms_cur_a'][2]:4.2f} "
                  f"{u['u_ms_next_in']:5.1f} {u['u_ms_drop_in']:6.1f}")
            last = u["synth_section"]
        pos += 2.0


if __name__ == "__main__":
    _main()
