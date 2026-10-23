"""Held-tone detector (scope.md 2026-10-11 "held tones"): a clear, long-held note -- a belted
vocal line, a sustained lead -- which the beat/onset/loudness features never see (no hits,
steady level).

Fed the stem splitter's TONAL row (120 semitone bands, norm_db 0..1 = -60..0 dB, band b
centred on 27.5 * 2^(b/12) Hz) at the renderer's 60 Hz stem cadence. A held tone =
  * one band in the singing range (~150 Hz .. 1.5 kHz) standing clearly above its
    neighbourhood (prominence over the median of +-PROM_HALF bands), and
  * the same band (+-1 for vibrato) staying the peak for HOLD_S.
Outputs (shader uniforms, every shader gets them):
  u_held        0..1  how strongly a clear note is being held right now
  u_held_pitch  the held note as a band index (fractional; 0 = A0, 12 per octave), -1 if none
Changes of state are logged to ~/.cache/shader-backdrop/held.log, to check against what
is heard before anything visual depends on it.

2026-10-20 v2 (user: "nothing reacts adequately" -- the Prince solo in While My Guitar Gently Weeps found NOTHING): v1 pinned
ONE band (+-1) -> every bend / wide vibrato reset the hold (offline: 27 holds in 6.5 min, none >= 0.5 s). v2 TRACKS the note:
the most salient tonal peak (prominence + 0.5 x its octave's), followed within +-TRACK_W bands while it glides, gaps < GAP_S
bridged, and it must DOMINATE every other peak (not its own harmonics) -- one line, not a chord; the note's quality smoothed
over Q_TAU so one weak frame never breaks it. Offline on the stem splitter's tonal row (tools: job tmp held_eval2.py): held
10-20 % of the time, Kite's long notes 2-5.5 s. Validate live against what is heard.
"""
import math
import os
import time

import numpy as np

B_LO, B_HI = 29, 70
PROM_HALF = 4
PROM0, PROM1 = 0.08, 0.18
LEVEL0, LEVEL1 = 0.40, 0.60   # (unused since the relative level below; kept for reference)
REL0, REL1 = 0.25, 0.12      # the note within this far (norm_db: 15 .. 7 dB) under the song's recent loudest tonal peak ...
FLOOR0, FLOOR1 = 0.18, 0.26  # ... and above this absolute floor (silence never counts) -- 2026-10-20: quiet ballads sat under 0.40
REF_TAU = 8.0                # s: how fast that recent-loudest reference falls
HARM_K = 0.5          # weight of the octave-up (2nd harmonic) prominence in the note's salience
DOM0, DOM1 = 0.04, 0.10   # the note's salience over the strongest OTHER peak (not its octaves): one line, not a chord
TRACK_W = 3           # bands the tracked note may move within (a bend / vibrato is the same note)
SWITCH_S = 0.10       # s: another peak DOMINATING the tracked note this long -> the melody moved there: drop it (2026-10-20 live: the
                      #   tracker latched onto a low accompaniment note while the voice sang above it)
GAP_S = 0.12          # a dip shorter than this does not end the note
HOLD0, HOLD1 = 0.30, 0.80
RELEASE_S = 0.20
Q_TAU = 0.15          # s: the note's quality (prominence, level, dominance) is smoothed over this -- one weak frame never breaks a note



DEBUG_S = 2.0         # every this many s a "dbg" line in held.log (0 = off)
LOG_PATH = os.path.expanduser("~/.cache/shader-backdrop/held.log")
NOTE = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def _ss(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def note_name(b):
    b = int(round(b))
    return f"{NOTE[b % 12]}{(b + 9) // 12}"


class HeldTone:
    def __init__(self):
        self.pitch, self.since, self.gap, self.out, self.t_prev = -1.0, 0.0, 0.0, 0.0, None
        self.lvl = 0.0
        self.p = 0.0
        self.d = 0.0
        self.q = 0.0
        self.on, self.on_t = False, 0.0
        self.dbg_t = 0.0
        self.lose = 0.0
        self.ref = 0.3

    def update(self, tonal, now=None):
        now = time.monotonic() if now is None else now
        dt = 0.0 if self.t_prev is None else min(max(now - self.t_prev, 0.0), 0.1)
        self.t_prev = now
        x = np.asarray(tonal, dtype=np.float64)
        lo, hi = B_LO - PROM_HALF, B_HI + 12 + PROM_HALF
        seg = x[lo:hi].copy()
        seg[:PROM_HALF] = x[B_LO]                       # the sub-bass is in the BASS stem (zeros here): edge-replicate it, or every low
                                                         # note would look hugely prominent against an empty neighbourhood (2026-10-20 live)
        base = np.median(np.lib.stride_tricks.sliding_window_view(seg, 2 * PROM_HALF + 1), axis=1)
        prom_all = seg[PROM_HALF:len(seg) - PROM_HALF] - base          # bands B_LO .. B_HI+12-1
        n = B_HI - B_LO
        prom = prom_all[:n]
        sal = prom + HARM_K * np.clip(prom_all[12:12 + n], 0.0, None)  # + the octave above
        vals = x[B_LO:B_HI]
        kr = 1.0 - math.exp(-dt / REF_TAU) if dt > 0 else 0.0
        self.ref = max(float(vals.max()), self.ref - (self.ref - float(vals.max())) * kr) if vals.max() > self.ref else self.ref + (float(vals.max()) - self.ref) * kr
        lv0 = max(self.ref - REL0, FLOOR0)
        ok = (prom >= PROM0) & (vals >= lv0)
        def dom(j):                               # how far j stands above every other peak (its own octaves excluded)
            m = np.ones(n, bool); m[max(0, j - 2):j + 3] = False
            for o in (-12, 12, 19, 24):
                if 0 <= j + o < n: m[max(0, j + o - 1):j + o + 2] = False
            return float(sal[j] - (sal[m].max() if m.any() else 0.0))
        if self.pitch >= 0:                       # follow the note we have, if it is still there (it may glide)
            c = int(round(self.pitch)) - B_LO
            w0, w1 = max(0, c - TRACK_W), min(n, c + TRACK_W + 1)
            cand = [j for j in range(w0, w1) if ok[j]]
            if cand:
                j = max(cand, key=lambda q: sal[q])
                self.pitch += 0.5 * ((B_LO + j) - self.pitch)
                self.since += dt; self.gap = 0.0
                self.p, self.lvl = float(prom[j]), float(vals[j]); self.d = dom(j)
                self.lose = self.lose + dt if self.d < 0.0 else 0.0
                if self.lose > SWITCH_S:
                    self.pitch, self.since, self.lose = -1.0, 0.0, 0.0
            else:
                self.gap += dt
                if self.gap > GAP_S:
                    self.pitch, self.since = -1.0, 0.0
        if self.pitch < 0 and ok.any():           # a new note: the most salient one
            j = int(np.argmax(np.where(ok, sal, -1e9)))
            self.pitch, self.since, self.gap = float(B_LO + j), 0.0, 0.0
            self.q = 0.0
            self.p, self.lvl = float(prom[j]), float(vals[j]); self.d = dom(j)
        k = 1.0 - math.exp(-dt / Q_TAU) if dt > 0 else 1.0
        self.q += (_ss(PROM0, PROM1, self.p) * _ss(self.ref - REL0, self.ref - REL1, self.lvl) * _ss(FLOOR0, FLOOR1, self.lvl) * _ss(DOM0, DOM1, self.d) - self.q) * k
        target = (_ss(HOLD0, HOLD1, self.since) * self.q
                  if self.pitch >= 0 and self.gap == 0.0 else (self.out if self.pitch >= 0 else 0.0))
        self.out = target if target > self.out else self.out * math.exp(-dt / RELEASE_S)
        self.pitch_log = self.pitch
        self._log(now)
        if DEBUG_S > 0 and now - self.dbg_t >= DEBUG_S:     # (2026-10-20: live tuning -- what the detector sees)
            self.dbg_t = now
            j = int(np.argmax(sal)); self._write(f"dbg  top {B_LO + j:3d} prom {prom[j]:+.3f} lvl {vals[j]:.3f} | tracked {self.pitch:5.1f}"
                                                  f" p {self.p:+.3f} lvl {self.lvl:.3f} dom {self.d:+.3f} since {self.since:4.2f} q {self.q:.2f} out {self.out:.2f}"
                                                  f" | row max {x.max():.3f} ref {self.ref:.3f}")
        return {"u_held": float(self.out), "u_held_pitch": float(self.pitch if self.out > 0.05 else -1.0)}

    def _log(self, now):
        if not self.on and self.out > 0.5:
            self.on, self.on_t = True, now
            self._write(f"held  {note_name(self.pitch):4s} ({27.5 * 2 ** (self.pitch / 12):6.1f} Hz)")
        elif self.on and self.out < 0.2:
            self.on = False
            self._write(f"  end after {now - self.on_t:4.1f} s")

    def _write(self, msg):
        try:
            with open(LOG_PATH, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        except OSError:
            pass
