"""FLOCK TUNING (scope.md 2026-10-20 "FLOCK TUNING"): a live parameter search while ember_merge runs.

The user: "train it on the AMD box whenever the swarm runs ... then downscale to the phone, we don't need to retrain".
Objective: "a combination of area covered for speed, clearly shaped formations, and a penalty for repetitive behaviour".

18 steering / picker knobs are live uniforms on the desktop (u_tune[i], 0 = the compile-time default; the knob list is the
#ifdef DESKTOP block per knob in tools/emberbox_src). A separable CMA-ES-style search runs in LOG-RATIO space around the
defaults (x = 0 -> the shipped value; |x| <= ln 2: never more than x2 / x0.5), one candidate per TRIAL_S of live, audible
music, scored from 0.25 s readbacks:
  coverage  = the share of the box (8 x 16 x 12 cells) the embers swept during the trial / (0.5 + their mean speed)
  clarity   = exp(-embers' mean distance to their nearest leader / 0.12) x exp(-leaders' mean slot error / 0.15)
  repetition= the share of 0.5 s flock states (centroids + spreads) that came back (within REP_EPS) from > 8 s earlier
  score     = W_COV * coverage + W_CLR * clarity - W_REP * repetition
Each generation re-tests the current mean first (same music conditions as its rivals). Everything persists in
~/.cache/shader-backdrop/flock_tune.json; 'tune:' lines go to renderer.log. Pause: the move lab has a number set, or the
file ~/.cache/shader-backdrop/flock_tune_off exists, or rec-pixel is recording (then the knobs are the shipped defaults; the
interrupted candidate restarts its window afterwards).
"""
import json
import math
import os
import sys
import time

import numpy as np

KNOBS = [("LEAD_PERIOD", 0.35), ("LEAD_DAMP", 0.75), ("LEAD_FF", 1.0), ("LEAD_VMAX", 2.5), ("LEAD_SOFT", 0.18),
         ("LEAD_SPREAD", 0.10), ("W_TGT", 2.5), ("TRACK_K", 4.0), ("CATCH_K", 2.0), ("SEP_R0", 0.090), ("W_SEP", 5.0),
         ("W_ALIGN1", 3.0), ("W_COH1", 0.6), ("JIT_K", 1.2), ("AIR_Q", 2.5), ("PICK_HASH", 0.70), ("BODY_HOLD_MIN", 12.0),
         ("FAR_GAP0", 10.0)]
N = len(KNOBS)
STATE = os.path.expanduser("~/.cache/shader-backdrop/flock_tune.json")
OFF_FILE = os.path.expanduser("~/.cache/shader-backdrop/flock_tune_off")
HOLD_FILE = os.path.expanduser("~/.cache/shader-backdrop/flock_tune_hold")
SHIPPED_HOLD = "[0.175772, 0.190355, -0.262576, 0.440268, -0.073076, -0.101626, 0.133446, 0.692401, 0.017464, 0.504365, -0.636902, -0.193003, -0.538231, 0.185822, -0.606331, 0.465235, -0.081312, 0.640184]"
REC_PID = os.path.expanduser("~/.cache/rec-pixel/ffmpeg.pid")
TRIAL_S = 45.0
SAMPLE_S = 0.25
LAM, MU = 6, 3
SIGMA0, SIGMA_MIN, SIGMA_MAX = 0.15, 0.03, 0.40
XMAX = math.log(2.0)
W_COV, W_CLR, W_REP = 1.0, 1.0, 1.0
REP_EPS, REP_LAG_S = 0.08, 8.0
GRID = (8, 16, 12)
EXPLODE_V = 8.0
QUIET_LOUD = 0.05

_rec = {"t": 0.0, "on": False}

def recording():
    """rec-pixel is recording (its ffmpeg pid alive); checked at most once a second."""
    now = time.monotonic()
    if now - _rec["t"] >= 1.0:
        _rec["t"] = now
        try:
            os.kill(int(open(REC_PID).read().strip()), 0)
            _rec["on"] = True
        except (OSError, ValueError):
            _rec["on"] = False
    return _rec["on"]

class Tuner:
    def __init__(self, n_all, n_per, box):
        self.n_all, self.n_per = n_all, n_per
        self.lo, self.hi = np.array(box[0], "f8"), np.array(box[1], "f8")
        self.rng = np.random.default_rng(int(time.time()))
        self.st = self._load()
        self.last = 0.0
        self.was_paused = False
        self._new_trial()

    def _load(self):
        try:
            st = json.load(open(STATE))
            if len(st.get("mean", [])) == N:
                return st
        except (OSError, ValueError):
            pass
        return {"mean": [0.0] * N, "var": [1.0] * N, "sigma": SIGMA0, "gen": 0, "pop": [], "scores": [], "history": [],
                "best": {"score": None, "x": [0.0] * N}}

    def _save(self):
        try:
            tmp = STATE + ".tmp"
            json.dump(self.st, open(tmp, "w"), indent=1)
            os.replace(tmp, STATE)
        except OSError:
            pass

    def _new_trial(self):
        st = self.st
        if not st["pop"]:
            m, sd = np.array(st["mean"]), st["sigma"] * np.sqrt(np.array(st["var"]))
            pop = [m.tolist()] + [np.clip(m + sd * self.rng.standard_normal(N), -XMAX, XMAX).tolist() for _ in range(LAM - 1)]
            st["pop"], st["scores"] = pop, []
        self.x = np.array(st["pop"][len(st["scores"])])
        self.t_active = 0.0
        self.cov = np.zeros(GRID, bool)
        self.v_sum = self.v_n = 0.0
        self.clr_sum = self.clr_n = 0.0
        self.desc = []
        self.rep_hit = self.rep_n = 0
        self.aborted = False
        self.fast = 0

    def _finish(self):
        st = self.st
        if self.aborted:
            score, parts = -1.0, (0.0, 0.0, 1.0)
        else:
            cov = self.cov.mean() / (0.5 + self.v_sum / max(self.v_n, 1))
            clr = self.clr_sum / max(self.clr_n, 1)
            rep = self.rep_hit / max(self.rep_n, 1)
            score, parts = W_COV * cov + W_CLR * clr - W_REP * rep, (cov, clr, rep)
        st["scores"].append(score)
        k = len(st["scores"]) - 1
        print(f"tune: gen {st['gen']} cand {k} score {score:+.3f} (cov {parts[0]:.3f} clr {parts[1]:.3f} rep {parts[2]:.3f})"
              + (" ABORTED (dispersal)" if self.aborted else "") + " | " + " ".join(
                  f"{n}={d * math.exp(x):.3g}" for (n, d), x in zip(KNOBS, self.x)), file=sys.stderr)
        st["history"].append({"t": time.strftime("%F %T"), "gen": st["gen"], "cand": k, "score": score,
                              "cov": parts[0], "clr": parts[1], "rep": parts[2], "x": self.x.tolist()})
        st["history"] = st["history"][-2000:]
        if len(st["scores"]) == LAM:
            order = np.argsort(st["scores"])[::-1][:MU]
            w = np.log(MU + 0.5) - np.log(np.arange(1, MU + 1)); w /= w.sum()
            X = np.array(st["pop"]); m = np.array(st["mean"])
            m_new = (w[:, None] * X[order]).sum(0)
            step = (X[order] - m) / max(st["sigma"], 1e-6)
            st["var"] = (0.8 * np.array(st["var"]) + 0.2 * (w[:, None] * step ** 2).sum(0)).clip(0.05, 4.0).tolist()
            improved = st["scores"][order[0]] > st["scores"][0] + 0.01
            st["sigma"] = float(np.clip(st["sigma"] * (1.15 if improved else 0.92), SIGMA_MIN, SIGMA_MAX))
            st["mean"] = np.clip(m_new, -XMAX, XMAX).tolist()
            if st["best"]["score"] is None or st["scores"][0] > st["best"]["score"]:
                st["best"] = {"score": st["scores"][0], "x": X[0].tolist(), "gen": st["gen"]}
            print(f"tune: gen {st['gen']} done: best cand {int(order[0])} {st['scores'][order[0]]:+.3f}, mean was "
                  f"{st['scores'][0]:+.3f}, sigma {st['sigma']:.3f}", file=sys.stderr)
            st["gen"] += 1
            st["pop"], st["scores"] = [], []
        self._save()
        self._new_trial()

    def _held(self):
        """The held setting (HOLD_FILE), re-read at most once a second; None = not holding."""
        now = time.monotonic()
        if now - getattr(self, "_hold_t", -9.0) >= 1.0:
            self._hold_t, self._hold = now, None
            try:
                raw = (open(HOLD_FILE).read().strip() if os.path.exists(HOLD_FILE) else SHIPPED_HOLD) or "best"
                x = self.st["best"]["x"] if raw == "best" else self.st["mean"] if raw == "mean" else json.loads(raw)
                if len(x) == N:
                    self._hold = list(x)
                    if not getattr(self, "_hold_said", False):
                        print(f"tune: HOLD ({raw}) | " + " ".join(f"{n}={d * math.exp(v):.3g}" for (n, d), v in zip(KNOBS, x)),
                              file=sys.stderr)
                self._hold_said = self._hold is not None
            except (OSError, ValueError, KeyError, TypeError):
                self._hold_said = False
        return self._hold

    def _paused(self, paused):
        p = paused or recording() or os.path.exists(OFF_FILE) or self._held() is not None
        if self.was_paused and not p:
            self._new_trial()
        self.was_paused = p
        return p

    def values(self, paused):
        """The u_tune array for this frame: the candidate under test, or the shipped defaults when paused (move lab, rec-pixel,
        the off file)."""
        if self._paused(paused):
            h = None if paused else self._held()
            return [0.0] * N if h is None else [d * math.exp(x) for (n, d), x in zip(KNOBS, h)]
        return [d * math.exp(x) for (n, d), x in zip(KNOBS, self.x)]

    def sample(self, boid, ldr, ctl, dt_s, paused):
        """Called every frame; reads back every SAMPLE_S (the textures: boid N_ALL x 3 rgba32f, ldr 768 x 72 r32f, ctl r32f)."""
        if self.was_paused:
            return
        now = time.monotonic()
        if now - self.last < SAMPLE_S:
            return
        self.last = now
        ctl = np.frombuffer(ctl.read(), dtype="f4")
        if float(ctl[0]) < QUIET_LOUD:
            return
        b = np.frombuffer(boid.read(), dtype="f4").reshape(3, self.n_all, 4)
        idx = np.arange(1, self.n_all, max(1, self.n_all // 768))
        idx = idx[idx % self.n_per != 0]
        p, v = b[0, idx, :3].astype("f8"), b[1, idx, :3].astype("f8")
        if not np.isfinite(p).all():
            self.aborted = True; self._finish(); return
        sp = float(np.linalg.norm(v, axis=1).mean())
        self.fast = self.fast + 1 if sp > EXPLODE_V else 0
        if self.fast >= 4:
            self.aborted = True; self._finish(); return
        self.t_active += SAMPLE_S
        self.v_sum += sp; self.v_n += 1
        g = ((p - self.lo) / (self.hi - self.lo) * np.array(GRID)).astype(int)
        g = np.clip(g, 0, np.array(GRID) - 1)
        self.cov[g[:, 0], g[:, 1], g[:, 2]] = True
        L = np.frombuffer(ldr.read(), dtype="f4").reshape(72, 768)
        lead = [L[2 * f].reshape(192, 4)[:48, :3] for f in (0, 1)]
        slot = [L[4 + f].reshape(192, 4)[:48, :3] for f in (0, 1)]
        de = []
        for f in (0, 1):
            pf = p[(idx // self.n_per) == f]
            if len(pf):
                d = np.linalg.norm(pf[:, None, :] - lead[f][None, :, :], axis=2).min(1)
                de.append(d.mean())
        dl = float(np.mean([np.linalg.norm(lead[f] - slot[f], axis=1).mean() for f in (0, 1)]))
        if de:
            self.clr_sum += math.exp(-float(np.mean(de)) / 0.12) * math.exp(-dl / 0.15); self.clr_n += 1
        if int(self.t_active / SAMPLE_S) % 2 == 0:
            cs = [p[(idx // self.n_per) == f] for f in (0, 1)]
            dsc = np.concatenate([c.mean(0) for c in cs] + [c.std(0).mean(keepdims=True) for c in cs])
            old = [dd for t, dd in self.desc if self.t_active - t > REP_LAG_S]
            if old:
                self.rep_n += 1
                if min(np.linalg.norm(np.array(old) - dsc, axis=1)) < REP_EPS:
                    self.rep_hit += 1
            self.desc.append((self.t_active, dsc))
            self.desc = [(t, dd) for t, dd in self.desc if self.t_active - t < 120.0]
        if self.t_active >= TRIAL_S:
            self._finish()
