"""Tempo and beat-phase tracking for the shader backdrop.

Why this lives here and not in a shader (scope.md, 2026-09-21): `common.glsl`
is prepended per-shader-directory, so a beat clock written in GLSL is private
to one shader out of ~110 -- and GLSL has nowhere to keep the history a real
tempo estimator needs. All its state has to fit in vec4 texels, so the best it
can do is a per-hit heuristic keyed off the last inter-onset interval, which is
exactly the thing that got "shoved around" by fills and dropped bars: one
missed kick makes the next interval read ~2x and drags the estimate halfway to
half-tempo.

Here we have ~8 s of onset history, so tempo comes from an AUTOCORRELATION and
a single fill cannot move it. Phase is then locked with a PLL that corrects the
accumulator's RATE, never its position -- the no-teleport invariant every
consumer in the orrery depends on.

Fed from renderer.py's audio reader thread at the stem split's own 60 Hz
cadence (16.7 ms, comfortably under the 33 ms render frame, so draining the
full 188 Hz block rate here would buy nothing).
"""

import math

import numpy as np

# Kick band range, mirroring the in-shader detector it replaces: band centres
# are 27.5 * 2^(b/12), so 4..27 is ~34.6-131 Hz. The DRUMS row already has the
# bassline removed by the HPSS masks; restricting to the low bands on top of
# that excludes hats and snares, which would otherwise double the tempo.
KICK_BAND_LO = 4
KICK_BAND_HI = 27

# Seconds of onset history the autocorrelation sees. Long enough to hold
# several bars at any tempo in range, so a fill or a dropped bar is a small
# perturbation of the whole window rather than the entire measurement.
HISTORY_S = 8.0
NOMINAL_RATE_HZ = 60.0

# Tempo search range. Deliberately WIDER than the 60-180 BPM a per-hit detector
# needs: a narrow clamp is that kind of detector's only defence against octave
# errors (it cannot tell a beat from a half-beat), and the harmonic comb below
# replaces it. Widening means genuinely slow material no longer sits on the
# boundary.
BPM_MIN = 50.0
BPM_MAX = 210.0

# Octave resolution. Scoring a candidate period by summing the autocorrelation
# at that lag AND its multiples is what makes the true period win: a
# half-period scores its own peak but its "harmonics" land between the real
# ones. The log-BPM prior then breaks remaining ties toward human-perceived
# tempo. (Both are the standard construction -- Ellis 2007.)
COMB_HARMONICS = (1, 2, 3, 4)
COMB_WEIGHTS = (1.0, 0.5, 0.34, 0.25)
PRIOR_BPM = 120.0
PRIOR_SIGMA_OCT = 0.9

# How often to re-run tempo + phase estimation. The autocorrelation is ~480
# samples; re-running it at the 60 Hz feed rate would spend real CPU moving a
# deliberately slow-moving result by almost nothing.
ESTIMATE_PERIOD_S = 0.10

# A challenger must beat the incumbent tempo's score by this factor before it
# is adopted. This is the actual fix for "gets shoved around": a fill scores
# well for a moment but not 15% better than a tempo that has been true for
# eight seconds.
TEMPO_SWITCH_MARGIN = 1.15
# Even an adopted tempo eases in rather than snapping.
TBEAT_TAU_S = 1.0

# PLL: a measured phase error of `err` beats is corrected by adding err/TAU
# beats-per-second to the count's rate, clamped to a fraction of the nominal
# rate so a bad measurement can never make the orrery lurch.
PLL_TAU_S = 0.5
PLL_RATE_CLAMP = 0.25
# Beats of history the phase comb looks back over, newest weighted highest.
PHASE_COMB_BEATS = 8
PHASE_COMB_DECAY = 0.85

# Below this confidence the tempo freezes and the PLL disengages: the count
# keeps advancing at the last known rate (flywheel) instead of chasing noise.
CONF_GATE = 0.15
CONF_TAU_S = 0.4

# Same wrap, and the same reason, as the shader's BEAT_WRAP: keeping the
# accumulated count bounded protects float precision (the h21 lesson), and
# every consumer period in common.glsl divides 1024.
BEAT_WRAP = 1024.0

# Rolling scale for the ODF, so a quiet passage and a loud one contribute
# comparably. Tracks upward fast and decays slowly -- the opposite of an
# envelope follower, because what matters is the recent PEAK.
ODF_SCALE_DECAY_S = 4.0
ODF_FLOOR = 1e-4
# "Is anything actually hitting right now" is measured over the RECENT past
# only, against that slow-decaying scale. Measuring it over the whole 8 s
# window instead would be self-normalising and could never report silence: the
# window mean and the scale fall together, so the ratio stays put.
ACTIVITY_WINDOW_S = 1.5


def _wrap_signed(x):
    """Wrap a phase difference in beats into [-0.5, 0.5)."""
    return (x + 0.5) % 1.0 - 0.5


def _parabolic(y_lo, y_mid, y_hi):
    """Sub-sample peak offset in [-0.5, 0.5] from three samples around a
    maximum. Returns 0.0 for a degenerate (flat) peak."""
    denom = y_lo - 2.0 * y_mid + y_hi
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y_lo - y_hi) / denom, -0.5, 0.5))


class BeatTracker:
    """Stateful across calls, unlike StemSplitter -- a tempo tracker IS its
    history. Safe to call at an irregular rate: every time-dependent quantity
    is driven by a passed-in wall-clock delta, not by an assumed cadence."""

    def __init__(self, n_bands=120):
        self.n_bands = int(n_bands)
        self._len = int(HISTORY_S * NOMINAL_RATE_HZ)
        self._odf = np.zeros(self._len, dtype=np.float64)
        self._filled = 0
        self._prev_row = None

        self._rate_hz = NOMINAL_RATE_HZ      # measured feed rate
        self._last_audio_t = None
        self._last_mono = None

        self._tbeat = 0.5                    # seconds per beat
        self._tbeat_target = 0.5
        self._count = 0.0                    # beats, wrapped at BEAT_WRAP
        self._conf = 0.0
        self._phase_err = 0.0                # beats, [-0.5, 0.5)
        self._since_estimate = 0.0
        self._scale = ODF_FLOOR
        # External phase/tempo reference (a cached beat map for a recognised
        # song). Deliberately fed through the SAME PLL as the live estimate
        # rather than replacing the count: the count must stay continuous, so
        # a recalled map slews the clock into alignment over ~a second instead
        # of snapping it and teleporting every body on screen.
        self._ref = None

    # -- feed ---------------------------------------------------------------

    def update(self, drums_row, audio_t, mono_t):
        """`drums_row` is the stem split's drums output (norm_db, n_bands).
        `audio_t` is the timestamp of the delay-aligned row it came from, used
        only to measure the feed rate; `mono_t` is wall clock, used to advance
        the count. Returns the uniform dict to publish."""
        row = np.asarray(drums_row, dtype=np.float64)

        # Wall-clock delta drives the accumulator. Clamped so a render stall or
        # a scheduler hiccup cannot jump the count, and so a stalled audio
        # stream (no new blocks published at all, which is NOT the same as
        # silence) still leaves the orrery moving at its last known tempo
        # rather than freezing dead.
        if self._last_mono is None:
            dt = 1.0 / NOMINAL_RATE_HZ
        else:
            dt = float(np.clip(mono_t - self._last_mono, 0.0, 0.1))
        self._last_mono = mono_t

        # Feed rate from the audio clock, which is what lags convert against.
        if self._last_audio_t is not None:
            step = audio_t - self._last_audio_t
            if 1e-4 < step < 0.5:
                self._rate_hz += (1.0 / step - self._rate_hz) * 0.02
        self._last_audio_t = audio_t

        self._push_odf(row)

        if self._ref is not None:
            # A recognised song: the map is exact, so it outranks anything the
            # live estimator could work out from the last eight seconds.
            ref_phase, ref_tbeat = self._ref
            self._tbeat_target = float(np.clip(ref_tbeat,
                                               60.0 / BPM_MAX, 60.0 / BPM_MIN))
            self._phase_err = _wrap_signed(ref_phase - (self._count % 1.0))
            self._conf = 1.0
            self._since_estimate = 0.0
        else:
            self._since_estimate += dt
            if self._since_estimate >= ESTIMATE_PERIOD_S:
                self._since_estimate = 0.0
                self._estimate(dt=ESTIMATE_PERIOD_S)

        # Tempo eases toward its target even when the target was not replaced.
        k = 1.0 - math.exp(-dt / TBEAT_TAU_S)
        self._tbeat += (self._tbeat_target - self._tbeat) * k

        # Advance. The correction is a RATE term, so the count stays continuous
        # under any tempo or phase change -- no consumer can ever teleport.
        rate = 1.0 / max(self._tbeat, 1e-3)
        corr = self._phase_err / PLL_TAU_S * (1.0 if self._conf > CONF_GATE else 0.0)
        corr = float(np.clip(corr, -PLL_RATE_CLAMP * rate, PLL_RATE_CLAMP * rate))
        self._count = (self._count + dt * (rate + corr)) % BEAT_WRAP

        return self.uniforms()

    def set_reference(self, phase, tbeat):
        """Steer to a known beat grid (phase in [0,1), tbeat in seconds).
        Call every update while a cached map is in play."""
        self._ref = (float(phase) % 1.0, float(tbeat))

    def clear_reference(self):
        """Back to tracking the audio live."""
        self._ref = None

    @property
    def last_odf(self):
        """The most recent onset value, for a caller recording the song."""
        return float(self._odf[-1])

    def uniforms(self):
        return {
            "u_tbeat": float(self._tbeat),
            "u_beat_phase": float(self._count % 1.0),
            "u_beat_count": float(self._count),
            "u_beat_conf": float(self._conf),
            "u_beat_bpm": float(60.0 / max(self._tbeat, 1e-3)),
        }

    # -- internals ----------------------------------------------------------

    def _push_odf(self, row):
        """Half-wave-rectified flux of the kick bands. norm_db IS log
        magnitude, and log-domain flux is the standard onset function: it
        reports a proportional rise, so a quiet kick and a loud one read
        comparably without any explicit gain tracking."""
        lo = min(KICK_BAND_LO, row.size - 1)
        hi = min(KICK_BAND_HI + 1, row.size)
        band = row[lo:hi]
        if self._prev_row is None or self._prev_row.shape != band.shape:
            self._prev_row = band.copy()
            value = 0.0
        else:
            value = float(np.sum(np.maximum(0.0, band - self._prev_row)))
            self._prev_row = band.copy()

        self._odf = np.roll(self._odf, -1)
        self._odf[-1] = value
        self._filled = min(self._filled + 1, self._len)

        # Peak-tracking scale: jump up immediately, decay slowly.
        if value > self._scale:
            self._scale = value
        else:
            self._scale *= math.exp(-1.0 / (ODF_SCALE_DECAY_S * self._rate_hz))
        self._scale = max(self._scale, ODF_FLOOR)

    def _history(self):
        if self._filled < self._len:
            return self._odf[self._len - self._filled:]
        return self._odf

    def _score_lag(self, ac, lag):
        """Comb score for a candidate period, in SAMPLES (may be fractional).
        Summing the autocorrelation at the lag and its multiples is what
        resolves the octave; the log-BPM prior breaks what is left."""
        if lag < 1.0:
            return 0.0
        total = 0.0
        for h, w in zip(COMB_HARMONICS, COMB_WEIGHTS):
            idx = lag * h
            i = int(idx)
            if i + 1 >= ac.size:
                break
            frac = idx - i
            total += w * (ac[i] * (1.0 - frac) + ac[i + 1] * frac)
        bpm = 60.0 * self._rate_hz / lag
        if not (BPM_MIN <= bpm <= BPM_MAX):
            return 0.0
        prior = math.exp(-0.5 * (math.log2(bpm / PRIOR_BPM) / PRIOR_SIGMA_OCT) ** 2)
        return total * prior

    def _estimate(self, dt):
        x = self._history()
        # Need enough history to hold a few of the slowest periods before the
        # autocorrelation means anything.
        if x.size < int(3.0 * self._rate_hz):
            return

        recent = x[-max(2, int(ACTIVITY_WINDOW_S * self._rate_hz)):]
        activity = float(np.mean(recent)) / max(self._scale, ODF_FLOOR)
        conf_k = 1.0 - math.exp(-dt / CONF_TAU_S)

        xc = x - x.mean()
        denom = float(np.dot(xc, xc))
        if denom <= 1e-12 or activity < 1e-3:
            # Silence. Decay confidence, freeze tempo, let the count fly on.
            self._conf += (0.0 - self._conf) * conf_k
            self._phase_err = 0.0
            return

        # Biased autocorrelation (divided by N, not by the overlap count). The
        # taper this introduces is mild across our lag range -- at the longest
        # lag in range it is ~0.85 -- and it is the safe choice: the unbiased
        # form amplifies exactly the long-lag noise that would invent slow
        # tempos out of nothing.
        ac = np.correlate(xc, xc, mode="full")[xc.size - 1:] / denom

        lag_min = max(2, int(round(60.0 / BPM_MAX * self._rate_hz)))
        lag_max = min(ac.size - 2, int(round(60.0 / BPM_MIN * self._rate_hz)))
        if lag_max <= lag_min:
            self._conf += (0.0 - self._conf) * conf_k
            return

        lags = np.arange(lag_min, lag_max + 1)
        scores = np.array([self._score_lag(ac, float(L)) for L in lags])
        best_i = int(np.argmax(scores))
        best_score = float(scores[best_i])
        if best_score <= 0.0:
            self._conf += (0.0 - self._conf) * conf_k
            return

        # Sub-sample refinement, so tempo resolution is not limited to whole
        # samples of lag (~4 BPM at 120 BPM and 60 Hz).
        if 0 < best_i < scores.size - 1:
            offset = _parabolic(scores[best_i - 1], best_score, scores[best_i + 1])
        else:
            offset = 0.0
        cand_lag = float(lags[best_i]) + offset
        cand_tbeat = cand_lag / self._rate_hz

        # Hysteresis: score the INCUMBENT on the same evidence and keep it
        # unless genuinely beaten. A fill scores well for a moment; it does not
        # score 15% better than a tempo that has been true for eight seconds.
        inc_lag = self._tbeat_target * self._rate_hz
        inc_score = self._score_lag(ac, inc_lag)
        if best_score > inc_score * TEMPO_SWITCH_MARGIN or inc_score <= 0.0:
            self._tbeat_target = float(np.clip(cand_tbeat,
                                               60.0 / BPM_MAX, 60.0 / BPM_MIN))
            lock_lag = cand_lag
            lock_score = best_score
        else:
            lock_lag = inc_lag
            lock_score = inc_score

        # Confidence: how much the winning period stands out from the field,
        # gated by there being real onset activity at all.
        mean_score = float(np.mean(scores)) + 1e-12
        peak_ratio = lock_score / mean_score
        target_conf = float(np.clip((peak_ratio - 1.0) / 1.5, 0.0, 1.0))
        target_conf *= float(np.clip(activity * 12.0, 0.0, 1.0))
        self._conf += (target_conf - self._conf) * conf_k

        if self._conf > CONF_GATE:
            self._phase_err = self._measure_phase_error(x, lock_lag)
        else:
            self._phase_err = 0.0

    def _measure_phase_error(self, x, lag):
        """Correlate the onset history against an impulse comb at the locked
        period. The best-fitting offset says how long ago the last beat was;
        the difference from where the accumulator thinks it is drives the PLL."""
        L = max(lag, 2.0)
        n_off = int(math.ceil(L))
        if x.size < L * 2:
            return 0.0
        k_max = min(PHASE_COMB_BEATS, int((x.size - 1) / L))
        if k_max < 1:
            return 0.0

        newest = x.size - 1
        best_off, best_val = 0, -1.0
        vals = np.empty(n_off, dtype=np.float64)
        for off in range(n_off):
            total = 0.0
            w = 1.0
            for k in range(k_max):
                idx = newest - off - int(round(k * L))
                if idx < 0:
                    break
                total += w * x[idx]
                w *= PHASE_COMB_DECAY
            vals[off] = total
            if total > best_val:
                best_val, best_off = total, off
        if best_val <= 0.0:
            return 0.0

        if 0 < best_off < n_off - 1:
            best = best_off + _parabolic(vals[best_off - 1], vals[best_off],
                                         vals[best_off + 1])
        else:
            best = float(best_off)

        # `best` samples ago was a beat, so the newest sample sits that far
        # into the current beat.
        measured_phase = (best / L) % 1.0
        return _wrap_signed(measured_phase - (self._count % 1.0))



# ---------------------------------------------------------------------------
# Non-causal re-analysis (2026-09-21)
#
# Everything above is causal, because live it has to be. But once a song has
# FINISHED, we are no longer causal about it -- the whole onset history is in
# memory. Re-running a dynamic-programming tracker over that history gives a
# beat map of nearly offline quality for a song we only ever heard streamed,
# with no audio file involved at all.
#
# SIBLING CODE: shader-musicvideo/features.py has the same two algorithms,
# applied to a decoded file instead of a recorded ODF. They are deliberately
# duplicated rather than shared -- separate repos, separate venvs, and a render
# pipeline should not import across a checkout. If you fix a bug in one, fix it
# in the other; the constants are intentionally identical.
# ---------------------------------------------------------------------------

DP_TIGHTNESS = 100.0
DP_ALPHA = 0.8


def _fft_autocorr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return None
    n = 1 << int(np.ceil(np.log2(2 * len(x))))
    f = np.fft.rfft(x, n)
    return np.fft.irfft(f * np.conj(f), n)[:len(x)] / denom


def tempo_period(odf, rate):
    """Global tempo over a whole recording, in frames per beat. Same comb +
    log-BPM prior as the live estimator, but over the entire song rather than
    a rolling window."""
    ac = _fft_autocorr(odf)
    if ac is None:
        return 0.0
    lag_min = max(2, int(round(60.0 / BPM_MAX * rate)))
    lag_max = min(len(ac) - 2, int(round(60.0 / BPM_MIN * rate)))
    if lag_max <= lag_min:
        return 0.0
    lags = np.arange(lag_min, lag_max + 1)
    scores = np.zeros(len(lags))
    for h, w in zip(COMB_HARMONICS, COMB_WEIGHTS):
        idx = lags * h
        ok = idx < len(ac)
        scores[ok] += w * ac[idx[ok]]
    bpms = 60.0 * rate / lags
    scores *= np.exp(-0.5 * (np.log2(bpms / PRIOR_BPM) / PRIOR_SIGMA_OCT) ** 2)
    i = int(np.argmax(scores))
    if scores[i] <= 0.0:
        return 0.0
    if 0 < i < len(scores) - 1:
        off = _parabolic(scores[i - 1], scores[i], scores[i + 1])
    else:
        off = 0.0
    return float(lags[i]) + off


def dp_beat_track(odf, period):
    """Ellis dynamic-programming beat tracker over a finished recording.

    Every path is scored to the end before any beat is committed, so a dropped
    bar or a fill costs almost nothing -- the surrounding beats still prefer
    the grid that keeps tempo regular. This is the property a causal tracker
    cannot have at any tuning."""
    o = np.asarray(odf, dtype=np.float64)
    sd = o.std()
    o = (o - o.mean()) / sd if sd > 0 else o - o.mean()
    n = len(o)
    if n < 4 or period < 2.0:
        return np.array([], dtype=np.int64)

    w_lo = max(1, int(round(period * 0.5)))
    w_hi = max(w_lo + 1, int(round(period * 2.0)))
    gaps = np.arange(w_lo, w_hi + 1, dtype=np.float64)
    penalty = -DP_TIGHTNESS * (np.log(gaps / period) ** 2)

    C = np.zeros(n)
    P = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        hi = i - w_lo
        if hi < 0:
            C[i] = o[i]
            continue
        lo = max(i - w_hi, 0)
        pen = penalty[(i - np.arange(lo, hi + 1)) - w_lo]
        score = C[lo:hi + 1] + DP_ALPHA * pen
        k = int(np.argmax(score))
        C[i] = o[i] + score[k]
        P[i] = lo + k

    loc = np.where((C[1:-1] >= C[:-2]) & (C[1:-1] >= C[2:]))[0] + 1
    if len(loc) == 0:
        return np.array([], dtype=np.int64)
    thresh = 0.5 * float(np.median(C[loc]))
    strong = loc[C[loc] > thresh]
    last = int(strong[-1]) if len(strong) else int(loc[-1])

    beats = []
    while last >= 0:
        beats.append(last)
        last = int(P[last])
    return np.array(beats[::-1], dtype=np.int64)


def reanalyse(odf, times, duration_s=None):
    """Turn a recorded (odf, song-position times) pair into a beat map.

    `times` are SONG-POSITION seconds, so the result is replayable against any
    later playback of the same track. Returns
    (beat_times, bpm, coverage, confidence) or None when there is not enough
    evidence.

    `duration_s` is the track's full length. It is the denominator coverage
    must be measured against: the recording window (`span` below) shrinks with
    a partial listen, so dividing by it made a 10 s fragment of a 3-min song
    report ~100% coverage (OPEN_THREADS 6a). Both gates are returned so the
    caller decides what to trust; nothing is rejected here."""
    odf = np.asarray(odf, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if odf.size < 64 or odf.size != times.size:
        return None
    span = times[-1] - times[0]
    if span <= 4.0:
        return None
    rate = (times.size - 1) / span

    period = tempo_period(odf, rate)
    if period <= 0.0:
        return None
    idx = dp_beat_track(odf, period)
    if idx.size < 4:
        return None

    beat_times = times[idx]
    bpm = 60.0 * rate / period
    grid_span = float(beat_times[-1] - beat_times[0])
    if duration_s is not None and duration_s > 0.0:
        coverage = grid_span / float(duration_s)
    else:
        coverage = grid_span / span      # degraded: falls back to the old bug
    coverage = float(np.clip(coverage, 0.0, 1.0))

    # Confidence: how far the chosen beat frames sit above the average ODF,
    # as a fraction of the total. A grid placed on random frames scores ~0
    # (measured p99 ~0.06 on real audio); real beat grids measured 0.14-0.26.
    peak = float(np.mean(odf[idx]))
    base = float(np.mean(odf))
    confidence = float(np.clip((peak - base) / (peak + base + 1e-12), 0.0, 1.0))
    return beat_times, bpm, coverage, confidence

# ---------------------------------------------------------------------------
# Offline self-test. Runs without the renderer, EasyEffects, or a GPU:
#   python3 beat_track.py
# Synthesises drums rows with known tempo, then checks the tracker survives the
# two things that were actually reported as broken -- a dropped bar and a
# double-time fill.
# ---------------------------------------------------------------------------

def _synth(bpm, seconds, rate=NOMINAL_RATE_HZ, drop=None, fill=None, seed=7):
    """Returns a list of (drums_row, t) with a kick every beat. `drop` and
    `fill` are (start_s, end_s) spans of, respectively, no kicks at all and
    kicks at double rate."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    rows = []
    period = 60.0 / bpm
    level = np.full(120, 0.05)
    for i in range(n):
        t = i / rate
        in_drop = drop is not None and drop[0] <= t < drop[1]
        in_fill = fill is not None and fill[0] <= t < fill[1]
        p = period * 0.5 if in_fill else period
        # Distance to the nearest kick in this regime.
        frac = (t % p) / p
        hit = min(frac, 1.0 - frac) < (0.5 / rate / p)
        row = level + rng.normal(0.0, 0.004, 120)
        if hit and not in_drop:
            row[KICK_BAND_LO:KICK_BAND_HI + 1] += 0.45
        rows.append((np.clip(row, 0.0, 1.0), t))
    return rows


def _run(rows, tracker=None):
    tr = tracker or BeatTracker()
    out = {}
    for row, t in rows:
        out = tr.update(row, t, t)
    return tr, out


def _selftest():
    ok = True

    for bpm in (72.0, 100.0, 128.0, 174.0):
        tr, u = _run(_synth(bpm, 20.0))
        got = u["u_beat_bpm"]
        err = abs(got - bpm) / bpm
        good = err < 0.04 and u["u_beat_conf"] > CONF_GATE
        ok &= good
        print(f"  {'ok ' if good else 'FAIL'} steady {bpm:6.1f} BPM -> "
              f"{got:6.1f} ({err*100:4.1f}% err, conf {u['u_beat_conf']:.2f})")

    # A dropped bar: the thing that used to drag tempo toward half speed.
    tr, u = _run(_synth(120.0, 24.0, drop=(12.0, 14.0)))
    good = abs(u["u_beat_bpm"] - 120.0) / 120.0 < 0.04
    ok &= good
    print(f"  {'ok ' if good else 'FAIL'} dropped bar            -> "
          f"{u['u_beat_bpm']:6.1f} BPM")

    # A double-time fill: the thing that used to shove tempo to 2x.
    tr, u = _run(_synth(120.0, 24.0, fill=(12.0, 15.0)))
    good = abs(u["u_beat_bpm"] - 120.0) / 120.0 < 0.06
    ok &= good
    print(f"  {'ok ' if good else 'FAIL'} double-time fill       -> "
          f"{u['u_beat_bpm']:6.1f} BPM")

    # Silence must not move the tempo, and the count must keep advancing.
    tr, u = _run(_synth(120.0, 16.0))
    before_bpm, before_count = u["u_beat_bpm"], u["u_beat_count"]
    silent = [(np.full(120, 0.02), 16.0 + i / NOMINAL_RATE_HZ)
              for i in range(int(6.0 * NOMINAL_RATE_HZ))]
    _, u2 = _run(silent, tracker=tr)
    advanced = (u2["u_beat_count"] - before_count) % BEAT_WRAP
    good = (abs(u2["u_beat_bpm"] - before_bpm) < 1.0
            and abs(advanced - 6.0 / (60.0 / before_bpm)) < 0.5
            and u2["u_beat_conf"] < CONF_GATE)
    ok &= good
    print(f"  {'ok ' if good else 'FAIL'} 6s silence: flywheel   -> "
          f"{u2['u_beat_bpm']:6.1f} BPM, +{advanced:.2f} beats, "
          f"conf {u2['u_beat_conf']:.2f}")

    # Continuity: the count must never jump, at any tempo change.
    tr = BeatTracker()
    prev = None
    max_jump = 0.0
    for row, t in _synth(96.0, 14.0) + [(r, 14.0 + t) for r, t in _synth(150.0, 14.0)]:
        u = tr.update(row, t, t)
        c = u["u_beat_count"]
        if prev is not None:
            step = (c - prev) % BEAT_WRAP
            max_jump = max(max_jump, step)
        prev = c
    # One feed step at the fastest allowed rate plus the PLL's clamp.
    limit = (1.0 / NOMINAL_RATE_HZ) * (BPM_MAX / 60.0) * (1.0 + PLL_RATE_CLAMP)
    good = max_jump <= limit
    ok &= good
    print(f"  {'ok ' if good else 'FAIL'} tempo change continuity-> "
          f"max step {max_jump:.4f} beats (limit {limit:.4f})")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
