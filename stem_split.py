#!/usr/bin/env python3
"""Instrument-group ("stem") separation on the 120-band envelope matrix.

WHY THIS EXISTS
---------------
Shaders could already ask "how loud is this semitone" (the `bands` texture)
and "did something just hit" (`u_onset`, one global scalar). They could not
ask "show me the bassline" or "show me the drums" -- song ELEMENTS rather
than frequencies. Every attempt to infer that inside a shader is a guess
made from one frame of magnitudes, and it has to be re-implemented in each
of the ~110 shaders because `common.glsl` is prepended per-shader-directory
only, i.e. there is no shared GLSL layer to put it in. So it lives here,
computed once, published to every shader. See scope.md 2026-09-20.

WHAT IT DOES
------------
Median-filter source separation (the classic HPSS idea), applied to the
120-band envelope matrix rather than to an STFT:

  * A sustained note occupies ONE band across MANY frames.
    -> a median across TIME, per band, keeps it and rejects transients.
  * A drum hit occupies MANY bands in ONE frame.
    -> a median across FREQUENCY, per frame, keeps it and rejects tones.

Those two estimates become soft Wiener-style masks that sum to 1, so no
energy is invented or lost -- every band's energy is DIVIDED between the
groups, never duplicated into both. The harmonic side is then split by
register into bass vs everything-else-tonal.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
Separate vocals from synths from guitar. That needs a learned model
(Demucs); this is the explicitly-chosen lighter option, and those three
share the `tonal` output by design, not by accident.

DOMAIN
------
The incoming band values are `norm_db`: magnitude in dB, normalized over a
-60..0 dB window (shader_bands.cpp). Medians and masks on dB are NOT medians
and masks on magnitude, so everything here converts to linear first. The
conversion is exact and cheap in both directions:

    db      = 60*x - 60          (x is the 0..1 norm_db value)
    log10(m)= db/20 = 3*(x - 1)
    m       = 10 ** (3*(x - 1))

Output is converted back to norm_db so shaders see the same 0..1 convention
the existing `bands` texture already uses.
"""
import numpy as np

N_BANDS = 120
FLOOR_DB = -60.0

# 2026-09-21: 40 (~277 Hz) -> 24 (~110 Hz). At 277 Hz this "bass" stem was
# really "everything low-ish": a male pop-vocal fundamental (~110-180 Hz) sits
# inside it, so an a cappella voice lit the disk. A subwoofer bassline is
# octaves below that -- bass guitar low E is 41 Hz, and all four open-string
# fundamentals (E1 41, A1 55, D2 73, G2 98 Hz) are bands 7-20. Band 24 is
# 110 Hz (27.5 * 2^(24/12) = 110 exactly), so this keeps those fundamentals and
# excludes voice. Crossfaded over CROSSOVER_WIDTH bands, not a hard cut, so a
# bassline walking across the boundary does not jump between outputs mid-phrase.
#
# KNOWN LIMIT (scope.md): still a REGISTER split, not a source split. A bass
# guitar's upper register and a baritone voice overlap above ~110 Hz, and a
# real stem separator would keep a bass note's overtones with the bass. That
# remains the accepted price of "no ML".
BASS_CROSSOVER_BAND = 24.0
CROSSOVER_WIDTH = 5.0

# Time window for the harmonic (across-time) median, in seconds. Long enough
# that a drum hit is a clear outlier inside it, short enough to track real
# chord changes. At the measured 188 Hz block rate this is ~75 frames.
HARMONIC_WINDOW_S = 0.40

# Width, in bands, of the percussive (across-frequency) median. A drum hit
# is broadband, so it survives a median over this many semitones; a single
# note does not. ~17 semitones is wider than any chord voicing sitting
# inside one octave, which is the point.
PERCUSSIVE_KERNEL_BANDS = 17

# Mask exponent. 1.0 = plain ratio masks, 2.0 = Wiener-style (squares the
# contrast between the two estimates before normalizing, so a clear winner
# takes almost all of the band and an ambiguous band stays split).
MASK_POWER = 2.0

# How far above its own recent median a band must rise to read as a
# full-strength onset, in norm_db units (0.15 = 9 dB).
ONSET_SPAN = 0.15
ONSET_BASELINE_S = 0.25


def to_linear(x):
    """norm_db (0..1) -> linear magnitude. Exact inverse of shader_bands.cpp's
    norm_db(), which is db = 20*log10(mag) clamped/scaled over -60..0 dB."""
    return np.power(10.0, 3.0 * (np.asarray(x, dtype=np.float64) - 1.0))


def to_norm_db(m):
    """linear magnitude -> norm_db (0..1). Matches the native plugin's own
    formula including its 1e-9 guard, so round-tripping a value is a no-op."""
    db = 20.0 * np.log10(np.asarray(m, dtype=np.float64) + 1e-9)
    return np.clip((db - FLOOR_DB) / (0.0 - FLOOR_DB), 0.0, 1.0)


def _median_across_frequency(frame_lin, kernel):
    """Median over a sliding band-neighbourhood. Edges use a shrinking
    window rather than zero-padding: padding with zeros would drag the
    median down at the spectrum's ends and make bands 0 and 119 read as
    falsely tonal (low percussive estimate), which is exactly where the
    sub-bass and the dead top octave live."""
    half = kernel // 2
    out = np.empty_like(frame_lin)
    n = frame_lin.shape[0]
    for b in range(n):
        lo = max(0, b - half)
        hi = min(n, b + half + 1)
        out[b] = np.median(frame_lin[lo:hi])
    return out


def bass_weight(n_bands=N_BANDS):
    """Per-band bass share, 1.0 in the low register falling to 0.0 above the
    crossover. `tonal` gets (1 - this), so the two always sum to the full
    harmonic component -- no band is counted twice or dropped."""
    b = np.arange(n_bands, dtype=np.float64)
    t = np.clip((b - (BASS_CROSSOVER_BAND - CROSSOVER_WIDTH * 0.5)) / CROSSOVER_WIDTH, 0.0, 1.0)
    return 1.0 - (t * t * (3.0 - 2.0 * t))  # smoothstep


class StemSplitter:
    """Stateless with respect to time: every call gets the whole window it
    needs passed in, so there is no internal filter state that could drift,
    get stale during a pause, or depend on being called at a fixed rate.
    That matters because the caller's rate is the RENDER rate, which varies
    per shader, while the data's rate is the audio block rate."""

    def __init__(self, n_bands=N_BANDS):
        self.n_bands = n_bands
        self._bass_w = bass_weight(n_bands)

    def split(self, window, center_index, rate_hz):
        """`window` is (T, n_bands) of norm_db values, oldest first, and
        `center_index` is the row being rendered -- normally NOT the newest
        row. The caller feeds a window centred on the delay-aligned frame,
        which is only possible because renderer.py already holds back the
        audio delay (config.toml, e.g. ~1.1s for a Bluetooth speaker): the "future" half of the window
        has already been published. A centred median is what HPSS wants, and
        here it costs no additional latency at all.

        `rate_hz` is the real measured publish rate of the rows, passed in
        rather than inferred from the window length -- the caller is the only
        thing that knows it, and inferring it would silently produce a wrong
        onset baseline the moment the window size or block size changed.

        Returns a dict of (n_bands,) norm_db arrays: drums, bass, tonal,
        onset."""
        w = np.asarray(window, dtype=np.float64)
        if w.ndim != 2 or w.shape[1] != self.n_bands:
            raise ValueError(f"window must be (T, {self.n_bands}), got {w.shape}")
        t_len = w.shape[0]
        center_index = int(np.clip(center_index, 0, t_len - 1))

        lin = to_linear(w)
        center_lin = lin[center_index]

        # Harmonic estimate: what this band has been doing for a while.
        harm_est = np.median(lin, axis=0)
        # Percussive estimate: what the whole neighbourhood is doing NOW.
        perc_est = _median_across_frequency(center_lin, PERCUSSIVE_KERNEL_BANDS)

        h_p = np.power(harm_est, MASK_POWER)
        p_p = np.power(perc_est, MASK_POWER)
        denom = h_p + p_p + 1e-18
        harm_mask = h_p / denom
        perc_mask = p_p / denom

        harmonic_lin = center_lin * harm_mask
        percussive_lin = center_lin * perc_mask

        bass_lin = harmonic_lin * self._bass_w
        tonal_lin = harmonic_lin * (1.0 - self._bass_w)

        # Per-band onset: how far above its own recent median this band sits
        # right now, in norm_db units. Deliberately measured against a
        # MEDIAN of the recent past rather than the previous frame -- a
        # single-frame difference at 188 Hz is dominated by envelope ripple,
        # while a median baseline ignores brief dips and only reports a
        # genuine, sustained-enough rise. This is the per-band equivalent of
        # the native plugin's global `onset` (flux vs a ~1.5s baseline).
        base_rows = max(1, int(round(ONSET_BASELINE_S * rate_hz)))
        lo = max(0, center_index - base_rows)
        baseline = np.median(w[lo:center_index + 1], axis=0) if center_index > lo else w[center_index]
        onset = np.clip((w[center_index] - baseline) / ONSET_SPAN, 0.0, 1.0)

        return {
            "drums": to_norm_db(percussive_lin),
            "bass": to_norm_db(bass_lin),
            "tonal": to_norm_db(tonal_lin),
            "onset": onset,
        }


def window_frames(rate_hz):
    """How many history rows a caller should hand to split(), given the real
    observed publish rate. Odd so there is an exact centre row."""
    n = int(round(HARMONIC_WINDOW_S * rate_hz))
    return max(3, n | 1)
