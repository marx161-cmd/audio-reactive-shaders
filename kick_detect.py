#!/usr/bin/env python3
"""Drum detection with look-ahead (kicks first, all drums since 2026-10-09) ->
a per-band "flatten" weight for the frame
being drawn. (scope.md 2026-10-08 "kick-only flattening"; design reviewed by
an audio engineer, see AUDIO_CLEANUP_REVIEW.md.)

WHY HERE
--------
A kick and a sharp low note attack look the same in the band filters for the
first ~50 ms: any sharp low attack excites every low filter, and they ring
for ~100 ms (measured, tools/stem_split_bench.py). What tells them apart is
what happens NEXT -- a note keeps its pitch, a kick dies away -- and only the
renderer has "next": it holds the analysis back by the speaker latency
(~1.1 s), so the frames after the one being drawn already exist.

WHAT IT DOES (per drawn frame)
------------------------------
1. WHEN: spectral flux of the low bands (sum of positive frame-to-frame rises
   of the norm_db levels, bands KICK_LO..KICK_HI-1, 35-131 Hz), smoothed, peak
   picked against a running median. Each peak is an onset; the envelope starts
   PRE_FRAMES before the peak (the band envelope's 20 ms attack puts the flux
   peak a few frames after the true impact), so nothing slips through.
2. WHAT, per band (bands 0..FLAT_TOP+FLAT_FADE): did this band rise at the
   onset, and has it DECAYED by more than DECAY_DB PERSIST_S (0.4 s) later? Decayed =
   kick-like (also the onset ringing of a plucked note -- an artefact, fine to
   flatten). Persisting = a note (incl. overtone-free sub-bass / 808s): weight 0.
3. Is the event a kick at all: at least KICK_MIN_BANDS kick-like bands in the
   detector range. A lone note's onset ringing can pass this; its root then
   persists and is protected per band, and the cleanup kernel additionally
   protects any root whose overtones rose in proportion (harmonic bypass).
Output: (n_bands,) weights 0..1 = envelope x kick-like, max over overlapping
events. The GPU cleanup flattens the broad low component by that much.
"""
import numpy as np

KICK_LO, KICK_HI = 4, 28      # detector range, bands (35-131 Hz; beat_track.py's kick range)
FLAT_TOP, FLAT_FADE = 40, 8   # weights cover bands 0..47, fading out above band 40 (~277 Hz)
SMOOTH = 3                    # frames, flux smoothing (boxcar)
PEAK_HALF = 3                 # frames: a peak is the max of +-PEAK_HALF frames
FLUX_MIN = 0.6                # norm_db units summed over the 24 bands (0.6 = 1.5 dB per band average)
FLUX_REL = 2.5                # ... and this many times the running median flux (last BASE_S)
BASE_S = 1.0
PRE_FRAMES = 3                # envelope starts this many frames before the flux peak
RISE_DB = 3.0                 # a band "rose" at the onset by at least this
PEAK_S = 0.20                 # its peak level = max over the first PEAK_S after the onset (catches the kick sweep
                              # kick's downward pitch sweep, which moves energy into lower bands later)
PERSIST_S = 0.40              # look this far past EACH BAND'S OWN peak (a kick body rings ~0.2 s: -17 dB by 0.4 s; an 808 /
                              # plucked bass -3..-4 dB; sustained 0 dB -- measured, tools/kick_bench.py) ...
DECAY_DB = 12.0               # ... a band that dropped more than this from its peak is kick-like
                              # (0.5 s: kick ~-22 dB, 808 bass ~-5, pluck ~-2, sustained 0; was 0.8 s / 15 dB until the
                              # band-delay compensation (up to ~0.22 s in the kick range) had to fit the 0.92 s speaker delay too)
KICK_MIN_BANDS = 8
HOLD_S = 0.30                 # envelope: full for HOLD_S from the onset (only kick-classified bands are affected) ...
FADE_S = 0.25                 # ... then linear fade to 0
DB = 60.0                     # norm_db span


# 2026-10-09: generalised from kicks to ALL drums (user: "high amount of drums
# coming through, a longer drum-only part" -- with the full bands as input,
# snares / hats / toms passed untouched). The same logic runs per frequency
# RANGE, each with its own onset flux, its own "is it broad" count and its own
# flatten region. A snare / hat / tom hit decays within ~100-300 ms; a note
# keeps its pitch and stays. Long cymbal crashes ring for seconds and mostly
# still show (not decayed 0.4 s after their peak).
#   (name, detect lo, hi (bands), flatten core lo, hi, min drum-like bands, hold s, fade s)
RANGES = (
    ("kick",  KICK_LO, KICK_HI, 0, FLAT_TOP, KICK_MIN_BANDS, HOLD_S, FADE_S),   # 35-131 Hz
    ("mid",   28, 60, 28, 64, 10, 0.20, 0.15),    # 131-880 Hz: snare body, toms
    ("high",  60, 108, 60, 107, 12, 0.12, 0.10),  # 880 Hz-14 kHz: snare rattle, hats
)
FLUX_MIN_PER_BAND = FLUX_MIN / (KICK_HI - KICK_LO)   # 0.025 norm_db per band, as the kick range had


def _flux(rows, lo=KICK_LO, hi=KICK_HI):
    d = np.diff(rows[:, lo:hi], axis=0)
    f = np.concatenate([[0.0], np.maximum(d, 0).sum(1)])
    k = np.ones(SMOOTH) / SMOOTH
    return np.convolve(f, k, mode="same")


def _region(n_bands, core_lo, core_hi):
    """Flatten profile: 1 inside the core, fading out over FLAT_FADE bands either side."""
    b = np.arange(n_bands)
    return np.clip(1.0 - np.maximum(core_lo - b, b - core_hi) / FLAT_FADE, 0.0, 1.0)


def _events(rows, rate_hz, t_lo, t_hi, rng=RANGES[0]):
    """Onsets (frame index of the envelope start) of range `rng` with frames in
    [t_lo, t_hi), each with its per-band drum-like mask (n_bands,) bool."""
    _, dlo, dhi, flo, fhi, min_bands, _, _ = rng
    n_bands = rows.shape[1]
    reg = _region(n_bands, flo, fhi) > 0
    flux = _flux(rows, dlo, dhi)
    fmin = FLUX_MIN_PER_BAND * (dhi - dlo)
    base_n = max(3, int(BASE_S * rate_hz))
    pk_n = max(1, int(round(PEAK_S * rate_hz)))
    per_n = max(1, int(round(PERSIST_S * rate_hz)))
    out = []
    for t in range(max(PEAK_HALF, t_lo), min(len(rows) - PEAK_HALF, t_hi)):
        f = flux[t]
        if f < fmin or f < flux[t - PEAK_HALF:t + PEAK_HALF + 1].max():
            continue
        base = np.median(flux[max(0, t - base_n):t]) if t > 0 else 0.0
        if f < FLUX_REL * base:
            continue
        s = max(0, t - PRE_FRAMES)
        before = rows[max(0, s - 2)]
        win = rows[s:min(len(rows), s + pk_n + 1)]
        if len(win) <= pk_n // 2:              # not enough look-ahead: can't classify, skip
            continue
        peak = win.max(0)
        # decay is measured PERSIST_S after each band's OWN peak (2026-10-08:
        # the kick's sweep reaches its last, lowest bands late; the sharper low
        # filters peak later still -- a fixed start missed them)
        pk_i = s + win.argmax(0)
        later_i = np.minimum(pk_i + per_n, len(rows) - 1)
        later = rows[later_i, np.arange(n_bands)]
        enough = later_i - pk_i >= per_n // 2   # bands without enough look-ahead: not drum
        rose = (peak - before) * DB >= RISE_DB
        decayed = ((peak - later) * DB >= DECAY_DB) & enough
        drum = rose & decayed & reg
        if drum[dlo:dhi].sum() >= min_bands:
            out.append((s, drum))
    return out


def _envelope(dt_s, hold, fade):
    if dt_s < 0:
        return 0.0
    if dt_s <= hold:
        return 1.0
    return max(0.0, 1.0 - (dt_s - hold) / fade)


def weights_at(rows, idx, rate_hz, n_bands):
    """Flatten weights for frame `idx` of `rows` ((T, n_bands) norm_db, oldest
    first; frames after idx are the look-ahead). Max over all drum ranges."""
    w = np.zeros(n_bands)
    for rng in RANGES:
        hold, fade = rng[6], rng[7]
        prof = _region(n_bands, rng[3], rng[4])
        reach = int((hold + fade) * rate_hz) + 1
        for s, drum in _events(rows, rate_hz, idx - reach - PRE_FRAMES, idx + PRE_FRAMES + 1, rng):
            e = _envelope((idx - s) / rate_hz, hold, fade)
            if e > 0:
                w = np.maximum(w, e * drum * prof)
    return w


def range_weight(w, name):
    """How much of drum range `name` ("kick" / "mid" / "high") is being flattened."""
    for rng in RANGES:
        if rng[0] == name:
            return float(np.max(w[rng[1]:rng[2]])) if len(w) >= rng[2] else 0.0
    return 0.0


def kick_weight(w):
    """How much of the KICK range is being flattened (for u_kick / the drum glow)."""
    return float(np.max(w[KICK_LO:KICK_HI])) if len(w) >= KICK_HI else 0.0


def history_frames(rate_hz):
    """(frames before, frames after) the drawn frame that weights_at needs."""
    hold = max(r[6] for r in RANGES)
    fade = max(r[7] for r in RANGES)
    back = int((BASE_S + hold + fade) * rate_hz) + PRE_FRAMES + PEAK_HALF + SMOOTH + 4
    fwd = int((PEAK_S + PERSIST_S + 0.02) * rate_hz) + PRE_FRAMES + PEAK_HALF + SMOOTH + 2
    return back, fwd


def weights_all(rows, rate_hz):
    """Offline: weights for every frame (bench)."""
    return np.array([weights_at(rows, i, rate_hz, rows.shape[1]) for i in range(len(rows))])
