#!/usr/bin/env python3
"""bands_stems.py -- offline reproduction of the live shader-backdrop audio feed.

The live pipeline (patched EasyEffects `shader_bands` + shader-backdrop
renderer.py) hands shaders two things this offline renderer did not previously
provide:

    bands   120x1  120 log-spaced bands, 27.5 Hz .. 28160 Hz, one semitone
                   apart, norm_db 0..1, per-band attack/release envelope.
    stems   120x4  row 0 drums, 1 bass, 2 tonal, 3 per-band onset -- the HPSS
                   separation the orrery's beat clock reads.

This module reproduces both from a decoded audio file so a shader written for
the live feed (e.g. schwarz_orrery_v4_opt.mp) renders in render_video.py with
no shader edits.

Stems reuse shader-backdrop's own `stem_split.StemSplitter` verbatim -- the
SAME code the live renderer runs -- so the separation matches by construction.
Only the rate differs (render fps here vs 188 Hz block rate live); split()
takes rate_hz explicitly and adapts.

Bands (2026-09-26): the SAME filter bank as live -- one RBJ constant-0dB-peak
bandpass per band (shader_bands.cpp BandFilter::design, n_band_stages = 1),
RMS of its output per block, norm_db, envelope. Only the block length differs
(one video frame here vs the PipeWire quantum live).
The old version bucketed an UN-normalised STFT: sum of |rfft| bins is ~50 dB
hotter than a band RMS, so 89% of all band values clipped at 1.0. A flat,
pinned spectrum is exactly what the orrery's leakage/overtone/floor cleanup
subtracts to zero, so bass and treble barely moved in every video render.
"""
import os
import sys

import numpy as np

import features as F  # noqa: E402

SB_DIR = F.backdrop_dir()
if SB_DIR not in sys.path:
    sys.path.insert(0, SB_DIR)
import stem_split  # noqa: E402  (needs SB_DIR on the path first)

N_BANDS = 120
LOG_LO = 27.5
LOG_HI = 28160.0
FLOOR_DB = -60.0
RATIO = (LOG_HI / LOG_LO) ** (1.0 / N_BANDS)  # exactly 2**(1/12)


def band_edges():
    edges = []
    lo = LOG_LO
    for _ in range(N_BANDS):
        hi = lo * RATIO
        edges.append((lo, hi))
        lo = hi
    return edges


def _norm_db(mag):
    db = 20.0 * np.log10(np.asarray(mag, dtype=np.float64) + 1e-9)
    return np.clip((db - FLOOR_DB) / (0.0 - FLOOR_DB), 0.0, 1.0)


def _envelope_time(x, fps, attack_ms, release_ms):
    """Asymmetric one-pole follower along axis 0 (time), matching the native
    plugin's BandEnvelope ballistics so the orrery's own AGC sees the same
    character it sees live."""
    a = np.exp(-1.0 / max(fps * attack_ms / 1000.0, 1e-6))
    r = np.exp(-1.0 / max(fps * release_ms / 1000.0, 1e-6))
    out = np.empty_like(x, dtype=np.float64)
    prev = np.zeros(x.shape[1], dtype=np.float64)
    for t in range(x.shape[0]):
        coef = np.where(x[t] > prev, a, r)
        prev = coef * prev + (1.0 - coef) * x[t]
        out[t] = prev
    return out.astype(np.float32)


def compute_bands(audio, sr, fps):
    """(num_frames, 120) norm_db band matrix, envelope-smoothed. Mirrors the
    live native bank exactly (see module docstring)."""
    from scipy.signal import lfilter
    x = np.asarray(audio, dtype=np.float64)
    hop = max(1, int(round(sr / fps)))
    num_frames = max(1, int(len(x) / sr * fps))
    if len(x) < num_frames * hop:
        x = np.pad(x, (0, num_frames * hop - len(x)))
    x = x[:num_frames * hop]
    nyq = sr * 0.5
    rms = np.zeros((num_frames, N_BANDS), dtype=np.float64)
    for i, (lo, hi) in enumerate(band_edges()):
        f0 = np.sqrt(lo * hi)
        if f0 >= nyq:                                # dead top octave (live: NaN -> 0)
            continue
        q = min(max(f0 / (hi - lo), 0.7), 30.0)
        w0 = 2.0 * np.pi * f0 / sr
        alpha = np.sin(w0) / (2.0 * q)
        a0 = 1.0 + alpha
        y = lfilter([alpha / a0, 0.0, -alpha / a0],
                    [1.0, -2.0 * np.cos(w0) / a0, (1.0 - alpha) / a0], x)
        rms[:, i] = np.sqrt((y * y).reshape(num_frames, hop).mean(axis=1))
    out = _envelope_time(_norm_db(rms), fps, attack_ms=20.0, release_ms=25.0)
    out[:, [i for i, (lo, hi) in enumerate(band_edges()) if np.sqrt(lo * hi) >= nyq]] = 0.0
    return out


def compute_stems(bands_nd, fps):
    """(num_frames, 4, 120) norm_db rows drums/bass/tonal/onset, reusing the
    live StemSplitter so the separation itself matches."""
    sp = stem_split.StemSplitter(N_BANDS)
    win = stem_split.window_frames(fps)
    half = win // 2
    n = bands_nd.shape[0]
    out = np.zeros((n, 4, N_BANDS), dtype=np.float32)
    for t in range(n):
        lo = max(0, t - half)
        hi = min(n, t + half + 1)
        r = sp.split(bands_nd[lo:hi], t - lo, fps)
        out[t, 0] = r["drums"]
        out[t, 1] = r["bass"]
        out[t, 2] = r["tonal"]
        out[t, 3] = r["onset"]
    return out
