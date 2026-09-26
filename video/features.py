#!/usr/bin/env python3
"""features.py -- offline audio feature extraction for the shader music video.

Decodes any ffmpeg-readable track to mono float32 and computes, per *video
frame*, the features that fragment shaders react to:

    rms          overall loudness (amplitude)
    sub / bass / lowmid / mid / highmid / presence / brilliance
                 band energy (amplitude, RMS-in-band)
    centroid     spectral centroid (Hz == "brightness")
    flux         spectral flux (how much the spectrum changed vs last frame)
    onset        onset strength 0..1 (kick / transient energy)

Everything is frame-aligned to FPS, so features[t] == video frame t == audio
time t/FPS. Deterministic: same file + same args -> byte-identical output.
Only numpy is required (ffmpeg is used only to decode the file). fps should be
an integer-ish rate (30 / 60) so hop = sr/fps lands cleanly.

Texture layout (for the renderer / shader side):
    build_texture(features_norm) -> float32 array, shape (num_features, num_frames)
    i.e. width = num_frames (time), height = num_features, row = feature.
    GLSL:  texture(iFeatures, vec2((t + 0.5)/num_frames, (f + 0.5)/num_features)).r

Usage:
    python features.py track.flac --fps 60 -o features.npz
    python features.py track.flac --fps 60 --texture features.raw   # (F, N) f32 for GL
    python features.py --selftest                                    # synthetic click+sweep
"""

import argparse
import shutil
import subprocess
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# (name, low_hz, high_hz) -- musically meaningful bands (log-ish).
DEFAULT_BANDS = [
    ("sub",        20.0,    60.0),
    ("bass",       60.0,   250.0),
    ("lowmid",    250.0,   500.0),
    ("mid",       500.0,  2000.0),
    ("highmid",  2000.0,  4000.0),
    ("presence", 4000.0,  8000.0),
    ("brilliance", 8000.0, 16000.0),
]

DEFAULT_FFMPEG = "/opt/ffmpeg-master-custom/bin/ffmpeg"   # tried first, then PATH


def backdrop_dir():
    """The renderer checkout (shaders/, textures/, stem_split.py): $SHADER_BACKDROP_DIR,
    else this file's parent dir (video/ inside the audio-reactive-shaders repo),
    else a sibling `shader-backdrop` checkout."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.environ.get("SHADER_BACKDROP_DIR", ""), os.path.dirname(here),
                 os.path.join(os.path.dirname(here), "shader-backdrop")):
        if cand and os.path.isfile(os.path.join(cand, "stem_split.py")):
            return cand
    raise SystemExit("renderer checkout not found: set SHADER_BACKDROP_DIR")


def resolve_ffmpeg(explicit=None):
    if explicit:
        return explicit
    for cand in (DEFAULT_FFMPEG, "ffmpeg"):
        if shutil.which(cand):
            return cand
    raise SystemExit("ffmpeg not found (pass --ffmpeg /path/to/ffmpeg)")


def decode_file(path, target_sr=48000, ffmpeg=None):
    """Decode any ffmpeg-readable audio file to mono float32 numpy array."""
    ff = resolve_ffmpeg(ffmpeg)
    # -nostdin: without it ffmpeg reads the controlling tty for key commands.
    # When the render is auto-backgrounded by the shell, that read raises
    # SIGTTIN and STOPS THE WHOLE JOB (seen via `smv`, 2026-09-23). This ffmpeg
    # only reads a file and writes PCM to stdout, so stdin is never needed.
    cmd = [ff, "-nostdin", "-v", "error", "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise SystemExit("ffmpeg decode failed:\n" + proc.stderr.decode(errors="replace"))
    return np.frombuffer(proc.stdout, dtype="<f4").copy(), target_sr


def _stft(x, nperseg, hop):
    """Magnitude spectrogram (rfft). Returns (n_bins, n_frames)."""
    if len(x) < nperseg:
        x = np.pad(x, (0, nperseg - len(x)))
    window = np.hanning(nperseg).astype(np.float32)
    frames = sliding_window_view(x, nperseg)[::hop]
    spec = np.abs(np.fft.rfft(frames * window, n=nperseg, axis=1))
    return spec.T


def _peak_pick(env, thresh, hop, sr, min_interval):
    idx = np.where((env[1:-1] > thresh[1:-1])
                   & (env[1:-1] >= env[:-2])
                   & (env[1:-1] >= env[2:]))[0] + 1
    picked = []
    last = -10 ** 9
    min_idx = int(min_interval * sr / hop)
    for i in idx:
        if i - last >= min_idx:
            picked.append(i)
            last = i
    return np.array(picked, dtype=np.int64)


def _bpm(beats_fine, hop_fine, sr):
    if len(beats_fine) < 2:
        return 0.0
    iv = np.diff(beats_fine) * hop_fine / sr
    iv = iv[(iv > 0.1) & (iv < 2.0)]  # 30..600 BPM sanity window
    if len(iv) == 0:
        return 0.0
    return 60.0 / float(np.median(iv))


# ---------------------------------------------------------------------------
# Beat tracking (2026-09-21)
#
# Offline, so none of the live tracker's compromises apply. renderer.py's
# beat_track.py is causal: it can only ever use the past, so it needs a PLL, a
# confidence gate and a flywheel to guess through fills and dropouts. Here the
# whole track is known up front, so we can find the GLOBALLY optimal beat
# sequence instead -- dynamic programming over the entire onset envelope,
# maximising onset strength and tempo regularity at once (Ellis 2007). A fill
# or a quiet bridge is filled in from the surrounding evidence rather than
# guessed at.
#
# What this replaces: `beats` used to be _peak_pick() output, i.e. ONSETS, with
# `bpm` = 60/median(inter-onset interval). Every syncopation and hi-hat counted
# as a beat and there was no tempo grid at all. Nothing consumed either value,
# so there was nothing to keep compatible.
# ---------------------------------------------------------------------------

BEAT_BPM_MIN = 50.0
BEAT_BPM_MAX = 210.0
BEAT_PRIOR_BPM = 120.0
BEAT_PRIOR_SIGMA_OCT = 0.9
BEAT_COMB = ((1, 1.0), (2, 0.5), (3, 0.34), (4, 0.25))
# Ellis' tightness on the log-Gaussian interval penalty. Higher = stricter
# about holding a constant tempo.
BEAT_TIGHTNESS = 100.0
BEAT_ALPHA = 0.8
# Same wrap as the live pipeline's BEAT_WRAP, so a shader sees an identical
# beat clock whether it is running live or being rendered.
BEAT_WRAP = 1024.0


def _autocorr(x):
    """FFT autocorrelation, normalised to 1.0 at lag 0. Direct np.correlate is
    O(N^2) and a 5-minute track is ~28k frames, which is minutes, not seconds."""
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return None
    n = 1 << int(np.ceil(np.log2(2 * len(x))))
    f = np.fft.rfft(x, n)
    ac = np.fft.irfft(f * np.conj(f), n)
    return ac[:len(x)] / denom


def _tempo_period(odf, rate):
    """Global tempo, in frames per beat. Scores each candidate period by the
    autocorrelation at that lag AND its multiples, which is what resolves the
    octave (a half-period scores its own peak, but its harmonics land between
    the real ones), times a log-BPM prior that breaks what is left."""
    ac = _autocorr(odf)
    if ac is None:
        return 0.0
    lag_min = max(2, int(round(60.0 / BEAT_BPM_MAX * rate)))
    lag_max = min(len(ac) - 2, int(round(60.0 / BEAT_BPM_MIN * rate)))
    if lag_max <= lag_min:
        return 0.0

    lags = np.arange(lag_min, lag_max + 1)
    scores = np.zeros(len(lags), dtype=np.float64)
    for h, w in BEAT_COMB:
        idx = lags * h
        ok = idx < len(ac)
        scores[ok] += w * ac[idx[ok]]
    bpms = 60.0 * rate / lags
    scores *= np.exp(-0.5 * (np.log2(bpms / BEAT_PRIOR_BPM) / BEAT_PRIOR_SIGMA_OCT) ** 2)

    i = int(np.argmax(scores))
    if scores[i] <= 0.0:
        return 0.0
    # Sub-frame refinement, so the period is not quantised to whole frames.
    if 0 < i < len(scores) - 1:
        a, b, c = scores[i - 1], scores[i], scores[i + 1]
        d = a - 2.0 * b + c
        off = float(np.clip(0.5 * (a - c) / d, -0.5, 0.5)) if abs(d) > 1e-12 else 0.0
    else:
        off = 0.0
    return float(lags[i]) + off


def _beat_track_dp(odf, period):
    """Ellis dynamic-programming beat tracker. Returns beat frame indices.

    Forward pass: C[i] = onset[i] + max over plausible predecessors j of
    (C[j] + alpha * penalty(gap)), where penalty is a log-Gaussian on
    gap/period -- so landing one period back is free and anything else pays.
    Because every path is scored to the end of the track before any beat is
    committed, a missed kick costs almost nothing: the surrounding beats still
    prefer the grid that keeps the tempo regular."""
    o = np.asarray(odf, dtype=np.float64)
    sd = o.std()
    o = (o - o.mean()) / sd if sd > 0 else o - o.mean()
    n = len(o)
    if n < 4 or period < 2.0:
        return np.array([], dtype=np.int64)

    w_lo = max(1, int(round(period * 0.5)))
    w_hi = max(w_lo + 1, int(round(period * 2.0)))
    gaps = np.arange(w_lo, w_hi + 1, dtype=np.float64)
    penalty = -BEAT_TIGHTNESS * (np.log(gaps / period) ** 2)

    C = np.zeros(n, dtype=np.float64)
    P = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        hi = i - w_lo
        if hi < 0:
            C[i] = o[i]
            continue
        lo = max(i - w_hi, 0)
        # penalty[] is indexed by gap - w_lo; gap runs hi..lo as j runs lo..hi.
        pen = penalty[(i - np.arange(lo, hi + 1)) - w_lo]
        score = C[lo:hi + 1] + BEAT_ALPHA * pen
        k = int(np.argmax(score))
        C[i] = o[i] + score[k]
        P[i] = lo + k

    # Start the backtrace from a strong late beat, not simply the last frame:
    # the track may end on silence, and a fade-out would otherwise anchor the
    # whole grid to noise.
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


def _beat_clock(beat_times, num_frames, fps):
    """Per-video-frame beat count and phase, by interpolating the beat map.

    This is the whole point of doing it offline: the count is EXACT at every
    beat and linear between them, so it cannot drift and needs no PLL. Ends are
    extrapolated with the nearest real interval so the clock runs before the
    first beat and after the last."""
    t = np.arange(num_frames, dtype=np.float64) / fps
    if len(beat_times) < 2:
        return np.zeros(num_frames), np.zeros(num_frames)

    iv = np.diff(beat_times)
    med = float(np.median(iv))
    # Virtual beats outside the real ones, so np.interp extrapolates instead of
    # clamping (a clamped count would freeze motion at both ends of the track).
    pad = max(1, int(np.ceil(t[-1] / max(med, 1e-6))) + 2)
    times = np.concatenate([beat_times[0] - iv[0] * np.arange(pad, 0, -1),
                            beat_times,
                            beat_times[-1] + iv[-1] * np.arange(1, pad + 1)])
    index = np.arange(-pad, len(beat_times) + pad, dtype=np.float64)
    count = np.interp(t, times, index)
    return np.mod(count, BEAT_WRAP), np.mod(count, 1.0)


def _norm01(a):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = np.percentile(a, [1, 99])
    rng = hi - lo
    if rng <= 0:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / rng, 0.0, 1.0).astype(np.float32)


def _envelope(x, fps, attack_ms, release_ms):
    """Asymmetric exponential envelope follower (VU-meter ballistics): fast
    attack so it still catches a beat, slower release so it decays smoothly
    instead of chattering every frame. Without this, feeding a raw per-frame
    feature straight into a shader uniform reads as flicker, not a pulse."""
    x = np.asarray(x, dtype=np.float64)
    a_coef = np.exp(-1.0 / max(fps * attack_ms / 1000.0, 1e-6))
    r_coef = np.exp(-1.0 / max(fps * release_ms / 1000.0, 1e-6))
    y = np.zeros_like(x)
    prev = 0.0
    for i in range(len(x)):
        coef = a_coef if x[i] > prev else r_coef
        prev = coef * prev + (1.0 - coef) * x[i]
        y[i] = prev
    return y.astype(np.float32)


def extract(audio, sr, fps=60.0):
    """audio: mono float32 array. Returns a dict of features + metadata."""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    fps = float(fps)
    duration = n / sr
    hop = max(1, int(round(sr / fps)))
    num_frames = max(1, int(duration * fps))

    # frame-aligned STFT (one column per video frame)
    nperseg = 1 << int(np.ceil(np.log2(max(1024, hop * 4))))
    spec = _stft(audio, nperseg, hop)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sr).astype(np.float32)

    if spec.shape[1] < num_frames:
        spec = np.pad(spec, ((0, 0), (0, num_frames - spec.shape[1])))
    else:
        spec = spec[:, :num_frames]

    # overall RMS per frame (time domain)
    rms = np.zeros(num_frames, dtype=np.float32)
    for t in range(num_frames):
        a = t * hop
        b = min(a + hop, n)
        if b > a:
            rms[t] = np.sqrt(np.mean(audio[a:b] ** 2))

    # band energy (amplitude units)
    band_names = [b[0] for b in DEFAULT_BANDS]
    band_cols = []
    for _name, lo, hi in DEFAULT_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        band_cols.append(np.sqrt((spec[mask, :] ** 2).sum(axis=0)))
    bands = np.stack(band_cols, axis=1)  # (frames, n_bands)

    # spectral centroid
    magsum = spec.sum(axis=0)
    centroid = np.zeros(num_frames, dtype=np.float32)
    nz = magsum > 0
    centroid[nz] = ((freqs[:, None] * spec).sum(axis=0)[nz] / magsum[nz])

    # spectral flux (frame rate)
    flux = np.zeros(num_frames, dtype=np.float32)
    if num_frames > 1:
        flux[1:] = np.maximum(np.diff(spec, axis=1), 0).sum(axis=0)

    # onset / beat detection on a finer grid
    hop_fine = 512
    nperseg_fine = 1024
    spec_fine = _stft(audio, nperseg_fine, hop_fine)
    flux_fine = np.maximum(np.diff(spec_fine, axis=1), 0).sum(axis=0)
    flux_fine = np.concatenate([[0.0], flux_fine])
    k = max(1, int(0.03 * sr / hop_fine))  # 30 ms smoothing
    sm = np.convolve(flux_fine, np.ones(k) / k, mode="same")
    w = max(1, int(0.5 * sr / hop_fine))    # 0.5 s adaptive baseline
    box = np.ones(w) / w
    mean = np.convolve(sm, box, mode="same")
    var = np.convolve(sm ** 2, box, mode="same") - mean ** 2
    thresh = mean + 1.5 * np.sqrt(np.maximum(var, 0))
    onsets_fine = _peak_pick(sm, thresh, hop_fine, sr, min_interval=0.25)

    onsets = (onsets_fine * hop_fine / hop).astype(np.int64)
    onsets = onsets[onsets < num_frames]

    # True beats: a tempo grid, not a list of onsets. See _beat_track_dp.
    rate_fine = sr / hop_fine
    period = _tempo_period(sm, rate_fine)
    beats_fine = _beat_track_dp(sm, period) if period > 0 else np.array([], dtype=np.int64)
    beat_times = beats_fine * hop_fine / sr
    beats = (beats_fine * hop_fine / hop).astype(np.int64)
    beats = beats[beats < num_frames]
    beat_count, beat_phase = _beat_clock(beat_times, num_frames, fps)

    # onset strength per video frame (max smoothed flux within that frame)
    onset = np.zeros(num_frames, dtype=np.float32)
    fidx = np.floor(np.arange(len(sm)) * hop_fine / hop).astype(np.int64)
    np.maximum.at(onset, np.clip(fidx, 0, num_frames - 1), sm.astype(np.float32))

    # BPM from the tracked grid, not from inter-onset intervals.
    bpm = 60.0 * rate_fine / period if period > 0 else 0.0

    names = ["rms"] + band_names + ["centroid", "flux", "onset"]
    feats = np.column_stack([rms, bands, centroid, flux, onset]).astype(np.float32)
    feats_norm = np.zeros_like(feats)
    for i in range(feats.shape[1]):
        feats_norm[:, i] = _norm01(feats[:, i])

    # Envelope-follow before handing features to the shaders: raw per-frame
    # values are jittery at 60fps and any shader multiplying color by them
    # directly reads as flicker. onset gets a fast/punchy attack with a
    # short decay tail (still reads as a "hit"); the continuous bands get a
    # slower, smoother ballistics curve so brightness/color modulation breathes
    # with the music instead of strobing every frame.
    onset_i = names.index("onset")
    feats_norm[:, onset_i] = _envelope(feats_norm[:, onset_i], fps, attack_ms=15, release_ms=180)
    for i, n in enumerate(names):
        if n == "onset":
            continue
        feats_norm[:, i] = _envelope(feats_norm[:, i], fps, attack_ms=60, release_ms=220)

    # Running integral (seconds) of the bass/onset envelopes, for shaders that
    # want the beat to nudge motion *speed* forward. NOT 0..1 -- see
    # wrap_shader.py's module docstring for why this exists (the "audio level
    # multiplied straight into iTime" bug: that rescales the whole elapsed-time
    # clock, so envelope wobble gets amplified by however long the track has
    # been playing). Use additively: iTime*base_rate + mod*this, never
    # iTime*(base_rate + mod*this).
    bass_i = names.index("bass")
    bass_accum = np.cumsum(feats_norm[:, bass_i]) / fps
    onset_accum = np.cumsum(feats_norm[:, onset_i]) / fps
    names = names + ["bass_accum", "onset_accum"]
    feats_norm = np.column_stack([feats_norm, bass_accum, onset_accum]).astype(np.float32)

    return {
        "features": feats,
        "features_norm": feats_norm,
        "names": names,
        "beats": beats,
        "beat_times": beat_times,
        "beat_count": beat_count.astype(np.float32),
        "beat_phase": beat_phase.astype(np.float32),
        "onsets": onsets,
        "bpm": bpm,
        "sample_rate": sr,
        "fps": fps,
        "num_frames": num_frames,
        "duration": duration,
    }


def build_texture(features_norm):
    """(num_features, num_frames) float32 buffer, row = feature, col = time."""
    return np.ascontiguousarray(features_norm.T, dtype=np.float32)


def _click_track(bpm, dur, sr, drop=None, fill=None, seed=1234):
    """Click track at a known tempo, plus a tonal bed so the band checks have
    something to measure. `drop` and `fill` are (start, end) spans of,
    respectively, no clicks at all and clicks at double rate -- the two things
    that used to shove the live detector around. Returns (audio, true_beats)."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    x = (0.3 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    true_beats = np.arange(0.0, dur, period)
    for bt in true_beats:
        if drop is not None and drop[0] <= bt < drop[1]:
            continue
        i0 = int(bt * sr)
        x[i0:i0 + 480] += (0.9 * rng.standard_normal(480)).astype(np.float32)
    if fill is not None:
        for bt in np.arange(fill[0] + period * 0.5, fill[1], period):
            i0 = int(bt * sr)
            x[i0:i0 + 480] += (0.9 * rng.standard_normal(480)).astype(np.float32)
    return x, true_beats


def _grid_error(detected, true_beats, dur):
    """Worst distance, in seconds, from any true beat to the nearest detected
    one. Only beats comfortably inside the track are checked -- the DP grid is
    not expected to place a beat past the last onset evidence."""
    if len(detected) == 0:
        return float("inf")
    inner = true_beats[(true_beats > 0.5) & (true_beats < dur - 0.5)]
    if len(inner) == 0:
        return 0.0
    return float(np.max(np.min(np.abs(inner[:, None] - detected[None, :]), axis=1)))


def selftest():
    sr, fps, dur = 48000, 60.0, 10.0
    ok = True

    x, true_beats = _click_track(120.0, dur, sr)
    res = extract(x, sr, fps)
    beats, bpm = res["beats"], res["bpm"]
    bands = res["features"][:, 1:8]  # sub,bass,lowmid,mid,highmid,presence,brilliance
    bass = float(bands[:, 1].mean())
    mid = float(bands[:, 3].mean())
    presence = float(bands[:, 5].mean())

    print(f"selftest: beats={len(beats)} bpm={bpm:.1f} "
          f"bass={bass:.3f} mid={mid:.3f} presence={presence:.3f}")
    if not (115 <= bpm <= 125):
        print(f"  FAIL: expected bpm ~120, got {bpm:.1f}"); ok = False
    if not (17 <= len(beats) <= 22):
        print(f"  FAIL: expected ~20 beats, got {len(beats)}"); ok = False
    if not (bass > 2 * mid and presence > 2 * mid):
        print("  FAIL: band separation wrong (expected bass+presence >> mid)"); ok = False

    # The beat grid itself. 40 ms is well inside one 512-sample hop pair at
    # 48 kHz (10.7 ms each), so this is a real placement check, not a loose one.
    err = _grid_error(res["beat_times"], true_beats, dur)
    if err > 0.040:
        print(f"  FAIL: beat grid off by {err*1000:.0f} ms"); ok = False
    else:
        print(f"  ok   beat grid within {err*1000:.0f} ms")

    # The two cases that used to shove the causal detector around. Both are
    # checked against the ORIGINAL grid, including through the gap: the DP
    # tracker is expected to carry the beat across a bar with no onsets at all.
    for label, kw in (("dropped bar", {"drop": (4.0, 6.0)}),
                      ("double-time fill", {"fill": (4.0, 6.0)})):
        xx, tb = _click_track(120.0, dur, sr, **kw)
        r = extract(xx, sr, fps)
        e = _grid_error(r["beat_times"], tb, dur)
        good = abs(r["bpm"] - 120.0) <= 3.0 and e <= 0.040
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {label:18s} bpm={r['bpm']:6.1f} "
              f"grid err {e*1000:4.0f} ms")

    # The per-frame clock must be continuous and monotonic: shaders drive
    # position from beat_count, so a jump teleports every body on screen.
    c = res["beat_count"].astype(np.float64)
    step = np.diff(c)
    step = step[step > -BEAT_WRAP / 2]  # ignore the single wrap, if any
    good = bool(np.all(step >= 0.0)) and float(np.max(step)) < 3.0 / (fps * 60.0 / 210.0)
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} clock continuity   "
          f"max step {float(np.max(step)):.4f} beats/frame")

    print("selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="offline audio features for shader music video")
    ap.add_argument("track", nargs="?", help="audio file (any ffmpeg-readable format)")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("-o", "--out", default="features.npz")
    ap.add_argument("--texture", help="also write (F,N) float32 raw buffer for GL upload")
    ap.add_argument("--ffmpeg", help="path to ffmpeg binary")
    ap.add_argument("--selftest", action="store_true", help="run synthetic self-test")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.track:
        ap.error("provide a track, or use --selftest")

    audio, sr = decode_file(args.track, args.sr, args.ffmpeg)
    res = extract(audio, sr, args.fps)

    np.savez_compressed(
        args.out,
        features=res["features"],
        features_norm=res["features_norm"],
        beats=res["beats"],
        beat_times=res["beat_times"],
        beat_count=res["beat_count"],
        beat_phase=res["beat_phase"],
        onsets=res["onsets"],
        names=np.array(res["names"]),
        bpm=res["bpm"],
        fps=res["fps"],
        sample_rate=res["sample_rate"],
        num_frames=res["num_frames"],
    )
    if args.texture:
        build_texture(res["features_norm"]).tofile(args.texture)

    names = res["names"]
    print(f"track:     {args.track}")
    print(f"audio:     {sr} Hz  fps={res['fps']:.0f}  duration={res['duration']:.1f}s  frames={res['num_frames']}")
    print(f"features:  {len(names)} per frame -> {', '.join(names)}")
    print(f"beats:     {len(res['beats'])} tracked  bpm={res['bpm']:.1f}  "
          f"({len(res['onsets'])} raw onsets)")
    print(f"saved:     {args.out}")
    if args.texture:
        print(f"texture:   {args.texture}  (shape {len(names)}x{res['num_frames']} f32, row=feature, col=time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
