"""The shared "music state" layer (scope.md 2026-10-11 "song cache in shaders:
ONE live-capable interface, the cache only makes it better").

Every shader gets the SAME uniforms whether the source is live audio or the
offline song analysis (song_features.py); the cache only makes them more
accurate and adds an optional look-ahead. Updated in the renderer's audio
reader thread at the 60 Hz stem cadence, on the delay-aligned frame -- the
same moment the shader is showing.

Uniforms (all floats / vec4; -1 = unknown, which a shader must handle):
  u_ms_source      0 live, 1 cache
  u_ms_cur_a       energy, weight, density, brightness   (0..1, song-relative, now)
  u_ms_cur_b       pulse, voice, tension, impact          (tension: cache only, else -1)
  u_ms_sec_a/b     the same 8 axes for the CURRENT section
  u_ms_next_a/b    the next section's 8 axes (cache only, else -1)
  u_ms_change      section-change COUNTER (+1 per change; compare with a stored copy --
                   a pulse could be missed at a low frame rate)
  u_ms_change_age  seconds since the last change
  u_ms_next_in     beats until the next change (cache only, else -1)
  u_ms_drop_in     seconds until the next drop (cache only, else -1)
  u_ms_bar_phase   0..1 through the bar (cache: real downbeats; live: beat count / 4
                   from an arbitrary start)
  u_ms_repeat      section repeat id (cache only, else -1)
  u_ms_song_pos    seconds into the song as heard (cache only, else -1)

Live features (scope: "values we can also technically get live"):
  energy     broadband rms level (dB-mapped)        weight   sub+bass energy share
  density    drum onsets per second (kick/mid/high) brightness  spectral centroid (log-Hz)
  pulse      the beat tracker's confidence          voice    tonal-stem energy in the
                                                             vocal range (rough)
Each live axis is scaled against THIS song so far (a slow running range with a
per-axis minimum span, reset on a song change or a real silence) so 0..1 means
the same thing as the cache's song-relative scores. A live section change is a
sustained jump of the short-term values away from the current section's mean,
with a minimum section length (hysteresis), applied on a beat.
"""

import json
import math
import os
import threading
from collections import deque

import numpy as np

import song_features
import synth_clock

DUMP_PATH = os.path.expanduser("~/.cache/shader-backdrop/music_state.json")
# where the beat gate (and the user) mark non-music: <basename>{.notmusic,.excluded}
NOT_MUSIC_DIR = os.path.expanduser("~/.local/share/shader-backdrop/not_music")
AXES = ("energy", "weight", "density", "brightness", "pulse", "voice", "tension", "impact")
LIVE_AXES = AXES[:6]
# smallest live span that counts as a real change (live units)
LIVE_MIN_RANGE = {"energy": 0.15, "weight": 0.10, "density": 1.5, "brightness": 0.06,
                  "pulse": 0.20, "voice": 0.08}
RANGE_TAU_S = 45.0          # the running range relaxes toward the mean over ~this
CUR_TAU_S = 0.5             # u_ms_cur smoothing
PULSE_TAU_S = 2.5           # the live beat confidence flickers 0<->1 within seconds
WARMUP_S = 4.0              # after a song start/reset: ranges settle, no section stats yet
SHORT_TAU_S = 2.0           # short-term window the change detector compares
NOVELTY_ON = 0.12           # distance (0..1 axes) that counts as a change ... (tuned 2026-10-11 on a
                            # replay of a live log: one change per analysed boundary, 3-6 s late)
NOVELTY_HOLD_S = 0.6        # ... if it persists this long
MIN_SECTION_BEATS = 12      # hysteresis: no change sooner than this after the last
SILENCE_LEVEL = 0.05        # energy (dB-mapped) below this is silence ...
SILENCE_S = 1.5             # ... and resets the song-relative ranges after this long
ONSET_ON = 0.5              # drum weight rising through this = one onset
ONSET_REFRACT_S = 0.08
DENSITY_WIN_S = 2.0

# -- music-vs-speech gate (scope.md 2026-10-13 "synthetic motion clock") ----
# Live-observed: on a non-song video the beat tracker's conf flickers and the
# section detector runs away. The raw band features separate cleanly though
# (measured 2026-10-13, Contra la Luna vs a stream speech segment): music has a
# strong onset-autocorrelation peak AND a steady level; speech has neither.
# So gate on periodicity * level-stability, computed off the shm meta the frame
# already carries -- NOT the tracker's conf (which overlaps between the two).
# Measured on the 2026-10-13 captures: level stability separates cleanly
# (music rms-cv ~0.05, speech ~0.3+), but onset periodicity is weak over a short
# window (music ac-peak ~0.22 at 4s vs 0.53 over 22s), so it only MODULATES the
# steady term rather than gating it -- otherwise real music scores ~0 and would
# be switched to synthetic.
GATE_WIN_S = 4.0            # rolling window the classifier looks at
GATE_MIN_N = 24             # ... no verdict until this many samples (hold last)
AC_LAG_MIN_S = 0.30         # onset-autocorrelation lags = 60..200 BPM
AC_LAG_MAX_S = 1.00
GATE_AC_LO = 0.15           # smoothstep on the autocorr peak (modulator)
GATE_AC_HI = 0.35
GATE_CV_LO = 0.08           # smoothstep on SMOOTHED rms-cv, reversed: LOW cv = music
GATE_CV_HI = 0.25           # 2026-10-15: was 0.16. Measured live on a boom-bap track: the raw rms-cv
                            # straddled 0.16, so the score slammed 0.00 <-> 0.92 and the gate latched
                            # to synthetic. Widened, and the rms is low-passed first (see _classify).
GATE_RMS_TAU = 1.0          # s: low-pass on the level BEFORE measuring cv. The raw rms carries the
                            # BEAT (kick up, gap down), so a hard 4/4 read as "unstable level" i.e.
                            # speech -- the clearer the beat, the more likely it was called non-music.
GATE_ENTER = 0.30           # music_score below this, held, -> synthetic
GATE_EXIT = 0.45            # 2026-10-15: was 0.60, unreachably high -- on beat-driven music the score
                            # only pokes above 0.60 in isolated samples, so ONE trip = the whole track.
GATE_DOWN_S = 8.0           # 2026-10-15: was 4.0. Must exceed GATE_WIN_S (4.0) by more than the
                            # interruption it should survive: at 4.0 vs 4.0 a 1-2 s burst of speech
                            # keeps the window contaminated for exactly as long as the hold, so three
                            # words before the music could trip it. 8 s sits well outside that.
GATE_UP_S = 1.0             # 2026-10-15: was 2.0 -- how long "clearly music" must hold to come back.
GATE_MIN_TRACK_S = 30.0     # s: shortest "meta:" length that still counts as a real recording


def _smoothstep(lo, hi, x):
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _e(x):
    """dB-normalised level (norm_db, -60 dB floor) -> linear energy."""
    return 10.0 ** (6.0 * (np.asarray(x, dtype=np.float64) - 1.0))


class _Range:
    """Song-relative scaling: a running lo/hi that snaps out to new extremes and
    slowly relaxes inward, with a minimum span."""

    def __init__(self, min_span):
        self.min_span = min_span
        self.lo = self.hi = None

    def reset(self):
        self.lo = self.hi = None

    def rel(self, x, dt):
        if self.lo is None:
            self.lo, self.hi = x - 0.5 * self.min_span, x + 0.5 * self.min_span
        a = 1.0 - math.exp(-dt / RANGE_TAU_S)
        mid = 0.5 * (self.lo + self.hi)
        self.lo = min(x, self.lo + (mid - self.lo) * a)
        self.hi = max(x, self.hi + (mid - self.hi) * a)
        span = max(self.hi - self.lo, self.min_span)
        c = 0.5 * (self.lo + self.hi)
        return min(max((x - (c - 0.5 * span)) / span, 0.0), 1.0)


class MusicState:
    def __init__(self, n_bands, np_source=None, delay_fn=None):
        self.n_bands = n_bands
        self.np = np_source              # nowplaying.NowPlaying (shared with beat_session)
        self.delay_fn = delay_fn or (lambda: 0.0)
        centres = 27.5 * 2.0 ** (np.arange(n_bands) / 12.0)
        self.vocal = (centres >= 300.0) & (centres <= 3400.0)
        self.ranges = {k: _Range(LIVE_MIN_RANGE[k]) for k in LIVE_AXES}
        self.cur = {k: 0.5 for k in LIVE_AXES}
        self.short = {k: 0.5 for k in LIVE_AXES}
        self.sec_sum = {k: 0.0 for k in LIVE_AXES}
        self.sec_n = 0
        self.sec_vals = {k: 0.5 for k in LIVE_AXES}
        self.impact = 0.0
        self.changes = 0
        self.change_t = 0.0
        self.change_beat = None
        self.novel_since = None
        self.pending_change = False
        self.onset_t = {"kick": -1.0, "mid": -1.0, "high": -1.0}
        self.onset_hist = []
        self.prev_w = {"kick": 0.0, "mid": 0.0, "high": 0.0}
        self.silent_since = None
        self.song_t = 0.0
        self.last_t = None
        # music/speech gate + the single published beat clock
        self.music_score = 1.0
        self.mode = "live"          # "live" | "synth"
        self.pub_count = None
        self.pub_off = 0.0          # live: published count = tracker count + this WHOLE-beat offset
        self._pub_synth = False     # was the last published frame synthetic?
        self.synth_origin = 0.0     # synthetic song: count at which it started (a beat) ...
        self.synth_seed = 0         # ... and which song (new one per synthetic stretch)
        self._below = None
        self._above = None
        self._was_synth = False
        self.gate_hist = deque()
        # cache
        self.key = None
        self.song = None
        self._ident = False         # player named a real recording -> gate is not the right evidence
        self.gate_mode = None       # what the music/speech gate WANTED (diagnostic; not used for the clock)
        self.loading = None
        self.cache_sec = None
        self._dump_t = 0.0

    # -- song (cache) -------------------------------------------------------

    def _load(self, key):
        # Runs in a fresh thread per song: open the index HERE. (A connection kept from the
        # first song's thread raised "SQLite objects created in a thread can only be used in
        # that same thread" on every later song, swallowed below -- so only the first song
        # after a renderer start ever got its analysis; fixed 2026-10-11.)
        db = None
        try:
            db = song_features.connect()
            path = song_features.lookup(db, key)
            song = song_features.Song(path) if path else None
        except Exception as e:
            print(f"music_state: no analysis for {key}: {e}", flush=True)
            song = None
        finally:
            if db is not None:
                db.close()
        if self.key == key:
            self.song = song
            self.cache_sec = None

    def _track(self):
        if self.np is None:
            return None, None
        track, pos = self.np.current()
        if track is None:
            if self.key is not None:
                self.key, self.song = None, None
            return None, None
        if track["key"] != self.key:
            self.key, self.song = track["key"], None
            self._ident = self._music_identity(track["key"])   # cached per track (no per-frame stat)
            self._reset_song()
            # A NEW TRACK starts on the audio gate from scratch (2026-10-15): never inherit the last
            # track's latch. Without this, one bad verdict poisoned every subsequent track.
            self.mode = "live"
            self._below = self._above = None
            self.gate_hist.clear()
            # the npz load must not stall the audio thread
            threading.Thread(target=self._load, args=(track["key"],), daemon=True).start()
        return track, pos - self.delay_fn()

    def _music_identity(self, key):
        """True when the PLAYER TOLD US what is on AND it is a recording -- the case where the audio
        gate is simply the wrong evidence. Key kinds are nowplaying.identity_key():

          file:<path>                        a local recording -> music, unless marked not-music
          meta:<artist>|<title>|<length_us>  a named recording -> music if the length is real
          yt:<videoid>                       a YouTube video   -> NOT overridden, see below

        `yt:` is deliberately excluded: a video id is exactly the "random audio" case this gate and
        the synth clock exist for. A stream never qualifies either -- it has no length, so it lands
        as `meta:..|0` and fails the length test.
        """
        k = key or ""
        if k.startswith("file:"):
            base = os.path.basename(k[5:])
            for suffix in (".notmusic", ".excluded"):
                if os.path.exists(os.path.join(NOT_MUSIC_DIR, base + suffix)):
                    return False        # already marked not-music by the beat gate (or by hand)
            return True
        if k.startswith("meta:"):
            p = k[5:].split("|")
            if len(p) < 3 or not p[0].strip() or not p[1].strip():
                return False
            try:
                return int(p[2]) >= int(GATE_MIN_TRACK_S * 1e6)
            except ValueError:
                return False
        return False

    def _reset_song(self):
        self.song_t = 0.0
        for r in self.ranges.values():
            r.reset()
        self.sec_n = 0
        self.sec_sum = {k: 0.0 for k in LIVE_AXES}
        self.change_beat = None

    # -- live ---------------------------------------------------------------

    def _onsets(self, kick_w, t):
        for name, w in kick_w.items():
            if w >= ONSET_ON > self.prev_w[name] and t - self.onset_t[name] > ONSET_REFRACT_S:
                self.onset_t[name] = t
                self.onset_hist.append(t)
            self.prev_w[name] = w
        cut = t - DENSITY_WIN_S
        while self.onset_hist and self.onset_hist[0] < cut:
            self.onset_hist.pop(0)
        return len(self.onset_hist) / DENSITY_WIN_S

    def _live_raw(self, frame, tonal, scalars, t):
        rms = min(1.0, max(0.0, 1.5 * float(frame.get("rms", 0.0)) - 0.5))
        e_rms = float(_e(frame.get("rms", 0.0))) + 1e-12
        weight = float((_e(frame.get("sub", 0.0)) + _e(frame.get("bass", 0.0))) / e_rms)
        hz = max(20.0, float(frame.get("centroid", 0.0)) * 28160.0)
        bright = min(1.0, math.log2(hz / 20.0) / math.log2(16000.0 / 20.0))
        density = self._onsets({"kick": scalars.get("u_kick", 0.0),
                                "mid": scalars.get("u_drum_mid", 0.0),
                                "high": scalars.get("u_drum_high", 0.0)}, t)
        pulse = float(scalars.get("u_beat_conf", 0.0))
        bands = np.asarray(frame.get("bands", ()), dtype=np.float64)
        if tonal is not None and len(bands) == self.n_bands:
            e_t = _e(np.clip(tonal, 0, 1))
            voice = float(e_t[self.vocal].sum() / (_e(np.clip(bands, 0, 1)).sum() + 1e-12))
        else:
            voice = 0.0
        return {"energy": rms, "weight": min(weight, 1.0), "density": density,
                "brightness": bright, "pulse": pulse, "voice": min(voice, 1.0)}

    def _live(self, raw, dt, t, beat_count):
        if raw["energy"] < SILENCE_LEVEL:
            self.silent_since = self.silent_since or t
            if t - self.silent_since >= SILENCE_S:
                self._reset_song()
            return
        self.silent_since = None
        a_cur = 1.0 - math.exp(-dt / CUR_TAU_S)
        a_short = 1.0 - math.exp(-dt / SHORT_TAU_S)
        a_pulse = 1.0 - math.exp(-dt / PULSE_TAU_S)
        for k in LIVE_AXES:
            r = self.ranges[k].rel(raw[k], dt)
            self.cur[k] += (r - self.cur[k]) * (a_pulse if k == "pulse" else a_cur)
            self.short[k] += (r - self.short[k]) * a_short
        self.song_t += dt
        if self.song_t < WARMUP_S:
            return
        for k in LIVE_AXES:
            self.sec_sum[k] += self.cur[k]
        self.sec_n += 1
        self.sec_vals = {k: self.sec_sum[k] / self.sec_n for k in LIVE_AXES}
        if self.change_beat is None:
            self.change_beat = beat_count
        # change detection: short-term vs this section's mean, held, hysteresis
        # pulse is left out: the live beat confidence is too jumpy to vote
        w = {"energy": 1.0, "weight": 1.0, "density": 0.8, "brightness": 0.6, "pulse": 0.0,
             "voice": 0.5}
        nov = math.sqrt(sum(w[k] * (self.short[k] - self.sec_vals[k]) ** 2 for k in LIVE_AXES)
                        / sum(w.values()))
        old_enough = beat_count - self.change_beat >= MIN_SECTION_BEATS and self.sec_n > 60
        if nov > NOVELTY_ON and old_enough:
            self.novel_since = self.novel_since or t
            if t - self.novel_since >= NOVELTY_HOLD_S:
                self.pending_change = True
        else:
            self.novel_since = None
        # apply on the next beat
        if self.pending_change and (beat_count % 1.0) < 0.25:
            self.pending_change = False
            self.novel_since = None
            self.impact = min(max(2.0 * (self.short["energy"] - self.sec_vals["energy"]), 0.0), 1.0)
            self.changes += 1
            self.change_t = t
            self.change_beat = beat_count
            self.sec_sum = {k: self.short[k] for k in LIVE_AXES}
            self.sec_n = 1

    # -- music/speech gate --------------------------------------------------

    def _classify(self, frame, t):
        """music_score 0..1 from raw band features over the rolling window:
        onset-autocorrelation peak (periodicity) x level stability."""
        self.gate_hist.append((t, float(frame.get("rms", 0.0)),
                               float(frame.get("onset", 0.0))))
        while self.gate_hist and t - self.gate_hist[0][0] > GATE_WIN_S:
            self.gate_hist.popleft()
        n = len(self.gate_hist)
        if n < GATE_MIN_N:
            return self.music_score               # warmup: hold the last verdict
        rms = np.fromiter((x[1] for x in self.gate_hist), float, n)
        onset = np.fromiter((x[2] for x in self.gate_hist), float, n)
        dt_avg = (self.gate_hist[-1][0] - self.gate_hist[0][0]) / max(n - 1, 1)
        # Low-pass the level before measuring stability (2026-10-15). The raw rms carries the BEAT
        # itself -- kick lifts it, the gap drops it -- so a hard-hitting 4/4 scored as an unstable
        # level and was classified as speech: the clearer the beat, the more likely "not music".
        # GATE_RMS_TAU is longer than a beat period, so the pulse is removed and what is left is
        # what this test actually wants -- how steady the TRACK is, not how hard it hits.
        a = math.exp(-dt_avg / GATE_RMS_TAU) if dt_avg > 0 else 0.0
        sm = np.empty(n)
        acc = float(rms[0])
        for i in range(n):
            acc = float(rms[i]) + a * (acc - float(rms[i]))
            sm[i] = acc
        mean = sm.mean()
        cv = sm.std() / mean if mean > 1e-6 else 1.0
        o = onset - onset.mean()
        denom = float(o @ o)
        peak = 0.0
        if denom > 1e-9 and dt_avg > 0:
            ac = np.correlate(o, o, "full")[n - 1:] / denom
            lo = int(AC_LAG_MIN_S / dt_avg)
            hi = min(int(AC_LAG_MAX_S / dt_avg), len(ac) - 1)
            if hi > lo:
                peak = float(ac[lo:hi + 1].max())
        steady = 1.0 - _smoothstep(GATE_CV_LO, GATE_CV_HI, min(cv, 1.0))
        p_term = _smoothstep(GATE_AC_LO, GATE_AC_HI, peak)
        return steady * (0.7 + 0.3 * p_term)

    def _gate(self, score, t):
        if score < GATE_ENTER:
            self._below = self._below or t
            self._above = None
            if self.mode != "synth" and t - self._below >= GATE_DOWN_S:
                self.mode = "synth"
                self._was_synth = True
        elif score > GATE_EXIT:
            self._above = self._above or t
            self._below = None
            if self.mode != "live" and t - self._above >= GATE_UP_S:
                self.mode = "live"
        else:
            self._below = None
            self._above = None

    def _synth_advance(self, dt, t):
        """Advance the published count on the synthetic song; a switch INTO synthetic
        starts a new song at the next whole beat (sections start on a beat)."""
        if self.pub_count is None:
            self.pub_count = 0.0
        if not self._pub_synth:
            self.synth_origin = float(math.ceil(self.pub_count))
            self.synth_seed = int(t * 1000.0) & 0xFFFFFF
        pos = max(self.pub_count - self.synth_origin, 0.0)
        self.pub_count += dt * synth_clock.bpm_at(pos, self.synth_seed) / 60.0
        self._pub_synth = True

    def _synthetic(self, pbc, t):
        """The full uniform set while the gate is on the synthetic clock (the
        synthetic song publishes the whole u_ms_* set itself, look-ahead included)."""
        u = synth_clock.uniforms_at(pbc, self.synth_origin, self.synth_seed)
        self.synth_section = u.pop("synth_section")
        return u

    # -- main ---------------------------------------------------------------

    def update(self, frame, tonal, scalars, t):
        return self._dumped(self._update(frame, tonal, scalars, t), t)

    def idle(self, t):
        """No fresh audio at all (sink idle / nothing playing): the per-frame
        update never runs then, so the backdrop would freeze on its last values.
        Feed the gate true silence instead -- it scores as non-music, and after
        the GATE_DOWN_S hold the synthetic clock takes over. Returns the
        synthetic uniform set, or None while still holding the last live values."""
        dt = 1.0 / 60.0 if self.last_t is None else min(max(t - self.last_t, 1e-3), 0.25)
        self.last_t = t
        self.music_score = self._classify({"rms": 0.0, "onset": 0.0}, t)
        # No override in this path -- and note the override keys on "we MATCHED AN ANALYSIS"
        # (self.song) or "the player named a recording" (self._ident), NOT on "something is playing".
        # The idle path has neither, so the gate's verdict stands and the synth clock takes over.
        self._gate(self.music_score, t)
        self.gate_mode = self.mode          # here synth means "no audio at all", not "not music"
        if self.mode != "synth":
            return None
        self._synth_advance(dt, t)
        return self._dumped(self._synthetic(self.pub_count, t), t)

    def _dumped(self, out, t):
        if t - self._dump_t >= 0.5:          # readout for checking it by eye
            self._dump_t = t
            try:
                d = dict(out, key=self.key, song=(self.song.path if self.song else None),
                         music_score=self.music_score, mode=self.mode, gate_mode=self.gate_mode,
                         identified=self._ident,
                         synth_section=(getattr(self, "synth_section", None) if self.mode == "synth" else None))
                tmp = DUMP_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(d, f, indent=1)
                os.replace(tmp, DUMP_PATH)
            except OSError:
                pass
        return out

    def _update(self, frame, tonal, scalars, t):
        """frame: the delay-aligned band frame dict (bands + meta features);
        tonal: the stem splitter's tonal row; scalars: this sample's beat/kick
        scalars; t: monotonic seconds. Returns the uniform dict."""
        dt = 1.0 / 60.0 if self.last_t is None else min(max(t - self.last_t, 1e-3), 0.25)
        self.last_t = t
        tracker_count = float(scalars.get("u_beat_count", 0.0))
        # music/speech gate, then advance the ONE published beat clock. Rate-
        # following (not value-copying) keeps it continuous across mode changes.
        # THE PUBLISHED CLOCK IS ALWAYS THE MUSIC'S WHEN THERE IS AUDIO (2026-10-15 user: "uses
        # anything at least related to music").
        #
        # _update() only runs when there IS fresh audio, so in here the count ALWAYS follows the live
        # beat tracker -- i.e. u_beat_count/u_beat_bpm come from the music itself. The music/speech
        # gate is still computed and published (music_score, and gate_mode showing what it WANTED) so
        # it stays diagnosable, but it no longer governs the clock.
        #
        # The synthetic clock is therefore reached ONLY from idle() -- sink idle, nothing playing --
        # which is what it was written for ("the backdrop would freeze on its last values").
        #
        # Consequence, stated plainly: a podcast / stream / unmarked spoken-word file is now
        # beat-tracked off its own audio instead of getting the synthetic clock. To keep such content
        # off the music path, mark it (.notmusic / .excluded) -- that IS honoured, in _music_identity.
        self.music_score = self._classify(frame, t)
        self._gate(self.music_score, t)
        self.gate_mode = self.mode          # what the gate wanted -- diagnostic only
        self.mode = "live"
        self._was_synth = False
        self._below = self._above = None
        if self.pub_count is None:
            self.pub_count = tracker_count
        if self.mode == "synth":
            # free-running from wherever live left off (continuous, no jump)
            self._synth_advance(dt, t)
        else:
            # LIVE = the tracker's own count, so its phase corrections and the
            # cached-song lock (set_reference) reach the shaders. Coming back from
            # synthetic, re-pick a WHOLE-beat offset so the count stays continuous
            # (one step of at most half a beat) and the phase is the music's again.
            if self._pub_synth:
                self.pub_off = float(round(self.pub_count - tracker_count))
            self.pub_count = tracker_count + self.pub_off
            self._pub_synth = False
        pbc = self.pub_count
        if self.mode == "synth":
            return self._synthetic(pbc, t)
        if self._was_synth:
            self.change_beat = None       # re-anchor after a synthetic stretch
            self._was_synth = False
        self._live(self._live_raw(frame, tonal, scalars, t), dt, t, pbc)
        track, pos = self._track()
        song = self.song
        if song is not None and pos is not None and 0.0 <= pos <= song.duration:
            out = self._cached(song, pos, t, pbc)
        else:
            cur = [self.cur[k] for k in LIVE_AXES]
            sec = [self.sec_vals[k] for k in LIVE_AXES]
            out = {
                "u_ms_source": 0.0,
                "u_ms_cur_a": tuple(cur[:4]), "u_ms_cur_b": (cur[4], cur[5], -1.0, self.impact),
                "u_ms_sec_a": tuple(sec[:4]), "u_ms_sec_b": (sec[4], sec[5], -1.0, self.impact),
                "u_ms_next_a": (-1.0,) * 4, "u_ms_next_b": (-1.0,) * 4,
                "u_ms_change": float(self.changes), "u_ms_change_age": t - self.change_t,
                "u_ms_next_in": -1.0, "u_ms_drop_in": -1.0,
                "u_ms_bar_phase": (pbc % 4.0) / 4.0,
                "u_ms_repeat": -1.0, "u_ms_song_pos": -1.0,
            }
        out["u_beat_count"] = pbc
        out["u_beat_phase"] = pbc % 1.0
        return out

    def _cached(self, song, pos, t, beat_count):
        i = song.section_at(pos)
        # Count a change only on a forward step (or a real jump, i.e. a seek): the
        # position jitters back and forth across a boundary for a moment and every
        # flip used to count (the counter jumped by 2).
        if self.cache_sec is None:
            self.cache_sec = i
        elif i > self.cache_sec or (i < self.cache_sec and pos < song.starts[self.cache_sec] - 1.0):
            self.changes += 1
            self.change_t = t
            self.cache_sec = i
        else:
            i = self.cache_sec
        sec = song.sections[i]
        sc = [sec["score"][k] for k in AXES]
        nxt = song.sections[i + 1] if i + 1 < len(song.sections) else None
        nsc = [nxt["score"][k] for k in AXES] if nxt else [-1.0] * 8
        bi = song.beat_index_at(pos)
        next_in = (nxt["start_beat"] - bi) if nxt else -1.0
        cur = [song.curve(k, pos) for k in LIVE_AXES]
        drops = song.drops[song.drops > pos]
        drop_in = float(drops[0] - pos) if len(drops) else -1.0
        db = song.downbeats
        k = int(np.searchsorted(db, pos, side="right") - 1)
        if 0 <= k < len(db) - 1:
            bar_phase = (pos - db[k]) / max(db[k + 1] - db[k], 1e-3)
        else:
            bar_phase = (bi % song.bpb) / song.bpb
        return {
            "u_ms_source": 1.0,
            "u_ms_cur_a": tuple(cur[:4]), "u_ms_cur_b": (cur[4], cur[5], sc[6], sc[7]),
            "u_ms_sec_a": tuple(sc[:4]), "u_ms_sec_b": tuple(sc[4:]),
            "u_ms_next_a": tuple(nsc[:4]), "u_ms_next_b": tuple(nsc[4:]),
            "u_ms_change": float(self.changes), "u_ms_change_age": t - self.change_t,
            "u_ms_next_in": float(next_in), "u_ms_drop_in": drop_in,
            "u_ms_bar_phase": float(bar_phase),
            "u_ms_repeat": float(sec.get("repeat_id", -1)), "u_ms_song_pos": float(pos),
        }
