"""Silent SONG REPLAY test mode (scope.md 2026-10-20): drive the backdrop from a computed song's offline analysis
instead of the live audio, so shaders can be tuned to real musical structure while the speakers play something
else (a video). Toggle with ~/bin/backdrop-replay; the renderer checks the flag file below.

Flag file ~/.cache/shader-backdrop/replay_song: exists = on. It SHUFFLES through all computed songs (user: "don't do
a chosen one, shuffle through all"): a random analysis, played to its end, then the next random one. Absent = normal
live audio. Published per tick: the beat grid (u_beat_*), the section / look-ahead set (u_ms_*, through
MusicState._cached, the same code a recognised song uses), drum pulses from the analysed kick/snare/hat hits, and a
stand-in audio frame whose rms/bass/onset follow the song's loudness curve (bands stay silent), so the loudness
accumulators (u_rms_accum etc.) keep stepping with the song.
"""

import glob
import math
import os
import random
import types

import numpy as np

import music_state
import song_features

FLAG = os.path.expanduser("~/.cache/shader-backdrop/replay_song")
PULSE_RATE = 5.0      # hit decay per beat (synth_clock's value: 1/e after 0.2 beat)
# loudness -> the plugin's rms scale ((dB + 60) / 60). Calibrated 2026-10-20: live music at the usual playback
# level read rms median 0.53; a typical master sits near -10 LUFS short-term, so rms = (LUFS + 42) / 60.
LUFS_OFFSET = 42.0


class Replay:
    def __init__(self):
        self.path = None
        self.song = None
        self.t0 = 0.0
        self.st = None
        self._checked = -1.0
        self._active = False
        self._mt = None

    def active(self, now):
        """Cheap flag check (4x a second). Loads / switches the song when the flag changes."""
        if now - self._checked < 0.25:
            return self._active
        self._checked = now
        try:
            mt = os.path.getmtime(FLAG)
        except OSError:
            self._active = False
            self.path = None
            return False
        # off -> on, the song ended, or the flag was touched again (backdrop-replay next): the next random song
        if self.path is None or now - self.t0 >= self.song.duration or mt != self._mt:
            self._mt = mt
            return self._next(now)
        self._active = True
        return True

    def _next(self, now):
        files = glob.glob(os.path.join(song_features.FEATURES_DIR, "*.npz"))
        for _ in range(10):                 # skip unreadable / old analyses
            path = random.choice(files) if files else None
            if path is None:
                break
            try:
                self.song = song_features.Song(path)
                z = np.load(path)
                self.lufs = z["curve_loudness_lufs"].astype(np.float32)
                self.hits = {k: np.sort(z["hits_" + k].astype(np.float64)) for k in ("kick", "snare", "hat")}
                self.chroma = z["chroma_beats"].astype(np.float32) if "chroma_beats" in z.files else None   # (12, beats), C = 0
            except Exception as e:
                print(f"song_replay: skip {os.path.basename(path)}: {e}", flush=True)
                continue
            changes = self.st.changes + 1 if self.st else 0      # the change counter keeps counting across songs
            self.path = path
            self.t0 = now
            self.st = types.SimpleNamespace(cache_sec=None, changes=changes, change_t=now)
            print(f"song_replay: replaying {os.path.basename(path)} ({self.song.duration:.0f} s)", flush=True)
            try:                                 # for the shader's plan feed (orrery_kerr disk_sim.py, director plans)
                with open(os.path.expanduser("~/.cache/shader-backdrop/replay_now.txt"), "w") as f:
                    f.write(path + "\n")
            except OSError:
                pass
            self._active = True
            return True
        self._active = False
        return False

    def _pulse(self, name, pos, spb):
        h = self.hits[name]
        i = int(np.searchsorted(h, pos, side="right") - 1)
        if i < 0:
            return 0.0
        return math.exp(-PULSE_RATE * (pos - h[i]) / spb)

    def tick(self, now, n_features):
        """-> (stand-in audio frame, uniform dict)."""
        s = self.song
        pos = min(now - self.t0, s.duration)
        bi = s.beat_index_at(pos)
        b = s.beats
        k = int(np.clip(np.searchsorted(b, pos, side="right") - 1, 0, len(b) - 2))
        spb = max(b[k + 1] - b[k], 0.2)
        u = music_state.MusicState._cached(self.st, s, pos, now, bi)
        kick = self._pulse("kick", pos, spb)
        u.update({
            "u_beat_count": float(bi), "u_beat_phase": float(bi % 1.0), "u_beat_conf": 1.0,
            "u_beat_bpm": float(60.0 / spb),
            "u_kick": float(kick), "u_drum_mid": self._pulse("snare", pos, spb), "u_drum_high": self._pulse("hat", pos, spb),
            "u_stem_drums": 0.0, "u_stem_bass": 0.0, "u_stem_tonal": 0.0,
        })
        lufs = float(self.lufs[int(np.clip(pos * s.fps, 0, len(self.lufs) - 1))])
        rms = float(np.clip((lufs + LUFS_OFFSET) / 60.0, 0.0, 1.0))
        # 2026-10-21: a stand-in SPECTRUM (was silent) so band-driven effects (bass vibration, treble chroma split) work
        # in replay: lows follow the kick hits + the bass-weight curve, highs the hat hits + the brightness curve, the
        # middle the loudness. Same 0..1 band scale as the live plugin (rms sits ~0.4-0.6 there).
        hat = self._pulse("hat", pos, spb)
        weight = s.curve("weight", pos) if "weight" in s.rel else 0.5
        bright = s.curve("brightness", pos) if "brightness" in s.rel else 0.5
        lo = min(1.0, rms * (0.55 + 0.45 * weight) + 0.35 * kick)
        hi = min(1.0, rms * (0.45 + 0.55 * bright) + 0.30 * hat)
        bands = [0.0] * 120
        for i in range(117):                    # 117-119 are dead bands in the plugin too
            x = i / 116.0
            bands[i] = lo * max(0.0, 1.0 - x / 0.4) + rms * 0.8 * max(0.0, 1.0 - abs(x - 0.5) / 0.3) + hi * max(0.0, (x - 0.6) / 0.4)
        if self.chroma is not None and self.chroma.shape[1] > 0:     # the high bands carry the song's real pitch per beat
            ch = self.chroma[:, int(np.clip(bi, 0, self.chroma.shape[1] - 1))]
            ch = ch / max(float(ch.max()), 1e-6)
            for i in range(60, 117):                                 # band i = 27.5 Hz x 2^(i/12): A = 0 -> C-based (i + 9) % 12
                bands[i] *= 0.25 + 0.75 * float(ch[(i + 9) % 12]) ** 2
        frame = {name: 0.0 for name in n_features}
        frame.update({"rms": rms, "bass": lo, "onset": kick, "bands": bands, "timestamp": now})
        return frame, u
