"""Per-song orchestration: recognise, recall, record, re-analyse.

The loop this implements, once per song, with no manual step anywhere:

    song starts   -> ask the DB whether we have heard it before
      cache hit   -> steer the live tracker with the stored grid from bar one
      cache miss  -> track live as usual, and record the onset history
    song ends     -> re-analyse that recorded history NON-causally and store it

So the first listen costs nothing and the second is locked. The audio file is
never needed: what gets stored is derived from what we heard.

Threading: everything here is driven from renderer.py's audio reader thread and
must only be touched from it. `NowPlaying` owns its own polling thread and is
the only part that is cross-thread; it is read through a lock.
"""

import threading
import time

import numpy as np

import beat_cache
import beat_track
import nowplaying

# Below this many recorded samples a song is not worth analysing (a skipped
# track, or the tail of one that was already playing when the renderer started).
MIN_RECORD_SAMPLES = 900          # 15 s at 60 Hz
# A map is only stored or trusted if its grid actually spans the song. The
# fragments this rejects used to pass a 0.70 gate because coverage divided by
# the recording window, not the track (OPEN_THREADS 6a); measured over 125
# rows, fragments topped out at 0.81 and real full listens started at 0.95,
# so 0.90 sits in the gap.
FULL_COVERAGE = 0.90
# ...and its beats must stand clear of the average ODF (OPEN_THREADS 6b).
# Garbage grids score ~0.06 at p99; real grids measured 0.14-0.26.
CONF_MIN = 0.10
# A `meta:` key is fuzzy, so the durations must agree before a stored map is
# trusted. `yt:` and `file:` keys are exact and skip this.
DURATION_TOL_S = 2.0


def map_lookup(beat_times, t, median_iv=None):
    """(beat count, phase, tbeat) for song position `t` against a beat map.

    Linear between beats, extrapolated with the nearest real interval outside
    them, so the clock still runs over an intro before the first detected beat
    and an outro after the last."""
    bt = np.asarray(beat_times, dtype=np.float64)
    if bt.size < 2:
        return None
    iv = np.diff(bt)
    med = median_iv if median_iv is not None else float(np.median(iv))
    if med <= 0.0:
        return None

    if t <= bt[0]:
        count = -(bt[0] - t) / max(iv[0], 1e-6)
        tbeat = iv[0]
    elif t >= bt[-1]:
        count = (bt.size - 1) + (t - bt[-1]) / max(iv[-1], 1e-6)
        tbeat = iv[-1]
    else:
        i = int(np.searchsorted(bt, t) - 1)
        i = max(0, min(i, iv.size - 1))
        tbeat = iv[i]
        count = i + (t - bt[i]) / max(tbeat, 1e-6)
    return count, count % 1.0, tbeat


class BeatSession:
    def __init__(self, tracker, delay_fn=None, db_path=None, enabled=True):
        self.tracker = tracker
        self.delay_fn = delay_fn or (lambda: 0.0)
        self.enabled = enabled
        self.np = nowplaying.NowPlaying().start() if enabled else None
        self._db_path = db_path
        self._db = None                # opened lazily, in the calling thread

        self.key = None
        self.track = None
        self._rec_t = []
        self._rec_o = []
        self._tainted = False          # a seek makes the recording unusable
        self._map = None               # (beat_times, median interval)
        self._map_meta = None
        self.last_status = "idle"

    # -- lifecycle ----------------------------------------------------------

    def _db_conn(self):
        if self._db is None:
            self._db = beat_cache.connect(self._db_path)
        return self._db

    def close(self):
        if self.np:
            self.np.stop()
        self._finalise()
        if self._db is not None:
            self._db.close()
            self._db = None

    # -- per-sample ---------------------------------------------------------

    def update(self):
        """Call once per ODF sample (the stem split's 60 Hz cadence), after
        `tracker.update(...)`. Cheap: a lock-read and some arithmetic."""
        if not self.enabled:
            return
        try:
            self._update()
        except Exception as e:
            # A media-player or DB problem must never stop the backdrop.
            self.last_status = f"error: {e}"
            self.tracker.clear_reference()

    def _update(self):
        for kind, key in self.np.drain_events():
            if kind == "end":
                self._finalise()
            elif kind == "start":
                self._begin(key)
            elif kind == "seek":
                # The position<->onset mapping we have been recording is now
                # wrong, so this song cannot be stored. A RECALLED map is
                # still fine: it is in song-position coordinates and the new
                # position is simply looked up.
                self._tainted = True

        track, pos = self.np.current()
        if track is None:
            self.tracker.clear_reference()
            self.last_status = "nothing playing"
            return

        if self.key is None:
            self._begin(track["key"], track)

        # The ODF sample the tracker just produced came from the delay-aligned
        # audio frame, so its song position is the player's position minus the
        # same delay the shader is already being fed at.
        t = pos - self.delay_fn()

        if self._map is not None:
            got = map_lookup(self._map[0], t, self._map[1])
            if got is not None:
                _, phase, tbeat = got
                self.tracker.set_reference(phase, tbeat)
                self.last_status = f"locked from cache ({self._map_meta['bpm']:.1f} BPM)"
        else:
            self.tracker.clear_reference()
            if t >= 0.0:
                self._rec_t.append(t)
                self._rec_o.append(self.tracker.last_odf)
            self.last_status = f"learning ({len(self._rec_t)} samples)"

    # -- song boundaries ----------------------------------------------------

    def _begin(self, key, track=None):
        if self.key == key:
            return
        self._finalise()
        self.key = key
        self.track = track
        self._rec_t, self._rec_o = [], []
        self._tainted = False
        self._map = self._map_meta = None

        if track is None:
            track, _ = self.np.current()
        # Assign AFTER the fallback lookup, or a song begun from a 'start'
        # event (which carries only the key) is stored with no title/artist.
        self.track = track
        try:
            db = self._db_conn()
            got = beat_cache.load_beatmap(db, key)
        except Exception:
            got = None
        if got is None:
            return

        times, meta = got
        if times.size < 2:
            return
        # Hard drop on recall too: a stored fragment must never drive beats.
        # Legacy rows predate the confidence field (NULL); those pass on
        # coverage alone until the song is re-learned.
        cov = meta.get("coverage")
        conf = meta.get("confidence")
        if cov is None or cov < FULL_COVERAGE:
            return
        if conf is not None and conf < CONF_MIN:
            return
        # A fuzzy key must agree on duration before its map is trusted; an
        # exact key (yt:/file:) does not need the check.
        if key.startswith("meta:") and track and track.get("duration_s"):
            span = float(times[-1]) if times.size else 0.0
            if abs(track["duration_s"] - span) > max(DURATION_TOL_S,
                                                     0.1 * track["duration_s"]):
                return
        self._map = (times, float(np.median(np.diff(times))))
        self._map_meta = meta

    def _finalise(self):
        """Song ended: re-analyse what we recorded and store it."""
        key, ts, os_ = self.key, self._rec_t, self._rec_o
        track, tainted = self.track, self._tainted
        self.key = self.track = None
        self._rec_t, self._rec_o = [], []
        self._map = self._map_meta = None
        self._tainted = False
        self.tracker.clear_reference()

        if key is None or tainted or len(ts) < MIN_RECORD_SAMPLES:
            return
        out = beat_track.reanalyse(os_, ts,
                                   duration_s=(track or {}).get("duration_s"))
        if out is None:
            return
        beat_times, bpm, coverage, confidence = out
        # Hard drop: an incomplete grid, or one whose beats do not stand clear
        # of the ODF, is not stored at all -- the next listen re-learns it
        # rather than recalling a fragment (OPEN_THREADS 6a/6b).
        if coverage < FULL_COVERAGE or confidence < CONF_MIN:
            self.last_status = (f"discarded: coverage {coverage:.0%}, "
                                f"confidence {confidence:.2f}")
            return
        try:
            db = self._db_conn()
            tid = beat_cache.touch_track(
                db, key,
                source=(track or {}).get("player"),
                title=(track or {}).get("title"),
                artist=(track or {}).get("artist"),
                album=(track or {}).get("album"),
                duration_s=(track or {}).get("duration_s"),
                count_play=True)
            beat_cache.save_beatmap(db, tid, beat_times, bpm,
                                    method="dp_live_odf", coverage=coverage,
                                    confidence=confidence)
            self.last_status = f"learned {bpm:.1f} BPM ({len(beat_times)} beats)"
        except Exception as e:
            self.last_status = f"save failed: {e}"


def _selftest():
    """Drives a whole play-through with a stubbed player, so it runs with
    nothing actually playing and touches no real DB."""
    import os
    import tempfile

    ok = True

    class FakeNP:
        def __init__(self, key, dur):
            self.key, self.dur = key, dur
            self.pos = 0.0
            self.ev = [("start", key)]
        def drain_events(self):
            e, self.ev = self.ev, []
            return e
        def current(self):
            return ({"key": self.key, "duration_s": self.dur, "player": "vlc",
                     "title": "T", "artist": "A", "album": None}, self.pos)
        def stop(self):
            pass

    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "t.db")

        def play(key, seconds=40.0, bpm=120.0, seek=False, track_len=None):
            tr = beat_track.BeatTracker()
            s = BeatSession(tr, db_path=db_path)
            s.np = FakeNP(key, track_len if track_len is not None else seconds)
            rows = beat_track._synth(bpm, seconds)
            locked = 0
            for i, (row, t) in enumerate(rows):
                s.np.pos = t
                tr.update(row, t, t)
                if seek and i == len(rows) // 2:
                    s._tainted = True
                s.update()
                if tr._ref is not None:
                    locked += 1
            # _finalise() is what stores the map, so read the status AFTER
            # it -- during playback the status is still "learning".
            s._finalise()
            status = s.last_status
            if s._db is not None:
                s._db.close()
            return locked, status, len(rows)

        # First listen: nothing cached, so it learns.
        locked, status, n = play("yt:test1")
        good = locked == 0 and "learned" in status
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} first listen learns  -> {status}")

        # Second listen: cached, so it locks from the reference immediately.
        locked, status, n = play("yt:test1")
        good = locked > 0.9 * n and "locked from cache" in status
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} second listen locked for {locked}/{n} "
              f"samples -> {status}")

        # A seek taints the recording, so nothing is stored for that song.
        play("yt:test2", seek=True)
        db = beat_cache.connect(db_path)
        n2 = db.execute("SELECT COUNT(*) FROM track_keys WHERE key='yt:test2'").fetchone()[0]
        good = n2 == 0
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} seek taints the recording (stored: {n2})")

        # A song too short to judge is also not stored.
        play("yt:test3", seconds=6.0)
        n3 = db.execute("SELECT COUNT(*) FROM track_keys WHERE key='yt:test3'").fetchone()[0]
        good = n3 == 0
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} too-short song not stored (stored: {n3})")

        # A partial listen -- long enough to analyse, but covering only a
        # fraction of the track -- must be dropped, not stored (6a).
        play("yt:test4", seconds=40.0, track_len=400.0)
        n4 = db.execute("SELECT COUNT(*) FROM track_keys WHERE key='yt:test4'").fetchone()[0]
        good = n4 == 0
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} partial listen dropped (stored: {n4})")

        st = beat_cache.stats(db)
        db.close()
        good = st["tracks"] == 1 and st["beatmaps"] == 1
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} db holds {st['tracks']} track(s), "
              f"{st['beatmaps']} map(s)")

    print("beat_session selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
