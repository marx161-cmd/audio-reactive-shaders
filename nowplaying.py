"""What is playing, and where we are in it -- via MPRIS (`playerctl`).

Needed because a beat map is only replayable if it is stored in SONG-POSITION
coordinates, which means we need the position on the FIRST listen too, not
only on recall. Covers the two players that matter here: YouTube Music in the
browser and VLC. mpv is deliberately not supported (it exposes no MPRIS and
the user does not use it -- 2026-09-21).

Anything without MPRIS simply reports nothing and the caller falls back to
live beat tracking, which is the existing behaviour and always correct.

GOTCHA that is easy to get silently wrong: `playerctl position` returns
SECONDS, but `{{position}}` inside a --format string returns MICROSECONDS.
This module uses the format string, so it divides. `mpris:length` is always
microseconds.

POSITION PRECISION (2026-10-10, scope.md "song cache REDESIGN"): LibreWolf
reports WHOLE seconds, stepping cleanly every 1.000 s. Interpolating from a
~1 Hz poll of that put recorded positions up to 1 s late, with the error
ramping and snapping back every ~100 s as the poll phase drifted -- it warped
721 of 733 browser-learned beat maps. So a coarse player is now anchored on
the step itself: a short fast-poll catches the moment the second ticks over
(~15 ms), and position = monotonic time since song zero. Precise players
(VLC) anchor straight from each reading.
"""

import re
import subprocess
import threading
import time

POLL_PERIOD_S = 1.0
# A position that disagrees with wall-clock extrapolation by more than this is
# a seek, not drift. Generous, because browsers report position coarsely (the
# observed LibreWolf value steps in whole seconds).
SEEK_TOL_S = 1.5
# Anchored positions are exact, so a coarse reading must sit in
# [expected - 1, expected] (it is truncated); outside this margin it is a seek
# (or drift) and the anchor is re-found.
ANCHOR_TOL_S = 0.1
# A whole-second reading has a fractional part of a few microseconds.
COARSE_EPS = 0.002
# Long enough to always contain one step.
EDGE_WINDOW_S = 1.3
# Players we will trust. Anything else is ignored rather than guessed at.
ACCEPT = ("firefox", "librewolf", "vlc", "chromium", "chrome")

_FMT = ("{{playerName}}\t{{status}}\t{{position}}\t{{mpris:length}}"
        "\t{{xesam:url}}\t{{xesam:artist}}\t{{xesam:title}}\t{{xesam:album}}")


def identity_key(url, artist, title, length_us):
    """The most stable key available for this source.

    A YouTube video id is exact and survives re-opening the tab, which a title
    never does -- two songs share a title, and YT Music rewrites titles. A
    local file is keyed by path. Metadata is the last resort and includes the
    length, because that is what separates two masters of one song."""
    url = url or ""
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url)
    if m:
        return "yt:" + m.group(1)
    if url.startswith("file://"):
        return "file:" + url[7:]
    if title:
        return f"meta:{artist}|{title}|{length_us}"
    return None


def _parse(fields):
    """One tab-split metadata line -> dict, or None if unusable.

    Tolerates 7 fields because a blank trailing field (usually album) can be
    eaten by whitespace stripping upstream; pad rather than drop the row.
    """
    if len(fields) < 7:
        return None
    fields = list(fields) + [""] * (8 - len(fields))
    player, status, pos, length, url, artist, title, album = fields[:8]
    try:
        position = float(pos) / 1e6              # microseconds -> seconds
    except ValueError:
        position = 0.0
    length_us = length if length.isdigit() else ""
    return {
        "player": player, "status": status,
        "key": identity_key(url, artist, title, length_us), "position": position,
        "duration_s": (int(length_us) / 1e6) if length_us else None,
        "title": title or None, "artist": artist or None,
        "album": album or None,
    }


def _query():
    """One playerctl call for every player. Returns the first ACCEPTed player
    that is actually Playing, or None."""
    try:
        r = subprocess.run(["playerctl", "-a", "metadata", "--format", _FMT],
                           capture_output=True, text=True, timeout=2.0)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in r.stdout.splitlines():
        obs = _parse(line.split("\t"))
        if obs is None or obs["status"] != "Playing":
            continue
        if not any(a in obs["player"].lower() for a in ACCEPT):
            continue
        if obs["key"]:
            return obs
    return None


def query_player(player):
    """Same fields as `_query()` but for one named player and ANY status.

    `_query()` deliberately only returns a Playing, ACCEPTed player -- right
    for "what is on now". A recording session also needs to see a player
    through a pause and to notice it going Stopped, so it needs this
    un-filtered view. Returns None if the player is gone.
    """
    try:
        r = subprocess.run(["playerctl", "-p", player, "metadata",
                            "--format", _FMT],
                           capture_output=True, text=True, timeout=2.0)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    lines = r.stdout.splitlines()
    return _parse(lines[0].split("\t")) if lines else None


def is_coarse(position):
    """A whole-second reading (LibreWolf), as opposed to a precise one."""
    f = position % 1.0
    return f < COARSE_EPS or f > 1.0 - COARSE_EPS


def _timed_query_player(player):
    """query_player plus the monotonic midpoint of the call (when the value
    was most likely read) and half its duration (the uncertainty)."""
    a = time.monotonic()
    obs = query_player(player)
    b = time.monotonic()
    return obs, 0.5 * (a + b), 0.5 * (b - a)


def find_song_zero(player, key, window=EDGE_WINDOW_S, stop=None):
    """Monotonic time of song position 0 for a whole-second player, from the
    moment its reading steps to the next second. Fast-polls for up to
    `window` s; None if the song changes, stops or skips meanwhile.

    The step happens between two consecutive readings, so its time is taken
    as the midpoint between them (~+-8 ms at playerctl's ~15 ms per call)."""
    prev = None
    end = time.monotonic() + window
    while time.monotonic() < end:
        if stop is not None and stop.is_set():
            return None
        obs, t, _ = _timed_query_player(player)
        if (obs is None or obs["key"] != key or obs["status"] != "Playing"):
            return None
        pos = obs["position"]
        if prev is not None and round(pos) != round(prev[0]):
            if 0.5 < pos - prev[0] < 1.5:
                return 0.5 * (prev[1] + t) - round(pos)
            return None                     # a seek, not a tick
        prev = (pos, t)
    return None


class NowPlaying:
    """Polls MPRIS on its own thread and exposes the current track plus a
    position that is exact between polls: anchored on the whole-second step
    for coarse players (see the module docstring), on each reading for
    precise ones. Until a coarse player's step is found, the old
    interpolation is used as a fallback."""

    def __init__(self, poll_period=POLL_PERIOD_S):
        self.poll_period = poll_period
        self._lock = threading.Lock()
        self._state = None          # dict, or None when nothing is playing
        self._at = 0.0              # monotonic time the position was observed
        self._zero = None           # monotonic time of song position 0, once anchored
        self._paused_at = None      # monotonic time a Paused player was first seen
        self._paused_pos = None     # frozen song position while paused
        self._events = []           # ('start'|'end'|'seek', key)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="nowplaying")
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                a = time.monotonic()
                obs = _query()
                if obs is None:
                    # Nothing is Playing. It may just be PAUSED -- _query() only
                    # ever returns a Playing player, so ask the player we were
                    # tracking directly before treating this as the song ending.
                    obs = self._query_tracked()
                self._tick(obs, 0.5 * (a + time.monotonic()))
                self._anchor_if_needed(obs)
            except Exception:
                # Never take the renderer down over a media-player query.
                pass
            self._stop.wait(self.poll_period)

    def _query_tracked(self):
        """The tracked player's metadata, ANY status. See query_player()."""
        with self._lock:
            cur = self._state
        if cur is None:
            return None
        return query_player(cur["player"])

    def _tick(self, obs, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            prev = self._state

            # A Paused player is not gone. Keep the session alive and freeze the
            # clock. Before 2026-10-14 this arrived as obs=None (because
            # _query() only returns a Playing player) and was read as the song
            # ending: BeatSession discarded the recording, and the resumed half
            # became a fresh fragment that was itself dropped at track end.
            if (obs is not None and obs["status"] == "Paused"
                    and prev is not None and prev["key"] == obs["key"]):
                if self._paused_at is None:
                    self._paused_at = now
                    self._paused_pos = self._position_at(now)
                self._state = obs
                return

            # Nothing playing, or the player stopped: the song is over.
            if obs is None or obs["status"] == "Stopped":
                if prev is not None:
                    self._events.append(("end", prev["key"]))
                self._state, self._at, self._zero = None, now, None
                self._paused_at = self._paused_pos = None
                return

            # Playing again.
            resumed = self._paused_at is not None
            if resumed:
                # The song did not advance while paused; the wall clock did.
                # Shift song-zero forward by the paused time so the position
                # picks up exactly where it paused, with no seek.
                if self._zero is not None:
                    self._zero += now - self._paused_at
                self._paused_at = self._paused_pos = None

            if prev is None:
                self._events.append(("start", obs["key"]))
                self._zero = None
            elif prev["key"] != obs["key"]:
                self._events.append(("end", prev["key"]))
                self._events.append(("start", obs["key"]))
                self._zero = None
            elif not resumed:
                if self._zero is not None:
                    # Anchored: a truncated reading sits in [expected-1, expected].
                    expected = now - self._zero
                    lag = expected - obs["position"]
                    hi = 1.0 if is_coarse(obs["position"]) else 0.0
                    if not (-ANCHOR_TOL_S <= lag <= hi + ANCHOR_TOL_S):
                        if abs(lag) > SEEK_TOL_S:
                            self._events.append(("seek", obs["key"]))
                        self._zero = None       # seek or drift: re-anchor
                else:
                    # Same track: is this a seek or just the clock advancing?
                    expected = prev["position"] + (now - self._at)
                    if abs(obs["position"] - expected) > SEEK_TOL_S:
                        self._events.append(("seek", obs["key"]))
            if self._zero is None and not is_coarse(obs["position"]):
                self._zero = now - obs["position"]
            self._state, self._at = obs, now

    def _anchor_if_needed(self, obs):
        """Fast-poll for the whole-second step of a coarse player (blocks this
        thread for up to EDGE_WINDOW_S, never the caller)."""
        if obs is None or obs["status"] != "Playing":
            return
        with self._lock:
            if self._zero is not None or self._state is None \
                    or self._state["key"] != obs["key"]:
                return
        zero = find_song_zero(obs["player"], obs["key"], stop=self._stop)
        if zero is None:
            return
        with self._lock:
            if self._state is not None and self._state["key"] == obs["key"]:
                self._zero = zero

    def anchored(self):
        """True once the position is exact (step found / precise player)."""
        with self._lock:
            return self._zero is not None

    def _position_at(self, now):
        """Song position at monotonic time `now`. Caller holds the lock."""
        if self._zero is not None:
            pos = now - self._zero
        else:
            pos = self._state["position"] + (now - self._at)
        if self._state["duration_s"]:
            pos = min(pos, self._state["duration_s"])
        return pos

    def current(self):
        """(track dict, position_seconds) at now, or (None, None)."""
        with self._lock:
            if self._state is None:
                return None, None
            if self._paused_pos is not None:
                return dict(self._state), self._paused_pos
            return dict(self._state), self._position_at(time.monotonic())

    def drain_events(self):
        with self._lock:
            ev, self._events = self._events, []
        return ev


def _pause_selftest():
    """Deterministic: drive _tick with a stubbed player through a pause. A
    pause must NOT emit 'end' (that discarded the recording), the position
    must freeze, and resuming must not look like a seek. Stopped must still
    emit 'end'. No subprocess, no real clock."""
    ok = True
    np_ = NowPlaying()
    base = 1000.0

    def obs(status, pos, key="yt:x", player="vlc"):
        return {"player": player, "status": status, "key": key, "position": pos,
                "duration_s": 200.0, "title": "t", "artist": "a", "album": None}

    np_._tick(obs("Playing", 30.3), base)
    np_._tick(obs("Playing", 31.3), base + 1.0)
    ev = np_.drain_events()
    good = ("start", "yt:x") in ev
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} play  -> {ev}")

    zero0 = np_._zero
    for i in range(3):
        np_._tick(obs("Paused", 31.3), base + 2.0 + i)
    ev = np_.drain_events()
    good = ev == [] and abs(np_._paused_pos - ((base + 2.0) - zero0)) < 1e-6
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} pause -> no events, pos frozen "
          f"at {np_._paused_pos!r} {ev}")

    np_._tick(obs("Playing", 38.3), base + 8.0)
    ev = np_.drain_events()
    good = ev == [] and abs(np_._position_at(base + 8.0)
                            - ((base + 2.0) - zero0)) < 1e-6
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} resume-> no seek, pos continues "
          f"({np_._position_at(base + 8.0):.3f}) {ev}")

    np_._tick(obs("Stopped", 38.3), base + 9.0)
    ev = np_.drain_events()
    good = ("end", "yt:x") in ev
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} stop  -> end {ev}")

    print("nowplaying pause selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _selftest():
    """Runs against whatever is actually playing right now. With nothing
    playing it verifies the quiet path instead of failing."""
    ok = _pause_selftest() == 0

    from shutil import which
    if which("playerctl") is None:
        print("  playerctl not installed -- cannot test")
        return 1

    # Key derivation is pure and always checkable.
    cases = [
        ("https://music.youtube.com/watch?v=zDicQfEaSWw&list=LR", "", "", "", "yt:zDicQfEaSWw"),
        ("file:///music/a.flac", "", "", "", "file:/music/a.flac"),
        ("", "Artist", "Title", "101000000", "meta:Artist|Title|101000000"),
        ("", "", "", "", None),
    ]
    for url, ar, ti, ln, want in cases:
        got = identity_key(url, ar, ti, ln)
        good = got == want
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} key {url[:34] or '(no url)':34s} -> {got}")

    np_ = NowPlaying(poll_period=0.5).start()
    time.sleep(1.6)
    track, pos = np_.current()
    if track is None:
        print("  ok   nothing playing -- quiet path returns (None, None)")
    else:
        print(f"  ok   live: {track['artist']} - {track['title']}")
        print(f"       key={track['key']}  player={track['player']}  "
              f"len={track['duration_s']}s")
        p1 = pos
        time.sleep(1.2)
        _, p2 = np_.current()
        advanced = p2 - p1
        # Position must advance roughly with wall clock, and must be in
        # SECONDS -- a microsecond bug shows up here as a vast number.
        good = 0.5 < advanced < 2.5 and 0 <= p2 < 100000
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} position {p1:.2f} -> {p2:.2f} "
              f"(+{advanced:.2f}s in 1.2s wall, units look like seconds)")
    np_.stop()

    print("nowplaying selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
