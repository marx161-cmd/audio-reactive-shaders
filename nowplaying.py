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


class NowPlaying:
    """Polls MPRIS on its own thread and exposes the current track plus a
    position interpolated with the wall clock between polls, so a caller
    running at 60 Hz gets a smooth position from a 1 Hz source."""

    def __init__(self, poll_period=POLL_PERIOD_S):
        self.poll_period = poll_period
        self._lock = threading.Lock()
        self._state = None          # dict, or None when nothing is playing
        self._at = 0.0              # monotonic time the position was observed
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
                self._tick(_query())
            except Exception:
                # Never take the renderer down over a media-player query.
                pass
            self._stop.wait(self.poll_period)

    def _tick(self, obs):
        now = time.monotonic()
        with self._lock:
            prev = self._state
            if obs is None:
                if prev is not None:
                    self._events.append(("end", prev["key"]))
                self._state, self._at = None, now
                return

            if prev is None or prev["key"] != obs["key"]:
                if prev is not None:
                    self._events.append(("end", prev["key"]))
                self._events.append(("start", obs["key"]))
            else:
                # Same track: is this a seek or just the clock advancing?
                expected = prev["position"] + (now - self._at)
                if abs(obs["position"] - expected) > SEEK_TOL_S:
                    self._events.append(("seek", obs["key"]))
            self._state, self._at = obs, now

    def current(self):
        """(track dict, position_seconds) with the position interpolated to
        now, or (None, None)."""
        with self._lock:
            if self._state is None:
                return None, None
            pos = self._state["position"] + (time.monotonic() - self._at)
            if self._state["duration_s"]:
                pos = min(pos, self._state["duration_s"])
            return dict(self._state), pos

    def drain_events(self):
        with self._lock:
            ev, self._events = self._events, []
        return ev


def _selftest():
    """Runs against whatever is actually playing right now. With nothing
    playing it verifies the quiet path instead of failing."""
    ok = True

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
