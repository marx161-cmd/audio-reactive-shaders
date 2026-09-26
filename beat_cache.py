"""Track database for the shader backdrop: what we have heard, and what we
worked out about it.

Why a DB and not one JSON per song: the first thing stored here is beat maps,
but the point is that it should take LYRICS later (user asked, 2026-09-21)
without re-laying-out what is already saved. So identity, metadata and derived
analyses are separate tables from the start, and a new kind of analysis is a
new table keyed on `tracks.id` -- never a change to an existing row shape.

Identity is deliberately indirect. The same song heard on YouTube Music and
played from a local file in VLC is ONE track with TWO keys, so `track_keys` is
its own table and a merge later is an UPDATE, not a data migration. A key is
whatever is most stable for the source:

    yt:<video id>    YouTube Music in the browser -- exact, survives re-opens,
                     and is not fooled by two songs sharing a title
    file:<path>      VLC playing a local file
    meta:<artist>|<title>|<length>   last resort
    fp:<hash>        reserved for the acoustic-fingerprint layer, which will
                     recognise a song with no metadata at all

Analyses are versioned rather than overwritten (`is_current`), because a
better re-analysis should not destroy the evidence of what the shader was
actually driven by last night.
"""

import os
import sqlite3
import time

import numpy as np

DEFAULT_DB = os.path.expanduser(
    os.environ.get("SHADER_TRACK_DB", "~/.local/share/shader-backdrop/tracks.db")
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id          INTEGER PRIMARY KEY,
    title       TEXT,
    artist      TEXT,
    album       TEXT,
    duration_s  REAL,
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL,
    play_count  INTEGER NOT NULL DEFAULT 0
);

-- Several keys may point at one track (same song, different source).
CREATE TABLE IF NOT EXISTS track_keys (
    key       TEXT PRIMARY KEY,
    track_id  INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    source    TEXT,
    added     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_track_keys_track ON track_keys(track_id);

-- Beat maps. `beat_times` is a float32 BLOB of seconds from song start, in
-- SONG-POSITION coordinates -- not wall clock and not "since we started
-- listening", or it could never be replayed against a different playback.
CREATE TABLE IF NOT EXISTS beatmaps (
    id          INTEGER PRIMARY KEY,
    track_id    INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    method      TEXT NOT NULL,      -- 'dp_live_odf', 'dp_file', ...
    created     REAL NOT NULL,
    bpm         REAL,
    confidence  REAL,
    coverage    REAL,               -- fraction of the track the grid spans
    n_beats     INTEGER,
    beat_times  BLOB NOT NULL,
    is_current  INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_beatmaps_track ON beatmaps(track_id, is_current);

-- Not written yet. Here now so that adding lyrics later is an INSERT and not a
-- schema rewrite: one row per fetched/derived set, its lines in lyric_lines.
-- `synced` distinguishes timed lines from a plain text dump; an unsynced set
-- simply has NULL timings, so both live in the same shape.
CREATE TABLE IF NOT EXISTS lyrics (
    id          INTEGER PRIMARY KEY,
    track_id    INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    source      TEXT,
    created     REAL NOT NULL,
    synced      INTEGER NOT NULL DEFAULT 0,
    language    TEXT,
    is_current  INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_lyrics_track ON lyrics(track_id, is_current);

CREATE TABLE IF NOT EXISTS lyric_lines (
    id         INTEGER PRIMARY KEY,
    lyrics_id  INTEGER NOT NULL REFERENCES lyrics(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    t_start    REAL,               -- song-position seconds, same base as beats
    t_end      REAL,
    text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_lyric_lines ON lyric_lines(lyrics_id, idx);
"""


def connect(path=None):
    """Open (creating if needed) the track DB. WAL so the reader thread can
    write while anything else reads, and foreign keys on so a deleted track
    takes its analyses with it."""
    path = path or DEFAULT_DB
    os.makedirs(os.path.dirname(path), exist_ok=True)
    db = sqlite3.connect(path, timeout=5.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(_SCHEMA)
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version < SCHEMA_VERSION:
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    db.commit()
    return db


def touch_track(db, key, source=None, title=None, artist=None, album=None,
                duration_s=None, count_play=False):
    """Find or create the track behind `key`, refresh what we know about it,
    and return its id. Metadata is only ever filled IN, never blanked: a
    player that reports no album should not erase an album we already had."""
    now = time.time()
    row = db.execute("SELECT track_id FROM track_keys WHERE key = ?", (key,)).fetchone()
    if row:
        tid = row["track_id"]
        db.execute(
            """UPDATE tracks SET last_seen = ?,
                   title      = COALESCE(NULLIF(?, ''), title),
                   artist     = COALESCE(NULLIF(?, ''), artist),
                   album      = COALESCE(NULLIF(?, ''), album),
                   duration_s = COALESCE(?, duration_s),
                   play_count = play_count + ?
               WHERE id = ?""",
            (now, title, artist, album, duration_s, 1 if count_play else 0, tid))
    else:
        cur = db.execute(
            """INSERT INTO tracks (title, artist, album, duration_s,
                                   first_seen, last_seen, play_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (title, artist, album, duration_s, now, now, 1 if count_play else 0))
        tid = cur.lastrowid
        db.execute(
            "INSERT INTO track_keys (key, track_id, source, added) VALUES (?, ?, ?, ?)",
            (key, tid, source, now))
    db.commit()
    return tid


def add_key(db, track_id, key, source=None):
    """Point another key at an existing track (same song, second source)."""
    db.execute(
        "INSERT OR IGNORE INTO track_keys (key, track_id, source, added) "
        "VALUES (?, ?, ?, ?)", (key, track_id, source, time.time()))
    db.commit()


def save_beatmap(db, track_id, beat_times, bpm, method="dp_live_odf",
                 confidence=None, coverage=None, make_current=True):
    """Store a beat map. Previous maps are kept but demoted, so what drove a
    render last week is still recoverable after a re-analysis."""
    bt = np.asarray(beat_times, dtype=np.float32)
    if make_current:
        db.execute("UPDATE beatmaps SET is_current = 0 WHERE track_id = ?", (track_id,))
    db.execute(
        """INSERT INTO beatmaps (track_id, method, created, bpm, confidence,
                                 coverage, n_beats, beat_times, is_current)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (track_id, method, time.time(), bpm, confidence, coverage,
         int(bt.size), bt.tobytes(), 1 if make_current else 0))
    db.commit()


def load_beatmap(db, key):
    """Current beat map for whatever track `key` belongs to, or None.
    Returns (beat_times float64 array, meta dict)."""
    row = db.execute(
        """SELECT b.* FROM beatmaps b
           JOIN track_keys k ON k.track_id = b.track_id
           WHERE k.key = ? AND b.is_current = 1
           ORDER BY b.created DESC LIMIT 1""", (key,)).fetchone()
    if row is None:
        return None
    times = np.frombuffer(row["beat_times"], dtype=np.float32).astype(np.float64)
    return times, {"bpm": row["bpm"], "method": row["method"],
                   "confidence": row["confidence"], "coverage": row["coverage"],
                   "created": row["created"], "n_beats": row["n_beats"]}


def stats(db):
    q = lambda s: db.execute(s).fetchone()[0]
    return {
        "tracks": q("SELECT COUNT(*) FROM tracks"),
        "keys": q("SELECT COUNT(*) FROM track_keys"),
        "beatmaps": q("SELECT COUNT(*) FROM beatmaps WHERE is_current = 1"),
        "lyrics": q("SELECT COUNT(*) FROM lyrics WHERE is_current = 1"),
    }


def _selftest():
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as d:
        db = connect(os.path.join(d, "t.db"))

        tid = touch_track(db, "yt:abc123", source="firefox", title="WGNO",
                          artist="Yassin & Mädness", duration_s=192.0,
                          count_play=True)
        save_beatmap(db, tid, np.arange(0.0, 192.0, 0.5), bpm=120.0,
                     confidence=0.9, coverage=0.99)

        got = load_beatmap(db, "yt:abc123")
        good = got is not None and got[0].size == 384 and abs(got[1]["bpm"] - 120) < 1e-6
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} save + load beat map ({got[0].size} beats)")

        # Same song via VLC: a second key, NOT a second track.
        add_key(db, tid, "file:/music/wgno.flac", source="vlc")
        got2 = load_beatmap(db, "file:/music/wgno.flac")
        good = got2 is not None and got2[0].size == 384
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} second source key resolves to the same track")

        # Metadata is filled in, never blanked.
        touch_track(db, "yt:abc123", album="Yassin & Mädness")
        touch_track(db, "yt:abc123", title="")
        r = db.execute("SELECT title, album, play_count FROM tracks WHERE id=?",
                       (tid,)).fetchone()
        good = r["title"] == "WGNO" and r["album"] == "Yassin & Mädness" and r["play_count"] == 1
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} metadata fills in, empty never blanks "
              f"(title={r['title']!r} plays={r['play_count']})")

        # Re-analysis demotes the old map instead of destroying it.
        save_beatmap(db, tid, np.arange(0.0, 192.0, 0.4), bpm=150.0, method="dp_file")
        cur = load_beatmap(db, "yt:abc123")
        n_all = db.execute("SELECT COUNT(*) FROM beatmaps WHERE track_id=?",
                           (tid,)).fetchone()[0]
        good = abs(cur[1]["bpm"] - 150.0) < 1e-6 and n_all == 2
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} re-analysis supersedes, old kept "
              f"({n_all} maps, current {cur[1]['bpm']:.0f} BPM)")

        # The lyrics tables exist and take a synced set, keyed on the same
        # track and the same song-position time base as the beats.
        lid = db.execute(
            "INSERT INTO lyrics (track_id, source, created, synced, language) "
            "VALUES (?, ?, ?, 1, 'de')", (tid, "manual", time.time())).lastrowid
        db.executemany(
            "INSERT INTO lyric_lines (lyrics_id, idx, t_start, t_end, text) "
            "VALUES (?, ?, ?, ?, ?)",
            [(lid, 0, 0.5, 3.0, "line one"), (lid, 1, 3.0, 6.2, "line two")])
        db.commit()
        n = db.execute("SELECT COUNT(*) FROM lyric_lines WHERE lyrics_id=?",
                       (lid,)).fetchone()[0]
        good = n == 2 and stats(db)["lyrics"] == 1
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} lyrics extension works with no schema change "
              f"({n} lines)")

        # Deleting a track must not leave orphans behind.
        db.execute("DELETE FROM tracks WHERE id = ?", (tid,))
        db.commit()
        left = (db.execute("SELECT COUNT(*) FROM beatmaps").fetchone()[0]
                + db.execute("SELECT COUNT(*) FROM lyric_lines").fetchone()[0]
                + db.execute("SELECT COUNT(*) FROM track_keys").fetchone()[0])
        good = left == 0
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} cascade delete leaves no orphans ({left} rows)")

    print("beat_cache selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
