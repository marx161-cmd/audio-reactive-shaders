"""Song -> offline analysis (analysis/analyze.py output) lookup, shared by
beat_session.py (beat steering) and music_state.py (the music-state uniforms).
scope.md 2026-10-11 "song cache in shaders: ONE live-capable interface".

Index = table song_features in tracks.db:
    src      the analysed audio file's path (UNIQUE)
    npz      the analysis file
    version  analyze.py ANALYSIS_VERSION
    yt       the YouTube id written into the file (comment=yt:<id>), or NULL
A playing song's key resolves through:
    file:<url-encoded path>  -> src (VLC plays the same files)
    any other key            -> its track in track_keys -> that track's yt:<id>
                                keys -> yt
Indexing is idempotent; analyze.py indexes each song it saves, and
`python3 song_features.py` (re)indexes the whole features folder.
"""

import json
import os
import sqlite3
import subprocess
import sys
import time
from urllib.parse import unquote

import numpy as np

FEATURES_DIR = os.path.expanduser("~/.local/share/shader-backdrop/features")
DB_PATH = os.path.expanduser("~/.local/share/shader-backdrop/tracks.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS song_features (
    src     TEXT PRIMARY KEY,
    npz     TEXT NOT NULL,
    version INTEGER,
    yt      TEXT,
    added   REAL
);
CREATE INDEX IF NOT EXISTS ix_song_features_yt ON song_features(yt);
"""


def connect(path=DB_PATH):
    db = sqlite3.connect(path, timeout=5.0)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(_SCHEMA)
    return db


def _yt_of(src):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=comment",
             "-of", "default=nw=1:nk=1", src],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out[3:] if out.startswith("yt:") else None


def index_json(db, json_path):
    """Index one analysis (its .json next to the .npz)."""
    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    src = meta.get("source")
    npz = json_path[:-5] + ".npz"
    if not src or not os.path.exists(npz):
        return False
    db.execute("INSERT OR REPLACE INTO song_features (src, npz, version, yt, added) "
               "VALUES (?, ?, ?, ?, ?)",
               (src, npz, meta.get("version"), _yt_of(src), time.time()))
    db.commit()
    return True


def lookup(db, key):
    """npz path for a nowplaying key, or None."""
    if not key:
        return None
    if key.startswith("file:"):
        row = db.execute("SELECT npz FROM song_features WHERE src = ?",
                         (unquote(key[5:]),)).fetchone()
        return row[0] if row else None
    ids = [key[3:]] if key.startswith("yt:") else []
    try:
        ids += [r[0][3:] for r in db.execute(
            """SELECT k2.key FROM track_keys k1 JOIN track_keys k2
               ON k2.track_id = k1.track_id
               WHERE k1.key = ? AND k2.key LIKE 'yt:%'""", (key,))]
    except sqlite3.Error:
        pass
    for vid in ids:
        row = db.execute("SELECT npz FROM song_features WHERE yt = ? ORDER BY added DESC",
                         (vid,)).fetchone()
        if row:
            return row[0]
    return None


class Song:
    """The parts of one analysis the runtime needs, loaded once per song."""

    def __init__(self, npz_path):
        z = np.load(npz_path)
        meta = json.loads(bytes(z["meta_json"]))
        self.path = npz_path
        self.fps = float(meta["fps"])
        self.duration = float(meta["duration_s"])
        self.bpb = int(meta.get("beats_per_bar", 4))
        # Phase-1 meter confidence (scope.md 2026-10-14); absent = legacy
        # analysis whose bpb is the old unreliable one, so treat as untrusted.
        self.meter_conf = float(meta.get("meter_conf", 0.0))
        self.beats = z["beats"].astype(np.float64)
        self.downbeats = z["downbeats"].astype(np.float64)
        # Per-bar map (meter_version >= 3): bar starts as beat indices + lock flags.
        self.bar_db_idx = (z["bar_db_idx"].astype(np.int64) if "bar_db_idx" in z.files
                           else None)
        self.bar_lock = (z["bar_lock"].astype(bool) if "bar_lock" in z.files else None)
        self.rel = {k: z["rel_" + k].astype(np.float32)
                    for k in ("energy", "weight", "density", "brightness", "pulse", "voice")}
        self.sections = meta["sections"]
        self.starts = np.array([s["start_s"] for s in self.sections])
        self.drops = np.array([e["t"] for e in meta["events"] if e["type"] == "drop"])
        self.builds = [(e["t"], e["until"]) for e in meta["events"] if e["type"] == "build"]

    def curve(self, name, t):
        a = self.rel[name]
        i = int(np.clip(t * self.fps, 0, len(a) - 1))
        return float(a[i])

    def section_at(self, t):
        return int(max(np.searchsorted(self.starts, t, side="right") - 1, 0))

    def beat_index_at(self, t):
        """Fractional beat index at song time t (beats before the first beat < 0)."""
        b = self.beats
        if t <= b[0]:
            return (t - b[0]) / max(b[1] - b[0], 1e-3)
        i = int(np.searchsorted(b, t, side="right") - 1)
        if i >= len(b) - 1:
            return (len(b) - 1) + (t - b[-1]) / max(b[-1] - b[-2], 1e-3)
        return i + (t - b[i]) / max(b[i + 1] - b[i], 1e-6)


def main():
    db = connect()
    n = 0
    for name in sorted(os.listdir(FEATURES_DIR)):
        if name.endswith(".json") and index_json(db, os.path.join(FEATURES_DIR, name)):
            n += 1
    linked = db.execute("SELECT COUNT(*) FROM song_features WHERE yt IS NOT NULL").fetchone()[0]
    print(f"indexed {n} analyses ({linked} with a YouTube id)")
    if len(sys.argv) > 1:
        print(sys.argv[1], "->", lookup(db, sys.argv[1]))


if __name__ == "__main__":
    main()
