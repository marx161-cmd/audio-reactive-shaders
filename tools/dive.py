#!/usr/bin/env python3
"""Dive path for the Nebulabrot-Infinite-Dive shader.

A ring of stops from shallow to deep. The last stop is the shader's loop copy:
arriving there it swaps the view back to the start, so `next` carries on with
stop 1 and the dive never ends. A flight takes 16 beats of the renderer's beat
clock. With auto-dive on, the renderer moves on to the next stop every 32 beats
(16 flying, 16 resting).

    tools/dive.py auto [on|off|toggle]   auto-dive to the beat
    tools/dive.py next | prev            fly one stop deeper / back
    tools/dive.py goto N                 fly to stop N
    tools/dive.py add                    save the current view as a stop
    tools/dive.py list | undo | clear | reset

Bind `next`, `prev` and `auto toggle` to window-manager keys; the backdrop
never takes keyboard focus. The stops live in shader_state/nebulabrot_dive.txt
("re im zoom rot" per line, sorted by zoom). On first use it is seeded with the
default path shipped next to the shader; `reset` restores that default.
The current view is read from shader_state/<shader>.txt, which the renderer
saves every 2 s, so wait a moment after moving before `add`.
"""
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADER = "nebulabrot-infinite-dive.mp"
STATE = os.path.join(ROOT, "shader_state")
VIEW = os.path.join(STATE, SHADER + ".txt")
PATH = os.path.join(STATE, "nebulabrot_dive.txt")
IDX = os.path.join(STATE, "nebulabrot_dive_idx.txt")
DEFAULT = os.path.join(ROOT, "shaders", SHADER, "dive_default.txt")
TARGET = os.path.join(ROOT, "cam_target.txt")
AUTO = os.path.join(ROOT, "autodive.txt")
START_VIEW = "-0.4 0 1 0"          # the whole bake, zoomed out, centred (= the loop copy)


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def stops():
    return [l.strip() for l in read(PATH).splitlines() if l.strip()]


def index():
    try:
        return int(read(IDX, "0").strip() or 0)
    except ValueError:
        return 0


def send(view, idx):
    """Write the flight target; a new id makes the shader start a new flight."""
    parts = read(TARGET).split()
    tid = (int(float(parts[4])) + 1) % 100000 if len(parts) >= 5 else 1
    write(TARGET, f"{view} {tid}\n")
    write(IDX, f"{idx}\n")


def fly(n, path):
    send(path[n - 1], n)
    print(f"stop {n}/{len(path)}: {path[n - 1]}")


def swapped(path):
    """Has the loop copy swapped the view back to the start (zoomed out past stop 1)?"""
    v = read(VIEW).split()
    return len(v) >= 3 and float(v[2]) < float(path[0].split()[2]) * 0.99


def flying_to_last(path):
    t = read(TARGET).split()
    try:
        recent = time.time() - os.path.getmtime(TARGET) < 20
    except OSError:
        return False
    return " ".join(t[:4]) == path[-1] and recent


def main():
    os.makedirs(STATE, exist_ok=True)
    if not os.path.exists(PATH) and os.path.exists(DEFAULT):
        shutil.copyfile(DEFAULT, PATH)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    path, idx = stops(), index()
    n = len(path)

    if cmd == "auto":
        cur = read(AUTO, "0").strip() == "1"
        arg = sys.argv[2] if len(sys.argv) > 2 else "toggle"
        new = {"on": True, "off": False}.get(arg, not cur)
        write(AUTO, "1\n" if new else "0\n")
        print("auto-dive", "on" if new else "off")
    elif cmd in ("next", "prev", "goto") and n == 0:
        sys.exit("no stops")
    elif cmd == "next":
        if idx < n:
            fly(idx + 1, path)
        elif swapped(path):
            fly(1, path)
        elif flying_to_last(path):
            print("diving into the loop copy -- wait for the swap")
        else:
            fly(n, path)                       # the dive into the copy got interrupted: finish it
    elif cmd == "prev":
        if idx == 1:
            send(f"{START_VIEW}", n)
            print(f"start view (= stop {n}, the loop copy)")
        else:
            fly(max(idx - 1, 1), path)
    elif cmd == "goto":
        fly(min(max(int(sys.argv[2]), 1), n), path)
    elif cmd == "add":
        v = read(VIEW).split()
        if len(v) < 4:
            sys.exit("no current view saved yet")
        cur = f"{v[0]} 0 {v[2]} {v[3]}"      # Im 0: on the mirror axis
        path = sorted(path + [cur], key=lambda l: float(l.split()[2]))
        write(PATH, "\n".join(path) + "\n")
        pos = path.index(cur) + 1
        write(IDX, f"{pos}\n")
        print(f"added stop {pos}/{len(path)}: {cur}")
    elif cmd == "undo":
        path = path[:-1]
        write(PATH, "\n".join(path) + ("\n" if path else ""))
        write(IDX, f"{min(idx, len(path))}\n")
        print(f"stops: {len(path)}")
    elif cmd == "clear":
        write(PATH, "")
        write(IDX, "0\n")
        print("cleared")
    elif cmd == "reset":
        shutil.copyfile(DEFAULT, PATH)
        write(IDX, "0\n")
        print(f"default path restored ({len(stops())} stops)")
    elif cmd == "list":
        for i, l in enumerate(path, 1):
            print(f"{i:4d}  {l}")
        print(f"current: {idx} / {n}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
