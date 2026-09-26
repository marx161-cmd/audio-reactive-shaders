#!/usr/bin/env python3
"""Download the Kerr-Black-Hole-Visualizer's data (~5.5 GB) into textures/.

Hosted as a Hugging Face dataset. Standard library only; resumes partial
downloads and checks every file's SHA-256. The Tonnetz-Hex-Lattice shader
needs no data.

    python3 download_data.py            # everything
    python3 download_data.py --check    # only verify what's already there

What the files are (all rebuildable with the bake scripts in this repo, see
README "Rebuilding the data"):
  textures/kerr_v7/kn/paths32.npy     light paths around the Kerr-Newman hole,
                                      32 layers of 2410x2410 (kn_bake_modal.py)
  textures/kerr_v7/kn/sky.npy         where each pixel's light ends up on the sky
  textures/kerr_v7/kn_disk_table.npy  accretion disk emission table (kn_disk.py)
  textures/starfield/sky64k/*         NASA SVS Deep Star Maps 2020, 64k, as a
                                      BC6H cube map (skycube_unpack/bake.py)
  textures/pulsar_orrery/*.npy        star / planet / ring surface maps
                                      (pulsar_orrery_prep.py)
"""
import argparse
import hashlib
import os
import sys
import urllib.request

REPO = "marx161-cmd/audio-reactive-shaders-data"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
ROOT = os.path.dirname(os.path.abspath(__file__))

# size, sha256, path (relative to the repo root and to the dataset root)
FILES = [
    (2973747328, "e852551dd73285218e51e2517535ca4938387a3292f79461e59965ab2302fc59", "textures/kerr_v7/kn/paths32.npy"),
    (371718528, "68da545889f63b11111b8527daa004ea766ec30dc023fdb1d4aee75872c54d61", "textures/kerr_v7/kn/sky.npy"),
    (16512, "aa7bcd6e8bb564059ec0efcf1b0335cc98dbcc5d10449b065e31ae62d71f4c9c", "textures/kerr_v7/kn_disk_table.npy"),
    (8513, "f08b2721949418b75a0c0adc3c364149de82c67d9235cb2fd52ff1c19d24b661", "textures/starfield/sky64k/sky64k.json"),
    (2147483808, "5b54fe410d92849dae40654e8d6b9cdcfc0d3ce1e3d16f3e0c2bb41bc53d43a6", "textures/starfield/sky64k/sky64k_bc6h.bin"),
    (6294656, "11d18263c4320f824778220f35e9fea97801e131a02fe11c519afa9e5928a012", "textures/pulsar_orrery/dwarf.npy"),
    (524928, "c27bb430bd96ba4ab87ca54cb7018cc9d7553ca45441f7508d40d35b7e86fd2b", "textures/pulsar_orrery/chromo.npy"),
    (1574528, "fb6c8e7dce56ad6bc481858949ee4ee3d6528e1f8a0a9d77e801397349a278d6", "textures/pulsar_orrery/giant.npy"),
    (1574528, "ac2cce8897839981ea4515fca1ad2777b342e67aebf327530f21bd114de28cb1", "textures/pulsar_orrery/rock.npy"),
    (2176, "12602f2d34b026d6c11b1160fabd8ed263aca682668e3247d42054fe498641cd", "textures/pulsar_orrery/ringprof.npy"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(size, rel):
    dst = os.path.join(ROOT, rel)
    part = dst + ".part"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(BASE + rel, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r, open(part, "ab" if have else "wb") as out:
        if have and r.status != 206:          # server ignored the range: start over
            out.seek(0); out.truncate(); have = 0
        done = have
        while chunk := r.read(1 << 22):
            out.write(chunk)
            done += len(chunk)
            print(f"\r  {rel}: {done / 1e9:.2f} / {size / 1e9:.2f} GB", end="", flush=True)
    print()
    os.replace(part, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verify only, download nothing")
    args = ap.parse_args()
    bad = 0
    for size, digest, rel in FILES:
        path = os.path.join(ROOT, rel)
        ok = os.path.isfile(path) and os.path.getsize(path) == size and sha256(path) == digest
        if not ok and not args.check:
            fetch(size, rel)
            ok = os.path.getsize(path) == size and sha256(path) == digest
        print(f"{'ok ' if ok else 'BAD'} {rel}")
        bad += not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
