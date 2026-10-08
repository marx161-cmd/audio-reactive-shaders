#!/usr/bin/env python3
"""Download the shaders' data into textures/ (~16 GB for everything).

Hosted as a Hugging Face dataset. Standard library only; resumes partial
downloads and checks every file's SHA-256. Pick what you need:

    python3 download_data.py                  # everything
    python3 download_data.py kerr             # Kerr-Black-Hole-Visualizer, ~6.6 GB
    python3 download_data.py nebulabrot       # Nebulabrot-Infinite-Dive, ~9.1 GB
    python3 download_data.py diamonds         # both stained-glass diamonds, ~44 MB
    python3 download_data.py --check [...]    # only verify what's already there

The Tonnetz-Hex-Lattice shader needs no data. What the files are (see README
"Rebuilding the data"):
  textures/kerr_v7/kn/paths32.npy     light paths around the Kerr-Newman hole,
                                      32 layers of 2410x2410 (kn_bake_modal.py)
  textures/kerr_v7/kn/sky.npy         where each pixel's light ends up on the sky
  textures/kerr_v7/kn_disk_table_r2_20.npy  disk emission table (kn_disk.py)
  textures/starfield/gaia_cube_g117_max/*   Gaia DR3 stars (G < 11.7) as a BC6H
                                      cube map (tools/gaia_fetch.py, gaia_cube_bake.py)
  textures/starfield/nebula_photo4.npy      Milky Way dust + nebula photo map
  textures/pulsar_orrery/*.npy        star / planet / ring surface maps
                                      (pulsar_orrery_prep.py)
  textures/nebulabrot/*.npy           Buddhabrot density bakes (nebula_bake_modal.py)
  textures/illusion_diamond/backdrops/*     stained-glass window + its pane data
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
FILES = [   # group, size, sha256, path
    ("kerr", 2973747328, "e852551dd73285218e51e2517535ca4938387a3292f79461e59965ab2302fc59", "textures/kerr_v7/kn/paths32.npy"),
    ("kerr", 371718528, "68da545889f63b11111b8527daa004ea766ec30dc023fdb1d4aee75872c54d61", "textures/kerr_v7/kn/sky.npy"),
    ("kerr", 16512, "aa0e1a6ae8d8cf350c470ec6eaf110fc5928d89435334a0c5d288f422fa083f5", "textures/kerr_v7/kn_disk_table_r2_20.npy"),
    ("kerr", 8595, "1f6689877e4d954a76590a1cab48f650a5436ece006539a9c4fc0ed844fb0daa", "textures/starfield/gaia_cube_g117_max/gaia_cube.json"),
    ("kerr", 2147483808, "1597f135630c724fa5693bc2bb3a37972740d5bed25de056267a51a551d2b8bd", "textures/starfield/gaia_cube_g117_max/sky64k_bc6h.bin"),
    ("kerr", 1073741952, "a2a3754eca8516916506b626152c067d453770c9705225b04f9e41bc6be32f3f", "textures/starfield/nebula_photo4.npy"),
    ("kerr", 6294656, "11d18263c4320f824778220f35e9fea97801e131a02fe11c519afa9e5928a012", "textures/pulsar_orrery/dwarf.npy"),
    ("kerr", 524928, "c27bb430bd96ba4ab87ca54cb7018cc9d7553ca45441f7508d40d35b7e86fd2b", "textures/pulsar_orrery/chromo.npy"),
    ("kerr", 1574528, "fb6c8e7dce56ad6bc481858949ee4ee3d6528e1f8a0a9d77e801397349a278d6", "textures/pulsar_orrery/giant.npy"),
    ("kerr", 1574528, "ac2cce8897839981ea4515fca1ad2777b342e67aebf327530f21bd114de28cb1", "textures/pulsar_orrery/rock.npy"),
    ("kerr", 2176, "12602f2d34b026d6c11b1160fabd8ed263aca682668e3247d42054fe498641cd", "textures/pulsar_orrery/ringprof.npy"),
    ("nebulabrot", 2147483776, "8903cda27cfa68fdb94312e49b7c8449d473ac5efbbaf1de500bbfc6915ff69e", "textures/nebulabrot/nebula16384_rgba.npy"),
    ("nebulabrot", 2147483776, "e60aeb13695617ae54c5323872f5d3b8c13556fb75a16e89c9abe236a4024602", "textures/nebulabrot/nebula_w0_rgba.npy"),
    ("nebulabrot", 2147483776, "1444b20a6c6732ada6daff186b3b0c9937d283ec78387c8a1c7eaae284671e99", "textures/nebulabrot/nebula_w1_rgba.npy"),
    ("nebulabrot", 2147483776, "31e7206337d3d41ef9f8ef012c2202715db9ea8ecf9fb35770d2d0bd3cb7199b", "textures/nebulabrot/nebula_glitter_mcmc16384_eq_rgba.npy"),
    ("nebulabrot", 536871040, "8cb9888513f513b65ce856fdd6678967f088b1828d177892d0a2422a361aeab4", "textures/nebulabrot/nebula_anti8192_rgba.npy"),
    ("diamonds", 19396705, "a2ba4304f3f399248b79c2be59238dd0d475a9358f53857b2558339c214024a4", "textures/illusion_diamond/backdrops/azathoth_glass.png"),
    ("diamonds", 15871139, "a5ff613922ae56ad793159cd0c30ecba71246a2b23d7aeba8644230d1c7bcff0", "textures/illusion_diamond/backdrops/azathoth_glass_data.png"),
    ("diamonds", 9155654, "61fc3a3e44090d770ba3bedc81870d7182d6feabf9f8fae5798e9d89c56975c3", "textures/illusion_diamond/backdrops/Picsart_26-10-04_19-14-14-181.png"),
]
GROUPS = sorted({g for g, *_ in FILES})


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
    ap.add_argument("groups", nargs="*", choices=GROUPS + [[]], metavar="group",
                    help=f"which data to fetch: {', '.join(GROUPS)} (default: all)")
    ap.add_argument("--check", action="store_true", help="verify only, download nothing")
    args = ap.parse_args()
    want = set(args.groups) or set(GROUPS)
    bad = 0
    for group, size, digest, rel in FILES:
        if group not in want:
            continue
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
