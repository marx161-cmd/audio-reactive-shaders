#!/usr/bin/env python3
"""Download the shaders' data into textures/ (and phone/backdrops/), ~17 GB for everything.

Hosted as a Hugging Face dataset. Standard library only; resumes partial
downloads and checks every file's SHA-256. Pick what you need:

    python3 download_data.py                  # everything
    python3 download_data.py kerr             # the three Kerr shaders, ~7 GB
    python3 download_data.py nebulabrot       # Nebulabrot-Infinite-Dive, ~9.1 GB
    python3 download_data.py diamonds         # both stained-glass diamonds, ~44 MB
    python3 download_data.py ringbox          # Ringbox, ~16 MB
    python3 download_data.py embers           # Ember-Flock-Metal-Pit, ~130 MB
    python3 download_data.py dwarf            # Brown-Dwarf + Brown-Dwarf-Glass-Box, ~3.3 GB (incl. the sky)
    python3 download_data.py gems             # Armillary-Gem-Sphere + Disnub-Diamond-Checker-Tunnel, ~175 MB
    python3 download_data.py tunnel           # Disnub-Diamond-Checker-Tunnel, ~18 MB
    python3 download_data.py phone            # wall images for the phone wallpapers, ~18 MB
    python3 download_data.py --check [...]    # only verify what's already there

Tonnetz-Hex-Lattice, Tonnetz-Hex-Pins and both Lagoon-Worms need no data. What
the files are (see README "Rebuilding the data"):
  textures/kerr_v7/kn/paths32.npy     light paths around the Kerr-Newman hole,
                                      32 layers of 2410x2410 (kn_bake_modal.py)
  textures/kerr_v7/kn/sky.npy         where each pixel's light ends up on the sky
  textures/kerr_v7/kn/lens_inv*.npy   the same bake inverted into a lens grid (lens_inv_bake.py)
  textures/kerr_v7/kn_disk_table_r2_20.npy  disk emission table (kn_disk.py)
  textures/slurp/gasbake_flip.npy     the torn star's gas, a pre-lensed flipbook (85 frames)
  textures/starfield/gaia_cube_g117_max/*   Gaia DR3 stars (G < 11.7) as a BC6H
                                      cube map (tools/gaia_fetch.py, gaia_cube_bake.py)
  textures/starfield/nebula_photo4.npy      Milky Way dust + nebula photo map
  textures/pulsar_orrery/*.npy        star / planet / ring surface maps
                                      (pulsar_orrery_prep.py)
  textures/nebulabrot/*.npy           Buddhabrot density bakes (nebula_bake_modal.py)
  textures/illusion_diamond/backdrops/*     stained-glass windows + their pane data,
                                      the checker tunnel
  textures/stained_glass_space/*.png  the glass box walls
  textures/moon_sphere/, ringbox/     moon and sun surface maps
  textures/brown_dwarf/, fbm_drop_v3/ lightning decal / bolt atlases (grown on the GPU, dielectric breakdown)
  textures/fbm_drop_v6/*              the gem sphere's backdrop image + its depth and normals
  phone/backdrops/*.png               wall images for the phone wallpapers
"""
import argparse
import hashlib
import os
import sys
import urllib.request

REPO = "marx161-cmd/audio-reactive-shaders-data"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
ROOT = os.path.dirname(os.path.abspath(__file__))

# groups (space-separated: a file shared by several shaders is in each group), size, sha256,
# path (relative to the repo root and to the dataset root)
FILES = [
    ("kerr", 2973747328, "e852551dd73285218e51e2517535ca4938387a3292f79461e59965ab2302fc59", "textures/kerr_v7/kn/paths32.npy"),
    ("kerr", 371718528, "68da545889f63b11111b8527daa004ea766ec30dc023fdb1d4aee75872c54d61", "textures/kerr_v7/kn/sky.npy"),
    ("kerr", 16512, "aa0e1a6ae8d8cf350c470ec6eaf110fc5928d89435334a0c5d288f422fa083f5", "textures/kerr_v7/kn_disk_table_r2_20.npy"),
    ("kerr dwarf", 8595, "1f6689877e4d954a76590a1cab48f650a5436ece006539a9c4fc0ed844fb0daa", "textures/starfield/gaia_cube_g117_max/gaia_cube.json"),
    ("kerr dwarf", 2147483808, "1597f135630c724fa5693bc2bb3a37972740d5bed25de056267a51a551d2b8bd", "textures/starfield/gaia_cube_g117_max/sky64k_bc6h.bin"),
    ("kerr dwarf", 1073741952, "a2a3754eca8516916506b626152c067d453770c9705225b04f9e41bc6be32f3f", "textures/starfield/nebula_photo4.npy"),
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
    ("embers", 22034017, "010ba3ac9c7594ff32944e7b2351149b6854b88b4446f0a5b609bf08c989741f", "textures/illusion_diamond/backdrops/sg_78149_glass.png"),
    ("embers", 20990411, "41229d11c376bbee5d0acaae6b71b852b23b9c46a9ddbed12ffd249771118679", "textures/illusion_diamond/backdrops/sg_78149_glass_data.png"),
    ("embers", 20066143, "8caa18fc7af424dd9f6fb2028432f0ab9eda2fa4e6ae5fd9ded8809f73c487b6", "textures/illusion_diamond/backdrops/sg_78150_glass.png"),
    ("embers", 22786177, "249bd9ecdaa148f9eeda19d57818ac47ce7b73fe6da8533e600c35f3bc06be36", "textures/illusion_diamond/backdrops/sg_78150_glass_data.png"),
    ("embers", 22087115, "cdff1c7ac530924af2e12552d2c92e5dacc9f91a9cd3a718d2eec85c61106ee0", "textures/illusion_diamond/backdrops/sg_78151_glass.png"),
    ("embers", 21706972, "1a1f7e9a39f0d50938cc62a72e32af54571a6ca66ad73aaf7bef5cae684e2d73", "textures/illusion_diamond/backdrops/sg_78151_glass_data.png"),
    ("tunnel gems", 18282545, "9dc21b4074536cab455d96f8994d6000ca2594cc3c8cf0b28ac384c51b760e60", "textures/illusion_diamond/backdrops/tunnel_cc.png"),
    ("phone", 3147687, "51e9d6d6f43f011302545b20e1b25f5497395036e842233ee0ad8c88dabc2b2d", "phone/backdrops/dio_backC.png"),
    ("phone", 1970166, "a18695736f56ee8d4da52b069c758304c7864bc9ee19656bb1de436f93e82204", "phone/backdrops/dio_botC.png"),
    ("phone", 1013644, "502d525cd73ba7611207f3c4fdbb9a1f96197b199d7948e06ed5da118049d965", "phone/backdrops/dio_botS.png"),
    ("phone", 2125875, "7021afd4df1b1dec918b50cd6c8112a7fe9ea799e04791c464f76dfa2a700b5b", "phone/backdrops/dio_topC.png"),
    ("phone", 1044656, "0f39953834ec3897342d2912eedfab9e1b05c04977dff6544a4a2e817b16a5ee", "phone/backdrops/dio_topS.png"),
    ("phone", 3530958, "6e92c751dd129a9a28e00d73e368031948c5518604bdd6b4fa176f785e8a1e28", "phone/backdrops/dio_wlC.png"),
    ("phone", 979830, "f2b27e3b545639f3224081d20ca570e3fc7614fe8fa8a397c6860b2e8bb00034", "phone/backdrops/dio_wlS.png"),
    ("phone", 3491918, "909d45d77d3b0288f8b08f228e3e1d1bcb5ce68ea0dbcaf807a1c9ab58c15781", "phone/backdrops/dio_wrC.png"),
    ("phone", 932987, "71bcf132e6e450dcaa977c1bca32d866e9362e93b1cb247b6df48e887da51c0c", "phone/backdrops/dio_wrS.png"),
    ("kerr", 401080448, "901e6c117484bb11f50a1f9f3bee4c462c974f715f5b3dd35a08aeed9c2b84e4", "textures/slurp/gasbake_flip.npy"),
    ("kerr", 9633920, "9d98f214a05a153c3c652db4a89bef988704b14f39ad16e56817c8f2ba6292f3", "textures/kerr_v7/kn/lens_inv.npy"),
    ("kerr", 2359424, "3f81b434a94c6b30fd57bfd7694b820ae1b20f0da0f58a318629af1747393c50", "textures/kerr_v7/kn/lens_inv_count.npy"),
    ("ringbox dwarf", 10063546, "83ee217582046454b579502397e223ebd02fb5de28b17bf9a2ebd6f5974978a2", "textures/moon_sphere/iChannel0.png"),
    ("ringbox", 3105072, "716722efb2313b5246d8dab55af4e75b8526efec4cf704c766bcd33be8f027d8", "textures/ringbox/sun_photo.png"),
    ("ringbox", 2767776, "a9fd336d55a5c7de2e6a9f42979a438944e7d1a9be506b354d7f388a6ee12d0a", "textures/ringbox/sun_euv.png"),
    ("dwarf", 5243008, "6da8c0d1fe4a1764324cda251e2b0933c3395ac14da187093b720c6afa5bc444", "textures/brown_dwarf/bolt_decals_u8.npy"),
    ("dwarf", 11286494, "0c878262cf86bdd2a5a6d3155a647c9d0907f5d63d14085444081d45d72c2f9e", "textures/stained_glass_space/floor-backwall.png"),
    ("dwarf", 13376380, "2aa49890513ef45a51bb5fe7c66737dda43495b28ac156b6f9b2b99dbe281a10", "textures/stained_glass_space/long-wall-l.png"),
    ("dwarf", 15214456, "ffa711bcdb6186f838025c09f957af6a587b1cf310a7b69f02b72cac2b9aabd4", "textures/stained_glass_space/long-wall-r.png"),
    ("dwarf", 21727378, "b13beffe0b6f4f99e84a2e0716634db765f4158407871ebbe8daf13d52f847ff", "textures/stained_glass_space/short-wall.png"),
    ("dwarf", 28745766, "a4fd29acc9be4783342aedd3f44b2ba55a5702299a196d551f283dd672980b0e", "textures/stained_glass_space/short-wall-top.png"),
    ("gems", 1310848, "6b317526d30ff93d78810b72d7e7bc1734be364b19a917950075ec82049464bc", "textures/fbm_drop_v3/bolts_u8.npy"),
    ("gems", 37136, "3722ed00aeefa02968bfb77d7753cd7ff6ec5703bfe58ad5f0a081e5df0bc297", "textures/fbm_drop_v3/hops_f32.npy"),
    ("gems", 35998016, "7c499eb5182a653942fa23599e03b69605f9275331cc873b9e883ebc6a71fef8", "textures/fbm_drop_v6/backdrop_u8.npy"),
    ("gems", 71995904, "cf95594cc9a61f3b97e76b236e82b18480cc2ccee7f86818d706253d2a614965", "textures/fbm_drop_v6/backdrop_normal_f16.npy"),
    ("gems", 47997312, "2e7362be46ef35c4c086ff660a361ba6460994c1f8c9d8f8a71d6e9d39c9c0bd", "textures/fbm_drop_v6/backdrop_depth_f32.npy"),
]
GROUPS = sorted({g for gs, *_ in FILES for g in gs.split()})


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
        if not want & set(group.split()):
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
