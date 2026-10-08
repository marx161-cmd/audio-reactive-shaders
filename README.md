# audio-reactive-shaders

Music-reactive GLSL shaders that run as a live desktop background on Linux/X11,
plus an offline pipeline that renders the same shaders into HDR10 music videos.

Proof of concept, not maintained. It runs every day on one machine
(Fedora, AMD RX 9060 XT, X11/i3); anything beyond that is untested.

## Shaders

**Kerr-Black-Hole-Visualizer** (`shaders/kerr-black-hole-visualizer.mp`):
a spinning Kerr-Newman black hole (spin a\* = 0.997, charge Q\* = 0.07) with a
pulsar and a red dwarf on beat-locked orbits that dive past it, plus planets.
Light is not approximated: every pixel follows a real null geodesic,
precomputed into a path table and walked as a polyline, so objects behind the
hole show up as true Einstein arcs and counter-images. The sky is real Gaia DR3
stars with a photographed Milky Way and nebulae, lensed the same way.
There is no permanent accretion disk. Each time a star swings close, a
volumetric disk winds out of its side, is full at the closest approach and is
eaten from the outside in as the star leaves; the red dwarf's disk is weaker
and warmer than the pulsar's, and at the horseshoe swap, when both stars dive
past within a few beats, they feed one disk. The disk is the one from the
[NPGS](https://github.com/baopinshui/NPGS) Kerr-Newman renderer, marched along
the precomputed light paths. The pulsar's beams turn twice per beat. The music
drives it: loudness slowly turns the sky, treble splits star light into a
spectrum outside the Einstein ring, and the orbits run on a beat clock that
tracks the song's tempo.

**Nebulabrot-Infinite-Dive** (`shaders/nebulabrot-infinite-dive.mp`): the
Nebulabrot (the Buddhabrot coloured by escape time) from precomputed
high-resolution density bakes, with an endless dive: a path of stops into the
set whose last stop is a copy of the start, so it loops forever. Auto-dive moves
on every 32 beats; see "Controls". Bass makes the dive run faster.

**Illusion-Diamond-Stained-Glass** and **Disnub-Diamond-Stained-Glass**
(`shaders/illusion-diamond-stained-glass.mp`, `shaders/disnub-diamond-stained-glass.mp`):
a ray-traced gem (total internal reflection, Fresnel, dispersion) in front of a
backlit stained-glass window whose light rises with the bass. The second gem is
cut as the great disnub dirhombidodecahedron (Skilling's figure).

**Tonnetz-Hex-Lattice** (`shaders/tonnetz-hex-lattice.mp`): the Tonnetz (the
music-theory lattice of pitch classes) as a polar hex field seen from above. Angle is pitch class, each ring outward is one octave
up, and each hex pin rises with its band's loudness, so a chord shows up as
the same shape repeated across octaves.

## How it fits together

```
audio ──> EasyEffects + shader_bands patch ──(shared memory: 120 bands)──> renderer.py ──> backdrop windows
                                                                              │
track file ──> video/render_video.py (same band analysis, offline) ───────────┴──> HDR10 AV1 video
```

- **[easyeffects-shader-bands](https://github.com/marx161-cmd/easyeffects-shader-bands)**:
  EasyEffects with a native band analyser: 120 semitone-spaced band-pass
  filters (27.5 Hz to 28 kHz), published to `/dev/shm/shader_bands` for
  any program to read. The renderer reads it via `shared_bands.py`.
- **`renderer.py`**: draws the active shader into one square 2410×2410
  canvas and shows each connected monitor its crop of it, as an
  override-redirect window below everything else (a wallpaper). Shaders
  hot-reload on save; `active_shader.txt` picks the shader (the patched
  EasyEffects has a picker for it).
- **`video/`**: the offline renderer. It runs the same shader code with
  higher sample counts (`#define OFFLINE 1`) and the same band analysis on
  the track file, and writes 10-bit HDR10 AV1 (VAAPI).

## Requirements

- Linux, X11 (the backdrop uses X11 override-redirect windows; Wayland is not supported)
- The patched EasyEffects from [easyeffects-shader-bands](https://github.com/marx161-cmd/easyeffects-shader-bands)
- An OpenGL 3.3 GPU. VRAM: the Kerr shader keeps about 6.6 GB of tables on
  the GPU (plan for 8 GB free), the Nebulabrot about 9 GB (realistically a 16 GB
  card), the diamonds and the Tonnetz almost none.
- Python 3.11+, `playerctl` (optional: per-song beat maps), ffmpeg with
  `av1_vaapi` for the video pipeline

## Setup

```sh
git clone https://github.com/marx161-cmd/audio-reactive-shaders ~/audio-reactive-shaders
cd ~/audio-reactive-shaders
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 download_data.py            # all data, ~16 GB; or pick: kerr | nebulabrot | diamonds
cp config.example.toml config.toml  # optional: audio delay, which monitors
echo kerr-black-hole-visualizer.mp > active_shader.txt
.venv/bin/python3 renderer.py
```

To run it as a service, see `contrib/audio-reactive-shaders.service`.
Frame times are printed as `perf:` lines.

In the patched EasyEffects, set the renderer location in
`~/.config/easyeffects/shader_bands.conf`:

```ini
[backdrop]
dir=~/audio-reactive-shaders
```

### Controls

The backdrop never takes keyboard focus. Bind `contrib/orrery-cam`
(`up|down|left|right|in|out|reset`) to window-manager keys to move the camera.

For the Nebulabrot, bind `tools/dive.py next`, `tools/dive.py prev` and
`tools/dive.py auto toggle`. `tools/dive.py add` saves the current view as a
new stop (the default path ships with the shader; `tools/dive.py reset`
restores it), `list` shows the path.

## Music videos

```sh
video/render.sh song.flac                                  # full track, HDR10 AV1
EXTRA_ARGS="--start 45 --frames 300" video/render.sh song.flac preview.mp4
SHADER=tonnetz-hex-lattice.mp MODE=sdr video/render.sh song.flac
```

The renderer loads its own copy of the shader's data, so on a 16 GB card stop
the live backdrop while rendering the Kerr or Nebulabrot shaders (both copies
do not fit).

The camera and sky position are taken from the live renderer's state files, so
the video frames the scene the way your backdrop does (a fresh install starts
on a framing with a nebula in view). The HDR10 metadata
defaults (MaxCLL 2163, MaxFALL 96) were measured for the Kerr shader at the
default grade. If you change the grade, measure again.

## Rebuilding the data

Everything `download_data.py` fetches can be rebuilt:

| data | script | source |
|---|---|---|
| `textures/kerr_v7/kn/paths32.npy`, `sky.npy` | `kn_bake_modal.py` (runs on a [Modal](https://modal.com) cloud H100) | Kerr-Newman geodesic integration |
| `textures/kerr_v7/kn_disk_table_r2_20.npy` | `kn_disk.py --rin 2 --rout 20` | disk emission model |
| `textures/starfield/gaia_cube_g117_max/` | `tools/gaia_fetch.py` (ESA Gaia archive, ~104 M stars), then `GAIA_G_MAX=11.7 GAIA_MIP_MAX=1 GAIA_OUT=textures/starfield/gaia_cube_g117_max gaia_cube_bake.py` | Gaia DR3 |
| `textures/starfield/nebula_photo4.npy` | not rebuildable from this repo: hand-edited photos (see credits) | ESO, see credits |
| `textures/pulsar_orrery/*.npy` | `pulsar_orrery_prep.py` | see credits |
| `textures/nebulabrot/*.npy` | `nebula_bake_modal.py` (Modal H100; uniform, window and Metropolis modes), then `nebula_prep.py`, `nebula_window_prep.py`, `nebula_eq_prep.py` | the Mandelbrot iteration |
| `textures/illusion_diamond/backdrops/*_glass_data.png` | `stained_glass_prep.py` | the stained-glass image |

The older NASA SVS star map (`textures/starfield/sky64k/`, `skycube_unpack.py` +
`skycube_bake.py`) is still in the dataset for the previous Kerr version but no
longer used.

`skycube_bake.py` encodes BC6H with the
[Intel ISPC Texture Compressor](https://github.com/GameTechDev/ISPCTextureCompressor),
built into `tools/ispc_texcomp/` (see `tools/bc6h_ispc.py`).

The published `paths32.npy` has one bake defect repaired: 57 rows where
sample #9 came out zero, which showed up as a streak of stars. Check a fresh
bake for zeroed samples before using it.

## Credits

- Accretion disk: the volumetric disk of [NPGS](https://github.com/baopinshui/NPGS)
  by baopinshui (GPL-3.0), ported onto this shader's precomputed light paths.
- Stars: ESA/Gaia/DPAC, Gaia DR3 (CC BY-SA 3.0 IGO).
- Milky Way: ESO/S. Brunier, eso0932a (CC BY 4.0), edited (stars removed, recoloured).
- Southern Ring Nebula (NGC 3132) and Helix Nebula (NGC 7293): ESA (CC BY 4.0),
  edited (stars removed), placed on the sky map.
- Stained-glass window and the diamond's portal picture: the author's own images.
- Older star map (still in the dataset): NASA/Goddard Space Flight Center
  Scientific Visualization Studio, *Deep Star Maps 2020*.
- Sun and Saturn textures (and the ring alpha): [Solar System Scope](https://www.solarsystemscope.com/textures/), CC BY 4.0.
- Chromosphere: NASA STEREO/EUVI + SDO/AIA 304 Å Carrington map.
- Moon surface: NASA LRO / LROC.

## License

GPL-3.0-or-later. See `LICENSE`.

## Related projects

- [easyeffects-shader-bands](https://github.com/marx161-cmd/easyeffects-shader-bands) — patched EasyEffects that produces the 120-band `/dev/shm/shader_bands` feed these shaders read.

## Demo

<!-- Upload a shader music video and paste the link here, e.g. (replace VIDEOID):
[![Watch the shader music video](https://img.youtube.com/vi/VIDEOID/maxresdefault.jpg)](https://youtu.be/VIDEOID)
-->
