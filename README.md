# audio-reactive-shaders

Music-reactive GLSL shaders that run as a live desktop background on Linux/X11,
plus an offline pipeline that renders the same shaders into HDR10 music videos.

Proof of concept, not maintained. It runs every day on one machine
(Fedora, AMD RX 9060 XT, X11/i3); anything beyond that is untested.

## Shaders

**Kerr-Black-Hole-Visualizer** (`shaders/kerr-black-hole-visualizer.mp`):
a spinning Kerr-Newman black hole (spin a\* = 0.997, charge Q\* = 0.07) with an
accretion disk and a small system of stars, a pulsar and planets passing behind
and in front of it. Light is not approximated: every pixel follows a real null
geodesic, precomputed into a path table and walked as a polyline, so objects
behind the hole show up as true Einstein arcs and counter-images. The sky is the
NASA SVS 2020 Deep Star Map at full 64k resolution, lensed the same way.
The music drives it: bass pushes on the sky like a rubber sheet, treble splits
star light into a spectrum outside the Einstein ring, and orbits run on a beat
clock that tracks the song's tempo.

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
- An OpenGL 3.3 GPU. The Kerr shader keeps about 5.6 GB of tables on the
  GPU, so plan for 6 to 7 GB of free VRAM. The Tonnetz shader needs almost none.
- Python 3.11+, `playerctl` (optional: per-song beat maps), ffmpeg with
  `av1_vaapi` for the video pipeline

## Setup

```sh
git clone https://github.com/marx161-cmd/audio-reactive-shaders ~/audio-reactive-shaders
cd ~/audio-reactive-shaders
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 download_data.py            # ~5.5 GB, only needed for the Kerr shader
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

## Music videos

```sh
video/render.sh song.flac                                  # full track, HDR10 AV1
EXTRA_ARGS="--start 45 --frames 300" video/render.sh song.flac preview.mp4
SHADER=tonnetz-hex-lattice.mp MODE=sdr video/render.sh song.flac
```

The camera and sky position are taken from the live renderer's state files, so
the video frames the scene the way your backdrop does. The HDR10 metadata
defaults (MaxCLL 2163, MaxFALL 96) were measured for the Kerr shader at the
default grade. If you change the grade, measure again.

## Rebuilding the data

Everything `download_data.py` fetches can be rebuilt:

| data | script | source |
|---|---|---|
| `textures/kerr_v7/kn/paths32.npy`, `sky.npy` | `kn_bake_modal.py` (runs on a [Modal](https://modal.com) cloud H100) | Kerr-Newman geodesic integration |
| `textures/kerr_v7/kn_disk_table.npy` | `kn_disk.py` | disk emission model |
| `textures/starfield/sky64k/` | `skycube_unpack.py`, then `skycube_bake.py` (torch, ~7 min on a GPU) | NASA SVS *Deep Star Maps 2020* (svs.gsfc.nasa.gov), `starmap_2020_64k.exr` |
| `textures/pulsar_orrery/*.npy` | `pulsar_orrery_prep.py` | see credits |

`skycube_bake.py` encodes BC6H with the
[Intel ISPC Texture Compressor](https://github.com/GameTechDev/ISPCTextureCompressor),
built into `tools/ispc_texcomp/` (see `tools/bc6h_ispc.py`).

The published `paths32.npy` has one bake defect repaired: 57 rows where
sample #9 came out zero, which showed up as a streak of stars. Check a fresh
bake for zeroed samples before using it.

## Credits

- Star map: NASA/Goddard Space Flight Center Scientific Visualization Studio,
  *Deep Star Maps 2020*. Gaia DR2 (ESA/Gaia/DPAC), Hipparcos-2, Tycho-2.
- Sun and Saturn textures (and the ring alpha): [Solar System Scope](https://www.solarsystemscope.com/textures/), CC BY 4.0.
- Chromosphere: NASA STEREO/EUVI + SDO/AIA 304 Å Carrington map.
- Moon surface: NASA LRO / LROC.

## License

GPL-3.0-or-later. See `LICENSE`.
