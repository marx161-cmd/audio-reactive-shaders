#!/usr/bin/env python3
"""Minimal test renderer: proves dGPU context + shared-memory band data +
a live shader work end to end, before wiring in real "big boy" shaders or
xwinwrap desktop-backdrop integration. One window, undecorated, positioned
at a target output's origin -- not yet click-through/always-below (that's
the xwinwrap step, once this is proven).
"""
import datetime
import json
import math
import os
import re
import subprocess
import sys
import threading
import tomllib
import time
from collections import deque

# Must be set before glfw/GLX touches the X connection -- this is what
# forces Mesa's DRI3 loader onto the dGPU instead of the default iGPU,
# same mechanism validated tonight's mpv tests.
#
# Hard assignment (NOT setdefault): ~/.profile globally exports
#   DRI_PRIME=pci-0000_11_00_0
#   MESA_VK_DEVICE_SELECT=1002:15bf
#   MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1
# to keep shell-launched desktop apps on the iGPU (Radeon 760M). setdefault
# silently deferred to that inherited iGPU selection, which put THIS renderer
# on the iGPU too -- measured 2026-09-06: iGPU pinned at 99% busy and the
# backdrop dropped to ~12 fps, lagging the whole streamed desktop. This
# renderer is the one process that must ALWAYS run on the dGPU (RX 9060 XT,
# pci-0000_03_00_0), so it overrides the inherited iGPU selection outright.
os.environ["DRI_PRIME"] = "1"
os.environ.pop("MESA_VK_DEVICE_SELECT", None)
os.environ.pop("MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE", None)
os.environ.setdefault("DISPLAY", ":0")

import glfw
import moderngl
import numpy as np
import png as png16  # pypng -- PIL silently truncates 16-bit RGB PNGs to
                      # 8-bit on load (confirmed: Image.open() round-trips a
                      # 16-bit test file back as dtype uint8), so a real
                      # 16-bit-source texture needs a loader that doesn't do
                      # that. Only used for the 16-bit path in
                      # load_channel_texture() below -- 8-bit PNGs still go
                      # through PIL as before.
from PIL import Image
from Xlib import X, XK
from Xlib.display import Display

sys.path.insert(0, os.path.dirname(__file__))
from shared_bands import BandsReader, N_BANDS
import beat_session
import beat_track
import stem_split
from audio_texture import AudioTexReader, WIDTH as AUDIO_TEX_WIDTH
import importlib.util

# Real Shadertoy audio sample rate our native EasyEffects plugin actually
# runs at (see shader-musicvideo's shader_bands.cpp) -- fed as iSampleRate
# for shaders that reference it, though almost none do anything meaningful
# with the literal value.
AUDIO_SAMPLE_RATE = 48000.0

TEXTURES_DIR = os.path.join(os.path.dirname(__file__), "textures")

# Every named scalar feature in the shared_bands contract, exposed as a
# plain uniform -- same auto-wire philosophy as ShaderEditor's
# BuiltinSystemUniforms on the phone: a shader declares `uniform float
# u_<name>;` for whichever ones it wants, the rest are silently unused.
# Tuning (gain, attack/release, band-curve shaping) lives one layer down,
# in live_tap_writer.py's Envelope constants -- not here, and not per-shader.
META_FEATURES = (
    "rms", "sub", "bass", "lowmid", "mid", "highmid",
    "presence", "brilliance", "centroid", "flux", "onset",
)

# Loudness features (dB-normalized 0..1) — these get a floor re-map before
# being exposed as uniforms (see condition_level below). centroid gets its
# own log-map (condition_centroid); flux/onset are already normalized
# ratios and pass through as-is.
LEVEL_FEATURES = (
    "rms", "sub", "bass", "lowmid", "mid", "highmid",
    "presence", "brilliance",
)


def condition_level(x: float) -> float:
    # The native plugin's norm_db() maps loudness as (db + 60) / 60, i.e. a
    # -60dB floor. Real music sits at -30..-10dB, so these cluster at
    # 0.4-0.8 with a tiny standard deviation and "react" as a faint shimmer.
    # Re-map the same dB value to a -40dB floor — smooth and linear-in-dB,
    # no gate/threshold (a hard gate was tried and reverted elsewhere) — so
    # the musical range spreads across 0..1:
    #   db = 60*x - 60   ->   y = (db + 40) / 40 = 1.5*x - 0.5.
    # This is what native norm_db()'s FLOOR_DB ought to be; kept here rather
    # than in shader_bands.cpp to avoid a full EasyEffects rebuild, same
    # reasoning as SHAPE_EXP further down.
    return min(1.0, max(0.0, 1.5 * x - 0.5))


def condition_centroid(x: float) -> float:
    # Native plugin writes centroid = centroid_hz / LOG_HI (LOG_HI = 28160,
    # linear in Hz), so music (centroids ~500Hz-4kHz) reads ~0.018-0.14 and
    # never approaches the top of its range. Recover Hz and log-map over the
    # audible 20Hz-16kHz span so it actually tracks brightness 0..1.
    # 2026-09-20: multiplier was 16000 -- stale since LOG_HI was widened
    # 16000 -> 28160 (the "exact 10 octaves" change), which read every
    # centroid Hz ~1.76x low and compressed mid brightness.
    hz = max(20.0, x * 28160.0)
    return min(1.0, math.log2(hz / 20.0) / math.log2(16000.0 / 20.0))


# Per-machine settings (config.toml next to this file, git-ignored; see
# config.example.toml). Read once at startup, except the audio delay file,
# which is re-read live so a fresh A/V calibration applies without a restart.
CONFIG_FILE = os.environ.get(
    "SHADER_BACKDROP_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml"))


def load_config() -> dict:
    try:
        with open(CONFIG_FILE, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}


CONFIG = load_config()
_AUDIO_CFG = CONFIG.get("audio", {})

# Audio/video delay: shifts WHICH audio snapshot feeds the shader, so the
# picture lines up with a speaker that plays late (e.g. a Bluetooth or network
# speaker ~1 s behind). Same clock as BandsReader's timestamp (CLOCK_MONOTONIC).
# Either a fixed `delay_ms`, or `delay_file`: a KEY=VALUE file re-read live
# (value in ms under `delay_file_key`), so an external calibration tool can
# update it while the renderer runs.
AUDIO_DELAY_MS = float(_AUDIO_CFG.get("delay_ms", 0.0))
AUDIO_DELAY_FILE = os.path.expanduser(_AUDIO_CFG.get("delay_file", ""))
AUDIO_DELAY_FILE_KEY = _AUDIO_CFG.get("delay_file_key", "VIDEO_DELAY_MS")


def read_audio_delay_seconds() -> float:
    if AUDIO_DELAY_FILE:
        try:
            with open(AUDIO_DELAY_FILE) as f:
                for line in f:
                    if line.startswith(AUDIO_DELAY_FILE_KEY + "="):
                        return max(0.0, float(line.strip().split("=", 1)[1]) / 1000.0)
        except (FileNotFoundError, ValueError, IndexError):
            pass
    return max(0.0, AUDIO_DELAY_MS / 1000.0)


def pick_delayed_index(history, target_t):
    """Index form of pick_delayed_frame. The stem separation needs a window
    CENTRED on the delay-aligned row, not just the row itself, so it needs to
    know where that row sits -- and the centring is only affordable because
    this pipeline already runs behind real time by the audio delay, meaning the
    "future" half of the window has already been published. Same walk-back and
    same fallbacks as pick_delayed_frame; kept separate so that function's
    existing callers are untouched."""
    if not history:
        return None
    if target_t <= history[0]["timestamp"]:
        return 0
    for i in range(len(history) - 1, -1, -1):
        if history[i]["timestamp"] <= target_t:
            return i
    return 0


def pick_delayed_frame(history, target_t):
    # `history` is time-ordered oldest-first (deque, appended on the right).
    # Walk back from the newest frame until we find one at or before the
    # target time; if the buffer doesn't reach back far enough yet (startup,
    # or the delay just increased), fall back to the oldest we have rather
    # than stalling on nothing.
    if not history:
        return None
    if target_t <= history[0]["timestamp"]:
        return history[0]
    for f in reversed(history):
        if f["timestamp"] <= target_t:
            return f
    return history[0]


SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")
# Written by the EasyEffects "Shader Bands" QML page (ShaderFileIO::setActiveShader)
# -- a bare filename, e.g. "sin_wave.glsl". Watched below alongside the shader's
# own mtime so picking a different shader in-app takes effect live, no restart,
# same as editing the .glsl text directly.
ACTIVE_SHADER_FILE = os.path.join(os.path.dirname(__file__), "active_shader.txt")
MOUSE_MODE_FILE = os.path.join(os.path.dirname(__file__), "mouse_mode.txt")

# Keyboard-driven camera state, same hot-read file convention as
# active_shader.txt / mouse_mode.txt. One line: "<elev> <yaw> <zoom>", radians
# and a zoom multiplier. Written by contrib/orrery-cam, which is what a WM
# keybinding actually calls -- the renderer never reads the keyboard itself.
#
# That indirection is the whole point: this backdrop is an override-redirect
# window that never receives focus, so it cannot be sent key events, and
# polling global key state (XQueryKeymap) would fire on keys pressed in other
# applications -- the same failure that made cursor steering unusable. A WM
# binding is an unambiguous "the user meant this" signal.
CAMERA_FILE = os.path.join(os.path.dirname(__file__), "camera_state.txt")
# 2026-09-21: the hold-to-steer free-fly offsets (u_cam_look / u_cam_move) used
# to live ONLY in this process, so a framing dialled in by hand was lost on any
# restart and could not be read back to freeze as shader constants. Persisted
# here as six floats, written when they change and reloaded at startup.
FREEFLY_FILE = os.path.join(os.path.dirname(__file__), "freefly_state.txt")
FREEFLY_KEYS = ("yaw", "pitch", "roll", "r", "u", "f")
# 2026-09-28: phone gyro as a second steering hand. cybersyn-hid-relay
# (--gyro-target shader) writes "<active> <gx> <gy>" here while vol-down is held
# on the Pixel: the phone's pointing offset since the press, already in canvas
# px (y up) like a cursor. Fed into the SAME path as Super + mouse below, so
# every shader that steers by mouse steers by gyro with no shader change.
# 4th value (2026-09-28): the twist about the pointing axis, as px of Super +
# Ctrl vertical drag (u_cam_move.z: nebulabrot zoom, free-fly forward/back).
# Treated as released when the file goes stale (relay died mid-hold).
GYRO_FILE = os.path.join(os.path.dirname(__file__), "gyro_state.txt")
GYRO_STALE_S = 1.0
# 2026-09-23: full-canvas video capture (scope.md "Music-session video capture").
# Same polled-state-file convention as the files above. Written by
# capture_session.py, which watches MPRIS; the recording itself lives in
# capture.py because nothing on screen shows the whole canvas to grab.
CAPTURE_FILE = os.path.join(os.path.dirname(__file__), "capture_state.txt")


def load_freefly():
    out = {k: 0.0 for k in FREEFLY_KEYS}
    try:
        with open(FREEFLY_FILE) as fh:
            parts = fh.read().split()
        for k, v in zip(FREEFLY_KEYS, parts):
            out[k] = float(v)
    except (OSError, ValueError):
        pass
    return out


# 2026-09-27: opt-in per-shader state that survives reloads/restarts. A
# multipass shader lists `"persist": {"buffer": "A"}` in its manifest; texel
# (0,0) of that buffer is read back every PERSIST_SECONDS and written here, and
# handed back to the shader as `uniform vec4 u_persist0` (+ `u_persist_valid`
# 1.0 when a saved value exists) so its seed frame can restore it. First user:
# nebulabrot.mp's parked view (centre, zoom, rotation).
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "shader_state")
PERSIST_SECONDS = 2.0
# 2026-09-27: `uniform float u_autozoom` (0/1), toggled by Super+Z via i3 ->
# ~/bin/backdrop-autozoom. A WM binding, not polled keys: see CAMERA_FILE's note.
AUTOZOOM_FILE = os.path.join(os.path.dirname(__file__), "autozoom.txt")
# 2026-09-27: dive path -- ~/bin/backdrop-dive writes the target view
# here ("re im zoom rot id"); passed as u_cam_target (vec4) + u_cam_target_id.
CAM_TARGET_FILE = os.path.join(os.path.dirname(__file__), "cam_target.txt")
# 2026-09-27: auto-dive -- `~/bin/backdrop-dive auto` (Super+Up) flips
# autodive.txt 0/1. While on, every AUTODIVE_BEATS beats of the beat clock the
# renderer sends the camera on to the next dive stop (the shader's flight takes
# 16 beats, then it rests 16 at the stop). Past the last stop it continues with
# stop 1: the last stop is nebulabrot's loop copy, whose swap has by then put
# the view back at the start. A manual jump (Super+Left/Right) restarts the count.
AUTODIVE_FILE = os.path.join(os.path.dirname(__file__), "autodive.txt")
DIVE_PATH_FILE = os.path.join(PERSIST_DIR, "nebulabrot_dive.txt")
DIVE_IDX_FILE = os.path.join(PERSIST_DIR, "nebulabrot_dive_idx.txt")
AUTODIVE_BEATS = 32.0


def persist_path(mp):
    return os.path.join(PERSIST_DIR, os.path.basename(mp["dir"].rstrip("/")) + ".txt")


def load_persist(mp):
    try:
        with open(persist_path(mp)) as fh:
            vals = [float(v) for v in fh.read().split()[:4]]
        return tuple(vals) if len(vals) == 4 else None
    except (OSError, ValueError):
        return None


def save_persist(mp):
    spec = mp["manifest"].get("persist")
    b = mp["buffers"].get((spec or {}).get("buffer", ""))
    if b is None:
        return
    raw = b["fbo"][b["cur"]].read(viewport=(0, 0, 1, 1), components=4, dtype="f4")
    vals = np.frombuffer(raw, dtype=np.float32)
    if not np.all(np.isfinite(vals)):
        return
    os.makedirs(PERSIST_DIR, exist_ok=True)
    tmp = persist_path(mp) + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(" ".join("%.9g" % v for v in vals) + "\n")
    os.replace(tmp, persist_path(mp))
    return tuple(float(v) for v in vals)


def save_freefly(cam):
    # Atomic: renderer-adjacent tooling reads this while we write it.
    try:
        tmp = FREEFLY_FILE + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(" ".join("%.6f" % cam[k] for k in FREEFLY_KEYS) + "\n")
        os.replace(tmp, FREEFLY_FILE)
    except OSError:
        pass
CAMERA_DEFAULT = (0.58, 0.35, 1.0)

# Hold-to-steer key. While this key is physically held, the shader may use
# cursor movement to aim the camera (u_steer = 1.0); otherwise cursor motion is
# ignored entirely. Read with XQueryKeymap, which reports global key state and
# needs no focus -- an override-redirect backdrop can never receive key events.
#
# A HELD KEY is the signal that makes mouse steering workable at all: cursor
# position on its own is ambiguous (the cursor moves all day for reasons that
# have nothing to do with the wallpaper), and gating on "cursor is over the
# bare desktop" is not enough either, because the cursor crosses the desktop
# constantly. Holding a key is unambiguous.
#
# Default Super_L: a big key held comfortably with the hand that is NOT on
# the mouse, and free in practice. i3's floating_modifier is Mod4, but that
# only acts when a mouse BUTTON is also held (Mod+drag moves a floating
# window, Mod+right-drag resizes) -- holding Super and merely moving the
# pointer does nothing in i3, and this steering involves no button at all,
# so the two do not overlap. The one real cost: holding Super for an i3
# chord while also moving the mouse nudges the view slightly.
# Override with any keysym name in steer_key.txt -- "Menu" (135),
# "Control_R" (105), "Scroll_Lock" (78) and "Caps_Lock" (66) all resolve on
# this pc105/de keyboard; "Alt_R" does NOT (it is AltGr/ISO_Level3_Shift
# here, keycode 0).
STEER_KEY_FILE = os.path.join(os.path.dirname(__file__), "steer_key.txt")
STEER_KEY_DEFAULT = "Super_L"
DEFAULT_SHADER_NAME = "kerr-black-hole-visualizer.mp"


def is_multipass_path(path):
    # A multipass shader is a directory (shaders/<name>.mp/manifest.json
    # + pass .glsl files) rather than a single .glsl file -- see
    # SHADER_REFERENCE.md's "Multipass shaders" section and scope.md's
    # 2026-08-23 entry for why this exists and how it's staged.
    return os.path.isdir(path)


def resolve_active_shader_path():
    try:
        with open(ACTIVE_SHADER_FILE) as f:
            name = f.read().strip()
    except FileNotFoundError:
        name = ""
    # Bare name only -- no path traversal via the pointer file. Either
    # a "<name>.glsl" single-pass file or a "<name>.mp" multipass
    # directory (must actually contain manifest.json, not just exist).
    valid_name = name and "/" not in name and "\\" not in name and (
        name.endswith(".glsl") or name.endswith(".mp")
    )
    if not valid_name:
        name = DEFAULT_SHADER_NAME
    path = os.path.join(SHADER_DIR, name)
    if name.endswith(".mp"):
        if not os.path.isfile(os.path.join(path, "manifest.json")):
            path = os.path.join(SHADER_DIR, DEFAULT_SHADER_NAME)
    elif not os.path.isfile(path):
        path = os.path.join(SHADER_DIR, DEFAULT_SHADER_NAME)
    return path


SHADER_PATH = resolve_active_shader_path()

# Output geometry, read live from `xrandr --query` at startup and re-read by
# sync_layout() -- never hardcoded (a hardcoded snapshot silently went stale
# whenever the display layout changed).
# Used only if xrandr can't be queried: one window, size from config.
_DISPLAY_CFG = CONFIG.get("display", {})
_FALLBACK_OUTPUTS = [
    {"name": "fallback", "w": int(_DISPLAY_CFG.get("fallback_w", 1920)),
     "h": int(_DISPLAY_CFG.get("fallback_h", 1080)), "x": 0, "y": 0, "hz": 60.0},
]

_XRANDR_LINE_RE = re.compile(
    r"^(?P<name>\S+) connected (?P<primary>primary )?"
    r"(?P<w>\d+)x(?P<h>\d+)\+(?P<x>\d+)\+(?P<y>\d+)"
    r"(?: (?P<rot>left|right|inverted))?"
)

# The refresh rate is NOT on the connector line -- it is on the indented mode
# line that follows it, with `*` marking the currently-active rate:
#       Virtual-2-3 connected primary 1280x2856+0+0 ...
#          1280x2856_120.00 119.92*
# (Hardcoding 60 once paced a 119.92 Hz primary wrongly for weeks.)
_XRANDR_MODE_RE = re.compile(r"^\s+\S+\s+.*?(?P<hz>\d+\.\d+)\*")


def detect_outputs():
    try:
        out = subprocess.run(
            ["xrandr", "--query"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
    except Exception as e:
        print(f"detect_outputs: xrandr query failed ({e}), using fallback geometry", file=sys.stderr)
        return _FALLBACK_OUTPUTS

    # Every connected output with an active mode gets a backdrop window; the
    # renderer draws ONE square canvas and each window blits its own crop of
    # it, so the number, names and orientation of outputs don't matter.
    # `[display] outputs = [...]` in config.toml restricts it to named ones.
    only = set(_DISPLAY_CFG.get("outputs", []))
    found = []
    hz_by_name = {}
    pending = None
    for line in out.splitlines():
        m = _XRANDR_LINE_RE.match(line)
        if m:
            pending = m.group("name") if (not only or m.group("name") in only) else None
            if pending:
                found.append(m)
            continue
        if pending:
            mode = _XRANDR_MODE_RE.match(line)
            if mode:
                hz_by_name[pending] = float(mode.group("hz"))
                pending = None

    if not found:
        print("detect_outputs: no connected output with an active mode; using fallback geometry",
              file=sys.stderr)
        return _FALLBACK_OUTPUTS

    # Primary first (it owns vsync pacing and the mouse mapping), the rest in
    # name order so the list is stable across re-detection (sync_layout
    # matches windows by index).
    found.sort(key=lambda m: (m.group("primary") is None, m.group("name")))
    return [{
        "name": m.group("name"),
        "w": int(m.group("w")),
        "h": int(m.group("h")),
        "x": int(m.group("x")),
        "y": int(m.group("y")),
        # Falls back to 60 only if the mode line was unreadable -- a wrong
        # LOW value just costs frames, a wrong high one would spin.
        "hz": hz_by_name.get(m.group("name"), 60.0),
    } for m in found]


# Canvas = FIXED 2856x2856. Square, and deliberately NOT derived from the
# detected outputs -- see HANDOFF.md item 2 for the render/blit spec (one real
# render at this size, cheap centered-crop blit to each window, NOT three
# independent full-resolution renders).
#
# It used to be computed as max(width) x max(height) over whatever xrandr
# reported. That is why the backdrop had to be restarted by hand every time
# the Pixel switched resolution: the canvas silently resized with it, so a
# shader's framing changed underneath it for reasons that had nothing to do
# with the shader. The canvas is a deliberate choice, not a computed minimum.
#
# Square because the outputs are in mixed orientations (2856x1280 landscape,
# 1080x1920 portrait). A non-square canvas makes composition depend on WHICH
# output is cropping it; a square one crops identically either way, so
# "fills the frame" means the same thing on every screen.
# 2026-09-20: 2856 -> 2410. Cost is almost exactly linear in pixel count for
# these raymarchers (measured: 71.2% of the pixels costs 72-73% of the time),
# so this is ~29% off every shader's frame cost -- a far bigger lever than any
# shader-local optimization available. schwarz_orrery_v3_opt goes 35.36 ->
# 25.49 ms, tonnetz_pins_v2 11.23 -> 8.21 ms.
#
# The Pixel head (1280x2856) is now BIGGER than the canvas, which is the case
# crop_for_spec below exists to handle correctly: it grabs a 1080x2410 region
# -- the same aspect ratio, since 1080/2410 = 0.4481 and 1280/2856 = 0.4482 --
# and scales that up wholesale, uniformly. Since 2026-09-27 the same rule
# covers every head (canvas_per_screen: long side = full canvas), so the two
# Dell appliance heads are downscaled ~0.8x instead of taking a 1:1 crop.
CANVAS_W = 2410
CANVAS_H = 2410


def canvas_per_screen(spec):
    """Canvas pixels per screen pixel for this output -- ONE uniform factor.

    2026-09-27 (user): every output's LONG side spans the full canvas, the
    short side is a centred crop. The canvas is square because the Pixel
    rotates (portrait/landscape) and the Dells are one of each, so "long side
    = whole canvas" is the one rule that treats them all alike. Heads smaller
    than the canvas are downscaled (the blit supersamples), bigger ones
    upscaled. Before this, heads that fit took a 1:1 centre crop, so the Dells
    showed only 45% x 80% of what was rendered.
    """
    return CANVAS_W / max(spec["w"], spec["h"])


def crop_for_spec(spec):
    """UV offset+scale for blitting this output's share of the shared canvas
    (see canvas_per_screen: long side = full canvas, uniform scale, centred).
    Used by presenter setup and sync_layout, which must agree.
    """
    k = canvas_per_screen(spec)
    uscale = min(1.0, (spec["w"] * k) / CANVAS_W)
    vscale = min(1.0, (spec["h"] * k) / CANVAS_H)
    return ((1.0 - uscale) / 2.0, (1.0 - vscale) / 2.0), (uscale, vscale)

OUTPUTS = detect_outputs()

BLIT_VERT = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

BLIT_FRAG = """
#version 330
uniform sampler2D src_tex;
uniform vec2 uv_offset;
uniform vec2 uv_scale;
uniform vec2 tap;       // a quarter output pixel, in canvas uv: 4-tap supersample
in vec2 v_uv;
out vec4 fragColor;
void main() {
    // 4 taps per output pixel: heads smaller than the canvas are DOWNscaled
    // (~0.8x for the Dells), and a single bilinear tap would skip texels, so
    // fine detail (star fields) would shimmer.
    vec2 uv = uv_offset + v_uv * uv_scale;
    vec4 col = 0.25 * (texture(src_tex, uv + vec2(-tap.x, -tap.y)) + texture(src_tex, uv + vec2(tap.x, -tap.y))
                     + texture(src_tex, uv + vec2(-tap.x, tap.y)) + texture(src_tex, uv + vec2(tap.x, tap.y)));
    // canvas_tex (every shader's render target) is 8-bit with no dtype
    // override, so any smooth in-shader gradient (lighting falloff, glow,
    // fog, a tonemap curve) quantizes to 256 visible steps here, on the way
    // to the screen -- independent of any source texture's own bit depth.
    // A small hash-based dither breaks the discrete steps up into
    // imperceptible noise instead. This runs once per output window per
    // frame in the shared blit stage, so it fixes banding for every shader,
    // not just whichever one is active.
    float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
    fragColor = vec4(col.rgb + dither / 255.0, col.a);
}
"""


def make_backdrop(x11_win_id, x, y, w, h):
    """Retrofit override-redirect onto an already-created (and already
    mapped/managed) toolkit window: unmap, set the attribute, reconfigure
    geometry + lowest stacking order, remap. Override-redirect windows
    never generate a MapRequest, so the WM never gets a chance to manage
    or tile it -- this is the same underlying mechanism xwinwrap uses,
    just applied after the fact instead of at window-creation time."""
    disp = Display()
    win = disp.create_resource_object("window", x11_win_id)

    win.unmap()
    disp.sync()

    win.change_attributes(override_redirect=1)
    win.configure(x=x, y=y, width=w, height=h, stack_mode=X.Below)
    disp.sync()

    win.map()
    win.configure(stack_mode=X.Below)
    disp.sync()
    return disp


# A bare Shadertoy export (just mainImage() + helpers, no harness -- that's
# all shadertoy.com itself ever gives you, since the site's own hidden JS
# supplies the entry point invisibly) gets exactly that same treatment here:
# auto-wrapped at load time so the file on disk never needs hand-editing.
# Declaring the full standard set unconditionally is deliberate, not lazy --
# SHADER_REFERENCE.md already documents that unused uniforms are silently
# dropped by both the GLSL compiler and set_uniform()'s KeyError guard, so
# there's no cost to declaring ones a given shader doesn't reference, and it
# avoids needing to text-scan each file for which ones it actually uses.
SHADERTOY_AUTOWRAP_UNIFORMS = """
uniform float iTime;
uniform float iTimeDelta;
uniform int iFrame;
uniform vec3 iResolution;
uniform float iSampleRate;
uniform vec4 iMouse;
uniform vec4 iDate;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform vec3 iChannelResolution[4];
out vec4 fragColor;
"""


def load_program(ctx, path):
    src = open(path).read()
    if "#if defined VERTEX_SHADER" not in src:
        # No hand-written harness in this file at all -- treat it as a raw
        # Shadertoy export and synthesize the same two-half structure our
        # hand-wrapped shaders use, around it, unmodified.
        src = (
            "#if defined VERTEX_SHADER\n\n"
            "in vec2 in_pos;\n\n"
            "void main() {\n"
            "    gl_Position = vec4(in_pos, 0.0, 1.0);\n"
            "}\n\n"
            "#elif defined FRAGMENT_SHADER\n\n"
            + SHADERTOY_AUTOWRAP_UNIFORMS + "\n"
            + src + "\n\n"
            "void main() {\n"
            "    mainImage(fragColor, gl_FragCoord.xy);\n"
            "}\n\n"
            "#endif\n"
        )
    vert = "#version 330\n#define VERTEX_SHADER 1\n" + src
    frag = "#version 330\n#define FRAGMENT_SHADER 1\n" + src
    return ctx.program(vertex_shader=vert, fragment_shader=frag)


def find_channel_texture_path(channel_idx, shader_path):
    """Per-shader texture wins if present (textures/<ShaderName>/iChannelN.png),
    otherwise fall back to a shared one (textures/iChannelN.png) -- lets a
    generic noise/font texture serve many shaders, while a shader that needs
    something specific gets its own. Drop PNGs in either place, named to
    match, no code changes needed."""
    stem = os.path.splitext(os.path.basename(shader_path))[0]
    per_shader = os.path.join(TEXTURES_DIR, stem, f"iChannel{channel_idx}.png")
    if os.path.isfile(per_shader):
        return per_shader
    shared = os.path.join(TEXTURES_DIR, f"iChannel{channel_idx}.png")
    if os.path.isfile(shared):
        return shared
    return None


# path -> (mtime, GL texture, size) -- shader hot-reload used to re-decode
# every channel texture from disk on every reload, even when only the .glsl
# text changed and no texture file was touched. Harmless when textures are
# small PIL-decoded PNGs, but a large 16-bit texture (pure-Python decode,
# see load_channel_texture below) turns every hot-reload into a multi-minute
# wait regardless of what actually changed. Keyed by mtime so an edited
# texture file still gets picked up -- this is a cache, not a "load once"
# shortcut.
_texture_cache = {}


def load_channel_texture_cached(ctx, path):
    mtime = os.path.getmtime(path)
    cached = _texture_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2]
    tex, size = load_channel_texture(ctx, path)
    if cached is not None:
        cached[1].release()  # the file actually changed -- old GL texture is now orphaned, free it
    _texture_cache[path] = (mtime, tex, size)
    return tex, size


def load_channel_texture(ctx, path):
    # Real 16-bit source PNGs (e.g. a graded HDR starmap with a smooth
    # low-frequency glow region) need to stay above 8 bits all the way to
    # the GPU, or the same banding this was meant to fix just gets baked
    # back in at load time. moderngl doesn't expose a normalized-16-bit-int
    # GL format through its simple dtype shorthand, so this decodes to
    # 0..1 float and uploads as f2 (half-float, GL_RGBA16F) -- same
    # category of format the multipass buffer textures already use
    # (dtype="f4"), just half precision, and still fully compatible with
    # normal sampler2D/LINEAR filtering/mipmaps (an integer GL format
    # would not be).
    if path.lower().endswith(".png"):
        reader = png16.Reader(filename=path)
        width, height, rows, info = reader.read()
        if info.get("bitdepth") == 16:
            planes = info["planes"]
            # pypng's Reader already decodes each row into a native-endian
            # array.array('H', ...) of correct integer values -- treating
            # that as raw big-endian bytes via np.frombuffer(row, ">u2")
            # silently corrupts every sample on a little-endian machine
            # (verified: this path was never previously exercised by any
            # real 16-bit PNG in textures/, so it's a genuinely latent bug,
            # not a regression). np.array() converts the already-correct
            # values directly, independent of platform byte order.
            arr = np.vstack([np.array(row, dtype=np.uint16) for row in rows])
            arr = arr.reshape(height, width, planes).astype(np.float32) / 65535.0
            if planes == 1:
                arr = np.repeat(arr, 3, axis=-1)
                planes = 3
            if planes == 3:
                alpha = np.ones((height, width, 1), dtype=np.float32)
                arr = np.concatenate([arr, alpha], axis=-1)
            arr = arr[::-1]  # GL texture origin is bottom-left
            arr = np.ascontiguousarray(arr.astype(np.float16))
            tex = ctx.texture((width, height), 4, arr.tobytes(), dtype="f2")
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.repeat_x = True
            tex.repeat_y = True
            return tex, (width, height)

    img = Image.open(path).convert("RGBA")
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # GL texture origin is bottom-left
    tex = ctx.texture(img.size, 4, img.tobytes())
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.repeat_x = True
    tex.repeat_y = True
    return tex, img.size


def load_static_data_texture(ctx, path, filter_mode=None):
    """Loads a raw 2D .npy array directly as a single-channel GL texture --
    no PNG round-trip, no PIL, no automatic row-flip like
    load_channel_texture()'s PNG path. For data too large/precise for a
    sane PNG (a baked light-transport atlas, a float32 density field) that
    also doesn't fit the 4-iChannel convention (see
    load_extra_static_textures below) -- the row-flip these need to match
    this codebase's GL-native texture convention is instead applied once,
    at prep time, by whatever script produced the .npy (see
    storm_cell_prep.py), so this loader stays a dumb, generic pass-through.
    uint8 arrays upload as normalized u8 (0..1 in-shader); float32 arrays
    upload as f4. No mipmaps -- callers needing exact tile addressing
    (an atlas) should sample with textureLod(..., 0.0)."""
    # 2026-10-04: a static texture may be a plain image file instead of a
    # pre-baked .npy -- so manifest.json's "static_textures" can point straight
    # at a .png/.jpg and there is no separate bake step to forget. Any
    # non-.npy path goes to load_static_image_texture below, which applies the
    # same GL row-flip and honours the same "static_texture_filters" as the
    # .npy path (so an image and its baked .npy look identical).
    if not path.lower().endswith(".npy"):
        return load_static_image_texture(ctx, path, filter_mode)
    # 2026-09-21: (L, H, W, C) float32 -> a 2D texture ARRAY (sampler2DArray),
    # uploaded layer by layer from a memmap so a multi-GB bake (kerr_bake.py,
    # ~3 GB) never sits in RAM, and CACHED across hot reloads -- re-uploading
    # it on every shader edit would stall the backdrop for seconds.
    head = np.load(path, mmap_mode="r")
    if head.ndim == 4 and filter_mode == "volume":
        return load_static_volume_texture(ctx, path, head)
    if head.ndim == 4:
        return load_static_array_texture(ctx, path, head, filter_mode)
    del head
    # 2026-09-27: big 2D fields (the 16k Nebulabrot, 1.6 GB) are cached across
    # hot reloads like the array textures -- re-uploading on every shader save
    # stalls the backdrop for seconds.
    big = os.path.getsize(path) > (256 << 20)
    key = (id(ctx), path, filter_mode)
    mtime = os.path.getmtime(path)
    if big:
        hit = _ARRAY_TEX_CACHE.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        if hit is not None:
            hit[1].release()
    arr = np.load(path, mmap_mode="r") if big else np.load(path)
    # (H, W) stays single-channel as before; (H, W, C) uploads C components, so a
    # table of vectors is ONE fetch per element instead of one per component.
    if arr.ndim == 2:
        h, w = arr.shape
        comps = 1
    elif arr.ndim == 3 and arr.shape[2] in (1, 2, 3, 4):
        h, w, comps = arr.shape
    else:
        raise ValueError(f"{path}: expected (H,W) or (H,W,C<=4), got shape {arr.shape}")
    if big:
        # 2026-09-27: one-shot upload of the 1.6 GB Nebulabrot fails in radeonsi
        # ("failed to create temporary texture to hold untiled copy"), so big
        # textures are allocated empty and filled in ~128 MB row slabs from the
        # memmap (never the whole array in RAM either).
        dt = {np.uint8: "f1", np.float32: "f4", np.float16: "f2"}.get(arr.dtype.type)
        if dt is None:
            raise ValueError(f"{path}: unsupported dtype {arr.dtype}")
        tex = ctx.texture((w, h), comps, dtype=dt)
        # 2026-09-28: slabs alone weren't enough -- cold starts with nebulabrot
        # still died in amdgpu_bo_alloc (-12) on a 128 MB GTT staging buffer: the
        # writes are queued, so every slab's staging copy stays allocated until the
        # GPU gets round to it, and five 2 GB textures fill the 2 GB GTT
        # (amdgpu.gttsize=2048 is global, the dGPU gets it too). 64 MB slabs, and
        # ctx.finish() after each so its staging is freed before the next.
        rows = max(1, (64 << 20) // (w * comps * arr.itemsize))
        for y0 in range(0, h, rows):
            y1 = min(h, y0 + rows)
            tex.write(np.ascontiguousarray(arr[y0:y1]).tobytes(), viewport=(0, y0, w, y1 - y0))
            ctx.finish()
    elif arr.dtype == np.uint8:
        tex = ctx.texture((w, h), comps, arr.tobytes())
    elif arr.dtype == np.float32:
        tex = ctx.texture((w, h), comps, arr.tobytes(), dtype="f4")
    elif arr.dtype == np.float16:
        # 2026-09-25: an HDR map that must stay HDR but doesn't need f32 --
        # v8's Milky Way diffuse layer spans ~123x from galactic centre to pole,
        # so 8 bits would band it, while f16 halves the upload of f32 at
        # precision far beyond what a smooth glow needs. The (L,H,W,C) array
        # path below already accepted f16; only this 2D/3D one didn't, and it
        # rejected the whole shader rather than the one texture.
        tex = ctx.texture((w, h), comps, arr.tobytes(), dtype="f2")
    else:
        raise ValueError(f"{path}: unsupported dtype {arr.dtype}, "
                         f"expected uint8, float16 or float32")
    # Honour the manifest's "static_texture_filters" here too -- it used to be
    # read only by the array path, so a "nearest" asked for on a 2D texture was
    # silently ignored and it got LINEAR anyway.
    if filter_mode == "nearest":
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    elif filter_mode == "linear_mip":
        # 2026-09-25: the nebula raster needs mipmaps. Without them, when the
        # lensed footprint grows the sampler takes uncorrelated texels and the
        # smooth map aliases into mush -- which is exactly what killed the first
        # diffuse layer. With the chain built, the hardware picks LOD from the
        # same fwidth(uv) footprint the star PSF uses, so magnification blurs
        # (correct for a smooth source) instead of aliasing.
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR_MIPMAP_LINEAR)
    else:
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = filter_mode == "linear_wrap"
    tex.repeat_y = False
    if big:
        _ARRAY_TEX_CACHE[key] = (mtime, tex)
    return tex


_ARRAY_TEX_CACHE = {}   # (id(ctx), path) -> (mtime, texture); survives reloads


def load_static_image_texture(ctx, path, filter_mode=None):
    """A static texture given as a plain image file (.png/.jpg/...) rather than
    a .npy -- see the dispatch in load_static_data_texture. Decodes with PIL
    and applies the GL row-flip the .npy path expects at prep time, so a PNG
    and its baked .npy look identical. Cached by mtime in the same
    _ARRAY_TEX_CACHE the big .npy fields use, so a hot reload that didn't
    touch the image does not re-decode it; honours the manifest's
    "static_texture_filters" exactly like the .npy path."""
    key = (id(ctx), path, filter_mode)
    mtime = os.path.getmtime(path)
    hit = _ARRAY_TEX_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    if hit is not None:
        hit[1].release()
    img = Image.open(path)
    if "A" in img.getbands():
        arr = np.asarray(img.convert("RGBA"))
    elif img.mode in ("L", "1", "I", "F"):
        arr = np.asarray(img.convert("L"))
    else:
        arr = np.asarray(img.convert("RGB"))
    arr = np.ascontiguousarray(arr[::-1])   # GL texture origin is bottom-left
    h, w = arr.shape[:2]
    comps = 1 if arr.ndim == 2 else arr.shape[2]
    tex = ctx.texture((w, h), comps, arr.tobytes())
    if filter_mode == "nearest":
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    elif filter_mode == "linear_mip":
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR_MIPMAP_LINEAR)
    else:
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = filter_mode == "linear_wrap"
    tex.repeat_y = False
    _ARRAY_TEX_CACHE[key] = (mtime, tex)
    return tex


def load_static_array_texture(ctx, path, mm, filter_mode=None):
    # filter_mode (manifest "static_texture_filters"): "nearest" | "linear_wrap".
    # Default by dtype: f32 -> nearest (exact per-pixel bake data), f16 ->
    # linear_wrap (the disk sim field). The Kerr-Newman disk passages are f16
    # but per-pixel exact, so they set "nearest" explicitly.
    key = (id(ctx), path, filter_mode)
    mtime = os.path.getmtime(path)
    hit = _ARRAY_TEX_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    if hit is not None:
        hit[1].release()
    layers, h, w, comps = mm.shape
    if mm.dtype not in (np.float32, np.float16) or comps not in (1, 2, 3, 4):
        raise ValueError(f"{path}: expected float32/float16 (L,H,W,C<=4), got {mm.dtype} {mm.shape}")
    # float32 arrays are exact per-pixel data (the Kerr bake): NEAREST, clamped.
    # float16 arrays are fields to be sampled smoothly (the disk sim, rows =
    # log r, columns = azimuth): LINEAR, and x wraps (azimuth is periodic).
    f16 = mm.dtype == np.float16
    mode = filter_mode or ("linear_wrap" if f16 else "nearest")
    tex = ctx.texture_array((w, h, layers), comps, dtype="f2" if f16 else "f4")
    for i in range(layers):
        tex.write(np.ascontiguousarray(mm[i]).tobytes(), viewport=(0, 0, i, w, h, 1))
    lin = mode == "linear_wrap"
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR) if lin else (moderngl.NEAREST, moderngl.NEAREST)
    tex.repeat_x = lin
    tex.repeat_y = False
    _ARRAY_TEX_CACHE[key] = (mtime, tex)
    print(f"array texture {path}: {layers}x{h}x{w}x{comps} f32 "
          f"({mm.nbytes / 1e9:.2f} GB) uploaded", file=sys.stderr)
    return tex


def load_static_volume_texture(ctx, path, mm):
    # 2026-09-27: (D, H, W, C) f16/f32 -> a real 3D texture (sampler3D), for
    # manifest filter "volume": the periodic turbulence box (turb_bake_modal.py).
    # LINEAR with a mip chain (a fly-through samples far cells at a coarse
    # footprint) and REPEAT on all three axes -- the bake is periodic, so the
    # camera can fly forever without a seam. Axis order: W = s (x), H = t (y),
    # D = r (z). Cached across hot reloads like the array textures.
    key = (id(ctx), path, "volume")
    mtime = os.path.getmtime(path)
    hit = _ARRAY_TEX_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    if hit is not None:
        hit[1].release()
    d, h, w, comps = mm.shape
    if mm.dtype not in (np.float32, np.float16) or comps not in (1, 2, 3, 4):
        raise ValueError(f"{path}: expected float32/float16 (D,H,W,C<=4), got {mm.dtype} {mm.shape}")
    f16 = mm.dtype == np.float16
    tex = ctx.texture3d((w, h, d), comps, dtype="f2" if f16 else "f4")
    slab = max(1, (256 << 20) // (h * w * comps * mm.itemsize))   # ~256 MB per write
    for z0 in range(0, d, slab):
        z1 = min(d, z0 + slab)
        tex.write(np.ascontiguousarray(mm[z0:z1]).tobytes(), viewport=(0, 0, z0, w, h, z1 - z0))
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = tex.repeat_z = True
    _ARRAY_TEX_CACHE[key] = (mtime, tex)
    print(f"volume texture {path}: {w}x{h}x{d}x{comps} {mm.dtype} "
          f"({mm.nbytes / 1e9:.2f} GB) uploaded", file=sys.stderr)
    return tex


def is_cached_array_texture(tex):
    return any(t is tex for _m, t in _ARRAY_TEX_CACHE.values())


def load_fallback_texture(ctx):
    """1x1 neutral gray -- bound when no PNG is present for a channel a
    shader references, so sampling it doesn't error."""
    tex = ctx.texture((1, 1), 4, bytes([128, 128, 128, 255]))
    return tex, (1, 1)


def load_channel_textures(ctx, shader_path):
    """Returns (textures, sizes, channel0_is_live_audio) for iChannel0..3,
    released by the caller when reloading. iChannel0 defaults to the live
    Shadertoy-style audio texture (see audio_texture.py) when no PNG
    override is present -- matches what a real Shadertoy shader gets when
    its author picked "Music" as the input, and is the whole point of the
    native plugin's shader_audio_tex_shm.h export. An explicit
    textures/iChannel0.png (or per-shader override) still wins."""
    textures = []
    sizes = []
    channel0_is_live_audio = False
    for i in range(4):
        path = find_channel_texture_path(i, shader_path)
        if path is not None:
            try:
                tex, size = load_channel_texture_cached(ctx, path)
                print(f"iChannel{i}: {path}", file=sys.stderr)
            except Exception as e:
                print(f"iChannel{i}: failed to load {path}: {e}", file=sys.stderr)
                tex, size = load_fallback_texture(ctx)
        elif i == 0:
            tex, size = load_fallback_texture(ctx)  # placeholder, swapped by the caller
            channel0_is_live_audio = True
            print("iChannel0: no PNG found, defaulting to live audio texture", file=sys.stderr)
        else:
            tex, size = load_fallback_texture(ctx)
        textures.append(tex)
        sizes.append(size)
    return textures, sizes, channel0_is_live_audio


# --- Multipass shaders (Shadertoy Buffer A/B/C/D + Image) -----------------
#
# Staged by shader-add-multipass-tab / shadertoy-multipass-finalize (see
# SHADER_REFERENCE.md's "Multipass shaders" section) into a
# shaders/<name>.mp/ directory:
#   manifest.json          -- {"buffers": ["A","B",...], "wiring": {...}}
#   common.glsl             (optional, prepended to every pass's source)
#   buffer_a.glsl .. buffer_d.glsl   (whichever buffers exist)
#   image.glsl              (required, the terminal pass)
#
# wiring format: manifest["wiring"][<pass name, "A".."D" or "Image">] is a
# dict keyed by channel string "0".."3". Three shapes matter after
# normalize_mp_wiring() below:
#   {"type": "buffer", "id": "A"}  -- sample that buffer's most-recently
#                                      -available texture (see render order
#                                      note below); "self" is normalized to
#                                      this with id == the buffer's own
#                                      letter, since a self-reference is by
#                                      far the most common case (feedback
#                                      trails).
#   {"type": "audio"}              -- live Shadertoy-style audio texture,
#                                      on ANY channel including 0 -- always
#                                      explicit, never guessed.
#   absent / no entry for a channel -- PNG texture convention
#                                      (textures/<stem>/iChannelN.png or
#                                      the shared fallback) if one exists,
#                                      otherwise a neutral gray fallback.
#                                      NOT audio, on any channel, even 0 --
#                                      a bare unwired channel means
#                                      "nothing", same as single-pass
#                                      shaders' textures/ convention.
#                                      (Was channel-0-defaults-to-audio
#                                      until 2026-08-23: silently wrong on
#                                      Revision_Qualification_2018-Cupe.mp,
#                                      whose Image pass genuinely leaves
#                                      iChannel0 unassigned on the real
#                                      Shadertoy page -- confirmed live,
#                                      not guessed. Blank should never be a
#                                      synonym for "probably audio".)
#
# Render order each frame is always Common(spliced in)+A, B, C, D, then
# Image -- this is Shadertoy's own actual fixed execution order, not a
# dependency-sorted one. A pass sampling a buffer that hasn't rendered yet
# *this* frame gets that buffer's previous-frame result; one that already
# rendered this frame gets the fresh one. Implemented with one ping-pong
# texture pair per buffer: "cur" is the index holding the most recently
# *completed* frame (what anyone samples), rendering writes into the other
# index and only flips "cur" after that pass finishes -- so within a single
# frame, a not-yet-rendered buffer is still exposing last frame's texture
# to anyone sampling it, exactly matching Shadertoy's behavior.
MULTIPASS_BUFFER_LETTERS = ("A", "B", "C", "D")

# Multipass buffer passes (A/B/C/D) render at a fraction of the full canvas
# resolution and are bilinear-upsampled by the full-res Image pass. The
# Wolf-Rayet star's Buffer A holds the carbon-vein / convection detail, which
# needs full canvas resolution to stay crisp when the Image pass upscales it.
MP_BUFFER_SCALE = 1.0

# Bloom-chain buffers (B/C/D in the sonicether "Gargantua With HDR Bloom"
# layout used by kerr_newman_bh.mp: B downsamples the scene into octaves, C/D
# gaussian-blur it) render *sonicether's* output, not a scene needing its own
# detail -- and get blurred twice more before the Image pass composites them.
# Buffer B alone is ~1300 texture taps/pixel (Grab16's 16x16 oversample x5);
# at MP_BUFFER_SCALE's full canvas res that's the single most expensive thing
# in the whole multipass chain after Buffer A's own ray tracing. Kept on its
# own, much lower scale -- bilinear upsampling an already-blurry bloom halo
# is visually indistinguishable from computing it at full res. Only used for
# letters other than "A"; Buffer A always uses MP_BUFFER_SCALE.
MP_BLOOM_SCALE = 0.25

# ---------------------------------------------------------------------------
# Frame pacing. Reworked 2026-09-20 -- see scope.md.
#
# THE OLD DESIGN, and why it was wrong: there was no time-based pacing in the
# render loop at all. The frame rate was whatever three serial, vsync-BLOCKING
# `glfw.swap_buffers` calls happened to allow, with `SWAP_INTERVAL = round(60 /
# TARGET_FPS)` hardcoding 60 as every output's refresh. Two consequences, both
# measured on 2026-09-20:
#   * The primary output (`Virtual-2-3`, the Pixel stream head) actually runs
#     at 119.92 Hz, not 60, so its interval-2 swap was a 60 fps cap, not the
#     intended 30.
#   * It never reached 60 anyway, because the same loop then blocked on two
#     59.93 Hz windows at interval 2 (~33 ms each). The two appliance displays
#     were pacing the 120 Hz Pixel output down to ~30.
# Using a blocking swap as the frame limiter binds render rate to display
# refresh by construction. A video player does not do this: it decodes on its
# own clock and the display re-scans whatever is in the front buffer, so 24 fps
# content on a 120 Hz panel simply repeats each frame. That is the model here
# now -- "holding" or "repeating" a frame needs no code, it is what a display
# already does when you do not swap.
#
# THE NEW DESIGN:
#   * The loop is paced by the MONOTONIC CLOCK against a target fps, which a
#     shader may declare for itself (see read_shader_target_fps).
#   * The primary window keeps swap_interval 1 -- one 119.92 Hz vsync is 8.3 ms
#     of quantization, cheap enough to be worth staying tear-free for.
#   * The secondary windows get swap_interval 0 so they can never brake the
#     shared loop. They are also only PRESENTED at their own refresh rate; on
#     the iterations in between, their front buffer simply holds the previous
#     frame. That is the "vkms drop down to whatever they want" behaviour.
#
# IMPORTANT: there is ONE canvas, rendered ONCE per iteration, that all three
# windows blit a crop of. So there is exactly one frame rate. Per-output
# swap_interval is not a per-display frame rate -- it only controls whether
# that window's vsync is allowed to brake the single shared loop.
# All three are env-overridable purely so the pacing can be A/B'd against the
# old behaviour without editing and re-editing this file (2026-09-20 -- the
# rework measured SLOWER than the design it replaced on a GPU-bound shader,
# and attributing that inside a saturated GPU queue is not reliable, so it
# gets settled by end-to-end fps across combinations instead).
DEFAULT_TARGET_FPS = float(os.environ.get("SHADER_TARGET_FPS", "120"))

# Only the primary blocks on vsync; see above.
PRIMARY_SWAP_INTERVAL = int(os.environ.get("SHADER_PRIMARY_SWAP_INTERVAL", "1"))
# 1, NOT 0 -- measured 2026-09-20. Setting this to 0 to "stop the secondaries
# braking the loop" cost ~1.4 fps / ~3 ms per frame on a GPU-bound shader
# (20.4 vs 22.4 fps on schwarz_orrery_v3), because an interval-0 swap on vkms
# does an immediate page-flip every iteration instead of a queued one that
# overlaps with GPU work already in flight. A/B across all four combinations
# showed the primary's interval makes NO difference (2/2 and 1/2 both 22.1
# fps) and the entire cost was this one.
#
# 1 does not reintroduce the braking, because swap_interval(n) means "no
# sooner than n vblanks SINCE THE PREVIOUS SWAP", not "always wait". The
# present loop below only presents a secondary once its own refresh period has
# elapsed, so by construction at least one vblank has always passed and the
# swap returns immediately. The throttle is what prevents the gating; the
# interval is only there to keep the flip cheap.
SECONDARY_SWAP_INTERVAL = int(os.environ.get("SHADER_SECONDARY_SWAP_INTERVAL", "1"))

# iChannel0-3 are rebound to different textures every pass, every frame --
# always GL texture units 0-3 (whichever texture -- a buffer, a static PNG,
# or the live audio texture -- is wired to that channel this pass, since
# only one pass renders at a time there's no cross-pass collision reusing
# 0-3 this way). `bands` is different: a pass can reference it *alongside*
# iChannel0-3 in the same draw, so it needs its own unit well out of that
# rotation's way, bound once and left alone (unlike single-pass's bands=
# unit 0, which is safe there only because single-pass's iChannels start at
# unit 1 -- see bind_static_samplers).
MP_BANDS_UNIT = 8

# Some shaders need more simultaneous large static data textures than the
# 4-channel iChannel wiring convention has room for (storm_cell.mp: 3 baked
# light-transport atlases + a density field, on top of iChannel0 already
# carrying Buffer A's audio-grid state). manifest.json's optional
# "static_textures" field ({uniform_name: path-relative-to-repo-root})
# covers this -- loaded once via load_static_data_texture (raw .npy, not
# the PNG convention), bound at fixed units starting here, well clear of
# both the iChannel0-3 rotation and MP_BANDS_UNIT.
EXTRA_STATIC_TEXTURE_UNIT_START = 9

# Instrument-group ("stem") feed, 2026-09-20 -- see scope.md. A 120x4 float
# texture on its own sampler:
#     row 0 = drums   (percussive component)
#     row 1 = bass    (low-register harmonic component)
#     row 2 = tonal   (everything else harmonic: vocals + synths + guitar,
#                      deliberately NOT separated from each other)
#     row 3 = onset   (per-band transient strength)
# Its OWN sampler rather than extra rows on `bands` on purpose: an undeclared
# uniform is silently dropped, so a new sampler cannot affect any of the ~110
# existing shaders, whereas widening `bands` to 120x5 could disturb any shader
# that samples it with a normalized v coordinate under LINEAR filtering.
# NEAREST filtering, because rows are discrete channels -- interpolating
# between "drums" and "bass" is meaningless.
STEMS_UNIT = 7
STEM_ROWS = 4

# manifest.json's optional "compute" field (2026-10-02, scope.md "fluid ball"):
#     "compute": {"script": "sim.py"}
# A per-shader compute stage. The shader's own folder holds the logic -- a
# Python step script (setup(api) / step(api, dt)) plus its .comp kernels --
# and the renderer only provides the generic plumbing (ComputeStage below):
# compiling kernels, allocating 3D textures, binding images/samplers for a
# dispatch, broadcasting the per-frame uniforms, and handing chosen textures to
# the buffer/Image passes as ordinary samplers. Nothing shader-specific lives
# here (the v11 disk's old one-off hook was moved out to
# shaders/_attic/schwarz_orrery_v11.mp/NOTE_disk_compute_hook.md).
#
# Exposed textures sit at fixed units from here up, clear of
# EXTRA_STATIC_TEXTURE_UNIT_START's range (currently 9..17). Samplers bound
# for a single dispatch use a scratch range above that, so a dispatch never
# disturbs what the render passes see.
COMPUTE_EXPOSE_UNIT_START = 24
COMPUTE_SCRATCH_UNIT_START = 40
# How often the stage's GPU time is measured and logged (seconds).
COMPUTE_TIMING_PERIOD = 10.0

# How often the separation is recomputed, in the reader thread. The result is
# a windowed median over ~400 ms, so it moves slowly; recomputing it at the
# full 188 Hz block rate would burn CPU to change almost nothing. 60 Hz is
# already faster than any shader currently presents.
STEM_UPDATE_PERIOD = 1.0 / 60.0


_GL_CUBE = None


def load_bc6h_cubemap(ctx, json_path):
    # 2026-09-26: a pre-compressed BC6H cube map (skycube_bake.py) -- the 64k
    # NASA sky as 6 x 16384^2 faces + full mip chain, ~2.15 GB. moderngl cannot
    # upload pre-compressed blocks, so the texture is allocated through moderngl
    # (so it binds like any other) and each face/level is filled with a raw
    # glCompressedTexImage2D. Mips are baked (compressed levels cannot be
    # generated on the GPU); seamless cube filtering + 16x anisotropy are on.
    global _GL_CUBE
    import ctypes
    if _GL_CUBE is None:
        gl = ctypes.CDLL("libOpenGL.so.0")
        gl.glBindTexture.argtypes = [ctypes.c_uint, ctypes.c_uint]
        gl.glCompressedTexImage2D.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_int,
                                              ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        gl.glTexParameteri.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_int]
        gl.glEnable.argtypes = [ctypes.c_uint]
        gl.glGetError.restype = ctypes.c_uint
        _GL_CUBE = gl
    gl = _GL_CUBE
    GL_TEXTURE_CUBE_MAP, GL_CUBE_POS_X = 0x8513, 0x8515
    BC6H_UF = 0x8E8F                       # GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT
    meta = json.load(open(json_path))
    blob = np.memmap(os.path.join(os.path.dirname(json_path), "sky64k_bc6h.bin"), np.uint8, mode="r")
    n = meta["face_size"]
    tex = ctx.texture_cube((n, n), 3, None, dtype="f2", internal_format=BC6H_UF)
    gl.glGetError()
    gl.glBindTexture(GL_TEXTURE_CUBE_MAP, tex.glo)
    for e in meta["entries"]:
        chunk = np.ascontiguousarray(blob[e["offset"]:e["offset"] + e["size"]])
        gl.glCompressedTexImage2D(GL_CUBE_POS_X + e["face"], e["level"], BC6H_UF,
                                  e["w"], e["w"], 0, e["size"], chunk.ctypes.data)
    gl.glTexParameteri(GL_TEXTURE_CUBE_MAP, 0x813C, 0)                    # BASE_LEVEL
    gl.glTexParameteri(GL_TEXTURE_CUBE_MAP, 0x813D, meta["levels"] - 1)   # MAX_LEVEL
    gl.glEnable(0x884F)                                                   # CUBE_MAP_SEAMLESS
    err = gl.glGetError()
    if err:
        raise RuntimeError(f"BC6H cube upload failed: GL error 0x{err:x}")
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.anisotropy = 16.0
    total = sum(e["size"] for e in meta["entries"])
    print(f"bc6h cube {json_path}: {n}^2 x6, {meta['levels']} levels, {total / 1e9:.2f} GB uploaded",
          file=sys.stderr)
    return tex


def load_extra_static_textures(ctx, manifest):
    result = []  # (uniform_name, texture, unit)
    unit = EXTRA_STATIC_TEXTURE_UNIT_START
    filters = manifest.get("static_texture_filters", {})
    for uniform_name, rel_path in manifest.get("static_textures", {}).items():
        path = os.path.join(os.path.dirname(__file__), rel_path)
        if path.endswith(".json"):
            tex = load_bc6h_cubemap(ctx, path)
        else:
            tex = load_static_data_texture(ctx, path, filters.get(uniform_name))
        print(f"{uniform_name}: {path} (unit {unit})", file=sys.stderr)
        result.append((uniform_name, tex, unit))
        unit += 1
    return result


_COMPUTE_FORMATS = {   # GLSL image format -> (components, moderngl dtype)
    "rgba32f": (4, "f4"), "rgba16f": (4, "f2"),
    "rg32f": (2, "f4"), "rg16f": (2, "f2"),
    "r32f": (1, "f4"), "r16f": (1, "f2"),
}


class ComputeStage:
    """The generic compute stage -- see COMPUTE_EXPOSE_UNIT_START. Built by
    load_multipass() from manifest.json's "compute" field; raises on any
    failure (missing script, kernel compile error), same contract as the
    rest of load_multipass(), so a broken edit keeps the last good shader.

    The script gets this object as `api`:
        api.kernel(file, variant=None, defines=None) -> compiled kernel. The
            source is prefixed with "#version 460", "#define K_<variant>" and
            compute_common.glsl (if the folder has one), so one .comp file can
            hold several kernels behind #ifdef K_<name>.
        api.texture3d((x, y, z), fmt) -> 3D texture, fmt in _COMPUTE_FORMATS,
            linear filtering, clamp-to-edge.
        api.texture2d((x, y), fmt, data=None, nearest=False, wrap=False) -> 2D
            texture (a lookup map, or a 2D sim field; wrap=True for periodic).
        api.run(kernel, groups, images={binding: tex}, samplers={name: tex},
                uniforms={name: value}) -> one dispatch + memory barrier.
        api.set(kernel, {name: value}) -> set uniforms without dispatching.
        api.expose(uniform, tex) -> the buffer/Image passes see `tex` under
            `uniform` from now on (call again whenever a ping-pong swaps).
        api.render_uniform(name, value) -> set a uniform on the render passes.
        api.buffer(letter) -> that buffer pass's latest texture (last frame's).
        api.state -> dict the script keeps its own data in.
    Every kernel also gets the same per-frame uniforms as the render passes
    (iTime, iTimeDelta, iFrame, u_bass, ...) and the `bands`/`stems` samplers.
    """

    def __init__(self, ctx, mp_dir, spec):
        self.ctx = ctx
        self.dir = mp_dir
        self.state = {}
        self._kernels = []
        self._textures = []
        self._exposed = {}          # uniform -> [unit, texture]
        self._render_progs = []
        self._mp = None
        self._timing_last = 0.0
        self.gpu_ms = None
        common_path = os.path.join(mp_dir, "compute_common.glsl")
        self._common = open(common_path).read() if os.path.isfile(common_path) else ""
        script_path = os.path.join(mp_dir, spec["script"])
        mod_spec = importlib.util.spec_from_file_location(
            f"compute_{os.path.basename(mp_dir).replace('.', '_')}", script_path)
        self._module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(self._module)
        self._module.setup(self)
        nbytes = sum(math.prod(t.size) * t.components * (2 if t.dtype == "f2" else 4)
                     for t in self._textures)
        print(f"compute stage: {script_path} ({len(self._kernels)} kernels, "
              f"{len(self._textures)} textures, {nbytes / 2**20:.0f} MiB)", file=sys.stderr)

    # ---- script API ----
    def kernel(self, file, variant=None, defines=None):
        head = "#version 460\n"
        if variant:
            head += f"#define K_{variant} 1\n"
        for k, v in (defines or {}).items():
            head += f"#define {k} {v}\n"
        src = head + self._common + "\n" + open(os.path.join(self.dir, file)).read()
        prog = self.ctx.compute_shader(src)
        for name, unit in (("bands", MP_BANDS_UNIT), ("stems", STEMS_UNIT)):
            try:
                prog[name].value = unit
            except KeyError:
                pass
        self._kernels.append(prog)
        return prog

    def texture3d(self, size, fmt):
        comps, dtype = _COMPUTE_FORMATS[fmt]
        t = self.ctx.texture3d(tuple(size), comps, dtype=dtype)
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = t.repeat_z = False
        self._textures.append(t)
        return t

    def texture2d(self, size, fmt, data=None, nearest=False, wrap=False):
        comps, dtype = _COMPUTE_FORMATS[fmt]
        t = self.ctx.texture(tuple(size), comps, data=data, dtype=dtype)
        f = moderngl.NEAREST if nearest else moderngl.LINEAR
        t.filter = (f, f)
        t.repeat_x = t.repeat_y = wrap
        self._textures.append(t)
        return t

    def run(self, kernel, groups, images=None, samplers=None, uniforms=None):
        for binding, tex in (images or {}).items():
            tex.bind_to_image(binding, read=True, write=True)
        for i, (name, tex) in enumerate((samplers or {}).items()):
            tex.use(location=COMPUTE_SCRATCH_UNIT_START + i)
            try:
                kernel[name].value = COMPUTE_SCRATCH_UNIT_START + i
            except KeyError:   # declared but unused -> compiled out
                pass
        self.set(kernel, uniforms or {})
        kernel.run(*groups)
        self.ctx.memory_barrier()

    def set(self, kernel, uniforms):
        for name, val in uniforms.items():
            try:
                kernel[name].value = val
            except KeyError:
                pass

    def expose(self, uniform, tex):
        if uniform not in self._exposed:
            unit = COMPUTE_EXPOSE_UNIT_START + len(self._exposed)
            self._exposed[uniform] = [unit, tex]
            for p in self._render_progs:
                try:
                    p[uniform].value = unit
                except KeyError:
                    pass
        self._exposed[uniform][1] = tex
        tex.use(location=self._exposed[uniform][0])

    def render_uniform(self, name, value):
        """Set a uniform on the buffer/Image passes (e.g. a blend weight the sim decides)."""
        for p in self._render_progs:
            try:
                p[name].value = value
            except KeyError:
                pass

    def buffer(self, letter):
        b = self._mp["buffers"][letter]
        return b["tex"][b["cur"]]

    # ---- renderer side ----
    def attach(self, mp, render_progs):
        self._mp = mp
        self._render_progs = render_progs
        for uniform, (unit, tex) in self._exposed.items():
            tex.use(location=unit)
            for p in render_progs:
                try:
                    p[uniform].value = unit
                except KeyError:
                    pass

    def programs(self):
        return self._kernels

    def step(self, dt, now):
        timed = now - self._timing_last > COMPUTE_TIMING_PERIOD
        if timed:
            self._timing_last = now
            q = self.ctx.query(time=True)
            with q:
                self._module.step(self, dt)
            self.gpu_ms = q.elapsed / 1e6
            print(f"compute stage: {self.gpu_ms:.2f} ms GPU time", file=sys.stderr)
        else:
            self._module.step(self, dt)
        # the dispatches rebound units; restore what the render passes read
        for unit, tex in self._exposed.values():
            tex.use(location=unit)

    def release(self):
        for k in self._kernels:
            k.release()
        for t in self._textures:
            t.release()


def load_compute_stage(ctx, mp_dir, manifest):
    spec = manifest.get("compute")
    if not spec:
        return None
    if "script" not in spec:
        # the pre-2026-10-02 disk-only format (v11): no longer supported here
        print(f"compute: {mp_dir} uses the old disk-only \"compute\" format, ignored "
              f"(see shaders/_attic/schwarz_orrery_v11.mp/NOTE_disk_compute_hook.md)", file=sys.stderr)
        return None
    return ComputeStage(ctx, mp_dir, spec)


def mp_newest_mtime(mp_dir):
    # Hot-reload trigger for a multipass directory: newest mtime across
    # every file that actually affects rendering (manifest + common + all
    # pass files), mirroring the single-.glsl-file mtime check.
    newest = 0.0
    for entry in os.scandir(mp_dir):
        if entry.is_file():
            newest = max(newest, entry.stat().st_mtime)
    return newest


def normalize_mp_wiring(manifest):
    wiring = manifest.get("wiring", {})
    for pass_name, channels in wiring.items():
        for ch, spec in channels.items():
            if spec.get("type") == "self":
                spec["type"] = "buffer"
                spec["id"] = pass_name
    return wiring


def compile_mp_pass(ctx, common_src, code_src):
    # Same auto-wrap harness as load_program()'s Shadertoy path (a bare
    # mainImage() + helpers, no hand-written harness) -- multipass passes
    # are staged as raw pasted tab code, never hand-wrapped.
    src = (
        "#if defined VERTEX_SHADER\n\n"
        "in vec2 in_pos;\n\n"
        "void main() {\n"
        "    gl_Position = vec4(in_pos, 0.0, 1.0);\n"
        "}\n\n"
        "#elif defined FRAGMENT_SHADER\n\n"
        + SHADERTOY_AUTOWRAP_UNIFORMS + "\n"
        + common_src + "\n"
        + code_src + "\n\n"
        "void main() {\n"
        "    mainImage(fragColor, gl_FragCoord.xy);\n"
        "}\n\n"
        "#endif\n"
    )
    vert = "#version 330\n#define VERTEX_SHADER 1\n" + src
    frag = "#version 330\n#define FRAGMENT_SHADER 1\n" + src
    prog = ctx.program(vertex_shader=vert, fragment_shader=frag)
    try:
        prog["bands"].value = MP_BANDS_UNIT
    except KeyError:
        pass
    try:
        prog["stems"].value = STEMS_UNIT
    except KeyError:
        pass
    return prog


# A shader may declare its own target frame rate, because the right rate is a
# property of what the shader costs and how it moves, not a global constant:
#   * multipass: {"fps": 60} in manifest.json
#   * single-pass: a `// fps: 60` line anywhere in the first 40 lines
# Omitted means DEFAULT_TARGET_FPS. A heavy shader does not need to declare
# anything -- it self-limits by being slow (the orrery at ~33 ms/frame cannot
# exceed ~30 however it is paced). Declaring a LOW value is the useful case:
# it stops a cheap shader from rendering flat-out and burning the dGPU for
# frames nothing benefits from.
_FPS_DIRECTIVE_RE = re.compile(r"^\s*//\s*fps\s*:\s*(?P<fps>\d+(?:\.\d+)?)\s*$")


def read_shader_target_fps(path):
    try:
        if is_multipass_path(path):
            with open(os.path.join(path, "manifest.json")) as f:
                value = json.load(f).get("fps")
            return float(value) if value else DEFAULT_TARGET_FPS
        with open(path) as f:
            for _ in range(40):
                line = f.readline()
                if not line:
                    break
                m = _FPS_DIRECTIVE_RE.match(line)
                if m:
                    return float(m.group("fps"))
    except Exception as e:
        print(f"read_shader_target_fps({path}): {e}; using default", file=sys.stderr)
    return DEFAULT_TARGET_FPS


def load_multipass(ctx, vbo, mp_dir):
    """Raises on any failure (missing manifest, bad JSON, compile error) --
    caller (try_reload) keeps the previous good state on failure, same
    contract as load_program()."""
    with open(os.path.join(mp_dir, "manifest.json")) as f:
        manifest = json.load(f)
    wiring = normalize_mp_wiring(manifest)

    common_path = os.path.join(mp_dir, "common.glsl")
    common_src = open(common_path).read() if os.path.isfile(common_path) else ""

    stem = os.path.basename(mp_dir)
    if stem.endswith(".mp"):
        stem = stem[: -len(".mp")]

    buf_w = max(1, int(round(CANVAS_W * MP_BUFFER_SCALE)))
    buf_h = max(1, int(round(CANVAS_H * MP_BUFFER_SCALE)))
    bloom_w = max(1, int(round(CANVAS_W * MP_BLOOM_SCALE)))
    bloom_h = max(1, int(round(CANVAS_H * MP_BLOOM_SCALE)))

    buffers = {}
    for letter in MULTIPASS_BUFFER_LETTERS:
        fpath = os.path.join(mp_dir, f"buffer_{letter.lower()}.glsl")
        if not os.path.isfile(fpath):
            continue
        prog = compile_mp_pass(ctx, common_src, open(fpath).read())
        vao = ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])
        w, h = (buf_w, buf_h) if letter == "A" else (bloom_w, bloom_h)
        tex_pair = []
        fbo_pair = []
        for _ in range(2):
            t = ctx.texture((w, h), 4, dtype="f4")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex_pair.append(t)
            fbo_pair.append(ctx.framebuffer(color_attachments=[t]))
        buffers[letter] = {
            "prog": prog, "vao": vao,
            "tex": tex_pair, "fbo": fbo_pair, "cur": 0,
            "res": (w, h),
            "wiring": wiring.get(letter, {}),
        }

    image_path = os.path.join(mp_dir, "image.glsl")
    image_prog = compile_mp_pass(ctx, common_src, open(image_path).read())
    image_vao = ctx.vertex_array(image_prog, [(vbo, "2f", "in_pos")])

    fallback_tex, _ = load_fallback_texture(ctx)

    # Static (non-buffer) channel textures -- resolved once at load time,
    # same PNG convention as single-pass (textures/<stem>/iChannelN.png or
    # shared textures/iChannelN.png), keyed by (pass_name, channel) since
    # different passes can each have their own override.
    static_textures = []  # released alongside everything else on reload
    # {(pass_name, ch), ...} using the live audio texture -- ONLY populated
    # by an explicit {"type": "audio"} wiring entry. No implicit
    # channel-0-defaults-to-audio behavior (removed 2026-08-23: a real bug
    # on Revision_Qualification_2018-Cupe.mp, where Image's genuinely-
    # unwired iChannel0 silently got audio instead of the correct "nothing"
    # -- confirmed against the real Shadertoy page. Blank now always means
    # blank/gray, matching single-pass's own textures/ fallback behavior;
    # audio must be asked for explicitly on every channel, including 0).
    live_audio_channels = set()

    def resolve_static(pass_name, ch):
        # Every pass in a multipass shader shares the one <stem> texture
        # set (textures/<stem>/iChannelN.png or the shared fallback) --
        # per-pass texture overrides aren't a thing single-pass textures/
        # supports either, so this matches existing convention.
        path = find_channel_texture_path(int(ch), os.path.join(TEXTURES_DIR, stem))
        if path is not None:
            tex, _size = load_channel_texture(ctx, path)
            static_textures.append(tex)
            return tex
        return fallback_tex

    all_passes = [(letter, buffers[letter]["wiring"]) for letter in MULTIPASS_BUFFER_LETTERS if letter in buffers]
    all_passes.append(("Image", wiring.get("Image", {})))
    for pass_name, channels in all_passes:
        for ch in ("0", "1", "2", "3"):
            spec = channels.get(ch)
            if spec is not None and spec.get("type") == "buffer":
                continue
            if spec is not None and spec.get("type") == "audio":
                live_audio_channels.add((pass_name, ch))
                channels[ch] = {"type": "tex", "_tex": fallback_tex}
                continue
            channels[ch] = {"type": "tex", "_tex": resolve_static(pass_name, ch)}

    extra_static = load_extra_static_textures(ctx, manifest)
    extra_progs = [b["prog"] for b in buffers.values()] + [image_prog]
    for uniform_name, tex, unit in extra_static:
        tex.use(location=unit)
        for prog in extra_progs:
            try:
                prog[uniform_name].value = unit
            except KeyError:
                pass

    compute = load_compute_stage(ctx, mp_dir, manifest)

    mp = {
        "dir": mp_dir, "manifest": manifest,
        "buffers": buffers,
        "image_prog": image_prog, "image_vao": image_vao,
        "image_wiring": wiring.get("Image", {}),
        "fallback_tex": fallback_tex,
        "static_textures": static_textures,
        "extra_static_textures": extra_static,
        "live_audio_channels": live_audio_channels,
        "compute": compute,
    }
    if compute is not None:
        compute.attach(mp, extra_progs)
    return mp


def release_multipass(mp):
    for b in mp["buffers"].values():
        b["prog"].release()
        b["vao"].release()
        for t in b["tex"]:
            t.release()
        for f in b["fbo"]:
            f.release()
    mp["image_prog"].release()
    mp["image_vao"].release()
    mp["fallback_tex"].release()
    for t in mp["static_textures"]:
        t.release()
    for _name, tex, _unit in mp["extra_static_textures"]:
        if not is_cached_array_texture(tex):   # cached arrays outlive reloads
            tex.release()
    if mp.get("compute") is not None:
        mp["compute"].release()


def bind_mp_wiring(prog, channels, mp, audio_channel_tex, pass_name):
    for ch in ("0", "1", "2", "3"):
        spec = channels.get(ch)
        unit = int(ch)
        if spec is not None and spec.get("type") == "buffer":
            buf = mp["buffers"].get(spec["id"])
            tex = buf["tex"][buf["cur"]] if buf is not None else mp["fallback_tex"]
        elif (pass_name, ch) in mp["live_audio_channels"]:
            tex = audio_channel_tex
        elif spec is not None:
            tex = spec.get("_tex", mp["fallback_tex"])
        else:
            tex = mp["fallback_tex"]
        tex.use(location=unit)
        try:
            prog[f"iChannel{ch}"].value = unit
        except KeyError:
            pass


def render_multipass(ctx, mp, canvas_fbo, audio_channel_tex):
    # Per-frame uniform values (u_time, u_bass, iFrame, ...) are pushed onto
    # every active program -- including every one of these -- by the
    # caller's set_uniform() broadcast *before* this runs, same mechanism
    # as the single-pass path; only wiring + the actual draw happens here.
    for letter in MULTIPASS_BUFFER_LETTERS:
        b = mp["buffers"].get(letter)
        if b is None:
            continue
        bind_mp_wiring(b["prog"], b["wiring"], mp, audio_channel_tex, letter)
        # Buffer passes render at reduced resolution, so iResolution (which
        # their code divides fragCoord by) must match the actual render target
        # size, not the full canvas. The Image pass keeps the full-canvas
        # value set by the per-frame uniform broadcast.
        buf_w, buf_h = b["res"]
        try:
            b["prog"]["iResolution"].value = (float(buf_w), float(buf_h), 1.0)
        except KeyError:
            pass
        write_idx = 1 - b["cur"]
        b["fbo"][write_idx].use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        b["vao"].render(moderngl.TRIANGLE_STRIP)
        b["cur"] = write_idx

    bind_mp_wiring(mp["image_prog"], mp["image_wiring"], mp, audio_channel_tex, "Image")
    canvas_fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 1.0)
    mp["image_vao"].render(moderngl.TRIANGLE_STRIP)


def create_output_window(spec, share=None):
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DECORATED, False)
    glfw.window_hint(glfw.FLOATING, True)

    window = glfw.create_window(spec["w"], spec["h"], f"shader-backdrop-{spec['name']}", None, share)
    if not window:
        raise SystemExit(f"glfw.create_window() failed for {spec['name']}")

    glfw.set_window_pos(window, spec["x"], spec["y"])

    x11_win_id = glfw.get_x11_window(window)
    xdisp = make_backdrop(x11_win_id, spec["x"], spec["y"], spec["w"], spec["h"])
    print(f"backdrop[{spec['name']}]: override-redirect set on X11 window 0x{x11_win_id:x}, "
          f"pos=({spec['x']},{spec['y']}) size={spec['w']}x{spec['h']}", file=sys.stderr)
    return window, xdisp


def main():
    if not glfw.init():
        raise SystemExit("glfw.init() failed")

    # Primary window/context does the real render (the expensive part).
    # The other two windows share its GL object namespace (glfw's `share=`)
    # so the offscreen canvas texture can be sampled from all three without
    # re-running the shader per window -- see HANDOFF.md item 2.
    primary_window, primary_xdisp = create_output_window(OUTPUTS[0])
    glfw.make_context_current(primary_window)
    # Primary is the only window allowed to block on vsync -- see the pacing
    # block near DEFAULT_TARGET_FPS.
    glfw.swap_interval(PRIMARY_SWAP_INTERVAL)
    print(f"pacing: primary '{OUTPUTS[0]['name']}' at {OUTPUTS[0].get('hz', 60.0):.2f} Hz, "
          f"swap_interval={PRIMARY_SWAP_INTERVAL}", file=sys.stderr)

    ctx = moderngl.create_context()
    renderer_str = ctx.info["GL_RENDERER"]
    print(f"GL_RENDERER: {renderer_str}", file=sys.stderr)
    if "9060" not in renderer_str and "gfx1200" not in renderer_str.lower():
        print(f"WARNING: expected the dGPU (RX 9060 XT / gfx1200), got: {renderer_str}", file=sys.stderr)

    quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
    vbo = ctx.buffer(quad.tobytes())

    def do_load(path):
        """Load whichever mode `path` resolves to -- a single-pass .glsl
        file (prog/vao set, mp None) or a multipass .mp directory (mp set,
        prog/vao None). Raises on failure; caller decides whether that's
        fatal (startup) or ignorable (try_reload keeps the previous good
        state, same contract load_program() always had)."""
        if is_multipass_path(path):
            return None, None, load_multipass(ctx, vbo, path), [], [], False
        new_prog = load_program(ctx, path)
        new_vao = ctx.vertex_array(new_prog, [(vbo, "2f", "in_pos")])
        new_tex, new_sizes, new_live_audio = load_channel_textures(ctx, path)
        return new_prog, new_vao, None, new_tex, new_sizes, new_live_audio

    prog, vao, mp, channel_textures, channel_sizes, channel0_is_live_audio = do_load(SHADER_PATH)

    def active_programs():
        # Every compiled program that needs this frame's uniform values --
        # just the one program in single-pass mode, every buffer pass plus
        # Image in multipass mode (see render_multipass()).
        if mp is not None:
            progs = [b["prog"] for b in mp["buffers"].values()] + [mp["image_prog"]]
            if mp.get("compute") is not None:
                progs += mp["compute"].programs()
            return progs
        return [prog]

    def set_uniform(name, value):
        # Shaders won't always reference every meta feature -- GLSL drops
        # unused uniforms, so a shader-dependent KeyError here is expected,
        # not an error.
        for p in active_programs():
            try:
                p[name].value = value
            except KeyError:
                pass

    bands_tex = ctx.texture((N_BANDS, 1), 1, dtype="f4")
    bands_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    # Bound to both units -- single-pass programs reference unit 0
    # (bind_static_samplers below), multipass programs reference
    # MP_BANDS_UNIT (set at compile time in compile_mp_pass) -- see that
    # constant's comment for why multipass can't reuse unit 0 the same way.
    bands_tex.use(location=0)
    bands_tex.use(location=MP_BANDS_UNIT)

    # One unit for both single-pass and multipass -- unlike `bands`, which
    # needs two because single-pass puts it on unit 0 where multipass's
    # iChannel rotation lives. Unit 7 is clear of the iChannel rotation (0-4),
    # MP_BANDS_UNIT (8) and the extra-static range (9+).
    stems_tex = ctx.texture((N_BANDS, STEM_ROWS), 1, dtype="f4")
    stems_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    stems_tex.write(np.zeros((STEM_ROWS, N_BANDS), dtype="f4").tobytes())
    stems_tex.use(location=STEMS_UNIT)

    # Live Shadertoy-style audio texture (real iChannel0 convention) --
    # (512,2), row0=freq (v~0.25), row1=wave (v~0.75), rewritten every
    # frame from the native EasyEffects plugin's shm export. Persistent,
    # not tied to any particular shader/reload cycle.
    audio_channel_tex = ctx.texture((AUDIO_TEX_WIDTH, 2), 1, dtype="f4")
    audio_channel_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    try:
        audio_reader = AudioTexReader()
    except FileNotFoundError:
        audio_reader = None
        print("audio_texture: /dev/shm/shader_audio_tex not found -- "
              "iChannel0 will fall back to gray/PNG until the native plugin writes it",
              file=sys.stderr)

    def bind_channel_textures(textures, is_live_audio):
        if is_live_audio:
            audio_channel_tex.use(location=1)
        else:
            textures[0].use(location=1)
        for i in range(1, 4):
            textures[i].use(location=i + 1)

    def bind_static_samplers(p):
        # bands=unit0, iChannel0..3=units1..4 -- set on every (re)compiled
        # program since sampler bindings aren't preserved across a new
        # ctx.program(). Shaders won't all reference every channel; KeyError
        # here is expected, not an error (see set_uniform below). Single-pass
        # only -- multipass programs get their sampler bindings from
        # compile_mp_pass (bands) and bind_mp_wiring (iChannel0-3, every
        # frame, since those rotate per-pass).
        try:
            p["bands"].value = 0
        except KeyError:
            pass
        try:
            p["stems"].value = STEMS_UNIT
        except KeyError:
            pass
        for i in range(4):
            try:
                p[f"iChannel{i}"].value = i + 1
            except KeyError:
                pass

    def apply_single_pass_bindings():
        if prog is None:
            return
        bind_channel_textures(channel_textures, channel0_is_live_audio)
        bind_static_samplers(prog)

    apply_single_pass_bindings()

    def current_mtime():
        return mp_newest_mtime(SHADER_PATH) if is_multipass_path(SHADER_PATH) else os.path.getmtime(SHADER_PATH)

    # Mutable holder rather than a nonlocal: try_reload already carries a long
    # nonlocal list, and the present loop only ever reads this.
    pacing = {"target_fps": read_shader_target_fps(SHADER_PATH)}
    print(f"pacing: target {pacing['target_fps']:.0f} fps for {os.path.basename(SHADER_PATH)}",
          file=sys.stderr)

    shader_mtime = current_mtime()
    try:
        pointer_mtime = os.path.getmtime(ACTIVE_SHADER_FILE)
    except FileNotFoundError:
        pointer_mtime = 0.0

    def try_reload():
        # Hot-reload: edit the shader live (a .glsl file's text, or any file
        # inside a .mp directory) while this keeps running, ShaderEditor-on-
        # the-phone style -- the editable surface is the shader text itself,
        # not a params/preset UI. A syntax error mid-edit must not kill the
        # backdrop -- keep the last good program and report the error, same
        # as ShaderEditor does.
        global SHADER_PATH
        nonlocal prog, vao, mp, shader_mtime, pointer_mtime, channel_textures, channel_sizes, channel0_is_live_audio

        # Which file/directory is "active" can itself change (picked in the
        # EasyEffects QML page, or by shadertoy-multipass-finalize) -- check
        # that first so a switch is picked up even if the newly-selected
        # target's own mtime happens to be older than what we last saw for
        # the previous one.
        try:
            new_pointer_mtime = os.path.getmtime(ACTIVE_SHADER_FILE)
        except FileNotFoundError:
            new_pointer_mtime = 0.0
        if new_pointer_mtime != pointer_mtime:
            pointer_mtime = new_pointer_mtime
            new_path = resolve_active_shader_path()
            if new_path != SHADER_PATH:
                SHADER_PATH = new_path
                shader_mtime = -1.0  # force the reload below regardless of the new target's mtime

        mtime = current_mtime()
        if mtime == shader_mtime:
            return
        shader_mtime = mtime
        # Re-read every reload: the declaration lives in the shader/manifest,
        # so editing it must take effect the same way editing the GLSL does.
        new_fps = read_shader_target_fps(SHADER_PATH)
        if new_fps != pacing["target_fps"]:
            print(f"pacing: target {pacing['target_fps']:.0f} -> {new_fps:.0f} fps",
                  file=sys.stderr)
            pacing["target_fps"] = new_fps
        try:
            new_prog, new_vao, new_mp, new_tex, new_sizes, new_live_audio = do_load(SHADER_PATH)
        except Exception as e:
            print(f"shader reload failed, keeping previous program:\n{e}", file=sys.stderr)
            return

        # Release whatever the previous mode was using -- a reload can
        # switch modes entirely (single-pass <-> multipass), not just swap
        # programs within the same mode.
        if mp is not None:
            release_multipass(mp)
        if vao is not None:
            vao.release()
        if prog is not None:
            prog.release()
        # Only release textures that actually got replaced -- one still
        # present in new_tex (by identity) came from _texture_cache and is
        # still in use, releasing it here would destroy a live GL object the
        # new state still needs.
        for tex in channel_textures:
            if tex not in new_tex:
                tex.release()

        prog, vao, mp = new_prog, new_vao, new_mp
        channel_textures, channel_sizes, channel0_is_live_audio = new_tex, new_sizes, new_live_audio
        apply_single_pass_bindings()

        mode = "multipass" if mp is not None else "single-pass"
        print(f"shader reloaded: {SHADER_PATH} ({mode})", file=sys.stderr)

    # Offscreen canvas the real shader renders into, once per frame, at the
    # full envelope size (widest x tallest output). Lives under the primary
    # context; sampled from the other two windows' contexts below via their
    # shared GL namespace.
    # f4 (float32), matching the multipass buffer textures below -- every
    # shader's smooth math (lighting falloff, glow, tonemap curves) needs to
    # survive at full precision until the ONE place it actually gets
    # quantized to 8-bit: the blit stage's dither, further down. An 8-bit
    # canvas_tex bakes banding in at this first write, before the blit ever
    # runs -- dithering there can't undo damage already done to the stored
    # values.
    canvas_tex = ctx.texture((CANVAS_W, CANVAS_H), 4, dtype="f4")
    canvas_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    canvas_fbo = ctx.framebuffer(color_attachments=[canvas_tex])

    # Video capture reads back from this same canvas -- the only place the full
    # square exists. Idle (and free) until capture_state.txt says otherwise.
    # Optional module: without capture.py there is simply no recording.
    try:
        import capture
    except ImportError:
        canvas_capture = None
    else:
        canvas_capture = capture.CanvasCapture(
            ctx, canvas_fbo, CANVAS_W, CANVAS_H, CAPTURE_FILE
        )

    # Each output window gets its own moderngl Context (moderngl wraps
    # "whatever GL context is current" -- one Python Context object per
    # window keeps that unambiguous) and its own tiny crop-blit program.
    # VAOs/framebuffers aren't shareable across GL contexts even with
    # `share=` (only buffers/textures/programs are), so each gets its own
    # trivial quad -- cheap, this isn't the expensive part.
    presenters = []
    for spec in OUTPUTS:
        if spec is OUTPUTS[0]:
            window, xdisp, win_ctx = primary_window, primary_xdisp, ctx
        else:
            window, xdisp = create_output_window(spec, share=primary_window)
            glfw.make_context_current(window)
            # Non-blocking: a secondary must never gate the shared loop.
            glfw.swap_interval(SECONDARY_SWAP_INTERVAL)
            win_ctx = moderngl.create_context()

        blit_prog = win_ctx.program(vertex_shader=BLIT_VERT, fragment_shader=BLIT_FRAG)
        blit_vbo = win_ctx.buffer(quad.tobytes())
        blit_vao = win_ctx.vertex_array(blit_prog, [(blit_vbo, "2f", "in_pos")])

        (u0, v0), (uscale, vscale) = crop_for_spec(spec)

        presenters.append({
            "window": window, "ctx": win_ctx, "prog": blit_prog, "vao": blit_vao,
            "uv_offset": (u0, v0), "uv_scale": (uscale, vscale),
            "spec": spec,
            # Last time this window was actually blitted+swapped. Secondaries
            # are skipped on iterations that would exceed their own refresh;
            # their front buffer just holds the previous frame.
            "last_present": 0.0,
        })

    glfw.make_context_current(primary_window)

    reader = None
    for _ in range(50):
        try:
            reader = BandsReader()
            break
        except FileNotFoundError:
            time.sleep(0.1)
    if reader is None:
        raise SystemExit("shared memory /dev/shm/shader_bands not found -- "
                          "is the \"Shader Bands\" EasyEffects effect enabled?")

    # 2026-09-20: this is now filled by a READER THREAD at the audio block
    # rate (~188 Hz), not once per rendered frame. Previously the loop read
    # the shm exactly once per frame, so at 30 fps roughly 5 of every 6
    # published blocks were discarded and no consumer could see transient
    # detail finer than a render frame. Decoupling also means the analysis
    # rate no longer changes when the shader's frame rate does.
    #
    # Sized in ROWS, so it must cover the audio delay (e.g. ~1.1s = ~207 rows) plus
    # half the stem window, with headroom for a larger recalibrated delay.
    # 2400 rows is ~12.8s at 188 Hz.
    band_history = deque(maxlen=2400)

    # Published by the reader thread, read by the render loop. A plain tuple
    # rebound as a whole: rebinding a name is atomic under the GIL, so the
    # loop can never observe a half-updated set of values, and no lock is
    # needed on a path that runs every frame.
    #   (delayed_frame, stems_array(4,120) float32 or None, scalars dict)
    audio_state = [(None, None, {})]
    stem_splitter = stem_split.StemSplitter(N_BANDS)
    # Tempo/phase tracking (2026-09-21, scope.md). Lives here rather than in a
    # shader because a tempo estimator needs HISTORY, and a shader has nowhere
    # to keep it -- see beat_track.py's header. Publishes u_tbeat /
    # u_beat_phase / u_beat_count / u_beat_conf / u_beat_bpm through the same
    # scalar dict the stem levels already use, so any shader can read them and
    # undeclared ones are dropped as usual.
    beat_tracker = beat_track.BeatTracker(N_BANDS)
    # Per-song beat cache (2026-09-21, scope.md). Recognises the track over
    # MPRIS (YouTube Music in the browser, VLC), recalls a stored beat grid so
    # a known song is locked from bar one, and re-analyses what it heard
    # NON-causally at song end to learn it for next time -- no audio file
    # involved. Always on; SHADER_BEAT_CACHE=0 disables it and leaves pure
    # live tracking. Any failure inside it degrades to live tracking rather
    # than affecting the backdrop.
    beat_cache_on = os.environ.get("SHADER_BEAT_CACHE", "1") != "0"
    beat_sess = beat_session.BeatSession(
        beat_tracker, delay_fn=read_audio_delay_seconds, enabled=beat_cache_on)

    def audio_reader_thread():
        last_ts = None
        last_split = 0.0
        rate_hz = 188.0          # refined from real timestamps below
        win = stem_split.window_frames(rate_hz)
        while True:
            try:
                frame = reader.read()
                if frame is not None and frame["timestamp"] != last_ts:
                    last_ts = frame["timestamp"]
                    band_history.append(frame)

                now = time.monotonic()
                # Recompute at a fixed cadence rather than per published block:
                # the separation is a windowed median, so re-running it 188x a
                # second would spend real CPU to move a slow-moving result by
                # almost nothing.
                if now - last_split >= STEM_UPDATE_PERIOD:
                    last_split = now
                    # list() on a deque is atomic under the GIL, so this cannot
                    # tear against our own appends above -- and it must be a
                    # snapshot, because iterating a deque that is being
                    # appended to raises "deque mutated during iteration".
                    snap = list(band_history)
                    if len(snap) >= 8:
                        span = snap[-1]["timestamp"] - snap[0]["timestamp"]
                        if span > 0.5:
                            rate_hz = (len(snap) - 1) / span
                            win = stem_split.window_frames(rate_hz)
                        idx = pick_delayed_index(
                            snap, time.monotonic() - read_audio_delay_seconds())
                        if idx is not None:
                            half = win // 2
                            lo = max(0, idx - half)
                            hi = min(len(snap), idx + half + 1)
                            rows = np.array([r["bands"] for r in snap[lo:hi]],
                                            dtype=np.float64)
                            np.nan_to_num(rows, copy=False, nan=0.0,
                                          posinf=0.0, neginf=0.0)
                            out = stem_splitter.split(rows, idx - lo, rate_hz)
                            tex = np.empty((STEM_ROWS, N_BANDS), dtype="f4")
                            tex[0] = out["drums"]
                            tex[1] = out["bass"]
                            tex[2] = out["tonal"]
                            tex[3] = out["onset"]
                            scalars = {
                                "u_stem_drums": float(out["drums"].max()),
                                "u_stem_bass": float(out["bass"].max()),
                                "u_stem_tonal": float(out["tonal"].max()),
                            }
                            # Fed the DRUMS row, at this same 60 Hz cadence.
                            # snap[idx]["timestamp"] is the delay-aligned audio
                            # clock (what the shader is actually showing);
                            # monotonic() advances the beat count, so a stalled
                            # audio stream cannot freeze the orrery.
                            scalars.update(beat_tracker.update(
                                out["drums"], snap[idx]["timestamp"],
                                time.monotonic()))
                            # Sets the tracker's reference for the NEXT sample
                            # when the song is recognised, and records this one
                            # when it is not.
                            beat_sess.update()
                            audio_state[0] = (snap[idx], tex, scalars)
            except Exception as e:
                # A crash in here must never take the backdrop down -- the
                # render loop falls back to its previous values.
                print(f"audio_reader_thread: {e}", file=sys.stderr)
                time.sleep(0.05)
            time.sleep(0.0008)

    threading.Thread(target=audio_reader_thread, daemon=True,
                     name="audio-reader").start()

    def read_mouse_real():
        # Real X11 pointer position, mapped into canvas space via the
        # primary output's own crop offset within the canvas (the primary
        # window is where iMouse is most meaningful -- no shader here
        # currently depends on cross-output mouse precision). Y flipped to
        # GL's bottom-left origin. No real click detection possible for an
        # override-redirect background window that never receives input
        # focus -- zw always 0 (Shadertoy's "never clicked" state).
        #
        # This is real global cursor position, not "cursor over the
        # backdrop" -- an override-redirect window never receives focus, so
        # there's no way to know if the cursor is meaningfully positioned
        # for this shader vs. just resting wherever the user's actual
        # foreground app happens to have it. Any already-ported shader that
        # branches on iMouse (e.g. grid_landscape.glsl's `if iMouse.xy ==
        # vec2(0,0)` auto-camera check) is effectively driven by that
        # incidental position whenever mouse mode is "real" -- see
        # read_mouse() below for the audio-driven default that replaces
        # this noise with something deliberate.

        primary_spec = OUTPUTS[0]
        # Must use the SAME uniform factor the blit uses, or the cursor lands
        # somewhere the picture isn't: when the head is bigger than the canvas
        # the mapping is not a plain centred offset any more, it is offset
        # PLUS the scale that crop_for_spec picked.
        k = canvas_per_screen(primary_spec)
        crop_x = (CANVAS_W - primary_spec["w"] * k) / 2.0
        crop_y = (CANVAS_H - primary_spec["h"] * k) / 2.0
        try:
            p = primary_xdisp.screen().root.query_pointer()
            mx = float((p.root_x - primary_spec["x"]) * k + crop_x)
            my = float(CANVAS_H - ((p.root_y - primary_spec["y"]) * k + crop_y))
            return (mx, my, 0.0, 0.0)
        except Exception:
            return (0.0, 0.0, 0.0, 0.0)

    def read_mouse_synthetic(frame):
        # Audio-driven iMouse: u_centroid (spectral brightness, independent
        # of loudness -- see SHADER_REFERENCE.md) drives x, u_rms drives y.
        # Both are already smoothed 0..1 scalars from shader_bands.cpp, so
        # no extra shaping needed here -- straight linear map into canvas
        # pixel space, matching how real mouse position is just raw pixels
        # too (no accumulator involved: position is fine to drive directly
        # from a live level, unlike *speed*, which is what the
        # additive-accumulator rule in SHADER_REFERENCE.md is actually
        # about).
        if frame is None:
            return (0.0, 0.0, 0.0, 0.0)
        mx = frame["centroid"] * CANVAS_W
        my = frame["rms"] * CANVAS_H
        return (mx, my, 0.0, 0.0)

    def resolve_steer_keycode():
        try:
            with open(STEER_KEY_FILE) as f:
                name = f.read().strip() or STEER_KEY_DEFAULT
        except OSError:
            name = STEER_KEY_DEFAULT
        keysym = XK.string_to_keysym(name)
        if not keysym:
            print(f"steer key: unknown keysym {name!r}, falling back to "
                  f"{STEER_KEY_DEFAULT}", file=sys.stderr)
            keysym = XK.string_to_keysym(STEER_KEY_DEFAULT)
        code = primary_xdisp.keysym_to_keycode(keysym)
        print(f"steer key: {name} -> keycode {code}", file=sys.stderr)
        return code

    steer_keycode = resolve_steer_keycode()

    def read_steer():
        # Global key state -- no focus required, no key events needed.
        if not steer_keycode:
            return 0.0
        try:
            km = primary_xdisp.query_keymap()
            return 1.0 if (km[steer_keycode >> 3] & (1 << (steer_keycode & 7))) else 0.0
        except Exception:
            return 0.0

    def resolve_keycode(name):
        try:
            keysym = XK.string_to_keysym(name)
            return primary_xdisp.keysym_to_keycode(keysym) if keysym else 0
        except Exception:
            return 0

    ctrl_keycode  = resolve_keycode("Control_L")
    shift_keycode = resolve_keycode("Shift_L")

    def read_key(code):
        if not code:
            return 0.0
        try:
            km = primary_xdisp.query_keymap()
            return 1.0 if (km[code >> 3] & (1 << (code & 7))) else 0.0
        except Exception:
            return 0.0

    # Mouse camera, gated on the hold-to-steer key (Super):
    #   Super + cursor            -> look (yaw/pitch)
    #   Super + Ctrl + cursor     -> move (strafe / forward-back)
    #   Super + Shift + cursor    -> up-down + roll
    # State lives in the renderer because a shader here has no spare buffer for
    # it; the shader reads it back as accumulated offsets (u_cam_look/u_cam_move).
    MOUSE_LOOK_SENS = 0.0022   # rad per canvas px
    MOUSE_MOVE_SENS = 0.030    # world(Rs) per canvas px
    MOUSE_ROLL_SENS = 0.0018   # rad per canvas px
    mouse_cam = load_freefly()
    freefly_dirty = {"v": False, "last": 0.0}
    prev_mouse = None
    prev_steering = False
    gyro = {"m": -1.0, "active": False, "xy": (0.0, 0.0, 0.0), "prev": None}
    persist_last = {"t": 0.0, "dir": None, "saved": None}
    autozoom = {"m": -1.0, "v": 0.0}
    cam_target = {"m": -1.0, "t": (0.0, 0.0, 0.0, 0.0), "id": 0.0}
    autodive = {"m": -1.0, "on": False, "beats": 0.0, "last": None, "id": None}

    def autodive_next():
        # the same step as `backdrop-dive next`: stop idx+1, or 1 past the last
        with open(DIVE_PATH_FILE) as fh:
            stops = [ln.split() for ln in fh if ln.strip()]
        if not stops:
            return
        try:
            with open(DIVE_IDX_FILE) as fh:
                idx = int(fh.read().strip() or 0)
        except (OSError, ValueError):
            idx = 0
        nxt = idx + 1 if idx < len(stops) else 1
        new_id = (int(cam_target["id"]) + 1) % 100000
        tmp = CAM_TARGET_FILE + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(" ".join(stops[nxt - 1][:4]) + f" {new_id}\n")
        os.replace(tmp, CAM_TARGET_FILE)
        with open(DIVE_IDX_FILE, "w") as fh:
            fh.write(f"{nxt}\n")
        autodive["id"] = float(new_id)
        bpm = audio_state[0][2].get("u_beat_bpm", 0.0)
        print(f"autodive: stop {nxt}/{len(stops)} ({bpm:.0f} bpm)", file=sys.stderr)

    camera_cache = {"mtime": -1.0, "value": CAMERA_DEFAULT}

    def read_camera():
        # Cached on mtime: this is polled every frame, but the file only
        # changes when a keybinding fires.
        try:
            mtime = os.path.getmtime(CAMERA_FILE)
        except OSError:
            return CAMERA_DEFAULT
        if mtime != camera_cache["mtime"]:
            try:
                with open(CAMERA_FILE) as f:
                    parts = f.read().split()
                camera_cache["value"] = (
                    float(parts[0]), float(parts[1]), float(parts[2]))
            except (OSError, ValueError, IndexError):
                camera_cache["value"] = CAMERA_DEFAULT
            camera_cache["mtime"] = mtime
        return camera_cache["value"]

    def read_mouse_mode():
        try:
            with open(MOUSE_MODE_FILE) as f:
                mode = f.read().strip()
        except FileNotFoundError:
            mode = ""
        # Default is "audio", not "real" -- real mouse is incidental noise
        # for an override-redirect window that never gets focus (see
        # read_mouse_real()'s docstring), so audio-driven is the more
        # meaningful default, not just an opt-in extra.
        return "real" if mode == "real" else "audio"

    def read_mouse(frame):
        if read_mouse_mode() == "real":
            return read_mouse_real()
        return read_mouse_synthetic(frame)

    last_frame = None
    start_time = time.monotonic()
    last_time = start_time
    frame_counter = 0
    bass_accum = 0.0
    onset_accum = 0.0
    rms_accum = 0.0
    mid_accum = 0.0
    presence_accum = 0.0
    onset_smooth = 0.0
    color_phase = 0.0
    print("renderer: entering main loop (close window or Ctrl+C to stop)", file=sys.stderr)
    perf_last = time.monotonic()
    perf_frames = 0
    perf_frame_ms_sum = 0.0
    perf_render_ms_sum = 0.0
    perf_min_ms = 1e9
    perf_max_ms = 0.0
    try:
        crop_for = crop_for_spec

        def sync_layout():
            """Re-detect outputs and resize/reposition the backdrop windows to
            match. Without this, rotating a display or changing its mode leaves
            the windows at their startup geometry, the centred crop visibly
            breaks, and the only fix is restarting the renderer by hand.

            Cheap because the canvas is a FIXED size: nothing is reallocated
            here, only window geometry and each window's crop rect."""
            fresh = detect_outputs()
            changed = False
            for i, spec in enumerate(fresh):
                if i >= len(presenters):
                    break
                cur = presenters[i]["spec"]
                if (cur["w"], cur["h"], cur["x"], cur["y"]) == \
                   (spec["w"], spec["h"], spec["x"], spec["y"]):
                    continue
                changed = True
                print(f"layout: {cur['name']} {cur['w']}x{cur['h']}+{cur['x']}+{cur['y']}"
                      f" -> {spec['w']}x{spec['h']}+{spec['x']}+{spec['y']}", file=sys.stderr)
                # hz too: a mode change is exactly when refresh moves, and a
                # stale value here would mis-throttle this window's presents.
                cur.update(w=spec["w"], h=spec["h"], x=spec["x"], y=spec["y"],
                           hz=spec.get("hz", cur.get("hz", 60.0)))
                win = presenters[i]["window"]
                glfw.set_window_size(win, spec["w"], spec["h"])
                glfw.set_window_pos(win, spec["x"], spec["y"])
                # Re-apply override-redirect + lowest stacking: the WM can
                # reassert itself when a window is reconfigured.
                make_backdrop(glfw.get_x11_window(win),
                              spec["x"], spec["y"], spec["w"], spec["h"])
                off, scale = crop_for(spec)
                presenters[i]["uv_offset"], presenters[i]["uv_scale"] = off, scale
            return changed

        LAYOUT_POLL_SECONDS = 5.0
        # Log any loop iteration whose wall time crosses this, with a phase
        # breakdown, so an intermittent freeze names its own culprit (poll /
        # layout xrandr / context+reload / render / present) instead of being
        # guessed at. 2026-09-20.
        STALL_LOG_MS = float(os.environ.get("SHADER_STALL_LOG_MS", "200"))
        frame_deadline = 0.0
        last_layout_check = time.monotonic()
        last_root_geom = None

        while not any(glfw.window_should_close(p["window"]) for p in presenters):
            t0 = time.monotonic()
            glfw.poll_events()
            t1 = time.monotonic()

            if time.monotonic() - last_layout_check >= LAYOUT_POLL_SECONDS:
                last_layout_check = time.monotonic()
                # Cheap X-only root-geometry check FIRST. `xrandr --query`
                # makes the dGPU probe every physical connector for EDID
                # (each a DDC read that fails here, ~ms-to-1s, and logs
                # "No EDID found" per connector). Polling it every 30s meant a
                # constant, needless DRM probe on the render/encode GPU --
                # seen in the kernel log at a metronomic 30s cadence. Only run
                # the full xrandr re-detect when the root geometry actually
                # changed, which is exactly what a rotate/resize does.
                # `get_geometry()` is a plain X round-trip, no DRM probe.
                try:
                    g = primary_xdisp.screen().root.get_geometry()
                    root_geom = (g.width, g.height)
                except Exception:
                    root_geom = None
                if root_geom != last_root_geom:
                    last_root_geom = root_geom
                    try:
                        sync_layout()
                    except Exception as e:
                        print(f"layout: re-detect failed ({e}), keeping current geometry",
                              file=sys.stderr)

            t2 = time.monotonic()
            glfw.make_context_current(primary_window)
            try_reload()
            frame_start = time.monotonic()
            t3 = frame_start

            now = time.monotonic()
            elapsed = now - start_time
            dt = now - last_time
            last_time = now

            # Our own convention, kept alongside the Shadertoy-standard
            # names below so bars_test.glsl and any hand-written shader
            # using u_time/u_resolution keep working unchanged. Always the
            # full canvas size -- the real render never knows about
            # individual output windows, only the shared envelope.
            set_uniform("u_time", elapsed)
            set_uniform("u_resolution", (float(CANVAS_W), float(CANVAS_H)))

            # Real Shadertoy uniform vocabulary -- matching these names/types
            # exactly means most of ~/shaders/Shaders (226 raw Shadertoy
            # .frag files, not just our own hand-adapted ones) work with zero
            # shader-side edits, since they already reference these names.
            set_uniform("iTime", elapsed)
            set_uniform("iTimeDelta", dt)
            set_uniform("iFrame", frame_counter)
            set_uniform("iResolution", (float(CANVAS_W), float(CANVAS_H), 1.0))
            set_uniform("iSampleRate", AUDIO_SAMPLE_RATE)
            # 2026-09-27: while the steer key is held, steering means the HAND --
            # always the real cursor, whatever mouse mode is set. With the audio
            # mouse (the default) the steer deltas below were the centroid/rms
            # wobble, so music shook the camera whenever Super was held.
            super_held = read_steer()
            steering = super_held > 0.5
            cur_mouse = read_mouse_real() if steering else read_mouse(last_frame)
            set_uniform("iMouse", cur_mouse)
            cam_elev, cam_yaw, cam_zoom = read_camera()
            set_uniform("u_cam_elev", cam_elev)
            set_uniform("u_cam_yaw", cam_yaw)
            set_uniform("u_cam_zoom", cam_zoom)
            set_uniform("u_steer", super_held)
            # first steered frame only records the position (prev was the audio mouse)
            if steering and prev_mouse is not None and prev_steering:
                mdx = cur_mouse[0] - prev_mouse[0]
                mdy = cur_mouse[1] - prev_mouse[1]
                if read_key(ctrl_keycode) > 0.5:
                    mouse_cam["r"] += mdx * MOUSE_MOVE_SENS
                    mouse_cam["f"] += mdy * MOUSE_MOVE_SENS
                elif read_key(shift_keycode) > 0.5:
                    mouse_cam["u"]    += mdy * MOUSE_MOVE_SENS
                    mouse_cam["roll"] += mdx * MOUSE_ROLL_SENS
                else:
                    mouse_cam["yaw"]   += mdx * MOUSE_LOOK_SENS
                    mouse_cam["pitch"] += mdy * MOUSE_LOOK_SENS
            prev_mouse = cur_mouse
            prev_steering = steering
            # phone gyro (see GYRO_FILE): its per-frame change drags exactly like
            # a Super + cursor move (look channel)
            try:
                g_m = os.path.getmtime(GYRO_FILE)
                if g_m != gyro["m"]:
                    gyro["m"] = g_m
                    with open(GYRO_FILE) as fh:
                        gv = [float(x) for x in fh.read().split()[:4]]
                    if len(gv) >= 3:
                        gyro["active"] = gv[0] > 0.5
                        gyro["xy"] = (gv[1], gv[2], gv[3] if len(gv) > 3 else 0.0)
                if time.time() - g_m > GYRO_STALE_S:
                    gyro["active"] = False
            except (OSError, ValueError):
                gyro["active"] = False
            # a release still applies its final offset (the relay writes the last
            # one on release), so no motion is lost to a missed sample
            if gyro["active"] or (gyro["prev"] is not None and time.time() - gyro["m"] <= GYRO_STALE_S):
                if gyro["prev"] is not None:
                    mouse_cam["yaw"]   += (gyro["xy"][0] - gyro["prev"][0]) * MOUSE_LOOK_SENS
                    mouse_cam["pitch"] += (gyro["xy"][1] - gyro["prev"][1]) * MOUSE_LOOK_SENS
                    # twist about the pointing axis = Super + Ctrl vertical drag
                    # (nebulabrot: zoom; free-fly shaders: forward/back)
                    mouse_cam["f"]     += (gyro["xy"][2] - gyro["prev"][2]) * MOUSE_MOVE_SENS
                gyro["prev"] = gyro["xy"] if gyro["active"] else None
                super_held = 1.0          # the free-fly pose gets saved like a Super steer
                set_uniform("u_steer", 1.0)   # shaders gating on it see a steer
            else:
                gyro["prev"] = None
            set_uniform("u_cam_look", (mouse_cam["yaw"], mouse_cam["pitch"], mouse_cam["roll"]))
            set_uniform("u_cam_move", (mouse_cam["r"], mouse_cam["u"], mouse_cam["f"]))
            # Super+Z (i3 -> ~/bin/backdrop-autozoom) flips autozoom.txt: 0/1
            try:
                az_m = os.path.getmtime(AUTOZOOM_FILE)
                if az_m != autozoom["m"]:
                    autozoom["m"] = az_m
                    with open(AUTOZOOM_FILE) as fh:
                        autozoom["v"] = 1.0 if fh.read().strip() == "1" else 0.0
            except (OSError, ValueError):
                autozoom["v"] = 0.0
            set_uniform("u_autozoom", autozoom["v"])
            # dive path (~/bin/backdrop-dive -> cam_target.txt:
            # "re im zoom rot id"); the shader flies to it when the id changes
            try:
                ct_m = os.path.getmtime(CAM_TARGET_FILE)
                if ct_m != cam_target["m"]:
                    cam_target["m"] = ct_m
                    with open(CAM_TARGET_FILE) as fh:
                        v = [float(x) for x in fh.read().split()[:5]]
                    if len(v) == 5:
                        cam_target["t"], cam_target["id"] = tuple(v[:4]), v[4]
                        print(f"cam target: {cam_target['t']} id {cam_target['id']:.0f}", file=sys.stderr)
            except (OSError, ValueError):
                pass
            set_uniform("u_cam_target", cam_target["t"])
            set_uniform("u_cam_target_id", cam_target["id"])
            # auto-dive (Super+Up): counted on the beat clock, like the flights
            try:
                ad_m = os.path.getmtime(AUTODIVE_FILE)
                if ad_m != autodive["m"]:
                    autodive["m"] = ad_m
                    with open(AUTODIVE_FILE) as fh:
                        on = fh.read().strip() == "1"
                    if on and not autodive["on"]:
                        autodive["beats"] = AUTODIVE_BEATS      # first jump right away
                        autodive["last"], autodive["id"] = None, cam_target["id"]
                    autodive["on"] = on
            except OSError:
                autodive["on"] = False
            if autodive["on"]:
                bc = audio_state[0][2].get("u_beat_count")
                if bc is not None:
                    if autodive["last"] is not None:
                        # forward steps only: the clock can tick back a hair
                        # (phase correction / cached-grid lock), and wrapped
                        # that reads as ~1024 beats -- which fired every frame
                        d = (bc - autodive["last"]) % 1024.0
                        autodive["beats"] += d if d < 4.0 else 0.0
                    autodive["last"] = bc
                if cam_target["id"] != autodive["id"]:          # a manual jump
                    autodive["beats"], autodive["id"] = 0.0, cam_target["id"]
                if autodive["beats"] >= AUTODIVE_BEATS:
                    autodive["beats"] = 0.0
                    try:
                        autodive_next()
                    except OSError as e:
                        print(f"autodive: {e}", file=sys.stderr)
            if mp is not None and mp["manifest"].get("persist"):
                if persist_last.get("dir") != mp["dir"]:
                    persist_last["dir"] = mp["dir"]
                    persist_last["saved"] = load_persist(mp)
                saved = persist_last["saved"]
                set_uniform("u_persist0", saved if saved else (0.0, 0.0, 0.0, 0.0))
                set_uniform("u_persist_valid", 1.0 if saved else 0.0)
            # Persist the free-fly pose, throttled: it only changes while the
            # steer key is held, and one write per second is plenty to survive a
            # restart or be read back and frozen into the shader.
            if super_held > 0.5:
                freefly_dirty["v"] = True
            if freefly_dirty["v"] and (elapsed - freefly_dirty["last"]) > 1.0:
                save_freefly(mouse_cam)
                freefly_dirty["last"] = elapsed
                freefly_dirty["v"] = False
            now_wall = datetime.datetime.now()
            seconds_since_midnight = (
                now_wall - now_wall.replace(hour=0, minute=0, second=0, microsecond=0)
            ).total_seconds()
            set_uniform("iDate", (float(now_wall.year), float(now_wall.month),
                                   float(now_wall.day), seconds_since_midnight))
            for i, size in enumerate(channel_sizes):
                set_uniform(f"iChannelResolution[{i}]", (float(size[0]), float(size[1]), 1.0))

            frame_counter += 1

            # The reader thread owns the shm now: it drains every published
            # block at ~188 Hz, does the delay-alignment, and runs the stem
            # separation. The render loop just consumes the latest published
            # result at whatever rate it happens to be running -- which is the
            # whole point of moving the analysis upstream.
            delayed, stems_arr, stem_scalars = audio_state[0]
            if delayed is not None:
                last_frame = delayed
            if stems_arr is not None:
                stems_tex.write(stems_arr.tobytes())
                for sname, svalue in stem_scalars.items():
                    set_uniform(sname, svalue)
            if last_frame is not None:
                bands = np.array(last_frame["bands"], dtype="f4")
                # Bands 117-119 are ALWAYS NaN (verified 2026-09-20: 100% of
                # frames, never any other index). Their centre frequencies are
                # 23.7/25.1/26.6 kHz and fs is 48 kHz, so those RBJ bandpasses
                # are designed at or above Nyquist -- w0 = 2*pi*f0/fs >= pi
                # gives degenerate coefficients. Every existing shader happens
                # to be immune because they all drop the dead top octave, but
                # ANY consumer that aggregates across the full 120 inherits the
                # NaN. Sanitized here, once, so no downstream consumer has to
                # know. The real fix is clamping the bank to Nyquist in
                # shader_bands.cpp; that needs a native rebuild.
                np.nan_to_num(bands, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                bands_tex.write(bands.tobytes())
                for name in META_FEATURES:
                    value = last_frame[name]
                    if name in LEVEL_FEATURES:
                        value = condition_level(value)
                    elif name == "centroid":
                        value = condition_centroid(value)
                    set_uniform(f"u_{name}", value)


                # Running integrals of bass/onset -- for driving motion
                # *speed* without the classic "multiply audio level straight
                # into iTime" bug (rescales the whole elapsed-time clock by a
                # live coefficient, so envelope wobble gets amplified by
                # however many seconds have already played). Additive use in
                # a shader instead: iTime*base_rate + mod*u_bass_accum.
                # Same convention as shader-musicvideo's wrap_shader.py.
                # Shaped BEFORE integration with a steep, smooth power curve
                # (no threshold/step anywhere -- a hard gate was tried and
                # reverted: it makes quiet content read exactly 0 and loud
                # content jump straight to a similar high value, which is
                # *more* on/off, not less). A single continuous curve instead:
                # raw rms/bass from shader_bands.cpp reads meaningfully above
                # zero even for quiet ambient content (its dB floor is
                # generous), so a high exponent is needed to keep quiet
                # content compressed low while still spreading real
                # loud-vs-louder content out proportionally above it.
                SHAPE_EXP = 3.0

                def shape(x: float) -> float:
                    return x ** SHAPE_EXP

                bass_accum += shape(last_frame["bass"]) * dt
                onset_accum += last_frame["onset"] * dt
                set_uniform("u_bass_accum", bass_accum)
                set_uniform("u_onset_accum", onset_accum)

                # u_onset_smooth: an EMA low-pass of raw u_onset, ~150ms time
                # constant. Exists because raw u_onset (already a punchy
                # 15ms-attack/180ms-decay envelope at the native-plugin
                # level -- see shader_bands.cpp) is still fast enough that
                # plugging it *directly* into anything spatial (a radius, a
                # camera zoom, a raymarched surface height) reads as a jerky
                # snap each hit rather than a punch, because those are
                # evaluated fresh every frame with no continuity of their
                # own -- unlike colour/brightness, which the eye reads as
                # forgiving of frame-to-frame wiggle. This is the same
                # "integrate, don't plug in the raw level" principle as
                # bass_accum/rms_accum and u_color_phase below, just for a
                # decaying pulse instead of an ever-advancing one. Use this
                # (never raw u_onset) for anything that visibly moves a
                # surface/position/scale; raw u_onset stays fine for
                # additive brightness/glow terms.
                ONSET_SMOOTH_TAU = 0.15
                onset_smooth += (last_frame["onset"] - onset_smooth) * min(1.0, dt / ONSET_SMOOTH_TAU)
                set_uniform("u_onset_smooth", onset_smooth)

                # Overall-loudness accumulator, same monotonic-clock contract
                # as u_bass_accum but driven by broadband RMS instead of bass.
                rms_accum += shape(last_frame["rms"]) * dt
                set_uniform("u_rms_accum", rms_accum)

                # Same monotonic-accumulator contract, for shaders (FBM_Opus.mp)
                # that want an independent motion rate per band instead of just
                # bass/rms. mid_accum drives a mid-scale detail layer;
                # presence_accum drives fine-detail/surface crawl.
                mid_accum += shape(last_frame["mid"]) * dt
                set_uniform("u_mid_accum", mid_accum)

                presence_accum += shape(last_frame["presence"]) * dt
                set_uniform("u_presence_accum", presence_accum)

                # Running phase for the shader's EXISTING colour oscillation
                # (its sin(t)-based brightness pulsing) -- NOT a hue offset,
                # never touches hue. Only the *speed* this phase advances at
                # reacts to treble (a direct multiplier on the instantaneous
                # envelope -- quiet ~1x, loud ~3x, no separate history/decay
                # needed since it's just today's level, not an integral).
                # Still has to be an always-forward-advancing phase rather
                # than `sin(iTime * multiplier)` directly, though: rescaling
                # elapsed time by a live-changing multiplier retroactively
                # warps everything already played, causing a visible jump
                # each time the multiplier changes. Advancing a running
                # phase by rate*dt avoids that -- and unlike the earlier hue
                # attempt, this phase only ever feeds into sin(), which is
                # bounded/periodic no matter how large the input gets, so it
                # can never drift through colours the way a raw hue offset
                # did.
                treble = max(last_frame["highmid"], last_frame["presence"], last_frame["brilliance"])
                COLOR_BASE_RATE = 0.05  # matches the shader's own BASE_SPEED
                COLOR_TREBLE_BOOST = 2.0  # +2x at full treble -> ~3x total
                color_phase += COLOR_BASE_RATE * (1.0 + COLOR_TREBLE_BOOST * treble) * dt
                set_uniform("u_color_phase", color_phase)

            if audio_reader is not None:
                audio_frame = audio_reader.read()
                if audio_frame is not None:
                    combined = np.concatenate([
                        np.array(audio_frame["freq"], dtype="f4"),
                        np.array(audio_frame["wave"], dtype="f4"),
                    ])
                    audio_channel_tex.write(combined.tobytes())

            # The one expensive render: the real shader, into the shared
            # offscreen canvas, at full envelope size. Everything above
            # this line only needed to happen once, under the primary
            # context -- everything below is cheap presentation. Multipass
            # renders every buffer pass first (see render_multipass()) and
            # ends up in the same place -- canvas_fbo holding this frame's
            # Image pass output -- so everything below is identical either
            # way.
            render_start = time.monotonic()
            if mp is not None:
                if mp.get("compute") is not None:
                    mp["compute"].step(dt, elapsed)
                render_multipass(ctx, mp, canvas_fbo, audio_channel_tex)
                if mp["manifest"].get("persist") and elapsed - persist_last["t"] > PERSIST_SECONDS:
                    persist_last["t"] = elapsed
                    try:
                        v = save_persist(mp)
                        if v:
                            persist_last["saved"] = v    # a hot reload restores the LATEST view
                    except Exception as e:
                        print(f"persist: save failed: {e}", file=sys.stderr)
            else:
                canvas_fbo.use()
                ctx.clear(0.0, 0.0, 0.0, 1.0)
                vao.render(moderngl.TRIANGLE_STRIP)
            render_ms = (time.monotonic() - render_start) * 1000.0

            # Capture, before the presenters: canvas_fbo holds this frame and
            # the blits below only read from it. Both calls are no-ops unless a
            # recording is armed.
            if canvas_capture is not None:
                canvas_capture.poll()
                canvas_capture.grab()

            t4 = time.monotonic()

            now_present = time.monotonic()
            for pi, p in enumerate(presenters):
                # The primary presents every iteration -- it is the vsync the
                # loop is aligned to. A secondary is skipped whenever it has
                # already been presented within its own refresh period: there
                # is only ONE canvas and it has not changed again yet, so a
                # second blit would be wasted work. Its front buffer holds the
                # previous frame, which is exactly the "repeat the frame"
                # behaviour a display gives for free.
                if pi > 0:
                    period = 1.0 / max(p["spec"].get("hz", 60.0), 1.0)
                    if now_present - p["last_present"] < period * 0.995:
                        continue
                p["last_present"] = now_present
                glfw.make_context_current(p["window"])
                p["ctx"].screen.use()
                canvas_tex.use(location=0)  # shared GL namespace via `share=`
                p["prog"]["src_tex"].value = 0
                p["prog"]["uv_offset"].value = p["uv_offset"]
                p["prog"]["uv_scale"].value = p["uv_scale"]
                p["prog"]["tap"].value = (0.25 * p["uv_scale"][0] / p["spec"]["w"],
                                          0.25 * p["uv_scale"][1] / p["spec"]["h"])
                p["ctx"].clear(0.0, 0.0, 0.0, 1.0)
                p["vao"].render(moderngl.TRIANGLE_STRIP)
                glfw.swap_buffers(p["window"])
            t5 = time.monotonic()

            total_ms = (t5 - t0) * 1000.0
            if total_ms >= STALL_LOG_MS:
                print(
                    f"stall: {total_ms:.1f}ms | poll {(t1 - t0) * 1000:.1f} "
                    f"layout {(t2 - t1) * 1000:.1f} ctx+reload {(t3 - t2) * 1000:.1f} "
                    f"render {(t4 - t3) * 1000:.1f} present {(t5 - t4) * 1000:.1f}",
                    file=sys.stderr,
                )

            # Clock-based limiter. This, not the swap, is what sets the frame
            # rate now. A shader that cannot reach its target just runs slower
            # and never sleeps here; the deadline then resyncs instead of
            # accumulating an ever-growing debt it would try to "catch up" by
            # rendering flat-out.
            target_fps = pacing["target_fps"]
            if target_fps and target_fps > 0.0:
                period = 1.0 / target_fps
                now_pace = time.monotonic()
                if frame_deadline <= 0.0:
                    frame_deadline = now_pace + period
                else:
                    slack = frame_deadline - now_pace
                    if slack > 0.0005:
                        time.sleep(slack)
                    frame_deadline += period
                    if frame_deadline < time.monotonic() - period:
                        frame_deadline = time.monotonic() + period

            frame_ms = (time.monotonic() - frame_start) * 1000.0
            perf_frames += 1
            perf_frame_ms_sum += frame_ms
            perf_render_ms_sum += render_ms
            perf_min_ms = min(perf_min_ms, frame_ms)
            perf_max_ms = max(perf_max_ms, frame_ms)
            now = time.monotonic()
            if now - perf_last >= 2.0:
                elapsed = now - perf_last
                n = max(1, perf_frames)
                print(
                    f"perf: {perf_frames / elapsed:5.1f} fps  "
                    f"frame {perf_frame_ms_sum / n:6.1f} ms (min {perf_min_ms:5.1f} max {perf_max_ms:6.1f})  "
                    f"render {perf_render_ms_sum / n:6.1f} ms",
                    file=sys.stderr,
                )
                perf_last = now
                perf_frames = 0
                perf_frame_ms_sum = 0.0
                perf_render_ms_sum = 0.0
                perf_min_ms = 1e9
                perf_max_ms = 0.0
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        if audio_reader is not None:
            audio_reader.close()
        glfw.terminate()


if __name__ == "__main__":
    main()
