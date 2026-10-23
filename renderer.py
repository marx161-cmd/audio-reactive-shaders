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
import music_state
import song_replay
import held_tone
import beat_track
import stem_split
import kick_detect
import band_filters
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
# 2026-10-15: the DroidPad control surface on the Pixel (com.github.umer0586.droidpad
# -> MQTT -> ~/bin/droidpad-shader-bridge -> this file). Same hot-read convention as
# the state files above. The bridge writes the raw widgets plus MONOTONIC event
# counters (the renderer only samples at frame rate, so counting happens where every
# MQTT message is seen); we diff the counters per frame and publish the whole surface
# as u_pad_* uniforms for a shader to steer itself with. Nothing here overrides the
# shader -- these are forces/bias/pressed-edges, and the shader decides. Treated as
# gone (u_pad_alive 0) when the file goes stale, so shaders fall back to their own
# motion rather than freezing on the last command.
PAD_FILE = os.path.join(os.path.dirname(__file__), "pad.json")
PAD_STALE_S = 3.0
# 2026-10-17 PERISCOPE (scope.md "PERISCOPE mode"): ~/bin/periscope-stream writes the Pixel's fused orientation here
# ("unix_t qx qy qz qw", Android world frame, phone -> world) while the pad's "Periscope" switch is ON. The free camera
# then follows the phone 1:1 as a full 3-axis window (camera forward = out of the phone's back), relative to the pose at
# engage. Stale (no sample for PERISCOPE_STALE_S) = released; the sticks take over again.
PERISCOPE_FILE = os.path.join(os.path.dirname(__file__), "periscope_state.txt")
PERISCOPE_STALE_S = 0.5


def read_periscope():
    """-> the phone's xyzw quaternion, or None if missing/stale."""
    try:
        with open(PERISCOPE_FILE) as fh:
            v = [float(x) for x in fh.read().split()]
        if len(v) != 5 or time.time() - v[0] > PERISCOPE_STALE_S:
            return None
        n = math.sqrt(sum(c * c for c in v[1:])) or 1.0
        return tuple(c / n for c in v[1:])
    except (OSError, ValueError):
        return None


def periscope_rel_cam(q0, q):
    """The phone's turn since engage (in its own frame), re-expressed in camera space. The phone frame is x right,
    y up the screen, z OUT of the screen; the camera is x right, y up, z forward = out of the phone's BACK, i.e. the
    phone frame mirrored in z. Conjugating a rotation by that mirror maps quaternion (x, y, z, w) -> (-x, -y, z, w)."""
    r = quat_mul((-q0[0], -q0[1], -q0[2], q0[3]), q)
    return (-r[0], -r[1], r[2], r[3])
PAD_SEP_STEP = 0.05          # per dpad click, the figures' separation nudge
# 2026-10-17 XBOX CONTROLLER (scope.md "NEW shader fishbox.mp"): ~/bin/xpad-shader-bridge reads the Elite
# controller over evdev and writes gamepad.json (sticks -1..1 in evdev sign, triggers 0..1, held buttons,
# monotonic press counters). Same hot-read contract as pad.json, but its OWN namespace: published as u_xb_*
# and nothing else -- the phone pads never see the controller and no shader but the one that asks reads it
# (pads/README.md "two namespaces"). Stale = controller gone -> u_xb_alive 0, the shader runs its own dance.
GAMEPAD_FILE = os.path.join(os.path.dirname(__file__), "gamepad.json")
XB_STALE_S = 3.0             # the bridge heartbeats once a second
XB_DEADZONE = 0.12           # stick slop around centre (the Elite's sticks rest a few % off)
XB_STICK_Y_UP = -1.0         # evdev reports +y = DOWN; the one place to flip "stick up = up"
# THE LAYER MAP (user: strict split -- left paddles / LT / LB belong to the LEFT stick = flock 0, right side
# = flock 1). The paddles arrive as the face buttons the controller's profile maps them to, so which face
# button is which paddle is a per-profile fact: remap HERE, live, on first test (user: "remap is sth we
# should be testing"). Layer 0 = none (MOVE), 1 = upper paddle (AXIS), 2 = lower paddle (SHAPE).
XB_LAYER_MAP = {"left": {"upper": "ls", "lower": "b"},    # read off the bridge log 2026-10-17 11:42 (this profile:
                "right": {"upper": "rs", "lower": "x"}}   #  the paddles send stick clicks + B / X)
# ... and the SAME shape in gamepad_map.json next to this file OVERRIDES it, hot-read (mtime), so a remap is an
# edit to that file, not a renderer restart: {"left": {"upper": "ls", "lower": "b"}, "right": {...}}
GAMEPAD_MAP_FILE = os.path.join(os.path.dirname(__file__), "gamepad_map.json")

# ---- PAD FLIGHT + CROSSHAIR (2026-10-15, user: "maybe we can actually add 3d camera
# flight", "it should fling whatever comes into crosshair since orbits are hard to hit")
# The pad's own free-fly, in the same shape as the mouse one: a HELD stick deflection is
# a RATE, so letting go parks the framing where you left it instead of swimming it back.
# Values are per second so the feel is framerate-independent.
PAD_CAM_LOOK_RATE = 0.9      # rad/s at full stick: how fast the Look stick swings the view
PAD_CAM_MOVE_RATE = 3.0      # world units/s at full stick (brown dwarf: 1.0 = the dwarf's radius)
PAD_CAM_DEADZONE = 0.10      # stick slop around centre, in stick units (-1..1)
PAD_BOOP_DEADZONE = 0.12     # crosshair stick: under this the reticle is parked and uncatches
# The general remote's one safety cap. The pad's Look/Move sticks also drive the SHARED
# hand camera (see the pad block in the render loop), whose pose is GLOBAL and PERSISTED
# (freefly_state.txt): one long frame -- a shader reload, an output-layout change -- with a
# stick held would otherwise turn one frame into one huge fling of every flying shader's
# framing, and leave it there. 0.10 s is far above a 120 fps frame and far below a reload.
PAD_STEER_DT_MAX = 0.10      # s: most sim-time one held stick may contribute per frame
# Spring-return controls go back to rest on their own -- but only if they hear about it. The app
# sends a JOYSTICK's value on movement and a 0 on drag END, so a release that never arrives (a pad
# switched mid-drag, an app backgrounded, one QoS-0 message lost) leaves the last deflection
# latched, and the bridge's heartbeat then keeps it alive forever. A widget nobody has refreshed
# for this long therefore reads as its RESTING value (0), for sticks and for the wheel; sliders and
# switches are NOT spring-return and keep their value indefinitely. Generous on purpose: a finger
# resting dead still on glass still jitters, so a genuine hold keeps re-stamping itself.
PAD_STICK_TTL_S = 8.0        # s without any event before a spring-return widget reads as rest
# The general pad's STEERING_WHEEL is the one camera axis no stick can reach (roll). Its event is
# an ABSOLUTE angle in degrees (the app animates a released wheel back to 0), so it is applied as a
# DELTA and the phone's own self-centring unwinds it -- see the pad block. Full lock (360) is about
# 69 degrees of roll, which is enough to lay the horizon right over without losing the scene.
PAD_WHEEL_ROLL_SENS = 1.2 / 360.0
# 2026-10-16: the wheel now also rolls the free camera (u_pad_cam_q, see the wheel block). view_quat
# applies its roll as Rz(-roll), which is the window-cal convention; this is the one place to flip if
# tilting the phone's right edge down rolls the horizon the wrong way.
PHONE_ROLL_SIGN = 1.0
# The ONE place to flip every stick if a control feels inverted. DroidPad plots a
# joystick in screen coordinates, so a stick pushed "up" can arrive as -y; +1 here means
# "a joystick's +y is UP" (pitch up, fly forward, reticle up the canvas).
PAD_STICK_Y_UP = 1.0


def pad_xy(widgets, wid, now=None):
    """A JOYSTICK's deflection, or (0, 0) when it has gone quiet.

    Spring-return widgets have no "released" state on the wire -- only a 0 event that may
    never arrive (see PAD_STICK_TTL_S) -- so pass `now` and an unrefreshed stick reads as
    the centre it would have sprung back to. Sliders/switches keep their value: pad_val
    and pad_sw are untouched by any of this."""
    v = widgets.get(wid) or {}
    if now is not None and now - float(v.get("t", 0.0)) > PAD_STICK_TTL_S:
        return (0.0, 0.0)
    return (float(v.get("x", 0.0)), float(v.get("y", 0.0)))


def pad_val(widgets, wid):
    return float((widgets.get(wid) or {}).get("value", 0.0))


def pad_sw(widgets, wid):
    return 1.0 if (widgets.get(wid) or {}).get("state") else 0.0


PAD_ALIASES = {"farming off": "ring_farm_off", "cam NORMAL/PORTRAIT": "cam_mode",
               "cam NORMAL/PORTRAIT/SHIP A/SHIP B/ON C": "cam_mode",   # 2026-10-23: 5-stop cam slider
               "NORMAL/SLINGSHOT/ROSETTE": "sys_mode", "Star WHITE/YELLOW/RED": "star_colour",
               "mouse OFF/BOOP/GRAB": "Mouse", "STAR SLING": "Dwarf Big Shock", "MOON": "refill"}


def pad_val_opt(widgets, wid):
    """A slider's value, or None when the pad has never sent that widget (2026-10-22, ringbox step sliders)."""
    w = widgets.get(wid) or {}
    v = w.get("value")
    return None if v is None else float(v)


# ---- REMOTE VARIANTS (2026-10-15) --------------------------------------------------
# A shader and its remote-only sibling are a name pair: "<x>.mp" <-> "ctrl_<x>.mp". The
# pad's "off" switch picks which half is active by writing active_shader.txt (see the
# pad block in the render loop). These helpers are deliberately dumb: they only ever
# swap between an existing pair, and never invent a name.
# ---- LEAN UNIFORMS (2026-10-19, user: "we skip every file check and setting copy that is not mandatory"): every uniform
# write goes through uset(). It remembers each program's uniform object (or that the program has none -- GLSL drops unused
# ones, and the old code paid a raised+caught KeyError for that every frame) and skips a write whose value did not change.
# Entries hold the program itself and are only trusted while `entry[0] is p`, so a reloaded program never inherits them.
class _SkipCheck(Exception):
    pass


_UCACHE = {}
_MISSING = object()


def uset(p, name, value):
    # (2026-10-19: a cached / skip-unchanged version was MEASURED slower -- 24 % vs 12 % of the main thread -- because the
    #  audio thread shares the GIL and more Python steps = more forced switches. So: the original direct write.)
    try:
        p[name].value = value
    except KeyError:
        pass


def read_active_shader_name():
    try:
        with open(ACTIVE_SHADER_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


def write_active_shader_name(name):
    tmp = ACTIVE_SHADER_FILE + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(name + "\n")
    os.replace(tmp, ACTIVE_SHADER_FILE)


def shader_pair(name):
    """(auto_name, ctrl_name) for a shader that has a ctrl_ sibling, else None.

    Both halves must exist as real .mp dirs (manifest.json present), so a lone shader
    has no remote variant and the trigger stays inert on it."""
    if not name or not name.endswith(".mp"):
        return None
    base = name[len("ctrl_"):] if name.startswith("ctrl_") else name
    pair = (base, "ctrl_" + base)
    for n in pair:
        if not os.path.isfile(os.path.join(SHADER_DIR, n, "manifest.json")):
            return None
    return pair
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


# 2026-10-15 (scope.md "SIDE VIEW"): WINDOWS. A head can look at the SAME world from a fixed offset in
# the ship's frame -- like a ship's window. The ship (the shader's own camera: orbit, buffer-A
# quaternion, u_cam_look ...) is never touched here; a window is just a fixed rotation + field-of-view
# scale handed to the Image pass as `u_view_q` (xyzw quaternion, rotates the camera-space ray) and
# `u_view_fov` (uv scale). The front/default view is the identity, so shaders that do not declare
# them, and heads without a window entry, are byte-for-byte unchanged. Buffers and compute run once;
# only the Image pass runs again per window, into that head's own texture.
#
# windows.json (hot-reloaded; edited by tools/window-cal):
#   {"windows": [{"match": {"primary": true}, "yaw": -60, "pitch": 0, "roll": 0, "zoom": 1.0,
#                 "locked": false}]}
# match: "primary" (the vsync head), "index" (OUTPUTS order), "size" ("1920x1080" -- survives the xrandr
# name drift when monitors are rearranged) or "name" (xrandr name, drifts!). An entry with yaw/pitch/roll 0
# is a pure per-head ZOOM. zoom 2 = twice as magnified.
# Degrees; yaw+ looks RIGHT, pitch+ looks UP, roll+ is clockwise as seen by the viewer. "enabled":
# false turns an entry off; "locked" is honoured by tools/window-cal (the renderer reads values only).
WINDOWS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "windows.json")
VIEW_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def view_quat(yaw_deg, pitch_deg, roll_deg):
    """xyzw quaternion of R = Ry(yaw) * Rx(-pitch) * Rz(-roll), camera space x right, y up, z forward."""
    def axis_q(ax, ang):
        s, c = math.sin(ang / 2.0), math.cos(ang / 2.0)
        return (ax[0] * s, ax[1] * s, ax[2] * s, c)

    def mul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return (aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz)
    q = mul(mul(axis_q((0, 1, 0), math.radians(yaw_deg)),
                axis_q((1, 0, 0), -math.radians(pitch_deg))),
            axis_q((0, 0, 1), -math.radians(roll_deg)))
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return tuple(v / n for v in q)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_rot(q, v):
    """Rotate v by xyzw q -- the same formula as the shaders' viewRot()."""
    x, y, z, w = q
    cx = (y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0])
    t = (cx[0] + w * v[0], cx[1] + w * v[1], cx[2] + w * v[2])
    c2 = (y * t[2] - z * t[1], z * t[0] - x * t[2], x * t[1] - y * t[0])
    return (v[0] + 2 * c2[0], v[1] + 2 * c2[1], v[2] + 2 * c2[2])


def load_windows(prev_mtime):
    """-> (mtime, entries|None). None entries = unchanged since prev_mtime. A bad file keeps the old
    state (returns the new mtime with entries=None) rather than dropping every window mid-calibration."""
    try:
        m = os.path.getmtime(WINDOWS_FILE)
    except OSError:
        return (None, []) if prev_mtime is not None else (None, None)
    if m == prev_mtime:
        return m, None
    try:
        with open(WINDOWS_FILE) as fh:
            raw = json.load(fh).get("windows", [])
        out = []
        for e in raw:
            if not e.get("enabled", True):
                continue
            out.append({
                "match": dict(e.get("match", {})),
                "q": view_quat(float(e.get("yaw", 0.0)), float(e.get("pitch", 0.0)), float(e.get("roll", 0.0))),
                # zoom 2 = twice as magnified (the ray's uv scale is its inverse). "fov" = the old key (uv scale).
                "fov": 1.0 / max(0.05, float(e["zoom"])) if "zoom" in e else max(0.05, float(e.get("fov", 1.0))),
            })
        return m, out
    except (OSError, ValueError, TypeError, AttributeError) as ex:
        print(f"windows: {WINDOWS_FILE} unreadable ({ex}); keeping previous", file=sys.stderr)
        return m, None


def window_for(entries, idx, spec):
    for e in entries:
        mt = e["match"]
        if (mt.get("primary") and idx == 0) or mt.get("index") == idx or \
                (mt.get("name") and mt["name"] == spec["name"]) or \
                (mt.get("size") and mt["size"] == "%dx%d" % (spec["w"], spec["h"])):
            return e
    return None


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

# Canvas size is declared BY the shader, not derived from the outputs (see the
# CANVAS block below for why it is not read off xrandr). Same shape as the fps
# directive: multipass `"canvas": [W, H]` in manifest.json, single-pass a
# `// canvas: WxH` line anywhere in the first 40 lines. Omitted -> the square
# default, so nothing existing changes. Orientation is free: landscape,
# portrait or square are all fine, and the blit quarter-turns per output only
# when the output's orientation differs from the canvas's (crop_for_spec /
# BLIT_FRAG) -- so a square canvas never rotates on any head.
#
# Read ONCE at startup: canvas_tex / canvas_fbo and the multipass buffers are
# allocated from it below, so changing the declaration needs a renderer restart
# (the ctl `start`), like any other Python-side change. try_reload() prints a
# loud "restart required" if a hot-reloaded shader declares a different one.
_CANVAS_DIRECTIVE_RE = re.compile(
    r"^\s*//\s*canvas\s*:\s*(?P<w>\d+)\s*[xX]\s*(?P<h>\d+)\s*$"
)
DEFAULT_CANVAS_W = 2410
DEFAULT_CANVAS_H = 2410


def read_shader_canvas(path):
    """(w, h) the active shader declares, else the square default."""
    try:
        if is_multipass_path(path):
            with open(os.path.join(path, "manifest.json")) as f:
                value = json.load(f).get("canvas")
            if value and len(value) == 2:
                return max(1, int(value[0])), max(1, int(value[1]))
        else:
            with open(path) as f:
                for _ in range(40):
                    line = f.readline()
                    if not line:
                        break
                    m = _CANVAS_DIRECTIVE_RE.match(line)
                    if m:
                        return max(1, int(m.group("w"))), max(1, int(m.group("h")))
    except Exception as e:
        print(f"read_shader_canvas({path}): {e}; using default canvas", file=sys.stderr)
    return DEFAULT_CANVAS_W, DEFAULT_CANVAS_H


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


# Canvas = FIXED size, and deliberately NOT derived from the detected outputs
# -- see HANDOFF.md item 2 for the render/blit spec (one real render at this
# size, cheap centered-crop blit to each window, NOT three independent
# full-resolution renders). Default is the square 2410x2410 below; since
# 2026-10-15 the shader may declare its own instead (read_shader_canvas).
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
# "fills the frame" means the same thing on every screen. Still the default and
# still the safest choice for that reason. 2026-10-15: a shader may now declare
# a non-square (portrait) canvas instead -- the quarter-turn in crop_for_spec /
# BLIT_FRAG is what makes that safe again on landscape heads.
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
CANVAS_W, CANVAS_H = read_shader_canvas(SHADER_PATH)
# 2026-10-15 (user): a NON-SQUARE canvas is a PORTRAIT canvas, always -- the
# mouse/reference space is portrait, and aspect is the one portrait shape the
# shader declared. A landscape digit order is normalised to portrait here (and
# logged), so declaring `// canvas: 2410x1080` and `// canvas: 1080x2410` mean
# the same portrait canvas and the cursor can never end up in a landscape
# frame. A SQUARE canvas is left exactly as today: nothing rotates, every head
# fits by design. Landscape heads quarter-turn a portrait canvas (crop_for_spec
# / BLIT_FRAG); square never rotates anywhere, so the default is unchanged.
if CANVAS_W != CANVAS_H and CANVAS_W > CANVAS_H:
    print(f"canvas: declared {CANVAS_W}x{CANVAS_H} normalised to portrait "
          f"{CANVAS_H}x{CANVAS_W} (non-square canvas is always portrait)",
          file=sys.stderr)
    CANVAS_W, CANVAS_H = CANVAS_H, CANVAS_W


def canvas_portrait():
    """True for a non-square (portrait) canvas -- the case that quarter-turns
    landscape heads. A square canvas is False everywhere, so it never rotates.
    """
    return CANVAS_W != CANVAS_H


def canvas_per_screen(spec):
    """Canvas px per output px -- ONE uniform factor, a COVER scale: the canvas
    always fills this output, cropping the excess (never letterboxing).

    2026-09-27 (user): every output's LONG side spans the full canvas, the
    short side is a centred crop. Heads smaller than the canvas are downscaled
    (the blit supersamples), bigger ones upscaled. For the square default
    canvas the long-side rule below and cover-fit are the same thing, so this
    is unchanged for every existing shader. For a non-square canvas only cover
    is correct (long-side fit can under-fill and would stretch); the rotated
    case swaps which canvas axis lines up with the output's width.
    """
    if output_rotated(spec):
        return min(CANVAS_H / spec["w"], CANVAS_W / spec["h"])
    return min(CANVAS_W / spec["w"], CANVAS_H / spec["h"])


def output_rotated(spec):
    """True when this output's orientation is perpendicular to the canvas's, so
    the blit must quarter-turn it. Square canvas -> always False (canonical
    case, unchanged). Portrait canvas -> True only on landscape heads.
    """
    if not canvas_portrait():
        return False
    return spec["w"] > spec["h"]


def crop_for_spec(spec):
    """UV offset+scale (+ quarter-turn flag) for blitting this output's share of
    the shared canvas (see canvas_per_screen: long side = full canvas, uniform
    scale, centred). Used by presenter setup and sync_layout, which must agree.

    Third return value: rotate -- 1.0 when the blit must quarter-turn the canvas
    for this head (a portrait canvas on a landscape head). The axes that map to
    the output's w/h swap when rotated, which is what the span_u/span_v swap
    below expresses; BLIT_FRAG does the actual 90 deg turn of the sample coords.
    """
    k = canvas_per_screen(spec)
    rot = output_rotated(spec)
    span_u = spec["h"] * k if rot else spec["w"] * k
    span_v = spec["w"] * k if rot else spec["h"] * k
    uscale = min(1.0, span_u / CANVAS_W)
    vscale = min(1.0, span_v / CANVAS_H)
    return ((1.0 - uscale) / 2.0, (1.0 - vscale) / 2.0), (uscale, vscale), (1.0 if rot else 0.0)

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
uniform float uv_rot;   // 1.0 = quarter-turn the canvas for this head
uniform vec2 tap;       // a quarter output pixel, in canvas uv: 4-tap supersample
in vec2 v_uv;
out vec4 fragColor;
void main() {
    // 4 taps per output pixel: heads smaller than the canvas are DOWNscaled
    // (~0.8x for the Dells), and a single bilinear tap would skip texels, so
    // fine detail (star fields) would shimmer.
    // uv_rot: a portrait canvas on a landscape head -- rotate the sample coords
    // a quarter turn so the canvas long side lands on the output long side.
    // The 4-tap cross is symmetric, so it needs no rotation of its own.
    vec2 tuv = v_uv;
    if (uv_rot > 0.5) tuv = vec2(tuv.y, 1.0 - tuv.x);
    vec2 uv = uv_offset + tuv * uv_scale;
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
# Kick flatten weights, 2026-10-08 (scope.md "kick-only flattening",
# kick_detect.py): a 120x1 float texture `kickflat`, per band 0..1 = how much
# of that band is kick at the drawn frame (look-ahead classified). Own sampler
# like `stems`, so shaders that don't declare it are unaffected. NEAREST.
KICK_UNIT = 6
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
AUDIO_IDLE_S = 0.5           # no NEW audio frame for this long = nothing playing -> music.idle() (synthetic clock)


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
        # 2026-10-12 THE HAND (scope.md): the main loop puts the boop / reset counters here every frame, so a
        # sim.py reads them as api.inputs without a readback: boop (count), boop_xy (canvas px), reset (count)
        self.inputs = {"boop": 0, "boop_xy": (0.0, 0.0), "reset": 0}
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
        for name, unit in (("bands", MP_BANDS_UNIT), ("stems", STEMS_UNIT), ("kickflat", KICK_UNIT)):
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
            uset(kernel, name, val)

    def expose(self, uniform, tex):
        if uniform not in self._exposed:
            unit = COMPUTE_EXPOSE_UNIT_START + len(self._exposed)
            self._exposed[uniform] = [unit, tex]
            for p in self._render_progs:
                uset(p, uniform, unit)
        self._exposed[uniform][1] = tex
        tex.use(location=self._exposed[uniform][0])

    def render_uniform(self, name, value):
        """Set a uniform on the buffer/Image passes (e.g. a blend weight the sim decides)."""
        for p in self._render_progs:
            uset(p, name, value)

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
                uset(p, uniform, unit)

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
    try:
        prog["kickflat"].value = KICK_UNIT
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


def render_windows(ctx, mp, presenters, win_state, audio_channel_tex):
    """Re-run ONLY the Image pass for every head that has a windows.json entry (buffers and compute
    already ran once for the ship). Each window renders into its OWN canvas-sized texture with the very
    same iResolution as the front render, and the head then shows it through its normal crop-blit --
    so at identity a window is pixel-for-pixel the front view for ANY shader, whatever its uv
    convention (min(res), res.y ...). (First cut rendered at the head's native size: the dwarf's
    min(iResolution) scale then framed ~2.2x wider than the front. Don't.)
    Cheap when windows.json is absent: one failed stat per frame."""
    now = time.monotonic()                                  # LEAN (2026-10-19): windows.json 4x a second, not every frame
    if now - win_state.get("t", 0.0) >= 0.25:
        win_state["t"] = now
        m, entries = load_windows(win_state["m"])
        win_state["m"] = m
        if entries is not None:
            win_state["entries"] = entries
            print(f"windows: {len(entries)} window(s) active", file=sys.stderr)
    prog = mp["image_prog"]
    drew = False
    for idx, p in enumerate(presenters):
        e = (window_for(win_state["entries"], idx, p["spec"])
             if win_state["entries"] and not win_state.get("suppress") else None)
        v = p["view"]
        if e is None:
            if v is not None:
                v["fbo"].release()
                v["tex"].release()
                p["view"] = None
            continue
        if v is None:
            tex = ctx.texture((CANVAS_W, CANVAS_H), 4, dtype="f4")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            v = p["view"] = {"tex": tex, "fbo": ctx.framebuffer(color_attachments=[tex])}
        try:
            prog["u_view_q"].value = e["q"]
        except KeyError:
            pass
        try:
            prog["u_view_fov"].value = e["fov"]
        except KeyError:
            pass
        v["fbo"].use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        mp["image_vao"].render(moderngl.TRIANGLE_STRIP)
        drew = True
    if drew:
        # put the identity back so next frame's front render (which also re-sets it) starts clean
        try:
            prog["u_view_q"].value = VIEW_IDENTITY
        except KeyError:
            pass
        try:
            prog["u_view_fov"].value = 1.0
        except KeyError:
            pass


def create_output_window(spec, share=None):
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DECORATED, False)
    glfw.window_hint(glfw.FLOATING, True)
    # 2026-10-09: created HIDDEN. A visible glfw window is first mapped as a
    # normal window, which goes through i3 -- and while another app is
    # fullscreen (VLC, a browser video) i3 leaves new windows unmapped, so
    # glfw's wait for the window to become visible never ended: the renderer
    # hung at startup and the backdrop stayed black. make_backdrop() below
    # maps it itself, already override-redirect, past the window manager.
    glfw.window_hint(glfw.VISIBLE, False)

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
            uset(p, name, value)

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

    kick_tex = ctx.texture((N_BANDS, 1), 1, dtype="f4")
    kick_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    kick_tex.write(np.zeros(N_BANDS, dtype="f4").tobytes())
    kick_tex.use(location=KICK_UNIT)

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
        try:
            p["kickflat"].value = KICK_UNIT
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
    reload_check_t = [0.0]                       # hot-reload polling: at most every 0.2 s
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
        if new_pointer_mtime == pointer_mtime:
            return                   # LEAN (2026-10-19, user): NO hot-reload of edited shader files -- restart, or
                                     #  `touch active_shader.txt` (= reload the active shader), or switch shaders
        pointer_mtime = new_pointer_mtime
        new_path = resolve_active_shader_path()
        if True:
            if new_path != SHADER_PATH:
                SHADER_PATH = new_path
            shader_mtime = -1.0  # force the reload below (a switch, or a touch of the pointer = manual reload)

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
        # Canvas size is fixed at startup (canvas_tex / canvas_fbo and the
        # multipass buffers are allocated from it). A hot-reloaded shader that
        # declares a different one cannot take effect without a restart -- say
        # so loudly instead of silently blitting through the old canvas. The
        # set-compare matches read_shader_canvas's portrait normalisation, so a
        # landscape-digit declaration of the same portrait canvas is not a
        # false alarm.
        decl_canvas = read_shader_canvas(SHADER_PATH)
        if set(decl_canvas) != {CANVAS_W, CANVAS_H}:
            print(f"canvas: {os.path.basename(SHADER_PATH)} declares "
                  f"{decl_canvas[0]}x{decl_canvas[1]} but this renderer started "
                  f"with {CANVAS_W}x{CANVAS_H}; RESTART REQUIRED to apply "
                  f"(~/bin/shader-renderer-ctl.sh start)", file=sys.stderr)
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

        (u0, v0), (uscale, vscale), rot = crop_for_spec(spec)

        presenters.append({
            "window": window, "ctx": win_ctx, "prog": blit_prog, "vao": blit_vao,
            "uv_offset": (u0, v0), "uv_scale": (uscale, vscale), "uv_rot": rot,
            "spec": spec,
            # WINDOW view (windows.json): None = show the shared canvas as always. Otherwise
            # {"tex","fbo"}: this head's own canvas-sized Image-pass render (see render_windows).
            "view": None,
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
    #   (delayed_frame, stems_array(4,120) float32 or None, scalars dict,
    #    kick weights (120,) float32 or None)
    audio_state = [(None, None, {}, None)]
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
    # Shared music-state layer (2026-10-11, scope.md "song cache in shaders: ONE
    # live-capable interface"): u_ms_* uniforms -- section scores, changes,
    # look-ahead -- live for every song, refined by the offline song analysis
    # when the playing song has one. Shares beat_sess's NowPlaying.
    held = held_tone.HeldTone()
    music = music_state.MusicState(N_BANDS, np_source=beat_sess.np,
                                   delay_fn=read_audio_delay_seconds)
    replay = song_replay.Replay()        # 2026-10-20 silent song-replay test mode (song_replay.py)
    # Diagnostic capture (scope.md 2026-10-13 synthetic clock): when
    # SHADER_CAPTURE_JSONL is set, append one JSON line per drawn frame with the
    # scalars the fallback gate will read. Off by default; never affects render.
    capture_path = os.environ.get("SHADER_CAPTURE_JSONL")
    capture_fh = open(capture_path, "a", buffering=1) if capture_path else None

    # Per-band delay compensation (2026-10-08, scope.md "sharper band filters").
    # The native band filters are Butterworth bandpasses of graded order
    # ([filters] orders in shader_bands.conf); a sharper filter answers later,
    # by its group delay (up to ~0.28 s at A0). band_filters.py mirrors the
    # exact design, so each band is read that much LATER in the history --
    # every consumer (bands texture, stems, kick detector) sees the bands lined
    # up with what you hear. The look-ahead the delay holds back pays for it.
    # SHADER_BAND_DELAY_COMP=0 disables it.
    band_delay_s = (band_filters.delays_s()
                    if os.environ.get("SHADER_BAND_DELAY_COMP", "1") != "0"
                    else np.zeros(N_BANDS))
    print(f"band delay compensation: max {band_delay_s.max() * 1000:.0f} ms "
          f"(orders {band_filters.read_orders()})", file=sys.stderr)

    def comp_rows(snap, lo, hi, rate_hz):
        """Rows lo..hi-1 of the history, each band taken its own delay later."""
        sh = np.round(band_delay_s * rate_hz).astype(int)
        top = min(len(snap), hi + int(sh.max()))
        # rows carry a float64 copy of their bands made once on arrival ("_b"): stacking those
        # is ~5x cheaper than re-converting the python lists every call (2026-10-13: this line
        # was ~46% of the audio thread, which starved the render loop at 60 fps via the GIL)
        b = np.stack([r["_b"] if "_b" in r else np.asarray(r["bands"], dtype=np.float64)
                      for r in snap[lo:top]])
        t = np.minimum(np.arange(hi - lo)[:, None] + sh[None, :], len(b) - 1)
        out = b[t, np.arange(b.shape[1])[None, :]]
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def audio_reader_thread():
        last_ts = None
        last_split = 0.0
        last_frame_t = 0.0       # monotonic time the last NEW audio frame arrived
        rate_hz = 188.0          # refined from real timestamps below
        win = stem_split.window_frames(rate_hz)
        while True:
            try:
                frame = reader.read()
                if frame is not None and frame["timestamp"] != last_ts:
                    last_ts = frame["timestamp"]
                    frame["_b"] = np.asarray(frame["bands"], dtype=np.float64)
                    band_history.append(frame)
                    last_frame_t = time.monotonic()

                now = time.monotonic()
                # 2026-10-20 SONG REPLAY test mode (song_replay.py, ~/bin/backdrop-replay): while the flag file names a
                # computed song, its analysis drives the backdrop (beats, sections, drum hits, loudness) and the live
                # audio is ignored -- tune shaders to real music while the speakers play a video.
                if replay.active(now):
                    if now - last_split >= STEM_UPDATE_PERIOD:
                        last_split = now
                        try:
                            rframe, ru = replay.tick(now, META_FEATURES)
                            prev = audio_state[0]
                            scalars = dict(prev[2])
                            scalars.update(ru)
                            audio_state[0] = (rframe, np.zeros((STEM_ROWS, N_BANDS), dtype="f4"), scalars,
                                              np.zeros(N_BANDS, dtype="f4"))
                        except Exception as e:      # never take the audio thread down
                            print(f"song_replay: {e}", file=sys.stderr)
                    time.sleep(0.0008)
                    continue
                # No fresh audio (nothing playing / sink idle): the block below would
                # never run (empty history) or keep re-feeding the last frame, freezing
                # the backdrop. Let the synthetic clock drive it instead (scope.md
                # 2026-10-13 synthetic clock); real frames take over again as they arrive.
                if now - last_split >= STEM_UPDATE_PERIOD and (
                        not band_history or now - last_frame_t > AUDIO_IDLE_S):
                    last_split = now
                    try:
                        synth = music.idle(now)
                    except Exception as e:      # never take the audio thread down
                        synth = None
                        print(f"music_state idle: {e}", file=sys.stderr)
                    if synth is not None:
                        prev = audio_state[0]
                        zeros = np.zeros((STEM_ROWS, N_BANDS), dtype="f4")
                        scalars = dict(prev[2])
                        scalars.update({"u_stem_drums": 0.0, "u_stem_bass": 0.0, "u_stem_tonal": 0.0})
                        scalars.update(synth)
                        audio_state[0] = (prev[0], zeros, scalars, np.zeros(N_BANDS, dtype="f4"))
                    time.sleep(0.0008)
                    continue
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
                            rows = comp_rows(snap, lo, hi, rate_hz)
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
                            try:                        # clear long-held notes (held_tone.py)
                                scalars.update(held.update(out["tonal"]))
                            except Exception as e:
                                print(f"held_tone: {e}", file=sys.stderr)
                            # Fed the DRUMS row, at this same 60 Hz cadence.
                            # snap[idx]["timestamp"] is the delay-aligned audio
                            # clock (what the shader is actually showing);
                            # monotonic() advances the beat count, so a stalled
                            # audio stream cannot freeze the orrery.
                            # Look-ahead window for the beat tracker's DP phase
                            # (beat_track.py LOOKAHEAD_*): the low-band onset
                            # function of the delay-compensated bands, past AND
                            # future around the drawn frame -- one source for
                            # both sides, so there is no seam.
                            lp = int(beat_track.LOOKAHEAD_PAST_S * rate_hz)
                            lf = int(beat_track.LOOKAHEAD_FUTURE_S * rate_hz)
                            blo = max(0, idx - lp)
                            bhi = min(len(snap), idx + lf + 1)
                            bodf = kick_detect._flux(comp_rows(snap, blo, bhi, rate_hz))
                            scalars.update(beat_tracker.update(
                                out["drums"], snap[idx]["timestamp"],
                                time.monotonic(), (bodf, idx - blo, rate_hz)))
                            # Sets the tracker's reference for the NEXT sample
                            # when the song is recognised, and records this one
                            # when it is not.
                            beat_sess.update()
                            # Kick flatten weights for the drawn frame, from
                            # the history around it -- the frames after idx
                            # are the look-ahead (kick_detect.py needs ~0.85 s,
                            # the speaker delay is >= ~1 s).
                            kb = kick_detect.history_frames(rate_hz)
                            klo = max(0, idx - kb[0])
                            khi = min(len(snap), idx + kb[1] + 1)
                            krows = comp_rows(snap, klo, khi, rate_hz)
                            np.nan_to_num(krows, copy=False, nan=0.0,
                                          posinf=0.0, neginf=0.0)
                            kick = kick_detect.weights_at(
                                krows, idx - klo, rate_hz, N_BANDS).astype("f4")
                            scalars["u_kick"] = kick_detect.kick_weight(kick)
                            scalars["u_drum_mid"] = kick_detect.range_weight(kick, "mid")    # snare body, toms
                            scalars["u_drum_high"] = kick_detect.range_weight(kick, "high")  # snare rattle, hats
                            drawn = dict(snap[idx])
                            drawn["bands"] = comp_rows(snap, idx, idx + 1, rate_hz)[0].tolist()
                            try:
                                scalars.update(music.update(drawn, out["tonal"], scalars,
                                                            time.monotonic()))
                            except Exception as e:      # never take the audio thread down
                                print(f"music_state: {e}", file=sys.stderr)
                            audio_state[0] = (drawn, tex, scalars, kick)
                            if capture_fh is not None:
                                try:
                                    capture_fh.write(json.dumps({
                                        "t": round(time.monotonic(), 3),
                                        "bt": round(float(scalars.get("u_beat_count", -1.0)), 3),
                                        "conf": round(float(scalars.get("u_beat_conf", -1.0)), 4),
                                        "bpm": round(float(scalars.get("u_beat_bpm", -1.0)), 2),
                                        "kick": round(float(scalars.get("u_kick", 0.0)), 4),
                                        "dmid": round(float(scalars.get("u_drum_mid", 0.0)), 4),
                                        "dhi": round(float(scalars.get("u_drum_high", 0.0)), 4),
                                        "rms": round(float(drawn.get("rms", 0.0)), 4),
                                        "onset": round(float(drawn.get("onset", 0.0)), 4),
                                        "flux": round(float(drawn.get("flux", 0.0)), 4),
                                        "ms_src": scalars.get("u_ms_source", -1.0),
                                        "ms_score": round(float(getattr(music, "music_score", -1.0)), 4),
                                        "mode": getattr(music, "mode", "?"),
                                        "cur_a": [round(float(x), 4) for x in scalars.get("u_ms_cur_a", ())],
                                        "cur_b": [round(float(x), 4) for x in scalars.get("u_ms_cur_b", ())],
                                        "ms_change": scalars.get("u_ms_change", -1.0),
                                    }) + "\n")
                                except Exception as e:
                                    print(f"capture: {e}", file=sys.stderr)
            except Exception as e:
                # A crash in here must never take the backdrop down -- the
                # render loop falls back to its previous values.
                print(f"audio_reader_thread: {e}", file=sys.stderr)
                time.sleep(0.05)
            time.sleep(0.0008)

    threading.Thread(target=audio_reader_thread, daemon=True,
                     name="audio-reader").start()

    # ---- X POLLER (2026-10-20, perfseg + py-spy: the X11 round trips query_pointer / query_keymap took a MEDIAN 4-5 ms and
    # a p90 16-18 ms each -- Xorg is busy feeding three Sunshine streams -- and the frame loop waited for both, every
    # frame: that WAS the ~28 ms "inputs" stage). Now a daemon thread with its OWN X connection asks ~60x a second; the
    # frame loop only reads the latest answer (never waits). Each call site is inside a try: no answer yet = its old
    # error path.
    xpoll = {"ptr": None, "km": None}

    def x_poller():
        try:
            from Xlib import display as _xdisplay
            xd = _xdisplay.Display(primary_xdisp.get_display_name())
            xroot = xd.screen().root
        except Exception as e:
            print(f"x poller: no own X connection ({e}) -- mouse/keys unavailable", file=sys.stderr)
            return
        while True:
            try:
                xpoll["ptr"] = xroot.query_pointer()
                xpoll["km"] = xd.query_keymap()
            except Exception:
                time.sleep(0.5)
            time.sleep(1.0 / 60.0)

    threading.Thread(target=x_poller, name="x_poller", daemon=True).start()

    def _xptr():
        p = xpoll["ptr"]
        if p is None:
            raise RuntimeError("x poller: no pointer yet")
        return p

    def _xkm():
        k = xpoll["km"]
        if k is None:
            raise RuntimeError("x poller: no keymap yet")
        return k

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
            p = _xptr()
            mx = float((p.root_x - primary_spec["x"]) * k + crop_x)
            my = float(CANVAS_H - ((p.root_y - primary_spec["y"]) * k + crop_y))
            return (mx, my, 0.0, 0.0)
        except Exception:
            return (0.0, 0.0, 0.0, 0.0)

    def read_pointer_canvas():
        # 2026-10-12 THE HAND: the cursor in canvas px on WHICHEVER output it is on (read_mouse_real assumes the
        # primary output, so a boop on the landscape head landed off-canvas at x ~3000+). Same crop math per output.
        try:
            p = _xptr()
            px, py = p.root_x, p.root_y
        except Exception:
            return (0.0, 0.0)
        spec = OUTPUTS[0]
        for sp in OUTPUTS:
            if sp["x"] <= px < sp["x"] + sp["w"] and sp["y"] <= py < sp["y"] + sp["h"]:
                spec = sp
                break
        k = canvas_per_screen(spec)
        crop_x = (CANVAS_W - spec["w"] * k) / 2.0
        crop_y = (CANVAS_H - spec["h"] * k) / 2.0
        return (float((px - spec["x"]) * k + crop_x), float(CANVAS_H - ((py - spec["y"]) * k + crop_y)))

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
            km = _xkm()
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
    alt_keycode   = resolve_keycode("Alt_L")      # 2026-10-12 THE HAND: Super + Alt tap = boop

    def read_key(code):
        if not code:
            return 0.0
        try:
            km = _xkm()
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
    hand = {"alt": False, "rst": False, "boop": 0, "reset": 0, "xy": (0.0, 0.0)}
    gyro = {"m": -1.0, "active": False, "xy": (0.0, 0.0, 0.0), "prev": None}
    persist_last = {"t": 0.0, "dir": None, "saved": None}
    autozoom = {"m": -1.0, "v": 0.0}
    cam_target = {"m": -1.0, "t": (0.0, 0.0, 0.0, 0.0), "id": 0.0}
    autodive = {"m": -1.0, "on": False, "beats": 0.0, "last": None, "id": None}
    # DroidPad control surface (see PAD_FILE). The *_prev dicts hold last frame's
    # counters so an impulse is the exact delta (never a missed press); sep is a
    # latched separation nudge the dpad moves and that stays put.
    #
    # cam / boop (2026-10-15, the brown-dwarf remote): the pad's own free-fly, kept
    # SEPARATE from mouse_cam. mouse_cam is a hand-dialled framing shared by every
    # flying shader (and persisted to freefly_state.txt), so a stick that wrote into
    # it would silently shove string_world/nebulabrot/the flyers around. cam holds
    # the same six offsets, but only the ctrl_ variant of a shader reads them.
    # boop is the reticle the third stick places: (canvas px x, canvas px y, active),
    # the SAME convention as hand["xy"] so sim.py treats it with identical code.
    pad = {"m": -1.0, "doc": None, "alive": 0.0,
           "prev_trigger": {}, "prev_dpad": {}, "sep": 0.0, "gate": 0.0,
           "cam": {k: 0.0 for k in FREEFLY_KEYS}, "cq": VIEW_IDENTITY, "mv": [0.0, 0.0, 0.0], "boop": (0.0, 0.0, 0.0),
           "shocks": (0.0, 0.0), "fire": (0.0, 0.0, 0.0), "speed": 1.0, "synced": False,
           "wheel": 0.0, "ids": None, "frenzy": 0.5, "pocket": 0.0,
           "peri_q0": None, "peri_cq0": VIEW_IDENTITY}   # PERISCOPE: the phone's / the camera's pose at engage
    # the Xbox controller (GAMEPAD_FILE): mtime cache + last press counters (taps are edge-diffed per frame)
    xb = {"m": -1.0, "doc": None, "alive": 0.0, "prev_press": None, "map": XB_LAYER_MAP, "map_m": None}

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
        # changes when a keybinding fires. LEAN (2026-10-19): stat at most 4x a second.
        now = time.monotonic()
        if now - camera_cache.get("t", 0.0) < 0.25:
            return camera_cache["value"]
        camera_cache["t"] = now
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

    def read_pad():
        """Poll pad.json (mtime-cached) -> the DroidPad state, or None if stale.

        Stale or absent means the phone is not connected: u_pad_alive goes 0 and
        shaders fall back to their own motion instead of holding the last command
        forever. See PAD_FILE's comment for the whole contract."""
        try:
            mtime = os.path.getmtime(PAD_FILE)
        except OSError:
            pad["alive"] = 0.0
            return None
        if time.time() - mtime > PAD_STALE_S:
            pad["alive"] = 0.0
            return None
        pad["alive"] = 1.0
        if mtime != pad["m"]:
            pad["m"] = mtime
            try:
                with open(PAD_FILE) as fh:
                    pad["doc"] = json.load(fh)
            except (OSError, ValueError):
                pass
        return pad["doc"]

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
    win_state = {"m": None, "entries": []}
    perf_last = time.monotonic()
    perf_frames = 0
    perf_frame_ms_sum = 0.0
    perf_render_ms_sum = 0.0
    perf_min_ms = 1e9
    perf_max_ms = 0.0
    seg = {}                                  # PERFSEG (2026-10-20): per-stage ms sums, printed with every perf line
    def seg_add(k, v):
        seg[k] = seg.get(k, 0.0) + v
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
                off, scale, rot = crop_for(spec)
                presenters[i]["uv_offset"], presenters[i]["uv_scale"] = off, scale
                presenters[i]["uv_rot"] = rot
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
            if t2 - reload_check_t[0] >= 0.25:      # (LEAN: one stat of active_shader.txt, 4x a second)
                reload_check_t[0] = t2
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
            # 2026-10-12 THE HAND (scope.md "THE HAND"): with the steer key held, a TAP of Alt_L = one BOOP at the
            # cursor (a single poke, user: "hold is aggressive, let's rather boop"), Ctrl_L + Shift_L together = RESET.
            # Both are counters (+1 per press), so a shader or sim.py reacts when the value changes and never
            # misses a one-frame press. u_hand = (boop x, boop y in canvas px, boop count, reset count).
            alt_on = steering and read_key(alt_keycode) > 0.5
            ctl_on = read_key(ctrl_keycode) > 0.5 if steering else False
            shf_on = read_key(shift_keycode) > 0.5 if steering else False
            reset_on = steering and ctl_on and shf_on
            if alt_on and not hand["alt"]:
                hand["boop"] += 1
                hand["xy"] = read_pointer_canvas()
            if reset_on and not hand["rst"]:
                hand["reset"] += 1
            hand["alt"], hand["rst"] = alt_on, reset_on
            set_uniform("u_hand", (hand["xy"][0], hand["xy"][1], float(hand["boop"]), float(hand["reset"])))
            # first steered frame only records the position (prev was the audio mouse)
            if steering and prev_mouse is not None and prev_steering and not alt_on and not reset_on:
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
                    dyaw   = (gyro["xy"][0] - gyro["prev"][0]) * MOUSE_LOOK_SENS   # rad
                    dpitch = (gyro["xy"][1] - gyro["prev"][1]) * MOUSE_LOOK_SENS   # rad
                    dturn  = (gyro["xy"][2] - gyro["prev"][2]) * MOUSE_MOVE_SENS   # world units
                    mouse_cam["yaw"]   += dyaw
                    mouse_cam["pitch"] += dpitch
                    # twist about the pointing axis = Super + Ctrl vertical drag
                    # (nebulabrot: zoom; free-fly shaders: forward/back)
                    mouse_cam["f"]     += dturn
                    # ---- the SAME motion into the free camera (2026-10-16) ----------------
                    # ctrl_brown_dwarf reads u_pad_cam_q / u_pad_move and none of the u_cam_* it
                    # stopped reading, so the Euler camera above is the legacy one. The phone drives
                    # BOTH, exactly as the pad's Look stick already does (see the pad block): its
                    # pointing turns the free camera about the camera's OWN axes (no clamps, and it
                    # composes with a stick that is doing the same thing), and its twist walks the
                    # camera along the axis it is already facing -- Move's own "go where you look".
                    # Same angles as the legacy path, so the gains the phone was tuned with hold.
                    if pad["alive"] > 0.5:
                        if abs(dyaw) > 1e-9 or abs(dpitch) > 1e-9:
                            dq = view_quat(math.degrees(dyaw), math.degrees(dpitch), 0.0)
                            q = quat_mul(pad["cq"], dq)
                            n = math.sqrt(sum(c * c for c in q)) or 1.0
                            pad["cq"] = tuple(c / n for c in q)
                        if abs(dturn) > 1e-9:
                            fwd = quat_rot(pad["cq"], (0.0, 0.0, dturn))
                            for i in range(3):
                                pad["mv"][i] += fwd[i]
                gyro["prev"] = gyro["xy"] if gyro["active"] else None
                super_held = 1.0          # the free-fly pose gets saved like a Super steer
                set_uniform("u_steer", 1.0)   # shaders gating on it see a steer
            else:
                gyro["prev"] = None
            # ---- DroidPad control surface (see PAD_FILE) ----
            # Publishes forces / bias / pressed-edges ONLY. It never sets a
            # position, angle, colour or cue: each shader decides how hard to lean
            # on these, so the sim keeps its own movementset. u_pad_alive 0 means
            # the phone is gone -- use your own motion.
            pdoc = read_pad()
            pwidgets = (pdoc or {}).get("widgets", {})
            # "3D POV" pad switch: explicitly OFF = suppress every ship window (heads show the
            # plain shared canvas again). Absent widget / no pad = windows as configured.
            _pov = pwidgets.get("3D POV")
            win_state["suppress"] = _pov is not None and not _pov.get("state")
            pderiv = (pdoc or {}).get("derived", {})
            ptrigger = pderiv.get("trigger", {})
            pheld = pderiv.get("held", {})
            pdpad = pderiv.get("dpad", {})
            pnow = time.time()      # one clock for every spring-return TTL test this frame
            # Pressed EDGES: the bridge counts every MQTT event, we diff the counters per frame.
            # Two different things hide in "a widget I have not seen before", and the difference
            # is exactly what makes a button's first press work:
            #   * the first fresh read of the WHOLE surface -- the phone may have been in use
            #     long before this renderer started, so ADOPT its counters as history, or a
            #     restart would replay the session so far as a burst of phantom presses;
            #   * a widget that shows up LATER -- that can only be a brand-new press (the app
            #     starts every widget at zero and the bridge writes as each event arrives), so
            #     count it from zero. Defaulting a missing widget to its own count is what made
            #     the first press of a button vanish on a pad profile the renderer had not
            #     seen yet.
            # And if a counter goes BACKWARDS, the bridge restarted (they live in that process):
            # the whole new count is real presses, and treating the dip as negative would eat
            # the one press that caused it.
            if not pad["synced"] and pdoc is not None:
                pad["synced"] = True
                pad["prev_trigger"] = dict(ptrigger)
                pad["prev_dpad"] = dict(pdpad)
            impulses = {}
            for wid, n in ptrigger.items():
                prev = pad["prev_trigger"].get(wid, 0)
                d = n if n < prev else n - prev
                if d > 0:
                    impulses[wid] = float(d)
            pad["prev_trigger"] = dict(ptrigger)
            # PAD ALIASES (2026-10-22): the user's "starbox" pad names its widgets by what they do; map them onto the
            # ids the shaders read. A switch/slider: the newer of the two wins. A press: both count.
            if any(src in pwidgets or src in impulses for src in PAD_ALIASES):
                pwidgets = dict(pwidgets)
                for src, dst in PAD_ALIASES.items():
                    if src in pwidgets:
                        cur = pwidgets.get(dst)
                        if cur is None or pwidgets[src].get("t", 0.0) >= cur.get("t", 0.0):
                            pwidgets[dst] = pwidgets[src]
                    if src in impulses:
                        impulses[dst] = impulses.get(dst, 0.0) + impulses[src]
            # Every button/trigger press is announced once: `u_pad_fly` / `u_pad_down` / `u_pad_fire`
            # go to every shader, so a shader that reads none of them and a button whose id is
            # mistyped look identical from here -- this line is what tells the two apart.
            if impulses:
                print("pad press: %s" % ", ".join(sorted(impulses)), file=sys.stderr)
            dclicks = {}
            for d, n in pdpad.items():
                prev = pad["prev_dpad"].get(d, 0)
                dclicks[d] = n if n < prev else n - prev
            pad["prev_dpad"] = dict(pdpad)
            # 2026-10-23 (ctrl_ringbox_v2 CAM STEERING): raw JOYSTICK deflections by widget id + this frame's dpad
            # clicks, handed to a sim as-is (no shared camera, no alias) -- the sim owns its own free camera.
            pad["joy"] = {k: pad_xy(pwidgets, k, pnow) for k, v in pwidgets.items() if (v or {}).get("type") == "JOYSTICK"}
            pad["dclicks"] = dict(dclicks)
            # dpad nudges the figures' distance to one another: each click a step,
            # and it stays where you nudged it
            pad["sep"] = max(-1.0, min(1.0, pad["sep"] + (
                dclicks.get("RIGHT", 0) - dclicks.get("LEFT", 0)) * PAD_SEP_STEP))
            set_uniform("u_pad_alive", pad["alive"])
            set_uniform("u_pad_tilt", pad_xy(pwidgets, "Tilt", pnow))
            set_uniform("u_pad_depth", pad_xy(pwidgets, "Depth", pnow))
            set_uniform("u_pad_hue", (pad_val(pwidgets, "Hue1"), pad_val(pwidgets, "Hue2")))
            # Rose-pocket override: the dial's REST (0) is NO override -- the shader falls back to the
            # music's own density drive. Scale is /10, not /5: step sliders arrive RAW on the app's
            # 0..10 scale (pads/README.md), so /5 saturated from half travel on.
            pad["pocket"] = max(0.0, min(1.0, pad_val(pwidgets, "Pocket") / 10.0))
            # The `Frenzy` 3-stop dial (raw 0 / 5 / 10): 0 = off, 0.5 = auto (the music gate),
            # 1 = on. An ABSENT dial publishes 0.5, not 0: a pad without the widget must behave
            # like the music halves, and 0 would silently kill the frenzy for every shader that
            # reads this.
            pad["frenzy"] = (max(0.0, min(1.0, pad_val(pwidgets, "Frenzy") / 10.0))
                             if "Frenzy" in pwidgets else 0.5)
            set_uniform("u_pad_pocket", pad["pocket"])
            set_uniform("u_pad_frenzy", pad["frenzy"])
            # `Auto Moves` RESTS AT AUTO, the same rule as the `Frenzy` dial's absent 0.5: an absent
            # switch must leave this half behaving like the pad-free one (the music's own,
            # phrase-paced specials), never like the free-for-all -- that is the LEAST INVASIVE
            # setting, and it is where a freshly reloaded pad rests, because a reload publishes
            # nothing until a widget is touched. Existence, not value, decides: only an explicit OFF
            # turns the busy mode on, and a pad that is gone cannot leave it latched.
            auto_moves = (pad_sw(pwidgets, "Auto Moves") if "Auto Moves" in pwidgets else 1.0)
            set_uniform("u_pad_sw", (pad_sw(pwidgets, "Half Frenzy"), auto_moves))
            set_uniform("u_pad_sep", pad["sep"])
            # ---- PAD FLIGHT: the Look / Move sticks as hold-to-steer camera -----------------
            # (user 2026-10-15: "maybe we can actually add 3d camera flight and improve the
            # kind of booping") Integrated HERE, once, and published twice -- as u_pad_look /
            # u_pad_move for the image pass AND into the compute inputs below -- because
            # brown_dwarf mirrors its camera in sim.py to turn a cursor into a world ray. A
            # camera that lives in two places must not be integrated in two places, or the
            # boop lands wherever the shader's copy of the camera happens to be.
            cam = pad["cam"]
            clx, cly = pad_xy(pwidgets, "Look", pnow)
            cmx, cmy = pad_xy(pwidgets, "Move", pnow)
            if pad["alive"] > 0.5:
                clx = 0.0 if abs(clx) < PAD_CAM_DEADZONE else clx
                cly = 0.0 if abs(cly) < PAD_CAM_DEADZONE else cly
                cmx = 0.0 if abs(cmx) < PAD_CAM_DEADZONE else cmx
                cmy = 0.0 if abs(cmy) < PAD_CAM_DEADZONE else cmy
                cam["yaw"]   += clx * PAD_CAM_LOOK_RATE * dt
                cam["pitch"] += cly * PAD_CAM_LOOK_RATE * dt * PAD_STICK_Y_UP
                cam["r"]     += cmx * PAD_CAM_MOVE_RATE * dt
                cam["f"]     += cmy * PAD_CAM_MOVE_RATE * dt * PAD_STICK_Y_UP
                # ---- ... AND A THIRD HAND ON THE SHARED CAMERA (the GENERAL REMOTE, 2026-10-15) --
                # (user) "one general kinda remote that does the simple lookaround/skybox rotation/
                # orbit camera kinda movement for all the shaders doing the mod+ctrl or mod+shift
                # with mouse stuff". The look-around family already reads ONE camera -- u_cam_look /
                # u_cam_move -- which the hand dials with Super+cursor and the phone gyro drags; so
                # the general remote is not a feature per shader, it is THIS block. The same two
                # sticks, same rates and same signs as the pad's own free-fly above, are added into
                # that same accumulator, so a stick drags the shared camera exactly the way a
                # cursor does: 22 shaders read u_cam_look, 10 read u_cam_move and 30 gate on
                # u_steer, and not one of them learns the phone exists. Shaders with their own bespoke
                # pad camera (brown dwarf, via u_pad_look) read no shared channel, so nothing here
                # doubles up on them. Idle sticks add nothing and raise no flag -- which is what keeps
                # an untouched pad, and the whole AUTO half it is paired with, untouched as authored.
                if clx or cly or cmx or cmy:
                    pdt = min(dt, PAD_STEER_DT_MAX)
                    mouse_cam["yaw"]   += clx * PAD_CAM_LOOK_RATE * pdt
                    mouse_cam["pitch"] += cly * PAD_CAM_LOOK_RATE * pdt * PAD_STICK_Y_UP
                    mouse_cam["r"]     += cmx * PAD_CAM_MOVE_RATE * pdt
                    mouse_cam["f"]     += cmy * PAD_CAM_MOVE_RATE * pdt * PAD_STICK_Y_UP
                    # "A hand is steering", said exactly the way the gyro path says it: shaders that
                    # gate their camera on u_steer apply the offsets they were just handed, and the
                    # pose is saved like a Super steer rather than only living in this process.
                    super_held = 1.0
                    set_uniform("u_steer", 1.0)
            # ---- FREE CAMERA (2026-10-16, ctrl_brown_dwarf): one quaternion about the camera's OWN
            # axes (no clamps, no Euler) + a position offset in the AUTHORED frame. Move's stick
            # vector is rotated by the quaternion, so forward goes where you are LOOKING. Holds still
            # when the sticks rest. The "Recenter" button returns the authored shot.
            recenter = impulses.get("Recenter", 0.0) > 0.5
            if recenter:
                pad["cq"], pad["mv"] = VIEW_IDENTITY, [0.0, 0.0, 0.0]
            # PERISCOPE (see PERISCOPE_FILE): switch ON + a fresh phone sample -> the camera IS the phone (relative to the
            # pose at engage, so switching on never jumps; Recenter re-zeros: the authored shot = where the phone points now).
            # The Look stick is superseded while it runs; Move still flies where you look.
            # The remote's switch IS the on/off (user): ON spawns ~/bin/periscope-stream --once (the phone's sensors run
            # only while it lives), OFF ends it -- it exits by itself within ~0.25 s, terminate() is the backstop.
            peri_sw = bool(pwidgets.get("Periscope", {}).get("state"))
            pp = pad.get("peri_proc")
            if peri_sw and (pp is None or pp.poll() is not None) and time.time() - pad.get("peri_t", 0.0) > 2.0:
                pad["peri_t"] = time.time()
                try:
                    with open(os.path.expanduser("~/.cache/shader-backdrop/periscope.log"), "a") as lf:
                        pad["peri_proc"] = subprocess.Popen([os.path.expanduser("~/bin/periscope-stream"), "--once"],
                                                            stdout=lf, stderr=lf, start_new_session=True)
                except OSError as e:
                    print(f"periscope: spawn failed: {e}", file=sys.stderr)
            if peri_sw:
                pad["peri_off_t"] = None
            elif pp is not None:
                if pad.get("peri_off_t") is None:
                    pad["peri_off_t"] = time.time()
                if pp.poll() is None and time.time() - pad["peri_off_t"] > 2.0:
                    pp.terminate()               # it should have exited by itself; backstop (it releases the phone on TERM)
                if pp.poll() is not None:
                    pad["peri_proc"], pad["peri_off_t"] = None, None
            peri_q = read_periscope() if peri_sw else None
            if peri_q is None:
                pad["peri_q0"] = None
            else:
                if pad["peri_q0"] is None or recenter:
                    pad["peri_q0"], pad["peri_cq0"] = peri_q, pad["cq"]
                q = quat_mul(pad["peri_cq0"], periscope_rel_cam(pad["peri_q0"], peri_q))
                n = math.sqrt(sum(c * c for c in q)) or 1.0
                pad["cq"] = tuple(c / n for c in q)
                clx = cly = 0.0
            if pad["alive"] > 0.5 and (clx or cly or cmx or cmy):
                pdt = min(dt, PAD_STEER_DT_MAX)
                dq = view_quat(math.degrees(clx * PAD_CAM_LOOK_RATE * pdt),
                               math.degrees(cly * PAD_CAM_LOOK_RATE * pdt * PAD_STICK_Y_UP), 0.0)
                q = quat_mul(pad["cq"], dq)
                n = math.sqrt(sum(c * c for c in q)) or 1.0
                pad["cq"] = tuple(c / n for c in q)
                mvv = quat_rot(pad["cq"], (cmx, 0.0, cmy * PAD_STICK_Y_UP))
                for i in range(3):
                    pad["mv"][i] += mvv[i] * PAD_CAM_MOVE_RATE * pdt
            # ---- THE MOUSE MODES (2026-10-19, the dwarf_box remote; user: "a toggle to switch desktop mouse into
            # different modes bc I'm away from the keyboard and have just my phone"). The pad's "Mouse" dial
            # (STEP_SLIDER 0..5) gives the PLAIN mouse -- the Cybersyn trackpad's real clicks -- the hand's jobs:
            #   0 OFF    a normal desktop mouse (also: no dial on any pad)
            #   1 BOOP   a click = one boop at the cursor (exactly Super + Alt tap: every hand shader gets it)
            #   2 GRAB   the held button catches + flings (sims read inputs["mouse"]: dwarf_box)
            #   3 LOOK   drag = turn the free camera (pad quaternion + the shared look-around camera)
            #   4 FLY    drag = fly: up/down along the view, sideways strafes
            #   5 METEOR a click = a meteor aimed at the cursor (sims read inputs["mouse"]: dwarf_box)
            mm = max(0, min(5, int(round(pad_val(pwidgets, "Mouse"))))) if "Mouse" in pwidgets else 0
            # (2026-10-19 py-spy: X round trips were 14 % of the main thread -- no query at all while the dial is OFF, and ONE
            # query for both the button and the position while it is on)
            mbtn, mxy = False, pad.get("m_prev") or (0.0, 0.0)
            if mm:
                try:
                    qp = _xptr()
                    mbtn = bool(qp.mask & 0x100)                                  # Button1Mask
                    spec = OUTPUTS[0]
                    for sp in OUTPUTS:
                        if sp["x"] <= qp.root_x < sp["x"] + sp["w"] and sp["y"] <= qp.root_y < sp["y"] + sp["h"]:
                            spec = sp
                            break
                    # the pointer -> canvas px through EXACTLY the blit's mapping (crop_for_spec + BLIT_FRAG): a landscape
                    # head shows the portrait canvas QUARTER-TURNED, which the old straight crop ignored -- the mouse modes
                    # missed the scene there (2026-10-19, user: "it doesn't seem to work on the landscape one")
                    (ou, ov), (su, sv), rot = crop_for_spec(spec)
                    vx = (qp.root_x - spec["x"]) / max(spec["w"], 1)
                    vy = 1.0 - (qp.root_y - spec["y"]) / max(spec["h"], 1)
                    tx, ty = (vy, 1.0 - vx) if rot > 0.5 else (vx, vy)
                    mxy = (float((ou + tx * su) * CANVAS_W), float((ov + ty * sv) * CANVAS_H))
                except Exception:
                    pass
            mpress = mbtn and not pad.get("m_btn", False)
            if mm == 1 and mpress:
                hand["boop"] += 1
                hand["xy"] = mxy
            if mm == 5 and mpress:
                pad["m_met"] = pad.get("m_met", 0) + 1
                pad["m_met_xy"] = mxy
            if mm in (3, 4) and mbtn and pad.get("m_btn") and pad.get("m_prev") is not None:
                ddx, ddy = mxy[0] - pad["m_prev"][0], mxy[1] - pad["m_prev"][1]
                if mm == 3:
                    dq = view_quat(math.degrees(ddx * MOUSE_LOOK_SENS), math.degrees(ddy * MOUSE_LOOK_SENS), 0.0)
                    q = quat_mul(pad["cq"], dq)
                    n = math.sqrt(sum(c * c for c in q)) or 1.0
                    pad["cq"] = tuple(c / n for c in q)
                    mouse_cam["yaw"] += ddx * MOUSE_LOOK_SENS
                    mouse_cam["pitch"] += ddy * MOUSE_LOOK_SENS
                else:
                    mvv = quat_rot(pad["cq"], (ddx * MOUSE_MOVE_SENS, 0.0, ddy * MOUSE_MOVE_SENS))
                    for i in range(3):
                        pad["mv"][i] += mvv[i]
                    mouse_cam["r"] += ddx * MOUSE_MOVE_SENS
                    mouse_cam["f"] += ddy * MOUSE_MOVE_SENS
                super_held = 1.0
                set_uniform("u_steer", 1.0)
            if mm != pad.get("m_mode"):
                print("mouse mode: %s" % ("off", "boop", "grab", "look", "fly", "meteor")[mm], file=sys.stderr)
                pad["m_mode"] = mm
            pad["m_btn"], pad["m_prev"] = mbtn, mxy
            mx2 = pad.get("m_met_xy", (0.0, 0.0))
            pad["mouse"] = (float(mm), float(mxy[0]), float(mxy[1]), 1.0 if mbtn else 0.0,
                            float(pad.get("m_met", 0)), float(mx2[0]), float(mx2[1]))
            set_uniform("u_pad_look", (cam["yaw"], cam["pitch"], cam["roll"]))
            set_uniform("u_pad_move", tuple(pad["mv"]))
            set_uniform("u_pad_cam_q", pad["cq"])
            # ---- PAD CROSSHAIR: the third stick ("boop") ------------------------------------
            # The stick PLACES a reticle on the canvas and whatever drifts into it is flung
            # ALONG THE RETICLE'S OWN RAY -- so a centred stick still means "straight away
            # from me" (the classic poke) while an off-centre one throws things that way.
            # Published as CANVAS PX, y up: the exact convention of the mouse's boop xy, so
            # the shader draws it and sim.py rays it with the same code path, and the ghost
            # of a cursor is the same number in both places.
            #
            # It is placed on the VISIBLE FRAME, not on the raw canvas: the canvas is square
            # and the Pixel head is portrait, so the blit cover-crops about 28% of the canvas
            # off each side (canvas_per_screen -- one uniform scale, no stretch). Mapping the
            # stick to the raw canvas would throw the reticle off-screen at roughly half
            # deflection; going through the SAME crop math the cursor uses (read_mouse_real)
            # means full deflection lands exactly on the frame edge.
            cbx, cby = pad_xy(pwidgets, "boop", pnow)
            cactive = 1.0 if (abs(cbx) > PAD_BOOP_DEADZONE or abs(cby) > PAD_BOOP_DEADZONE) else 0.0
            if pad["alive"] < 0.5:
                cactive = 0.0          # phone gone: no reticle, nothing catches
            pspec = OUTPUTS[0]         # the Pixel stream head: what the phone actually shows
            pk = canvas_per_screen(pspec)
            pcx = (CANVAS_W - pspec["w"] * pk) / 2.0
            pcy = (CANVAS_H - pspec["h"] * pk) / 2.0
            pad["boop"] = ((0.5 + 0.5 * cbx) * pspec["w"] * pk + pcx,
                           CANVAS_H - ((0.5 - 0.5 * cby * PAD_STICK_Y_UP) * pspec["h"] * pk + pcy),
                           cactive)
            set_uniform("u_pad_boop", pad["boop"])
            # ---- PAD SHOCKS / AURORA / SPEED (2026-10-15, remote variant) --------------------
            # Two SWITCH GATES, three BUTTON FIRES and one STEP dial, for the widgets on the
            # user's brown-dwarf pad. House rule applied to each: ON = the shader's own
            # automatic system is ENGAGED, OFF = it is quiet. A BUTTON is an explicit press,
            # so it is NOT gated by its switch -- that is what keeps "switched off" meaning
            # quiet rather than broken, and it is the only way to fire while the switches sit
            # at their power-on default (the app initialises every switch to off and only
            # publishes on touch). Published as gates/edges/number only; the shader decides
            # what a shock is.
            pad["shocks"] = (pad_sw(pwidgets, "Dwarf Shocks"), pad_sw(pwidgets, "Moon Shocks"))
            pad["fire"] = (impulses.get("Dwarf Big Shock", 0.0),
                           impulses.get("Moon Shock", 0.0),
                           impulses.get("Ring Aurora", 0.0))
            pad["refill"] = impulses.get("refill", 0.0)
            pad["star_tint"] = impulses.get("star_tint", 0.0)
            pad["farm_off"] = pad_sw(pwidgets, "ring_farm_off")   # ringbox RING FARM OFF switch (2026-10-22)
            pad["sliders"] = {k: pad_val_opt(pwidgets, k) for k in ("cam_mode", "sys_mode", "star_colour")}
            pad["mode_fire"] = (impulses.get("mode_recover", 0.0),
                                impulses.get("mode_rosette", 0.0))
            # "Speed": a DroidPad STEP_SLIDER is a FLOAT over minValue..maxValue -- 0..10 here
            # (its 5 steps sit between the ends, so there are 7 stops and the phone prints
            # 5.00 dead centre). Half of it is therefore "the rate the shader was authored
            # at": 0 = still, 5 = as authored, 10 = double. It publishes the MULTIPLIER, not
            # the raw number, so no shader has to know where the middle of someone's slider is.
            # An ABSENT dial publishes 1.0 and not 0.0: the app starts every slider at its
            # MINIMUM, so "absent = 0" would freeze a shader the first time any pad carrying a
            # Speed dial was even loaded.
            pad["speed"] = (max(0.0, min(2.0, pad_val(pwidgets, "Speed") / 5.0))
                            if (pwidgets.get("Speed") or {}).get("type") else 1.0)
            set_uniform("u_pad_shocks", pad["shocks"])
            set_uniform("u_pad_fire", pad["fire"])
            set_uniform("u_pad_speed", pad["speed"])
            # ---- SPECIAL THROTTLE (both behaviours live in the remote variant) --------------
            # Shared rule: ON = the semi-regular throttle is ENGAGED. Switch "gate" (or "throttle"/"3").
            # No such switch -> 0 = free-for-all. Read with the switch's own existence each frame (not
            # "last value seen"): pads get swapped, and a general pad carries none of these, so a gate
            # left latched by the last pad with one would otherwise stay shut with no widget to reopen.
            gsw = max(pad_sw(pwidgets, "gate"), pad_sw(pwidgets, "throttle"), pad_sw(pwidgets, "3"))
            pad["gate"] = (gsw if any((pwidgets.get(n) or {}).get("type") == "SWITCH"
                                      for n in ("gate", "throttle", "3")) else 0.0)
            set_uniform("u_pad_gate", pad["gate"])
            set_uniform("u_pad_fly", (impulses.get("Normal Figure", 0.0),
                                      impulses.get("Shadybusiness", 0.0),
                                      impulses.get("Emitter", 0.0)))
            set_uniform("u_pad_down", (float(pheld.get("Normal Figure", 0)),
                                       float(pheld.get("Shadybusiness", 0)),
                                       float(pheld.get("Emitter", 0))))
            # ---- THE EMBER REMOTE (ctrl_ember_flock_v3, 2026-10-16) -------------------------
            # A SEPARATE pad with its own widget ids, so the two remotes cannot cross. This is
            # not redundant with the block above: the bridge's surface is the UNION of every pad
            # touched since it started, keyed by widget id and LATCHED forever, so wiring
            # ctrl_ember_flock_v3 to MyControlPad's ids (`Frenzy`, `Tilt`, `Pocket`, ...) meant
            # the worms remote drove the flock for as long as it had been touched once. Under
            # these ids the worms pad publishes nothing this shader reads, and vice versa.
            # Same contracts as the block above: an ABSENT widget reads exactly where the app
            # would leave an untouched one, and no dial may treat "absent" as "zero" unless zero
            # is its real rest (see pads/README.md -- `Split` needs the -1 sentinel for that).
            set_uniform("u_pad_e_tilt", pad_xy(pwidgets, "Plane", pnow))
            set_uniform("u_pad_e_skew", pad_xy(pwidgets, "Skew", pnow))
            # (There is deliberately NO water uniform here: user 2026-10-16 -- the flock with the
            # mirror floor is the AUTO half, the REMOTE half is the blackhole variant, and the `off`
            # switch's shader swap is what picks between them. Nothing keys off the sticks.)
            # `Blaze` 3-stop dial, `Rose` pocket dial: same shapes as `Frenzy` / `Pocket`.
            set_uniform("u_pad_e_blaze", (max(0.0, min(1.0, pad_val(pwidgets, "Blaze") / 10.0))
                                          if pwidgets.get("Blaze") else 0.5))
            set_uniform("u_pad_e_rose", max(0.0, min(1.0, pad_val(pwidgets, "Rose") / 10.0)))
            # `Split`: the dial reads bottom = the two flocks act as ONE swarm, top = fully
            # independent. Published as the merge amount (u_pad_e_one, 1 = one swarm), NOT the raw
            # dial, for two reasons: the shader's rest then IS its zero, so no sentinel is needed,
            # and an unpublished uniform (a renderer that does not know this name) reads 0 =
            # independent, i.e. the plain desktop pair -- a safe degradation, where a `-1` sentinel
            # scheme would have had a missing uniform silently merge the flock. `if present` is
            # still required: absent reads 0, and 1 - 0 would be a full merge.
            set_uniform("u_pad_e_one", (1.0 - max(0.0, min(1.0, pad_val(pwidgets, "Split") / 10.0))
                                        if pwidgets.get("Split") else 0.0))
            # `Align`: a MULTIPLIER on the music's own alignment weight, 0..10 -> 0..2 over the
            # dial's 5 stops, so its centre (5) is "as authored" and an absent dial reads 1.0 --
            # the same rest rule as `Speed` (a bare 0 in either would mean "no alignment at all").
            set_uniform("u_pad_e_align", (max(0.0, min(2.0, pad_val(pwidgets, "Align") / 5.0))
                                          if pwidgets.get("Align") else 1.0))
            # `Solo` = one flock possessed at a time (rest 0 = both, the music's own frenzy);
            # `Auto` = the music's own move pacing (rest 1 = auto, as `Auto Moves` above).
            set_uniform("u_pad_e_solo", pad_sw(pwidgets, "Solo") if pwidgets.get("Solo") else 0.0)
            set_uniform("u_pad_e_auto", pad_sw(pwidgets, "Auto") if pwidgets.get("Auto") else 1.0)
            # Buttons are impulses this frame, so they are inherently id-keyed: nothing latches.
            set_uniform("u_pad_e_fly", (impulses.get("Figure", 0.0),
                                        impulses.get("Hawk", 0.0),
                                        impulses.get("Special", 0.0)))
            # ---- THE XBOX CONTROLLER (fishbox.mp, 2026-10-17; see GAMEPAD_FILE) --------------
            # Published as themselves -- deflections, held buttons, press edges -- never a position
            # or a command: the shader decides what a layer means (scope.md "NEW shader fishbox.mp").
            xdoc = None
            try:
                xm = os.path.getmtime(GAMEPAD_FILE)
                if time.time() - xm <= XB_STALE_S:
                    if xm != xb["m"]:
                        xb["m"] = xm
                        with open(GAMEPAD_FILE) as fh:
                            xb["doc"] = json.load(fh)
                    xdoc = xb["doc"]
            except (OSError, ValueError):
                pass
            xb["alive"] = 1.0 if xdoc else 0.0
            xd = xdoc or {}
            xbtn = xd.get("btn", {})
            xpress = xd.get("press", {})

            def xb_stick(key):
                x, y = (xd.get(key) or [0.0, 0.0])[:2]
                x = 0.0 if abs(x) < XB_DEADZONE else x
                y = 0.0 if abs(y) < XB_DEADZONE else y
                return (float(x), float(y) * XB_STICK_Y_UP)
            # press EDGES: the bridge counts every press (it sees every event, we sample at frame
            # rate); adopt the counters on the first read so a renderer restart replays nothing, and
            # a counter that went backwards is a bridge restart -> the whole count is real presses.
            if xb["prev_press"] is None and xdoc:
                xb["prev_press"] = dict(xpress)
            xtaps = {}
            for k, n in xpress.items():
                prev = (xb["prev_press"] or {}).get(k, 0)
                d = n if n < prev else n - prev
                if d > 0:
                    xtaps[k] = float(d)
            if xdoc:
                xb["prev_press"] = dict(xpress)
            if xtaps:
                print("xb press: %s" % ", ".join(sorted(xtaps)), file=sys.stderr)

            try:                                     # the layer map: the file wins when it exists and parses
                mm = os.path.getmtime(GAMEPAD_MAP_FILE)
                if mm != xb["map_m"]:
                    xb["map_m"] = mm
                    with open(GAMEPAD_MAP_FILE) as fh:
                        doc = json.load(fh)
                    xb["map"] = {s: dict(XB_LAYER_MAP[s], **doc.get(s, {})) for s in ("left", "right")}
                    print("xb layer map: %s" % json.dumps(xb["map"], sort_keys=True), file=sys.stderr)
            except (OSError, ValueError, TypeError):
                pass

            def xb_layer(side):
                m = xb["map"][side]
                if xbtn.get(m["lower"]):
                    return 2.0
                return 1.0 if xbtn.get(m["upper"]) else 0.0
            set_uniform("u_xb_alive", xb["alive"])
            set_uniform("u_xb_ls", xb_stick("ls"))
            set_uniform("u_xb_rs", xb_stick("rs"))
            set_uniform("u_xb_trig", (float(xd.get("lt", 0.0)), float(xd.get("rt", 0.0))))
            set_uniform("u_xb_face", tuple(float(xbtn.get(k, 0)) for k in ("a", "b", "x", "y")))
            set_uniform("u_xb_bump", (float(xbtn.get("lb", 0)), float(xbtn.get("rb", 0))))
            set_uniform("u_xb_tap", (xtaps.get("lb", 0.0), xtaps.get("rb", 0.0),
                                     xtaps.get("view", 0.0), xtaps.get("menu", 0.0)))
            set_uniform("u_xb_layer", (xb_layer("left"), xb_layer("right")))
            set_uniform("u_xb_hat", tuple(float(v) for v in (xd.get("hat") or [0, 0])[:2]))
            # ---- GENERIC SPARE CONTROLS (the "general" pad, 2026-10-15) ----------------------
            # DroidPad's remaining widget classes, published AS THEMSELVES, so any shader can use
            # them without this file pretending to know what they mean. Same contract as the rest
            # of the surface: a number or an angle, never a command, and a widget the CURRENT pad
            # does not carry reads 0 -- exactly where the app leaves an untouched one.
            #   "steering_wheel" -> u_pad_wheel, DEGREES, -(maxAngle)..+(maxAngle), where maxAngle is
            #     the phone's own setting (360 on the general pad, whose wheel is freeRotation, so it
            #     can pass a full turn either way). The app also SELF-CENTRES a released wheel: it
            #     animates the angle back to 0 (verified in the bridge log -- a push to -196 degrees
            #     swung back to 0 over ~0.9 s). So read it as an ANGLE and DIFF it for motion; an
            #     integrator would wind the scene and then have it unwound by the phone at speed.
            #   "slider" / "step_slider" -> u_pad_slider / u_pad_step, the RAW value in whatever
            #     min..max the phone was configured with (the general pad's slider is 0..1, its step
            #     slider 0..10 in 16 steps). Deliberately NOT normalised to 0..1: the event carries
            #     no range (only the value), so guessing one would silently rescale somebody's dial.
            #     NB the app sends a STEP_SLIDER as type "SLIDER" like any other -- the only thing
            #     that survives the wire is the widget ID, so the two are told apart by name.
            wheelw = pwidgets.get("steering_wheel") or {}
            wheel_now = (float(wheelw.get("angle", 0.0))
                         if wheelw.get("type") == "STEERING_WHEEL" else 0.0)
            if pnow - float(wheelw.get("t", 0.0)) > PAD_STICK_TTL_S:
                # A wheel nobody has refreshed has sprung back: 0 IS its resting angle, so a
                # release that never arrived costs one unwind instead of a permanently rolled
                # view -- and the next real turn still diffs correctly from 0. (An absent
                # widget has no stamp at all, so this is also the "no wheel on this pad" case.)
                wheel_now = 0.0
            pad_slider, pad_step = pad_val(pwidgets, "slider"), pad_val(pwidgets, "step_slider")
            set_uniform("u_pad_wheel", wheel_now)
            set_uniform("u_pad_slider", pad_slider)
            set_uniform("u_pad_step", pad_step)
            # A one-line summary the first time each pad's WIDGET SET is seen, and whenever the two
            # DERIVED dials move: which surface the renderer is really reading, and what the spare
            # controls it does not interpret report. (The id list is the thing to check when a widget
            # seems dead -- a typo in a widget's name is indistinguishable from a widget nobody
            # touched, this side of the wire. `frenzy`/`pocket` are the 0..1 values actually published
            # as u_pad_frenzy / u_pad_pocket, so the `Frenzy` and `Pocket` dials are traceable here.)
            ids = ",".join(sorted(pwidgets))
            seen = (ids, round(pad["frenzy"], 2), round(pad["pocket"], 2))
            if seen != pad["ids"]:
                pad["ids"] = seen
                print("pad: %d widgets [%s] wheel=%.1f slider=%.3f step=%.3f frenzy=%.2f pocket=%.2f"
                      % (len(pwidgets), ids, wheel_now, pad_slider, pad_step,
                         pad["frenzy"], pad["pocket"]), file=sys.stderr)
            if pad["alive"] > 0.5:
                # The wheel is the one camera axis no stick can reach (ROLL), and it drives the SHARED
                # camera as a DELTA, exactly like the phone gyro -- so the phone's own return animation
                # unwinds precisely as much as the hand wound. (As a rate, a released full lock would
                # instead spin every flying shader back at the wheel's return speed, ~1200 deg/s.)
                d_deg = wheel_now - pad["wheel"]
                if abs(d_deg) > 180.0:          # a reconnect or a jump rope, not a hand movement
                    d_deg = 0.0
                if d_deg:
                    mouse_cam["roll"] += d_deg * PAD_WHEEL_ROLL_SENS
                    # ...and the free camera's roll (2026-10-16). The free camera has no roll input at
                    # all otherwise -- the pad's Look stick only does yaw/pitch -- so the wheel (finger
                    # or the phone's tilt lane) is what tilts the horizon. It is the same delta the
                    # legacy path just took, applied about the axis the quaternion camera actually
                    # has: the view axis. An ATTITUDE, not a rate: the phone's lane sends the posture
                    # it is holding, so levelling the phone levels the horizon again.
                    dq = view_quat(0.0, 0.0, math.degrees(d_deg * PAD_WHEEL_ROLL_SENS) * PHONE_ROLL_SIGN)
                    q = quat_mul(pad["cq"], dq)
                    n = math.sqrt(sum(c * c for c in q)) or 1.0
                    pad["cq"] = tuple(c / n for c in q)
                    super_held = 1.0
                    set_uniform("u_steer", 1.0)
            pad["wheel"] = wheel_now
            # ---- REMOTE TRIGGER: swap the shader for its remote-only variant ----------------
            # (user 2026-10-15) "lets make the remote shaders just drive remote only shader variants,
            # we settle for a prefix like ctrl_worms.mp ... hard cut doesnt matter ... just repurpose
            # the off switch". A SWITCH named "off" picks between "<name>.mp" and "ctrl_<name>.mp" by
            # writing active_shader.txt -- the pointer this renderer already watches -- so a swap is an
            # ordinary shader reload (a HARD CUT: the context rebuilds and sim state resets, which the
            # user explicitly does not mind). Switched off means the AUTO half, so this also works as
            # "get me back out of the remote". Only acts when the sibling really exists, and only
            # writes when the content actually has to change, so it never fights the prefs page or
            # shader-mp-stage, and it does nothing at all on a pad without the switch.
            offw = pwidgets.get("off") or {}
            if offw.get("type") == "SWITCH" and pad["m"] != pad.get("off_m"):   # LEAN: only when pad.json changed
                pad["off_m"] = pad["m"]
                cur = read_active_shader_name()
                pair = shader_pair(cur)
                if pair:
                    want = pair[1] if pad_sw(pwidgets, "off") > 0.5 else pair[0]
                    if want != cur:
                        write_active_shader_name(want)
                        # A swap reloads the shader, so the sim state is gone anyway: put the pad's
                        # own flight back to the authored framing too, or you would land looking out
                        # from wherever you last flew. This is also the only "recentre" there is --
                        # flip off and back on. (The pad has no widget for it in the dwarf profile.)
                        pad["cam"] = {k: 0.0 for k in FREEFLY_KEYS}
                        pad["cq"], pad["mv"] = VIEW_IDENTITY, [0.0, 0.0, 0.0]
                        print("pad: remote trigger -> %s" % want, file=sys.stderr)
            set_uniform("u_cam_look", (mouse_cam["yaw"], mouse_cam["pitch"], mouse_cam["roll"]))
            set_uniform("u_cam_move", (mouse_cam["r"], mouse_cam["u"], mouse_cam["f"]))
            # Super+Z (i3 -> ~/bin/backdrop-autozoom) flips autozoom.txt: 0/1
            slow_files = t2 - autozoom.get("t", 0.0) >= 0.25       # LEAN (2026-10-19): these 3 files 4x a second
            if slow_files:
                autozoom["t"] = t2
            try:
                if not slow_files:
                    raise _SkipCheck
                az_m = os.path.getmtime(AUTOZOOM_FILE)
                if az_m != autozoom["m"]:
                    autozoom["m"] = az_m
                    with open(AUTOZOOM_FILE) as fh:
                        autozoom["v"] = 1.0 if fh.read().strip() == "1" else 0.0
            except _SkipCheck:
                pass
            except (OSError, ValueError):
                autozoom["v"] = 0.0
            set_uniform("u_autozoom", autozoom["v"])
            # dive path (~/bin/backdrop-dive -> cam_target.txt:
            # "re im zoom rot id"); the shader flies to it when the id changes
            try:
                if not slow_files:
                    raise _SkipCheck
                ct_m = os.path.getmtime(CAM_TARGET_FILE)
                if ct_m != cam_target["m"]:
                    cam_target["m"] = ct_m
                    with open(CAM_TARGET_FILE) as fh:
                        v = [float(x) for x in fh.read().split()[:5]]
                    if len(v) == 5:
                        cam_target["t"], cam_target["id"] = tuple(v[:4]), v[4]
                        print(f"cam target: {cam_target['t']} id {cam_target['id']:.0f}", file=sys.stderr)
            except _SkipCheck:
                pass
            except (OSError, ValueError):
                pass
            set_uniform("u_cam_target", cam_target["t"])
            set_uniform("u_cam_target_id", cam_target["id"])
            # auto-dive (Super+Up): counted on the beat clock, like the flights
            try:
                if not slow_files:
                    raise _SkipCheck
                ad_m = os.path.getmtime(AUTODIVE_FILE)
                if ad_m != autodive["m"]:
                    autodive["m"] = ad_m
                    with open(AUTODIVE_FILE) as fh:
                        on = fh.read().strip() == "1"
                    if on and not autodive["on"]:
                        autodive["beats"] = AUTODIVE_BEATS      # first jump right away
                        autodive["last"], autodive["id"] = None, cam_target["id"]
                    autodive["on"] = on
            except _SkipCheck:
                pass
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
            delayed, stems_arr, stem_scalars, kick_arr = audio_state[0]
            if delayed is not None:
                last_frame = delayed
            if kick_arr is not None:
                kick_tex.write(kick_arr.tobytes())
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
                    mp["compute"].inputs = {"boop": hand["boop"], "boop_xy": hand["xy"], "reset": hand["reset"],
                                            "canvas": (float(CANVAS_W), float(CANVAS_H)),
                                            # pad flight + reticle: the SAME numbers the image pass
                                            # gets, so a sim that mirrors the camera (brown_dwarf's
                                            # mt_camera -> the boop ray) stays in lockstep with it.
                                            "pad_look": (pad["cam"]["yaw"], pad["cam"]["pitch"],
                                                         pad["cam"]["roll"]),
                                            "pad_move": tuple(pad["mv"]),
                                            "pad_cam_q": pad["cq"],
                                            "pad_boop": pad["boop"],
                                            "pad_alive": pad["alive"],
                                            # the rest of the surface: the two shock GATES, the
                                            # three fire EDGES (1 on the frame the press lands)
                                            # and the Speed multiplier. The sim owns these --
                                            # it fires the ambient bolts itself and hands the
                                            # gates/edges on to the NOTES kernel and the aurora.
                                            "pad_shocks": pad["shocks"],
                                            "pad_fire": pad["fire"],
                                            "pad_refill": pad["refill"],
                                            "pad_star_tint": pad["star_tint"],
                                            "pad_farm_off": pad.get("farm_off", 0.0),
                                            "pad_sliders": pad.get("sliders", {}),
                                            "pad_joy": pad.get("joy", {}),          # raw sticks by id (2026-10-23)
                                            "xb": (xd if xdoc else {}),             # the Xbox controller (2026-10-23): gamepad.json as is, {} when gone
                                            "pad_dpad": pad.get("dclicks", {}),     # dpad clicks this frame
                                            "pad_mode": pad["mode_fire"],
                                            "pad_speed": pad["speed"],
                                            # THE MOUSE MODES: (mode, x, y, button held, meteor clicks, meteor x, y)
                                            "mouse": pad.get("mouse", (0.0,) * 7),
                                            # THE AUDIO SCALARS (u_kick, u_ms_*: the dict the shaders get), read-only, for
                                            # CPU-side music reactions (2026-10-19, dwarf_box's chase: the swarm's hawk dash)
                                            "audio": audio_state[0][2]}
                    _ts = time.monotonic()
                    mp["compute"].step(dt, elapsed)
                    seg_add("sim", (time.monotonic() - _ts) * 1000.0)
                # The front/canvas render always sees the identity view, so a shader that declares
                # u_view_q is unchanged for every head without a window.
                try:
                    mp["image_prog"]["u_view_q"].value = VIEW_IDENTITY
                except KeyError:
                    pass
                try:
                    mp["image_prog"]["u_view_fov"].value = 1.0
                except KeyError:
                    pass
                render_multipass(ctx, mp, canvas_fbo, audio_channel_tex)
                render_windows(ctx, mp, presenters, win_state, audio_channel_tex)
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
                # windows need a multipass Image pass; a single-pass shader shows the canvas as ever
                for p in presenters:
                    if p["view"] is not None:
                        p["view"]["fbo"].release()
                        p["view"]["tex"].release()
                        p["view"] = None
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
                # A window head shows ITS canvas (same size, same crop rule), the rest the shared one.
                (p["view"]["tex"] if p["view"] is not None else canvas_tex).use(location=0)  # shared GL namespace via `share=`
                uv_off, uv_scl, uv_rot = p["uv_offset"], p["uv_scale"], p["uv_rot"]
                p["prog"]["src_tex"].value = 0
                p["prog"]["uv_offset"].value = uv_off
                p["prog"]["uv_scale"].value = uv_scl
                p["prog"]["uv_rot"].value = uv_rot
                p["prog"]["tap"].value = (0.25 * uv_scl[0] / p["spec"]["w"],
                                          0.25 * uv_scl[1] / p["spec"]["h"])
                p["ctx"].clear(0.0, 0.0, 0.0, 1.0)
                p["vao"].render(moderngl.TRIANGLE_STRIP)
                _ts = time.monotonic()
                glfw.swap_buffers(p["window"])
                seg_add("swap_phone" if pi == 0 else "swap_dells", (time.monotonic() - _ts) * 1000.0)
            t5 = time.monotonic()
            seg_add("poll", (t1 - t0) * 1000.0); seg_add("layout", (t2 - t1) * 1000.0)
            seg_add("inputs", (render_start - frame_start) * 1000.0)
            seg_add("draw", render_ms); seg_add("capture", (t4 - render_start) * 1000.0 - render_ms)
            seg_add("present", (t5 - t4) * 1000.0)

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
                        seg_add("sleep", slack * 1000.0)
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
                print("perfseg: " + "  ".join(f"{k} {v / n:.1f}" for k, v in seg.items())
                      + "  (ms per frame; draw includes sim, present includes the swaps)", file=sys.stderr)
                seg.clear()
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
