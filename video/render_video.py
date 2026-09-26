#!/usr/bin/env python3
"""render_video.py -- headless dGPU fragment-shader renderer -> AV1 .mp4.

Renders a wrapped Shadertoy-style fragment shader frame-by-frame into a float
FBO on the **dGPU** (surfaceless EGL, no display/scanout), CPU-reads back, and
pipes raw pixels to ffmpeg (`av1_amf`). Default output is 10-bit HDR10 AV1
(BT.2020 / SMPTE 2084 / p010le); pass --sdr for 10-bit BT.709 SDR.

Safety: never touches a monitor. The kernel panic only happens when GL renders
to a display surface; here everything goes to an FBO and out to a file.

Usage:
    .venv/bin/python render_video.py \
        --shader wrapped/Colorful_FBM.glsl \
        --audio track.flac --fps 60 \
        --width 1080 --height 2410 -o out.mp4
    .venv/bin/python render_video.py --shader ... --features features.npz \
        --frames 60 --width 320 --height 640 -o smoke.mp4   # quick smoke render
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "0")

import moderngl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bands_stems as BS
import features as F
import wrap_shader as W

MULTIPASS_LETTERS = ("A", "B", "C", "D")

DEFAULT_FFMPEG = "/opt/ffmpeg-master-custom/bin/ffmpeg"

VERT = """#version 330
in vec2 in_vert;
out vec2 uv;
void main() { uv = (in_vert + 1.0) * 0.5; gl_Position = vec4(in_vert, 0.0, 1.0); }
"""

# Second fullscreen pass: sample the main render and apply the HDR10 PQ OETF on
# the GPU (much faster than doing the pow() in numpy). uHdrScale = nits/10000
# for HDR, 0.0 = SDR passthrough (just clamp).
POST_FRAG = """#version 330
in vec2 uv;
// INTEGER output target. The shader does the 16-bit quantisation itself, so
// the readback is a straight copy of bytes that are already rgba64le -- no
// numpy conversion at all. Measured 2026-09-21: an f4 readback runs at
// 1.13 GB/s and then costs another 34.7 ms in numpy; this costs 27.5 ms and
// nothing after it.
out uvec4 outColor;
uniform sampler2D uTex;
uniform float uHdrScale;    // nits that shader value 1.0 maps to, /10000
uniform float uHdrPeak;     // display peak for the highlight roll, /10000
uniform float uGamutNative; // 1 = shader RGB is BT.2020 already (no matrix)
uniform float uRollMode;    // 0 = none, 1 = luma, 2 = per-channel clip
uniform float uSat;         // linear saturation multiplier after the roll
void main() {
    // Flipped here, not in numpy: GL framebuffers are bottom-row-first and
    // ffmpeg's rawvideo wants top-row-first. Doing it on the GPU costs
    // nothing and lets the readback go straight to the encoder untouched.
    vec3 c = texture(uTex, vec2(uv.x, 1.0 - uv.y)).rgb;
    if (uHdrScale > 0.0) {
        // Gamut. "rec709" converts the shader's Rec.709/sRGB coordinates into
        // BT.2020 (linear 3x3) -- technically faithful, but it pulls the
        // stylized colours back to standard Rec.709 saturation. "native"
        // (default) declares the shader's raw RGB to already BE BT.2020, so the
        // disk blues / redshifted reds sit out near the display's real gamut
        // edge. Neither is a "bug": the shaders are stylized, not broadcast
        // footage. (The 2026-09-23 first attempt accidentally got the native
        // look while ALSO mapping SDR white to 1000 nits, which is what blew
        // the dwarf into an orange blob -- the roll below is the fix.)
        const mat3 M709_2020 = mat3(
            0.6274040, 0.0690970, 0.0163916,   // column 0
            0.3292820, 0.9195400, 0.0880132,   // column 1
            0.0433136, 0.0113612, 0.8955950);  // column 2
        vec3 s = max(c, 0.0);
        vec3 lin = (uGamutNative > 0.5 ? s : M709_2020 * s) * uHdrScale;
        // Highlight handling above --hdr-peak. Which is "right" is a look
        // decision, so it is selectable:
        //   none  - encode the raw channel ratios; the display's own tone
        //           mapper decides what to do with the top end.
        //   clip  - per-channel clamp. A single dominant primary (a red star,
        //           a blue disk) keeps its subpixel saturated instead of being
        //           dragged toward white.
        //   luma  - scale RGB by a rolled luminance; hue-exact but any body
        //           brighter than the knee is dimmed as a whole, and on a
        //           luminance-ceiling display the bright coloured bodies read
        //           grey (this was the "disk/star desaturated" report).
        if (uRollMode > 1.5) {
            lin = min(lin, vec3(uHdrPeak));
        } else if (uRollMode > 0.5) {
            float L = dot(lin, vec3(0.2627, 0.6780, 0.0593));   // BT.2020 luma
            if (L > 1e-6) {
                float knee = min(uHdrScale * 2.0, uHdrPeak);
                float span = max(uHdrPeak - knee, 1e-6);
                float Lp = (L <= knee) ? L
                         : knee + span * (1.0 - exp(-(L - knee) / span));
                lin *= Lp / L;
            }
        }
        // Re-saturate after the tone map. PQ (and any luminance compression)
        // lifts the low channels, so a bright blue disk or a red star comes out
        // pale even when the channel ratios were preserved. This multiplies the
        // distance from luma back out, in linear light, before the PQ encode.
        if (uSat != 1.0) {
            float y = dot(lin, vec3(0.2627, 0.6780, 0.0593));
            lin = max(vec3(y) + uSat * (lin - vec3(y)), 0.0);
        }
        float m1 = 0.1593017578125, m2 = 78.84375;
        float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
        vec3 y = pow(max(lin, 0.0), vec3(m1));
        c = pow((c1 + c2 * y) / (1.0 + c3 * y), vec3(m2));
    } else {
        c = clamp(c, 0.0, 1.0);
    }
    outColor = uvec4(clamp(c, 0.0, 1.0) * 65535.0 + 0.5, 65535u);
}
"""


def find_dgpu_index():
    """EGL device_index to render on. $EGL_DEVICE_INDEX wins; otherwise the
    first known discrete card (RX 9060 XT / gfx1200 here), otherwise the first
    hardware (non-software) renderer among EGL devices 0-3."""
    if os.environ.get("EGL_DEVICE_INDEX"):
        return int(os.environ["EGL_DEVICE_INDEX"])
    hw = []
    for idx in range(4):
        try:
            ctx = moderngl.create_standalone_context(backend="egl", device_index=idx)
        except Exception:
            continue
        r = ctx.info.get("GL_RENDERER", "")
        ctx.release()
        rl = r.lower()
        if any(k in rl for k in ("llvmpipe", "softpipe", "swrast")):
            continue
        if any(k in rl for k in ("9060", "navi", "gfx1200", "rx 90")):
            return idx
        hw.append(idx)
    return hw[0] if hw else None


def _beat_clock_arrays(res_or_npz, fps):
    """Per-frame beat clock for the shader-backdrop `u_beat_*` contract.

    The map is PRECOMPUTED (features.py tracks the whole track at once), so
    unlike the live path there is nothing to estimate here and no confidence
    to report -- conf is 1.0 because the grid is exact. `u_tbeat` is the
    INSTANTANEOUS beat period, recovered from the count's own slope, so a
    shader scheduling against it (the orrery's meteor does) follows real tempo
    drift rather than a single album-wide average.

    Returns None for an npz predating the beat map, in which case the uniforms
    are simply never set and a shader takes its own fallback path."""
    try:
        count = np.asarray(res_or_npz["beat_count"], dtype=np.float64)
        phase = np.asarray(res_or_npz["beat_phase"], dtype=np.float64)
    except (KeyError, IndexError):
        return None
    if count.size < 2:
        return None

    d = np.diff(count)
    d[d < 0] += F.BEAT_WRAP          # the single wrap, if the track is long enough
    d = np.concatenate([d, d[-1:]])
    tbeat = np.where(d > 1e-9, (1.0 / fps) / np.maximum(d, 1e-9), 0.5)
    return {"count": count, "phase": phase, "tbeat": tbeat}


def load_features(audio, features_path, fps):
    if features_path:
        d = np.load(features_path)
        return (d["features_norm"], int(d["num_frames"]), float(d["fps"]),
                _beat_clock_arrays(d, float(d["fps"])))
    if not audio:
        raise SystemExit("need --audio or --features")
    audio_np, sr = F.decode_file(audio)
    res = F.extract(audio_np, sr, fps)
    return (res["features_norm"], res["num_frames"], res["fps"],
            _beat_clock_arrays(res, res["fps"]))


# Per-phase timing, off unless RENDER_PHASE_LOG=1. Exists because "why is an
# offline render ~11x slower than the same shader live at the same resolution"
# is not answerable by looking at CPU or GPU utilisation -- when the pipeline
# is serialised, nothing is saturated and everything is waiting.
_PHASE_LOG = os.environ.get("RENDER_PHASE_LOG", "0") == "1"
_phases = {"render": 0.0, "readback": 0.0, "convert": 0.0, "pipe": 0.0, "n": 0}


def report_phases():
    n = _phases["n"]
    if not n:
        return
    total = sum(v for k, v in _phases.items() if k != "n")
    print(f"\nper-frame phase breakdown over {n} frames "
          f"({total / n * 1000:.1f} ms/frame total):")
    for k in ("render", "readback", "convert", "pipe"):
        ms = _phases[k] / n * 1000
        print(f"  {k:9s} {ms:7.1f} ms  {_phases[k] / total * 100:5.1f}%")


def _align16(v):
    """Round up to a multiple of 16, the VAAPI coded-frame alignment."""
    return (int(v) + 15) // 16 * 16


def make_noise(seed, size=512):
    rng = np.random.default_rng(seed)
    return (rng.random((size, size, 4)) * 255).astype("u1")


def is_multipass(shader_path):
    return os.path.isdir(shader_path)


def normalize_mp_wiring(wiring):
    """Shadertoy manifests express a pass sampling its OWN previous frame as
    {"type": "self"}. The binder only understands
    {"type": "buffer", "id": <pass>}. shader-backdrop's live renderer does this
    rewrite; this offline loader did NOT, so a "self" channel silently fell
    through to the 1x1 gray fallback -- the buffer never read its own previous
    frame, its state (camera, gears, beat count) never persisted, and the whole
    scene came out FROZEN. Ported verbatim from the live renderer."""
    for pass_name, channels in wiring.items():
        for _ch, spec in channels.items():
            if spec.get("type") == "self":
                spec["type"] = "buffer"
                spec["id"] = pass_name
    return wiring


def load_multipass(ctx, vbo, mp_dir, width, height, dtype):
    """Buffer chain (buffer_a.glsl..buffer_d.glsl -> image.glsl), manifest-
    driven wiring -- same model as shader-backdrop's live renderer.py, ported
    here since this offline pipeline only ever handled single-pass shaders.
    Every buffer renders at the full target resolution (no bloom-style
    downscaling) -- there's no frame-time budget to protect here, so no
    reason to trade quality for it.

    Buffer textures are ALWAYS f4 (float32), ignoring the `dtype` param
    entirely (defends against a future --dtype f2 override) --
    a self-feeding buffer accumulates tiny per-frame deltas (e.g. a camera
    orbit's yaw/position state machine) and f2's precision at those
    magnitudes is comparable to the delta itself, so each frame's nudge
    gets quantized away on write. shader-backdrop's live renderer already
    gets this right (always f4 for multipass buffers); this offline
    version incorrectly inherited the output dtype instead."""
    with open(os.path.join(mp_dir, "manifest.json")) as f:
        manifest = json.load(f)
    wiring = normalize_mp_wiring(manifest.get("wiring", {}))

    # common.glsl (optional, prepended to every pass's source) -- matches
    # shader-backdrop's own renderer.py convention. Missed in the initial
    # port of this loader (see module docstring), which broke any shader
    # whose buffer_*.glsl/image.glsl rely on common.glsl for shared #defines
    # (ROWS/COLS/etc) instead of duplicating them locally.
    common_path = os.path.join(mp_dir, "common.glsl")
    common_src = open(common_path).read() if os.path.isfile(common_path) else ""
    # 2026-09-26: shaders can pick offline-only quality settings (no frame-time
    # budget here -- user: "we dont care at all about fps in the offline path").
    common_src = "#define OFFLINE 1\n" + common_src

    fallback_tex = ctx.texture((1, 1), 4, b"\x80\x80\x80\xff")
    photo_textures = {}  # path -> moderngl.Texture, populated lazily by bind_mp_wiring

    buffers = {}
    for letter in MULTIPASS_LETTERS:
        fpath = os.path.join(mp_dir, f"buffer_{letter.lower()}.glsl")
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            src = f.read()
        prog = ctx.program(vertex_shader=VERT, fragment_shader=W.wrap(common_src + src))
        vao = ctx.vertex_array(prog, [(vbo, "2f", "in_vert")])
        tex_pair = [ctx.texture((width, height), 4, dtype="f4") for _ in range(2)]
        fbo_pair = [ctx.framebuffer(color_attachments=[t]) for t in tex_pair]
        buffers[letter] = {
            "prog": prog, "vao": vao,
            "tex": tex_pair, "fbo": fbo_pair, "cur": 0,
            "wiring": wiring.get(letter, {}),
        }

    with open(os.path.join(mp_dir, "image.glsl")) as f:
        img_src = f.read()
    image_prog = ctx.program(vertex_shader=VERT, fragment_shader=W.wrap(common_src + img_src))
    image_vao = ctx.vertex_array(image_prog, [(vbo, "2f", "in_vert")])

    return {
        "buffers": buffers,
        "image_prog": image_prog, "image_vao": image_vao,
        "image_wiring": wiring.get("Image", {}),
        "fallback_tex": fallback_tex,
        "photo_textures": photo_textures,
        "ctx": ctx,
        "res": (width, height),
        "all_progs": [b["prog"] for b in buffers.values()] + [image_prog],
        "dir": mp_dir,
        "manifest": manifest,
    }


def load_static_array_texture(ctx, path, mm, filter_mode=None):
    """(L,H,W,C) float .npy -> a 2D texture ARRAY (sampler2DArray), uploaded
    layer by layer from a memmap so a multi-GB bake (the Kerr-Newman path /
    disk / sky tables are ~5 GB together) never sits in RAM. Ported from
    shader-backdrop's renderer.load_static_array_texture: this offline loader
    previously only handled 2D data textures, so anything whose manifest listed
    a (L,H,W,C) bake (schwarz_orrery_v7_*.mp) died here with a ValueError.

    filter_mode (manifest "static_texture_filters"): "nearest" | "linear_wrap".
    Defaults by dtype: f32 -> nearest (exact per-pixel bake data), f16 ->
    linear_wrap (the disk-sim field; azimuth is periodic so x wraps). The
    Kerr-Newman disk passages are f16 but per-pixel exact, so the manifest sets
    "nearest" for them explicitly."""
    layers, h, w, comps = mm.shape
    if mm.dtype not in (np.float32, np.float16) or comps not in (1, 2, 3, 4):
        raise ValueError(
            f"{path}: expected float32/float16 (L,H,W,C<=4), got {mm.dtype} {mm.shape}")
    f16 = mm.dtype == np.float16
    mode = filter_mode or ("linear_wrap" if f16 else "nearest")
    tex = ctx.texture_array((w, h, layers), comps, dtype="f2" if f16 else "f4")
    for i in range(layers):
        tex.write(np.ascontiguousarray(mm[i]).tobytes(),
                  viewport=(0, 0, i, w, h, 1))
    lin = mode == "linear_wrap"
    tex.filter = ((moderngl.LINEAR, moderngl.LINEAR) if lin
                  else (moderngl.NEAREST, moderngl.NEAREST))
    tex.repeat_x = lin
    tex.repeat_y = False
    print(f"  array {path}: {layers}x{h}x{w}x{comps} "
          f"{'f16' if f16 else 'f32'} ({mm.nbytes / 1e9:.2f} GB)")
    return tex


def load_static_data_texture(ctx, path, filter_mode=None):
    """Load a raw .npy as a GL texture -- same convention as shader-backdrop's
    live renderer.load_static_data_texture: (H,W) stays single channel,
    (H,W,C<=4) uploads C components, uint8 -> normalized, float16 -> f2,
    float32 -> f4. Mipmaps only for filter_mode "linear_mip" (v8's nebula);
    everything else is unmipped so tile/atlas addressing stays exact.
    Row-flip is baked into the .npy at prep time.

    A (L,H,W,C) float array is a 2D texture ARRAY (sampler2DArray) -- see
    load_static_array_texture -- because several shaders ship multi-layer
    baked path/volume tables that must be sampled with texture(sampler2DArray,
    vec3). filter_mode comes from the manifest's static_texture_filters."""
    head = np.load(path, mmap_mode="r")
    if head.ndim == 4:
        return load_static_array_texture(ctx, path, head, filter_mode)
    del head
    arr = np.load(path)
    if arr.ndim == 2:
        h, w = arr.shape
        comps = 1
    elif arr.ndim == 3 and arr.shape[2] in (1, 2, 3, 4):
        h, w, comps = arr.shape
    else:
        raise ValueError(f"{path}: expected (H,W) or (H,W,C<=4), got {arr.shape}")
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.uint8:
        tex = ctx.texture((w, h), comps, arr.tobytes())
    elif arr.dtype == np.float32:
        tex = ctx.texture((w, h), comps, arr.tobytes(), dtype="f4")
    elif arr.dtype == np.float16:
        # 2026-09-25: ported from shader-backdrop's renderer.py. v8's nebula is
        # a float16 2D HDR map; this path accepted only uint8/float32 and raised,
        # which rejects the WHOLE shader over one texture (that is exactly how
        # v8 failed to load live before the same fix went in there).
        tex = ctx.texture((w, h), comps, arr.tobytes(), dtype="f2")
    else:
        raise ValueError(f"{path}: unsupported dtype {arr.dtype}, "
                         f"expected uint8, float16 or float32")
    # Honour the manifest's "static_texture_filters" for 2D textures too. It was
    # read only by the array path, so a "nearest" asked for on a 2D texture was
    # silently ignored. Harmless for u_stars/u_cells (texelFetch ignores
    # filtering) but NOT for the nebula, which needs the mip chain below.
    if filter_mode == "nearest":
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    elif filter_mode == "linear_mip":
        # Without mipmaps, where the lens magnifies the footprint the sampler
        # takes uncorrelated texels and the smooth map aliases into mush --
        # which is what killed v8's first diffuse layer. With the chain built,
        # the hardware picks LOD from the same fwidth(uv) footprint the star PSF
        # uses, so magnification blurs instead of aliasing.
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR_MIPMAP_LINEAR)
    else:
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = filter_mode == "linear_wrap"
    tex.repeat_y = False
    return tex


def scan_mp_feed_uses(mp_dir):
    """Which live-feed samplers does this multipass shader reference?"""
    uses = set()
    for fn in os.listdir(mp_dir):
        if fn.endswith(".glsl"):
            s = open(os.path.join(mp_dir, fn)).read()
            if "sampler2D bands" in s:
                uses.add("bands")
            if "sampler2D stems" in s:
                uses.add("stems")
    return uses


_GL_CUBE = None


def load_bc6h_cubemap(ctx, json_path):
    """Pre-compressed BC6H cube map (shader-backdrop/skycube_bake.py): the 64k
    NASA sky as 6 x 16384^2 faces + mips, ~2.15 GB. Same loader as the live
    renderer.py (2026-09-26): moderngl allocates the cube so it binds like any
    texture, raw glCompressedTexImage2D fills each face/level (moderngl cannot
    upload pre-compressed blocks), seamless cube filtering + 16x anisotropy."""
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
    GL_TEXTURE_CUBE_MAP, GL_CUBE_POS_X, BC6H_UF = 0x8513, 0x8515, 0x8E8F
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
    gl.glTexParameteri(GL_TEXTURE_CUBE_MAP, 0x813C, 0)
    gl.glTexParameteri(GL_TEXTURE_CUBE_MAP, 0x813D, meta["levels"] - 1)
    gl.glEnable(0x884F)                                   # CUBE_MAP_SEAMLESS
    err = gl.glGetError()
    if err:
        raise RuntimeError(f"BC6H cube upload failed: GL error 0x{err:x}")
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.anisotropy = 16.0
    print(f"  bc6h cube {json_path}: {n}^2 x6, {meta['levels']} levels, "
          f"{sum(e['size'] for e in meta['entries']) / 1e9:.2f} GB")
    return tex


def prepare_live_feed(ctx, mp, args, audio):
    """Build the live-style feed (bands / stems) + the manifest's static
    data textures, and bind them on every program. Texture units: 0-3 are
    iChannel0-3 (rebound per pass), 4 is iAudio, so the live extras start at
    5. Static textures are immutable, so they are uploaded and left bound;
    bands/stems are rewritten per frame in write_live_feed()."""
    d = mp["dir"]
    uses = scan_mp_feed_uses(d)
    static = mp["manifest"].get("static_textures", {})
    filters = mp["manifest"].get("static_texture_filters", {})
    feed = {"bands": None, "stems": None, "bands_tex": None, "stems_tex": None,
            "static_tex": [], "uses": uses}
    units = {}
    unit = 5
    if "bands" in uses:
        feed["bands_tex"] = ctx.texture((BS.N_BANDS, 1), 1, dtype="f4")
        feed["bands_tex"].filter = (moderngl.LINEAR, moderngl.LINEAR)
        feed["bands_tex"].use(location=unit)
        units["bands"] = unit
        unit += 1
    if "stems" in uses:
        feed["stems_tex"] = ctx.texture((BS.N_BANDS, 4), 1, dtype="f4")
        feed["stems_tex"].filter = (moderngl.NEAREST, moderngl.NEAREST)
        feed["stems_tex"].use(location=unit)
        units["stems"] = unit
        unit += 1
    for name, rel in static.items():
        p = rel if os.path.isabs(rel) else os.path.join(args.textures_root, rel)
        if p.endswith(".json"):
            tex = load_bc6h_cubemap(ctx, p)
        else:
            tex = load_static_data_texture(ctx, p, filter_mode=filters.get(name))
        tex.use(location=unit)
        units[name] = unit
        feed["static_tex"].append(tex)
        print(f"  static {name}: {p} (unit {unit})")
        unit += 1

    for prog in mp["all_progs"]:
        for name, u in units.items():
            try:
                prog[name].value = u
            except KeyError:
                pass

    # Fixed per-shader camera pose (no interaction offline). CLI > manifest
    # "camera" > fallback.
    cam = mp["manifest"].get("camera", {}) or {}
    cam_elev = args.cam_elev if args.cam_elev is not None else float(cam.get("elev", 0.58))
    cam_yaw = args.cam_yaw if args.cam_yaw is not None else float(cam.get("yaw", 0.35))
    cam_zoom = args.cam_zoom if args.cam_zoom is not None else float(cam.get("zoom", 1.0))
    cam_move = tuple(args.cam_move) if args.cam_move is not None else (0.0, 0.0, 0.0)
    print(f"  camera: elev={cam_elev} yaw={cam_yaw} zoom={cam_zoom} (fixed, steer=0)")
    print(f"  camera move: {cam_move} (u_cam_move -- live free-fly offsets; "
          f"drives the sky yaw/roll knobs on locked v8)")
    for prog in mp["all_progs"]:
        for name, val in (("u_cam_elev", cam_elev), ("u_cam_yaw", cam_yaw),
                          ("u_cam_zoom", cam_zoom), ("u_cam_move", cam_move),
                          ("u_steer", 0.0)):
            try:
                prog[name].value = val
            except KeyError:
                pass

    if uses:
        if not audio:
            raise SystemExit(
                f"shader uses {sorted(uses)} (live feed) -- needs --audio, not just --features")
        audio_np, sr = F.decode_file(audio)
        print(f"  live feed: {sorted(uses)} ...")
        feed["bands"] = BS.compute_bands(audio_np, sr, args.fps)
        if "stems" in uses:
            feed["stems"] = BS.compute_stems(feed["bands"], args.fps)
    return feed


def write_live_feed(feed, band_frame):
    if feed["bands"] is not None and feed["bands_tex"] is not None:
        i = min(band_frame, feed["bands"].shape[0] - 1)
        feed["bands_tex"].write(np.ascontiguousarray(feed["bands"][i], dtype="f4").tobytes())
    if feed["stems"] is not None and feed["stems_tex"] is not None:
        i = min(band_frame, feed["stems"].shape[0] - 1)
        feed["stems_tex"].write(np.ascontiguousarray(feed["stems"][i], dtype="f4").tobytes())


def load_photo_texture(ctx, path):
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    tex = ctx.texture(img.size, 4, img.tobytes())
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.repeat_x = True
    tex.repeat_y = False
    return tex


def bind_mp_wiring(prog, channels, mp):
    for ch in ("0", "1", "2", "3"):
        spec = channels.get(ch)
        unit = int(ch)
        if spec is not None and spec.get("type") == "buffer":
            buf = mp["buffers"].get(spec["id"])
            tex = buf["tex"][buf["cur"]] if buf is not None else mp["fallback_tex"]
        elif spec is not None and spec.get("type") == "photo":
            path = spec["path"]
            tex = mp["photo_textures"].get(path)
            if tex is None:
                tex = load_photo_texture(mp["ctx"], path)
                mp["photo_textures"][path] = tex
        else:
            tex = mp["fallback_tex"]
        tex.use(location=unit)
        try:
            prog[f"iChannel{ch}"].value = unit
        except KeyError:
            pass


def render_multipass(mp, fbo_main):
    """Fixed render order (A,B,C,D,Image) matching Shadertoy's own actual
    execution order -- a pass sampling a buffer that hasn't rendered yet this
    frame gets that buffer's previous-frame result. Per-frame uniforms
    (iTime, iFrame, iAudio, ...) must already be set on every one of
    mp["all_progs"] by the caller before this runs. Image renders straight
    into fbo_main, same target the single-pass path would have used."""
    w, h = mp["res"]
    for letter in MULTIPASS_LETTERS:
        b = mp["buffers"].get(letter)
        if b is None:
            continue
        bind_mp_wiring(b["prog"], b["wiring"], mp)
        if "iResolution" in b["prog"]:
            b["prog"]["iResolution"].value = (float(w), float(h))
        write_idx = 1 - b["cur"]
        b["fbo"][write_idx].use()
        b["vao"].render(moderngl.TRIANGLE_STRIP)
        b["cur"] = write_idx

    bind_mp_wiring(mp["image_prog"], mp["image_wiring"], mp)
    if "iResolution" in mp["image_prog"]:
        mp["image_prog"]["iResolution"].value = (float(w), float(h))
    fbo_main.use()
    mp["image_vao"].render(moderngl.TRIANGLE_STRIP)


def build_ffmpeg_cmd(ffmpeg, args, w, h, fps, audio):
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    if args.hdr:
        # HDR10 also requires Mastering Display Color Volume + Content Light
        # Level SEI metadata -- without it, many players (this was silently
        # breaking playback on the Pixel and in every local preview) don't
        # trust the PQ/BT.2020 tags and fall back to displaying the raw PQ
        # curve values as if they were flat/linear, which looks exactly like
        # a washed-out, no-true-blacks, blown-highlights mess regardless of
        # how correct the actual rendered pixels are. This ffmpeg build
        # treats these as input-side options (attached to the raw frame
        # stream so they flow through the filter graph as side data), not
        # output/encoder options -- hence placed before -i here. Standard
        # BT.2020/D65 primaries (scaled x50000 per the master_display spec),
        # max/min luminance derived from --hdr-scale (scaled x10000).
        max_nits = int(round(args.hdr_scale))
        # 2026-09-25: MaxCLL/MaxFALL are now overridable, because deriving them
        # from --hdr-scale alone is measurably wrong in BOTH directions and a
        # display tone-maps using exactly these two numbers. Measured on the v8
        # orrery at scale 1100: declared 1100/440, actual 5202/11.1 -- MaxCLL
        # understated 4.7x (so the panel is never told to roll, and the brightest
        # body hard-clips: the "blown red dwarf") and MaxFALL overstated 40x (so
        # the panel compresses a picture that is 92% black). Every earlier attempt
        # to fix the look by tuning gamut/white/roll/sat was tuning the pixels
        # while the side data lied about them.
        cll = int(round(args.hdr_maxcll if args.hdr_maxcll is not None else max_nits))
        fall = int(round(args.hdr_maxfall if args.hdr_maxfall is not None
                         else max_nits * 0.4))
        # 2026-09-26: the MASTERING display peak is the target display's peak
        # (--hdr-peak), not --hdr-scale. It used to be the scale (314), so a
        # file with 1864-nit content claimed a 314-nit master -- another side-
        # data lie a TV may tone-map from. MaxCLL/MaxFALL stay the measured values.
        master_nits = int(round(args.hdr_peak))
        cmd += ["-mastering_display",
                f"G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)"
                f"L({master_nits * 10000},1)",
                "-content_light", f"{cll},{fall}"]
    cmd += ["-f", "rawvideo", "-pixel_format", "rgba64le",
           "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", "-"]
    if audio:
        if args.start:
            cmd += ["-ss", str(args.start)]
        cmd += ["-i", audio]
    cmd += ["-map", "0:v:0"]
    if audio:
        cmd += ["-map", "1:a:0"]
    # VAAPI hardware encode on the dGPU (VCN 5). rgba64le -> p010le (software
    # RGB->YUV, BT.2020nc when HDR) -> hwupload -> av1_vaapi.
    if args.hdr:
        sp = "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=limited"
    else:
        sp = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=limited"
    # Plain bitrate-mode rate control, not QVBR: confirmed 2026-08-27 that
    # av1_vaapi's QVBR path (rc_mode QVBR + global_quality, on this Mesa
    # 25.3.6/gfx1200 driver) hard-faults the VCN encoder ("context is lost")
    # at higher resolutions -- reproduced with a bare synthetic testsrc, no
    # shader/Python involved, and confirmed codec-independent (hevc_vaapi
    # QVBR faults identically). Plain -b:v with no rc_mode/global_quality
    # verified clean at 2856x2856. This drops quality-targeted encoding in
    # favor of a flat bitrate target -- fine for this bitrate, but if a
    # future track looks soft, that's the trade this made.
    cmd += ["-vaapi_device", args.vaapi_device,
            "-vf", f"{sp},format=p010le,hwupload",
            "-c:v", "av1_vaapi",
            "-b:v", args.bitrate,
            "-pix_fmt", "p010le"]
    if args.hdr:
        cmd += ["-color_primaries", "bt2020", "-color_trc", "smpte2084",
                "-colorspace", "bt2020nc", "-color_range", "limited"]
    else:
        cmd += ["-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709", "-color_range", "limited"]
    if audio:
        cmd += ["-c:a", "libopus", "-b:a", "160k"]
    cmd += ["-shortest", args.out]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description="headless dGPU shader -> AV1 mp4")
    ap.add_argument("--shader", required=True)
    ap.add_argument("--audio", help="audio file (any ffmpeg-readable format)")
    ap.add_argument("--features", help="pre-computed features.npz (skip analysis)")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=2410)
    ap.add_argument("-o", "--out", default="out.mp4")
    ap.add_argument("--dtype", default="f4", choices=["f2", "f4"])
    ap.add_argument("--sdr", action="store_true", help="10-bit SDR (BT.709), no PQ")
    ap.add_argument("--hdr-gamut", choices=["native", "rec709"], default="native",
                    help="'native' (default) treats the shader's raw RGB as "
                         "BT.2020 directly -- wide-gamut, vibrant, right for "
                         "these stylized shaders; 'rec709' applies the linear "
                         "Rec.709->BT.2020 matrix for a faithful SDR remaster "
                         "(pulls stylized colours back to standard saturation)")
    ap.add_argument("--hdr-scale", type=float, default=500.0,
                    help="nits that shader value 1.0 maps to (HDR only); higher "
                         "= brighter/punchier midtones. 203 = BT.2408 faithful; "
                         "500 = stylized punch; the highlight roll caps the top")
    ap.add_argument("--hdr-peak", type=float, default=1000.0,
                    help="display peak (nits) the highlight roll eases toward "
                         "(HDR only); raise toward the panel's real peak")
    ap.add_argument("--hdr-roll", choices=["luma", "clip", "none"], default="luma",
                    help="highlight handling above --hdr-peak: 'luma' scales RGB "
                         "toward the peak (hue-exact, but bright coloured bodies "
                         "read grey); 'clip' per-channel clamps (keeps a single "
                         "dominant primary vivid); 'none' encodes raw ratios and "
                         "lets the display tone-map")
    ap.add_argument("--hdr-sat", type=float, default=1.0,
                    help="linear saturation multiplier after the tone map (HDR "
                         "only); PQ lifts the low channels so bright coloured "
                         "bodies read pale -- raise (1.3-2.0) to put the chroma "
                         "back without touching the luminance")
    ap.add_argument("--hdr-maxcll", type=float, default=None,
                    help="MaxCLL (nits) to declare in the HDR side data. Default "
                         "derives it from --hdr-scale, which is WRONG: the shader "
                         "peaks well above value 1.0, so the master understates "
                         "its own peak and the display never rolls the highlights "
                         "-- that is what blows out a bright body. Measure the "
                         "real value (max channel x --hdr-scale) and pass it.")
    ap.add_argument("--hdr-maxfall", type=float, default=None,
                    help="MaxFALL (nits) to declare. Default derives it as 0.4 x "
                         "--hdr-scale, also wrong: this content is ~92%% black, so "
                         "the real frame average is a couple of nits. Overstating "
                         "it makes a display compress an almost entirely dark "
                         "picture.")
    ap.add_argument("--quality", type=int, default=28,
                    help="QVBR quality for av1_vaapi (0-51, lower = better)")
    ap.add_argument("--bitrate", default="16M",
                    help="QVBR target bitrate for av1_vaapi")
    ap.add_argument("--vaapi-device", default="/dev/dri/renderD128",
                    help="VAAPI device for hardware encode (dGPU VCN 5)")
    ap.add_argument("--frames", type=int, default=0, help="render only N frames after --start (0 = all)")
    ap.add_argument("--start", type=float, default=0.0,
                    help="seconds into the track to start at (for quick previews mid-track, "
                         "e.g. right on a bass hit) -- iTime keeps counting from this real "
                         "offset, it does not reset to 0, so the preview reflects the shader's "
                         "actual behavior at that elapsed time")
    ap.add_argument("--channel0", default="noise", help="noise[:seed] or photo:PATH for iChannel0")
    ap.add_argument("--textures-root",
                    default=None,
                    help="base dir for manifest static_textures relative paths "
                         "(default: the renderer checkout, see features.backdrop_dir)")
    # Offline there is no interaction: the camera pose is FIXED for the whole
    # render. Predefined per shader -- CLI flag > manifest "camera" block >
    # fallback. `u_steer` is always 0 (no steering input exists offline).
    ap.add_argument("--cam-elev", type=float, default=None)
    ap.add_argument("--cam-yaw", type=float, default=None)
    ap.add_argument("--cam-zoom", type=float, default=None)
    ap.add_argument("--cam-move", type=float, nargs=3, default=None,
                    help="u_cam_move (r,u,f) -- the live free-fly move offsets. On "
                         "locked v8 these drive the sky yaw/roll knobs, so passing the "
                         "live freefly_state.txt values reproduces the live sky position.")
    ap.add_argument("--ffmpeg", default=DEFAULT_FFMPEG)
    ap.add_argument("--push-adb", metavar="SERIAL", default=None,
                    help="adb push the finished file to this device's /sdcard/Download/ "
                         "(adb serial or host:port, connected first) for a quick look on a phone")
    args = ap.parse_args(argv)
    if args.textures_root is None:
        args.textures_root = F.backdrop_dir()

    # VAAPI aligns the CODED frame up to a multiple of 16. Hand it 2410 and it
    # encodes 2416 and fills the 6 extra columns with luma but ZERO chroma,
    # which decodes as pure green (verified 2026-09-21 on the L'Enfer render:
    # cols 2410-2415 read R=0 G=76 B=0). The container says 2410 but players
    # show the coded frame, so the padding is visible as a green strip down
    # the right edge. Rendering at an aligned size means there is no padding
    # to leak in the first place.
    aw, ah = _align16(args.width), _align16(args.height)
    if (aw, ah) != (args.width, args.height):
        print(f"size {args.width}x{args.height} -> {aw}x{ah} "
              f"(16-aligned for the encoder; avoids green edge padding)")
        args.width, args.height = aw, ah
    args.hdr = not args.sdr

    feats_norm, num_frames, fps, beat = load_features(args.audio, args.features, args.fps)
    start_frame = int(round(args.start * fps))
    end_frame = num_frames if not args.frames else min(num_frames, start_frame + args.frames)
    feats_norm = feats_norm[start_frame:end_frame]
    num_frames = feats_norm.shape[0]
    if num_frames <= 0:
        raise SystemExit(f"--start {args.start}s is past the end of the track")
    print(f"frames={num_frames} (start={args.start}s) fps={fps} res={args.width}x{args.height} "
          f"{'HDR10' if args.hdr else 'SDR10'}")

    idx = find_dgpu_index()
    if idx is None:
        raise SystemExit("FAIL: no hardware GPU found on EGL devices 0-3 (set EGL_DEVICE_INDEX)")
    ctx = moderngl.create_standalone_context(backend="egl", device_index=idx)
    print(f"dGPU device_index={idx}: {ctx.info.get('GL_RENDERER')}")

    multipass = is_multipass(args.shader)

    quad = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
    vbo = ctx.buffer(quad.tobytes())

    if multipass:
        print(f"multipass shader: {args.shader}")
        mp = load_multipass(ctx, vbo, args.shader, args.width, args.height, args.dtype)
        progs = mp["all_progs"]
    else:
        with open(args.shader) as f:
            frag = f.read()
        prog = ctx.program(vertex_shader=VERT, fragment_shader=frag)
        vao = ctx.vertex_array(prog, [(vbo, "2f", "in_vert")])
        progs = [prog]

    # Live-style feed (bands/stems) + manifest static data textures, for
    # shaders written against shader-backdrop's live uniforms. Multipass only.
    feed = prepare_live_feed(ctx, mp, args, args.audio) if multipass else None

    # iChannel0 (unit 0) -- noise or photo; iChannel1..3 stubbed (units 1..3).
    # Multipass buffers get their iChannel0-3 rebound per-pass from the
    # manifest wiring instead (render_multipass()), so this block is
    # single-pass only.
    if not multipass:
        seed = 7
        ch0 = args.channel0
        if ch0.startswith("photo:"):
            print("warning: photo channel0 not implemented; using noise", file=sys.stderr)
            ch0 = "noise"
        if ch0.startswith("noise") and ":" in ch0:
            seed = int(ch0.split(":", 1)[1])
        noise = make_noise(seed)
        tex_ic0 = ctx.texture((512, 512), 4, noise.tobytes())
        tex_ic0.use(location=0)
        stub = b"\x00\x00\x00\xff"
        ctx.texture((1, 1), 4, stub).use(location=1)
        ctx.texture((1, 1), 4, stub).use(location=2)
        ctx.texture((1, 1), 4, stub).use(location=3)

    # iAudio (unit 4): width=num_frames (time), height=num_features, f32 single
    # channel. Shared by every program, single-pass or multipass -- moderngl
    # auto-assigns iChannel0..3/iAudio to units 0..4 in declaration order
    # (same order wrap_shader.py's header always emits them in), so this one
    # binding is visible to every buffer/image program without per-program setup.
    num_feats = feats_norm.shape[1]
    audio_tex = np.ascontiguousarray(feats_norm.T, dtype="f4")  # (num_feats, frames)
    tex_ia = ctx.texture((num_frames, num_feats), 1, audio_tex.tobytes(), dtype="f4")
    tex_ia.use(location=4)

    for prog in progs:
        if "iResolution" in prog:
            prog["iResolution"].value = (args.width, args.height)
        if "iAudio" in prog:
            # Sampler uniforms default to texture unit 0 in OpenGL -- there is
            # no such thing as moderngl "auto-assigning" units by declaration
            # order, that was a wrong assumption. Without this, iAudio reads
            # whatever's actually bound to unit 0 (buffer A's iChannel0
            # fallback texture) instead of the real feature texture, so every
            # audio()-driven effect silently read a constant, non-reactive
            # value the whole time.
            prog["iAudio"].value = 4
        if "iAudioFrames" in prog:
            prog["iAudioFrames"].value = float(num_frames)
        if "iMouse" in prog:
            prog["iMouse"].value = (0.0, 0.0, 0.0, 0.0)
        if "iDate" in prog:
            prog["iDate"].value = (2026.0, 8.0, 16.0, 0.0)
        if "iTimeDelta" in prog:
            prog["iTimeDelta"].value = 1.0 / fps
        if "iChannelResolution" in prog:
            # Some shaders only ever index a constant element (e.g. iChannelResolution[0]),
            # so the GLSL compiler can shrink the introspected array below the declared
            # size 4 -- a shader-dependent quirk, not something to let crash the batch.
            try:
                prog["iChannelResolution"].value = [(512, 512, 0)] * 4
            except Exception:
                try:
                    prog["iChannelResolution"].value = (512, 512, 0)
                except Exception as e:
                    print(f"  warning: could not set iChannelResolution ({e}), skipping")
        if "iChannelTime" in prog:
            try:
                prog["iChannelTime"].value = [0.0] * 4
            except Exception:
                try:
                    prog["iChannelTime"].value = 0.0
                except Exception as e:
                    print(f"  warning: could not set iChannelTime ({e}), skipping")

    # main render FBO (float32 by default, linear HDR -- values may exceed 1.0.
    # not time-constrained here, so no reason to accept f2 quantization on the
    # scene's own extreme dynamic range -- event horizon glow vs. faint stars).
    tex_main = ctx.texture((args.width, args.height), 4, dtype=args.dtype)
    fbo_main = ctx.framebuffer(color_attachments=[tex_main])

    # post FBO (float32, PQ-encoded 0..1 output) + PQ pass program -- same
    # reasoning, f2 here would re-quantize the PQ curve's dark-region precision.
    # GL_RGBA16UI. Measured on this GPU at 2416^2, per frame:
    #   f4 readback 82.4 ms (1.13 GB/s) + 34.7 ms numpy  = 117.1 ms
    #   f2 readback 27.4 ms              + 70.9 ms numpy = 98.3 ms
    #     (and a DIRECT clip/scale on float16 costs 333 ms and OVERFLOWS --
    #      65535 is outside half-float range. Do not "simplify" to that.)
    #   u2 readback 27.5 ms              + 0 ms          = 27.5 ms
    # The readback was never limited by the bus (PCIe 4.0 x16 is ~25 GB/s) --
    # it was limited by asking the driver to detile and convert a float
    # surface, and then converting again on the CPU. With an integer target
    # the post shader quantises once and the bytes are already what ffmpeg
    # wants.
    tex_post = ctx.texture((args.width, args.height), 4, dtype="u2")
    fbo_post = ctx.framebuffer(color_attachments=[tex_post])
    post_prog = ctx.program(vertex_shader=VERT, fragment_shader=POST_FRAG)
    post_vao = ctx.vertex_array(post_prog, [(vbo, "2f", "in_vert")])
    post_prog["uHdrScale"].value = args.hdr_scale / 10000.0 if args.hdr else 0.0
    post_prog["uHdrPeak"].value = args.hdr_peak / 10000.0
    post_prog["uGamutNative"].value = 1.0 if args.hdr_gamut == "native" else 0.0
    post_prog["uRollMode"].value = {"none": 0.0, "luma": 1.0, "clip": 2.0}[args.hdr_roll]
    post_prog["uSat"].value = args.hdr_sat

    import atexit
    atexit.register(report_phases)
    cmd = build_ffmpeg_cmd(args.ffmpeg, args, args.width, args.height, fps, args.audio)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # One consumer thread, so ordering is preserved by the queue itself.
    frame_q = queue.Queue(maxsize=4)
    writer_err = []

    def _writer():
        while True:
            buf = frame_q.get()
            if buf is None:
                break
            try:
                proc.stdin.write(buf)
            except Exception as e:      # ffmpeg died; stop, do not spin
                writer_err.append(e)
                break

    writer = threading.Thread(target=_writer, daemon=True, name="frame-writer")
    writer.start()

    try:
        for t in range(num_frames):
            for prog in progs:
                if "iTime" in prog:
                    prog["iTime"].value = (start_frame + t) / fps
                if "iFrame" in prog:
                    prog["iFrame"].value = t
                if "u_bass" in prog:
                    prog["u_bass"].value = float(feats_norm[t, 2])
                if beat is not None:
                    bt = min(start_frame + t, len(beat["count"]) - 1)
                    if "u_beat_count" in prog:
                        prog["u_beat_count"].value = float(beat["count"][bt])
                    if "u_beat_phase" in prog:
                        prog["u_beat_phase"].value = float(beat["phase"][bt])
                    if "u_tbeat" in prog:
                        prog["u_tbeat"].value = float(beat["tbeat"][bt])
                    if "u_beat_conf" in prog:
                        # Exact by construction -- see _beat_clock_arrays.
                        prog["u_beat_conf"].value = 1.0
            if feed is not None:
                write_live_feed(feed, start_frame + t)
            if _PHASE_LOG:
                ctx.finish(); _ph_t = time.perf_counter()
            if multipass:
                render_multipass(mp, fbo_main)
            else:
                fbo_main.use()
                vao.render(moderngl.TRIANGLE_STRIP)
            fbo_post.use()
            tex_main.use(location=0)
            post_vao.render(moderngl.TRIANGLE_STRIP)
            if _PHASE_LOG:
                # finish() forces the GPU to actually complete before we time
                # the readback, otherwise the render's cost hides inside it.
                ctx.finish(); _ph_r = time.perf_counter()
            # Deliberately f4, NOT u2. Reading a FLOATING-POINT framebuffer as
            # an integer type does not apply the normalised mapping -- the
            # float is converted directly, so everything below 1.0 truncates
            # to zero and the whole frame comes out black. Measured that the
            # hard way 2026-09-21. The conversion therefore stays in numpy,
            # but it is much cheaper now than it was: the Y flip moved into
            # POST_FRAG, so these operate on CONTIGUOUS memory instead of the
            # reversed-stride view `px[::-1]` used to produce.
            raw = fbo_post.read(components=4, dtype="u2", alignment=1)
            if _PHASE_LOG:
                _ph_rd = _ph_cv = time.perf_counter()
            # Queued, not written inline: the encoder's backpressure was 38.6%
            # of every frame, and stalling the GPU on it is pure waste. The
            # queue is short so we cannot run far ahead and hoard frames.
            frame_q.put(raw)
            if _PHASE_LOG:
                _ph_w = time.perf_counter()
                _phases["render"] += _ph_r - _ph_t
                _phases["readback"] += _ph_rd - _ph_r
                _phases["convert"] += _ph_cv - _ph_rd
                _phases["pipe"] += _ph_w - _ph_cv
                _phases["n"] += 1
            if t % 60 == 0:
                print(f"  frame {t}/{num_frames}")
    finally:
        frame_q.put(None)
        writer.join(timeout=120)
        proc.stdin.close()
        if writer_err:
            print(f"frame writer failed: {writer_err[0]}", file=sys.stderr)
        proc.wait()

    if proc.returncode != 0:
        print(f"ffmpeg failed (rc={proc.returncode})")
        return 1
    print(f"done -> {args.out}")

    if args.push_adb:
        dev = args.push_adb
        dest = f"/sdcard/Download/{os.path.basename(args.out)}"
        subprocess.run(["adb", "connect", dev], check=False,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        r = subprocess.run(["adb", "-s", dev, "push", args.out, dest])
        if r.returncode == 0:
            print(f"pushed -> {dev}:{dest}")
        else:
            print(f"adb push failed -- is {dev} reachable?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
