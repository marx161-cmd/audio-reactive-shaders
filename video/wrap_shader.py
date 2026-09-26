#!/usr/bin/env python3
"""wrap_shader.py — turn a Shadertoy-style mainImage() .frag into a GLSL 330
fragment shader the offline renderer can compile, with an injected audio-feature
sampler (iAudio) so wiring audio reactivity is a one-liner per shader.

The feature texture (from features.py) is (width=num_frames, height=13) float32,
single channel: row 0..10 = rms, sub, bass, lowmid, mid, highmid, presence,
brilliance, centroid, flux, onset; row 11..12 = bass_accum, onset_accum. In a
shader:

    float bass = audio(FEAT_BASS);   // 0..1 loudness in the bass band, this frame
    float kick = audio(FEAT_ONSET);  // 0..1 transient/kick energy, this frame

FEAT_BASS_ACCUM / FEAT_ONSET_ACCUM are NOT 0..1 -- they're an unbounded
running integral (seconds) of the bass/onset envelope, for driving motion
*speed* without the classic "multiply audio level straight into iTime" bug:
that construction rescales the WHOLE elapsed-time clock by a live coefficient,
so any envelope wobble gets amplified by however many seconds have already
played -- mild at the start of a track, violent by the end. Use the additive
form instead: `iTime*base_rate + mod*audio(FEAT_BASS_ACCUM)` nudges the phase
forward with the beat without ever rescaling elapsed time.

Usage:
    python wrap_shader.py in.frag -o out.glsl
    python wrap_shader.py shaderwallpaper_bestof/ -o wrapped/      # batch
"""

import argparse
import os
import re
import sys

FEATURE_NAMES = ["rms", "sub", "bass", "lowmid", "mid", "highmid",
                 "presence", "brilliance", "centroid", "flux", "onset",
                 "bass_accum", "onset_accum"]
FEATURE_COUNT = len(FEATURE_NAMES)


def make_header():
    defines = "\n".join(
        f"#define FEAT_{n.upper()} {i}.0" for i, n in enumerate(FEATURE_NAMES)
    )
    return f'''#version 330
// ---- header injected by wrap_shader.py ----
out vec4 outColor;

uniform vec2  iResolution;
uniform float iTime;
uniform int   iFrame;
uniform vec4  iMouse;
uniform vec4  iDate;
uniform float iTimeDelta;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform vec3  iChannelResolution[4];
uniform float iChannelTime[4];

// audio feature texture (from features.py): width = iAudioFrames (time),
// height = {FEATURE_COUNT} (features), single channel float32.
uniform sampler2D iAudio;
uniform float    iAudioFrames;

{defines}

// sample feature `f` (a FEAT_* row index) at the current frame
float audio(float f) {{
    vec2 uv = vec2((float(iFrame) + 0.5) / max(iAudioFrames, 1.0),
                   (f + 0.5) / {FEATURE_COUNT}.0);
    return texture(iAudio, uv).r;
}}
'''

MAIN_WRAPPER = '''
void main() {
    vec4 fragColor = vec4(0.0);
    mainImage(fragColor, gl_FragCoord.xy);
    // No global post-process here on purpose: each shader now drives its own
    // internal parameters (warp speed, distortion amount, hue) via audio(),
    // tailored to what that shader actually is. A flat brightness/flash
    // layered on top of that just reads as flicker again.
    outColor = fragColor;
}
'''


def wrap(src):
    lines = [l.rstrip("\\") for l in src.splitlines()]  # drop stray line continuations
    # strip any #version / precision the source carried; we inject our own
    body = "\n".join(
        l for l in lines
        if not l.lstrip().startswith("#version")
        and not l.lstrip().startswith("precision")
    )
    has_main = re.search(r"\bvoid\s+main\s*\(", body) is not None
    out = make_header() + "\n" + body
    if not has_main:
        if "mainImage" in body:
            out += MAIN_WRAPPER
        else:
            sys.stderr.write("warning: no mainImage/main found; wrapping as-is\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help=".frag file or directory")
    ap.add_argument("-o", "--out", help="output file (or dir for batch)")
    args = ap.parse_args(argv)

    if os.path.isdir(args.input):
        outdir = args.out or "wrapped"
        os.makedirs(outdir, exist_ok=True)
        for fn in sorted(os.listdir(args.input)):
            if not fn.endswith(".frag"):
                continue
            with open(os.path.join(args.input, fn)) as f:
                src = f.read()
            outp = os.path.join(outdir, fn[: -len(".frag")] + ".glsl")
            with open(outp, "w") as f:
                f.write(wrap(src))
            print(f"wrapped {fn} -> {outp}")
    else:
        with open(args.input) as f:
            src = f.read()
        out = wrap(src)
        if args.out:
            with open(args.out, "w") as f:
                f.write(out)
            print(f"wrapped {args.input} -> {args.out}")
        else:
            sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
