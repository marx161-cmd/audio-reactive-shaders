#!/usr/bin/env bash
# Render a music video: a track through a shader, offline, as HDR10 AV1.
#
#   video/render.sh <track> [output.mp4] [size] [fps]
#
#   track   any ffmpeg-readable audio or video file
#   output  default: video/rendered/<shader>/<track>.mp4
#   size    square side in px, default 2410 (the live canvas)
#   fps     default 30
#
# Env:
#   SHADER       shader dir name under shaders/ (default kerr-black-hole-visualizer.mp)
#   MODE         hdr (default) or sdr (10-bit Rec.709)
#   HDR_SCALE    nits for shader value 1.0 (default 314)
#   HDR_PEAK     mastering display peak / roll-off target (default 1000)
#   HDR_ROLL     luma | clip | none (default luma)
#   HDR_SAT, HDR_GAMUT (native|rec709), BITRATE
#   HDR_MAXCLL / HDR_MAXFALL   declared HDR10 content light levels. The defaults
#                (2163 / 96) were MEASURED for the Kerr shader at the defaults
#                above; re-measure if you change the grade or the shader.
#   EXTRA_ARGS   extra render_video.py flags, e.g. "--start 45 --frames 300"
#                for a 10 s preview, or "--push-adb <serial>" to copy the result
#                to a phone
#   STOP_UNIT    systemd --user unit to stop during the render and restart
#                afterwards (default audio-reactive-shaders.service if it is
#                running: the live backdrop and the render both need the GPU
#                memory for the Kerr tables)
#
# Offline there is no frame budget: render_video.py prepends "#define OFFLINE 1",
# which raises the shader's sample counts. The camera pose and sky position are
# taken from the live renderer's camera_state.txt / freefly_state.txt if present.
set -euo pipefail
VIDEO="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$VIDEO")"
cd "$VIDEO"

TRACK="${1:?usage: video/render.sh <track> [output.mp4] [size] [fps]}"
SHADER_NAME="${SHADER:-kerr-black-hole-visualizer.mp}"
BASE="$(basename "$TRACK")"; BASE="${BASE%.*}"
SIZE="${3:-2410}"
FPS="${4:-30}"
OUT="${2:-$VIDEO/rendered/${SHADER_NAME%.mp}/${BASE}.mp4}"
MODE="${MODE:-hdr}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PY="${PYTHON:-python3}"

if [[ "$MODE" == "hdr" ]]; then
  MODE_ARGS="--hdr-gamut ${HDR_GAMUT:-native} --hdr-scale ${HDR_SCALE:-314} \
--hdr-peak ${HDR_PEAK:-1000} --hdr-roll ${HDR_ROLL:-luma} --hdr-sat ${HDR_SAT:-1.1} \
--hdr-maxcll ${HDR_MAXCLL:-2163} --hdr-maxfall ${HDR_MAXFALL:-96} \
--bitrate ${BITRATE:-20M}"
else
  MODE_ARGS="--sdr --bitrate ${BITRATE:-16M}"
fi

read -r CE CY CZ _ < <(cat "$ROOT/camera_state.txt" 2>/dev/null || echo "0.58 0.35 1.0")
read -r _FY _FP _FR MR MU MF < <(cat "$ROOT/freefly_state.txt" 2>/dev/null || echo "0 0 0 0 0 0")

UNIT="${STOP_UNIT:-audio-reactive-shaders.service}"
STOPPED=0
if systemctl --user is-active --quiet "$UNIT" 2>/dev/null; then
  systemctl --user stop "$UNIT"; STOPPED=1
fi
trap '[[ $STOPPED == 1 ]] && systemctl --user start "$UNIT"' EXIT

mkdir -p "$(dirname "$OUT")"
echo "shader: $SHADER_NAME   output: $OUT (${SIZE}x${SIZE} @ ${FPS} fps, $MODE)"
env -u DISPLAY -u WAYLAND_DISPLAY PYTHONUNBUFFERED=1 "$PY" render_video.py \
  --shader "$ROOT/shaders/$SHADER_NAME" --audio "$TRACK" \
  --fps "$FPS" --width "$SIZE" --height "$SIZE" \
  --cam-elev "${CE:-0.58}" --cam-yaw "${CY:-0.35}" --cam-zoom "${CZ:-1.0}" \
  --cam-move "${MR:-0}" "${MU:-0}" "${MF:-0}" \
  $MODE_ARGS ${EXTRA_ARGS} \
  -o "$OUT" </dev/null
