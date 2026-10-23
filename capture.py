"""Record the full shader canvas plus EasyEffects-processed audio to a video.

See scope.md 2026-09-23 "Music-session video capture" (and the later
"capture fixes + silent mode" entry). Short version: the 2410x2410 canvas is
never shown whole on any output (every head takes a crop, see crop_for_spec),
and it does not fit in the free desktop region either (3000x1280), so there is
nothing on screen to x11grab. Frames are read back from canvas_fbo in-process
and piped to ffmpeg, which pulls the audio itself and does the muxing.

AUDIO SOURCE, READ THIS BEFORE CHANGING IT: the tap is
`ee_processed_sink.monitor` -- the POST-effects sink. `easyeffects_sink` is the
DEFAULT sink that applications play into, i.e. pre-effects, and recording that
one gets you audio with none of the processing applied. A relay in this homelab
already shipped that exact bug once (fixed 2026-08-19).

TIMING, READ THIS BEFORE CHANGING IT: rawvideo over a pipe carries NO
timestamps ffmpeg will honour. `-use_wallclock_as_timestamps 1` is silently
ignored by the rawvideo demuxer (verified 2026-09-23: 40 frames fed over 4s
produced a 1.33s file), which stamps frames purely by frame COUNT at the
declared -r. The first version trusted it, so when the capture readback pulled
the render loop from 30 to ~15 fps the muxed video ran ~2x fast and ended at
~2 min while the live audio kept going -- the "frozen after 2 min" report.
grab()/_emit() now earn the declared rate on the wall clock instead: the newest
frame is duplicated until the stream is back in step, so N frames always equal
N/fps seconds of real time whatever the render loop is doing.

Driven by `capture_state.txt` next to renderer.py, polled like camera_state.txt
and friends, so arming it is a one-line write and needs no IPC:

    REC /abs/path/to/output.mp4      start (or keep) recording to that file
    <file absent, empty, or IDLE>    stop and finalise

`capture_session.py` is the MPRIS driver that writes this file.
"""
import os
import signal
import subprocess
import sys
import time

# The post-effects EasyEffects sink. See the module docstring.
AUDIO_SOURCE = "ee_processed_sink.monitor"

# Encoder placement. DEFAULT IS CPU (libx264) -- see vaapi_device() for the
# live-loop measurements that made that the choice. The dGPU/iGPU VAAPI lanes
# below remain available as opt-in overrides.
#
# Pinned by PCI address, NOT by renderD number: those are handed out in probe
# order and can swap across a reboot. As of 2026-09-23 renderD128 happens to be
# the RX 9060 XT and renderD129 the 760M -- the opposite of the usual
# assumption that the iGPU comes first -- so a hardcoded renderD128 is right
# today and silently wrong the day the order changes.
DGPU_PCI = "0000:03:00.0"                         # RX 9060 XT
IGPU_PCI = "0000:11:00.0"                         # Radeon 760M
VAAPI_DEVICE = os.environ.get("SHADER_CAPTURE_VAAPI_DEVICE") or \
    f"/dev/dri/by-path/pci-{DGPU_PCI}-render"


def vaapi_device():
    """The dGPU render node, or None to use libx264 instead.

    SHADER_CAPTURE_ENCODER=cpu forces the libx264 path. Measured 2026-09-23,
    live render loop during a 2410x2410 recording:
      dGPU h264_vaapi ~15-17 fps (shader and VCN fight over the dGPU)
      iGPU h264_vaapi ~25-29 fps (but competes with Sunshine's streams)
      CPU libx264 ultrafast ~100 fps standalone, uses neither GPU
    so CPU is the default lane and the GPU choices are opt-in overrides.
    """
    if os.environ.get("SHADER_CAPTURE_ENCODER", "cpu").lower() == "cpu":
        return None
    if os.path.exists(VAAPI_DEVICE):
        return VAAPI_DEVICE
    print(f"capture: {VAAPI_DEVICE} missing, falling back to libx264",
          file=sys.stderr)
    return None


def _encoder_args(vaapi):
    """Video encoder args. VAAPI keeps the CPU free; libx264 is the fallback."""
    if vaapi:
        # vflip first: glReadPixels hands back rows bottom-up. The flip has to
        # happen on the CPU side of hwupload, so it is part of the same chain.
        return ["-vf", "vflip,format=nv12,hwupload",
                "-c:v", "h264_vaapi", "-qp", "22"]
    return ["-vf", "vflip", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "20", "-pix_fmt", "yuv420p"]


class CanvasCapture:
    """Reads canvas_fbo into a pair of PBOs and pipes frames to ffmpeg.

    Double-buffered so frame N's transfer overlaps frame N+1's render, but note
    the readback still costs ~30 ms/frame in this process (measured 2026-09-23),
    which drops the render loop to ~15 fps while recording -- that is why the
    timeline has to be re-derived from the wall clock in _emit() rather than
    trusted from the frame count.
    """

    def __init__(self, ctx, fbo, width, height, state_file, fps=30, vaapi=True):
        self.ctx, self.fbo = ctx, fbo
        self.w, self.h, self.fps = width, height, fps
        self.state_file = state_file
        self.vaapi = vaapi
        self.proc = None
        self.path = None
        self.bufs = None
        self.slot = 0
        self.primed = False
        self.frames = 0
        self.dropped = 0
        self._mtime = None
        self._next_out = None      # wall-clock time the next frame is due

    # ---- state file -------------------------------------------------------
    def poll(self):
        """Start/stop according to capture_state.txt. Cheap; call every frame."""
        try:
            try:
                mtime = os.path.getmtime(self.state_file)
            except OSError:
                # Absent = disarmed. (Removing the file is how capture_session
                # signals stop, so this branch runs on every normal finish.)
                if self.proc is not None:
                    self.stop()
                self._mtime = None
                return
            if mtime == self._mtime:
                return
            self._mtime = mtime
            try:
                with open(self.state_file) as fh:
                    line = fh.readline().strip()
            except OSError:
                return
            if line.startswith("REC "):
                want = line[4:].strip()
                if self.proc is None or want != self.path:
                    self.stop()
                    self.start(want)
            elif self.proc is not None:
                self.stop()
        except Exception as exc:
            # This is called every frame from the render loop; nothing here may
            # ever raise, or a capture glitch takes the live wallpaper down
            # (which is exactly what happened on 2026-09-23: an unhandled
            # TimeoutExpired out of stop() crashed the renderer).
            print(f"capture: poll error: {exc}", file=sys.stderr)

    # ---- lifecycle --------------------------------------------------------
    def start(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        dev = vaapi_device() if self.vaapi else None
        use_vaapi = dev is not None
        cmd = ["ffmpeg", "-loglevel", "error", "-y"]
        if use_vaapi:
            cmd += ["-vaapi_device", dev]
        cmd += [
            # rawvideo's timeline is frame-count x -r (see the module docstring),
            # so the input is declared at the target rate and grab()/_emit()
            # does the work of delivering that many frames per second.
            "-thread_queue_size", "1024",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.w}x{self.h}", "-r", str(self.fps),
            "-i", "pipe:0",
            # Pull the audio from the post-effects monitor. thread_queue_size +
            # nobuffer stop the capture queue overflowing while the main thread
            # is busy with 17 MB video frames; that overflow was the source of
            # the audio dropouts/crackle (the long-lived musiccast / oppo
            # relays pass the same flags on the very same source).
            "-thread_queue_size", "512", "-fflags", "+nobuffer",
            "-f", "pulse", "-i", AUDIO_SOURCE,
        ]
        cmd += _encoder_args(use_vaapi)
        cmd += ["-c:a", "aac", "-b:a", "256k", "-fps_mode", "cfr", path]

        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except OSError as exc:
            print(f"capture: cannot start ffmpeg: {exc}", file=sys.stderr)
            self.proc = None
            return
        if self.bufs is None:
            nbytes = self.w * self.h * 3
            self.bufs = [self.ctx.buffer(reserve=nbytes) for _ in range(2)]
        self.path = path
        self.slot = 0
        self.primed = False
        self.frames = 0
        self.dropped = 0
        self._next_out = None
        print(f"capture: recording {self.w}x{self.h} + {AUDIO_SOURCE} -> {path}",
              file=sys.stderr)

    def grab(self):
        """Queue this frame. Call once per rendered frame, after the canvas."""
        if self.proc is None:
            return
        try:
            cur, prev = self.bufs[self.slot], self.bufs[1 - self.slot]
            self.fbo.read_into(cur, components=3, dtype="f1")
            if self.primed:                  # prev finished transferring last frame
                self._emit(prev.read())
            self.primed = True
            self.slot = 1 - self.slot
        except (BrokenPipeError, OSError) as exc:
            print(f"capture: ffmpeg went away ({exc})", file=sys.stderr)
            self.stop()
        except Exception as exc:
            print(f"capture: grab error: {exc}", file=sys.stderr)
            self.stop()

    def _emit(self, data):
        """Write `data` enough times to hold the declared fps on the wall clock.

        rawvideo gets no usable timestamps, so when the render loop supplies
        fewer than fps frames per second the stream silently runs fast and
        ends early (the freeze). Duplicating the newest frame until the next
        slot is due pins the timeline to real time instead. If the loop is
        *faster* than the target the frame is simply dropped.
        """
        now = time.monotonic()
        period = 1.0 / self.fps
        if self._next_out is None:
            self._next_out = now
        # Recover from a long stall (e.g. a shader reload) without emitting a
        # runaway burst of duplicates to catch up.
        if self._next_out < now - period:
            self._next_out = now - period
        while self._next_out <= now:
            self.proc.stdin.write(data)
            self.frames += 1
            self._next_out += period

    def stop(self):
        if self.proc is None:
            return
        proc, path, frames = self.proc, self.path, self.frames
        self.proc = None
        self.path = None
        self.primed = False
        self._next_out = None
        try:
            proc.stdin.close()
        except OSError:
            pass
        # The pulse input is a live source that never ends by itself, so
        # closing stdin (video EOF) does NOT make ffmpeg exit -- it waits
        # forever for audio. The old code called wait(timeout=30) here, which
        # raised TimeoutExpired, propagated out of poll() and crashed the whole
        # renderer. Signal it to finalise the container instead, and only kill
        # as a last resort.
        try:
            proc.send_signal(signal.SIGINT)
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print(f"capture: saved {path} ({frames} frames)", file=sys.stderr)
