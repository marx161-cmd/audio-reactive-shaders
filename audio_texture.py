"""Python side of shader_audio_tex_shm.h's contract (from the native
EasyEffects plugin, src/shader_audio_tex_shm.h in easyeffects-shader-bands --
keep byte-for-byte in sync, same rule as shared_bands.py/shared_bands.h).

Real Shadertoy-style audio texture: 512 columns, two 0..1 rows (freq,
wave -- wave centered on 0.5 for silence, not 0.0, matching how real
Shadertoy audio textures store waveform data). Read-only from this side --
only the native plugin writes it.
"""
import mmap
import os
import struct

SHM_NAME = "shader_audio_tex"
SHM_PATH = f"/dev/shm/{SHM_NAME}"
WIDTH = 512

# '<' = little-endian, no padding (matches the C header's #pragma pack(1))
_FORMAT = f"<I{WIDTH}f{WIDTH}f"
STRUCT_SIZE = struct.calcsize(_FORMAT)
_struct = struct.Struct(_FORMAT)


class AudioTexReader:
    """Opens the existing shm segment read-only, retries on torn reads --
    same generation-counter protocol as shared_bands.py's BandsReader."""

    def __init__(self):
        fd = os.open(SHM_PATH, os.O_RDONLY)
        self._mm = mmap.mmap(fd, STRUCT_SIZE, prot=mmap.PROT_READ)
        os.close(fd)

    def read(self):
        for _ in range(8):
            raw = bytes(self._mm)
            gen_first = struct.unpack("<I", raw[0:4])[0]
            unpacked = _struct.unpack(raw)
            gen_check = unpacked[0]
            if gen_first % 2 == 0 and gen_first == gen_check:
                freq = unpacked[1:1 + WIDTH]
                wave = unpacked[1 + WIDTH:1 + 2 * WIDTH]
                return {"freq": freq, "wave": wave}
        return None  # gave up after 8 retries, caller should reuse last good frame

    def close(self):
        self._mm.close()
