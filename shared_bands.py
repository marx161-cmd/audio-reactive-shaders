"""Python side of the shared_bands.h contract. Keep byte-for-byte in sync
with that header -- this is the single source of truth for the struct
layout on the Python side, mirrored manually on the C side.
"""
import mmap
import os
import struct
import time

SHM_NAME = "shader_bands"
SHM_PATH = f"/dev/shm/{SHM_NAME}"
N_BANDS = 120  # 27.5Hz-28160Hz -- must match shared_bands.h exactly

# '<' = little-endian, no padding (matches the C header's #pragma pack(1))
# d = timestamp, I I = generation/n_bands, Nf = bands, 11f = meta features
_META_FIELDS = (
    "rms", "sub", "bass", "lowmid", "mid", "highmid",
    "presence", "brilliance", "centroid", "flux", "onset",
)
_FORMAT = f"<dII{N_BANDS}f{len(_META_FIELDS)}f"
STRUCT_SIZE = struct.calcsize(_FORMAT)
_struct = struct.Struct(_FORMAT)


class BandsWriter:
    """Writer side: creates/truncates the shm segment, writes frames with
    the even/odd generation counter for torn-read safety."""

    def __init__(self):
        fd = os.open(SHM_PATH, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, STRUCT_SIZE)
        self._mm = mmap.mmap(fd, STRUCT_SIZE)
        os.close(fd)
        self._generation = 0

    def write(self, bands, meta: dict):
        assert len(bands) == N_BANDS
        self._generation += 1  # now odd: mid-write
        self._mm[:] = _struct.pack(
            time.monotonic(),
            self._generation,
            N_BANDS,
            *[float(b) for b in bands],
            *[float(meta.get(k, 0.0)) for k in _META_FIELDS],
        )
        self._generation += 1  # now even: stable
        # patch just the generation field to the final even value without
        # re-serializing the whole payload
        self._mm[8:12] = struct.pack("<I", self._generation)

    def close(self):
        self._mm.close()


class BandsReader:
    """Reader side: opens the existing shm segment read-only, retries on
    torn reads."""

    def __init__(self):
        fd = os.open(SHM_PATH, os.O_RDONLY)
        self._mm = mmap.mmap(fd, STRUCT_SIZE, prot=mmap.PROT_READ)
        os.close(fd)

    def read(self):
        for _ in range(8):
            raw = bytes(self._mm)
            unpacked = _struct.unpack(raw)
            gen_first = unpacked[1]
            gen_check = struct.unpack("<I", raw[8:12])[0]
            if gen_first % 2 == 0 and gen_first == gen_check:
                timestamp, generation, n_bands = unpacked[0], unpacked[1], unpacked[2]
                bands = unpacked[3:3 + N_BANDS]
                meta = dict(zip(_META_FIELDS, unpacked[3 + N_BANDS:]))
                return {
                    "timestamp": timestamp,
                    "n_bands": n_bands,
                    "bands": bands,
                    **meta,
                }
        return None  # gave up after 8 retries, caller should reuse last good frame

    def close(self):
        self._mm.close()
