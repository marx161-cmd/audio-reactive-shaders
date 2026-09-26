"""BC6H (unsigned half) encoder: ctypes wrapper around Intel ISPC Texture Compressor
(tools/ispc_texcomp, built with: make -f Makefile.linux ISPC=/usr/bin/ispc
ISPC_ARCH=x86-64 ISPC_TARGETS=avx2-i32x8 ISPC_OBJS= ARCH_CXXFLAGS=-march=znver4).
Measured on a 4096^2 Milky Way crop, profile basic: flux 1.008, median rel err
~5% (stars/glow/darks), 20 Mtexel/s on the 8600G. slow = same quality, 3.6x slower.
Mesa's upload-time encoder lost 15% flux / 45% star error -- do not use it."""
import ctypes, os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ispc_texcomp", "build", "libispc_texcomp.so"))
class Surf(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("width", ctypes.c_int32), ("height", ctypes.c_int32), ("stride", ctypes.c_int32)]
class Set(ctypes.Structure):
    _fields_ = [("slow_mode", ctypes.c_bool), ("fast_mode", ctypes.c_bool), ("r1", ctypes.c_int), ("r2", ctypes.c_int), ("skip", ctypes.c_int)]
_lib.CompressBlocksBC6H.argtypes = [ctypes.POINTER(Surf), ctypes.c_void_p, ctypes.POINTER(Set)]
_lib.CompressBlocksBC6H.restype = None
def encode(rgb16, profile="basic", threads=os.cpu_count()):
    """rgb16: (H, W, 3) float16, H and W multiples of 4. Returns BC6H blocks, bytes."""
    H, W, _ = rgb16.shape
    rgba = np.empty((H, W, 4), np.float16); rgba[..., :3] = rgb16; rgba[..., 3] = 1
    out = np.empty((H // 4, W // 4, 16), np.uint8)
    st = Set(); getattr(_lib, f"GetProfile_bc6h_{profile}")(ctypes.byref(st))
    band = 64
    def job(y):
        src = rgba[y:y + band]
        s = Surf(src.ctypes.data, W, src.shape[0], W * 8)
        _lib.CompressBlocksBC6H(ctypes.byref(s), ctypes.c_void_p(out[y // 4:(y + band) // 4].ctypes.data), ctypes.byref(st))
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(job, range(0, H, band)))
    return out.tobytes()
