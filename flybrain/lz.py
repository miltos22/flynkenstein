import ctypes
import math
import os
import subprocess
from pathlib import Path
import numpy as np
_SRC = Path(__file__).resolve().parent / 'csrc' / 'lz76.c'
_LIB = Path(__file__).resolve().parents[1] / 'cache' / 'liblz76.so'
_lib = None

def _load():
    global _lib
    if _lib is None:
        if not _LIB.exists() or _LIB.stat().st_mtime < _SRC.stat().st_mtime:
            _LIB.parent.mkdir(exist_ok=True)
            subprocess.run(['gcc', '-O2', '-shared', '-fPIC', '-o', str(_LIB), str(_SRC)], check=True)
        _lib = ctypes.CDLL(str(_LIB))
        for f in (_lib.lz76, _lib.lz76_linear):
            f.restype = ctypes.c_longlong
            f.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
    return _lib

def lz76(bits):
    b = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    c = int(_load().lz76_linear(b.ctypes.data, b.size))
    if c < 0:
        raise MemoryError(f'lz76_linear could not allocate for {b.size} bits')
    return c

def lz76_quadratic(bits):
    b = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    return int(_load().lz76(b.ctypes.data, b.size))

def pci_lz(S):
    S = np.asarray(S, dtype=np.uint8)
    L = S.size
    p = float(S.mean())
    if L == 0 or p in (0.0, 1.0):
        return 0.0
    H = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return lz76(S.T.reshape(-1)) * math.log2(L) / (L * H)
