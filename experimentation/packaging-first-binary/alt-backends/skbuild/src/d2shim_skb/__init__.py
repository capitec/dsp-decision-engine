"""Plain shared library (no Python.h, no PyInit) loaded with ctypes."""
import ctypes, glob, os
_here = os.path.dirname(__file__)
_cands = glob.glob(os.path.join(_here, "*nashim*.so")) + glob.glob(os.path.join(_here, "*nashim*.dylib")) + glob.glob(os.path.join(_here, "*nashim*.dll")) + glob.glob(os.path.join(_here, "*nashim*.pyd"))
if not _cands:
    raise ImportError(f"no nashim shared library in {_here}")
lib = ctypes.CDLL(_cands[0])
lib.sm_nanoarrow_version.restype = ctypes.c_char_p
NANOARROW_VERSION = lib.sm_nanoarrow_version().decode()
LIBRARY = _cands[0]
