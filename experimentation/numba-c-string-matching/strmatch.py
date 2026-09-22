"""ctypes binding to c/libstrmatch.so, and the ONE thing a kernel needs
from it: the raw address of `sm_match` as a plain integer.

Shape lifted from ../rust-cabi-in-kernel/tree_walk_cabi.py: every pointer
is `c_void_p` (numba's ctypes bridge rejects `c_char_p`), lengths cross as
separate integers. But unlike that experiment, the kernels here never
capture the ctypes function object as a global -- they receive
`MATCH_ADDR` inside a `uint64` table ARGUMENT, and pattern IDs (not pointers) in an `int64` table argument, and call through
`call_ptr.call_match` (inttoptr + call). That is the §W distinction that
makes the kernel disk-cacheable: a pointer passed as an argument caches,
a symbol captured as a global does not.
"""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB_PATH = HERE / "c" / "libstrmatch.so"
if not LIB_PATH.exists():
    subprocess.run([str(HERE / "c" / "build.sh")], check=True)

_lib = ctypes.CDLL(str(LIB_PATH))

REGEX, EXACT, PREFIX, SUFFIX, SUBSTRING, POSIX = 0, 1, 2, 3, 4, 5
FLAG_UTF, FLAG_NO_JIT = 1, 2
KIND_NAMES = {REGEX: "regex", EXACT: "exact", PREFIX: "prefix", SUFFIX: "suffix", SUBSTRING: "substring", POSIX: "posix"}

_lib.sm_compile.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64)]
_lib.sm_compile.restype = ctypes.c_int64
_lib.sm_match_id.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64]
_lib.sm_match_id.restype = ctypes.c_int32
_lib.sm_match.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
_lib.sm_match.restype = ctypes.c_int32
_lib.sm_handle_ptr.argtypes = [ctypes.c_int64]
_lib.sm_handle_ptr.restype = ctypes.c_void_p
_lib.sm_free.argtypes = [ctypes.c_int64]
_lib.sm_free.restype = None
_lib.sm_is_jit.argtypes = [ctypes.c_int64]
_lib.sm_is_jit.restype = ctypes.c_int32
_lib.sm_set_match_limit.argtypes = [ctypes.c_uint32]
_lib.sm_set_match_limit.restype = None
_lib.sm_get_match_limit.argtypes = []
_lib.sm_get_match_limit.restype = ctypes.c_uint32
_lib.sm_error_message.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32]
_lib.sm_error_message.restype = ctypes.c_int32
_lib.sm_trivial.argtypes = [ctypes.c_int32, ctypes.c_int32]
_lib.sm_trivial.restype = ctypes.c_int32

# Raw addresses -- what goes into the kernel's `fn_table` argument.
MATCH_ADDR: int = ctypes.cast(_lib.sm_match_id, ctypes.c_void_p).value       # (id, ptr, len) -- what kernels use
MATCH_RAW_ADDR: int = ctypes.cast(_lib.sm_match, ctypes.c_void_p).value      # (ptr, ptr, len) -- microbench only
TRIVIAL_ADDR: int = ctypes.cast(_lib.sm_trivial, ctypes.c_void_p).value

# The ctypes function objects, for the (non-cacheable) global-capture
# comparison and for Python-side use.
sm_match_c = _lib.sm_match_id
sm_trivial_c = _lib.sm_trivial
set_match_limit = _lib.sm_set_match_limit
get_match_limit = _lib.sm_get_match_limit
handle_ptr = _lib.sm_handle_ptr


class PatternError(ValueError):
    pass


def compile_pattern(pattern: str, kind: int = REGEX, flags: int = FLAG_UTF) -> int:
    """Compile once, off the hot path. Returns the registry id (an int >= 0).
    A malformed pattern raises PatternError HERE, in Python, at build time
    -- it never reaches a kernel. `flags=FLAG_UTF` matches polars' UTF-8
    semantics; 0 gives byte-mode PCRE2."""
    b = pattern.encode("utf-8")
    buf = np.frombuffer(b, dtype=np.uint8) if b else np.zeros(1, dtype=np.uint8)
    err = ctypes.c_int32(0)
    off = ctypes.c_int64(0)
    h = _lib.sm_compile(buf.ctypes.data, len(b), kind, flags, ctypes.byref(err), ctypes.byref(off))
    if h < 0:
        msg = ctypes.create_string_buffer(256)
        _lib.sm_error_message(err.value, ctypes.cast(msg, ctypes.c_void_p), 256)
        raise PatternError(
            f"{KIND_NAMES.get(kind, kind)} pattern {pattern!r} rejected at offset "
            f"{off.value}: {msg.value.decode(errors='replace')} (code {err.value})"
        )
    return h


def is_jit(pid: int) -> bool:
    return _lib.sm_is_jit(pid) == 1


def free_pattern(pid: int) -> None:
    _lib.sm_free(pid)


def match_py(pid: int, s: str) -> int:
    """Python-side call for tests/oracles only."""
    b = s.encode("utf-8")
    buf = np.frombuffer(b, dtype=np.uint8) if b else np.zeros(1, dtype=np.uint8)
    return _lib.sm_match_id(pid, buf.ctypes.data, len(b))
