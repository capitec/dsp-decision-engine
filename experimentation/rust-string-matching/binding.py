"""ctypes bridge to `rust_string_match`'s `extern "C"` entry points, plus
the Arrow-buffer extraction and the raw-address `@intrinsic` that lets an
`@njit(cache=True)` kernel call `match_one` WITHOUT capturing a ctypes
symbol as a compile-time global.

Why an intrinsic and not the ctypes object directly: EXPERIMENTS.md §V
found that a kernel referencing a ctypes function object can never be
numba-disk-cached ("dynamic globals"); §W found the precise distinction —
a pointer passed as an ARGUMENT caches, a symbol captured as a GLOBAL does
not. So the kernel takes a `uint64[:]` pointer table as an ordinary
argument and calls through it with `inttoptr` + `call`, exactly
`../cfunc-pointer-interpreter/call_ptr.py`'s mechanism (lifted, not
reinvented — only the signature is new).
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import llvmlite.ir as ll
import numpy as np
import polars as pl
from numba import types
from numba.extending import intrinsic

HERE = Path(__file__).resolve().parent
LIB_PATH = HERE / "rust_string_match" / "target" / "release" / "librust_string_match.so"
if not LIB_PATH.exists():
    raise FileNotFoundError(f"{LIB_PATH} not built — `cargo build --release` in {LIB_PATH.parents[2]}")

_lib = ctypes.CDLL(str(LIB_PATH))

_lib.pattern_compile.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_int64]
_lib.pattern_compile.restype = ctypes.c_int32

_MATCH_ONE_ARGS = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_int64,
                   ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
_lib.match_one.argtypes = _MATCH_ONE_ARGS
_lib.match_one.restype = ctypes.c_int32
_lib.match_one_unprotected.argtypes = _MATCH_ONE_ARGS
_lib.match_one_unprotected.restype = ctypes.c_int32

_lib.match_rows.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_int64,
                            ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                            ctypes.c_int64, ctypes.c_void_p]
_lib.match_rows.restype = ctypes.c_int32

_lib.trivial_i32.argtypes = [ctypes.c_int32, ctypes.c_int32]
_lib.trivial_i32.restype = ctypes.c_int32
_lib.deliberately_panic_protected.argtypes = [ctypes.c_int32]
_lib.deliberately_panic_protected.restype = ctypes.c_int32
_lib.deliberately_panic_unprotected.argtypes = [ctypes.c_int32]
_lib.deliberately_panic_unprotected.restype = ctypes.c_int32


def _addr(fn) -> int:
    return ctypes.cast(fn, ctypes.c_void_p).value


# Slot layout of the pointer table the kernel receives as an ARGUMENT.
PTR_MATCH_ONE = 0
PTR_MATCH_ONE_UNPROTECTED = 1
PTR_TRIVIAL = 2
PTR_PANIC_PROTECTED = 3
PTR_PANIC_UNPROTECTED = 4


def make_ptr_table() -> np.ndarray:
    """Raw `extern "C"` addresses as DATA (`uint64[:]`). Built once at bind
    time; the kernel's compiled body never sees these values."""
    return np.array([
        _addr(_lib.match_one), _addr(_lib.match_one_unprotected), _addr(_lib.trivial_i32),
        _addr(_lib.deliberately_panic_protected), _addr(_lib.deliberately_panic_unprotected),
    ], dtype=np.uint64)


# ---------------------------------------------------------------------------
# Bind-time: compile a pattern once, get an id.
# ---------------------------------------------------------------------------

# decider2.trees.schema.TStringMatchType, in declaration order.
KINDS = {"exact": 0, "starts_with": 1, "contains": 2, "ends_with": 3, "regex": 4}


def compile_pattern(pattern: str, kind: str = "regex") -> int:
    b = pattern.encode("utf-8")
    buf = np.frombuffer(b, dtype=np.uint8)
    pid = _lib.pattern_compile(KINDS[kind], buf.ctypes.data, len(b))
    if pid < 0:
        raise ValueError(f"invalid pattern: {pattern!r}")
    return pid


# ---------------------------------------------------------------------------
# Arrow buffers from polars — zero-copy, asserted.
# ---------------------------------------------------------------------------

class StringBuffers:
    """`(offsets: int64[n+1], values: uint8[total_bytes])` for a Utf8
    series. Keeps a reference to the polars buffers so the memory stays
    alive as long as the numpy views do.

    NOT free, measured: polars 1.41 stores `String` as Utf8View, and
    `_get_buffers()` materialises a fresh offsets/values pair every call
    (the values pointer differs between two calls; 7-18 ns/row). The numpy
    views of THOSE buffers are zero-copy (`allow_copy=False` asserts it),
    the buffers themselves are a per-batch O(rows) conversion."""

    __slots__ = ("offsets", "values", "n", "_keep")

    def __init__(self, series: pl.Series):
        if series.dtype != pl.Utf8:
            raise TypeError(f"expected Utf8, got {series.dtype}")
        bufs = series._get_buffers()
        if bufs["validity"] is not None:
            raise ValueError("null strings not handled in this experiment")
        # allow_copy=False: if polars could not hand these out zero-copy this
        # raises rather than silently paying a copy the numbers would hide.
        self.offsets = bufs["offsets"].to_numpy(allow_copy=False)
        self.values = bufs["values"].to_numpy(allow_copy=False)
        assert self.offsets.dtype == np.int64 and self.values.dtype == np.uint8
        self.n = len(series)
        assert self.offsets.shape[0] == self.n + 1
        self._keep = bufs


def match_rows_py(pattern_id: int, sb: StringBuffers, row_codes: np.ndarray | None = None) -> np.ndarray:
    """ONE Rust call over a whole string set — the column (per-row mask) or
    the category dictionary (per-category mask), or dictionary + codes."""
    n = sb.n if row_codes is None else row_codes.shape[0]
    out = np.empty(n, dtype=np.uint8)
    codes_ptr = None
    if row_codes is not None:
        assert row_codes.dtype == np.int32
        codes_ptr = row_codes.ctypes.data
    rc = _lib.match_rows(pattern_id, sb.offsets.ctypes.data, sb.n,
                         sb.values.ctypes.data, sb.values.shape[0],
                         codes_ptr, n, out.ctypes.data)
    if rc != 0:
        raise RuntimeError(f"match_rows returned {rc}")
    return out


def match_one_py(pattern_id: int, sb: StringBuffers, idx: int, *, protected: bool = True) -> int:
    fn = _lib.match_one if protected else _lib.match_one_unprotected
    return fn(pattern_id, sb.offsets.ctypes.data, sb.n, sb.values.ctypes.data, sb.values.shape[0], idx)


# ---------------------------------------------------------------------------
# The in-kernel call: `int32 match_one(int32, i64*, i64, u8*, i64, i64)`
# through a raw address read from the pointer-table ARGUMENT.
# ---------------------------------------------------------------------------

@intrinsic
def call_match_one(typingctx, addr_t, pid_t, off_t, n_t, val_t, vlen_t, idx_t):
    if addr_t not in (types.uint64, types.int64):
        return None
    sig = types.int32(addr_t, types.int32, types.uint64, types.int64, types.uint64, types.int64, types.int64)

    def codegen(context, builder, signature, args):
        addr, pid, off, n, val, vlen, idx = args
        i8p = ll.IntType(8).as_pointer()
        i64p = ll.IntType(64).as_pointer()
        fnty = ll.FunctionType(ll.IntType(32), [ll.IntType(32), i64p, ll.IntType(64), i8p, ll.IntType(64), ll.IntType(64)])
        fnptr = builder.inttoptr(addr, fnty.as_pointer())
        return builder.call(fnptr, [pid, builder.inttoptr(off, i64p), n, builder.inttoptr(val, i8p), vlen, idx])

    return sig, codegen


@intrinsic
def call_i32_i32(typingctx, addr_t, a_t, b_t):
    """`int32(int32, int32)` through a raw address — the call-overhead probe
    (same shape §V/§W measured, so the overhead number is comparable)."""
    if addr_t not in (types.uint64, types.int64):
        return None
    sig = types.int32(addr_t, types.int32, types.int32)

    def codegen(context, builder, signature, args):
        addr, a, b = args
        fnty = ll.FunctionType(ll.IntType(32), [ll.IntType(32), ll.IntType(32)])
        fnptr = builder.inttoptr(addr, fnty.as_pointer())
        return builder.call(fnptr, [a, b])

    return sig, codegen


@intrinsic
def call_i32_i32_1arg(typingctx, addr_t, a_t):
    """`int32(int32)` through a raw address — the panic demo's shape."""
    if addr_t not in (types.uint64, types.int64):
        return None
    sig = types.int32(addr_t, types.int32)

    def codegen(context, builder, signature, args):
        addr, a = args
        fnty = ll.FunctionType(ll.IntType(32), [ll.IntType(32)])
        fnptr = builder.inttoptr(addr, fnty.as_pointer())
        return builder.call(fnptr, [a])

    return sig, codegen
