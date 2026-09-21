"""ctypes bridge to `rust_cabi_tree`'s `extern "C"` entry points, and the
njit wrapper functions that call them from INSIDE a compiled kernel — not
through PyO3.

numba's ctypes bridge rejects `c_char_p` and accepts `c_void_p` (§T,
confirmed again here): every pointer argument below is declared
`ctypes.c_void_p` on the Python side and a concrete `*const T` on the Rust
side (see `rust_cabi_tree/src/lib.rs`); lengths cross as separate `c_int32`/
`c_int64` arguments rather than being bundled into a fat pointer, because a
`(ptr, len)` Rust slice has no C-ABI-stable representation numba's bridge
could reconstruct.

The two njit-callable wrappers below (`walk_row_cabi`, one call per row;
`walk_batch_cabi`, one call for the whole batch) are what `bench_perf.py`
times against the numba array walker. `walk_row_cabi` is also what
`fusion_check/pipeline_module.py` calls from inside a real decider2 step —
same function, same binding, reused rather than duplicated.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

from numba import njit

HERE = Path(__file__).resolve().parent
LIB_PATH = HERE / "rust_cabi_tree" / "target" / "release" / "librust_cabi_tree.so"

if not LIB_PATH.exists():
    raise FileNotFoundError(
        f"{LIB_PATH} not built yet — run "
        f"`cargo build --release` in {LIB_PATH.parent.parent}"
    )

_lib = ctypes.CDLL(str(LIB_PATH))

# ---------------------------------------------------------------------------
# walk_row / walk_row_unprotected — 16 args: 10 tree-array pointers + 2
# lengths (n_patterns, n_nodes) + row pointer + n_numeric + str_row pointer
# + n_string. Matches rust_cabi_tree::walk_row's parameter order exactly.
# ---------------------------------------------------------------------------
_WALK_ROW_ARGTYPES = (
    [ctypes.c_void_p] * 7 + [ctypes.c_int32]      # kind..patterns, n_patterns
    + [ctypes.c_void_p] * 3 + [ctypes.c_int32]     # left, right, leaf_value, n_nodes
    + [ctypes.c_void_p, ctypes.c_int32]            # row, n_numeric
    + [ctypes.c_void_p, ctypes.c_int32]            # str_row, n_string
)
_lib.walk_row.argtypes = _WALK_ROW_ARGTYPES
_lib.walk_row.restype = ctypes.c_double
_lib.walk_row_unprotected.argtypes = _WALK_ROW_ARGTYPES
_lib.walk_row_unprotected.restype = ctypes.c_double
walk_row_c = _lib.walk_row
walk_row_unprotected_c = _lib.walk_row_unprotected

_lib.walk_batch.argtypes = (
    [ctypes.c_void_p] * 7 + [ctypes.c_int32]
    + [ctypes.c_void_p] * 3 + [ctypes.c_int32]
    + [ctypes.c_void_p, ctypes.c_int32]
    + [ctypes.c_void_p, ctypes.c_int32]
    + [ctypes.c_int64, ctypes.c_void_p]
)
_lib.walk_batch.restype = ctypes.c_int32
walk_batch_c = _lib.walk_batch

_lib.noop.argtypes = []
_lib.noop.restype = ctypes.c_int64
noop_c = _lib.noop

_lib.trivial.argtypes = [ctypes.c_int32, ctypes.c_int32]
_lib.trivial.restype = ctypes.c_int32
trivial_c = _lib.trivial

_lib.regex_compile.argtypes = [ctypes.c_void_p, ctypes.c_int32]
_lib.regex_compile.restype = ctypes.c_void_p
regex_compile_c = _lib.regex_compile

_lib.regex_is_match.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
_lib.regex_is_match.restype = ctypes.c_int32
regex_is_match_c = _lib.regex_is_match

_lib.regex_free.argtypes = [ctypes.c_void_p]
_lib.regex_free.restype = None
regex_free_c = _lib.regex_free

_lib.deliberately_panic_protected.argtypes = [ctypes.c_int32]
_lib.deliberately_panic_protected.restype = ctypes.c_double
deliberately_panic_protected_c = _lib.deliberately_panic_protected

_lib.deliberately_panic_unprotected.argtypes = [ctypes.c_int32]
_lib.deliberately_panic_unprotected.restype = ctypes.c_double
deliberately_panic_unprotected_c = _lib.deliberately_panic_unprotected


# ---------------------------------------------------------------------------
# njit wrappers — called from inside a compiled kernel, per row.
# ---------------------------------------------------------------------------


@njit(cache=False)
def walk_row_cabi(kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns,
                   left, right, leaf_value, row, str_row):
    """One row through the C-ABI walker. `kind`/... are the flat tree
    arrays (shared across every row); `row`/`str_row` are this row's
    numeric/string-code slices. The whole call is `extern "C"`; nothing
    about it requires the Python interpreter or the GIL."""
    return walk_row_c(
        kind.ctypes.data, feat_idx.ctypes.data, op_code.ctypes.data, thresh.ctypes.data,
        pat_start.ctypes.data, pat_count.ctypes.data, patterns.ctypes.data, patterns.shape[0],
        left.ctypes.data, right.ctypes.data, leaf_value.ctypes.data, kind.shape[0],
        row.ctypes.data, row.shape[0], str_row.ctypes.data, str_row.shape[0],
    )


@njit(cache=False)
def walk_row_cabi_unprotected(kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns,
                               left, right, leaf_value, row, str_row):
    """Same call, to `walk_row_unprotected` — no `catch_unwind` on the Rust
    side. Used only by the panic-safety demonstration, never in the
    performance benchmarks."""
    return walk_row_unprotected_c(
        kind.ctypes.data, feat_idx.ctypes.data, op_code.ctypes.data, thresh.ctypes.data,
        pat_start.ctypes.data, pat_count.ctypes.data, patterns.ctypes.data, patterns.shape[0],
        left.ctypes.data, right.ctypes.data, leaf_value.ctypes.data, kind.shape[0],
        row.ctypes.data, row.shape[0], str_row.ctypes.data, str_row.shape[0],
    )


@njit(cache=False)
def walk_batch_per_row_cabi(kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns,
                             left, right, leaf_value, numeric_cols, string_cols, out):
    """The PER-ROW calling shape: an `njit` row loop making one `extern
    "C"` call per row. This is the shape that preserves fusion — the exact
    thing item 2 (`fusion_check/`) confirms stays fused when this same
    per-row call sits inside a real decider2 step among neighbours."""
    n = numeric_cols.shape[0]
    for i in range(n):
        out[i] = walk_row_cabi(
            kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns,
            left, right, leaf_value, numeric_cols[i], string_cols[i],
        )


@njit(cache=False)
def deliberately_panic_protected(bad):
    """Called from inside njit, matching the real calling context the
    panic-safety demo cares about (an njit caller has no Rust unwind
    tables of its own — the boundary is what matters, not whether the
    immediate caller happens to be CPython or numba-JIT-compiled code)."""
    return deliberately_panic_protected_c(bad)


@njit(cache=False)
def deliberately_panic_unprotected(bad):
    return deliberately_panic_unprotected_c(bad)


@njit(cache=False)
def walk_batch_single_call_cabi(kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns,
                                 left, right, leaf_value, numeric_cols, string_cols, out):
    """The PER-BATCH calling shape: ONE `extern "C"` call for the whole
    array, looped inside Rust. Faster (amortises the call across all rows)
    but reintroduces a boundary — nothing between rows is numba-compiled,
    so this shape cannot be interleaved with a neighbouring njit step the
    way the per-row shape can."""
    n = numeric_cols.shape[0]
    n_numeric = numeric_cols.shape[1]
    n_string = string_cols.shape[1]
    status = walk_batch_c(
        kind.ctypes.data, feat_idx.ctypes.data, op_code.ctypes.data, thresh.ctypes.data,
        pat_start.ctypes.data, pat_count.ctypes.data, patterns.ctypes.data, patterns.shape[0],
        left.ctypes.data, right.ctypes.data, leaf_value.ctypes.data, kind.shape[0],
        numeric_cols.ctypes.data, n_numeric, string_cols.ctypes.data, n_string,
        n, out.ctypes.data,
    )
    return status
