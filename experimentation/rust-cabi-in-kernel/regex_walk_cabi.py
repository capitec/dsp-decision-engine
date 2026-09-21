"""Regex via the C ABI, called from inside njit — compile once, get an
opaque handle back, then cross only a `(pointer, length)` pair per call.
No `Vec<String>` marshalling (§U's naive PyO3 binding: ~79 ns/call on top
of the 25.6 ns/call pure match) and no Python object at all on the hot
path — this is what "raw byte pointers" buys, concretely.

Text is held in ONE contiguous byte buffer (`numpy.uint8`) built once
outside the timed loop, exactly the shape decider2's own boundary already
uses for string columns dictionary-encoded at the frame tier (doc 05
§1.5) — Arrow-style `(data, offsets)` for variable-length text, or a
flat fixed-width slab when every row's text is the same length.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from tree_walk_cabi import regex_compile_c, regex_free_c, regex_is_match_c


def compile_pattern(pattern: str) -> int:
    """Compile ONCE, off the hot path. Returns the opaque handle (an
    address, as a plain Python int) or raises if the pattern is invalid."""
    pattern_bytes = pattern.encode("utf-8")
    buf = np.frombuffer(pattern_bytes, dtype=np.uint8)
    handle = regex_compile_c(buf.ctypes.data, len(pattern_bytes))
    if not handle:
        raise ValueError(f"invalid pattern: {pattern!r}")
    return handle


def free_pattern(handle: int) -> None:
    regex_free_c(handle)


def encode_variable_length(strings: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Arrow-style `(data, offsets)`: one big contiguous byte buffer plus
    `len(strings) + 1` start offsets. Built once, outside any timed loop —
    exactly the boundary-time cost decider2 already pays once per batch for
    a dictionary-encoded string column, never per row."""
    encoded = [s.encode("utf-8") for s in strings]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    for i, e in enumerate(encoded):
        offsets[i + 1] = offsets[i] + len(e)
    data = np.zeros(offsets[-1], dtype=np.uint8)
    pos = 0
    for e in encoded:
        data[pos:pos + len(e)] = np.frombuffer(e, dtype=np.uint8)
        pos += len(e)
    return data, offsets


def encode_fixed_width(strings: list[str], width: int) -> np.ndarray:
    """A flat slab, `width` bytes per row — every row the same length, so
    no offsets array is needed at all, just `i * width`."""
    data = np.zeros(len(strings) * width, dtype=np.uint8)
    for i, s in enumerate(strings):
        b = s.encode("utf-8")
        assert len(b) == width, f"{s!r} is not exactly {width} bytes"
        data[i * width:(i + 1) * width] = np.frombuffer(b, dtype=np.uint8)
    return data


@njit(cache=False)
def regex_match_variable(handle, data, offsets, out):
    """One `extern "C"` call per row, over an Arrow-style `(data,
    offsets)` pair — zero-copy, zero per-call allocation on either side of
    the boundary."""
    n = offsets.shape[0] - 1
    base = data.ctypes.data
    for i in range(n):
        start = offsets[i]
        length = offsets[i + 1] - start
        out[i] = regex_is_match_c(handle, base + start, length)


@njit(cache=False)
def regex_match_variable_repeated(handle, data, offsets, out, repeats):
    """Same per-call cost as `regex_match_variable`, but the whole scan
    runs `repeats` times inside ONE njit dispatch — needed because a tiny
    `n` (12 categories) makes a single Python->numba call's own dispatch
    overhead (~1-2 us, unrelated to this experiment) dominate a wall-clock
    measurement of one call. Repeating amortises that fixed cost away and
    leaves the steady-state per-call number, the same convention §T/§U use
    for their own 'pure compute' figures."""
    n = offsets.shape[0] - 1
    base = data.ctypes.data
    for _ in range(repeats):
        for i in range(n):
            out[i] = regex_is_match_c(handle, base + offsets[i], offsets[i + 1] - offsets[i])


@njit(cache=False)
def regex_match_fixed(handle, data, width, out):
    """Same call, over a flat fixed-width slab — the per-category-mask
    case's shape (small n, short fixed strings) and the per-row case's
    shape (n = row count) are the SAME function; only n and width differ."""
    n = data.shape[0] // width
    base = data.ctypes.data
    for i in range(n):
        out[i] = regex_is_match_c(handle, base + i * width, width)
