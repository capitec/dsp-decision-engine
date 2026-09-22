"""The owner's shape — `ft1 > c1 AND ft2 > c2 AND ft3 ~ /^dog/` — as four
`@njit(cache=True)` kernels, one per string strategy. All four take the
SAME numeric inputs and write the SAME bool result; only how the string
condition is answered differs. The AND short-circuits exactly as decider2's
own encoded AND does (each condition's `otherwise` edge jumps to the
overall else), so the string node is reached only by rows that passed the
two numeric tests.

`cache=True` on every one of them is the decisive non-performance question
(RESULTS.md, caching section): `lazy_rust` calls Rust through a pointer
read from `ptr_table`, an ordinary argument — never a ctypes global.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from binding import call_match_one, call_i32_i32, call_i32_i32_1arg, PTR_MATCH_ONE, PTR_TRIVIAL


@njit(cache=True)
def baseline_no_string(ft1, ft2, c1, c2, out):
    """Floor: the two numeric tests with no string node at all."""
    n = ft1.shape[0]
    hits = 0
    for i in range(n):
        r = False
        if ft1[i] > c1:
            if ft2[i] > c2:
                r = True
        out[i] = r
        hits += r
    return hits


@njit(cache=True)
def frame_tier(ft1, ft2, matched, c1, c2, out):
    """Strategy 1 — the incumbent: the string condition was precomputed
    for EVERY row in the frame tier (`pl.col.str.contains`) into a bool
    column; the kernel just reads it."""
    n = ft1.shape[0]
    hits = 0
    for i in range(n):
        r = False
        if ft1[i] > c1:
            if ft2[i] > c2:
                if matched[i]:
                    r = True
        out[i] = r
        hits += r
    return hits


@njit(cache=True)
def per_category(ft1, ft2, codes, mask, c1, c2, out):
    """Strategy 2 — per-category mask: `mask` is O(distinct) long, built by
    ONE Rust `match_rows` call over the dictionary; the kernel indexes it
    by the row's dictionary code."""
    n = ft1.shape[0]
    hits = 0
    for i in range(n):
        r = False
        if ft1[i] > c1:
            if ft2[i] > c2:
                if mask[codes[i]] != 0:
                    r = True
        out[i] = r
        hits += r
    return hits


@njit(cache=True)
def lazy_rust(ft1, ft2, offsets, values, ptr_table, pattern_id, c1, c2, out):
    """Strategy 3 — lazy Rust AT THE NODE: the regex runs only for rows
    that reach the node, over the column's own Arrow buffers, zero-copy.
    `offsets`/`values` are the polars buffers; `ptr_table[PTR_MATCH_ONE]`
    is `match_one`'s address, passed as data."""
    n = ft1.shape[0]
    n_strings = offsets.shape[0] - 1
    off_ptr = offsets.ctypes.data
    val_ptr = values.ctypes.data
    vlen = values.shape[0]
    addr = ptr_table[PTR_MATCH_ONE]
    hits = 0
    for i in range(n):
        r = False
        if ft1[i] > c1:
            if ft2[i] > c2:
                if call_match_one(addr, pattern_id, off_ptr, n_strings, val_ptr, vlen, i) == 1:
                    r = True
        out[i] = r
        hits += r
    return hits


@njit(cache=True)
def lazy_rust_via_dict(ft1, ft2, codes, dict_offsets, dict_values, ptr_table, pattern_id, c1, c2, out):
    """Strategy 3b — same lazy call, but over the DICTIONARY's buffers
    indexed by the row's code (the shape when the column is already
    Categorical/Enum at the boundary). Same Rust function, same
    intrinsic — only which buffers and which index are handed over."""
    n = ft1.shape[0]
    n_strings = dict_offsets.shape[0] - 1
    off_ptr = dict_offsets.ctypes.data
    val_ptr = dict_values.ctypes.data
    vlen = dict_values.shape[0]
    addr = ptr_table[PTR_MATCH_ONE]
    hits = 0
    for i in range(n):
        r = False
        if ft1[i] > c1:
            if ft2[i] > c2:
                if call_match_one(addr, pattern_id, off_ptr, n_strings, val_ptr, vlen, codes[i]) == 1:
                    r = True
        out[i] = r
        hits += r
    return hits


@njit(cache=True)
def call_overhead_probe(ptr_table, n):
    """`n` calls of `trivial_i32` through the pointer table — isolates the
    boundary cost from the regex cost."""
    addr = ptr_table[PTR_TRIVIAL]
    acc = 0
    for i in range(n):
        acc += call_i32_i32(addr, np.int32(i), np.int32(1))
    return acc


@njit(cache=True)
def call_one_i32(ptr_table, slot, arg):
    """Generic `int32(int32)` call through slot `slot` — the panic demo."""
    return call_i32_i32_1arg(ptr_table[slot], np.int32(arg))


@njit(cache=True)
def match_one_from_kernel(ptr_table, slot, pattern_id, offsets, values, idx):
    """One `match_one` (or `_unprotected`) call from inside njit, for the
    panic demo — `idx` may be hostile."""
    return call_match_one(ptr_table[slot], pattern_id, offsets.ctypes.data,
                          offsets.shape[0] - 1, values.ctypes.data, values.shape[0], idx)
