"""Pure-numba byte matchers over polars' Arrow string buffers. No C, no
Rust, no new dependency.

Two families, same algorithms:

  ptr_*  -- every input is an INTEGER: the haystack is (address, length),
            the needle is (address, length). Bytes are read through
            `rawload.load_u8`. This is the shape a per-row tree walker can
            call without the NRT refcount trap (see rawload.py).
  arr_*  -- the haystack is a uint8 array + (start, end), the needle a
            uint8 array + (start, len). Ordinary numba indexing. Fine
            inside a single-loop kernel; measured here to show the
            difference and to show the refcount trap if it appears.

A needle table is one flat uint8 buffer plus (start, len, kind) per
pattern, so ONE compiled kernel serves every pattern ever -- there is no
per-pattern compile and nothing pattern-specific in the compiled object,
which is what lets the kernel disk-cache.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from rawload import load_u8, load_i64, load_u64

# match kinds -- decider2's TStringMatchType minus `regex`
EXACT, PREFIX, SUFFIX, SUBSTRING, SUBSTRING_SKIP, PREFIX_CI, PREFIX_W = 0, 1, 2, 3, 4, 5, 6
KIND_NAMES = {EXACT: "exact", PREFIX: "prefix", SUFFIX: "suffix", SUBSTRING: "substring(naive)",
              SUBSTRING_SKIP: "substring(first-byte skip)", PREFIX_CI: "prefix(ascii-ci)", PREFIX_W: "prefix(8-byte words)"}

# ---------------------------------------------------------------------------
# Pointer family: all-scalar inputs.
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def ptr_bytes_eq(a, b, n):
    """memcmp(a, b, n) == 0, byte at a time."""
    for k in range(n):
        if load_u8(a + np.uint64(k)) != load_u8(b + np.uint64(k)):
            return False
    return True


@njit(cache=True, inline="always")
def ptr_bytes_eq_w(a, b, n):
    """memcmp(a, b, n) == 0, eight bytes at a time then a byte tail --
    what glibc's memcmp does for short inputs."""
    k = 0
    while k + 8 <= n:
        if load_u64(a + np.uint64(k)) != load_u64(b + np.uint64(k)):
            return False
        k += 8
    while k < n:
        if load_u8(a + np.uint64(k)) != load_u8(b + np.uint64(k)):
            return False
        k += 1
    return True


@njit(cache=True, inline="always")
def ptr_exact(hay, hlen, nd, nlen):
    return hlen == nlen and ptr_bytes_eq(hay, nd, nlen)


@njit(cache=True, inline="always")
def ptr_prefix(hay, hlen, nd, nlen):
    return hlen >= nlen and ptr_bytes_eq(hay, nd, nlen)


@njit(cache=True, inline="always")
def ptr_prefix_w(hay, hlen, nd, nlen):
    return hlen >= nlen and ptr_bytes_eq_w(hay, nd, nlen)


@njit(cache=True, inline="always")
def ptr_suffix(hay, hlen, nd, nlen):
    return hlen >= nlen and ptr_bytes_eq(hay + np.uint64(hlen - nlen), nd, nlen)


@njit(cache=True, inline="always")
def ptr_substring(hay, hlen, nd, nlen):
    """Naive two-loop search. O(hlen * nlen) worst case, but with hlen <= 22
    and a short needle that bound is tiny and has no pathological input
    beyond it (no backtracking, no state)."""
    if nlen == 0:
        return True
    last = hlen - nlen
    for i in range(last + 1):
        if ptr_bytes_eq(hay + np.uint64(i), nd, nlen):
            return True
    return False


@njit(cache=True, inline="always")
def ptr_substring_skip(hay, hlen, nd, nlen):
    """Scan for the needle's first byte, compare the rest only on a hit --
    the shape glibc's memmem takes for short needles."""
    if nlen == 0:
        return True
    last = hlen - nlen
    first = load_u8(nd)
    for i in range(last + 1):
        if load_u8(hay + np.uint64(i)) == first:
            if ptr_bytes_eq(hay + np.uint64(i + 1), nd + np.uint64(1), nlen - 1):
                return True
    return False


@njit(cache=True, inline="always")
def ascii_lower(c):
    # 'A'..'Z' -> 'a'..'z'; every other byte (incl. UTF-8 continuation
    # bytes >= 0x80) is left alone, so multi-byte code points pass through
    # untouched and non-ASCII letters are NOT folded. See RESULTS.md §7.
    if c >= 65 and c <= 90:
        return c + np.uint8(32)
    return c


@njit(cache=True, inline="always")
def ptr_prefix_ci(hay, hlen, nd, nlen):
    """ASCII-case-insensitive prefix. The needle is pre-lowered in Python;
    the haystack is folded byte by byte in the loop."""
    if hlen < nlen:
        return False
    for k in range(nlen):
        if ascii_lower(load_u8(hay + np.uint64(k))) != load_u8(nd + np.uint64(k)):
            return False
    return True


@njit(cache=True, inline="always")
def ptr_match(kind, hay, hlen, nd, nlen):
    """The one dispatch a tree node needs: (kind, haystack, needle) -> bool."""
    if kind == EXACT:
        return ptr_exact(hay, hlen, nd, nlen)
    elif kind == PREFIX:
        return ptr_prefix(hay, hlen, nd, nlen)
    elif kind == SUFFIX:
        return ptr_suffix(hay, hlen, nd, nlen)
    elif kind == SUBSTRING:
        return ptr_substring(hay, hlen, nd, nlen)
    elif kind == SUBSTRING_SKIP:
        return ptr_substring_skip(hay, hlen, nd, nlen)
    elif kind == PREFIX_CI:
        return ptr_prefix_ci(hay, hlen, nd, nlen)
    else:
        return ptr_prefix_w(hay, hlen, nd, nlen)


# ---------------------------------------------------------------------------
# Array family: the same algorithms over numpy arrays.
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def arr_bytes_eq(vals, s, nd, ns, n):
    for k in range(n):
        if vals[s + k] != nd[ns + k]:
            return False
    return True


@njit(cache=True, inline="always")
def arr_match(kind, vals, s, e, nd, ns, nlen):
    hlen = e - s
    if kind == EXACT:
        return hlen == nlen and arr_bytes_eq(vals, s, nd, ns, nlen)
    elif kind == PREFIX:
        return hlen >= nlen and arr_bytes_eq(vals, s, nd, ns, nlen)
    elif kind == SUFFIX:
        return hlen >= nlen and arr_bytes_eq(vals, e - nlen, nd, ns, nlen)
    elif kind == SUBSTRING:
        if nlen == 0:
            return True
        for i in range(hlen - nlen + 1):
            if arr_bytes_eq(vals, s + i, nd, ns, nlen):
                return True
        return False
    elif kind == SUBSTRING_SKIP:
        if nlen == 0:
            return True
        first = nd[ns]
        for i in range(hlen - nlen + 1):
            if vals[s + i] == first and arr_bytes_eq(vals, s + i + 1, nd, ns + 1, nlen - 1):
                return True
        return False
    elif kind == PREFIX_CI:
        if hlen < nlen:
            return False
        for k in range(nlen):
            if ascii_lower(vals[s + k]) != nd[ns + k]:
                return False
        return True
    else:  # PREFIX_W has no array form; fall back to byte prefix
        return hlen >= nlen and arr_bytes_eq(vals, s, nd, ns, nlen)


# ---------------------------------------------------------------------------
# Whole-column kernels (one match per row) for the per-call measurement.
# `tab` is uint64 [5]: (offsets addr, values addr, n_offsets, n_bytes,
#   validity addr -- a uint8[n] array, 1 = valid, or address 0 when the
#   column has no nulls). A null row has ZERO length in the offsets, so it
#   is indistinguishable from "" without the validity buffer: an empty
#   needle or `exact ""` would match a null. Null -> no match, always.
# `nd_tab` is uint64 [2]: (needle addr, needle len).
# ---------------------------------------------------------------------------

@njit(cache=True)
def col_ptr(kind, tab, nd_tab, out):
    off_addr = tab[0]; val_addr = tab[1]; valid_addr = tab[4]
    nd = nd_tab[0]; nlen = np.int64(nd_tab[1])
    n = np.int64(tab[2]) - 1
    for i in range(n):
        if valid_addr != 0 and load_u8(valid_addr + np.uint64(i)) == 0:
            out[i] = 0
            continue
        s = load_i64(off_addr + np.uint64(8 * i))
        e = load_i64(off_addr + np.uint64(8 * (i + 1)))
        out[i] = ptr_match(kind, val_addr + np.uint64(s), e - s, nd, nlen)


@njit(cache=True)
def col_arr(kind, offs, vals, valid, needle, ns, nlen, out):
    """`valid` is uint8[n] or a zero-length array (no nulls)."""
    has_valid = valid.shape[0] > 0
    for i in range(offs.shape[0] - 1):
        if has_valid and valid[i] == 0:
            out[i] = 0
            continue
        out[i] = arr_match(kind, vals, offs[i], offs[i + 1], needle, ns, nlen)


@njit(cache=True)
def col_floor(tab, out):
    """Same loop, no match: load the two offsets and store the length."""
    off_addr = tab[0]
    n = np.int64(tab[2]) - 1
    for i in range(n):
        s = load_i64(off_addr + np.uint64(8 * i))
        e = load_i64(off_addr + np.uint64(8 * (i + 1)))
        out[i] = np.uint8(e - s)


# --- Python-side helpers ------------------------------------------------------

def string_buffers(series):
    """polars string Series -> (offsets int64[n+1], values uint8[], valid
    uint8[n] or uint8[0]) numpy arrays, as the sibling strand does it plus
    the validity buffer. NOTE: _get_buffers() is O(n) on polars 1.41
    (binview -> offsets+values), ~11 ns/row; that cost is charged to the
    lazy strategies in the selectivity benchmark."""
    b = series._get_buffers()
    offs = b["offsets"].to_numpy()
    vals = b["values"].to_numpy()
    v = b.get("validity")
    valid = np.zeros(0, np.uint8) if v is None else np.ascontiguousarray(v.to_numpy().astype(np.uint8))
    assert offs.dtype == np.int64 and vals.dtype == np.uint8
    return offs, vals, valid


def buffer_table(offs, vals, valid):
    """The uint64[5] pointer table. The caller keeps offs/vals/valid alive."""
    assert offs.flags.c_contiguous and vals.flags.c_contiguous and valid.flags.c_contiguous
    return np.array([offs.ctypes.data, vals.ctypes.data, offs.shape[0], vals.shape[0],
                     valid.ctypes.data if valid.shape[0] else 0], np.uint64)


class NeedleTable:
    """Many patterns, one flat buffer. `kind[i]`, `start[i]`, `length[i]`."""

    def __init__(self, patterns):
        """patterns: [(kind, str_or_bytes), ...]"""
        parts, kinds, starts, lens = [], [], [], []
        pos = 0
        for kind, p in patterns:
            b = p.encode("utf-8") if isinstance(p, str) else bytes(p)
            if kind == PREFIX_CI:
                b = bytes(ascii_lower_py(c) for c in b)
            parts.append(b); kinds.append(kind); starts.append(pos); lens.append(len(b)); pos += len(b)
        self.buf = np.frombuffer(b"".join(parts) + b"\0", dtype=np.uint8).copy()  # +1 so an empty table has an address
        self.kind = np.asarray(kinds, np.int64)
        self.start = np.asarray(starts, np.int64)
        self.length = np.asarray(lens, np.int64)

    def addr_table(self, i):
        return np.array([self.buf.ctypes.data + int(self.start[i]), int(self.length[i])], np.uint64)


def ascii_lower_py(c):
    return c + 32 if 65 <= c <= 90 else c
