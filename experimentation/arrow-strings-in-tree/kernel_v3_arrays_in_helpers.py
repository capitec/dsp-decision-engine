"""A walk_tree-shaped kernel with a STR node that matches a string straight
out of polars' own Utf8View ("binview") buffers. Pure numba; no C, no Rust,
no dictionary encoding, no preprocessing pass over the column.

How a row's string is found (Arrow Utf8View, exported by `arrowc.export`):

    view = 16 bytes at views[16*i]
    length = u32 at +0
    length <= 12 : bytes are inline at +4 .. +4+length
    length  > 12 : buffer_index = u32 at +8, offset = u32 at +12,
                   bytes are data[buffer_index][offset : offset+length]

Every address (views, validity, data buffers) is passed as a plain uint64
ARGUMENT in an array -- never captured as a global -- so the kernel disk-
caches (EXPERIMENTS.md §W). `_u8_at` turns an address into a numba array
view of that memory with no owner (meminfo=None): no copy, no refcount.

Node kinds: LEAF, CMP (float64 feature vs threshold), IS_TRUE (float64
feature != 0), STR (string column vs pattern, op = EXACT/PREFIX/SUFFIX/
CONTAINS). A null string never matches (takes `else_`). A malformed view
(buffer index or byte range out of bounds) writes ERR_LEAF and stops that
row -- never reads outside the buffers.
"""
from __future__ import annotations

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

LEAF, CMP, IS_TRUE, STR = 0, 1, 2, 3
LT, LE, EQ, GT, GE, NE = 0, 1, 2, 3, 4, 5
EXACT, PREFIX, SUFFIX, CONTAINS = 0, 1, 2, 3
ERR_LEAF = -1


@intrinsic
def _u8_at(typingctx, addr, n):
    """uint8[n] array view over raw memory at `addr`. No owner, no copy."""
    if not isinstance(addr, types.Integer) or not isinstance(n, types.Integer):
        return None
    arytype = types.Array(types.uint8, 1, "C", readonly=True)

    def codegen(context, builder, sig, args):
        addr_v, n_v = args
        ptr = builder.inttoptr(addr_v, ir.IntType(8).as_pointer())
        n64 = context.cast(builder, n_v, sig.args[1], types.intp)
        ary = context.make_array(arytype)(context, builder)
        one = context.get_constant(types.intp, 1)
        context.populate_array(ary, data=ptr, shape=[n64], strides=[one],
                               itemsize=one, meminfo=None)
        return ary._getvalue()

    return arytype(addr, n), codegen


@njit(cache=True, inline="always")
def _u32(views, at):
    return (np.uint32(views[at]) | (np.uint32(views[at + 1]) << np.uint32(8))
            | (np.uint32(views[at + 2]) << np.uint32(16)) | (np.uint32(views[at + 3]) << np.uint32(24)))


@njit(cache=True, inline="always")
def _is_valid(validity, i):
    if validity.shape[0] == 0:
        return True
    return (validity[i >> 3] >> (i & 7)) & 1 == 1


@njit(cache=True, inline="always")
def _eq_at(s, s_off, pat, p_off, n):
    for j in range(n):
        if s[s_off + j] != pat[p_off + j]:
            return False
    return True


@njit(cache=True, inline="always")
def _match_at(s, off, n, pat, poff, m, mode):
    """Match s[off:off+n] against pat[poff:poff+m]. No slices: numba builds a
    fresh array struct per slice, which is what made the first version cost
    ~100 ns/row (see results.jsonl probe=slice_cost)."""
    if mode == EXACT:
        return n == m and _eq_at(s, off, pat, poff, m)
    elif mode == PREFIX:
        return n >= m and _eq_at(s, off, pat, poff, m)
    elif mode == SUFFIX:
        return n >= m and _eq_at(s, off + n - m, pat, poff, m)
    else:  # CONTAINS
        if m == 0:
            return True
        first = pat[poff]
        for k in range(n - m + 1):
            if s[off + k] == first and _eq_at(s, off + k, pat, poff, m):
                return True
        return False


@njit(cache=True, inline="always")
def _str_node(views, i, data_addr, data_size, base, nd, pat, poff, plen, mode):
    """Locate row i's bytes in the binview layout and match them in place.
    Returns (ok, matched). ok=False means the view pointed outside its
    buffers (corrupt input) -- the caller writes ERR_LEAF.

    `data_addr`/`data_size` are the FLAT tables over all string columns and
    this column's buffers start at `base`. Flat on purpose: indexing a row of
    a 2D table (`tab[col]`) makes numba build a refcounted array view per row,
    and that alone cost ~140 ns/row (results.jsonl probe=ablate, bench_v2)."""
    at = 16 * i
    ln = np.int64(_u32(views, at))
    if ln <= 12:
        return True, _match_at(views, at + 4, ln, pat, poff, plen, mode)
    bi = np.int64(_u32(views, at + 8))
    off = np.int64(_u32(views, at + 12))
    if bi < 0 or bi >= nd:
        return False, False
    size = np.int64(data_size[base + bi])
    if off < 0 or off + ln > size:
        return False, False
    buf = _u8_at(data_addr[base + bi], size)
    return True, _match_at(buf, off, ln, pat, poff, plen, mode)


@njit(cache=True)
def walk_chunk(
    # per-row float features for this chunk: [n, nf]
    feats,
    # the string columns of this chunk (one row each): addresses are ARGUMENTS
    str_views_addr, str_validity_addr, str_validity_len, str_validity_off,
    str_n_data, str_data_base, str_data_addr, str_data_size,
    # pattern table: flat bytes + [start, end) per pattern
    pat_bytes, pat_bounds,
    # tree structure
    thresholds, kind, feat_idx, op, thr_slot, then_, else_, leaf_value,
    # output slice for this chunk
    out, out_off, n,
):
    for i in range(n):
        pc = 0
        while True:
            k = kind[pc]
            if k == LEAF:
                out[out_off + i] = leaf_value[pc]
                break
            elif k == CMP:
                a = feats[i, feat_idx[pc]]
                b = thresholds[thr_slot[pc]]
                o = op[pc]
                if o == LT:
                    c = a < b
                elif o == LE:
                    c = a <= b
                elif o == EQ:
                    c = a == b
                elif o == GT:
                    c = a > b
                elif o == GE:
                    c = a >= b
                else:
                    c = a != b
                pc = then_[pc] if c else else_[pc]
            elif k == IS_TRUE:
                pc = then_[pc] if feats[i, feat_idx[pc]] != 0.0 else else_[pc]
            else:  # STR
                col = feat_idx[pc]
                validity = _u8_at(str_validity_addr[col], str_validity_len[col])
                if not _is_valid(validity, i + str_validity_off[col]):
                    pc = else_[pc]
                    continue
                views = _u8_at(str_views_addr[col], 16 * n)
                p = thr_slot[pc]
                poff = pat_bounds[p, 0]
                ok, hit = _str_node(views, i, str_data_addr, str_data_size, str_data_base[col], str_n_data[col],
                                    pat_bytes, poff, pat_bounds[p, 1] - poff, op[pc])
                if not ok:
                    out[out_off + i] = ERR_LEAF
                    break
                pc = then_[pc] if hit else else_[pc]


# ---------------------------------------------------------------------------
# Python side: build the argument tables from a StringView and run per chunk.
# ---------------------------------------------------------------------------

def pattern_table(patterns: list[str]):
    bs = [p.encode("utf-8") for p in patterns]
    flat = np.frombuffer(b"".join(bs), dtype=np.uint8).copy() if bs else np.empty(0, np.uint8)
    bounds = np.zeros((len(bs), 2), dtype=np.int64)
    pos = 0
    for k, b in enumerate(bs):
        bounds[k] = (pos, pos + len(b))
        pos += len(b)
    return flat, bounds


def string_tables(chunks_by_col):
    """chunks_by_col: list over string columns of ONE arrowc.Chunk each (same
    chunk index). Returns the eight per-chunk argument arrays. The data-buffer
    tables are FLAT (all columns end to end) with a per-column `base` -- see
    `_str_node` for why."""
    ncol = len(chunks_by_col)
    views = np.zeros(ncol, np.uint64)
    valid = np.zeros(ncol, np.uint64)
    valid_len = np.zeros(ncol, np.int64)
    valid_off = np.zeros(ncol, np.int64)
    nd = np.zeros(ncol, np.int64)
    base = np.zeros(ncol, np.int64)
    daddr, dsize = [], []
    for k, c in enumerate(chunks_by_col):
        # A polars slice is zero-copy: the chunk carries an Arrow `offset`.
        # Views are 16 bytes each so the address just moves; the validity
        # bitmap cannot be re-based to a byte, so the bit offset travels.
        views[k] = c.views + 16 * c.offset
        valid[k] = c.validity
        valid_len[k] = (c.offset + c.length + 7) // 8 if c.validity else 0
        valid_off[k] = c.offset
        nd[k] = len(c.data)
        base[k] = len(daddr)
        daddr.extend(c.data)
        dsize.extend(c.data_sizes)
    return (views, valid, valid_len, valid_off, nd, base,
            np.array(daddr, dtype=np.uint64), np.array(dsize, dtype=np.uint64))


def run_tree(float_cols, string_views, tree, patterns, out=None):
    """float_cols: [n, nf] float64 (may be empty [n, 0]); string_views: list of
    arrowc.StringView (one per string column, all from the same frame so the
    chunk boundaries agree); tree: dict of structure arrays."""
    n = string_views[0].n_rows if string_views else float_cols.shape[0]
    if out is None:
        out = np.empty(n, dtype=np.int64)
    pat_bytes, pat_bounds = pattern_table(patterns)
    n_chunks = len(string_views[0].chunks) if string_views else 1
    off = 0
    for ci in range(n_chunks):
        chunks = [v.chunks[ci] for v in string_views]
        cn = chunks[0].length if chunks else n
        tabs = string_tables(chunks)
        walk_chunk(
            float_cols[off: off + cn], *tabs, pat_bytes, pat_bounds,
            tree["thresholds"], tree["kind"], tree["feat_idx"], tree["op"], tree["thr_slot"],
            tree["then_"], tree["else_"], tree["leaf_value"], out, off, cn,
        )
        off += cn
    return out


def build_tree(nodes, thresholds=()):
    """nodes: list of tuples (kind, feat_idx, op, thr_slot, then_, else_, leaf_value)."""
    cols = list(zip(*nodes))
    return {
        "thresholds": np.array(thresholds, np.float64),
        "kind": np.array(cols[0], np.int64), "feat_idx": np.array(cols[1], np.int64),
        "op": np.array(cols[2], np.int64), "thr_slot": np.array(cols[3], np.int64),
        "then_": np.array(cols[4], np.int64), "else_": np.array(cols[5], np.int64),
        "leaf_value": np.array(cols[6], np.int64),
    }
