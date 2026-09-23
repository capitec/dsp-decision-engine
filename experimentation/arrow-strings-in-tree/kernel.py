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
caches (EXPERIMENTS.md §W). Bytes are read through two tiny intrinsics
(`_load_u8`, `_load_u32`: an LLVM inttoptr + load); no array is ever built
over polars' memory, so nothing is copied and nothing is refcounted.

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
from numba.extending import intrinsic

LEAF, CMP, IS_TRUE, STR = 0, 1, 2, 3
LT, LE, EQ, GT, GE, NE = 0, 1, 2, 3, 4, 5
EXACT, PREFIX, SUFFIX, CONTAINS = 0, 1, 2, 3
ERR_LEAF = -1


@intrinsic
def _load_u8(typingctx, addr):
    """One byte at a raw address. Scalars in, scalar out: no array, no refcount."""
    if not isinstance(addr, types.Integer):
        return None

    def codegen(context, builder, sig, args):
        ptr = builder.inttoptr(args[0], ir.IntType(8).as_pointer())
        return builder.load(ptr)
    return types.uint8(addr), codegen


@intrinsic
def _load_u32(typingctx, addr):
    """Unaligned little-endian u32 at a raw address (a binview field)."""
    if not isinstance(addr, types.Integer):
        return None

    def codegen(context, builder, sig, args):
        ptr = builder.inttoptr(args[0], ir.IntType(32).as_pointer())
        return builder.load(ptr, align=1)
    return types.uint32(addr), codegen


# ---------------------------------------------------------------------------
# WHY EVERY HELPER BELOW TAKES ONLY INTEGERS
#
# The first three versions of this kernel passed numpy arrays (the views, the
# pattern bytes, the buffer tables) into `inline="always"` helpers, and cost
# 100-250 ns/row. numba's IR inlining keeps an NRT incref/decref pair for
# every array argument at every call, and the tables have real meminfos so
# those are atomic ops: ablate2.py kernel A = 170 ns/row with 8 incref/23
# decref in its IR, against 13 ns/row for the identical logic written inline
# (ablate.py K4). Address arithmetic and byte loads have nothing to refcount.
# The C strand hit the same wall from the other side (its 365 ns/row).
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _is_valid(validity_addr, bit):
    """Arrow validity bitmap: bit `bit` of the buffer at `validity_addr`;
    a zero address means 'no nulls'."""
    if validity_addr == 0:
        return True
    byte = _load_u8(validity_addr + np.uint64(bit >> 3))
    return (byte >> (bit & 7)) & 1 == 1


@njit(cache=True, inline="always")
def _eq_at(s_addr, p_addr, n):
    for j in range(n):
        if _load_u8(s_addr + np.uint64(j)) != _load_u8(p_addr + np.uint64(j)):
            return False
    return True


@njit(cache=True, inline="always")
def _match_at(s_addr, n, p_addr, m, mode):
    """Match the n bytes at s_addr against the m bytes at p_addr."""
    if mode == EXACT:
        return n == m and _eq_at(s_addr, p_addr, m)
    elif mode == PREFIX:
        return n >= m and _eq_at(s_addr, p_addr, m)
    elif mode == SUFFIX:
        return n >= m and _eq_at(s_addr + np.uint64(n - m), p_addr, m)
    else:  # CONTAINS
        if m == 0:
            return True
        first = _load_u8(p_addr)
        for k in range(n - m + 1):
            a = s_addr + np.uint64(k)
            if _load_u8(a) == first and _eq_at(a, p_addr, m):
                return True
        return False


@njit(cache=True, inline="always")
def _view(views_addr, i):
    """Decode row i's 16-byte binview: (length, buffer_index, offset).
    For an inline string (length <= 12) the bytes sit at views_addr+16*i+4
    and buffer_index/offset are meaningless."""
    at = views_addr + np.uint64(16 * i)
    ln = np.int64(_load_u32(at))
    bi = np.int64(_load_u32(at + np.uint64(8)))
    off = np.int64(_load_u32(at + np.uint64(12)))
    return ln, bi, off


@njit(cache=True)
def walk_chunk(
    # per-row float features for this chunk: [n, nf]
    feats,
    # the string columns of this chunk (one row each): addresses are ARGUMENTS
    str_views_addr, str_validity_addr, str_validity_off,
    str_n_data, str_data_base, str_data_addr, str_data_size,
    # pattern table: flat bytes + [start, end) per pattern
    pat_bytes, pat_bounds,
    # tree structure
    thresholds, kind, feat_idx, op, thr_slot, then_, else_, leaf_value,
    # output slice for this chunk
    out, out_off, n,
):
    pat_base = np.uint64(pat_bytes.ctypes.data)
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
            else:  # STR -- everything here is scalar; the tables are indexed
                #        in this body and never passed to a helper (see the
                #        note above `_is_valid`).
                col = feat_idx[pc]
                if not _is_valid(str_validity_addr[col], i + str_validity_off[col]):
                    pc = else_[pc]
                    continue
                va = str_views_addr[col]
                ln, bi, off = _view(va, i)
                if ln <= 12:
                    s_addr = va + np.uint64(16 * i + 4)
                else:
                    if bi < 0 or bi >= str_n_data[col]:
                        out[out_off + i] = ERR_LEAF
                        break
                    slot = str_data_base[col] + bi
                    if off < 0 or off + ln > np.int64(str_data_size[slot]):
                        out[out_off + i] = ERR_LEAF
                        break
                    s_addr = str_data_addr[slot] + np.uint64(off)
                p = thr_slot[pc]
                poff = pat_bounds[p, 0]
                hit = _match_at(s_addr, ln, pat_base + np.uint64(poff), pat_bounds[p, 1] - poff, op[pc])
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
    chunk index). Returns the seven per-chunk argument arrays. The data-buffer
    tables are FLAT (all columns end to end) with a per-column `base` -- see
    `_str_node` for why."""
    ncol = len(chunks_by_col)
    views = np.zeros(ncol, np.uint64)
    valid = np.zeros(ncol, np.uint64)
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
        valid_off[k] = c.offset
        nd[k] = len(c.data)
        base[k] = len(daddr)
        daddr.extend(c.data)
        dsize.extend(c.data_sizes)
    return (views, valid, valid_off, nd, base,
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
    # Polars does not promise that two columns of one frame share chunk
    # boundaries (a fresh Series added with `with_columns` is one chunk on a
    # frame that has two). The kernel walks chunk-by-chunk, so a mismatch
    # would silently pair row i of one column with row j of another. Refuse
    # loudly; the caller can `df.rechunk()` (a copy) or align by rows.
    for v in string_views:
        if v.format != "vu":
            raise ValueError(f"expected Arrow Utf8View ('vu') from polars, got {v.format!r}; "
                             "this kernel reads the binview layout only")
    layouts = [[c.length for c in v.chunks] for v in string_views]
    if any(l != layouts[0] for l in layouts):
        raise ValueError(f"string columns have different chunk layouts {layouts}; call df.rechunk() first")
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
