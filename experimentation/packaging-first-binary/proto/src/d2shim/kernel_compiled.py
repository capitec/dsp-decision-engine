"""kernel_b.py from experimentation/nanoarrow-accessors, unchanged except
for imports: the STR node asks nanoarrow for each row's bytes through
`sm_get_string` (a function-pointer ARGUMENT), then byte-matches with the
pure kernel's matcher. The tree walk is the pure kernel's, verbatim."""
from __future__ import annotations

import numpy as np
from numba import njit

from .compiled import DEFAULT, GET_STRING_ADDR, GET_STRING_CHECKED_ADDR, NanoView, call_get_string
from .pure import (CMP, EQ, ERR_LEAF, GE, GT, IS_TRUE, LE, LEAF, LT, NE, _match_at,  # noqa: F401
                   pattern_table)


@njit(cache=True)
def walk_chunk_b(
    feats,
    get_string_addr,   # uint64: address of sm_get_string (an ARGUMENT)
    str_view_addr,     # uint64[ncol]: struct ArrowArrayView* per string column
    pat_bytes, pat_bounds,
    thresholds, kind, feat_idx, op, thr_slot, then_, else_, leaf_value,
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
            else:  # STR: one call into nanoarrow, then a byte match
                s_addr, ln = call_get_string(get_string_addr, str_view_addr[feat_idx[pc]], i)
                if ln == -1:          # null never matches
                    pc = else_[pc]
                    continue
                if ln < -1:           # the checked accessor refused this element
                    out[out_off + i] = ERR_LEAF
                    break
                p = thr_slot[pc]
                poff = pat_bounds[p, 0]
                hit = _match_at(np.uint64(s_addr), ln, pat_base + np.uint64(poff),
                                pat_bounds[p, 1] - poff, op[pc])
                pc = then_[pc] if hit else else_[pc]


def run_tree_compiled(float_cols, series_list, tree, patterns, out=None, checked=True):
    """Same contract as pure.run_tree but takes polars Series (it does its own
    import). `checked=True` uses sm_get_string_checked (bounds-checked, the
    variant that matches the pure kernel's behaviour on corrupt input)."""
    nano_views = [NanoView(s) for s in series_list]
    try:
        n = nano_views[0].n_rows if nano_views else float_cols.shape[0]
        if out is None:
            out = np.empty(n, dtype=np.int64)
        pat_bytes, pat_bounds = pattern_table(patterns)
        fn = np.uint64(GET_STRING_CHECKED_ADDR if checked else GET_STRING_ADDR)
        if checked:
            for v in nano_views:
                v.validate(DEFAULT)
        n_chunks = len(nano_views[0].chunks) if nano_views else 1
        layouts = [[c.array.length for c in v.chunks] for v in nano_views]
        if any(l != layouts[0] for l in layouts):
            raise ValueError(f"string columns have different chunk layouts {layouts}; call df.rechunk() first")
        off = 0
        for ci in range(n_chunks):
            cn = nano_views[0].chunks[ci].array.length if nano_views else n
            addrs = np.array([v.chunks[ci].view_addr for v in nano_views], dtype=np.uint64)
            walk_chunk_b(
                float_cols[off: off + cn], fn, addrs, pat_bytes, pat_bounds,
                tree["thresholds"], tree["kind"], tree["feat_idx"], tree["op"], tree["thr_slot"],
                tree["then_"], tree["else_"], tree["leaf_value"], out, off, cn,
            )
            off += cn
        return out
    finally:
        for v in nano_views:
            v.release()
