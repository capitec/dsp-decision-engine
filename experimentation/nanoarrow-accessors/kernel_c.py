"""Approach C: nanoarrow validates the batch once (`ArrowArrayViewValidate`
at a chosen level), then the incumbent's pure-numba per-row decode runs.

Two per-row kernels:
  * `walk_chunk` -- the incumbent, imported unchanged (keeps its own bounds
    checks; the validation is an extra gate in front of it).
  * `walk_chunk_trusting` -- the incumbent with its two bounds checks
    REMOVED, i.e. what one would write if nanoarrow's ValidateFull really
    made every element provably in bounds. Exists to test that hypothesis.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from kernel import (CMP, EQ, GE, GT, IS_TRUE, LE, LEAF, LT, NE, _is_valid, _match_at, _view,  # noqa: F401
                    pattern_table, string_tables, walk_chunk)
from nashim import FULL, NanoView


@njit(cache=True)
def walk_chunk_trusting(
    feats,
    str_views_addr, str_validity_addr, str_validity_off,
    str_n_data, str_data_base, str_data_addr, str_data_size,
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
            else:
                col = feat_idx[pc]
                if not _is_valid(str_validity_addr[col], i + str_validity_off[col]):
                    pc = else_[pc]
                    continue
                va = str_views_addr[col]
                ln, bi, off = _view(va, i)
                if ln <= 12:
                    s_addr = va + np.uint64(16 * i + 4)
                else:
                    # NO bounds checks: trusting the up-front validation
                    s_addr = str_data_addr[str_data_base[col] + bi] + np.uint64(off)
                p = thr_slot[pc]
                poff = pat_bounds[p, 0]
                hit = _match_at(s_addr, ln, pat_base + np.uint64(poff), pat_bounds[p, 1] - poff, op[pc])
                pc = then_[pc] if hit else else_[pc]


def run_tree_c(float_cols, nano_views: list[NanoView], tree, patterns, out=None, level=FULL,
               trusting=False, pat=None):
    """Validate every string column at `level` (None = skip), then walk."""
    n = nano_views[0].n_rows if nano_views else float_cols.shape[0]
    if out is None:
        out = np.empty(n, dtype=np.int64)
    pat_bytes, pat_bounds = pat if pat is not None else pattern_table(patterns)
    for v in nano_views:
        if v.format != "vu":
            raise ValueError(f"expected Arrow Utf8View ('vu') from polars, got {v.format!r}; "
                             "the per-row numba decode reads the binview layout only")
        if level is not None:
            v.validate(level)
    kern = walk_chunk_trusting if trusting else walk_chunk
    n_chunks = len(nano_views[0].chunks) if nano_views else 1
    layouts = [[c.array.length for c in v.chunks] for v in nano_views]
    if any(l != layouts[0] for l in layouts):
        raise ValueError(f"string columns have different chunk layouts {layouts}; call df.rechunk() first")
    off = 0
    for ci in range(n_chunks):
        chunks = [v.arrowc_chunks()[ci] for v in nano_views]
        cn = chunks[0].length if chunks else n
        tabs = string_tables(chunks)
        kern(
            float_cols[off: off + cn], *tabs, pat_bytes, pat_bounds,
            tree["thresholds"], tree["kind"], tree["feat_idx"], tree["op"], tree["thr_slot"],
            tree["then_"], tree["else_"], tree["leaf_value"], out, off, cn,
        )
        off += cn
    return out
