"""ONE generic njit kernel that walks ANY array-encoded tree.

Compiled once, ever, for a given (numeric-columns, string-columns) dtype
pair — which is always (float64 2D, int32 2D) here, so in practice this
compiles exactly once for the whole process, regardless of how many
different tree *shapes* are thrown at it afterward. That is the entire
claim under test; `run_experiment.py` verifies `WALK_BATCH.signatures`
stays at length 1 across all four shapes.

Design choices, per the brief:
  * struct-of-arrays (`tree_shapes.FlatTree`), not array-of-structs.
  * explicit index loop over nodes (`node = left/right[node]`), no
    recursion — no stack limit, unlike a naive recursive walker.
  * flat typed numpy arrays only: no Python objects, no dicts, no
    `typed.List`.
  * `njit(cache=True)`, matching decider2's own `_try_njit` convention for
    every compiled kernel it ships.
"""
from __future__ import annotations

from numba import njit

KIND_LEAF, KIND_NUM, KIND_STR = 0, 1, 2
OP_LT, OP_LE, OP_EQ, OP_GT, OP_GE, OP_NE = 0, 1, 2, 3, 4, 5


@njit(cache=True)
def _walk_one(row, str_row, kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns, left, right, leaf_value):
    node = 0
    while kind[node] != KIND_LEAF:
        if kind[node] == KIND_NUM:
            f = row[feat_idx[node]]
            t = thresh[node]
            op = op_code[node]
            if op == OP_LT:
                cond = f < t
            elif op == OP_LE:
                cond = f <= t
            elif op == OP_EQ:
                cond = f == t
            elif op == OP_GT:
                cond = f > t
            elif op == OP_GE:
                cond = f >= t
            else:
                cond = f != t
        else:  # KIND_STR: OR of exact dictionary-code matches
            code = str_row[feat_idx[node]]
            cond = False
            start = pat_start[node]
            end = start + pat_count[node]
            for k in range(start, end):
                if code == patterns[k]:
                    cond = True
                    break
        node = left[node] if cond else right[node]
    return leaf_value[node]


@njit(cache=True)
def walk_batch(numeric_cols, string_cols, kind, feat_idx, op_code, thresh,
                pat_start, pat_count, patterns, left, right, leaf_value, out):
    """Explicit row loop over `_walk_one` — numba inlines the call, so this
    is one compiled loop over the batch, the same shape decider2's own
    decision-table kernel uses (rows in `shared` arrays, one generic
    loop). No recursion anywhere: `_walk_one`'s `while` walks parent to
    child by array index, so tree depth never touches the call stack."""
    n = numeric_cols.shape[0]
    for i in range(n):
        out[i] = _walk_one(
            numeric_cols[i], string_cols[i], kind, feat_idx, op_code, thresh,
            pat_start, pat_count, patterns, left, right, leaf_value,
        )
