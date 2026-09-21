"""Q2 prototype: a tree walked by COMPOSED, CACHED njit callables, not by
codegen and not by one generic interpreter loop with an in-loop kind-switch.

This is the third shape named in the brief: "each function returns a
callable next type that gets pushed up into a single function" (the owner's
words, describing polars' physical-plan executor tree) — ported to numba.

Design, and why it is shaped exactly this way:

Three kernels total, ever, for the whole process, regardless of how many
trees or how many nodes any of them has:

  * `leaf_eval`   — kind=LEAF
  * `num_eval`    — kind=NUM  (numeric threshold test)
  * `str_eval`    — kind=STR  (string membership test)

Each has the SAME signature — `(numeric_row, string_row, node_idx, kind,
feat_idx, op_code, thresh, pat_start, pat_count, patterns, left, right,
leaf_value) -> (is_leaf: bool, next_index_or_value: float64)` — reading this
node's own data out of the SAME struct-of-arrays `FlatTree` the sibling
interpreted-kernel experiment uses (`../tree-codegen-vs-interpreted/tree_shapes.py`,
imported unmodified below). A "plan" for a specific tree is NOT new source
and is NOT a new compile: it is just those arrays, plus a 3-entry
`numba.typed.List[FunctionType]` (`FN_TABLE`, built once at import time)
mapping `kind -> which kernel`. Composing/recomposing a tree — the thing a
config UI's structural edit does — is pure data assembly (numpy arrays),
never a numba compile, which is the entire promise of the pattern.

The mechanism under test is the DISPATCH, not the data layout. The sibling
interpreter's `_walk_one` picks the next behaviour with an in-loop
`if kind == NUM: ... elif kind == STR: ... else: leaf`, a direct compile-time
branch LLVM can predict and inline around. This module's `walk_one_composed`
instead does `fn = FN_TABLE[kind[idx]]; is_leaf, val = fn(...)` — an INDIRECT
call through a first-class function value pulled out of a runtime list.
Numba cannot know at compile time which of the three concrete functions
`fn` will be, so it cannot inline the call — exactly the risk the brief
names: "an indirect call through a function value prevents inlining."
Every other design choice (SoA arrays, explicit index loop, no recursion,
no Python objects in the hot path, `njit(cache=True)`) is held identical to
the interpreter so that ns/row differences are attributable to ONE variable:
direct branch vs. indirect call.

`node_idx` is threaded through explicitly rather than baking any per-node
constant (threshold, feature index, pattern) into a closure — matching doc
05 §4.2's rule that a decision-relevant constant is a *value*, never
something that forces a recompile. Retuning a threshold is a data change to
`thresh[idx]`; restructuring the tree is a data change to
`left`/`right`/`kind`/... plus a `FN_TABLE` re-lookup per node kind used
(itself already built, never rebuilt). Neither ever touches numba.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, types
from numba.typed import List as NList

sys.path.insert(0, str(Path(__file__).parent.parent / "tree-codegen-vs-interpreted"))
from tree_shapes import FlatTree, KIND_LEAF, KIND_NUM, KIND_STR  # noqa: E402

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_num_row_ty = types.float64[::1]
_str_row_ty = types.int32[::1]
_i8a = types.int8[::1]
_i32a = types.int32[::1]
_f64a = types.float64[::1]

_ret_ty = types.Tuple((types.boolean, types.float64))
NODE_SIG = _ret_ty(
    _num_row_ty, _str_row_ty, types.int64,
    _i8a, _i32a, _i8a, _f64a, _i32a, _i32a, _i32a, _i32a, _i32a, _f64a,
)
NODE_ARG_TYPES = NODE_SIG.args
NODE_FNTY = types.FunctionType(NODE_SIG)


# ---------------------------------------------------------------------------
# The three kernels — compiled exactly once, ever (verified in run_experiment.py
# via `.signatures`), reused unchanged across every tree and every structural edit.
# ---------------------------------------------------------------------------

def _leaf_eval_py(numeric_row, string_row, idx, kind, feat_idx, op_code, thresh,
                   pat_start, pat_count, patterns, left, right, leaf_value):
    return (True, leaf_value[idx])


def _num_eval_py(numeric_row, string_row, idx, kind, feat_idx, op_code, thresh,
                  pat_start, pat_count, patterns, left, right, leaf_value):
    f = numeric_row[feat_idx[idx]]
    t = thresh[idx]
    op = op_code[idx]
    if op == 0:
        cond = f < t
    elif op == 1:
        cond = f <= t
    elif op == 2:
        cond = f == t
    elif op == 3:
        cond = f > t
    elif op == 4:
        cond = f >= t
    else:
        cond = f != t
    nxt = left[idx] if cond else right[idx]
    return (False, np.float64(nxt))


def _str_eval_py(numeric_row, string_row, idx, kind, feat_idx, op_code, thresh,
                  pat_start, pat_count, patterns, left, right, leaf_value):
    code = string_row[feat_idx[idx]]
    cond = False
    start = pat_start[idx]
    end = start + pat_count[idx]
    for k in range(start, end):
        if code == patterns[k]:
            cond = True
            break
    nxt = left[idx] if cond else right[idx]
    return (False, np.float64(nxt))


leaf_eval = njit(cache=True)(_leaf_eval_py)
num_eval = njit(cache=True)(_num_eval_py)
str_eval = njit(cache=True)(_str_eval_py)

# Explicit-signature compile now (decider2's own `_try_njit` convention:
# `njit(cache=True)(fn)` then `.compile(sig)`), which is also what forces
# these three to exist as concrete FunctionType-castable dispatchers before
# they go into FN_TABLE.
leaf_eval.compile(NODE_ARG_TYPES)
num_eval.compile(NODE_ARG_TYPES)
str_eval.compile(NODE_ARG_TYPES)

# The plan-independent, built-once function table. Index == kind code.
FN_TABLE = NList.empty_list(NODE_FNTY)
FN_TABLE.append(leaf_eval)   # index 0 == KIND_LEAF
FN_TABLE.append(num_eval)    # index 1 == KIND_NUM
FN_TABLE.append(str_eval)    # index 2 == KIND_STR
assert (KIND_LEAF, KIND_NUM, KIND_STR) == (0, 1, 2)


# ---------------------------------------------------------------------------
# The driver — the ONLY thing that changes shape is which arrays it is
# handed; no new source is ever generated for a new tree.
# ---------------------------------------------------------------------------

@njit(cache=True)
def walk_one_composed(numeric_row, string_row, kind, feat_idx, op_code, thresh,
                       pat_start, pat_count, patterns, left, right, leaf_value, fn_table):
    idx = 0
    while True:
        fn = fn_table[kind[idx]]
        is_leaf, val = fn(
            numeric_row, string_row, idx, kind, feat_idx, op_code, thresh,
            pat_start, pat_count, patterns, left, right, leaf_value,
        )
        if is_leaf:
            return val
        idx = np.int64(val)


@njit(cache=True)
def walk_batch_composed(numeric_cols, string_cols, kind, feat_idx, op_code, thresh,
                         pat_start, pat_count, patterns, left, right, leaf_value,
                         fn_table, out):
    n = numeric_cols.shape[0]
    for i in range(n):
        out[i] = walk_one_composed(
            numeric_cols[i], string_cols[i], kind, feat_idx, op_code, thresh,
            pat_start, pat_count, patterns, left, right, leaf_value, fn_table,
        )


def build_plan(flat: FlatTree):
    """'Compose the plan': pure data assembly, no numba compile call at all.
    Returns the exact positional-arg tuple `walk_batch_composed`/`walk_one_composed`
    expect, plus the shared FN_TABLE (already built at import time)."""
    return (
        flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
        flat.pat_start, flat.pat_count, flat.patterns, flat.left, flat.right,
        flat.leaf_value, FN_TABLE,
    )


def run_batch(flat: FlatTree, numeric_cols: np.ndarray, string_cols: np.ndarray) -> np.ndarray:
    out = np.empty(numeric_cols.shape[0], dtype=np.float64)
    args = build_plan(flat)
    walk_batch_composed(numeric_cols, string_cols, *args, out)
    return out


def signature_counts() -> dict:
    return {
        "leaf_eval": len(leaf_eval.signatures),
        "num_eval": len(num_eval.signatures),
        "str_eval": len(str_eval.signatures),
        "walk_one_composed": len(walk_one_composed.signatures),
        "walk_batch_composed": len(walk_batch_composed.signatures),
    }
