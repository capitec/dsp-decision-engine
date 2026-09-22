"""The generic tree walker (doc 08 §3.4) — replaces `trees/codegen.py`'s
former nested-`if`/`elif` source emission.

**One `@njit(cache=True)` function, `walk_tree`, walks any tree, ever.**
`decider2.trees.codegen` no longer emits per-node source: every node and
condition class in `trees/schema.py` implements `encode(ctx, ...)`, appending
its own row(s) to flat int32/float64 arrays instead of returning a source
string. Building a NEW tree is building new arrays in Python — a data
transformation — never a new compile of branching logic. EXPERIMENTS.md §Q
found this shape (inline array walk, not a call per node) already beats a
per-node function-pointer interpreter for a tree specifically, because a
tree node is a comparison, not a computation — see that section and its
sibling §W before touching this file.

**Registers, not identifiers.** The historical bug this migration exists to
retire — two sibling conditions emitting the same generated parameter name,
so `5 < x < 10` compiled to `(x > root_thr) and (x < root_thr)`, never
satisfiable — was a defect of *naming* comparisons in source text. Here a
comparison never has a name: it is one row of `kind`/`feat_idx`/`op`/
`thr_slot` addressed purely by array position, assigned once, in the fixed
order `codegen.EncodeContext` builds them. Two sibling thresholds cannot
collide on a slot by construction — there is no name to collide on.

**Two runtime-argument tuples, not the module's compiled kernel arguments
directly.** `feats` carries every feature this ONE tree reads (numeric
columns and hoisted string-matcher outputs alike, the latter cast to
float64 — see `codegen.EncodeContext.feature_slot`); `thresholds` carries
every `param()`-backed threshold plus every anonymous literal (a string
matcher's pattern index) this tree needs. Both are built by the tiny
per-tree wrapper function `codegen.emit_tree` writes, from that function's
own named arguments — never from a module-level global — so this kernel's
own compiled body has nothing tree-specific baked into it at all. The
STRUCTURE arrays (`kind`/`feat_idx`/`op`/`thr_slot`/`then_`/`else_`/
`leaf_value`), by contrast, ARE per-tree module-level globals in that
wrapper file, and that is deliberately fine: doc 05 §4.1's caching contract
only requires a real, content-addressed source file, and EXPERIMENTS.md's
own cache probe (this migration's report) confirms a plain numpy-array
global referenced inside an `@njit(cache=True)` function disk-caches
cleanly, cold and warm, across a fresh process — unlike a `ctypes`/`cfunc`
pointer captured as a global (§V/§W), which numba refuses to cache at all.
"""
from __future__ import annotations

from numba import njit

__all__ = [
    "LT", "LE", "EQ", "GT", "GE", "NE",
    "LEAF", "CMP", "IS_TRUE", "IS_FALSE",
    "compare",
    "walk_tree",
]

# --- comparison opcodes — decider2.trees.schema._ThresholdedUnaryOp's six
# operators, and nothing else (`between`/`isin`/`string_match`/`is_true`/
# `is_false` all decompose to a chain of these plus the two boolean-only
# opcodes below; see `codegen.py`'s `_encode_*` helpers). ---------------------
LT = 0
LE = 1
EQ = 2
GT = 3
GE = 4
NE = 5


@njit(cache=True)
def compare(op, a, b):
    """One primitive comparison. A closed switch over six operators — the
    same six `_ThresholdedUnaryOp` subclasses already were — compiled once,
    shared by every tree's every comparison node."""
    if op == LT:
        return a < b
    elif op == LE:
        return a <= b
    elif op == EQ:
        return a == b
    elif op == GT:
        return a > b
    elif op == GE:
        return a >= b
    else:  # NE
        return a != b


# --- node kinds ---------------------------------------------------------
LEAF = 0       # leaf_value[pc] is this tree's result_idx; walk stops here
CMP = 1        # compare(op[pc], feats[feat_idx[pc]], thresholds[thr_slot[pc]])
IS_TRUE = 2    # feats[feat_idx[pc]] != 0.0
IS_FALSE = 3   # feats[feat_idx[pc]] == 0.0


@njit(cache=True)
def walk_tree(feats, thresholds, kind, feat_idx, op, thr_slot, then_, else_, leaf_value, start_pc):
    """Walk one tree for one row, returning the leaf's `result_idx`.

    `feats`/`thresholds` are the per-call, per-row homogeneous tuples the
    wrapper function builds from its own arguments (never module globals);
    everything else is that tree's fixed structure, addressed by plain
    array index — no recursion (tree depth never touches the call stack,
    same reasoning as EXPERIMENTS.md §Q's `_walk_one`), no per-node call.
    """
    pc = start_pc
    while True:
        k = kind[pc]
        if k == LEAF:
            return leaf_value[pc]
        elif k == CMP:
            a = feats[feat_idx[pc]]
            b = thresholds[thr_slot[pc]]
            pc = then_[pc] if compare(op[pc], a, b) else else_[pc]
        elif k == IS_TRUE:
            pc = then_[pc] if feats[feat_idx[pc]] != 0.0 else else_[pc]
        else:  # IS_FALSE
            pc = then_[pc] if feats[feat_idx[pc]] == 0.0 else else_[pc]
