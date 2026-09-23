"""The generic tree walker (doc 08 §3.4) — replaces `trees/codegen.py`'s
former nested-`if`/`elif` source emission.

**One `@njit(inline="always")` function, `walk_tree`, walks any tree, ever.**
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

**Two runtime arguments, not the module's compiled kernel arguments
directly.** `feats` carries every feature this ONE tree reads (numeric
columns and hoisted string-matcher outputs alike, the latter cast to
float64 — see `encode.EncodeContext`); `thresholds` carries every
`param()`-backed threshold plus every anonymous literal (a string
matcher's pattern index) this tree needs. Both come from the per-tree
`path_fn` closure `encode._build_path_fn` builds, out of that closure's
own `(args, params)` arguments — never from a module-level global — so
this walker's body has nothing tree-specific in it at all. The STRUCTURE
arrays (`kind`/`feat_idx`/`op`/`thr_slot`/`then_`/`else_`/`leaf_value`),
by contrast, ARE per-tree numpy arrays captured by that closure, and that
is deliberately fine: a closure over plain numpy arrays disk-caches
cleanly under `cache=True`, cold and warm, across a fresh process
(EXPERIMENTS.md §X's cache probe, and this migration's report before it)
— unlike a `ctypes`/`cfunc` pointer captured as a global (§V/§W), which
numba refuses to cache at all.

**The walker is inlined into `path_fn`; `path_fn` is the cached entry.**
`walk_tree` is `@njit(inline="always")`, not `cache=True`: numba splices
its body into every caller at the IR level (the `InlineInlinables` pass,
which runs before typing), so it is never compiled as a function of its
own and has no cache entry of its own — `path_fn` (`cache=True,
inline="always"`) is where this body lives on disk, and in fused mode
`compile.driver.build_packed_kernel`'s per-row loop absorbs `path_fn` in
turn. The reason, and the rule that follows from it, are in `walk_tree`'s
own docstring: **do not put a non-inlined layer between the per-row
kernel and the walker.**
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


@njit(inline="always")
def walk_tree(feats, thresholds, kind, feat_idx, op, thr_slot, then_, else_, leaf_value, start_pc):
    """Walk one tree for one row, returning the leaf's `result_idx`.

    `feats`/`thresholds` are the per-call, per-row row buffer and
    homogeneous tuple `encode._build_path_fn`'s `path_fn` builds from its
    own arguments (never module globals); everything else is that tree's
    fixed structure, addressed by plain array index — no recursion (tree
    depth never touches the call stack, same reasoning as EXPERIMENTS.md
    §Q's `_walk_one`), no per-node call.

    **`inline="always"`, not `cache=True`.** numba splices this body into
    each caller before typing (`InlineInlinables`), so `walk_tree` is never
    compiled as a function of its own: `walk_tree.overloads` stays empty in
    every process and it has no disk-cache entry — `cache=True` here would
    cache nothing. The cached entry is the caller's, `path_fn` (`cache=True,
    inline="always"`), whose entry carries this body and serves the
    interpreted/stepped modes; in fused mode the driver's per-row loop
    absorbs `path_fn`, and this body with it, so the structure arrays are
    constants of the one compiled loop and the row buffer stays in
    registers. Why it matters: a numba array crossing a REAL call is seven
    scalars (meminfo, parent, nitems, itemsize, data, shape, stride) pushed
    and reloaded per call, and this call carries eight arrays per row.
    EXPERIMENTS.md §X measured it on a 16-feature, 40-leaf tree, 200k rows:
    `apply()` 187–234 ns/row with the real call, 100–146 inlined (the walk
    alone 170–199 → 81–102), output digests identical, cold compile
    cheaper, cache saves/loads intact.

    **Do not put a non-inlined layer between the per-row kernel and this
    walker.** Any `@njit` function that takes `feats` plus the structure
    arrays and calls `walk_tree` must itself be `inline="always"` (as
    `path_fn` is), and so must anything wrapped around `path_fn` — one
    real call on that path and the whole cost above comes back.
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
