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
comparison never has a name: it is one row of `kind`/`feat_kind`/`feat_idx`/
`op`/`thr_slot` addressed purely by array position, assigned once, in the
fixed order `encode.EncodeContext` builds them. Two sibling thresholds cannot
collide on a slot by construction — there is no name to collide on.

**Features are split by type, and the type is program data.** `feats` is
the six-tuple `(f64, i64, b8, i32, s64, sbytes)` — one row array per
`decider2.types.FeatureKind` — and every node carries `feat_kind[pc]`
alongside `feat_idx[pc]`, so a node says "read slot 7 of the int64 array",
never "read slot 7 of the one array". The previous walker coerced every
feature into ONE float64 array, which silently collapses any integer above
2**53 (`9007199254740993 == 9007199254740992` answered True) — exactly the
wrong-answer-with-no-signal failure doc 03 §2.1 calls the worst the design
can have, on exactly the scaled-int64 money columns doc 03 §1.2 mandates.
Thresholds are split the same way (`thr_f`/`thr_i`): an int64 feature is
compared against an int64 threshold, or the promotion to float64 would
re-introduce the defect one operand over. `feat_kind` also decides which
threshold tuple `thr_slot` indexes, so one program-data field does both
jobs. Its cost — one extra int32 load and a small switch per node, over a
walk that already loads `kind`/`feat_idx`/`op`/`thr_slot` — is measured,
not assumed, in `evaluation/typed-features/`.

`CODE` (an int32 dictionary code, doc 05 §1.5) and `STR` (the reserved raw-
string span slot, `FeatureKind.STR`) are carried in the tuple so the row
representation is the same one everywhere; nothing in this walker matches
on either yet (a string test is hoisted into its own matcher step, and a
categorical has no order — `encode.EncodeContext` rejects `<` on one at
build time, which the single float64 array could never detect).

**Two runtime-argument bundles, not the module's compiled kernel arguments
directly.** `feats` carries every feature this ONE tree reads (numeric
columns and hoisted string-matcher outputs alike, the latter now an int64
slot); `thr_f`/`thr_i` carry every `param()`-backed threshold plus every
anonymous literal (a string matcher's pattern index) this tree needs. All
come from the per-tree `path_fn` closure `encode._build_path_fn` builds,
out of that closure's own `(args, params)` arguments — never from a
module-level global — so this walker's body has nothing tree-specific in
it at all. The STRUCTURE arrays (`kind`/`feat_kind`/`feat_idx`/`op`/
`thr_slot`/`then_`/`else_`/`leaf_value`), by contrast, ARE per-tree numpy
arrays captured by that closure, and that is deliberately fine: a closure
over plain numpy arrays disk-caches cleanly under `cache=True`, cold and
warm, across a fresh process (EXPERIMENTS.md §X's cache probe, and this
migration's report before it) — unlike a `ctypes`/`cfunc` pointer captured
as a global (§V/§W), which numba refuses to cache at all.

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

from decider2.types import FeatureKind

__all__ = [
    "LT", "LE", "EQ", "GT", "GE", "NE",
    "LEAF", "CMP", "IS_TRUE", "IS_FALSE",
    "F64", "I64", "BOOL", "CODE", "STR",
    "compare",
    "walk_tree",
]

# --- comparison opcodes — decider2.trees.schema._ThresholdedUnaryOp's six
# operators, and nothing else (`between`/`isin`/`string_match`/`is_true`/
# `is_false` all decompose to a chain of these plus the two boolean-only
# opcodes below; see `encode.py`'s `_encode_*` helpers). ---------------------
LT = 0
LE = 1
EQ = 2
GT = 3
GE = 4
NE = 5

# --- feature kinds — `decider2.types.FeatureKind`'s values, bound here as
# plain ints so the njit body below reads them as compile-time constants.
F64 = int(FeatureKind.F64)
I64 = int(FeatureKind.I64)
BOOL = int(FeatureKind.BOOL)
CODE = int(FeatureKind.CODE)
STR = int(FeatureKind.STR)


@njit(cache=True)
def compare(op, a, b):
    """One primitive comparison. A closed switch over six operators — the
    same six `_ThresholdedUnaryOp` subclasses already were — compiled once
    PER OPERAND TYPE PAIR (float64/float64, int64/int64, int32/int64) and
    shared by every tree's every comparison node of that pair."""
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
CMP = 1        # compare(op[pc], <kind array>[feat_idx[pc]], <kind thresholds>[thr_slot[pc]])
IS_TRUE = 2    # <kind array>[feat_idx[pc]] is truthy
IS_FALSE = 3   # <kind array>[feat_idx[pc]] is falsy


@njit(inline="always")
def walk_tree(
    feats, thr_f, thr_i,
    kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value, start_pc,
):
    """Walk one tree for one row, returning the leaf's `result_idx`.

    `feats` is the per-row six-tuple of typed row arrays (`types.Step.
    typed_args`); `thr_f`/`thr_i` the per-call float64/int64 threshold
    tuples; everything else is that tree's fixed structure, addressed by
    plain array index — no recursion (tree depth never touches the call
    stack, same reasoning as EXPERIMENTS.md §Q's `_walk_one`), no per-node
    call. `feat_kind[pc]` picks the row array AND the threshold tuple: a
    `BOOL`/`CODE` node compares against `thr_i` (a bool threshold rides as
    0/1; a code against an int code), an `F64` node against `thr_f`.

    **`inline="always"`, not `cache=True`.** numba splices this body into
    each caller before typing (`InlineInlinables`), so `walk_tree` is never
    compiled as a function of its own: `walk_tree.overloads` stays empty in
    every process and it has no disk-cache entry — `cache=True` here would
    cache nothing. The cached entry is the caller's, `path_fn` (`cache=True,
    inline="always"`), whose entry carries this body and serves the
    interpreted/stepped modes; in fused mode the driver's per-row loop
    absorbs `path_fn`, and this body with it, so the structure arrays are
    constants of the one compiled loop and the row buffers stay in
    registers. Why it matters: a numba array crossing a REAL call is seven
    scalars (meminfo, parent, nitems, itemsize, data, shape, stride) pushed
    and reloaded per call, and this call carries fourteen arrays per row —
    the six row arrays plus eight of structure. EXPERIMENTS.md §X measured
    the inlining on the single-array walker, 16 features, 40 leaves, 200k
    rows: `apply()` 187–234 ns/row with the real call, 100–146 inlined (the
    walk alone 170–199 → 81–102), output digests identical, cold compile
    cheaper, cache saves/loads intact. The typed split is NOT a speed win
    of its own: with the same inlining, the single-array walker measures
    125–133 ns/row `apply()` end to end and this typed walker 136–154 — the
    type discriminator costs ~12–35% against inlining alone
    (`experimentation/verification-probes/README.md`, `run3.sh`). It is
    paid for the int64 correctness above, not for speed.

    **Do not put a non-inlined layer between the per-row kernel and this
    walker.** Any `@njit` function that takes `feats` plus the structure
    arrays and calls `walk_tree` must itself be `inline="always"` (as
    `path_fn` is), and so must anything wrapped around `path_fn` — one
    real call on that path and the whole cost above comes back.
    """
    f64, i64, b8, i32, s64, sbytes = feats
    pc = start_pc
    while True:
        k = kind[pc]
        if k == LEAF:
            return leaf_value[pc]
        fk = feat_kind[pc]
        j = feat_idx[pc]
        if k == CMP:
            o = op[pc]
            if fk == F64:
                r = compare(o, f64[j], thr_f[thr_slot[pc]])
            elif fk == I64:
                r = compare(o, i64[j], thr_i[thr_slot[pc]])
            elif fk == BOOL:
                r = compare(o, b8[j], thr_i[thr_slot[pc]] != 0)
            else:  # CODE
                r = compare(o, i32[j], thr_i[thr_slot[pc]])
        else:
            if fk == F64:
                t = f64[j] != 0.0
            elif fk == I64:
                t = i64[j] != 0
            elif fk == BOOL:
                t = b8[j]
            else:  # CODE
                t = i32[j] != 0
            r = t if k == IS_TRUE else not t
        pc = then_[pc] if r else else_[pc]
