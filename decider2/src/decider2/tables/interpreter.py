"""The generic decision-table row scanner (doc 08 §3.4) — replaces
`tables/codegen.py`'s former per-condition `if ok and ...: ok = False`
source lines.

**One `@njit(cache=True)` function, `scan_table`, scans any table, ever.**
A table was already doc 08 §3.4's "generic kernel" case — "N rows x M
condition columns, uniform operators" — and every ROW was already data,
passed through `shared` (doc 03 §4.2), never emitted source. What this
module retires is the last piece that was still text: the EXPRESSION shape
itself (how many conditions, which operator each is, how And/Or groups
them). `decider2.tables.schema.Expression.emit()` now returns one `CondOp`
per leaf condition instead of source lines; `decider2.tables.codegen`
flattens the DNF groups `to_dnf()` already produces into flat `op_*`
arrays; this module walks them.

**Reuses `decider2.trees.interpreter`'s comparison opcodes.** A `between`
condition's bound test is the exact same "which of the four ops, decided
once by `BoundMode`, not per row" shape a tree's `RangeCondition` already
has — importing `compare`/`LT`/`LE`/`GT`/`GE` rather than a second copy
keeps the two engines from drifting on what `lower_inclusive` means.

**Four condition kinds, four fixed shapes — never a fifth via a generic
"N-ary" path.** `BETWEEN`/`EQ` read from a per-condition `(bound, second
bound, has, has-second)`-shaped row of table data (`between_bounds`/
`eq_bounds`, one homogeneous tuple of arrays each — `EQ` uses two of the
four slots BETWEEN uses, kept as a *separate*, narrower tuple type rather
than padded into BETWEEN's, so there is nothing to accidentally read past);
`IS_TRUE` reads no per-row-of-TABLE data at all; `IN` is CSR (decider 1's
own ragged-set shape, `InExpression`'s docstring), a separate pair of
homogeneous tuples. Each is exactly the shape `tables/schema.py`'s
corresponding `Expression.emit()` already built as `EmittedCondition.
arrays` before this migration — unchanged by it; only how those arrays get
read at scan time has moved, from generated `if` lines to this fixed
switch.
"""
from __future__ import annotations

from numba import njit

from decider2.trees.interpreter import GE, GT, LE, LT, compare

__all__ = [
    "BETWEEN", "EQ", "IS_TRUE", "IN", "scan_table",
    # Re-exported so a caller building op_lo_op/op_hi_op arrays
    # (decider2.tables.codegen) names the same four opcodes
    # decider2.trees.interpreter does, without importing that module
    # purely for four integers.
    "LT", "LE", "GT", "GE",
]

# --- condition kinds — decider2.tables.schema's four leaf Expression types,
# and nothing else (AndExpression/OrExpression flatten away in `to_dnf()`
# before any of this runs). -------------------------------------------------
BETWEEN = 0
EQ = 1
IS_TRUE = 2
IN = 3


@njit(cache=True)
def scan_table(
    vars_, n_rows,
    group_start, group_end,
    op_kind, op_var_idx, op_local, op_lo_op, op_hi_op,
    between_bounds, eq_bounds, in_offsets, in_values,
):
    """Index of the first row that matches every condition in some DNF
    group, in group order — decider 1's `calculate_decision_table_output`
    semantics exactly (first `when` wins), or -1.

    `vars_` is the per-call, per-record homogeneous tuple the wrapper
    function builds from its own arguments (the table's runtime inputs —
    plain columns and hoisted string-matcher outputs alike, the latter cast
    to float64, same convention as a tree's `feats`). Everything else is
    this table's fixed shape plus its rows, both addressed by plain array
    index — no recursion, no per-condition Python-level dispatch, one
    `@njit` function shared by every table regardless of row count,
    condition count or And/Or nesting.
    """
    n_groups = group_start.shape[0]
    for r in range(n_rows):
        for g in range(n_groups):
            ok = True
            for i in range(group_start[g], group_end[g]):
                k = op_kind[i]
                var = vars_[op_var_idx[i]]
                local = op_local[i]
                if k == BETWEEN:
                    b = between_bounds[local]
                    lo = b[0][r]
                    hi = b[1][r]
                    has_lo = b[2][r]
                    has_hi = b[3][r]
                    if has_lo != 0.0 and not compare(op_lo_op[i], var, lo):
                        ok = False
                        break
                    if has_hi != 0.0 and not compare(op_hi_op[i], var, hi):
                        ok = False
                        break
                elif k == EQ:
                    e = eq_bounds[local]
                    val = e[0][r]
                    has = e[1][r]
                    if has != 0.0 and var != val:
                        ok = False
                        break
                elif k == IS_TRUE:
                    if var == 0.0:
                        ok = False
                        break
                else:  # IN — CSR membership, decider 1's ragged set shape
                    off = in_offsets[local]
                    vals = in_values[local]
                    start = off[r]
                    end = off[r + 1]
                    if start == end:
                        ok = False
                        break
                    hit = False
                    for j in range(start, end):
                        if var == vals[j]:
                            hit = True
                            break
                    if not hit:
                        ok = False
                        break
            if ok:
                return r
    return -1
