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
bound, has, has-second)`-shaped row of table data — `between_lo`/
`between_hi`/`between_has_lo`/`between_has_hi` and `eq_val`/`eq_has`, each
one 2D float64 array of shape `(conditions of that kind, n_rows)`, indexed
`arr[local, r]` (`EQ` uses two of the four arrays BETWEEN uses, kept
*separate* rather than padded alongside BETWEEN's, so there is nothing to
accidentally read past); `IS_TRUE` reads no per-row-of-TABLE data at all;
`IN` is CSR (decider 1's own ragged-set shape, `InExpression`'s
docstring), `in_off` a 2D `(conditions, n_rows + 1)` array (uniform width:
every condition's own offsets array is `n_rows + 1` long) and `in_vals` a
SECOND-LEVEL CSR — every condition's own (ragged-length) values
concatenated end to end, `in_vals_start[local]` the base a given
condition's own offsets are relative to.

**Why 2D arrays, not `decider2.tables.encode`'s previous one-array-tuple-
per-condition.** A condition COUNT used to be encoded directly in a numba
tuple TYPE (`between_bounds: tuple of N (four-array) tuples`) — one
hand-written Python closure body assembled that tuple per condition COUNT
`decider2.tables.encode` had ever seen (`_compose0`..`_compose8`), a wall
past 8 conditions of one kind in a whole table. A 2D array's numba TYPE
carries only its dtype and dimensionality, never its shape — so `local`
becomes an ordinary runtime array index (`arr[local, r]`) instead of a
runtime index into a fixed-length tuple, and there is no condition count
past which this raises. See this module's report.
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
    between_lo, between_hi, between_has_lo, between_has_hi,
    eq_val, eq_has,
    in_off, in_vals, in_vals_start,
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
    condition count or And/Or nesting (see this module's docstring for why
    `local` indexes a 2D array now, not a tuple).
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
                    has_lo = between_has_lo[local, r]
                    has_hi = between_has_hi[local, r]
                    if has_lo != 0.0 and not compare(op_lo_op[i], var, between_lo[local, r]):
                        ok = False
                        break
                    if has_hi != 0.0 and not compare(op_hi_op[i], var, between_hi[local, r]):
                        ok = False
                        break
                elif k == EQ:
                    if eq_has[local, r] != 0.0 and var != eq_val[local, r]:
                        ok = False
                        break
                elif k == IS_TRUE:
                    if var == 0.0:
                        ok = False
                        break
                else:  # IN — CSR membership, decider 1's ragged set shape,
                    # a SECOND level of CSR across conditions: condition
                    # `local`'s own values are `in_vals[base:base+len]`,
                    # `base = in_vals_start[local]` (this module's docstring).
                    base = in_vals_start[local]
                    start = base + in_off[local, r]
                    end = base + in_off[local, r + 1]
                    if start == end:
                        ok = False
                        break
                    hit = False
                    for j in range(start, end):
                        if var == in_vals[j]:
                            hit = True
                            break
                    if not hit:
                        ok = False
                        break
            if ok:
                return r
    return -1
