"""Null handling at the boundary — doc 05 §2, and doc 03 §1's four situations.

**The slot under a null contains leftover garbage, not zero.** A measured left
join produced `9.0` at a position that was genuinely null (EXPERIMENTS.md §A;
doc 05 §2). Nothing here — and nothing in the C gather — reads a value whose
validity bit is clear (docs/BOUNDARY-REWORK.md §1.5: "sentinels are gone"):

    * `NullPolicy.MISSING_AS` / `NOT_APPLICABLE_AS` (tiers 2 & 4) — filled
      by `sm_gather_row` itself, from the fill value that rides in the
      column's `ColDesc` (`_arrow/frame.py`, `ColumnSpec.fill`), so the
      compiled kernel sees a plain number and there is nothing to forget.
      No `np.where` copy any more. They differ only in which reason code
      the fill is tagged with (`FillReason`) — the step body never knows
      which situation produced the value it sees (doc 03 §1, "the fourth
      situation: not-applicable is not missing"). `FillInfo` is computed
      from the gather's per-column validity output.
    * `NullPolicy.OPTIONAL` (tier 3) — *not* filled. `extract.py` passes the
      gather's per-column validity alongside the values, and the compiled
      driver builds a numba `Optional` per row (doc 05 §2).
    * `NullPolicy.REQUIRED` (tier 1) — **routed, not raised, by default**,
      at the frame level, before anything is exported (`route_required_
      nulls`, below — unchanged in what it does).

The last point is the design's highest-ranked production risk (doc 03 §1, "A
null must be able to produce a decision, not an exception"): on a 400-input
realtime path a hard fail-fast per null turns an ordinary business outcome
(an applicant who cannot be scored) into a 500. So `route_required_nulls`
implements doc 03 §1's `pipeline.on_missing_input(default=..., raise_for=...)`
contract — a null in a required column routes its whole row to a `Decision`
(`policy.default`, e.g. `Decision.REFER`) and never reaches the kernel, unless
that column is named in `policy.raise_for`, which is doc 05 §2's hard-fail
message shape: a column declared genuinely incapable of being absent (an
application id) still fails the whole batch loudly, by name.

**Reconciling two spec passages that read as contradictory.** Doc 05 §2's
"Required-input validation" section shows only the hard-fail message and
calls it the mechanism for `x: float` (no default). Doc 03 §1, written later
in the same review pass that produced doc 00-BUILD.md's "the fable review"
commit, overrides that as the *default* pipeline behaviour and keeps doc 05
§2's message for exactly the `raise_for` carve-out. That is what this module
implements; see this package's report for the ambiguity flagged in full.

The per-tier strategy registry this module used to carry (`NullTierStrategy`,
`NULL_TIER_STRATEGIES`, `fill_column`, `validity_mask`) went with the dtype
ladder: there is one consumer of a null policy left (`extract.py`, one
place), and the fill itself is a C `switch` now.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import polars as pl

from decider2.types import Decision, Input, MissingInputPolicy, NullPolicy

__all__ = [
    "FillReason",
    "FillInfo",
    "NullRouting",
    "fill_reason",
    "route_required_nulls",
]


class FillReason(Enum):
    """Doc 03 §1's fourth situation: filling is one mechanism, but the reason
    code it is tagged with must distinguish "this did not apply" (a complete
    application) from "this was absent" (an incomplete one) — collapsing them
    silently approves the second. `observe/record.py` owns the numeric
    registry; this is the structural fact that module needs to consume.
    """

    MISSING = "missing_as"
    NOT_APPLICABLE = "not_applicable_as"


_FILL_REASON: dict[NullPolicy, FillReason] = {
    NullPolicy.MISSING_AS: FillReason.MISSING,
    NullPolicy.NOT_APPLICABLE_AS: FillReason.NOT_APPLICABLE,
}


def fill_reason(policy: NullPolicy) -> FillReason | None:
    """The reason a fill under `policy` is tagged with — `None` for the two
    policies that never fill (REQUIRED routes, OPTIONAL masks)."""
    return _FILL_REASON.get(policy)


@dataclass(frozen=True)
class FillInfo:
    """What happened when a tier-2/4 column was filled."""

    reason: FillReason
    filled_count: int
    filled_mask: np.ndarray | None  # True where a fill was substituted; None if nothing was


@dataclass(frozen=True)
class NullRouting:
    """Doc 03 §1's routing decision, computed over a whole frame at once.

    `mask[i]` is True for a row that must NOT reach the kernel — it carries a
    null in at least one `REQUIRED` column not listed in `raise_for`.
    `column[i]` names the first such column (declaration order), for the
    audit record (doc 03 §6). `decision`/`reason` are the pipeline's policy
    values, carried alongside so a caller doesn't have to re-derive which
    policy this routing was computed under — `MissingInputPolicy` is one
    policy per pipeline (doc 03 §1), not one per column.
    """

    mask: np.ndarray                    # bool[n_rows]
    column: tuple[str | None, ...]      # offending input name per row, or None
    decision: Decision
    reason: int

    @property
    def routed_count(self) -> int:
        return int(self.mask.sum())


def _required_absent_message(name: str) -> str:
    """Doc 05 §2's message shape, for a `raise_for` column that is not
    merely null but entirely absent from the frame (review finding 4:
    absent and null share one routing path, so a `raise_for` column being
    absent gets the same hard, named failure a `raise_for` column full of
    nulls gets — not a bare `KeyError` deep inside the kernel call)."""
    return (
        f"step argument '{name}' is declared required (no `| None`) but "
        f"column '{name}' is not present in the input frame at all. Either "
        f"supply the column or declare `{name}: <type> | None` (doc 03 §1)."
    )


def _required_null_message(name: str, series: pl.Series, bad_rows: np.ndarray) -> str:
    """Doc 05 §2's exact message shape, used only for `raise_for` columns."""
    n_bad = len(bad_rows)
    first = int(bad_rows[0])
    examples = ", ".join(str(int(i)) for i in bad_rows[:3])
    py_type = {
        pl.Float64: "float", pl.Int64: "int", pl.Int32: "int", pl.UInt8: "int",
        pl.Boolean: "bool", pl.String: "str",
    }.get(series.dtype.base_type(), str(series.dtype))
    return (
        f"step argument '{name}' is declared required (`{series.dtype}`, no "
        f"`| None`) but column '{name}' has {n_bad} null(s) in {series.len()} "
        f"rows (first at row {first}, e.g. rows [{examples}]). Either fix the "
        f"input or declare `{name}: {py_type} | None`."
    )


def _null_rows(series: pl.Series) -> np.ndarray:
    """Row indices of the nulls of a column known to have some."""
    return np.flatnonzero(series.is_null().to_numpy())


def route_required_nulls(
    frame: pl.DataFrame,
    inputs: Sequence[Input],
    policy: MissingInputPolicy | None = None,
) -> NullRouting:
    """Doc 03 §1: a null in a `REQUIRED` input routes its row to a `Decision`
    instead of raising, unless the column is in `policy.raise_for`.

    `policy=None` uses `MissingInputPolicy()`'s own defaults
    (`default=Decision.REFER`, `reason=4101`, `raise_for=()`) — doc 03 §1 is
    explicit that refer-not-raise is the default even with no pipeline-level
    `on_missing_input(...)` call at all.

    A column entirely absent from `frame` — not merely null on some rows —
    is NOT a typo to leave for `graph/resolve.py` (doc 03 §2.2, O23): that
    catches a WIRING mistake between modules at build time, over a fixed
    set of declared names, never whether the actual `frame` handed to one
    particular `.apply()` call happens to carry every declared leaf column.
    Review finding 4: absent and null are the same situation from here on —
    a `REQUIRED` column that simply is not in `frame` routes every row
    exactly as if that column existed and were null in all of them (or, for
    a `raise_for` column, fails the whole batch by name, same as a
    `raise_for` column full of nulls does).

    The null counts come from ONE `frame.null_count()` (a single polars
    call, ~6 µs at 17 columns) rather than one `Series.null_count()` per
    required column (~1 µs each): the gate doc 05 §1.3 wants is still "a
    column with no nulls costs nothing beyond the count".
    """
    policy = policy or MissingInputPolicy()
    n = frame.height
    required = [decl for decl in inputs if decl.null_policy is NullPolicy.REQUIRED]
    if not required:
        return NullRouting(mask=np.zeros(n, dtype=bool), column=(None,) * n,
                           decision=policy.default, reason=policy.reason)
    present = set(frame.columns)
    null_counts = dict(zip(frame.columns, frame.null_count().row(0))) if n else {}

    # Structural columns first, and fully checked before any routing work is
    # done on the rest: a `raise_for` violation fails the whole batch, so
    # there is no point computing a routing mask the caller will never see.
    for decl in required:
        if decl.name not in policy.raise_for:
            continue
        if decl.name not in present:
            raise ValueError(_required_absent_message(decl.name))
        if not null_counts.get(decl.name, 0):
            continue
        series = frame.get_column(decl.name)
        raise ValueError(_required_null_message(decl.name, series, _null_rows(series)))

    mask = np.zeros(n, dtype=bool)
    column: list[str | None] = [None] * n
    for decl in required:
        if decl.name in policy.raise_for:
            continue  # already cleared above
        if decl.name not in present:
            # The whole column is absent: every row is missing it, exactly
            # as if every row's value were null (finding 4).
            for i in range(n):
                if not mask[i]:
                    mask[i] = True
                    column[i] = decl.name
            continue
        if not null_counts.get(decl.name, 0):
            continue
        for idx in _null_rows(frame.get_column(decl.name)):
            i = int(idx)
            if not mask[i]:
                mask[i] = True
                column[i] = decl.name

    return NullRouting(mask=mask, column=tuple(column), decision=policy.default, reason=policy.reason)
