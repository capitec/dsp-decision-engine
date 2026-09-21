"""Null handling at the boundary — doc 05 §2, and doc 03 §1's four situations.

**The slot under a null contains leftover garbage, not zero.** A measured left
join produced `9.0` at a position that was genuinely null (EXPERIMENTS.md §A;
doc 05 §2). Every function here exists so a step never reads an unchecked
value:

    * `NullPolicy.MISSING_AS` / `NOT_APPLICABLE_AS` (tiers 2 & 4) — filled
      *here*, at extraction, so the compiled kernel sees a plain number and
      there is nothing to forget. They differ only in which reason code the
      fill is tagged with (`FillReason`) — the step body never knows which
      situation produced the value it sees (doc 03 §1, "the fourth
      situation: not-applicable is not missing").
    * `NullPolicy.OPTIONAL` (tier 3) — *not* filled here. `extract.py` passes
      the raw values alongside the validity mask this module exposes, and the
      compiled driver builds a numba `Optional` per row (doc 05 §2).
    * `NullPolicy.REQUIRED` (tier 1) — **routed, not raised, by default.**

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

**`NullPolicy` dispatch: a registry keyed on the fixed enum.**
`decider2.types.NullPolicy` is a plain `Enum` in a fixed seam (`types.py`)
this package never edits — it can't become a pydantic discriminated union
itself. So the enum stays exactly the declaration value it always was, and
the *behaviour* for each of its four members lives in one `NullTierStrategy`
subclass apiece, looked up through the `NULL_TIER_STRATEGIES` registry below
instead of an `if null_policy is X` / `elif null_policy is Y` chain at every
call site that cares (this module had two such chains; `extract.py` had
another two). The tier-2/tier-4 split doc 03 §1 calls "the fourth situation"
is the one non-obvious case this buys: `_FillTier` is instantiated *twice*
(once per `FillReason`) rather than branching internally, so the two tiers
share every line of behaviour and differ only in the one constructor
argument each registry entry supplies — exactly the shape a registry keyed
on the enum is for, versus a hand-written `if`/`else` that a future edit
could silently collapse.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, NamedTuple, Sequence

import numpy as np
import polars as pl

from decider2.types import Decision, Input, MissingInputPolicy, NullPolicy

from .dtypes import ColumnPlan

__all__ = [
    "FillReason",
    "FillInfo",
    "NullRouting",
    "NullTierStrategy",
    "NULL_TIER_STRATEGIES",
    "fill_column",
    "route_required_nulls",
    "validity_mask",
]


def validity_mask(series: pl.Series) -> np.ndarray | None:
    """`True` where valid, `False` where null. `None` when the column has no
    validity buffer at all — the cheapest possible "definitely clean" signal
    (doc 05 §1.3) — so callers can skip masking work entirely rather than
    materialise an all-`True` array.

    A column whose dtype is polars' own `Null` (every value absent, and no
    other dtype was ever declared for it — a brand-new field with nothing
    captured yet is a real credit-data shape) has no buffers at all:
    `_get_buffers()` raises `TypeError` for it. Handled explicitly rather
    than left to crash the routing pass over one all-empty column.
    """
    if series.dtype == pl.Null:
        return np.zeros(series.len(), dtype=bool)
    validity = series._get_buffers()["validity"]
    if validity is None:
        return None
    # The validity buffer is itself a Boolean Series, and Boolean is never
    # zero-copy (arrow bitpacks it, doc 05 §1.4) — allow_copy=False always
    # raises here, so this doesn't even try it.
    return validity.to_numpy()


class FillReason(Enum):
    """Doc 03 §1's fourth situation: filling is one mechanism, but the reason
    code it is tagged with must distinguish "this did not apply" (a complete
    application) from "this was absent" (an incomplete one) — collapsing them
    silently approves the second. `observe/record.py` owns the numeric
    registry; this is the structural fact that module needs to consume.
    """

    MISSING = "missing_as"
    NOT_APPLICABLE = "not_applicable_as"


@dataclass(frozen=True)
class FillInfo:
    """What happened when a tier-2/4 column was filled."""

    reason: FillReason
    filled_count: int
    filled_mask: np.ndarray | None  # True where a fill was substituted; None if nothing was


class TierResult(NamedTuple):
    """What one `NullTierStrategy` produces for a column: the pieces
    `extract.py` assembles into an `ExtractedColumn` (`values` always;
    `validity`/`fill`/`categories` only when that tier populates them —
    `ExtractedColumn`'s own fields are already optional for exactly this
    reason)."""

    values: np.ndarray
    validity: np.ndarray | None = None
    fill: FillInfo | None = None
    categories: tuple[str, ...] | None = None


class NullTierStrategy:
    """One of doc 03 §1's four situations. `NullPolicy` (the fixed enum)
    only *names* a tier; a strategy instance, looked up from
    `NULL_TIER_STRATEGIES`, is what actually knows what that tier does to a
    column, both when the column is present (`extract_column`) and when a
    declared input is entirely absent from the frame
    (`synthesize_absent` — review finding 4's placeholder path).
    """

    routes_at_frame_level: ClassVar[bool] = False
    reason: ClassVar[FillReason | None] = None  # only a fill tier sets this

    def extract_column(self, series: pl.Series, decl: Input | None, plan: ColumnPlan) -> TierResult:
        raise NotImplementedError

    def synthesize_absent(self, decl: Input, n: int) -> TierResult:
        raise NotImplementedError


class _RequiredTier(NullTierStrategy):
    """Tier 1 — routed away before extraction ever runs (`route_required_
    nulls`), so by the time a column reaches `extract_column` it must
    already be clean; if it isn't (this function called directly, outside
    `extract_frame`'s routing pass), that is refused rather than silently
    fed to a kernel."""

    routes_at_frame_level = True

    def extract_column(self, series: pl.Series, decl: Input | None, plan: ColumnPlan) -> TierResult:
        if series.null_count() > 0:
            raise ValueError(
                f"extract_column('{series.name}') is REQUIRED and still has "
                f"{series.null_count()} null(s); route it with "
                "boundary.nulls.route_required_nulls first (doc 03 §1)"
            )
        values, categories = plan.extract(series)
        return TierResult(values, categories=categories)

    def synthesize_absent(self, decl: Input, n: int) -> TierResult:
        # Every row was already routed away by route_required_nulls (or a
        # raise_for violation already raised the whole batch), so `n` must
        # be 0 here — this only has to be a validly-shaped empty array.
        return TierResult(np.zeros(n, dtype=np.float64))


class _OptionalTier(NullTierStrategy):
    """Tier 3 — not filled: the raw values ride alongside a validity mask,
    and the compiled driver builds a numba `Optional` per row (doc 05 §2)."""

    def extract_column(self, series: pl.Series, decl: Input | None, plan: ColumnPlan) -> TierResult:
        values, categories = plan.extract(series)
        validity = validity_mask(series)
        if validity is None:
            validity = np.ones(series.len(), dtype=bool)
        return TierResult(values, validity, categories=categories)

    def synthesize_absent(self, decl: Input, n: int) -> TierResult:
        values = np.zeros(n, dtype=np.float64)
        validity = np.zeros(n, dtype=bool)
        return TierResult(values, validity)


class _FillTier(NullTierStrategy):
    """Tiers 2 & 4 (`MISSING_AS`/`NOT_APPLICABLE_AS`) — identical filling,
    a distinct `FillReason` per member (doc 03 §1's "fourth situation": the
    one thing that must never drift between the two)."""

    def __init__(self, reason: FillReason) -> None:
        self.reason = reason

    def extract_column(self, series: pl.Series, decl: Input | None, plan: ColumnPlan) -> TierResult:
        assert decl is not None  # only reachable via a decl whose null_policy selected this tier
        values, fill_info = fill_column(series, decl)
        return TierResult(values, fill=fill_info)

    def synthesize_absent(self, decl: Input, n: int) -> TierResult:
        values = np.full(n, decl.fill, dtype=np.float64)
        fill = FillInfo(reason=self.reason, filled_count=n, filled_mask=np.ones(n, dtype=bool))
        return TierResult(values, fill=fill)


NULL_TIER_STRATEGIES: dict[NullPolicy, NullTierStrategy] = {
    NullPolicy.REQUIRED: _RequiredTier(),
    NullPolicy.OPTIONAL: _OptionalTier(),
    NullPolicy.MISSING_AS: _FillTier(FillReason.MISSING),
    NullPolicy.NOT_APPLICABLE_AS: _FillTier(FillReason.NOT_APPLICABLE),
}


def fill_column(series: pl.Series, decl: Input) -> tuple[np.ndarray, FillInfo]:
    """Doc 03 §1 tiers 2 & 4: substitute at extraction so the kernel sees a
    plain number — the step body has nothing to check and nothing to forget.

    Tiers 2 (`MISSING_AS`) and 4 (`NOT_APPLICABLE_AS`) fill identically and
    differ only in `FillInfo.reason`. This never routes a row away (that is
    `route_required_nulls`, for `REQUIRED` only) and never raises: a declared
    fill means the author already decided this null is not exceptional.
    """
    strategy = NULL_TIER_STRATEGIES.get(decl.null_policy)
    if strategy is None or strategy.reason is None:
        raise ValueError(
            f"fill_column called on '{decl.name}' with null_policy="
            f"{decl.null_policy!r}; only MISSING_AS/NOT_APPLICABLE_AS are fillable here"
        )
    reason = strategy.reason

    raw = series._get_buffers()["values"]
    valid = validity_mask(series)

    if valid is None:
        # Nothing to fill — still return real values, not a shortcut that
        # skips the buffer read (a clean column with a MISSING_AS declaration
        # is ordinary: every value is already present).
        values = raw.to_numpy(allow_copy=False)
        return values, FillInfo(reason=reason, filled_count=0, filled_mask=None)

    garbage = raw.to_numpy()  # never allow_copy=False here: about to be read unconditionally
    filled_mask = ~valid
    values = np.where(valid, garbage, decl.fill)
    return values, FillInfo(reason=reason, filled_count=int(filled_mask.sum()), filled_mask=filled_mask)


@dataclass(frozen=True)
class NullRouting:
    """Doc 03 §1's routing decision, computed over a whole frame at once.

    `mask[i]` is `True` when row `i` must be routed to `decision` instead of
    reaching the kernel, because some `REQUIRED` input (not in
    `policy.raise_for`) is null there. `column[i]` names the first offending
    input, in declaration order — "first match" mirrors the four-kind
    algebra's own ordering rule (doc 00-BUILD.md Layer 4). `decision` and
    `reason` are uniform across every routed row: `MissingInputPolicy` is one
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
    """
    policy = policy or MissingInputPolicy()
    n = frame.height
    required = [decl for decl in inputs if NULL_TIER_STRATEGIES[decl.null_policy].routes_at_frame_level]

    # Structural columns first, and fully checked before any routing work is
    # done on the rest: a `raise_for` violation fails the whole batch, so
    # there is no point computing a routing mask the caller will never see.
    for decl in required:
        if decl.name not in policy.raise_for:
            continue
        if decl.name not in frame.columns:
            raise ValueError(_required_absent_message(decl.name))
        series = frame[decl.name]
        if series.null_count() == 0:
            continue
        valid = validity_mask(series)
        assert valid is not None  # null_count() > 0 implies a validity buffer exists
        bad_rows = np.flatnonzero(~valid)
        raise ValueError(_required_null_message(decl.name, series, bad_rows))

    mask = np.zeros(n, dtype=bool)
    column: list[str | None] = [None] * n
    for decl in required:
        if decl.name in policy.raise_for:
            continue  # already cleared above
        if decl.name not in frame.columns:
            # The whole column is absent: every row is missing it, exactly
            # as if every row's value were null (finding 4).
            for i in range(n):
                if not mask[i]:
                    mask[i] = True
                    column[i] = decl.name
            continue
        series = frame[decl.name]
        if series.null_count() == 0:
            continue
        valid = validity_mask(series)
        assert valid is not None
        for idx in np.flatnonzero(~valid):
            i = int(idx)
            if not mask[i]:
                mask[i] = True
                column[i] = decl.name

    return NullRouting(mask=mask, column=tuple(column), decision=policy.default, reason=policy.reason)
