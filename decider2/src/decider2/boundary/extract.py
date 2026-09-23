"""Getting data out of polars, per column, without `to_arrow` — doc 05 §1.

`series.to_arrow()` requires pyarrow, which is not a project dependency, and
raised `ModuleNotFoundError` for all 26 dtype/nullability combinations tested
(EXPERIMENTS.md §A). `Series._get_buffers()` is the replacement: no new
dependency, the same cost (~1.9us/col at 100k rows), pre-sliced validity and
pre-split offsets.

A column with no nulls has no validity bitmap allocated at all — that is the
cheapest possible gate (§1.3, ~0.375us total, fully zero-copy). Everything
else copies; `boundary.dtypes` says how, and `boundary.nulls` says what a
null in each column means.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl

from decider2.types import Input, MissingInputPolicy, NullPolicy

from .dtypes import ColumnPlan, DtypeTier, NeedsKernelSplit, SpanPlan, ZeroCopyPlan, plan_column
from .nulls import NULL_TIER_STRATEGIES, FillInfo, NullRouting, route_required_nulls

__all__ = [
    "ExtractedColumn",
    "ExtractedFrame",
    "NeedsKernelSplit",
    "rechunk_once",
    "is_clean",
    "extract_column",
    "extract_frame",
]


@dataclass(frozen=True)
class ExtractedColumn:
    """One column, ready for the calling convention (doc 05 §3).

    `validity` is populated only for `NullPolicy.OPTIONAL` columns — it is
    the mask the compiled driver uses to build a numba `Optional` per row
    (doc 05 §2). For every other null policy the values array is already
    unconditionally safe to read: `REQUIRED` nulls were routed away before
    this ran (see `extract_frame`), and `MISSING_AS`/`NOT_APPLICABLE_AS`
    nulls were filled.

    `categories` is populated only for `EntryMode.CODES` columns (doc 05
    §1.5's Utf8/Categorical/Enum row) — the dictionary a code is an index
    into, in code order (`.cat.get_categories().to_list()`). This is what
    lets a `str`-typed param be resolved to the matching code at
    param-resolution time (`runtime.invoke.resolve_params`) instead of the
    kernel ever seeing text.
    """

    name: str
    values: np.ndarray
    validity: np.ndarray | None
    plan: ColumnPlan
    fill: FillInfo | None = None
    categories: tuple[str, ...] | None = None


def rechunk_once(frame: pl.DataFrame) -> pl.DataFrame:
    """Doc 05 §1.1 — once, at frame entry, never per column.

    0.33us if already single-chunk; a genuine two-chunk rechunk costs
    585us/1M rows, and doing it per column multiplies that by the column
    count.
    """
    return frame.rechunk()


def is_clean(series: pl.Series) -> bool:
    """The cheapest possible gate (doc 05 §1.3): a column with no nulls has
    no validity bitmap at all. `null_count()` is the spec's named
    alternative to checking `_get_buffers()["validity"] is None` directly;
    it needs no buffer materialisation.
    """
    return series.null_count() == 0


def extract_column(series: pl.Series, decl: Input | None = None) -> ExtractedColumn:
    """Extract one already-routed column into calling-convention-ready
    arrays.

    `decl` carries the null policy (doc 03 §1); omitting it treats the
    column as `REQUIRED` with no remaining nulls — the state `extract_frame`
    guarantees by the time this runs on a `REQUIRED` column. Passing `decl`
    explicitly is what lets this function be tested column-by-column,
    independent of a whole-frame routing pass.

    Raises `NeedsKernelSplit` for a dtype with no flat-array form (doc 05
    §1.5's `KERNEL_SPLIT` entry mode) — never silently drops or crashes the
    column, since the ladder rejects nothing. `plan.guard()` is what raises
    it here, uniformly, before any null-tier-specific handling runs (a
    `KernelSplitPlan` column must fail the same way whether its `NullPolicy`
    would otherwise have filled, masked, or passed it straight through).

    The rest of this function used to be an `if`/`elif` over `null_policy`
    (OPTIONAL builds a mask; MISSING_AS/NOT_APPLICABLE_AS fill; REQUIRED
    passes through) — now a single dispatch into `NULL_TIER_STRATEGIES`,
    the same registry `boundary.nulls` uses for its own two null-tier
    branches (doc 03 §1's four situations, one place each).
    """
    nullable = not is_clean(series)
    plan = plan_column(series.name, series.dtype, nullable=nullable)
    null_policy = decl.null_policy if decl is not None else NullPolicy.REQUIRED
    if decl is not None and decl.annotation is bytes:
        # Declared as a STRING SPAN (a tree's string feature, docs/BOUNDARY-
        # REWORK.md §2.1): the column crosses as (address, length) pairs
        # through the Arrow shim, not as dictionary codes. Decided by the
        # declaration, since a `str`-declared input still takes `CodesPlan`.
        plan = SpanPlan(name=series.name, dtype=series.dtype, tier=plan.tier, nullable=nullable,
                        note="string spans through the Arrow shim (BOUNDARY-REWORK.md §2.1)")

    plan.guard()

    strategy = NULL_TIER_STRATEGIES[null_policy]
    values, validity, fill, categories = strategy.extract_column(series, decl, plan)
    return ExtractedColumn(series.name, values, validity, plan, fill=fill, categories=categories)


def _synthesize_absent_column(decl: Input, n: int) -> ExtractedColumn:
    """Review finding 4: a declared input entirely absent from the frame —
    not merely null on some rows — shares the same fill/route path a null
    does; there is no polars `Series` to route/fill/mask against, so this
    builds the placeholder `extract_column` would have produced had the
    column existed and been null on every row.

    A REQUIRED `decl` reaching here has already had every one of `n` rows
    routed away by `route_required_nulls` (or the whole batch already
    raised, for a `raise_for` column) — `n` is 0 in that case, and this only
    has to produce a correctly-shaped EMPTY array so the compiled kernel's
    calling convention still has something to bind its `array` argument
    role against (without even an empty array in the registry, `fused` mode
    raises a bare `KeyError` building the kernel call, three frames deep in
    `modes._build_call_args` — the exact failure this finding names).
    MISSING_AS/NOT_APPLICABLE_AS/OPTIONAL never route rows away, so `n` here
    is the full `kernel_frame` height, filled/masked exactly as a genuinely
    all-null column of that tier would be.

    The synthesized dtype is a placeholder (`float64`), not `decl`'s
    declared annotation: `runtime.invoke.apply` re-casts every column via
    `numpy_dtype(annotation)` right after extraction regardless of what
    dtype arrives here (see review finding 2's fix), so this only has to be
    a valid, correctly-shaped numeric array, never the final typed one.

    Dispatches through the same `NULL_TIER_STRATEGIES` registry
    `extract_column` uses (`synthesize_absent` rather than
    `extract_column`, since there is no polars `Series` here to route/fill/
    mask against) — previously its own second copy of the tier-2/4 reason
    ternary `boundary.nulls.fill_column` already had one of.
    """
    if decl.annotation is bytes:
        # A string-span input (a tree's string feature) has a fixed shape,
        # `(n, 2)` int64 with length -1 for a null, that no float64
        # placeholder re-cast can produce — so the placeholder is built in
        # that shape here (REQUIRED: every row already routed, n == 0).
        plan = SpanPlan(
            name=decl.name, dtype=pl.Null(), tier=DtypeTier.ZERO_COPY,
            nullable=decl.null_policy is NullPolicy.OPTIONAL,
            note="column absent from the input frame — synthesized null spans (doc 03 §1)",
        )
        values = np.tile(np.array([0, -1], dtype=np.int64), (n, 1))
        return ExtractedColumn(decl.name, values, None, plan)
    plan = ZeroCopyPlan(
        name=decl.name, dtype=pl.Null(), tier=DtypeTier.ZERO_COPY,
        nullable=decl.null_policy is NullPolicy.OPTIONAL,
        note="column absent from the input frame — synthesized placeholder (doc 03 §1)",
    )
    strategy = NULL_TIER_STRATEGIES[decl.null_policy]
    values, validity, fill, categories = strategy.synthesize_absent(decl, n)
    return ExtractedColumn(decl.name, values, validity, plan, fill=fill, categories=categories)


@dataclass(frozen=True)
class ExtractedFrame:
    """The whole-frame result of `extract_frame`: every input column, ready
    for the calling convention, plus the routing decision for rows a
    `REQUIRED` null pulled out before the kernel ever saw them.

    `categories` is the frame-level counterpart of `ExtractedColumn.
    categories` — every `CODES`-entry-mode column's name mapped to its
    category tuple, gathered here so a caller (`runtime.invoke.resolve_
    params`) can resolve a `str`-typed param's literal without walking
    `columns` itself.
    """

    columns: dict[str, ExtractedColumn]
    routing: NullRouting
    kernel_frame: pl.DataFrame  # the row subset actually handed to the kernel
    categories: dict[str, tuple[str, ...]] = field(default_factory=dict)


def extract_frame(
    frame: pl.DataFrame,
    inputs: Sequence[Input],
    *,
    policy: MissingInputPolicy | None = None,
) -> ExtractedFrame:
    """The whole-column, whole-frame extraction pass (doc 05 §1 + §2 tied
    together):

    1. rechunk once (§1.1);
    2. route `REQUIRED` nulls over the *whole* frame — a `raise_for` column
       fails the whole batch here, before any array work happens (doc 03 §1);
    3. filter to the rows that pass, and extract every declared input from
       that subset. A column with no flat-array form (`NeedsKernelSplit`)
       still fails per-column here — `extract_frame` does not implement the
       kernel-split escape itself (that is `compile/numba/fallback.py`'s
       job); it surfaces the same exception `extract_column` raises, with
       every other column already having done its (wasted, but correct)
       work first is an accepted cost of "report, don't half-extract".

    The rows `routing.mask` selects are **not** dropped — they are excluded
    from `kernel_frame` and left for the caller to route to
    `routing.decision` (doc 03 §1: "the violation lands in the decision
    record with the column, the reason code and the count").
    """
    frame = rechunk_once(frame)
    routing = route_required_nulls(frame, inputs, policy)

    if routing.routed_count:
        kernel_frame = frame.filter(pl.Series(~routing.mask))
        kernel_frame = rechunk_once(kernel_frame)  # a boolean filter is cheap to re-check, not assumed clean
    else:
        kernel_frame = frame

    columns: dict[str, ExtractedColumn] = {}
    for decl in inputs:
        if decl.name not in kernel_frame.columns:
            # Absent, not merely null (finding 4) — synthesize the same
            # placeholder `route_required_nulls` already priced this
            # column's absence in for (REQUIRED: every remaining row was
            # already routed away, so `kernel_frame.height` is 0 here;
            # MISSING_AS/NOT_APPLICABLE_AS/OPTIONAL: filled/masked over the
            # full `kernel_frame` height, exactly as an all-null column
            # would be).
            columns[decl.name] = _synthesize_absent_column(decl, kernel_frame.height)
            continue
        columns[decl.name] = extract_column(kernel_frame[decl.name], decl)

    categories = {
        name: ec.categories for name, ec in columns.items() if ec.categories is not None
    }
    return ExtractedFrame(columns=columns, routing=routing, kernel_frame=kernel_frame, categories=categories)
