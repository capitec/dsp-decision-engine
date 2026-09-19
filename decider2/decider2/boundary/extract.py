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

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from decider2.types import Input, MissingInputPolicy, NullPolicy

from .dtypes import ColumnPlan, EntryMode, plan_column
from .nulls import FillInfo, NullRouting, fill_column, route_required_nulls, validity_mask

__all__ = [
    "ExtractedColumn",
    "ExtractedFrame",
    "NeedsKernelSplit",
    "rechunk_once",
    "is_clean",
    "extract_column",
    "extract_frame",
]


class NeedsKernelSplit(Exception):
    """Raised by `extract_column` when a dtype has no flat-array
    representation here (doc 05 §1.5: `List`, `Struct`, and anything else not
    yet on the ladder). **Not a rejection** — the ladder rejects nothing —
    this is the boundary handing the column to the escape mechanism one layer
    up: `compile/numba/fallback.py` splits the *kernel* around the column,
    never a per-row `objmode` escape (EXPERIMENTS.md §B measured that at 77x
    a pure kernel, worse than falling back to plain Python for the whole
    driver).
    """

    def __init__(self, name: str, dtype: pl.DataType, reason: str = ""):
        self.column = name
        self.dtype = dtype
        message = f"column '{name}' ({dtype}) has no flat-array extraction; needs the kernel-split escape"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


@dataclass(frozen=True)
class ExtractedColumn:
    """One column, ready for the calling convention (doc 05 §3).

    `validity` is populated only for `NullPolicy.OPTIONAL` columns — it is
    the mask the compiled driver uses to build a numba `Optional` per row
    (doc 05 §2). For every other null policy the values array is already
    unconditionally safe to read: `REQUIRED` nulls were routed away before
    this ran (see `extract_frame`), and `MISSING_AS`/`NOT_APPLICABLE_AS`
    nulls were filled.
    """

    name: str
    values: np.ndarray
    validity: np.ndarray | None
    plan: ColumnPlan
    fill: FillInfo | None = None


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


def _raw_values(series: pl.Series) -> pl.Series:
    """The values buffer, always — garbage under any null slot (doc 05 §2),
    never returned to a caller that hasn't already handled validity.
    """
    return series._get_buffers()["values"]


def _extract_numeric_like(series: pl.Series) -> tuple[np.ndarray, bool]:
    """Zero-copy when polars allows it, an ordinary copy otherwise — tried
    directly rather than assumed from the dtype/nullability table, because
    which combinations are actually zero-copy is a fact about the installed
    polars/numpy build (EXPERIMENTS.md §A pins it to this environment).
    Covers `EntryMode.NATIVE`, `COPY_VALIDITY`, `AS_INTEGER` and
    `BUFFER_COPY` — all four are "get the values buffer as a numpy array";
    they differ only in whether that succeeds without a copy.
    """
    values = _raw_values(series)
    try:
        return values.to_numpy(allow_copy=False), True
    except RuntimeError:
        return values.to_numpy(), False


def _extract_codes(series: pl.Series) -> np.ndarray:
    """Categorical/Enum/Utf8 all enter as integer codes (doc 05 §1.5) — a
    string never enters a kernel as a string. Categorical/Enum already store
    codes; Utf8 is dictionary-encoded first (`cast(pl.Categorical)`), then
    the same code extraction applies.
    """
    base = series.dtype.base_type()
    if base in (pl.Categorical, pl.Enum):
        return _raw_values(series).to_numpy(allow_copy=False)
    encoded = series.cast(pl.Categorical)
    return _raw_values(encoded).to_numpy(allow_copy=False)


def extract_decimal_as_cents(series: pl.Series, *, money_scale: int = 2) -> np.ndarray:
    """Doc 03 §1.2: `Decimal` converts to a money-scaled int64 at the
    boundary — never crosses as `Decimal` itself.

    Must run under `except BaseException`: `_get_buffers()` on a `Decimal`
    column succeeds and hands back a usable `Int128` series (doc 00-BUILD.md
    §O11) — the panic fires one call deeper, on `.to_numpy()`/an unchecked
    `.cast()` of *that* buffer, as `pyo3_runtime.PanicException`, which does
    not inherit `Exception` (EXPERIMENTS.md §A).
    """
    dtype = series.dtype
    if not isinstance(dtype, pl.Decimal):
        raise TypeError(f"extract_decimal_as_cents expects a Decimal column, got {dtype}")

    raw = _raw_values(series)  # Int128 series — the unscaled mantissa at `dtype.scale`
    scale = dtype.scale if dtype.scale is not None else 0
    try:
        cents = raw.cast(pl.Int64, strict=True)
        shift = money_scale - scale
        if shift > 0:
            cents = cents * (10**shift)
        elif shift < 0:
            cents = cents // (10 ** (-shift))
        return cents.to_numpy(allow_copy=False)
    except BaseException as exc:  # noqa: BLE001 — doc 05 §1.5's hard requirement
        raise NeedsKernelSplit(
            series.name, dtype,
            reason=f"Decimal->int64 cents conversion failed: {exc!r}",
        ) from exc


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
    column, since the ladder rejects nothing.
    """
    nullable = not is_clean(series)
    plan = plan_column(series.name, series.dtype, nullable=nullable)
    null_policy = decl.null_policy if decl is not None else NullPolicy.REQUIRED

    if plan.entry_mode is EntryMode.KERNEL_SPLIT:
        raise NeedsKernelSplit(series.name, series.dtype, reason=plan.note)

    if null_policy is NullPolicy.OPTIONAL:
        values = _extract_by_plan(series, plan)
        validity = validity_mask(series)
        if validity is None:
            validity = np.ones(series.len(), dtype=bool)
        return ExtractedColumn(series.name, values, validity, plan)

    if null_policy in (NullPolicy.MISSING_AS, NullPolicy.NOT_APPLICABLE_AS):
        assert decl is not None
        values, fill_info = fill_column(series, decl)
        return ExtractedColumn(series.name, values, None, plan, fill=fill_info)

    # REQUIRED: by the time a column reaches here, `extract_frame` has
    # already removed any row a null in it would have routed away (doc 03
    # §1 — routing happens at extraction, so the kernel never sees an
    # unresolved null). Called directly on a column that still has nulls,
    # this still must not silently feed garbage to a kernel:
    if series.null_count() > 0:
        raise ValueError(
            f"extract_column('{series.name}') is REQUIRED and still has "
            f"{series.null_count()} null(s); route it with "
            "boundary.nulls.route_required_nulls first (doc 03 §1)"
        )
    values = _extract_by_plan(series, plan)
    return ExtractedColumn(series.name, values, None, plan)


def _extract_by_plan(series: pl.Series, plan: ColumnPlan) -> np.ndarray:
    mode = plan.entry_mode
    if mode in (EntryMode.NATIVE, EntryMode.COPY_VALIDITY, EntryMode.AS_INTEGER, EntryMode.BUFFER_COPY):
        values, _zero_copy = _extract_numeric_like(series)
        return values
    if mode is EntryMode.CODES:
        return _extract_codes(series)
    if mode is EntryMode.SCALED_INT64:
        return extract_decimal_as_cents(series)
    raise NeedsKernelSplit(series.name, series.dtype, reason=plan.note)  # pragma: no cover — guarded above


@dataclass(frozen=True)
class ExtractedFrame:
    """The whole-frame result of `extract_frame`: every input column, ready
    for the calling convention, plus the routing decision for rows a
    `REQUIRED` null pulled out before the kernel ever saw them.
    """

    columns: dict[str, ExtractedColumn]
    routing: NullRouting
    kernel_frame: pl.DataFrame  # the row subset actually handed to the kernel


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
            continue  # unbound input: graph/resolve.py's job (doc 03 §2.2), not this function's
        columns[decl.name] = extract_column(kernel_frame[decl.name], decl)

    return ExtractedFrame(columns=columns, routing=routing, kernel_frame=kernel_frame)
