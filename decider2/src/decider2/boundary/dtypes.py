"""The Arrow table: declared kind × polars dtype → what crosses, and how.

docs/BOUNDARY-REWORK.md §1.2. The dtype LADDER this module used to be —
`DtypeTier`, `EntryMode`, five `ColumnPlan` classes discriminated on how a
column was pulled out of polars, `probe_column` trying each conversion on
real data — is gone. Nullability is not a tier any more (the C gather reads
the validity bit, `_arrow/c/shim.c`), Boolean and the temporal types are
not copies any more (nanoarrow decodes the bitmap and the storage integer),
and nothing is probed because nothing is converted column by column any
more: the whole frame crosses once through `__arrow_c_stream__()`.

What is left is ONE decision per declared input, made from the frame's
schema alone and cached per (inputs, frame schema) by `extract.py`:

    kind    which typed-row buffer the column lands in (`types.FeatureKind`,
            from the input's annotation via `types.feature_kind` — the same
            table `compile.driver.numpy_dtype` reads, so `int | None` is F64
            here because it is float64 there, and the two never disagree);
    cast    the ONE explicit polars cast inserted in the frame tier before
            export when the frame's dtype is not one the kind reads
            natively — a Float64 column declared `int`, a String column
            declared `str` (→ Categorical, the dictionary-code convention
            hand-written `str` steps keep until Stage 7), a Decimal (→
            scaled int64 cents, doc 03 §1.2, as before), an all-null `Null`
            column (→ the kind's dtype, so its nulls fill or route like any
            other column's);
    reject  a column with no flat form — List, Array, Struct, Object,
            Binary — raises `NeedsKernelSplit` before anything is exported:
            the kernel-split escape one layer up, exactly as before (the
            ladder rejects nothing; it reports), never a per-row `objmode`
            escape (EXPERIMENTS.md §B: 77x).

Everything a kind reads natively — Float32/64, every Int/UInt width,
Boolean, Date/Datetime/Duration/Time (their storage integer), String
(Utf8View spans), Categorical/Enum (dictionary indices) — crosses with NO
polars work at all: nanoarrow decodes it and `sm_gather_row` widens it per
row. A pair this table has no cast for (a String column declared `float`,
a Categorical declared `int`) is not guessed at: the import refuses it by
name, Arrow type and kind (`_arrow.frame.ArrowKindError`).

`explain_boundary` keeps its job (doc 05 §1.5, "what the framework owes the
author"): one row per column with the Arrow type nanoarrow reports, the
kind, and whether a frame-tier cast was inserted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl

from decider2._arrow.frame import ArrowKindError
from decider2.types import FeatureKind, Input, feature_kind

__all__ = [
    "ArrowKindError",
    "ColumnPlan",
    "NeedsKernelSplit",
    "kind_for",
    "plan_column",
    "cast_series",
    "cast_frame",
    "explain_boundary",
]

F64, I64, BOOL, CODE, STR = (
    FeatureKind.F64, FeatureKind.I64, FeatureKind.BOOL, FeatureKind.CODE, FeatureKind.STR,
)


class NeedsKernelSplit(ArrowKindError):
    """Raised for a declared input whose dtype has no flat-array form here
    (`List`, `Struct`, `Array`, `Object`, `Binary` — and a Decimal whose
    scaled-int64 cast overflows on real data). **Not a rejection** — the
    table rejects nothing — this is the boundary handing the column to the
    escape mechanism one layer up: the kernel splits around the column
    (doc 05 §1.5), never a per-row `objmode` escape (EXPERIMENTS.md §B).
    An `ArrowKindError`, so one `except` covers "the kernel cannot read
    this column" whether the table or the import said so.
    """

    def __init__(self, name: str, dtype: pl.DataType, reason: str = ""):
        self.column = name
        self.dtype = dtype
        message = f"column '{name}' ({dtype}) has no flat-array extraction; needs the kernel-split escape"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


@dataclass(frozen=True)
class ColumnPlan:
    """One declared input's row of the table: where it lands (`kind`), the
    frame-tier cast inserted before export (`cast`, `None` when nanoarrow
    reads the frame's own dtype), and — from `explain_boundary` only, since
    it needs a bound view — the Arrow type nanoarrow reported, or the
    error the import raised."""

    name: str
    dtype: pl.DataType          # the frame's dtype, as handed in
    kind: FeatureKind
    cast: pl.DataType | None    # the polars dtype exported instead, or None
    note: str = ""
    arrow_type: str | None = None
    error: str | None = None


def kind_for(annotation) -> FeatureKind:
    """The kind a declared annotation lands in: `types.feature_kind`, whose
    table (`float`→F64, `int`→I64, `bool`→BOOL, `str`→CODE, `bytes`→STR,
    anything else — including an OPTIONAL input's `int | None` — →F64) is
    row for row the one `compile.driver.numpy_dtype` types the kernel by."""
    return feature_kind(annotation)


# What each kind reads with no cast: the polars dtypes whose Arrow storage
# `sm_resolve_col` accepts for it (shim.c). Temporal types are their
# storage integer (`tdD` int32, `tsu:` int64, ...) — an I64 feature.
_INTEGERS = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_TEMPORAL = (pl.Date, pl.Datetime, pl.Duration, pl.Time)
_NATIVE: dict[FeatureKind, tuple] = {
    F64: (pl.Float64, pl.Float32),
    I64: _INTEGERS + _TEMPORAL,
    BOOL: (pl.Boolean,),
    CODE: (pl.Categorical, pl.Enum),
    STR: (pl.String,),
}
# The dtype a kind is cast TO when the frame's dtype is castable but not native.
_CANONICAL: dict[FeatureKind, pl.DataType] = {
    F64: pl.Float64(), I64: pl.Int64(), BOOL: pl.Boolean(), CODE: pl.Categorical(), STR: pl.String(),
}
_NO_FLAT_FORM = (pl.Object, pl.Binary, pl.Unknown)


def plan_column(name: str, dtype: pl.DataType, kind: FeatureKind) -> ColumnPlan:
    """The table, one row. Decides from the dtype alone — no data is
    touched — whether `kind` reads `dtype` natively, needs one frame-tier
    cast, or (`NeedsKernelSplit`) has no flat form at all. A pair with no
    cast here is left for the import to refuse by name and Arrow type."""
    base = dtype.base_type()
    if dtype.is_nested() or base in _NO_FLAT_FORM:
        raise NeedsKernelSplit(name, dtype, reason="nested/object columns do not enter the kernel (doc 05 §1.5)")
    if base in _NATIVE[kind]:
        return ColumnPlan(name, dtype, kind, None, note="native: nanoarrow reads the frame's own buffers")
    if base is pl.Decimal:
        if kind in (F64, I64):
            return ColumnPlan(
                name, dtype, kind, _CANONICAL[kind],
                note="Decimal → scaled int64 cents in the frame tier (doc 03 §1.2), then the declared kind",
            )
    elif base is pl.Null:
        return ColumnPlan(
            name, dtype, kind, _CANONICAL[kind],
            note="all-null Null column cast to the kind's dtype so its nulls fill/route like any other",
        )
    elif kind in (F64, I64, BOOL) and (dtype.is_numeric() or base is pl.Boolean or base in _TEMPORAL):
        return ColumnPlan(
            name, dtype, kind, _CANONICAL[kind],
            note=f"frame-tier cast {dtype} → {_CANONICAL[kind]}: the declared kind is not the column's dtype",
        )
    elif kind is CODE and base is pl.String:
        return ColumnPlan(
            name, dtype, kind, _CANONICAL[kind],
            note="String → Categorical: a `str` input enters as a dictionary code until Stage 7 (doc 05 §1.5)",
        )
    elif kind is STR and base in (pl.Categorical, pl.Enum):
        return ColumnPlan(
            name, dtype, kind, _CANONICAL[kind],
            note="Categorical/Enum → String for a `bytes` input; Stage 6 reads the dictionary instead",
        )
    return ColumnPlan(
        name, dtype, kind, None,
        note=f"no frame-tier cast from {dtype} to {kind.name}; the import refuses it by name and Arrow type",
    )


def _decimal_as_cents(series: pl.Series, *, money_scale: int = 2) -> pl.Series:
    """Doc 03 §1.2: `Decimal` converts to a money-scaled int64 at the
    boundary — never crosses as `Decimal` itself. `to_physical()` is the
    unscaled Int128 mantissa at the column's own scale; the strict cast to
    Int64 is the one step that can fail on real data (an overflow), and the
    caller turns that into `NeedsKernelSplit`."""
    scale = series.dtype.scale if series.dtype.scale is not None else 0
    cents = series.to_physical().cast(pl.Int64, strict=True)
    shift = money_scale - scale
    if shift > 0:
        cents = cents * (10 ** shift)
    elif shift < 0:
        cents = cents // (10 ** (-shift))
    return cents


def cast_series(series: pl.Series, plan: ColumnPlan) -> pl.Series:
    """Apply `plan.cast` to one column — the one explicit, reported polars
    cast of the frame tier. Under `except BaseException`, as the old
    Decimal probe was (EXPERIMENTS.md §A: a Rust panic is a `BaseException`,
    not an `Exception`), so a failed cast reports the column by name."""
    if plan.cast is None:
        return series
    try:
        base = series.dtype.base_type()
        if base is pl.Decimal:
            out = _decimal_as_cents(series)
            return out if plan.cast == pl.Int64() else out.cast(plan.cast, strict=True)
        if base in _TEMPORAL:
            return series.to_physical().cast(plan.cast, strict=True)
        return series.cast(plan.cast, strict=True)
    except BaseException as exc:  # noqa: BLE001 — doc 05 §1.5's hard requirement
        raise NeedsKernelSplit(
            plan.name, series.dtype, reason=f"frame-tier cast to {plan.cast} failed: {exc!r}",
        ) from exc


def cast_frame(frame: pl.DataFrame, casts: Sequence[tuple[int, ColumnPlan]]) -> pl.DataFrame:
    """`frame` with each `(position, plan)` column replaced by its cast, as
    a NEW frame: the caller's columns are never re-typed under it. `clone`
    + `replace_column` + `Series.cast` is ~10 µs per column at n=1 against
    ~40 µs for `with_columns`, which goes through the lazy planner."""
    out = frame.clone()
    for pos, plan in casts:
        out.replace_column(pos, cast_series(frame.get_column(plan.name), plan))
    return out


def _kind_by_dtype(dtype: pl.DataType) -> FeatureKind:
    """`explain_boundary`'s kind for a column nobody declared: what its
    dtype would most naturally be read as."""
    base = dtype.base_type()
    if base in _NATIVE[STR]:
        return STR
    if base in _NATIVE[CODE]:
        return CODE
    if base in _NATIVE[BOOL]:
        return BOOL
    if base in _NATIVE[I64] or base is pl.Decimal:
        return I64
    return F64


def explain_boundary(frame: pl.DataFrame, inputs: Sequence[Input] | None = None) -> list[ColumnPlan]:
    """One row per column: the kind it lands in, the Arrow type nanoarrow
    reports for what is actually exported, and whether a frame-tier cast
    was inserted (doc 05 §1.5: "what the framework owes the author, since
    nothing is rejected"). Reading this answers "why is my batch doing a
    cast" from a table instead of a guess.

    `inputs` (the pipeline's `interface.inputs`) decides the kinds; without
    it every column is read as its dtype's natural kind. A column the
    table rejects, or the import refuses, is reported with `error` set
    rather than raised — this is diagnostics, it never fails."""
    from decider2._arrow.frame import FramePlan, FrameView

    if inputs is not None:
        wanted = [(decl.name, kind_for(decl.annotation)) for decl in inputs if decl.name in frame.columns]
    else:
        wanted = [(name, _kind_by_dtype(dtype)) for name, dtype in zip(frame.columns, frame.dtypes)]
    position = {name: k for k, name in enumerate(frame.columns)}
    plans: list[ColumnPlan] = []
    casts: list[tuple[int, ColumnPlan]] = []
    for name, kind in wanted:
        dtype = frame.dtypes[position[name]]
        try:
            plan = plan_column(name, dtype, kind)
        except NeedsKernelSplit as exc:
            plans.append(ColumnPlan(name, dtype, kind, None, note="", error=str(exc)))
            continue
        if plan.cast is not None:
            casts.append((position[name], plan))
        plans.append(plan)

    try:
        exported = cast_frame(frame, casts) if casts else frame
    except NeedsKernelSplit as exc:
        return [
            ColumnPlan(p.name, p.dtype, p.kind, p.cast, p.note, error=str(exc)) if p.name == exc.column else p
            for p in plans
        ]
    # Bind ONE column at a time so a refused column is reported on its own
    # row and the others still get their Arrow type.
    out: list[ColumnPlan] = []
    for p in plans:
        if p.error is not None:
            out.append(p)
            continue
        view = FrameView(FramePlan([(p.name, p.kind)], exported.columns))
        try:
            view.bind(exported)
            out.append(ColumnPlan(p.name, p.dtype, p.kind, p.cast, p.note, arrow_type=view.arrow_type(p.name)))
        except ArrowKindError as exc:
            out.append(ColumnPlan(p.name, p.dtype, p.kind, p.cast, p.note,
                                  arrow_type=getattr(exc, "arrow_type", None), error=str(exc)))
        finally:
            view.release()
    return out
