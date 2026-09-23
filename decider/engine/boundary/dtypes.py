"""Which polars dtypes each declared kind reads natively, which need one cast first, and which can't cross."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import polars as pl

from decider.engine.boundary._arrow.plan import ArrowKindError, FramePlan
from decider.engine.ir.decls import FeatureKind, Input, feature_kind

F64, I64, BOOL, CODE, STR = (
    FeatureKind.F64, FeatureKind.I64, FeatureKind.BOOL, FeatureKind.CODE, FeatureKind.STR,
)


class NeedsKernelSplit(ArrowKindError):
    """A declared column has no flat-array form (List, Struct, Array, Object, Binary, an overflowing Decimal).

    The caller splits the kernel around the column; `column` and `dtype` name it.
    """

    def __init__(self, name: str, dtype: pl.DataType, reason: str = ""):
        self.column = name
        self.dtype = dtype
        message = f"column '{name}' ({dtype}) has no flat-array extraction; needs the kernel-split escape"
        super().__init__(f"{message} ({reason})" if reason else message)


@dataclass(frozen=True)
class ColumnPlan:
    """How one declared input crosses the boundary.

    Args:
        dtype: the frame's own dtype.
        cast: the polars dtype exported instead, or `None` when the kind reads `dtype` as is.
        arrow_type: set by `explain_boundary` only: the Arrow type nanoarrow reported.
        error: set by `explain_boundary` only: why the column can't cross.
    """

    name: str
    dtype: pl.DataType
    kind: FeatureKind
    cast: pl.DataType | None
    note: str = ""
    arrow_type: str | None = None
    error: str | None = None


# Temporal types cross as their storage integer.
_INTEGERS = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_TEMPORAL = (pl.Date, pl.Datetime, pl.Duration, pl.Time)
_NATIVE: dict[FeatureKind, tuple] = {
    F64: (pl.Float64, pl.Float32),
    I64: _INTEGERS + _TEMPORAL,
    BOOL: (pl.Boolean,),
    CODE: (pl.Categorical, pl.Enum),
    STR: (pl.String,),
}
_CANONICAL: dict[FeatureKind, pl.DataType] = {
    F64: pl.Float64(), I64: pl.Int64(), BOOL: pl.Boolean(), CODE: pl.Categorical(), STR: pl.String(),
}
_NO_FLAT_FORM = (pl.Object, pl.Binary, pl.Unknown)


def plan_column(name: str, dtype: pl.DataType, kind: FeatureKind) -> ColumnPlan:
    """Decide from the dtype alone whether `kind` reads it natively, needs one cast, or can't read it.

    A pair with no cast (a String declared `float`) is left for the import to
    refuse by Arrow type. Nested and object dtypes raise `NeedsKernelSplit`.

    >>> plan_column("amount", pl.Float64(), FeatureKind.I64).cast
    Int64
    """
    base = dtype.base_type()
    if dtype.is_nested() or base in _NO_FLAT_FORM:
        raise NeedsKernelSplit(name, dtype, reason="nested/object columns do not enter the kernel")
    if base in _NATIVE[kind]:
        return ColumnPlan(name, dtype, kind, None, note="native: nanoarrow reads the frame's own buffers")
    cast = _CANONICAL[kind]
    if base is pl.Decimal:
        if kind in (F64, I64):
            return ColumnPlan(name, dtype, kind, cast, note="Decimal -> scaled int64 cents, then the declared kind")
    elif base is pl.Null:
        return ColumnPlan(name, dtype, kind, cast, note="all-null column cast to the kind's dtype")
    elif kind in (F64, I64, BOOL) and (dtype.is_numeric() or base is pl.Boolean or base in _TEMPORAL):
        return ColumnPlan(name, dtype, kind, cast, note=f"frame-tier cast {dtype} -> {cast}")
    elif (kind is CODE and base is pl.String) or (kind is STR and base in _NATIVE[CODE]):
        return ColumnPlan(name, dtype, kind, cast, note=f"frame-tier cast {dtype} -> {cast}")
    return ColumnPlan(name, dtype, kind, None,
                      note=f"no frame-tier cast from {dtype} to {kind.name}; the import refuses it")


def _decimal_as_cents(series: pl.Series, *, money_scale: int = 2) -> pl.Series:
    # Money never enters a kernel as Decimal: the unscaled mantissa is rescaled to cents.
    scale = series.dtype.scale or 0
    cents = series.to_physical().cast(pl.Int64, strict=True)
    shift = money_scale - scale
    if shift > 0:
        return cents * (10 ** shift)
    if shift < 0:
        return cents // (10 ** -shift)
    return cents


def cast_series(series: pl.Series, plan: ColumnPlan) -> pl.Series:
    """Apply `plan.cast` to one column; a failed cast raises `NeedsKernelSplit` naming it."""
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
    # A Rust panic surfaces as pyo3's PanicException, which is not an Exception.
    except BaseException as exc:  # noqa: BLE001
        raise NeedsKernelSplit(
            plan.name, series.dtype, reason=f"frame-tier cast to {plan.cast} failed: {exc!r}",
        ) from exc


def cast_frame(frame: pl.DataFrame, casts: Sequence[tuple[int, ColumnPlan]]) -> pl.DataFrame:
    """A new frame with each `(position, plan)` column cast; the caller's frame is never re-typed."""
    # clone + replace_column is ~10 us a column against ~40 us for with_columns' lazy planner.
    out = frame.clone()
    for pos, plan in casts:
        out.replace_column(pos, cast_series(frame.get_column(plan.name), plan))
    return out


def _natural_kind(dtype: pl.DataType) -> FeatureKind:
    base = dtype.base_type()
    for kind in (STR, CODE, BOOL, I64):
        if base in _NATIVE[kind]:
            return kind
    return I64 if base is pl.Decimal else F64


def explain_boundary(frame: pl.DataFrame, inputs: Sequence[Input] | None = None) -> list[ColumnPlan]:
    """One `ColumnPlan` per column: the kind it lands in, the cast inserted, and the Arrow type exported.

    Diagnostics: a column that can't cross is reported with `error` set, never
    raised. Without `inputs`, every column is read as its dtype's natural kind.

    >>> [p.arrow_type for p in explain_boundary(pl.DataFrame({"x": [1.0]}))]  # doctest: +SKIP
    ['double']
    """
    from decider.engine.boundary._arrow.view import FrameView

    if inputs is not None:
        wanted = [(d.name, feature_kind(d.annotation)) for d in inputs if d.name in frame.columns]
    else:
        wanted = [(name, _natural_kind(dtype)) for name, dtype in frame.schema.items()]
    position = {name: k for k, name in enumerate(frame.columns)}
    plans: list[ColumnPlan] = []
    casts: list[tuple[int, ColumnPlan]] = []
    for name, kind in wanted:
        dtype = frame.dtypes[position[name]]
        try:
            plan = plan_column(name, dtype, kind)
        except NeedsKernelSplit as exc:
            plans.append(ColumnPlan(name, dtype, kind, None, error=str(exc)))
            continue
        if plan.cast is not None:
            casts.append((position[name], plan))
        plans.append(plan)
    try:
        exported = cast_frame(frame, casts)
    except NeedsKernelSplit as exc:
        return [replace(p, error=str(exc)) if p.name == exc.column else p for p in plans]
    out: list[ColumnPlan] = []
    for p in plans:
        if p.error is not None:
            out.append(p)
            continue
        # One column per view, so a refused column doesn't hide the others' Arrow types.
        view = FrameView(FramePlan([(p.name, p.kind)], exported.columns))
        try:
            view.bind(exported)
            out.append(replace(p, arrow_type=view.arrow_type(p.name)))
        except ArrowKindError as exc:
            out.append(replace(p, arrow_type=getattr(exc, "arrow_type", None), error=str(exc)))
        finally:
            view.release()
    return out

