"""Which polars dtypes each declared kind reads natively, which need one cast first, and which can't cross."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import polars as pl

from decider._engine.boundary._arrow.plan import BOOL, CODE, F64, I64, STR
from decider._engine.ir.decls import FeatureKind, Input, NullPolicy, feature_kind
from decider._engine.boundary.errors import ArrowKindError, NeedsKernelSplit


@dataclass(frozen=True)
class ColumnPlan:
    """How one declared input crosses the boundary.

    Args:
        dtype: the frame's own dtype.
        cast: the polars dtype exported instead, or `None` when the kind reads `dtype` as is.
        scale: for a Decimal input, its scale: the column reads as the int64 mantissa at that scale.
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
    scale: int | None = None


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


def plan_column(name: str, dtype: pl.DataType, kind: FeatureKind, scale: int | None = None) -> ColumnPlan:
    """Decide from the dtype alone whether `kind` reads it natively, needs one cast, or can't read it.

    A pair with no cast (a String declared `float`) is left for the import to
    refuse by Arrow type. Nested and object dtypes raise `NeedsKernelSplit`.
    A Decimal input (`scale` set) reads a polars Decimal or integer column as
    its int64 mantissa at that scale.

    >>> plan_column("amount", pl.Float64(), FeatureKind.I64).cast
    Int64
    """
    base = dtype.base_type()
    if dtype.is_nested() or base in _NO_FLAT_FORM:
        raise NeedsKernelSplit(name, dtype, reason="nested/object columns do not enter the kernel")
    if scale is not None and (base is pl.Decimal or base in _INTEGERS):
        return ColumnPlan(name, dtype, kind, pl.Int64(), note=f"Decimal: {dtype} -> int64 mantissa at scale {scale}", scale=scale)
    if scale is not None and base is not pl.Null:
        return ColumnPlan(name, dtype, kind, None, note=f"{dtype} is not a Decimal column: cast it to pl.Decimal in the frame tier; the import refuses it")
    if base in _NATIVE[kind]:
        return ColumnPlan(name, dtype, kind, None, note="native: nanoarrow reads the frame's own buffers")
    cast = _CANONICAL[kind]
    if base is pl.Null:
        return ColumnPlan(name, dtype, kind, cast, note="all-null column cast to the kind's dtype")
    elif kind in (F64, I64, BOOL) and (dtype.is_numeric() or base is pl.Boolean or base in _TEMPORAL):
        return ColumnPlan(name, dtype, kind, cast, note=f"frame-tier cast {dtype} -> {cast}")
    elif (kind is CODE and base is pl.String) or (kind is STR and base in _NATIVE[CODE]):
        return ColumnPlan(name, dtype, kind, cast, note=f"frame-tier cast {dtype} -> {cast}")
    return ColumnPlan(name, dtype, kind, None,
                      note=f"no frame-tier cast from {dtype} to {kind.name}; the import refuses it")


def decimal_mantissa(series: pl.Series, scale: int) -> pl.Series:
    """A polars Decimal or integer Series as the int64 mantissa at `scale`, rescaled half-even; overflow raises."""
    # Polars rescales a Decimal half-even and an integer to whole units; either overflow is an error.
    if not (isinstance(series.dtype, pl.Decimal) and series.dtype.scale == scale):
        series = series.cast(pl.Decimal(38, scale), strict=True)
    return series.to_physical().cast(pl.Int64, strict=True)


def cast_series(series: pl.Series, plan: ColumnPlan) -> pl.Series:
    """Apply `plan.cast` to one column; a failed cast raises `NeedsKernelSplit` naming it."""
    if plan.cast is None:
        return series
    try:
        base = series.dtype.base_type()
        if plan.scale is not None:
            return decimal_mantissa(series, plan.scale)
        if base in _TEMPORAL:
            return series.to_physical().cast(plan.cast, strict=True)
        return series.cast(plan.cast, strict=True)
    # A Rust panic surfaces as pyo3's PanicException, which is not an Exception.
    except BaseException as exc:  # noqa: BLE001
        why = (f"a value does not fit a Decimal at scale {plan.scale} in int64 (at most "
               f"{(2 ** 63 - 1) / 10 ** plan.scale}): {exc!r}" if plan.scale is not None
               else f"frame-tier cast to {plan.cast} failed: {exc!r}")
        raise NeedsKernelSplit(plan.name, series.dtype, reason=why) from exc


def cast_frame(frame: pl.DataFrame, casts: Sequence[tuple[int, ColumnPlan]]) -> pl.DataFrame:
    """A new frame with each `(position, plan)` column cast; the caller's frame is never re-typed."""
    # clone + replace_column is ~10 us a column against ~40 us for with_columns' lazy planner.
    out = frame.clone()
    for pos, plan in casts:
        out.replace_column(pos, cast_series(frame.get_column(plan.name), plan))
    return out


_NATURAL = {STR: bytes, CODE: str, BOOL: bool, I64: int}


def _natural(dtype: pl.DataType) -> type:
    base = dtype.base_type()
    for kind in (STR, CODE, BOOL, I64):
        if base in _NATIVE[kind]:
            return _NATURAL[kind]
    return float


def explain_boundary(frame: pl.DataFrame, inputs: Sequence[Input] | None = None) -> list[ColumnPlan]:
    """One `ColumnPlan` per column: the kind it lands in, the cast inserted, and the Arrow type exported.

    Diagnostics: a column that can't cross is reported with `error` set, never
    raised. Without `inputs`, every column is read as its dtype's natural kind.

    >>> [p.arrow_type for p in explain_boundary(pl.DataFrame({"x": [1.0]}))]  # doctest: +SKIP
    ['double']
    """
    from decider._engine.boundary._arrow.view import FrameView
    from decider._engine.boundary.extract import _schema_plan

    schema = frame.schema
    if inputs is None:
        inputs = [Input(name, _natural(dtype)) for name, dtype in schema.items()]
    columns, dtypes = tuple(schema), tuple(schema.values())
    out: list[ColumnPlan] = []
    # One column at a time, so a refused column doesn't hide the others' Arrow types.
    for decl in inputs:
        if decl.name not in schema:
            continue
        try:
            sp = _schema_plan((Input(decl.name, decl.annotation, NullPolicy.OPTIONAL),), columns, dtypes)
            plan = sp.plans[decl.name]
            exported = cast_frame(frame, sp.casts)
        except NeedsKernelSplit as exc:
            out.append(ColumnPlan(decl.name, schema[decl.name], feature_kind(decl.annotation), None, error=str(exc)))
            continue
        view = FrameView(sp.frame_plan)
        try:
            view.bind(exported)
            out.append(replace(plan, arrow_type=view.arrow_type(decl.name)))
        except ArrowKindError as exc:
            out.append(replace(plan, arrow_type=getattr(exc, "arrow_type", None), error=str(exc)))
        finally:
            view.release()
    return out
