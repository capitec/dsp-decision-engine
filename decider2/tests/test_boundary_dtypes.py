"""decider2.boundary.dtypes — the Arrow table (docs/BOUNDARY-REWORK.md §1.2).

One decision per (declared kind, polars dtype): read natively, ONE explicit
frame-tier cast, or `NeedsKernelSplit`. Nullability is not a tier any more
and nothing is probed; the old ladder's `DtypeTier`/`EntryMode`/five plan
classes/`probe_column` are gone (§9, Stage 3).
"""
from datetime import date, datetime
from decimal import Decimal

import polars as pl
import pytest

from decider2._arrow.frame import ArrowKindError
from decider2.boundary.dtypes import (
    ColumnPlan,
    NeedsKernelSplit,
    cast_frame,
    cast_series,
    explain_boundary,
    kind_for,
    plan_column,
)
from decider2.compile.driver import numpy_dtype
from decider2.types import FeatureKind as K
from decider2.types import Input, NullPolicy


# --- kind_for: row for row the driver's own dtype table ----------------------

@pytest.mark.parametrize("annotation,kind", [
    (float, K.F64), (int, K.I64), (bool, K.BOOL), (str, K.CODE), (bytes, K.STR),
    (float | None, K.F64), (int | None, K.F64), (None, K.F64), (object, K.F64),
])
def test_kind_for_mirrors_numpy_dtype(annotation, kind):
    """`int | None` is F64 here BECAUSE `compile.driver.numpy_dtype` types
    it float64 (pre-existing, doc 03 §1's OPTIONAL annotation is stored
    raw): the boundary and the kernel must never disagree about where a
    value lands."""
    assert kind_for(annotation) is kind
    expected = {K.F64: "float64", K.I64: "int64", K.BOOL: "bool", K.CODE: "int32"}
    if kind is not K.STR:
        assert numpy_dtype(annotation).name == expected[kind]


# --- plan_column: native / cast / reject ------------------------------------

@pytest.mark.parametrize("dtype,kind", [
    (pl.Float64(), K.F64), (pl.Float32(), K.F64),
    (pl.Int8(), K.I64), (pl.Int16(), K.I64), (pl.Int32(), K.I64), (pl.Int64(), K.I64),
    (pl.UInt8(), K.I64), (pl.UInt16(), K.I64), (pl.UInt32(), K.I64), (pl.UInt64(), K.I64),
    (pl.Date(), K.I64), (pl.Datetime("us"), K.I64), (pl.Duration("ms"), K.I64), (pl.Time(), K.I64),
    (pl.Boolean(), K.BOOL), (pl.Categorical(), K.CODE), (pl.Enum(["a"]), K.CODE), (pl.String(), K.STR),
])
def test_every_row_of_the_table_that_nanoarrow_reads_natively_gets_no_cast(dtype, kind):
    plan = plan_column("x", dtype, kind)
    assert plan.cast is None and plan.kind is kind and "native" in plan.note


@pytest.mark.parametrize("dtype,kind,cast", [
    (pl.Int64(), K.F64, pl.Float64()),          # an Int64 column declared `float` (or `int | None`)
    (pl.Boolean(), K.F64, pl.Float64()),
    (pl.Date(), K.F64, pl.Float64()),
    (pl.Float64(), K.I64, pl.Int64()),          # a Float64 column declared `int`
    (pl.Boolean(), K.I64, pl.Int64()),
    (pl.Int64(), K.BOOL, pl.Boolean()),
    (pl.Decimal(18, 2), K.I64, pl.Int64()),     # doc 03 §1.2: scaled int64 cents
    (pl.Decimal(18, 2), K.F64, pl.Float64()),
    (pl.String(), K.CODE, pl.Categorical()),    # a `str` input: the dictionary-code convention until Stage 7
    (pl.Categorical(), K.STR, pl.String()),     # a `bytes` input on an encoded column (Stage 6 does better)
    (pl.Null(), K.F64, pl.Float64()),           # an all-null column with no dtype of its own
    (pl.Null(), K.CODE, pl.Categorical()),
])
def test_a_castable_mismatch_gets_exactly_one_reported_frame_tier_cast(dtype, kind, cast):
    plan = plan_column("x", dtype, kind)
    assert plan.cast == cast and plan.kind is kind


@pytest.mark.parametrize("dtype", [
    pl.List(pl.Float64()), pl.Array(pl.Int64(), 2), pl.Struct({"a": pl.Float64()}), pl.Object(), pl.Binary(),
])
def test_a_column_with_no_flat_form_is_reported_for_the_kernel_split_by_name(dtype):
    with pytest.raises(NeedsKernelSplit) as exc_info:
        plan_column("amounts", dtype, K.F64)
    exc = exc_info.value
    assert exc.column == "amounts" and exc.dtype == dtype
    assert "'amounts'" in str(exc) and str(dtype) in str(exc) and "kernel-split" in str(exc)


def test_needs_kernel_split_is_an_arrow_kind_error():
    """One `except` covers "the kernel cannot read this column" whether the
    table said so (nested) or the import did (a String declared float)."""
    assert issubclass(NeedsKernelSplit, ArrowKindError)
    assert issubclass(NeedsKernelSplit, TypeError)


@pytest.mark.parametrize("dtype,kind", [
    (pl.String(), K.F64), (pl.String(), K.I64), (pl.Categorical(), K.I64), (pl.Float64(), K.CODE),
    (pl.Int64(), K.STR),
])
def test_a_pair_with_no_cast_is_left_for_the_import_to_refuse_by_arrow_type(dtype, kind):
    plan = plan_column("x", dtype, kind)
    assert plan.cast is None and "refuses it" in plan.note


# --- cast_series / cast_frame ------------------------------------------------

def test_decimal_casts_to_money_scaled_int64_cents():
    s = pl.Series("amount", [Decimal("19.99"), Decimal("5.00"), None], dtype=pl.Decimal(18, 2))
    out = cast_series(s, plan_column("amount", s.dtype, K.I64))
    assert out.dtype == pl.Int64 and out.to_list() == [1999, 500, None]


def test_decimal_rescales_from_a_different_source_scale_to_money_scale_2():
    s = pl.Series("amount", [Decimal("19.9950")], dtype=pl.Decimal(18, 4))
    assert cast_series(s, plan_column("amount", s.dtype, K.I64)).to_list() == [1999]
    s = pl.Series("amount", [Decimal("7")], dtype=pl.Decimal(18, 0))
    assert cast_series(s, plan_column("amount", s.dtype, K.I64)).to_list() == [700]


def test_decimal_declared_float_arrives_as_cents_too():
    """Cents on the wire regardless of the declared kind — what the old
    `ScaledInt64Plan` + `astype(float64)` produced, kept."""
    s = pl.Series("amount", [Decimal("1.25")], dtype=pl.Decimal(10, 2))
    out = cast_series(s, plan_column("amount", s.dtype, K.F64))
    assert out.dtype == pl.Float64 and out.to_list() == [125.0]


def test_a_decimal_that_overflows_int64_is_the_kernel_split_case_not_a_crash():
    s = pl.Series("m", [Decimal("9" * 30)], dtype=pl.Decimal(38, 0))
    with pytest.raises(NeedsKernelSplit, match=r"column 'm'.*frame-tier cast to Int64 failed"):
        cast_series(s, plan_column("m", s.dtype, K.I64))


def test_a_failing_cast_is_caught_as_baseexception_not_just_exception(monkeypatch):
    """EXPERIMENTS.md §A: a Rust panic is `pyo3_runtime.PanicException`,
    which does not inherit `Exception`."""
    import decider2.boundary.dtypes as mod

    class FakePanic(BaseException):
        pass

    def _boom(series, **kw):
        raise FakePanic("pretend Rust panic")

    monkeypatch.setattr(mod, "_decimal_as_cents", _boom)
    s = pl.Series("m", [Decimal("1.23")], dtype=pl.Decimal(18, 2))
    with pytest.raises(NeedsKernelSplit, match="FakePanic"):
        cast_series(s, plan_column("m", s.dtype, K.I64))


def test_temporal_columns_declared_float_cast_through_their_storage_integer():
    s = pl.Series("t", [datetime(1970, 1, 1, 0, 0, 1), None], dtype=pl.Datetime("us"))
    assert cast_series(s, plan_column("t", s.dtype, K.F64)).to_list() == [1_000_000.0, None]
    d = pl.Series("d", [date(1970, 1, 3)])
    assert cast_series(d, plan_column("d", d.dtype, K.F64)).to_list() == [2.0]


def test_cast_frame_returns_a_new_frame_and_never_retypes_the_callers_columns():
    frame = pl.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"], "k": [3, 4]})
    casts = [(0, plan_column("a", pl.Float64(), K.I64)), (1, plan_column("s", pl.String(), K.CODE))]
    out = cast_frame(frame, casts)
    assert out.dtypes == [pl.Int64, pl.Categorical, pl.Int64] and out.columns == frame.columns
    assert frame.dtypes == [pl.Float64, pl.String, pl.Int64]        # untouched
    assert out["a"].to_list() == [1, 2] and out["k"].to_list() == [3, 4]


# --- explain_boundary: the reporting contract -------------------------------

def test_explain_boundary_reports_arrow_type_kind_and_cast_per_declared_input():
    frame = pl.DataFrame({
        "clean_f": [1.0, 2.0], "n": [1, None], "cat": ["a", "b"], "flag": [True, False],
        "money": pl.Series([Decimal("1.00"), None], dtype=pl.Decimal(18, 2)),
    })
    inputs = [
        Input("clean_f", float), Input("n", int | None, NullPolicy.OPTIONAL), Input("cat", str),
        Input("flag", bool), Input("money", int),
    ]
    by_name = {p.name: p for p in explain_boundary(frame, inputs)}
    assert all(isinstance(p, ColumnPlan) and p.error is None for p in by_name.values())
    assert (by_name["clean_f"].kind, by_name["clean_f"].cast, by_name["clean_f"].arrow_type) == (K.F64, None, "double")
    assert (by_name["n"].kind, by_name["n"].cast, by_name["n"].arrow_type) == (K.F64, pl.Float64(), "double")
    assert (by_name["cat"].kind, by_name["cat"].cast) == (K.CODE, pl.Categorical())
    assert by_name["cat"].arrow_type.startswith("dictionary(")
    assert (by_name["flag"].kind, by_name["flag"].arrow_type) == (K.BOOL, "bool")
    assert (by_name["money"].kind, by_name["money"].cast, by_name["money"].arrow_type) == (K.I64, pl.Int64(), "int64")


def test_explain_boundary_without_inputs_reads_every_column_as_its_natural_kind():
    frame = pl.DataFrame({"x": [1.0], "i": [1], "b": [True], "s": ["a"], "c": pl.Series(["a"], dtype=pl.Categorical)})
    kinds = {p.name: p.kind for p in explain_boundary(frame)}
    assert kinds == {"x": K.F64, "i": K.I64, "b": K.BOOL, "s": K.STR, "c": K.CODE}


def test_explain_boundary_reports_a_refused_column_instead_of_raising():
    frame = pl.DataFrame({"l": [[1, 2]], "x": [1.0], "s": ["a"]})
    inputs = [Input("l", float), Input("x", float), Input("s", float)]
    by_name = {p.name: p for p in explain_boundary(frame, inputs)}
    assert "kernel-split" in by_name["l"].error and by_name["l"].arrow_type is None
    assert by_name["x"].error is None and by_name["x"].arrow_type == "double"
    assert "string_view" in by_name["s"].error and by_name["s"].arrow_type == "string_view"   # nanoarrow named it
