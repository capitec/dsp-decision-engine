from datetime import date, datetime
from decimal import Decimal

import polars as pl
import pytest

import decider.engine.boundary.dtypes as dtypes_mod
from decider.engine.boundary._arrow import available
from decider.engine.boundary._arrow.plan import ArrowKindError
from decider.engine.boundary.dtypes import (
    ColumnPlan, NeedsKernelSplit, cast_frame, cast_series, explain_boundary, plan_column,
)
from decider.engine.ir.decls import FeatureKind as K
from decider.engine.ir.decls import Input, NullPolicy, feature_kind

needs_shim = pytest.mark.skipif(not available(), reason="the Arrow shim can't be built here (no C compiler)")


@pytest.mark.parametrize("annotation,kind", [
    (float, K.F64), (int, K.I64), (bool, K.BOOL), (str, K.CODE), (bytes, K.STR),
    (float | None, K.F64), (int | None, K.F64), (None, K.F64), (object, K.F64),
])
def test_an_annotation_lands_in_its_feature_kind_and_optional_int_is_f64(annotation, kind):
    assert feature_kind(annotation) is kind


@pytest.mark.parametrize("dtype,kind", [
    (pl.Float64(), K.F64), (pl.Float32(), K.F64),
    (pl.Int8(), K.I64), (pl.Int16(), K.I64), (pl.Int32(), K.I64), (pl.Int64(), K.I64),
    (pl.UInt8(), K.I64), (pl.UInt16(), K.I64), (pl.UInt32(), K.I64), (pl.UInt64(), K.I64),
    (pl.Date(), K.I64), (pl.Datetime("us"), K.I64), (pl.Duration("ms"), K.I64), (pl.Time(), K.I64),
    (pl.Boolean(), K.BOOL), (pl.Categorical(), K.CODE), (pl.Enum(["a"]), K.CODE), (pl.String(), K.STR),
])
def test_a_dtype_nanoarrow_reads_natively_gets_no_cast(dtype, kind):
    plan = plan_column("x", dtype, kind)
    assert plan.cast is None and plan.kind is kind and "native" in plan.note


@pytest.mark.parametrize("dtype,kind,cast", [
    (pl.Int64(), K.F64, pl.Float64()),
    (pl.Boolean(), K.F64, pl.Float64()),
    (pl.Date(), K.F64, pl.Float64()),
    (pl.Float64(), K.I64, pl.Int64()),
    (pl.Boolean(), K.I64, pl.Int64()),
    (pl.Int64(), K.BOOL, pl.Boolean()),
    (pl.Decimal(18, 2), K.I64, pl.Int64()),
    (pl.Decimal(18, 2), K.F64, pl.Float64()),
    (pl.String(), K.CODE, pl.Categorical()),
    (pl.Categorical(), K.STR, pl.String()),
    (pl.Null(), K.F64, pl.Float64()),
    (pl.Null(), K.CODE, pl.Categorical()),
])
def test_a_castable_mismatch_gets_exactly_one_frame_tier_cast(dtype, kind, cast):
    plan = plan_column("x", dtype, kind)
    assert plan.cast == cast and plan.kind is kind


@pytest.mark.parametrize("dtype", [
    pl.List(pl.Float64()), pl.Array(pl.Int64(), 2), pl.Struct({"a": pl.Float64()}), pl.Object(), pl.Binary(),
])
def test_a_column_with_no_flat_form_needs_a_kernel_split_by_name(dtype):
    with pytest.raises(NeedsKernelSplit) as exc_info:
        plan_column("amounts", dtype, K.F64)
    exc = exc_info.value
    assert exc.column == "amounts" and exc.dtype == dtype
    assert "'amounts'" in str(exc) and str(dtype) in str(exc) and "kernel-split" in str(exc)


def test_needs_kernel_split_is_an_arrow_kind_error():
    assert issubclass(NeedsKernelSplit, ArrowKindError) and issubclass(NeedsKernelSplit, TypeError)


@pytest.mark.parametrize("dtype,kind", [
    (pl.String(), K.F64), (pl.String(), K.I64), (pl.Categorical(), K.I64), (pl.Float64(), K.CODE),
    (pl.Int64(), K.STR),
])
def test_a_pair_with_no_cast_is_left_for_the_import_to_refuse(dtype, kind):
    plan = plan_column("x", dtype, kind)
    assert plan.cast is None and "refuses it" in plan.note


def test_decimal_casts_to_money_scaled_int64_cents():
    s = pl.Series("amount", [Decimal("19.99"), Decimal("5.00"), None], dtype=pl.Decimal(18, 2))
    out = cast_series(s, plan_column("amount", s.dtype, K.I64))
    assert out.dtype == pl.Int64 and out.to_list() == [1999, 500, None]


def test_decimal_rescales_from_any_source_scale_to_cents():
    s = pl.Series("amount", [Decimal("19.9950")], dtype=pl.Decimal(18, 4))
    assert cast_series(s, plan_column("amount", s.dtype, K.I64)).to_list() == [1999]
    s = pl.Series("amount", [Decimal("7")], dtype=pl.Decimal(18, 0))
    assert cast_series(s, plan_column("amount", s.dtype, K.I64)).to_list() == [700]


def test_decimal_declared_float_arrives_as_cents_too():
    s = pl.Series("amount", [Decimal("1.25")], dtype=pl.Decimal(10, 2))
    out = cast_series(s, plan_column("amount", s.dtype, K.F64))
    assert out.dtype == pl.Float64 and out.to_list() == [125.0]


def test_a_decimal_that_overflows_int64_needs_a_kernel_split_not_a_crash():
    s = pl.Series("m", [Decimal("9" * 30)], dtype=pl.Decimal(38, 0))
    with pytest.raises(NeedsKernelSplit, match=r"column 'm'.*frame-tier cast to Int64 failed"):
        cast_series(s, plan_column("m", s.dtype, K.I64))


def test_a_rust_panic_during_a_cast_is_caught_even_though_it_is_not_an_exception(monkeypatch):
    class FakePanic(BaseException):
        pass

    def _boom(series, **kw):
        raise FakePanic("pretend Rust panic")

    monkeypatch.setattr(dtypes_mod, "_decimal_as_cents", _boom)
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
    assert frame.dtypes == [pl.Float64, pl.String, pl.Int64]
    assert out["a"].to_list() == [1, 2] and out["k"].to_list() == [3, 4]


@needs_shim
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


@needs_shim
def test_explain_boundary_without_inputs_reads_every_column_as_its_natural_kind():
    frame = pl.DataFrame({"x": [1.0], "i": [1], "b": [True], "s": ["a"], "c": pl.Series(["a"], dtype=pl.Categorical)})
    kinds = {p.name: p.kind for p in explain_boundary(frame)}
    assert kinds == {"x": K.F64, "i": K.I64, "b": K.BOOL, "s": K.STR, "c": K.CODE}


@needs_shim
def test_explain_boundary_reports_a_refused_column_instead_of_raising():
    frame = pl.DataFrame({"l": [[1, 2]], "x": [1.0], "s": ["a"]})
    inputs = [Input("l", float), Input("x", float), Input("s", float)]
    by_name = {p.name: p for p in explain_boundary(frame, inputs)}
    assert "kernel-split" in by_name["l"].error and by_name["l"].arrow_type is None
    assert by_name["x"].error is None and by_name["x"].arrow_type == "double"
    assert "string_view" in by_name["s"].error and by_name["s"].arrow_type == "string_view"
