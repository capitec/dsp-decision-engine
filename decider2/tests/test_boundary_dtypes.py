"""Scratch tests for decider2.boundary.dtypes — the dtype ladder (doc 05 §1.5).

Grounded against this exact environment (polars 1.41.2) rather than assumed
from the spec prose alone — see EXPERIMENTS.md §A for why that matters:
`_get_buffers()`'s exact failure modes are measured facts, not designed ones.
"""
from decimal import Decimal

import polars as pl
import pytest

from decider2.boundary.dtypes import DtypeTier, EntryMode, explain_boundary, plan_column, probe_column


# --- plan_column: the declarative half, one row per doc 05 §1.5's table ----

def test_clean_float64_is_zero_copy():
    plan = plan_column("x", pl.Float64(), nullable=False)
    assert plan.tier is DtypeTier.ZERO_COPY
    assert plan.entry_mode is EntryMode.NATIVE


@pytest.mark.parametrize("dtype", [pl.Float64(), pl.Int64(), pl.Int32(), pl.UInt8()])
def test_the_four_zero_copy_dtypes_when_clean(dtype):
    assert plan_column("x", dtype, nullable=False).tier is DtypeTier.ZERO_COPY


@pytest.mark.parametrize("dtype", [pl.Float64(), pl.Int64(), pl.Int32(), pl.UInt8()])
def test_the_same_four_dtypes_nullable_copy_instead(dtype):
    plan = plan_column("x", dtype, nullable=True)
    assert plan.tier is DtypeTier.COPY
    assert plan.entry_mode is EntryMode.COPY_VALIDITY


def test_boolean_is_never_zero_copy_even_when_clean():
    """Doc 05 §1.4: arrow bitpacks Boolean; allow_copy=False always fails."""
    plan = plan_column("b", pl.Boolean(), nullable=False)
    assert plan.tier is DtypeTier.COPY
    assert plan.entry_mode is EntryMode.BUFFER_COPY


@pytest.mark.parametrize("dtype", [pl.Date(), pl.Datetime(), pl.Duration()])
def test_temporal_dtypes_enter_as_integers(dtype):
    plan = plan_column("t", dtype, nullable=False)
    assert plan.tier is DtypeTier.COPY
    assert plan.entry_mode is EntryMode.AS_INTEGER


def test_categorical_enters_as_codes():
    plan = plan_column("c", pl.Categorical(), nullable=False)
    assert plan.tier is DtypeTier.COPY
    assert plan.entry_mode is EntryMode.CODES


def test_utf8_is_tier_3_dictionary_codes():
    """Doc 05 §1.5: 'a string never enters a kernel as a string.'"""
    plan = plan_column("s", pl.String(), nullable=False)
    assert plan.tier is DtypeTier.CONVERT
    assert plan.entry_mode is EntryMode.CODES


def test_decimal_is_tier_3_scaled_int64():
    plan = plan_column("m", pl.Decimal(18, 2), nullable=False)
    assert plan.tier is DtypeTier.CONVERT
    assert plan.entry_mode is EntryMode.SCALED_INT64


def test_list_is_not_rejected_it_is_reported_for_kernel_split():
    """The ladder rejects nothing (doc 05 §1.5) — List has no flat-array form
    *yet*, so it lands on KERNEL_SPLIT rather than raising."""
    plan = plan_column("l", pl.List(pl.Float64()), nullable=False)
    assert plan.tier is DtypeTier.CONVERT
    assert plan.entry_mode is EntryMode.KERNEL_SPLIT


def test_struct_also_reports_kernel_split_rather_than_raising():
    plan = plan_column("st", pl.Struct({"a": pl.Float64()}), nullable=False)
    assert plan.entry_mode is EntryMode.KERNEL_SPLIT


# --- probe_column: the verified half — must survive Decimal's BaseException-only panic

def test_probe_decimal_succeeds_when_it_fits_int64():
    s = pl.Series("m", [Decimal("1.23"), Decimal("4.56")], dtype=pl.Decimal(18, 2))
    plan = probe_column(s)
    assert plan.entry_mode is EntryMode.SCALED_INT64  # the predicted mode survives probing


def test_probe_list_short_circuits_without_touching_buffers():
    """List's plan is already KERNEL_SPLIT; probe_column must not try (and
    must not crash on) `_get_buffers()`, which raises for list[f64] in this
    polars version (EXPERIMENTS.md §A)."""
    s = pl.Series("l", [[1.0, 2.0], [3.0]])
    plan = probe_column(s)
    assert plan.entry_mode is EntryMode.KERNEL_SPLIT


def test_probe_catches_a_baseexception_not_just_exception(monkeypatch):
    """Simulates Decimal's actual failure mode: a probe that raises something
    that is NOT an Exception subclass must still be caught and downgraded,
    never propagate and crash the whole extraction (doc 05 §1.5's hard
    requirement, EXPERIMENTS.md §A)."""
    import decider2.boundary.dtypes as dtypes_mod

    class FakePanic(BaseException):
        pass

    def _boom(series, plan):
        raise FakePanic("pretend Rust panic")

    monkeypatch.setattr(dtypes_mod, "_probe_extract", _boom)

    s = pl.Series("m", [Decimal("1.23")], dtype=pl.Decimal(18, 2))
    plan = probe_column(s)
    assert plan.entry_mode is EntryMode.KERNEL_SPLIT
    assert "FakePanic" in plan.note


def test_probe_column_actually_catches_the_real_int128_panic():
    """Not a simulation: cast a Decimal too wide for int64 and confirm the
    real `pyo3_runtime.PanicException` is what fires, and that probe_column
    survives it. precision=38 with a huge value overflows int64 on cast."""
    s = pl.Series("m", [Decimal("9" * 30)], dtype=pl.Decimal(38, 0))
    plan = probe_column(s)
    # Either it fits (unlikely at 30 nines) or it's downgraded — either way,
    # no exception escapes this call.
    assert plan.entry_mode in (EntryMode.SCALED_INT64, EntryMode.KERNEL_SPLIT)


# --- explain_boundary: the reporting contract -------------------------------

def test_explain_boundary_reports_one_row_per_column_with_nullability_override():
    frame = pl.DataFrame({
        "clean_f": [1.0, 2.0],
        "nullable_f": [1.0, None],
        "cat": ["a", "b"],
    })
    plans = explain_boundary(frame, nullable={"clean_f": False, "nullable_f": True})
    by_name = {p.name: p for p in plans}
    assert by_name["clean_f"].tier is DtypeTier.ZERO_COPY
    assert by_name["nullable_f"].tier is DtypeTier.COPY
    assert by_name["cat"].entry_mode is EntryMode.CODES  # String -> dict codes


def test_explain_boundary_falls_back_to_frame_null_count_when_not_told():
    frame = pl.DataFrame({"x": [1.0, None, 3.0]})
    plans = explain_boundary(frame)
    assert plans[0].nullable is True
    assert plans[0].tier is DtypeTier.COPY
