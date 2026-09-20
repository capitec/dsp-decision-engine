"""Scratch tests for decider2.boundary.extract — doc 05 §1 + §2 tied together."""
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from decider2.boundary.dtypes import EntryMode
from decider2.boundary.extract import (
    NeedsKernelSplit,
    extract_column,
    extract_frame,
    is_clean,
    rechunk_once,
)
from decider2.types import Decision, Input, MissingInputPolicy, NullPolicy


# --- rechunk_once / is_clean -------------------------------------------------

def test_rechunk_once_is_idempotent_and_cheap():
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    out = rechunk_once(frame)
    assert out.equals(frame)


def test_is_clean_true_for_no_nulls():
    assert is_clean(pl.Series("x", [1.0, 2.0]))


def test_is_clean_false_with_a_null():
    assert not is_clean(pl.Series("x", [1.0, None]))


# --- extract_column: REQUIRED, tier 1 zero-copy ------------------------------

def test_required_clean_float64_extracts_zero_copy():
    s = pl.Series("net_income", [9200.0, 4100.0])
    col = extract_column(s)
    assert col.values.tolist() == [9200.0, 4100.0]
    assert col.validity is None
    assert col.plan.entry_mode is EntryMode.NATIVE
    assert col.values.flags["OWNDATA"] is False  # zero-copy: shares polars' buffer


def test_required_with_remaining_nulls_refuses_to_silently_extract():
    """extract_column is only safe to call on an already-routed REQUIRED
    column (doc 03 §1) — calling it directly on one with nulls must not
    silently feed garbage into what would become a kernel array."""
    s = pl.Series("instalment", [100.0, None])
    with pytest.raises(ValueError, match="route it with"):
        extract_column(s)


# --- extract_column: MISSING_AS / NOT_APPLICABLE_AS --------------------------

def test_missing_as_column_is_filled_not_routed():
    decl = Input(name="bureau_score", annotation=float, null_policy=NullPolicy.MISSING_AS, fill=0.0)
    col = extract_column(pl.Series("bureau_score", [10.0, None]), decl)
    assert col.values.tolist() == [10.0, 0.0]
    assert col.fill.filled_count == 1
    assert col.validity is None  # the kernel sees a plain number, no mask needed


# --- extract_column: OPTIONAL -------------------------------------------------

def test_optional_column_keeps_its_validity_mask_and_does_not_fill():
    decl = Input(name="bureau_score", annotation=float | None, null_policy=NullPolicy.OPTIONAL)
    col = extract_column(pl.Series("bureau_score", [10.0, None]), decl)
    assert col.validity.tolist() == [True, False]
    # index 1's raw value is untouched garbage, not the fill semantics of tier 2/4
    assert col.fill is None


def test_optional_clean_column_still_gets_an_all_true_mask():
    decl = Input(name="x", annotation=float | None, null_policy=NullPolicy.OPTIONAL)
    col = extract_column(pl.Series("x", [1.0, 2.0]), decl)
    assert col.validity.tolist() == [True, True]


# --- extract_column: Boolean, Categorical, Decimal, List ---------------------

def test_boolean_extracts_via_copy():
    col = extract_column(pl.Series("flag", [True, False, True]))
    assert col.values.tolist() == [True, False, True]
    assert col.plan.entry_mode is EntryMode.BUFFER_COPY


def test_categorical_extracts_as_codes():
    s = pl.Series("segment", ["retail", "sme", "retail"]).cast(pl.Categorical)
    col = extract_column(s)
    assert col.plan.entry_mode is EntryMode.CODES
    assert col.values[0] == col.values[2]  # same category -> same code
    assert col.values[0] != col.values[1]


def test_utf8_extracts_as_dictionary_codes():
    col = extract_column(pl.Series("segment", ["retail", "sme", "retail"]))
    assert col.plan.entry_mode is EntryMode.CODES
    assert col.values[0] == col.values[2]
    assert col.values.dtype.kind in ("u", "i")  # integer codes, never strings


def test_decimal_extracts_as_scaled_int64_cents():
    s = pl.Series("amount", [Decimal("19.99"), Decimal("5.00")], dtype=pl.Decimal(18, 2))
    col = extract_column(s)
    assert col.plan.entry_mode is EntryMode.SCALED_INT64
    assert col.values.tolist() == [1999, 500]
    assert col.values.dtype == np.int64


def test_decimal_rescales_from_a_different_source_scale_to_money_scale_2():
    """precision/scale(18,4) source -> money_scale=2 cents on the wire."""
    s = pl.Series("amount", [Decimal("19.9950")], dtype=pl.Decimal(18, 4))
    col = extract_column(s)
    assert col.values.tolist() == [1999]  # 19.9950 -> 1999 cents (truncating the sub-cent digit)


def test_list_column_raises_needs_kernel_split_not_a_silent_drop():
    s = pl.Series("amounts", [[1.0, 2.0], [3.0]])
    with pytest.raises(NeedsKernelSplit) as exc_info:
        extract_column(s)
    assert exc_info.value.column == "amounts"


# --- extract_frame: whole-frame orchestration --------------------------------

def test_extract_frame_end_to_end_matches_the_flagship_shape():
    frame = pl.DataFrame({
        "net_income": [9200.0, 4100.0, 15000.0, 4999.0],
        "expenses": [3100.0, 1500.0, 6000.0, 2000.0],
    })
    inputs = [
        Input(name="net_income", annotation=float, null_policy=NullPolicy.REQUIRED),
        Input(name="expenses", annotation=float, null_policy=NullPolicy.REQUIRED),
    ]
    result = extract_frame(frame, inputs)
    assert result.routing.routed_count == 0
    assert result.kernel_frame.height == 4
    assert result.columns["net_income"].values.tolist() == [9200.0, 4100.0, 15000.0, 4999.0]


def test_extract_frame_removes_routed_rows_from_the_kernel_frame():
    frame = pl.DataFrame({"instalment": [100.0, None, 300.0], "id": [1, 2, 3]})
    inputs = [
        Input(name="instalment", annotation=float, null_policy=NullPolicy.REQUIRED),
        Input(name="id", annotation=int, null_policy=NullPolicy.REQUIRED),
    ]
    result = extract_frame(frame, inputs)
    assert result.routing.routed_count == 1
    assert result.kernel_frame.height == 2
    assert result.columns["instalment"].values.tolist() == [100.0, 300.0]
    assert result.routing.decision is Decision.REFER


def test_extract_frame_raise_for_fails_before_any_column_is_extracted():
    frame = pl.DataFrame({"application_id": [1, None]})
    inputs = [Input(name="application_id", annotation=int, null_policy=NullPolicy.REQUIRED)]
    policy = MissingInputPolicy(raise_for=("application_id",))
    with pytest.raises(ValueError):
        extract_frame(frame, inputs, policy=policy)


def test_extract_frame_synthesizes_a_required_column_entirely_absent_from_the_frame():
    """Review finding 4: absent and null share one routing path. A REQUIRED
    input with no matching frame column at all routes every row away
    (mirroring `route_required_nulls`) rather than being silently skipped —
    the old behaviour left nothing in `columns` for it, and the kernel
    later raised a bare `KeyError` three frames deep."""
    frame = pl.DataFrame({"a": [1.0]})
    inputs = [
        Input(name="a", annotation=float, null_policy=NullPolicy.REQUIRED),
        Input(name="missing_col", annotation=float, null_policy=NullPolicy.REQUIRED),
    ]
    result = extract_frame(frame, inputs)
    assert set(result.columns) == {"a", "missing_col"}
    assert result.routing.routed_count == 1
    assert result.kernel_frame.height == 0
    assert result.columns["missing_col"].values.tolist() == []


def test_extract_frame_fills_a_missing_as_column_entirely_absent_from_the_frame():
    decl = Input(name="bureau_score", annotation=float, null_policy=NullPolicy.MISSING_AS, fill=7.0)
    frame = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = extract_frame(frame, [Input(name="a", annotation=float, null_policy=NullPolicy.REQUIRED), decl])
    assert result.columns["bureau_score"].values.tolist() == [7.0, 7.0, 7.0]
    assert result.columns["bureau_score"].fill.filled_count == 3
