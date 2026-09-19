"""Scratch tests for decider2.boundary.nulls — doc 03 §1's four situations.

The centrepiece is `route_required_nulls`: doc 03 §1 calls this "the
highest-ranked production risk in the design" — a null in a required column
must produce a Decision, not an exception, by default.
"""
import numpy as np
import polars as pl
import pytest

from decider2.boundary.nulls import (
    FillReason,
    fill_column,
    route_required_nulls,
    validity_mask,
)
from decider2.types import Decision, Input, MissingInputPolicy, NullPolicy


# --- validity_mask -----------------------------------------------------------

def test_validity_mask_is_none_for_a_clean_column():
    assert validity_mask(pl.Series("x", [1.0, 2.0, 3.0])) is None


def test_validity_mask_true_means_valid():
    mask = validity_mask(pl.Series("x", [1.0, None, 3.0]))
    assert mask.tolist() == [True, False, True]


# --- fill_column: tiers 2 and 4 ----------------------------------------------

def test_missing_as_fills_the_null_slot_with_the_declared_value():
    decl = Input(name="bureau_score", annotation=float, null_policy=NullPolicy.MISSING_AS, fill=0.0)
    values, info = fill_column(pl.Series("bureau_score", [10.0, None, 30.0]), decl)
    assert values.tolist() == [10.0, 0.0, 30.0]
    assert info.reason is FillReason.MISSING
    assert info.filled_count == 1
    assert info.filled_mask.tolist() == [False, True, False]


def test_not_applicable_as_fills_identically_but_tags_a_different_reason():
    decl = Input(name="spouse_income", annotation=float, null_policy=NullPolicy.NOT_APPLICABLE_AS, fill=0.0)
    values, info = fill_column(pl.Series("spouse_income", [None, 500.0]), decl)
    assert values.tolist() == [0.0, 500.0]
    assert info.reason is FillReason.NOT_APPLICABLE


def test_fill_column_on_a_clean_column_fills_nothing_but_still_returns_values():
    decl = Input(name="x", annotation=float, null_policy=NullPolicy.MISSING_AS, fill=-1.0)
    values, info = fill_column(pl.Series("x", [1.0, 2.0]), decl)
    assert values.tolist() == [1.0, 2.0]
    assert info.filled_count == 0
    assert info.filled_mask is None


def test_fill_column_rejects_a_required_input():
    decl = Input(name="x", annotation=float, null_policy=NullPolicy.REQUIRED)
    with pytest.raises(ValueError):
        fill_column(pl.Series("x", [1.0, None]), decl)


def test_the_garbage_slot_is_never_read_unfiltered():
    """Doc 05 §2: 'the slot under a null contains leftover garbage, not
    zero' — measured there via a left join. In this exact polars build
    (1.41.2) a left join happens to zero-fill the values buffer instead (see
    this test's sibling check below), so a `when/then/otherwise` construction
    is used here to genuinely reproduce a nonzero value sitting under a null
    slot, and confirm the fill overwrites it rather than only working by
    coincidence on a buffer that happens to already be zero.
    """
    picked = pl.Series("m", [True, False, True])
    source = pl.Series("v", [9.0, 9.0, 9.0])
    s = pl.DataFrame({"m": picked, "v": source}).select(
        pl.when(pl.col("m")).then(pl.col("v")).otherwise(None).alias("v")
    )["v"]
    assert s.null_count() == 1
    assert s._get_buffers()["values"].to_numpy(allow_copy=False)[1] == 9.0  # the garbage, confirmed present

    decl = Input(name="v", annotation=float, null_policy=NullPolicy.MISSING_AS, fill=0.0)
    values, info = fill_column(s, decl)
    assert values[1] == 0.0  # not 9.0 — the garbage was overwritten, not trusted


def test_a_left_join_null_is_handled_too_even_though_this_polars_build_zero_fills_it():
    """Belt and braces: whether or not the values buffer under a join-induced
    null happens to be garbage or zero in a given polars build, `fill_column`
    must not depend on which — it must always substitute the declared value
    at every invalid position."""
    left = pl.DataFrame({"k": [1, 2, 3]})
    right = pl.DataFrame({"k": [1, 3], "v": [9.0, 9.0]})
    s = left.join(right, on="k", how="left")["v"]
    assert s.null_count() == 1

    decl = Input(name="v", annotation=float, null_policy=NullPolicy.MISSING_AS, fill=-1.0)
    values, info = fill_column(s, decl)
    assert values.tolist() == [9.0, -1.0, 9.0]


# --- route_required_nulls: the production-risk contract ----------------------

def test_default_policy_refers_rather_than_raising():
    frame = pl.DataFrame({"instalment": [100.0, None, 300.0]})
    inputs = [Input(name="instalment", annotation=float, null_policy=NullPolicy.REQUIRED)]
    routing = route_required_nulls(frame, inputs)  # no policy passed -> defaults
    assert routing.decision is Decision.REFER  # MissingInputPolicy()'s own default
    assert routing.mask.tolist() == [False, True, False]
    assert routing.column == (None, "instalment", None)
    assert routing.routed_count == 1


def test_raise_for_hard_fails_the_whole_batch_by_name():
    frame = pl.DataFrame({"application_id": [1, None, 3]})
    inputs = [Input(name="application_id", annotation=int, null_policy=NullPolicy.REQUIRED)]
    policy = MissingInputPolicy(raise_for=("application_id",))
    with pytest.raises(ValueError, match="application_id.*required.*1 null"):
        route_required_nulls(frame, inputs, policy)


def test_raise_for_message_matches_doc_05_2s_shape():
    frame = pl.DataFrame({"instalment": [100.0] * 7 + [None, 100.0] * 2 + [100.0] * 991})
    inputs = [Input(name="instalment", annotation=float, null_policy=NullPolicy.REQUIRED)]
    policy = MissingInputPolicy(raise_for=("instalment",))
    with pytest.raises(ValueError) as exc_info:
        route_required_nulls(frame, inputs, policy)
    msg = str(exc_info.value)
    assert "step argument 'instalment' is declared required" in msg
    assert "2 null(s)" in msg
    assert "Either fix the input or declare" in msg


def test_a_non_raise_for_required_null_never_reaches_the_kernel_frame_untreated():
    """This is the property doc 03 §1 actually cares about: routing, not raising."""
    frame = pl.DataFrame({"bureau_score": [1.0, None]})
    inputs = [Input(name="bureau_score", annotation=float, null_policy=NullPolicy.REQUIRED)]
    routing = route_required_nulls(frame, inputs, MissingInputPolicy(default=Decision.DECLINE, reason=9001))
    assert routing.decision is Decision.DECLINE
    assert routing.reason == 9001
    assert routing.mask.tolist() == [False, True]


def test_first_match_wins_when_two_required_columns_are_both_null_on_one_row():
    frame = pl.DataFrame({"a": [None, 1.0], "b": pl.Series([None, None], dtype=pl.Float64)})
    inputs = [
        Input(name="a", annotation=float, null_policy=NullPolicy.REQUIRED),
        Input(name="b", annotation=float, null_policy=NullPolicy.REQUIRED),
    ]
    routing = route_required_nulls(frame, inputs)
    assert routing.mask.tolist() == [True, True]
    # row 0: 'a' is null first (declaration order) -> 'a' wins even though 'b' is also null
    assert routing.column[0] == "a"
    # row 1: only 'b' is null
    assert routing.column[1] == "b"


def test_a_column_not_present_in_the_frame_is_not_this_functions_problem():
    """Unbound-input detection is graph/resolve.py's job (doc 03 §2.2, O23)."""
    frame = pl.DataFrame({"a": [1.0, 2.0]})
    inputs = [Input(name="typo_name", annotation=float, null_policy=NullPolicy.REQUIRED)]
    routing = route_required_nulls(frame, inputs)
    assert routing.routed_count == 0


def test_a_clean_required_column_routes_nothing():
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    inputs = [Input(name="x", annotation=float, null_policy=NullPolicy.REQUIRED)]
    routing = route_required_nulls(frame, inputs)
    assert routing.routed_count == 0
    assert not routing.mask.any()
