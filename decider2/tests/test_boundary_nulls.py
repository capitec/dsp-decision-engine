"""decider2.boundary.nulls — doc 03 §1's four situations.

The centrepiece is `route_required_nulls`: doc 03 §1 calls this "the
highest-ranked production risk in the design" — a null in a required column
must produce a Decision, not an exception, by default. Filling
(MISSING_AS / NOT_APPLICABLE_AS) and masking (OPTIONAL) happen in the C
gather now (docs/BOUNDARY-REWORK.md §1.5) and are tested through
`extract_frame` in `test_boundary_extract.py`; this file keeps what is
still Python: routing, and the reason a fill is tagged with.
"""
import numpy as np
import polars as pl
import pytest

from decider2.boundary.nulls import FillReason, fill_reason, route_required_nulls
from decider2.types import Decision, Input, MissingInputPolicy, NullPolicy


# --- fill_reason: the one thing tiers 2 and 4 differ in ------------------------

def test_only_the_two_fill_tiers_have_a_reason_and_they_differ():
    assert fill_reason(NullPolicy.MISSING_AS) is FillReason.MISSING
    assert fill_reason(NullPolicy.NOT_APPLICABLE_AS) is FillReason.NOT_APPLICABLE
    assert fill_reason(NullPolicy.REQUIRED) is None and fill_reason(NullPolicy.OPTIONAL) is None
    assert FillReason.MISSING is not FillReason.NOT_APPLICABLE


# --- route_required_nulls: the production-risk contract -------------------------

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
    assert "first at row 7" in msg
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


def test_a_required_column_absent_from_the_frame_routes_every_row():
    """Review finding 4: absent and null share one routing path. A REQUIRED
    column that is not in `frame` at all routes every row exactly as a
    REQUIRED column full of nulls would — not silently ignored (that was
    the bug: a bare `KeyError` three frames deep in the kernel, since
    nothing ever populated the registry for it)."""
    frame = pl.DataFrame({"a": [1.0, 2.0]})
    inputs = [Input(name="missing_col", annotation=float, null_policy=NullPolicy.REQUIRED)]
    routing = route_required_nulls(frame, inputs)
    assert routing.routed_count == 2
    assert routing.mask.tolist() == [True, True]
    assert routing.column == ("missing_col", "missing_col")


def test_a_required_raise_for_column_absent_from_the_frame_raises_by_name():
    frame = pl.DataFrame({"a": [1.0, 2.0]})
    inputs = [Input(name="missing_col", annotation=float, null_policy=NullPolicy.REQUIRED)]
    policy = MissingInputPolicy(raise_for=("missing_col",))
    with pytest.raises(ValueError, match="missing_col"):
        route_required_nulls(frame, inputs, policy)


def test_a_clean_required_column_routes_nothing():
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    inputs = [Input(name="x", annotation=float, null_policy=NullPolicy.REQUIRED)]
    routing = route_required_nulls(frame, inputs)
    assert routing.routed_count == 0
    assert not routing.mask.any()


def test_only_required_inputs_route_and_a_null_dtype_column_counts_as_all_null():
    frame = pl.DataFrame({"x": [1.0, None], "z": [None, None], "o": [None, 2.0]})
    assert frame.schema["z"] == pl.Null
    inputs = [
        Input("x", float, NullPolicy.MISSING_AS, fill=0.0),
        Input("o", float | None, NullPolicy.OPTIONAL),
        Input("z", float, NullPolicy.REQUIRED),
    ]
    routing = route_required_nulls(frame, inputs)
    assert routing.mask.tolist() == [True, True] and routing.column == ("z", "z")


def test_routing_a_zero_row_frame_is_empty_not_an_error():
    frame = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)})
    routing = route_required_nulls(frame, [Input("x", float)])
    assert routing.routed_count == 0 and routing.mask.dtype == np.bool_ and routing.column == ()
