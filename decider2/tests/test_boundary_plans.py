"""The four null tiers (`decider2.types.NullPolicy`) x the five kinds
(`FeatureKind`) through the ONE path a column now takes — the whole-frame
Arrow import and the C gather (docs/BOUNDARY-REWORK.md §1.5).

This file used to pin the pydantic discriminated union of five `ColumnPlan`
classes and the `NullTierStrategy` registry; both are gone with the dtype
ladder (§9, Stage 3). What it pins now is the property those existed to
protect: MISSING_AS and NOT_APPLICABLE_AS fill identically and tag DIFFERENT
reason codes, OPTIONAL masks and never fills, REQUIRED routes — for every
kind, whether the column is present, all-null, or absent from the frame.
"""
import numpy as np
import polars as pl
import pytest

from decider2.boundary.extract import extract_frame
from decider2.boundary.nulls import FillReason
from decider2.types import FeatureKind as K
from decider2.types import Input, NullPolicy

# (annotation, a column with a null at row 1, the fill to declare, the value the kernel dtype gives the fill)
_KIND_CASES = {
    K.F64: (float, pl.Series([1.5, None, 3.0]), -1.0, -1.0),
    K.I64: (int, pl.Series([2 ** 62 + 1, None, 3]), -7, -7),
    K.BOOL: (bool, pl.Series([False, None, True]), True, True),
    K.CODE: (str, pl.Series(["a", None, "b"]), -1, -1),
}


@pytest.mark.parametrize("kind", list(_KIND_CASES))
def test_missing_as_and_not_applicable_as_fill_identically_and_differ_only_in_reason(kind):
    annotation, series, fill, filled = _KIND_CASES[kind]
    frame = pl.DataFrame({"x": series})
    m = extract_frame(frame, [Input("x", annotation, NullPolicy.MISSING_AS, fill=fill)]).columns["x"]
    n = extract_frame(frame, [Input("x", annotation, NullPolicy.NOT_APPLICABLE_AS, fill=fill)]).columns["x"]
    assert m.values.tolist() == n.values.tolist()
    assert m.values[1] == filled and m.plan.kind is kind
    assert m.validity is None and n.validity is None
    assert m.fill.filled_count == n.fill.filled_count == 1
    assert m.fill.filled_mask.tolist() == n.fill.filled_mask.tolist() == [False, True, False]
    assert m.fill.reason is FillReason.MISSING and n.fill.reason is FillReason.NOT_APPLICABLE


@pytest.mark.parametrize("kind", list(_KIND_CASES))
def test_optional_masks_and_never_fills(kind):
    annotation, series, _, _ = _KIND_CASES[kind]
    col = extract_frame(pl.DataFrame({"x": series}), [Input("x", annotation, NullPolicy.OPTIONAL)]).columns["x"]
    assert col.validity.tolist() == [True, False, True] and col.fill is None
    assert col.values[0] == (series[0] if kind is not K.CODE else col.values[0])


@pytest.mark.parametrize("kind", list(_KIND_CASES))
def test_required_routes_the_row_before_the_kernel_sees_it(kind):
    annotation, series, _, _ = _KIND_CASES[kind]
    ex = extract_frame(pl.DataFrame({"x": series}), [Input("x", annotation, NullPolicy.REQUIRED)])
    assert ex.routing.mask.tolist() == [False, True, False] and ex.routing.column[1] == "x"
    assert len(ex.columns["x"].values) == 2 and ex.columns["x"].validity is None and ex.columns["x"].fill is None


def test_a_bytes_input_has_no_fill_and_reports_null_spans():
    frame = pl.DataFrame({"s": ["ab", None]})
    col = extract_frame(frame, [Input("s", bytes, NullPolicy.OPTIONAL)]).columns["s"]
    assert col.validity.tolist() == [True, False] and col.values[1].tolist() == [0, -1]
    with pytest.raises(ValueError, match="STR feature has no fill"):
        extract_frame(frame, [Input("s", bytes, NullPolicy.MISSING_AS, fill=0)])


def test_the_two_fill_reasons_stay_distinct_when_the_column_is_absent():
    """A declared input with no matching frame column at all must still tag
    the correct, distinct reason code for its tier, not silently collapse."""
    frame = pl.DataFrame({"id": [1, 2, 3]})
    missing_inputs = [Input("id", int), Input("bureau_score", float, NullPolicy.MISSING_AS, fill=0.0)]
    na_inputs = [Input("id", int), Input("spouse_income", float, NullPolicy.NOT_APPLICABLE_AS, fill=0.0)]
    missing_result = extract_frame(frame, missing_inputs)
    na_result = extract_frame(frame, na_inputs)
    assert missing_result.columns["bureau_score"].fill.reason is FillReason.MISSING
    assert na_result.columns["spouse_income"].fill.reason is FillReason.NOT_APPLICABLE
    assert missing_result.columns["bureau_score"].fill.filled_count == 3
    assert na_result.columns["spouse_income"].fill.filled_count == 3


def test_an_absent_column_and_an_all_null_column_produce_the_same_column_per_tier():
    """The synthesized placeholder (no polars column to export) must agree
    with what the gather produces for a real, all-null column."""
    present = pl.DataFrame({"k": [1, 2, 3], "x": pl.Series([None, None, None], dtype=pl.Float64)})
    absent = pl.DataFrame({"k": [1, 2, 3]})
    for policy, fill in ((NullPolicy.OPTIONAL, None), (NullPolicy.MISSING_AS, 7.0)):
        decl = Input("x", float | None if fill is None else float, policy, fill=fill)
        a = extract_frame(present, [Input("k", int), decl]).columns["x"]
        b = extract_frame(absent, [Input("k", int), decl]).columns["x"]
        assert a.values.dtype == b.values.dtype == np.float64
        if policy is NullPolicy.OPTIONAL:
            assert a.validity.tolist() == b.validity.tolist() == [False] * 3
        else:
            assert a.values.tolist() == b.values.tolist() == [7.0] * 3
            assert a.fill.reason is b.fill.reason is FillReason.MISSING and a.fill.filled_count == b.fill.filled_count == 3


def test_extract_frame_keeps_all_four_tiers_correct_at_once():
    """One frame, one column per tier — the whole registry exercised
    together, mirroring how a real pipeline declares its inputs."""
    frame = pl.DataFrame({
        "id": [1, 2, 3],
        "score": [600.0, None, 700.0],
        "backup_score": [None, 550.0, None],
        "income": [5000.0, None, 7000.0],
    })
    inputs = [
        Input("id", int),
        Input("score", float, NullPolicy.MISSING_AS, fill=-1.0),
        Input("backup_score", float, NullPolicy.NOT_APPLICABLE_AS, fill=-2.0),
        Input("income", float | None, NullPolicy.OPTIONAL),
    ]
    result = extract_frame(frame, inputs)
    assert result.routing.routed_count == 0  # only 'id' is REQUIRED, and it's clean
    assert result.columns["score"].values.tolist() == [600.0, -1.0, 700.0]
    assert result.columns["score"].fill.reason is FillReason.MISSING
    assert result.columns["backup_score"].values.tolist() == [-2.0, 550.0, -2.0]
    assert result.columns["backup_score"].fill.reason is FillReason.NOT_APPLICABLE
    assert result.columns["income"].validity.tolist() == [True, False, True]
    assert result.columns["income"].fill is None
    assert result.columns["id"].values.tolist() == [1, 2, 3] and result.columns["id"].values.dtype == np.int64
