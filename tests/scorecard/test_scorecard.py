"""Scorecards bin inputs into points and sum them, with the answers the polars scorecard gave."""
from __future__ import annotations

import json

import polars as pl
import pytest

from decider.steps.scorecard import (
    AdjustedVariable,
    BoundBin,
    ConstantScore,
    DefaultBin,
    ScorecardConfig,
    ScoredVariable,
    ValuesBin,
)


def _sc(variables, output_name="score") -> ScorecardConfig:
    return ScorecardConfig(name="sc", variables=variables, output_name=output_name)


def _age(**kw) -> ScoredVariable:
    return ScoredVariable(
        variable_name="age",
        bins=[BoundBin(value=10.0, upper_bound=40.0), BoundBin(value=20.0, lower_bound=40.0)],
        default=DefaultBin(value=0.0), **kw,
    )


def test_bound_bins_assign_values_by_range():
    sc = _sc([ScoredVariable(
        variable_name="age",
        bins=[
            BoundBin(value=5.0, upper_bound=25.0),
            BoundBin(value=10.0, lower_bound=25.0, upper_bound=60.0),
            BoundBin(value=15.0, lower_bound=60.0),
        ],
        default=DefaultBin(value=0.0),
    )])
    result = sc.run(pl.DataFrame({"age": [20.0, 40.0, 70.0]}))
    assert result["score"].to_list() == pytest.approx([5.0, 10.0, 15.0])


def test_multiple_variables_score_is_sum():
    sc = _sc([
        _age(),
        ScoredVariable(
            variable_name="income",
            bins=[BoundBin(value=5.0, upper_bound=50_000.0), BoundBin(value=15.0, lower_bound=50_000.0)],
            default=DefaultBin(value=0.0),
        ),
    ])
    result = sc.run(pl.DataFrame({"age": [30.0, 50.0], "income": [30_000.0, 80_000.0]}))
    assert result["score"].to_list() == pytest.approx([15.0, 35.0])
    assert result["age_score"].to_list() == [10.0, 20.0]
    assert result["income_score"].to_list() == [5.0, 15.0]


def test_value_bins_match_categorical_inputs():
    sc = _sc([ScoredVariable(
        variable_name="status",
        bins=[ValuesBin(value=50.0, items=["vip"]), ValuesBin(value=10.0, items=["member"])],
        default=DefaultBin(value=0.0),
    )])
    result = sc.run(pl.DataFrame({"status": ["vip", "member", "unknown"]}))
    assert result["score"].to_list() == pytest.approx([50.0, 10.0, 0.0])


def test_default_bin_used_when_no_bin_matches():
    sc = _sc([ScoredVariable(
        variable_name="risk",
        bins=[BoundBin(value=100.0, lower_bound=1.0, upper_bound=2.0)],
        default=DefaultBin(value=-999.0),
        strict=False,
    )])
    result = sc.run(pl.DataFrame({"risk": [1.5, 5.0, -1.0]}))
    assert result["score"].to_list() == pytest.approx([100.0, -999.0, -999.0])


def test_constant_score_adds_fixed_value_to_all_rows():
    sc = _sc([_age(), ConstantScore(score=100.0, output_name="base")])
    result = sc.run(pl.DataFrame({"age": [30.0, 50.0]}))
    assert result["score"].to_list() == pytest.approx([110.0, 120.0])
    assert result["base"].to_list() == [100.0, 100.0]


def test_adjusted_variable_applies_scale_and_offset():
    sc = _sc([AdjustedVariable(variable=_age(), scale=2.0, offset=5.0)])
    result = sc.run(pl.DataFrame({"age": [30.0, 50.0]}))
    assert result["score"].to_list() == pytest.approx([25.0, 45.0])
    assert result["age_score"].to_list() == [10.0, 20.0]
    assert result["age_adjusted_score"].to_list() == [25.0, 45.0]


def test_upper_bounds_are_inclusive_and_lower_bounds_exclusive():
    result = _sc([_age()]).run(pl.DataFrame({"age": [40.0, 40.000001]}))
    assert result["score"].to_list() == [10.0, 20.0]


def test_values_bins_win_over_bound_bins_whatever_their_order():
    sc = _sc([ScoredVariable(
        variable_name="age",
        bins=[BoundBin(value=10.0, upper_bound=40.0), ValuesBin(value=99.0, items=[30, 50.0])],
        default=DefaultBin(value=0.0),
    )])
    assert sc.run(pl.DataFrame({"age": [30.0, 50.0, 60.0]}))["score"].to_list() == [99.0, 99.0, 0.0]


def test_a_null_input_scores_the_default_and_nan_clears_every_lower_bound():
    result = _sc([_age()]).run(pl.DataFrame({"age": [None, float("nan")]}, schema={"age": pl.Float64}))
    assert result["score"].to_list() == [0.0, 20.0]


def test_a_zero_bound_counts_as_unset():
    sc = _sc([ScoredVariable(
        variable_name="x", bins=[BoundBin(value=1.0, lower_bound=0.0, upper_bound=10.0)],
        default=DefaultBin(value=-1.0), strict=False,
    )])
    assert sc.run(pl.DataFrame({"x": [-5.0, 5.0, 11.0]}))["score"].to_list() == [1.0, 1.0, -1.0]


def test_a_legacy_scorecard_document_loads_and_round_trips():
    doc = {
        "type": "scorecard", "name": "sc", "output_name": "total",
        "variables": [
            {"type": "scored", "variable_name": "age", "strict": True, "default": {"value": 0, "name": "other"},
             "bins": [{"value": 5, "upper_bound": 25, "name": "young"}, {"value": 10, "lower_bound": 25}]},
            {"type": "adjusted", "scale": 2, "offset": 1,
             "variable": {"type": "scored", "variable_name": "status", "default": {"value": 0},
                          "bins": [{"value": 7, "items": ["vip", "gold"]}]}},
            {"type": "constant", "score": 3, "output_name": "base"},
        ],
    }
    card = ScorecardConfig.load(doc)
    df = pl.DataFrame({"age": [20.0, 30.0], "status": ["gold", "none"]})
    before = card.run(df)
    assert before["total"].to_list() == [5.0 + 15.0 + 3.0, 10.0 + 1.0 + 3.0]
    reloaded = ScorecardConfig.load(json.loads(card.model_dump_json()))
    assert reloaded == card
    assert reloaded.run(df).equals(before)


@pytest.mark.parametrize("bins, strict, message", [
    ([BoundBin(value=1, lower_bound=1, upper_bound=2), BoundBin(value=2, lower_bound=0, upper_bound=1)], True,
     "less than previous highest_bound"),
    ([BoundBin(value=1, lower_bound=0, upper_bound=1.5), BoundBin(value=2, lower_bound=1, upper_bound=2)], False,
     "less than previous highest_bound"),
    ([BoundBin(value=1, lower_bound=2, upper_bound=1)], True, "lower_bound >= upper_bound"),
    ([BoundBin(value=1, lower_bound=0, upper_bound=0.5), BoundBin(value=2, lower_bound=1, upper_bound=2)], True,
     "equal to \\(in strict mode\\)"),
    ([BoundBin(value=1, lower_bound=0), BoundBin(value=2, upper_bound=2)], True, "must define a upper_bound"),
    ([ValuesBin(value=1, items=["A"]), ValuesBin(value=2, items=["A"])], True, "Duplicate values"),
])
def test_invalid_bins_are_rejected(bins, strict, message):
    with pytest.raises(ValueError, match=message):
        ScoredVariable(variable_name="x", bins=bins, default=DefaultBin(value=0), strict=strict)


def test_gaps_are_allowed_when_not_strict():
    ScoredVariable(variable_name="x", default=DefaultBin(value=0), strict=False,
                   bins=[BoundBin(value=1, lower_bound=0, upper_bound=0.5), BoundBin(value=2, lower_bound=1)])


def test_a_variable_may_appear_only_once():
    with pytest.raises(ValueError, match="Duplicate variable_name 'age'"):
        _sc([_age(), AdjustedVariable(variable=_age())])
