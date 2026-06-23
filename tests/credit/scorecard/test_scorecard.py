"""
Integration tests for ScoreCard.

Exercises bound bins, value bins, default fallback, constant scores,
adjusted variables, and multi-variable score summing.
"""

import polars as pl
import pytest

from decider.modules.credit.scorecard import (
    ScoreCard,
    ScoredVariable,
    AdjustedVariable,
    ConstantScore,
    BoundBin,
    ValuesBin,
    DefaultBin,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _sc(variables, output_name="score") -> ScoreCard:
    return ScoreCard(name="sc", variables=variables, output_name=output_name)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_bound_bins_assign_values_by_range():
    """BoundBins map numeric ranges to their configured values."""
    sc = _sc([
        ScoredVariable(
            type="scored",
            variable_name="age",
            bins=[
                BoundBin(value=5.0,  upper_bound=25.0),
                BoundBin(value=10.0, lower_bound=25.0, upper_bound=60.0),
                BoundBin(value=15.0, lower_bound=60.0),
            ],
            default=DefaultBin(value=0.0),
        )
    ])
    df = pl.DataFrame({"age": [20.0, 40.0, 70.0]})
    result = sc({"input": df})
    assert result["score"].to_list() == pytest.approx([5.0, 10.0, 15.0])


def test_multiple_variables_score_is_sum():
    """Each variable contributes independently; the final score is their sum."""
    sc = _sc([
        ScoredVariable(
            type="scored",
            variable_name="age",
            bins=[
                BoundBin(value=10.0, upper_bound=40.0),
                BoundBin(value=20.0, lower_bound=40.0),
            ],
            default=DefaultBin(value=0.0),
        ),
        ScoredVariable(
            type="scored",
            variable_name="income",
            bins=[
                BoundBin(value=5.0,  upper_bound=50_000.0),
                BoundBin(value=15.0, lower_bound=50_000.0),
            ],
            default=DefaultBin(value=0.0),
        ),
    ])
    df = pl.DataFrame({
        "age":    [30.0,   50.0],
        "income": [30_000.0, 80_000.0],
    })
    result = sc({"input": df})
    # Row 0: age=10 + income=5  = 15
    # Row 1: age=20 + income=15 = 35
    assert result["score"].to_list() == pytest.approx([15.0, 35.0])


def test_value_bins_match_categorical_inputs():
    """ValuesBins assign scores to specific string values; unmatched rows get the default."""
    sc = _sc([
        ScoredVariable(
            type="scored",
            variable_name="status",
            bins=[
                ValuesBin(value=50.0, items=["vip"]),
                ValuesBin(value=10.0, items=["member"]),
            ],
            default=DefaultBin(value=0.0),
        )
    ])
    df = pl.DataFrame({"status": ["vip", "member", "unknown"]})
    result = sc({"input": df})
    assert result["score"].to_list() == pytest.approx([50.0, 10.0, 0.0])


def test_default_bin_used_when_no_bin_matches():
    """Rows outside every defined bin receive the default score."""
    sc = _sc([
        ScoredVariable(
            type="scored",
            variable_name="risk",
            bins=[BoundBin(value=100.0, lower_bound=1.0, upper_bound=2.0)],
            default=DefaultBin(value=-999.0),
            strict=False,
        )
    ])
    df = pl.DataFrame({"risk": [1.5, 5.0, -1.0]})
    result = sc({"input": df})
    assert result["score"].to_list() == pytest.approx([100.0, -999.0, -999.0])


def test_constant_score_adds_fixed_value_to_all_rows():
    """ConstantScore contributes the same amount to every row regardless of input."""
    sc = _sc([
        ScoredVariable(
            type="scored",
            variable_name="age",
            bins=[BoundBin(value=10.0, upper_bound=40.0), BoundBin(value=20.0, lower_bound=40.0)],
            default=DefaultBin(value=0.0),
        ),
        ConstantScore(type="constant", score=100.0, output_name="base"),
    ])
    df = pl.DataFrame({"age": [30.0, 50.0]})
    result = sc({"input": df})
    # age=30 → 10 + 100 = 110; age=50 → 20 + 100 = 120
    assert result["score"].to_list() == pytest.approx([110.0, 120.0])


def test_adjusted_variable_applies_scale_and_offset():
    """AdjustedVariable multiplies the inner score by scale and adds offset before summing."""
    inner = ScoredVariable(
        type="scored",
        variable_name="age",
        bins=[BoundBin(value=10.0, upper_bound=40.0), BoundBin(value=20.0, lower_bound=40.0)],
        default=DefaultBin(value=0.0),
    )
    sc = _sc([
        AdjustedVariable(type="adjusted", variable=inner, scale=2.0, offset=5.0)
    ])
    df = pl.DataFrame({"age": [30.0, 50.0]})
    result = sc({"input": df})
    # age=30 → bin=10 → 10*2+5 = 25; age=50 → bin=20 → 20*2+5 = 45
    assert result["score"].to_list() == pytest.approx([25.0, 45.0])
