"""
Integration tests for DecisionTableModule and ScoreCard.

Mirrors the style of tests/rules/test_parameters.py — call the module with
{"input": df} and assert on the resulting DataFrame columns.
"""

import polars as pl
import pytest

from decider.modules.credit.decision_table import DecisionTableModule, ParametersConfig
from decider.modules.credit.decision_table.config import (
    BetweenExpression,
    InExpression,
    IsTrueExpression,
    AndExpression,
)
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

def _params(rows: list, dtypes: dict) -> ParametersConfig:
    return ParametersConfig(data=rows, dtypes=dtypes)


def _run_dt(module: DecisionTableModule, df: pl.DataFrame) -> pl.DataFrame:
    result = module({"input": df})
    return result.select(pl.col("output").struct.unnest())


# ─────────────────────────────────────────────────────────────────────────────
# DecisionTableModule
# ─────────────────────────────────────────────────────────────────────────────

def test_decision_table_between_maps_ranges_to_outputs():
    """BetweenExpression maps numeric ranges to labelled output rows."""
    m = DecisionTableModule(
        name="dt",
        parameters=_params(
            [
                {"lo": None, "hi": 30.0, "band": "low"},
                {"lo": 30.0, "hi": 70.0, "band": "mid"},
                {"lo": 70.0, "hi": None, "band": "high"},
            ],
            dtypes={"lo": "Float64", "hi": "Float64", "band": "String"},
        ),
        expression=BetweenExpression(
            type="between",
            variable="score",
            lower_bound_column="lo",
            upper_bound_column="hi",
        ),
        outputs=["band"],
        default=["other"],
    )
    df = pl.DataFrame({"score": [10.0, 50.0, 90.0, -5.0]})
    result = _run_dt(m, df)
    assert result["band"].to_list() == ["low", "mid", "high", "low"]


def test_decision_table_default_when_no_row_matches():
    """Rows that fall outside all ranges use the default output."""
    m = DecisionTableModule(
        name="dt",
        parameters=_params(
            [{"lo": 0.0, "hi": 100.0, "label": "in_range"}],
            dtypes={"lo": "Float64", "hi": "Float64", "label": "String"},
        ),
        expression=BetweenExpression(
            type="between", variable="v", lower_bound_column="lo", upper_bound_column="hi",
            allow_gaps=True,
        ),
        outputs=["label"],
        default=["out_of_range"],
    )
    df = pl.DataFrame({"v": [50.0, 200.0, -10.0]})
    result = _run_dt(m, df)
    assert result["label"].to_list() == ["in_range", "out_of_range", "out_of_range"]


def test_decision_table_in_expression_categorical_lookup():
    """InExpression matches rows where a column value is in a list."""
    m = DecisionTableModule(
        name="dt",
        parameters=_params(
            [
                {"vals": ["A", "B"], "tier": "premium"},
                {"vals": ["C", "D"], "tier": "standard"},
            ],
            dtypes={"vals": "List[str]", "tier": "String"},
        ),
        expression=InExpression(type="in", variable="code", values_column="vals"),
        outputs=["tier"],
        default=["unknown"],
    )
    df = pl.DataFrame({"code": ["A", "C", "X"]})
    result = _run_dt(m, df)
    assert result["tier"].to_list() == ["premium", "standard", "unknown"]


def test_decision_table_and_expression_multiple_variables():
    """AndExpression combines two conditions — both must hold for a row to match."""
    m = DecisionTableModule(
        name="dt",
        parameters=_params(
            [
                {"age_lo": 18.0, "age_hi": 65.0, "flag": True, "outcome": "eligible"},
            ],
            dtypes={"age_lo": "Float64", "age_hi": "Float64", "flag": "bool", "outcome": "String"},
        ),
        expression=AndExpression(
            type="and",
            expressions=[
                BetweenExpression(
                    type="between",
                    variable="age",
                    lower_bound_column="age_lo",
                    upper_bound_column="age_hi",
                    allow_gaps=True,
                ),
                IsTrueExpression(type="is_true", variable="verified"),
            ],
        ),
        outputs=["outcome"],
        default=["ineligible"],
    )
    df = pl.DataFrame({
        "age":      [30.0,  17.0,  40.0,  70.0],
        "verified": [True,  True,  False, True],
    })
    result = _run_dt(m, df)
    assert result["outcome"].to_list() == ["eligible", "ineligible", "ineligible", "ineligible"]


def test_decision_table_multiple_output_columns():
    """The module returns all declared output columns in the output struct."""
    m = DecisionTableModule(
        name="dt",
        parameters=_params(
            [
                {"lo": None, "hi": 50.0, "label": "low",  "pts": 10},
                {"lo": 50.0, "hi": None, "label": "high", "pts": 20},
            ],
            dtypes={"lo": "Float64", "hi": "Float64", "label": "String", "pts": "int"},
        ),
        expression=BetweenExpression(
            type="between", variable="v", lower_bound_column="lo", upper_bound_column="hi",
        ),
        outputs=["label", "pts"],
        default=["other", 0],
    )
    df = pl.DataFrame({"v": [20.0, 80.0, 200.0]})
    result = _run_dt(m, df)
    assert result["label"].to_list() == ["low", "high", "other"]
    assert result["pts"].to_list() == [10, 20, 0]


# ─────────────────────────────────────────────────────────────────────────────
# ScoreCard
# ─────────────────────────────────────────────────────────────────────────────

def _sc(variables, output_name="score") -> ScoreCard:
    return ScoreCard(name="sc", variables=variables, output_name=output_name)


def test_scorecard_single_variable_bound_bins():
    """A single ScoredVariable with BoundBins contributes its bin value to the score."""
    sc = _sc([
        ScoredVariable(
            type="scored",
            variable_name="age",
            bins=[
                BoundBin(value=5.0, upper_bound=25.0),
                BoundBin(value=10.0, lower_bound=25.0, upper_bound=60.0),
                BoundBin(value=15.0, lower_bound=60.0),
            ],
            default=DefaultBin(value=0.0),
        )
    ])
    df = pl.DataFrame({"age": [20.0, 40.0, 70.0]})
    result = sc({"input": df})
    assert result["score"].to_list() == pytest.approx([5.0, 10.0, 15.0])


def test_scorecard_multiple_variables_sum():
    """Multiple variables each contribute independently; score is their sum."""
    sc = _sc([
        ScoredVariable(
            type="scored",
            variable_name="age",
            bins=[BoundBin(value=10.0, upper_bound=40.0), BoundBin(value=20.0, lower_bound=40.0)],
            default=DefaultBin(value=0.0),
        ),
        ScoredVariable(
            type="scored",
            variable_name="income",
            bins=[BoundBin(value=5.0, upper_bound=50_000.0), BoundBin(value=15.0, lower_bound=50_000.0)],
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


def test_scorecard_value_bin_takes_priority_over_bound_bin():
    """ValuesBin matches override BoundBin ranges when both could apply."""
    sc = _sc([
        ScoredVariable(
            type="scored",
            variable_name="status",
            bins=[
                ValuesBin(value=50.0, items=["vip"]),
                BoundBin(value=10.0, upper_bound=100.0),
            ],
            default=DefaultBin(value=0.0),
            strict=False,
        )
    ])
    df = pl.DataFrame({"status": ["vip", "standard", "unknown"]})
    result = sc({"input": df})
    # "vip" → ValuesBin=50; "standard" is a string so BoundBin won't match; → default=0
    assert result["score"][0] == pytest.approx(50.0)
    assert result["score"][1] == pytest.approx(0.0)


def test_scorecard_default_bin_used_when_no_bin_matches():
    """Rows that fall outside all bins get the default value."""
    sc = _sc([
        ScoredVariable(
            type="scored",
            variable_name="risk",
            bins=[BoundBin(value=100.0, lower_bound=0.0, upper_bound=1.0)],
            default=DefaultBin(value=-999.0),
            strict=False,
        )
    ])
    df = pl.DataFrame({"risk": [0.5, 5.0, -1.0]})
    result = sc({"input": df})
    assert result["score"].to_list() == pytest.approx([100.0, -999.0, -999.0])


def test_scorecard_constant_score_adds_fixed_offset():
    """ConstantScore adds a fixed value to every row regardless of input."""
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


def test_scorecard_adjusted_variable_applies_scale_and_offset():
    """AdjustedVariable wraps a ScoredVariable and applies scale/offset before summing."""
    inner = ScoredVariable(
        type="scored",
        variable_name="age",
        bins=[BoundBin(value=10.0, upper_bound=40.0), BoundBin(value=20.0, lower_bound=40.0)],
        default=DefaultBin(value=0.0),
    )
    sc = _sc([
        AdjustedVariable(
            type="adjusted",
            variable=inner,
            scale=2.0,
            offset=5.0,
        )
    ])
    df = pl.DataFrame({"age": [30.0, 50.0]})
    result = sc({"input": df})
    # age=30 → bin=10, adjusted = 10*2 + 5 = 25
    # age=50 → bin=20, adjusted = 20*2 + 5 = 45
    assert result["score"].to_list() == pytest.approx([25.0, 45.0])
