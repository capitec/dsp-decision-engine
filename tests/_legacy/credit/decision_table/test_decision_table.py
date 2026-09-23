"""
Integration tests for DecisionTableModule.

Exercises range lookups, categorical matching, boolean gates,
compound (AND) conditions, multi-output rows, and default fallback.
"""

import polars as pl

from decider.modules.credit.decision_table import DecisionTableModule, ParametersConfig
from decider.modules.credit.decision_table.config import (
    AndExpression,
    BetweenExpression,
    InExpression,
    IsTrueExpression,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(module: DecisionTableModule, df: pl.DataFrame) -> pl.DataFrame:
    result = module({"input": df})
    return result.select(pl.col("output").struct.unnest())


def _between_module(rows, dtypes, variable, lo_col, hi_col, outputs, default, **kwargs):
    return DecisionTableModule(
        name="dt",
        parameters=ParametersConfig(data=rows, dtypes=dtypes),
        expression=BetweenExpression(
            type="between",
            variable=variable,
            lower_bound_column=lo_col,
            upper_bound_column=hi_col,
            **kwargs,
        ),
        outputs=outputs,
        default=default,
    )


# ── tests ─────────────────────────────────────────────────────────────────────

def test_between_maps_ranges_to_output_labels():
    """BetweenExpression buckets a numeric column into labelled output rows."""
    m = _between_module(
        rows=[
            {"lo": None, "hi": 30.0, "band": "low"},
            {"lo": 30.0, "hi": 70.0, "band": "mid"},
            {"lo": 70.0, "hi": None, "band": "high"},
        ],
        dtypes={"lo": "Float64", "hi": "Float64", "band": "String"},
        variable="score", lo_col="lo", hi_col="hi",
        outputs=["band"], default=["other"],
    )
    df = pl.DataFrame({"score": [10.0, 30.0, 70.0, 90.0]})
    result = _run(m, df)
    # BetweenExpression is lower_inclusive by default: [lo, hi)
    assert result["band"].to_list() == ["low", "mid", "high", "high"]


def test_between_default_when_outside_all_ranges():
    """Rows that fall outside every range use the configured default."""
    m = _between_module(
        rows=[{"lo": 10.0, "hi": 90.0, "label": "in_range"}],
        dtypes={"lo": "Float64", "hi": "Float64", "label": "String"},
        variable="v", lo_col="lo", hi_col="hi", allow_gaps=True,
        outputs=["label"], default=["out_of_range"],
    )
    df = pl.DataFrame({"v": [50.0, 5.0, 200.0]})
    result = _run(m, df)
    assert result["label"].to_list() == ["in_range", "out_of_range", "out_of_range"]


def test_in_expression_categorical_lookup():
    """InExpression routes rows based on whether a column value is in a list."""
    m = DecisionTableModule(
        name="dt",
        parameters=ParametersConfig(
            data=[
                {"vals": ["A", "B"], "tier": "premium"},
                {"vals": ["C", "D"], "tier": "standard"},
            ],
            dtypes=[("vals", {"type": "List", "inner": "String"}), ("tier", "String")],
        ),
        expression=InExpression(type="in", variable="code", values_column="vals"),
        outputs=["tier"],
        default=["unknown"],
    )
    df = pl.DataFrame({"code": ["A", "C", "X"]})
    result = _run(m, df)
    assert result["tier"].to_list() == ["premium", "standard", "unknown"]


def test_and_expression_requires_all_conditions():
    """AndExpression only matches when every sub-condition holds simultaneously."""
    m = DecisionTableModule(
        name="dt",
        parameters=ParametersConfig(
            data=[{"age_lo": 18.0, "age_hi": 65.0, "flag": True, "outcome": "eligible"}],
            dtypes={"age_lo": "Float64", "age_hi": "Float64", "flag": "Boolean", "outcome": "String"},
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
    result = _run(m, df)
    assert result["outcome"].to_list() == ["eligible", "ineligible", "ineligible", "ineligible"]


def test_multiple_output_columns_all_populated():
    """All declared output columns are present in the returned struct."""
    m = DecisionTableModule(
        name="dt",
        parameters=ParametersConfig(
            data=[
                {"lo": None,  "hi": 50.0, "label": "low",  "pts": 10},
                {"lo": 50.0,  "hi": None, "label": "high", "pts": 20},
            ],
            dtypes={"lo": "Float64", "hi": "Float64", "label": "String", "pts": "Int64"},
        ),
        expression=BetweenExpression(
            type="between", variable="v", lower_bound_column="lo", upper_bound_column="hi",
        ),
        outputs=["label", "pts"],
        default=["other", 0],
    )
    df = pl.DataFrame({"v": [20.0, 80.0, 200.0]})
    result = _run(m, df)
    assert result["label"].to_list() == ["low", "high", "high"]
    assert result["pts"].to_list() == [10, 20, 20]
