"""BANDS -> a decider2 DecisionTable, compiled into a one-table Pipeline."""
from __future__ import annotations

from decider2 import flow
from decider2.tables import BetweenExpression, BoundMode, DecisionTable, ParametersConfig, table_module

from rules import BANDS


def build_table() -> DecisionTable:
    rows = [
        {"lo": lo, "hi": hi, "tier_code": code, "limit": limit}
        for lo, hi, code, _name, limit in BANDS
    ]
    return DecisionTable(
        name="income_band",
        parameters=ParametersConfig(
            data=rows,
            dtypes={"lo": "Float64", "hi": "Float64", "tier_code": "Int64", "limit": "Float64"},
        ),
        expression=BetweenExpression(
            type="between",
            variable="income",
            lower_bound_column="lo",
            upper_bound_column="hi",
            mode=BoundMode.lower_inclusive,
        ),
        outputs=["tier_code", "limit"],
        default=[-1, 0.0],
    )


def build(build_dir):
    """Returns (pipeline, shared). Compiles the numba kernel (build_dir is
    the on-disk cache directory the caller controls, for cold/warm control)."""
    table = build_table()
    tm = table_module(table, build_dir=build_dir)
    pipeline = flow(tm.module)
    return pipeline, tm.shared
