"""Ported from decider 1:
`tests/credit/decision_table/test_decision_table.py` (5 tests, all of
them).

Migration-conformance suite — see `decider2/tests/PORTED.md`. This is the
most mechanical of the five ported files: the decision-table vocabulary
(`DecisionTableModule` -> `DecisionTable`, `ParametersConfig`,
`AndExpression`/`BetweenExpression`/`InExpression`/`IsTrueExpression`) was
kept unchanged across the migration (`decider2/tables/schema.py`'s own
docstring: "Every type name, field name and default below is decider 1's
... A `DecisionTableModule` config that decider 1 accepts parses here with
only its wrapper renamed"), so every one of decider 1's five configs
constructs a `decider2.tables.DecisionTable` with the identical keyword
arguments and produces the identical answer.

Two mechanical differences, present in every test:
  * decider 1 runs a module directly (`module({"input": df})`) and unnests
    an `"output"` struct column. decider2 compiles first
    (`table_module(table, build_dir=tmp_path)`) and the flat output columns
    are already top-level (doc 03 §7) — `built.decode(flow(built.module).
    apply(frame, shared=built.shared))` in place of `module({"input": df})`
    + `.struct.unnest()`.
  * A string-valued output column comes back through `.decode()` (doc
    08 §3.4's row-array shape can't carry a string out of the kernel
    directly — see `tables/build.py`'s `TableModule.decode` docstring).

This file also duplicates coverage already present in this session's
`test_trees_migration.py` (all five of decider 1's table tests are ported
there too, under different names, alongside tree fixtures). That overlap
is intentional, not an oversight: this task's target file list
(`test_tables_ported.py`) asks for a complete, independent port of this
one decider 1 source file, not a dedupe against work already done in a
prior session.
"""
from __future__ import annotations

import polars as pl

from decider2 import flow
from decider2.tables import (
    AndExpression,
    BetweenExpression,
    DecisionTable,
    InExpression,
    IsTrueExpression,
    ParametersConfig,
    table_module,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(built, frame: pl.DataFrame) -> pl.DataFrame:
    return built.decode(flow(built.module).apply(frame, shared=built.shared))


# ---------------------------------------------------------------------------
# Tests — decider 1's five, verbatim configs and expected answers
# ---------------------------------------------------------------------------


def test_between_maps_ranges_to_output_labels(tmp_path):
    """decider 1: `BetweenExpression` buckets a numeric column into
    labelled output rows. decider 1's own comment: "BetweenExpression is
    lower_inclusive by default: [lo, hi)"."""
    table = DecisionTable(
        name="dt",
        parameters=ParametersConfig(
            data=[
                {"lo": None, "hi": 30.0, "band": "low"},
                {"lo": 30.0, "hi": 70.0, "band": "mid"},
                {"lo": 70.0, "hi": None, "band": "high"},
            ],
            dtypes={"lo": "Float64", "hi": "Float64", "band": "String"},
        ),
        expression=BetweenExpression(
            type="between", variable="score", lower_bound_column="lo", upper_bound_column="hi",
        ),
        outputs=["band"],
        default=["other"],
    )
    built = table_module(table, build_dir=tmp_path)
    out = _run(built, pl.DataFrame({"score": [10.0, 30.0, 70.0, 90.0]}))

    assert out["band"].to_list() == ["low", "mid", "high", "high"]


def test_between_default_when_outside_all_ranges(tmp_path):
    """decider 1: rows that fall outside every range use the configured
    default."""
    table = DecisionTable(
        name="dt",
        parameters=ParametersConfig(
            data=[{"lo": 10.0, "hi": 90.0, "label": "in_range"}],
            dtypes={"lo": "Float64", "hi": "Float64", "label": "String"},
        ),
        expression=BetweenExpression(
            type="between", variable="v", lower_bound_column="lo",
            upper_bound_column="hi", allow_gaps=True,
        ),
        outputs=["label"],
        default=["out_of_range"],
    )
    built = table_module(table, build_dir=tmp_path)
    out = _run(built, pl.DataFrame({"v": [50.0, 5.0, 200.0]}))

    assert out["label"].to_list() == ["in_range", "out_of_range", "out_of_range"]


def test_in_expression_categorical_lookup(tmp_path):
    """decider 1: `InExpression` routes rows based on whether a column
    value is in a list."""
    table = DecisionTable(
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
    built = table_module(table, build_dir=tmp_path)
    out = _run(built, pl.DataFrame({"code": ["A", "C", "X"]}))

    assert out["tier"].to_list() == ["premium", "standard", "unknown"]


def test_and_expression_requires_all_conditions(tmp_path):
    """decider 1: `AndExpression` only matches when every sub-condition
    holds simultaneously."""
    table = DecisionTable(
        name="dt",
        parameters=ParametersConfig(
            data=[{"age_lo": 18.0, "age_hi": 65.0, "flag": True, "outcome": "eligible"}],
            dtypes={"age_lo": "Float64", "age_hi": "Float64", "flag": "Boolean", "outcome": "String"},
        ),
        expression=AndExpression(
            type="and",
            expressions=[
                BetweenExpression(
                    type="between", variable="age", lower_bound_column="age_lo",
                    upper_bound_column="age_hi", allow_gaps=True,
                ),
                IsTrueExpression(type="is_true", variable="verified"),
            ],
        ),
        outputs=["outcome"],
        default=["ineligible"],
    )
    built = table_module(table, build_dir=tmp_path)
    out = _run(built, pl.DataFrame({
        "age": [30.0, 17.0, 40.0, 70.0],
        "verified": [True, True, False, True],
    }))

    assert out["outcome"].to_list() == ["eligible", "ineligible", "ineligible", "ineligible"]


def test_multiple_output_columns_all_populated(tmp_path):
    """decider 1: all declared output columns are present in the returned
    struct (here: flat columns)."""
    table = DecisionTable(
        name="dt",
        parameters=ParametersConfig(
            data=[
                {"lo": None, "hi": 50.0, "label": "low", "pts": 10},
                {"lo": 50.0, "hi": None, "label": "high", "pts": 20},
            ],
            dtypes={"lo": "Float64", "hi": "Float64", "label": "String", "pts": "Int64"},
        ),
        expression=BetweenExpression(
            type="between", variable="v", lower_bound_column="lo", upper_bound_column="hi",
        ),
        outputs=["label", "pts"],
        default=["other", 0],
    )
    built = table_module(table, build_dir=tmp_path)
    out = _run(built, pl.DataFrame({"v": [20.0, 80.0, 200.0]}))

    assert out["label"].to_list() == ["low", "high", "high"]
    assert out["pts"].to_list() == [10, 20, 20]
