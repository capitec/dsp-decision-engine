"""Migration: decider 1's own tree and table fixtures, same answers.

Every config in this file is lifted from decider 1's test suite, at the path
named above each one, and every expected value is decider 1's own assertion
— not re-derived here. That is the point: the claim under test is not "this
engine is self-consistent", it is "this engine answers what decider 1
answered", and a fixture invented for decider2 could not make that claim.

Sources:
  tests/rules/test_tree_migration.py::test_v3_numerical_and_range_nodes_execute_correctly
  tests/rules/test_tree_migration.py::test_v1_tree_as_basemodule_produces_correct_output
  tests/rules/test_tree_end_to_end.py::test_nested_unary_then_cases_ranges
  tests/credit/decision_table/test_decision_table.py (all five)
"""
from __future__ import annotations

import polars as pl
import pytest

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
from decider2.testing import assert_equivalent
from decider2.trees import (
    CasesRanges,
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    Position,
    PositionedNode,
    RangeCondition,
    Tree,
    TreeOutput,
    UnaryLessThan,
    UnaryNode,
    UnaryStringMatch,
    tree_module,
)


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


def test_v3_cases_ranges_bands_exactly_as_decider1(tmp_path):
    """decider 1 `test_v3_numerical_and_range_nodes_execute_correctly`.

    Its own comment on the expected answer: "lower_inclusive: [min, max) —
    30 is start of mid".
    """
    tree = Tree(
        name="band",
        edges=[
            MultiSourceEdge(id="e0", source="root", target="leaf_low", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(id="e1", source="root", target="leaf_mid", data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(id="e2", source="root", target="leaf_high", data=MultiEdgeData(sourceIndex=[2])),
            MultiSourceEdge(id="e3", source="root", target="leaf_unk", data=MultiEdgeData(sourceIndex=[3])),
        ],
        nodes=[
            PositionedNode(id="root", position=Position(x=0, y=0), data=CasesRanges(
                feature="score",
                conditions=[
                    RangeCondition(max=30.0),
                    RangeCondition(min=30.0, max=70.0),
                    RangeCondition(min=70.0),
                ],
                end_logic="lower_inclusive",
                strict=False,
            )),
            PositionedNode(id="leaf_low", position=Position(x=-150, y=100), data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_mid", position=Position(x=0, y=100), data=LeafNode(result_idx=1)),
            PositionedNode(id="leaf_high", position=Position(x=150, y=100), data=LeafNode(result_idx=2)),
            PositionedNode(id="leaf_unk", position=Position(x=300, y=100), data=LeafNode(result_idx=-1)),
        ],
        output=TreeOutput(
            data=[{"band": "low"}, {"band": "mid"}, {"band": "high"}],
            default={"band": "unknown"},
            dtypes=[("band", "String")],
        ),
    )
    built = tree_module(tree, build_dir=tmp_path)
    frame = pl.DataFrame({"score": [10.0, 30.0, 50.0, 70.0, 90.0]})
    out = built.decode(flow(built.module).apply(frame))

    assert out["band"].to_list() == ["low", "mid", "mid", "high", "high"]


def test_v1_string_match_subtree_exactly_as_decider1(tmp_path):
    """decider 1 `test_v1_tree_as_basemodule_produces_correct_output`'s
    numeric subtree, plus the string-match subtree from the same `_V1_DICT`.

    decider 1 runs the two as separate subtrees and asserts only the numeric
    one (its root). Both are asserted here, on the same rows, because the
    string half is what exercises the hoisted-matcher path.

    `case_sensitive` is True here where decider 1's fixture sets False: doc
    05 §1.5 has no case folding inside a kernel, so the migrated document
    lower-cases its own patterns and the column is normalised in the frame
    tier. That difference is the point of the `UnsupportedInKernel` test in
    `test_trees.py` — it is refused loudly, never silently ignored.
    """
    tree = Tree(
        name="group",
        edges=[
            MultiSourceEdge(id="e1", source="num_root", target="leaf_young", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(id="e2", source="num_root", target="str_root", data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(id="e3", source="str_root", target="leaf_vip", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(id="e4", source="str_root", target="leaf_std", data=MultiEdgeData(sourceIndex=[1])),
        ],
        nodes=[
            PositionedNode(id="num_root", data=UnaryNode(
                condition=UnaryLessThan(feature="age", threshold=30.0))),
            PositionedNode(id="leaf_young", data=LeafNode(result_idx=0)),
            PositionedNode(id="str_root", data=UnaryNode(
                condition=UnaryStringMatch(feature="status", patterns=["vip"]))),
            PositionedNode(id="leaf_vip", data=LeafNode(result_idx=2)),
            PositionedNode(id="leaf_std", data=LeafNode(result_idx=3)),
        ],
        output=TreeOutput(
            data=[{"group": "young"}, {"group": "adult"}, {"group": "vip"}, {"group": "standard"}],
            default={"group": "unknown"},
            dtypes=[("group", "String")],
        ),
    )
    built = tree_module(tree, build_dir=tmp_path)
    frame = pl.DataFrame({
        "age": [20.0, 40.0, 25.0, 35.0],
        "status": ["vip", "standard", "regular", "vip"],
    })
    out = built.decode(flow(built.module).apply(frame))

    # age < 30 -> young (decider 1's assertion for rows 0 and 2);
    # otherwise the string subtree decides.
    assert out["group"].to_list() == ["young", "standard", "young", "vip"]


def test_nested_unary_then_cases_ranges_exactly_as_decider1(tmp_path):
    """decider 1 `test_nested_unary_then_cases_ranges`:

        age < 30  -> CasesRanges on score (low / mid / high)
        age >= 30 -> LeafRule("adult")

    "Verifies that the then-branch of a UnaryRule can itself be a CasesRule."
    """
    tree = Tree(
        name="nested",
        edges=[
            MultiSourceEdge(source="root", target="buckets", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="root", target="leaf_adult", data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(source="buckets", target="leaf_low", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="buckets", target="leaf_mid", data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(source="buckets", target="leaf_high", data=MultiEdgeData(sourceIndex=[2])),
            MultiSourceEdge(source="buckets", target="leaf_none", data=MultiEdgeData(sourceIndex=[3])),
        ],
        nodes=[
            PositionedNode(id="root", data=UnaryNode(
                condition=UnaryLessThan(feature="age", threshold=30.0))),
            PositionedNode(id="buckets", data=CasesRanges(
                feature="score",
                conditions=[
                    RangeCondition(max=40.0),
                    RangeCondition(min=40.0, max=70.0),
                    RangeCondition(min=70.0),
                ],
                end_logic="lower_inclusive",
                strict=False,
            )),
            PositionedNode(id="leaf_low", data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_mid", data=LeafNode(result_idx=1)),
            PositionedNode(id="leaf_high", data=LeafNode(result_idx=2)),
            PositionedNode(id="leaf_none", data=LeafNode(result_idx=-1)),
            PositionedNode(id="leaf_adult", data=LeafNode(result_idx=3)),
        ],
        output=TreeOutput(
            data=[{"r": "low"}, {"r": "mid"}, {"r": "high"}, {"r": "adult"}],
            default={"r": "default"},
            dtypes=[("r", "String")],
        ),
    )
    built = tree_module(tree, build_dir=tmp_path)
    frame = pl.DataFrame({
        "age": [25.0, 25.0, 25.0, 45.0],
        "score": [10.0, 50.0, 90.0, 10.0],
    })
    out = built.decode(flow(built.module).apply(frame))

    assert out["r"].to_list() == ["low", "mid", "high", "adult"]


# ---------------------------------------------------------------------------
# Decision tables — all five of decider 1's integration tests
# ---------------------------------------------------------------------------


def _run_table(built, frame: pl.DataFrame) -> pl.DataFrame:
    return built.decode(flow(built.module).apply(frame, shared=built.shared))


def test_between_maps_ranges_to_output_labels(tmp_path):
    """decider 1: "BetweenExpression is lower_inclusive by default: [lo, hi)"."""
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
            type="between", variable="score", lower_bound_column="lo", upper_bound_column="hi"
        ),
        outputs=["band"],
        default=["other"],
    )
    built = table_module(table, build_dir=tmp_path)
    out = _run_table(built, pl.DataFrame({"score": [10.0, 30.0, 70.0, 90.0]}))

    assert out["band"].to_list() == ["low", "mid", "high", "high"]


def test_between_default_when_outside_all_ranges(tmp_path):
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
    out = _run_table(built, pl.DataFrame({"v": [50.0, 5.0, 200.0]}))

    assert out["label"].to_list() == ["in_range", "out_of_range", "out_of_range"]


def test_in_expression_categorical_lookup(tmp_path):
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
    out = _run_table(built, pl.DataFrame({"code": ["A", "C", "X"]}))

    assert out["tier"].to_list() == ["premium", "standard", "unknown"]


def test_and_expression_requires_all_conditions(tmp_path):
    table = DecisionTable(
        name="dt",
        parameters=ParametersConfig(
            data=[{"age_lo": 18.0, "age_hi": 65.0, "flag": True, "outcome": "eligible"}],
            dtypes={
                "age_lo": "Float64", "age_hi": "Float64",
                "flag": "Boolean", "outcome": "String",
            },
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
    out = _run_table(built, pl.DataFrame({
        "age": [30.0, 17.0, 40.0, 70.0],
        "verified": [True, True, False, True],
    }))

    assert out["outcome"].to_list() == [
        "eligible", "ineligible", "ineligible", "ineligible",
    ]


def test_multiple_output_columns_all_populated(tmp_path):
    """decider 1: "All declared output columns are present"."""
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
            type="between", variable="v", lower_bound_column="lo", upper_bound_column="hi"
        ),
        outputs=["label", "pts"],
        default=["other", 0],
    )
    built = table_module(table, build_dir=tmp_path)
    out = _run_table(built, pl.DataFrame({"v": [20.0, 80.0, 200.0]}))

    assert out["label"].to_list() == ["low", "high", "high"]
    assert out["pts"].to_list() == [10.0, 20.0, 20.0]


# ---------------------------------------------------------------------------
# The equivalence ladder, on a migrated tree and a migrated table
# ---------------------------------------------------------------------------


def test_a_migrated_tree_agrees_across_all_three_modes(tmp_path):
    """doc 02 §3.1's core correctness test, on decider 1's own config."""
    tree = Tree(
        name="band",
        edges=[
            MultiSourceEdge(source="root", target="leaf_low", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="root", target="leaf_mid", data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(source="root", target="leaf_high", data=MultiEdgeData(sourceIndex=[2])),
        ],
        nodes=[
            PositionedNode(id="root", data=CasesRanges(
                feature="score",
                conditions=[
                    RangeCondition(max=30.0),
                    RangeCondition(min=30.0, max=70.0),
                    RangeCondition(min=70.0),
                ],
                strict=False,
            )),
            PositionedNode(id="leaf_low", data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_mid", data=LeafNode(result_idx=1)),
            PositionedNode(id="leaf_high", data=LeafNode(result_idx=2)),
        ],
        output=TreeOutput(
            data=[{"pts": 1.0}, {"pts": 2.0}, {"pts": 3.0}],
            default={"pts": 0.0},
            dtypes=[("pts", "Float64")],
        ),
    )
    built = tree_module(tree, build_dir=tmp_path)
    frame = pl.DataFrame({"score": [10.0, 30.0, 50.0, 70.0, 90.0, -1.0]})

    assert_equivalent(flow(built.module), frame)


def test_a_migrated_table_agrees_across_all_three_modes(tmp_path):
    table = DecisionTable(
        name="dt",
        parameters=ParametersConfig(
            data=[
                {"lo": None, "hi": 50.0, "pts": 10},
                {"lo": 50.0, "hi": None, "pts": 20},
            ],
            dtypes={"lo": "Float64", "hi": "Float64", "pts": "Int64"},
        ),
        expression=BetweenExpression(
            type="between", variable="v", lower_bound_column="lo", upper_bound_column="hi"
        ),
        outputs=["pts"],
        default=[0],
    )
    built = table_module(table, build_dir=tmp_path)
    frame = pl.DataFrame({"v": [20.0, 80.0, 50.0, 200.0]})

    assert_equivalent(flow(built.module), frame, shared=built.shared)
