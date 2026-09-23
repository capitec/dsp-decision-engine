"""Acceptance test for removing the hand-unrolled N-variant closure
families this migration deletes:

    _compose_args1..16   decider2.compile.driver    (packed-step row gather)
    _compose0..8         decider2.tables.encode      (per-kind condition gather)
    _path0..6            decider2.trees.encode       (computed-feature count)

Doc 01 §4d sizes a realistic credit flow at ~400 inputs; the OLD families
raised `ValueError` past 16 (driver), 8 (tables, per condition kind) and 6
(trees, computed features). These tests build past every one of those
numbers and assert the pipeline still builds AND answers correctly — not
merely that it builds.
"""
from __future__ import annotations

import polars as pl

from decider2 import flow
from decider2.tables import AndExpression, BetweenExpression, DecisionTable, EqExpression, InExpression, ParametersConfig, table_module
from decider2.testing import assert_equivalent
from decider2.trees import (
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    PositionedNode,
    Tree,
    TreeOutput,
    UnaryGreaterThan,
    UnaryNode,
    tree_module,
)

N_WIDE = 400  # doc 01 §4d's own figure for a realistic credit flow


# ---------------------------------------------------------------------------
# Trees — decider2.compile.driver's row gather (_compose_args1..16) AND
# decider2.trees.encode's computed-feature family (_path0..6), both at once:
# N_WIDE plain features (the row-gather wall) plus more than 6 computed
# features (the tree-specific wall).
# ---------------------------------------------------------------------------


def _wide_chain_tree(n: int) -> Tree:
    """A chain of `n` comparison nodes, one distinct feature each:
    `f{i} > i`. Node `i`'s FALSE edge goes to its own leaf (`pts=i`); the
    last node's TRUE edge goes to one more leaf (`pts=n`). So a row's
    answer names exactly which node (if any) it failed at — a real,
    per-feature-position correctness check, not just "did this build"."""
    nodes: list = []
    edges: list = []
    leaf_rows: list = []
    for i in range(n):
        nodes.append(PositionedNode(
            id=f"n{i}",
            data=UnaryNode(condition=UnaryGreaterThan(feature=f"f{i}", threshold=float(i))),
        ))
        nodes.append(PositionedNode(id=f"leaf{i}", data=LeafNode(result_idx=i)))
        leaf_rows.append({"pts": float(i)})
        edges.append(MultiSourceEdge(
            source=f"n{i}", target=f"leaf{i}", data=MultiEdgeData(sourceIndex=[1]),
        ))
        if i + 1 < n:
            edges.append(MultiSourceEdge(
                source=f"n{i}", target=f"n{i + 1}", data=MultiEdgeData(sourceIndex=[0]),
            ))
    nodes.append(PositionedNode(id="leaf_final", data=LeafNode(result_idx=n)))
    leaf_rows.append({"pts": float(n)})
    edges.append(MultiSourceEdge(
        source=f"n{n - 1}", target="leaf_final", data=MultiEdgeData(sourceIndex=[0]),
    ))
    return Tree(
        name="wide",
        nodes=nodes,
        edges=edges,
        output=TreeOutput(data=leaf_rows, default={"pts": -1.0}, dtypes=[("pts", "Float64")]),
    )


def test_a_400_feature_tree_builds_and_answers_correctly(tmp_path):
    """The literal acceptance test the report asks for: a decision tree
    with 400 features (doc 01 §4d's own number) builds and answers
    correctly. `_compose_args1..16` raised ValueError at feature 17;
    this tree has 400."""
    tree = _wide_chain_tree(N_WIDE)
    built = tree_module(tree, build_dir=tmp_path)
    pipeline = flow(built.module)

    columns = {f"f{i}": [] for i in range(N_WIDE)}
    rows = []

    # Row 0: passes every node -> reaches leaf_final -> pts == N_WIDE.
    rows.append({f"f{i}": float(N_WIDE) for i in range(N_WIDE)})
    # Row 1: fails immediately (f0 > 0 is false) -> pts == 0.
    rows.append({f"f{i}": 0.0 for i in range(N_WIDE)})
    # Row 2: passes the first 250 nodes, fails node 250 -> pts == 250.
    row2 = {f"f{i}": float(N_WIDE) for i in range(N_WIDE)}
    row2["f250"] = 0.0
    rows.append(row2)
    # Row 3: fails at the very last node (399) -> pts == 399.
    row3 = {f"f{i}": float(N_WIDE) for i in range(N_WIDE)}
    row3["f399"] = 0.0
    rows.append(row3)

    for key in columns:
        columns[key] = [r[key] for r in rows]
    frame = pl.DataFrame(columns)

    out = pipeline.apply(frame, mode="fused")
    assert out["pts"].to_list() == [400.0, 0.0, 250.0, 399.0]

    # The full interpreted/stepped/fused/score ladder, exact agreement —
    # doc 05 §9.1, not merely "it built".
    assert_equivalent(pipeline, frame)


def test_more_than_six_computed_features_in_one_tree(tmp_path):
    """`_path0..6` raised ValueError past 6 computed features in one tree
    (`decider2.trees.encode._MAX_COMPUTED`, now removed). Ten computed
    expressions, each gating its own node, each on a DIFFERENT pair of
    plain columns, so a wrong slot assignment would show up as a wrong
    branch taken, not just a crash."""
    n_computed = 10
    nodes: list = []
    edges: list = []
    leaf_rows: list = []
    for i in range(n_computed):
        nodes.append(PositionedNode(
            id=f"n{i}",
            data=UnaryNode(condition=UnaryGreaterThan(
                feature={"type": "computed", "expression": f"a{i} - b{i}"},
                threshold=0.0,
            )),
        ))
        nodes.append(PositionedNode(id=f"leaf{i}", data=LeafNode(result_idx=i)))
        leaf_rows.append({"pts": float(i)})
        edges.append(MultiSourceEdge(
            source=f"n{i}", target=f"leaf{i}", data=MultiEdgeData(sourceIndex=[1]),
        ))
        if i + 1 < n_computed:
            edges.append(MultiSourceEdge(
                source=f"n{i}", target=f"n{i + 1}", data=MultiEdgeData(sourceIndex=[0]),
            ))
    nodes.append(PositionedNode(id="leaf_final", data=LeafNode(result_idx=n_computed)))
    leaf_rows.append({"pts": float(n_computed)})
    edges.append(MultiSourceEdge(
        source=f"n{n_computed - 1}", target="leaf_final", data=MultiEdgeData(sourceIndex=[0]),
    ))
    tree = Tree(
        name="many_computed",
        nodes=nodes,
        edges=edges,
        output=TreeOutput(data=leaf_rows, default={"pts": -1.0}, dtypes=[("pts", "Float64")]),
    )
    built = tree_module(tree, build_dir=tmp_path)
    pipeline = flow(built.module)

    columns = {}
    for i in range(n_computed):
        columns[f"a{i}"] = [1.0]
        columns[f"b{i}"] = [0.0]  # a_i - b_i == 1.0 > 0 for every i -> passes all -> leaf_final
    frame = pl.DataFrame(columns)
    out = pipeline.apply(frame, mode="fused")
    assert out["pts"].to_list() == [float(n_computed)]

    # Fail at computed node 7 specifically (mid-array, not an edge case).
    columns2 = dict(columns)
    columns2["a7"] = [0.0]
    columns2["b7"] = [1.0]  # a7 - b7 == -1.0, not > 0 -> fails node 7 -> pts == 7
    frame2 = pl.DataFrame(columns2)
    out2 = pipeline.apply(frame2, mode="fused")
    assert out2["pts"].to_list() == [7.0]

    assert_equivalent(pipeline, frame)
    assert_equivalent(pipeline, frame2)


# ---------------------------------------------------------------------------
# Tables — decider2.tables.encode's per-condition-kind family
# (_compose0..8, an 8-conditions-of-one-kind-per-table wall).
# ---------------------------------------------------------------------------


def test_a_table_of_comparable_width_builds_and_answers_correctly(tmp_path):
    """A table of ~400 columns, one BETWEEN condition per column, ANDed
    together — the table analogue of the 400-feature tree above.
    `_compose0..8` raised ValueError past 8 BETWEEN conditions in one
    table; this table has 400. Bounds are DISTINCT per condition
    (`[i, i + 10)`) so a stacking/indexing bug (reading condition `local`'s
    neighbour instead of its own row) would show up as a wrong match, not
    just a crash."""
    row = {}
    for i in range(N_WIDE):
        row[f"lo{i}"] = float(i)
        row[f"hi{i}"] = float(i + 10)
    row["pts"] = 1
    dtypes = {f"lo{i}": "Float64" for i in range(N_WIDE)}
    dtypes.update({f"hi{i}": "Float64" for i in range(N_WIDE)})
    dtypes["pts"] = "Int64"

    table = DecisionTable(
        name="wide_table",
        parameters=ParametersConfig(data=[row], dtypes=dtypes),
        expression=AndExpression(type="and", expressions=[
            BetweenExpression(
                type="between", variable=f"v{i}",
                lower_bound_column=f"lo{i}", upper_bound_column=f"hi{i}",
            )
            for i in range(N_WIDE)
        ]),
        outputs=["pts"],
        default=[0],
    )
    built = table_module(table, build_dir=tmp_path)
    pipeline = flow(built.module)

    # Row 0: every v_i == i + 5 -> inside [i, i+10) for every condition -> matches.
    match_row = {f"v{i}": float(i + 5) for i in range(N_WIDE)}
    # Row 1: same, except v200 is pushed WAY outside its own [200, 210)
    # band -> exactly one condition (mid-array) fails -> no match -> default.
    fail_row = dict(match_row)
    fail_row["v200"] = 9999.0

    columns = {f"v{i}": [match_row[f"v{i}"], fail_row[f"v{i}"]] for i in range(N_WIDE)}
    frame = pl.DataFrame(columns)

    out = pipeline.apply(frame, shared=built.shared, mode="fused")
    assert out["pts"].to_list() == [1, 0]

    assert_equivalent(pipeline, frame, shared=built.shared)


def test_more_than_eight_eq_and_in_conditions_in_one_table(tmp_path):
    """`_compose0..8` raised ValueError past 8 conditions of one KIND —
    tested here directly for EQ and IN (BETWEEN is covered at width above),
    20 of each, well past the old per-kind ceiling."""
    n_eq = 20
    n_in = 20
    row = {}
    for i in range(n_eq):
        row[f"e{i}"] = float(i)
    for i in range(n_in):
        row[f"s{i}"] = [float(i), float(i) + 100.0]
    row["pts"] = 1
    dtypes = {f"e{i}": "Float64" for i in range(n_eq)}
    dtypes.update({f"s{i}": "List" for i in range(n_in)})
    dtypes["pts"] = "Int64"

    expressions = [
        EqExpression(type="eq", variable=f"ev{i}", value_column=f"e{i}") for i in range(n_eq)
    ] + [
        InExpression(type="in", variable=f"iv{i}", values_column=f"s{i}") for i in range(n_in)
    ]
    table = DecisionTable(
        name="eq_in_wide",
        parameters=ParametersConfig(data=[row], dtypes=dtypes),
        expression=AndExpression(type="and", expressions=expressions),
        outputs=["pts"],
        default=[0],
    )
    built = table_module(table, build_dir=tmp_path)
    pipeline = flow(built.module)

    match_cols = {f"ev{i}": [float(i)] for i in range(n_eq)}
    match_cols.update({f"iv{i}": [float(i)] for i in range(n_in)})
    frame = pl.DataFrame(match_cols)
    out = pipeline.apply(frame, shared=built.shared, mode="fused")
    assert out["pts"].to_list() == [1]

    # Break EQ condition 15 (mid-array) specifically.
    fail_cols = dict(match_cols)
    fail_cols["ev15"] = [999.0]
    frame2 = pl.DataFrame(fail_cols)
    out2 = pipeline.apply(frame2, shared=built.shared, mode="fused")
    assert out2["pts"].to_list() == [0]

    assert_equivalent(pipeline, frame, shared=built.shared)
    assert_equivalent(pipeline, frame2, shared=built.shared)
