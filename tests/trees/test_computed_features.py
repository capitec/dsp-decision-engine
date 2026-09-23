"""Computed features: an expression over columns, evaluated as data by the compiled walker."""
import polars as pl
import pytest

from decider import engine
from decider.engine import Engine
from decider.engine.compile import kernel
from decider.steps.expr import parse
from decider.steps.trees import TreeConfig, walker

FRAME = pl.DataFrame({"x": [25.0, 15.0], "y": [10.0, 10.0]})


def two_leaf(condition: dict, name: str = "risk") -> TreeConfig:
    return TreeConfig(name=name, tree={
        "nodes": [{"id": "root", "data": {"type": "unary", "condition": condition}},
                  {"id": "hi", "data": {"type": "leaf", "result_idx": 0}},
                  {"id": "lo", "data": {"type": "leaf", "result_idx": 1}}],
        "edges": [{"source": "root", "target": "hi", "data": {"sourceIndex": 0}},
                  {"source": "root", "target": "lo", "data": {"sourceIndex": 1}}],
        "output": {"data": [{"r": 1.0}, {"r": 0.0}], "default": {"r": -1.0}, "dtypes": [["r", "Float64"]]},
    })


def computed(expression: str) -> dict:
    return {"type": "computed", "expression": expression}


def test_a_computed_feature_used_as_a_tree_condition(run):
    tree = two_leaf({"op": ">", "feature": computed("x - y"), "threshold": 10.0})
    # 25 - 10 = 15 > 10: hi; 15 - 10 = 5: lo.
    assert run(tree, FRAME)["r"].to_list() == [1.0, 0.0]


def test_a_whole_condition_as_one_computed_boolean(run):
    tree = two_leaf({"op": "is_true", "feature": computed("x - y > 10")})
    assert run(tree, FRAME)["r"].to_list() == [1.0, 0.0]


def test_a_computed_features_columns_are_the_tree_inputs():
    tree = two_leaf({"op": ">", "feature": computed("x - y"), "threshold": 10.0})
    assert sorted(i.name for i in engine.to_ir(tree).inputs) == ["x", "y"]


@pytest.mark.parametrize("expression, x, y", [
    ("min(x, y, 3) + max(x, y) * 2", 4.0, 7.0),
    ("abs(x - y) ** 2 // 3 % 5", 4.0, 7.5),
    ("-(x / y) + (not x > y) - (x > 1 and y < 1) + (x < 1 or y > 1)", 2.0, 8.0),
    ("(x != y) + (x == y) * 10 + (x <= y) * 100 + (x >= y) * 1000", 3.0, 3.0),
])
def test_every_operator_agrees_between_the_walkers(run, expression, x, y):
    expected = parse(expression).evaluate({"x": x, "y": y})
    tree = two_leaf({"op": "==", "feature": computed(expression), "threshold": float(expected)})
    assert run(tree, pl.DataFrame({"x": [x], "y": [y]}))["r"].to_list() == [1.0]


def test_editing_a_literal_inside_an_expression_never_recompiles():
    frame = pl.DataFrame({"x": [4.0]})
    Engine().bind(two_leaf({"op": ">", "feature": computed("x * 2"), "threshold": 10.0}), mode="fused").run(frame)
    before = len(walker.walk.signatures), len(kernel._KERNELS)
    answers = [Engine().bind(two_leaf({"op": ">", "feature": computed(f"x * {k}"), "threshold": 10.0}),
                             mode="fused").run(frame)["r"].to_list() for k in (2.0, 100.0, 0.1, 3, 4, 5, 6, 7)]
    assert (len(walker.walk.signatures), len(kernel._KERNELS)) == before
    assert answers[0] != answers[1]
