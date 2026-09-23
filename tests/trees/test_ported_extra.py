"""Null numeric inputs are errors, and a param is one value for the whole call."""
import polars as pl
import pytest

from decider.engine import Engine
from decider.exceptions import MissingInputError
from decider.steps.trees import TreeConfig

MODES = ("interpreted", "stepped", "fused")


def _two_leaf(condition: dict, name: str, output: dict) -> TreeConfig:
    return TreeConfig(name=name, tree={
        "nodes": [{"id": "root", "data": {"type": "unary", "condition": condition}},
                  {"id": "yes", "data": {"type": "leaf", "result_idx": 0}}],
        "edges": [{"source": "root", "target": "yes", "data": {"sourceIndex": 0}}],
        "output": output,
    })


@pytest.mark.parametrize("mode", MODES)
def test_a_null_numeric_feature_is_a_missing_input_not_a_silent_otherwise(mode):
    tree = _two_leaf({"op": "between", "feature": "v", "min": 0.0, "max": 100.0}, "routed",
                     {"data": [{"r": "match"}], "default": {"r": "no"}, "dtypes": [["r", "String"]]})
    with pytest.raises(MissingInputError, match="'v'"):
        Engine().bind(tree, mode=mode).run(pl.DataFrame({"v": pl.Series([50.0, None], dtype=pl.Float64)}))


def test_a_param_threshold_is_one_value_per_call_not_per_row(run):
    # decider_old read a per-row `parameters` column; a params document sets one value for every row.
    tree = TreeConfig(name="t3", tree={
        "nodes": [{"id": "root", "data": {"type": "unary", "condition": {
                      "op": "<", "feature": "score", "threshold": {"param": "thresh", "default": 50.0}}}},
                  {"id": "low", "data": {"type": "leaf", "result_idx": 0}},
                  {"id": "high", "data": {"type": "leaf", "result_idx": 1}}],
        "edges": [{"source": "root", "target": "low", "data": {"sourceIndex": 0}},
                  {"source": "root", "target": "high", "data": {"sourceIndex": 1}}],
        "output": {"data": [{"r": "low"}, {"r": "high"}], "dtypes": [["r", "String"]]},
    })
    frame = pl.DataFrame({"score": [30.0, 70.0]})
    assert run(tree, frame, {"t3": {"thresh": 20.0}})["r"].to_list() == ["high", "high"]
    assert run(tree, frame, {"t3": {"thresh": 80.0}})["r"].to_list() == ["low", "low"]
