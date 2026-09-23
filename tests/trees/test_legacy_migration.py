"""Older tree formats: v1 documents are refused with the version named; their logic runs as v3 or flat rules."""
import polars as pl
import pydantic
import pytest

from decider.steps.trees import TreeConfig

V1 = {
    "formatVersion": 1,
    "features": ["age", "status"],
    "edges": [{"id": "e1", "source": "num_root", "target": "leaf_young", "data": {"sourceIndex": [0]}},
              {"id": "e2", "source": "num_root", "target": "leaf_adult", "data": {"sourceIndex": [1]}}],
    "nodes": [
        {"id": "num_root", "position": {"x": 0, "y": 0}, "data": {
            "node_type": "numerical_test_node", "split_feature_id": 0, "comparison_op": "<", "threshold": 30.0}},
        {"id": "leaf_young", "data": {"node_type": "leaf", "output_data": {"group": "young"}}},
        {"id": "leaf_adult", "data": {"node_type": "leaf", "output_data": {"group": "adult"}}},
    ],
    "subtrees": [{"rootNodeId": "num_root", "name": "age_split", "order": 0}],
}

GROUPS = {"data": [{"group": "young"}, {"group": "adult"}], "dtypes": [["group", "String"]]}
AGES = pl.DataFrame({"age": [20.0, 40.0, 25.0, 35.0]})


@pytest.mark.parametrize("doc", [V1, {**V1, "formatVersion": 2}, {k: v for k, v in V1.items() if k != "nodes"}])
def test_a_v1_or_v2_tree_is_refused_naming_its_version(doc):
    version = f"v{doc['formatVersion']}"
    with pytest.raises(pydantic.ValidationError, match=f"{version} tree documents are deprecated"):
        TreeConfig(name="t", tree=doc)


def test_the_v1_age_split_runs_the_same_as_a_v3_tree_and_as_flat_rules(run):
    v3 = TreeConfig(name="t", tree={
        "nodes": [{"id": "num_root", "data": {"type": "unary", "condition": {"op": "<", "feature": "age",
                                                                              "threshold": 30.0}}},
                  {"id": "leaf_young", "data": {"type": "leaf", "result_idx": 0}},
                  {"id": "leaf_adult", "data": {"type": "leaf", "result_idx": 1}}],
        "edges": [{"source": "num_root", "target": "leaf_young", "data": {"sourceIndex": [0]}},
                  {"source": "num_root", "target": "leaf_adult", "data": {"sourceIndex": [1]}}],
        "output": GROUPS,
    })
    flat = TreeConfig(name="t", tree={"type": "flat_rule", "output": GROUPS, "rule": {"rule": {
        "type": "unary", "condition": {"op": "<", "feature": "age", "threshold": 30.0},
        "then": {"type": "leaf", "result_idx": 0}, "otherwise": {"type": "leaf", "result_idx": 1}}}})
    out = run(v3, AGES)
    assert out["group"].to_list() == ["young", "adult", "young", "adult"]
    assert out.equals(run(flat, AGES))


def test_v3_range_cases_band_lower_inclusive(run):
    tree = TreeConfig(name="t", tree={
        "nodes": [
            {"id": "root", "position": {"x": 0, "y": 0}, "data": {
                "type": "cases", "op": "ranges", "feature": "score", "end_logic": "lower_inclusive", "strict": False,
                "conditions": [{"max": 30.0}, {"min": 30.0, "max": 70.0}, {"min": 70.0}]}},
            {"id": "leaf_low", "position": {"x": -150, "y": 100}, "data": {"type": "leaf", "result_idx": 0}},
            {"id": "leaf_mid", "position": {"x": 0, "y": 100}, "data": {"type": "leaf", "result_idx": 1}},
            {"id": "leaf_high", "position": {"x": 150, "y": 100}, "data": {"type": "leaf", "result_idx": 2}},
            {"id": "leaf_unk", "position": {"x": 300, "y": 100}, "data": {"type": "leaf", "result_idx": -1}},
        ],
        "edges": [{"id": f"e{i}", "source": "root", "target": t, "data": {"sourceIndex": [i]}}
                  for i, t in enumerate(("leaf_low", "leaf_mid", "leaf_high", "leaf_unk"))],
        "output": {"data": [{"band": "low"}, {"band": "mid"}, {"band": "high"}], "default": {"band": "unknown"},
                   "dtypes": [["band", "String"]]},
        "parameters": {},
    })
    out = run(tree, pl.DataFrame({"score": [10.0, 30.0, 50.0, 70.0, 90.0]}))
    assert out["band"].to_list() == ["low", "mid", "mid", "high", "high"]
