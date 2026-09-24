"""`trace_output`: the ordered path of node ids walked, from both walkers, in every mode."""
import polars as pl
import pytest

from decider import engine
from decider.engine import Engine
from decider.steps.trees import TreeConfig


def _node(nid, op, feature, threshold):
    return {"id": nid, "data": {"type": "unary", "condition": {"op": op, "feature": feature, "threshold": threshold}}}


def _leaf(nid, k):
    return {"id": nid, "data": {"type": "leaf", "result_idx": k}}


def _edges(source, then, otherwise):
    return [{"source": source, "target": then, "data": {"sourceIndex": 0}},
            {"source": source, "target": otherwise, "data": {"sourceIndex": 1}}]


# A campaign tree shaped like the worked example: node 6 has two parents (4 and 5), so the leaf alone
# doesn't say how a client got to offer 912.
CAMPAIGN = TreeConfig(name="campaign", path_output="leaf", trace_output="path", feature_types={"age": "int"}, tree={
    "nodes": [
        _node("1", ">=", "age", 18),
        {"id": "2", "data": {"type": "unary", "condition": {"op": "is_true", "feature": "holds_product"}}},
        _node("4", "<=", "risk_grade", 3.0),
        _node("5", ">", "income", 20000.0),
        _node("6", ">=", "pre_assessed_amount", 50000.0),
        _node("8", "<", "utilisation", 0.8),
        _node("10", ">=", "pre_assessed_amount", 80000.0),
        _leaf("901", 0), _leaf("902", 0), _leaf("905", 0), _leaf("906", 0), _leaf("908", 0),
        _leaf("911", 1), _leaf("912", 2),
    ],
    "edges": [*_edges("1", "2", "901"), *_edges("2", "902", "4"), *_edges("4", "6", "5"), *_edges("5", "6", "905"),
              *_edges("6", "8", "906"), *_edges("8", "10", "908"), *_edges("10", "912", "911")],
    "output": {"data": [{"offer": 0}, {"offer": 15000}, {"offer": 20800}], "default": {"offer": 0},
               "dtypes": [["offer", "Int64"]]},
})
CLIENT = {"age": 30, "holds_product": False, "risk_grade": 2.0, "income": 9000.0, "pre_assessed_amount": 84000.0,
          "utilisation": 0.5}


def test_the_trace_is_the_ordered_path_through_a_shared_node(run):
    df = pl.DataFrame([CLIENT, {**CLIENT, "risk_grade": 5.0, "income": 30000.0}, {**CLIENT, "holds_product": True},
                       {**CLIENT, "age": None}, {**CLIENT, "pre_assessed_amount": 60000.0}])
    out = run(CAMPAIGN, df)
    assert out["path"].to_list() == ["1>2>4>6>8>10>912", "1>2>4>5>6>8>10>912", "1>2>902", "1>901",
                                     "1>2>4>6>8>10>911"]
    assert out["leaf"].to_list() == ["912", "912", "902", "901", "911"]
    assert out.schema["path"] == pl.String


@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_a_scored_record_carries_its_path(mode):
    record = Engine().bind(CAMPAIGN, mode=mode).score(CLIENT)
    assert (record["offer"], record["leaf"], record["path"]) == (20800, "912", "1>2>4>6>8>10>912")


def test_a_trace_output_must_not_clash_with_another_column():
    with pytest.raises(ValueError, match="trace_output 'offer'"):
        engine.to_ir(CAMPAIGN.model_copy(update={"trace_output": "offer"}))
    with pytest.raises(ValueError, match="both 'leaf'"):
        engine.to_ir(CAMPAIGN.model_copy(update={"trace_output": "leaf"}))


def test_a_tree_without_rules_traces_a_null(run):
    tree = TreeConfig(name="t", trace_output="path", tree={
        "type": "prioritized_flat_rule", "rules": [],
        "output": {"data": [], "default": {"y": 0}, "dtypes": [["y", "Int64"]]}})
    assert run(tree, pl.DataFrame({"x": [1.0]}))["path"].to_list() == [None]


def test_a_tree_with_too_many_paths_to_trace_is_refused():
    # Each node sends both branches to the next one: 2**18 paths through 18 nodes.
    nodes = [_node(f"n{k}", ">", "x", float(k)) for k in range(18)] + [_leaf("end", 0)]
    edges = [e for k in range(18) for e in _edges(f"n{k}", f"n{k + 1}" if k < 17 else "end",
                                                   f"n{k + 1}" if k < 17 else "end")]
    tree = TreeConfig(name="t", trace_output="path", tree={
        "nodes": nodes, "edges": edges, "output": {"data": [{"y": 1}], "dtypes": [["y", "Int64"]]}})
    with pytest.raises(ValueError, match="262144 distinct paths"):
        engine.to_ir(tree)
