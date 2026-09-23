"""decider_old's own tree fixtures give decider_old's answers."""
import polars as pl

from decider.steps.trees import TreeConfig


def tree(nodes: list, edges: list, output: dict, name: str = "t") -> TreeConfig:
    return TreeConfig(name=name, tree={
        "nodes": [{"id": i, "position": {"x": 10 * k, "y": 0}, "data": d} for k, (i, d) in enumerate(nodes)],
        "edges": [{"source": s, "target": t, "data": {"sourceIndex": [b]}} for s, t, b in edges],
        "output": output,
    })


def labels(column: str, *values: str, default: str) -> dict:
    return {"data": [{column: v} for v in values], "default": {column: default}, "dtypes": [[column, "String"]]}


def leaf(i: int) -> dict:
    return {"type": "leaf", "result_idx": i}


BANDS = {"type": "cases", "op": "ranges", "feature": "score", "end_logic": "lower_inclusive", "strict": False,
         "conditions": [{"max": 30.0}, {"min": 30.0, "max": 70.0}, {"min": 70.0}]}


def test_v3_range_cases_band_exactly_as_decider_old(run):
    t = tree([("root", BANDS), ("low", leaf(0)), ("mid", leaf(1)), ("high", leaf(2)), ("unk", leaf(-1))],
             [("root", "low", 0), ("root", "mid", 1), ("root", "high", 2), ("root", "unk", 3)],
             labels("band", "low", "mid", "high", default="unknown"))
    # lower_inclusive, [min, max): 30 starts mid.
    out = run(t, pl.DataFrame({"score": [10.0, 30.0, 50.0, 70.0, 90.0]}))
    assert out["band"].to_list() == ["low", "mid", "mid", "high", "high"]


def test_a_string_match_subtree_exactly_as_decider_old(run):
    t = tree(
        [("num_root", {"type": "unary", "condition": {"op": "<", "feature": "age", "threshold": 30.0}}),
         ("young", leaf(0)),
         ("str_root", {"type": "unary", "condition": {"op": "string_match", "feature": "status",
                                                      "patterns": ["VIP"], "case_sensitive": False}}),
         ("vip", leaf(2)), ("std", leaf(3))],
        [("num_root", "young", 0), ("num_root", "str_root", 1), ("str_root", "vip", 0), ("str_root", "std", 1)],
        labels("group", "young", "adult", "vip", "standard", default="unknown"))
    frame = pl.DataFrame({"age": [20.0, 40.0, 25.0, 35.0], "status": ["vip", "standard", "regular", "vip"]})
    assert run(t, frame)["group"].to_list() == ["young", "standard", "young", "vip"]


def test_a_unary_node_leading_to_range_cases_exactly_as_decider_old(run):
    buckets = {**BANDS, "conditions": [{"max": 40.0}, {"min": 40.0, "max": 70.0}, {"min": 70.0}]}
    t = tree([("root", {"type": "unary", "condition": {"op": "<", "feature": "age", "threshold": 30.0}}),
              ("buckets", buckets), ("low", leaf(0)), ("mid", leaf(1)), ("high", leaf(2)), ("none", leaf(-1)),
              ("adult", leaf(3))],
             [("root", "buckets", 0), ("root", "adult", 1), ("buckets", "low", 0), ("buckets", "mid", 1),
              ("buckets", "high", 2), ("buckets", "none", 3)],
             labels("r", "low", "mid", "high", "adult", default="default"))
    frame = pl.DataFrame({"age": [25.0, 25.0, 25.0, 45.0], "score": [10.0, 50.0, 90.0, 10.0]})
    assert run(t, frame)["r"].to_list() == ["low", "mid", "high", "adult"]


def test_a_migrated_numeric_tree_agrees_in_every_mode(run):
    t = tree([("root", BANDS), ("low", leaf(0)), ("mid", leaf(1)), ("high", leaf(2))],
             [("root", "low", 0), ("root", "mid", 1), ("root", "high", 2)],
             {"data": [{"pts": 1.0}, {"pts": 2.0}, {"pts": 3.0}], "default": {"pts": 0.0},
              "dtypes": [["pts", "Float64"]]})
    out = run(t, pl.DataFrame({"score": [10.0, 30.0, 50.0, 70.0, 90.0, -1.0]}))
    assert out["pts"].to_list() == [1.0, 2.0, 2.0, 3.0, 3.0, 1.0]
