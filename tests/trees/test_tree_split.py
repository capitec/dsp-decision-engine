"""A tree with many rule outputs is several walker calls, with the same columns and answers as one."""
import polars as pl
import pytest

from decider import _engine as engine
from decider.steps.trees import TreeConfig
from decider.steps.trees._split import MAX_OUTPUTS

W = 8  # part width the tests ask for, so a wide tree stays a few dozen rules

OUTPUT = {"data": [{"hit": True}], "default": {"hit": False}, "dtypes": [["hit", "Boolean"]]}


def _wide(n: int, threshold=None, **config) -> TreeConfig:
    config.setdefault("part_outputs", W)
    leaf = {"type": "leaf", "result_idx": 0}
    rules = [{"meta": {"name": f"R{k}"} if k % 2 else {},
              "rule": {"type": "unary", "condition": {"op": ">", "feature": f"f{k % 3}",
                                                        "threshold": k if threshold is None else threshold},
                       "then": leaf}}
             for k in range(n)]
    return TreeConfig(name="wide", **config, tree={"type": "prioritized_flat_rule", "mode": "all", "rules": rules,
                                                    "output": OUTPUT})


def _nodes(cfg: TreeConfig):
    node = engine.to_ir(cfg)
    return node.children() or (node,)


def test_a_tree_with_many_outputs_is_several_row_nodes_of_bounded_width():
    nodes = _nodes(_wide(3 * W + 5))
    assert len(nodes) == 4
    assert all(len(n.outputs) <= W for n in nodes)
    assert [o.name for n in nodes for o in n.outputs] == [
        f"{'R' + str(k) if k % 2 else 'rule_' + str(k)}.hit" for k in range(3 * W + 5)]


def test_the_split_tree_answers_like_the_rules_it_holds(run):
    n = 2 * W + 3
    df = pl.DataFrame({"f0": [0.0, 50.0], "f1": [10.0, 50.0], "f2": [100.0, 0.0]})
    out = run(_wide(n, path_output="leaf"), df)
    for k in range(n):
        name = f"R{k}" if k % 2 else f"rule_{k}"
        assert out[f"{name}.hit"].to_list() == [df[f"f{k % 3}"][i] > k for i in range(2)]
        assert out[f"{name}.leaf"].is_null().to_list() == (~out[f"{name}.hit"]).to_list()


def test_a_tree_with_params_stays_one_row_node():
    cfg = _wide(2 * W, threshold={"param": "limit", "default": 1.0})
    (node,) = _nodes(cfg)
    assert len(node.outputs) == 2 * W
    assert [p.name for p in node.params] == ["limit"]


def _rules(n: int, condition) -> dict:
    leaf = {"type": "leaf", "result_idx": 0}
    rules = [{"meta": {"name": f"R{k}"}, "rule": {"type": "unary", "condition": condition(k), "then": leaf}}
             for k in range(n)]
    return {"type": "prioritized_flat_rule", "mode": "all", "rules": rules, "output": OUTPUT}


def test_a_part_reads_a_feature_as_the_whole_tree_does(run):
    n = W + 4
    tree = _rules(n, lambda k: {"op": "is_true", "feature": "flag"} if k < W
                  else {"op": ">", "feature": "flag", "threshold": 0.5})
    out = run(TreeConfig(name="t", part_outputs=W, tree=tree), pl.DataFrame({"flag": [1.0, 0.0]}))
    assert out["R0.hit"].to_list() == out[f"R{n - 1}.hit"].to_list() == [True, False]


def test_a_declared_feature_type_applies_to_the_part_that_reads_it(run):
    n = W + 4
    tree = _rules(n, lambda k: {"op": ">", "feature": "a" if k < W else "b", "threshold": 0})
    out = run(TreeConfig(name="t", part_outputs=W, feature_types={"b": "int"}, tree=tree), pl.DataFrame({"a": [1.0], "b": [1]}))
    assert out[f"R{n - 1}.hit"].to_list() == [True]


def test_a_node_breakpoint_on_the_tree_pauses_after_the_part_that_reached_it():
    n = W + 4
    tree = _rules(n, lambda k: {"op": ">", "feature": "x", "threshold": k})
    cfg = TreeConfig(name="t", part_outputs=W, tree=tree)
    s = engine.Engine().bind(cfg).session(pl.DataFrame({"x": [1000.0]}))
    s.break_at(f"t#{cfg.tree.to_tree().rules[-1].root}")
    at = s.resume()
    assert (at.origin.path, at.when) == ("t/part_1", "after")


def test_part_outputs_sets_the_width_of_each_part():
    nodes = _nodes(_wide(100, part_outputs=64))
    assert [len(n.outputs) for n in nodes] == [64, 36]


def test_part_outputs_none_keeps_a_wide_tree_whole(run):
    n = 3 * W
    (node,) = _nodes(_wide(n, part_outputs=None))
    assert len(node.outputs) == n
    df = pl.DataFrame({"f0": [0.0, 50.0], "f1": [10.0, 50.0], "f2": [100.0, 0.0]})
    assert run(_wide(n, part_outputs=None), df).equals(run(_wide(n), df))


def test_part_outputs_below_one_is_invalid():
    with pytest.raises(ValueError, match="part_outputs"):
        _wide(10, part_outputs=0)


def test_a_tree_splits_into_parts_of_the_default_width_unless_told_otherwise():
    assert TreeConfig(name="t", tree=_rules(1, lambda k: {"op": ">", "feature": "x", "threshold": 0})).part_outputs == MAX_OUTPUTS
