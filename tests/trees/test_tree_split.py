"""A tree with many rule outputs is several walker calls, with the same columns and answers as one."""
import polars as pl

from decider import _engine as engine
from decider.steps.trees import TreeConfig
from decider.steps.trees._split import MAX_OUTPUTS

OUTPUT = {"data": [{"hit": True}], "default": {"hit": False}, "dtypes": [["hit", "Boolean"]]}


def _wide(n: int, threshold=None, **config) -> TreeConfig:
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
    nodes = _nodes(_wide(3 * MAX_OUTPUTS + 5))
    assert len(nodes) == 4
    assert all(len(n.outputs) <= MAX_OUTPUTS for n in nodes)
    assert [o.name for n in nodes for o in n.outputs] == [
        f"{'R' + str(k) if k % 2 else 'rule_' + str(k)}.hit" for k in range(3 * MAX_OUTPUTS + 5)]


def test_the_split_tree_answers_like_the_rules_it_holds(run):
    n = 2 * MAX_OUTPUTS + 3
    df = pl.DataFrame({"f0": [0.0, 50.0], "f1": [10.0, 50.0], "f2": [100.0, 0.0]})
    out = run(_wide(n, path_output="leaf"), df)
    for k in range(n):
        name = f"R{k}" if k % 2 else f"rule_{k}"
        assert out[f"{name}.hit"].to_list() == [df[f"f{k % 3}"][i] > k for i in range(2)]
        assert out[f"{name}.leaf"].is_null().to_list() == (~out[f"{name}.hit"]).to_list()


def test_a_tree_with_params_stays_one_row_node():
    cfg = _wide(2 * MAX_OUTPUTS, threshold={"param": "limit", "default": 1.0})
    (node,) = _nodes(cfg)
    assert len(node.outputs) == 2 * MAX_OUTPUTS
    assert [p.name for p in node.params] == ["limit"]
