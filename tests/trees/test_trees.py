"""A tree's own properties: params, retuning without recompiling, node kinds, composition, size."""
import numpy as np
import polars as pl
import pytest

from decider import engine, flow
from decider.engine import Engine
from decider.engine.compile import kernel
from decider.steps.trees import TreeConfig, walker


def v3(nodes: dict, edges: list, output: dict) -> dict:
    """A v3 document from `{id: data}` and `(source, target, branch)` edges."""
    return {"nodes": [{"id": k, "data": d} for k, d in nodes.items()],
            "edges": [{"source": s, "target": t, "data": {"sourceIndex": [i]}} for s, t, i in edges],
            "output": output}


def unary(op: str, feature, threshold) -> dict:
    return {"type": "unary", "condition": {"op": op, "feature": feature, "threshold": threshold}}


def leaf(i: int) -> dict:
    return {"type": "leaf", "result_idx": i}


def pts(*values, default=0.0) -> dict:
    return {"data": [{"pts": v} for v in values], "default": {"pts": default}, "dtypes": [["pts", "Float64"]]}


def two_level(age=30.0, score=700.0, name="risk") -> TreeConfig:
    """age < A, then score >= B: two thresholds, three leaves."""
    return TreeConfig(name=name, tree=v3(
        {"age_node": unary("<", "age", age), "score_node": unary(">=", "score", score),
         "leaf_good": leaf(0), "leaf_bad": leaf(1), "leaf_old": leaf(2)},
        [("age_node", "score_node", 0), ("age_node", "leaf_old", 1),
         ("score_node", "leaf_good", 0), ("score_node", "leaf_bad", 1)],
        pts(10.0, 20.0, 30.0),
    ))


FRAME = pl.DataFrame({"age": [25.0, 25.0, 45.0], "score": [800.0, 600.0, 0.0]})


def compiled_counts() -> tuple[int, int]:
    return len(walker.walk.signatures), len(kernel._KERNELS)


def test_a_tree_walks_to_the_leaf_each_row_reaches(run):
    assert run(two_level(), FRAME)["pts"].to_list() == [10.0, 20.0, 30.0]


def test_retuning_a_param_threshold_never_recompiles():
    tree = two_level(age={"param": "age_floor", "default": 30.0})
    exe = Engine().bind(tree, mode="fused")
    frame = pl.DataFrame({"age": [25.0, 45.0], "score": [800.0, 600.0]})
    exe.run(frame)
    before = compiled_counts()
    answers = [exe.run(frame, params={"risk": {"age_floor": a}})["pts"].to_list() for a in (30.0, 40.0, 50.0, 18.0)]
    assert compiled_counts() == before
    assert answers[0] != answers[3]


def test_editing_a_literal_threshold_rebuilds_the_ir_but_never_recompiles():
    frame = pl.DataFrame({"age": [25.0, 45.0], "score": [800.0, 600.0]})
    Engine().bind(two_level(), mode="fused").run(frame)
    before = compiled_counts()
    answers = [Engine().bind(two_level(age=a), mode="fused").run(frame)["pts"].to_list() for a in (40.0, 50.0, 18.0)]
    assert compiled_counts() == before
    assert answers == [[10.0, 30.0], [10.0, 20.0], [30.0, 30.0]]


def test_a_literal_threshold_and_a_param_threshold_give_the_same_answers(run):
    literal = two_level(age=40.0)
    ref = two_level(age={"key": "age_floor"}, name="risk_ref")
    assert "risk" not in literal.parameters()
    assert list(ref.parameters()["risk_ref"]) == ["age_floor"]
    frame = pl.DataFrame({"age": [25.0, 45.0], "score": [800.0, 600.0]})
    assert (run(literal, frame)["pts"].to_list()
            == run(ref, frame, params={"risk_ref": {"age_floor": 40.0}})["pts"].to_list())


def test_one_param_named_twice_is_one_knob(run):
    tree = TreeConfig(name="shared_ref", tree=v3(
        {"a": unary("<", "x", {"key": "floor"}), "b": unary("<", "y", {"key": "floor"}),
         "leaf_0": leaf(0), "leaf_1": leaf(1), "leaf_2": leaf(2)},
        [("a", "b", 0), ("a", "leaf_2", 1), ("b", "leaf_0", 0), ("b", "leaf_1", 1)],
        pts(0.0, 1.0, 2.0, default=-1.0),
    ))
    assert list(tree.parameters()["shared_ref"]) == ["floor"]
    frame = pl.DataFrame({"x": [1.0, 1.0, 9.0], "y": [1.0, 9.0, 1.0]})
    assert run(tree, frame, params={"shared_ref": {"floor": 5.0}})["pts"].to_list() == [0.0, 1.0, 2.0]


def test_an_unconnected_branch_reaches_the_default_row(run):
    tree = TreeConfig(name="partial", tree=v3(
        {"root": unary("<", "x", 10.0), "leaf_yes": leaf(0)}, [("root", "leaf_yes", 0)], pts(1.0, default=99.0)))
    assert run(tree, pl.DataFrame({"x": [5.0, 50.0]}))["pts"].to_list() == [1.0, 99.0]


def test_composite_between_and_isin_nodes(run):
    tree = TreeConfig(name="kinds", tree=v3(
        {"comp": {"type": "composite", "op": "and", "conditions": [
            {"op": "between", "feature": "age", "min": 18.0, "max": 65.0},
            {"op": "is_true", "feature": "verified"}]},
         "cases": {"type": "cases", "op": "isin", "feature": "region",
                   "conditions": [{"values": [1, 2]}, {"values": [3]}]},
         "leaf_a": leaf(0), "leaf_b": leaf(1), "leaf_other": leaf(2), "leaf_no": leaf(-1)},
        [("comp", "cases", 0), ("comp", "leaf_no", 1),
         ("cases", "leaf_a", 0), ("cases", "leaf_b", 1), ("cases", "leaf_other", 2)],
        pts(1.0, 2.0, 3.0),
    ))
    frame = pl.DataFrame({
        "age": [30.0, 30.0, 30.0, 70.0, 18.0, 65.0],
        "verified": [True, True, True, True, True, False],
        "region": [1.0, 3.0, 9.0, 1.0, 2.0, 1.0],
    })
    # rows 0-2: composite holds, region routes; row 3: age > 65; row 4: 18 is inclusive; row 5: not verified.
    assert run(tree, frame)["pts"].to_list() == [1.0, 2.0, 3.0, 0.0, 1.0, 0.0]


def test_two_same_shaped_sibling_thresholds_do_not_collide(run):
    band = {"type": "composite", "op": "and", "conditions": [
        {"op": ">", "feature": "x", "threshold": {"param": "lo", "default": 5.0}},
        {"op": "<", "feature": "x", "threshold": {"param": "hi", "default": 10.0}}]}
    tree = TreeConfig(name="band", tree=v3(
        {"root": band, "leaf_in": leaf(0), "leaf_out": leaf(1)}, [("root", "leaf_in", 0), ("root", "leaf_out", 1)],
        pts(1.0, 0.0, default=-1.0)))
    schema = tree.parameters()["band"]
    assert {k: v["default"] for k, v in schema.items()} == {"lo": 5.0, "hi": 10.0}
    frame = pl.DataFrame({"x": [7.0, 3.0, 12.0, 5.0, 10.0]})
    assert run(tree, frame)["pts"].to_list() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert run(tree, frame, params={"band": {"lo": 2.0}})["pts"].to_list() == [1.0, 1.0, 0.0, 1.0, 0.0]


def test_a_tree_composes_with_ordinary_steps(run):
    def disposable_income(net_income: float, expenses: float) -> float:
        return net_income - expenses

    def final_score(pts: float) -> float:
        return pts * 2.0

    tree = TreeConfig(name="affordability_band", tree=v3(
        {"root": {"type": "cases", "op": "ranges", "feature": "disposable_income", "strict": False,
                  "conditions": [{"max": 1000.0}, {"min": 1000.0}]},
         "leaf_low": leaf(0), "leaf_high": leaf(1)},
        [("root", "leaf_low", 0), ("root", "leaf_high", 1)], pts(5.0, 50.0)))
    pipeline = flow(disposable_income, tree, final_score)
    frame = pl.DataFrame({"net_income": [5000.0, 2000.0], "expenses": [1500.0, 1500.0]})
    assert run(pipeline, frame)["final_score"].to_list() == [100.0, 10.0]


def chain(length: int, arm: int) -> TreeConfig:
    """`length` nodes `x < i`, continuing down branch `arm`; the other branch is leaf 0."""
    nodes, edges = {}, []
    for i in range(length):
        nodes[f"n{i}"], nodes[f"l{i}"] = unary("<", "x", float(i)), leaf(0)
        edges.append((f"n{i}", f"l{i}", 1 - arm))
        if i + 1 < length:
            edges.append((f"n{i}", f"n{i + 1}", arm))
    return TreeConfig(name="chain", tree=v3(nodes, edges, pts(1.0)))


def test_a_very_large_tree_builds_and_runs(run):
    # x < 0 fails, x < 1 holds: leaf 0. Past the end of the chain: the default row.
    assert run(chain(400, arm=1), pl.DataFrame({"x": [0.5, 1000.0]}))["pts"].to_list() == [1.0, 0.0]


def test_a_then_chain_thousands_deep_builds_and_runs(run):
    tree = chain(3000, arm=0)
    assert len(tree.nodes) == 6000
    assert run(tree, pl.DataFrame({"x": [-1.0, 0.5]}))["pts"].to_list() == [0.0, 1.0]


def test_a_computed_feature_reads_the_columns_its_expression_names():
    tree = TreeConfig(name="c", tree=v3(
        {"root": unary("<", {"type": "computed", "expression": "income - expenses"}, 1.0), "yes": leaf(0)},
        [("root", "yes", 0)], pts(1.0)))
    assert sorted(i.name for i in engine.to_ir(tree).inputs) == ["expenses", "income"]


@pytest.mark.parametrize("kwargs, expected", [
    ({"match_type": "regex", "patterns": ["^a.c$"]}, [1.0, 0.0, 0.0]),
    ({"case_sensitive": False, "patterns": ["abc"]}, [1.0, 1.0, 0.0]),
    ({"trim_whitespace": True, "patterns": ["abc"]}, [1.0, 0.0, 1.0]),
    ({"match_type": "contains", "patterns": ["b"]}, [1.0, 0.0, 1.0]),
])
def test_every_string_match_option_gives_the_python_answer(run, kwargs, expected):
    cond = {"type": "unary", "condition": {"op": "string_match", "feature": "s", **kwargs}}
    tree = TreeConfig(name="t", tree=v3({"root": cond, "hit": leaf(0)}, [("root", "hit", 0)], pts(1.0)))
    assert run(tree, pl.DataFrame({"s": ["abc", "ABC", " abc "]}))["pts"].to_list() == expected


def test_the_same_document_encodes_to_the_same_arrays():
    first, second = engine.to_ir(two_level()), engine.to_ir(two_level())
    assert dict(first.consts)["layout"] == dict(second.consts)["layout"]
    assert dict(first.consts)["outputs"] == dict(second.consts)["outputs"]
    assert all(np.array_equal(a, b) for a, b in zip(first.reference.arrays, second.reference.arrays))


def test_strict_ranges_must_be_contiguous():
    with pytest.raises(ValueError, match="not continuous"):
        TreeConfig(name="t", tree=v3(
            {"root": {"type": "cases", "op": "ranges", "feature": "x", "conditions": [{"max": 10.0}, {"min": 20.0}]}},
            [], pts(1.0)))


def test_retuning_changes_score_without_recompiling():
    tree = two_level(score={"param": "score_floor", "default": 700.0})
    assert tree.parameters()["risk"]["score_floor"]["default"] == 700.0
    exe = Engine().bind(tree, mode="fused")
    record = {"age": 25.0, "score": 800.0}
    assert exe.score(record)["pts"] == 10.0
    before = compiled_counts()
    assert exe.score(record, params={"risk": {"score_floor": 900.0}})["pts"] == 20.0
    assert compiled_counts() == before
