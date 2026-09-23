"""Tree features keep their type: an int64 is compared as an int64, a bool as a bool."""
import polars as pl
import pytest

from decider import engine
from decider.engine import Engine
from decider.engine.compile import kernel
from decider.engine.ir.decls import base_annotation
from decider.steps.trees import TreeConfig, walker

BIG = 9007199254740992  # 2**53: the first integer float64 can't tell from its successor
MODES = ("interpreted", "stepped", "fused")


def one_node(cond: dict, name: str = "t", **kw) -> TreeConfig:
    """`root -> yes (hit 1) / no (hit 0)`."""
    return TreeConfig(name=name, tree={
        "nodes": [{"id": "root", "data": {"type": "unary", "condition": cond}},
                  {"id": "yes", "data": {"type": "leaf", "result_idx": 1}},
                  {"id": "no", "data": {"type": "leaf", "result_idx": 0}}],
        "edges": [{"source": "root", "target": "yes", "data": {"sourceIndex": 0}},
                  {"source": "root", "target": "no", "data": {"sourceIndex": 1}}],
        "output": {"data": [{"hit": 0}, {"hit": 1}], "default": {"hit": -1}, "dtypes": [["hit", "Int64"]]},
    }, **kw)


def chain(conds: list, name: str = "chain", **kw) -> TreeConfig:
    """`n0 -> n1 -> ... -> leaf_all (pts n)`; node i's false branch is leaf i."""
    nodes, edges = [], []
    for i, cond in enumerate(conds):
        nodes += [{"id": f"n{i}", "data": {"type": "unary", "condition": cond}},
                  {"id": f"leaf{i}", "data": {"type": "leaf", "result_idx": i}}]
        edges.append({"source": f"n{i}", "target": f"leaf{i}", "data": {"sourceIndex": 1}})
        nxt = f"n{i + 1}" if i + 1 < len(conds) else "leaf_all"
        edges.append({"source": f"n{i}", "target": nxt, "data": {"sourceIndex": 0}})
    nodes.append({"id": "leaf_all", "data": {"type": "leaf", "result_idx": len(conds)}})
    return TreeConfig(name=name, tree={"nodes": nodes, "edges": edges, "output": {
        "data": [{"pts": i} for i in range(len(conds) + 1)], "default": {"pts": -1}, "dtypes": [["pts", "Int64"]]}},
        **kw)


def kinds(tree: TreeConfig) -> dict:
    return {i.name: base_annotation(i.annotation) for i in engine.to_ir(tree).inputs}


def test_int64_above_2_53_is_compared_exactly_in_every_mode(run):
    tree = one_node({"op": "==", "feature": "n", "threshold": BIG}, feature_types={"n": "int"})
    frame = pl.DataFrame({"n": pl.Series([BIG, BIG + 1], dtype=pl.Int64)})
    assert run(tree, frame)["hit"].to_list() == [1, 0]
    for mode in MODES:
        exe = Engine().bind(tree, mode=mode)
        assert (exe.score({"n": BIG})["hit"], exe.score({"n": BIG + 1})["hit"]) == (1, 0), mode


def test_an_undeclared_int_feature_is_read_as_a_float_and_collapses(run):
    tree = one_node({"op": "==", "feature": "n", "threshold": BIG}, name="undeclared")
    assert kinds(tree) == {"n": float}
    frame = pl.DataFrame({"n": pl.Series([BIG, BIG + 1], dtype=pl.Int64)})
    assert run(tree, frame)["hit"].to_list() == [1, 1]


def test_a_bool_feature_stays_a_bool_end_to_end(run):
    tree = one_node({"op": "is_true", "feature": "flag"}, name="b")
    assert kinds(tree) == {"flag": bool}
    assert run(tree, pl.DataFrame({"flag": [True, False, True]}))["hit"].to_list() == [1, 0, 1]
    for mode in MODES:
        assert Engine().bind(tree, mode=mode).score({"flag": False})["hit"] == 0, mode


def test_an_int_feature_stays_an_int_end_to_end(run):
    tree = chain([
        {"op": ">", "feature": "cents", "threshold": 100},
        {"op": "between", "feature": "cents", "min": 150, "max": 250},
        {"op": "isin", "feature": "cents", "values": [200, 210]},
    ], feature_types={"cents": "Int64"})
    assert kinds(tree) == {"cents": int}
    frame = pl.DataFrame({"cents": pl.Series([50, 120, 200, 240, 300], dtype=pl.Int64)})
    assert run(tree, frame)["pts"].to_list() == [0, 1, 3, 2, 1]


def test_a_mixed_tree_agrees_across_every_mode(run):
    tree = chain([
        {"op": "<", "feature": "score", "threshold": 700.0},
        {"op": ">=", "feature": "income_cents", "threshold": 5_000_00},
        {"op": "is_true", "feature": "is_staff"},
        {"op": "string_match", "feature": "sector", "patterns": ["retail", "public"]},
        {"op": ">", "feature": {"type": "computed", "expression": "score - burden"}, "threshold": 100.0},
        {"op": "isin", "feature": "defaults", "values": [0, 1]},
    ], name="mixed", feature_types={"income_cents": "int", "defaults": "int"})
    assert kinds(tree) == {"score": float, "burden": float, "income_cents": int, "defaults": int,
                           "is_staff": bool, "sector": bytes}
    # Row i fails at node i (its false branch is leaf i); row 6 passes every node.
    frame = pl.DataFrame({
        "score": [720.0, 650.0, 650.0, 650.0, 650.0, 650.0, 650.0],
        "income_cents": pl.Series([6_000_00, 4_000_00] + [6_000_00] * 5, dtype=pl.Int64),
        "is_staff": [True, True, False, True, True, True, True],
        "sector": ["retail", "retail", "retail", "mining", "retail", "retail", "public"],
        "burden": [0.0, 0.0, 0.0, 0.0, 700.0, 0.0, 0.0],
        "defaults": pl.Series([0, 0, 0, 0, 0, 5, 1], dtype=pl.Int64),
    })
    assert run(tree, frame)["pts"].to_list() == [0, 1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("tree, message", [
    (one_node({"op": "<", "feature": "sector", "threshold": 5.0}, feature_types={"sector": "str"}),
     "numeric comparison on the string feature"),
    (chain([{"op": "string_match", "feature": "sector", "patterns": ["a"]},
            {"op": "<", "feature": "sector", "threshold": 5.0}]), "numeric comparison on the string feature"),
    (one_node({"op": "==", "feature": "flag", "threshold": 1}, feature_types={"flag": "bool"}),
     "threshold on the boolean feature"),
    (one_node({"op": "is_true", "feature": "sector"}, feature_types={"sector": "str"}),
     "is_true / is_false on the string feature"),
    (one_node({"op": "string_match", "feature": "n", "patterns": ["a"]}, feature_types={"n": "int"}),
     "string_match on the int feature"),
    (one_node({"op": "<", "feature": "n", "threshold": 5000.5}, feature_types={"n": "int"}), "not a whole number"),
    (one_node({"op": ">", "feature": {"type": "computed", "expression": "n - 1"}, "threshold": 0.0},
              feature_types={"n": "int"}), "computed feature reads the int feature"),
    (chain([{"op": "<", "feature": "a", "threshold": {"key": "floor"}},
            {"op": "<", "feature": "b", "threshold": {"key": "floor"}}], feature_types={"a": "int", "b": "float"}),
     "one param has one type"),
    (one_node({"op": "<", "feature": "x", "threshold": 1.0}, feature_types={"y": "int"}), "does not read"),
])
def test_meaningless_uses_of_a_feature_are_errors_naming_it(tree, message):
    with pytest.raises(ValueError, match=message):
        engine.to_ir(tree)


def test_an_unknown_feature_type_is_rejected_when_the_config_loads():
    with pytest.raises(ValueError, match="not a type a tree understands"):
        one_node({"op": "<", "feature": "x", "threshold": 1.0}, feature_types={"x": "Decimal"})


def test_polars_spellings_are_accepted():
    tree = chain([{"op": "<", "feature": "a", "threshold": 1}, {"op": "is_true", "feature": "b"},
                  {"op": "<", "feature": "c", "threshold": 1.0}],
                 feature_types={"a": "Int64", "b": "Boolean", "c": "Float64"})
    assert kinds(tree) == {"a": int, "b": bool, "c": float}


def test_retuning_an_int_threshold_never_recompiles():
    tree = one_node({"op": ">", "feature": "cents", "threshold": {"param": "floor", "default": 100}},
                    name="retune", feature_types={"cents": "int"})
    exe = Engine().bind(tree, mode="fused")
    frame = pl.DataFrame({"cents": pl.Series([50, 150, 250], dtype=pl.Int64)})
    assert exe.run(frame)["hit"].to_list() == [0, 1, 1]
    before = len(walker.walk.signatures), len(kernel._KERNELS)
    assert exe.run(frame, params={"retune": {"floor": 200}})["hit"].to_list() == [0, 0, 1]
    assert (len(walker.walk.signatures), len(kernel._KERNELS)) == before


def test_an_undeclared_tree_reads_every_feature_as_a_float():
    tree = chain([{"op": "<", "feature": f"f{i}", "threshold": float(i)} for i in range(5)])
    assert set(kinds(tree).values()) == {float}
