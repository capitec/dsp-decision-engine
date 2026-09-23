"""Nulls in trees: decider_old's three-valued routing by default, a strict error on request."""
import polars as pl
import pytest

from decider.engine import Engine
from decider.exceptions import MissingInputError
from decider.steps.trees import TreeConfig

MODES = ("interpreted", "stepped", "fused")
OUT = {"data": [{"r": "yes"}], "default": {"r": "no"}, "dtypes": [["r", "String"]]}


def _tree(condition: dict, **config) -> TreeConfig:
    return TreeConfig(name="t", **config, tree={
        "nodes": [{"id": "root", "data": {"type": "unary", "condition": condition}},
                  {"id": "yes", "data": {"type": "leaf", "result_idx": 0}}],
        "edges": [{"source": "root", "target": "yes", "data": {"sourceIndex": 0}}],
        "output": OUT,
    })


def _composite(op: str, *conditions: dict, **config) -> TreeConfig:
    return TreeConfig(name="t", **config, tree={
        "nodes": [{"id": "root", "data": {"type": "composite", "op": op, "conditions": list(conditions)}},
                  {"id": "yes", "data": {"type": "leaf", "result_idx": 0}}],
        "edges": [{"source": "root", "target": "yes", "data": {"sourceIndex": 0}}],
        "output": OUT,
    })


def lt(feature: str, threshold: float) -> dict:
    return {"op": "<", "feature": feature, "threshold": threshold}


def nt(*conditions: dict) -> dict:
    return {"type": "composite", "op": "not", "conditions": list(conditions)}


def _col(*values) -> pl.Series:
    return pl.Series(values, dtype=pl.Float64)


def test_a_null_number_takes_the_otherwise_branch(run):
    out = run(_tree(lt("x", 50.0)), pl.DataFrame({"x": _col(10.0, None, 90.0)}))
    assert out["r"].to_list() == ["yes", "no", "no"]


def test_not_of_a_null_test_is_still_unknown_so_it_takes_otherwise_too(run):
    out = run(_composite("not", lt("x", 50.0)), pl.DataFrame({"x": _col(10.0, None, 90.0)}))
    assert out["r"].to_list() == ["no", "no", "yes"]


def test_and_and_or_follow_three_valued_logic(run):
    df = pl.DataFrame({"x": _col(None, None, None), "y": _col(10.0, 90.0, None)})
    assert run(_composite("and", lt("x", 50.0), lt("y", 50.0)), df)["r"].to_list() == ["no", "no", "no"]
    assert run(_composite("or", lt("x", 50.0), lt("y", 50.0)), df)["r"].to_list() == ["yes", "no", "no"]


def test_not_over_and_or_tells_false_from_unknown(run):
    df = pl.DataFrame({"x": _col(None, None, None), "y": _col(10.0, 90.0, None)})
    # AND(unknown, false) is false, so its NOT holds; AND(unknown, true) stays unknown.
    assert run(_composite("not", {"type": "composite", "op": "and", "conditions": [lt("x", 50.0), lt("y", 50.0)]}),
               df)["r"].to_list() == ["no", "yes", "no"]
    # OR(unknown, true) is true, so its NOT fails; OR(unknown, false) stays unknown.
    assert run(_composite("not", {"type": "composite", "op": "or", "conditions": [lt("x", 50.0), lt("y", 50.0)]}),
               df)["r"].to_list() == ["no", "no", "no"]
    assert run(_composite("and", nt(lt("x", 50.0)), nt(lt("y", 50.0))), df)["r"].to_list() == ["no", "no", "no"]


def test_a_null_tries_every_case_then_takes_otherwise(run):
    tree = TreeConfig(name="t", tree={
        "nodes": [{"id": "root", "data": {"type": "cases", "op": "ranges", "feature": "x",
                                          "conditions": [{"max": 30.0}, {"min": 30.0}]}},
                  {"id": "low", "data": {"type": "leaf", "result_idx": 0}},
                  {"id": "high", "data": {"type": "leaf", "result_idx": 1}}],
        "edges": [{"source": "root", "target": "low", "data": {"sourceIndex": 0}},
                  {"source": "root", "target": "high", "data": {"sourceIndex": 1}}],
        "output": {"data": [{"r": "low"}, {"r": "high"}], "default": {"r": "none"}, "dtypes": [["r", "String"]]},
    })
    assert run(tree, pl.DataFrame({"x": _col(10.0, None, 40.0)}))["r"].to_list() == ["low", "none", "high"]


def test_a_computed_feature_over_a_null_is_unknown(run):
    cond = {"op": ">", "feature": {"type": "computed", "expression": "x - y"}, "threshold": 0.0}
    df = pl.DataFrame({"x": _col(5.0, None, 1.0), "y": _col(1.0, 1.0, 5.0)})
    assert run(_tree(cond), df)["r"].to_list() == ["yes", "no", "no"]
    assert run(_composite("not", cond), df)["r"].to_list() == ["no", "no", "yes"]


def test_null_int_and_bool_features_are_unknown(run):
    int_tree = _composite("not", lt("n", 5), feature_types={"n": "int"})
    ints = pl.DataFrame({"n": pl.Series([1, None, 9], dtype=pl.Int64)})
    assert run(int_tree, ints)["r"].to_list() == ["no", "no", "yes"]
    flags = pl.DataFrame({"b": pl.Series([True, None, False], dtype=pl.Boolean)})
    assert run(_composite("not", {"op": "is_true", "feature": "b"}), flags)["r"].to_list() == ["no", "no", "yes"]


def _match(null_handling: str) -> dict:
    return {"op": "string_match", "feature": "s", "patterns": ["a"], "null_handling": null_handling}


def test_a_null_string_is_a_plain_miss_or_hit_by_its_null_handling(run):
    df = pl.DataFrame({"s": pl.Series(["a", None, "b"], dtype=pl.String)})
    assert run(_tree(_match("no_match")), df)["r"].to_list() == ["yes", "no", "no"]
    # A miss is false, not unknown, so its NOT holds.
    assert run(_composite("not", _match("no_match")), df)["r"].to_list() == ["no", "yes", "yes"]
    assert run(_tree(_match("match")), df)["r"].to_list() == ["yes", "yes", "no"]
    assert run(_composite("not", _match("match")), df)["r"].to_list() == ["no", "no", "yes"]


@pytest.mark.parametrize("mode", MODES)
def test_strict_null_handling_raises_for_numbers_and_strings(mode):
    numbers = _tree(lt("x", 50.0), null_handling="error")
    with pytest.raises(MissingInputError, match="'x'"):
        Engine().bind(numbers, mode=mode).run(pl.DataFrame({"x": _col(10.0, None)}))
    strings = _tree(_match("error"))
    with pytest.raises(MissingInputError, match="'s'"):
        Engine().bind(strings, mode=mode).run(pl.DataFrame({"s": pl.Series(["a", None], dtype=pl.String)}))


@pytest.mark.parametrize("mode", MODES)
def test_score_routes_a_null_like_run(mode):
    exe = Engine().bind(_composite("or", lt("x", 50.0), lt("y", 50.0)), mode=mode)
    assert exe.score({"x": None, "y": 10.0})["r"] == "yes"
    assert exe.score({"x": None, "y": 90.0})["r"] == "no"
