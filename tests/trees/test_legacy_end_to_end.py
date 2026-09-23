"""Larger rule flows from the polars rule engine, run through `TreeConfig` in every mode: nested trees,
prioritized rules in `first_match` and `all` mode, multi-column outputs and empty cases."""
import polars as pl
import pydantic
import pytest

from decider.engine import Engine
from decider.exceptions import MissingInputError
from decider.steps.trees import TreeConfig


def leaf(i=-1):
    return {"type": "leaf", "result_idx": i}


def unary(cond, then, otherwise, **extra):
    return {"type": "unary", "condition": cond, "then": then, "otherwise": otherwise, **extra}


def lt(feature, threshold):
    return {"op": "<", "feature": feature, "threshold": threshold}


def cases(op, feature, whens, branches, **extra):
    conditions = [{"when": w, "then": i} for i, w in enumerate(whens)]
    return {"type": "cases", "op": op, "feature": feature, "conditions": conditions, "otherwise": len(whens),
            "branches": branches, **extra}


def ranges(feature, bands, branches):
    return cases("ranges", feature, bands, branches, end_logic="lower_inclusive", strict=False)


def output(*labels, default="default"):
    return {"data": [{"r": x} for x in labels], "default": {"r": default}, "dtypes": [["r", "String"]]}


def flat(rule, out):
    return TreeConfig(name="t", tree={"type": "flat_rule", "rule": {"rule": rule}, "output": out})


def prioritized(rules, out, mode="first_match"):
    return TreeConfig(name="t", tree={"type": "prioritized_flat_rule", "mode": mode, "output": out,
                                      "rules": [{"meta": {"name": n}, "rule": r} for n, r in rules]})


# --- nested trees ------------------------------------------------------------------------------

SCORE_BUCKETS = ranges("score", [{"max": 40.0}, {"min": 40.0, "max": 70.0}, {"min": 70.0}],
                       [leaf(0), leaf(1), leaf(2), leaf(-1)])
YOUNG_BY_SCORE = unary(lt("age", 30.0), SCORE_BUCKETS, leaf(3))
YOUNG_OUT = output("young_low", "young_mid", "young_high", "adult", default="unknown")
YOUNG_DF = pl.DataFrame({"age": [20.0, 25.0, 25.0, 35.0], "score": [20.0, 55.0, 80.0, 99.0]})


def test_a_unary_rule_can_lead_to_a_cases_rule(run):
    assert run(flat(YOUNG_BY_SCORE, YOUNG_OUT), YOUNG_DF)["r"].to_list() == [
        "young_low", "young_mid", "young_high", "adult"]


def test_the_same_nested_tree_as_v3_and_as_flat_rules_gives_identical_outputs(run):
    v3 = TreeConfig(name="t", tree={
        "nodes": [
            {"id": "root", "data": {"type": "unary", "condition": lt("age", 30.0)}},
            {"id": "buckets", "data": {"type": "cases", "op": "ranges", "feature": "score", "strict": False,
                                       "conditions": [{"max": 40.0}, {"min": 40.0, "max": 70.0}, {"min": 70.0}]}},
            *({"id": f"l{i}", "data": leaf(i)} for i in range(4)),
        ],
        "edges": [
            {"source": "root", "target": "buckets", "data": {"sourceIndex": 0}},
            {"source": "root", "target": "l3", "data": {"sourceIndex": 1}},
            *({"source": "buckets", "target": f"l{i}", "data": {"sourceIndex": i}} for i in range(3)),
        ],
        "output": YOUNG_OUT,
    })
    assert run(v3, YOUNG_DF).equals(run(flat(YOUNG_BY_SCORE, YOUNG_OUT), YOUNG_DF))


def test_a_cases_branch_can_be_a_string_match_rule(run):
    region = unary({"op": "string_match", "feature": "region", "patterns": ["rural", "farm"], "match_type": "exact"},
                   leaf(0), leaf(1))
    root = ranges("income", [{"max": 50_000.0}, {"min": 50_000.0}], [region, leaf(2), leaf(-1)])
    df = pl.DataFrame({"income": [30_000.0, 30_000.0, 80_000.0], "region": ["rural", "city", "city"]})
    out = run(flat(root, output("rural_low", "urban_low", "wealthy", default="unknown")), df)
    assert out["r"].to_list() == ["rural_low", "urban_low", "wealthy"]


def test_a_three_level_tree_routes_every_row(run):
    status = unary({"op": "string_match", "feature": "status", "patterns": ["active"], "match_type": "exact"},
                   leaf(2), leaf(3))
    root = unary(lt("age", 18.0), leaf(0), unary(lt("score", 50.0), status, leaf(4)))
    df = pl.DataFrame({"age": [15.0, 25.0, 25.0, 30.0], "score": [99.0, 30.0, 30.0, 75.0],
                       "status": ["x", "active", "inactive", "active"]})
    out = run(flat(root, output("minor", "unused", "active_low", "inactive_low", "high_score", default="unknown")), df)
    assert out["r"].to_list() == ["minor", "active_low", "inactive_low", "high_score"]


def test_a_composite_rule_works_inside_a_nested_tree(run):
    vip = {"type": "composite", "op": "and",
           "conditions": [{"op": ">", "feature": "score", "threshold": 60.0},
                          {"op": "string_match", "feature": "status", "patterns": ["vip"], "match_type": "exact"}],
           "then": leaf(1), "otherwise": leaf(2)}
    root = unary({"op": ">=", "feature": "age", "threshold": 18.0}, vip, leaf(0))
    df = pl.DataFrame({"age": [15.0, 25.0, 25.0, 30.0], "score": [90.0, 80.0, 80.0, 40.0],
                       "status": ["vip", "vip", "basic", "vip"]})
    out = run(flat(root, output("minor", "vip_adult", "regular_adult", default="unknown")), df)
    assert out["r"].to_list() == ["minor", "vip_adult", "regular_adult", "regular_adult"]


# --- prioritized rules: `all` mode ---------------------------------------------------------------


def test_all_mode_returns_each_rules_result_under_its_name(run):
    fraud = unary({"op": ">", "feature": "amount", "threshold": 10_000.0}, leaf(0), leaf(-1))
    vip = unary({"op": "string_match", "feature": "tier", "patterns": ["gold", "platinum"], "match_type": "exact"},
                leaf(1), leaf(-1))
    df = pl.DataFrame({"amount": [500.0, 15_000.0, 15_000.0], "tier": ["gold", "silver", "gold"]})
    tree = prioritized([("fraud_flag", fraud), ("vip_flag", vip)], output("high_amount", "vip", default="none"), "all")
    out = run(tree, df)
    assert out["fraud_flag.r"].to_list() == ["none", "high_amount", "high_amount"]
    assert out["vip_flag.r"].to_list() == ["vip", "none", "vip"]


def test_all_mode_gives_every_rule_the_default_when_nothing_matches(run):
    r1 = unary(lt("x", 0.0), leaf(0), leaf(-1))
    r2 = unary({"op": ">", "feature": "x", "threshold": 100.0}, leaf(0), leaf(-1))
    out = run(prioritized([("a", r1), ("b", r2)], output("match", default="none"), "all"), pl.DataFrame({"x": [50.0, 50.0]}))
    assert out["a.r"].to_list() == ["none", "none"]
    assert out["b.r"].to_list() == ["none", "none"]


def test_an_unnamed_rule_in_all_mode_is_named_by_its_position(run):
    r = unary(lt("x", 0.0), leaf(0), leaf(-1))
    tree = TreeConfig(name="t", tree={"type": "prioritized_flat_rule", "mode": "all", "output": output("neg"),
                                      "rules": [{"rule": r}, {"rule": r}]})
    out = run(tree, pl.DataFrame({"x": [-1.0, 1.0]}))
    assert out["rule_0.r"].to_list() == out["rule_1.r"].to_list() == ["neg", "default"]


# --- which nodes a row passes (the reference walker's visits) -----------------------------------


def _visits(tree, record):
    exe = Engine().bind(tree)
    seen = []
    exe.runner.visit = seen.append
    exe.score(record)
    return seen


def test_a_nested_tree_visits_one_node_per_decision_and_the_leaf():
    inner = unary(lt("score", 50.0), leaf(0), leaf(1))
    tree = flat(unary(lt("age", 30.0), inner, leaf(2)), output("young_low", "young_high", "adult"))
    assert _visits(tree, {"age": 20.0, "score": 30.0}) == ["0", "0.0", "0.0.0"]
    assert _visits(tree, {"age": 20.0, "score": 70.0}) == ["0", "0.0", "0.0.1"]
    assert _visits(tree, {"age": 40.0, "score": 99.0}) == ["0", "0.1"]


def test_a_cases_rule_visits_the_branch_of_the_matched_condition_or_otherwise():
    tree = flat(cases("isin", "code", [{"values": [1]}, {"values": [2]}], [leaf(0), leaf(1), leaf(-1)]),
                output("one", "two", default="other"))
    assert [_visits(tree, {"code": c})[-1] for c in (1.0, 2.0, 9.0)] == ["0.0", "0.1", "0.2"]


def test_a_missing_numeric_input_is_an_error_under_strict_null_handling():
    tree = TreeConfig(name="t", null_handling="error", tree={
        "type": "flat_rule", "rule": {"rule": unary(lt("v", 50.0), leaf(0), leaf(-1))},
        "output": output("low", default="other")})
    for mode in ("interpreted", "stepped", "fused"):
        with pytest.raises(MissingInputError):
            Engine().bind(tree, mode=mode).run(pl.DataFrame({"v": pl.Series([30.0, None], dtype=pl.Float64)}))


# --- multi-column outputs -----------------------------------------------------------------------

MULTI = {
    "data": [{"label": "low", "score": 0.1, "flag": False}, {"label": "mid", "score": 0.5, "flag": False},
             {"label": "high", "score": 0.9, "flag": True}],
    "default": {"label": "none", "score": 0.0, "flag": False},
    "dtypes": [["label", "String"], ["score", "Float64"], ["flag", "Boolean"]],
}


def test_every_output_column_comes_from_the_reached_row(run):
    root = ranges("x", [{"max": 30.0}, {"min": 30.0, "max": 70.0}, {"min": 70.0, "max": 100.0}],
                  [leaf(0), leaf(1), leaf(2), leaf(-1)])
    out = run(flat(root, MULTI), pl.DataFrame({"x": [10.0, 50.0, 80.0, 200.0]}))
    assert out["label"].to_list() == ["low", "mid", "high", "none"]
    assert out["score"].to_list() == pytest.approx([0.1, 0.5, 0.9, 0.0])
    assert out["flag"].to_list() == [False, False, True, False]


def test_every_output_column_comes_from_the_default_row_when_nothing_matches(run):
    out = run(flat(unary(lt("x", 0.0), leaf(0), leaf(-1)), MULTI), pl.DataFrame({"x": [50.0, 100.0]}))
    assert out["label"].to_list() == ["none", "none"]
    assert out["score"].to_list() == [0.0, 0.0]
    assert out["flag"].to_list() == [False, False]


def test_first_match_with_several_output_columns_picks_the_winning_rules_row(run):
    r1 = unary({"op": ">=", "feature": "x", "threshold": 80.0}, leaf(2), leaf(-1))
    r2 = unary({"op": ">=", "feature": "x", "threshold": 30.0}, leaf(1), leaf(-1))
    out = run(prioritized([("high", r1), ("mid", r2)], MULTI), pl.DataFrame({"x": [90.0, 50.0, 10.0]}))
    assert out["label"].to_list() == ["high", "mid", "none"]
    assert out["flag"].to_list() == [True, False, False]


# --- edge shapes ----------------------------------------------------------------------------------


@pytest.mark.parametrize("op, feature, df", [
    ("ranges", "x", pl.DataFrame({"x": [1.0, 2.0]})),
    ("string_match", "s", pl.DataFrame({"s": ["a", "b"]})),
    ("isin", "x", pl.DataFrame({"x": [1.0, 2.0]})),
])
def test_a_cases_rule_without_conditions_always_takes_otherwise(run, op, feature, df):
    rule = {"type": "cases", "op": op, "feature": feature, "conditions": [], "otherwise": 0, "branches": [leaf(-1)]}
    if op == "ranges":
        rule["strict"] = False
    assert run(flat(rule, output(default="always")), df)["r"].to_list() == ["always", "always"]


def test_a_composite_rule_without_conditions_is_rejected():
    rule = {"type": "composite", "op": "and", "conditions": [], "then": leaf(0), "otherwise": leaf(-1)}
    with pytest.raises(pydantic.ValidationError, match="at least one condition"):
        flat(rule, output("never", default="always"))


def test_params_referenced_inside_cases_rules_are_the_steps_params():
    lo, hi = {"key": "lo"}, {"key": "hi"}
    trees = {
        ("lo", "hi"): cases("ranges", "x", [{"min": lo, "max": hi}], [leaf(0), leaf(-1)], strict=False),
        ("pat",): cases("string_match", "s", [{"patterns": [{"key": "pat"}]}], [leaf(0), leaf(-1)]),
        ("allowed",): cases("isin", "x", [{"values": {"key": "allowed"}}], [leaf(0), leaf(-1)]),
    }
    for names, rule in trees.items():
        assert set(flat(rule, output("hit")).parameters()["t"]) == set(names)
