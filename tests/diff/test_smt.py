"""prove with z3: trees, decision tables and computed features, checked against the real steps."""
import random
import sys

import polars as pl
import pytest

pytest.importorskip("z3")

from decider import diff, flow  # noqa: E402
from decider.steps.tables import DecisionTableConfig  # noqa: E402
from decider.steps.trees import TreeConfig  # noqa: E402

OUT = {"data": [{"o": 0}, {"o": 1}], "default": {"o": -1}, "dtypes": [["o", "Int64"]]}


def cond(feature, op="<=", threshold=0):
    return {"type": "unary", "condition": {"op": op, "feature": feature, "threshold": threshold}}


def leaf(idx):
    return {"type": "leaf", "result_idx": idx}


def tree(spec, name="t", **extra):
    """`spec` is a leaf index, or (data, then, otherwise) with each branch a spec."""
    nodes, edges = [], []

    def add(s):
        nid = f"n{len(nodes)}"
        nodes.append({"id": nid, "data": leaf(s) if isinstance(s, int) else s[0]})
        if not isinstance(s, int):
            for i, branch in enumerate(s[1:]):
                edges.append({"source": nid, "target": add(branch), "data": {"sourceIndex": i}})
        return nid

    add(spec)
    return TreeConfig.load({"type": "tree", "name": name, **extra, "tree": {"nodes": nodes, "edges": edges, "output": OUT}})


def table(rows, expression=None, **extra):
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "t", "columns": {"g": "String", "o": "Float64"}, "rows": rows,
        "expression": expression or {"type": "eq", "variable": "grade", "value_column": "g"},
        "outputs": ["o"], "default": [0.0], **extra})


def frame_of(row):
    return pl.DataFrame({k: [v] for k, v in row.items()}, schema={
        k: pl.String if isinstance(v, str) else pl.Int64 if isinstance(v, int) else pl.Float64
        for k, v in row.items()})


def test_trees_with_the_same_decisions_in_a_different_layout_are_proven_identical():
    either = (cond("x", ">", 5), 1, (cond("y", ">", 3), 1, 0))
    swapped = (cond("y", ">", 3), 1, (cond("x", ">", 5), 1, 0))
    proof = diff.prove(tree(either), tree(swapped))
    assert (proof.tier, proof.identical) == ("smt", True)
    assert proof.evidence.steps == ("t",) and "float64" in proof.evidence.modelled


def test_a_moved_threshold_gives_a_counterexample_the_real_steps_disagree_on():
    old, new = tree((cond("x", ">", 5), 1, 0)), tree((cond("x", ">", 6), 1, 0))
    proof = diff.prove(old, new)
    assert (proof.tier, proof.identical) == ("smt", None)
    row = proof.evidence.counterexample
    assert 5 < row["x"] <= 6
    assert old.run(frame_of(row))["o"].to_list() != new.run(frame_of(row))["o"].to_list()


def test_an_int_feature_compares_exactly_so_greater_than_five_is_at_least_six():
    ints = {"feature_types": {"n": "int"}}
    assert diff.prove(tree((cond("n", ">", 5), 1, 0), **ints), tree((cond("n", ">=", 6), 1, 0), **ints)).identical is True
    proof = diff.prove(tree((cond("n", ">", 5), 1, 0), **ints), tree((cond("n", ">=", 7), 1, 0), **ints))
    assert proof.identical is None and proof.evidence.counterexample == {"n": 6}


def test_not_over_the_opposite_comparison_is_identical_even_for_a_null():
    negated = {"type": "composite", "op": "not", "conditions": [cond("x", "<=", 5)["condition"]]}
    assert diff.prove(tree((cond("x", ">", 5), 1, 0)), tree((negated, 1, 0))).identical is True


def test_a_null_taking_another_branch_is_a_counterexample():
    # A null is unknown, so it takes the otherwise branch: 1 for `x <= 5`, 0 for `x > 5`.
    proof = diff.prove(tree((cond("x", "<=", 5), 0, 1)), tree((cond("x", ">", 5), 1, 0)))
    assert proof.identical is None and proof.evidence.counterexample["x"] is None


def test_the_leaf_id_output_distinguishes_trees_that_answer_from_different_leaves():
    one = tree((cond("x", ">", 5), 1, 0), path_output="leaf")
    other = tree((cond("x", ">", 5), 1, 0), path_output="leaf")
    assert diff.prove(one, other).identical is True
    swapped = tree((cond("x", "<=", 5), 0, 1), path_output="leaf")
    assert diff.prove(one, swapped).identical is None


def test_computed_features_compare_in_float64():
    a = tree((cond({"type": "computed", "expression": "x - y"}, ">", 1.0), 1, 0))
    b = tree((cond({"type": "computed", "expression": "x + -y"}, ">", 1.0), 1, 0))
    c = tree((cond({"type": "computed", "expression": "x - y"}, ">", 2.0), 1, 0))
    assert diff.prove(a, b).identical is True
    proof = diff.prove(a, c)
    assert proof.identical is None and 1.0 < proof.evidence.counterexample["x"] - proof.evidence.counterexample["y"] <= 2.0


def test_a_division_in_a_computed_feature_falls_back_with_a_reason():
    ratio = lambda n: tree((cond({"type": "computed", "expression": "x / y"}, ">", n), 1, 0))  # noqa: E731
    proof = diff.prove(ratio(1.0), ratio(2.0))
    assert (proof.tier, proof.identical) == ("none", None) and "not modelled" in proof.reason


def test_string_matches_exact_and_contains_differ_on_a_string_around_the_pattern():
    def match(kind):
        return tree(({"type": "unary", "condition": {"op": "string_match", "feature": "s", "patterns": ["ab"],
                                                      "match_type": kind}}, 1, 0))
    assert diff.prove(match("exact"), match("exact")).identical is True
    proof = diff.prove(match("exact"), match("contains"))
    assert proof.identical is None and "ab" in proof.evidence.counterexample["s"]


def test_tables_with_reordered_non_overlapping_rows_are_proven_identical():
    rows = [{"g": "A", "o": 1.0}, {"g": "B", "o": 2.0}, {"g": "C", "o": 3.0}]
    proof = diff.prove(table(rows), table(rows[::-1]))
    assert (proof.tier, proof.identical) == ("smt", True)


def test_tables_with_overlapping_rows_in_another_order_have_a_real_counterexample():
    band = {"type": "between", "variable": "score", "lower_bound_column": "lo", "upper_bound_column": "hi",
            "allow_gaps": True}
    rows = [{"lo": 0.0, "hi": 10.0, "o": 1.0}, {"lo": 5.0, "hi": 20.0, "o": 2.0}]

    def banded(r):
        return DecisionTableConfig.load({
            "type": "decision_table", "name": "t", "columns": {"lo": "Float64", "hi": "Float64", "o": "Float64"},
            "rows": r, "expression": band, "outputs": ["o"], "default": [0.0]})

    old, new = banded(rows), banded(rows[::-1])
    proof = diff.prove(old, new)
    assert proof.identical is None
    row = proof.evidence.counterexample
    assert 5 <= row["score"] < 10
    assert old.run(frame_of(row))["o"].to_list() != new.run(frame_of(row))["o"].to_list()


def test_a_step_that_is_plain_python_falls_back_with_a_reason():
    def versions(offset):
        def f(x: float) -> float:
            return x + offset
        return flow(f, name="p")

    proof = diff.prove(versions(1.0), versions(2.0))
    assert (proof.tier, proof.identical) == ("none", None)
    assert "tree or a decision table" in proof.reason


def test_a_tree_with_a_param_falls_back_with_a_reason():
    def param_tree(default):
        return tree((cond("x", ">", {"param": "limit", "default": default}), 1, 0))

    proof = diff.prove(param_tree(5.0), param_tree(6.0))
    assert proof.identical is None and "param" in proof.reason


def test_without_z3_the_proof_stops_at_the_hash_and_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "z3", None)
    proof = diff.prove(tree((cond("x", ">", 5), 1, 0)), tree((cond("x", ">", 6), 1, 0)))
    assert (proof.tier, proof.identical) == ("none", None) and "decider[prove]" in proof.reason


def _random_spec(rng, depth=2):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice([0, 1, -1])
    return (cond(rng.choice("xy"), rng.choice(["<", "<=", ">", ">=", "==", "!="]), rng.randint(1, 3)),
            _random_spec(rng, depth - 1), _random_spec(rng, depth - 1))


def _split_leaves(rng, spec):
    # The same decisions with some leaves behind a test that sends both ways to them.
    if isinstance(spec, int):
        return (cond("x", ">", rng.randint(1, 3)), spec, spec) if rng.random() < 0.5 else spec
    return (spec[0], _split_leaves(rng, spec[1]), _split_leaves(rng, spec[2]))


def test_the_verdict_matches_running_both_trees_over_every_region_of_the_inputs():
    # Thresholds are whole numbers 1..3, so half-steps around each (and a null) reach every region.
    values = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    grid = pl.DataFrame([{"x": x, "y": y} for x in values for y in values], schema={"x": pl.Float64, "y": pl.Float64})
    rng = random.Random(7)
    verdicts = set()
    for _ in range(25):
        a = _random_spec(rng)
        b = _random_spec(rng) if rng.random() < 0.5 else _split_leaves(rng, a)
        if isinstance(a, int) or isinstance(b, int):
            continue
        old, new = tree(a), tree(b)
        same = old.run(grid)["o"].to_list() == new.run(grid)["o"].to_list()
        proof = diff.prove(old, new)
        verdicts.add(proof.identical)
        if proof.tier == "ir_hash":
            assert same
        else:
            assert proof.tier == "smt" and (proof.identical is True) == same, (a, b)
            assert (proof.evidence.counterexample is None) == same
    assert verdicts >= {True, None}
