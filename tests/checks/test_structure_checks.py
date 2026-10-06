"""The structure checks DCD201-208: each fires on its defect and stays quiet on clean code."""
import sys

import polars as pl
import pytest

from decider import branch, flow, frame_step, loop, param, step
from decider.lab import checks
from decider.ops.decimal import Currency
from decider.steps.tables import DecisionTableConfig
from decider.steps.trees import TreeConfig

OUT = {"data": [{"o": 0}, {"o": 1}], "default": {"o": -1}, "dtypes": [["o", "Int64"]]}
STRUCTURE = ["unreachable_branch", "dead_output", "shadowed_write", "unused_input", "unbounded_loop",
             "opaque_frame_step"]


def _found(pipeline, alias):
    return [f for f in checks.run(pipeline, STRUCTURE).findings if f.check == alias]


def _tree(spec):
    """`spec` is a leaf index, or (condition, then, otherwise), each a spec."""
    nodes, edges = [], []

    def add(s):
        nid = f"n{len(nodes)}"
        nodes.append({"id": nid, "data": {"type": "leaf", "result_idx": s} if isinstance(s, int)
                      else {"type": "unary", "condition": s[0]}})
        if not isinstance(s, int):
            for i, child in enumerate(s[1:]):
                edges.append({"source": nid, "target": add(child), "data": {"sourceIndex": i}})
        return nid

    add(spec)
    return TreeConfig.load({"type": "tree", "name": "t", "tree": {"nodes": nodes, "edges": edges, "output": OUT}})


def _gt(threshold):
    return {"op": ">", "feature": "x", "threshold": threshold}


def _bands(rows, default=None):
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "bands", "columns": {"lo": "Float64", "hi": "Float64", "band": "String"},
        "rows": rows, "outputs": ["band"], "default": default,
        "expression": {"type": "between", "variable": "score", "lower_bound_column": "lo",
                       "upper_bound_column": "hi", "allow_gaps": True}})


@step(output="x")
def double(x: float) -> float:
    return x * 2


# -- DCD201 --

def test_constant_condition_leaves_an_arm_unreachable():
    def always(x: float) -> bool:
        return True

    @step(output="y")
    def low(x: float) -> float:
        return x

    @step(output="y")
    def high(x: float) -> float:
        return x + 1

    (f,) = _found(flow(branch(always, high, low, modifies=["y"], name="pick"), name="term"), "unreachable_branch")
    assert f.code == "DCD201" and f.detail["index"] == 1 and "term/pick/low" in f.message and f.line


def test_a_condition_that_depends_on_data_keeps_both_arms():
    def big(x: float) -> bool:
        return x > 5

    @step(output="y")
    def low(x: float) -> float:
        return x

    @step(output="y")
    def high(x: float) -> float:
        return x + 1

    assert not _found(flow(branch(big, high, low, modifies=["y"], name="pick"), name="term"), "unreachable_branch")


def test_an_index_with_no_arm_is_reported():
    def which(x: float) -> int:
        if x > 1:
            return 2
        return 0

    def constant(x: float) -> int:
        return 5

    @step(output="y")
    def a(x: float) -> float:
        return x

    @step(output="y")
    def b(x: float) -> float:
        return -x

    for cond in (which, constant):
        (f,) = [f for f in _found(flow(branch(cond, a, b, modifies=["y"], name="pick"), name="t"),
                                  "unreachable_branch") if "arms" in f.message]
        assert f.detail["index"] in (2, 5)


def test_a_tree_branch_no_input_can_take_is_reported():
    # Below the root's otherwise branch x <= 5 holds, so x > 10 never does.
    tree = _tree((_gt(5), 1, (_gt(10), 1, 0)))
    pytest.importorskip("z3")
    (f,) = _found(flow(tree, name="term"), "unreachable_branch")
    assert (f.code, f.detail["tree_node"], f.detail["branch"]) == ("DCD201", "n2", 0)
    assert f.step.path == "term/t"


def test_a_tree_whose_branches_all_matter_is_quiet():
    pytest.importorskip("z3")
    assert not _found(flow(_tree((_gt(5), (_gt(10), 1, 0), 0)), name="term"), "unreachable_branch")


def test_without_z3_a_tree_is_skipped_not_an_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "decider.lab.diff._smt", raising=False)
    monkeypatch.setitem(sys.modules, "z3", None)
    report = checks.run(flow(_tree((_gt(5), 1, 0)), name="term"), ["unreachable_branch"])
    assert not report.findings
    (s,) = report.skipped
    assert (s.code, "decider[prove]" in s.reason, "term/t" in s.reason) == ("DCD201", True, True)


def test_a_tree_the_encoding_cannot_model_is_skipped_with_the_reason():
    pytest.importorskip("z3")
    tree = TreeConfig.load({**_tree((_gt(5), 1, 0)).model_dump(mode="json"), "null_handling": "error"})
    report = checks.run(flow(tree, name="term"), ["unreachable_branch"])
    assert not report.findings and "null_handling" in report.skipped[0].reason


def test_a_table_row_the_rows_above_cover_is_reported():
    pytest.importorskip("z3")
    rows = [{"lo": 0.0, "hi": 100.0, "band": "wide"}, {"lo": 10.0, "hi": 20.0, "band": "inside"}]
    dead = [f for f in _found(flow(_bands(rows), name="term"), "unreachable_branch")]
    assert [f.detail["row"] for f in dead] == [1] and "term/bands#1" in dead[0].message


def test_a_table_with_both_ends_covered_has_no_findings():
    pytest.importorskip("z3")
    rows = [{"lo": None, "hi": 30.0, "band": "low"}, {"lo": 30.0, "hi": None, "band": "high"}]
    report = checks.run(flow(_bands(rows), name="term"), ["unreachable_branch", "table_overlap", "table_gap"])
    assert not report.findings and not report.skipped


def test_rows_that_can_both_match_overlap():
    pytest.importorskip("z3")
    rows = [{"lo": 0.0, "hi": 10.0, "band": "a"}, {"lo": 5.0, "hi": 15.0, "band": "b"}]
    found = _found_all(flow(_bands(rows), name="term"), "table_overlap")
    (f,) = found
    assert (f.code, f.detail["rows"]) == ("DCD207", [0, 1]) and 5.0 <= f.detail["example"]["score"] < 10.0


def test_disabling_table_overlap_drops_its_findings_but_not_the_gap(tmp_path):
    pytest.importorskip("z3")
    rows = [{"lo": 0.0, "hi": 10.0, "band": "a"}, {"lo": 5.0, "hi": 15.0, "band": "b"}]
    path = tmp_path / "checks.yaml"
    path.write_text("checks:\n  - {type: unreachable_branch}\ndisable: [table_overlap]\n")
    codes = {f.code for f in checks.run(flow(_bands(rows), name="term"), config=path).findings}
    assert "DCD207" not in codes and "DCD208" in codes


def test_an_input_no_row_matches_is_a_gap_and_a_warning_without_default():
    pytest.importorskip("z3")
    rows = [{"lo": 0.0, "hi": 10.0, "band": "a"}, {"lo": 20.0, "hi": 30.0, "band": "b"}]
    (f,) = _found_all(flow(_bands(rows), name="term"), "table_gap")
    assert f.code == "DCD208" and f.severity is checks.Severity.WARNING
    score = f.detail["example"]["score"]
    assert not (0.0 <= score < 10.0 or 20.0 <= score < 30.0)
    (g,) = _found_all(flow(_bands(rows, default=["none"]), name="term"), "table_gap")
    assert g.severity is checks.Severity.INFO


def _found_all(pipeline, alias):
    return [f for f in checks.run(pipeline, STRUCTURE).findings if f.check == alias]


def test_a_table_with_rows_from_params_is_skipped():
    pytest.importorskip("z3")
    table = DecisionTableConfig.load({
        "type": "decision_table", "name": "bands", "columns": {"lo": "Float64", "hi": "Float64", "band": {"type": "Enum", "categories": ["a"]}},
        "rows": {"table": "prices"}, "outputs": ["band"],
        "expression": {"type": "between", "variable": "score", "lower_bound_column": "lo",
                       "upper_bound_column": "hi"}, "default": None})
    report = checks.run(flow(table, name="term"), STRUCTURE)
    assert not [f for f in report.findings if f.code in ("DCD201", "DCD207", "DCD208")]
    assert "params" in report.skipped[0].reason


# -- DCD202 / DCD203 --

def _arm_with_scratch():
    @step(output="scratch")
    def scratch(x: float) -> float:
        return x * 3

    @step(output="y")
    def high(x: float) -> float:
        return x + 1

    @step(output="y")
    def low(x: float) -> float:
        return x

    def big(x: float) -> bool:
        return x > 5

    return branch(big, flow(scratch, high, name="up"), low, modifies=["y"], name="pick")


def test_a_value_trapped_in_a_branch_arm_is_dead():
    (f,) = _found(flow(_arm_with_scratch(), name="term"), "dead_output")
    assert (f.code, f.detail["name"], f.step.path) == ("DCD202", "scratch", "term/pick/up/scratch")


def test_final_values_and_merged_values_are_not_dead():
    report = checks.run(flow(double, _arm_with_scratch(), name="term"), STRUCTURE)
    assert [f.detail["name"] for f in report.findings if f.check == "dead_output"] == ["scratch"]
    assert not [f for f in report.findings if f.check == "shadowed_write"]


def test_a_write_overwritten_before_it_is_read_is_shadowed():
    @step(output="z")
    def first(x: float) -> float:
        return x

    @step(output="z")
    def second(x: float) -> float:
        return x + 1

    (f,) = _found(flow(first, second, name="term"), "shadowed_write")
    assert (f.code, f.step.path, f.detail["overwritten_by"]) == ("DCD203", "term/first", "term/second")
    assert not _found(flow(first, second, name="term"), "dead_output")


def test_a_write_that_is_read_before_the_next_write_is_quiet():
    @step(output="z")
    def first(x: float) -> float:
        return x

    @step(output="z")
    def second(z: float) -> float:
        return z + 1

    assert not _found(flow(first, second, name="term"), "shadowed_write")


# -- DCD204 --

def test_an_argument_the_body_ignores_is_reported():
    @step(output="y")
    def ignores(x: float, spare: float) -> float:
        return x

    (f,) = _found(flow(ignores, name="term"), "unused_input")
    assert (f.code, f.detail["argument"], f.severity) == ("DCD204", "spare", checks.Severity.INFO) and f.line


def test_arguments_that_are_all_used_are_quiet():
    assert not _found(flow(double, name="term"), "unused_input")


# -- DCD205 --

@step(output="n")
def tick(n: int) -> int:
    return n + 1


def _loop(condition, max_iterations=10):
    return flow(loop(condition, tick, carries=["n"], max_iterations=max_iterations, name="count"), name="term")


def test_a_loop_condition_that_ignores_the_carries_is_reported():
    def forever(limit: int) -> bool:
        return limit > 0

    (f,) = _found(_loop(forever), "unbounded_loop")
    assert f.code == "DCD205" and f.detail["carries"] == ["n"]


def test_a_huge_max_iterations_is_reported():
    def below(n: int) -> bool:
        return n < 5

    (f,) = _found(_loop(below, 1_000_000), "unbounded_loop")
    assert f.detail["max_iterations"] == 1_000_000


def test_a_bounded_loop_over_its_carries_is_quiet():
    def below(n: int) -> bool:
        return n < 5

    assert not _found(_loop(below), "unbounded_loop")


# -- DCD206 --

def test_a_frame_step_with_unknown_lineage_is_reported():
    @frame_step
    def opaque(df: pl.DataFrame) -> pl.DataFrame:
        return df

    (f,) = _found(flow(opaque, name="term"), "opaque_frame_step")
    assert (f.code, f.step.path) == ("DCD206", "term/opaque")


def test_a_frame_step_declaring_its_columns_is_quiet():
    @frame_step(reads=["x"], writes=["y"])
    def known(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(y=pl.col("x"))

    assert not _found(flow(known, name="term"), "opaque_frame_step")


# -- the whole family --

def test_the_money_example_has_no_structure_findings():
    def inst(amount: Currency, rate: float = param(0.1)) -> Currency:
        return round(amount * rate / 12, 2)

    def ratio(a: float, b: float) -> float:
        return a / b

    assert checks.run(flow(inst, ratio, name="term"), STRUCTURE).findings == ()


def test_structure_checks_are_in_standard_and_suppressible_by_code():
    @step(output="y", suppress_checks=["DCD204"])
    def ignores(x: float, spare: float) -> float:
        return x

    report = checks.run(flow(ignores, name="term"))
    (f,) = [f for f in report.findings if f.code == "DCD204"]
    assert f.suppressed is not None
    assert not [f for f in report.findings if f.code == "DCD403"]
