"""Decision tables: matching, defaults, bounds, strings, lists, nulls, validation and composition."""
import polars as pl
import pytest

from decider import flow
from decider.engine import Engine
from decider.steps.tables import BoundMode, DecisionTableConfig
from decider.steps.trees import RangeEndLogic

BANDS = [{"lo": None, "hi": 30.0, "pts": 1}, {"lo": 30.0, "hi": 70.0, "pts": 2}, {"lo": 70.0, "hi": None, "pts": 3}]


def bands(rows=None, name="bands", variable="score", **between) -> DecisionTableConfig:
    return DecisionTableConfig(
        name=name, columns={"lo": "Float64", "hi": "Float64", "pts": "Int64"}, rows=BANDS if rows is None else rows,
        expression={"type": "between", "variable": variable, "lower_bound_column": "lo", "upper_bound_column": "hi",
                    **between},
        outputs=["pts"], default=[0])


def test_no_match_takes_the_default(run):
    table = bands([{"lo": 10.0, "hi": 90.0, "pts": 7}], allow_gaps=True)
    assert run(table, pl.DataFrame({"score": [50.0, 5.0, 200.0]}))["pts"].to_list() == [7, 0, 0]


def test_upper_inclusive_moves_the_closed_end(run):
    frame = pl.DataFrame({"score": [30.0, 70.0]})
    assert run(bands(), frame)["pts"].to_list() == [2, 3]
    assert run(bands(mode="upper_inclusive"), frame)["pts"].to_list() == [1, 2]


def test_bound_mode_is_the_trees_range_end_logic():
    assert BoundMode is RangeEndLogic


def test_the_first_matching_row_wins(run):
    table = DecisionTableConfig(
        name="first", columns={"key": "String", "pts": "Int64"},
        rows=[{"key": "a", "pts": 1}, {"key": "a", "pts": 2}, {"key": "b", "pts": 3}],
        expression={"type": "eq", "variable": "k", "value_column": "key"}, outputs=["pts"], default=[0])
    assert run(table, pl.DataFrame({"k": ["a", "b", "c"]}))["pts"].to_list() == [1, 3, 0]


def test_or_of_two_bands(run):
    table = DecisionTableConfig(
        name="either", columns={"lo": "Float64", "hi": "Float64", "alt_lo": "Float64", "alt_hi": "Float64",
                                "pts": "Int64"},
        rows=[{"lo": 0.0, "hi": 10.0, "alt_lo": 90.0, "alt_hi": 100.0, "pts": 1}],
        expression={"type": "or", "expressions": [
            {"type": "between", "variable": "v", "lower_bound_column": "lo", "upper_bound_column": "hi",
             "allow_gaps": True},
            {"type": "between", "variable": "v", "lower_bound_column": "alt_lo", "upper_bound_column": "alt_hi",
             "allow_gaps": True}]},
        outputs=["pts"], default=[0])
    assert run(table, pl.DataFrame({"v": [5.0, 50.0, 95.0]}))["pts"].to_list() == [1, 0, 1]


def _bureau(rows=None) -> DecisionTableConfig:
    return DecisionTableConfig(
        name="bureau", columns={"key": "String", "pts": "Int64"},
        rows=rows or [{"key": "experian", "pts": 10}, {"key": "transunion", "pts": 20}],
        expression={"type": "eq", "variable": "BureauKey", "value_column": "key"}, outputs=["pts"], default=[0])


def test_eq_on_a_string_column(run):
    frame = pl.DataFrame({"BureauKey": ["experian", "transunion", "other", None]})
    assert run(_bureau(), frame)["pts"].to_list() == [10, 20, 0, 0]


@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_score_takes_a_string_input(mode):
    exe = Engine().bind(_bureau(), mode=mode)
    assert exe.score({"BureauKey": "experian"})["pts"] == 10
    assert exe.score({"BureauKey": "equifax"})["pts"] == 0


def test_in_on_numeric_sets(run):
    table = DecisionTableConfig(
        name="regions", columns={"vals": {"type": "List", "inner": "Float64"}, "pts": "Int64"},
        rows=[{"vals": [1.0, 2.0, 3.0], "pts": 5}, {"vals": [9.0], "pts": 6}, {"vals": [], "pts": 7}],
        expression={"type": "in", "variable": "region", "values_column": "vals"}, outputs=["pts"], default=[0])
    assert run(table, pl.DataFrame({"region": [2.0, 9.0, 4.0]}))["pts"].to_list() == [5, 6, 0]


def test_and_of_between_and_is_true(run):
    table = DecisionTableConfig(
        name="elig", columns={"age_lo": "Float64", "age_hi": "Float64", "pts": "Int64"},
        rows=[{"age_lo": 18.0, "age_hi": 65.0, "pts": 1}],
        expression={"type": "and", "expressions": [
            {"type": "between", "variable": "age", "lower_bound_column": "age_lo", "upper_bound_column": "age_hi",
             "allow_gaps": True},
            {"type": "is_true", "variable": "verified"}]},
        outputs=["pts"], default=[0])
    frame = pl.DataFrame({"age": [30.0, 17.0, 40.0, 70.0, 30.0], "verified": [True, True, False, True, None]})
    assert run(table, frame)["pts"].to_list() == [1, 0, 0, 0, 0]


def test_a_null_input_or_a_null_row_value_never_matches(run):
    table = DecisionTableConfig(
        name="nulls", columns={"v": "Float64", "pts": "Int64"}, rows=[{"v": None, "pts": 1}, {"v": 2.0, "pts": 2}],
        expression={"type": "eq", "variable": "x", "value_column": "v"}, outputs=["pts"], default=[0])
    assert run(table, pl.DataFrame({"x": pl.Series([2.0, None, 5.0])}))["pts"].to_list() == [2, 0, 0]


def test_an_int_column_is_compared_as_a_number_and_outputs_keep_their_types(run):
    table = DecisionTableConfig(
        name="typed", columns={"lo": "Int64", "hi": "Int64", "rate": "Float64", "ok": "Boolean", "n": "Int64"},
        rows=[{"lo": 0, "hi": 10, "rate": 0.5, "ok": True, "n": 1}, {"lo": 10, "hi": 20, "rate": 1.5, "ok": False,
                                                                       "n": 2}],
        expression={"type": "between", "variable": "months", "lower_bound_column": "lo", "upper_bound_column": "hi"},
        outputs=["rate", "ok", "n"], default=[0.0, False, 0])
    out = run(table, pl.DataFrame({"months": pl.Series([3, 12, 25], dtype=pl.Int64)}))
    assert out.select("rate", "ok", "n").rows() == [(0.5, True, 1), (1.5, False, 2), (0.0, False, 0)]
    assert out.schema["rate"] == pl.Float64 and out.schema["ok"] == pl.Boolean and out.schema["n"] == pl.Int64


def test_no_default_writes_nulls(run):
    table = DecisionTableConfig(
        name="nodefault", columns={"lo": "Float64", "hi": "Float64", "pts": "Int64", "label": "String"},
        rows=[{"lo": 0.0, "hi": 1.0, "pts": 4, "label": "in"}],
        expression={"type": "between", "variable": "v", "lower_bound_column": "lo", "upper_bound_column": "hi"},
        outputs=["pts", "label"])
    out = run(table, pl.DataFrame({"v": [0.5, 3.0]}))
    assert out.select("pts", "label").rows() == [(4, "in"), (None, None)]


def test_a_table_composes_with_ordinary_steps(run):
    def disposable_income(net_income: float, expenses: float) -> float:
        return net_income - expenses

    def final_score(pts: int) -> float:
        return pts * 10.0

    table = DecisionTableConfig(
        name="afford_bands", columns={"lo": "Float64", "hi": "Float64", "pts": "Int64"},
        rows=[{"lo": None, "hi": 1000.0, "pts": 1}, {"lo": 1000.0, "hi": None, "pts": 5}],
        expression={"type": "between", "variable": "disposable_income", "lower_bound_column": "lo",
                    "upper_bound_column": "hi"},
        outputs=["pts"], default=[0])
    frame = pl.DataFrame({"net_income": [5000.0, 2000.0], "expenses": [1500.0, 1500.0]})
    assert run(flow(disposable_income, table, final_score), frame)["final_score"].to_list() == [50.0, 10.0]


def test_the_python_matcher_visits_each_row_it_tries():
    exe = Engine().bind(bands())
    seen = []
    exe.runner.visit = seen.append
    exe.score({"score": 50.0})
    assert seen == ["0", "1"]


@pytest.mark.parametrize("change, message", [
    ({"rows": [{"lo": None, "hi": 30.0, "pts": 1}, {"lo": 50.0, "hi": None, "pts": 2}]}, "ranges are not contiguous"),
    ({"outputs": ["missing_column"]}, "not found in parameters columns"),
    ({"default": [0, 1]}, "must match outputs length"),
    ({"expression": {"type": "between", "variable": "v", "lower_bound_column": "nope"}}, "not found"),
    ({"expression": {"type": "between", "variable": "v"}}, "At least one of"),
    ({"expression": {"type": "in", "variable": "v", "values_column": "lo"}}, "must be a list type"),
    ({"rows": [{"lo": "abc", "hi": 1.0, "pts": 1}]}, "don't fit the declared columns"),
    ({"rows": [{"lo": 0.0, "hi": 1.0, "pts": 1, "extra": 2}]}, "does not declare"),
    ({"expression": {"type": "and", "expressions": [
        {"type": "between", "variable": "v", "lower_bound_column": "lo", "upper_bound_column": "hi"},
        {"type": "is_true", "variable": "v"}]}}, "one variable has one type"),
])
def test_a_malformed_table_is_rejected_when_it_loads(change, message):
    doc = {"type": "decision_table", "name": "dt", "columns": {"lo": "Float64", "hi": "Float64", "pts": "Int64"},
           "rows": BANDS, "outputs": ["pts"], "default": [0],
           "expression": {"type": "between", "variable": "v", "lower_bound_column": "lo", "upper_bound_column": "hi"}}
    with pytest.raises(ValueError, match=message):
        DecisionTableConfig.load({**doc, **change})


def test_a_table_round_trips_through_json(run):
    table = bands()
    again = DecisionTableConfig.load(table.model_dump_json())
    # ponytail: compares dumps; the rows' DataFrame model has no usable ==.
    assert again.model_dump() == table.model_dump()
    frame = pl.DataFrame({"score": [10.0, 50.0, 90.0]})
    assert run(again, frame).equals(run(table, frame))
