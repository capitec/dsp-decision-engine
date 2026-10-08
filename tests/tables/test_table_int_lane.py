"""Decision tables compare integer inputs as int64: exact above 2**53, and shareable with int steps."""
import polars as pl
import pytest

from decider import flow, step
from decider._engine import Engine
from decider.exceptions import WiringError
from decider.steps.tables import DecisionTableConfig

BIG = 2**60


def by_code(rows, variable="code", op="eq", **extra) -> DecisionTableConfig:
    column = {"eq": {"value_column": "code"}, "in": {"values_column": "codes"}}[op]
    return DecisionTableConfig(
        name="by_code", columns={"code": "Int64", "codes": {"type": "List", "inner": "Int64"}, "pts": "Int64"},
        rows=rows, expression={"type": op, "variable": variable, **column}, outputs=["pts"], default=[0], **extra)


EQ_ROWS = [{"code": 7, "codes": None, "pts": 1}, {"code": BIG, "codes": None, "pts": 2},
           {"code": None, "codes": None, "pts": 3}]
IN_ROWS = [{"code": None, "codes": [7, 8], "pts": 1}, {"code": None, "codes": [BIG, BIG + 2], "pts": 2},
           {"code": None, "codes": None, "pts": 3}]


def test_a_table_and_an_int_step_share_an_int_column(run):
    @step(output="doubled")
    def double(code: int) -> int:
        return code * 2

    pipeline = flow(double, by_code(EQ_ROWS), name="lookup")
    out = run(pipeline, pl.DataFrame({"code": [7, BIG, 9]}))
    assert out["pts"].to_list() == [1, 2, 0]


def test_a_float_step_feeding_an_int_table_is_refused():
    @step(output="code")
    def code_f(x: float) -> float:
        return x

    with pytest.raises(WiringError, match="lookup/by_code reads 'code' as int, but lookup/code_f writes it as float"):
        Engine().bind(flow(code_f, by_code(EQ_ROWS), name="lookup"))


def test_int_eq_tells_apart_ids_above_2_53(run):
    out = run(by_code(EQ_ROWS), pl.DataFrame({"code": [BIG, BIG + 1, BIG - 1]}))
    assert out["pts"].to_list() == [2, 0, 0]


def test_int_in_tells_apart_ids_above_2_53(run):
    out = run(by_code(IN_ROWS, op="in"), pl.DataFrame({"code": [BIG, BIG + 1, BIG + 2, 8, 9]}))
    assert out["pts"].to_list() == [2, 0, 2, 1, 0]


@pytest.mark.parametrize("op, rows, expected", [("eq", EQ_ROWS, [1, 0, 0]), ("in", IN_ROWS, [1, 0, 0])])
def test_a_null_int_input_never_matches(run, op, rows, expected):
    frame = pl.DataFrame({"code": pl.Series([7, None, 9], dtype=pl.Int64)})
    assert run(by_code(rows, op=op), frame)["pts"].to_list() == expected


def test_a_null_int_input_does_not_match_a_null_row_value(run):
    frame = pl.DataFrame({"code": pl.Series([None], dtype=pl.Int64)})
    assert run(by_code([{"code": None, "codes": None, "pts": 3}]), frame)["pts"].to_list() == [0]


def test_an_absent_int_column_is_null_and_matches_nothing(run):
    assert run(by_code(EQ_ROWS), pl.DataFrame({"other": [1, 2]}))["pts"].to_list() == [0, 0]


def test_an_eq_variable_also_in_a_between_stays_float(run):
    table = DecisionTableConfig(
        name="mixed", columns={"code": "Int64", "lo": "Float64", "hi": "Float64", "pts": "Int64"},
        rows=[{"code": 1, "lo": None, "hi": 10.0, "pts": 1}, {"code": 1, "lo": 10.0, "hi": None, "pts": 2}],
        expression={"type": "and", "expressions": [
            {"type": "eq", "variable": "x", "value_column": "code"},
            {"type": "between", "variable": "x", "lower_bound_column": "lo", "upper_bound_column": "hi",
             "allow_gaps": True}]},
        outputs=["pts"], default=[0])
    assert run(table, pl.DataFrame({"x": [1.0, 1.5, 2.0]}))["pts"].to_list() == [1, 0, 0]


def test_an_input_also_used_by_a_between_stays_float():
    table = DecisionTableConfig(
        name="mixed", columns={"code": "Int64", "lo": "Float64", "hi": "Float64", "pts": "Int64"},
        rows=[{"code": 1, "lo": 0.0, "hi": 10.0, "pts": 1}],
        expression={"type": "or", "expressions": [
            {"type": "eq", "variable": "x", "value_column": "code"},
            {"type": "between", "variable": "x", "lower_bound_column": "lo", "upper_bound_column": "hi",
             "allow_gaps": True}]},
        outputs=["pts"], default=[0])

    @step(output="y")
    def uses_int(x: int) -> int:
        return x

    with pytest.raises(WiringError, match="'x' as int, but .* as float"):
        Engine().bind(flow(uses_int, table, name="p"))


def test_int_eq_and_string_eq_on_one_variable_is_refused():
    with pytest.raises(ValueError, match="'k' is compared as int and as str; one variable has one type"):
        DecisionTableConfig(
            name="clash", columns={"code": "Int64", "text": "String", "pts": "Int64"},
            rows=[{"code": 1, "text": "a", "pts": 1}],
            expression={"type": "and", "expressions": [
                {"type": "eq", "variable": "k", "value_column": "code"},
                {"type": "eq", "variable": "k", "value_column": "text"}]},
            outputs=["pts"])


def test_an_int_eq_table_compiles_without_falling_back():
    exe = Engine(strict_compile=True).bind(by_code(EQ_ROWS), mode="fused")
    assert exe.run(pl.DataFrame({"code": [BIG, BIG + 1]}))["pts"].to_list() == [2, 0]


def test_int_rows_from_params_are_matched_exactly(run):
    table = DecisionTableConfig(
        name="ids", columns={"code": "Int64", "pts": "Int64"}, rows={"table": "ids"},
        expression={"type": "eq", "variable": "code", "value_column": "code"}, outputs=["pts"], default=[0])
    params = {"ids": {"ids": [{"code": BIG, "pts": 5}, {"code": None, "pts": 6}]}}
    out = run(table, pl.DataFrame({"code": [BIG, BIG + 1, None]}), params=params)
    assert out["pts"].to_list() == [5, 0, 0]
