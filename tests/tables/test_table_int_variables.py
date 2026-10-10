"""A table variable declared `int` in `variable_types` is compared as int64: exact above 2**53, shareable with int steps."""
import math

import polars as pl
import pytest

from decikit import flow, step
from decikit._engine import Engine
from decikit.exceptions import NotIntegralError, WiringError
from decikit.steps.tables import DecisionTableConfig

BIG = 2**60


def by_code(rows, op="eq", declare=True, **extra) -> DecisionTableConfig:
    column = {"eq": {"value_column": "code"}, "in": {"values_column": "codes"}}[op]
    return DecisionTableConfig(
        name="by_code", columns={"code": "Int64", "codes": {"type": "List", "inner": "Int64"}, "pts": "Int64"},
        rows=rows, expression={"type": op, "variable": "code", **column}, outputs=["pts"], default=[0],
        **{"variable_types": {"code": "int"} if declare else {}, **extra})


EQ_ROWS = [{"code": 7, "codes": None, "pts": 1}, {"code": BIG, "codes": None, "pts": 2},
           {"code": None, "codes": None, "pts": 3}]
IN_ROWS = [{"code": None, "codes": [7, 8], "pts": 1}, {"code": None, "codes": [BIG, BIG + 2], "pts": 2},
           {"code": None, "codes": None, "pts": 3}]
BANDS = DecisionTableConfig(
    name="bands", columns={"lo": "Int64", "hi": "Int64", "pts": "Int64"}, variable_types={"n": "int"},
    rows=[{"lo": None, "hi": BIG, "pts": 1}, {"lo": BIG, "hi": BIG + 2, "pts": 2}, {"lo": BIG + 2, "hi": None, "pts": 3}],
    expression={"type": "between", "variable": "n", "lower_bound_column": "lo", "upper_bound_column": "hi"},
    outputs=["pts"], default=[0])


def test_an_undeclared_table_still_conflicts_with_an_int_step():
    @step(output="doubled")
    def double(code: int) -> int:
        return code * 2

    with pytest.raises(WiringError, match="'code' as int, but .* as float"):
        Engine().bind(flow(double, by_code(EQ_ROWS, declare=False), name="p"))


def test_an_undeclared_table_still_conflates_ids_above_2_53(run):
    out = run(by_code(EQ_ROWS, declare=False), pl.DataFrame({"code": [BIG, BIG + 1]}))
    assert out["pts"].to_list() == [2, 2]


def test_a_declared_int_variable_shares_an_int_column_with_a_step(run):
    @step(output="doubled")
    def double(code: int) -> int:
        return code * 2

    out = run(flow(double, by_code(EQ_ROWS), name="lookup"), pl.DataFrame({"code": [7, BIG, 9]}))
    assert out["pts"].to_list() == [1, 2, 0]


def test_int_eq_tells_apart_ids_above_2_53(run):
    out = run(by_code(EQ_ROWS), pl.DataFrame({"code": [BIG, BIG + 1, BIG - 1]}))
    assert out["pts"].to_list() == [2, 0, 0]


def test_int_in_tells_apart_ids_above_2_53(run):
    out = run(by_code(IN_ROWS, op="in"), pl.DataFrame({"code": [BIG, BIG + 1, BIG + 2, 8, 9]}))
    assert out["pts"].to_list() == [2, 0, 2, 1, 0]


def test_int_between_compares_as_ints_above_2_53_with_open_edges(run):
    out = run(BANDS, pl.DataFrame({"n": [-5, BIG - 1, BIG, BIG + 1, BIG + 2, 2**62]}))
    assert out["pts"].to_list() == [1, 1, 2, 2, 3, 3]


def test_int_between_honours_an_upper_inclusive_mode(run):
    table = DecisionTableConfig.load(
        {**BANDS.model_dump(mode="json"), "expression": {**BANDS.expression.model_dump(mode="json"),
                                                          "mode": "upper_inclusive"}})
    assert run(table, pl.DataFrame({"n": [BIG, BIG + 1, BIG + 2]}))["pts"].to_list() == [1, 2, 2]


@pytest.mark.parametrize("op, rows, expected", [("eq", EQ_ROWS, [1, 0, 0]), ("in", IN_ROWS, [1, 0, 0])])
def test_a_null_int_input_never_matches(run, op, rows, expected):
    frame = pl.DataFrame({"code": pl.Series([7, None, 9], dtype=pl.Int64)})
    assert run(by_code(rows, op=op), frame)["pts"].to_list() == expected


def test_a_null_int_input_does_not_match_a_null_row_value(run):
    frame = pl.DataFrame({"code": pl.Series([None], dtype=pl.Int64)})
    assert run(by_code([{"code": None, "codes": None, "pts": 3}]), frame)["pts"].to_list() == [0]


def test_an_absent_int_column_is_null_and_matches_nothing(run):
    assert run(by_code(EQ_ROWS), pl.DataFrame({"other": [1, 2]}))["pts"].to_list() == [0, 0]


def test_the_null_marker_value_never_matches_even_in_a_row(run):
    rows = [{"code": -(2**63), "codes": None, "pts": 1}]
    assert run(by_code(rows), pl.DataFrame({"code": [-(2**63), 0]}))["pts"].to_list() == [0, 0]


def test_a_step_that_writes_a_declared_int_variable_as_float_is_a_wiring_error():
    @step(output="code")
    def to_float(raw: int) -> float:
        return float(raw)

    with pytest.raises(WiringError, match="reads 'code' as int, but .* writes it as float"):
        Engine().bind(flow(to_float, by_code(EQ_ROWS), name="p"))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_nan_or_infinite_float_into_an_int_variable_is_refused(bad):
    exe = Engine().bind(by_code(EQ_ROWS))
    with pytest.raises(Exception, match="(?i)nan|inf|integer|Int64"):
        exe.run(pl.DataFrame({"code": [bad]}))
    with pytest.raises((ValueError, OverflowError)):
        exe.score({"code": bad})


def test_declared_int_over_a_float_column_is_refused():
    with pytest.raises(ValueError, match="'code' is declared int, but column 'f' is float"):
        DecisionTableConfig(
            name="t", columns={"f": "Float64", "pts": "Int64"}, rows=[{"f": 1.0, "pts": 1}], variable_types={"code": "int"},
            expression={"type": "eq", "variable": "code", "value_column": "f"}, outputs=["pts"])


def test_declared_int_with_float_bounds_is_refused():
    with pytest.raises(ValueError, match="'n' is declared int, but column 'hi' is float"):
        DecisionTableConfig(
            name="t", columns={"hi": "Float64", "pts": "Int64"}, rows=[{"hi": 1.0, "pts": 1}], variable_types={"n": "int"},
            expression={"type": "between", "variable": "n", "upper_bound_column": "hi"}, outputs=["pts"])


def test_declared_int_over_a_float_list_is_refused():
    with pytest.raises(ValueError, match="'code' is declared int, but column 'codes' is list\\[float\\]"):
        DecisionTableConfig(
            name="t", columns={"codes": {"type": "List", "inner": "Float64"}, "pts": "Int64"},
            rows=[{"codes": [1.0], "pts": 1}], variable_types={"code": "int"},
            expression={"type": "in", "variable": "code", "values_column": "codes"}, outputs=["pts"])


def test_declaring_a_variable_the_table_does_not_read_is_refused():
    with pytest.raises(ValueError, match=r"variable_types names \['nope'\], which the table does not read"):
        by_code(EQ_ROWS, variable_types={"code": "int", "nope": "int"})


def test_only_int_can_be_declared():
    with pytest.raises(ValueError, match="variable_types declares int only"):
        by_code(EQ_ROWS, variable_types={"code": "float"})


def test_an_unsigned_value_from_2_63_is_refused_when_the_document_loads():
    table = {"type": "decision_table", "name": "t", "columns": {"code": "UInt64", "pts": "Int64"},
             "rows": [{"code": 2**63, "pts": 1}], "outputs": ["pts"], "variable_types": {"code": "int"},
             "expression": {"type": "eq", "variable": "code", "value_column": "code"}}
    with pytest.raises(ValueError, match="2\\*\\*63|9223372036854775808.*does not fit int64"):
        DecisionTableConfig.load(table)
    DecisionTableConfig.load({**table, "variable_types": {}})


def test_an_int_table_compiles_without_falling_back():
    exe = Engine(strict_compile=True).bind(by_code(EQ_ROWS), mode="fused")
    assert exe.run(pl.DataFrame({"code": [BIG, BIG + 1]}))["pts"].to_list() == [2, 0]


def test_int_rows_from_params_are_matched_exactly(run):
    table = DecisionTableConfig(
        name="ids", columns={"code": "Int64", "pts": "Int64"}, rows={"table": "ids"}, variable_types={"code": "int"},
        expression={"type": "eq", "variable": "code", "value_column": "code"}, outputs=["pts"], default=[0])
    params = {"ids": {"ids": [{"code": BIG, "pts": 5}, {"code": None, "pts": 6}]}}
    out = run(table, pl.DataFrame({"code": [BIG, BIG + 1, None]}), params=params)
    assert out["pts"].to_list() == [5, 0, 0]


def test_a_float_variable_beside_an_int_variable_keeps_both_lanes(run):
    table = DecisionTableConfig(
        name="both", columns={"code": "Int64", "lo": "Float64", "hi": "Float64", "pts": "Int64"},
        variable_types={"code": "int"},
        rows=[{"code": BIG, "lo": None, "hi": 1.5, "pts": 1}, {"code": BIG, "lo": 1.5, "hi": None, "pts": 2}],
        expression={"type": "and", "expressions": [
            {"type": "eq", "variable": "code", "value_column": "code"},
            {"type": "between", "variable": "x", "lower_bound_column": "lo", "upper_bound_column": "hi"}]},
        outputs=["pts"], default=[0])
    frame = pl.DataFrame({"code": [BIG, BIG, BIG + 1], "x": [1.0, 2.0, 1.0]})
    assert run(table, frame)["pts"].to_list() == [1, 2, 0]


@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_a_fractional_float_into_an_int_variable_is_refused_and_a_whole_one_matches(mode):
    exe = Engine().bind(by_code(EQ_ROWS), mode=mode)
    with pytest.raises(NotIntegralError, match="input 'code'"):
        exe.score({"code": 7.5})
    with pytest.raises(NotIntegralError, match=r"input 'code' \(row 1\)"):
        exe.run(pl.DataFrame({"code": [7.0, 7.5]}))
    assert exe.score({"code": 7.0})["pts"] == 1


@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_an_int_step_writing_a_fraction_into_an_int_variable_is_refused_naming_the_step(mode):
    @step(output="code")
    def halved(raw: float) -> int:
        return raw / 2

    exe = Engine().bind(flow(halved, by_code(EQ_ROWS), name="p"), mode=mode)
    assert exe.score({"raw": 14.0})["pts"] == 1
    with pytest.raises(NotIntegralError, match="p/halved"):
        exe.score({"raw": 15.0})
