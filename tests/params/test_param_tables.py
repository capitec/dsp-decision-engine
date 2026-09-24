"""`param_table()`: a plain step's table-valued param, rows from the params document."""
import numpy as np
import polars as pl
import pytest

from decider import Engine, ParamRef, Table, flow, param, param_table
from decider.exceptions import ParamsError
from decider.steps.tables import TableRef
from decider.testing import no_recompile

ROWS = [{"floor": 0, "rate": 0.2}, {"floor": 600, "rate": 0.1}]


def rate(score: int, rates: Table = param_table({"floor": int, "rate": float}, default=ROWS)) -> float:
    r = 0.0
    for i in range(len(rates.floor)):
        if score >= rates.floor[i]:
            r = rates.rate[i]
    return r


def fee(amount: float, fees: Table = param_table({"cap": float, "on": bool}, required=True)) -> float:
    total = 0.0
    for i in range(len(fees.cap)):
        if fees.on[i]:
            total += min(amount, fees.cap[i])
    return total


def points(score: int, bands: Table = param_table({"floor": int, "rate": float}, shared_key="bands",
                                                  default=ROWS)) -> float:
    return rate(score, bands) * 100


def other(score: int, bands: Table = param_table({"floor": int, "rate": float}, shared_key="bands",
                                                 default=ROWS)) -> float:
    return rate(score, bands) * 100


def doc(rows, step="rate", param="rates"):
    return {"p": {step: {param: rows}}}


PIPELINE = flow(rate, name="p")
FRAME = pl.DataFrame({"score": [100, 700]})


def test_the_step_receives_read_only_column_arrays():
    seen = []

    def spy(score: int, rates: Table = param_table({"floor": int, "rate": float}, default=ROWS)) -> float:
        seen.append(rates)
        return 0.0

    flow(spy, name="p").run(FRAME.head(1))
    (t,) = seen
    assert t._fields == ("floor", "rate") and t.floor.dtype == np.int64 and t.rate.dtype == np.float64
    assert not t.floor.flags.writeable


def test_a_direct_call_uses_the_default_rows():
    assert rate(700) == 0.1


def test_parameters_report_the_schema_and_default_rows():
    schema = PIPELINE.parameters()
    assert schema == {"p/rate": {"rates": {"type": "table", "schema": {"floor": "int", "rate": "float"},
                                           "default": ROWS}}}
    assert schema.defaults() == doc(ROWS)
    rates = schema.json_schema()["properties"]["p"]["properties"]["rate"]["properties"]["rates"]
    assert rates["items"]["properties"] == {"floor": {"type": "integer"}, "rate": {"type": "number"}}
    assert rates["default"] == ROWS


def test_a_required_table_shows_as_no_rows_in_the_defaults():
    schema = flow(fee, name="p").parameters()
    assert schema["p/fee"]["fees"]["required"] is True
    assert schema.defaults() == doc([], "fee", "fees")
    assert flow(fee, name="p").run(pl.DataFrame({"amount": [5.0]}), params=schema.defaults())["fee"].to_list() == [0.0]


def test_a_missing_required_table_names_its_schema():
    with pytest.raises(ParamsError, match=r"param 'fees' is required but missing; expected a list of rows like "
                                          r'\[\{"cap": float, "on": bool\}, \.\.\.\]'):
        flow(fee, name="p").run(pl.DataFrame({"amount": [5.0]}))


@pytest.mark.parametrize("rows, error", [
    ([{"floor": 0}], "row 0, column 'rate': Field required; expected a list of rows"),
    ([{"floor": 0, "rate": 0.1}, {"floor": 1.5, "rate": 0.1}], "row 1, column 'floor': Input should be a valid integer"),
    ([{"floor": 0, "rate": 0.1, "cap": 3}], "row 0, column 'cap': Extra inputs"),
    ({"floor": 0}, "p/rate: param 'rates': Input should be a valid list"),
    ([{"floor": 2 ** 70, "rate": 0.1}], "column 'floor' holds an int that doesn't fit in int64"),
])
def test_bad_rows_name_the_step_param_row_and_column(rows, error):
    with pytest.raises(ParamsError, match=error.replace("[", r"\[")):
        PIPELINE.run(FRAME, params=doc(rows))


def test_an_int_is_accepted_for_a_float_column():
    assert PIPELINE.run(FRAME, params=doc([{"floor": 0, "rate": 1}]))["rate"].to_list() == [1.0, 1.0]


def test_a_missing_nested_field_is_not_reported_as_a_missing_param():
    with pytest.raises(ParamsError) as e:
        PIPELINE.run(FRAME, params=doc([{"floor": 0}]))
    assert "is required but missing" not in str(e.value)


def test_a_shared_table_is_read_from_shared():
    both = flow(points, other, name="p")
    schema = both.parameters()
    assert schema["shared"]["bands"]["used_by"] == ["p/points", "p/other"]
    assert schema.defaults() == {"shared": {"bands": ROWS}}
    out = both.run(FRAME, params={"shared": {"bands": [{"floor": 0, "rate": 0.5}]}})
    assert out["points"].to_list() == [50.0, 50.0] and out["other"].to_list() == [50.0, 50.0]


@pytest.mark.parametrize("columns, error", [
    ({"a-b": int}, "'a_b'"), ({"name": str}, "int, float or bool"), ({}, "at least one column"),
])
def test_bad_columns_are_refused_at_declaration(columns, error):
    with pytest.raises(TypeError, match=error):
        param_table(columns, default=[])


def test_default_rows_are_checked_at_declaration():
    with pytest.raises(ValueError, match="valid integer"):
        param_table({"a": int}, default=[{"a": "x"}])


@pytest.mark.parametrize("ref", [lambda: ParamRef(param="hi-cut"), lambda: TableRef(table="my-rows")])
def test_a_non_identifier_param_name_is_refused_with_a_suggestion(ref):
    with pytest.raises(ValueError, match="_"):
        ref()


@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_rows_are_retuned_in_every_mode_without_recompiling(mode):
    exe = Engine(strict_compile=True).bind(flow(rate, name="p"), mode=mode)
    assert exe.run(FRAME)["rate"].to_list() == [0.2, 0.1]
    assert exe.score({"score": 700})["rate"] == 0.1
    with no_recompile():
        more = [{"floor": 0, "rate": 0.3}, {"floor": 50, "rate": 0.25}, {"floor": 650, "rate": 0.05}]
        assert exe.run(FRAME, params=doc(more))["rate"].to_list() == [0.25, 0.05]
        assert exe.score({"score": 60}, doc(more))["rate"] == 0.25
        assert exe.score({"score": 60}, doc([]))["rate"] == 0.0


def capped(score: int, rates: Table = param_table({"floor": int, "rate": float}, default=ROWS),
           cap: float = param(0.15)) -> float:
    r = 0.0
    for i in range(len(rates.floor)):
        if score >= rates.floor[i]:
            r = rates.rate[i]
    return min(r, cap)


def test_a_table_and_a_plain_param_mix_in_a_fused_kernel():
    exe = Engine(strict_compile=True).bind(flow(capped, name="p"), mode="fused")
    assert exe.run(FRAME)["capped"].to_list() == [0.15, 0.1]
