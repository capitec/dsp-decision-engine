"""Rows from the params document: checked when the document arrives, and never a recompile."""
import polars as pl
import pytest

from decider import flow
from decider.engine import Engine
from decider.exceptions import ParamsError
from decider.steps.tables import DecisionTableConfig
from decider.testing import no_recompile

COLUMNS = {"lo": "Float64", "hi": "Float64", "band": {"type": "Enum", "categories": ["low", "mid", "high"]},
           "pts": "Int64", "keys": {"type": "List", "inner": "String"}}
EXPRESSION = {"type": "and", "expressions": [
    {"type": "between", "variable": "score", "lower_bound_column": "lo", "upper_bound_column": "hi"},
    {"type": "in", "variable": "channel", "values_column": "keys"}]}
THREE = [{"lo": None, "hi": 30.0, "band": "low", "pts": 1, "keys": ["app", "web"]},
         {"lo": 30.0, "hi": 70.0, "band": "mid", "pts": 2, "keys": ["app"]},
         {"lo": 70.0, "hi": None, "band": "high", "pts": 3, "keys": ["app", "branch"]}]
FIVE = [{"lo": None, "hi": 10.0, "band": "high", "pts": 9, "keys": ["web"]},
        {"lo": 10.0, "hi": 20.0, "band": "mid", "pts": 8, "keys": ["web"]},
        {"lo": 20.0, "hi": 40.0, "band": "low", "pts": 7, "keys": ["app"]},
        {"lo": 40.0, "hi": 80.0, "band": None, "pts": None, "keys": ["app"]},
        {"lo": 80.0, "hi": None, "band": "mid", "pts": 5, "keys": ["branch"]}]
FRAME = pl.DataFrame({"score": [5.0, 25.0, 50.0, 90.0, None], "channel": ["web", "app", "app", "branch", "app"]})


def table(**change) -> DecisionTableConfig:
    doc = {"type": "decision_table", "name": "bands", "columns": COLUMNS, "rows": {"table": "rows"},
           "expression": EXPRESSION, "outputs": ["band", "pts"], "default": ["low", 0], **change}
    return DecisionTableConfig.load(doc)


def rows(data: list) -> dict:
    return {"bands": {"rows": data}}


def test_rows_come_from_the_params_document(run):
    out = run(table(), FRAME, rows(THREE))
    assert out.select("band", "pts").rows() == [("low", 1), ("low", 1), ("mid", 2), ("high", 3), ("low", 0)]
    out = run(table(), FRAME, rows(FIVE))
    assert out.select("band", "pts").rows() == [("high", 9), ("low", 7), (None, None), ("mid", 5), ("low", 0)]


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_retuning_rows_and_their_count_never_recompiles(mode):
    exe = Engine().bind(table(), mode=mode)
    exe.run(FRAME, params=rows(THREE))
    exe.score({"score": 5.0, "channel": "web"}, params=rows(THREE))
    with no_recompile():
        assert exe.run(FRAME, params=rows(FIVE))["pts"].to_list() == [9, 7, None, 5, 0]
        assert exe.score({"score": 90.0, "channel": "branch"}, params=rows(FIVE))["pts"] == 5
        assert exe.run(FRAME, params=rows(THREE[:1]))["pts"].to_list() == [1, 1, 0, 0, 0]
        assert exe.run(FRAME, params=rows([]))["pts"].to_list() == [0, 0, 0, 0, 0]


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_editing_inline_rows_rebuilds_the_node_but_never_recompiles(mode):
    def inline(data):
        return DecisionTableConfig.load({"type": "decision_table", "name": "bands", "columns": COLUMNS,
                                         "rows": data, "expression": EXPRESSION, "outputs": ["pts"],
                                         "default": [0]})

    # A null among inline outputs would make the output `int | None`, a new type; these rows have none.
    five = [{**r, "band": r["band"] or "low", "pts": r["pts"] or 6} for r in FIVE]
    Engine().bind(inline(THREE), mode=mode).run(FRAME)
    with no_recompile():
        assert Engine().bind(inline(five), mode=mode).run(FRAME)["pts"].to_list() == [9, 7, 6, 5, 0]


@pytest.mark.parametrize("data, message", [
    ([{**THREE[0], "band": "other"}], "don't fit the declared columns"),
    ([{**THREE[0], "lo": "abc"}], "don't fit the declared columns"),
    ([{**THREE[0], "surplus": 1}], "does not declare"),
    ([{k: v for k, v in THREE[0].items() if k != "pts"}], "lack the declared column"),
    ([THREE[0], THREE[2]], "ranges are not contiguous"),
])
def test_rows_that_break_the_declared_columns_are_invalid_params(data, message):
    with pytest.raises(ParamsError, match=message):
        Engine().bind(table()).run(FRAME, params=rows(data))


def test_missing_rows_are_invalid_params():
    with pytest.raises(ParamsError, match="required"):
        Engine().bind(table()).run(FRAME)


def test_a_string_output_of_param_rows_must_declare_its_values():
    with pytest.raises(ValueError, match="Enum"):
        Engine().bind(table(columns={**COLUMNS, "band": "String"}))


def test_the_param_carries_its_schema():
    schema = table().parameters()["bands"]["rows"]
    assert schema["type"] == "table"
    assert set(schema["schema"]) == set(COLUMNS)


def test_a_shared_table_feeds_two_tables(run):
    grid = {"table": "grid", "shared": True}
    both = flow(table(rows=grid, outputs=["band"], default=["low"]),
                table(name="points", rows=grid, outputs=["pts"], default=[0]))
    out = run(both, FRAME, {"shared": {"grid": THREE}})
    assert out.select("band", "pts").rows() == [("low", 1), ("low", 1), ("mid", 2), ("high", 3), ("low", 0)]
