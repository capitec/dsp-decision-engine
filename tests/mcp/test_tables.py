from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from decider._tooling.mcp import _configs as configs, _scorecards as scorecards, _tables as tables

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "example_projects/07-credit-limit-management/sonnet/configs/0.1.0/matrix.json"

BAND = {"type": "decision_table", "name": "band", "columns": {"lo": "Float64", "hi": "Float64", "band": "String"},
        "rows": [{"lo": None, "hi": 600.0, "band": "C"}, {"lo": 600.0, "hi": 700.0, "band": "B"},
                 {"lo": 700.0, "hi": None, "band": "A"}],
        "expression": {"type": "between", "variable": "bureau_score", "lower_bound_column": "lo", "upper_bound_column": "hi"},
        "outputs": ["band"], "default": ["C"]}

PRICES = {"type": "decision_table", "name": "price", "columns": {"product": "String", "rate": "Float64"},
          "rows": {"table": "prices"}, "expression": {"type": "eq", "variable": "product", "value_column": "product"},
          "outputs": ["rate"], "default": [0.0]}

CARD = {"type": "scorecard", "name": "card", "variables": [{
    "type": "scored", "variable_name": "age", "default": {"value": 0},
    "bins": [{"value": 5, "upper_bound": {"param": "young", "default": 25}},
             {"value": 10, "lower_bound": {"param": "young", "default": 25}}]}]}


def _write(tmp_path, doc, name="doc.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return str(path)


def _score(path, *scores):
    return [r["band"] for r in configs.try_config(path, [{"bureau_score": s} for s in scores])["rows"]]


def test_table_view_shows_the_match_rule_and_rows_with_indexes(tmp_path):
    view = tables.table_view(_write(tmp_path, BAND))
    assert view["match"].startswith("row.lo <= bureau_score < row.hi")
    assert view["rows_in"] == "document" and view["row_count"] == 3 and view["rows"][1]["band"] == "B"
    assert "| 1 | 600.0 | 700.0 | B |" in view["rows_markdown"] and "| 0 | null | 600.0 | C |" in view["rows_markdown"]


def test_table_view_of_params_rows_points_at_params_set(tmp_path):
    view = tables.table_view(_write(tmp_path, PRICES))
    assert view["rows_in"] == "params" and "params_set" in view["edit"] and "rows" not in view


def test_table_view_summarises_and_or_expressions(tmp_path):
    doc = {**BAND, "columns": {**BAND["columns"], "product": "String"},
           "rows": [{**r, "product": "loan"} for r in BAND["rows"]],
           "expression": {"type": "and", "expressions": [
               {"type": "eq", "variable": "product_code", "value_column": "product"}, BAND["expression"]]}}
    assert tables.table_view(_write(tmp_path, doc))["match"].startswith("product_code == row.product AND row.lo <=")


def test_added_rows_change_the_answer(tmp_path):
    path = _write(tmp_path, BAND)
    assert _score(path, 640.0, 800.0) == ["B", "A"]
    # open a hole in the 700+ band by hand, then fill it with the tool
    doc = json.loads(Path(path).read_text())
    doc["rows"][2]["lo"] = 800.0
    Path(path).write_text(json.dumps(doc))
    assert tables.table_add_rows(path, [{"lo": 700.0, "hi": 800.0, "band": "A-"}], at=2)["saved"]
    assert _score(path, 750.0, 850.0) == ["A-", "A"]


def test_adding_a_row_that_leaves_a_gap_is_refused_and_the_file_is_unchanged(tmp_path):
    path = _write(tmp_path, BAND)
    before = Path(path).read_text()
    result = tables.table_add_rows(path, [{"lo": 900.0, "hi": 950.0, "band": "X"}], at=1)
    assert not result["saved"] and "contiguous" in result["problems"][0]
    assert Path(path).read_text() == before


def test_updated_cells_change_the_answer(tmp_path):
    path = _write(tmp_path, BAND)
    result = tables.table_update_rows(path, [{"index": 1, "values": {"band": "B+"}}])
    assert result["saved"] and _score(path, 650.0) == ["B+"]


def test_update_with_an_unknown_column_is_refused(tmp_path):
    path = _write(tmp_path, BAND)
    result = tables.table_update_rows(path, [{"index": 1, "values": {"bnad": "B"}}])
    assert not result["saved"] and "bnad" in result["problems"][0]
    assert tables.table_update_rows(path, [{"index": 9, "values": {}}])["problems"] == ["row index 9 is outside 0..2"]


def test_deleting_the_last_row_opens_the_top_band(tmp_path):
    path = _write(tmp_path, BAND)
    doc = json.loads(Path(path).read_text())
    doc["rows"][1]["hi"] = None
    Path(path).write_text(json.dumps(doc))
    assert tables.table_delete_rows(path, [2])["saved"]
    assert _score(path, 650.0, 900.0) == ["B", "B"]


def test_delete_rows_rejects_an_index_outside_the_table(tmp_path):
    result = tables.table_delete_rows(_write(tmp_path, BAND), [7])
    assert not result["saved"] and "outside" in result["problems"][0]


def test_added_output_column_is_in_every_row_the_outputs_and_the_default(tmp_path):
    path = _write(tmp_path, BAND)
    assert tables.table_add_column(path, "limit", "Float64", value=1000.0, output=True)["saved"]
    doc = json.loads(Path(path).read_text())
    assert doc["outputs"] == ["band", "limit"] and doc["default"] == ["C", 1000.0]
    assert all(r["limit"] == 1000.0 for r in doc["rows"])
    assert configs.try_config(path, [{"bureau_score": 650.0}])["rows"][0]["limit"] == 1000.0
    assert tables.table_update_rows(path, [{"index": 1, "values": {"limit": 2000.0}}])["saved"]
    assert configs.try_config(path, [{"bureau_score": 650.0}])["rows"][0]["limit"] == 2000.0


def test_adding_an_existing_column_is_refused(tmp_path):
    result = tables.table_add_column(_write(tmp_path, BAND), "band", "String")
    assert not result["saved"] and "already exists" in result["problems"][0]


def test_dropping_a_column_the_expression_uses_is_refused(tmp_path):
    path = _write(tmp_path, BAND)
    before = Path(path).read_text()
    result = tables.table_drop_column(path, "lo")
    assert not result["saved"] and "used by the match expression" in result["problems"][0]
    assert Path(path).read_text() == before


def test_dropping_a_spare_column_removes_it_everywhere(tmp_path):
    path = _write(tmp_path, BAND)
    tables.table_add_column(path, "limit", "Float64", value=1.0, output=True)
    assert tables.table_drop_column(path, "limit")["saved"]
    doc = json.loads(Path(path).read_text())
    assert "limit" not in doc["columns"] and doc["outputs"] == ["band"] and doc["default"] == ["C"]
    assert all("limit" not in r for r in doc["rows"])


def test_row_edits_on_a_params_table_point_at_params_set(tmp_path):
    result = tables.table_add_rows(_write(tmp_path, PRICES), [{"product": "x", "rate": 1.0}])
    assert not result["saved"] and "params_set" in result["problems"][0]


def test_params_rows_are_tried_with_params(tmp_path):
    path = _write(tmp_path, PRICES)
    params = {"price": {"prices": [{"product": "loan", "rate": 0.12}]}}
    rows = configs.try_config(path, [{"product": "loan"}, {"product": "car"}], params)["rows"]
    assert [r["rate"] for r in rows] == [0.12, 0.0]


def test_edit_of_the_example_matrix_changes_the_looked_up_multiplier(tmp_path):
    path = str(shutil.copy(MATRIX, tmp_path / "matrix.json"))
    cell = {"product_code": 20, "behaviour_grade": 1, "utilisation_band": 1, "mob_band": 1}
    assert configs.try_config(path, [cell])["rows"][0]["multiplier"] == 1.0037
    view = tables.table_view(path, limit=2)
    assert view["row_count"] == 1152 and len(view["rows"]) == 2
    assert tables.table_update_rows(path, [{"index": 0, "values": {"multiplier": 1.25}}])["saved"]
    assert configs.try_config(path, [cell])["rows"][0]["multiplier"] == 1.25


def test_scorecard_view_shows_bins_defaults_and_params(tmp_path):
    view = scorecards.scorecard_view(_write(tmp_path, CARD))
    age = view["variables"][0]
    assert age["variable"] == "age" and "x <= {young=25}" in age["bins_markdown"] and "| default |" in age["bins_markdown"]
    assert view["params"] == {"card/age": {"young": {"type": "float", "default": 25}}}


def test_new_scorecard_bins_change_the_score(tmp_path):
    path = _write(tmp_path, CARD)
    bins = [{"value": 5, "upper_bound": 25}, {"value": 8, "lower_bound": 25, "upper_bound": 40}, {"value": 12, "lower_bound": 40}]
    assert scorecards.scorecard_set_bins(path, "age", bins)["saved"]
    scores = configs.try_config(path, [{"age": 20.0}, {"age": 30.0}, {"age": 50.0}])["rows"]
    assert [r["score"] for r in scores] == [5.0, 8.0, 12.0]


def test_scorecard_bins_that_overlap_are_refused(tmp_path):
    path = _write(tmp_path, CARD)
    before = Path(path).read_text()
    result = scorecards.scorecard_set_bins(path, "age", [{"value": 1, "upper_bound": 30}, {"value": 2, "lower_bound": 20}])
    assert not result["saved"] and "non-overlapping" in result["problems"][0]
    assert Path(path).read_text() == before


def test_scorecard_set_bins_names_the_variables_when_it_is_unknown(tmp_path):
    result = scorecards.scorecard_set_bins(_write(tmp_path, CARD), "income", [])
    assert not result["saved"] and "['age']" in result["problems"][0]


def test_table_tools_refuse_other_document_types(tmp_path):
    with pytest.raises(ValueError, match="not a decision_table"):
        tables.table_view(_write(tmp_path, CARD))
    with pytest.raises(ValueError, match="not a scorecard"):
        scorecards.scorecard_view(_write(tmp_path, BAND))


def test_one_patch_splits_a_band_where_single_row_edits_would_each_leave_a_gap(tmp_path):
    path = _write(tmp_path, BAND)
    ops = [{"op": "replace", "path": "/rows/2/lo", "value": 800.0},
           {"op": "add", "path": "/rows/2", "value": {"lo": 700.0, "hi": 800.0, "band": "A-"}}]
    assert configs.patch_config(path, ops)["saved"]
    assert _score(path, 750.0, 850.0) == ["A-", "A"]


def test_a_refused_edit_names_the_problem_without_a_pydantic_prefix(tmp_path):
    result = tables.table_add_rows(_write(tmp_path, BAND), [{"lo": 700.0, "hi": 800.0, "band": "A-"}], at=2)
    assert result["problems"][0].startswith("Row 2 upper") and "added_at" not in result
