from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from decikit._tooling.mcp import Policy, _configs as configs, _params as params, build_server
from decikit._tooling.mcp._patch import PatchError, apply_patch

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_07 = ROOT / "examples/credit-limit-management"
BUREAU_TREE = ROOT / "ui/vscode/examples/bank/trees/bureau_grade.json"

PIPELINE = '''\
from decikit import flow, param


def ratio(income: float, debt: float) -> float:
    return debt / income


def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit


def build(band):
    return flow(ratio, approved, band, name="credit")
'''

BAND = {"type": "decision_table", "name": "band", "columns": {"lo": "Float64", "hi": "Float64", "band": "String"},
        "rows": [{"lo": None, "hi": 600.0, "band": "C"}, {"lo": 600.0, "hi": 700.0, "band": "B"},
                 {"lo": 700.0, "hi": None, "band": "A"}],
        "expression": {"type": "between", "variable": "bureau_score", "lower_bound_column": "lo", "upper_bound_column": "hi"},
        "outputs": ["band"], "default": ["C"]}


@pytest.fixture()
def project(tmp_path):
    version = tmp_path / "configs" / "0.1.0"
    version.mkdir(parents=True)
    (tmp_path / "pipeline.py").write_text(PIPELINE)
    (version / "band.json").write_text(json.dumps(BAND, indent=2) + "\n")
    (version / "params.json").write_text(json.dumps({"credit": {"approved": {"limit": 0.4}}}, indent=2) + "\n")
    return tmp_path


@pytest.fixture()
def bureau(tmp_path):
    return str(shutil.copy(BUREAU_TREE, tmp_path / "bureau_grade.json"))


def test_list_configs_finds_the_example_projects_documents():
    found = {(c["kind"], c["name"], c["version"], c["key"]) for c in configs.list_configs(str(EXAMPLE_07))["configs"]}
    assert ("decision_table", "limit_assignment_matrix", "0.1.0", "matrix") in found
    assert ("params", "params", "0.1.0", "params") in found


def test_list_configs_skips_files_that_are_not_documents(project):
    (project / "broken.json").write_text("{not json")
    (project / "other.json").write_text('{"type": "not-a-step"}')
    (project / "list.json").write_text("[1, 2]")
    kinds = sorted(c["kind"] for c in configs.list_configs(str(project))["configs"])
    assert kinds == ["decision_table", "params"]


def test_list_configs_ignores_hidden_and_vendored_directories(project):
    for skipped in (".venv", "node_modules", ".scratch"):
        (project / skipped).mkdir()
        (project / skipped / "band.json").write_text(json.dumps(BAND))
    assert len(configs.list_configs(str(project))["configs"]) == 2


def test_patch_that_breaks_the_document_is_refused_and_the_file_is_unchanged(bureau):
    before = Path(bureau).read_text()
    result = configs.patch_config(bureau, [{"op": "replace", "path": "/tree/edges/0/target", "value": "nowhere"}])
    assert not result["saved"] and "nowhere" in result["problems"][0]
    assert Path(bureau).read_text() == before


def test_patch_that_keeps_the_document_valid_is_saved_and_changes_the_answer(bureau):
    rows = [{"bureau_score": 710.0}]
    assert configs.try_config(bureau, rows)["rows"][0]["grade"] == 2
    ops = [{"op": "test", "path": "/tree/nodes/0/data/condition/threshold/default", "value": 720.0},
           {"op": "replace", "path": "/tree/nodes/0/data/condition/threshold/default", "value": 700.0}]
    assert configs.patch_config(bureau, ops)["saved"]
    assert configs.try_config(bureau, rows)["rows"][0]["grade"] == 1


def test_failing_patch_op_names_its_index_and_pointer(bureau):
    ops = [{"op": "replace", "path": "/name", "value": "grade"}, {"op": "remove", "path": "/tree/nodes/99"}]
    problem = configs.patch_config(bureau, ops)["problems"][0]
    assert "op 1" in problem and "/tree/nodes/99" in problem
    assert json.loads(Path(bureau).read_text())["name"] == "bureau_grade"


def test_a_failed_test_op_stops_the_patch(bureau):
    result = configs.patch_config(bureau, [{"op": "test", "path": "/name", "value": "other"}])
    assert not result["saved"] and "test failed" in result["problems"][0]


def test_dry_run_returns_the_patched_document_without_saving(bureau):
    before = Path(bureau).read_text()
    result = configs.patch_config(bureau, [{"op": "replace", "path": "/name", "value": "renamed"}], dry_run=True)
    assert result["doc"]["name"] == "renamed" and result["problems"] == []
    assert Path(bureau).read_text() == before


def test_patch_pointer_escapes_address_keys_with_slash_and_tilde():
    doc = {"a/b": {"c~d": 1}}
    out = apply_patch(doc, [{"op": "replace", "path": "/a~1b/c~0d", "value": 2}])
    assert out == {"a/b": {"c~d": 2}} and doc == {"a/b": {"c~d": 1}}


def test_patch_dash_appends_and_index_inserts():
    out = apply_patch({"xs": [1, 3]}, [{"op": "add", "path": "/xs/-", "value": 4}, {"op": "add", "path": "/xs/1", "value": 2}])
    assert out == {"xs": [1, 2, 3, 4]}


def test_patch_move_copy_remove():
    out = apply_patch({"a": {"x": 1}, "b": []},
                      [{"op": "copy", "from": "/a/x", "path": "/b/-"}, {"op": "move", "from": "/a", "path": "/c"},
                       {"op": "remove", "path": "/b/0"}])
    assert out == {"b": [], "c": {"x": 1}}


@pytest.mark.parametrize("op,fragment", [
    ({"op": "add", "path": "/xs/5", "value": 0}, "out of range"),
    ({"op": "replace", "path": "/missing", "value": 0}, "no key 'missing'"),
    ({"op": "add", "path": "xs", "value": 0}, "starts with '/'"),
    ({"op": "move", "from": "/xs", "path": "/xs/0"}, "into itself"),
    ({"op": "frobnicate", "path": "/xs"}, "unknown op"),
])
def test_patch_errors_say_what_is_wrong(op, fragment):
    with pytest.raises(PatchError, match=fragment):
        apply_patch({"xs": [1]}, [op])


def test_write_config_refuses_an_existing_file_without_overwrite(project):
    path = str(project / "configs/0.1.0/band.json")
    result = configs.write_config(path, BAND)
    assert not result["saved"] and "overwrite" in result["problems"][0]
    assert configs.write_config(path, {**BAND, "default": ["A"]}, overwrite=True)["saved"]
    assert json.loads(Path(path).read_text())["default"] == ["A"]


def test_write_config_creates_parent_directories_and_validates(project):
    path = str(project / "configs/0.2.0/band.json")
    assert not configs.write_config(path, {**BAND, "outputs": ["nope"]})["saved"]
    assert not Path(path).exists()
    assert configs.write_config(path, BAND)["saved"]


def test_validate_config_reports_problems_for_a_file_or_an_unsaved_document(project):
    assert configs.validate_config(path=str(project / "configs/0.1.0/band.json")) == {"valid": True, "problems": []}
    bad = configs.validate_config(doc={**BAND, "outputs": ["nope"]})
    assert not bad["valid"] and "nope" in bad["problems"][0]
    with pytest.raises(ValueError):
        configs.validate_config()


def test_read_config_returns_the_document_and_its_problems(bureau):
    out = configs.read_config(bureau)
    assert out["doc"]["name"] == "bureau_grade" and out["problems"] == []


def test_try_config_returns_json_safe_rows(project):
    out = configs.try_config(str(project / "configs/0.1.0/band.json"), [{"bureau_score": 550}, {"bureau_score": 720}])
    assert [r["band"] for r in out["rows"]] == ["C", "A"] and "band" in out["columns"]
    json.dumps(out)


def test_try_config_renders_dates_as_iso_strings():
    import datetime
    assert configs.json_safe([{"d": datetime.date(2026, 1, 15)}]) == [{"d": "2026-01-15"}]


def test_params_set_writes_a_nested_value_keeping_the_layout(project):
    path = str(project / "configs/0.1.0/params.json")
    assert params.params_set(path, {"credit/approved/limit": 0.3, "credit/other/x": 1})["saved"]
    assert json.loads(Path(path).read_text()) == {"credit": {"approved": {"limit": 0.3}, "other": {"x": 1}}}
    assert Path(path).read_text().startswith('{\n  "credit"')


def test_params_set_reports_an_unknown_key_under_an_entry_and_leaves_the_file(project):
    path = str(project / "configs/0.1.0/params.json")
    before = Path(path).read_text()
    result = params.params_set(path, {"credit/approved/limt": 0.3}, entry="pipeline:build")
    assert not result["saved"] and "unknown param 'limt'" in result["problems"][0] and "limit" in result["problems"][0]
    assert Path(path).read_text() == before


def test_params_set_under_an_entry_saves_a_valid_change(project):
    path = str(project / "configs/0.1.0/params.json")
    result = params.params_set(path, {"credit/approved/limit": 0.25}, entry="pipeline:build")
    assert result["saved"] and result["problems"] == []
    assert json.loads(Path(path).read_text())["credit"]["approved"]["limit"] == 0.25


def test_params_view_with_an_entry_reports_a_step_path_that_does_not_exist(project):
    path = str(project / "configs/0.1.0/params.json")
    Path(path).write_text(json.dumps({"credit": {"approvd": {"limit": 0.4}}}))
    out = params.params_view(path, entry="pipeline:build")
    assert "approvd" in out["problems"][0] and "credit/approved" in out["schema"]


def test_params_set_rejects_a_path_through_a_scalar(project):
    path = str(project / "configs/0.1.0/params.json")
    result = params.params_set(path, {"credit/approved/limit/deeper": 1})
    assert not result["saved"] and "limit" in result["problems"][0]


def test_params_of_the_example_project_validate_against_its_pipeline(monkeypatch):
    for project in sorted((ROOT / "examples").glob("*/")):
        monkeypatch.syspath_prepend(str(project))
    out = params.params_view(str(EXAMPLE_07 / "configs/0.1.0/params.json"), entry="limit_mgmt.pipeline:build")
    assert out["problems"] == [] and "limit_mgmt/caps/cap_income_multiple" in out["schema"]


def test_server_registers_the_config_tools_and_reports_refusals(bureau):
    mcp = build_server(Policy())
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"list_configs", "read_config", "validate_config", "write_config", "patch_config", "try_config",
            "table_view", "table_add_rows", "table_update_rows", "table_delete_rows", "table_add_column",
            "table_drop_column", "scorecard_view", "scorecard_set_bins", "params_view", "params_set"} <= names
    r = asyncio.run(mcp.call_tool("patch_config", {"path": bureau, "ops": [{"op": "remove", "path": "/tree"}]}))
    assert r.structured_content["saved"] is False and r.structured_content["problems"]
