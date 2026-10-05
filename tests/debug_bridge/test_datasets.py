import json
import os
import subprocess
import sys

import polars as pl
import pytest

from decider.debug_bridge.bridge import Bridge
from decider.debug_bridge.run import main

FLOW = {"flow": "credit", "steps": [{"step": "double"}]}
STEPS = """\
from decider import step


@step(output="twice")
def double(client_id: int) -> int:
    return client_id * 2
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    (tmp_path / "configs").mkdir()
    (tmp_path / "flow.yaml").write_text(json.dumps({**FLOW, "components": "dbl_steps"}))
    (tmp_path / "dbl_steps.py").write_text(STEPS)
    data = tmp_path / "datasets"
    (data / "apps").mkdir(parents=True)
    pl.DataFrame({"client_id": [1, 2, 3]}).write_parquet(data / "loans.parquet")
    pl.DataFrame({"client_id": [10]}).write_csv(data / "apps" / "a.csv")
    pl.DataFrame({"client_id": [20, 30]}).write_csv(data / "apps" / "b.csv")
    (data / "config.json").write_text(json.dumps({
        "keyed": {"type": "file", "path": "loans.parquet", "key": ["client_id"]},
        "remote": {"type": "file", "path": "s3://b/x.parquet"},
        "broken": {"type": "file"}}))
    yield tmp_path
    sys.modules.pop("dbl_steps", None)


def ask(project, cmd, **args):
    return Bridge().handle({"cmd": cmd, "file": str(project / "flow.yaml"), **args})


def test_listing_reports_types_keys_and_paths_without_failing_on_a_broken_entry(project):
    out = ask(project, "datasets")
    by_name = {d["name"]: d for d in out["datasets"]}
    assert out["root"] == str((project / "datasets").resolve())
    assert set(by_name) == {"apps", "keyed", "loans", "remote", "broken"}
    assert by_name["apps"]["type"] == "folder" and by_name["apps"]["path"] == str(project / "datasets" / "apps")
    assert by_name["keyed"]["key"] == ["client_id"] and by_name["loans"]["key"] is None
    assert by_name["remote"]["path"] is None and "error" not in by_name["remote"]
    assert by_name["broken"]["error"] and by_name["broken"]["key"] is None and by_name["broken"]["path"] is None


def test_schema_has_a_row_count_for_parquet_only(project):
    assert ask(project, "dataset_schema", name="loans") == {"columns": [["client_id", "Int64"]], "row_count": 3}
    assert ask(project, "dataset_schema", name="apps")["row_count"] is None


def test_head_returns_the_first_rows(project):
    out = ask(project, "dataset_head", name="loans", n=2)
    assert out == {"columns": [["client_id", "Int64"]], "rows": [{"client_id": 1}, {"client_id": 2}]}


def test_unknown_dataset_is_an_error_naming_the_known_ones(project):
    with pytest.raises(Exception, match="unknown dataset 'nope'.*loans"):
        ask(project, "dataset_head", name="nope")


def test_trace_runs_on_a_named_dataset(project):
    out = ask(project, "trace", dataset="loans")
    assert [r["client_id"] for r in out["data"]] == [1, 2, 3] and not out["error"]


def test_start_reports_the_dataset_name(project):
    b = Bridge()
    status = b.handle({"cmd": "start", "file": str(project / "flow.yaml"), "dataset": "apps"})
    assert status["dataset"]["dataset"] == "apps"


def test_data_and_dataset_together_are_an_error(project):
    with pytest.raises(ValueError, match="not both"):
        ask(project, "trace", dataset="loans", data=[{"client_id": 1}])


def test_run_script_takes_a_dataset_name(project, capsys):
    main([str(project / "flow.yaml"), "--dataset", "loans"])
    assert "6" in capsys.readouterr().out


def test_the_json_lines_protocol_answers_datasets_from_the_project_cwd(project):
    r, w = os.pipe()
    proc = subprocess.Popen([sys.executable, "-m", "decider.debug_bridge", "--fd", str(w)], cwd=project,
                            stdin=subprocess.PIPE, pass_fds=(w,), text=True)
    os.close(w)
    proc.stdin.write('{"id": 1, "cmd": "datasets"}\n{"id": 2, "cmd": "exit"}\n')
    proc.stdin.flush()
    with os.fdopen(r) as replies:
        first = json.loads(replies.readline())
    proc.stdin.close()
    proc.wait(timeout=30)
    assert first["ok"] and {d["name"] for d in first["result"]["datasets"]} >= {"loans", "apps"}
