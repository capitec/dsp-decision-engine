"""decikit check: output formats, exit status, --select and --list."""
import json

import pytest
from click.testing import CliRunner

from decikit._tooling.cli import cli

PIPELINE = '''
from decikit import flow, step


@step(output="cents", id="0123abcdef45")
def scale(cents: int) -> int:
    return cents * 1000


def build():
    return flow(scale, name="term", id="fedcba543210")
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "pipeline.py").write_text(PIPELINE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _check(*args):
    return CliRunner().invoke(cli, ["check", *args])


def test_warnings_pass_the_default_gate(project):
    result = _check("shop.pipeline:build")
    assert result.exit_code == 0 and "DCD101 int64_overflow" in result.output


def test_fail_on_warning_exits_1(project):
    assert _check("shop.pipeline:build", "--fail-on", "warning").exit_code == 1


def test_the_gate_comes_from_the_project_config(project):
    (project / "configs").mkdir()
    (project / "checks.yaml").write_text("checks: [{type: int64_overflow, severity: error}]\n")
    assert _check("shop.pipeline:build").exit_code == 1
    (project / "other.yaml").write_text("disable: [int64_overflow]\n")
    assert _check("shop.pipeline:build", "--config", "other.yaml").exit_code == 0


def test_json_and_sarif_formats(project):
    doc = json.loads(_check("shop.pipeline:build", "--format", "json").output)
    assert doc["flow"]["name"] == "term" and doc["findings"][0]["code"] == "DCD101"
    sarif = json.loads(_check("shop.pipeline:build", "--format", "sarif").output)
    assert sarif["runs"][0]["results"][0]["partialFingerprints"]["decikit/v1"] == "fedcba543210/0123abcdef45/DCD101"


def test_select_keeps_findings_by_code_prefix(project):
    out = _check("shop.pipeline:build", "--select", "DCD4").output
    assert "DCD101" not in out
    assert "DCD101" in _check("shop.pipeline:build", "--select", "DCD1").output


def test_list_shows_every_check_without_an_entry(project):
    result = _check("--list")
    assert result.exit_code == 0
    assert "DCD001  wall_clock  determinism  warning" in result.output
    assert "DCD501  crash_probe  runtime  error [executes]" in result.output
    assert any(r["alias"] == "int64_overflow" for r in json.loads(_check("--list", "--format", "json").output))


def test_an_entry_is_required_without_list(project):
    assert _check().exit_code == 2


def test_a_bad_config_is_a_clean_error(project):
    (project / "bad.yaml").write_text("extends: other\n")
    result = _check("shop.pipeline:build", "--config", "bad.yaml")
    assert result.exit_code == 1 and "bad.yaml" in result.output
