"""checks.yaml / checks.json: extends, checks with options and severity, disable, fail_on."""
import json

import pytest

from decider import checks, flow, step
from decider.checks import Severity
from decider.exceptions import DeciderError


@step(output="cents")
def scale(cents: int) -> int:
    return cents * 1000


def _write(tmp_path, text, name="checks.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_checks_override_extends_severity_and_set_the_gate(tmp_path):
    path = _write(tmp_path, "extends: standard\nchecks:\n  - {type: numeric, severity: error}\nfail_on: warning\n")
    report = checks.run(flow(scale, name="term"), config=path)
    assert {f.check for f in report.findings} >= {"numeric", "durable_ids"}
    assert next(f for f in report.findings if f.check == "numeric").severity is Severity.ERROR
    assert report.gate is Severity.WARNING


def test_disable_by_alias_and_by_code(tmp_path):
    path = _write(tmp_path, "disable: [numeric, DCD401]\n")
    report = checks.run(flow(scale, name="term"), config=path)
    assert not {f.check for f in report.findings} & {"numeric", "durable_ids"}


def test_without_extends_only_the_listed_checks_run(tmp_path):
    path = _write(tmp_path, "extends: null\nchecks: [{type: numeric}]\n")
    assert {f.check for f in checks.run(flow(scale, name="term"), config=path).findings} == {"numeric"}


def test_json_is_read_too_and_options_reach_the_check(tmp_path):
    path = _write(tmp_path, json.dumps({"checks": [{"type": "mode_divergence", "modes": ["interpreted"]}]}),
                  "checks.json")
    assert checks.run(flow(scale, name="term"), config=path).findings


def test_the_project_file_is_the_default(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    _write(tmp_path, "extends: null\nchecks: [{type: numeric}]\n")
    monkeypatch.chdir(tmp_path)
    assert {f.check for f in checks.run(flow(scale, name="term")).findings} == {"numeric"}


def test_an_explicit_check_list_ignores_the_project_file(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    _write(tmp_path, "extends: null\nchecks: [{type: numeric}]\n")
    monkeypatch.chdir(tmp_path)
    assert {f.check for f in checks.run(flow(scale, name="term"), ["durable_ids"]).findings} == {"durable_ids"}


@pytest.mark.parametrize("text, message", [
    ("extends: other\n", "extends"),
    ("checks: [{type: no_such_check}]\n", "no_such_check"),
    ("checks: [{type: numeric, severity: loud}]\n", "severity"),
    ("checks: [{type: numeric, bogus: 1}]\n", "bogus"),
    ("fail_on: critical\n", "fail_on"),
    ("surprise: 1\n", "surprise"),
    ("- a\n- b\n", "mapping"),
])
def test_an_invalid_config_is_a_clear_error_naming_the_file(tmp_path, text, message):
    path = _write(tmp_path, text)
    with pytest.raises(DeciderError, match=message) as e:
        checks.run(flow(scale, name="term"), config=path)
    assert str(path) in str(e.value)
