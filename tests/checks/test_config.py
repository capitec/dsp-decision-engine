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
    path = _write(tmp_path, "extends: standard\nchecks:\n  - {type: int64_overflow, severity: error}\nfail_on: warning\n")
    report = checks.run(flow(scale, name="term"), config=path)
    assert {f.check for f in report.findings} >= {"int64_overflow", "durable_ids"}
    assert next(f for f in report.findings if f.check == "int64_overflow").severity is Severity.ERROR
    assert report.gate is Severity.WARNING


def test_disable_by_alias_and_by_code(tmp_path):
    path = _write(tmp_path, "disable: [int64_overflow, DCD401]\n")
    report = checks.run(flow(scale, name="term"), config=path)
    assert not {f.check for f in report.findings} & {"int64_overflow", "durable_ids"}


def test_without_extends_only_the_listed_checks_run(tmp_path):
    path = _write(tmp_path, "extends: null\nchecks: [{type: int64_overflow}]\n")
    assert {f.check for f in checks.run(flow(scale, name="term"), config=path).findings} == {"int64_overflow"}


def test_json_is_read_too_and_options_reach_the_check(tmp_path):
    path = _write(tmp_path, json.dumps({"checks": [{"type": "mode_divergence", "modes": ["interpreted"]}]}),
                  "checks.json")
    assert checks.run(flow(scale, name="term"), config=path).findings


def test_the_project_file_is_the_default(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    _write(tmp_path, "extends: null\nchecks: [{type: int64_overflow}]\n")
    monkeypatch.chdir(tmp_path)
    assert {f.check for f in checks.run(flow(scale, name="term")).findings} == {"int64_overflow"}


def test_an_explicit_check_list_ignores_the_project_file(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    _write(tmp_path, "extends: null\nchecks: [{type: int64_overflow}]\n")
    monkeypatch.chdir(tmp_path)
    assert {f.check for f in checks.run(flow(scale, name="term"), ["durable_ids"]).findings} == {"durable_ids"}


@pytest.mark.parametrize("text, message", [
    ("extends: other\n", "extends"),
    ("checks: [{type: no_such_check}]\n", "no_such_check"),
    ("checks: [{type: int64_overflow, severity: loud}]\n", "severity"),
    ("checks: [{type: int64_overflow, bogus: 1}]\n", "bogus"),
    ("fail_on: critical\n", "fail_on"),
    ("surprise: 1\n", "surprise"),
    ("- a\n- b\n", "mapping"),
])
def test_an_invalid_config_is_a_clear_error_naming_the_file(tmp_path, text, message):
    path = _write(tmp_path, text)
    with pytest.raises(DeciderError, match=message) as e:
        checks.run(flow(scale, name="term"), config=path)
    assert str(path) in str(e.value)


def test_an_explicit_check_list_still_honours_the_project_disable_and_fail_on(tmp_path, monkeypatch):
    _write(tmp_path, "disable: [durable_ids]\nfail_on: warning\n")
    monkeypatch.chdir(tmp_path)
    report = checks.run(flow(scale, name="term"), ["durable_ids", "int64_overflow"])
    assert {f.check for f in report.findings} == {"int64_overflow"} and report.gate is Severity.WARNING


def test_two_entries_of_one_type_with_different_options_both_run():
    from decider.checks._config import ChecksConfig

    cfg = ChecksConfig.model_validate({"extends": None, "checks": [
        {"type": "durable_ids", "severity": "error"}, {"type": "durable_ids", "severity": "info"},
        {"type": "durable_ids", "severity": "info"}]})
    assert [c.severity.value for c in cfg.instances()] == ["error", "info"]
