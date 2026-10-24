"""The built-in checks: each fires on its defect and stays quiet on clean code."""
import json
import time

import pytest

from decikit import flow, step
from decikit.lab import checks
from decikit.lab.checks import Severity


@step(output="x")
def stamped(x: float) -> float:
    return x + time.time()


@step(output="cents")
def scale(cents: int) -> int:
    return cents * 1000


@step(output="x")
def cap(x: float) -> float:
    return x


@step(output="x", id="0123abcdef45")
def cap_id(x: float) -> float:
    return x


def _by(report, alias):
    return [f for f in report.findings if f.check == alias]


def test_a_wall_clock_read_is_reported_with_its_source():
    wall = _by(checks.run(flow(stamped, name="term")), "wall_clock")
    assert len(wall) == 1
    finding = wall[0]
    assert finding.code == "DCD001" and finding.severity is Severity.WARNING
    assert finding.step.path == "term/stamped"
    assert finding.file.endswith("test_checks.py") and finding.line > 0
    assert finding.detail["call"] == "time.time()"
    assert finding.fix


def test_a_clean_step_has_no_wall_clock_or_overflow_findings():
    report = checks.run(flow(cap, name="term"))
    assert not _by(report, "wall_clock") and not _by(report, "int64_overflow") and not _by(report, "impure")


def test_impure_reports_globals_and_mutable_defaults():
    state = []

    @step(output="y")
    def leaky(x: float) -> float:
        global state
        state = [x]

        def helper(seen=[]):  # noqa: B006
            return seen

        return x

    found = {f.detail["kind"] for f in _by(checks.run(flow(leaky, name="term")), "impure")}
    assert found == {"global", "mutable_default"}


def test_int64_overflow_flags_integer_scaling():
    found = _by(checks.run(flow(scale, name="term")), "int64_overflow")
    assert [f.code for f in found] == ["DCD101"]


def test_a_flow_without_durable_ids_is_reported_per_flow_and_step():
    missing = _by(checks.run(flow(cap, name="term")), "durable_ids")
    assert {f.detail["kind"] for f in missing} == {"flow", "step"}
    assert all(f.code == "DCD401" for f in missing)
    flow_finding = next(f for f in missing if f.detail["kind"] == "flow")
    assert flow_finding.message == "flow 'term' has no committed durable id"
    assert next(f for f in missing if f.detail["kind"] == "step").step.path == "term/cap"


def test_a_flow_with_committed_ids_has_no_missing_id_findings():
    assert not _by(checks.run(flow(cap_id, name="term", id="fedcba543210")), "durable_ids")


def test_missing_ids_can_be_promoted_to_an_error():
    report = checks.run(flow(cap, name="term"), [{"type": "durable_ids", "severity": "error"}])
    assert report.findings and all(f.severity is Severity.ERROR for f in report.findings)
    assert not report.ok and report.fail_on("error")


def test_standard_finds_known_defects_without_arguments():
    found = {f.check for f in checks.run(flow(stamped, scale, name="term")).findings}
    assert {"wall_clock", "int64_overflow", "durable_ids"} <= found


def test_findings_carry_the_contract_step_reference():
    from decikit.lab import describe

    pipeline = flow(stamped, name="term")
    nodes = {n.path: n for n in describe(pipeline).nodes}
    finding = _by(checks.run(pipeline), "wall_clock")[0]
    assert finding.step == nodes["term/stamped"]


def test_report_serialises_to_json_with_string_severities():
    data = json.loads(checks.run(flow(stamped, name="term")).model_dump_json())
    assert data["flow"]["name"] == "term" and data["ok"] is True
    assert data["counts"]["warning"] >= 1
    assert all(isinstance(f["severity"], str) and f["code"] for f in data["findings"])


def test_crash_probe_surfaces_a_generated_regime_defect():
    def ratio(x: float, y: float) -> float:
        return x / y

    report = checks.run(flow(ratio, name="term"), ["crash_probe"])
    found = _by(report, "crash_probe")
    assert found and all(f.severity is Severity.ERROR and f.code == "DCD501" for f in found)
    assert any("division by zero" in f.message for f in found)
    assert all(f.detail["case"] for f in found)


def test_crash_probe_is_quiet_on_a_safe_flow():
    assert not checks.run(flow(cap, name="term"), ["crash_probe"]).findings


def test_mode_divergence_is_quiet_when_modes_agree():
    assert not checks.run(flow(cap, name="term"), ["mode_divergence"]).findings


def test_mode_divergence_reports_a_disagreement(monkeypatch):
    import decikit.lab.checks._builtin.runtime as runtime

    def disagree(*args, **kwargs):
        raise AssertionError("run() in fused mode differs from interpreted")

    monkeypatch.setattr(runtime, "assert_equivalent", disagree)
    found = checks.run(flow(cap, name="term"), ["mode_divergence"]).findings
    assert found and all(f.code == "DCD505" and "differs" in f.message for f in found)


@pytest.mark.parametrize("alias", ["crash_probe", "mode_divergence"])
def test_executing_checks_are_not_in_standard(alias):
    assert alias not in checks.standard
