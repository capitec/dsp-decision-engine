"""suppress_checks: on the node or an ancestor, by alias or code; unknown names are DCD403."""
import pytest

from decikit import flow, step
from decikit.lab import checks


@step(output="cents", suppress_checks=["int64_overflow"])
def by_alias(cents: int) -> int:
    return cents * 1000


@step(output="cents", suppress_checks=["DCD101"])
def by_code(cents: int) -> int:
    return cents * 1000


@step(output="cents")
def bare(cents: int) -> int:
    return cents * 1000


def _overflow(report):
    return [f for f in report.findings if f.check == "int64_overflow"]


@pytest.mark.parametrize("fn, name", [(by_alias, "int64_overflow"), (by_code, "DCD101")])
def test_a_step_suppresses_by_alias_or_code_and_the_finding_stays(fn, name):
    report = checks.run(flow(fn, name="term"))
    (finding,) = _overflow(report)
    assert finding.suppressed.name == name and finding.suppressed.path == "term/" + fn.name
    assert report.counts[checks.Severity.WARNING] == sum(
        f.suppressed is None and f.severity is checks.Severity.WARNING for f in report.findings)


def test_a_composite_suppresses_every_step_inside_it():
    report = checks.run(flow(bare, name="term", suppress_checks=["int64_overflow"]))
    (finding,) = _overflow(report)
    assert finding.suppressed.path == "term"


def test_suppressed_findings_do_not_fail_the_gate():
    pipeline = flow(bare, name="term", suppress_checks=["int64_overflow"])
    report = checks.run(pipeline, [{"type": "int64_overflow", "severity": "error"}])
    assert report.findings and report.ok and not report.fail_on("info")


def test_an_unsuppressed_sibling_still_reports():
    report = checks.run(flow(by_alias, bare, name="term"))
    assert sorted(bool(f.suppressed) for f in _overflow(report)) == [False, True]


def test_an_unknown_suppression_name_is_a_warning_naming_the_step():
    @step(output="x", suppress_checks=["int64_overflw"])
    def typo(x: float) -> float:
        return x

    (found,) = [f for f in checks.run(flow(typo, name="term")).findings if f.code == "DCD403"]
    assert found.check == "unknown_suppression" and found.step.path == "term/typo"
    assert found.severity is checks.Severity.WARNING and found.detail["name"] == "int64_overflw"


def test_a_known_alias_and_code_are_not_unknown():
    findings = checks.run(flow(by_alias, by_code, name="term")).findings
    assert not [f for f in findings if f.code == "DCD403"]
