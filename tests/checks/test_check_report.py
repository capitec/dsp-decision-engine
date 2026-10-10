"""Report: skipped checks, text, SARIF, counts and the gate."""
from decikit import flow, step
from decikit.lab import checks
from decikit.lab.checks import Check, Severity
from typing import Literal


@step(output="cents")
def scale(cents: int) -> int:
    return cents * 1000


class NeedsData(Check):
    type: Literal["needs_data_probe"] = "needs_data_probe"
    code = "ACME900"
    needs_data = True

    def run(self, ctx):
        yield ctx.finding(ctx.ir, f"saw {ctx.frame.height} rows")


def test_a_check_that_needs_data_is_skipped_with_a_reason_without_it():
    report = checks.run(flow(scale, name="term"), [NeedsData()])
    assert not report.findings
    (skipped,) = report.skipped
    assert (skipped.check, skipped.code) == ("needs_data_probe", "ACME900") and "data" in skipped.reason


def test_a_check_that_needs_data_runs_with_a_frame():
    import polars as pl

    report = checks.run(flow(scale, name="term"), [NeedsData()], data=pl.DataFrame({"cents": [1, 2]}))
    assert [f.message for f in report.findings] == ["saw 2 rows"] and not report.skipped


def test_sarif_carries_rules_levels_locations_and_fingerprints():
    pipeline = flow(scale, name="term", id="fedcba543210")
    sarif = checks.run(pipeline, ["int64_overflow"]).to_sarif()
    assert sarif["version"] == "2.1.0"
    (run,) = sarif["runs"]
    assert run["tool"]["driver"]["rules"] == [{"id": "DCD101", "name": "int64_overflow"}]
    (result,) = run["results"]
    assert result["ruleId"] == "DCD101" and result["level"] == "warning"
    assert result["partialFingerprints"]["decikit/v1"] == "fedcba543210/term/scale/DCD101"
    location = result["locations"][0]["physicalLocation"]
    assert location["artifactLocation"]["uri"].endswith("test_check_report.py") and location["region"]["startLine"] > 0


def test_sarif_marks_suppressed_results():
    sarif = checks.run(flow(scale, name="term", suppress_checks=["int64_overflow"]), ["int64_overflow"]).to_sarif()
    assert sarif["runs"][0]["results"][0]["suppressions"][0]["kind"] == "inSource"


def test_text_lists_findings_fixes_skips_and_a_summary():
    text = checks.run(flow(scale, name="term"), ["int64_overflow", NeedsData()]).to_text()
    assert "warning DCD101 int64_overflow" in text and "fix:" in text
    assert "skipped ACME900" in text and text.splitlines()[-1] == "term: 0 info, 1 warning, 0 error"


def test_fail_on_compares_severity_and_defaults_to_the_gate():
    report = checks.run(flow(scale, name="term"), ["int64_overflow"])
    assert report.ok and report.fail_on("warning") and report.fail_on(Severity.WARNING) and not report.fail_on("error")
    assert report.model_copy(update={"gate": Severity.WARNING}).fail_on()
