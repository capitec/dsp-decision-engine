"""The check interface: static diagnostics, durable ids, extension seam, equivalence."""
import json
import time

import numpy as np

from decider import ConfigurableStep, Table, Value, checks, flow, param_table, step
from decider.checks import Finding, Severity
from decider.steps.trees import TreeConfig


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


def net(a: float, b: float) -> float:
    return a - b


def scaled(net: float) -> float:
    return net * 2.0


def scaled_changed(net: float) -> float:
    return net * 3.0


PIPELINE = flow(net, scaled, name="term")
CHANGED = flow(net, scaled_changed, name="term")


def test_a_wall_clock_read_is_reported_with_its_source():
    report = checks.run(flow(stamped, name="term"))
    wall = [f for f in report.findings if f.check == "wall_clock"]
    assert len(wall) == 1
    finding = wall[0]
    assert finding.severity is Severity.WARNING
    assert finding.step.path == "term/stamped"
    assert finding.step.source
    assert finding.line is not None and finding.line > 0
    assert finding.detail["call"] == "time.time()"


def test_a_flow_without_durable_ids_is_reported_per_flow_and_step():
    report = checks.run(flow(cap, name="term"))
    missing = [f for f in report.findings if f.check == "durable_ids"]
    assert {f.detail["kind"] for f in missing} == {"flow", "step"}
    flow_finding = next(f for f in missing if f.detail["kind"] == "flow")
    assert flow_finding.message == "flow 'term' has no committed durable id"
    step_finding = next(f for f in missing if f.detail["kind"] == "step")
    assert step_finding.step.path == "term/cap"


def test_a_flow_with_committed_ids_has_no_missing_id_findings():
    report = checks.run(flow(cap_id, name="term", id="fedcba543210"))
    assert not [f for f in report.findings if f.check == "durable_ids"]


def test_missing_durable_id_diagnostic_is_client_promotable():
    strict = checks.durable_ids(Severity.ERROR)
    report = checks.run(flow(cap, name="term"), suites=[(strict,)])
    assert all(f.severity is Severity.ERROR for f in report.findings)
    assert not report.ok


def test_a_custom_check_runs_and_is_named_after_its_function():
    def always_warn(pipeline):
        return [Finding(severity=Severity.WARNING, message="custom finding")]

    report = checks.run(flow(cap, name="term"), suites=[(always_warn,)])
    assert [f.check for f in report.findings] == ["always_warn"]
    assert report.findings[0].message == "custom finding"


def test_default_suite_finds_known_defects_without_arguments():
    report = checks.run(flow(stamped, scale, name="term"))
    checks_found = {f.check for f in report.findings}
    assert {"wall_clock", "numeric", "durable_ids"} <= checks_found
    assert all(f.step is not None or f.detail.get("kind") == "flow" for f in report.findings)


def test_report_serialises_to_json_with_string_severities():
    report = checks.run(flow(stamped, name="term"))
    data = json.loads(report.model_dump_json())
    assert data["flow"]["name"] == "term"
    assert data["findings"]
    assert all(isinstance(f["severity"], str) for f in data["findings"])
    assert all("step" in f and "message" in f for f in data["findings"])


def test_compare_reports_structural_equivalence_when_sources_match():
    result = checks.compare(PIPELINE, PIPELINE)
    assert result.structurally_equivalent
    assert result.method == "structure"
    assert result.equivalent


def test_compare_falls_back_to_corpus_evidence_when_revisions_differ():
    result = checks.compare(PIPELINE, CHANGED)
    assert not result.structurally_equivalent
    assert result.method == "corpus"
    assert not result.equivalent
    assert result.frames_compared >= 1
    assert result.divergences


def _threshold_tree(threshold):
    rule = {"meta": {"name": "r"}, "rule": {"type": "unary", "then": {"id": "hit", "type": "leaf", "result_idx": 0},
            "condition": {"op": ">", "feature": "amount", "threshold": threshold}}}
    return TreeConfig.load({"type": "tree", "name": "t", "path_output": "leaf", "tree": {
        "type": "prioritized_flat_rule", "mode": "all", "rules": [rule],
        "output": {"data": [{"hit": 1}], "default": {"hit": 0}, "dtypes": [["hit", "Int64"]]}}})


def test_compare_does_not_prove_flows_equivalent_when_only_a_const_differs():
    assert checks.compare(_threshold_tree(5.0), _threshold_tree(6.0)).method == "corpus"


def test_compare_does_not_prove_flows_equivalent_when_only_a_param_default_differs():
    low, high = ({"param": "limit", "default": d} for d in (5.0, 6.0))
    assert checks.compare(_threshold_tree(low), _threshold_tree(high)).method == "corpus"


class _Banded(ConfigurableStep):
    column: str
    factor: Value[float]
    bump: int = -1

    def to_ir(self, ctx):
        table = np.zeros(2000)
        table[self.bump] = 1.0
        return ctx.call(self, _banded, inputs={"x": self.column}, values={"factor": self.factor, "table": table})


def _banded(x: float, factor: float, table: object) -> float:
    return x * factor


def test_compare_does_not_prove_flows_equivalent_when_a_large_const_differs_in_an_elided_element():
    one, other = (_Banded(name="s", column="x", factor=2.0, bump=b) for b in (500, 1000))
    assert checks.compare(one, other).method == "corpus"


def test_compare_does_not_prove_flows_equivalent_when_a_param_table_differs_in_one_middle_row():
    def ladder(rate_at_25):
        rows = [{"floor": i, "rate": rate_at_25 if i == 25 else 0.2} for i in range(50)]

        def rate(score: int, ladder: Table = param_table({"floor": int, "rate": float}, default=rows)) -> float:
            return 0.0
        return flow(rate, name="pricing")

    assert checks.compare(ladder(0.1), ladder(0.3)).method == "corpus"


def test_equivalence_check_surfaces_a_generated_regime_defect():
    def ratio(x: float, y: float) -> float:
        return x / y

    report = checks.run(flow(ratio, name="term"), suites=[(checks.equivalence(),)])
    findings = [f for f in report.findings if f.check == "equivalence"]
    assert findings and all(f.severity is Severity.ERROR for f in findings)
    assert any("division by zero" in f.message for f in findings)
