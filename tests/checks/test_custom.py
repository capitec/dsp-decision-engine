"""Custom checks: by class, by decorator, from an entry point, and a ConfigurableStep's own hook."""
import sys
from typing import Literal

from decider import ConfigurableStep, checks, flow, step
from decider.checks import Check, Severity


@step(output="fee")
def fee(x: float) -> float:
    return float(x)


class NoFloatCall(Check):
    type: Literal["no_float_call"] = "no_float_call"
    code = "ACME001"
    allow: list[str] = []

    def run(self, ctx):
        import ast

        for node in ctx.calls():
            src = ctx.source(node)
            for call in ast.walk(src.tree):
                if (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "float"
                        and node.origin.path.rpartition("/")[2] not in self.allow):
                    yield ctx.finding(node, "calls float()", fix="keep cents", line=src.line(call))


def test_a_check_class_runs_in_a_flat_list_with_options_and_severity():
    pipeline = flow(fee, name="term")
    (found,) = checks.run(pipeline, [checks.standard, NoFloatCall(severity="error")]).findings[-1:]
    assert (found.code, found.check, found.severity) == ("ACME001", "no_float_call", Severity.ERROR)
    assert found.step.path == "term/fee" and found.fix == "keep cents"
    assert not [f for f in checks.run(pipeline, [NoFloatCall(allow=["fee"])]).findings]


def test_a_custom_check_loads_from_a_config_dict_by_alias():
    report = checks.run(flow(fee, name="term"), [{"type": "no_float_call", "allow": ["fee"]}])
    assert not report.findings


def test_a_custom_check_can_be_suppressed_by_code():
    pipeline = flow(fee, name="term", suppress_checks=["ACME001"])
    (found,) = checks.run(pipeline, [NoFloatCall()]).findings
    assert found.suppressed.name == "ACME001"


@step(output="y")
def _second(x: float) -> float:
    return x


@checks.check(code="ACME002", severity="info")
def no_wide_flows(ctx):
    if len(ctx.calls()) > 1:
        yield ctx.finding(ctx.ir, "more than one step")


def test_a_decorated_function_is_a_check_with_its_alias_and_code():
    from decider.checks import catalogue

    report = checks.run(flow(fee, _second, name="term"), [no_wide_flows])
    (found,) = report.findings
    assert (found.code, found.check, found.severity) == ("ACME002", "no_wide_flows", Severity.INFO)
    assert any(r["alias"] == "no_wide_flows" and r["family"] == "custom" for r in catalogue())


def test_a_third_party_check_loads_from_an_entry_point(tmp_path, monkeypatch):
    (tmp_path / "ep_checks.py").write_text('''
from typing import Literal
from decider.checks import Check

class EpCheck(Check):
    type: Literal["ep_check"] = "ep_check"
    code = "EP001"

    def run(self, ctx):
        yield ctx.finding(ctx.ir, "from a plugin")
''')
    dist = tmp_path / "ep_checks-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ep-checks\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text("[decider.checks]\nep_check = ep_checks:EpCheck\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "ep_checks", raising=False)
    assert "ep_checks" not in sys.modules
    (found,) = checks.run(flow(fee, name="term"), ["ep_check"]).findings
    assert found.code == "EP001"
    assert any(r["alias"] == "ep_check" for r in checks.catalogue())


class Banded(ConfigurableStep):
    type: Literal["banded_probe"] = "banded_probe"
    column: str
    low: float
    high: float

    def to_ir(self, ctx):
        return ctx.call(self, _ident, inputs={"x": self.column})

    def check(self, ctx):
        if self.low > self.high:
            yield ctx.finding(ctx.node, f"{self.name}: low is above high", fix="swap them", code="ACME010",
                              check="empty_band")


def _ident(x: float) -> float:
    return x


def test_a_configurable_step_reports_through_its_own_check_hook():
    pipeline = flow(Banded(name="b", column="x", low=2.0, high=1.0), name="term")
    (found,) = [f for f in checks.run(pipeline).findings if f.code == "ACME010"]
    assert found.step.path == "term/b" and found.check == "empty_band"
    assert not [f for f in checks.run(flow(Banded(name="b", column="x", low=1.0, high=2.0), name="term")).findings
                if f.code == "ACME010"]


def test_a_hook_finding_is_suppressed_by_the_step_and_disabled_by_config(tmp_path):
    suppressed = Banded(name="b", column="x", low=2.0, high=1.0, suppress_checks=["empty_band"])
    (found,) = [f for f in checks.run(flow(suppressed, name="term")).findings if f.code == "ACME010"]
    assert found.suppressed
    path = tmp_path / "checks.yaml"
    path.write_text("disable: [ACME010]\n")
    assert not [f for f in checks.run(flow(Banded(name="b", column="x", low=2.0, high=1.0), name="term"), config=path)
                .findings if f.code == "ACME010"]
