from __future__ import annotations

import asyncio

import pytest
from fastmcp.exceptions import ToolError

from decider.data import load
from decider.experiments import Experiment, load_flow
from decider.mcp import Policy, build_server
from decider.mcp.policy import REDACTED

PIPELINE_SRC = '''\
from decider import flow, param, step


def ratio(income: float, debt: float) -> float:
    return debt / income


def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit


@step(output="tier", id="0123abcdef45")
def tier(approved: bool) -> float:
    return 1.0 if approved else 0.0


PIPELINE = flow(ratio, approved, tier, name="credit", id="abcdef012345")
'''

DATA = '{"income": 1000.0, "debt": 200.0}\n{"income": 500.0, "debt": 400.0}\n'


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "pipeline.py").write_text(PIPELINE_SRC)
    data = tmp_path / "data.json"
    data.write_text(DATA)
    monkeypatch.syspath_prepend(str(tmp_path))
    return {"entry": "mypkg.pipeline:PIPELINE", "data": str(data), "root": str(tmp_path)}


def _call(mcp, tool, **args):
    r = asyncio.run(mcp.call_tool(tool, args))
    assert not r.is_error, f"{tool}: {r}"
    return r.structured_content


def _tools(mcp):
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


def test_server_exposes_the_headless_read_tools():
    tools = _tools(build_server(Policy()))
    assert {
        "discover_flows", "describe_flow", "inspect_step", "lineage", "source_context",
        "run_summary", "check_report", "experiment_definition", "experiment_results",
        "raw_record", "raw_trace",
    } <= set(tools)
    assert not {"highlight", "reveal_source", "editor_selection"} & set(tools)


def test_describe_flow_returns_stable_identities(workspace):
    mcp = build_server(Policy())
    desc = _call(mcp, "describe_flow", entry=workspace["entry"])
    assert desc["flow"]["flow_id"] == "abcdef012345"
    step_ids = {n["step_id"] for n in desc["nodes"]}
    assert "0123abcdef45" in step_ids
    assert all(n["flow_id"] == "abcdef012345" for n in desc["nodes"])


def test_describe_flow_restricts_to_a_subgraph(workspace):
    mcp = build_server(Policy())
    desc = _call(mcp, "describe_flow", entry=workspace["entry"], subgraph="credit/approved")
    assert {n["path"] for n in desc["nodes"]} == {"credit/approved"}


def test_inspect_step_returns_identity_source_and_edges(workspace):
    mcp = build_server(Policy())
    out = _call(mcp, "inspect_step", entry=workspace["entry"], path="credit/tier")
    assert out["step"]["step_id"] == "0123abcdef45"
    assert out["step"]["flow_id"] == "abcdef012345"
    assert out["step"]["source"]
    assert any(e["kind"] == "data" for e in out["edges_in"])


def test_lineage_reports_slots_writers_and_readers(workspace):
    mcp = build_server(Policy())
    out = _call(mcp, "lineage", entry=workspace["entry"], name="ratio")
    assert "credit/ratio" in out["writers"]
    assert "credit/approved" in out["readers"]


def test_discover_flows_finds_the_pipeline(workspace):
    mcp = build_server(Policy())
    flows = _call(mcp, "discover_flows", root=workspace["root"])["flows"]
    entries = [f["entry"] for f in flows if "entry" in f]
    assert workspace["entry"] in entries


def test_run_summary_returns_tallies_not_per_record_values(workspace):
    mcp = build_server(Policy())
    summary = _call(mcp, "run_summary", entry=workspace["entry"], data=workspace["data"])
    assert summary["rows"] == 2
    assert {"value": 1.0, "count": 1} in summary["outputs"]["tier"]


def test_check_report_returns_the_default_suite(workspace):
    mcp = build_server(Policy())
    report = _call(mcp, "check_report", entry=workspace["entry"])
    assert report["flow"]["name"] == "credit"
    assert report["findings"]  # missing durable ids on ratio/approved


def test_mutating_tools_are_confirmation_gated():
    tools = _tools(build_server(Policy()))
    for name in ("run_flow", "run_experiment", "start_debug", "generate_ids"):
        assert getattr(tools[name].annotations, "destructive_hint", None) is True, name
    for name in ("describe_flow", "run_summary", "check_report"):
        assert getattr(tools[name].annotations, "destructive_hint", None) is not True, name


def test_raw_tools_are_redacted_by_default(workspace):
    mcp = build_server(Policy(raw=False))
    assert _call(mcp, "raw_record", entry=workspace["entry"], data=workspace["data"], record=0)["redacted"] == REDACTED
    assert _call(mcp, "raw_trace", entry=workspace["entry"], data=workspace["data"], record=0)["redacted"] == REDACTED


def test_raw_tools_return_data_when_enabled(workspace):
    mcp = build_server(Policy(raw=True))
    record = _call(mcp, "raw_record", entry=workspace["entry"], data=workspace["data"], record=0)
    assert record["values"]["credit/ratio"]["ratio"] == 0.2
    assert record["values"]["credit/tier"]["tier"] == 1.0
    trace = _call(mcp, "raw_trace", entry=workspace["entry"], data=workspace["data"], record=0)
    assert trace["events"]
    assert any(e["kind"] == "STEP" and e["path"] == "credit/tier" for e in trace["events"])


def test_run_flow_is_gated_but_executes(workspace):
    mcp = build_server(Policy())
    out = _call(mcp, "run_flow", entry=workspace["entry"], data=workspace["data"])
    assert out["ran"] is True
    assert out["rows"] == 2


def test_an_error_surfaces_as_a_tool_error():
    mcp = build_server(Policy())
    with pytest.raises(ToolError):
        asyncio.run(mcp.call_tool("describe_flow", {"entry": "no.such:module"}))


def test_experiment_definition_and_results_read_files(workspace, tmp_path):
    mcp = build_server(Policy())
    exp = (Experiment("drift", load_flow(workspace["entry"]), load(workspace["data"]).frame,
                      entry=workspace["entry"])
           .scenario("baseline")
           .compare(outputs=("tier",)))
    result = exp.run()
    result_path = tmp_path / "result.json"
    result_path.write_text(result.model_dump_json())
    assert _call(mcp, "experiment_results", path=str(result_path))["status"] == "completed"

    from decider.experiments import dump_yaml
    def_path = tmp_path / "experiment.yaml"
    def_path.write_text(dump_yaml(exp.def_.model_dump()))
    assert _call(mcp, "experiment_definition", path=str(def_path))["name"] == "drift"
