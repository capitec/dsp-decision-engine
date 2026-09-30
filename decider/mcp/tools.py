"""The headless MCP tools: pure functions over the core modules.

These are the read and confirmation-gated actions an agent drives over `decider`
core. They are plain functions (no FastMCP), so they test without a server; the
server in `decider.mcp.server` wraps each one as a tool. Flows are named by
import path (`module:qualname`), the same identity `decider.experiments.load_flow`
resolves, and every returned step carries the contract's stable `flow_id`/`step_id`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import polars as pl

from decider import check
from decider.contract import StepRef, describe, resolve_step, source_location
from decider.contract.describing import find_pipelines
from decider.data import load
from decider.debug_bridge.loading import load_module
from decider.engine import Engine
from decider.engine.trace import TraceSink
from decider.experiments import ExperimentDef, ExperimentResult, load_flow, load_yaml, run_experiment
from decider.experiments.runs import trace as run_trace
from decider.mcp.policy import REDACTED, Policy

_EXCLUDED = {".git", "__pycache__", ".venv", "venv", "node_modules",
             ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages", "dist", "build"}


def discover_flows(root: str = ".") -> dict:
    """The decider pipelines under `root`, each as `{name, entry, file, line, kind}`.

    A candidate file has a top-level assignment or a `def build`; it is imported
    and its pipelines listed with the contract finder. A file that fails to
    import is reported with its error rather than guessed at.
    """
    flows: list[dict] = []
    for path in _pipeline_files(root):
        try:
            mod = load_module(str(path))
            pipelines = find_pipelines(mod, str(path))
        except Exception as e:  # noqa: BLE001  an unimportable candidate is reported, not fatal
            flows.append({"file": str(path), "error": f"{type(e).__name__}: {e}"})
            continue
        for p in pipelines:
            flows.append({"name": p["name"], "entry": f"{mod.__name__}:{p['name']}",
                          "file": str(path), "line": p["line"], "kind": p["kind"]})
    return {"flows": flows}


def describe_flow(entry: str, subgraph: str | None = None) -> dict:
    """The static structure of `entry` as a `FlowDescription`, or of the subtree under `subgraph`."""
    desc = describe(load_flow(entry))
    if subgraph is None:
        return desc.model_dump(mode="json")
    keep = {n.path for n in desc.nodes if n.path == subgraph or n.path.startswith(subgraph + "/")}
    return {
        "contract_version": desc.contract_version,
        "flow": desc.flow.model_dump(mode="json"),
        "nodes": [n.model_dump(mode="json") for n in desc.nodes if n.path in keep],
        "edges": [e.model_dump(mode="json") for e in desc.edges
                  if e.from_path in keep and e.to_path in keep],
        "value_slots": [s.model_dump(mode="json") for s in desc.value_slots if s.path in keep],
    }


def inspect_step(entry: str, path: str) -> dict:
    """One node of `entry` (stable identity, source location) and the edges touching it."""
    desc = describe(load_flow(entry))
    node = resolve_step(desc, StepRef(path=path))
    loc = source_location(node.source) if node.source else None
    return {
        "step": node.model_dump(mode="json"),
        "file": loc[0] if loc else None,
        "line": loc[1] if loc else None,
        "edges_in": [e.model_dump(mode="json") for e in desc.edges if e.to_path == node.path],
        "edges_out": [e.model_dump(mode="json") for e in desc.edges if e.from_path == node.path],
    }


def lineage(entry: str, name: str) -> dict:
    """Every slot, writer and reader of value `name` across `entry`."""
    desc = describe(load_flow(entry))
    slots = [s for s in desc.value_slots if s.name == name]
    return {
        "name": name,
        "slots": [s.model_dump(mode="json") for s in slots],
        "writers": [s.path for s in slots if s.path is not None],
        "readers": sorted({e.to_path for e in desc.edges if e.kind == "data" and name in e.values}),
    }


def source_context(entry: str, path: str) -> dict:
    """The file, line and a short window of source around the step `path` in `entry`."""
    desc = describe(load_flow(entry))
    node = resolve_step(desc, StepRef(path=path))
    file, line, lines = None, None, []
    if node.source:
        loc = source_location(node.source)
        if loc is not None:
            file, line = loc
            try:
                text = Path(file).read_text().splitlines()
                lines = text[max(0, line - 6):line + 5]
            except OSError:
                pass
    return {"path": node.path, "source": node.source, "file": file, "line": line, "lines": lines}


def run_summary(entry: str, data: str) -> dict:
    """Aggregate tallies for one run of `entry` over `data`: row count and per-output value counts.

    Runs the flow but returns no per-record or raw data, so it is summarised and
    ungated; `run_flow` is the confirmation-gated action that also runs.
    """
    return _summarise(entry, data)


def check_report(entry: str) -> dict:
    """The default check suite over `entry`, as the structured `Report`."""
    return check.run(load_flow(entry)).model_dump(mode="json")


def experiment_definition(path: str) -> dict:
    """An `experiment.yaml` (or JSON) definition, read from `path`."""
    return ExperimentDef.model_validate(load_yaml(Path(path).read_text())).model_dump(mode="json")


def experiment_results(path: str) -> dict:
    """A persisted experiment result (JSON) read from `path`."""
    return ExperimentResult.model_validate_json(Path(path).read_text()).model_dump(mode="json")


def raw_record(policy: Policy, entry: str, data: str, record: int) -> dict:
    """Per-step values for one record of `entry` (raw data, opt-in)."""
    if not policy.raw:
        return {"redacted": REDACTED}
    step = load_flow(entry)
    result = run_trace(step, load(data).frame, None)
    return {
        "entry": entry, "record": record,
        "values": {path: {name: vals[record] for name, vals in written.items()}
                   for path, written in result["steps"].items()},
    }


def raw_trace(policy: Policy, entry: str, data: str, record: int) -> dict:
    """The decision-trace events for one record of `entry` (raw data, opt-in)."""
    if not policy.raw:
        return {"redacted": REDACTED}
    step = load_flow(entry)
    sink = TraceSink()
    Engine().bind(step).run(load(data).frame, trace=sink)
    return {"entry": entry, "record": record,
            "events": [_event(e) for e in sink.events() if e.record is None or e.record == record]}


def run_flow(entry: str, data: str) -> dict:
    """Run `entry` end to end over `data`; confirmation-gated (it runs code)."""
    return {"ran": True, **_summarise(entry, data)}


def run_experiment(definition: str, data: str) -> dict:
    """Run the experiment in `definition` (an experiment.yaml) over `data`; confirmation-gated."""
    def_ = ExperimentDef.model_validate(load_yaml(Path(definition).read_text()))
    step = load_flow(def_.flow.entry)
    result = run_experiment(def_, step, load(data).frame)
    return result.model_dump(mode="json")


def start_debug(entry: str, data: str) -> dict:
    """Open a debug session over `entry`/`data`, paused before anything runs; confirmation-gated.

    Stepping is a stateful editor-bound interaction (task 11b); this opens and
    describes the paused session only.
    """
    step = load_flow(entry)
    frame = load(data).frame
    session = Engine().bind(step).session(frame)
    return {"entry": entry, "rows": frame.height, "finished": session.finished,
            "structure": session.structure()}


def generate_ids(path: str = ".", check: bool = True) -> dict:
    """Add durable `id=` tokens to the source under `path`; confirmation-gated persistence."""
    from decider.ids import generate

    report = generate(path, check=check)
    return {"changes": [[str(f), lineno, name, id_] for f, lineno, name, id_ in report.changes],
            "files": report.files}


def _summarise(entry: str, data: str) -> dict:
    step = load_flow(entry)
    frame = load(data).frame
    exe = Engine().bind(step)
    out = exe.run(frame)
    produced = [c for c in out.columns if c not in frame.columns]
    return {
        "entry": entry,
        "rows": frame.height,
        "outputs": {c: _tally(out[c]) for c in produced},
        "params": {"validated": exe.report.validated, "invalid": exe.report.invalid,
                   "warnings": exe.report.warnings},
    }


def _tally(series: pl.Series) -> list[dict]:
    counts = series.value_counts()
    return [{"value": _json(v), "count": int(n)}
            for v, n in zip(counts[series.name].to_list(), counts["count"].to_list())]


def _json(value):
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else value


def _event(e) -> dict:
    origin = e.origin
    return {"kind": e.kind.name, "path": origin.path if origin else None,
            "id": origin.id if origin else None, "source": origin.source if origin else None,
            "arm": e.arm, "iteration": e.iteration, "record": e.record, "value": e.value}


def _pipeline_files(root: str):
    base = Path(root).resolve()
    for path in sorted(base.rglob("*.py")):
        if any(part in _EXCLUDED or part.startswith(".") for part in path.parts):
            continue
        if _candidate(path):
            yield path


def _candidate(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return False
    return any(isinstance(n, ast.Assign) or (isinstance(n, ast.FunctionDef) and n.name == "build")
               for n in tree.body)
