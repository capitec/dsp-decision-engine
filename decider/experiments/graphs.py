"""Built-in visualisations, as data the UI renders.

`path_sankey` runs the baseline once more with a trace sink to capture each
record's branch arms, then lays out a Sankey of record flow: input → branch
arms → final outcome. `tree_flows` reuses the baseline trace's tree-path counts
when there is no branch to fan out on.
"""
from __future__ import annotations

from typing import Any

from decider.engine import Engine
from decider.engine.ir.context import to_ir
from decider.engine.ir.nodes import BranchNode
from decider.engine.trace import Kind, TraceSink


def path_sankey(step, frame, params, baseline: dict, outputs: tuple[str, ...]) -> dict[str, Any]:
    """A Sankey of record flow through the baseline's branches, ending at the outcome.

    Returns `{"nodes": [...], "links": [...], "outcome": <column>}`: one node
    per branch arm and one per outcome value, with links carrying record counts.
    """
    sink = TraceSink()
    out = Engine().bind(step).run(frame, params, trace=sink)
    arms = _arm_names(step)
    per_record: list[list[str]] = [[] for _ in range(frame.height)]
    for e in sink.events():
        if e.kind is Kind.BRANCH_ARM and e.record is not None and e.origin is not None:
            per_record[e.record].append(f"{e.origin.path}#{e.arm}")
    outcome = _outcome(outputs, out.columns)
    nodes: dict[str, dict[str, Any]] = {"in": {"id": "in", "label": f"{frame.height} records", "kind": "input"}}
    links: dict[tuple[str, str], int] = {}
    for r, path in enumerate(per_record):
        value = out[outcome][r] if outcome in out.columns else None
        terminal = f"out:{value}"
        nodes.setdefault(terminal, {"id": terminal, "label": f"{outcome} {value}", "kind": "outcome"})
        prev = "in"
        for arm in path:
            nodes.setdefault(arm, {"id": arm, "label": _arm_label(arm, arms), "kind": "arm"})
            links[(prev, arm)] = links.get((prev, arm), 0) + 1
            prev = arm
        links[(prev, terminal)] = links.get((prev, terminal), 0) + 1
    return {
        "outcome": outcome,
        "nodes": list(nodes.values()),
        "links": [{"source": s, "target": t, "value": v} for (s, t), v in links.items()],
        "paths": baseline.get("paths", {}),
    }


def _outcome(outputs: tuple[str, ...], columns) -> str:
    for name in outputs:
        if name in columns:
            return name
    return columns[-1] if len(columns) else ""


def _arm_names(step) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for node in _walk(to_ir(step)):
        if isinstance(node, BranchNode):
            out[node.origin.path] = [arm.origin.path.split("/")[-1] for arm in node.arms]
    return out


def _arm_label(arm: str, names: dict[str, list[str]]) -> str:
    path, _, index = arm.partition("#")
    short = path.split("/")[-1]
    arm_names = names.get(path)
    if arm_names and index.isdigit() and int(index) < len(arm_names):
        return arm_names[int(index)]
    return f"{short} arm {index}"


def _walk(node):
    yield node
    for child in node.children():
        yield from _walk(child)
