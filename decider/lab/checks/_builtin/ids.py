from __future__ import annotations

from typing import Iterator, Literal

from decider.lab.checks._base import Check, CheckContext, registered
from decider.lab.checks._report import Finding, Severity
from decider._suggest import hint
from decider.ir import BranchNode, CallNode, FiniteStateMachineNode, IRNode, LoopNode, ScatterGatherNode

__all__ = ["DuplicateIds", "DurableIds", "UnknownSuppression"]


class DurableIds(Check):
    """Flows and steps missing a committed durable id.

    A condition or transition call of a composite carries its parent's identity and is never flagged.
    """

    type: Literal["durable_ids"] = "durable_ids"
    code = "DCD401"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        root = ctx.ir
        if root.origin.path and root.origin.id is None:
            yield ctx.finding(root, f"flow {root.origin.path!r} has no committed durable id",
                              fix="run `decider ids`", kind="flow")
        helpers = _helper_calls(ctx.nodes)
        for node in ctx.nodes:
            if isinstance(node, CallNode) and id(node) not in helpers and node.origin.id is None:
                yield ctx.finding(node, f"{node.origin.path} has no committed durable id",
                                  fix="run `decider ids`", kind="step")


class DuplicateIds(Check):
    """Two steps or flows that carry one durable id; a copied definition keeps its id and then collides."""

    type: Literal["duplicate_ids"] = "duplicate_ids"
    code = "DCD402"
    severity: Severity = Severity.ERROR

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        first: dict[str, IRNode] = {}
        for node in ctx.nodes:
            id_ = node.origin.id
            if id_ is None:
                continue
            seen = first.setdefault(id_, node)
            # Nodes inside one row kernel share a path, and so an id, by design.
            if seen.origin.path != node.origin.path:
                yield ctx.finding(node, f"{node.origin.path} has durable id {id_}, which {seen.origin.path} already has",
                                  fix="give one of them a new id with `decider ids`", id=id_, other=seen.origin.path)


class UnknownSuppression(Check):
    """A `suppress_checks` entry that names no registered check: a typo, or a check that was renamed."""

    type: Literal["unknown_suppression"] = "unknown_suppression"
    code = "DCD403"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        names = registered()
        known = set(names) | {c.code for c in names.values()}
        for node in ctx.nodes:
            for name in node.origin.suppress_checks:
                if name not in known:
                    yield ctx.finding(node, f"{node.origin.path} suppresses {name!r}, which names no check."
                                      f"{hint(name, known)}", fix="fix the name or remove it", name=name)


def _helper_calls(nodes: list[IRNode]) -> set[int]:
    # The condition, transition or accumulate call of a composite is part of its parent's identity.
    helpers: set[int] = set()
    for node in nodes:
        if isinstance(node, (BranchNode, LoopNode)):
            helpers.add(id(node.condition))
        elif isinstance(node, ScatterGatherNode) and node.accumulate is not None:
            helpers.add(id(node.accumulate))
        elif isinstance(node, FiniteStateMachineNode):
            helpers.add(id(node.transition_function))
    return helpers
