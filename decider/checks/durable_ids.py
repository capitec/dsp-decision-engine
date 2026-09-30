from __future__ import annotations

from typing import Any

from decider.check import Check, Finding, Severity
from decider.checks._common import call_nodes, helper_calls, step_ref
from decider.engine.ir.context import to_ir
from decider.engine.ir.nodes import CallNode


def durable_ids(severity: Severity = Severity.WARNING) -> Check:
    """Detect flows and steps missing a committed durable id.

    Reports a finding for the flow root (when named) and for every step whose
    `Origin.id` is unset. A condition/transition call of a composite carries
    its parent's identity and is never flagged. `severity` lets a client
    promote the diagnostic to a CI failure by running `durable_ids(Severity.ERROR)`
    — or the client can map the default warning itself.

    Example::

        check.run(pipeline, suites=[(durable_ids(Severity.ERROR),)])
    """

    def check(step: Any) -> list[Finding]:
        ir = to_ir(step)
        findings: list[Finding] = []
        if ir.origin.path and ir.origin.id is None:
            findings.append(Finding(check="durable_ids", severity=severity,
                                    message=f"flow {ir.origin.path!r} has no committed durable id",
                                    detail={"kind": "flow"}))
        helpers = helper_calls(ir)
        for node in call_nodes(ir):
            if id(node) in helpers or node.origin.id is not None:
                continue
            findings.append(Finding(check="durable_ids", severity=severity,
                                    step=step_ref(node, ir.origin.id),
                                    message=f"{node.origin.path} has no committed durable id",
                                    detail={"kind": "step"}))
        return findings

    check.__name__ = "durable_ids"
    return check
