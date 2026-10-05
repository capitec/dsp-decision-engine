from __future__ import annotations

from typing import Iterator, Literal

from decider.checks._base import Check, CheckContext
from decider.checks._report import Finding, Severity

__all__ = ["InvalidParamsDoc"]


class InvalidParamsDoc(Check):
    """Every problem in the params document: a key that matches no step (a typo), a wrong type, an out-of-bounds value. Needs `params=`."""

    type: Literal["invalid_params_doc"] = "invalid_params_doc"
    code = "DCD304"
    severity: Severity = Severity.ERROR

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        if ctx.params is None:
            return
        by_path = {n.origin.path: n for n in ctx.nodes}
        for error in ctx.pipeline.parameters().validate(ctx.params).errors:
            # Node errors start with the step path; document-level ones don't, and sit at the flow.
            node = by_path.get(error.split(":", 1)[0], ctx.ir)
            yield ctx.finding(node, error, fix="use a step path and values from `pipeline.parameters().defaults()`")
