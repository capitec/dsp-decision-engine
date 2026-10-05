from __future__ import annotations

from typing import Iterator, Literal

from decider.checks._base import Check, CheckContext
from decider.checks._report import Finding, Severity
from decider.ir import NullPolicy

__all__ = ["MissingRequired"]


class MissingRequired(Check):
    """Required inputs that are absent from the frame or null in it, reported at the first step that reads them."""

    type: Literal["missing_required"] = "missing_required"
    code = "DCD504"
    needs_data = True
    severity: Severity = Severity.ERROR

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        readers: dict[str, object] = {}
        for call in ctx.plan.calls:
            for v in call.reads or ():
                if v.producer is None:
                    readers.setdefault(v.name, call.node)
        for inp in ctx.plan.inputs:
            if inp.null_policy is not NullPolicy.REQUIRED:
                continue
            node = readers.get(inp.name, ctx.ir)
            if inp.name not in ctx.frame.columns:
                yield ctx.finding(node, f"required input column {inp.name!r} is not in the data and has no default",
                                  fix="add the column, or give the parameter a default", column=inp.name)
            elif nulls := ctx.frame[inp.name].null_count():
                yield ctx.finding(node, f"required input column {inp.name!r} has {nulls} null row(s) and no default to fill them",
                                  fix="fill the nulls, or declare a default with `missing_as`", column=inp.name, nulls=nulls)
