from __future__ import annotations

import ast
import typing
from typing import Annotated, Iterator, Literal

from pydantic import TypeAdapter, ValidationError

from decikit.ops.measures import Percent, metadata_of
from decikit.lab.checks._base import Check, CheckContext
from decikit.lab.checks._report import Finding, Severity

__all__ = ["InvalidParamsDoc", "ParamDefaultOutOfBounds", "ParamUnconstrained", "ParamUnused"]


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


def _params(ctx: CheckContext):
    for node in ctx.calls():
        for p in node.params:
            yield node, p


def _bounds(p) -> tuple[float | None, float | None]:
    meta = p.field_info.metadata if p.field_info is not None else []
    lows = [m.ge if hasattr(m, "ge") else m.gt for m in meta if hasattr(m, "ge") or hasattr(m, "gt")]
    highs = [m.le if hasattr(m, "le") else m.lt for m in meta if hasattr(m, "le") or hasattr(m, "lt")]
    return (max(lows) if lows else None, min(highs) if highs else None)


class ParamUnconstrained(Check):
    """A numeric `param()` with no bounds; a `Percent` without `ge=0, le=1` is a warning."""

    type: Literal["param_unconstrained"] = "param_unconstrained"
    code = "DCD301"
    severity: Severity = Severity.INFO

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node, p in _params(ctx):
            kind = typing.get_args(p.annotation)[0] if typing.get_origin(p.annotation) is Annotated else p.annotation
            if kind not in (int, float):
                continue
            low, high = _bounds(p)
            meta = metadata_of(p.annotation)
            if isinstance(meta, Percent) and not (low is not None and low >= 0 and high is not None and high <= 1):
                bound, fix = "a Percent without ge=0, le=1", "add `ge=0, le=1`"
            elif low is None and high is None:
                bound, fix = "no bounds", "add `ge=`/`le=` so a bad params document is refused"
            else:
                continue
            severity = max(self.severity, Severity.WARNING, key=lambda s: s.rank) if isinstance(meta, Percent) else None
            yield ctx.finding(node, f"param {p.name!r} of {node.origin.path} has {bound}", fix=fix,
                              severity=severity, param=p.name)


class ParamDefaultOutOfBounds(Check):
    """A param whose default breaks its own bounds, or whose `on_invalid="default"` silently swaps bad values for it."""

    type: Literal["param_default_out_of_bounds"] = "param_default_out_of_bounds"
    code = "DCD302"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node, p in _params(ctx):
            if p.required:
                continue
            if p.field_info is not None and p.field_info.metadata:
                try:
                    TypeAdapter(Annotated[p.annotation, p.field_info]).validate_python(p.default)
                except ValidationError as e:
                    yield ctx.finding(node, f"param {p.name!r} of {node.origin.path} has default {p.default!r}, "
                                      f"which breaks its own bounds: {e.errors()[0]['msg']}",
                                      fix="change the default or the bounds", param=p.name, kind="default")
            if p.on_invalid == "default":
                yield ctx.finding(node, f"param {p.name!r} of {node.origin.path} replaces an invalid value with "
                                  "its default without a warning", fix='use on_invalid="warn", or "error"',
                                  param=p.name, kind="on_invalid")


class ParamUnused(Check):
    """A declared param the step's body never reads."""

    type: Literal["param_unused"] = "param_unused"
    code = "DCD303"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node in ctx.calls():
            src = ctx.source(node)
            # A row kernel receives its params by position, so a name there proves nothing.
            if src is None or node.kind != "scalar":
                continue
            used = {n.id for n in ast.walk(src.tree) if isinstance(n, ast.Name)}
            for p in node.params:
                if p.arg not in used:
                    yield ctx.finding(node, f"param {p.name!r} of {node.origin.path} is declared but never read",
                                      fix="remove it, or use it", param=p.name)
