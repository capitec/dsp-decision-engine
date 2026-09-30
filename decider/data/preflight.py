"""Pre-flight a run: data compatibility, params, capabilities and a cost estimate, before anything executes."""
from __future__ import annotations

from typing import Any, Mapping

import polars as pl
from pydantic import BaseModel, ConfigDict

from decider.contract import capabilities
from decider.engine.ir.decls import NullPolicy
from decider.engine.run.params import NodeParams, ParamsCache, RunParams, check_namespaces
from decider.engine.wiring import Plan, resolve
from decider.exceptions import ParamsError, WiringError


class CostEstimate(BaseModel):
    """A first-order run-cost estimate: rows times non-frame calls, plus one run per frame call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: int = 0
    calls: int = 0
    frame_calls: int = 0

    @property
    def row_operations(self) -> int:
        return self.rows * (self.calls - self.frame_calls)


class PreflightReport(BaseModel):
    """What pre-flight found: whether the run can proceed, and why not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    missing_values: tuple[tuple[str, int], ...] = ()
    cost: CostEstimate | None = None
    contract_version: str = ""
    modes: tuple[str, ...] = ()


def preflight(step: Any, frame: pl.DataFrame, *, params: Mapping[str, Any] | None = None) -> PreflightReport:
    """Check that `frame` and `params` can run `step`, before executing it.

    Reports required input columns absent from the data and required values
    that are null (the framework's missing-value semantics stay authoritative:
    nothing here fills a value in), validates the params document, and gives a
    first-order cost estimate. Returns a report; it does not raise for a bad
    run, so a caller surfaces the reasons instead of failing on the first one.

    Example::

        report = preflight(pipeline, loaded.frame, params=doc)
        report.ok, report.missing_columns, report.cost
    """
    caps = capabilities()
    try:
        plan = resolve(step)
    except WiringError as e:
        return PreflightReport(ok=False, errors=(str(e),), contract_version=caps.contract_version, modes=caps.modes)

    errors: list[str] = []
    missing_columns, missing_values = _data_compatibility(plan, frame, errors)
    warnings = _params_warnings(plan, params, frame.height, errors)

    return PreflightReport(
        ok=not errors,
        errors=tuple(errors),
        warnings=warnings,
        missing_columns=missing_columns,
        missing_values=missing_values,
        cost=CostEstimate(rows=frame.height, calls=len(plan.calls),
                          frame_calls=sum(1 for c in plan.calls if c.node.kind == "frame")),
        contract_version=caps.contract_version,
        modes=caps.modes,
    )


def _data_compatibility(plan: Plan, frame: pl.DataFrame, errors: list[str]) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
    missing_columns: list[str] = []
    missing_values: list[tuple[str, int]] = []
    for inp in plan.inputs:
        if inp.null_policy is not NullPolicy.REQUIRED:
            continue
        if inp.name not in frame.columns:
            missing_columns.append(inp.name)
            errors.append(f"required input column {inp.name!r} is not in the data and has no default")
        else:
            nulls = frame[inp.name].null_count()
            if nulls:
                missing_values.append((inp.name, nulls))
                errors.append(f"required input column {inp.name!r} has {nulls} null row(s) and no default to fill them")
    return tuple(missing_columns), tuple(missing_values)


def _params_warnings(plan: Plan, params: Mapping[str, Any] | None, rows: int, errors: list[str]) -> tuple[str, ...]:
    if params is None:
        return ()
    nodes = {c.id: NodeParams(c.node.origin.path, c.node.params) for c in plan.calls if c.node.params}
    if not nodes:
        return ()
    run = RunParams(nodes, params, ParamsCache(), lazy=False)
    try:
        check_namespaces(run.doc, nodes)
        for call_id in nodes:
            run.bundle(call_id, rows)
    except ParamsError as e:
        errors.append(str(e))
    return tuple(run.report.warnings)
