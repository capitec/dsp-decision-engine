from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from decider.check.models import Finding, Report
from decider.contract.refs import FlowRef
from decider.engine.ir.context import to_ir
from decider.exceptions import DeciderError

Check = Callable[[Any], Iterable[Finding]]
"""A check: a callable taking a pipeline and returning findings.

A client's own check is any such function (or callable object) — it does not
need to register anywhere. Return a `Finding` per problem (or none), leaving
`check` empty to have `run` name it after the function.
"""

Suite = Sequence[Check]
"""A suite: a sequence of checks run together (see `decider.checks.default_suite`)."""


def run(step: Any, suites: Sequence[Suite] | None = None) -> Report:
    """Run every check in `suites` over `step` and collect a `Report`.

    `suites` is a sequence of suites, each a sequence of checks; with no suites
    (the default), run `decider.checks.default_suite`. A check is a callable
    taking the pipeline and returning findings; add a custom one by putting it
    in a suite:

    Example::

        def my_check(pipeline):
            return [Finding(severity=Severity.WARNING, message="...")]

        report = check.run(pipeline, suites=[checks.default_suite, (my_check,)])
        report.ok
        report.model_dump_json()   # machine-readable, for CI / notebooks / editors

    A check that raises aborts the run with that error, so a broken custom
    check never silently disappears; catch inside the check to report a
    `Finding` instead.
    """
    if suites is None:
        from decider.checks import default_suite

        suites = (default_suite,)
    ir = to_ir(step)
    flow = FlowRef(name=ir.origin.path, source=ir.origin.source, flow_id=ir.origin.id)
    findings: list[Finding] = []
    for suite in suites:
        for check in suite:
            findings.extend(_run_one(check, step))
    return Report(flow=flow, findings=tuple(findings))


def _run_one(check: Check, step: Any) -> list[Finding]:
    name = getattr(check, "__name__", type(check).__name__)
    out: list[Finding] = []
    for finding in check(step):
        if not isinstance(finding, Finding):
            raise DeciderError(f"check {name!r} returned {type(finding).__name__}, not a Finding")
        if not finding.check:
            finding = finding.model_copy(update={"check": name})
        out.append(finding)
    return out
