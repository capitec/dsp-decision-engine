"""Run checks over a flow and collect a structured report.

`run(pipeline)` runs the default suite; pass `suites=` to add or replace
checks. A check is any callable taking a pipeline and returning `Finding`s —
see `run`'s docstring for the extension seam and `decider.checks` for the
built-in suites.

Example::

    from decider import check, checks
    report = check.run(pipeline, suites=[checks.default_suite])
    report.ok
"""
from decider.check.models import Finding, Report, Severity
from decider.check.run import Check, Suite, run

__all__ = ["Check", "Finding", "Report", "Severity", "Suite", "run"]
