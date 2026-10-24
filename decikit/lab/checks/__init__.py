"""Run checks over a flow and collect a structured report.

`run(pipeline)` runs `standard`, or the project's `checks.yaml` when there is
one. A check is a `Check` subclass (or a function decorated with `check`) with
a stable code; a step silences findings with `suppress_checks=[alias_or_code]`.
`decikit check ENTRY` is the same from the command line, with an exit status.

Example::

    from decikit.lab import checks
    report = checks.run(pipeline)
    report = checks.run(pipeline, [checks.standard, NoFloatMoney(allow=["fee"])])
    print(report.to_text()); report.ok; report.counts
"""
from decikit.lab.checks._base import STANDARD, Check, CheckContext, check
from decikit.lab.checks._report import Finding, Report, Severity, SkippedCheck, Suppression
from decikit.lab.checks._run import catalogue, run

standard = STANDARD
"""The static checks (no data, no execution), as a check set for a `run` list."""

__all__ = [
    "Check", "CheckContext", "Finding", "Report", "Severity", "SkippedCheck", "Suppression", "catalogue", "check",
    "run", "standard",
]
