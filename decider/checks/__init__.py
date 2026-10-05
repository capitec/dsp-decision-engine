"""Run checks over a flow and collect a structured report.

`run(pipeline)` runs the default suite; pass `suites=` to add or replace
checks. A check is any callable taking a pipeline and returning `Finding`s;
see `run`'s docstring for the extension seam.

The built-in suites: static diagnostics, durable-id presence, whole-path equivalence.
`default_suite` is the static set (wall-clock reads, shared-state mutation,
numeric risk, missing durable ids) that needs no execution. `equivalence` reuses
`decider.testing` for whole-path comparison; to compare two versions, see `decider.diff`.

Example::

    from decider import checks
    report = checks.run(pipeline)                       # default_suite
    checks.run(pipeline, suites=[checks.default_suite, (checks.equivalence(),)])
"""
from decider.checks._model import Finding, Report, Severity
from decider.checks._run import Check, Suite, run
from decider.checks._durable_ids import durable_ids
from decider.checks._equivalence import equivalence
from decider.checks._static import impure, numeric, wall_clock

default_suite = (wall_clock, impure, numeric, durable_ids())

__all__ = [
    "Check", "Finding", "Report", "Severity", "Suite", "default_suite", "durable_ids",
    "equivalence", "impure", "numeric", "run", "wall_clock",
]
