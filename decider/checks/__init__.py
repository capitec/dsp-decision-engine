"""Built-in check suites: static diagnostics, durable-id presence, whole-path equivalence.

`default_suite` is the static set (wall-clock reads, shared-state mutation,
numeric risk, missing durable ids) that needs no execution. `equivalence` and
`compare` reuse `decider.testing` for whole-path and revision comparison.

Example::

    from decider import check, checks
    report = check.run(pipeline)                       # default_suite
    check.run(pipeline, suites=[checks.default_suite, (checks.equivalence(),)])
    checks.compare(term_old, term_new).equivalent
"""
from decider.checks.durable_ids import durable_ids
from decider.checks.equivalence import Equivalence, compare, equivalence
from decider.checks.static import impure, numeric, wall_clock

default_suite = (wall_clock, impure, numeric, durable_ids())

__all__ = [
    "Equivalence", "compare", "default_suite", "durable_ids", "equivalence", "impure", "numeric", "wall_clock",
]
