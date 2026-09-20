"""A tiny pipeline to `decider2 serve` and to play with.

Doc 03 §1.1's flagship example, reused verbatim (not imported) from
`tests/test_flagship.py` — that file is fixed acceptance-test content this
agent was told not to edit, so this is an independent copy rather than a
test file importing an example importing a test.

Run it:

    decider2 serve decider2/examples/flagship.py

then:

    curl localhost:8000/ping
    curl -X POST localhost:8000/invocations -d '{"net_income": 4100.0,
        "expenses": 1500.0, "instalment": 800.0, "term_cap": 60.0,
        "min_net_salary": 4100.0}'
    curl localhost:8000/params/schema
    curl -X POST localhost:8000/params -d '{"cap_by_income_band": {"cap": 24.0}}'
"""
from __future__ import annotations

from decider2 import flow, param


def disposable_income(net_income: float, expenses: float) -> float:
    """Income remaining after committed expenses."""
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    """Affordability ratio."""
    return disposable_income / instalment


def cap_by_income_band(
    term_cap: float,
    min_net_salary: float,
    cap: float = param(48.0, ge=6, le=60),
    income_threshold: float = param(5000.0, ge=0),
) -> float:
    """Cap term at 48 months below the income floor.

    Implements: Credit Policy §7.4.2
    """
    if min_net_salary < income_threshold:
        return min(term_cap, cap)
    return term_cap


pipeline = flow(disposable_income, affordability_ratio, cap_by_income_band)
