"""`core.affordability` -- discretionary income and capacity (spec 00 §6.5).

The unit only: `net_monthly_income`, `living_expenses`, `existing_obligations`
in, `discretionary_income` and `max_affordable_instalment` out, with a
verdict. Project 02 owns the *assessment* built on top of this (household
framing, the four modes, the evidence ladder) -- 00-ADDENDUM §B2 "Ownership"
note: 00 implements the units' arithmetic and interface, 02 composes them,
so project 10 can depend on 00 alone and stay standalone.

`affordability_buffer` is a plain `param()`, not a library-policy constant
baked into code: 07 supplies its own (18% vs. this module's 12% default,
addendum B3) without forking the capability, and a tighten-only overlay
(`core.adjustments`, kind `"buffer_adjustment"`) can raise it further at
runtime -- see `tests/test_affordability.py::test_buffer_overlay_composes`.

Monotone in `proposed_instalment` by construction: the verdict function
only ever compares two already-computed numbers, so a larger proposed
instalment can never produce a better verdict (02 §5.7.2(c), which this
unit must not break).
"""
from __future__ import annotations

from decider import missing_as, param, step

PASS = 1
MARGINAL = 2
FAIL = 3
INDETERMINATE = 4

_MARGINAL_BAND = 0.05  # within 5% over the max is "marginal", not an outright fail


def discretionary_income(
    net_monthly_income: float, living_expenses: float, existing_obligations: float,
    court_ordered_deductions: float = missing_as(0.0),
) -> float:
    """Addendum item 10: `court_ordered_deductions` is subtracted here, outside `statutory_deductions`."""
    return round(net_monthly_income - living_expenses - existing_obligations - court_ordered_deductions, 2)


def max_affordable_instalment(
    discretionary_income: float, affordability_buffer_applied: float,
) -> float:
    return round(max(0.0, discretionary_income) * (1.0 - affordability_buffer_applied), 2)


def affordability_buffer_applied(affordability_buffer: float = param(0.12, ge=0.0, le=0.9)) -> float:
    return affordability_buffer


def affordability_verdict_code(
    max_affordable_instalment: float, net_monthly_income: float,
    proposed_instalment: float | None = None,
) -> int:
    if net_monthly_income is None or net_monthly_income <= 0:
        return INDETERMINATE
    if proposed_instalment is None:
        return PASS if max_affordable_instalment > 0 else FAIL
    if proposed_instalment <= max_affordable_instalment:
        return PASS
    if proposed_instalment <= max_affordable_instalment * (1.0 + _MARGINAL_BAND):
        return MARGINAL
    return FAIL


discretionary_income_step = step(discretionary_income)
max_affordable_instalment_step = step(max_affordable_instalment)
affordability_buffer_applied_step = step(affordability_buffer_applied)
affordability_verdict_code_step = step(affordability_verdict_code)
