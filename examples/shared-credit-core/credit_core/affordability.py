"""Discretionary income, the most the applicant can commit to a new instalment, and a verdict on a proposal.

The buffer is a param, so a consumer with a stricter policy sets its own without forking the capability.
"""
from __future__ import annotations

import math

from decikit import missing_as, param, step

# affordability_verdict_code
PASS = 1
MARGINAL = 2
FAIL = 3
INDETERMINATE = 4


def discretionary_income(
    net_monthly_income: float, living_expenses: float, existing_obligations: float,
    court_ordered_deductions: float = missing_as(0.0),
) -> float:
    """Court-ordered deductions come off here, not in `statutory_deductions`."""
    return round(net_monthly_income - living_expenses - existing_obligations - court_ordered_deductions, 2)


def affordability_buffer_applied(affordability_buffer: float = param(0.12, ge=0.0, le=0.9)) -> float:
    """Publishes the buffer in force as a column, so the decision record shows it."""
    return affordability_buffer


def max_affordable_instalment(discretionary_income: float, affordability_buffer_applied: float) -> float:
    return round(max(0.0, discretionary_income) * (1.0 - affordability_buffer_applied), 2)


def affordability_verdict_code(
    max_affordable_instalment: float, net_monthly_income: float,
    proposed_instalment: float = missing_as(math.nan),
    marginal_band: float = param(0.05, ge=0.0, le=1.0),
) -> int:
    """`PASS` within the maximum, `MARGINAL` up to `marginal_band` over it, else `FAIL`.

    Without a proposal, `PASS` means there is capacity at all. A larger proposal never gives a better verdict.
    """
    if net_monthly_income <= 0:
        return INDETERMINATE
    if math.isnan(proposed_instalment):
        return PASS if max_affordable_instalment > 0 else FAIL
    if proposed_instalment <= max_affordable_instalment:
        return PASS
    if proposed_instalment <= max_affordable_instalment * (1.0 + marginal_band):
        return MARGINAL
    return FAIL


discretionary_income_step = step(discretionary_income)
affordability_buffer_applied_step = step(affordability_buffer_applied)
max_affordable_instalment_step = step(max_affordable_instalment)
affordability_verdict_code_step = step(affordability_verdict_code)
