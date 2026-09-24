"""`core.instalment` -- amortisation, and its inverse (spec 00 §6.6).

Level-payment (annuity) amortisation over `term_months` at a fixed monthly
rate. Called inside iterative solves (projects 03, 05, 06) many times per
application, so both directions live in one module: the forward
calculation and "what advance produces this instalment", because addendum
A8/00 §13 Q7 settle that these are the same unit run differently, not two
capabilities.

# ponytail: effective_annual_rate is the compounded nominal rate
# ((1+r)^12-1), not a fees-inclusive APR solve (which needs a numeric
# root-find over the whole payment stream). Upgrade to a true APR solve
# if a consumer needs disclosure-grade EAR; none in this project's scope
# does (03 §5.11 disclosure outputs are explicitly out of scope here).
"""
from __future__ import annotations

from decider import missing_as, param, step


def monthly_rate(nominal_annual_rate: float) -> float:
    return nominal_annual_rate / 12.0


def instalment_before_fees(offered_amount: float, term_months: float, nominal_annual_rate: float) -> float:
    """The capital + interest instalment, before fees and premium are added."""
    r = nominal_annual_rate / 12.0
    n = term_months
    if r == 0.0:
        return offered_amount / n
    return offered_amount * r / (1.0 - (1.0 + r) ** (-n))


def instalment(
    instalment_before_fees: float,
    monthly_service_fee: float = missing_as(0.0),
    credit_life_premium: float = missing_as(0.0),
) -> float:
    """The total monthly payment: capital + interest + fees + premium (00 §4 Offer and pricing)."""
    return instalment_before_fees + monthly_service_fee + credit_life_premium


def total_cost_of_credit(instalment: float, term_months: float, initiation_fee: float = missing_as(0.0)) -> float:
    return instalment * term_months + initiation_fee


def total_interest(total_cost_of_credit: float, offered_amount: float) -> float:
    return total_cost_of_credit - offered_amount


def effective_annual_rate(nominal_annual_rate: float) -> float:
    r = nominal_annual_rate / 12.0
    return (1.0 + r) ** 12 - 1.0


def solve_advance_for_instalment(
    target_instalment_before_fees: float, term_months: int, nominal_annual_rate: float
) -> float:
    """The inverse of `instalment_before_fees`: the advance that produces a given instalment.

    Not a step -- called directly by a solver (project 03 §5.8) that needs
    it many times per application without paying pipeline overhead each
    call.
    """
    r = nominal_annual_rate / 12.0
    n = term_months
    if r == 0.0:
        return target_instalment_before_fees * n
    return target_instalment_before_fees * (1.0 - (1.0 + r) ** (-n)) / r


instalment_before_fees_step = step(instalment_before_fees)
instalment_step = step(instalment)
total_cost_of_credit_step = step(total_cost_of_credit)
total_interest_step = step(total_interest)
effective_annual_rate_step = step(effective_annual_rate)
