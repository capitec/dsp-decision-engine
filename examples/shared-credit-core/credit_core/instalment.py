"""Level-payment amortisation and its inverse: the same formula run in both directions."""
from __future__ import annotations

from decider import missing_as, step


def instalment_before_fees(offered_amount: float, term_months: float, nominal_annual_rate: float) -> float:
    """Capital and interest per month, before fees and premium."""
    r = nominal_annual_rate / 12.0
    if r == 0.0:
        return offered_amount / term_months
    return offered_amount * r / (1.0 - (1.0 + r) ** (-term_months))


def instalment(
    instalment_before_fees: float,
    monthly_service_fee: float = missing_as(0.0),
    credit_life_premium: float = missing_as(0.0),
) -> float:
    """The monthly payment: capital, interest, service fee and credit life premium."""
    return instalment_before_fees + monthly_service_fee + credit_life_premium


def total_cost_of_credit(instalment: float, term_months: float, initiation_fee: float = missing_as(0.0)) -> float:
    return round(instalment * term_months + initiation_fee, 2)


def total_interest(total_cost_of_credit: float, offered_amount: float) -> float:
    return round(total_cost_of_credit - offered_amount, 2)


def effective_annual_rate(nominal_annual_rate: float) -> float:
    # ponytail: the nominal rate compounded monthly, not a fees-inclusive APR solve; add the solve for disclosures.
    return (1.0 + nominal_annual_rate / 12.0) ** 12 - 1.0


def solve_advance_for_instalment(
    target_instalment_before_fees: float, term_months: int, nominal_annual_rate: float
) -> float:
    """The advance whose instalment is `target_instalment_before_fees` (the inverse of `instalment_before_fees`).

    Plain function for solvers that call it many times; `solve_advance_for_instalment(instalment_before_fees(a, n, r), n, r) == a`.
    """
    r = nominal_annual_rate / 12.0
    if r == 0.0:
        return target_instalment_before_fees * term_months
    return target_instalment_before_fees * (1.0 - (1.0 + r) ** (-term_months)) / r


instalment_before_fees_step = step(instalment_before_fees)
instalment_step = step(instalment)
total_cost_of_credit_step = step(total_cost_of_credit)
total_interest_step = step(total_interest)
effective_annual_rate_step = step(effective_annual_rate)
