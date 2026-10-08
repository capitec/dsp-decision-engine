"""Offer assembly: whether there is an offer at all, and the figures of the one the solve found."""
from __future__ import annotations

from decider import allow_fallback, step

from credit_core.affordability import MARGINAL, PASS
from credit_core.vocab import FraudVerdictCode


def decision_open(is_eligible: bool, product_10_eligible: bool, p09_declined: bool, fraud_verdict_code: int) -> bool:
    """False once something has already declined the request, so no offer is made and no reason about the amount is given.
    A fraud refer leaves the decision open: the outcome refers, the offer is indicative."""
    return is_eligible and product_10_eligible and not p09_declined and fraud_verdict_code != FraudVerdictCode.DECLINE


@allow_fallback
def assemble_offer(
    decision_open: bool, offer_affordability_verdict_code: int, solved_amount: float, solve_binding_constraint: str,
) -> tuple[float, str, bool]:
    """(offer_amount, offer_binding_constraint, has_offer): an offer needs an open decision, an affordable basis and
    an amount the solve found."""
    if decision_open and offer_affordability_verdict_code in (PASS, MARGINAL) and solved_amount > 0.0:
        return solved_amount, solve_binding_constraint, True
    return 0.0, "", False


@allow_fallback
def priced_offer(has_offer: bool, record: dict | None) -> tuple[float, float, int, int, float, float, float, float, float, float]:
    """The winning candidate's figures, or zeros when there is no offer."""
    names = ("nominal_annual_rate", "nominal_annual_rate_unadjusted", "amt_band", "term_band", "initiation_fee",
             "initiation_fee_incl_tax", "amount_financed", "credit_life_premium", "instalment_before_fees", "instalment")
    return tuple(record[name] for name in names) if has_offer and record else (0.0, 0.0, -1, -1) + (0.0,) * 6


def total_cost_of_credit(instalment: float, offer_term_months: float) -> float:
    """Every instalment paid: the capitalised initiation fee is inside them, so it is not added again."""
    return round(instalment * offer_term_months, 2)


def total_interest(instalment_before_fees: float, offer_term_months: float, amount_financed: float) -> float:
    return round(instalment_before_fees * offer_term_months - amount_financed, 2) if amount_financed else 0.0


assemble_offer_step = step(assemble_offer, outputs=("offer_amount", "offer_binding_constraint", "has_offer"))
priced_offer_step = step(priced_offer, outputs=(
    "nominal_annual_rate", "nominal_annual_rate_unadjusted", "amt_band", "term_band", "initiation_fee",
    "initiation_fee_incl_tax", "amount_financed", "credit_life_premium", "instalment_before_fees", "instalment"))
