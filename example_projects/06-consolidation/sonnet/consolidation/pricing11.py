"""Product 11 (Flex Loan Consolidation) pricing (spec 06 §5.6.4).

Reuses project 03's `PriceEvaluator` / `RateCardIndex` / `CreditLifeIndex`
**unchanged** -- "the two 06/07 want" per 03's own NOTES.md -- over 06's own
rate card (`rate_cards.generate_product11_card`). This is DEPS.md's "reused
four times over inputs project 03 never contemplated": a different card, the
identical lookup-and-price mechanism.

**The required-advance circular solve** (§5.6.2 item 3) is *not* 03's
`solve_term` (the max-affordable-amount search) -- 06's amount is not a free
variable to maximise, it is the settlement total plus new money plus its own
capitalised fees, and those fees depend on the amount they are capitalised
into. That circularity is a fixed-point iteration (declared bound: 6
iterations, R1 tolerance -- §5.6.2), a different problem from 03's
bisection-for-a-maximum even though DEPS.md calls both "the circular solve".
See NOTES.md "Framework friction" for why `solve_term` itself could not be
reused unchanged here regardless (`PRODUCT_MIN_AMOUNT`/`PRODUCT_MAX_AMOUNT`
are module constants in `loan_granting.solve`, not parameters).
"""
from __future__ import annotations

from dataclasses import dataclass

from loan_granting.pricing import CreditLifeIndex, PriceEvaluator, PricingResult, RateCardIndex

from consolidation.rate_cards import generate_product11_card, margin_adjustment_bps

MAX_ITERATIONS = 6
CONVERGENCE_TOLERANCE = 1.0  # rand


@dataclass(frozen=True)
class RequiredAdvanceResult:
    converged: bool
    iterations: int
    offered_amount: float
    pricing: PricingResult


def build_product11_evaluator(statutory_ceiling: float, rows: list[dict] | None = None) -> PriceEvaluator:
    """`rows` normally comes from `configs/<version>/rate_card_product11.json`
    (`RateCardIndex.from_configurable_step`, 03's own convention: 00 §9, "no
    redeploy"); regenerated in-process only as a fallback (tests, or no config
    loaded) so this stays usable standalone."""
    rows = rows if rows is not None else generate_product11_card()["rows"]
    return PriceEvaluator(
        rate_card_index=RateCardIndex(rows), credit_life_index=CreditLifeIndex(),
        statutory_ceiling=statutory_ceiling, credit_life_substitution_declared=False,
    )


def solve_required_advance(
    evaluator: PriceEvaluator, base_amount: float, term_months: int, risk_grade: int,
    applicant_age_years: float, external_proportion: float, rate_addon_bps: float = 0.0,
) -> RequiredAdvanceResult:
    """§5.6.2 item 3: capitalised initiation fee and credit-life premium both depend on
    the advance; the advance depends on them. Fixed-point iterate from `base_amount`
    (the uncapitalised settlement total + buffer + new money) until the amount stops
    moving by more than R1, or the iteration bound is reached -- non-convergence is a
    scenario rejection (`interventions.REJ_SOLVE_NOT_CONVERGED`), never a thrown error."""
    amount = base_amount
    margin = margin_adjustment_bps(external_proportion, risk_grade)
    pricing = evaluator.evaluate(amount, term_months, risk_grade, applicant_age_years, False,
                                  rate_addon_bps=rate_addon_bps + margin)
    for i in range(1, MAX_ITERATIONS + 1):
        if not pricing.priced:
            return RequiredAdvanceResult(False, i, amount, pricing)
        next_amount = round(base_amount + (pricing.amount_financed - amount), 2)
        # `amount_financed = amount + initiation_fee`, so the capitalised-fee delta is
        # `amount_financed - amount`; credit life is priced on `amount_financed`
        # already, so it is captured in the same delta.
        if abs(next_amount - amount) <= CONVERGENCE_TOLERANCE:
            return RequiredAdvanceResult(True, i, next_amount, pricing)
        amount = next_amount
        pricing = evaluator.evaluate(amount, term_months, risk_grade, applicant_age_years, False,
                                      rate_addon_bps=rate_addon_bps + margin)
    return RequiredAdvanceResult(False, MAX_ITERATIONS, amount, pricing)
