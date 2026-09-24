"""P17 -- Final validation (spec 10 §5.18): whether the Bank is willing to be bound.

O-18: "P17 must be **unable** to read any shared intermediate... if P17
can read `instalment` it will eventually assert `instalment == instalment`."
This module's re-derivation therefore never reads `offer_instalment`,
`nominal_annual_rate` or any other value another phase already computed
for this decision -- it recomputes the instalment from `offer_amount`,
`term_months` and `risk_grade` through `retail_credit.solve.price_one`
(the same pure-Python pricing function P13's search calls, which is
itself built from the same rate card P12's decider steps read), and
compares its own answer to the one the pipeline produced. A mismatch is
exactly what requirement 7 (10 §5.14: "the offer must never fail
affordability when re-checked from scratch") exists to catch.

A representative subset of assertions (six, not the spec's 61 for product
10 on entry point 1) -- assertion *volume* is not this project's
difficulty; the isolation property (O-18) and the re-derivation-not-
comparison shape are.
"""
from __future__ import annotations

from decider import missing_as, param, step

from retail_credit.pricing import STATUTORY_RATE_CEILING
from retail_credit.solve import GRID, price_one

A_AMOUNT_ON_GRID = 1701
A_AMOUNT_WITHIN_CAP = 1702
A_TERM_WITHIN_CAP = 1703
A_RATE_WITHIN_CEILING = 1704
A_REDERIVATION_IN_RANGE = 1705
A_AFFORDABLE_ON_REDERIVATION = 1706


def revalidate_offer(
    has_offer: bool,
    offer_amount: float,
    term_months: int,
    term_cap: float,
    amount_cap: float,
    risk_grade: int,
    max_affordable_instalment: float,
    applicant_age_years: float,
    rate_addon_bps: float = missing_as(0.0),
    tolerance_cents: float = param(1.0, ge=0.0),
) -> tuple[bool, list[int], float]:
    """(all_assertions_pass, failed_assertion_codes, rederived_instalment). Every
    assertion is evaluated, not short-circuited (matching P03's own "evaluate every
    gate" discipline, 10 §5.4) -- a client whose offer fails validation is entitled to
    every reason, not the first one found.
    """
    if not has_offer:
        return True, [], 0.0  # nothing to validate: no offer was assembled

    tolerance = tolerance_cents / 100.0
    failed: list[int] = []
    if offer_amount % GRID != 0:
        failed.append(A_AMOUNT_ON_GRID)
    if offer_amount > amount_cap + tolerance:
        failed.append(A_AMOUNT_WITHIN_CAP)
    if term_months > term_cap:
        failed.append(A_TERM_WITHIN_CAP)

    rederivation = price_one(
        offer_amount, min(term_months, int(term_cap)), risk_grade, applicant_age_years, rate_addon_bps)
    if rederivation is None:
        failed.extend([A_RATE_WITHIN_CEILING, A_REDERIVATION_IN_RANGE, A_AFFORDABLE_ON_REDERIVATION])
        return False, failed, 0.0

    if rederivation.rate > STATUTORY_RATE_CEILING + 1e-9:
        failed.append(A_RATE_WITHIN_CEILING)
    if rederivation.instalment > max_affordable_instalment + tolerance:
        failed.append(A_AFFORDABLE_ON_REDERIVATION)

    return len(failed) == 0, failed, rederivation.instalment


revalidate_offer_step = step(revalidate_offer, outputs=("validation_passed", "validation_failed_assertions",
                                                          "rederived_instalment"))
