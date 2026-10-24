"""Final validation: whether the Bank is willing to be bound by the offer, checked on the figures it will be bound to."""
from __future__ import annotations

from decikit import allow_fallback, param, step

A_AMOUNT_ON_GRID = 1701
A_AMOUNT_WITHIN_CAP = 1702
A_TERM_WITHIN_CAP = 1703
A_RATE_WITHIN_CEILING = 1704
A_INSTALMENT_AFFORDABLE = 1705


@allow_fallback
def revalidate_offer(
    has_offer: bool, offer_amount: float, offer_term_months: float, term_cap: float, amount_cap: float,
    nominal_annual_rate: float, instalment: float, instalment_ceiling: float, amount_grid: float,
    rate_ceiling: float = param(0.2825, gt=0.0), tolerance: float = param(0.01, ge=0.0),
) -> tuple[bool, list[int]]:
    """(validation_passed, validation_failed_assertions): every assertion is checked, so a failed offer lists all its
    failures. The instalment is checked against the ceiling of the basis the offer was priced on."""
    if not has_offer:
        return True, []
    failed = [
        (offer_amount % amount_grid != 0, A_AMOUNT_ON_GRID), (offer_amount > amount_cap + tolerance, A_AMOUNT_WITHIN_CAP),
        (offer_term_months > term_cap, A_TERM_WITHIN_CAP), (nominal_annual_rate > rate_ceiling, A_RATE_WITHIN_CEILING),
        (instalment > instalment_ceiling + tolerance, A_INSTALMENT_AFFORDABLE),
    ]
    codes = [code for hit, code in failed if hit]
    return not codes, codes


revalidate_offer_step = step(revalidate_offer, outputs=("validation_passed", "validation_failed_assertions"))
