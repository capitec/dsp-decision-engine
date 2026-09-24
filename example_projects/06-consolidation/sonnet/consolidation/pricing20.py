"""Product 20 (Everyday Card balance transfer) pricing (spec 06 §5.6.6).

"The odd one out": no term, no instalment, a promotional rate for a fixed
period then a reversion rate. Written from scratch -- nothing in project 03
has this shape (its solve and pricing are both term-loan, single-rate). Uses
`consolidation.rate_cards.Product20CardIndex` for the two small rate tables
and `credit_core.fees`/`credit_core.rounding` (project 00) for the shared
fee and rounding conventions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from credit_core.rounding import round_advance, round_instalment

from consolidation.rate_cards import Product20CardIndex

MIN_LIMIT = 1_000.0
MAX_LIMIT = 300_000.0
TRANSFER_UTILISATION_CAP = 0.80
TRANSFER_FEE_PCT = 0.025
MIN_PAYMENT_FLOOR = 50.0
MIN_PAYMENT_PCT = 0.03
MANDATORY_PAYDOWN_MONTHS = 36
STRESSED_AMORTISATION_MONTHS = 36
_MAX_SIMULATED_MONTHS = 84  # a bound on the total-cost simulation, not a product term


@dataclass(frozen=True)
class Product20PricingResult:
    approved_limit: float
    transferred_amount: float
    transfer_fee: float
    promo_rate: float | None
    promo_duration_months: int
    reversion_rate: float | None
    promo_cell_id: str | None
    reversion_cell_id: str | None
    monthly_payment: float                # promotional-period contractual payment
    stressed_payment: float                # affordability is tested against this, never the promo payment
    total_cost_of_credit: float            # over the mandatory paydown schedule, promo then reversion
    priced: bool


def required_limit(transferred_amount: float) -> float:
    """§5.6.6: transferred balance <= 80% of the approved limit."""
    raw = transferred_amount / TRANSFER_UTILISATION_CAP
    return round_advance(max(MIN_LIMIT, min(MAX_LIMIT, raw)))


def _minimum_payment(balance: float, monthly_rate: float) -> float:
    interest = balance * monthly_rate
    return max(MIN_PAYMENT_FLOOR, balance * MIN_PAYMENT_PCT + interest)


def _simulate_total_cost(
    opening_balance: float, promo_rate: float, promo_duration_months: int, reversion_rate: float,
) -> tuple[float, float]:
    """Amortises the transferred balance under its mandatory paydown schedule -- the
    minimum payment plus 1/36th of the transferred amount every month -- at the
    promotional rate for the promotional period and the reversion rate thereafter
    (§5.6.6 "Total cost comparison": never at the promotional rate throughout).
    Returns (total_paid, first_month_payment)."""
    mandatory_component = opening_balance / MANDATORY_PAYDOWN_MONTHS
    balance = opening_balance
    total_paid = 0.0
    first_payment = None
    for month in range(1, _MAX_SIMULATED_MONTHS + 1):
        if balance <= 0.01:
            break
        rate = promo_rate if month <= promo_duration_months else reversion_rate
        monthly_rate = rate / 12.0
        payment = round_instalment(max(_minimum_payment(balance, monthly_rate), mandatory_component))
        interest = balance * monthly_rate
        payment = min(payment, balance + interest)  # never overpay past the final settlement
        if first_payment is None:
            first_payment = payment
        balance = balance + interest - payment
        total_paid += payment
    return round(total_paid, 2), round(first_payment or 0.0, 2)


def price_product20(
    card_index: Product20CardIndex, transferred_amount: float, risk_grade: int, promo_duration_months: int,
) -> Product20PricingResult:
    limit = required_limit(transferred_amount)
    transfer_fee = round(transferred_amount * TRANSFER_FEE_PCT, 2)
    opening_balance = round(transferred_amount + transfer_fee, 2)

    promo_hit = card_index.lookup_promo(limit, promo_duration_months, risk_grade)
    reversion_hit = card_index.lookup_reversion(limit, risk_grade)
    if promo_hit is None or reversion_hit is None:
        return Product20PricingResult(
            limit, transferred_amount, transfer_fee, None, promo_duration_months, None, None, None,
            0.0, 0.0, 0.0, priced=False,
        )
    promo_rate, promo_cell_id = promo_hit
    reversion_rate, reversion_cell_id = reversion_hit

    total_cost, first_payment = _simulate_total_cost(opening_balance, promo_rate, promo_duration_months, reversion_rate)

    # Stressed affordability (§5.6.6): the reversion rate applied to the *full approved
    # limit*, amortised over 36 months -- never the promotional minimum payment.
    r = reversion_rate / 12.0
    n = STRESSED_AMORTISATION_MONTHS
    stressed = limit * r / (1.0 - (1.0 + r) ** (-n)) if r else limit / n
    stressed_payment = round_instalment(stressed)

    return Product20PricingResult(
        approved_limit=limit, transferred_amount=transferred_amount, transfer_fee=transfer_fee,
        promo_rate=promo_rate, promo_duration_months=promo_duration_months, reversion_rate=reversion_rate,
        promo_cell_id=promo_cell_id, reversion_cell_id=reversion_cell_id,
        monthly_payment=first_payment, stressed_payment=stressed_payment,
        total_cost_of_credit=total_cost, priced=True,
    )
