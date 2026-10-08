"""Pricing one offered amount: rate card cell, long-term loading, rate add-on, fees, credit life and instalment."""
from __future__ import annotations

from typing import TypedDict

from decider import flow, param, step
from decider.ops import Struct

from credit_core import credit_life, fees, instalment, rounding
from retail_credit import flex_card

STATUTORY_RATE_CEILING = 0.2825  # 28.25% at a 7.25% repo rate (illustrative)


class Quote(TypedDict):
    """What the solve keeps of its winning candidate, so the offer is not priced a second time."""

    offered_amount: float
    nominal_annual_rate: float
    nominal_annual_rate_unadjusted: float
    amt_band: int
    term_band: int
    initiation_fee: float
    initiation_fee_incl_tax: float
    amount_financed: float
    credit_life_premium: float
    monthly_service_fee: float
    instalment_before_fees: float
    instalment: float


def offer_term_months(term_months: int, term_cap: float) -> float:
    return float(min(term_months, term_cap))


def long_term_loading(
    offer_term_months: float,
    from_46: float = param(0.006, ge=0.0, le=0.05), from_61: float = param(0.011, ge=0.0, le=0.05),
    from_73: float = param(0.017, ge=0.0, le=0.05),
) -> float:
    """Terms past the card's last column (45 months) are priced from it plus a loading."""
    if offer_term_months <= 45:
        return 0.0
    return from_46 if offer_term_months <= 60 else from_61 if offer_term_months <= 72 else from_73


def initiation_fee_cap(excl_tax_cap: float = param(1480.0, ge=0.0)) -> float:
    return excl_tax_cap


def monthly_service_fee(incl_tax_fee: float = param(87.98, ge=0.0)) -> float:
    return incl_tax_fee


def nominal_annual_rate_unadjusted(card_rate: float, long_term_loading: float) -> float:
    return round(card_rate + long_term_loading, 6)


def nominal_annual_rate(nominal_annual_rate_unadjusted: float, rate_addon_bps: float) -> float:
    """`rate_addon_bps` is the overlay register resolved once per decision, before the solve."""
    return round(nominal_annual_rate_unadjusted + rate_addon_bps, 6)


def initiation_fee_incl_tax(initiation_fee: float, tax_rate: float = param(0.15, ge=0.0, le=0.25)) -> float:
    return round(initiation_fee * (1.0 + tax_rate), 2)


def amount_financed(offered_amount: float, initiation_fee_incl_tax: float) -> float:
    """The fee is capitalised: the client receives `offered_amount` and finances this larger sum."""
    return round(offered_amount + initiation_fee_incl_tax, 2)


def record(
    offered_amount: float, nominal_annual_rate: float, nominal_annual_rate_unadjusted: float, amt_band: int,
    term_band: int, initiation_fee: float, initiation_fee_incl_tax: float, amount_financed: float,
    credit_life_premium: float, monthly_service_fee: float, instalment_before_fees: float, instalment: float,
) -> Struct[Quote]:
    return (offered_amount, nominal_annual_rate, nominal_annual_rate_unadjusted, amt_band, term_band, initiation_fee,
            initiation_fee_incl_tax, amount_financed, credit_life_premium, monthly_service_fee,
            instalment_before_fees, instalment)


def amount_independent_unit():
    """Everything the price needs that does not depend on the amount, looked up once per decision.

    Reads `offer_term_months`, `applicant_age_years` and `cover_type_code`; writes `long_term_loading`,
    `initiation_fee_cap`, `monthly_service_fee`, `credit_life_rate` and `credit_life_cell_id`.
    """
    life = credit_life.build_credit_life_table().relabel(
        reads={"term_months": "offer_term_months"}, writes={"rate": "credit_life_rate", "cell_id": "credit_life_cell_id"})
    return flow(long_term_loading, initiation_fee_cap, monthly_service_fee, life, name="fixed_terms")


def candidate_unit():
    """Prices `offered_amount`: reads the unit above, `risk_grade`, `rate_addon_bps`; writes `record` (a `Quote`).

    The solve scores every candidate with this unit, so the offer it ships is priced by the steps that chose it.
    """
    return flow(
        step(flex_card.card_rate, outputs=("card_rate", "amt_band", "term_band")),
        nominal_annual_rate_unadjusted, nominal_annual_rate, fees.initiation_fee_step, initiation_fee_incl_tax,
        amount_financed,
        credit_life.credit_life_premium_step.relabel(reads={"offered_amount": "amount_financed", "rate": "credit_life_rate"}),
        instalment.instalment_before_fees_step.relabel(reads={"offered_amount": "amount_financed", "term_months": "offer_term_months"}),
        instalment.instalment_step, rounding.round_instalment_step.relabel(reads={"amount": "instalment"}),
        step(record, output="record"),
        name="candidate",
    )


def statutory_ceiling_ok(nominal_annual_rate: float, ceiling: float = param(STATUTORY_RATE_CEILING)) -> bool:
    """Checked on the rate after the add-on: a card that was compliant plus an add-on can still breach the ceiling."""
    return nominal_annual_rate <= ceiling
