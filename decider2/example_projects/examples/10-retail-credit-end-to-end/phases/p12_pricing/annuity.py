"""P12(d) the instalment and effective rate. Spec 5.13(d). Last in O-14's
within-phase order: rate -> fee -> premium -> instalment, because each
depends on the last.

`instalment` is a `shared_value` with all three axes (values/register.py) —
the only one with all three — because it is read by P10 back through the
loop (back_edges_permitted_via=("L1","L3")) as well as forward by six phases.
"""

from __future__ import annotations

from decider2 import module, round_half_up

def ordinary_annuity_instalment(amount_financed: float, nominal_annual_rate: float,
                                term_months: int) -> float:
    pass  # standard monthly annuity formula, amount_financed at nominal_annual_rate/12

def instalment(ordinary_annuity_instalment: float, monthly_service_fee: float,
              credit_life_premium: float) -> float:
    """Rounded to the cent — O-22's one place per value."""
    pass  # round_half_up(ordinary_annuity_instalment + monthly_service_fee + credit_life_premium, 2)

def total_cost_of_credit(instalment: float, term_months: int) -> float:
    pass  # instalment * term_months

def effective_annual_rate(amount_financed: float, instalment: float, term_months: int) -> float:
    """The annualised IRR on the amount ADVANCED against the full instalment
    stream — exceeds the nominal rate substantially at short terms."""
    pass  # solve the IRR of the instalment stream against amount_financed, annualise

Instalment = module(ordinary_annuity_instalment, instalment, total_cost_of_credit,
                    effective_annual_rate, name="annuity")
