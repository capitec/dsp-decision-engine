"""What every product that can carry a consolidation offers the search, and the policy each product team owns.

A product answers four questions: may this client use it, can it carry this set of accounts, which terms are worth
pricing, and what does it cost. The search, the interventions and the objective only ever ask these, so a new product
is a new module and one line in `product_policies`; none of them change.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from decider import allow_fallback, step


@dataclass(frozen=True)
class Quote:
    """One priced scenario. `tested_payment` is what affordability is tested against: the instalment for a loan, the
    stressed payment for a card (never its promotional minimum)."""

    amount: float
    instalment: float
    tested_payment: float
    nominal_annual_rate: float
    total_cost_of_credit: float
    effective_annual_rate: float
    rate_cell_id: str
    rate_card_version: str
    fees: dict = field(default_factory=dict)


@allow_fallback
@step(output="pricing_context")
def pricing_context(statutory_ceiling: float, initiation_fee_cap: float, monthly_service_fee_cap: float) -> dict:
    """The ceiling and fee caps in force on the decision date, which every product prices under."""
    return {"statutory_ceiling": statutory_ceiling, "initiation_fee_cap": initiation_fee_cap,
            "monthly_service_fee_cap": monthly_service_fee_cap}


@allow_fallback
@step(output="product_policies")
def product_policies(flex_loan_policy: dict, everyday_card_policy: dict) -> dict:
    """Every product's policy, by product code. A product team changes its own params and nobody else's."""
    # workaround: a dict output cannot have int keys in a batch run, so the product code is text
    return {str(p["product_code"]): p for p in (flex_loan_policy, everyday_card_policy)}
