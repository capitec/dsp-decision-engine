"""Flex Loan assessment pipeline.

Composed from shared library capabilities:
- Income: income determination
- Affordability: discretionary income and capacity
- Fees: fee calculation

This is the main flow for a generic unsecured term loan.
Implements: Product Design §10 (Flex Loan)
"""

from pydantic import BaseModel, Field

from decider2 import flow, module

from ..modules.affordability import Affordability, AffordabilityParams
from ..modules.fees import Fees, FeesParams
from ..modules.income import Income, IncomeParams


class FlexLoanSharedParams(BaseModel):
    """Shared parameters across the Flex Loan pipeline."""

    prime_rate: float = Field(
        8.5, ge=0.0, le=20.0, description="Base prime lending rate"
    )
    product_margin: float = Field(
        2.5, ge=0.0, le=10.0, description="Margin on top of prime rate"
    )


def offered_amount(
    requested_amount: float,
    max_affordable_instalment: float,
    params,
    shared,
) -> float:
    """Determine offered amount based on affordability and policy limits.

    Implements: Product Policy §10.2
    """
    # Simplified: capped at requested amount and at a product maximum
    product_max = params.product_maximum_amount
    return min(requested_amount, product_max)


def nominal_annual_rate(shared) -> float:
    """Determine offer rate.

    Simplified: prime + margin
    Implements: Credit Policy §6.9
    """
    return shared.prime_rate + shared.product_margin


def term_months(
    requested_term: int,
    params,
) -> int:
    """Determine loan term.

    Applies product constraints.
    Implements: Product Policy §10.3
    """
    return min(requested_term, params.product_maximum_term)


def total_cost_of_credit(
    offered_amount: float,
    term_months: int,
    nominal_annual_rate: float,
    monthly_service_fee: float,
    initiation_fee: float,
) -> float:
    """Simplified total cost of credit calculation.

    Actual implementation would use proper amortisation.
    Implements: Credit Policy §6.6
    """
    monthly_rate = nominal_annual_rate / 100.0 / 12.0
    if monthly_rate == 0:
        interest = 0.0
    else:
        interest = offered_amount * monthly_rate * term_months

    total_fees = initiation_fee + (monthly_service_fee * term_months)
    return offered_amount + interest + total_fees


class FlexLoanParams(BaseModel):
    """Flex Loan product-specific parameters."""

    product_maximum_amount: float = Field(
        500000.0, ge=0.0, description="Product maximum advance"
    )
    product_minimum_amount: float = Field(
        2000.0, ge=0.0, description="Product minimum advance"
    )
    product_maximum_term: int = Field(
        84, ge=1, description="Product maximum term in months"
    )
    product_minimum_term: int = Field(
        6, ge=1, description="Product minimum term in months"
    )


# Assemble inline modules for term and rate
TermDecision = module(term_months, name="term_decision", params=FlexLoanParams)
OfferedAmount = module(
    offered_amount, name="offered_amount", params=FlexLoanParams
)
RateCard = module(nominal_annual_rate, name="rate_card")
CostCalculation = module(total_cost_of_credit, name="cost")

# Main pipeline
flex_loan = (
    Income
    | Affordability
    | Fees
    | TermDecision
    | OfferedAmount
    | RateCard
    | CostCalculation
)

__all__ = [
    "flex_loan",
    "FlexLoanSharedParams",
    "FlexLoanParams",
    "IncomeParams",
    "AffordabilityParams",
    "FeesParams",
]
