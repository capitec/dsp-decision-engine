"""Access Facility (revolving credit facility) assessment pipeline.

Demonstrates module reuse with relabeling to adapt the shared library
to different column names.

The Access Facility product uses the same income and affordability logic
as Flex Loan, but with different terminology and constraints.

Implements: Product Design §21 (Access Facility)
"""

from pydantic import BaseModel, Field

from decider2 import flow, module

from ..modules.affordability import Affordability
from ..modules.fees import Fees
from ..modules.income import Income


class AccessFacilitySharedParams(BaseModel):
    """Shared parameters across the Access Facility pipeline."""

    prime_rate: float = Field(
        8.5, ge=0.0, le=20.0, description="Base prime lending rate"
    )
    product_margin: float = Field(
        3.0, ge=0.0, le=10.0, description="Margin for revolving products"
    )


def approved_limit(
    max_affordable_instalment: float,
    params,
) -> float:
    """Determine approved credit limit.

    For revolving products, this is the maximum the client may draw.
    Implements: Product Policy §21.2
    """
    return min(
        max_affordable_instalment * params.limit_multiplier,
        params.product_maximum_limit,
    )


def monthly_interest_rate(shared) -> float:
    """Interest rate for revolving facility.

    Implements: Credit Policy §6.9 (adapted for revolving)
    """
    return (shared.prime_rate + shared.product_margin) / 100.0 / 12.0


class AccessFacilityParams(BaseModel):
    """Access Facility product-specific parameters."""

    product_maximum_limit: float = Field(
        300000.0, ge=0.0, description="Product maximum credit limit"
    )
    product_minimum_limit: float = Field(
        1000.0, ge=0.0, description="Product minimum credit limit"
    )
    limit_multiplier: float = Field(
        12.0, ge=1.0, description="Limit as multiple of monthly affordability"
    )


# Assemble inline modules for Access Facility-specific logic
LimitDecision = module(
    approved_limit, name="limit_decision", params=AccessFacilityParams
)
InterestRate = module(monthly_interest_rate, name="interest_rate")

# Main pipeline: reuse Income and Affordability from shared library
# The income module produces 'net_monthly_income', which Affordability expects
access_facility = (
    Income
    | Affordability
    | Fees
    | LimitDecision
    | InterestRate
)

__all__ = [
    "access_facility",
    "AccessFacilitySharedParams",
    "AccessFacilityParams",
]
