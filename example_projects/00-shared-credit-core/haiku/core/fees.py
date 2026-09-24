from __future__ import annotations
"""Fee calculation (00 §6.7)."""
from dataclasses import dataclass


@dataclass
class FeesResult:
    """Result of fee calculation."""
    initiation_fee: float
    monthly_service_fee: float
    credit_life_premium: float
    total_fees: float


def calculate_fees(
    amount: float,
    product_code: int,
    term_months: int
) -> FeesResult:
    """Calculate statutory capped fees."""
    # Simplified fee calculation
    # Real implementation: table-driven by amount bands and product

    # Initiation fee: capped at ~1% of amount
    initiation_fee = min(amount * 0.01, 250.0)

    # Monthly service fee: ~R50
    monthly_service_fee = 50.0

    # Credit life premium: ~0.1% of outstanding monthly
    credit_life_premium = amount / term_months * 0.001 if term_months > 0 else 0.0

    total = initiation_fee + (monthly_service_fee * term_months) + (credit_life_premium * term_months)

    return FeesResult(
        initiation_fee=initiation_fee,
        monthly_service_fee=monthly_service_fee,
        credit_life_premium=credit_life_premium,
        total_fees=total
    )
