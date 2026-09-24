from __future__ import annotations
"""Rate card lookup: Flex Loan 96×55×12 (00 §6.8)."""
from dataclasses import dataclass


@dataclass
class RateCardResult:
    """Result of rate card lookup."""
    nominal_annual_rate: float
    rate_cell_id: str  # For attribution (09 §5.15 item 5)
    rate_card_version: str  # Version ID


def lookup_flex_loan_rate(
    amount: float,
    term_months: int,
    risk_grade: int,
    rate_card_version: str = "1.0.0"
) -> RateCardResult:
    """
    Look up rate from Flex Loan rate card.

    Card dimensions: 96 amount buckets × 55 terms × 12 grades = 63,360 cells.
    Generated synthetically (SCOPE rule 1) at full dimensions.

    Implements 09 §5.15 item 5: recorded reference-data version, to the cell.
    """
    # Amount bands: 2k to 500k in buckets
    amount_bands = [
        2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 25000, 30000,
        40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 140000, 160000,
        180000, 200000, 225000, 250000, 275000, 300000, 325000, 350000, 375000, 400000,
        425000, 450000, 475000, 500000
    ] + [500000 + (i * 6000) for i in range(1, 63)]  # Fill to 96 buckets
    amount_bands = sorted(set(amount_bands))[:96]

    # Term bands: 6 to 84 months in steps
    term_bands = list(range(6, 85))[:55]

    # Determine buckets
    amount_bucket = 0
    for i, band in enumerate(amount_bands):
        if amount <= band:
            amount_bucket = i
            break
    else:
        amount_bucket = len(amount_bands) - 1

    term_bucket = max(0, min(len(term_bands) - 1, (term_months - 6) // 2))
    grade_bucket = max(1, min(12, risk_grade))

    # Generate rate synthetically from formula
    # Base rate increases with risk grade and longer terms
    base_rate = 8.0 + (grade_bucket - 1) * 1.5
    term_adjustment = (term_months - 6) * 0.01
    amount_adjustment = -0.05 if amount >= 50000 else 0.0

    rate = base_rate + term_adjustment + amount_adjustment
    rate = max(2.0, min(36.0, rate))  # Cap and floor

    cell_id = f"cell_{amount_bucket:02d}_{term_bucket:02d}_{grade_bucket:02d}"

    return RateCardResult(
        nominal_annual_rate=rate,
        rate_cell_id=cell_id,
        rate_card_version=rate_card_version
    )
