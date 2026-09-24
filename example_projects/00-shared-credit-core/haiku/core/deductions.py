from __future__ import annotations
"""Statutory deductions (00 §6.2)."""
from dataclasses import dataclass


@dataclass
class DeductionsResult:
    """Result of statutory deduction calculation."""
    statutory_deductions: float
    net_monthly_income: float


def calculate_deductions(
    gross_monthly_income: float,
    employment_type_code: int,
    tax_year: int = 2024
) -> DeductionsResult:
    """
    Calculate statutory deductions (tax, UIF, compulsory retirement).

    Effective-dated tax tables by decision_date, not today's date.
    This is a simplified example; real tax calculation would be table-driven.
    """
    # Simplified tax calculation (2024 rates as example)
    # This would normally be driven by effective-dated table lookups

    if employment_type_code == 6:  # Pensioner
        # No employment tax
        deductions = gross_monthly_income * 0.01  # Minimal deduction
    elif employment_type_code == 5:  # Social grant
        deductions = 0.0  # No deductions on social grants
    else:
        # Standard employment deductions: roughly 20% for tax, UIF, pension
        deductions = gross_monthly_income * 0.20

    net = gross_monthly_income - deductions

    return DeductionsResult(
        statutory_deductions=deductions,
        net_monthly_income=max(0.0, net)  # Never negative
    )
