from __future__ import annotations
"""Instalment calculation (00 §6.6)."""
from dataclasses import dataclass
from .rounding import round_instalment
import math


@dataclass
class InstalmentResult:
    """Result of instalment calculation."""
    instalment: float  # Monthly payment including fees
    total_cost_of_credit: float  # Sum of all payments
    effective_annual_rate: float  # All-in annual rate


def calculate_instalment(
    amount: float,
    term_months: int,
    nominal_annual_rate: float,
    initiation_fee: float = 0.0,
    monthly_service_fee: float = 0.0,
    credit_life_premium: float = 0.0
) -> InstalmentResult:
    """
    Calculate the monthly instalment and total cost of credit.
    Implements the standard amortization formula for a term loan.

    Implements 09 §5.15 items 6 & 16 by capturing all inputs and returning
    all outputs (including score contributions, handled by caller).
    """
    monthly_rate = nominal_annual_rate / 12 / 100

    # Principal portion of instalment
    if monthly_rate > 0:
        principal_instalment = (
            amount * (monthly_rate * (1 + monthly_rate) ** term_months) /
            ((1 + monthly_rate) ** term_months - 1)
        )
    else:
        principal_instalment = amount / term_months

    # Total monthly payment includes all fees
    total_monthly = principal_instalment + monthly_service_fee + credit_life_premium
    total_monthly = round_instalment(total_monthly)

    # Total cost of credit
    total_paid = total_monthly * term_months + initiation_fee
    total_cost = total_paid - amount

    # Effective annual rate (IRR approximation)
    # For simplicity, using approximation: (total_cost / amount) / (term_months / 12) * 100
    if term_months > 0:
        ear = (total_cost / amount) / (term_months / 12) * 100
    else:
        ear = 0.0

    return InstalmentResult(
        instalment=total_monthly,
        total_cost_of_credit=round_instalment(total_cost),
        effective_annual_rate=round(ear, 2)
    )


def calculate_max_affordable_amount(
    net_income: float,
    living_expenses: float,
    existing_obligations: float,
    term_months: int,
    nominal_annual_rate: float,
    max_dti_ratio: float = 0.35,
    monthly_service_fee: float = 0.0,
    credit_life_premium: float = 0.0,
    initiation_fee: float = 0.0
) -> float:
    """
    Calculate the maximum affordable loan amount based on affordability constraints.
    Inverse of instalment calculation (00-ADDENDUM C item 6).

    Uses iterative approximation to solve for maximum amount.
    """
    discretionary = net_income - living_expenses - existing_obligations
    if discretionary <= 0:
        return 0.0

    max_instalment = discretionary * max_dti_ratio

    # Binary search for maximum affordable amount
    low, high = 0.0, discretionary * 10
    epsilon = 0.01  # Tolerance of 1 cent

    for _ in range(50):  # Max iterations to avoid infinite loop
        mid = (low + high) / 2
        result = calculate_instalment(
            mid, term_months, nominal_annual_rate,
            initiation_fee, monthly_service_fee, credit_life_premium
        )

        if abs(result.instalment - max_instalment) < epsilon:
            return round_instalment(mid)
        elif result.instalment < max_instalment:
            low = mid
        else:
            high = mid

    return round_instalment((low + high) / 2)
