from __future__ import annotations
"""Income determination (00 §6.1)."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class IncomeResult:
    """Result of income determination."""
    gross_monthly_income: float
    income_source_code: int  # Evidence tier (1-6)
    income_verification_tier: str  # "declared", "payslip", "statement", "employer", "bureau"
    income_haircut_applied: float  # Percentage (0-1)
    income_variability_ratio: float  # Ratio of variation


def determine_income(
    declared_income: Optional[float] = None,
    payslip_income: Optional[float] = None,
    statement_income: Optional[float] = None,
    statement_confidence: float = 0.0,
    employer_confirmed: Optional[float] = None,
    bureau_estimated: Optional[float] = None,
    employment_type_code: int = 1,
    months_employed: float = 12.0,
    variable_pay_history: Optional[list[float]] = None
) -> IncomeResult:
    """
    Establish gross_monthly_income from whichever evidence exists,
    applying a verification haircut per evidence tier.

    Evidence waterfall (highest to lowest):
    1. Employer-confirmed
    2. Payslip-derived
    3. Statement-derived
    4. Declared
    5. Bureau-estimated
    """
    # Evidence waterfall
    if employer_confirmed is not None:
        selected_income = employer_confirmed
        tier = "employer"
        source_code = 2
        haircut = 0.0  # No haircut for employer-confirmed
    elif payslip_income is not None:
        selected_income = payslip_income
        tier = "payslip"
        source_code = 3
        haircut = 0.05  # 5% haircut for payslip
    elif statement_income is not None:
        selected_income = statement_income
        tier = "statement"
        source_code = 4
        haircut = 0.10  # 10% haircut for statement
    elif declared_income is not None:
        selected_income = declared_income
        tier = "declared"
        source_code = 5
        haircut = 0.15  # 15% haircut for declared
    elif bureau_estimated is not None:
        selected_income = bureau_estimated
        tier = "bureau"
        source_code = 6
        haircut = 0.20  # 20% haircut for bureau
    else:
        selected_income = 0.0
        tier = "none"
        source_code = 9
        haircut = 1.0

    # Apply haircut
    gross_monthly = selected_income * (1.0 - haircut)

    # Variability calculation (if variable pay history provided)
    if variable_pay_history and len(variable_pay_history) > 1:
        mean = sum(variable_pay_history) / len(variable_pay_history)
        if mean > 0:
            variance = sum((x - mean) ** 2 for x in variable_pay_history) / len(variable_pay_history)
            std_dev = variance ** 0.5
            variability_ratio = std_dev / mean
        else:
            variability_ratio = 0.0
    else:
        variability_ratio = 0.0

    return IncomeResult(
        gross_monthly_income=gross_monthly,
        income_source_code=source_code,
        income_verification_tier=tier,
        income_haircut_applied=haircut,
        income_variability_ratio=variability_ratio
    )
