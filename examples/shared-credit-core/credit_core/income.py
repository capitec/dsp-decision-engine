"""Income: which evidence tier wins, and the haircut that tier carries.

The tiers are tried best first, so the order is code; the haircuts are policy and live in the params document.
Absent evidence is NaN, which is not the same as a verified income of zero.
"""
from __future__ import annotations

import math
import statistics

from decider import allow_fallback, flow, missing_as, param

EMPLOYER_CONFIRMED = 1
PAYSLIP = 2
BANK_STATEMENT = 3
DECLARED = 4
BUREAU_ESTIMATED = 5
NONE_ESTABLISHED = 0

_NA = float("nan")


def income_source_code(
    declared_income: float = missing_as(_NA),
    payslip_income: float = missing_as(_NA),
    statement_income: float = missing_as(_NA),
    employer_confirmed_income: float = missing_as(_NA),
    bureau_estimated_income: float = missing_as(_NA),
) -> int:
    """The best tier with evidence."""
    if not math.isnan(employer_confirmed_income):
        return EMPLOYER_CONFIRMED
    if not math.isnan(payslip_income):
        return PAYSLIP
    if not math.isnan(statement_income):
        return BANK_STATEMENT
    if not math.isnan(declared_income):
        return DECLARED
    if not math.isnan(bureau_estimated_income):
        return BUREAU_ESTIMATED
    return NONE_ESTABLISHED


def income_verification_tier(income_source_code: int) -> int:
    """The same tier under the name evidence records key on."""
    return income_source_code


def income_haircut_applied(
    income_source_code: int,
    statement_confidence: float = missing_as(_NA),
    statement_confidence_threshold: float = param(0.7, ge=0.0, le=1.0),
    haircut_payslip: float = param(0.05, ge=0.0, le=1.0),
    haircut_statement_high_confidence: float = param(0.10, ge=0.0, le=1.0),
    haircut_statement_low_confidence: float = param(0.25, ge=0.0, le=1.0),
    haircut_declared: float = param(0.20, ge=0.0, le=1.0),
    haircut_bureau_estimated: float = param(0.30, ge=0.0, le=1.0),
) -> float:
    """The share of the evidenced income that is discounted; a statement's depends on its confidence."""
    if income_source_code == PAYSLIP:
        return haircut_payslip
    if income_source_code == BANK_STATEMENT:
        confident = statement_confidence >= statement_confidence_threshold  # False for NaN
        return haircut_statement_high_confidence if confident else haircut_statement_low_confidence
    if income_source_code == DECLARED:
        return haircut_declared
    if income_source_code == BUREAU_ESTIMATED:
        return haircut_bureau_estimated
    return 0.0


def gross_monthly_income(
    income_source_code: int,
    income_haircut_applied: float,
    declared_income: float = missing_as(_NA),
    payslip_income: float = missing_as(_NA),
    statement_income: float = missing_as(_NA),
    employer_confirmed_income: float = missing_as(_NA),
    bureau_estimated_income: float = missing_as(_NA),
) -> float:
    """The income of the winning tier after its haircut, to the cent (0 when nothing is evidenced)."""
    raw = 0.0
    if income_source_code == EMPLOYER_CONFIRMED:
        raw = employer_confirmed_income
    elif income_source_code == PAYSLIP:
        raw = payslip_income
    elif income_source_code == BANK_STATEMENT:
        raw = statement_income
    elif income_source_code == DECLARED:
        raw = declared_income
    elif income_source_code == BUREAU_ESTIMATED:
        raw = bureau_estimated_income
    return round(raw * (1.0 - income_haircut_applied), 2)


@allow_fallback
def income_variability_ratio(variable_pay_history: list[float] = missing_as([])) -> float:
    """Coefficient of variation of the monthly variable pay (0 with fewer than two values)."""
    values = list(variable_pay_history)
    if len(values) < 2 or statistics.mean(values) == 0:
        return 0.0
    return round(statistics.pstdev(values) / statistics.mean(values), 4)


income_unit = flow(income_source_code, income_verification_tier, income_haircut_applied, gross_monthly_income,
                   income_variability_ratio, name="income")
