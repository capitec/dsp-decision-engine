"""Simple pipeline for testing."""
from __future__ import annotations
from decider import flow, param, missing_as
from datetime import date
from core import generate_decision_id
from core.income import determine_income
from core.deductions import calculate_deductions
from core.expense_norms import apply_expense_norms
from core.affordability import assess_affordability as assess_aff
from core.instalment import calculate_instalment
from core.fees import calculate_fees
from core.rate_card import lookup_flex_loan_rate


def income_step(
    payslip_income: float = param(15000.0, ge=0.0)
) -> float:
    """Calculate gross income."""
    result = determine_income(payslip_income=payslip_income)
    return result.gross_monthly_income


def deductions_step(
    gross_monthly_income: float,
    employment_type_code: int = param(1, ge=1, le=6)
) -> float:
    """Calculate net income."""
    result = calculate_deductions(gross_monthly_income, employment_type_code)
    return result.net_monthly_income


def expenses_step(
    gross_monthly_income: float,
    declared_living_expenses: float = param(1000.0, ge=0.0)
) -> float:
    """Calculate living expenses."""
    result = apply_expense_norms(gross_monthly_income, 0, declared_living_expenses)
    return result.living_expenses


def affordability_step(
    net_monthly_income: float,
    living_expenses: float,
    affordability_buffer: float = param(0.12, ge=0.0, le=0.3)
) -> float:
    """Calculate max affordable instalment."""
    result = assess_aff(
        net_monthly_income,
        living_expenses,
        0.0,
        affordability_buffer=affordability_buffer
    )
    return result.max_affordable_instalment


def pricing_step(
    amount: float = param(50000.0, ge=1000.0),
    term_months: int = param(36, ge=6, le=84)
) -> float:
    """Calculate instalment."""
    rate_result = lookup_flex_loan_rate(amount, term_months, 6, "1.0.0")
    fees_result = calculate_fees(amount, 10, term_months)
    instalment_result = calculate_instalment(
        amount, term_months, rate_result.nominal_annual_rate,
        initiation_fee=fees_result.initiation_fee,
        monthly_service_fee=fees_result.monthly_service_fee,
        credit_life_premium=fees_result.credit_life_premium
    )
    return instalment_result.instalment


def build():
    """Build the core library pipeline."""
    return flow(
        income_step,
        deductions_step,
        expenses_step,
        affordability_step,
        pricing_step,
        name="core-library"
    )
