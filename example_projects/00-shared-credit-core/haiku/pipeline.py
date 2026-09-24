from __future__ import annotations
"""Shared credit core library pipeline.

Composes the 22 published capabilities into a decision flow.
Implements 09 §5.15: every decision records what it needs to be replayed.
"""
from decider import flow, param, missing_as
from datetime import date
from core import generate_decision_id
from core.dates import resolve_effective_dated
from core.reason_codes import rank_reasons, get_primary_reason
from core.adjustments import apply_adjustments
from core.rounding import round_instalment
from core.income import determine_income
from core.deductions import calculate_deductions
from core.expense_norms import apply_expense_norms
from core.obligations import calculate_obligations, Account
from core.affordability import assess_affordability
from core.instalment import calculate_instalment, calculate_max_affordable_amount
from core.fees import calculate_fees
from core.rate_card import lookup_flex_loan_rate
from core.scorecard import evaluate_scorecard
from core.calibration import calibrate_score
from core.risk_grade import assign_risk_grade
from core.bureau import normalize_bureau_response
from core.stubs import (
    check_eligibility, get_appetite, calculate_exposure,
    check_consent, calculate_credit_life, classify_adverse_events
)


def generate_decision_context(
    client_id: int,
    application_id: int,
    decision_date: date = missing_as(date.today())
) -> dict:
    """Generate decision context (09 §5.15 item 1: stable decision identifier)."""
    return {
        "decision_id": generate_decision_id(),
        "client_id": client_id,
        "application_id": application_id,
        "decision_date": decision_date
    }


def assess_income(
    declared_income: float = missing_as(0.0),
    payslip_income: float = missing_as(0.0),
    months_employed: float = param(12.0, ge=0.0),
    employment_type_code: int = param(1, ge=1, le=6)
) -> dict:
    """Core.income capability (00 §6.1)."""
    result = determine_income(
        declared_income=declared_income if declared_income > 0 else None,
        payslip_income=payslip_income if payslip_income > 0 else None,
        months_employed=months_employed,
        employment_type_code=employment_type_code
    )
    return {
        "gross_monthly_income": result.gross_monthly_income,
        "income_source_code": result.income_source_code,
        "income_verification_tier": result.income_verification_tier,
        "income_haircut_applied": result.income_haircut_applied
    }


def assess_deductions(
    gross_monthly_income: float,
    employment_type_code: int = param(1, ge=1, le=6)
) -> dict:
    """Core.deductions capability (00 §6.2)."""
    result = calculate_deductions(gross_monthly_income, employment_type_code)
    return {
        "statutory_deductions": result.statutory_deductions,
        "net_monthly_income": result.net_monthly_income
    }


def assess_expenses(
    gross_monthly_income: float,
    dependants_count: int = param(0, ge=0),
    declared_living_expenses: float = param(1000.0, ge=0.0)
) -> dict:
    """Core.expense_norms capability (00 §6.3)."""
    result = apply_expense_norms(
        gross_monthly_income,
        dependants_count,
        declared_living_expenses
    )
    return {
        "living_expenses": result.living_expenses,
        "expense_basis_code": result.expense_basis_code,
        "norm_table_version": result.norm_table_version
    }


def assess_obligations(
    monthly_payment_sum: float = missing_as(0.0),
    bureau_count: int = missing_as(0)
) -> dict:
    """Core.obligations capability (00 §6.4)."""
    return {
        "existing_obligations": monthly_payment_sum,
        "per_account_obligations": [],
        "account_count": bureau_count
    }


def assess_affordability(
    net_monthly_income: float,
    living_expenses: float,
    existing_obligations: float,
    proposed_instalment: float = missing_as(0.0),
    affordability_buffer: float = param(0.12, ge=0.0, le=0.3),
    appetite_haircut: float = param(0.85, ge=0.5, le=1.0)
) -> dict:
    """Core.affordability capability (00 §6.5)."""
    from core.affordability import assess_affordability as assess
    result = assess(
        net_monthly_income,
        living_expenses,
        existing_obligations,
        proposed_instalment if proposed_instalment > 0 else None,
        affordability_buffer=affordability_buffer,
        appetite_haircut=appetite_haircut
    )
    return {
        "affordability_verdict_code": result.verdict_code,
        "discretionary_income": result.discretionary_income,
        "max_affordable_instalment": result.max_affordable_instalment,
        "pass_fail": result.pass_fail
    }


def score_client(
    gross_monthly_income: float = missing_as(0.0),
    dependants: int = missing_as(0)
) -> dict:
    """Core.scorecard capability (00 §6.10)."""
    # Simplified: score based on income and dependants
    characteristics = {
        "income_level": gross_monthly_income,
        "employment_tenure": 24.0,
        "adverse_events": 0.0
    }

    bins = {
        "income_level": [(0, 5000), (5000, 10000), (10000, 20000), (20000, float('inf'))],
        "employment_tenure": [(0, 6), (6, 24), (24, 60), (60, float('inf'))],
        "adverse_events": [(0, 1), (1, 3), (3, float('inf'))]
    }

    result = evaluate_scorecard(1, characteristics, bins)
    return {
        "scorecard_id": result.scorecard_id,
        "score": result.score,
        "score_contributions": [
            {"characteristic": c.characteristic_id, "points": c.points}
            for c in result.contributions
        ]
    }


def calibrate_and_grade(
    score: float
) -> dict:
    """Core.calibration and core.risk_grade (00 §6.11, §6.12)."""
    segment_code = "retail"
    cal_params = {}
    cal_result = calibrate_score(score, segment_code, cal_params)

    grade_boundaries = {
        "retail": [0.01, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 0.85, 1.0],
        "sme": [0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 0.85, 0.95, 1.0]
    }

    grade_result = assign_risk_grade(
        cal_result.probability_of_default,
        segment_code,
        grade_boundaries
    )

    return {
        "probability_of_default": cal_result.probability_of_default,
        "risk_grade": grade_result.risk_grade,
        "grade_bucket": grade_result.grade_bucket
    }


def price_product(
    amount: float,
    term_months: int = param(36, ge=6, le=84),
    risk_grade: int = missing_as(6)
) -> dict:
    """Core.rate_card, core.fees, core.instalment (00 §6.8, §6.7, §6.6)."""
    product_code = 10  # Flex Loan
    rate_card_version = "1.0.0"
    # Get rate
    rate_result = lookup_flex_loan_rate(amount, term_months, risk_grade, rate_card_version)

    # Get fees
    fees_result = calculate_fees(amount, product_code, term_months)

    # Calculate instalment
    instalment_result = calculate_instalment(
        amount,
        term_months,
        rate_result.nominal_annual_rate,
        initiation_fee=fees_result.initiation_fee,
        monthly_service_fee=fees_result.monthly_service_fee,
        credit_life_premium=fees_result.credit_life_premium
    )

    return {
        "nominal_annual_rate": rate_result.nominal_annual_rate,
        "rate_cell_id": rate_result.rate_cell_id,
        "initiation_fee": fees_result.initiation_fee,
        "monthly_service_fee": fees_result.monthly_service_fee,
        "credit_life_premium": fees_result.credit_life_premium,
        "instalment": round_instalment(instalment_result.instalment),
        "effective_annual_rate": instalment_result.effective_annual_rate,
        "total_cost_of_credit": instalment_result.total_cost_of_credit
    }


def check_gates(
    client_id: int,
    risk_grade: int = missing_as(6)
) -> dict:
    """Stub capabilities: eligibility, appetite, exposure, consent (00 §6.13-19)."""
    product_code = 10
    eligibility = check_eligibility(client_id, product_code)
    appetite = get_appetite(risk_grade, product_code)
    exposure = calculate_exposure(client_id)
    consent = check_consent(client_id)

    return {
        "eligible": eligibility.outcome,
        "appetite_limit": appetite.appetite_limit,
        "total_exposure": exposure.total_exposure,
        "consent_sms": consent.sms_consent
    }


def build():
    """Build the shared credit core library pipeline."""
    return flow(
        generate_decision_context,
        assess_income,
        assess_deductions,
        assess_expenses,
        assess_obligations,
        assess_affordability,
        score_client,
        calibrate_and_grade,
        price_product,
        check_gates,
        name="core-library"
    )
