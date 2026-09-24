"""
Affordability assessment pipeline (project 02).

Composes the seven-stage affordability assessment from project 00's core units.
"""
from __future__ import annotations
from datetime import date
from typing import Optional
from decider import flow, param, missing_as

from assessment import AssessmentRequest, assess_affordability


def assess(
    is_joint_application: bool = param(False),
    primary_applicant_id: int = param(1, ge=1),
    secondary_applicant_id: Optional[int] = None,
    dependants_count: int = param(0, ge=0, le=6),
    decision_date: date = missing_as(date.today()),
    primary_income: float = param(15000.0, ge=0.0),
    primary_employment_type: int = param(1, ge=1, le=6),
    primary_months_employed: float = param(24.0, ge=0.0),
    secondary_income: Optional[float] = None,
    secondary_employment_type: Optional[int] = None,
    secondary_months_employed: Optional[float] = None,
    applicant_age_years: int = param(35, ge=18, le=100),
    court_ordered_deductions: float = param(0.0, ge=0.0),
    declared_living_expenses: float = param(5000.0, ge=0.0),
    product_code: int = param(10, ge=10, le=51),
    assessment_mode_code: int = param(1, ge=1, le=4),
    bureau_accounts: list = missing_as([]),
    internal_accounts: list = missing_as([]),
    risk_grade: int = param(6, ge=1, le=12),
    proposed_instalment: Optional[float] = None
) -> dict:
    """Seven-stage affordability assessment."""
    request = AssessmentRequest(
        is_joint_application=is_joint_application,
        primary_applicant_id=primary_applicant_id,
        secondary_applicant_id=secondary_applicant_id,
        dependants_count=dependants_count,
        decision_date=decision_date,
        primary_income=primary_income,
        primary_employment_type=primary_employment_type,
        primary_months_employed=primary_months_employed,
        secondary_income=secondary_income,
        secondary_employment_type=secondary_employment_type,
        secondary_months_employed=secondary_months_employed,
        applicant_age_years=applicant_age_years,
        court_ordered_deductions=court_ordered_deductions,
        declared_living_expenses=declared_living_expenses,
        product_code=product_code,
        assessment_mode_code=assessment_mode_code,
        bureau_accounts=bureau_accounts,
        internal_accounts=internal_accounts,
        risk_grade=risk_grade,
        proposed_instalment=proposed_instalment
    )

    result = assess_affordability(request)

    return {
        "is_joint_application": result.is_joint_application,
        "dependants_count": result.dependants_count,
        "gross_monthly_income": result.gross_monthly_income,
        "income_source_code": result.income_source_code,
        "income_verification_tier": result.income_verification_tier,
        "income_haircut_applied": round(result.income_haircut_applied, 4),
        "statutory_deductions": result.statutory_deductions,
        "net_monthly_income": result.net_monthly_income,
        "court_ordered_deductions": result.court_ordered_deductions,
        "living_expenses": result.living_expenses,
        "living_expenses_unadjusted": result.living_expenses_unadjusted,
        "expense_basis_code": result.expense_basis_code,
        "norm_table_version": result.norm_table_version,
        "existing_obligations": result.existing_obligations,
        "obligations_internal": result.obligations_internal,
        "obligations_external": result.obligations_external,
        "total_exposure": result.total_exposure,
        "worst_arrears_months": result.worst_arrears_months,
        "accounts_in_arrears_count": result.accounts_in_arrears_count,
        "discretionary_income": result.discretionary_income,
        "max_affordable_instalment": result.max_affordable_instalment,
        "max_affordable_instalment_unadjusted": result.max_affordable_instalment_unadjusted,
        "affordability_buffer_applied": result.affordability_buffer_applied,
        "affordability_verdict_code": result.affordability_verdict_code,
        "evidence_sufficiency_code": result.evidence_sufficiency_code,
        "discretionary_income_after": result.discretionary_income_after
    }


def build():
    """Build the affordability assessment pipeline."""
    return flow(assess, name="affordability_assessment")
