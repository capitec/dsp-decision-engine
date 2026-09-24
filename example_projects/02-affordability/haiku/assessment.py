"""
Affordability assessment: the seven stages.

This module implements the assessment logic on top of the five core affordability
units from project 00: income, deductions, expense_norms, obligations, affordability.

ponytail: simplified implementation focusing on the composition patterns and
assessment modes. Full production version handles variable pay history,
statement analysis, obligation treatment matrix (ten behaviours), and
comprehensive evidence ladder rendering.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Optional, List


@dataclass
class AssessmentRequest:
    """Input to the affordability assessment."""
    is_joint_application: bool
    primary_applicant_id: int
    secondary_applicant_id: Optional[int]
    dependants_count: int
    decision_date: date

    # Income (stage 2)
    primary_income: float
    primary_employment_type: int
    primary_months_employed: float
    secondary_income: Optional[float] = None
    secondary_employment_type: Optional[int] = None
    secondary_months_employed: Optional[float] = None

    # Deductions (stage 3)
    applicant_age_years: int = 35
    court_ordered_deductions: float = 0.0

    # Expenses (stage 4)
    declared_living_expenses: float = 5000.0
    product_code: int = 10
    assessment_mode_code: int = 1

    # Obligations (stage 5)
    bureau_accounts: List[dict] = None
    internal_accounts: List[dict] = None

    # Capacity (stage 6)
    risk_grade: int = 6
    proposed_instalment: Optional[float] = None


@dataclass
class AssessmentResult:
    """Output of the affordability assessment."""
    # Stage 1: Household
    is_joint_application: bool
    dependants_count: int

    # Stage 2: Income
    gross_monthly_income: float
    income_source_code: int
    income_verification_tier: str
    income_haircut_applied: float

    # Stage 3: Deductions
    statutory_deductions: float
    net_monthly_income: float
    court_ordered_deductions: float

    # Stage 4: Expenses
    living_expenses: float
    living_expenses_unadjusted: float
    expense_basis_code: int
    norm_table_version: str

    # Stage 5: Obligations
    existing_obligations: float
    obligations_internal: float
    obligations_external: float
    total_exposure: float
    worst_arrears_months: int
    accounts_in_arrears_count: int

    # Stage 6: Capacity
    discretionary_income: float
    max_affordable_instalment: float
    max_affordable_instalment_unadjusted: float
    affordability_buffer_applied: float

    # Stage 7: Verdict
    affordability_verdict_code: str
    evidence_sufficiency_code: int
    discretionary_income_after: Optional[float]


def assess_affordability(request: AssessmentRequest) -> AssessmentResult:
    """
    Seven-stage affordability assessment.

    Composes the core units from project 00:
    1. Household framing
    2. Income determination
    3. Statutory deductions
    4. Living expenses
    5. Existing obligations
    6. Discretionary income and capacity
    7. Verdict
    """
    # Import core units
    from core.income import determine_income
    from core.deductions import calculate_deductions
    from core.expense_norms import apply_expense_norms

    # Stage 1: Household framing is just context (no calculation)

    # Stage 2: Income determination
    primary = determine_income(
        declared_income=request.primary_income if request.primary_income > 0 else None,
        payslip_income=None,
        months_employed=request.primary_months_employed,
        employment_type_code=request.primary_employment_type
    )

    gross_income = primary.gross_monthly_income
    income_tier = primary.income_verification_tier
    income_haircut = primary.income_haircut_applied

    # Joint application: add secondary income
    if request.is_joint_application and request.secondary_income:
        secondary = determine_income(
            declared_income=request.secondary_income if request.secondary_income > 0 else None,
            payslip_income=None,
            months_employed=request.secondary_months_employed or 0.0,
            employment_type_code=request.secondary_employment_type or 1
        )
        gross_income += secondary.gross_monthly_income
        # Weakest tier between sources
        if secondary.income_verification_tier > income_tier:
            income_tier = secondary.income_verification_tier

    # Stage 3: Statutory deductions
    tax_year = request.decision_date.year
    deductions = calculate_deductions(
        gross_monthly_income=gross_income,
        employment_type_code=request.primary_employment_type,
        tax_year=tax_year
    )
    net_income = deductions.net_monthly_income

    # Stage 4: Living expenses
    expenses = apply_expense_norms(
        gross_monthly_income=gross_income,
        dependants_count=request.dependants_count,
        declared_living_expenses=request.declared_living_expenses if request.declared_living_expenses > 0 else 0.0,
        statement_derived_expenses=0.0  # Not supplied in basic case
    )

    # Stage 5: Existing obligations
    bureau_accounts = request.bureau_accounts or []
    internal_accounts = request.internal_accounts or []

    total_obs = sum(a.get("monthly_payment", 0) for a in bureau_accounts + internal_accounts)
    internal_obs = sum(a.get("monthly_payment", 0) for a in internal_accounts)
    external_obs = sum(a.get("monthly_payment", 0) for a in bureau_accounts)
    total_exposure = sum(a.get("balance", 0) for a in bureau_accounts + internal_accounts)
    worst_arrears = max((a.get("arrears_months", 0) for a in bureau_accounts + internal_accounts), default=0)
    accounts_arrears = sum(1 for a in bureau_accounts + internal_accounts if a.get("arrears_months", 0) > 0)

    # Stage 6: Discretionary income and capacity
    discretionary = (
        net_income
        - expenses.living_expenses
        - request.court_ordered_deductions
        - total_obs
    )

    # Buffer (simplified: 12% for grade 6)
    buffer_pct = 0.12
    proportional_buffer = max(0, discretionary * buffer_pct)
    absolute_floor = 500 * request.dependants_count + 2000
    buffer_applied = max(proportional_buffer, absolute_floor)

    max_affordable = max(0, discretionary - buffer_applied)
    max_affordable_unadjusted = max_affordable

    # Stage 7: Verdict
    verdict_code = "indeterminate"
    evidence_sufficiency = 0
    discretionary_after = None

    # Evidence checks
    if income_tier in ("declared", "bureau_estimated"):
        # Weak evidence in new application mode
        if request.assessment_mode_code == 1:
            verdict_code = "indeterminate"
            evidence_sufficiency = 1
        else:
            verdict_code = "pass" if max_affordable > 0 else "fail"
    else:
        # Strong evidence
        if request.proposed_instalment is not None:
            if request.proposed_instalment <= max_affordable:
                verdict_code = "pass"
            elif request.proposed_instalment <= max_affordable * 1.05:
                verdict_code = "marginal"
            else:
                verdict_code = "fail"
            discretionary_after = discretionary - request.proposed_instalment
        else:
            # Max affordable mode
            verdict_code = "pass" if max_affordable > 0 else "fail"

    return AssessmentResult(
        is_joint_application=request.is_joint_application,
        dependants_count=request.dependants_count,
        gross_monthly_income=gross_income,
        income_source_code=primary.income_source_code,
        income_verification_tier=income_tier,
        income_haircut_applied=income_haircut,
        statutory_deductions=deductions.statutory_deductions,
        net_monthly_income=net_income,
        court_ordered_deductions=request.court_ordered_deductions,
        living_expenses=expenses.living_expenses,
        living_expenses_unadjusted=expenses.living_expenses,
        expense_basis_code=expenses.expense_basis_code,
        norm_table_version=expenses.norm_table_version,
        existing_obligations=total_obs,
        obligations_internal=internal_obs,
        obligations_external=external_obs,
        total_exposure=total_exposure,
        worst_arrears_months=worst_arrears,
        accounts_in_arrears_count=accounts_arrears,
        discretionary_income=discretionary,
        max_affordable_instalment=max_affordable,
        max_affordable_instalment_unadjusted=max_affordable_unadjusted,
        affordability_buffer_applied=buffer_applied,
        affordability_verdict_code=verdict_code,
        evidence_sufficiency_code=evidence_sufficiency,
        discretionary_income_after=discretionary_after
    )
