"""Affordability for a limit increase, through the shared assessment (`assessment.pipeline`), not a copy of it.

The programme has no payslip and no client in the room, so it differs from origination only in what it feeds the
assessment and what it layers on top:

- its own evidence tiers (verified salary, irregular deposits, declared) are translated into the income fields the
  assessment already knows how to haircut;
- the assessment runs in limit-increase mode against a notional instalment: the minimum payment on the new limit;
- a wider buffer (an overlay) and its own staleness rule decide automatic, conditional or not offered.
"""
from __future__ import annotations

import math
from datetime import date

from decider import allow_fallback, flow, helper, missing_as, param, step

from assessment import modes, pipeline as assessment, verdict
from limit_mgmt.caps import round_down_500
from limit_mgmt.overlays import buffer_adjustment_step
from limit_mgmt.vocab import PRODUCT_EVERYDAY_CARD, Cap, EvidenceTier, IncreasePath

_NO_DECLARED_INCOME = 9_999
_NAN = math.nan


@helper(signatures=[((float,), float)])
def _minimum_payment_rate(product_code: float) -> float:
    return 0.035 if product_code == PRODUCT_EVERYDAY_CARD else 0.05


def assessment_mode_code(mode: int = param(modes.LIMIT_INCREASE, ge=1, le=4)) -> int:
    """The programme assesses in limit-increase mode; it is a param so the evidence record shows it."""
    return mode


def is_joint_application(joint: bool = param(False)) -> bool:
    """A programme account has one client; the assessment reads a second applicant only when this is true."""
    return joint


def notional_instalment(policy_proposed_limit: float, product_code: float) -> float:
    """The commitment if the facility were drawn to the new limit and serviced at the minimum payment (3.5% of the
    limit, at least R100, on the card; 5%, at least R50, on the facility)."""
    floor = 100.0 if product_code == PRODUCT_EVERYDAY_CARD else 50.0
    return round(max(policy_proposed_limit * _minimum_payment_rate(product_code), floor if policy_proposed_limit > 0 else 0.0), 2)


@allow_fallback
def evidence_tier_code(
    verified_salary_present_3m: bool = missing_as(False), salary_deposit_variability_pct: float = missing_as(1.0),
    declared_income_age_days: int = missing_as(_NO_DECLARED_INCOME),
    steady_deposit_variability: float = param(0.15, ge=0.0, le=1.0),
    declared_income_fresh_days: int = param(365, ge=0),
) -> int:
    """A salary credited in each of the last three months is tier A when steady and tier B when not, then declared
    income by age. No deposits and no declared income is tier E."""
    if verified_salary_present_3m:
        return EvidenceTier.A_VERIFIED_SALARY if salary_deposit_variability_pct <= steady_deposit_variability else EvidenceTier.B_IRREGULAR_DEPOSITS
    if declared_income_age_days <= declared_income_fresh_days:
        return EvidenceTier.C_DECLARED_REFRESHED
    return EvidenceTier.D_DECLARED_STALE if declared_income_age_days < _NO_DECLARED_INCOME else EvidenceTier.E_NO_EVIDENCE


@allow_fallback
def income_staleness_days(
    decision_date: date, evidence_tier_code: int, salary_deposit_as_of_date: date | None = None,
    declared_income_age_days: int = missing_as(_NO_DECLARED_INCOME),
) -> int:
    """Age of the income basis on the decision date."""
    if evidence_tier_code in (EvidenceTier.A_VERIFIED_SALARY, EvidenceTier.B_IRREGULAR_DEPOSITS):
        return _NO_DECLARED_INCOME if salary_deposit_as_of_date is None else (decision_date - salary_deposit_as_of_date).days
    return declared_income_age_days


def programme_income_evidence(
    evidence_tier_code: int, verified_salary_amount: float = missing_as(0.0), declared_income_raw: float = missing_as(0.0),
    declared_income_age_days: int = missing_as(_NO_DECLARED_INCOME),
    cpi_annual_rate: float = param(0.052, ge=0.0, le=0.30),
    steady_confidence: float = param(0.95, ge=0.0, le=1.0), irregular_confidence: float = param(0.40, ge=0.0, le=1.0),
) -> tuple[float, float, float]:
    """The evidence as the assessment's income fields: `(statement_income, statement_confidence, declared_income)`.

    Deposits become a bank statement, steady ones with high confidence. A refreshed declared income is indexed
    forward at CPI for up to 12 months. What is not evidenced is NaN, which the assessment reads as absent.
    """
    if evidence_tier_code == EvidenceTier.A_VERIFIED_SALARY:
        return verified_salary_amount, steady_confidence, _NAN
    if evidence_tier_code == EvidenceTier.B_IRREGULAR_DEPOSITS:
        return verified_salary_amount, irregular_confidence, _NAN
    if evidence_tier_code == EvidenceTier.C_DECLARED_REFRESHED:
        months = min(12, max(0, declared_income_age_days // 30))
        return _NAN, _NAN, round(declared_income_raw * (1.0 + cpi_annual_rate) ** (months / 12.0), 2)
    if evidence_tier_code == EvidenceTier.D_DECLARED_STALE:
        return _NAN, _NAN, declared_income_raw  # usable, but only for a conditional increase
    return _NAN, _NAN, _NAN


@allow_fallback
def increase_path_code(
    evidence_tier_code: int, income_staleness_days: int, bureau_as_of_date: date | None, decision_date: date,
    evidence_sufficiency_code: int, current_limit: float, policy_proposed_limit: float, cap_affordability: float,
    expenses_refreshed_or_normed: bool = missing_as(True), obligations_growth_pct: float = missing_as(0.0),
    income_staleness_max_days: int = param(45, ge=1), bureau_staleness_max_days: int = param(35, ge=1),
    obligations_growth_threshold: float = param(0.25, ge=0.0),
) -> int:
    """Automatic only on tier A or B income within 45 days, a bureau view within 35, refreshed expenses and obligations
    that have not grown over 25%. Anything else the assessment can still carry is conditional on the client confirming
    income. Not offered when the evidence cannot support an assessment, or when the client cannot carry any of the
    increase the caps allowed."""
    if evidence_sufficiency_code != verdict.EVIDENCE_OK or (policy_proposed_limit > current_limit and cap_affordability <= current_limit):
        return IncreasePath.FAIL
    bureau_age = _NO_DECLARED_INCOME if bureau_as_of_date is None else (decision_date - bureau_as_of_date).days
    fresh = (evidence_tier_code in (EvidenceTier.A_VERIFIED_SALARY, EvidenceTier.B_IRREGULAR_DEPOSITS)
             and income_staleness_days <= income_staleness_max_days and bureau_age <= bureau_staleness_max_days
             and expenses_refreshed_or_normed and obligations_growth_pct <= obligations_growth_threshold)
    return IncreasePath.AUTOMATIC if fresh else IncreasePath.CONDITIONAL


def affordability_cap(
    policy_proposed_limit: float, policy_binding_cap_code: int, max_affordable_instalment: float,
    current_limit: float, product_code: float,
) -> tuple[float, int]:
    """C7: the limit whose notional instalment the client can carry, rounded down to R500. It only holds an increase
    back, never pushes the limit under the current one. Returns the limit and the code of the cap that bound."""
    implied = round_down_500(max(current_limit, max_affordable_instalment / _minimum_payment_rate(product_code)))
    if implied < policy_proposed_limit:
        return implied, Cap.C7_AFFORDABILITY
    return policy_proposed_limit, policy_binding_cap_code


def unit():
    # The assessment's inputs are named for origination; this renames the ones that mean something else here.
    evidence = assessment.evidence_unit().relabel(reads={
        "risk_grade": "behaviour_grade", "applicant1_statement_income": "programme_statement_income",
        "applicant1_statement_confidence": "programme_statement_confidence",
        "applicant1_declared_income": "programme_declared_income"})
    capacity = assessment.capacity_unit().relabel(
        reads={"proposed_instalment": "notional_instalment"},
        writes={"decline_reason_codes": "affordability_decline_reason_codes",
                "primary_reason_code": "affordability_primary_reason_code",
                "reason_registry_version": "affordability_reason_registry_version"})
    return flow(
        step(assessment_mode_code), step(is_joint_application), step(notional_instalment), step(evidence_tier_code), step(income_staleness_days),
        step(programme_income_evidence, outputs=("programme_statement_income", "programme_statement_confidence",
                                                 "programme_declared_income")),
        evidence, buffer_adjustment_step, capacity,
        step(affordability_cap, outputs=("cap_affordability", "binding_cap_code")), step(increase_path_code),
        name="affordability_bridge")
