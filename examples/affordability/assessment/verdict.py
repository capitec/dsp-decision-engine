"""The verdict: why the evidence was not enough, pass/marginal/fail against a proposal, and the reasons to report.

`indeterminate` is not a failure of affordability: the remedy is more evidence, not a decline.
"""
from __future__ import annotations

import math
from datetime import date

from decider import allow_fallback, flow, missing_as, param, step

from credit_core import affordability, reason_codes
from credit_core.affordability import FAIL, INDETERMINATE, PASS

# evidence_sufficiency_code
EVIDENCE_OK = 0
APPLICANT1_INCOME_UNESTABLISHED = 1
APPLICANT2_INCOME_UNESTABLISHED = 2
BOTH_APPLICANTS_INCOME_UNESTABLISHED = 3
BUREAU_STALE = 4
REFER_ACCOUNT = 5
INCOME_BELOW_MINIMUM_TIER = 6
NO_NET_INCOME = 7

# Reason codes reported with an indeterminate or failed verdict: 200 + the evidence code, and one for a failed proposal.
R_INSTALMENT_EXCEEDS_CAPACITY = 208

REASON_REGISTRY = reason_codes.ReasonCodeRegistry("affordability-reasons-2026.09", [
    reason_codes.ReasonCode(201, 10, "Applicant 1 income could not be established", True),
    reason_codes.ReasonCode(202, 10, "Applicant 2 income could not be established", True),
    reason_codes.ReasonCode(203, 5, "Neither applicant's income could be established", True),
    reason_codes.ReasonCode(204, 30, "Bureau view older than the permitted window", True),
    reason_codes.ReasonCode(205, 25, "An account type requires manual review", True),
    reason_codes.ReasonCode(206, 20, "Income evidence below the minimum for this assessment", True),
    reason_codes.ReasonCode(207, 15, "No net income established", True),
    reason_codes.ReasonCode(R_INSTALMENT_EXCEEDS_CAPACITY, 40, "Proposed instalment exceeds the affordable maximum", False),
])


@allow_fallback
def bureau_is_stale(
    decision_date: date, product_code: float = missing_as(-1.0), bureau_as_of_date: date | None = None,
    staleness_window_days: int = param(7, ge=1, le=60), staleness_window_days_mortgage: int = param(14, ge=1, le=60),
) -> bool:
    """A bureau view older than the window at the decision date (longer for mortgage-secured product 40), or none at all."""
    if bureau_as_of_date is None:
        return True
    window = staleness_window_days_mortgage if product_code == 40 else staleness_window_days
    return (decision_date - bureau_as_of_date).days > window


def evidence_sufficiency_code(
    income_verification_tier: int, minimum_income_tier: int, applicant_income_evidence_gap: int,
    bureau_is_stale: bool, has_refer_account: bool, net_monthly_income: float,
) -> int:
    """0 when the evidence supports a verdict, else the first cause, whose income is missing before anything else."""
    if applicant_income_evidence_gap:
        return applicant_income_evidence_gap  # 1, 2 or 3 are the codes for applicant 1, 2 and both
    if bureau_is_stale:
        return BUREAU_STALE
    if has_refer_account:
        return REFER_ACCOUNT
    if income_verification_tier == 0 or income_verification_tier > minimum_income_tier:
        return INCOME_BELOW_MINIMUM_TIER
    if net_monthly_income <= 0:
        return NO_NET_INCOME
    return EVIDENCE_OK


# credit_core's comparison of a proposal with the maximum, kept under its own name so the final verdict can override it.
proposal_verdict = affordability.affordability_verdict_code_step.named("proposal_verdict").relabel(
    writes={"affordability_verdict_code": "proposal_verdict_code"})


def affordability_verdict_code(
    evidence_sufficiency_code: int, proposal_verdict_code: int, proposed_instalment: float = missing_as(math.nan),
) -> int:
    """`INDETERMINATE` when the evidence is short, `PASS` when there is no proposal to fail (the caller wants the capacity), else the comparison."""
    if evidence_sufficiency_code != EVIDENCE_OK:
        return INDETERMINATE
    if math.isnan(proposed_instalment):
        return PASS
    return proposal_verdict_code


def discretionary_income_after(discretionary_income: float, proposed_instalment: float = missing_as(math.nan)) -> float:
    """What is left once the proposal is committed; NaN without one."""
    return round(discretionary_income - proposed_instalment, 2)


@allow_fallback
def reasons_fired(evidence_sufficiency_code: int, affordability_verdict_code: int) -> list[int]:
    reasons = [200 + evidence_sufficiency_code] if evidence_sufficiency_code else []
    return [*reasons, R_INSTALMENT_EXCEEDS_CAPACITY] if affordability_verdict_code == FAIL else reasons


# `decline_reason_codes` is written as fired, then ranked, so the two steps form a flow.
reasons_unit = flow(step(reasons_fired, output="decline_reason_codes"), REASON_REGISTRY.resolve_step(), name="reasons")
