"""The outcome, the referral queue and the reasons: the first stage that decides wins."""
from __future__ import annotations

from decikit import allow_fallback, flow, missing_as, step

from credit_core.affordability import INDETERMINATE
from credit_core.vocab import (
    OUTCOME_APPROVE, OUTCOME_APPROVE_WITH_CONDITIONS, OUTCOME_DECLINE, OUTCOME_REFER, ConsentVerdict,
)

from loan_granting import reasons
from loan_granting.bureau import NO_HIT
from loan_granting.fraud import BYPASSED, DECLINE, FORCED_REFER

QUEUE_NONE, QUEUE_CONSENT, QUEUE_BUREAU, QUEUE_FRAUD, QUEUE_NO_HIT, QUEUE_AFFORDABILITY, QUEUE_SOLVE, QUEUE_VALIDATION = range(8)


@allow_fallback
@step(outputs=("outcome_code", "referral_queue_code", "decline_reason_codes"))
def outcome(
    is_eligible: bool, eligibility_decline_reasons: list[int], fraud_handling_code: int, consent_verdict: int,
    bureau_referral_required: bool, bureau_availability_code: int, waterfall_decline_reason_codes: list[int],
    score_reason_codes: list[int], affordability_verdict_code: int, suppression_reason_codes: list[int],
    has_any_offer: bool, final_validation_passed: bool, internal_tenure_months: float = missing_as(0.0),
    credit_life_substitution_declared: bool = missing_as(False),
) -> tuple[int, int, list[int]]:
    """The outcome, the queue a referral goes to, and the reasons of the stage that decided (unranked).

    Ineligible and fraud declines come first, then the referrals that need no further assessment (consent, fraud
    refer, bureau data quality, a bureau no-hit with no internal history), then policy declines, then affordability,
    the solve, and last the offers and their validation. A bureau problem is never a decline.
    """
    if not is_eligible:
        return OUTCOME_DECLINE, QUEUE_NONE, [*eligibility_decline_reasons]
    if fraud_handling_code == DECLINE:
        return OUTCOME_DECLINE, QUEUE_NONE, [reasons.R_FRAUD_DECLINE]
    if consent_verdict != ConsentVerdict.PERMITTED:
        return OUTCOME_REFER, QUEUE_CONSENT, [reasons.R_CONSENT_MISSING]
    if fraud_handling_code == FORCED_REFER:
        return OUTCOME_REFER, QUEUE_FRAUD, []
    if bureau_referral_required:
        return OUTCOME_REFER, QUEUE_BUREAU, []
    if bureau_availability_code == NO_HIT and not internal_tenure_months > 0:
        return OUTCOME_REFER, QUEUE_NO_HIT, []
    if waterfall_decline_reason_codes:
        # a grade beyond appetite is explained by the characteristics that cost the client points
        explained = score_reason_codes if reasons.R_BEYOND_APPETITE in waterfall_decline_reason_codes else []
        return OUTCOME_DECLINE, QUEUE_NONE, [*waterfall_decline_reason_codes, *explained]
    if affordability_verdict_code == INDETERMINATE:
        return OUTCOME_REFER, QUEUE_AFFORDABILITY, [reasons.R_AFFORDABILITY_INDETERMINATE]
    if reasons.R_EVALUATION_CEILING_REACHED in suppression_reason_codes:
        return OUTCOME_REFER, QUEUE_SOLVE, [reasons.R_EVALUATION_CEILING_REACHED]
    if not has_any_offer:
        return OUTCOME_DECLINE, QUEUE_NONE, suppression_reason_codes or [reasons.R_AFFORDABILITY_FAIL]
    if not final_validation_passed:
        return OUTCOME_REFER, QUEUE_VALIDATION, [reasons.R_FINAL_VALIDATION_FAILED]
    if fraud_handling_code == BYPASSED or credit_life_substitution_declared:  # proof of cover, or the bypass, is a condition
        return OUTCOME_APPROVE_WITH_CONDITIONS, QUEUE_NONE, []
    return OUTCOME_APPROVE, QUEUE_NONE, []


def unit(reason_codes):
    """The outcome, then `reason_codes` (a `ReasonCodes` document) ranks `decline_reason_codes` in place."""
    return flow(outcome, reason_codes, name="decision").emit(
        "outcome_code", "referral_queue_code", "decline_reason_codes", "primary_reason_code", "reason_registry_version")
