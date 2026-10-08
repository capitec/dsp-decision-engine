"""The outcome and the reasons: what each stage said, and which of them decides."""
from __future__ import annotations

from decider import allow_fallback, flow, missing_as, step

from credit_core.affordability import INDETERMINATE
from credit_core.vocab import (
    OUTCOME_APPROVE, OUTCOME_APPROVE_WITH_CONDITIONS, OUTCOME_DECLINE, OUTCOME_REFER, ConsentVerdict,
)

from loan_granting import reasons
from loan_granting.bureau import NO_HIT
from loan_granting.fraud import BYPASSED, DECLINE, FORCED_REFER

QUEUE_NONE, QUEUE_CONSENT, QUEUE_BUREAU, QUEUE_FRAUD, QUEUE_NO_HIT, QUEUE_AFFORDABILITY, QUEUE_SOLVE, QUEUE_VALIDATION = range(8)


@allow_fallback
def fired_reason_codes(
    is_eligible: bool, eligibility_decline_reasons: list[int], fraud_handling_code: int, consent_verdict: int,
    waterfall_decline_reason_codes: list[int], affordability_verdict_code: int, has_any_offer: bool,
    suppression_reason_codes: list[int], score_reason_codes: list[int],
) -> list[int]:
    """Every reason any stage gave, unranked. `outcome_code` says which of them decided."""
    fired = [] if is_eligible else [*eligibility_decline_reasons]
    fired += [reasons.R_FRAUD_DECLINE] if fraud_handling_code == DECLINE else []
    fired += [] if consent_verdict == ConsentVerdict.PERMITTED else [reasons.R_CONSENT_MISSING]
    fired += waterfall_decline_reason_codes
    if reasons.R_BEYOND_APPETITE in waterfall_decline_reason_codes:  # the grade is the reason, and the score explains it
        fired += score_reason_codes
    fired += [reasons.R_EVALUATION_CEILING_REACHED] if reasons.R_EVALUATION_CEILING_REACHED in suppression_reason_codes else []
    if affordability_verdict_code == INDETERMINATE:
        fired.append(reasons.R_AFFORDABILITY_INDETERMINATE)
    elif not has_any_offer and not fired:
        fired += suppression_reason_codes or [reasons.R_AFFORDABILITY_FAIL]
    return fired


@step(outputs=("outcome_code", "referral_queue_code"))
def outcome(
    is_eligible: bool, fraud_handling_code: int, consent_verdict: int, bureau_referral_required: bool,
    bureau_availability_code: int, waterfall_decline_reason_codes: list[int], affordability_verdict_code: int,
    suppression_reason_codes: list[int], has_any_offer: bool, final_validation_passed: bool,
    internal_tenure_months: float = missing_as(0.0), credit_life_substitution_declared: bool = missing_as(False),
) -> tuple[int, int]:
    """The outcome and, for a referral, the queue. The first stage that decides wins, in this order.

    Ineligible and fraud declines come first, then the referrals that need no further assessment (consent, fraud
    refer, bureau data quality, a bureau no-hit with no internal history), then policy declines, then affordability,
    the solve, and last the offers and their validation. A bureau problem is never a decline.
    """
    if not is_eligible or fraud_handling_code == DECLINE:
        return OUTCOME_DECLINE, QUEUE_NONE
    if consent_verdict != ConsentVerdict.PERMITTED:
        return OUTCOME_REFER, QUEUE_CONSENT
    if fraud_handling_code == FORCED_REFER:
        return OUTCOME_REFER, QUEUE_FRAUD
    if bureau_referral_required:
        return OUTCOME_REFER, QUEUE_BUREAU
    if bureau_availability_code == NO_HIT and not internal_tenure_months > 0:
        return OUTCOME_REFER, QUEUE_NO_HIT
    if waterfall_decline_reason_codes:
        return OUTCOME_DECLINE, QUEUE_NONE
    if affordability_verdict_code == INDETERMINATE:
        return OUTCOME_REFER, QUEUE_AFFORDABILITY
    if reasons.R_EVALUATION_CEILING_REACHED in suppression_reason_codes:
        return OUTCOME_REFER, QUEUE_SOLVE
    if not has_any_offer:
        return OUTCOME_DECLINE, QUEUE_NONE
    if not final_validation_passed:
        return OUTCOME_REFER, QUEUE_VALIDATION
    if fraud_handling_code == BYPASSED or credit_life_substitution_declared:  # proof of cover, or the bypass, is a condition
        return OUTCOME_APPROVE_WITH_CONDITIONS, QUEUE_NONE
    return OUTCOME_APPROVE, QUEUE_NONE


reasons_unit = flow(step(fired_reason_codes, output="decline_reason_codes"), reasons.REGISTRY.resolve_step(), name="reasons")
