"""Final outcome resolution: combines every stage's verdict into one
`outcome_code`, `referral_queue_code` and ranked reason set (spec 03 §7.1).

Every stage above runs unconditionally for every application (see NOTES.md
"What I left out" -- no `branch`-based short-circuit after eligibility/
fraud/DQ, unlike the spec's own §5.1 short-circuit language); this module
is where their signals are given precedence and collapsed into one answer,
so an ineligible or fraud-declined application's (otherwise meaningless)
downstream scoring/waterfall/solve numbers never leak into the outcome.
"""
from __future__ import annotations

from decider import missing_as, step

from credit_core.affordability import FAIL as AFF_FAIL, INDETERMINATE as AFF_INDETERMINATE, MARGINAL as AFF_MARGINAL
from credit_core.vocab import ConsentVerdict

from loan_granting import reasons
from loan_granting.fraud import (
    DECLINE as FRAUD_DECLINE, FORCED_REFER, QUEUE_AFFORDABILITY_INDETERMINATE, QUEUE_CONSENT,
    QUEUE_DQ, QUEUE_EVALUATION_CEILING, QUEUE_FINAL_VALIDATION, QUEUE_FRAUD_REFER,
)

OUTCOME_APPROVE = 1
OUTCOME_APPROVE_WITH_CONDITIONS = 2
OUTCOME_REFER = 3
OUTCOME_DECLINE = 4


def combined_decline_reason_codes(
    is_eligible: bool, eligibility_decline_reasons: list[int] = missing_as([]),
    fraud_handling_path: str = missing_as(""), waterfall_decline_reason_codes: list[int] = missing_as([]),
    affordability_verdict_code: int = missing_as(0), has_any_offer: bool = missing_as(False),
    suppression_reason_codes: list[int] = missing_as([]), score_reason_codes: list[int] = missing_as([]),
) -> list[int]:
    fired: list[int] = []
    if is_eligible is not None and not is_eligible:
        fired.extend([c for c in (eligibility_decline_reasons if eligibility_decline_reasons is not None else [])])
    if fraud_handling_path == FRAUD_DECLINE:
        fired.append(reasons.R_FRAUD_DECLINE)
    waterfall_declined = waterfall_decline_reason_codes is not None and len(waterfall_decline_reason_codes) > 0
    if waterfall_declined:
        fired.extend(list(waterfall_decline_reason_codes))
    if affordability_verdict_code == AFF_FAIL:
        fired.append(reasons.R_AFFORDABILITY_FAIL)
        fired.extend(list(score_reason_codes) if score_reason_codes is not None else [])
    blocked = (is_eligible is False) or (fraud_handling_path == FRAUD_DECLINE) or waterfall_declined or \
        (affordability_verdict_code == AFF_FAIL)
    if not blocked and not has_any_offer:
        codes = list(suppression_reason_codes) if suppression_reason_codes is not None else []
        fired.extend(codes if codes else [reasons.R_AFFORDABILITY_FAIL])
    return sorted(set(fired))


def application_outcome_code(
    is_eligible: bool, fraud_handling_path: str, consent_verdict: int, bureau_referral_required: bool,
    waterfall_decline_reason_codes: list[int] = missing_as([]), affordability_verdict_code: int = missing_as(0),
    has_any_offer: bool = missing_as(False), final_validation_passed: bool = missing_as(True),
    fraud_bypass_applied: bool = missing_as(False), credit_life_substitution_declared: bool = missing_as(False),
) -> int:
    waterfall_declined = waterfall_decline_reason_codes is not None and len(waterfall_decline_reason_codes) > 0
    if not is_eligible:
        return OUTCOME_DECLINE
    if fraud_handling_path == FRAUD_DECLINE:
        return OUTCOME_DECLINE
    if consent_verdict not in (int(ConsentVerdict.PERMITTED), int(ConsentVerdict.REGULATED_NOTICE)):
        return OUTCOME_REFER
    if fraud_handling_path == FORCED_REFER:
        return OUTCOME_REFER
    if bureau_referral_required:
        return OUTCOME_REFER
    if waterfall_declined:
        return OUTCOME_DECLINE
    if affordability_verdict_code == AFF_FAIL:
        return OUTCOME_DECLINE
    if affordability_verdict_code == AFF_INDETERMINATE:
        return OUTCOME_REFER
    if not has_any_offer:
        return OUTCOME_DECLINE
    if not final_validation_passed:
        return OUTCOME_REFER
    if affordability_verdict_code == AFF_MARGINAL or fraud_bypass_applied or credit_life_substitution_declared:
        return OUTCOME_APPROVE_WITH_CONDITIONS
    return OUTCOME_APPROVE


def referral_queue_code(
    outcome_code: int, consent_verdict: int, fraud_handling_path: str, bureau_referral_required: bool,
    affordability_verdict_code: int = missing_as(0), final_validation_passed: bool = missing_as(True),
) -> int:
    if outcome_code != OUTCOME_REFER:
        return 0
    if consent_verdict not in (int(ConsentVerdict.PERMITTED), int(ConsentVerdict.REGULATED_NOTICE)):
        return QUEUE_CONSENT
    if bureau_referral_required:
        return QUEUE_DQ
    if fraud_handling_path == FORCED_REFER:
        return QUEUE_FRAUD_REFER
    if affordability_verdict_code == AFF_INDETERMINATE:
        return QUEUE_AFFORDABILITY_INDETERMINATE
    if not final_validation_passed:
        return QUEUE_FINAL_VALIDATION
    return QUEUE_EVALUATION_CEILING


combined_decline_reason_codes_step = step(combined_decline_reason_codes, output="decline_reason_codes")
application_outcome_code_step = step(application_outcome_code, output="outcome_code")
referral_queue_code_step = step(referral_queue_code, output="referral_queue_code")
