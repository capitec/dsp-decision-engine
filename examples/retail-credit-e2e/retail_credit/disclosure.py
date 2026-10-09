"""What the client is told: the ranked reasons, and the outcome."""
from __future__ import annotations

from decider import allow_fallback, flow, missing_as, step

from credit_core.affordability import FAIL, INDETERMINATE, MARGINAL, PASS
from credit_core.vocab import (
    OUTCOME_APPROVE, OUTCOME_APPROVE_WITH_CONDITIONS, OUTCOME_DECLINE, OUTCOME_REFER, FraudVerdictCode,
)
from retail_credit.vocab import BASIS_HYPOTHETICAL, OUTCOME_REJECTED

R_AFFORDABILITY_FAIL = 2140
R_NO_AFFORDABLE_AMOUNT = 2420
R_FRAUD_DECLINE = 2001
R_FRAUD_REFER = 2002
R_VALIDATION_FAILED = 2701


@allow_fallback
def decline_reason_codes(
    p01_valid: bool, hard_eligibility_reasons: list[int], product_routing_reasons: list[int], fraud_verdict_code: int,
    p09_declined: bool, p09_decline_reason: int, offer_affordability_verdict_code: int, decision_open: bool,
    has_offer: bool, validation_failed_assertions: list[int],
) -> list[int]:
    """Every reason any phase raised, before ranking, which the `reason_codes` document then ranks in place.
    A rejected request has none: a rejection is not a decline."""
    if not p01_valid:
        return []
    reasons = [*hard_eligibility_reasons, *product_routing_reasons, *validation_failed_assertions]
    if fraud_verdict_code == FraudVerdictCode.DECLINE:
        reasons.append(R_FRAUD_DECLINE)
    elif fraud_verdict_code == FraudVerdictCode.REFER:
        reasons.append(R_FRAUD_REFER)
    if p09_declined:
        reasons.append(p09_decline_reason)
    if not has_offer and offer_affordability_verdict_code == FAIL:
        reasons.append(R_AFFORDABILITY_FAIL)
    if decision_open and not has_offer and offer_affordability_verdict_code in (PASS, MARGINAL):
        reasons.append(R_NO_AFFORDABLE_AMOUNT)
    if validation_failed_assertions:
        reasons.append(R_VALIDATION_FAILED)
    return reasons


def outcome_code(
    p01_valid: bool, is_eligible: bool, product_10_eligible: bool, fraud_verdict_code: int, p09_declined: bool,
    identity_is_trustworthy: bool, offer_affordability_verdict_code: int, offer_basis_code: int, has_offer: bool,
    validation_passed: bool, bureau_down_envelope_ok: bool = missing_as(True),
) -> int:
    """Rejected, declined, referred, approved with conditions or approved, in that order of precedence.

    Refer means a person must look: the bureau is down and the request is outside the reduced envelope, the fraud
    check or identity match is unsure, the income is not established, or the solve or final validation produced no
    offer. An offer priced on settling the client's other debts, or on marginal affordability, carries conditions.
    """
    if not p01_valid:
        return OUTCOME_REJECTED
    if (not is_eligible or not product_10_eligible or p09_declined or fraud_verdict_code == FraudVerdictCode.DECLINE
            or (offer_affordability_verdict_code == FAIL and not has_offer)):
        return OUTCOME_DECLINE
    if (not bureau_down_envelope_ok or fraud_verdict_code != FraudVerdictCode.APPROVE or not identity_is_trustworthy
            or offer_affordability_verdict_code == INDETERMINATE or not has_offer or not validation_passed):
        return OUTCOME_REFER
    if offer_basis_code == BASIS_HYPOTHETICAL or offer_affordability_verdict_code == MARGINAL:
        return OUTCOME_APPROVE_WITH_CONDITIONS
    return OUTCOME_APPROVE


def reasons_unit(reason_codes):
    """`decline_reason_codes` fired, then ranked in place; writes the ranked list, `primary_reason_code` and the version."""
    return flow(step(decline_reason_codes), reason_codes, name="reasons").emit(
        "decline_reason_codes", "primary_reason_code", "reason_registry_version")


outcome_code_step = step(outcome_code)
