"""P18 -- Disclosure and decision record emission (spec 10 §5.19): outcome and reasons.

O-19 (P18's reason ranking runs after every phase that can raise a reason,
**including P17**) and O-20 (disclosure only after P17 passes) are both
structural facts about `pipeline.py`'s wiring (this phase runs last), not
mechanisms this module itself enforces -- there is nothing left to check
once the dag has already ordered it that way. What this module owns: the
registry (reused: `credit_core.reason_codes.ReasonCodeRegistry`, this
project's own 412-code-taxonomy slice) and `outcome_code` (10 §7.1's
entry point 1 shape: approve / approve-with-conditions / refer / decline).
"""
from __future__ import annotations

from decider import missing_as, step

from credit_core import reason_codes
from retail_credit.affordability_phase import FAIL, INDETERMINATE, MARGINAL, PASS
from retail_credit.cap_waterfall import R_APPETITE_DECLINE, R_CHANNEL_CLOSED, R_ENQUIRY_VELOCITY_DECLINE
from retail_credit.eligibility_consent import R_BUREAU_CONSENT_MISSING, R_IN_FLIGHT_DUPLICATE, R_MAX_AGE_AT_MATURITY
from retail_credit.routing import (
    R_AMOUNT_OUT_OF_RANGE, R_EVIDENCE_TIER_TOO_LOW, R_GRADE_BELOW_FLOOR, R_TERM_OUT_OF_RANGE,
)
from retail_credit.solve import EVALUATION_CEILING_PER_TERM  # noqa: F401 -- documents where 2420 comes from
from retail_credit.validation import (
    A_AFFORDABLE_ON_REDERIVATION, A_AMOUNT_ON_GRID, A_AMOUNT_WITHIN_CAP, A_RATE_WITHIN_CEILING,
    A_REDERIVATION_IN_RANGE, A_TERM_WITHIN_CAP,
)
from retail_credit.vocab import DEGRADED_BUREAU_DOWN

R_AFFORDABILITY_FAIL = 2140      # matches the spec's own worked example code (10 §5.11)
R_SOLVE_EXHAUSTED = 2420         # matches the spec's own worked example code (10 §5.14)
R_FRAUD_DECLINE = 2001
R_FRAUD_REFER = 2002
R_VALIDATION_FAILED = 2701

REASON_REGISTRY = reason_codes.ReasonCodeRegistry("p10-reasons-2026.09", [
    reason_codes.ReasonCode(1001, 10, "Applicant below minimum age", True),
    reason_codes.ReasonCode(1002, 15, "Applicant not resident", True),
    reason_codes.ReasonCode(1003, 5, "No contractual capacity", True),
    reason_codes.ReasonCode(1004, 40, "Product not available to applicant", False),
    reason_codes.ReasonCode(1005, 8, "On an internal exclusion list", True),
    reason_codes.ReasonCode(1006, 1, "Sanctioned party", True),
    reason_codes.ReasonCode(1007, 2, "Deceased on record", True),
    reason_codes.ReasonCode(1008, 12, "Under debt review", True),
    reason_codes.ReasonCode(1009, 12, "Under administration", True),
    reason_codes.ReasonCode(R_MAX_AGE_AT_MATURITY, 11, "Age at maturity exceeds the product maximum", True),
    reason_codes.ReasonCode(R_IN_FLIGHT_DUPLICATE, 30, "A duplicate application is already in flight", False),
    reason_codes.ReasonCode(R_BUREAU_CONSENT_MISSING, 3, "Bureau enquiry consent not given", True),
    reason_codes.ReasonCode(R_AMOUNT_OUT_OF_RANGE, 41, "Requested amount outside the product's range", False),
    reason_codes.ReasonCode(R_TERM_OUT_OF_RANGE, 42, "Requested term outside the product's range", False),
    reason_codes.ReasonCode(R_EVIDENCE_TIER_TOO_LOW, 25, "Income evidence tier below the product minimum", False),
    reason_codes.ReasonCode(R_GRADE_BELOW_FLOOR, 20, "Risk grade below the product's floor", False),
    reason_codes.ReasonCode(R_CHANNEL_CLOSED, 44, "Channel not open for this product", False),
    reason_codes.ReasonCode(R_APPETITE_DECLINE, 21, "Risk grade outside the Bank's appetite", False),
    reason_codes.ReasonCode(R_ENQUIRY_VELOCITY_DECLINE, 22, "Bureau enquiry velocity too high", False),
    reason_codes.ReasonCode(R_AFFORDABILITY_FAIL, 20, "Proposed instalment exceeds affordable capacity", True),
    reason_codes.ReasonCode(R_SOLVE_EXHAUSTED, 26, "No affordable amount found within the evaluation budget", False),
    reason_codes.ReasonCode(R_FRAUD_DECLINE, 1, "Application fraud verdict: decline", True),
    reason_codes.ReasonCode(R_FRAUD_REFER, 6, "Application fraud verdict: refer", True),
    reason_codes.ReasonCode(R_VALIDATION_FAILED, 27, "Offer failed final validation", False),
    reason_codes.ReasonCode(A_AMOUNT_ON_GRID, 50, "Offer amount not on the R250 grid", False),
    reason_codes.ReasonCode(A_AMOUNT_WITHIN_CAP, 23, "Offer amount exceeds the cap in force", False),
    reason_codes.ReasonCode(A_TERM_WITHIN_CAP, 24, "Offer term exceeds the cap in force", False),
    reason_codes.ReasonCode(A_RATE_WITHIN_CEILING, 4, "Rate exceeds the statutory ceiling", True),
    reason_codes.ReasonCode(A_REDERIVATION_IN_RANGE, 45, "Offer amount could not be re-derived", False),
    reason_codes.ReasonCode(A_AFFORDABLE_ON_REDERIVATION, 20, "Re-derived instalment exceeds affordable capacity",
                             True),
])

OUTCOME_REJECTED = 0    # structural (P01), never a credit decision (10 §5.2) -- distinct from 4
OUTCOME_APPROVE = 1
OUTCOME_APPROVE_WITH_CONDITIONS = 2
OUTCOME_REFER = 3
OUTCOME_DECLINE = 4


def combine_decline_reasons(
    hard_eligibility_reasons: list[int],
    product_routing_reasons: list[int],
    fraud_verdict_code: int,
    fraud_reason_codes: list[int],
    p09_declined: bool,
    p09_decline_reason: int,
    affordability_verdict_code: int,
    has_offer: bool,
    validation_failed_assertions: list[int],
) -> list[int]:
    reasons: list[int] = list(hard_eligibility_reasons or [])
    reasons += list(product_routing_reasons or [])
    if fraud_verdict_code == 3:
        reasons.append(R_FRAUD_DECLINE)
    elif fraud_verdict_code == 2:
        reasons.append(R_FRAUD_REFER)
    if p09_declined and p09_decline_reason:
        reasons.append(p09_decline_reason)
    if affordability_verdict_code == FAIL and not has_offer:
        reasons.append(R_AFFORDABILITY_FAIL)
    if not has_offer and affordability_verdict_code in (PASS, MARGINAL):
        reasons.append(R_SOLVE_EXHAUSTED)
    if validation_failed_assertions:
        reasons.append(R_VALIDATION_FAILED)
        reasons += list(validation_failed_assertions)
    return sorted(set(reasons))


combine_decline_reasons_step = step(combine_decline_reasons, output="decline_reason_codes")
rank_reasons_step = REASON_REGISTRY.resolve_step()


def outcome_code(
    p01_valid: bool, is_eligible: bool, product_10_eligible: bool, fraud_verdict_code: int, p09_declined: bool,
    affordability_verdict_code: int, has_offer: bool, validation_passed: bool,
    degraded_mode_code: int = missing_as(0), bureau_down_envelope_ok: bool = missing_as(True),
) -> int:
    if not p01_valid:
        return OUTCOME_REJECTED
    if not is_eligible or not product_10_eligible or fraud_verdict_code == 3 or p09_declined:
        return OUTCOME_DECLINE
    # Bureau down (10 §5.25, this project's one declared degraded mode): "approve only
    # within a reduced envelope... everyone else refers" -- never a decline, never a
    # silent approval on an assumption.
    if degraded_mode_code == DEGRADED_BUREAU_DOWN and not bureau_down_envelope_ok:
        return OUTCOME_REFER
    if affordability_verdict_code == INDETERMINATE:
        return OUTCOME_REFER
    if affordability_verdict_code == FAIL and not has_offer:
        return OUTCOME_DECLINE
    if not has_offer or not validation_passed:
        return OUTCOME_REFER
    if fraud_verdict_code == 2 or affordability_verdict_code == MARGINAL:
        return OUTCOME_APPROVE_WITH_CONDITIONS
    return OUTCOME_APPROVE


outcome_code_step = step(outcome_code, output="outcome_code")
