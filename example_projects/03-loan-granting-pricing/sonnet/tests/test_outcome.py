"""Outcome precedence (spec 03 §7.1): ineligible > fraud decline > consent/DQ/fraud
referral > policy decline > affordability > no offer > final validation > approve."""
from __future__ import annotations

from credit_core.affordability import FAIL as AFF_FAIL, INDETERMINATE as AFF_INDETERMINATE, PASS as AFF_PASS
from credit_core.vocab import ConsentVerdict

from loan_granting.outcome import OUTCOME_APPROVE, OUTCOME_DECLINE, OUTCOME_REFER, application_outcome_code

BASE = dict(
    is_eligible=True, fraud_handling_path="continue", consent_verdict=int(ConsentVerdict.PERMITTED),
    bureau_referral_required=False, waterfall_decline_reason_codes=[], affordability_verdict_code=AFF_PASS,
    has_any_offer=True, final_validation_passed=True, fraud_bypass_applied=False,
    credit_life_substitution_declared=False,
)


def test_ineligible_always_declines():
    assert application_outcome_code(**{**BASE, "is_eligible": False}) == OUTCOME_DECLINE


def test_fraud_decline_beats_everything_downstream():
    assert application_outcome_code(**{**BASE, "fraud_handling_path": "decline",
                                        "affordability_verdict_code": AFF_FAIL}) == OUTCOME_DECLINE


def test_missing_consent_refers():
    assert application_outcome_code(**{**BASE, "consent_verdict": int(ConsentVerdict.SUPPRESSED)}) == OUTCOME_REFER


def test_waterfall_decline_declines():
    assert application_outcome_code(**{**BASE, "waterfall_decline_reason_codes": [1183]}) == OUTCOME_DECLINE


def test_affordability_fail_declines():
    assert application_outcome_code(**{**BASE, "affordability_verdict_code": AFF_FAIL}) == OUTCOME_DECLINE


def test_affordability_indeterminate_refers():
    assert application_outcome_code(**{**BASE, "affordability_verdict_code": AFF_INDETERMINATE}) == OUTCOME_REFER


def test_no_surviving_offer_declines():
    assert application_outcome_code(**{**BASE, "has_any_offer": False}) == OUTCOME_DECLINE


def test_final_validation_failure_refers_rather_than_ships_an_offer():
    assert application_outcome_code(**{**BASE, "final_validation_passed": False}) == OUTCOME_REFER


def test_clean_application_approves():
    assert application_outcome_code(**BASE) == OUTCOME_APPROVE
