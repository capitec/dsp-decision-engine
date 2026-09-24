from retail_credit.disclosure import (
    OUTCOME_APPROVE, OUTCOME_APPROVE_WITH_CONDITIONS, OUTCOME_DECLINE, OUTCOME_REFER, OUTCOME_REJECTED,
    combine_decline_reasons, outcome_code,
)


def test_p01_rejection_is_never_a_decline():
    """10 §5.2: a structured rejection must never be recorded as a credit decline."""
    code = outcome_code(
        p01_valid=False, is_eligible=True, product_10_eligible=True, fraud_verdict_code=1, p09_declined=False,
        affordability_verdict_code=1, has_offer=True, validation_passed=True,
    )
    assert code == OUTCOME_REJECTED
    assert code != OUTCOME_DECLINE


def test_bureau_down_outside_envelope_refers_not_declines():
    """10 §5.25: bureau down, outside the reduced envelope -> refer, never decline, never a
    silent approval."""
    code = outcome_code(
        p01_valid=True, is_eligible=True, product_10_eligible=True, fraud_verdict_code=1, p09_declined=False,
        affordability_verdict_code=1, has_offer=True, validation_passed=True,
        degraded_mode_code=11, bureau_down_envelope_ok=False,
    )
    assert code == OUTCOME_REFER


def test_affordability_fail_without_an_offer_declines():
    code = outcome_code(
        p01_valid=True, is_eligible=True, product_10_eligible=True, fraud_verdict_code=1, p09_declined=False,
        affordability_verdict_code=3, has_offer=False, validation_passed=True,
    )
    assert code == OUTCOME_DECLINE


def test_marginal_verdict_is_approve_with_conditions():
    code = outcome_code(
        p01_valid=True, is_eligible=True, product_10_eligible=True, fraud_verdict_code=1, p09_declined=False,
        affordability_verdict_code=2, has_offer=True, validation_passed=True,
    )
    assert code == OUTCOME_APPROVE_WITH_CONDITIONS


def test_clean_approve():
    code = outcome_code(
        p01_valid=True, is_eligible=True, product_10_eligible=True, fraud_verdict_code=1, p09_declined=False,
        affordability_verdict_code=1, has_offer=True, validation_passed=True,
    )
    assert code == OUTCOME_APPROVE


def test_every_gate_contributes_to_the_reason_set_not_only_the_first():
    """10 §5.4's short-circuit tension: a client failing two independent gates gets both
    reasons, not only the first one found."""
    reasons = combine_decline_reasons(
        hard_eligibility_reasons=[1001, 1005], product_routing_reasons=[1201],
        fraud_verdict_code=1, fraud_reason_codes=[], p09_declined=False, p09_decline_reason=0,
        affordability_verdict_code=1, has_offer=True, validation_failed_assertions=[],
    )
    assert {1001, 1005, 1201} <= set(reasons)
