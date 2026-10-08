"""Fraud: every rule is checked and the verdict has its own precedence."""
import pytest

PROCEED, REFER, DECLINE = 1, 2, 3


def test_the_sample_fires_two_minor_rules_and_proceeds(decide):
    out = decide()
    assert (out["fraud_verdict_code"], out["fraud_reason_codes"], out["fraud_rules_fired_count"]) == (PROCEED, [4211, 4221], 2)
    assert out["fraud_weighted_score"] == 0.1


def test_a_severity_four_watchlist_hit_declines_whatever_else_fired(decide):
    out = decide(consortium_watchlist_hit=True)
    assert out["fraud_verdict_code"] == DECLINE and out["outcome_code"] == 4
    assert out["decline_reason_codes"] == [2001]


def test_two_serious_identity_hits_refer(decide):
    out = decide(identity_number_reused=True, synthetic_identity_score=0.9)
    assert out["fraud_verdict_code"] == REFER and out["outcome_code"] == 3
    assert out["fraud_reason_codes"] == [4201, 4202, 4203, 4211, 4221]


@pytest.mark.parametrize("change, verdict", [
    ({"income_employer_mismatch": True, "shared_device_count": 2, "address_change_count_90d": 3}, REFER),
    ({"income_employer_mismatch": True, "shared_device_count": 2, "address_change_count_90d": 3,
      "application_burst_count": 3, "referral_chain_flag": True, "sector_employer_mismatch": True}, DECLINE),
])
def test_the_weighted_score_of_everything_that_fired_refers_then_declines(decide, change, verdict):
    assert decide(**change)["fraud_verdict_code"] == verdict


def test_a_small_request_from_a_long_standing_client_skips_the_consortium_families(decide):
    hit = {"consortium_watchlist_hit": True, "requested_amount": 6_000.0, "internal_tenure_months": 36.0}
    assert decide(**hit)["fraud_verdict_code"] == PROCEED  # F5 is not evaluated, which is not the same as not firing
    assert decide(**{**hit, "has_adverse_internal_history": True})["fraud_verdict_code"] == DECLINE
    assert decide(**{**hit, "requested_amount": 9_000.0})["fraud_verdict_code"] == DECLINE
