"""Admission, identity and the bureau-down envelope."""
import pytest


def test_a_malformed_request_is_rejected_with_every_reason_and_is_not_a_decline(decide):
    out = decide(requested_amount=0.0, term_months=0, product_code=20, entry_point_code=3)
    assert out["p01_valid"] is False
    assert out["p01_rejection_reasons"] == [9001, 9002, 9003, 9005]
    assert out["outcome_code"] == 0 and out["decline_reason_codes"] == []


@pytest.mark.parametrize("flags, path, confidence", [
    ({}, 1, 1.0),
    ({"has_authenticated_session": False, "identity_number_match": True}, 2, 0.97),
    ({"has_authenticated_session": False, "identity_verification_confirmed": True}, 3, 0.92),
])
def test_the_first_identity_path_that_holds_decides_the_confidence(decide, flags, path, confidence):
    out = decide(**flags)
    assert (out["identity_resolution_path"], out["identity_confidence"], out["identity_is_trustworthy"]) == (path, confidence, True)


def test_a_probabilistic_match_below_the_floor_refers_instead_of_approving(decide):
    out = decide(has_authenticated_session=False, probabilistic_match_confidence=0.80)
    assert (out["identity_resolution_path"], out["identity_is_trustworthy"]) == (4, False)
    assert out["outcome_code"] == 3


def test_a_working_bureau_is_not_degraded(decide):
    out = decide()
    assert (out["degraded_mode_code"], out["source_degradation_codes"], out["bureau_down_envelope_ok"]) == (0, [], True)


def test_bureau_down_inside_the_reduced_envelope_still_approves(decide):
    out = decide(bureau_available=False, requested_amount=20_000.0, term_months=36, internal_tenure_months=30.0)
    assert (out["degraded_mode_code"], out["source_degradation_codes"]) == (11, [11])
    assert out["bureau_down_envelope_ok"] is True and out["outcome_code"] == 1


@pytest.mark.parametrize("change", [
    {"requested_amount": 26_000.0}, {"term_months": 48}, {"internal_tenure_months": 20.0},
    {"is_staff": True},  # segment 11 is outside segments 3 to 5
])
def test_bureau_down_outside_the_reduced_envelope_refers_and_never_declines(decide, change):
    out = decide(bureau_available=False, **{"requested_amount": 20_000.0, "term_months": 36,
                                            "internal_tenure_months": 30.0, **change})
    assert out["bureau_down_envelope_ok"] is False
    assert out["outcome_code"] == 3
