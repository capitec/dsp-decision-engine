"""Hard eligibility: every gate is checked, and the reasons are ranked."""


def test_every_failed_gate_is_reported_not_only_the_first(decide):
    out = decide(is_sanctioned=True, on_exclusion_list=True, bureau_enquiry_consent=False)
    assert out["hard_eligibility_reasons"] == [1005, 1006, 1103]
    assert out["decline_reason_codes"] == [1006, 1103, 1005]  # most severe first
    assert (out["primary_reason_code"], out["outcome_code"], out["has_offer"]) == (1006, 4, False)


def test_a_client_too_old_at_maturity_is_declined(decide):
    out = decide(applicant_age_years=72.0, term_months=60)
    assert out["hard_eligibility_reasons"] == [1101] and out["outcome_code"] == 4


def test_a_duplicate_application_in_flight_is_declined(decide):
    assert decide(has_in_flight_application=True)["hard_eligibility_reasons"] == [1102]


def test_an_enquiry_without_bureau_consent_is_a_hard_stop(decide):
    out = decide(bureau_enquiry_consent=None)
    assert out["hard_eligibility_reasons"] == [1103] and out["outcome_code"] == 4
