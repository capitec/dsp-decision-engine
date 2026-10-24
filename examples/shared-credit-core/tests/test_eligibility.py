import pytest

from credit_core import eligibility as e

ADULT = {"applicant_age_years": 30.0}


def test_an_applicant_who_passes_every_gate_has_no_reasons(score):
    out = score(e.decline_reason_codes_step, ADULT)
    assert out["decline_reason_codes"] == []
    assert e.is_eligible([]) and not e.is_eligible([e.R_SANCTIONED])


@pytest.mark.parametrize("failure,code", [
    ({"applicant_age_years": 17.0}, e.R_UNDERAGE),
    ({"is_resident": False}, e.R_NOT_RESIDENT),
    ({"has_contractual_capacity": False}, e.R_NO_CAPACITY),
    ({"product_available": False}, e.R_PRODUCT_UNAVAILABLE),
    ({"on_exclusion_list": True}, e.R_EXCLUSION_LIST),
    ({"is_sanctioned": True}, e.R_SANCTIONED),
    ({"is_deceased": True}, e.R_DECEASED),
    ({"in_debt_review": True}, e.R_DEBT_REVIEW),
    ({"in_administration": True}, e.R_ADMINISTRATION),
])
def test_each_gate_raises_its_own_reason(score, failure, code):
    assert score(e.decline_reason_codes_step, {**ADULT, **failure})["decline_reason_codes"] == [code]


def test_every_failed_gate_is_reported_not_only_the_first(score):
    out = score(e.decline_reason_codes_step, {"applicant_age_years": 15.0, "is_sanctioned": True, "in_debt_review": True})
    assert out["decline_reason_codes"] == [e.R_UNDERAGE, e.R_SANCTIONED, e.R_DEBT_REVIEW]


def test_the_minimum_age_is_a_param_and_the_boundary_age_passes(score):
    assert score(e.decline_reason_codes_step, {"applicant_age_years": 18.0})["decline_reason_codes"] == []
    params = {"decline_reason_codes": {"minimum_age": 21.0}}
    assert score(e.decline_reason_codes_step, {"applicant_age_years": 20.0}, params)["decline_reason_codes"] == [e.R_UNDERAGE]
