"""The 14 gates: every one is evaluated, every failure is reported, and a missing input is not a pass."""
import pytest

from loan_granting import reasons

GATES = ["product_channel", "min_age", "max_age_at_maturity", "capacity", "residency", "employment_type",
         "debt_review", "administration_order", "insolvency", "deceased_estate", "exclusion_lists",
         "duplicate_application", "in_flight_application", "insolvency_seasoning"]


def failed(out):
    return [gate for gate, verdict in out["eligibility_gate_verdicts"].items() if verdict == "fail"]


def test_a_clean_application_passes_every_gate(decide):
    out = decide()
    assert list(out["eligibility_gate_verdicts"]) == GATES
    assert set(out["eligibility_gate_verdicts"].values()) == {"pass"}
    assert (out["is_eligible"], out["eligibility_decline_reasons"]) == (True, [])


def test_every_gate_is_evaluated_after_the_first_failure_and_every_reason_is_given(decide):
    out = decide(applicant_age_years=15.0, residency_code=5, deceased_flag=True, debt_review_status_code=2)
    assert failed(out) == ["min_age", "residency", "debt_review", "deceased_estate"]
    assert len(out["eligibility_gate_verdicts"]) == 14
    assert out["is_eligible"] is False
    assert out["eligibility_decline_reasons"] == [1110, 1120, 1140, 1150]
    assert (out["outcome_code"], out["decline_reason_codes"], out["primary_reason_code"]) == (
        4, [1150, 1140, 1110, 1120], 1150)  # ranked by severity: deceased first


def test_a_term_that_was_not_asked_for_leaves_its_gate_not_evaluated_rather_than_passed(decide):
    out = decide(requested_term_months=None)
    assert out["eligibility_gate_verdicts"]["max_age_at_maturity"] == "not_evaluated"
    assert out["is_eligible"] is True


def test_the_age_at_maturity_counts_the_term(decide):
    assert failed(decide(applicant_age_years=72.0, requested_term_months=60)) == ["max_age_at_maturity"]
    assert failed(decide(applicant_age_years=72.0, requested_term_months=36)) == []


@pytest.mark.parametrize("change, gate, code", [
    ({"channel_code": 5, "requested_amount": 9_000.0}, "product_channel", reasons.R_PRODUCT_CHANNEL),
    ({"channel_code": 6}, "product_channel", reasons.R_PRODUCT_CHANNEL),
    ({"has_contractual_capacity": False}, "capacity", reasons.R_NO_CAPACITY),
    ({"employment_type_code": 6}, "employment_type", reasons.R_EMPLOYMENT_EXCLUDED),
    ({"administration_order_flag": True}, "administration_order", reasons.R_ADMIN_ORDER),
    ({"insolvency_status_code": 1}, "insolvency", reasons.R_INSOLVENCY),
    ({"estate_flag": True}, "deceased_estate", reasons.R_DECEASED_ESTATE),
    ({"is_duplicate_application": True}, "duplicate_application", reasons.R_DUPLICATE_APPLICATION),
    ({"in_flight_application_count": 1}, "in_flight_application", reasons.R_IN_FLIGHT_APPLICATION),
])
def test_each_gate_fails_on_its_own_condition_with_its_own_reason(decide, change, gate, code):
    out = decide(**change)
    assert failed(out) == [gate]
    assert code in out["decline_reason_codes"] and out["outcome_code"] == 4


def test_a_cleared_debt_review_passes_and_any_other_status_fails(decide):
    assert failed(decide(debt_review_status_code=4)) == []
    assert failed(decide(debt_review_status_code=1)) == ["debt_review"]


def test_a_rehabilitated_client_needs_two_years_of_seasoning(decide):
    assert failed(decide(insolvency_status_code=2, insolvency_rehabilitated_months=12.0)) == ["insolvency_seasoning"]
    assert failed(decide(insolvency_status_code=2, insolvency_rehabilitated_months=30.0)) == []


def test_every_exclusion_list_hit_gives_its_own_reason(decide):
    out = decide(exclusion_list_hits=[1, 3])
    assert failed(out) == ["exclusion_lists"]
    assert out["eligibility_decline_reasons"] == [reasons.R_EXCLUSION_SANCTIONS, reasons.R_EXCLUSION_STAFF]
    assert out["primary_reason_code"] == reasons.R_EXCLUSION_SANCTIONS
