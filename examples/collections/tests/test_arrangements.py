import math

import pytest

from collections_treatment import arrangements


def test_an_instalment_that_leaves_room_is_sustainable():
    sustainable, residual, detail = arrangements.arrangement_sustainability(
        net_monthly_income=10_000.0, discretionary_income=4_000.0, proposed_instalment=500.0)
    assert (sustainable, residual, detail) == (True, 3_500.0, "")


def test_an_instalment_that_leaves_under_the_minimum_residual_fails():
    sustainable, residual, detail = arrangements.arrangement_sustainability(
        net_monthly_income=10_000.0, discretionary_income=4_000.0, proposed_instalment=3_900.0)
    assert sustainable is False and residual == 100.0 and "below_minimum_residual" in detail.split(",")


def test_an_instalment_above_85_percent_of_discretionary_income_fails():
    _, _, detail = arrangements.arrangement_sustainability(
        net_monthly_income=20_000.0, discretionary_income=4_000.0, proposed_instalment=3_600.0)
    assert detail == "instalment_exceeds_discretionary_income_pct"


def test_a_residual_under_five_percent_of_net_income_fails_even_above_the_rand_minimum():
    _, residual, detail = arrangements.arrangement_sustainability(
        net_monthly_income=30_000.0, discretionary_income=1_350.0, proposed_instalment=1_000.0)
    assert residual == 350.0 and detail == "below_minimum_residual_income_pct"


def test_no_discretionary_income_is_never_sustainable():
    sustainable, _, detail = arrangements.arrangement_sustainability(
        net_monthly_income=5_000.0, discretionary_income=-200.0, proposed_instalment=100.0)
    assert sustainable is False and "instalment_exceeds_discretionary_income_pct" in detail


def test_no_proposal_is_reported_as_such_not_as_sustainable():
    sustainable, residual, detail = arrangements.arrangement_sustainability(
        net_monthly_income=5_000.0, discretionary_income=2_000.0)
    assert sustainable is False and math.isnan(residual) and detail == "no_proposal"


def test_the_sample_arrangement_is_assessed_in_arrangement_mode_and_is_sustainable(decide):
    out = decide()
    assert out["assessment_mode_code"] == 3
    assert (out["affordability_verdict_code"], out["arrangement_sustainable"]) == (1, True)
    assert out["arrangement_residual"] == round(out["discretionary_income"] - 900.0, 2)


def test_a_large_instalment_fails_sustainability_on_the_same_assessment(decide):
    out = decide(proposed_instalment=2_400.0)
    assert out["arrangement_sustainable"] is False
    assert "instalment_exceeds_discretionary_income_pct" in out["arrangement_failure_detail"]


def test_arrangement_mode_accepts_a_declared_income_that_a_new_application_would_not(decide):
    out = decide(applicant1_payslip_income=None, applicant1_declared_income=12_500.0)
    assert out["affordability_verdict_code"] != 4  # not indeterminate
    assert decide(applicant1_payslip_income=None, applicant1_declared_income=12_500.0,
                  proposed_instalment=None)["arrangement_failure_detail"] == "no_proposal"


def test_the_assessment_mode_is_a_param_not_a_request_field(decide, params):
    mode = params["collections_treatment"]["arrangement_assessment"]["arrangement_assessment_mode"]
    assert mode == {"mode": 3}
    assert decide(assessment_mode_code=1)["assessment_mode_code"] == 3  # a request cannot change the mode
