import math

import pytest

from credit_core.calibration import build_calibration_table, probability_of_default, score_for_probability
from credit_core.risk_grade import build_risk_grade_table
from credit_core.scorecard import VARIABLES, adverse_action_codes, build_scorecard

APPLICANT = {
    "bureau_score": 720.0, "months_employed": 50.0, "worst_arrears_months": 0.0, "accounts_in_arrears_count": 0,
    "revolving_utilisation": 0.2, "applicant_age_years": 35.0, "dependants_count": 1, "employment_type_code": 1,
}


def test_the_scorecard_returns_points_per_characteristic_and_their_sum_on_a_base(score):
    out = score(build_scorecard(), APPLICANT)
    points = {v: out[f"{v}_score"] for v in VARIABLES}
    assert points == {"bureau_score": 20, "months_employed": 20, "worst_arrears_months": 25,
                      "accounts_in_arrears_count": 15, "revolving_utilisation": 10, "applicant_age_years": 5,
                      "dependants_count": 5, "employment_type_code": 10}
    assert out["score"] == 600 + 110


@pytest.mark.parametrize("bureau_score,points", [
    (499.9, -40), (500.0, -40), (500.1, -10), (650.0, -10), (650.1, 20), (750.0, 20), (750.1, 45)])
def test_a_scorecard_bin_includes_its_upper_edge(score, bureau_score, points):
    assert score(build_scorecard(), {**APPLICANT, "bureau_score": bureau_score})["bureau_score_score"] == points


def test_a_missing_characteristic_scores_its_default_bin(score):
    out = score(build_scorecard(), {**APPLICANT, "months_employed": None})
    assert out["months_employed_score"] == -5


def test_bureau_cut_offs_are_params(score):
    params = {"retail_unsecured": {"bureau_score": {"bureau_mid": 700.0}}}
    assert score(build_scorecard(), {**APPLICANT, "bureau_score": 680.0}, params)["bureau_score_score"] == -10


def test_adverse_action_codes_are_the_most_negative_contributors_first():
    points = {"bureau_score": -40.0, "months_employed": 20.0, "worst_arrears_months": -35.0, "revolving_utilisation": -5.0}
    assert adverse_action_codes(points, n=2) == [4101, 4103]
    assert adverse_action_codes({"bureau_score": 5.0}) == []


def test_probability_of_default_is_one_half_at_the_anchor_and_falls_with_the_score():
    assert probability_of_default(600.0, anchor=600.0, scale=60.0) == 0.5
    assert probability_of_default(660.0, 600.0, 60.0) == pytest.approx(1 / (1 + math.e) )
    assert probability_of_default(700.0, 600.0, 60.0) < probability_of_default(600.0, 600.0, 60.0)


def test_the_inverse_recovers_the_score():
    for s in (500.0, 600.0, 700.0):
        assert score_for_probability(probability_of_default(s, 600.0, 60.0), 600.0, 60.0) == pytest.approx(s)


def test_the_inverse_clamps_probabilities_of_zero_and_one():
    assert math.isfinite(score_for_probability(0.0, 600.0, 60.0)) and math.isfinite(score_for_probability(1.0, 600.0, 60.0))


def test_each_segment_is_calibrated_on_its_own_anchor(score):
    one = score(build_calibration_table(), {"segment_code": 1})
    three = score(build_calibration_table(), {"segment_code": 3})
    assert (one["anchor"], one["scale"], three["anchor"], three["scale"]) == (600.0, 60.0, 560.0, 70.0)
    assert one["calibration_cell_id"] == "calibration@cal-2026.09#1"
    assert three["calibration_cell_id"] == "calibration@cal-2026.09#3"


def grade(score, pd, product=10, segment=1):
    return score(build_risk_grade_table(), {"product_code": product, "segment_code": segment, "probability_of_default": pd})


@pytest.mark.parametrize("pd,expected", [(0.001, 1), (0.0099, 1), (0.01, 2), (0.03, 3), (0.035, 4), (0.49, 11), (0.5, 12), (0.9, 12)])
def test_grades_run_from_best_to_worst_with_the_cut_point_in_the_worse_grade(score, pd, expected):
    assert grade(score, pd)["risk_grade"] == expected


def test_the_grade_records_the_band_that_produced_it(score):
    out = grade(score, 0.03)
    assert (out["risk_grade_boundary_lo"], out["risk_grade_boundary_hi"]) == (0.02, 0.035)
    assert out["risk_grade_cell_id"] == "risk_grade@rg-2026.09#10|1|3"


def test_the_new_to_bank_segment_has_wider_bands(score):
    assert grade(score, 0.025, segment=3)["risk_grade"] == 2 and grade(score, 0.025, segment=1)["risk_grade"] == 3


def test_a_product_and_segment_without_boundaries_gets_the_worst_grade(score):
    assert grade(score, 0.001, product=30)["risk_grade"] == 12
