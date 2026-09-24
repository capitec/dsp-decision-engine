"""`core.scorecard`, `core.calibration`, `core.risk_grade` (spec 00 §6.10-§6.12)."""
import pytest
from decider import Engine, dag

from credit_core.calibration import build_calibration_table, probability_of_default, score_for_probability
from credit_core.risk_grade import build_risk_grade_table
from credit_core.scorecard import VARIABLES, adverse_action_codes, build_scorecard

RECORD = {
    "bureau_score": 720.0, "months_employed": 50.0, "worst_arrears_months": 0.0,
    "accounts_in_arrears_count": 0, "revolving_utilisation": 0.2, "applicant_age_years": 35.0,
    "dependants_count": 1, "employment_type_code": 1,
}


def test_scorecard_emits_a_per_characteristic_contribution_for_every_variable():
    """00 §6.10 "Hard part": per-characteristic points are a required output, not a diagnostic."""
    exe = Engine().bind(build_scorecard(), mode="interpreted")
    out = exe.score(RECORD)
    for variable in VARIABLES:
        assert f"{variable}_score" in out
    assert out["score"] == sum(out[f"{v}_score"] for v in VARIABLES) + 600  # + the base offset


def test_a_null_characteristic_scores_its_declared_default_not_an_error():
    exe = Engine().bind(build_scorecard(), mode="interpreted")
    out = exe.score({**RECORD, "bureau_score": None})
    assert out["bureau_score_score"] == 0  # the "no_bureau" default bin


def test_adverse_action_codes_are_the_largest_negative_contributors():
    scores = {"bureau_score": -40.0, "months_employed": 20.0, "worst_arrears_months": -35.0,
              "accounts_in_arrears_count": 15.0, "revolving_utilisation": -5.0}
    codes = adverse_action_codes(scores, n=2)
    assert codes == [4101, 4103]  # bureau_score (-40) then worst_arrears_months (-35): most negative first


def test_calibration_is_invertible_to_the_cent():
    """00 §6.11 "Hard part": both directions must agree to tolerance."""
    for score in (500.0, 600.0, 700.0):
        pd = probability_of_default(score, anchor=600.0, scale=60.0)
        back = score_for_probability(pd, anchor=600.0, scale=60.0)
        assert back == pytest.approx(score, abs=1e-9)


def test_calibration_segments_use_different_settings_in_one_table():
    """Acceptance §10 item 4: the same capability, different settings, per record."""
    table = build_calibration_table()
    exe = Engine().bind(table, mode="interpreted")
    seg1 = exe.score({"segment_code": 1})
    seg3 = exe.score({"segment_code": 3})
    assert seg1["anchor"] != seg3["anchor"]
    assert seg1["calibration_cell_id"] != seg3["calibration_cell_id"]


def test_risk_grade_boundaries_are_returned_as_evidence():
    """00 §6.12 "Produces": the probability band edges that produced the grade."""
    table = build_risk_grade_table()
    exe = Engine().bind(table, mode="interpreted")
    out = exe.score({"product_code": 10, "segment_code": 1, "probability_of_default": 0.03})
    assert out["grade"] == 3
    assert out["risk_grade_boundary_lo"] < 0.03 < out["risk_grade_boundary_hi"]


def test_risk_grade_1_is_best_grade_12_is_worst():
    table = build_risk_grade_table()
    exe = Engine().bind(table, mode="interpreted")
    best = exe.score({"product_code": 10, "segment_code": 1, "probability_of_default": 0.001})
    worst = exe.score({"product_code": 10, "segment_code": 1, "probability_of_default": 0.9})
    assert best["grade"] == 1
    assert worst["grade"] == 12
