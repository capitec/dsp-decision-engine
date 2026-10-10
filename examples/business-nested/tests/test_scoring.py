"""Entity scoring: scorecard, overlay, calibration and the grade ladder."""
import math
from datetime import date

import pytest
from decikit import Engine

from business_nested import scoring

score = Engine().bind(scoring.entity_scoring_unit(), mode="interpreted")
PERSON = {"is_natural_person": True, "bureau_score": 705.0, "months_on_record": 140.0, "worst_delinquency_months": 0.0,
          "effective_ownership_pct": 55.0, "decision_date": date(2026, 9, 24)}


def scored(params=None, **changes):
    return score.score(PERSON | changes, params)


def pd(points):
    return 1.0 / (1.0 + math.exp((points - 600.0) / 60.0))


def test_a_natural_person_scores_the_scorecard_less_the_new_to_bank_shift():
    out = scored()
    assert (out["score_before_overlay"], out["entity_score"], out["entity_score_unadjusted"]) == (775.0, 763.0, 775.0)
    assert out["score_adjustments_applied"] == ["ADJ-05-ENT-001"]


def test_the_probability_of_default_is_calibrated_from_the_score_with_and_without_the_overlay():
    out = scored()
    assert out["entity_pd"] == pytest.approx(pd(763.0)) and out["entity_pd_unadjusted"] == pytest.approx(pd(775.0))
    assert out["entity_pd"] > out["entity_pd_unadjusted"]


def test_the_grade_comes_from_the_ladder_of_the_entitys_segment_and_the_unadjusted_grade_is_kept():
    out = scored()
    assert (out["entity_grade"], out["entity_grade_unadjusted"]) == (5, 4)
    juristic = scored(is_natural_person=False)  # ladder 5, whose cut points are lower
    assert (juristic["entity_grade"], juristic["entity_grade_unadjusted"]) == (5, 5)


def test_a_juristic_entity_is_outside_the_scope_of_the_natural_person_overlay():
    out = scored(is_natural_person=False)
    assert out["entity_score"] == out["entity_score_unadjusted"] == 775.0
    assert out["score_adjustments_applied"] == []


def test_the_overlay_does_not_apply_before_it_takes_effect():
    out = scored(decision_date=date(2025, 12, 31))
    assert out["entity_score"] == 775.0 and out["score_adjustments_applied"] == []


@pytest.mark.parametrize("bureau, points", [(499.0, -35), (500.0, -35), (501.0, -5), (650.0, -5), (651.0, 20), (750.0, 20), (751.0, 40)])
def test_the_bureau_score_bands_award_their_points_up_to_and_including_their_upper_edge(bureau, points):
    assert scored(bureau_score=bureau)["score_before_overlay"] == 700 + points + 20 + 25 + 10


def test_a_missing_characteristic_scores_its_default_bin():
    out = scored(bureau_score=None, months_on_record=None)
    assert out["score_before_overlay"] == 700 - 20 - 10 + 25 + 10


def test_the_bureau_cut_offs_are_params():
    params = {"scoring": {"entity_scorecard": {"bureau_score": {"bureau_mid": 700.0}}}}
    assert scored(params, bureau_score=690.0)["score_before_overlay"] == scored(bureau_score=640.0)["score_before_overlay"]


grade = Engine().bind(scoring.grade_lookup("pd", "grade"), mode="interpreted")


@pytest.mark.parametrize("segment, probability, expected", [
    (4, 0.0, 1), (4, 0.0099, 1), (4, 0.0101, 2), (4, 0.0599, 4), (4, 0.0601, 5), (4, 0.5999, 11), (4, 0.99, 12),
    (5, 0.0079, 1), (5, 0.0081, 2), (5, 0.99, 12), (9, 0.001, 12)])
def test_the_grade_ladder_by_segment(segment, probability, expected):
    assert grade.score({"pd": probability, "segment_code": float(segment)})["grade"] == expected


def test_the_ladder_cell_that_gave_the_grade_is_recorded():
    out = grade.score({"pd": 0.05, "segment_code": 4.0})
    assert out["grade_cell_id"] == "grade_ladder@bus-rg-2026.09#4|4"
