"""Segment, score, probability of default and grade."""
import pytest


@pytest.mark.parametrize("change, segment", [
    ({}, 4),                                                        # existing client with a clean credit holding
    ({"is_joint_application": True, "is_staff": True}, 12),         # joint beats staff
    ({"is_staff": True, "employment_type_code": 3}, 11),            # staff beats self-employed
    ({"is_non_resident": True, "employment_type_code": 4}, 10),
    ({"employment_type_code": 3}, 6), ({"employment_type_code": 4}, 7),
    ({"employment_type_code": 5}, 8), ({"employment_type_code": 6}, 9),
    ({"bureau_accounts": [{"account_type_code": 10, "balance": 1.0, "limit": 100.0, "instalment": 1.0,
                           "months_in_arrears": 3, "opened_date": "2022-03-01", "closed": False, "is_internal": False}]}, 5),
    ({"internal_credit_holding": False}, 3),                         # an existing transactional client
    ({"internal_credit_holding": False, "internal_tenure_months": 2.0}, 2),   # new to bank, thick bureau file
    ({"internal_credit_holding": False, "internal_tenure_months": 2.0, "bureau_history_months": 10.0}, 1),
])
def test_the_first_segment_that_fits_in_a_fixed_order_is_assigned(decide, change, segment):
    assert decide(**change)["segment_code"] == segment


def test_the_sample_scores_eight_characteristics_on_a_base_of_600(decide):
    out = decide()
    assert out["score"] == 674.0  # 600 - 12 bureau + 18 tenure + 28 arrears + 15 accounts - 6 utilisation + 20 + 6 + 5


def test_a_worse_bureau_score_lowers_the_score_and_worsens_the_grade(decide):
    low, high = decide(bureau_score=480.0), decide(bureau_score=780.0)
    assert (low["score"], decide()["score"], high["score"]) == (641.0, 674.0, 734.0)
    assert low["risk_grade"] > decide()["risk_grade"] > high["risk_grade"]


def test_the_channel_overlay_raises_the_probability_before_grading_and_is_recorded(decide):
    out = decide()
    assert out["probability_of_default"] == pytest.approx(out["probability_of_default_unadjusted"] * 1.18)
    assert out["pd_adjustments_applied"] == ["ADJ-0061"] and out["risk_grade_cell_id"] == "risk_grade@rg-2026.09#4|7"
    assert decide(channel_code=2)["pd_adjustments_applied"] == []


def test_the_overlay_can_move_a_client_a_grade_worse(decide):
    assert decide(channel_code=2)["risk_grade"] == 6 and decide()["risk_grade"] == 7
