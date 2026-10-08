import pytest
from decider import DeciderError
from decider.exceptions import ParamsError

from conftest import switch_off_every_overlay
from limit_mgmt import pipeline
from limit_mgmt.matrix import build_matrix_table
from limit_mgmt.vocab import ChangeType, NoticeClass


def test_the_sample_account_is_offered_an_automatic_increase_after_the_dial(decide):
    out = decide()
    assert (out["matrix_cell_id"], out["behaviour_grade"], out["utilisation_band"], out["mob_band"]) == (
        "limit_matrix@limit-matrix-2026.09#20|2|5|5", 2, 5, 5)
    assert (out["policy_proposed_limit"], out["final_proposed_limit"]) == (20_000.0, 20_000.0)
    assert (out["change_type_code"], out["notice_class_code"]) == (ChangeType.INCREASE_AUTOMATIC, NoticeClass.PRESCRIBED_NOTICE)
    assert out["decline_reason_codes"] == [] and out["exclusion_codes"] == []


def test_the_unadjusted_matrix_value_stays_beside_the_adjusted_one(decide):
    out = decide()
    assert (out["matrix_multiplier_unadjusted"], out["matrix_multiplier"]) == (1.4331, 1.3465)  # the 80% dial on the excess
    assert out["matrix_adjustments_applied"] == ["ADJ-07-DIAL-2026-03"]
    assert (out["uncapped_target_limit_unadjusted"], out["uncapped_target_limit"]) == (21_496.5, 20_197.5)
    assert (out["policy_proposed_limit_unadjusted"], out["policy_proposed_limit"]) == (21_000.0, 20_000.0)


def test_switching_every_overlay_off_gives_the_unadjusted_answer_through_the_same_pipeline(decide, params):
    out = decide(params_document=switch_off_every_overlay(params))
    assert out["matrix_multiplier"] == out["matrix_multiplier_unadjusted"] == 1.4331
    assert out["final_proposed_limit"] == 21_000.0 and out["affordability_buffer_applied"] == 0.1227
    assert out["matrix_adjustments_applied"] == [] and out["buffer_adjustments_applied"] == []


def test_a_cycle_cap_holds_the_increase_while_the_unadjusted_proposal_keeps_the_cell(decide, params):
    capped = {**params, "limit_mgmt": {**params["limit_mgmt"], "matrix": {
        **params["limit_mgmt"]["matrix"], "apply_cycle_cap": {"cycle_cap_enabled": True, "cycle_cap_amount": 3_000.0}}}}
    out = decide(params_document=capped)
    assert (out["cycle_cap_bound"], out["matrix_max_increase"], out["final_proposed_limit"]) == (True, 3_000.0, 18_000.0)
    assert (out["binding_cap_code"], out["policy_proposed_limit_unadjusted"]) == (6, 21_000.0)


def test_an_account_without_consent_is_still_scored_but_keeps_its_limit(decide):
    out = decide(consent_automatic_increase=False)
    assert out["exclusion_codes"] == [12] and out["is_excluded"] is True
    assert (out["final_proposed_limit"], out["change_type_code"], out["decline_reason_codes"]) == (15_000.0, ChangeType.NO_CHANGE, [7001])
    assert out["behaviour_grade"] == 2 and out["matrix_cell_id"]  # "we did consider your account" is itself an answer


def test_arrears_this_month_excludes_from_increases_and_proposes_a_decrease_to_what_is_owed(decide):
    out = decide(worst_arrears_months_now=1, worst_arrears_months_6=1)
    assert out["exclusion_codes"] == [1, 2]
    assert (out["change_type_code"], out["final_proposed_limit"], out["decrease_trigger_codes"]) == (ChangeType.DECREASE, 8_445.0, ["D03"])
    assert out["notice_class_code"] == NoticeClass.IMMEDIATE and out["decline_reason_codes"] == []


def test_a_tiny_spender_is_held_at_the_current_limit_and_the_observed_spend_cap_is_named(decide):
    out = decide(current_limit=6_000.0, cycle_purchase_values=[400.0, 300.0, 350.0, 200.0, 250.0, 100.0])
    assert (out["binding_cap_code"], out["final_proposed_limit"], out["decline_reason_codes"]) == (5, 6_000.0, [7002])


def test_a_grade_nine_account_gets_no_increase(decide):
    out = decide(worst_arrears_months_24=3, months_since_last_arrears=1, payment_to_balance_ratio_6m=0.05, over_limit_cycles_12m=3,
                 cash_withdrawal_ratio_12m=0.6, bureau_enquiry_velocity_6m=9.0)
    assert out["behaviour_grade"] >= 9 and out["matrix_multiplier_unadjusted"] == 1.0
    assert out["final_proposed_limit"] == out["current_limit"]


def test_an_account_that_has_never_drawn_reads_the_zero_utilisation_cell(decide):
    out = decide(cycle_balances=[0.0] * 6)
    assert out["utilisation_band"] == 1 and out["matrix_cell_id"].endswith("|1|5")


def test_a_ninety_percent_drawn_account_reads_the_top_utilisation_cell(decide):
    out = decide(cycle_balances=[13_500.0] * 6)
    assert out["utilisation_band"] == 8 and out["matrix_cell_id"].endswith("|8|5")


def test_the_score_shift_and_pd_multiplier_apply_in_scope_and_keep_the_unadjusted_values(decide):
    partner = decide(channel_code=5)
    assert (partner["behaviour_score_unadjusted"], partner["behaviour_score"]) == (700.0, 685.0)
    assert partner["score_adjustments_applied"] == ["ADJ-07-001"]
    facility = decide(product_code=21)
    assert facility["probability_of_default"] == pytest.approx(facility["probability_of_default_unadjusted"] * 1.25)
    assert facility["pd_adjustments_applied"] == ["ADJ-07-002"]


def test_a_param_changes_the_decision_without_touching_the_code(decide, params):
    assert decide()["final_proposed_limit"] == 20_000.0
    tighter = {**params, "limit_mgmt": {**params["limit_mgmt"], "caps": {
        **params["limit_mgmt"]["caps"], "cap_observed_spend": {"spend_multiple": 1.0, "spend_floor": 2_000.0}}}}
    out = decide(params_document=tighter)  # 1 x R7 900 of spend no longer supports a rise
    assert (out["binding_cap_code"], out["final_proposed_limit"]) == (5, 15_000.0)


def test_a_missing_required_input_is_an_error(decide):
    with pytest.raises(DeciderError, match="current_limit"):
        decide(current_limit=None)


def test_the_params_document_on_disk_is_valid_for_the_pipeline(params):
    pipeline.build(build_matrix_table()).parameters().validate(params).check()
    assert pipeline.build(build_matrix_table()).parameters().validate({"limit_mgmt": {"caps": {"cap_income_multiple": {"income_multiple": 99}}}}).errors
