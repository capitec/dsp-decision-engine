"""One client's assignment record, through the pipeline: suppressions, the pre-assessment, the tree and the overlays."""
import copy

import pytest

from campaign_trees import vocab

# A client with room for a big loan, who reaches leaf 910 (tier A, in-app).
TIER_A = {"settlement_ratio": 0.5, "behaviour_score": 700, "worst_arrears_months_12m": 0, "discretionary_income": 5000.0,
          "income_band_code": 7, "estimated_discretionary_income": 40_000.0, "app_logins_90d": 5, "current_flex_balance": 5_000.0}
OCTOBER = {"cycle_date": "2026-10-05"}


def test_the_sample_client_is_offered_what_the_pre_assessment_says_they_can_afford(decide):
    out = decide()
    assert out["tree_path"] == "n1>n2>n4>n6>n8>n10>leaf_912"
    assert (out["assignment_outcome_code"], out["offer_tier_code"], out["reason_label"]) == (vocab.TARGET, 2, 9103)
    assert (out["pre_assessed_amount"], out["binding_constraint_code"]) == (20_800.0, "BIND-AFF")
    assert out["advertised_amount"] == 20_800.0
    assert (str(out["pre_assessment_valid_from"]), str(out["pre_assessment_valid_to"])) == ("2026-09-24", "2026-10-29")


def test_the_tree_reads_the_pre_assessed_amount_the_same_run_computed(decide):
    out = decide(estimated_discretionary_income=1000.0)
    assert out["pre_assessed_amount"] < 15_000.0
    assert (out["tree_path"], out["reason_label"]) == ("n1>n2>n4>n6>n8>leaf_904", 9204)
    assert out["assignment_outcome_code"] == vocab.DO_NOT_TARGET and out["advertised_amount"] == 0.0


def test_a_daily_delta_pre_assessment_expires_sooner(decide):
    assert str(decide(is_daily_delta=True)["pre_assessment_valid_to"]) == "2026-10-04"


@pytest.mark.parametrize("change, code", [
    ({"is_deceased": True}, "S01"), ({"in_debt_review": True}, "S02"), ({"fraud_marker": True}, "S11"),
    ({"on_do_not_target_list": True}, "S14"),
])
def test_an_absolutely_suppressed_client_is_not_evaluated_at_all(decide, change, code):
    out = decide(**change)
    assert code in out["suppression_codes"] and out["evaluated"] is False
    assert out["tree_path"] is None and out["pre_assessed_amount"] is None
    assert (out["assignment_outcome_code"], out["advertised_amount"]) == (vocab.DO_NOT_TARGET, 0.0)


def test_every_suppression_that_applies_is_recorded(decide):
    out = decide(is_deceased=True, in_debt_review=True, fraud_marker=True, on_do_not_target_list=True)
    assert out["suppression_codes"] == ["S01", "S02", "S11", "S14"]


def test_a_measurement_relevant_suppression_still_walks_the_tree_but_never_contacts(decide):
    out = decide(contacts_30d=4)
    assert out["suppression_codes"] == ["S30"]
    assert out["tree_path"] == "n1>n2>n4>n6>n8>n10>leaf_912"
    assert (out["assignment_outcome_code"], out["advertised_amount"]) == (vocab.SUPPRESSED, 0.0)


def test_a_client_without_consent_on_their_channel_is_suppressed_on_it(decide):
    assert decide(sms_permitted=False)["suppression_codes"] == ["S08"]
    assert decide(consent_channel_code=4)["suppression_codes"] == []  # email is permitted


def test_an_ineligible_client_is_not_evaluated(decide):
    out = decide(applicant_age_years=16.0)
    assert out["eligibility_decline_reasons"] == [1001] and out["evaluated"] is False
    assert out["assignment_outcome_code"] == vocab.DO_NOT_TARGET


def test_a_client_in_the_control_group_is_evaluated_and_held_out(decide, params):
    everyone_is_control = copy.deepcopy(params)
    everyone_is_control["campaign_trees"]["is_control"]["control_percentage"] = 1.0
    out = decide(params_document=everyone_is_control)
    assert out["is_control"] is True and out["tree_path"] is not None
    assert (out["assignment_outcome_code"], out["advertised_amount"]) == (vocab.HOLD_OUT, 0.0)


def test_the_volume_dial_changes_the_leaf_and_the_unadjusted_leaf_survives(decide):
    out = decide(discretionary_income=2400.0)
    assert out["overlay_stack_id"] == "OS-ADJ-C23-VOL-001"
    assert (out["leaf"], out["unadjusted_leaf"]) == ("leaf_903", "leaf_912")
    assert out["tree_path"] == "n1>n2>n4>n6>leaf_903"
    assert out["unadjusted_tree_path"] == "n1>n2>n4>n6>n8>n10>leaf_912"
    assert (out["assignment_outcome_code"], out["unadjusted_leaf_outcome_code"]) == (vocab.DO_NOT_TARGET, vocab.TARGET)


def test_without_the_overlay_the_adjusted_and_unadjusted_answers_agree(decide, params):
    no_overlay = copy.deepcopy(params)
    no_overlay["campaign_trees"]["evaluate"]["evaluation"]["tree_adjusted"]["n6_discretionary_income_floor"] = 2200
    out = decide(params_document=no_overlay, discretionary_income=2400.0)
    assert out["leaf"] == out["unadjusted_leaf"] == "leaf_912"


def test_the_cap_reduction_lowers_the_advertised_amount_in_october_only(decide):
    september, october = decide(**TIER_A), decide(**TIER_A, **OCTOBER)
    assert september["leaf"] == october["leaf"] == "leaf_910"
    assert september["pre_assessed_amount"] > 200_000.0
    assert (september["advertised_tier_ceiling"], september["advertised_amount"]) == (250_000.0, 250_000.0)
    assert (october["advertised_tier_ceiling"], october["advertised_amount"]) == (200_000.0, 200_000.0)
    assert october["tier_ceiling_before_cap_overlay"] == 250_000.0 and october["cap_adjustments_applied"] == ["ADJ-C23-CAP-001"]


def test_the_cap_reduction_can_be_switched_off_for_the_shadow_run(decide, params):
    shadow = copy.deepcopy(params)
    shadow["campaign_trees"]["evaluate"]["evaluation"]["cap_reduction"]["adjustment_stack_enabled"] = False
    out = decide(params_document=shadow, **TIER_A, **OCTOBER)
    assert out["advertised_amount"] == 250_000.0 and out["cap_adjustments_applied"] == []


def test_a_campaign_never_advertises_more_than_the_pre_assessed_amount(decide):
    out = decide(**{**TIER_A, "estimated_discretionary_income": 15_000.0})
    assert out["pre_assessed_amount"] < out["advertised_tier_ceiling"]
    assert out["advertised_amount"] == out["pre_assessed_amount"]
