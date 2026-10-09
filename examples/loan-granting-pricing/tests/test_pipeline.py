"""The whole decision on the sample application and its variations, scored through the pipeline."""
import copy
import math

import polars as pl
import pytest
from decider import Engine

from loan_granting import reasons
from loan_granting.pipeline import build


def annuity(principal, annual_rate, months):
    monthly = annual_rate / 12
    return principal * monthly / (1 - (1 + monthly) ** -months)


def recommended(out):
    return out["recommended_offer"]


def with_params(params, step, **changes):
    """A copy of the params document with `changes` on the named step. Never edit one in place after scoring with it."""
    changed = copy.deepcopy(params)
    changed["loan_granting"][step].update(changes)
    return changed


def test_the_sample_application_is_approved_with_the_requested_amount_over_36_months(decide):
    out = decide()
    assert (out["outcome_code"], out["referral_queue_code"], out["decline_reason_codes"]) == (1, 0, [])
    assert (out["risk_grade"], out["max_affordable_instalment"]) == (9, 5309.05)
    best = recommended(out)
    assert (best["term_months"], best["offered_amount"], best["instalment"], best["binding_constraint_code"]) == (
        36, 120_000.0, 4_969.47, "BIND-REQ")
    assert sum(offer["is_recommended"] for offer in out["offers"]) == 1


def test_the_offer_set_has_one_offer_per_surviving_term_and_says_why_the_others_are_missing(decide):
    out = decide()
    assert [o["term_months"] for o in out["offers"]] == [6, 12, 18, 24, 36, 48]
    assert [(s["term_months"], s["reason_codes"]) for s in out["suppressed_terms"]] == [
        (60, [reasons.R_TOTAL_COST_RATIO_EXCEEDED]),
        (72, [reasons.R_TOTAL_COST_RATIO_EXCEEDED, reasons.R_IN_DUPLUM_EXCEEDED]),
        (84, [reasons.R_TOTAL_COST_RATIO_EXCEEDED, reasons.R_IN_DUPLUM_EXCEEDED])]


def test_every_offer_is_affordable_and_its_instalment_recomputes_from_the_annuity(decide):
    out = decide()
    for offer in out["offers"]:
        assert offer["instalment"] <= out["max_affordable_instalment"]
        by_hand = annuity(offer["amount_financed"], offer["nominal_annual_rate"], offer["term_months"])
        assert offer["instalment"] == round(by_hand + offer["monthly_service_fee"] + offer["credit_life_premium"], 2)
        assert offer["amount_financed"] == offer["offered_amount"] + offer["initiation_fee"]


def test_the_cap_waterfall_chain_and_the_rule_that_bound_each_ceiling_are_recorded(decide):
    out = decide()
    assert [(c["rule_id"], c["value"]) for c in out["amount_cap_chain"]] == [
        ("seed", 500_000.0), ("CAP-0100", 185_000.0), ("CAP-0210", 120_000.0), ("CAP-0420", 150_000.0)]
    assert (out["amount_cap"], out["amount_cap_binding_rule_id"], out["uplift_authority_reference"]) == (
        150_000.0, "CAP-0420", "CC-2026-14")
    assert (out["worst_acceptable_grade"], out["worst_acceptable_grade_binding_rule_id"]) == (9.0, "CAP-0210")
    assert len(out["waterfall_rules"]) == 12


def test_the_decision_records_which_table_cell_and_version_each_figure_came_from(decide):
    out = decide()
    assert recommended(out)["rate_cell_id"].startswith("rate_card.flex_loan@rc-2026.09#")
    assert recommended(out)["rate_card_version"] == "rc-2026.09"
    assert out["risk_grade_cell_id"].startswith("risk_grade@") and out["appetite_cell_id"].startswith("appetite@")
    assert out["fee_caps_cell_id"] == "fee_caps@fee_caps@2026#caps"
    assert out["norm_table_version"] and out["applicant1_tax_table_cell_id"]


def test_asking_for_no_amount_and_no_term_offers_what_the_client_can_have(decide):
    out = decide(requested_amount=None, requested_term_months=None)
    assert out["outcome_code"] == 1
    assert out["eligibility_gate_verdicts"]["max_age_at_maturity"] == "not_evaluated"
    by_term = {o["term_months"]: o for o in out["offers"]}
    assert by_term[36]["offered_amount"] == 129_400.0 and by_term[36]["binding_constraint_code"] == "BIND-AFF"
    assert by_term[48]["offered_amount"] == out["amount_cap"] == 150_000.0
    assert by_term[48]["binding_constraint_code"] == "BIND-CAP"
    assert recommended(out)["term_months"] == 48


@pytest.mark.parametrize("objective, term, amount", [
    ("largest_amount", 36, 120_000.0), ("lowest_total_cost", 6, 26_500.0), ("best_expected_value", 12, 49_900.0)])
def test_the_recommendation_objective_is_a_param_not_a_release(decide, params, objective, term, amount):
    out = decide(params_document=with_params(params, "recommendation_objective", recommendation_objective=objective))
    assert (out["recommendation_objective"], recommended(out)["term_months"], recommended(out)["offered_amount"]) == (
        objective, term, amount)


def test_a_lower_statutory_ceiling_unprices_the_cells_above_it(decide, params):
    assert decide()["statutory_ceiling"] == 0.28
    gazetted = decide(params_document=with_params(params, "statutory_ceiling", repo_rate=0.0))
    assert gazetted["statutory_ceiling"] == 0.21 and gazetted["outcome_code"] == 1  # the cells in use are under 21%
    out = decide(params_document=with_params(params, "statutory_ceiling", repo_rate=0.0, margin_over_repo=0.05))
    assert (out["outcome_code"], out["has_any_offer"], out["decline_reason_codes"]) == (4, False, [reasons.R_AFFORDABILITY_FAIL])
    assert {s["term_months"] for s in out["suppressed_terms"]} == {6, 12, 18, 24, 36, 48, 60, 72, 84}


def test_an_income_too_small_for_any_instalment_is_declined_for_affordability(decide):
    out = decide(applicant1_payslip_income=3_000.0)
    assert (out["max_affordable_instalment"], out["has_any_offer"]) == (0.0, False)
    assert (out["outcome_code"], out["decline_reason_codes"], out["primary_reason_code"]) == (4, [1310], 1310)
    assert [s["reason"] for s in out["suppressed_terms"]] == ["no affordable amount"] * 9


def test_a_joint_application_is_assessed_on_both_incomes(decide):
    solo = decide()
    joint = decide(is_joint_application=True, applicant2_payslip_income=9_000.0, applicant2_employment_type_code=2,
                   applicant2_dependants_count=3)
    assert joint["gross_monthly_income"] == 26_125.0
    assert joint["max_affordable_instalment"] > solo["max_affordable_instalment"]
    assert max(o["offered_amount"] for o in joint["offers"]) >= max(o["offered_amount"] for o in solo["offers"])


def test_a_declared_own_credit_life_policy_drops_the_premium_and_approves_with_conditions(decide):
    out = decide(credit_life_substitution_declared=True)
    assert recommended(out)["credit_life_premium"] == 0.0
    assert recommended(out)["instalment"] == round(
        annuity(126_500.0, recommended(out)["nominal_annual_rate"], 36) + 69.0, 2)
    assert out["outcome_code"] == 2


def test_a_client_new_to_the_bank_is_scored_on_the_new_to_bank_card_capped_and_not_offered_84_months(decide):
    out = decide(internal_tenure_months=0.0)
    assert (out["scorecard_id"], out["segment_code"], out["amount_cap"]) == (1010, 3.0, 75_000.0)  # R60 000 plus the uplift
    assert [s["reason"] for s in out["suppressed_terms"] if s["term_months"] == 84] == ["not offered to new-to-bank clients"]


def test_a_thin_file_takes_the_thin_file_score_shift(decide):
    out = decide(bureau_account_count=2)
    assert (out["scorecard_id"], out["score_unadjusted"], out["score"]) == (1012, 685.0, 667.0)
    assert "ADJ-03-001" in out["adjustments_applied"]


def test_an_overlay_moves_the_grade_and_the_unadjusted_values_survive(decide):
    base, channel_4 = decide(), decide(channel_code=4)
    assert (base["risk_grade"], channel_4["risk_grade"]) == (9, 10)
    assert channel_4["probability_of_default_unadjusted"] == base["probability_of_default_unadjusted"]
    assert channel_4["probability_of_default"] == pytest.approx(base["probability_of_default"] * 1.35)
    assert "ADJ-03-003" in channel_4["adjustments_applied"] and "ADJ-03-003" not in base["adjustments_applied"]


def test_a_grade_worse_than_the_arrears_rule_allows_is_declined_beyond_appetite_with_the_score_reasons(decide):
    out = decide(channel_code=4, bureau_score=480.0, revolving_utilisation=0.8)
    assert out["outcome_code"] == 4 and out["primary_reason_code"] == reasons.R_BEYOND_APPETITE
    assert out["decline_reason_codes"][1:] == [4101, 4105]  # the characteristics that cost the most points
    assert out["risk_grade"] > out["worst_acceptable_grade"]


def test_the_overlay_stack_runs_off_through_the_same_code_by_one_switch_per_point(decide, params):
    switched_off = copy.deepcopy(params)
    for point in switched_off["loan_granting"]["scoring"].values():
        point.update({"adjustment_stack_enabled": False} if "adjustment_stack_enabled" in point else {})
    out = decide(channel_code=4, params_document=switched_off)
    assert (out["adjustments_applied"], out["risk_grade"], out["outcome_code"]) == ([], 9, 1)
    assert out["probability_of_default"] == out["probability_of_default_unadjusted"]


def test_a_fraud_decline_declines_and_a_timeout_refers_unless_the_small_loan_bypass_applies(decide):
    declined = decide(fraud_verdict_code=3)
    assert (declined["outcome_code"], declined["decline_reason_codes"]) == (4, [reasons.R_FRAUD_DECLINE])
    refer = decide(fraud_response_ms=900.0)
    assert (refer["outcome_code"], refer["referral_queue_code"], refer["fraud_bypass_applied"]) == (3, 3, False)
    bypassed = decide(fraud_response_ms=900.0, requested_amount=10_000.0)
    assert (bypassed["outcome_code"], bypassed["fraud_bypass_applied"]) == (2, True)
    unknown_client = decide(fraud_response_ms=900.0, requested_amount=10_000.0, internal_tenure_months=6.0)
    assert (unknown_client["outcome_code"], unknown_client["referral_queue_code"]) == (3, 3)


def test_a_fraud_refer_continues_to_the_end_but_the_outcome_is_a_referral(decide):
    out = decide(fraud_verdict_code=2)
    assert (out["outcome_code"], out["referral_queue_code"]) == (3, 3) and out["has_any_offer"]


def test_bureau_consent_that_is_missing_or_expired_refers_to_the_channel(decide):
    for change in ({"bureau_consent_given": False}, {"bureau_consent_expires_on": "2026-09-01"}):
        out = decide(**change)
        assert (out["outcome_code"], out["referral_queue_code"], out["decline_reason_codes"]) == (3, 1, [reasons.R_CONSENT_MISSING])
    assert decide(bureau_consent_expires_on="2027-01-01")["outcome_code"] == 1


def test_the_consent_field_cannot_be_left_out(decide):
    with pytest.raises(Exception, match="bureau_consent_given"):
        decide(bureau_consent_given=None)


@pytest.mark.parametrize("change, queue", [
    ({"bureau_identity_subject_count": 2}, 2), ({"bureau_account_list_truncated": True}, 2),
    ({"bureau_identity_mismatch": True}, 2), ({"bureau_response_status": "failed"}, 2),
])
def test_a_bureau_data_quality_problem_refers_and_never_declines(decide, change, queue):
    out = decide(**change)
    assert (out["outcome_code"], out["referral_queue_code"], out["bureau_referral_required"]) == (3, queue, True)


def test_a_minor_bureau_defect_is_recorded_and_the_decision_goes_on(decide):
    out = decide(bureau_has_unparseable_account=True)
    assert (out["data_quality_verdict"], out["outcome_code"]) == (1, 1)


def test_a_bureau_no_hit_is_a_manual_underwriting_referral_only_without_internal_history(decide):
    no_hit = {"bureau_response_status": "no_hit", "bureau_account_count": 0}
    assert (decide(**no_hit, internal_tenure_months=0.0)["referral_queue_code"]) == 4
    assert decide(**no_hit)["outcome_code"] == 1  # 41 months of internal history stands in for the bureau


def test_a_stale_bureau_view_makes_affordability_indeterminate_and_refers(decide):
    out = decide(bureau_as_of_date="2026-07-01")
    assert out["bureau_is_stale"] is True
    assert (out["outcome_code"], out["referral_queue_code"], out["decline_reason_codes"]) == (3, 5, [1320])
    assert decide(bureau_as_of_date="2026-08-20")["bureau_is_stale"] is False  # 35 days is inside Flex Loan's 45


def test_a_waterfall_decline_declines_with_its_reason(decide):
    assert decide(bureau_enquiry_count_60d=12)["decline_reason_codes"] == [reasons.R_ENQUIRY_VELOCITY_EXCESSIVE]
    out = decide(months_employed=2.0)  # the short tenure also costs points, so the grade falls beyond appetite
    assert out["outcome_code"] == 4 and reasons.R_EMPLOYMENT_TENURE_TOO_SHORT in out["decline_reason_codes"]


def test_the_batch_run_gives_the_same_answers_as_scoring_one_record_at_a_time(documents, sample, params, parse):
    executable = Engine().bind(build(**documents), mode="interpreted")
    records = [parse(sample), parse({**sample, "requested_amount": 40_000.0, "channel_code": 4}),
               parse({**sample, "applicant_age_years": 15.0})]
    batch = executable.run(pl.DataFrame(records), params=params)
    for i, record in enumerate(records):
        single = executable.score(record, params)
        for name in ("outcome_code", "risk_grade", "amount_cap", "max_affordable_instalment", "solve_total_evaluations"):
            assert batch[name][i] == single[name], (i, name)
        assert batch["recommended_offer"][i] == single["recommended_offer"] or (
            not single["recommended_offer"] and not batch["recommended_offer"][i])


def test_the_params_document_validates_against_the_pipeline(documents, params):
    build(**documents).parameters().validate(params).check()
