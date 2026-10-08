"""The Flex Loan quote pipeline over the sample request: every figure is worked out by hand in the comments."""
import copy
from datetime import date

import pytest

from credit_core.pipeline import ADJUSTMENT_SET_ID, REASON_REGISTRY, R_AFFORDABILITY_FAIL
from credit_core.vocab import OUTCOME_APPROVE, OUTCOME_DECLINE, OUTCOME_REFER


def test_sample_request_is_approved_with_the_hand_worked_figures(quote, params, sample):
    out = quote.score(sample, params)
    # Income: payslip 18 500 less the 5% payslip haircut. Tax 2026: (17 575 - 8 900) * 18%.
    assert out["gross_monthly_income"] == 17575.0
    assert out["statutory_deductions"] == 1561.5
    assert out["net_monthly_income"] == 16013.5
    # Debts: the Flex Loan's stated 1 150, and the card at 3% of its 15 000 limit.
    assert out["existing_obligations"] == 1600.0
    # The declared expenses (6 200) are above the norm floor, so they stand.
    assert (out["living_expenses"], out["expense_basis_code"]) == (6200.0, 1)
    assert out["discretionary_income"] == 8213.5  # 16 013.5 - 6 200 - 1 600
    assert out["max_affordable_instalment"] == 7227.88  # 8 213.5 less the 12% buffer
    assert out["affordability_verdict_code"] == 1
    assert (out["score_unadjusted"], out["score"]) == (665.0, 642.0)
    assert out["risk_grade"] == 8
    assert out["offered_amount"] == 65000.0
    # Rate card: 5% + 7 grades * 1.5 + amount band 12 * 0.01 + term band 21 * 0.02 = 16.04%.
    assert out["nominal_annual_rate"] == 0.1604
    r = 0.1604 / 12
    assert out["instalment_before_fees"] == pytest.approx(65000 * r / (1 - (1 + r) ** -36))
    assert out["initiation_fee"] == 6500.0  # 1 050 + 10% of 55 000 is 6 550, capped
    assert out["instalment"] == 2355.49  # 2 286.49 plus the 69 service fee
    assert out["total_cost_of_credit"] == 91297.64  # 36 * 2 355.49 + 6 500
    assert out["total_interest"] == 26297.64
    assert out["outcome_code"] == OUTCOME_APPROVE
    assert out["decline_reason_codes"] == [] and out["primary_reason_code"] is None


def test_decision_id_comes_back_unchanged(quote, params, sample):
    assert quote.score(sample, params)["decision_id"] == sample["decision_id"]


def test_every_table_read_names_its_version_and_cell(quote, params, sample):
    out = quote.score(sample, params)
    assert out["tax_table_cell_id"] == "tax_table@tax@2026#1"
    assert out["statutory_norm_cell_id"] == "expense_norms.statutory@norms.statutory@2025#10"
    assert out["internal_norm_cell_id"] == "expense_norms.internal@norms.internal@2026#10|10"
    assert out["norm_table_version"] == "norms.internal@2026"
    assert out["calibration_cell_id"] == "calibration@cal-2026.09#3"
    assert out["risk_grade_cell_id"] == "risk_grade@rg-2026.09#10|3|8"
    assert out["rate_cell_id"] == "rate_card.flex_loan@rc-2026.09#12|21|8"
    assert out["rate_card_version"] == "rc-2026.09"
    assert out["fee_caps_cell_id"] == "fee_caps@fee_caps@2026#caps"


def test_both_score_overlays_apply_in_stack_order_and_the_unadjusted_score_survives(quote, params, sample):
    out = quote.score(sample, params)
    assert out["adjustment_set_id"] == ADJUSTMENT_SET_ID
    assert out["adjustments_applied"] == ["ADJ-2026-001", "ADJ-2026-002"]
    assert out["score"] == out["score_unadjusted"] - 18.0 - 5.0


def test_overlays_apply_only_inside_their_scope_and_window(quote, params, sample):
    other_segment = quote.score({**sample, "segment_code": 1}, params)
    assert other_segment["adjustments_applied"] == ["ADJ-2026-002"]
    before_channel_overlay = quote.score({**sample, "decision_date": date(2026, 5, 31)}, params)
    assert before_channel_overlay["adjustments_applied"] == ["ADJ-2026-001"]
    after_channel_overlay = quote.score({**sample, "decision_date": date(2027, 6, 1)}, params)
    assert after_channel_overlay["adjustments_applied"] == ["ADJ-2026-001"]


def test_switching_the_stack_off_in_params_returns_the_unadjusted_score(quote, params, sample):
    off = copy.deepcopy(params)
    off["flex_loan_quote"]["apply_score_adjustments"]["adjustment_stack_enabled"] = False
    out = quote.score(sample, off)
    assert (out["score"], out["score_unadjusted"], out["adjustments_applied"]) == (665.0, 665.0, [])


def test_a_decision_keeps_the_tables_of_its_own_date(quote, params, sample):
    gazetted = quote.score({**sample, "decision_date": date(2026, 10, 1)}, params)
    assert gazetted["statutory_norm_cell_id"].endswith("norms.statutory@2026#10")
    before_tax_change = quote.score({**sample, "decision_date": date(2026, 2, 28)}, params)
    assert before_tax_change["tax_table_cell_id"] == "tax_table@tax@2025#1"
    assert before_tax_change["statutory_deductions"] == 1633.5  # (17 575 - 8 500) * 18%


def test_replaying_a_decision_gives_the_same_answer(quote, params, sample):
    assert quote.score(sample, params) == quote.score(sample, params)


def test_an_instalment_over_capacity_is_declined_with_the_affordability_reason(quote, params, sample):
    out = quote.score({**sample, "requested_amount": 500000.0, "term_months": 6}, params)
    assert out["instalment"] > out["max_affordable_instalment"]
    assert out["outcome_code"] == OUTCOME_DECLINE
    assert out["decline_reason_codes"] == [R_AFFORDABILITY_FAIL] == [out["primary_reason_code"]]
    assert out["reason_registry_version"] == REASON_REGISTRY.version


def test_an_instalment_just_over_capacity_is_referred(quote, params, sample):
    out = quote.score({**sample, "requested_amount": 202800.0}, params)
    assert out["max_affordable_instalment"] < out["instalment"] <= out["max_affordable_instalment"] * 1.05
    assert out["affordability_verdict_code"] == 2
    assert out["outcome_code"] == OUTCOME_REFER


def test_ineligible_applicants_are_declined_with_every_reason_most_severe_first(quote, params, sample):
    out = quote.score({**sample, "is_sanctioned": True, "applicant_age_years": 17.0}, params)
    assert out["outcome_code"] == OUTCOME_DECLINE
    assert out["decline_reason_codes"] == [1006, 1001]  # sanctioned (rank 1) before underage (rank 10)
    assert out["primary_reason_code"] == 1006


def test_a_request_outside_the_product_range_is_flagged_and_priced_from_the_edge_band(quote, params, sample):
    out = quote.score({**sample, "requested_amount": 1_000_000.0}, params)
    assert out["out_of_range"] is True
    assert out["rate_cell_id"] == "rate_card.flex_loan@rc-2026.09#95|21|8"
    assert quote.score(sample, params)["out_of_range"] is False
