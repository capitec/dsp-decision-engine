"""The combined grade and the offer: blend, overlay, ladder, appetite, rate card, fees and instalment."""
import math

import pytest
from conftest import entity

from business_nested import pricing


def logit(p):
    return math.log(p / (1 - p))


def test_the_combined_probability_blends_the_components_on_log_odds_at_forty_sixty(assess):
    out = assess()
    blended = 1 / (1 + math.exp(-(0.4 * logit(out["financial_pd"]) + 0.6 * logit(out["people_pd"]))))
    assert out["probability_of_default_before_overlay"] == pytest.approx(blended)


def test_the_component_weights_are_params_and_are_rescaled_to_sum_to_one(assess, with_params):
    only_financial = with_params(("business_credit", "decision", "business_grade", "combined_pd_before_overlay"),
                                 financial_weight=0.5, people_weight=0.0)
    out = assess(params_document=only_financial)
    assert out["probability_of_default_before_overlay"] == pytest.approx(out["financial_pd"])
    equal = with_params(("business_credit", "decision", "business_grade", "combined_pd_before_overlay"),
                        financial_weight=0.2, people_weight=0.2)
    middle = assess(params_document=equal)["probability_of_default_before_overlay"]
    assert middle == pytest.approx(1 / (1 + math.exp(-0.5 * (logit(out["financial_pd"]) + logit(out["people_pd"])))))


def test_term_facilities_carry_the_odds_overlay_and_revolving_facilities_do_not(assess):
    term, revolving = assess(product_code=50), assess(product_code=51)
    assert term["adjustments_applied"] == ["ADJ-05-BIZ-001"]
    assert term["probability_of_default"] == pytest.approx(1.25 * term["probability_of_default_unadjusted"])
    assert revolving["adjustments_applied"] == []
    assert revolving["probability_of_default"] == revolving["probability_of_default_unadjusted"]


def test_the_unadjusted_grade_shows_what_the_overlay_cost(assess):
    out = assess()
    assert (out["risk_grade"], out["risk_grade_unadjusted"]) == (7, 6)
    assert out["risk_grade_cell_id"] == "grade_ladder@biz-rg-2026.09#50|7"


def test_the_revolving_ladder_is_stricter_than_the_term_ladder(assess):
    term, revolving = assess(product_code=50), assess(product_code=51)
    assert (term["risk_grade_unadjusted"], revolving["risk_grade"]) == (6, 7)  # the same probability, a stricter ladder


def test_the_offer_is_the_request_when_appetite_allows(assess):
    out = assess(requested_amount=1_000_000.0)
    assert (out["offered_amount"], out["max_amount"], out["binding_constraint_code"]) == (1_000_000.0, 3_500_000.0, pricing.REQUESTED_AMOUNT)


def test_the_offer_is_capped_by_the_appetite_of_the_grade(assess):
    out = assess(requested_amount=5_000_000.0)
    assert (out["offered_amount"], out["binding_constraint_code"]) == (3_500_000.0, pricing.APPETITE_CAP)
    assert out["appetite_cell_id"] == "business_appetite@biz-rate-2026.09#7"


@pytest.mark.parametrize("requested, offered", [(150_049.0, 150_000.0), (150_050.0, 150_100.0), (150_000.0, 150_000.0)])
def test_the_amount_is_rounded_to_the_nearest_hundred_half_up(assess, requested, offered):
    out = assess(requested_amount=requested)
    assert out["offered_amount"] == offered and out["binding_constraint_code"] == pricing.REQUESTED_AMOUNT


def test_a_request_the_rounding_trims_is_not_blamed_on_appetite(assess):
    assert assess(requested_amount=1_234_549.0)["binding_constraint_code"] == pricing.REQUESTED_AMOUNT


def test_the_rate_is_read_from_the_card_by_grade_and_security_type(assess):
    out = assess()  # grade 7, security 2
    assert out["nominal_annual_rate"] == pytest.approx(0.08 + 6 * 0.018 + 0.01)
    assert out["rate_card_cell_id"] == "business_rate_card@biz-rate-2026.09#7|2"
    assert assess(security_type_code=9)["nominal_annual_rate"] == 0.30


def test_the_instalment_is_the_annuity_payment_plus_the_service_fee_to_the_cent(assess):
    out = assess()
    r = out["nominal_annual_rate"] / 12
    annuity = 3_500_000.0 * r / (1 - (1 + r) ** -48)
    assert out["instalment"] == pytest.approx(round(annuity + 69.0, 2), abs=0.005)
    assert out["total_cost_of_credit"] == pytest.approx(round(out["instalment"] * 48 + out["initiation_fee"], 2))
    assert (out["initiation_fee"], out["monthly_service_fee"]) == (6_500.0, 69.0)


def test_the_term_defaults_to_three_years_when_the_request_has_none(assess):
    assert assess(requested_term_months=None)["term_months"] == 36
    assert assess(requested_term_months=0)["term_months"] == 36
    assert assess()["term_months"] == 48


def test_grade_twelve_has_no_appetite_and_is_declined(assess, with_params):
    people_only = with_params(("business_credit", "decision", "business_grade", "combined_pd_before_overlay"),
                              financial_weight=0.0, people_weight=1.0)
    bad_owner = entity(1, 5, 100.0, bureau_score=300.0, months_on_record=1.0, worst_delinquency_months=12.0)
    out = assess([bad_owner], params_document=people_only)
    assert (out["risk_grade"], out["offered_amount"], out["binding_constraint_code"]) == (12, 0.0, pricing.APPETITE_CAP)
    assert (out["outcome_code"], out["decline_reason_codes"]) == (4, [5602])
