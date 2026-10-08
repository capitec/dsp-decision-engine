"""The financial component: three ratios, a scorecard and a grade."""
import pytest

from business_nested import financial


def test_the_sample_statements_give_the_three_ratios(assess):
    out = assess()
    assert out["interest_cover"] == pytest.approx(1_100_000 / 210_000)
    assert out["current_ratio"] == pytest.approx(2_400_000 / 1_500_000)
    assert out["gearing"] == pytest.approx(1_800_000 / 2_600_000)


def test_the_ratios_score_700_plus_the_points_of_their_bands(assess):
    out = assess()  # cover 5.2 strong +30, current 1.6 strong +15, gearing 0.69 low +15
    assert out["financial_score"] == 760.0
    assert out["financial_pd"] == pytest.approx(1 / (1 + 2.718281828459045 ** (160 / 60)))
    assert out["financial_grade"] == 5 and out["financial_confidence_code"] == financial.CONFIDENCE_HIGH


def test_weak_statements_score_lower_and_grade_worse(assess):
    out = assess(ebitda=100_000.0, current_assets=900_000.0, interest_bearing_debt=9_000_000.0)
    assert out["financial_score"] == 700 - 30 - 20 - 25
    assert out["financial_grade"] > assess()["financial_grade"]


def test_no_finance_charges_is_full_cover_unless_earnings_are_negative():
    assert financial.interest_cover(500_000.0, 0.0) == financial.RATIO_CAP
    assert financial.interest_cover(-1.0, 0.0) == 0.0
    assert financial.interest_cover(-100.0, 50.0) == -2.0


def test_a_ratio_without_a_denominator_is_capped_rather_than_infinite():
    assert financial.current_ratio(1_000.0, 0.0) == financial.RATIO_CAP
    assert financial.gearing(1_000.0, 0.0) == financial.RATIO_CAP
    assert financial.gearing(1_000.0, -5.0) == financial.RATIO_CAP


def test_missing_statement_lines_are_scored_in_the_default_bins_and_lower_the_confidence(assess):
    out = assess(finance_charges=None, current_liabilities=None, tangible_net_worth=None)
    assert out["interest_cover"] == financial.RATIO_CAP and out["financial_confidence_code"] == financial.CONFIDENCE_LOW
    assert out["financial_score"] == 700 + 30 + 15 - 25  # capped ratios land in the best, best and worst bins
