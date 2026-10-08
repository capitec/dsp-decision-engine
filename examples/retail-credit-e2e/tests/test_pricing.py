"""Pricing: the card, the long-term loading and the rate add-on."""
import pytest

from reference import card_rate, instalment
from retail_credit.flex_card import card_tables


def test_the_card_has_a_rate_for_every_amount_band_term_and_grade():
    tables = card_tables()
    assert (len(tables["amount_bands"]), len(tables["cells"])) == (72, 72 * 40 * 12)
    assert tables["amount_bands"][0]["floor"] == 0.0 and tables["amount_bands"][36]["floor"] == 60_000.0


def test_rates_rise_with_grade_and_the_card_drops_at_the_sixty_thousand_edge(params):
    rates = [card_rate(params, 50_000.0, 60, grade)[0] for grade in range(1, 13)]
    assert rates == sorted(rates) and len(set(rates)) == 12
    below, at_edge = card_rate(params, 59_999.0, 60, 7)[0], card_rate(params, 60_000.0, 60, 7)[0]
    assert round(below - at_edge, 4) == 0.0157  # 1.55 points at the edge plus the 0.02 an ordinary band step is worth


@pytest.mark.parametrize("term, loading", [(45, 0.0), (46, 0.006), (60, 0.006), (61, 0.011), (72, 0.011), (73, 0.017), (84, 0.017)])
def test_terms_past_the_cards_last_column_are_priced_from_it_plus_a_loading(solve_once, params, term, loading):
    out = solve_once(offer_term_months=float(term), requested_amount=30_000.0, amount_cap=30_000.0,
                     instalment_ceiling=100_000.0)
    quote = out["record"]
    card, band, term_band = card_rate(params, 30_000.0, term, 7)
    assert term_band == min(term, 45) - 6
    assert quote["nominal_annual_rate_unadjusted"] == pytest.approx(card + loading)
    assert quote["instalment"] == instalment(params, 30_000.0, term, 7, out["credit_life_rate"], loading)


def test_the_rate_addon_applies_to_segment_four_only(decide):
    assert decide()["rate_addon_adjustments_applied"] == ["ADJ-0132"]
    staff = decide(is_staff=True)
    assert staff["segment_code"] == 11 and staff["rate_addon_bps"] == 0.0 and staff["rate_addon_adjustments_applied"] == []
    assert staff["nominal_annual_rate"] == staff["nominal_annual_rate_unadjusted"]


def test_the_add_on_is_resolved_once_and_the_offer_is_priced_with_it(decide):
    out = decide()
    assert out["rate_addon_bps"] == 0.006
    assert out["nominal_annual_rate"] == round(out["nominal_annual_rate_unadjusted"] + out["rate_addon_bps"], 6)


def test_the_amount_independent_terms_are_looked_up_once_beside_the_offer(decide):
    out = decide()
    assert (out["monthly_service_fee"], out["credit_life_rate"], out["credit_life_cell_id"]) == (87.98, 5.6, "credit_life@cl-2026.09#5")
    assert out["credit_life_premium"] == round(out["amount_financed"] / 1000 * 5.6, 2)
