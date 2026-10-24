import pytest
from decikit import DecisionTableConfig

from credit_core.rate_card import (
    FLEX_LOAN_AMOUNT_BANDS, FLEX_LOAN_GRADES, FLEX_LOAN_TERM_BANDS, diff_cards, generate_flex_loan_card, lookup,
    out_of_range, rate_card_rate,
)


@pytest.fixture(scope="module")
def doc():
    return generate_flex_loan_card("v1")


@pytest.fixture(scope="module")
def price(doc):
    card = lookup(DecisionTableConfig.load(doc)).compile("interpreted")

    def run(amount, term, grade):
        return card.run({"offered_amount": amount, "term_months": term, "risk_grade": grade})

    return run


def test_the_card_has_a_cell_for_every_amount_term_and_grade(doc):
    assert len(doc["rows"]) == FLEX_LOAN_AMOUNT_BANDS * FLEX_LOAN_TERM_BANDS * FLEX_LOAN_GRADES == 63_360
    assert len({r["cell_id"] for r in doc["rows"]}) == 63_360


def test_a_lookup_returns_the_rate_and_the_cell_it_came_from(price):
    # R50 000 is amount band 9 (bands are R5 187.50 wide from R2 000), 36 months is term band 21.
    # Rate: 5% + 4 grades * 1.5 + 9 * 0.01 + 21 * 0.02 = 11.51%.
    out = price(50_000.0, 36, 5)
    assert out["nominal_annual_rate"] == 0.1151
    assert out["rate_cell_id"] == "rate_card.flex_loan@v1#9|21|5"
    assert out["rate_card_version"] == "v1"


def test_a_value_on_a_band_edge_belongs_to_the_upper_band(price):
    assert price(2_000.0 + 10 * 5_187.5, 36, 5)["rate_cell_id"].endswith("#10|21|5")
    assert price(2_000.0 + 10 * 5_187.5 - 0.01, 36, 5)["rate_cell_id"].endswith("#9|21|5")


def test_the_cell_id_in_the_document_matches_the_one_the_lookup_reports(doc, price):
    row = next(r for r in doc["rows"] if r["amt_band"] == 9 and r["term_band"] == 21 and r["grade"] == 5)
    assert row["cell_id"] == price(50_000.0, 36, 5)["rate_cell_id"]
    assert row["rate"] == 11.51


@pytest.mark.parametrize("amount,term,band", [(1_000.0, 3, "#0|0"), (900_000.0, 120, "#95|54")])
def test_values_outside_the_product_range_are_priced_from_the_edge_band(price, amount, term, band):
    assert band in price(amount, term, 5)["rate_cell_id"]


def test_worse_grades_price_higher_all_else_equal(price):
    rates = [price(50_000.0, 36, g)["nominal_annual_rate"] for g in range(1, 13)]
    assert rates == sorted(rates) and len(set(rates)) == 12


def test_longer_terms_and_larger_amounts_price_slightly_higher(price):
    assert price(50_000.0, 60, 5)["nominal_annual_rate"] > price(50_000.0, 36, 5)["nominal_annual_rate"]
    assert price(300_000.0, 36, 5)["nominal_annual_rate"] > price(50_000.0, 36, 5)["nominal_annual_rate"]


def test_the_card_holds_percentages_and_pricing_uses_fractions():
    assert rate_card_rate(16.04) == 0.1604


@pytest.mark.parametrize("amount,term,flagged", [
    (50_000.0, 36.0, False), (2_000.0, 6.0, False), (500_000.0, 84.0, False),
    (1_000_000.0, 36.0, True), (1_999.0, 36.0, True), (50_000.0, 200.0, True), (50_000.0, 5.0, True)])
def test_out_of_range_flags_amounts_and_terms_outside_the_product(amount, term, flagged):
    assert out_of_range(amount, term) is flagged


def test_diff_cards_reports_repriced_added_and_removed_cells_in_key_order(doc):
    old = doc["rows"][:4]
    new = [dict(old[0], rate=old[0]["rate"] + 0.25), old[1], old[2], {**old[3], "amt_lo": 1.0}]
    assert [(c["change"], c["key"][2]) for c in diff_cards(old, new)] == [
        ("repriced", 1), ("removed", 4), ("added", 4)]
    assert diff_cards(old, new)[0]["delta"] == 0.25


def test_an_unchanged_card_has_an_empty_diff(doc):
    assert diff_cards(doc["rows"], doc["rows"]) == []
