"""Each product prices from its own card under its own policy; the search does not know which product it is asking."""
import pytest
from credit_core.instalment import instalment_before_fees


def product_scenarios(out, code):
    return [s for s in out["scenarios"] if s["product_code"] == code and "offered_amount" in s]


def test_the_consolidation_loan_lends_the_settlements_their_buffer_and_the_new_money_and_capitalises_its_fee(assess):
    for s in product_scenarios(assess(), 11):
        assert s["offered_amount"] == pytest.approx(s["settlement_total"] + s["settlement_buffer"] + s["new_money"], abs=0.01)
        financed = s["offered_amount"] + s["fees"]["initiation_fee"]
        monthly = (instalment_before_fees(financed, s["term_months"], s["nominal_annual_rate"])
                   + s["fees"]["monthly_service_fee"] + s["fees"]["credit_life_premium"])
        assert s["instalment"] == pytest.approx(monthly, abs=0.01)


def test_new_money_is_the_request_capped_at_a_quarter_of_the_settlement_and_r50_000(assess, sample, account):
    out = assess(requested_amount=500_000.0)
    for s in product_scenarios(out, 11):
        assert s["new_money"] == pytest.approx(min(500_000.0, 0.25 * s["settlement_total"], 50_000.0), abs=0.01)
    assert all(s["new_money"] == 0.0 for s in product_scenarios(out, 20))


def test_the_rate_margin_is_smaller_the_more_of_the_advance_pays_external_creditors(assess):
    by_external = sorted((s["external_proportion"], s["fees"]["margin_bps"]) for s in product_scenarios(assess(), 11)
                         if s["term_months"] == 60)
    margins = [m for _, m in by_external]
    assert margins[0] > margins[-1]


def test_the_card_tests_affordability_against_a_stressed_payment_not_its_promotional_one(assess):
    card = [s for s in product_scenarios(assess(), 20) if s["promotional_months"] == 24][0]
    assert card["tested_payment"] > card["instalment"] * 1.2
    # the reversion rate on the whole limit, over 36 months
    r = card["fees"]["reversion_rate"] / 12
    assert card["tested_payment"] == pytest.approx(card["offered_amount"] * r / (1 - (1 + r) ** -36), abs=0.01)


def test_the_card_total_cost_uses_the_promotional_rate_only_for_the_promotional_period(assess):
    by_promotion = {s["promotional_months"]: s["total_cost_of_credit"] for s in product_scenarios(assess(), 20)
                    if s["settlement_set"] == [2, 4]}
    assert list(by_promotion) == [0, 6, 12, 18, 24]
    assert list(by_promotion.values()) == sorted(by_promotion.values(), reverse=True)  # a longer promotion never costs more
    assert by_promotion[0] > by_promotion[24]


def test_the_card_limit_keeps_the_transfer_within_80_percent_of_it(assess):
    for s in product_scenarios(assess(), 20):
        assert s["settlement_total"] + s["settlement_buffer"] <= 0.8 * s["offered_amount"] + 1e-9


def test_the_card_is_unavailable_when_more_than_one_revolving_account_is_in_arrears(assess, sample, account):
    rest = [a for a in sample["accounts"] if a["account_ref"] not in (4, 5)]
    out = assess(accounts=[account(4, months_in_arrears=1), account(5, months_in_arrears=2), *rest])
    assert out["search_record"]["products_unavailable"] == {"20": "more than one revolving account in arrears"}
    assert not [s for s in out["scenarios"] if s["product_code"] == 20]


def test_the_card_is_unavailable_within_twelve_months_of_a_balance_transfer(assess):
    recent = assess(last_balance_transfer_date="2026-03-01")["search_record"]["products_unavailable"]
    assert recent == {"20": "a balance transfer within the last 12 months"}
    assert assess(last_balance_transfer_date="2025-06-01")["search_record"]["products_unavailable"] == {}


def test_an_advance_below_the_product_minimum_is_rejected_by_the_product(assess, params, with_params):
    out = assess(params_document=with_params(params, "flex_loan_policy", min_amount=250_000.0))
    too_small = [s for s in out["scenarios"] if s["product_code"] == 11
                 and s["settlement_total"] + s["settlement_buffer"] + s["new_money"] < 250_000.0]
    assert too_small and all(s["rejection_reason_codes"] == [1103] and s["unpriced_reason"] == "amount_outside_product_limits"
                             for s in too_small)


def test_a_product_team_changes_its_own_limits_without_touching_the_other_products(assess, params, with_params):
    base = assess()
    tighter = assess(params_document=with_params(params, "everyday_card_policy", max_limit=20_000.0))
    flex = [s for s in base["scenarios"] if s["product_code"] == 11]
    assert flex == [s for s in tighter["scenarios"] if s["product_code"] == 11]
    cards = [s for s in tighter["scenarios"] if s["product_code"] == 20]
    assert any(s.get("unpriced_reason") == "transfer_above_utilisation_cap" for s in cards)
    assert not any("unpriced_reason" in s for s in base["scenarios"] if s["product_code"] == 20)
