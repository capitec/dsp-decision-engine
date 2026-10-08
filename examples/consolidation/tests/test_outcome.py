"""The decision, its conditions, and the comparison the client is shown."""
import pytest


def test_the_sample_client_is_approved_with_conditions_because_most_settlement_amounts_are_estimates(assess):
    out = assess()
    assert (out["outcome_code"], out["decline_reason_codes"], out["primary_reason_code"]) == (2, [], None)
    assert "obtain a settlement quotation for account 6" in out["conditions"]
    assert not any("account 3" in c and "quotation" in c for c in out["conditions"])  # account 3's quotation is in hand


def test_a_set_of_accounts_whose_amounts_are_all_exact_is_approved_outright(assess, sample, account):
    accounts = [account(1), account(3), account(6, quotation_reference="Q-102-1", quotation_amount=55_400.0,
                                                 quotation_expiry_date="2026-11-10")]
    out = assess(accounts=accounts, requested_amount=None, client_nominated_settle=[1, 3, 6])
    assert out["chosen_offer"] and {s["amount_basis"] for s in out["chosen_offer"]["settlement_details"]} <= {"quoted", "derived-internal"}
    assert (out["outcome_code"], out["conditions"]) == (1, [])


def test_the_offer_is_valid_until_the_earliest_of_the_offer_validity_and_the_quotation_expiries(assess, sample, account):
    assert assess()["offer_valid_until"] == "2026-10-15"  # 21 days from the decision date; the quotation expires later
    soon = [account(3, quotation_expiry_date="2026-10-01"), *[a for a in sample["accounts"] if a["account_ref"] != 3]]
    assert assess(accounts=soon, client_nominated_settle=[3])["offer_valid_until"] == "2026-10-01"


def test_the_comparison_puts_before_and_after_side_by_side(assess):
    comparison = assess()["before_and_after"]
    assert comparison["before"]["accounts"] == 9 and comparison["after"]["accounts"] == 2
    assert comparison["before"]["providers"] > comparison["after"]["providers"]
    assert comparison["after"]["monthly_instalment"] < comparison["before"]["monthly_instalment"]
    assert comparison["after"]["weighted_average_rate"] < comparison["before"]["weighted_average_rate"]


def test_a_rise_in_total_cost_is_stated_in_rands_and_percent_with_the_reason(assess):
    comparison = assess()["before_and_after"]
    assert comparison["total_cost_change"] == 35_692.28 and comparison["total_cost_change_pct"] == 0.0677
    assert comparison["total_cost_note"] == ("you will pay R35,692 more in total (6.8%) because the new facility runs 72 months "
                                            "against 60 on the accounts it settles")


def test_the_execution_package_costs_every_settled_account_with_its_components(assess):
    out = assess()
    package = out["execution_package"]
    assert [p["account_ref"] for p in package] == out["chosen_offer"]["settlement_set"]
    assert sum(p["settlement_amount"] for p in package) == pytest.approx(out["chosen_offer"]["settlement_total"], abs=0.01)
    assert all("components" in p and p["assumed_settlement_date"] for p in package)


def test_the_conditions_name_the_cards_to_close_and_the_security_to_release(assess, sample, account):
    assert "close account 2 once settled" in assess()["conditions"]
    dear_vehicle = account(9, nominal_annual_rate=0.295, instalment=5_600.0, remaining_term_months=48.0, balance=120_000.0)
    rest = [a for a in sample["accounts"] if a["account_ref"] != 9]
    out = assess(accounts=[dear_vehicle, *rest], client_nominated_settle=[9], requested_amount=None)
    assert 9 in out["chosen_offer"]["settlement_set"]
    assert "release the security on account 9 (10 working days)" in out["conditions"]


def test_no_viable_scenario_is_a_decline_with_its_own_reason(assess):
    out = assess(consolidations_last_24_months=2)
    assert (out["outcome_code"], out["decline_reason_codes"], out["chosen_offer"]) == (4, [2008], {})
    assert out["before_and_after"] == {} and out["execution_package"] == []


def test_a_client_whose_affordability_cannot_be_established_is_referred_not_declined(assess):
    out = assess(bureau_as_of_date="2026-07-01")  # a bureau view older than the assessment's window
    assert (out["outcome_code"], out["decline_reason_codes"]) == (3, [2010])


def test_reasons_come_most_severe_first_with_one_primary(assess):
    out = assess(client_under_administration=True, bureau_unobtainable=True)
    assert out["decline_reason_codes"] == [2002, 2004] and out["primary_reason_code"] == 2002
