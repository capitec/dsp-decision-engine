"""What it costs to close each account: never just the balance."""
import pytest

from consolidation.settlement import settlement_buffer


def by_ref(out):
    return {s["account_ref"]: s for s in out["settlements"]}


def test_every_settleable_account_is_costed_and_no_other(assess):
    out = assess()
    assert sorted(by_ref(out)) == out["settleable_account_refs"]


def test_an_internal_amount_is_the_balance_plus_interest_to_the_assumed_settlement_date(assess):
    one = by_ref(assess())[1]
    # R45 000 at 29.5% a year for the 2-day disbursement lag
    assert one["components"]["accrued_interest"] == pytest.approx(45_000 * 0.295 / 365 * 2, abs=0.01)
    assert one["settlement_amount"] == pytest.approx(45_000 + one["components"]["accrued_interest"], abs=0.01)
    assert (one["amount_basis"], one["assumed_settlement_date"], one["estimation_tolerance"]) == (
        "derived-internal", "2026-09-26", 0.0)


def test_a_held_quotation_is_the_providers_firm_figure_and_carries_its_expiry(assess):
    three = by_ref(assess())[3]
    assert (three["settlement_amount"], three["amount_basis"], three["quotation_reference"], three["quotation_expiry_date"]) == (
        38_900.0, "quoted", "Q-105-8891", "2026-11-01")
    assert three["estimation_tolerance"] == 0.0


def test_an_account_without_a_quotation_is_an_estimate_with_a_tolerance(assess):
    six = by_ref(assess())[6]
    assert six["amount_basis"] == "estimated"
    assert six["estimation_tolerance"] == pytest.approx(six["settlement_amount"] * 0.03, abs=0.01)


def test_the_assumed_settlement_date_includes_the_providers_quotation_turnaround(assess):
    settled = by_ref(assess())
    assert settled[5]["assumed_settlement_date"] == "2026-09-28"  # provider 101 is on the exchange: 2 days + 2 lag
    assert settled[7]["assumed_settlement_date"] == "2026-10-06"  # provider 201 still posts: 10 days + 2 lag


def test_every_amount_is_the_sum_of_its_named_components(assess):
    for s in by_ref(assess()).values():
        if s["amount_basis"] != "quoted":
            assert s["settlement_amount"] == pytest.approx(sum(s["components"].values()), abs=0.01)


def test_a_secured_account_adds_the_security_release_cost(assess):
    assert by_ref(assess())[9]["components"]["security_release_cost"] == 500.0


def test_an_early_settlement_charge_applies_only_from_the_large_agreement_threshold(assess, sample, account):
    small = by_ref(assess())[9]["components"]["early_settlement_charge"]
    big = account(9, balance=300_000.0)
    rest = [a for a in sample["accounts"] if a["account_ref"] != 9]
    charged = by_ref(assess(accounts=[big, *rest]))[9]["components"]["early_settlement_charge"]
    assert small == 0.0
    assert charged == pytest.approx(300_000 * 0.115 / 12 * 1, abs=0.01)  # provider 103's rule: one month of interest


def test_unearned_fees_are_refunded_and_unpaid_fees_are_added(assess, sample, account):
    rest = [a for a in sample["accounts"] if a["account_ref"] != 6]
    six = by_ref(assess(accounts=[account(6, fees_unpaid=120.0, unearned_rebate=69.0), *rest]))[6]
    assert (six["components"]["fees_unpaid"], six["components"]["rebate"]) == (120.0, -69.0)
    assert six["settlement_amount"] == pytest.approx(55_000 + six["components"]["accrued_interest"] + 120.0 - 69.0, abs=0.01)


def test_the_settlement_buffer_is_a_share_of_the_total_capped(assess):
    assert settlement_buffer(100_000.0, 0.015, 2_500.0) == 1_500.0
    assert settlement_buffer(400_000.0, 0.015, 2_500.0) == 2_500.0
    assert assess()["settlement_buffer_policy"] == {"buffer_pct": 0.015, "buffer_cap": 2_500.0}


def test_the_buffer_is_added_to_each_scenarios_advance(assess):
    scenario = next(s for s in assess()["scenarios"] if s["product_code"] == 11)
    assert scenario["settlement_buffer"] == settlement_buffer(scenario["settlement_total"], 0.015, 2_500.0)
