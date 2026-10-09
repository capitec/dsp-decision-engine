"""Policy caps: the rule chain, the overlay, and the campaign uplift."""
import pytest


def statuses(out: dict) -> dict:
    return {entry["rule_id"]: entry["status"] for entry in out["amount_cap_chain"]}


def test_the_sample_chain_names_the_rule_that_set_the_cap_and_those_that_did_nothing(decide):
    out = decide()
    assert statuses(out) == {
        "SEED": "seed", "CAP-0029": "not_applicable", "CAP-0104": "bound", "CAP-0131": "evaluated_not_bound",
        "CAP-0158": "not_applicable", "CAP-0176": "evaluated_not_bound", "CAP-0212": "evaluated_not_bound",
        "CAP-0212-decline": "not_applicable", "CAP-0248": "not_applicable", "CAP-0266": "not_applicable",
        "CAP-0301": "bound", "CAP-0310": "not_applicable", "CAP-0118": "not_applicable"}
    assert [(e["before"], e["after"]) for e in out["amount_cap_chain"] if e["rule_id"] == "CAP-0104"] == [(500_000.0, 180_000.0)]
    assert (out["amount_cap_binding_rule"], out["amount_cap_unadjusted"]) == ("CAP-0301", 112_000.0)  # group headroom


def test_the_channel_overlay_comes_off_the_cap_after_the_register_and_is_recorded_beside_it(decide):
    out = decide()
    assert (out["amount_cap"], out["amount_cap_adjustments_applied"]) == (100_750.0, ["ADJ-0087"])  # 112 000 x 0.90, down to the grid
    other = decide(channel_code=2)
    assert (other["amount_cap"], other["amount_cap_adjustments_applied"]) == (112_000.0, [])


def test_a_campaign_raises_the_cap_by_a_fifth_and_it_is_the_only_entry_that_can(decide):
    out = decide(campaign_id="7712")
    assert statuses(out)["CAP-0118"] == "raised"
    assert (out["amount_cap_unadjusted"], out["amount_cap"]) == (134_400.0, 120_750.0)
    assert out["amount_cap_binding_rule"] == "CAP-0118"


def test_a_campaign_cannot_raise_the_cap_past_its_authority_ceiling(decide):
    out = decide(campaign_id="7712", group_limit=600_000.0)  # headroom 452 000, but the grade caps it at R180 000
    assert out["amount_cap_unadjusted"] == 200_000.0


def test_recent_arrears_cut_the_amount_and_tighten_the_grade_cap(decide, sample):
    late = {**sample["bureau_accounts"][0], "months_in_arrears": 3}
    out = decide(bureau_accounts=[late], bureau_score=800.0)
    assert out["worst_arrears_months"] == 3 and out["worst_acceptable_grade"] == 9 and out["risk_grade"] == 8
    assert statuses(out)["CAP-0176"] == "bound" and out["amount_cap_unadjusted"] == 35_000.0


def test_a_grade_worse_than_the_arrears_cap_allows_is_declined_by_routing(decide, sample):
    late = {**sample["bureau_accounts"][0], "months_in_arrears": 3}
    out = decide(bureau_accounts=[late], bureau_score=700.0)
    assert out["risk_grade"] == 10 > out["worst_acceptable_grade"]
    assert out["product_routing_reasons"] == [1204] and out["outcome_code"] == 4


def test_a_closed_channel_declines_outright_and_no_offer_is_made(decide):
    out = decide(channel_open=False)
    assert (out["p09_declined"], out["p09_decline_reason"], out["amount_cap"]) == (True, 2011, 0.0)
    assert statuses(out)["CAP-0029"] == "declined"
    assert (out["outcome_code"], out["has_offer"], out["decline_reason_codes"]) == (4, False, [2011])


def test_twelve_enquiries_in_ninety_days_decline_where_seven_only_cap(decide):
    assert statuses(decide(enquiry_velocity_90d=7))["CAP-0212"] == "bound"
    out = decide(enquiry_velocity_90d=12)
    assert statuses(out)["CAP-0212-decline"] == "declined" and out["outcome_code"] == 4


def test_a_name_on_the_concentration_watchlist_caps_the_amount_at_sixty_thousand(decide):
    out = decide(on_concentration_watchlist=True)
    assert statuses(out)["CAP-0266"] == "bound" and out["amount_cap_unadjusted"] == 60_000.0


def test_a_new_to_bank_client_is_capped_at_seventy_thousand(decide):
    out = decide(is_existing_client=False, group_limit=None, bureau_score=800.0)
    assert statuses(out)["CAP-0158"] == "bound" and out["amount_cap_unadjusted"] == 70_000.0


def test_income_evidence_that_is_only_declared_caps_the_amount_at_eighty_thousand(decide):
    out = decide(payslip_income=None, declared_income=18_500.0, group_limit=None, bureau_score=800.0, channel_code=2)
    assert out["income_evidence_tier"] == 7 and statuses(out)["CAP-0310"] == "bound"
    assert out["amount_cap_unadjusted"] == 80_000.0


@pytest.mark.parametrize("grade, cap", [(1, 84), (4, 72), (7, 60), (9, 48), (11, 24)])
def test_the_term_cap_follows_the_grade(grade, cap):
    from retail_credit.caps import term_cap
    assert term_cap(grade, 84) == cap
