"""The cap waterfall: which rule set each ceiling, the chain of values, and the one rule that may raise."""
from loan_granting.waterfall import (
    AMOUNT, BOUND, COINCIDENT, DECLINED, DEFAULT_BANDS, DID_NOT_BIND, NOT_APPLICABLE, RAISED, REGISTER, Effect, Rule,
    run_waterfall,
)

CLIENT = {
    "risk_grade": 7, "channel_code": 2, "employment_type_code": 1.0, "internal_tenure_months": 41.0, "months_employed": 41.0,
    "worst_arrears_months": 2.0, "bureau_enquiry_count_60d": 3, "employer_on_watchlist": 0.0,
    "group_exposure_limit": 180_000.0, "internal_exposure_total": 104_000.0, "campaign_id": 4471,
    "campaign_is_preapproved": True, "appetite_max_amount": 150_000.0, "appetite_max_term": 72.0,
}


def status(out, rule_id):
    return next(r["status"] for r in out["waterfall_rules"] if r["rule_id"] == rule_id)


def test_the_amount_chain_records_every_rule_that_moved_it_in_order():
    out = run_waterfall(CLIENT)
    assert [(c["rule_id"], c["value"]) for c in out["amount_cap_chain"]] == [
        ("seed", 500_000.0), ("CAP-0100", 150_000.0), ("CAP-0210", 120_000.0), ("CAP-0361", 76_000.0),
        ("CAP-0420", 95_000.0)]
    assert (out["amount_cap"], out["amount_cap_binding_rule_id"]) == (95_000.0, "CAP-0420")


def test_the_term_and_grade_ceilings_each_name_the_rule_that_set_them():
    out = run_waterfall(CLIENT)
    assert (out["term_cap"], out["term_cap_binding_rule_id"]) == (72.0, "CAP-0118")
    assert (out["worst_acceptable_grade"], out["worst_acceptable_grade_binding_rule_id"]) == (9.0, "CAP-0210")
    assert [c["rule_id"] for c in out["worst_acceptable_grade_chain"]] == ["seed", "CAP-0210"]


def test_a_rule_that_does_not_apply_is_told_apart_from_one_that_applied_and_did_not_bind():
    out = run_waterfall(CLIENT)
    assert status(out, "CAP-0140") == NOT_APPLICABLE  # not a new-to-bank client
    assert status(out, "CAP-0305") == NOT_APPLICABLE  # employer not on the watchlist
    assert status(out, "CAP-0175") == DID_NOT_BIND  # 41 months employed is past every tenure band
    assert status(out, "CAP-0210") == BOUND and status(out, "CAP-0361") == BOUND
    assert status(out, "CAP-0420") == RAISED


def test_every_rule_in_the_register_gets_one_entry():
    out = run_waterfall(CLIENT)
    assert [r["rule_id"] for r in out["waterfall_rules"]] == [rule.rule_id for rule in REGISTER]


def test_each_entry_names_the_ceilings_it_changed_with_the_values_before_and_after():
    entry = next(r for r in run_waterfall(CLIENT)["waterfall_rules"] if r["rule_id"] == "CAP-0210")
    assert entry["changes"] == {"amount": [150_000.0, 120_000.0], "grade": [12.0, 9.0]}


def test_the_uplift_is_capped_by_its_authority_ceiling():
    out = run_waterfall({**CLIENT, "appetite_max_amount": 220_000.0, "worst_arrears_months": 0.0,
                         "internal_exposure_total": 0.0, "group_exposure_limit": 1_000_000.0})
    assert out["amount_cap"] == 250_000.0  # 220 000 plus 25% would be 275 000
    assert out["uplift_authority_reference"] == "CC-2026-14"


def test_without_a_pre_approved_campaign_nothing_is_raised():
    out = run_waterfall({**CLIENT, "campaign_is_preapproved": False})
    assert (out["amount_cap"], out["uplift_authority_reference"]) == (76_000.0, None)
    assert status(out, "CAP-0420") == NOT_APPLICABLE


def test_the_uplift_never_lowers_a_ceiling_that_is_already_above_its_authority():
    out = run_waterfall({**CLIENT, "appetite_max_amount": 450_000.0, "worst_arrears_months": 0.0,
                         "internal_exposure_total": 0.0, "group_exposure_limit": 1_000_000.0})
    assert out["amount_cap"] == 450_000.0
    assert (status(out, "CAP-0420"), out["uplift_authority_reference"]) == (DID_NOT_BIND, None)


def test_the_uplift_cannot_pass_a_cap_set_by_a_regulatory_rule_and_says_which_one_held_it_back():
    by_id = {rule.rule_id: rule for rule in REGISTER}
    regulatory = Rule("CAP-0050", "Financial Crime", "regulatory", lambda app, ctx: Effect(amount=80_000.0))
    register = (by_id["CAP-0100"], by_id["CAP-0210"], regulatory, by_id["CAP-0361"], by_id["CAP-0420"])
    out = run_waterfall(CLIENT, register=register)
    # headroom took the cap to 76 000; the uplift was authorised to 95 000 but may only reach the regulatory 80 000
    assert out["amount_cap"] == 80_000.0
    assert out["uplift_restrained_by_rule_id"] == "CAP-0050"
    assert status(out, "CAP-0420") == RAISED


def test_two_rules_reducing_to_the_same_value_leave_the_earlier_as_binder_and_the_later_coincident():
    same = Rule("CAP-0900", "Credit Risk Policy", "policy", lambda app, ctx: Effect(amount=150_000.0))
    out = run_waterfall(CLIENT, register=(REGISTER[1], same))
    assert out["amount_cap_binding_rule_id"] == "CAP-0100"
    assert status(out, "CAP-0900") == COINCIDENT


def test_a_decline_rule_declines_and_gives_its_reason():
    out = run_waterfall({**CLIENT, "months_employed": 2.0, "bureau_enquiry_count_60d": 12})
    assert out["waterfall_decline_reason_codes"] == [1181, 1182]
    assert status(out, "CAP-0175") == DECLINED


def test_a_closed_channel_declines():
    assert run_waterfall({**CLIENT, "channel_code": 6})["waterfall_decline_reason_codes"] == [1180]


def test_no_appetite_for_the_grade_declines():
    out = run_waterfall({**CLIENT, "appetite_max_amount": 0.0})
    assert 1183 in out["waterfall_decline_reason_codes"] and status(out, "CAP-0100") == DECLINED


def test_a_pensioner_is_capped_at_80_000_and_a_new_to_bank_client_at_60_000():
    pensioner = run_waterfall({**CLIENT, "employment_type_code": 4.0, "campaign_is_preapproved": False})
    assert pensioner["amount_cap"] <= 80_000.0 and status(pensioner, "CAP-0150") == BOUND
    new = run_waterfall({**CLIENT, "internal_tenure_months": float("nan"), "campaign_is_preapproved": False})
    assert new["amount_cap"] <= 60_000.0 and status(new, "CAP-0140") == BOUND


def test_a_partner_application_is_capped_at_100_000_in_the_chain():
    out = run_waterfall({**CLIENT, "channel_code": 5, "internal_exposure_total": 22_000.0})
    assert [(c["rule_id"], c["value"]) for c in out["amount_cap_chain"]][-2:] == [("CAP-0330", 100_000.0), ("CAP-0420", 125_000.0)]


def test_a_broker_application_is_declined_and_still_capped_in_term():
    out = run_waterfall({**CLIENT, "channel_code": 6})
    assert out["waterfall_decline_reason_codes"] == [1180]
    assert (out["term_cap"], out["term_cap_binding_rule_id"]) == (60.0, "CAP-0330")


def test_changing_a_band_changes_the_ceiling_without_touching_the_register():
    tighter = [{**band, "amount": 100_000.0} if band["rule"] == 210 and band["lo"] == 2 else band for band in DEFAULT_BANDS]
    out = run_waterfall({**CLIENT, "campaign_is_preapproved": False}, bands=tighter)
    assert out["amount_cap"] == 76_000.0 and next(c for c in out["amount_cap_chain"] if c["rule_id"] == "CAP-0210")["value"] == 100_000.0


def test_a_grade_worse_than_the_worst_acceptable_grade_declines_and_a_better_one_passes():
    assert run_waterfall(CLIENT)["waterfall_decline_reason_codes"] == []  # grade 7 against a worst acceptable 9
    assert run_waterfall({**CLIENT, "risk_grade": 10})["waterfall_decline_reason_codes"] == [1183]
