"""The fourteen interventions: all evaluated, each recorded with actual and threshold, the anti-harm ceiling overlaid."""
import pytest
from conftest import with_params


def priced(out):
    return [s for s in out["scenarios"] if s["interventions"]]


def by_number(scenario):
    return {i["intervention"]: i for i in scenario["interventions"]}


def off(params):
    return {**params, "shared": {"overlay_stack_enabled": False}}


def test_every_priced_scenario_carries_all_fourteen_interventions(assess):
    scenarios = priced(assess())
    assert scenarios and all(sorted(by_number(s)) == list(range(1, 15)) for s in scenarios)


def test_not_applicable_is_distinct_from_passed(assess):
    flex, card = (next(s for s in priced(assess()) if s["product_code"] == p) for p in (11, 20))
    assert by_number(flex)[10]["status"] in ("passed", "failed") and by_number(card)[10]["status"] == "not_applicable"
    assert by_number(card)[10]["threshold"] is None
    assert by_number(flex)[2]["status"] == by_number(flex)[7]["status"] == "not_applicable"  # enforced earlier


def test_every_violation_is_recorded_not_only_the_first(assess):
    several = [s for s in assess()["scenarios"] if len(s["rejection_reason_codes"]) > 1]
    assert several
    for s in several:
        failed = sorted(1000 + n for n, i in by_number(s).items() if i["status"] == "failed")
        assert failed == [c for c in s["rejection_reason_codes"] if c < 1100]


def test_a_rejection_carries_the_actual_and_threshold_values(assess):
    s = next(s for s in priced(assess()) if 1003 in s["rejection_reason_codes"])
    relief = by_number(s)[3]
    assert relief["status"] == "failed" and relief["actual"] < relief["threshold"] == 0.10


def test_a_violated_intervention_removes_the_scenario_from_selection(assess):
    out = assess()
    rejected = {s["scenario_id"] for s in out["scenarios"] if not s["viable"]}
    chosen = {out["chosen_offer"]["scenario_id"], *(r["scenario_id"] for r in out["runners_up"])}
    assert rejected and not chosen & rejected
    assert all(s["viable"] == (s["rejection_reason_codes"] == []) for s in out["scenarios"])


def test_the_anti_harm_overlay_names_itself_and_the_base_value_it_replaced_on_product_11(assess):
    record = by_number(next(s for s in priced(assess()) if s["product_code"] == 11))[4]
    assert (record["overlay_id"], record["base_threshold"]) == ("ADJ-CON-INT-2026-001", 0.15)
    assert record["threshold"] == pytest.approx(0.12)


def test_the_anti_harm_overlay_does_not_touch_the_card(assess):
    for s in priced(assess()):
        if s["product_code"] == 20:
            assert by_number(s)[4]["overlay_id"] is None and by_number(s)[4]["threshold"] == 0.15


def test_the_overlay_rejects_scenarios_the_base_threshold_would_admit(assess, params):
    strict = with_params(params, "flex_loan_policy", anti_harm=0.11)  # a base ceiling the overlay tightens to 8.8%
    with_overlay, without = assess(params_document=strict), assess(params_document=off(strict))
    newly_rejected = [s for s, base in zip(with_overlay["scenarios"], without["scenarios"])
                      if base["viable"] and not s["viable"] and s["rejection_reason_codes"] == [1004]]
    assert newly_rejected
    for s in newly_rejected:
        record = by_number(s)[4]
        assert record["overlay_id"] == "ADJ-CON-INT-2026-001"
        assert record["threshold"] < record["actual"] <= record["base_threshold"] == 0.11


def test_with_the_stack_off_the_base_threshold_applies_and_no_overlay_is_named(assess, params):
    out = assess(params_document=off(params))
    assert out["search_record"]["overlay_stack_enabled"] is False
    for s in priced(out):
        assert by_number(s)[4]["overlay_id"] is None and by_number(s)[4]["threshold"] == 0.15


def test_the_anti_harm_threshold_is_a_product_param(assess, params):
    strict = assess(params_document=with_params(params, "flex_loan_policy", anti_harm=0.01))
    assert not any(s["viable"] for s in strict["scenarios"] if s["product_code"] == 11)
    assert all(by_number(s)[4]["base_threshold"] == 0.01 for s in priced(strict) if s["product_code"] == 11)


def test_the_maximum_consolidations_in_24_months_is_enforced(assess):
    out = assess(consolidations_last_24_months=2)
    assert out["scenarios"] and not any(s["viable"] for s in out["scenarios"])
    assert all(1013 in s["rejection_reason_codes"] for s in priced(out))
    assert out["outcome_code"] == 4 and out["decline_reason_codes"] == [2008]
