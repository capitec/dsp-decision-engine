import pytest
from pydantic import ValidationError

from fraud_interdiction.rules import RuleSet

MS_0208 = {
    "condition": {"type": "composite", "op": "and", "conditions": [
        {"op": "is_true", "feature": "beneficiary_first_payment"},
        {"op": "<", "feature": "beneficiary_age_hours", "threshold": 2},
        {"op": ">", "feature": {"type": "computed", "expression": "amount - 8000.0 * mule_scam_amount_multiplier"},
         "threshold": 0.0},
        {"op": ">=", "feature": "beneficiary_bank_mule_rate_band", "threshold": 4},
        {"op": "<", "feature": "device_change_hours", "threshold": 72}]},
    "base_condition": {"type": "composite", "op": "and", "conditions": [
        {"op": "is_true", "feature": "beneficiary_first_payment"},
        {"op": "<", "feature": "beneficiary_age_hours", "threshold": 2},
        {"op": ">", "feature": "amount", "threshold": 8000.0},
        {"op": ">=", "feature": "beneficiary_bank_mule_rate_band", "threshold": 4},
        {"op": "<", "feature": "device_change_hours", "threshold": 72}]},
}


def test_a_rule_set_with_a_repeated_rule_id_is_invalid(rule):
    with pytest.raises(ValidationError, match="duplicate rule ids"):
        RuleSet.load({"type": "rule_set", "name": "r", "rules": [rule("MS-0001"), rule("MS-0001")]})


def test_an_overlay_exempt_rule_cannot_read_an_overlay_adjusted_threshold(rule):
    with pytest.raises(ValidationError, match="overlay-exempt"):
        rule_set = {"type": "rule_set", "name": "r", "rules": [rule("MS-0001", overlay_exempt=True, **MS_0208)]}
        RuleSet.load(rule_set)


def test_an_unknown_rule_field_is_invalid(rule):
    with pytest.raises(ValidationError):
        RuleSet.load({"type": "rule_set", "name": "r", "rules": [rule("MS-0001", colour="red")]})


def test_every_applicable_rule_that_fires_is_recorded_not_only_the_winner(decide, rule, sample):
    out = decide([rule("MS-0001"), rule("AT-0001"), rule("CF-0001", event_types=[111])], sample)
    assert out["fired_rule_ids"] == ["MS-0001", "AT-0001"]  # CF-0001 is a card rule: not applicable to a payment


def test_a_rule_that_does_not_hold_does_not_fire(decide, rule, sample):
    quiet = rule("MS-0001", condition={"op": ">", "feature": "amount", "threshold": 1e9})
    assert decide([quiet, rule("AT-0001")], sample)["fired_rule_ids"] == ["AT-0001"]


def test_a_rule_applies_from_its_effective_from_date_until_the_day_before_effective_to(decide, rule, sample):
    rules = [rule("MS-0001", effective_from="2026-01-10"), rule("MS-0002", effective_from="2026-01-11"),
             rule("MS-0003", effective_to="2026-01-10"), rule("MS-0004", effective_to="2026-01-11")]
    assert decide(rules, sample)["fired_rule_ids"] == ["MS-0001", "MS-0004"]  # decision date 2026-01-10


def test_a_rule_scoped_to_segments_needs_one_of_the_clients(decide, rule, sample):
    rules = [rule("MS-0001", segments=["new_to_bank", "vulnerable"]), rule("MS-0002", segments=["high_value"]),
             rule("MS-0003", segments=None)]
    assert decide(rules, sample)["fired_rule_ids"] == ["MS-0001", "MS-0003"]  # the client is new_to_bank
    assert decide(rules, sample, client_segments=[])["fired_rule_ids"] == ["MS-0003"]


def test_a_shadow_rule_is_recorded_apart_and_never_moves_the_action(decide, rule, sample):
    shadow = rule("AA-0001", status="shadow", severity=5, action_code=70)
    out = decide([rule("MS-0001", action_code=30), shadow], sample)
    assert out["shadow_fired_rule_ids"] == ["AA-0001"] and out["fired_rule_ids"] == ["MS-0001"]
    assert (out["action_code"], out["action_source_rule_id"]) == (30, "MS-0001")
    assert decide([shadow], sample)["action_code"] == 10


def test_a_rule_that_fires_only_under_the_overlay_says_so_and_the_counterfactual_leaves_it_out(decide, rule, sample):
    rules = [rule("MS-0208", action_code=40, **MS_0208), rule("AT-0001", action_code=20)]
    over_base = decide(rules, sample, amount=9200.0)  # above R8 000 and above the overlay's R5 600
    assert over_base["fired_on_overlay_ids"] == []
    assert over_base["counterfactual_fired_rule_ids"] == ["MS-0208", "AT-0001"]

    between = decide(rules, sample, amount=6000.0)  # festive x0.7: R5 600 < R6 000 <= R8 000
    assert between["fired_rule_ids"] == ["MS-0208", "AT-0001"]
    assert between["fired_on_overlay_ids"] == ["MS-0208"]
    assert between["counterfactual_fired_rule_ids"] == ["AT-0001"]
    assert (between["action_code"], between["counterfactual_action_code"]) == (40, 20)


def test_switching_the_overlay_stack_off_makes_the_same_amount_fire_at_the_authored_threshold(decide, rule, sample):
    rules = [rule("MS-0208", **MS_0208)]
    off = decide(rules, sample, stack_enabled=False, amount=6000.0)
    assert off["fired_rule_ids"] == [] and off["mule_scam_amount_multiplier"] == 1.0
    assert off["applied_adjustment_ids"] == []


def test_a_rule_with_a_stale_velocity_aggregate_follows_the_behaviour_it_declares(decide, rule, sample):
    velocity = {"velocity_feature": "velocity_count_1min",
                "condition": {"op": ">=", "feature": "velocity_count_1min", "threshold": 1.0}}
    rules = [rule("MS-0001", stale_absent="suppress", **velocity), rule("MS-0002", stale_absent="evaluate_false", **velocity),
             rule("MS-0003", stale_absent="last_known_good", **velocity)]
    fresh = decide(rules, sample)
    assert fresh["fired_rule_ids"] == ["MS-0001", "MS-0002", "MS-0003"]
    stale = decide(rules, sample, velocity_count_1min_state=1)
    assert stale["fired_rule_ids"] == ["MS-0003"]
    assert stale["unevaluable_rule_ids"] == ["MS-0001"]  # suppressed, so recorded as unevaluable; evaluate_false just says no


def test_restricted_mode_suspends_the_rules_that_read_velocity_and_keeps_the_rest(decide, rule, sample):
    rules = [rule("MS-0001", velocity_feature="velocity_count_1min"), rule("MS-0002")]
    out = decide(rules, sample, device_session_degraded=True, counterparty_degraded=True)
    assert out["degraded_mode_code"] == 2
    assert out["fired_rule_ids"] == ["MS-0002"] and out["unevaluable_rule_ids"] == []
