import pytest

from fraud_interdiction.vocab import Action


def test_no_firing_allows_with_no_source_and_no_reason(decide, rule, sample):
    out = decide([rule("MS-0001", condition={"op": ">", "feature": "amount", "threshold": 1e9})], sample)
    assert (out["action_code"], out["action_source_rule_id"], out["decline_reason_codes"]) == (10, "", [])
    assert out["primary_reason_code"] is None


def test_the_most_severe_action_wins(decide, rule, sample):
    out = decide([rule("MS-0001", action_code=20), rule("AT-0001", action_code=50, severity=1)], sample)
    assert (out["action_code"], out["action_source_rule_id"]) == (50, "AT-0001")


@pytest.mark.parametrize("ids,expected", [
    ({"MS-0001": {"severity": 5}, "MS-0002": {"severity": 4}}, "MS-0001"),       # higher severity first
    ({"MS-0001": {"priority": 90}, "MS-0002": {"priority": 10}}, "MS-0002"),     # then lower priority number
    ({"CF-0001": {}, "MS-0001": {}, "FP-0001": {}}, "MS-0001"),                 # then family seniority: AA MS AT CF FP
    ({"MS-0002": {}, "MS-0001": {}}, "MS-0001"),                                 # last, the lower rule id
])
def test_rules_asking_for_the_same_action_are_ranked_severity_priority_family_then_id(decide, rule, sample, ids, expected):
    rules = [rule(rule_id, **fields) for rule_id, fields in ids.items()]
    assert decide(rules, sample)["action_source_rule_id"] == expected


def test_a_lower_severity_loses_whatever_its_priority(decide, rule, sample):
    rules = [rule("MS-0001", severity=3, priority=0), rule("MS-0002", severity=4, priority=999)]
    assert decide(rules, sample)["action_source_rule_id"] == "MS-0002"


def test_the_reason_set_holds_every_firing_ranked_and_the_primary_reason_is_the_winners(decide, rule, sample):
    rules = [rule("MS-0001", action_code=20, reason_code=3001), rule("AT-0001", action_code=50, reason_code=2001)]
    out = decide(rules, sample)
    assert out["decline_reason_codes"] == [3001, 2001]  # mule and scam ranks above account takeover
    assert out["primary_reason_code"] == 2001           # but the account-takeover rule decided
    assert out["reason_registry_version"] == "fraud-reasons-2026.01"


def test_a_reason_code_missing_from_the_registry_fails_loudly(decide, rule, sample):
    with pytest.raises(LookupError, match="not in registry"):
        decide([rule("MS-0001", reason_code=9999)], sample)


def test_a_critical_rule_is_binding_even_when_a_more_severe_ordinary_rule_fires(decide, rule, sample):
    rules = [rule("MS-0001", action_code=50, severity=5), rule("AA-0001", action_code=40, critical=True)]
    out = decide(rules, sample)
    assert (out["action_code"], out["action_source_rule_id"], out["governance_exception"]) == (40, "AA-0001", False)


def test_a_critical_allow_overrides_ordinary_firings(decide, rule, sample):
    rules = [rule("CF-0001", action_code=10, critical=True), rule("CF-0002", action_code=40)]
    out = decide(rules, sample)
    assert (out["action_code"], out["action_source_rule_id"]) == (10, "CF-0001")
    assert out["fired_rule_ids"] == ["CF-0001", "CF-0002"]  # still recorded


def test_two_critical_rules_that_disagree_are_a_governance_exception_and_the_more_severe_wins(decide, rule, sample):
    rules = [rule("AA-0001", action_code=50, critical=True), rule("MS-0001", action_code=40, critical=True)]
    out = decide(rules, sample)
    assert (out["action_code"], out["action_source_rule_id"], out["governance_exception"]) == (50, "AA-0001", True)


def test_two_critical_rules_that_agree_are_not(decide, rule, sample):
    rules = [rule("AA-0001", action_code=50, critical=True), rule("MS-0001", action_code=50, critical=True)]
    assert decide(rules, sample)["governance_exception"] is False


@pytest.mark.parametrize("channel,stack,expected", [(4, True, 30), (1, True, 20), (4, False, 20)])
def test_an_account_takeover_monitor_becomes_a_step_up_on_the_web_channel_under_the_overlay(
        decide, rule, sample, channel, stack, expected):
    out = decide([rule("AT-0001", action_code=20)], sample, stack_enabled=stack, channel_code=channel,
                 decision_date=__import__("datetime").date(2026, 3, 1))
    assert out["action_code"] == expected


def test_the_counterfactual_is_the_action_without_the_escalation(decide, rule, sample):
    out = decide([rule("AT-0001", action_code=20)], sample, channel_code=4)
    assert (out["action_code"], out["counterfactual_action_code"]) == (30, 20)


def test_a_hard_block_is_a_floor_even_when_no_rule_fires(decide, rule, sample):
    out = decide([], sample, account_frozen=True)
    assert (out["action_code"], out["action_source_rule_id"]) == (50, "HARD_BLOCK:account_frozen")
    assert out["decline_reason_codes"] == [6002] and out["primary_reason_code"] == 6002
    assert out["counterfactual_action_code"] == 50


def test_a_hard_block_still_evaluates_and_records_every_rule(decide, rule, sample):
    out = decide([rule("MS-0001", action_code=20)], sample, account_frozen=True)
    assert out["fired_rule_ids"] == ["MS-0001"]
    assert out["decline_reason_codes"] == [3001, 6002] or out["decline_reason_codes"] == [6002, 3001]


def test_a_rule_asking_for_more_than_the_hard_block_wins(decide, rule, sample):
    out = decide([rule("MS-0001", action_code=60)], sample, account_frozen=True)
    assert (out["action_code"], out["action_source_rule_id"], out["primary_reason_code"]) == (60, "MS-0001", 3001)


def test_when_two_gates_hold_the_more_severe_names_the_source_and_the_primary_reason(decide, rule, sample):
    out = decide([], sample, account_frozen=True, sanctions_list_match=True)
    assert out["action_code"] == Action.BLOCK_CHANNEL
    assert out["action_source_rule_id"] == "HARD_BLOCK:sanctions,account_frozen"
    assert out["primary_reason_code"] == 6001
    assert sorted(out["decline_reason_codes"]) == [6001, 6002]
