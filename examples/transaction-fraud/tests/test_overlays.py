from datetime import date

import pytest
from decikit import Engine, flow

from fraud_interdiction.vocab import Action


@pytest.fixture(scope="module")
def multiplier(documents):
    executable = Engine().bind(flow(documents["amount_overlays"], name="overlays"))

    def run(on, enabled=True):
        out = executable.run({"decision_date": on, "adjustment_stack_enabled": enabled})
        return (out["mule_scam_amount_multiplier"], out["mule_scam_amount_multiplier_unadjusted"],
                out["adjustment_set_id"], out["applied_adjustment_ids"])

    return run


def test_no_overlay_in_force_leaves_the_multiplier_at_one(multiplier):
    assert multiplier(date(2025, 6, 1)) == (1.0, 1.0, "ADJS-2026.01", [])


def test_the_festive_multiplier_tightens_mule_amounts_by_thirty_percent(multiplier):
    assert multiplier(date(2025, 12, 15)) == (0.7, 1.0, "ADJS-2026.01", ["ADJ-2026-002"])


def test_the_sensitivity_dial_is_its_inverse_during_the_attack_window(multiplier):
    assert multiplier(date(2026, 3, 12))[0::3] == (0.7143, ["ADJ-2026-001"])


def test_an_overlay_lapses_on_its_effective_to_date(multiplier):
    assert multiplier(date(2026, 1, 14))[3] == ["ADJ-2026-002"]
    assert multiplier(date(2026, 1, 15))[3] == []


def test_the_stack_off_switch_returns_the_unadjusted_value_through_the_same_code(multiplier):
    assert multiplier(date(2025, 12, 15), enabled=False) == (1.0, 1.0, "ADJS-2026.01", [])


@pytest.mark.parametrize("family,channel,stack,expected", [
    ("AT", 4, True, Action.STEP_UP),      # in scope
    ("AT", 1, True, Action.MONITOR),      # another channel
    ("MS", 4, True, Action.MONITOR),      # another family
    ("AT", 4, False, Action.MONITOR),     # stack off
])
def test_account_takeover_monitor_escalates_to_step_up_only_on_the_web_channel(
        decide, rule, sample, family, channel, stack, expected):
    out = decide([rule(f"{family}-0001", action_code=Action.MONITOR)], sample, stack_enabled=stack,
                 channel_code=channel, decision_date=date(2026, 3, 1))
    assert out["action_code"] == expected


def test_escalation_never_touches_a_more_severe_action(decide, rule, sample):
    out = decide([rule("AT-0001", action_code=Action.DECLINE)], sample, channel_code=4, decision_date=date(2026, 3, 1))
    assert out["action_code"] == Action.DECLINE


def test_escalation_lapses_with_its_window(decide, rule, sample):
    out = decide([rule("AT-0001", action_code=Action.MONITOR)], sample, channel_code=4, decision_date=date(2027, 1, 1))
    assert out["action_code"] == Action.MONITOR
