from datetime import date

from fraud_interdiction import overlays, vocab


def test_no_overlay_in_force_gives_unadjusted_value():
    adjusted, unadjusted, _, applied = overlays.effective_threshold_multiplier(date(2025, 6, 1))
    assert adjusted == unadjusted == 1.0
    assert applied == []


def test_festive_threshold_multiplier_alone():
    adjusted, unadjusted, set_id, applied = overlays.effective_threshold_multiplier(date(2025, 12, 15))
    assert unadjusted == 1.0
    assert adjusted == 0.7
    assert applied == ["ADJ-2026-002"]
    assert set_id == overlays.ADJUSTMENT_SET_ID


def test_sensitivity_dial_alone_during_the_attack_window():
    adjusted, unadjusted, _, applied = overlays.effective_threshold_multiplier(date(2026, 3, 12))
    assert unadjusted == 1.0
    assert adjusted == round(1.0 / 1.4, 4)
    assert applied == ["ADJ-2026-001"]


def test_stack_disabled_returns_unadjusted_through_the_same_call():
    """§7.6 / acceptance §10 item 8: same code path, one flag flipped."""
    adjusted, unadjusted, _, applied = overlays.effective_threshold_multiplier(
        date(2025, 12, 15), adjustment_stack_enabled=False)
    assert adjusted == unadjusted == 1.0
    assert applied == []


def test_overlay_exempt_rules_are_never_tagged_eligible():
    from fraud_interdiction.rules import build_rule_catalog
    catalog = build_rule_catalog()
    for rule in catalog.overlay_eligible():
        assert not rule.overlay_exempt


def test_at_web_escalation_scoped_to_web_channel_and_at_family():
    escalated, adj_id = overlays.at_web_action_escalation(vocab.FAMILY_ACCOUNT_TAKEOVER, 4, date(2026, 3, 1))
    assert escalated is True
    assert adj_id == "ADJ-2026-003"

    not_web, _ = overlays.at_web_action_escalation(vocab.FAMILY_ACCOUNT_TAKEOVER, 1, date(2026, 3, 1))
    assert not_web is False

    not_at, _ = overlays.at_web_action_escalation(vocab.FAMILY_MULE_SCAM, 4, date(2026, 3, 1))
    assert not_at is False
