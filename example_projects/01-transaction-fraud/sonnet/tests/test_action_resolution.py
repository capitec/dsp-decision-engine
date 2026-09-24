from datetime import date

from fraud_interdiction import action_resolution, vocab
from fraud_interdiction.rules import RuleCatalog, RuleDefinition


def _rule(rule_id, family, severity, action_code, priority=100, critical=False, **kw) -> RuleDefinition:
    defaults = dict(
        rule_id=rule_id, rule_version=1, family=family, severity=severity, action_code=action_code,
        priority=priority, status=vocab.STATUS_LIVE, effective_from=date(2025, 1, 1), effective_to=None,
        event_types=(210,), segments=None, overlay_exempt=False, critical=critical, suppressible=False,
        reason_code=3001, queue=None, stale_absent_behavior=vocab.STALE_ABSENT_LAST_KNOWN, max_fire_rate_pct=5.0,
        condition={}, predicate_count=1, overlay_eligible=False, base_condition=None, references_velocity=False,
        velocity_feature=None, description_internal="t", description_client_safe="t",
    )
    defaults.update(kw)
    return RuleDefinition(**defaults)


def test_most_severe_action_wins():
    catalog = RuleCatalog([
        _rule("MS-0001", "MS", 3, vocab.ACTION_MONITOR),
        _rule("AT-0001", "AT", 4, vocab.ACTION_DECLINE),
    ])
    action, source, reasons, exc = action_resolution._resolve(
        ["MS-0001", "AT-0001"], catalog, channel_code=1, decision_date=date(2025, 6, 1),
        adjustment_stack_enabled=True, apply_escalation=False)
    assert action == vocab.ACTION_DECLINE
    assert source == "AT-0001"
    assert exc is False


def test_tie_break_severity_then_priority_then_family_then_rule_id():
    """§5.12 item 5: severity desc, priority asc, family precedence, rule_id asc."""
    catalog = RuleCatalog([
        _rule("CF-0001", "CF", 4, vocab.ACTION_HOLD_FOR_REVIEW, priority=50),
        _rule("MS-0001", "MS", 4, vocab.ACTION_HOLD_FOR_REVIEW, priority=50),  # MS outranks CF in family precedence
        _rule("MS-0002", "MS", 3, vocab.ACTION_HOLD_FOR_REVIEW, priority=10),  # lower severity loses regardless
    ])
    _, source, _, _ = action_resolution._resolve(
        ["CF-0001", "MS-0001", "MS-0002"], catalog, channel_code=1, decision_date=date(2025, 6, 1),
        adjustment_stack_enabled=True, apply_escalation=False)
    assert source == "MS-0001"


def test_critical_overrides_a_more_frequent_non_critical_action():
    catalog = RuleCatalog([
        _rule("MS-0001", "MS", 2, vocab.ACTION_MONITOR),
        _rule("AA-0001", "AA", 5, vocab.ACTION_DECLINE, critical=True),
    ])
    action, source, _, exc = action_resolution._resolve(
        ["MS-0001", "AA-0001"], catalog, channel_code=1, decision_date=date(2025, 6, 1),
        adjustment_stack_enabled=True, apply_escalation=False)
    assert action == vocab.ACTION_DECLINE
    assert source == "AA-0001"
    assert exc is False


def test_two_disagreeing_critical_rules_raise_a_governance_exception():
    """§5.12 item 3: the more severe wins, and the collision is recorded."""
    catalog = RuleCatalog([
        _rule("AA-0001", "AA", 5, vocab.ACTION_DECLINE, critical=True),
        _rule("MS-0001", "MS", 5, vocab.ACTION_HOLD_FOR_REVIEW, critical=True),
    ])
    action, source, _, exc = action_resolution._resolve(
        ["AA-0001", "MS-0001"], catalog, channel_code=1, decision_date=date(2025, 6, 1),
        adjustment_stack_enabled=True, apply_escalation=False)
    assert action == vocab.ACTION_DECLINE  # more severe of the two critical actions
    assert exc is True


def test_critical_allow_suppresses_non_critical_firings_of_its_family():
    catalog = RuleCatalog([
        _rule("CF-0001", "CF", 5, vocab.ACTION_ALLOW, critical=True),
        _rule("CF-0002", "CF", 3, vocab.ACTION_HOLD_FOR_REVIEW),  # would otherwise win
    ])
    action, source, _, _ = action_resolution._resolve(
        ["CF-0001", "CF-0002"], catalog, channel_code=1, decision_date=date(2025, 6, 1),
        adjustment_stack_enabled=True, apply_escalation=False)
    assert action == vocab.ACTION_ALLOW
    assert source == "CF-0001"


def test_no_firings_allows():
    catalog = RuleCatalog([_rule("MS-0001", "MS", 3, vocab.ACTION_MONITOR)])
    action, source, reasons, exc = action_resolution._resolve(
        [], catalog, channel_code=1, decision_date=date(2025, 6, 1), adjustment_stack_enabled=True,
        apply_escalation=False)
    assert action == vocab.ACTION_ALLOW
    assert source == ""
    assert reasons == []


def test_at_web_monitor_escalates_to_step_up_under_the_overlay():
    catalog = RuleCatalog([_rule("AT-0001", "AT", 2, vocab.ACTION_MONITOR)])
    escalated_action, _, _, _ = action_resolution._resolve(
        ["AT-0001"], catalog, channel_code=4, decision_date=date(2026, 3, 1), adjustment_stack_enabled=True,
        apply_escalation=True)
    not_escalated, _, _, _ = action_resolution._resolve(
        ["AT-0001"], catalog, channel_code=4, decision_date=date(2026, 3, 1), adjustment_stack_enabled=True,
        apply_escalation=False)
    assert escalated_action == vocab.ACTION_STEP_UP
    assert not_escalated == vocab.ACTION_MONITOR
