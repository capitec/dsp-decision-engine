from datetime import date

from fraud_interdiction import firing, vocab
from fraud_interdiction.rules import RuleDefinition


def _rule(**overrides) -> RuleDefinition:
    defaults = dict(
        rule_id="MS-0001", rule_version=1, family=vocab.FAMILY_MULE_SCAM, severity=4,
        action_code=vocab.ACTION_HOLD_FOR_REVIEW, priority=100, status=vocab.STATUS_LIVE,
        effective_from=date(2025, 1, 1), effective_to=None,
        event_types=(vocab.EVENT_TYPE_INSTANT_PAYMENT,), segments=None,
        overlay_exempt=False, critical=False, suppressible=False, reason_code=3001, queue="MS-1",
        stale_absent_behavior=vocab.STALE_ABSENT_LAST_KNOWN, max_fire_rate_pct=5.0,
        condition={"op": ">", "feature": "amount", "threshold": 1000.0}, predicate_count=1,
        overlay_eligible=False, base_condition=None, references_velocity=False, velocity_feature=None,
        description_internal="test", description_client_safe="test",
    )
    defaults.update(overrides)
    return RuleDefinition(**defaults)


def test_applicable_gates_on_event_type_dates_and_segments():
    rule = _rule(event_types=(210,), effective_from=date(2025, 1, 1), effective_to=date(2026, 1, 1),
                 segments=("high_value",))
    assert firing._applicable(rule, 210, date(2025, 6, 1), ["high_value"], "normal")
    assert not firing._applicable(rule, 111, date(2025, 6, 1), ["high_value"], "normal")  # wrong event type
    assert not firing._applicable(rule, 210, date(2026, 6, 1), ["high_value"], "normal")  # past effective_to
    assert not firing._applicable(rule, 210, date(2025, 6, 1), ["youth"], "normal")  # no segment overlap
    assert firing._applicable(rule, 210, date(2025, 6, 1), [], "normal") is False  # no segments at all


def test_segments_none_means_all():
    rule = _rule(segments=None)
    assert firing._applicable(rule, 210, date(2025, 6, 1), ["anything"], "normal")


def test_restricted_mode_suspends_suppressible_rules_only():
    suppressible = _rule(suppressible=True)
    not_suppressible = _rule(suppressible=False)
    assert not firing._applicable(suppressible, 210, date(2025, 6, 1), None, "restricted")
    assert firing._applicable(not_suppressible, 210, date(2025, 6, 1), None, "restricted")


def test_unevaluable_only_when_suppress_declared_and_state_degraded():
    rule = _rule(velocity_feature="velocity_count_1min", stale_absent_behavior=vocab.STALE_ABSENT_SUPPRESS)
    fresh_row = {"velocity_count_1min_state": firing.FRESH}
    stale_row = {"velocity_count_1min_state": firing.STALE}
    assert not firing._unevaluable(rule, fresh_row)
    assert firing._unevaluable(rule, stale_row)

    evaluate_false_rule = _rule(velocity_feature="velocity_count_1min",
                                 stale_absent_behavior=vocab.STALE_ABSENT_EVALUATE_FALSE)
    assert not firing._unevaluable(evaluate_false_rule, stale_row)
    assert firing._forced_false(evaluate_false_rule, stale_row)


def test_resolve_row_separates_fired_unevaluable_and_overlay_induced():
    base_rule = _rule(rule_id="MS-0001")
    overlay_rule = _rule(rule_id="MS-0002", overlay_eligible=True)
    suppressed_rule = _rule(rule_id="MS-0003", velocity_feature="velocity_count_1min",
                             stale_absent_behavior=vocab.STALE_ABSENT_SUPPRESS)
    shadow_rule = _rule(rule_id="MS-0100", status=vocab.STATUS_SHADOW)

    row = {
        "event_type_code": 210, "decision_date": date(2025, 6, 1), "client_segments": None,
        "degraded_mode_code": "normal",
        "velocity_count_1min_state": firing.STALE, "velocity_count_10min_state": firing.FRESH,
        "velocity_sum_amount_1h_state": firing.FRESH,
        "MS-0001.fired": True, "MS-0002.fired": True, "MS-0002.fired_base": False,
        "MS-0003.fired": True, "MS-0100.fired": True,
    }
    result = firing._resolve_row(row, (base_rule, overlay_rule, suppressed_rule), (shadow_rule,),
                                  overlay_eligible_ids=frozenset({"MS-0002"}))
    assert result["fired_rule_ids"] == ["MS-0001", "MS-0002"]
    assert result["unevaluable_rule_ids"] == ["MS-0003"]
    assert result["fired_on_overlay_ids"] == ["MS-0002"]
    assert result["counterfactual_fired_rule_ids"] == ["MS-0001"]  # MS-0002 wouldn't have fired at base
    assert result["shadow_fired_rule_ids"] == ["MS-0100"]
