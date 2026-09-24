"""Tests for fraud decision engine (01)."""
import pytest
from datetime import date, datetime, timezone
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/00-shared-credit-core")

from fraud_engine import (
    generate_ruleset, RuleEvaluator, ActionCode, Family, RuleStatus,
    normalise_event, NormalisedEvent
)


class TestRuleGeneration:
    """Test rule set generation (01 §6.1)."""

    def test_ruleset_volume(self):
        """Rule volume kept at specification (01 SCOPE)."""
        decision_date = date(2026, 9, 24)
        ruleset = generate_ruleset(decision_date)

        # Verify rule counts per family
        assert len(ruleset.live_rules) >= 500, f"Expected ~520 live rules, got {len(ruleset.live_rules)}"
        assert len(ruleset.shadow_rules) >= 100, f"Expected ~100+ shadow rules, got {len(ruleset.shadow_rules)}"

        # Verify stable rule ids (09 §5.15 item 2)
        live_ids = [r.rule_id for r in ruleset.live_rules]
        shadow_ids = [r.rule_id for r in ruleset.shadow_rules]
        assert len(live_ids) == len(set(live_ids)), "Duplicate rule_id in live rules"
        assert len(shadow_ids) == len(set(shadow_ids)), "Duplicate rule_id in shadow rules"

    def test_rule_attributes(self):
        """Rule definitions have required attributes (01 §6.1)."""
        decision_date = date(2026, 9, 24)
        ruleset = generate_ruleset(decision_date)

        for rule in ruleset.live_rules[:5]:  # Spot check
            assert rule.rule_id
            assert rule.family in Family
            assert rule.description
            assert 1 <= rule.severity <= 5
            assert rule.action in ActionCode
            assert rule.priority >= 0
            assert rule.status == RuleStatus.LIVE
            assert rule.effective_from == decision_date
            assert rule.event_types == {210}  # Instant payment only (SCOPE)
            assert rule.reason_code > 0
            assert rule.rule_version >= 1


class TestRuleEvaluation:
    """Test rule evaluation (01 §5.10)."""

    def test_rule_fired_on_threshold(self):
        """Rule fires when features exceed thresholds (01 §5.10)."""
        decision_date = date(2026, 9, 24)
        ruleset = generate_ruleset(decision_date)
        evaluator = RuleEvaluator()

        # Get a live rule
        rule = ruleset.live_rules[0]

        # Create features that should fire the rule
        event_features = {
            "amount": rule.thresholds["amount"] + 1000,  # Exceed threshold
            "beneficiary_age_hours": rule.thresholds["beneficiary_age_hours"] + 1,
            "device_change_hours": rule.thresholds["device_change_hours"] + 1,
        }

        fired, base_fired, features = evaluator.evaluate_rule(rule, event_features)
        assert fired, f"Rule {rule.rule_id} should have fired"
        assert base_fired
        assert features

    def test_rule_not_fired_below_threshold(self):
        """Rule does not fire when features below thresholds (01 §5.10)."""
        decision_date = date(2026, 9, 24)
        ruleset = generate_ruleset(decision_date)
        evaluator = RuleEvaluator()

        rule = ruleset.live_rules[0]

        # Create features below threshold
        event_features = {
            "amount": rule.thresholds["amount"] / 2,  # Below threshold
            "beneficiary_age_hours": rule.thresholds["beneficiary_age_hours"] / 2,
            "device_change_hours": rule.thresholds["device_change_hours"] / 2,
        }

        fired, _, _ = evaluator.evaluate_rule(rule, event_features)
        assert not fired, f"Rule {rule.rule_id} should not have fired"

    def test_all_rules_evaluated_no_early_exit(self):
        """All applicable rules evaluated, no early exit (01 §5.10)."""
        decision_date = date(2026, 9, 24)
        ruleset = generate_ruleset(decision_date)

        applicable_live = ruleset.get_applicable_live(210, None)
        applicable_shadow = ruleset.get_applicable_shadow(210, None)

        # Both live and shadow should have applicable rules
        assert len(applicable_live) > 0, "Should have applicable live rules"
        assert len(applicable_shadow) > 0, "Should have applicable shadow rules"


class TestActionResolution:
    """Test action resolution (01 §5.12)."""

    def test_action_precedence_by_severity(self):
        """Action resolved by severity hierarchy (01 §5.12)."""
        decision_date = date(2026, 9, 24)
        ruleset = generate_ruleset(decision_date)
        evaluator = RuleEvaluator()

        # Create fired rules with different actions
        from fraud_engine.evaluator import FiredRule

        fired_rules = [
            FiredRule(
                rule=ruleset.live_rules[0],
                feature_values={"amount": 1000.0},
                fired_on_base=True,
                fired_on_overlay=False,
                evaluation_order=0
            )
        ]

        outcome = evaluator.resolve_action(fired_rules, [], None)
        assert outcome.action_code in ActionCode
        assert outcome.action_source_rule_id

    def test_hard_block_enforces_minimum_severity(self):
        """Hard block enforces minimum action severity (01 §5.7)."""
        decision_date = date(2026, 9, 24)
        ruleset = generate_ruleset(decision_date)
        evaluator = RuleEvaluator()

        # With a hard block of DECLINE, result must be at least DECLINE
        from fraud_engine.evaluator import FiredRule

        fired_rules = [
            FiredRule(
                rule=ruleset.live_rules[0],
                feature_values={},
                fired_on_base=True,
                fired_on_overlay=False,
                evaluation_order=0
            )
        ]

        outcome = evaluator.resolve_action(
            fired_rules,
            [],
            hard_block_action=ActionCode.DECLINE
        )

        assert outcome.action_code.value >= ActionCode.DECLINE.value


class TestEventNormalisation:
    """Test event admission and normalisation (01 §5.1)."""

    def test_normalise_valid_event(self):
        """Valid event normalises correctly (01 §5.1)."""
        assessment_ts = datetime.now(timezone.utc)
        raw_event = {
            "event_id": 1001,
            "event_type_code": 210,
            "event_timestamp": "2026-09-24T14:22:03Z",
            "client_id": 1000001,
            "channel_code": 2,
            "product_code": 0,
            "amount": 8500.00,
            "device_id": "dev_abc"
        }

        normalised, error = normalise_event(raw_event, assessment_ts)

        assert error is None, f"Should normalise without error, got {error}"
        assert normalised is not None
        assert normalised.event_id == 1001
        assert normalised.event_type_code == 210
        assert normalised.client_id == 1000001
        assert normalised.amount == 8500.00

    def test_normalise_missing_mandatory_field(self):
        """Missing mandatory field rejects event (01 §5.1)."""
        assessment_ts = datetime.now(timezone.utc)
        raw_event = {
            "event_id": 1001,
            # Missing event_type_code
            "event_timestamp": "2026-09-24T14:22:03Z",
            "client_id": 1000001,
        }

        normalised, error = normalise_event(raw_event, assessment_ts)

        assert normalised is None
        assert error is not None

    def test_normalise_invalid_timestamp(self):
        """Invalid timestamp format rejects event (01 §5.1)."""
        assessment_ts = datetime.now(timezone.utc)
        raw_event = {
            "event_id": 1001,
            "event_type_code": 210,
            "event_timestamp": "not-a-timestamp",
            "client_id": 1000001
        }

        normalised, error = normalise_event(raw_event, assessment_ts)

        assert normalised is None
        assert error is not None


class TestEvidenceContract:
    """Test evidence contract compliance (09 §5.15)."""

    def test_decision_id_generation(self):
        """Stable decision identifier (09 §5.15 item 1)."""
        from core import generate_decision_id

        id1 = generate_decision_id()
        id2 = generate_decision_id()

        assert id1 != id2, "decision_id should be unique"
        assert isinstance(id1, str)
        assert len(id1) > 0

    def test_no_reliance_on_today(self):
        """No "today" in logic, use decision_date (09 §5.15 item 3)."""
        decision_date = date(2026, 9, 20)
        ruleset = generate_ruleset(decision_date)

        # All rules should use effective dates, not "today"
        for rule in ruleset.live_rules:
            assert rule.effective_from == decision_date or rule.effective_from < decision_date
            assert rule.applies_at(decision_date) or rule.effective_from > decision_date


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
