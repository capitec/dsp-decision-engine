"""Tests for the governance and replay harness (09-H)."""
import pytest
from datetime import date
from governance import (
    ReplayEngine, ReplayVerdict, ReplayResult,
    ExplanationRenderer, ExplanationOutput,
    VersionDiffer, SwapSetAnalyzer, OverlayRegister, OverlayRecord,
    DeadLogicDetector
)


class TestReplayEngine:
    """Test decision replay (09 §5.1)."""

    def test_replay_identical_output(self):
        """Replay with identical output should return REPRODUCED."""
        engine = ReplayEngine()

        evidence = {
            "inputs_as_received": {"amount": 50000},
            "outputs": {"outcome_code": "approved", "offered_amount": 50000},
            "parameters": {"version": 1},
        }

        def flow_fn(**kwargs):
            return {"outcome_code": "approved", "offered_amount": 50000}

        result = engine.replay_decision(
            decision_id="DEC-001",
            evidence=evidence,
            flow_fn=flow_fn,
            decision_date=date(2024, 9, 24),
        )

        assert result.verdict == ReplayVerdict.REPRODUCED
        assert result.field_differences == {}

    def test_replay_diverges(self):
        """Replay with different output should return NOT_REPRODUCED."""
        engine = ReplayEngine()

        evidence = {
            "inputs_as_received": {"amount": 50000},
            "outputs": {"outcome_code": "approved"},
            "parameters": {"version": 1},
        }

        def flow_fn(**kwargs):
            return {"outcome_code": "declined"}

        result = engine.replay_decision(
            decision_id="DEC-002",
            evidence=evidence,
            flow_fn=flow_fn,
            decision_date=date(2024, 9, 24),
        )

        assert result.verdict == ReplayVerdict.NOT_REPRODUCED
        assert "outcome_code" in result.field_differences

    def test_replay_within_tolerance(self):
        """Replay within tolerance bounds should return REPRODUCED_WITHIN_TOLERANCE."""
        engine = ReplayEngine()

        evidence = {
            "inputs_as_received": {"amount": 50000},
            "outputs": {"score": 600.0},
            "parameters": {"version": 1},
        }

        def flow_fn(**kwargs):
            return {"score": 600.0000001}  # Within 1e-6 tolerance

        result = engine.replay_decision(
            decision_id="DEC-003",
            evidence=evidence,
            flow_fn=flow_fn,
            decision_date=date(2024, 9, 24),
        )

        # Should be within tolerance or reproduced
        assert result.verdict in (ReplayVerdict.REPRODUCED, ReplayVerdict.REPRODUCED_WITHIN_TOLERANCE)


class TestExplanationRenderer:
    """Test explanation rendering for three audiences (09 §5.2)."""

    def test_consultant_rendering(self):
        """Consultant rendering should be brief and plain language."""
        renderer = ExplanationRenderer()
        evidence = {
            "outcome_code": "declined",
            "primary_reason_code": "AFF-001",
            "product": "Flex Loan",
            "decision_date": date(2024, 9, 24),
        }

        text = renderer.render_consultant(evidence)

        assert "Your application" in text
        assert "Flex Loan" in text
        assert "declined" in text
        assert len(text) < 300  # Should be short

    def test_analyst_rendering(self):
        """Analyst rendering should include gates, rules, cap chain."""
        renderer = ExplanationRenderer()
        evidence = {
            "outcome_code": "declined",
            "decision_date": date(2024, 9, 24),
        }
        explanation = ExplanationOutput(
            decision_id="DEC-001",
            audience="analyst",
            flow_name="01-fraud",
            decision_date=date(2024, 9, 24),
            gates_evaluated=[
                {"gate_id": "G-001", "name": "Sanctions", "outcome": "passed", "value": "no_match"},
            ],
            rules_fired=[
                {"rule_id": "FR-001", "description": "Velocity check"},
            ],
        )

        text = renderer.render_analyst(evidence, explanation)

        assert "Gates:" in text
        assert "G-001" in text
        assert "Rules fired:" in text
        assert "FR-001" in text

    def test_ombud_rendering(self):
        """Ombud rendering should show policy, adjusted vs unadjusted."""
        renderer = ExplanationRenderer()
        evidence = {
            "outcome_code": "declined",
            "decision_date": date(2024, 9, 24),
        }
        explanation = ExplanationOutput(
            decision_id="DEC-001",
            audience="ombud",
            flow_name="03-granting",
            decision_date=date(2024, 9, 24),
            cap_chain=[
                {"value": 150000, "reason": "Affordability limit"},
            ],
            adjusted_values={"grade": 8},
            unadjusted_values={"grade": 6},
        )

        text = renderer.render_ombud(evidence, explanation)

        assert "Decision ID" in text
        assert "Policy in Force" in text
        assert "Cap Chain" in text
        assert "Adjustments Applied" in text


class TestVersionDiffer:
    """Test semantic version diffing (09 §5.4)."""

    def test_diff_added_rules(self):
        """Should detect added rules."""
        differ = VersionDiffer()

        rules_a = [
            {"rule_id": "R-001", "description": "Old rule"},
        ]
        rules_b = [
            {"rule_id": "R-001", "description": "Old rule"},
            {"rule_id": "R-002", "description": "New rule"},
        ]

        diff = differ.diff_rule_set(rules_a, rules_b)

        assert len(diff.changes) == 1
        assert diff.changes[0]["type"] == "add"
        assert diff.changes[0]["rule_id"] == "R-002"

    def test_diff_removed_rules(self):
        """Should detect removed rules."""
        differ = VersionDiffer()

        rules_a = [
            {"rule_id": "R-001", "description": "Rule to remove"},
            {"rule_id": "R-002", "description": "Keep this"},
        ]
        rules_b = [
            {"rule_id": "R-002", "description": "Keep this"},
        ]

        diff = differ.diff_rule_set(rules_a, rules_b)

        assert len(diff.changes) == 1
        assert diff.changes[0]["type"] == "remove"
        assert diff.changes[0]["rule_id"] == "R-001"


class TestOverlayRegister:
    """Test overlay management (09 §5.14)."""

    def test_add_overlay(self):
        """Should add overlay to register."""
        register = OverlayRegister()

        overlay = OverlayRecord(
            overlay_id="OVL-001",
            flow_name="01-fraud",
            kind="sensitivity_dial",
            scope="all",
            magnitude=1.2,
            stack_position=1,
            owner="fraud-team",
        )

        register.add_overlay(overlay)

        active = register.get_overlays_at_date(date.today())
        assert len(active) == 1
        assert active["OVL-001"].overlay_id == "OVL-001"

    def test_overlays_at_date(self):
        """Should return only overlays active at a given date."""
        register = OverlayRegister()

        # Overlay effective 2024-01-01 to 2024-12-31
        overlay = OverlayRecord(
            overlay_id="OVL-001",
            flow_name="01",
            kind="sensitivity_dial",
            scope="all",
            magnitude=1.2,
            stack_position=1,
            owner="team",
            effective_from=date(2024, 1, 1),
            effective_to=date(2024, 12, 31),
        )
        register.add_overlay(overlay)

        # Should be active on 2024-06-01
        active = register.get_overlays_at_date(date(2024, 6, 1))
        assert len(active) == 1

        # Should not be active on 2025-01-01
        active = register.get_overlays_at_date(date(2025, 1, 1))
        assert len(active) == 0


class TestDeadLogicDetector:
    """Test dead logic detection (09 §5.8)."""

    def test_detect_dead_rules(self):
        """Should identify rules that never fired."""
        detector = DeadLogicDetector()

        rule_set = [
            {"rule_id": "R-001", "description": "Fires regularly"},
            {"rule_id": "R-002", "description": "Never fires"},
        ]

        evaluation_log = [
            {"rule_id": "R-001", "outcome": "fired"},
            {"rule_id": "R-001", "outcome": "fired"},
        ]

        report = detector.detect_dead_rules(
            rule_set=rule_set,
            evaluation_log=evaluation_log,
            reference_date=date(2024, 9, 24),
            window_days=90,
        )

        assert len(report.dead_rules) == 1
        assert report.dead_rules[0]["rule_id"] == "R-002"


class TestSwapSetAnalyzer:
    """Test swap-set analysis (09 §5.5)."""

    def test_compare_versions(self):
        """Should measure approval and amount changes."""
        analyzer = SwapSetAnalyzer()

        population = [
            {"client_id": 1, "amount": 100000},
            {"client_id": 2, "amount": 50000},
        ]

        def version_a_fn(**kwargs):
            return {"outcome_code": "approved", "offered_amount": 100000}

        def version_b_fn(**kwargs):
            return {"outcome_code": "approved", "offered_amount": 105000}

        report = analyzer.compare_versions(
            population=population,
            version_a_fn=version_a_fn,
            version_b_fn=version_b_fn,
            decision_date=date(2024, 9, 24),
        )

        assert report.population_size == 2
        assert report.approvals_lost_count == 0  # Both versions approved
        assert report.amount_changed_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
