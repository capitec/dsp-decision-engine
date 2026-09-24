"""
Tests for project 05: business credit assessment with nested entities.
"""
import pytest
from datetime import date
from assessment import (
    Entity,
    AdverseEvent,
    Severity,
    Criticality,
    EntityVerdict,
    AssessmentRequest,
    assess_business_credit,
    determine_criticality,
    classify_adverse_event,
    roll_up_entity_verdict
)


class TestCriticality:
    """Tests for entity criticality classification."""

    def test_critical_controlling_entity(self):
        """Entity marked as controlling should be critical."""
        entity = Entity(
            entity_id=1,
            entity_key="ctrl_123",
            is_natural_person=True,
            relationship_type_code=1,
            path=["root"],
            depth=1,
            effective_ownership_pct=35.0,
            is_controlling=True
        )
        entities = [entity]

        criticality = determine_criticality(entity, entities)
        assert criticality == Criticality.CRITICAL

    def test_critical_high_ownership(self):
        """Entity with >=25% ownership should be critical."""
        entity = Entity(
            entity_id=1,
            entity_key="shareholder_123",
            is_natural_person=True,
            relationship_type_code=5,
            path=["root"],
            depth=1,
            direct_ownership_pct=30.0,
            effective_ownership_pct=30.0,
            is_controlling=False
        )
        entities = [entity]

        criticality = determine_criticality(entity, entities)
        assert criticality == Criticality.CRITICAL

    def test_significant_mid_ownership(self):
        """Entity with 10-25% ownership should be significant."""
        entity = Entity(
            entity_id=1,
            entity_key="shareholder_456",
            is_natural_person=True,
            relationship_type_code=5,
            path=["root"],
            depth=1,
            direct_ownership_pct=15.0,
            effective_ownership_pct=15.0,
            is_controlling=False
        )
        entities = [entity]

        criticality = determine_criticality(entity, entities)
        assert criticality == Criticality.SIGNIFICANT

    def test_peripheral_low_ownership(self):
        """Entity with <10% ownership should be peripheral."""
        entity = Entity(
            entity_id=1,
            entity_key="shareholder_789",
            is_natural_person=True,
            relationship_type_code=5,
            path=["root"],
            depth=1,
            direct_ownership_pct=5.0,
            effective_ownership_pct=5.0,
            is_controlling=False
        )
        entities = [entity]

        criticality = determine_criticality(entity, entities)
        assert criticality == Criticality.PERIPHERAL


class TestAdverseEventClassification:
    """Tests for adverse event classification."""

    def test_classify_satisfied_judgment_old(self):
        """Satisfied judgment >24 months old should be immaterial."""
        entity = Entity(
            entity_id=1,
            entity_key="person_123",
            is_natural_person=True,
            relationship_type_code=1,
            path=["root"],
            depth=1,
            effective_ownership_pct=100.0,
            is_controlling=True
        )
        entity.criticality_class = Criticality.CRITICAL

        event = AdverseEvent(
            event_id=1,
            entity_key="person_123",
            event_type_code=1,
            amount=30000.0,
            event_date=date(2023, 1, 1),
            is_satisfied=True,
            satisfaction_date=date(2023, 6, 1)
        )

        decision_date = date(2026, 3, 15)
        entities = [entity]
        thresholds = {
            Criticality.CRITICAL: {
                'event_1_material': 10000,
                'event_1_disqualifying': 50000
            }
        }

        classify_adverse_event(event, entity, entities, decision_date, thresholds)

        assert event.event_severity_code == Severity.IMMATERIAL
        assert event.classification_rule == "AE-C-01"

    def test_classify_unsatisfied_judgment_disqualifying(self):
        """Unsatisfied judgment above disqualifying threshold should be disqualifying."""
        entity = Entity(
            entity_id=1,
            entity_key="person_123",
            is_natural_person=True,
            relationship_type_code=1,
            path=["root"],
            depth=1,
            effective_ownership_pct=100.0,
            is_controlling=True
        )
        entity.criticality_class = Criticality.CRITICAL

        event = AdverseEvent(
            event_id=1,
            entity_key="person_123",
            event_type_code=1,
            amount=75000.0,
            event_date=date(2025, 6, 15),
            is_satisfied=False
        )

        decision_date = date(2026, 3, 15)
        entities = [entity]
        thresholds = {
            Criticality.CRITICAL: {
                'event_1_material': 10000,
                'event_1_disqualifying': 50000
            }
        }

        classify_adverse_event(event, entity, entities, decision_date, thresholds)

        assert event.event_severity_code == Severity.DISQUALIFYING
        assert event.classification_rule == "AE-C-03"


class TestEntityVerdict:
    """Tests for entity adverse verdict roll-up."""

    def test_verdict_any_disqualifying_event(self):
        """Any disqualifying event should produce disqualifying verdict."""
        entity = Entity(
            entity_id=1,
            entity_key="person_123",
            is_natural_person=True,
            relationship_type_code=1,
            path=["root"],
            depth=1,
            effective_ownership_pct=100.0,
            is_controlling=True
        )

        event = AdverseEvent(
            event_id=1,
            entity_key="person_123",
            event_type_code=1,
            amount=75000.0,
            event_date=date(2025, 6, 15),
            is_satisfied=False,
            event_severity_code=Severity.DISQUALIFYING,
            classification_rule="AE-C-03"
        )

        decision_date = date(2026, 3, 15)

        verdict = roll_up_entity_verdict(entity, [event], decision_date, 750000.0)

        assert verdict.verdict == EntityVerdict.DISQUALIFYING
        assert verdict.binding_rule == "AE-R-01"
        assert event.event_id in verdict.event_ids

    def test_verdict_multiple_minor_events_recent(self):
        """Three or more minor events within 12 months should be material."""
        entity = Entity(
            entity_id=1,
            entity_key="person_456",
            is_natural_person=True,
            relationship_type_code=5,
            path=["root"],
            depth=1,
            effective_ownership_pct=20.0,
            is_controlling=False
        )

        decision_date = date(2026, 3, 15)

        events = [
            AdverseEvent(
                event_id=1,
                entity_key="person_456",
                event_type_code=2,
                amount=5000.0,
                event_date=date(2025, 10, 1),
                is_satisfied=False,
                event_severity_code=Severity.MINOR,
                classification_rule="AE-C-08"
            ),
            AdverseEvent(
                event_id=2,
                entity_key="person_456",
                event_type_code=2,
                amount=3000.0,
                event_date=date(2025, 11, 15),
                is_satisfied=False,
                event_severity_code=Severity.MINOR,
                classification_rule="AE-C-08"
            ),
            AdverseEvent(
                event_id=3,
                entity_key="person_456",
                event_type_code=2,
                amount=4000.0,
                event_date=date(2026, 1, 10),
                is_satisfied=False,
                event_severity_code=Severity.MINOR,
                classification_rule="AE-C-08"
            )
        ]

        verdict = roll_up_entity_verdict(entity, events, decision_date, 750000.0)

        assert verdict.verdict == EntityVerdict.MATERIAL
        assert verdict.binding_rule == "AE-R-02"
        assert len(verdict.event_ids) == 3

    def test_verdict_no_events_clear(self):
        """Entity with no events should be clear."""
        entity = Entity(
            entity_id=1,
            entity_key="person_789",
            is_natural_person=True,
            relationship_type_code=1,
            path=["root"],
            depth=1,
            effective_ownership_pct=100.0,
            is_controlling=True
        )

        decision_date = date(2026, 3, 15)

        verdict = roll_up_entity_verdict(entity, [], decision_date, 750000.0)

        assert verdict.verdict == EntityVerdict.CLEAR
        assert verdict.binding_rule == "AE-R-11"


class TestFullAssessment:
    """Tests for full assessment flow."""

    def test_assessment_basic_business(self):
        """Test assessment of a simple business with one director and one event."""
        entities = [
            Entity(
                entity_id=1,
                entity_key="person_123",
                is_natural_person=True,
                relationship_type_code=1,
                path=["company_root"],
                depth=1,
                effective_ownership_pct=100.0,
                is_controlling=True
            )
        ]

        events = [
            AdverseEvent(
                event_id=101,
                entity_key="person_123",
                event_type_code=2,
                amount=12000.0,
                event_date=date(2025, 8, 15),
                is_satisfied=False
            )
        ]

        request = AssessmentRequest(
            application_id=1001,
            client_id=None,
            decision_date=date(2026, 3, 15),
            product_code=50,
            requested_amount=750000.0,
            declared_annual_turnover=2000000.0,
            entities=entities,
            adverse_events=events
        )

        result = assess_business_credit(request)

        assert result.application_id == 1001
        assert result.resolved_entity_count == 1
        assert len(result.entity_verdicts) == 1
        assert result.entity_verdicts[0].verdict == EntityVerdict.MATERIAL

    def test_assessment_multi_entity_business(self):
        """Test assessment with multiple entities (director, shareholder, surety)."""
        entities = [
            Entity(
                entity_id=1,
                entity_key="person_123",
                is_natural_person=True,
                relationship_type_code=1,
                path=["company_root"],
                depth=1,
                effective_ownership_pct=35.0,
                is_controlling=True
            ),
            Entity(
                entity_id=2,
                entity_key="person_456",
                is_natural_person=True,
                relationship_type_code=5,
                path=["company_root"],
                depth=1,
                direct_ownership_pct=25.0,
                effective_ownership_pct=25.0,
                is_controlling=False
            ),
            Entity(
                entity_id=3,
                entity_key="person_789",
                is_natural_person=True,
                relationship_type_code=1,
                path=["company_root"],
                depth=1,
                effective_ownership_pct=8.0,
                is_controlling=False,
                is_required_surety=True
            )
        ]

        events = [
            AdverseEvent(
                event_id=101,
                entity_key="person_123",
                event_type_code=1,
                amount=45000.0,
                event_date=date(2024, 11, 20),
                is_satisfied=False
            ),
            AdverseEvent(
                event_id=102,
                entity_key="person_456",
                event_type_code=2,
                amount=12000.0,
                event_date=date(2025, 8, 15),
                is_satisfied=False
            )
        ]

        request = AssessmentRequest(
            application_id=1001,
            client_id=None,
            decision_date=date(2026, 3, 15),
            product_code=50,
            requested_amount=750000.0,
            declared_annual_turnover=3500000.0,
            entities=entities,
            adverse_events=events
        )

        result = assess_business_credit(request)

        assert result.resolved_entity_count == 3
        assert len(result.entities_with_criticality) == 3
        assert len(result.entity_verdicts) == 3
        # Verify criticality classification
        assert result.entities_with_criticality[0]['criticality_class'] == Criticality.CRITICAL  # director, 35% ownership
        assert result.entities_with_criticality[1]['criticality_class'] == Criticality.CRITICAL  # 25% ownership (>=25% is critical)
        assert result.entities_with_criticality[2]['criticality_class'] == Criticality.CRITICAL  # required surety


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
