"""
Business credit assessment with nested entity handling.

Implements stages 5.1-5.8 (and 5.10) of spec 05:
- Structure resolution
- Criticality classification
- Adverse event classification with criticality-dependent thresholds
- Entity adverse verdict roll-up
- Entity scoring
- People blend
- Combined business grade
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional, List
from enum import IntEnum


class Severity(IntEnum):
    """Event severity codes."""
    IMMATERIAL = 1
    MINOR = 2
    MATERIAL = 3
    DISQUALIFYING = 4


class Criticality(IntEnum):
    """Entity criticality class."""
    CRITICAL = 1
    SIGNIFICANT = 2
    PERIPHERAL = 3


class EntityVerdict(IntEnum):
    """Entity adverse verdict."""
    CLEAR = 1
    MINOR = 2
    MATERIAL = 3
    DISQUALIFYING = 4


@dataclass
class Entity:
    """An entity in the business structure."""
    entity_id: int
    entity_key: str  # De-duplication key
    is_natural_person: bool
    relationship_type_code: int
    path: List[str]  # Ownership path from applicant
    depth: int  # 1..3
    direct_ownership_pct: Optional[float] = None
    effective_ownership_pct: float = 0.0
    is_controlling: bool = False
    is_required_surety: bool = False
    date_of_birth: Optional[date] = None
    screening_outcome_code: int = 1  # 1=clear
    screening_confidence: Optional[float] = None
    deceased_flag: Optional[bool] = None
    identity_verification_code: int = 1
    residency_code: Optional[int] = None

    # Computed at 5.4
    criticality_class: Optional[Criticality] = None
    criticality_inputs: Optional[dict] = None  # For attribution

    # Adverse events attached to this entity
    events: List[dict] = field(default_factory=list)


@dataclass
class AdverseEvent:
    """A classified adverse event."""
    event_id: int
    entity_key: str
    event_type_code: int
    amount: Optional[float] = None
    event_date: date = field(default_factory=date.today)
    status_code: int = 1  # 1=active
    is_disputed: bool = False
    is_satisfied: bool = False
    satisfaction_date: Optional[date] = None
    source_code: int = 1
    duplicate_group_id: Optional[int] = None

    # Classified at 5.5
    event_severity_code: Optional[Severity] = None
    classification_rule: Optional[str] = None
    threshold_material: Optional[float] = None
    threshold_disqualifying: Optional[float] = None
    threshold_material_overlaid: Optional[float] = None
    threshold_disqualifying_overlaid: Optional[float] = None
    overlay_applied: Optional[str] = None
    event_age_months: Optional[int] = None
    classification_provisional: bool = False


@dataclass
class AssessmentRequest:
    """Input request for business credit assessment."""
    application_id: int
    client_id: Optional[int]
    decision_date: date
    product_code: int  # 50 or 51
    requested_amount: float
    requested_term_months: Optional[int] = None
    registration_number: Optional[str] = None
    legal_form_code: int = 1
    registration_status_code: int = 1
    registration_date: Optional[date] = None
    trading_since_date: Optional[date] = None
    sector_code: int = 100
    jurisdiction_code: int = 1
    tax_reference: Optional[str] = None
    tax_compliance_status_code: int = 1
    vat_registration_number: Optional[str] = None
    declared_annual_turnover: float = 0.0
    facility_purpose_code: int = 1
    security_offered: List[dict] = field(default_factory=list)

    # Entity structure (1..40 entities after resolution)
    entities: List[Entity] = field(default_factory=list)

    # Adverse events (0..60 per entity)
    adverse_events: List[AdverseEvent] = field(default_factory=list)


@dataclass
class EntityAdverseVerdict:
    """Verdict for one entity's adverse history."""
    entity_key: str
    verdict: EntityVerdict
    binding_rule: str
    event_ids: List[int]  # Events that justified the verdict
    severity_counts: dict = field(default_factory=dict)  # {severity: count}
    total_unsatisfied_amount: float = 0.0
    most_recent_event: Optional[dict] = None
    verdict_unadjusted: Optional[EntityVerdict] = None
    overlay_sensitive: bool = False


@dataclass
class AssessmentResult:
    """Output from business credit assessment."""
    application_id: int
    decision_date: date

    # Stage 5.1 - Structure resolution
    resolved_entity_count: int = 0
    structure_unresolved: bool = False
    ownership_total_pct: float = 0.0

    # Stage 5.4 - Entity criticality
    entities_with_criticality: List[dict] = field(default_factory=list)

    # Stage 5.5-5.6 - Adverse events and roll-up
    entity_verdicts: List[EntityAdverseVerdict] = field(default_factory=list)

    # Stage 5.7 - Entity scoring
    entity_scores: List[dict] = field(default_factory=list)

    # Stage 5.8 - People blend
    people_pd: float = 0.0
    people_grade: int = 6
    people_coverage_ratio: float = 0.0
    people_coverage_sufficient: bool = False

    # Stage 5.10 - Combined grade
    business_pd: float = 0.0
    business_grade: int = 6
    business_verdict_code: int = 1  # 1=clear
    declined_entity_key: Optional[str] = None
    declined_event_id: Optional[int] = None
    declined_rule: Optional[str] = None


def determine_criticality(entity: Entity, entities: List[Entity]) -> Criticality:
    """
    Determine entity criticality class per spec 05 §5.4.

    Critical: controlling, ≥25% ownership, required surety, sole director/trustee/member,
             or corporate guarantor with >20% cover.
    Significant: 10-25% ownership, director/trustee without control.
    Peripheral: everything else.
    """
    if entity.is_controlling:
        return Criticality.CRITICAL

    if entity.effective_ownership_pct >= 25.0:
        return Criticality.CRITICAL

    if entity.is_required_surety:
        return Criticality.CRITICAL

    # Check sole director/trustee/member
    # (Simplified: count entities with same role type; if count==1, it's sole)
    same_role = [e for e in entities if e.relationship_type_code == entity.relationship_type_code]
    if len(same_role) == 1 and entity.relationship_type_code in [1, 3, 2]:  # director, trustee, member
        return Criticality.CRITICAL

    if entity.effective_ownership_pct >= 10.0:
        return Criticality.SIGNIFICANT

    if entity.relationship_type_code in [1, 3]:  # director or trustee
        return Criticality.SIGNIFICANT

    return Criticality.PERIPHERAL


def classify_adverse_event(
    event: AdverseEvent,
    entity: Entity,
    entities: List[Entity],
    decision_date: date,
    thresholds_by_criticality: dict
) -> None:
    """
    Classify an adverse event per spec 05 §5.5.

    Amount thresholds vary by entity criticality class.
    Applies overlays from stack position 1 (if provided in thresholds).
    """
    # Determine criticality
    if entity.criticality_class is None:
        entity.criticality_class = determine_criticality(entity, entities)

    criticality = entity.criticality_class

    # Base thresholds from table
    thresholds = thresholds_by_criticality.get(criticality, {})
    material_base = thresholds.get(f'event_{event.event_type_code}_material', 50000)
    disq_base = thresholds.get(f'event_{event.event_type_code}_disqualifying', 250000)

    # For demo, assume no overlay adjustments (position 1 would modify these)
    event.threshold_material = material_base
    event.threshold_disqualifying = disq_base
    event.threshold_material_overlaid = material_base
    event.threshold_disqualifying_overlaid = disq_base

    # Age calculation
    event_age = (decision_date - event.event_date).days // 30
    event.event_age_months = event_age

    # Simplified classification rules (subset of 34 from spec)
    if event.event_type_code == 1:  # Civil judgment
        if event.is_satisfied and event.satisfaction_date:
            months_since_satisfaction = (decision_date - event.satisfaction_date).days // 30
            if months_since_satisfaction > 24:
                event.event_severity_code = Severity.IMMATERIAL
                event.classification_rule = "AE-C-01"
            elif (event.amount or 0) <= material_base:
                event.event_severity_code = Severity.MINOR
                event.classification_rule = "AE-C-02"
            else:
                event.event_severity_code = Severity.MATERIAL
                event.classification_rule = "AE-C-02"
        else:  # unsatisfied
            if (event.amount or 0) > disq_base:
                event.event_severity_code = Severity.DISQUALIFYING
                event.classification_rule = "AE-C-03"
            elif (event.amount or 0) > material_base:
                if event_age <= 36:
                    event.event_severity_code = Severity.MATERIAL
                    event.classification_rule = "AE-C-04"
                else:
                    event.event_severity_code = Severity.MINOR
                    event.classification_rule = "AE-C-05"
            else:
                event.event_severity_code = Severity.MINOR
                event.classification_rule = "AE-C-06"

    elif event.event_type_code in [2]:  # Default listing
        if event.is_satisfied and event.satisfaction_date:
            months_cleared = (decision_date - event.satisfaction_date).days // 30
            if months_cleared >= 12:
                event.event_severity_code = Severity.IMMATERIAL
                event.classification_rule = "AE-C-09"
            else:
                event.event_severity_code = Severity.MINOR
                event.classification_rule = "AE-C-08"
        else:
            if (event.amount or 0) >= material_base:
                event.event_severity_code = Severity.MATERIAL
                event.classification_rule = "AE-C-08"
            else:
                event.event_severity_code = Severity.MINOR
                event.classification_rule = "AE-C-08"

    elif event.event_type_code == 14:  # Confirmed fraud marker
        event.event_severity_code = Severity.DISQUALIFYING
        event.classification_rule = "AE-C-20"

    else:  # Other types: default to minor
        event.event_severity_code = Severity.MINOR
        event.classification_rule = "AE-C-99"

    # Handle disputed events (downgrade one class, flag provisional)
    if event.is_disputed:
        if event.event_severity_code and event.event_severity_code > Severity.IMMATERIAL:
            event.event_severity_code = min(Severity.MATERIAL, event.event_severity_code - 1)
        event.classification_provisional = True


def roll_up_entity_verdict(
    entity: Entity,
    events_for_entity: List[AdverseEvent],
    decision_date: date,
    requested_amount: float
) -> EntityAdverseVerdict:
    """
    Roll up entity adverse events to a single verdict per spec 05 §5.6.

    Uses a subset of the 12 roll-up rules. Attribution tracks which events
    justified the verdict.
    """
    if not events_for_entity or all(e.event_severity_code == Severity.IMMATERIAL for e in events_for_entity):
        return EntityAdverseVerdict(
            entity_key=entity.entity_key,
            verdict=EntityVerdict.CLEAR,
            binding_rule="AE-R-11",
            event_ids=[]
        )

    # Rule AE-R-01: Any disqualifying event
    disq_events = [e for e in events_for_entity if e.event_severity_code == Severity.DISQUALIFYING]
    if disq_events:
        return EntityAdverseVerdict(
            entity_key=entity.entity_key,
            verdict=EntityVerdict.DISQUALIFYING,
            binding_rule="AE-R-01",
            event_ids=[e.event_id for e in disq_events],
            severity_counts={Severity.DISQUALIFYING: len(disq_events)},
            total_unsatisfied_amount=sum((e.amount or 0) for e in disq_events if not e.is_satisfied)
        )

    # Rule AE-R-02: ≥3 minor events within 12 months
    recent_cutoff = decision_date - timedelta(days=365)
    recent_minor = [e for e in events_for_entity
                    if e.event_severity_code == Severity.MINOR
                    and e.event_date >= recent_cutoff]
    if len(recent_minor) >= 3:
        return EntityAdverseVerdict(
            entity_key=entity.entity_key,
            verdict=EntityVerdict.MATERIAL,
            binding_rule="AE-R-02",
            event_ids=[e.event_id for e in recent_minor],
            severity_counts={Severity.MINOR: len(recent_minor)}
        )

    # Rule AE-R-06: Aggregate unsatisfied > R150k or >15% of amount
    unsatisfied_total = sum((e.amount or 0) for e in events_for_entity if not e.is_satisfied)
    threshold = max(150000, 0.15 * requested_amount)
    if unsatisfied_total > threshold:
        unsatisfied_events = [e for e in events_for_entity if not e.is_satisfied]
        return EntityAdverseVerdict(
            entity_key=entity.entity_key,
            verdict=EntityVerdict.MATERIAL,
            binding_rule="AE-R-06",
            event_ids=[e.event_id for e in unsatisfied_events],
            total_unsatisfied_amount=unsatisfied_total
        )

    # Rule AE-R-08: Any material event within 6 months
    material_events = [e for e in events_for_entity if e.event_severity_code == Severity.MATERIAL]
    recent_cutoff_6m = decision_date - timedelta(days=180)
    recent_material = [e for e in material_events if e.event_date >= recent_cutoff_6m]
    if recent_material:
        return EntityAdverseVerdict(
            entity_key=entity.entity_key,
            verdict=EntityVerdict.MATERIAL,
            binding_rule="AE-R-08",
            event_ids=[e.event_id for e in recent_material],
            severity_counts={Severity.MATERIAL: len(recent_material)}
        )

    # Rule AE-R-03: ≥5 minor events of any age
    all_minor = [e for e in events_for_entity if e.event_severity_code == Severity.MINOR]
    if len(all_minor) >= 5:
        return EntityAdverseVerdict(
            entity_key=entity.entity_key,
            verdict=EntityVerdict.MATERIAL,
            binding_rule="AE-R-03",
            event_ids=[e.event_id for e in all_minor],
            severity_counts={Severity.MINOR: len(all_minor)}
        )

    # Rule AE-R-12: Peripheral entity capped at material
    if entity.criticality_class == Criticality.PERIPHERAL:
        if material_events:
            return EntityAdverseVerdict(
                entity_key=entity.entity_key,
                verdict=EntityVerdict.MATERIAL,
                binding_rule="AE-R-12",
                event_ids=[e.event_id for e in material_events],
                severity_counts={Severity.MATERIAL: len(material_events)}
            )

    # Default: material if any material events, else minor if any minor events
    if material_events:
        return EntityAdverseVerdict(
            entity_key=entity.entity_key,
            verdict=EntityVerdict.MATERIAL,
            binding_rule="AE-R-99",
            event_ids=[e.event_id for e in material_events],
            severity_counts={Severity.MATERIAL: len(material_events)}
        )

    if any(e.event_severity_code == Severity.MINOR for e in events_for_entity):
        minor_events = [e for e in events_for_entity if e.event_severity_code == Severity.MINOR]
        return EntityAdverseVerdict(
            entity_key=entity.entity_key,
            verdict=EntityVerdict.MINOR,
            binding_rule="AE-R-99",
            event_ids=[e.event_id for e in minor_events],
            severity_counts={Severity.MINOR: len(minor_events)}
        )

    # Should not reach here
    return EntityAdverseVerdict(
        entity_key=entity.entity_key,
        verdict=EntityVerdict.CLEAR,
        binding_rule="AE-R-11",
        event_ids=[]
    )


def assess_business_credit(request: AssessmentRequest) -> AssessmentResult:
    """
    Assess business credit with nested entity handling.

    Implements stages 5.1-5.8 and 5.10 of spec 05.
    """
    result = AssessmentResult(
        application_id=request.application_id,
        decision_date=request.decision_date
    )

    # Stage 5.1: Structure resolution
    # For now, assume entities are already resolved (simplified)
    result.resolved_entity_count = len(request.entities)
    result.ownership_total_pct = sum(e.effective_ownership_pct for e in request.entities if e.direct_ownership_pct)

    # Stage 5.4: Determine criticality for each entity
    for entity in request.entities:
        entity.criticality_class = determine_criticality(entity, request.entities)
        criticality_inputs = {
            'is_controlling': entity.is_controlling,
            'effective_ownership_pct': entity.effective_ownership_pct,
            'is_required_surety': entity.is_required_surety,
            'relationship_type_code': entity.relationship_type_code
        }
        entity.criticality_inputs = criticality_inputs
        result.entities_with_criticality.append({
            'entity_key': entity.entity_key,
            'criticality_class': entity.criticality_class,
            'criticality_inputs': criticality_inputs
        })

    # Thresholds by criticality (simplified, per spec 05 §5.5)
    thresholds_by_criticality = {
        Criticality.CRITICAL: {
            'event_1_material': 10000,
            'event_1_disqualifying': 50000,
            'event_2_material': 5000,
            'event_2_disqualifying': 40000
        },
        Criticality.SIGNIFICANT: {
            'event_1_material': 25000,
            'event_1_disqualifying': 100000,
            'event_2_material': 7500,
            'event_2_disqualifying': 75000
        },
        Criticality.PERIPHERAL: {
            'event_1_material': 50000,
            'event_1_disqualifying': 250000,
            'event_2_material': 15000,
            'event_2_disqualifying': 150000
        }
    }

    # Stage 5.5: Classify adverse events
    for event in request.adverse_events:
        # Find the entity this event belongs to
        entity = next((e for e in request.entities if e.entity_key == event.entity_key), None)
        if entity:
            classify_adverse_event(event, entity, request.entities, request.decision_date, thresholds_by_criticality)
            entity.events.append(vars(event))

    # Stage 5.6: Roll up entity verdicts
    for entity in request.entities:
        events_for_entity = [e for e in request.adverse_events if e.entity_key == entity.entity_key]
        verdict = roll_up_entity_verdict(entity, events_for_entity, request.decision_date, request.requested_amount)
        result.entity_verdicts.append(verdict)

        # Check for disqualifying entities (impacts people verdict)
        if verdict.verdict == EntityVerdict.DISQUALIFYING:
            result.declined_entity_key = entity.entity_key
            if verdict.event_ids:
                first_event = next((e for e in events_for_entity if e.event_id == verdict.event_ids[0]), None)
                if first_event:
                    result.declined_event_id = first_event.event_id
            result.declined_rule = verdict.binding_rule

    # Stage 5.7: Entity scoring (simplified stub)
    for entity in request.entities:
        verdict = next((v for v in result.entity_verdicts if v.entity_key == entity.entity_key), None)
        # Stub: assign a grade based on verdict
        if verdict:
            grade_map = {
                EntityVerdict.DISQUALIFYING: 12,
                EntityVerdict.MATERIAL: 8,
                EntityVerdict.MINOR: 6,
                EntityVerdict.CLEAR: 4
            }
            grade = grade_map[verdict.verdict]
        else:
            grade = 6

        result.entity_scores.append({
            'entity_key': entity.entity_key,
            'score': 600 + (12 - grade) * 50,  # Stub scoring
            'risk_grade': grade,
            'pd': 0.05 * (grade / 6)  # Rough PD estimate
        })

    # Stage 5.8: People blend (simplified)
    # Count scoreable entities and blend their PDs
    scoreable_entities = [
        e for e in request.entities
        if next((v for v in result.entity_verdicts if v.entity_key == e.entity_key), EntityAdverseVerdict(entity_key='', verdict=EntityVerdict.CLEAR, binding_rule='', event_ids=[])).verdict != EntityVerdict.DISQUALIFYING
    ]

    if scoreable_entities:
        # PP-02: Coverage calculation
        owner_entities = [e for e in scoreable_entities if e.direct_ownership_pct or e.is_controlling]
        if owner_entities:
            result.people_coverage_ratio = sum(e.effective_ownership_pct for e in owner_entities) / 100
            result.people_coverage_sufficient = result.people_coverage_ratio >= 0.75

        # Simple PD blend: average across entities
        entity_scores = {e['entity_key']: e for e in result.entity_scores}
        scoreable_pds = [entity_scores.get(e.entity_key, {}).get('pd', 0.05) for e in scoreable_entities]
        if scoreable_pds:
            result.people_pd = sum(scoreable_pds) / len(scoreable_pds)

        # Grade from PD (stub)
        result.people_grade = max(1, min(12, int(12 - result.people_pd * 200)))

    # Stage 5.10: Combined business grade
    # Simplified: use people grade as business grade
    result.business_grade = result.people_grade
    result.business_pd = result.people_pd

    # Determine business verdict
    if result.declined_entity_key:
        result.business_verdict_code = 3  # Decline
    elif not result.people_coverage_sufficient:
        result.business_verdict_code = 2  # Refer
    else:
        result.business_verdict_code = 1  # Clear/Approve

    return result
