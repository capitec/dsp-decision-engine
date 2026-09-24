"""
Business credit assessment pipeline (project 05).

Composes the nested entity assessment stages from assessment.py.
"""
from __future__ import annotations
from datetime import date
from typing import Optional, List
from decider import flow, param, missing_as

from assessment import (
    AssessmentRequest,
    Entity,
    AdverseEvent,
    assess_business_credit
)


def assess(
    application_id: int = param(1, ge=1),
    decision_date: date = missing_as(date.today()),
    product_code: int = param(50, ge=50, le=51),
    requested_amount: float = param(500000.0, ge=50000, le=10000000),
    requested_term_months: Optional[int] = param(60, ge=12, le=360),
    registration_number: Optional[str] = None,
    legal_form_code: int = param(3, ge=1, le=9),  # 3=private company
    registration_status_code: int = param(1, ge=1, le=7),  # 1=registered
    registration_date: Optional[date] = None,
    trading_since_date: Optional[date] = None,
    sector_code: int = param(100, ge=1, le=420),
    jurisdiction_code: int = param(1, ge=1, le=10),
    tax_reference: Optional[str] = None,
    tax_compliance_status_code: int = param(1, ge=1, le=4),  # 1=compliant
    vat_registration_number: Optional[str] = None,
    declared_annual_turnover: float = param(2000000.0, ge=0.0),
    facility_purpose_code: int = param(1, ge=1, le=24),
    security_offered: List[dict] = missing_as([]),

    # Entities (1..40 after resolution)
    entities_list: List[dict] = missing_as([]),

    # Adverse events
    adverse_events_list: List[dict] = missing_as([])
) -> dict:
    """
    Assess business credit with nested entity handling.

    Stages 5.1-5.10 of spec 05:
    - Structure resolution
    - Entity criticality
    - Adverse event classification
    - Entity verdict roll-up
    - Entity scoring
    - People blend
    - Combined business grade
    """

    # Build entity objects from input list
    entities = []
    for e_dict in entities_list:
        entity = Entity(
            entity_id=e_dict.get('entity_id', 0),
            entity_key=e_dict.get('entity_key', ''),
            is_natural_person=e_dict.get('is_natural_person', True),
            relationship_type_code=e_dict.get('relationship_type_code', 1),
            path=e_dict.get('path', []),
            depth=e_dict.get('depth', 1),
            direct_ownership_pct=e_dict.get('direct_ownership_pct'),
            effective_ownership_pct=e_dict.get('effective_ownership_pct', 0.0),
            is_controlling=e_dict.get('is_controlling', False),
            is_required_surety=e_dict.get('is_required_surety', False),
            date_of_birth=e_dict.get('date_of_birth'),
            screening_outcome_code=e_dict.get('screening_outcome_code', 1),
            screening_confidence=e_dict.get('screening_confidence'),
            deceased_flag=e_dict.get('deceased_flag'),
            identity_verification_code=e_dict.get('identity_verification_code', 1),
            residency_code=e_dict.get('residency_code')
        )
        entities.append(entity)

    # Build adverse event objects from input list
    events = []
    for ev_dict in adverse_events_list:
        event = AdverseEvent(
            event_id=ev_dict.get('event_id', 0),
            entity_key=ev_dict.get('entity_key', ''),
            event_type_code=ev_dict.get('event_type_code', 1),
            amount=ev_dict.get('amount'),
            event_date=ev_dict.get('event_date', date.today()),
            status_code=ev_dict.get('status_code', 1),
            is_disputed=ev_dict.get('is_disputed', False),
            is_satisfied=ev_dict.get('is_satisfied', False),
            satisfaction_date=ev_dict.get('satisfaction_date'),
            source_code=ev_dict.get('source_code', 1),
            duplicate_group_id=ev_dict.get('duplicate_group_id')
        )
        events.append(event)

    # Build request
    request = AssessmentRequest(
        application_id=application_id,
        client_id=None,
        decision_date=decision_date,
        product_code=product_code,
        requested_amount=requested_amount,
        requested_term_months=requested_term_months,
        registration_number=registration_number,
        legal_form_code=legal_form_code,
        registration_status_code=registration_status_code,
        registration_date=registration_date,
        trading_since_date=trading_since_date,
        sector_code=sector_code,
        jurisdiction_code=jurisdiction_code,
        tax_reference=tax_reference,
        tax_compliance_status_code=tax_compliance_status_code,
        vat_registration_number=vat_registration_number,
        declared_annual_turnover=declared_annual_turnover,
        facility_purpose_code=facility_purpose_code,
        security_offered=security_offered,
        entities=entities,
        adverse_events=events
    )

    # Run assessment
    result = assess_business_credit(request)

    # Format output
    return {
        'application_id': result.application_id,
        'decision_date': result.decision_date.isoformat(),
        'resolved_entity_count': result.resolved_entity_count,
        'structure_unresolved': result.structure_unresolved,
        'ownership_total_pct': round(result.ownership_total_pct, 2),
        'entities_with_criticality': result.entities_with_criticality,
        'entity_verdicts': [
            {
                'entity_key': v.entity_key,
                'verdict': v.verdict,
                'binding_rule': v.binding_rule,
                'event_ids': v.event_ids,
                'severity_counts': v.severity_counts,
                'total_unsatisfied_amount': round(v.total_unsatisfied_amount, 2)
            }
            for v in result.entity_verdicts
        ],
        'entity_scores': result.entity_scores,
        'people_pd': round(result.people_pd, 6),
        'people_grade': result.people_grade,
        'people_coverage_ratio': round(result.people_coverage_ratio, 4),
        'people_coverage_sufficient': result.people_coverage_sufficient,
        'business_pd': round(result.business_pd, 6),
        'business_grade': result.business_grade,
        'business_verdict_code': result.business_verdict_code,
        'declined_entity_key': result.declined_entity_key,
        'declined_event_id': result.declined_event_id,
        'declined_rule': result.declined_rule
    }


def build():
    """Build the business credit assessment pipeline."""
    return flow(assess, name="business_credit_assessment")
