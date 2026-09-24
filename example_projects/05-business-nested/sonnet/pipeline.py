"""Business credit granting over nested entities (spec 05): structure resolution,
per-entity criticality, adverse event classification, the entity adverse
roll-up, entity scoring, the people blend, the combined business grade, and a
single-lookup offer -- see NOTES.md for the exact slice and what was left out.

Thirteen stages in the spec; this pipeline wires stages 1 (`structure.py`), 4
(`structure.py`'s criticality), 5 (`events.py`), 6 (`rollup.py`), 7 (`scoring.py`),
8 (`blend.py`), 9 (`financial.py`, 3 ratios), 10 (`grade.py`, reduced overlay
stack), 12 (`pricing.py`, single lookup) and the sole-proprietor regulated
affordability call (`sole_proprietor.py`, §5.2/§5.12 item 7) into one `dag`.
"""
from __future__ import annotations

from decider import dag

from business_nested import blend, events, financial, grade, outcome, pricing, rollup, scoring, sole_proprietor
from business_nested import structure
from business_nested.overlays import EVENT_ADJUSTMENT_SET_ID, EVENT_THRESHOLD_OVERLAYS
from business_nested.reasons_unit import build_reasons_unit


def build():
    return dag(
        structure.resolve_structure,
        events.make_classify_step(EVENT_THRESHOLD_OVERLAYS, EVENT_ADJUSTMENT_SET_ID),
        rollup.rollup_step,
        scoring.score_entities_step,
        blend.people_component_step,
        financial.build_financial_unit(),
        grade.build_grade_unit(),
        pricing.build_pricing_unit(),
        sole_proprietor.sole_proprietor_step,
        build_reasons_unit(),
        name="business_nested",
    ).emit(
        # Structure (§5.1, §5.4)
        "resolved_entity_count", "structure_unresolved", "ownership_reconciliation_pct",
        "entity_id", "entity_key", "entity_is_natural_person", "entity_relationship_type_code",
        "entity_effective_ownership_pct", "entity_is_controlling", "entity_is_owner", "entity_path_count",
        "entity_criticality_class", "entity_bureau_score", "entity_months_on_record",
        "entity_worst_delinquency_months",
        # Events (§5.5)
        "ev_entity_id", "ev_event_id", "ev_severity_code", "ev_severity_code_unadjusted", "ev_threshold_used",
        "ev_threshold_base", "ev_overlay_ids", "ev_classification_provisional", "ev_age_months",
        "ev_threshold_cell_id",
        # Roll-up (§5.6)
        "entity_verdict_code", "entity_verdict_code_unadjusted", "entity_verdict_binding_rule",
        "entity_verdict_overlay_sensitive", "attr_entity_id", "attr_event_id", "attr_rule_id",
        # Entity scoring (§5.7)
        "entity_score", "entity_score_unadjusted", "entity_pd", "entity_pd_unadjusted", "entity_grade",
        "entity_adjustments_applied", "entity_scoring_adjustment_set_id",
        # People blend (§5.8)
        "people_pd", "people_pd_unadjusted", "people_grade", "people_grade_unadjusted", "people_weight_vector",
        "insufficient_people_coverage", "no_scoreable_natural_person", "business_decline_from_entity",
        "business_decline_entity_id", "people_pd_overlay_contribution", "people_cap_rule",
        # Financial (§5.9)
        "interest_cover", "current_ratio", "gearing", "financial_confidence_code", "financial_pd",
        "financial_grade",
        # Combined grade (§5.10)
        "probability_of_default_before_overlay", "probability_of_default", "probability_of_default_unadjusted",
        "adjustment_set_id", "adjustments_applied", "risk_grade", "business_risk_grade_cell_id",
        # Pricing (§5.12)
        "offered_amount", "binding_constraint_code", "term_months", "nominal_annual_rate", "rate_card_cell_id",
        "initiation_fee", "monthly_service_fee", "instalment", "total_cost_of_credit", "effective_annual_rate",
        # Sole proprietor (§5.2, §5.12 item 7)
        "sole_proprietor_assessment_ran", "sole_proprietor_affordability_verdict_code",
        "sole_proprietor_max_affordable_instalment",
        # Outcome (§7.1, §9.2)
        "outcome_code", "decline_reason_codes", "primary_reason_code", "reason_registry_version",
        "attributing_entity_id", "attributing_event_ids", "business_decline_binding_rule_actual",
    )
