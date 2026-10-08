"""The business credit decision: structure, entities, people, financials, grade, price and outcome in one pipeline.

`decider build` and `decider serve` use `build`. The two halves are public for callers that re-run one of them:
`entities_unit()` assesses each entity, `decision_unit()` is everything that reads the assessed entities.
"""
from __future__ import annotations

from decider import dag, flow

from business_nested import blend, entity, financial, grade, outcome, pricing, sole_proprietor, structure

# What the decision unit writes that the caller sees; the entity-level detail is inside `assessed_entities`.
DECISION_OUTPUTS = (
    "resolved_entity_count", "ownership_reconciliation_pct", "structure_unresolved",
    "people_pd", "people_pd_unadjusted", "people_pd_overlay_contribution", "people_weights", "people_grade_blended",
    "people_grade_unadjusted", "people_grade", "people_cap_rule", "insufficient_people_coverage",
    "no_scoreable_natural_person", "business_decline_from_entity", "business_decline_entity_id",
    "business_decline_from_disputed_event",
    "interest_cover", "current_ratio", "gearing", "financial_confidence_code", "financial_score", "financial_pd",
    "financial_grade",
    "probability_of_default_before_overlay", "probability_of_default", "probability_of_default_unadjusted",
    "adjustment_set_id", "adjustments_applied", "risk_grade", "risk_grade_cell_id", "risk_grade_unadjusted",
    "max_amount", "appetite_cell_id", "offered_amount", "binding_constraint_code", "term_months",
    "nominal_annual_rate", "rate_card_cell_id", "initiation_fee", "monthly_service_fee", "instalment",
    "total_cost_of_credit", "effective_annual_rate",
    "sole_proprietor_assessment_ran", "sole_proprietor_affordability_verdict_code",
    "sole_proprietor_max_affordable_instalment",
    "outcome_code", "decline_reason_codes", "primary_reason_code", "reason_registry_version",
    "attributing_entity_id", "attributing_event_ids", "attributing_rule_id",
)


def decision_unit():
    """Everything after the entities: reads `assessed_entities` and the business's own fields."""
    return dag(
        structure.structure_check, blend.people_unit(), financial.financial_unit(), grade.business_grade_unit(),
        pricing.pricing_unit(), sole_proprietor.sole_proprietor_unit(), outcome.outcome_unit,
        name="decision",
    ).emit(*DECISION_OUTPUTS)


def build():
    """The whole decision, as `decider serve` runs it and as other projects embed it."""
    # workaround: a dag counts the field names inside an `each` as names the `each` writes, and they clash with
    # the decision's (`segment_code`, `anchor`, ...), so the two halves are a flow, in written order.
    return flow(entity.entities_unit(), decision_unit(), name="business_credit").emit("assessed_entities", *DECISION_OUTPUTS)
