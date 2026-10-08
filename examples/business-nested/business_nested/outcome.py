"""The outcome: approve, refer or decline, every reason that applies, and which entity and events a decline rests on."""
from __future__ import annotations

from decider import allow_fallback, flow, param, step

from assessment.verdict import FAIL, INDETERMINATE, MARGINAL
from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry
from credit_core.vocab import OUTCOME_APPROVE, OUTCOME_DECLINE, OUTCOME_REFER

R_STRUCTURE_UNRESOLVED = 5001
R_INSUFFICIENT_PEOPLE_COVERAGE = 5401
R_ENTITY_DISQUALIFYING = 5402
R_NO_SCOREABLE_NATURAL_PERSON = 5403
R_DISPUTED_EVENT = 5404
R_SOLE_PROPRIETOR_STATUTORY_FAIL = 5601
R_BUSINESS_GRADE_BELOW_CUTOFF = 5602
R_BUSINESS_GRADE_IN_REFERRAL_BAND = 5603
R_SOLE_PROPRIETOR_NOT_ESTABLISHED = 5604

REASON_REGISTRY = ReasonCodeRegistry("business-nested-reasons-2026.09", [
    ReasonCode(R_ENTITY_DISQUALIFYING, 1, "An entity behind the business carries a disqualifying adverse record", True),
    ReasonCode(R_SOLE_PROPRIETOR_STATUTORY_FAIL, 2, "Statutory affordability assessment failed", True),
    ReasonCode(R_DISPUTED_EVENT, 3, "The decline rests on an adverse event that is disputed", True),
    ReasonCode(R_STRUCTURE_UNRESOLVED, 5, "Disclosed ownership structure could not be resolved", True),
    ReasonCode(R_INSUFFICIENT_PEOPLE_COVERAGE, 20, "Scoreable ownership below the coverage requirement", True),
    ReasonCode(R_SOLE_PROPRIETOR_NOT_ESTABLISHED, 22, "Statutory affordability could not be established", True),
    ReasonCode(R_NO_SCOREABLE_NATURAL_PERSON, 25, "No scoreable natural person behind the business", True),
    ReasonCode(R_BUSINESS_GRADE_BELOW_CUTOFF, 30, "Business grade below the minimum acceptable grade", False),
    ReasonCode(R_BUSINESS_GRADE_IN_REFERRAL_BAND, 35, "Business grade in the referral band", False),
])


@allow_fallback
@step(outputs=("outcome_code", "decline_reason_codes", "attributing_entity_id", "attributing_event_ids",
               "attributing_rule_id"))
def business_outcome(
    structure_unresolved: bool, insufficient_people_coverage: bool, no_scoreable_natural_person: bool,
    business_decline_from_entity: bool, business_decline_entity_id: int | None, business_decline_from_disputed_event: bool,
    risk_grade: int, sole_proprietor_assessment_ran: bool, sole_proprietor_affordability_verdict_code: int,
    assessed_entities: list[dict],
    decline_cutoff_grade: int = param(11, ge=1, le=12), refer_cutoff_grade: int = param(9, ge=1, le=12),
) -> tuple[int, list[int], int | None, list[int], str]:
    """The outcome and every reason that fired (a referral's too), unranked.

    An unresolved structure is always a referral. A decline resting only on a disputed event is a referral that
    names the dispute. When an entity's record is the reason, its id, the events and the rule are reported.
    """
    if structure_unresolved:
        return OUTCOME_REFER, [R_STRUCTURE_UNRESOLVED], None, [], ""
    declines = [code for fired, code in (
        (business_decline_from_entity, R_ENTITY_DISQUALIFYING),
        (sole_proprietor_assessment_ran and sole_proprietor_affordability_verdict_code == FAIL, R_SOLE_PROPRIETOR_STATUTORY_FAIL),
        (risk_grade >= decline_cutoff_grade, R_BUSINESS_GRADE_BELOW_CUTOFF)) if fired]
    refers = [code for fired, code in (
        (insufficient_people_coverage, R_INSUFFICIENT_PEOPLE_COVERAGE),
        (no_scoreable_natural_person, R_NO_SCOREABLE_NATURAL_PERSON),
        (sole_proprietor_assessment_ran and sole_proprietor_affordability_verdict_code in (MARGINAL, INDETERMINATE),
         R_SOLE_PROPRIETOR_NOT_ESTABLISHED),
        (refer_cutoff_grade <= risk_grade < decline_cutoff_grade, R_BUSINESS_GRADE_IN_REFERRAL_BAND)) if fired]
    if declines == [R_ENTITY_DISQUALIFYING] and business_decline_from_disputed_event:
        declines, refers = [], [R_DISPUTED_EVENT, *refers]
    reasons = declines + refers
    entity = next((e for e in assessed_entities if e["entity_id"] == business_decline_entity_id), None)
    rests_on_entity = entity is not None and {R_ENTITY_DISQUALIFYING, R_DISPUTED_EVENT} & set(reasons)
    outcome = OUTCOME_DECLINE if declines else OUTCOME_REFER if refers else OUTCOME_APPROVE
    return (outcome, reasons, entity["entity_id"] if rests_on_entity else None,
            entity["adverse_event_ids"] if rests_on_entity else [], entity["adverse_binding_rule"] if rests_on_entity else "")


# `decline_reason_codes` is written as fired, then ranked, so the two steps form a flow.
outcome_unit = flow(business_outcome, REASON_REGISTRY.resolve_step(), name="outcome")
