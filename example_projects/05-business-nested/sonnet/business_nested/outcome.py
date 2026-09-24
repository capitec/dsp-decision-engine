"""Final stage -- outcome, reason codes and the headline attribution (spec 05 §7.1,
§9.2). `attributing_entity_id`/`attributing_event_ids` is "the output the project
exists for" (spec 05 §7.1): pulled from `rollup.py`'s flat attribution table
(`attr_entity_id`/`attr_event_id`), filtered to the one entity that bound the
business decline -- spec 05 §9.2's internal attribution triple (entity, event set,
rule).

Two reason sets are *not* built here (spec 05 §9.2's internal-vs-communicable
split) -- out of this slice; every reason code below is treated as internal.
"""
from __future__ import annotations  # noqa: F404 (kept first; needed for `int | None` in tuple annotations below)

from decider import step

from business_nested import vocab
from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry

R_STRUCTURE_UNRESOLVED = 5001
R_INSUFFICIENT_PEOPLE_COVERAGE = 5401
R_NO_SCOREABLE_NATURAL_PERSON = 5403
R_ENTITY_DISQUALIFYING = 5402
R_SOLE_PROPRIETOR_STATUTORY_FAIL = 5601
R_BUSINESS_GRADE_BELOW_CUTOFF = 5602

REASON_REGISTRY = ReasonCodeRegistry("business-nested-reasons-2026.09", [
    ReasonCode(R_STRUCTURE_UNRESOLVED, 5, "Disclosed ownership structure could not be resolved", True),
    ReasonCode(R_INSUFFICIENT_PEOPLE_COVERAGE, 20, "Scoreable ownership below the 75% coverage requirement", True),
    ReasonCode(R_NO_SCOREABLE_NATURAL_PERSON, 25, "No scoreable natural person behind the business", True),
    ReasonCode(R_ENTITY_DISQUALIFYING, 1, "An entity behind the business carries a disqualifying adverse record",
               True),
    ReasonCode(R_SOLE_PROPRIETOR_STATUTORY_FAIL, 2, "Statutory affordability assessment failed", True),
    ReasonCode(R_BUSINESS_GRADE_BELOW_CUTOFF, 30, "Combined business grade below the minimum acceptable grade", False),
])

BUSINESS_GRADE_DECLINE_CUTOFF = 11
BUSINESS_GRADE_REFER_CUTOFF = 9
SOLE_PROPRIETOR_STATUTORY_FAIL = 3  # project 02's `assessment.verdict.FAIL` (PASS=1, MARGINAL=2, FAIL=3, INDETERMINATE=4)


def business_outcome(
    structure_unresolved: bool, insufficient_people_coverage: bool, no_scoreable_natural_person: bool,
    business_decline_from_entity: bool, business_decline_entity_id: int | None, risk_grade: int,
    sole_proprietor_assessment_ran: bool, sole_proprietor_affordability_verdict_code: int,
    attr_entity_id: list[int], attr_event_id: list[int], entity_id: list[int],
    entity_verdict_binding_rule: list[str],
) -> tuple[int, list[int], int | None, list[int], str]:
    """(outcome_code, decline_reason_codes, attributing_entity_id, attributing_event_ids,
    primary_binding_rule). Ranking through the registry happens in a following step
    (`REASON_REGISTRY.resolve_step()`, spec 09 §5.15 item 11)."""
    reasons: list[int] = []
    attributing_entity_id = None
    attributing_event_ids: list[int] = []
    binding_rule = ""

    if structure_unresolved:
        return vocab.OUTCOME_REFER, [R_STRUCTURE_UNRESOLVED], None, [], ""

    if business_decline_from_entity:
        # `list(...)`, not the numpy array decider hands a plain step() for a list-typed
        # input in its row loop -- `.index()` doesn't exist on `numpy.ndarray` (00
        # NOTES.md "Framework friction" 4.4, hit here for the same reason).
        entity_id = list(entity_id)
        entity_verdict_binding_rule = list(entity_verdict_binding_rule)
        reasons.append(R_ENTITY_DISQUALIFYING)
        attributing_entity_id = business_decline_entity_id
        attributing_event_ids = sorted(
            ev for eid, ev in zip(attr_entity_id, attr_event_id) if eid == business_decline_entity_id
        )
        idx = entity_id.index(business_decline_entity_id) if business_decline_entity_id in entity_id else -1
        binding_rule = entity_verdict_binding_rule[idx] if idx >= 0 else ""
        return vocab.OUTCOME_DECLINE, reasons, attributing_entity_id, attributing_event_ids, binding_rule

    if insufficient_people_coverage:
        return vocab.OUTCOME_REFER, [R_INSUFFICIENT_PEOPLE_COVERAGE], None, [], ""

    if no_scoreable_natural_person:
        return vocab.OUTCOME_REFER, [R_NO_SCOREABLE_NATURAL_PERSON], None, [], ""

    if sole_proprietor_assessment_ran and sole_proprietor_affordability_verdict_code == SOLE_PROPRIETOR_STATUTORY_FAIL:
        return vocab.OUTCOME_DECLINE, [R_SOLE_PROPRIETOR_STATUTORY_FAIL], None, [], ""

    if risk_grade >= BUSINESS_GRADE_DECLINE_CUTOFF:
        return vocab.OUTCOME_DECLINE, [R_BUSINESS_GRADE_BELOW_CUTOFF], None, [], ""

    if risk_grade >= BUSINESS_GRADE_REFER_CUTOFF:
        return vocab.OUTCOME_REFER, [], None, [], ""

    return vocab.OUTCOME_APPROVE, [], None, [], ""


_OUTCOME_OUTPUTS = (
    "outcome_code", "decline_reason_codes", "attributing_entity_id", "attributing_event_ids",
    "business_decline_binding_rule_actual",
)

business_outcome_step = step(business_outcome, name="business_outcome", outputs=_OUTCOME_OUTPUTS)
