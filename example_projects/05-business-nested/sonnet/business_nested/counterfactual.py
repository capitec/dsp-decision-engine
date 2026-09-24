"""Single-event counterfactuals (spec 05 §9.4) and the partial one-entity
re-assessment (spec 05 H5, acceptance §10 item 11; SCOPE.md: "because 11
depends on it", spec 11 §5.17.5).

Both are built on the same pure functions the pipeline's `frame_step`s already
wrap (`structure.resolve_entities`, `events.classify_events`,
`rollup.rollup_entities`, `scoring.score_entities`) -- no second implementation
of the classification or roll-up rules, exactly the "same implementation"
requirement 00's `core.adjustments` module states for its own stack-on/off runs.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

from decider import Engine

from business_nested import blend, events, grade as grade_mod, outcome, pricing, rollup, scoring, structure
from business_nested.overlays import EVENT_ADJUSTMENT_SET_ID, EVENT_THRESHOLD_OVERLAYS


_PIPELINE_ENGINE: Engine | None = None


def _load_own_pipeline_build():
    """`pipeline.py` sits one directory above this package (BRIEF's own layout), and every
    project in this set uses the same filename for its entry point -- `import pipeline`
    here would be exactly as fragile as `sole_proprietor.py`'s note on consuming project
    02's `pipeline.py` by name. Loading it by its own known path (relative to this file,
    not `sys.path` order) sidesteps that entirely for *this* project's own entry point."""
    path = Path(__file__).resolve().parent.parent / "pipeline.py"
    spec = importlib.util.spec_from_file_location("business_nested._own_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build


def _pipeline_engine() -> Engine:
    global _PIPELINE_ENGINE
    if _PIPELINE_ENGINE is None:
        _PIPELINE_ENGINE = Engine().bind(_load_own_pipeline_build()())
    return _PIPELINE_ENGINE


# --- §9.4 single-event counterfactual --------------------------------------------------

def without_event(record: dict, event_id: int) -> dict:
    """A deep copy of `record` with one event removed from wherever it sits."""
    modified = copy.deepcopy(record)
    for entity in modified["entities"]:
        entity["adverse_events"] = [ev for ev in entity.get("adverse_events", []) if ev["event_id"] != event_id]
    return modified


def single_event_counterfactual(record: dict, event_id: int) -> dict:
    """The four answers spec 05 §9.4 requires: how the event classified, what the
    entity verdict/people grade/business grade/offer would have been without it --
    all on the artefacts in force at `record`'s own `decision_date` (the same
    pipeline, the same overlay stack; nothing here re-derives against "today")."""
    engine = _pipeline_engine()
    before = engine.score(record, {})
    after = engine.score(without_event(record, event_id), {})

    entity_of_event = None
    for eid, ev_id in zip(before["ev_entity_id"], before["ev_event_id"]):
        if ev_id == event_id:
            entity_of_event = eid
            break
    idx_before = list(before["entity_id"]).index(entity_of_event) if entity_of_event is not None else None
    idx_after = list(after["entity_id"]).index(entity_of_event) if entity_of_event is not None else None

    return {
        "event_id": event_id, "entity_id": entity_of_event,
        "entity_verdict_before": before["entity_verdict_code"][idx_before] if idx_before is not None else None,
        "entity_verdict_after": after["entity_verdict_code"][idx_after] if idx_after is not None else None,
        "people_grade_before": before["people_grade"], "people_grade_after": after["people_grade"],
        "risk_grade_before": before["risk_grade"], "risk_grade_after": after["risk_grade"],
        "offered_amount_before": before["offered_amount"], "offered_amount_after": after["offered_amount"],
        "outcome_before": before["outcome_code"], "outcome_after": after["outcome_code"],
    }


# --- H5 partial one-entity re-assessment ------------------------------------------------

def _splice(original: list, start: int, count: int, replacement: list) -> list:
    return original[:start] + replacement + original[start + count:]


def partial_reassess_entity(record: dict, entity_id: int, refreshed_entity: dict, original_result: dict) -> dict:
    """Re-derives **one** entity (its criticality, its events' classification, its
    roll-up verdict and its score) and re-runs only the cheap, non-ragged
    application-level stages (blend, grade, pricing, outcome) on the spliced result --
    skipping the event-threshold table lookup and the scorecard/calibration/risk-grade
    lookup for every *other* entity, which is where this stage's cost actually sits
    (spec 05 §13 Q8). `original_result` is the prior full `pipeline.build()` output for
    this same `record` -- financial assessment doesn't depend on the entity collection
    at all, so it is reused unchanged rather than recomputed (the same "don't re-run what
    the change can't have affected" principle applied one level up). Returns a result
    dict comparable field-for-field against a full re-run (the equivalence test SCOPE.md
    and spec 11 §5.17.5 ask for)."""
    decision_date = record["decision_date"]
    entities = copy.deepcopy(record["entities"])
    entities = [refreshed_entity if e["entity_id"] == entity_id else e for e in entities]

    # Structure resolution is pure Python and O(entities) -- cheap enough to always
    # re-run in full; it is the two nested-`Engine` calls (event thresholds, entity
    # scoring) this function actually economises on.
    resolved = structure.resolve_entities(entities, decision_date)

    idx = resolved["entity_id"].index(entity_id)
    # Positions of this entity's events in the freshly-resolved (sorted) event arrays.
    event_positions = [i for i, eid in enumerate(resolved["ev_entity_id"]) if eid == entity_id]

    provenance = {"sector_code": record["sector_code"]}
    only_this_entity_events = {
        k: [resolved[k][i] for i in event_positions]
        for k in ("ev_entity_id", "ev_event_id", "ev_event_type_code", "ev_amount", "ev_status_is_active",
                   "ev_is_disputed", "ev_is_satisfied")
    }
    classified = events.classify_events(
        only_this_entity_events["ev_entity_id"], only_this_entity_events["ev_event_id"],
        only_this_entity_events["ev_event_type_code"], only_this_entity_events["ev_amount"],
        only_this_entity_events["ev_status_is_active"], only_this_entity_events["ev_is_disputed"],
        only_this_entity_events["ev_is_satisfied"], [entity_id], [resolved["entity_criticality_class"][idx]],
        decision_date, EVENT_THRESHOLD_OVERLAYS, EVENT_ADJUSTMENT_SET_ID, True, provenance,
    )

    rolled = rollup.rollup_entities(
        [entity_id], [resolved["entity_criticality_class"][idx]], record["requested_amount"],
        only_this_entity_events["ev_entity_id"], only_this_entity_events["ev_event_id"],
        classified["ev_severity_code"], only_this_entity_events["ev_amount"],
        only_this_entity_events["ev_is_satisfied"],
        [resolved.get("ev_age_months", [0] * len(resolved["ev_entity_id"]))[i] for i in event_positions],
        only_this_entity_events["ev_event_type_code"],
    )

    scored = scoring.score_entities(
        [entity_id], [resolved["entity_is_natural_person"][idx]], [resolved["entity_bureau_score"][idx]],
        [resolved["entity_months_on_record"][idx]], [resolved["entity_worst_delinquency_months"][idx]],
        [resolved["entity_effective_ownership_pct"][idx]], decision_date,
    )

    # Now assemble the *full* entity table: every other entity keeps its already-known
    # verdict/score from `original_result` (unchanged inputs -> unchanged classification
    # and score, by construction, so they are never re-run); only this entity's row --
    # at position `idx` in the freshly re-sorted, still entity_id-ordered arrays -- is
    # replaced with what was actually recomputed above.
    original_by_id = dict(zip(
        list(original_result["entity_id"]),
        zip(list(original_result["entity_pd"]), list(original_result["entity_pd_unadjusted"]),
            list(original_result["entity_grade"]), list(original_result["entity_verdict_code"]),
            list(original_result["entity_verdict_binding_rule"])),
    ))

    entity_pd, entity_pd_unadj, entity_grade, entity_verdict, entity_rule = [], [], [], [], []
    for eid in resolved["entity_id"]:
        if eid == entity_id:
            entity_pd.append(scored["entity_pd"][0])
            entity_pd_unadj.append(scored["entity_pd_unadjusted"][0])
            entity_grade.append(scored["entity_grade"][0])
            entity_verdict.append(rolled["entity_verdict_code"][0])
            entity_rule.append(rolled["entity_verdict_binding_rule"][0])
        else:
            pd, pd_unadj, grade, verdict, rule = original_by_id[eid]
            entity_pd.append(pd)
            entity_pd_unadj.append(pd_unadj)
            entity_grade.append(grade)
            entity_verdict.append(verdict)
            entity_rule.append(rule)

    full_entity = {
        "entity_id": resolved["entity_id"], "entity_is_owner": resolved["entity_is_owner"],
        "entity_is_controlling": resolved["entity_is_controlling"],
        "entity_relationship_type_code": resolved["entity_relationship_type_code"],
        "entity_effective_ownership_pct": resolved["entity_effective_ownership_pct"],
        "entity_criticality_class": resolved["entity_criticality_class"],
        "entity_verdict_code": entity_verdict, "entity_verdict_binding_rule": entity_rule,
        "entity_pd": entity_pd, "entity_pd_unadjusted": entity_pd_unadj, "entity_grade": entity_grade,
    }

    people = blend.people_component(
        full_entity["entity_id"], full_entity["entity_is_owner"], full_entity["entity_is_controlling"],
        full_entity["entity_relationship_type_code"], full_entity["entity_effective_ownership_pct"],
        full_entity["entity_criticality_class"], full_entity["entity_verdict_code"],
        full_entity["entity_verdict_code"], full_entity["entity_pd"], full_entity["entity_pd_unadjusted"],
        full_entity["entity_grade"],
    )

    business_pd_before = grade_mod.combined_pd_before_overlay(original_result["financial_pd"], people["people_pd"])
    overlay_result = grade_mod._BUSINESS_OVERLAYS.apply_stack(
        "probability_of_default", business_pd_before, {"product_code": record["product_code"]},
        decision_date, "AS-05-BIZ-2026.09", True,
    )
    business_grade = _business_grade(overlay_result.adjusted_value, record["product_code"])

    return {
        "entity_id": full_entity["entity_id"], "entity_verdict_code": full_entity["entity_verdict_code"],
        "people_grade": people["people_grade"], "people_pd": people["people_pd"],
        "business_decline_from_entity": people["business_decline_from_entity"],
        "business_decline_entity_id": people["business_decline_entity_id"],
        "risk_grade": business_grade, "probability_of_default": overlay_result.adjusted_value,
    }


def _business_grade(pd: float, product_code: int) -> int:
    cuts = grade_mod._BUSINESS_BOUNDARIES[product_code]
    g = 1
    for cut in cuts:
        if pd < cut:
            return g
        g += 1
    return g
