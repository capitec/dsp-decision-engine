"""Questions asked of a decision after the fact: what if one event were not there, and what if one entity were corrected.

Both run the same flows as the decision itself, so an answer cannot drift from it. Records are the pipeline's
input with its dates already parsed, exactly as `Executable.run` takes them.
"""
from __future__ import annotations

import copy
from functools import cache

from decikit import Executable, InterpretedEngine, flow

from business_nested import structure
from business_nested.entity import entity_unit
from business_nested.pipeline import decision_unit


def without_event(record: dict, event_id: int) -> dict:
    """A copy of `record` with the event removed from every path it was reported on; `KeyError` if there is none."""
    changed = copy.deepcopy(record)
    found = False
    for entity in changed["entities"]:
        kept = [e for e in entity.get("adverse_events", []) if e["event_id"] != event_id]
        found |= len(kept) != len(entity.get("adverse_events", []))
        entity["adverse_events"] = kept
    if not found:
        raise KeyError(f"no adverse event {event_id} in the record")
    return changed


def single_event_counterfactual(executable: Executable, record: dict, event_id: int, params: dict | None = None) -> dict:
    """How one event was classified, and what the decision would have been without it.

    `executable` is the bound `build()` pipeline. Works for any event, not only the one a decline rests on.
    Example: `single_event_counterfactual(exe, record, 41)["outcome_before"]` is 4 and `["outcome_after"]` 1.
    """
    before, after = executable.run(record, params), executable.run(without_event(record, event_id), params)
    entity = next(e for e in before["assessed_entities"] if any(ev["event_id"] == event_id for ev in e["adverse_events"]))
    event = next(ev for ev in entity["adverse_events"] if ev["event_id"] == event_id)
    verdict_after = next(e for e in after["assessed_entities"] if e["entity_id"] == entity["entity_id"])
    answer = {"event_id": event_id, "entity_id": entity["entity_id"], "criticality_class": entity["criticality_class"]}
    answer |= {name: event[name] for name in (
        "event_severity_code", "event_severity_code_unadjusted", "material_threshold", "material_threshold_base",
        "disqualifying_threshold", "disqualifying_threshold_base", "threshold_overlay_ids", "threshold_cell_id")}
    answer |= {"entity_verdict_before": entity["adverse_verdict_code"],
               "entity_verdict_after": verdict_after["adverse_verdict_code"]}
    for name in ("people_grade", "risk_grade", "offered_amount", "outcome_code"):
        answer |= {f"{name}_before": before[name], f"{name}_after": after[name]}
    return answer


@cache
def _resolver() -> Executable:
    return InterpretedEngine().bind(flow(structure.resolved_entities))


@cache
def _entity_executable() -> Executable:
    return InterpretedEngine().bind(entity_unit())


@cache
def _decision_executable() -> Executable:
    return InterpretedEngine().bind(decision_unit())


def reassess_entity(record: dict, original: dict, entity_id: int) -> dict:
    """The decision for a corrected `record`, assessing only the entity `entity_id` again.

    `original` is the earlier result for the same application. Every other entity keeps its assessed values; the
    people blend, grade, price and outcome are run again over them, with the default policy. The result is the one
    a full run of `build()` gives when nothing but that entity changed. `KeyError` if `record` has an entity that
    `original` lacks.
    """
    assessed_before = {e["entity_id"]: e for e in original["assessed_entities"]}
    assessed = [_entity_executable().run(e) if e["entity_id"] == entity_id else assessed_before[e["entity_id"]]
                for e in _resolver().run(record)["resolved_entities"]]
    return _decision_executable().run({**record, "assessed_entities": assessed})
