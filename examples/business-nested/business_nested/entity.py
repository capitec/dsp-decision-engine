"""One entity, assessed: its events classified and rolled up to a verdict, and the entity scored.

`entity_unit()` is a flow over the fields of a single resolved entity, so it can be run on one entity alone
or on every entity of an application with `each`. `entities_unit()` is the structure step followed by that `each`.
"""
from __future__ import annotations

from decider import dag, each, flow

from business_nested import events, rollup, scoring, structure


def entity_unit():
    """Classify the entity's `adverse_events` (written back in place), roll them up, and score the entity."""
    return dag(
        # In place, so each event appears once in the audit record, with its classification beside its facts.
        each("adverse_events", events.event_unit(), name="events"),
        rollup.adverse_verdict, scoring.entity_scoring_unit(),
        name="entity")


def entities_unit():
    """Resolve the request's `entities`, then assess each one: writes `assessed_entities`."""
    return flow(structure.resolved_entities,
                each("resolved_entities", entity_unit(), name="entities", output="assessed_entities"),
                name="entities")
