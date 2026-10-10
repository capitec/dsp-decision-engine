"""One entity, assessed: its events classified and rolled up to a verdict, and the entity scored.

`entity_unit()` is a flow over the fields of a single resolved entity, so it can be run on one entity alone
or on every entity of an application with `each`. `entities_unit()` is the structure step followed by that `each`.
"""
from __future__ import annotations

from decikit import dag, each, flow

from business_nested import events, rollup, scoring, structure


def entity_unit(event_overlays=None, score_adjustments=None):
    """Classify the entity's `adverse_events` (written back in place), roll them up, and score the entity.

    The two arguments are the `event_threshold_overlays` and `entity_score_adjustments` documents; by default the ones
    in `configs/0.2.0`.
    """
    return dag(
        # In place, so each event appears once in the audit record, with its classification beside its facts.
        each("adverse_events", events.event_unit(event_overlays), name="events"),
        rollup.adverse_verdict, scoring.entity_scoring_unit(score_adjustments),
        name="entity")


def entities_unit(event_overlays=None, score_adjustments=None):
    """Resolve the request's `entities`, then assess each one: writes `assessed_entities`."""
    return flow(structure.resolved_entities,
                each("resolved_entities", entity_unit(event_overlays, score_adjustments), name="entities",
                     output="assessed_entities"),
                name="entities")
