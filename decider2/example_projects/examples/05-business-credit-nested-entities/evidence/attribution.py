"""The attribution spine -- how a business decline names an entity and an event.

s7.1 calls `attributing_entity_id` and `attributing_event_ids` "the output the
project exists for". s10 acceptance criterion 1 requires zero exceptions across
5 000 applications.

--------------------------------------------------------------------------
THE CLAIM
--------------------------------------------------------------------------
Nothing in this file assembles an attribution. It declares a JOIN over
attributions each grain already recorded about itself.

Every `verdict(...)` in this project emits four values at its own grain:

    <name>_verdict            the outcome
    <name>_binding_rule_id    which rule governed
    <name>_fired_rule_mask    which rules fired at all (collect="all")
    <name>_witness            which CHILD rows satisfied the binding rule

and every `Gather` fold emits a witness alongside its quantity. The spine is
those values joined by grain key. No level knows about the level below it, and
adding a third level (change scenario 13) adds a row to the join, not a rewrite.

Spec s13 Q4 -- "does the framework carry attribution upward, or does each
project rebuild it -- and the previous generation's answer was that each project
rebuilt it, badly, and one of them dropped it entirely" -- is answered by making
the witness a property of the fold rather than a discipline of the author.
"""

from decider2 import module, step, attribution_spine, level, resolve_witness
from grains import Application, Entity, Event


# --------------------------------------------------------------------------
# The declaration. Three levels, each naming the grain it attributes to and the
# value that carries the witness.
# --------------------------------------------------------------------------
SPINE = attribution_spine(
    name="business_decline_attribution",
    levels=[
        level(Application, "people_caps",               names=Entity),
        level(Application, "business_disqualification", names=None),  # no entity
        level(Entity,      "entity_adverse_verdict",    names=Event),
        level(Entity,      "entity_disqualification",   names=None),
        level(Event,       "classify_event",            names=None),
    ],
    identity={Entity: "entity_key", Event: "event_id"},
    primary="primary_reason_code",
)


def attributing_entity_id(
    outcome_code: int,
    people_caps_binding_rule_id: int,
    people_caps_names_entity_id: int,
    business_disqualification_binding_rule_id: int,
) -> int | None:
    """The entity a decline is attributable to, or None where it is not.

    None is correct and common: B-AROD-07 (excluded sector) has nothing to do
    with any entity. The build rejects a `fires(gives="business_decline")` that
    declares neither `names_entity` nor `attributes`, so "None because the author
    forgot" and "None because no entity is involved" are different states and
    only the second can occur.
    """
    pass


def attributing_event_ids(
    attributing_entity_id: int | None,
    entity_adverse_witness: int,
    entity_adverse_binding_rule_id: int,
) -> list[int]:
    """The event set, resolved from the witness bitmask against Event.identity.

    For AE-R-01 this is one event; for AE-R-02 it is every event that matched
    `is_minor_recent`, which is three or more; for AE-R-06 it is every event the
    sum summed. s10 acceptance criterion 2 -- "a count-based roll-up rule
    attributes to ALL the events that satisfied it, not to one of them" -- is
    the bitmask's definition, not a behaviour to test.
    """
    pass


def attribution_chain(
    attributing_entity_id: int | None,
    attributing_event_ids: list[int],
) -> str:
    """The rendered chain. NOT a string built by this project -- the framework
    renders it from the spine and the rule descriptions, which is what keeps the
    wording and the codes from drifting.

    The chain s5.5 requires, in full:

      entity 7 (Ms M. Dlamini, entity_key ...4821, 32.0% effective ownership
      across two paths, required surety, criticality CRITICAL because
      critical_by_surety and critical_by_ownership)
        <- entity_adverse_verdict DISQUALIFYING, binding rule AE-R-01,
           witness {event 41}
          <- event 41, civil judgment R184 000 dated 2025-03-14, unsatisfied,
             classified DISQUALIFYING by AE-C-03 against a disqualifying
             threshold of R25 000
            <- threshold base R50 000, halved by overlay ADJ-05-014
               (sector 412, criticality CRITICAL, approved by Credit Committee
               2026-02-11, expires 2026-08-31)
        -> people_caps PP-06 row 1, business decline, reason 5402

    Every fragment of that is a recorded value. None of it is a format string in
    this project.
    """
    pass


AttributionSpine = module(
    attributing_entity_id, attributing_event_ids, attribution_chain,
    name="attribution", grain=Application,
    taps=["attributing_entity_id", "entity_adverse_binding_rule_id"],
)


# --------------------------------------------------------------------------
# s9.6 -- the same person on two applications.
#
# "A natural person appearing on two applications on the same day must be
# evaluated consistently: the same event list, the same classifications where
# the criticality class is the same, and DIFFERENT classifications where it is
# not -- because the thresholds legitimately differ by role. Both applications'
# records must be able to show the other's existence and the role difference."
#
# The first half is free: the classification is a pure function of the event and
# the broadcast criticality class, so two applications with the same event list
# and the same class produce identical classifications by construction, and
# different classes produce different ones for a recorded reason.
#
# The second half is a frame-tier self-join on `Entity.identity` across the
# batch, which is set-shaped work and belongs where set-shaped work belongs. It
# is a report over the Entity frame, not a stage of the assessment -- which
# matters, because making it a stage would make one application's outcome depend
# on another's, and the determinism requirement in s8 forbids that.
# --------------------------------------------------------------------------
from decider2.frame import Join, Aggregate
import polars as pl

CrossApplicationView = (
    Aggregate(
        Entity, by="entity_key",
        metrics={"appearance_count": pl.len(),
                 "distinct_criticality_classes": pl.col("criticality_class").n_unique(),
                 "applications": pl.col("application_id").unique()},
    )
    | Join(Entity, on="entity_key", how="left",
           schema={"appearance_count": pl.Int64,
                   "distinct_criticality_classes": pl.Int64})
)
