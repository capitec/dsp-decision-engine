"""Building the PinSet: the seven things spec 09 §5.1 says must have been
captured, assembled into one closed world.

THE DEVIATION FROM DOC 03/08, AND WHY THE SPEC FORCED IT
--------------------------------------------------------
Doc 08 §6.2 gives `pipeline.resolve_params(document, origin=..., complete=True)`
and stops there. Params are pinned; nothing else is. Tables are "provisional"
(doc 03 §4.4, `tables` arrives as an ambient argument); `decision_date` is an
ordinary input column; the overlay stack is not mentioned anywhere in the
framework docs; capability versions are import-time facts.

Replaying a decision under that surface means the caller reconstructs seven
different things from seven different places and hopes. Spec §5.1's worst
failure mode - "a replay that succeeds with the wrong answer because something
resolved against today" - is not a bug in that design, it is that design's
ordinary behaviour when one of the seven is forgotten.

So the harness demands a PinSet: a single, total, self-describing resolution
environment, and a pipeline transform that SEALS a pipeline into it.

    evidence = store.fetch("FLX-2027-03-0418822")
    pins     = pin.resolve(evidence)              # -> PinSet, or raises Incomplete
    sealed   = flow03.at_pin(pins)                # -> a pipeline with no outside
    result   = sealed.score(**pins.inputs_as_received)

`at_pin` is the third invented construct and it is SEALING, not configuring.
The difference is the whole guarantee: inside a sealed pipeline there is no
"today", no live table family, no registry to consult, no default to fall back
to. A read of anything not in the PinSet raises `UnpinnedRead` naming the read.
Structural impossibility, not diligence.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping

from manifest.model import CapabilityPin, GovernanceManifest


class PinSet:
    """The seven categories of spec §5.1, one object, self-describing on disk."""

    # 1. inputs as received - BEFORE normalisation, with the three null situations
    #    of 00 §7.4 distinct. Carried as decider2's own null policy per field
    #    (doc 03 §1): `required` / `missing_as(x)` / `Optional`, so the
    #    distinction is in the TYPE, not in a convention the store might flatten.
    inputs_as_received: Mapping[str, "Received"]

    # 2. reference data as at the decision date - the bureau view AS RETURNED
    #    (never re-queried; spec §11.4's restatement case is why), internal
    #    account state as read, and every table version with the cells read.
    reference: "ReferenceSnapshot"

    # 3. every parameter value in force, including those that fell back to a
    #    code default, NAMED AS SUCH. doc 08 §2.1's `model_fields_set` gives
    #    this for free - the one place the framework already does the right thing.
    params: "ResolvedParams"
    params_schema_hash: str          # see seal.py: this is the one that bites

    # 4. the structure of the logic - build + structure_fingerprint, resolving
    #    for seven years. Not a tag. See engines.py for what "resolving" means.
    build: str
    structure_fingerprint: str

    # 5. per-capability versions. All twenty-one (00 §7.1), independently.
    capabilities: tuple[CapabilityPin, ...]

    # 6. every random or hashed assignment: the derivation AND the derived value.
    assignments: Mapping[str, "Assignment"]

    # 7. decision_date, distinct from computed_at, and distinct from today.
    decision_date: date
    computed_at: datetime

    # and the one spec §5.14.5 adds, which the seven do not contain:
    overlay_stack: "OverlayStack"     # ids, order, scope, magnitude, enabled state, and the
                                      # unadjusted input each overlay consumed


class Received:
    """One input as it arrived. The null situation is a type, not a value."""
    value: float | int | str | bool | None
    null_situation: "Literal['present', 'not_collected', 'collected_as_zero', 'could_not_establish'] "
    source_system: str
    as_at: date


class Assignment:
    """Spec §5.15.3. Both halves, because a change to the derivation must be
    detectable rather than silently absorbed."""
    name: str                # "challenger_split", "holdout_group", "detail_sample"
    stable_key: str          # application_id - never a sequence, never a timestamp
    seed_name: str           # named, registered, effective-dated
    derived: int             # WHAT IT ACTUALLY WAS
    derivation_hash: str     # WHAT PRODUCED IT


def resolve(evidence: "DecisionEvidence", *, manifest: GovernanceManifest) -> PinSet:
    """Assemble a PinSet, or raise `Incomplete` naming every missing category.

    Spec §13 Q1 asks whether the minimum re-derivable artefact is one thing or
    an assembly, and if an assembly, what guarantees completeness. Answer: it
    is an assembly of six stores, and NOTHING guarantees completeness at
    assembly time - completeness is only ever demonstrated by `seal.py`
    observing that the replay read exactly what was pinned. So incompleteness
    is detected at replay, which is seven years too late...

    ...unless the flow proves it at emission. Which is what `attest()` does:
    the witness set is derived from the graph, so the set of things the flow
    CAN read is the set of things it pins. That is the only answer that is not
    a hope, and it is why contract item 4 is structural.
    """
    pass  # fetch from the six stores by decision id, validate against the manifest's witness set


def resolve_as_at(family: str, d: date, snapshot: "ReferenceSnapshot") -> "TableRef":
    """The pinned replacement for `core.dates.in_force`. Reads the snapshot, never a live family."""
    pass  # KeyError here is UnpinnedRead, not a fetch
