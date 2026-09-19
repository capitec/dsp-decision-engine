"""The four grains this project computes at.

A **grain** is a declared nesting level: a key, a parent, a capacity and a
canonical order. It is the one concept this project needs that doc 03 does not
have (doc 06 O5, "ragged per-record collections", deprioritised and unresolved).

Everything else here is ordinary `decider2`. A step is still a pure scalar
function over one record -- it is just that "one record" now means *one event*,
or *one entity*, or *one pricing candidate*, and the grain says which.

Four properties are load-bearing, and each one is doing work that the spec
(05-business-credit-nested-entities.md) forces:

1. `capacity` is part of the **type**, not of the data. A 4-entity application
   and a 40-entity application compile to the same kernel and run on the same
   signature. Nothing recompiles when fan-out changes (doc 08 s4.2).
2. `capacity` is a machine bound; the **policy** limit is a param (see
   `structure/resolve.py`). Change scenario 3 raises the policy depth limit from
   3 to 4 for applications above R5m. That must not be a recompile, so the
   capacity is 4 and the limit is tunable underneath it. Conflating the two is
   the mistake this split exists to prevent.
3. `identity` is what attribution and de-duplication resolve on. The key is a
   position; the identity is a person. `spec s13 Q5` -- "positional identity
   fails the moment an event is removed" -- is answered here and nowhere else.
4. `order` is the **canonical materialisation order**, not an evaluation order.
   Evaluation order is undefined by construction (see `Gather` in
   `entities/adverse/gather.py`); the canonical order exists so that the
   persisted candidate ledger and the committee pack are byte-identical on
   replay (spec s8, determinism).
"""

from decider2 import grain, Capacity, asc, desc

# --------------------------------------------------------------------------
# Level 0 -- one row per credit application. The grain every other one hangs
# off, and the only grain the caller sees.
# --------------------------------------------------------------------------
Application = grain(
    "application",
    key="application_id",
    order=(asc("application_id"),),
)

# --------------------------------------------------------------------------
# Level 1 -- the resolved entity list. s4.2: 1..40 after structure resolution,
# median 4, mean 6.2, p95 14, p99 27.
#
# `on_exceed="suspend"` is the whole of s5.1's "when the bound is exceeded":
# the application is not declined and not approved, it is suspended with the
# unexpanded remainder listed. A framework that raised, or that silently took
# the first 40, would both be wrong -- and both are what an unbounded list
# column would give you.
# --------------------------------------------------------------------------
Entity = grain(
    "entity",
    key=("application_id", "entity_id"),
    parent=Application,
    identity="entity_key",
    capacity=Capacity(40, on_exceed="suspend", flag="structure_unresolved"),
    order=(desc("effective_ownership_pct"), asc("entity_key")),
)

# --------------------------------------------------------------------------
# Level 2 -- adverse events. s4.4: 0..60 per entity, median 6, p95 40, p99 52.
# Worst observed application: 600 events across 40 entities, 187 on one entity.
#
# Note `parent_identity="entity_key"`, not `entity_id`. s4.4 is explicit: "an
# event follows the person, not the path". After de-duplication two paths
# collapse to one entity and the events attach **once**. Binding events to the
# positional key would double-count them, which s5.1 records as having actually
# happened.
# --------------------------------------------------------------------------
Event = grain(
    "event",
    key=("application_id", "entity_id", "event_id"),
    parent=Entity,
    parent_identity="entity_key",
    identity="event_id",
    capacity=Capacity(60, on_exceed="suspend", flag="event_list_truncated"),
    order=(desc("event_date"), asc("event_id")),
)

# --------------------------------------------------------------------------
# Level 1' -- the pricing candidate space. s5.12: up to 2 200 (amount, term)
# pairs, of which 200-600 are admissible.
#
# This grain is *derived*, not received: `pricing/candidates.py` enumerates it.
# Declaring the search space as a grain rather than as a `Loop` is the single
# largest departure this sketch makes from doc 03 -- see FRAMEWORK-DEMANDS D14.
#
# capacity 2 200 > 64, so the free witness bitset (see `gather.py`) is not
# available here. Nothing needs it: the only fold over candidates is a `best_of`,
# which carries one key.
# --------------------------------------------------------------------------
Candidate = grain(
    "candidate",
    key=("application_id", "candidate_id"),
    parent=Application,
    identity="candidate_id",
    derived=True,
    capacity=Capacity(2200, on_exceed="truncate", flag="search_truncated"),
    order=(asc("amount_band_index"), asc("term_months")),
)

# --------------------------------------------------------------------------
# Severity and criticality orderings. Declared as total orders because
# `verdict(...)` resolves "most severe wins" against them and `Gather` needs a
# total tie-break to be order-independent. A partial order here would make the
# shuffle-invariance test in `tests/` unsatisfiable rather than merely failing.
# --------------------------------------------------------------------------
IMMATERIAL, MINOR, MATERIAL, DISQUALIFYING = 0, 1, 2, 3
SEVERITY_ORDER = (IMMATERIAL, MINOR, MATERIAL, DISQUALIFYING)

PERIPHERAL, SIGNIFICANT, CRITICAL = 1, 2, 3
CRITICALITY_ORDER = (PERIPHERAL, SIGNIFICANT, CRITICAL)

CLEAR, REFER, DECLINE = 0, 1, 2
DISPOSITION_ORDER = (CLEAR, REFER, DECLINE)
