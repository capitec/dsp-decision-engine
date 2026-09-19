"""The real-time pipeline. One expression; the whole flow is legible in it.

Read it top to bottom and every hard requirement in the spec is visible as a
position in the sequence:

  * hard blocks are computed BEFORE evaluation and consumed AFTER it, so they
    floor the action without truncating the evidence;
  * the counterfactual is the same resolver relabelled, so it cannot drift;
  * ShadowRules is AFTER Resolve, which is the shadow isolation guarantee —
    not a flag, not a review rule, a topological fact;
  * EmitRecord is last and asynchronous, and off the latency budget.

Latency budget, card authorisation p99 (spec §8), against this expression:

    Normalise                       1.5 ms
    ClientContext + fan(...)        8.0 ms   concurrent, model call inside
    Completeness + HardBlocks       0.5 ms
    Applicability + OverlayStack    1.0 ms
    LiveRules                       6.0 ms   635 evaluations, no early exit
    Resolve + ResolveBase + ...     1.0 ms
    ------------------------------------
    serial total                   18.0 ms   headroom 7.0 ms
    EmitRecord                      async, off budget

The interesting thing about that table is that the 6 ms for rule evaluation is
wildly over-provisioned and the 8 ms of enrichment is the whole problem. 2 700
predicates over a 3 KB feature vector that fits in L1 is a few microseconds of
compiled scalar code. The record tier makes the rules a rounding error; the
budget is spent in the frame/IO tier, which is where the two-tier split says it
should be. If this design fails on latency it will fail on enrichment, and no
amount of rule-engine cleverness will help.
"""

from __future__ import annotations

from decider2 import fan, pipeline, record_contract

from fraud_interdiction.decision.applicability import Applicability
from fraud_interdiction.decision.dispatch import Dispatch, Wording
from fraud_interdiction.decision.hard_blocks import HardBlocks
from fraud_interdiction.decision.overlays import OverlayStack
from fraud_interdiction.decision.resolve import OverlayAttribution, Resolve, ResolveBase
from fraud_interdiction.enrichment import Enrichment
from fraud_interdiction.events.normalise import Normalise
from fraud_interdiction.ruleset import LiveRules, ShadowRules

Interdiction = (
    Normalise
    | Enrichment                       # -> the feature vector, and nothing else
    | HardBlocks
    | Applicability
    | OverlayStack
    | LiveRules                        # -> live_fire_bits, live_base_fire_bits
    | OverlayAttribution               # -> fired_on_overlay_ids
    | Resolve                          # -> action_code, reasons, trace
    | ResolveBase                      # -> counterfactual_*, same module relabelled
    | Dispatch
    | Wording
    | ShadowRules                      # AFTER Resolve. Cannot influence it.
)

# The record contract is a static assertion, not a serialiser. It names every
# field spec §5.15 requires and fails the build if the pipeline does not produce
# one of them. "A record that fails to persist is an incident, not a dropped
# metric" — this is the half of that sentence the framework can enforce.
Interdiction = record_contract(Interdiction, "contracts/decision_record.json")


# --- the synchronous response ----------------------------------------------
# Deliberately a projection, and deliberately NOT the firing set.
#
# "A merchant acquirer does not learn which of the Bank's rules stopped a card."
# The projection is declared here, in code, reviewed, so that adding
# fired_rule_ids to the authorising system's response is a code change with a
# reviewer — not a JSON serialisation accident.
SYNCHRONOUS_RESPONSE = (
    "event_id", "action_code", "primary_reason_code",
    "client_wording", "challenge_selection", "effective_queue",
    "enrichment_degradation_code", "degraded_mode_code",
    "rule_set_version", "adjustment_stack_version",
)


def serve(event: dict) -> dict:
    """Single-record entry point. `score()`, not `apply()` — no polars on the
    request path at all (doc 02 §3.5). Compiled variants are selected at
    warmup, never compiled at runtime: this is a `sealed` deployment for the
    baseline and `live` only for staged rule sets (doc 08 §4.1)."""
    pass  # Interdiction.score(**event, params=RT.params) -> project SYNCHRONOUS_RESPONSE
