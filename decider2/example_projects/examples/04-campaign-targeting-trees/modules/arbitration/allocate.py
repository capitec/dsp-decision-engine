"""Arbitration — spec §5.6, and the answer to §13 Q9.

*"Where does a population-level constraint sit relative to per-client decisions?
Arbitration cannot be decided one client at a time, yet everything before it is
per client.  What is the shape of the thing that sits on top, and how does it
stay deterministic and re-runnable?"*

It sits **on top**, as a frame-tier module with `partition="cycle"`, and that
declaration is the whole answer to Q10 as well.  Everything before it declares
`partition="campaign"` or `partition="record"`, so
`pipeline.rerun_unit("leaf") == "campaign"` and
`pipeline.rerun_unit("contact_sequence") == "cycle"` are static facts a runbook
can quote at 03:00 instead of discovering.

`allocate(...)` IS A NEW FRAME-TIER KIND
-----------------------------------------
Doc 02 §5 ships `join`, `aggregate`, `filter` and offers `@breaks_lineage` for
everything else.  Using `@breaks_lineage` here would put the single most contested
output in the project — "why did my campaign lose 340 000 clients" — behind a
lineage gap, which is the opposite of what the escape hatch is for.  So this
sketch invents a fourth declarative frame kind: a **constrained assignment** with
a declared schema transform, a declared tie-break, and a declared refusal channel.

Three properties the kind guarantees, which is what makes it worth being a kind
rather than a function:

  1.  **Totality.**  `demand == admitted ⊎ refused`.  Every input row leaves with
      exactly one outcome.  "Not selected" without a reason is not an expressible
      state, because `refusal=` is a required argument (spec §5.6 req 3).
  2.  **Determinism.**  The tie-break is declared, never row order, never a
      frame's physical ordering.  Identical inputs produce an identical contact
      list, row for row (spec §8, acceptance criterion 4).
  3.  **Accounting.**  Per-constraint utilisation — allocated, used, unused, and
      the value of refused demand — is an output of the kind, not something the
      project computes afterwards and hopes agrees (§5.6 req 4).
"""

from __future__ import annotations

import polars as pl

from decider2.frame import allocate, refusal, tie_break
from decider2 import Table, param

from .reasons import REF_CAPACITY, REF_CONTROL, REF_FAIRNESS, REF_FATIGUE, REF_RANK, REF_SUSPENDED


# --------------------------------------------------------------------------
# Unconstrained demand: ~61 M qualifications from ~11.6 M clients, mean 5.3
# campaigns per contactable client, 9 at p99.  At most ~19 M contacts may be
# issued (2 per client), and after channel limits substantially fewer.
# --------------------------------------------------------------------------

Arbitration = allocate(
    name="arbitration",
    partition="cycle",                      # THE declared dependency.  See above.

    demand="qualifications",                # client x campaign x permitted channel
    unit=["client_id", "campaign_id"],

    # The bid.  `priority_weight` is the campaign's own statement; expected value
    # is response propensity x expected margin; the weights between them are the
    # campaign forum's monthly artefact.
    score=pl.col("priority_weight") * pl.col("forum_weight")
          + pl.col("expected_value_cents") * pl.col("ev_weight"),

    # Deterministic tie-break, declared.  NEVER row order.
    order=tie_break(
        "score", descending=True,
        then="stable_hash64(client_id, campaign_id, cycle_id)",
    ),

    # Constraints, evaluated in declared order.  Each names the refusal it emits.
    constraints=[
        # per-client, from the leaf outcome and the holdout register
        refusal.when(pl.col("campaign_suspended"), REF_SUSPENDED),
        refusal.when(pl.col("is_control") | pl.col("is_universal_holdout"), REF_CONTROL),
        refusal.when(pl.col("fatigue_blocked"), REF_FATIGUE),

        # per-client cap: 2 contacts this cycle.  Losers name the WINNER.
        allocate.cap(by="client_id", limit=param(2, ge=1, le=5),
                     refusal=REF_RANK, refusal_detail="winning_campaign_id"),

        # population-level.  This is the part that cannot be decided one client
        # at a time: whether client 8 412 907 gets an SMS depends on how many
        # other clients want one, which is not knowable while evaluating them.
        allocate.capacity(
            by="channel_code",
            limit=Table("registries/channel_capacity.csv", key="channel_code",
                        column="monthly_capacity"),
            daily_limit_column="daily_cap",
            refusal=REF_CAPACITY,
        ),

        # fairness: no campaign above 35% of a client's contacts over 6 cycles,
        # and every live campaign receives at least 60% of its rank-one demand.
        allocate.fairness(
            share_cap=param(0.35, ge=0.0, le=1.0),
            over_cycles=param(6, ge=1, le=24),
            floor_of_rank_one=param(0.60, ge=0.0, le=1.0),
            refusal=REF_FAIRNESS,
            floor_breach="raise_to_forum",       # a report, not a cycle failure
        ),
    ],

    # The mechanism.  The spec deliberately does not prescribe one; this sketch
    # picks rank-and-cut with a bounded repair pass and says so out loud, because
    # a mechanism that is not named is a mechanism nobody can reason about.
    #
    #   pass 1  rank all demand, admit greedily against every constraint
    #   pass 2  for each channel whose capacity bound, re-offer refused pairs on
    #           their next permitted channel from the leaf's channel preference
    #   pass 3  fairness repair: bounded swaps, max 3 rounds, declared
    #
    # It is not an optimum and does not claim to be.  It is deterministic,
    # re-runnable, explainable per row, and finishes in bounded time, which the
    # spec requires and optimality does not.
    method="rank_cut_repair",
    repair_rounds=param(3, ge=0, le=10),

    writes=["contacted", "channel_code", "contact_sequence",
            "arbitration_reason_code", "arbitration_detail"],
    reports=["channel_utilisation", "refused_demand_value", "rank_one_fill_rate"],
)
