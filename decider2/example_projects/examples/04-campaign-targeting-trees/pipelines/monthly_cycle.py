"""The monthly cycle — 14.2 M clients, up to 60 campaigns, ~400 M evaluations,
six hours, 22:00 to 06:00.

READ THE SHAPE FIRST.  The cycle is  chunk x all campaigns,  not
campaign x all clients:

    for chunk in chunks(64):            # ~220 000 clients, by stable hash bucket
        extract 287 columns once        # the union of every live tree's reads
        for campaign in live:           # 60 kernels over the same resident columns
            evaluate, capture the route, write per-campaign outputs
    arbitrate once over everything      # partition="cycle"

Why this way round and not the obvious one:

  * the boundary crossing, not the arithmetic, is the cost.  400 M evaluations at
    ~100 ns is 40 seconds of kernel time against a six-hour window; 14.2 M x 287
    columns is 32 GB of extraction.  Extract once per chunk, run 60 kernels while
    the columns are resident.
  * partial failure resumes at chunk granularity (spec §8): a cycle that fails at
    70% resumes at chunk 45 and does not re-evaluate chunks 0..44, and cannot
    produce different answers for them because the chunk is a pure function of
    the frozen snapshot.
  * chunk membership is `stable_hash64(client_id) % 64`, never a sort or a row
    number, so the boundaries are identical on a re-run in 2029.

The per-tree column set is narrower still (9..74 features), computed statically
from each tree document's `reads`.  Static lineage is doing real work here: it is
what lets the boundary extract 74 columns for campaign 23 instead of 600.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from decider2 import Branch, Cycle, fuse, parallel, shadow
from decider2.frame import Aggregate, Filter, Join

from modules.arbitration.allocate import Arbitration
from modules.fatigue import AttachFatigueState, CarriedContactState
from modules.holdout import HoldoutAssignment
from modules.overlays.sites import AmountCapOverlay, ScoreShift
from modules.overlays.resolve import resolve as resolve_overlays
from modules.suppression import ChannelSuppressions, EvaluationGate, GlobalSuppressions
from modules.tree import CampaignTree, load_tree

# ---------------------------------------------------------------------------
# Stage 3, per campaign.  The variant split is an n-way Branch over the live
# tree versions, which keeps "which tree ran" visible in the composition
# expression and in render(), rather than hidden in a lookup.
# ---------------------------------------------------------------------------

def campaign_stage(campaign_id: int, cycle_date: date):
    """Compose one campaign's per-record work.  Called 60 times; the result is
    60 independent `partition="campaign"` sub-pipelines."""
    champion, challenger_a, challenger_b = load_tree(campaign_id, as_at=cycle_date, variant=None)

    TreeStage = Branch(
        "variant_index",
        [CampaignTree.of(champion),
         CampaignTree.of(challenger_a),
         CampaignTree.of(challenger_b)],
        modifies=["leaf_key", "route_digest", "route_depth", "leaf_outcome_code",
                  "offer_tier_code", "advertised_amount_cents", "channel_pref_code",
                  "priority_weight", "reason_label"],
    )

    return (
        HoldoutAssignment                    # partition="record", before the tree
        | GlobalSuppressions                 # partition="record"
        | ChannelSuppressions
        | ScoreShift                         # overlay site 1
        | EvaluationGate(TreeStage)          # absolute suppressions skip; others don't
        | AmountCapOverlay                   # overlay site 3
    )


# ---------------------------------------------------------------------------
# The cycle.
# ---------------------------------------------------------------------------

cycle = Cycle(
    name="campaign_targeting_monthly",
    cycle_date=...,                          # supplied per run; NEVER date.today()
    chunks=64,
    chunk_by="stable_hash64(client_id)",
    resume="chunk",

    # A Cycle pins ONE generation and ONE document manifest for its whole life.
    # doc 08 §4 property 1 guarantees the generation pointer is read once per
    # *invocation*; a cycle is tens of thousands of invocations over six hours and
    # needs the guarantee at cycle scope.  DEMANDS #27.
    manifest="manifests/2026-09-M.json",
)

pipeline = (
    # --- stage 1: the frozen snapshot.  Preconditions raise, they do not warn. ---
    Filter(pl.col("mart_as_at") >= pl.col("mart_freshness_floor"), on_fail="raise")
    | Join("feature_mart", on="client_id", how="inner", snapshot="frozen")
    | AttachFatigueState                     # the carried input, pinned by version

    # --- stages 2,3,4,7 per campaign, all record-tier, inside one chunk --------
    | parallel(
        fuse(campaign_stage(23, cycle.cycle_date)),
        # ... 59 more, assembled from registries/campaigns.csv by
        #     `live_campaigns(as_at=cycle.cycle_date)`; written out here for
        #     exactly one campaign so the shape is legible.
      )

    # --- the overlays-off twin.  Same kernels, second thresholds array. --------
    | shadow(
        "campaign_stage:*",
        thresholds="unadjusted",             # the tree document's own slot values
        params={"score_shift": 0.0, "odds_multiplier": 1.0, "cap_reduction_pct": 0.0},
        prefix="unadjusted_",
      )

    # --- stage 5: bulk pre-assessment, 7.9 M clients, project 03 in batch ------
    | Join("project03_preassessment", on=["client_id", "product_code"], how="left")

    # --- stage 6: arbitration.  partition="cycle" — the declared dependency. ---
    | Arbitration

    # --- stage 8: outputs -----------------------------------------------------
    | Aggregate(by=["campaign_id", "tree_version", "overlay_stack_id", "variant",
                    "is_control", "route_digest"],
                metrics={"evaluations": pl.len()},
                emit="paths_rollup")
)


# ---------------------------------------------------------------------------
# What the cycle emits, and the two lines that make the audit story work.
# ---------------------------------------------------------------------------
#
#   assignments     61 M   client x campaign x cycle
#   paths          400 M   one uint64 route_digest per evaluation + its twin
#   suppressions    11 M   one bitmask row per (client, campaign)
#   arbitration     52 M   one refusal per qualifying non-contact
#   holdout         61 M   re-derivable; stored for convenience, not for truth
#
# assert pipeline.rerun_unit("leaf")             == "campaign"
# assert pipeline.rerun_unit("contact_sequence") == "cycle"
#
# Those two lines are spec §13 Q10 ("Can one campaign be re-run alone?  What is
# the honest answer, and how is the dependency made explicit rather than
# discovered at 03:00?").  The honest answer is: through stage 5 yes, exactly;
# from stage 6 no.  It is a static property of the composition and it is
# assertable in CI.
