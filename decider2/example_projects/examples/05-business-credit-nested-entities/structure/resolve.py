"""Stage 5.1 -- the disclosed graph collapsed to a bounded, de-duplicated tree.

This is the seam spec s13 Q12 names: "set-shaped work feeding record-shaped
work, the same seam `core.exposure` has, and this project needs both on the same
application."

It is frame tier, and it is **not** `@breaks_lineage`. It is three rounds of
declared joins under a bounded frame-tier `Loop`, which is the one frame-tier
construct this project needs that doc 02 s1 does not ship (`join`, `aggregate`,
`filter` only). See FRAMEWORK-DEMANDS D09.

Writing it as an opaque polars function would be two hours' work and would cost
static lineage on every downstream entity value -- which means losing the answer
to "which inputs can affect the people grade" for the whole flow. That trade is
not worth making for the stage that produces the collection everything else is
written over.
"""

import polars as pl
from decider2 import module, step, param, table
from decider2.frame import Loop as FrameLoop, Join, AntiJoin, Aggregate, Filter

from grains import Application, Entity


# --------------------------------------------------------------------------
# One round of expansion. Declared schema in, declared schema out, so lineage
# and dtype propagation survive (doc 02 s5).
#
# The bound is `Entity.capacity` -- 40, a machine capacity. The *policy* limits
# are params, and they are different numbers that happen to look the same:
#   * expansion depth limit          3   (s6.1, "Rare")
#   * expansion materiality floor  5.0%  (s6.1, "On regulation")
# Change scenario 1 moves the materiality floor from 5.0 to 3.0 and moves
# nothing else. Change scenario 3 moves the depth limit from 3 to 4 for
# applications above R5m -- a *record-varying* limit, which is why the limit is
# a param and the capacity is a type. If they were one number, scenario 3 would
# be a recompile per application.
# --------------------------------------------------------------------------
ExpandOneLevel = (
      Join(
          "disclosed_holdings",
          left_on="entity_key", right_on="parent_entity_key", how="inner",
          schema={"child_entity_key": pl.Utf8,
                  "direct_ownership_pct": pl.Float64,
                  "is_controlling": pl.Boolean},
      )
    | AntiJoin("path_so_far", on="child_entity_key", flag="cycle_truncated")
    | Filter(pl.col("effective_ownership_pct") >= pl.col("materiality_floor"))
)

ExpandStructure = FrameLoop(
    ExpandOneLevel,
    carries=["frontier", "resolved", "path_so_far"],
    max_iterations=3,                     # == Entity.capacity depth dimension
    on_exhausted="flag:depth_limit_reached",
)


# --------------------------------------------------------------------------
# De-duplication. s5.1: the same natural person reached by two paths is ONE
# entity, with summed ownership, the most senior role, disjoined control, events
# attached once, and *every* path retained.
#
# This is a Gather in the frame tier -- over the *pre-resolution* path rows,
# into the Entity grain. It is the same construct as every other Gather in this
# project (entities/adverse/gather.py), which is the point: the de-duplication
# is a roll-up, and it gets the same order-independence guarantee and the same
# free witness set as the adverse roll-up does.
#
# `paths_witness` is what makes the committee pack's "every path by which they
# reach the applicant" (s9.3 item 2) a lookup rather than a reconstruction.
# --------------------------------------------------------------------------
from decider2 import sum_of, max_of, any_of, count, collect_of

DeduplicateEntities = Gather(
    "path",                                        # the pre-resolution grain
    into=Entity,
    on="entity_key",                               # identity, not position
    effective_ownership_pct = sum_of("path_ownership_pct"),
    relationship_type_code  = max_of("role_seniority_rank"),   # 11>2>10>3>1>5>6>7>8>4>12
    is_controlling          = any_of("path_is_controlling"),
    path_count              = count(),
    paths                   = collect_of("path_string", max=8),
    name="deduplicate_entities",
)


# --------------------------------------------------------------------------
# Ownership reconciliation -- FV-06. Application grain, ordinary step.
# --------------------------------------------------------------------------
def ownership_reconciles(
    resolved_ownership_total: float,
    tolerance_pct: float = param(0.5, ge=0.0, le=5.0,
                                 description="FV-06 reconciliation tolerance"),
    minimum_disclosed_pct: float = param(90.0, ge=50.0, le=100.0,
                                         description="s5.1 disclosed-equity floor"),
) -> bool:
    """Effective ownership reconciles to 100% within tolerance (FV-06)."""
    pass  # abs(total - 100) <= tolerance and total >= minimum_disclosed


def structure_unresolved(
    resolved_entity_count: int,
    depth_limit_reached: bool,
    ownership_reconciles: bool,
    cycle_truncated: bool,
    entity_capacity_exceeded: bool,
) -> bool:
    """s5.1: bound exceeded, depth exceeded, or ownership under 90% -- suspend.

    Never a decline and never an approval. An application the Bank cannot see
    through is a referral. The outcome gate in validation/consistency.py is what
    enforces the "never an approval" half; this step only raises the flag.
    """
    pass  # or-reduce the five causes; the cause codes travel separately


def structure_unresolved_cause(
    depth_limit_reached: bool,
    ownership_reconciles: bool,
    cycle_truncated: bool,
    entity_capacity_exceeded: bool,
) -> int:
    """Which of the four causes suspended the structure. Ranked, not or-ed.

    s5.1 requires the *specific* cause and the unexpanded remainder, because a
    committee handling a capacity breach does something different from a
    committee handling a 78%-disclosed shareholder register.
    """
    pass  # rank: capacity > depth > reconciliation > cycle


ResolveStructure = (
    ExpandStructure
    | DeduplicateEntities
    | Entity.materialise()          # applies Entity.capacity and Entity.order;
                                    # a breach raises the declared flag and
                                    # suspends -- it does not truncate and it
                                    # does not raise. FRAMEWORK-DEMANDS D04.
)

ReconcileOwnership = module(
    ownership_reconciles,
    structure_unresolved,
    structure_unresolved_cause,
    name="reconcile_ownership",
)
