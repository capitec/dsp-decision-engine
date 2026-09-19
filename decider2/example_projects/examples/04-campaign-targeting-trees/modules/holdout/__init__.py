"""Holdout, control and challenger — spec §5.7.

Everything here is a function of identifiers and design versions alone.  No
randomness at run time, no stored assignment table that can drift, no dependence
on processing order, and — the requirement most often lost when a run is
optimised for the contacts it actually produces — **no gate on evaluation**.

Control clients go through the tree, reach a leaf and get a route recorded,
exactly as treated clients do.  The control flag enters at `Arbitration` as a
refusal reason (`REF_CONTROL`), and nowhere earlier.  That placement is the whole
mechanism: there is no `if control: skip` anywhere in the skeleton, so there is
nothing for a performance optimisation to find.
"""

from __future__ import annotations

from decider2 import Table, param, module, step
from decider2.hash import stable_hash64        # frozen algorithm; see DEMANDS #19


@step(description="Deterministic control-group membership for one (client, campaign)")
def is_control(
    client_id: int,
    campaign_id: int,
    holdout_design_version: int,
    control_fraction: float = param(0.05, ge=0.0, le=0.5,
                                    description="Per-campaign control share. 12 campaigns "
                                                "run 0.10 while their trees are new."),
) -> bool:
    """u = stable_hash64(client_id, campaign_id, holdout_design_version) / 2**64

    `stable_hash64` is a framework primitive with a pinned algorithm, its own
    version tag and a golden test vector checked in CI.  It is not `hash()`, not
    `hashlib` with a default, and not "whatever the library does this release".
    A silent change to it moves every client in every campaign between groups and
    invalidates every in-flight measurement, and nothing in the output would look
    wrong.  Doc 03 has no such primitive; DEMANDS #19 asks for one.
    """
    pass  # return (stable_hash64(client_id, campaign_id, holdout_design_version) >> 11) * 2**-53 < control_fraction


@step(description="Universal holdout — 1% of the client base, excluded from all campaigns")
def is_universal_holdout(
    client_id: int,
    universal_holdout_version: int,
    universal_fraction: float = param(0.01, ge=0.0, le=0.05),
) -> bool:
    """Measures the total effect of the programme rather than of any one campaign.
    Campaign id is deliberately absent from the hash input."""
    pass


@step(description="Champion/challenger variant index for one (client, campaign)")
def variant_index(
    client_id: int,
    campaign_id: int,
    variant_design_version: int,
    split: Table = Table("registries/variant_designs.csv", key="campaign_id"),
) -> int:
    """0 = champion, 1 = challenger_a, 2 = challenger_b.  90/10 by default,
    70/20/10 where three are live.

    Must be established **before** stage 3, because it selects which tree version
    is evaluated (spec §5.7 preconditions).  In the pipeline this is a `Branch`
    with the live tree versions as arms — the variant is not a parameter of the
    tree, it is a choice of tree, and modelling it as an n-way Branch (doc 03
    §8.2) keeps it visible in the composition expression and in `render()`.
    """
    pass


@step(description="Policy arm — 5% of clients dispatched on the overlays-off answer")
def policy_arm(
    client_id: int,
    campaign_id: int,
    policy_arm_version: int,
    shadow_fraction: float = param(0.05, ge=0.0, le=0.2),
) -> bool:
    """Spec §8 requires an overlays-disabled shadow evaluation of every campaign
    inside the same six hours.  We evaluate 100% of clients under both bundles
    (see `shadow(...)` in pipelines/monthly_cycle.py) — the cost is one extra tree
    traversal, which is not the constraint — and separately **dispatch** 5% of
    them on the unadjusted answer, so that the tree's own performance is
    measurable and not merely computable.  Those are two different requirements
    that the spec's "5% shadow" phrasing runs together."""
    pass


def enumerate_movers(old_version: int, new_version: int, client_ids) -> "pl.DataFrame":
    """Spec §5.7 req 2 and scenario 6: when a design version changes, the set of
    clients who moved must be enumerable.

    Because assignment is a pure function of identifiers, this is computable
    offline for any pair of design versions without either cycle having run — and
    the measurement team's "which historical comparisons are now broken" question
    is answered by joining the movers to the measurement windows they sit in.
    """
    pass


HoldoutAssignment = module(
    is_control, is_universal_holdout, variant_index, policy_arm,
    name="holdout",
    partition="record",       # depends on identifiers only — re-runnable per client
    taps=["is_control", "variant_index"],
)
