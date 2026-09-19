"""`Allocate` — a population-level constraint over per-account decisions.

Spec §13.5: "Capacity allocation is not a rule about an account; it is a rule
about all of them at once, and its outcome must be recorded as a per-account
explanation. Is this the same shape as project 07's portfolio budget, or a
different one?"

It is the same SHAPE and a harder instance: nine constrained pools instead of one
budget, fairness floors that bind before value, monthly pacing, a per-client
quota, and a neutrality requirement with respect to an experiment split.

DEVIATION FROM DOC 03 / DOC 02. The frame tier ships `join`, `aggregate`,
`filter` and later `sort` and `union`. None of those is a top-K-subject-to-
constraints, and composing one from them loses the thing that matters: the
per-account explanation. `Allocate` is a new frame-tier combinator with a
declared schema transform, so lineage and wiring still survive it.

It also needs a new lineage annotation. An account's `allocated` flag depends on
185 999 other accounts. `lineage("allocated")` must not claim it depends only on
this account's inputs, and must not report `unknown` either, because the
dependency is precise and declarable. So:

    @population_dependent(over="pool_code", key="rank_key")

is a declared, greppable marker in the same family as `@breaks_lineage`, and
`lineage()` reports the population dependency explicitly. FRAMEWORK-DEMANDS #14.

STAGES. Allocation is four ordered stages and the stage that placed an account
IS its explanation. This is what turns non-selection code 250 from a
population-level fact into a per-account one.
"""

from decider2 import Allocate, Quota, Reserve, Floor, Pace, population_dependent, param
from decider2.types import cents, f8, i1, i2, i4

from .pools import POOLS
from .ranking import RANK_KEY, rank_tie_break


CapacityAllocation = Allocate(
    name="capacity_allocation",
    pools=POOLS,
    demand="proposed_treatment_code",
    eligible="permitted_treatment_mask",     # from the suspension panel. Suspended
                                             # accounts are not ranked at all; they
                                             # get code 210, not code 250.
    rank_by=RANK_KEY,
    tie_break=rank_tie_break,

    stages=[
        # ---- stage 1: mandatory pulls. These bind BEFORE value. -------------
        Floor(name="untouched_days_floor",
              predicate="untouched_days >= limit",
              limit=param({"buckets_3_8": 21, "buckets_1_2": 30},
                          owner="collections_strategy", co_sign="regulatory_compliance"),
              exempt_when="all_contact_suspended",
              description="No account goes untouched for more than 21 days while "
                          "in buckets 3-8, or 30 days in buckets 1-2, unless a "
                          "suspension forbids all contact."),

        Floor(name="bucket3_entry_coverage",
              predicate="is_first_entry_to_bucket_3 and days_in_bucket <= 5",
              target_pct=param(60, ge=0, le=100, owner="collections_strategy"),
              pool="early_agents",
              description="At least 60% of accounts newly entering bucket 3 receive "
                          "an agent attempt within 5 days."),

        Floor(name="channel_starvation",
              predicate="automated_only_streak_days >= limit and arrears_bucket_code >= 4",
              limit=param(45, ge=7, le=180, owner="collections_strategy"),
              description="An account may not receive only automated treatments for "
                          "more than 45 consecutive days while in buckets 4-8."),

        Floor(name="pre_prescription_legal",
              predicate="pre_prescription_flag and recovery_estimate > threshold",
              threshold=param(500_000, owner="recoveries_and_legal"),
              pool="legal_handover",
              description="An account worth suing that is about to prescribe is "
                          "pulled ahead of higher-value candidates. The priority "
                          "competes for the same scarce capacity as everything else."),

        # ---- stage 2: reserved shares --------------------------------------
        Reserve(name="high_balance_reserve",
                pools=("early_agents", "late_agents"),
                share_pct=param(8, ge=0, le=50, owner="collections_strategy"),
                scope={"balance_band_code": [6, 7]},
                overlayable=True,      # allocation.weight surface targets THIS
                description="Volume plays in bands 1-3 have better cost-per-rand "
                            "ratios and will starve balance bands 6-7 if value "
                            "alone rules."),

        # ---- stage 3: value-ranked fill ------------------------------------
        "rank_fill",

        # ---- stage 4: pacing -----------------------------------------------
        Pace(pools=("legal_handover", "agency"), basis="month_to_date",
             description="Monthly contracted volumes consumed daily. Do not take "
                         "the best candidates on the first and starve the rest."),
    ],

    # --- cross-cutting constraints, applied within every stage --------------
    quotas=[
        Quota(by="client_id", cap="frequency_cap_headroom",
              non_selection_code=260,
              description="Three accounts of one client each politely observing a "
                          "per-account cap produce a harassment complaint. The cap "
                          "is per client, so the ALLOCATOR decides which sibling "
                          "account gets the contact, and the others record 260 "
                          "naming the sibling that was actioned."),
        Quota(by="region_code", pool="field", cap="max_sub_grains"),
    ],

    neutral_over=["cohort_code"],
    # Spec §5.9: "Challenger cohorts are allocated before champion overflow. If
    # rationing falls harder on challengers than on champions, the experiment
    # measures rationing rather than treatment."
    #
    # MECHANISM: each pool's capacity is split across cohorts in proportion to
    # each cohort's DEMAND share, and rank-fill runs within cohort strata. The
    # residual imbalance — integer rounding and floor interactions cannot be
    # made exactly zero — is computed and REPORTED per pool per day, which is
    # what acceptance criterion 9 means by "demonstrably uncorrelated" rather
    # than "assumed".
    neutrality_report="cohort_rationing_imbalance",

    emits={
        "allocated": bool,
        "pool_code": i1,
        "pool_rank": i4,
        "pool_ranked_count": i4,
        "pool_cutoff_rank": i4,
        "allocation_stage_code": i1,       # WHICH stage placed it — the explanation
        "binding_constraint_code": i2,
        "non_selection_reason_code": i2,
        "non_selection_detail": "struct",
    },
)


@population_dependent(over="pool_code", key="rank_key",
                      outputs=["allocated", "pool_rank", "pool_cutoff_rank",
                               "non_selection_reason_code"])
def _lineage_declaration():
    """Declared, not inferred. `grep -r "@population_dependent"` yields every
    place in the codebase where an account's answer depends on other accounts —
    the governance property, in the same family as `@breaks_lineage`."""


# ---------------------------------------------------------------------------
# NON-SELECTION. Spec §13.6: "How is non-selection made explainable without
# materialising a rank and a cut-off for every one of 2.3M accounts every day?"
#
# THE ANSWER IS THAT YOU DO MATERIALISE IT, AND IT IS CHEAP.
#
# Only accounts that were RANKED need a rank, and an account is ranked only if it
# reached stage 3 in a pool. The 2.3M split on an observed day:
#
#   matrix recommended no action (code 200)        980 000   no rank needed
#   suspended (code 210)                           187 000   no rank needed
#   cooling-off / interval / cap (220, 230, 240)   410 000   no rank needed
#   no permitted channel (270)                      94 000   no rank needed
#   ranked and allocated                           343 000   rank + pool
#   ranked and below the cut-off (code 250)        286 000   rank + pool + cutoff
#
# So 629 000 accounts carry (i1 pool, i4 rank) = 3 MB, the cut-off is one scalar
# per pool per day, and the remaining 1.67M carry an i2 reason code and a small
# struct. Total ~14 MB/day, 36 GB over the seven-year retention.
#
# "You were 91 412th of 186 000 in the early agent pool, and we stopped at
# 27 000" is therefore a stored per-account fact, not a reconstruction. The
# instinct to avoid materialising it is the instinct that makes §9.5 — "why was
# this account never contacted for 40 days?" — a data-science project.
# ---------------------------------------------------------------------------

NON_SELECTION_CODES = {
    200: ("matrix_no_action",        ("matrix_cell_id",)),
    205: ("suppressed_by_overlay",   ("overlay_id", "scope", "expiry",
                                      "treatment_code_unadjusted")),
    210: ("suspended",               ("suspension_mask", "evidence_rows")),
    220: ("cooling_off",             ("treatment_code", "last_applied_on",
                                      "cooling_off_days_remaining")),
    230: ("interval_or_cap",         ("interval_test_results", "current_count",
                                      "window_days", "cap")),
    240: ("promise_in_force",        ("promise_id", "promised_on",
                                      "promise_grace_ends_on")),
    250: ("below_capacity_cutoff",   ("pool_code", "pool_rank", "pool_ranked_count",
                                      "pool_cutoff_rank", "ranking_basis_code")),
    260: ("sibling_account_actioned", ("sibling_account_id", "sibling_treatment_instance_id")),
    270: ("no_permitted_channel",    ("channel_failure_mask",)),
}
