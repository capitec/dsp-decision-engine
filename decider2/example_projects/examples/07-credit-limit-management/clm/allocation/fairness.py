"""Fairness floors and the conditional sub-envelope (s5.8 constraints 5 and the
conditional list). Two more sweeps, before and beside the main one.

FAIRNESS. No segment -- product x grade x mob band, 144 of them -- may have a
funded proportion below 40% of the population-wide funded proportion in three
consecutive cycles. 15% of the envelope is held as reserved sub-budgets
allocated to satisfy these floors BEFORE the main pool is allocated by rank.
"A segment that is starved must be visible as a starved segment, not as 6 000
individually unlucky accounts."

THE SHAPE PROBLEM AND HOW ORDER SOLVES IT. 144 reserved purses is a vector of
carried state, and a sweep carries scalars. Rather than extend the carry model,
the reserved pass is ORDERED BY SEGMENT THEN RANK, and the purse resets when
the sweep crosses a segment boundary. Two scalar carries -- the current segment
id and its remaining purse -- replace a 144-wide vector, and the reset is one
comparison.

That works here and it is worth being honest that it works because the
partition is a single key. FRAMEWORK-DEMANDS #4 asks for fixed-length array
carries for the case where it is not.
"""

import polars as pl
from decider2 import Sweep, module, param
from decider2.frame import Aggregate, Join, Sort

SegmentHistory = Aggregate(
    source="cycle_records",
    by=["product_code", "behaviour_grade", "mob_band"],
    windows=3,                             # three consecutive cycles
    metrics={"funded_proportion_3c": pl.col("was_funded").mean(),
             "starved_cycles": pl.col("was_starved").sum()},
    declares={"funded_proportion_3c": pl.Float64, "starved_cycles": pl.Int8},
)

AttachSegment = Join(source=SegmentHistory, on=["product_code", "behaviour_grade",
                                                "mob_band"], how="left")


def segment_id(product_code: int, behaviour_grade: int, mob_band: int) -> int:
    """One of 144 segments. Declared here so the fairness report and the
    allocation agree about what a segment is."""
    pass


def segment_reserve_c(
    segment_id: int, funded_proportion_3c: float, starved_cycles: int, shared,
    floor_ratio: float = param(0.40, ge=0, le=1.0,
                               description="floor as a fraction of the book-wide rate"),
    reserved_share: float = param(0.15, ge=0, le=0.5,
                                  description="share of the envelope held back"),
) -> int:
    """This segment's purse from the reserved 15%, sized to lift it to the floor."""
    pass


def reserve_remaining_c(reserve_remaining_c: int, current_segment_id: int,
                        segment_id: int, segment_reserve_c: int,
                        additional_limit_c: int, take_reserved: bool) -> int:
    """Reset the purse on crossing a segment boundary; draw from it otherwise."""
    pass


def current_segment_id(segment_id: int) -> int:
    """The carry that makes the reset detectable."""
    return segment_id


def take_reserved(additional_limit_c: int, reserve_remaining_c: int,
                  current_segment_id: int, segment_id: int,
                  segment_reserve_c: int) -> bool:
    """Funded from the segment's reserved purse, ahead of the main pool."""
    pass


ReservedPass = Sweep(
    module(segment_id, segment_reserve_c, take_reserved,
           reserve_remaining_c, current_segment_id, name="allocate_reserved"),
    carries=["reserve_remaining_c", "current_segment_id"],
    requires_order=["segment_id asc", "rank_key asc", "account_id asc"],
    name="fairness_reserve",
    evidence=["take_reserved", "segment_reserve_c", "reserve_remaining_c@entry"],
)

FairnessReserve = (
    AttachSegment
    | Sort(by=["segment_id", "rank_key", "account_id"])
    | ReservedPass
)

# The conditional list (s5.6, 213 000 accounts) is ranked separately against a
# reserved sub-envelope and charged against the budget at its expected 22%
# conversion rate rather than at face value, because most will never confirm
# their income. Same combinator, a third instance; the only differences are the
# seed and a charge rate.
ConditionalPass = Sweep(
    "clm.allocation.sweep:AllocateOne",
    seed=module(..., name="conditional_seed"),
    carries=["limit_budget_remaining_c", "rwa_remaining_c", "el_remaining_c",
             "funded_count", "tail_skips_used"],
    requires_order=["rank_key asc", "account_id asc"],
    charge_rate=param(0.22, ge=0, le=1.0,
                      description="expected conversion; provisioned vs realised "
                                  "is reconciled separately in s9.4"),
    name="conditional_allocation",
    evidence=["allocation_rank", "allocation_outcome_code"],
)
