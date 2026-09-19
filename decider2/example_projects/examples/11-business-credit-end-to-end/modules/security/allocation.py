"""Cross-facility security allocation. Spec 5.11, H4, 13-Q8.

    One property, at a 70% advance rate, secures a term facility and a revolving
    facility simultaneously. Growing one re-opens the other. The facilities are
    assessed at different times, by different people, under different entry
    points.

This is the second of the two hardest requirements in the spec and it is the one
doc 03 has no shape for at all. Every construct in doc 03 -- step, module,
Branch, Loop -- is about ONE RECORD. The allocation is inherently about a SET of
records that are decided separately, months apart.

---------------------------------------------------------------------------
Three things the framework has to give us
---------------------------------------------------------------------------
  1. A cross-record operation with a DECLARED INVARIANT, so that
     "no collateral item's allocations sum above its adjusted value at that
     date" (spec 10 acceptance 12) is checkable nightly over the whole book by
     the same declaration that the decision path uses. Two expressions of one
     invariant is how 340 double-allocations happen (change scenario 20).
  2. `reopens=`: a declared CONSEQUENCE EDGE. Changing facility F's claim on an
     item re-opens every other facility sharing it. From that declaration the
     framework computes, STATICALLY:
        - the affected set, for spec 5.11.3 requirement 4;
        - the authority union, for requirement 5 and spec 10 acceptance 14;
        - the list of clients who need a conversation nobody has had.
  3. Ordering independence. Spec 8.2: outcomes invariant under shuffling --
     "extended here to the cross-facility allocation, which is the case most
     likely to violate it". An allocation decided by whoever ran first is the
     defect; `policy=` is what replaces run order with a declared rule.

FRAMEWORK-DEMANDS D3.
"""

from decider2 import cross_record, module, param, sum_le, every_member_sharing
from decider2.frame import Join
from consumed.p05_origination import CollateralCover

# --------------------------------------------------------------------------
# The declared allocation policy. Spec 5.11.3 requirement 2.
#
# The worked case: C1 (industrial property, R4 760 000 adjusted) and C2 (debtors
# book, R1 640 000) against F1 (R4 000 000 term) and F2 (R2 500 000 overdraft).
#
#   pro rata      -> F1 ratio 0.98 type 2, F2 ratio 0.98 type 2
#   specific      -> F1 ratio 1.19 type 1, F2 ratio 0.66 type 2
#
# `security_type` keys the rate card. Under specific allocation F1 prices 140 bps
# lower -- R56 000 a year. The allocation is worth R56 000 a year and in the
# absence of a declared policy it is decided by whoever ran the assessment first.
# That sentence is why this is a `policy=`, versioned and approved, and not a
# sort order.
# --------------------------------------------------------------------------
allocation_policy = param.table(
    "collateral_allocation_priority",
    owner="legal",
    co_owner="portfolio_management",
    order=[
        "specific_where_contractually_linked",   # product 52's financed asset
        "priority_by_ranking",                   # first bond before second
        "maximise_total_adjusted_cover",
        "tie_break_longest_dated_facility",
    ],
    alternatives_evaluable=True,   # a client asking "why is my term loan priced
                                   # as partially secured when you hold a bond
                                   # worth more than it" is asking about policy
)

Allocation = cross_record(
    "collateral_allocation",
    over="collateral_item_id",
    members="facility_id",
    as_at="allocation_date",
    policy=allocation_policy,
    invariant=sum_le("allocated_cents", "adjusted_value_cents", tolerance=0),
    reopens=every_member_sharing("collateral_item_id"),
    # Spec 5.11.3 requirement 6: every facility's decision of record pins the
    # allocation it relied on, with its date and the underpinning valuation.
    # Reconstructing the pool's position as at any past date is then a query.
    pins=["allocation_id", "valuation_id", "valuation_date", "advance_rate"],
    # Consumed, not reimplemented: the advance rates and the cover-ratio ->
    # security_type derivation are project 05's (its 5.11), extended from 8
    # collateral classes to 14. The ALLOCATION is ours; the COVER is theirs.
    cover=CollateralCover,
)


def reallocate(trigger_facility_id: int, proposed_claim_cents: int) -> dict:
    """Run the policy over the whole pool and report every member that moved.

    Spec 5.11.3 requirement 4: any assessment that changes a facility's claim
    must re-test every other facility sharing it and STATE WHETHER ITS
    `security_type` CHANGED AND WHAT THAT DOES TO ITS PRICE.

    The worked amendment: raising F2 from R2 500 000 to R4 000 000 re-allocates
    C1, F1's ratio falls to 0.80, F1's type moves 1 -> 2, and F1 reprices upward
    by 140 bps. F1 is a facility nobody asked about, owned by a different
    relationship, whose client conversation has not happened, and whose own last
    decision of record said type 1.
    """
    pass  # apply policy; diff security_type per member; emit the moved set


def authority_of_the_affected_set(moved: list) -> int:
    """Spec 5.11.3 requirement 5, spec 10 acceptance 14.

    "An amendment to F2 that worsens F1 is not approvable at F2's authority
     level alone."

    Derived from `reopens=`, not recomputed here: the framework already knows
    the affected set statically, so the authority module takes the union rather
    than trusting whoever wrote the amendment to remember. See
    modules/authority/routing.py -- this function returns the set, routing takes
    the max over it.
    """
    pass  # union of member facilities; hand to routing


def revaluation_consequences(collateral_item_id: int, new_value_cents: int,
                             valued_on) -> list:
    """A revaluation is an EVENT, not a field update. Spec 5.11.3 requirement 7.

    C1 revalued on 2029-09-02 at R5 900 000 drops adjusted cover from R4 760 000
    to R4 130 000. Every facility relying on C1 changes its cover ratio on that
    date, whether or not anyone was assessing it.

    The defined consequence set: re-allocation, re-derived security types, an
    LTV covenant test where one exists (product 53), a signal to L3, a flag to
    the next review, and an amendment where the resulting position breaches a
    cover condition.

    Cadences by class -- property 24 months, plant 12, debtors monthly, listed
    securities DAILY -- so this fires continually. It is an EP-8-shaped event
    (a change to one subject re-opening decisions about others) that is not in
    spec 5.12's nine cascade triggers, and this project treats it as a tenth,
    bounded the same way. FRAMEWORK-DEMANDS D3 note 2.
    """
    pass  # emit the six consequences; route through Cascade with hops=1


def release_authority(collateral_item_id: int, post_release_cover: dict) -> int:
    """Spec 5.11.3 requirement 8: release requires the authority of the RESULTING
    POSITION across all facilities in the pool, not the size of the item.
    Releasing a R400 000 debtors cession that drops three facilities from type 1
    to type 4 is a level 5 act.
    """
    pass  # authority of the post-release position over the affected set


Security = module(reallocate, authority_of_the_affected_set,
                  revaluation_consequences, release_authority,
                  name="security_allocation", owner="legal",
                  co_owners=["portfolio_management", "business_credit_risk_policy"],
                  taps=["security_type@*", "allocation_id", "branch_path"])
