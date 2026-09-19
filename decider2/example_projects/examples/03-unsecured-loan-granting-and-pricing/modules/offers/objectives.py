"""Stage 5.9 (part 2) -- ranking and recommendation.

§13.13: "Where does the recommendation objective live?  It is a parameter with
three values that changes a scoring computation over a variable-length
collection.  Is that configuration, or is it three implementations selected by
configuration, and what does the evidence record look like either way?"

THREE IMPLEMENTATIONS, SELECTED BY CONFIGURATION.  And the mechanism needs no
extension at all -- this is doc 03 §8.2's routing `Branch` with a router that
reads a param, and it is the one hard part of this spec that doc 03 answers
cleanly and completely.  Worth saying plainly, because most of this sketch is
complaint.

Why not one module with an `if params.objective == ...`:

  * change scenario 5 says `lowest_total_cost` "currently picks the smallest
    loan, which nobody wanted" and "is re-specified twice".  Three separately
    versioned artefacts means that re-specification touches one of them, is
    reviewed by one owner, and appears in one diff.
  * all three arms are in `lineage()` and in the rendered artefact whether or
    not they are selected, so "what are the other two?" is answerable without
    reading config.
  * `branch_path` records which fired, as one int64 compile-time immediate --
    so "which objective was in force?" is already in the audit record with no
    extra machinery.  A client asking in 2029 why 60 months was recommended is
    asking about a parameter value in 2026, and the record has it.
  * change scenario 4 -- switched on a Tuesday and back on the Thursday -- is
    then a params swap: microseconds, no compile, no deploy, and
    `decider2.impact()` prices it beforehand.
"""

from __future__ import annotations

from decider2 import Branch, Rank, module, param, step
from decider2.money import Money

LARGEST_AMOUNT, LOWEST_TOTAL_COST, BEST_EXPECTED_VALUE = 1, 2, 3


@step(output="objective_index")
def objective_index(
    recommendation_objective: int = param(
        LARGEST_AMOUNT, choices=[1, 2, 3], owner="product",
        description="1 largest_amount | 2 lowest_total_cost | 3 best_expected_value",
    ),
) -> int:
    """Which objective is in force. Recorded on every application."""
    pass


LargestAmount = Rank(
    name="largest_amount",
    over="offers",
    by=[("offered_amount", "desc"), ("total_cost_of_credit", "asc")],
    flag="is_recommended",
)

LowestTotalCost = Rank(
    name="lowest_total_cost",
    over="offers",
    by=[("total_cost_ratio", "asc")],       # per rand advanced, NOT absolute
    flag="is_recommended",
)


@step(output="expected_margin")
def expected_margin(
    total_cost_of_credit: Money,
    offered_amount: Money,
    term_months: int,
    probability_of_default: float,
    lgd: float = param(0.72, ge=0, le=1, owner="treasury"),
    average_ead: float = param(0.55, ge=0, le=1, owner="treasury"),
    funding_cost: float = param(0.0875, ge=0, le=1, owner="treasury"),
) -> float:
    """(total cost - advance - funding cost) x (1 - PD over term)
       - LGD x EAD x PD over term, with PD over the term derived from the 12-month PD."""
    pass


BestExpectedValue = module(
    expected_margin,
    Rank(name="rank_by_margin", over="offers", by=[("expected_margin", "desc")],
         flag="is_recommended"),
    name="best_expected_value",
)

Recommend = Branch(
    objective_index,
    [LargestAmount, LowestTotalCost, BestExpectedValue],
    modifies=["is_recommended", "offer_rank"],
    taps=["branch_path", "recommendation_objective"],
)

# ---------------------------------------------------------------------------
# The ragged outcome.  Between 0 and 9 offers.
#
# "Zero offers with no policy decline is a DISTINCT OUTCOME -- 'declined on
# affordability' -- and must not be reported as an approval with an empty set."
# So `outcome_code` is derived from (offer_count, decline_reason_codes,
# referral_queue_code) by a three-row decision table rather than by
# `if len(offers) == 0: approve`, which is the shape that produces the bug.
# ---------------------------------------------------------------------------
