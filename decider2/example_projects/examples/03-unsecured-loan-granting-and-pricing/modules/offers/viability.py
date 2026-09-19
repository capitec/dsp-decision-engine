"""Stage 5.9 (part 1) -- suppression and deduplication over a ragged offer set.

The shape problem: everything in doc 03 is a scalar.  `Loop` carries scalars.
There is no map, no filter over a collection, no reduce, and no way for a
module to produce between zero and nine of anything.  This project's OUTPUT is
a ragged collection -- "between 0 and 9 offers per application" -- and so are
its bureau accounts (0..80), its obligations (0..80), its enquiries (0..120),
its consent records (1..12) and its in-flight applications (0..6).

So: `Collection[T]` with a DECLARED CAPACITY, `Map`, `MapFilter`, `Collapse`
and `Rank`.  Capacity is required for the same reason `max_iterations` is
required -- the record tier has no heap, so a fixed-capacity struct-of-arrays
is the only representation, and declaring it is how an overflow becomes a
handled value rather than an exception.  See FRAMEWORK-DEMANDS #18.

The suppression rules themselves are five rows a product manager edits weekly.
Tabular, so a generic kernel: adding Compliance's sixth rule (change scenario 7
-- "no offer whose instalment exceeds 35% of net monthly income") is a row plus
one registered predicate, not a release.
"""

from __future__ import annotations

from decider2 import Collapse, MapFilter, param, predicate
from decider2.money import Money
from decider2.offers import SuppressionRules, pct

# ---------------------------------------------------------------------------
# Terms removed BEFORE any pricing happens.  Client W's term_cap of 72 removes
# 84, and that removal is recorded with its reason -- so a branch consultant
# asked "why is there no 84-month option?" has an answer that is not "there
# just isn't one".
# ---------------------------------------------------------------------------


@predicate(id="term_within_cap")
def term_within_cap(term_months: int, term_cap: int) -> bool:
    """Term at or below the policy `term_cap`. Reason 1450 when it is not."""
    pass


@predicate(id="term_in_segment_list")
def term_in_segment_list(term_months: int, segment_code: int, tables) -> bool:
    """Term in the per-segment permitted list. New-to-bank is not offered 84 months.
    Reason 1451."""
    pass


# ---------------------------------------------------------------------------
# Minimum viable offer rules.  Five, each with its own reason code, each
# independently tunable by Unsecured Lending Product, all applicable reasons
# recorded and the most severe reported.
# ---------------------------------------------------------------------------

MinimumViableOffers = MapFilter(
    name="minimum_viable_offers",
    over="offers",
    rules=SuppressionRules(
        interior="config/flex_loan/interiors/minimum_viable_offers.json",
        capacity=8,
        # ALL applicable reasons are recorded, ranked, and the most severe is
        # reported.  Client W's 72-month offer is suppressed by two rules --
        # total cost ratio 2.02 against 1.85, AND scheduled in duplum, charges
        # of R96 759.76 against an advance of R95 000.  A filter that returns a
        # bool records one of those, arbitrarily.  `MapFilter` collects them
        # all, which is why it is a kind and not a lambda.
        collect_all_reasons=True,
        rank_by="core.reason_codes.severity",
    ),
    writes=["offers", "suppression_reason_codes", "primary_suppression_reason_code"],
)

# ---------------------------------------------------------------------------
# Deduplication.  "Two surviving offers whose amounts are EQUAL and whose
# instalments differ by less than 2% collapse to the one with the lower total
# cost."  The tolerance is a parameter.
#
# Client W's 36, 48 and 60 month offers are all at R95 000 -- equal amounts --
# but their instalments differ by far more than 2%, so all three survive.  The
# rule is therefore not "same amount collapses"; it is "same amount AND
# near-identical instalment collapses", and getting that wrong costs the client
# two of their three best offers.
# ---------------------------------------------------------------------------

Deduplicate = Collapse(
    name="deduplicate",
    over="offers",
    key=["offered_amount"],
    within={"instalment": pct(param(0.02, ge=0, le=0.25, owner="product"))},
    keep="min_by:total_cost_of_credit",
    writes=["offers", "deduplication_decisions"],
    # The DECISIONS are output, not just the survivors.  §7.2 requires "the
    # deduplication decisions" in the persisted record, and a collapse that
    # emits only its survivors cannot supply them.
)
