"""The partition.  This file is the answer to §13.5.

    "How does non-monotonicity get expressed as a REQUIREMENT that the
     implementation must satisfy, rather than as a property somebody hopes
     holds?  What does a specification of 'correct' look like here that is
     checkable without evaluating all 4 981 candidates?"

The answer is that the instalment is not non-monotone.  It is **piecewise
monotone, with breakpoints that every artefact in the probe already knows.**

    * the rate card knows its 96 amount-band edges -- they are an axis, in a
      file, authored by Treasury (rate_card.py)
    * the fee schedule knows its one kink, at R12 700, and knows it is derived
      rather than written down (fees.py)
    * the credit life table knows its term-band edges -- irrelevant within a
      single term, and declared anyway so the declaration is uniform
    * a rate add-on overlay knows its amount-band scope edges, and says so with
      `contributes_breakpoints` (adjustments/points.py)

Between two adjacent breakpoints every step function in the probe is constant,
and every remaining factor is declared monotone non-decreasing in the amount
(`@monotone_in` on the fee, the premium and the annuity).  So within a segment,
instalment(a) is monotone non-decreasing in a.  That is a THEOREM given the
partition, not an empirical property, and it holds for any card -- including
one with 58 inversions instead of 41, and one re-banded to 120 bands.

The specification of "correct" is therefore three checkable obligations on the
ARTEFACTS, not on the search:

    (P1) every artefact the probe reads declares its breakpoints in the search
         variable;
    (P2) the partition is the union of those declarations, clipped to the
         domain;
    (P3) every remaining factor declares its monotonicity in the search
         variable.

If all three hold, "the search returns the true maximum over the R100 grid" is
a proof, not a test.  The 10 000-application exhaustive check in acceptance
criterion 3 then changes meaning entirely: it stops being a test of the search
and becomes a **regression test on the breakpoint declarations** -- and when it
fails, it fails naming an artefact that forgot to declare something, which is a
fixable defect, rather than saying "the search is wrong somewhere", which is
not.  See FRAMEWORK-DEMANDS #11, #12.
"""

from __future__ import annotations

from decider2 import module, step
from decider2.collections import Collection
from decider2.money import Money
from decider2.search import Partition, breakpoints_of

from modules.pricing.probe import PriceCandidate


@step(output="search_domain")
def search_domain(
    requested_amount: "Maybe[Money]",
    amount_cap: Money,
    product_minimum: Money,
    product_maximum: Money,
    rounding_unit: Money,
) -> "Interval[Money]":
    """[R2 000, min(requested, amount_cap, R500 000)], snapped to the R100 grid.

    THE CAPS ARE APPLIED TO THE DOMAIN, NEVER TO THE ANSWER.  This is the
    structural reason acceptance criterion 2 holds -- "every offer passes an
    independent from-scratch re-check, zero failures over 250 000".  The silent
    failure mode §5.8(7) describes is "an offer that was affordable when the
    search evaluated it but is not affordable at the amount finally written,
    because a cap moved it into another band afterwards".  If no stage between
    the solve and the offer may reduce `offered_amount`, that failure is
    unreachable -- and "no stage downstream of Solve writes `offered_amount`"
    is a STATIC LINEAGE ASSERTION, checked at build.  See
    modules/validation/independent.py.

    A null `requested_amount` -- 31% of app-channel volume, the "tell me what I
    qualify for" journey -- simply drops out of the min.  It is not defaulted
    to the product maximum, because BIND-REQ and BIND-MAX are different binding
    constraints and the client is entitled to know which one applied.
    """
    pass


AMOUNT_PARTITION = Partition(
    variable="offered_amount",
    unit=Money("100.00"),           # the R100 grid.  Advances round DOWN to R100.
    sources=[
        breakpoints_of("flex_rate_card.amount"),          # 96 band edges
        breakpoints_of("fees.initiation_fee"),            # 1 derived kink, R12 700
        breakpoints_of("credit_life.credit_life_premium"),  # inherited from the fee
        breakpoints_of("pricing.rate"),                   # the RATE_ADD_ON's scope edges
    ],
    # (P3).  Every factor not constant within a segment must be declared
    # monotone in the search variable.  A probe containing an undeclared
    # non-monotone factor is a BUILD ERROR naming the step, not a wrong answer
    # discovered by the 10 000-application sample four months later.
    require_monotone_within_segment=PriceCandidate.outputs("instalment"),
)


@step(output="segments")
def segments(search_domain: "Interval[Money]", tables) -> Collection["Segment"]:
    """The domain cut at every declared breakpoint, ordered descending.

    For client W's grade-9 twin at 60 months over [R2 000, R95 000] this is 87
    segments.  The bracket eliminates 78 of them before a single probe runs.
    """
    pass


Partitioning = module(search_domain, segments, name="partition",
                      taps=["search_domain", "segment_count"])
