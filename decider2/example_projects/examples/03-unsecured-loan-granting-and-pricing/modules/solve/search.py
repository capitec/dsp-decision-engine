"""Stage 5.8 -- the solve.  The heart of the specification.

WHY `Loop` IS NOT ENOUGH.

Doc 03 §8.3 gives `Loop(should_continue, Body, carries=[...], max_iterations=N)`.
It is a good while-loop and it is not a search.  Everything §5.8 requires
beyond "it terminates" would have to be hand-written inside the body, where the
framework cannot see it, cannot render it, cannot verify it and cannot stop the
next author from writing it differently:

    requirement (§5.8)            with `Loop`                with `Search`
    --------------------------    -----------------------    -------------------
    1 bounded, 24 EVALUATIONS     max_iterations counts      `budget=` counts probes;
                                  ITERATIONS, and one        an iteration may probe
                                  iteration may probe        zero, one or many times
                                  twice
    2 terminating                 yes                        yes
    3 deterministic               yes                        yes
    4 CORRECT (true maximum)      hand-written, unverified   a theorem from the
                                                             partition (partition.py)
    5 tie-broken by a stated      an `if` in the body        `tie_break=`, declared,
      parameter                                              recorded, config-owned
    6 attributed: binding         hand-written into an       `binding=`, a closed
      constraint code             output the author          vocabulary the framework
                                  remembers to set           assigns and records
    7 re-checkable from scratch   nothing                    `independent_of=` on the
                                                             validator (lineage)
    8 exhaustively verified       a second implementation    `Search.exhaustive()`,
                                                             the SAME object

The last row is the one that decides it.  Writing the exhaustive verifier
separately means acceptance criterion 3 compares the search against a second
implementation of the search, and a shared misunderstanding passes.  Making
`exhaustive()` a MODE of the same declaration -- same domain, same probe, same
feasibility test, different strategy -- means the only thing that can differ is
the strategy, which is exactly what criterion 3 exists to test.

See FRAMEWORK-DEMANDS #12, #14, #16.
"""

from __future__ import annotations

from decider2 import Map, module, param, step
from decider2.money import Money
from decider2.search import (
    Binding,
    Budget,
    Evidence,
    InverseBracket,
    Search,
    Strategy,
    asc,
    desc,
    maximise,
)

from modules.pricing.probe import PriceCandidate
from modules.solve.partition import AMOUNT_PARTITION, Partitioning


@step(output="feasible")
def is_affordable(instalment: Money, max_affordable_instalment: Money) -> bool:
    """The feasibility test. One line, and it is the whole of it.

    Note what is NOT in here: no tolerance, no epsilon, no 'close enough'.  The
    spec's worked failure turns on four cents (R48 600 -> R1 560.04 against a
    ceiling of R1 560.00, "no, by 4 cents"), which is representable exactly in
    int64 cents and is not representable reliably in float64 rands.  This is
    the concrete reason for `money="int64_cents"` in pipelines/numerics.py.
    """
    pass


# ---------------------------------------------------------------------------
# THE SEARCH.
# ---------------------------------------------------------------------------

MaximumAffordableAmount = Search(
    name="max_affordable_amount",
    # -- the domain -------------------------------------------------------
    over="search_domain",                 # produced by partition.search_domain
    partition=AMOUNT_PARTITION,           # the union of declared breakpoints
    # -- the probe --------------------------------------------------------
    probe=PriceCandidate,                 # ONE composite, so a probe is countable
    feasible=is_affordable,
    objective=maximise("candidate_amount"),
    # -- the strategy, DECLARED rather than coded -------------------------
    #
    # Three phases, each provably sound given the partition:
    #
    #   1. BRACKET.  One inverse-annuity evaluation at the cheapest rate in the
    #      domain with minimal loadings gives a provable upper bound: no amount
    #      above it is feasible at ANY rate.  `tighten=2` re-runs it against
    #      the cheapest rate in the reduced range.  For the worked failure this
    #      cuts [R2 000, R95 000] to [R2 000, R59 500] -- 78 of 87 segments
    #      eliminated before the first probe.
    #
    #   2. SEGMENT SCAN, descending.  Probe the LEFT EDGE of each segment from
    #      the top down.  Within a segment the instalment is monotone
    #      non-decreasing, so the left edge is the segment's minimum: if it is
    #      infeasible the ENTIRE segment is infeasible and dies in one probe.
    #      The first segment whose left edge is feasible contains the answer,
    #      because every higher segment is entirely infeasible and every lower
    #      segment holds only smaller amounts.
    #
    #   3. BISECT WITHIN.  Binary search on the R100 grid inside that one
    #      segment, where monotonicity is a theorem.  ceil(log2(width)) probes.
    #
    # Worst case = 1 + S + ceil(log2(W)), where S and W come from the card's
    # axis.  That is a CLOSED FORM, evaluated at card validation against the
    # declared budget of 24 -- see rate_card.search_budget_is_satisfiable.
    # Acceptance criterion 4 ("no application EVER performs more than 24
    # evaluations, including on adversarial inputs") stops being a property
    # sampling can only fail to disprove.
    strategy=Strategy.bracket_scan_bisect(
        bracket=InverseBracket(
            inverse="instalment.advance_upper_bound",
            rate="cheapest_in_domain",
            loadings="minimal",
            tighten=param(2, ge=1, le=4, owner="credit_systems"),
        ),
        scan="segments_descending",
        within="bisect",
    ),
    # -- bounds -----------------------------------------------------------
    budget=Budget(
        probes=param(24, ge=8, le=64, owner="credit_systems"),
        # Change scenario 13: Product wants 32, Credit Systems wants the search
        # improved instead, and "both must be testable cheaply".  Both are one
        # number here -- the param for the first, `tighten=3` for the second --
        # and `decider2.impact(active, candidate, sample)` prices each.
        on_exhausted="no_offer",
        exhausted_outcome="refer",
        exhausted_queue=6,
        exhausted_reason=1420,
        # Not a truncated answer.  Not a best-effort.  An explicit, recorded,
        # MONITORED outcome -- because a search that can fail to find an answer
        # must be able to SAY so, and §13.6 asks whether that is expressible at
        # all.  It is, and it costs one field.
    ),
    # -- the tie-break, as a parameter of the specification ----------------
    tie_break=(desc("offered_amount"), asc("total_cost_of_credit"), asc("term_months")),
    # "The tie-break must be a parameter of the specification, not an accident
    # of evaluation order" (§5.8.5).  Declared here it is both: it is recorded
    # on every application, it is diffable, and it appears in the generated
    # review artefact as a sentence.
    # -- attribution -------------------------------------------------------
    binding=Binding(
        output="binding_constraint_code",
        vocabulary=[
            "BIND-AFF",   # the instalment ceiling
            "BIND-CAP",   # the policy amount_cap
            "BIND-REQ",   # the requested amount
            "BIND-MIN",   # the product minimum, R2 000
            "BIND-MAX",   # the product maximum, R500 000
            "BIND-TCR",   # the total cost ratio threshold
            "BIND-DUP",   # the scheduled in duplum test
            "BIND-CEIL",  # no cell at or below the statutory rate ceiling
            "BIND-EXH",   # evaluation ceiling reached
        ],
        # The framework assigns the code from WHICH CONSTRAINT PRODUCED THE
        # DOMAIN EDGE THE ANSWER LANDED ON, which it knows because the domain
        # was built from named constraints.  The author does not write a
        # fifteen-branch `if` and cannot get it wrong.
        attribute_from="domain_provenance",
    ),
    # -- evidence ----------------------------------------------------------
    #
    # 11 fields x up to 216 probes = 2 376 values per application.  At 14.2 M
    # applications a month that is 34 billion values, so §13.7 asks: "is
    # evidence capture a parameter, and if so, does turning it down change the
    # answer?"
    #
    # It is a parameter, and it CANNOT change the answer, because the framework
    # enforces that the evidence sink is write-only: no step may read from it,
    # so it is not in any output's lineage.  That is assertable statically, and
    # `assert_modes_agree` additionally runs the corpus at every evidence level
    # and requires identical decisions.
    #
    # `sampled` keys on `client_id`, not on a counter, so a replay reproduces
    # WHICH records were sampled.  A sample that cannot be reproduced is not
    # evidence.
    evidence=Evidence.level(
        realtime="full",
        batch=Evidence.sampled(rate=0.01, keyed_on="client_id"),
        always=["binding_constraint_code", "probe_count", "winning_probe"],
    ),
    writes=["offered_amount", "probe_count", "binding_constraint_code",
            "no_feasible_amount", "no_feasible_reason_code"],
)


# ---------------------------------------------------------------------------
# One term.  The Map in the pipeline runs this nine times.
# ---------------------------------------------------------------------------

SolveOneTerm = module(
    Partitioning,
    MaximumAffordableAmount,
    PriceCandidate.at(inputs={"candidate_amount": "offered_amount"}).named("final_pricing"),
    name="solve_term",
    taps=["probe_count", "binding_constraint_code", "segment_count"],
)


# ---------------------------------------------------------------------------
# Acceptance criterion 3, as a MODE of the same object rather than a second
# implementation.  Runs on every rate card version, because a new card can
# introduce a band-edge inversion no previous card had.
#
#     MaximumAffordableAmount.exhaustive()
#
# Same domain, same probe, same feasibility test; strategy replaced by
# "evaluate every candidate".  The only thing that can differ is the strategy,
# which is what the criterion exists to test.  Zero disagreements permitted.
# ---------------------------------------------------------------------------
