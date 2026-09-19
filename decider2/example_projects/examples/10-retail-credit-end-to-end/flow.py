"""THE flow.  Eighteen phases, eight entry points, one expression.

This file is 120 lines and it is the whole composition.  There is exactly one
of it.  There is not one per entry point, not one per product and not one per
latency class, and the reason is spec 4.1.1: eight copies work on day one and
guarantee that within two years the tax table is right in six of them.

Read the expression at the bottom top to bottom and you have read the flow.
Everything else in this project is a phase, a decision point, a table, an
interior document, a parameter or an overlay.

---------------------------------------------------------------------------
WHAT IS NOT HERE, DELIBERATELY
---------------------------------------------------------------------------
1. No `if entry_point_code == 7`.  Applicability is derived (entrypoints/
   manifest.py) and the build emits eleven specialisations.  Entry point 7 is
   not this graph with eleven phases switched off; it is a seven-phase kernel.

2. No `if product_code == 10`.  The product fan-out is a `Map` over the routed
   product set, so P09-P13 run n times over one implementation and P16 compares
   n commensurable results.  Ordering constraint O-17.

3. No date, anywhere, in this file or under phases/.  `decision_date` is fixed
   in P01 and a lint forbids `datetime.now` / `date.today` / `time.time` under
   phases/ and values/.  O-04: a phase that re-reads "today" replays
   SUCCESSFULLY with the wrong answer, which is worse than failing.

4. No budget numbers.  They are in budgets/*.toml because a budget is a
   property of (phase, entry point) and this file has no entry point in it.

5. No ordering assertions.  They are in ordering.py, where they are checked
   against this graph rather than implied by the order of the `|`s.  Written
   order here IS execution order (doc 03 8.1) -- ordering.py exists because
   24 of the ~40 constraints are not expressible as "A before B in one list":
   six are entry-point-conditional, two are genuine cycles, one is an isolation
   constraint and one is a concurrency constraint.
"""

from __future__ import annotations

from decider2 import Map, fuse
from decider2.collections import Collection

from loops.l1_consolidation import ConsolidationLoop
from phases import (
    P01, P02, P03, P04, P05, P06, P07, P08, P09, P10,
    P11, P12, P13, P15, P16, P17, P18,
)

# ---------------------------------------------------------------------------
# The routed product set.  Up to three eligible products on entry point 1; up
# to four inside P14's search.  Capacity is declared because the record tier
# has no heap, and a product dropped from routing is dropped WITH A REASON --
# spec 5.12(b): "every substitution is recorded with the rule that made it".
# ---------------------------------------------------------------------------

routed_products: Collection[int] = Collection(
    name="routed_products", capacity=6, source=P11.outputs.eligible_products,
    absent_because="product_withdrawal_reason_code",
)

# ---------------------------------------------------------------------------
# The flow.
# ---------------------------------------------------------------------------

retail_credit = (
    # -- Admission.  The only place the phase set is decided, and therefore the
    #    only place "why did phase 14 not run for this client" is answerable
    #    without inference (spec 5.2).
    P01
    # -- Identity, then consent, then acquisition.  O-03 then O-02: consent is
    #    held against `client_id`, and a bureau enquiry without consent is an
    #    offence that leaves a footprint that cannot be withdrawn.
    | P02
    | P03
    # -- Fraud and acquisition.  Their order INVERTS by entry point (O-06) and
    #    that is a genuine cycle in the dependency graph.  `cycle_break` in
    #    ordering.py resolves it; here both appear once, in the order that
    #    holds for the majority case, and the specialisation for entry points
    #    5 and 6 emits them the other way round.  There is no second flow.
    | P04
    | P05
    # -- Feature derivation.  510 values, 11 downstream phases, and the first
    #    four links of the affordability chain -- pulled forward out of P10
    #    because scoring, grading and the cap waterfall all need income before
    #    the affordability verdict exists (spec 5.7).
    | P06
    | P07
    | P08                                   # resolves the overlay stack ONCE (O-05)
    # -- The first cycle break.  P10 runs product-neutrally with the most
    #    conservative buffer across the candidate set; P11 routes on the
    #    resulting provisional amount; P10 re-runs with the routed product's
    #    own buffer, which can only be the same or more generous.  Routing does
    #    NOT re-open.  O-09 / L3.
    | P09.pass_one                          # amount, term, grade entries
    | P10.at_site("product_neutral")
    | P11
    # -- The product fan-out.  One implementation, n products, n commensurable
    #    results.  O-17.
    | Map(
        over=routed_products,
        as_="product_code",
        body=(
            P09.pass_one
            | P10.at_site("routed")
            | P12
            | P13
            | P09.pass_two                  # instalment-class entries only (O-10)
        ),
        capacity=6,
        collect=["offers", "amount_cap", "term_cap", "instalment_cap",
                 "nominal_annual_rate", "max_affordable_instalment"],
        absent_because="product_withdrawal_reason_code",
    )
    # -- The four-phase loop.  Entered only when affordability failed AND the
    #    client is consolidation-eligible.  Its body is the phase objects
    #    above, not copies of them.  loops/l1_consolidation.py.
    | ConsolidationLoop
    # -- Limit assignment.  Entry points 2, 3, 4 only, by derivation; its
    #    population-level allocation is suppressed on 2 and 4 by DECLARATION.
    | P15
    # -- Assembly and arbitration.  The phase no isolated flow owns, because
    #    composing four notions of "best" requires a fifth (spec 5.17).
    | P16
    # -- Final validation, which is UNABLE to read a shared intermediate (O-18),
    #    and disclosure.
    | P17
    | P18
).named("retail_credit")


# ---------------------------------------------------------------------------
# Build.  Eleven specialisations from one graph.
# ---------------------------------------------------------------------------
#
#   decider2 build retail_credit --entrypoints entrypoints/ --verify
#
# emits one compiled variant per `phase_set_id`, asserts each against its
# manifest's [derived] block, writes the decision-point registry and the
# per-phase-set bitmap layout (navigability/index.py), checks every ordering
# constraint in ordering.py against this graph, and asserts a runtime load
# triggers zero compilations.
#
# Fusion: `apply()` never fuses across module boundaries (doc 02 1.2), which
# here means never across PHASE boundaries -- and that is not a performance
# choice, it is what makes per-phase budget measurement possible at all.  The
# realtime path deviates from doc 02's "fuse maximally" for the same reason.
# FRAMEWORK-DEMANDS #16.
