"""The twenty-four declared ordering constraints, and the two genuine cycles.

Spec 5.22: roughly forty hard ordering constraints hold between the eighteen
phases; twenty-four are non-obvious, entry-point-dependent or genuinely
circular.  "A cycle broken by declaration is defensible, explainable and
testable.  A cycle broken by accident is a defect that appears when a rule is
added on the far side of it, and it will be found by a client whose second
application was declined for a reason the first one did not mention."

===========================================================================
WHY THIS FILE EXISTS AT ALL
===========================================================================
Doc 03 8.1 settles ordering: "`|` is a sequence: written order is execution
order."  For a six-module pipeline that is complete and this file would be
noise.  Here it is not sufficient, in four distinct ways:

  1. SIX CONSTRAINTS ARE ENTRY-POINT-CONDITIONAL.  O-06's P04/P05 inversion is
     not "A before B"; it is "A before B on entry points 1 and 2 below R40 000,
     B before A on 5 and 6".  A single `|` order cannot state it, and eight
     `|` orders is eight flows.

  2. TWO ARE GENUINE CYCLES.  `|` cannot express a cycle at all, so the break
     is invisible: the reader sees P10 | P11 | P10 and has no way to know that
     the second P10 is a DECLARED break of a real circularity rather than
     someone running affordability twice for luck.

  3. ONE IS AN ISOLATION CONSTRAINT.  O-18: P17 must be UNABLE to read a shared
     intermediate.  That is not an order, it is a scope restriction, and doc 03
     3's pipeline scope has no mechanism for it -- every name in scope is
     visible to every later module.  Without it, spec 5.18's warning comes
     true: "if P17 can read `instalment` it will eventually assert
     `instalment == instalment`."

  4. ONE IS A CONCURRENCY CONSTRAINT.  O-24, in-flight application aggregation
     serialised per client: "order-dependence here is a concurrency constraint
     masquerading as an ordering one, and it is the only place in the flow
     where two decisions can affect each other."

So constraints are DATA, checked against flow.py at build.  The check runs in
both directions, and the second direction is the one that matters:

    forward   every declared constraint holds in the graph
    reverse   every cycle the graph contains is named by a `cycle_break`, and
              every `cycle_break` still names a real cycle

The reverse check is what makes spec 13 Q9 answerable -- "impossible to
reintroduce by adding a rule on the far side of it".  A new cap entry that
reads `instalment` creates a P12->P09 edge; if that closes a cycle nobody
declared, the build fails naming both edges and the rule that added one.  A
`cycle_break` that no longer breaks anything ALSO fails, so the declarations
cannot rot into folklore.

FRAMEWORK-DEMANDS #5, #6, #7.
"""

from __future__ import annotations

from decider2.ordering import (
    ConcurrencyConstraint,
    TwoPass,
    cycle_break,
    isolates,
    precedes,
    resolves_once,
)

from entrypoints.manifest import EP01_NEW_APPLICATION as EP1
from entrypoints.manifest import ENTRY_POINTS

# ---------------------------------------------------------------------------
# The ordinary ones.  Each is a requirement with a reason and an owner, not a
# convention.  `because=` is rendered verbatim into the reviewable artefact:
# spec 9.3's governance gap is that nobody owns the composition, and these
# strings are the only prose anywhere that describes it.
# ---------------------------------------------------------------------------

O01 = precedes(
    "p08.adjustments", "p08.grading", declared_by="T5",
    because="Grading keys off ADJUSTED PD. Grading first and adjusting after "
            "produces a grade that does not correspond to the PD recorded "
            "beside it, and a rate read from the wrong row.",
)

O02 = precedes(
    "p03.consent", "p04.bureau_wave", declared_by="T3", severity="statutory",
    because="A bureau enquiry without consent is an offence, and the enquiry "
            "leaves a footprint on the client's file that cannot be withdrawn. "
            "Not reversible by apology.",
)

O03 = precedes(
    "p02.resolution", "p03.consent", declared_by="T2",
    because="Consent is held against `client_id`. A probabilistic match at 0.89 "
            "confidence reading someone else's marketing preferences is a "
            "privacy incident, not a rounding error.",
)

O04 = resolves_once(
    "decision_date", at="p01", declared_by="T1", severity="statutory",
    forbids_symbols=("datetime.now", "date.today", "time.time"),
    because="Every effective-dated artefact resolves against it. A phase that "
            "re-reads 'today' replays SUCCESSFULLY with the wrong answer, "
            "which is worse than failing.",
)

O05 = resolves_once(
    "adjustment_set_id", at="p08.overlay_stack", declared_by="T4",
    consumed_by=["p09", "p10", "p12", "p13", "p14", "p15", "p16", "p17", "p18"],
    because="Two phases resolving independently, with a register change between "
            "them, put two different overlay stacks in one decision and its "
            "record would be indefensible.",
)

O07 = precedes("p06.income", "p07", declared_by="T5",
               because="14 of SC-A3's 61 characteristics are income-derived. "
                       "This is why the affordability chain is split across P06 "
                       "and P10 at all.")

O08 = precedes("p06.bureau_normalise", "p06.segment", "p07.selection",
               declared_by="T2",
               because="Segments 1 and 2 are distinguished by thin-file status, "
                       "which is bureau-derived. A wrong segment silently selects "
                       "a VALID scorecard and produces a plausible wrong score.")

O11 = precedes("p02.related_parties", "p09.CAP-0301", declared_by="T1",
               because="The graph query is expensive and set-shaped; the cap "
                       "check is cheap and record-shaped. Computing it late "
                       "means computing it inside a 7.5 ms budget.")

O12 = precedes("p09.class:regulatory", "p09.CAP-0118", declared_by="T3",
               severity="statutory", position="last_in_register",
               because="CAP-0118 is the only entry that may raise a ceiling. "
                       "Raising before a regulatory entry runs would let a "
                       "campaign authority exceed a statutory maximum, which is "
                       "not a policy error but an unlawful agreement.")

O13 = precedes("p12.rate_add_on", "p12.statutory_ceiling_check", declared_by="T3",
               severity="statutory",
               because="A card validated as compliant, plus a 60 basis point "
                       "overlay, can be non-compliant. Checking only at card "
                       "validation checks the wrong artefact.")

O14 = precedes("p12.rate", "p12.fee", "p12.premium", "p12.instalment",
               declared_by="T6", within_phase=True,
               because="The premium is a function of the amount FINANCED, which "
                       "includes the capitalised fee. Any other order computes a "
                       "premium on the wrong base.")

O19 = precedes("p17", "p18.reason_ranking", declared_by="T3",
               because="A validation failure is a reason. Ranking before P17 "
                       "produces a reason set that omits the reason the offer "
                       "was withdrawn.")

O20 = precedes("p17.pass", "p18.disclosure", declared_by="T3", severity="statutory",
               because="A quotation issued from an offer that failed validation "
                       "is a representation the Bank is bound by.")

O22 = resolves_once("rounding", per_value=True, declared_by="T1",
                    because="Rounding to R250 twice, or rounding an already-"
                            "rounded advance after a cap, changes the answer by "
                            "up to R250 and breaks the multiple-of-R250 assertion.")

O23 = resolves_once("consent_state", at="p03", read_by=["p04", "p14", "p16", "p18"],
                    declared_by="T3",
                    because="P18 needs consent state for the marketing disclosure. "
                            "Re-reading gives a FRESHER answer and an internally "
                            "inconsistent decision. Freshness loses to consistency, "
                            "and the choice is declared rather than defaulted.")

# ---------------------------------------------------------------------------
# The isolation constraint.  O-18.
#
# `isolates` is a SCOPE restriction, not an order.  It closes P17's input set:
# the phase may read only the three values named, plus effective-dated
# artefacts resolved from `decision_date`.  A step inside P17 whose signature
# names anything else does not compile, and the error names the value and says
# that P17 must re-derive it.
#
# Doc 03 3 has no mechanism for this.  Its pipeline scope is a flat pool where
# "most recent wins", and a module can read anything upstream of it.  For
# seventeen phases that is correct and convenient; for the eighteenth it is the
# defect.  FRAMEWORK-DEMANDS #7.
# ---------------------------------------------------------------------------

O18 = isolates(
    "p17",
    may_read=("offered_amount", "term_months", "product_code"),
    plus="effective_dated_artefacts",
    declared_by=("T1", "T3"),
    because="'Carry nothing forward' is easy in a flow with six intermediates "
            "and hard in a flow with 41 shared ones and up to 251 live versions "
            "of the obligations figure. The phase must be UNABLE to read them, "
            "not merely disciplined about it.",
    # The one exception, and it is the subtle one.  Spec 5.18: assertion 7 must
    # test against the obligations basis THE OFFER WAS PRICED ON, identified by
    # `loop_pass_index` and `value_basis_code`, not against "the" affordability
    # answer -- of which there are four.  So P17 may read the offer's
    # PROVENANCE, which is not a shared intermediate; it is a property of the
    # thing under test.  The distinction is load-bearing and is why `basis_of()`
    # exists (values/bases.py).
    may_read_provenance_of=("shipped_offer",),
)

# ---------------------------------------------------------------------------
# The concurrency constraint.  O-24.
# ---------------------------------------------------------------------------

O24 = ConcurrencyConstraint(
    "in_flight_application_aggregation",
    serialised_by="client_id",
    declared_by="T1",
    because="Two simultaneous applications from one client each see the other's "
            "exposure or neither does, depending on interleaving. This is the "
            "only place in the flow where two decisions can affect each other, "
            "and it is therefore the only place where acceptance criterion 10 "
            "(bit-for-bit determinism 'at any concurrency') has a real cost.",
    # Stated because it is the honest consequence: serialising per client puts a
    # lock on the 0.4% of entry point 1 volume that is a duplicate application
    # inside 48 hours, and the lock is outside the 120 ms budget.
    cost="excluded from the 120 ms; measured and published separately",
)

# ===========================================================================
# THE TWO GENUINE CYCLES
# ===========================================================================

O06 = cycle_break(
    name="O-06 fraud / bureau",
    edges=(("p05.rules", "p04.bureau_wave"), ("p04.bureau_wave", "p05.rules")),
    # The break is PER ENTRY POINT, and that is the whole difficulty.  This is
    # the constraint that a single `|` order cannot express and that eight
    # copies of the flow would express eight times, wrongly, within two years.
    broken_by={
        (1, 2): TwoPass(first="p05.rules", then="p04.bureau_wave",
                        when="requested_amount < R40 000"),
        (5, 6): TwoPass(first="p04.bureau_wave", then="p05.rules"),
    },
    because=("Low value: no sense paying for a bureau enquiry on a known "
             "fraudulent identity, and the enquiry itself helps a synthetic "
             "identity build a file. High value and re-price: 34 of the 188 "
             "fraud rules consume bureau-derived velocity and address history."),
    declared_by="T10", approved="FC-2027-02", review="2028-06-30",
    # What the reverse check enforces.  A 189th fraud rule that reads a
    # bureau-derived feature is fine on entry points 5 and 6 and is a BUILD
    # ERROR on 1 and 2, naming the rule and the feature.  Without this, the
    # rule would simply read a null on the low-value path and never fire -- a
    # silent dead rule that a firing count cannot distinguish from a rare one.
    enforces="no_new_edge_across_the_break",
    residual_recorded_as="fraud_rules_unevaluable_by_ordering",
)

O09 = cycle_break(
    name="O-09 product / affordability / amount",
    edges=(("p10", "p11"), ("p11", "p13"), ("p13", "p10")),
    broken_by=TwoPass(
        first="p10@product_neutral",   # most conservative buffer across candidates
        then="p11",
        again="p10@routed",            # the routed product's own buffer
        monotone="buffer_can_only_loosen",
    ),
    because=("Product eligibility is amount-keyed; the amount comes from the "
             "solve; the solve is bounded by affordability; affordability's "
             "buffer is grade- AND product-keyed. Running P10 twice with a "
             "conservative first pass is a declared, arguable compromise."),
    declared_by="T4", approved="CC-2027-03", review="2028-03-31",
    # Spec 5.12: if the re-run changes the routing decision, THE ROUTING STANDS
    # and the discrepancy is recorded -- because a second routing pass opens an
    # unbounded loop and this project has enough of those.  The residual is a
    # recorded number, not a silence, and it is monitored: a rising
    # `routing_provisional_delta` means the conservative buffer is drifting
    # away from the routed ones and the compromise is costing more than it was
    # approved to cost.
    residual_recorded_as="routing_provisional_delta",
    residual_monitored=True,
    # And the visibility requirement: spec 5.12 says this is "exactly the kind
    # of thing that must be VISIBLE IN THE REVIEWABLE ARTEFACT rather than
    # buried."  `surfaces=` puts it at the head of the P10 and P11 sections of
    # the generated rendering, with this `because=` string.
    surfaces_in_reviewable_artefact=True,
)

O10 = cycle_break(
    name="O-10 instalment cap (the third, in miniature)",
    edges=(("p09.instalment_entries", "p10"), ("p10", "p09.instalment_entries")),
    broken_by=TwoPass(first="p09.pass_one", then="p10", again="p09.pass_two",
                      partition_by="narrows"),
    because=("`instalment_cap` is seeded from affordability and two register "
             "entries reduce it further. The register runs in two passes and "
             "THE ENTRIES DO NOT KNOW WHICH PASS THEY ARE IN -- the pass is "
             "derived from which ceiling an entry declares `narrows=`, which "
             "keeps the split invisible in the register, as it must be."),
    declared_by="T4", approved="CC-2027-03",
)

CYCLE_BREAKS = (O06, O09, O10)
CONSTRAINTS = (O01, O02, O03, O04, O05, O07, O08, O11, O12, O13, O14,
               O18, O19, O20, O22, O23, O24) + CYCLE_BREAKS

# Asserted at build.  Spec 5.22 tabulates 24 of ~40; the other ~16 are the
# obvious ones (pricing before the solve, scoring before grading) which the
# graph proves on its own and which are therefore NOT declared here -- a
# declaration that restates what `|` already says is a second source of truth.
EXPECTED_DECLARED_CONSTRAINTS = 24
EXPECTED_CYCLES_IN_GRAPH = 3
