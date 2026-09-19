"""The 41 shared intermediates, declared.

Spec 5.21: forty-one values produced in one phase and read in another; twelve
have five or more consumers; three have nine or more.  "This is the connective
tissue that no isolated specification can show, and it is where a large flow's
dependencies actually live."

---------------------------------------------------------------------------
WHY THESE ARE DECLARED AND NOT JUST WIRED BY NAME
---------------------------------------------------------------------------
Doc 03 2 wires by parameter name and nothing else, and that is right for a
six-module pipeline.  At eighteen phases it fails in three ways the spec names
directly:

 1. Spec 5.21 "Producing a value is an obligation to consumers you cannot see."
    P02 builds a related-party set nothing in P02 needs, because an exposure
    rule five phases later is too expensive to compute in a 7.5 ms budget.  In
    a name-wired graph that computation looks like dead code to whoever is
    tidying up.  `consumed_by=` makes the obligation visible AT THE PRODUCER,
    and deleting a declared consumer's read is a build error, not a cleanup.

 2. Doc 03 2.1's ambiguity rule fires on frame-column shadowing only.  Here the
    dangerous case is a phase re-DERIVING a value another phase produced.  Spec
    5.7: "an implementation that re-derives income inside P10 is wrong EVEN WHEN
    IT HAPPENS TO AGREE."  `produced_by=` is single and exclusive, so a second
    producer of `net_monthly_income` anywhere in the flow does not compile.

 3. The three axes (values/bases.py) have to attach somewhere.

This file is also the N1 index's spine: `where("net_monthly_income")` starts
here and walks out to the eleven consumers and to every place the value can be
set.  Spec 5.27 N1 is "find EVERY place it can be set", and a human cannot
enumerate 132 places by reading -- so the enumeration is generated, and this is
what it is generated from.
"""

from __future__ import annotations

from decider2.values import shared_value

from values.bases import ADJUSTMENT, BASIS, PASS

# ---------------------------------------------------------------------------
# The three with nine or more consumers.
# ---------------------------------------------------------------------------

decision_date = shared_value(
    "decision_date",
    dtype="date",
    produced_by="p01.admission",
    consumed_by="*",                       # all 18.  The only `*` in the file.
    axes=(),
    # Spec 5.2 and ordering O-04.  This is the mechanism by which a 2027
    # decision reproduces in 2034, so it is immutable after P01 and a lint
    # forbids `date.today` / `datetime.now` anywhere under phases/.
    immutable_after="p01",
    forbids_symbols=("datetime.now", "date.today", "time.time"),
)

net_monthly_income = shared_value(
    "net_monthly_income",
    dtype="Money",
    produced_by="p06.income",
    consumed_by=["p07", "p09", "p10", "p11", "p12", "p13", "p14",
                 "p15", "p16", "p17", "p18"],          # 11, spec 5.21
    axes=(BASIS,),                          # EP_SUBSTITUTED only; never hypothetical
    permitted_bases=("actual", "entry_point_substituted"),
    # Spec 5.7: `income_band_code` derived from a payslip on entry point 1 and
    # from internal deposit history on entry points 3 and 4 is the same name,
    # the same band table and a different number.  `value_basis_code` 4 exists
    # so the difference is VISIBLE IN THE RECORD rather than inferred from the
    # entry point by whoever is reading it.  Acceptance criterion 9 lives here.
    substitution_by_entry_point={3: "tier5_deposit_history", 4: "tier5_deposit_history"},
)

risk_grade = shared_value(
    "risk_grade",
    dtype="int8",
    produced_by="p08.grading",
    consumed_by=["p09", "p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17"],
    axes=(BASIS, ADJUSTMENT),
    # Four live versions in a consolidation assessment: actual-adjusted,
    # actual-unadjusted, hypothetical-adjusted, hypothetical-unadjusted.  Some
    # scorecard characteristics are obligation-derived, so a hypothetical
    # settlement changes the grade, which changes the rate, which changes the
    # instalment, which changes the affordability test.  Spec 5.21.1.
)

adjustment_set_id = shared_value(
    "adjustment_set_id",
    dtype="int16",
    produced_by="p08.overlay_stack",
    consumed_by=["p09", "p10", "p12", "p13", "p14", "p15", "p16", "p17", "p18"],
    axes=(),
    # O-05.  Resolved ONCE per decision, for the whole decision, by a phase that
    # applies almost none of the overlays it resolves.  Five later phases consume
    # overlays P08 does not apply.  If two phases resolved the register
    # independently and it changed between them, one decision would contain two
    # overlay stacks and its record would be indefensible.
    resolved_once=True,
    resolution_is_pinned=True,
)

# ---------------------------------------------------------------------------
# The hard case.  Spec 5.21.1.
# ---------------------------------------------------------------------------

existing_obligations = shared_value(
    "existing_obligations",
    dtype="Money",
    produced_by="p06.obligations",
    consumed_by=["p07", "p09", "p10", "p13", "p14", "p15", "p17"],
    axes=(BASIS, ADJUSTMENT),
    permitted_bases=("actual", "hypothetical"),
    max_live_versions=251,                  # 1 actual + up to 250 hypothetical
    #
    # The per-consumer basis CONTRACT.  This is the declaration that makes spec
    # 5.21.1 requirement 2 enforceable: the build walks every read of this value
    # and checks it against this table.  A read from a phase not listed, or a
    # read in a basis the phase is not permitted, does not compile.
    #
    basis_contract={
        "p07": "inherited",   # obligation-derived characteristics; scenario-aware
        "p09": "actual",      # the Bank's real exposure does not fall on a scenario
        "p10": "inherited",   # one arithmetic, four evidence modes
        "p13": "inherited",
        "p14": "explicit",    # it is the producer of the hypotheticals
        "p15": "actual",
        "p17": "basis_of",    # the basis the offer was PRICED on.  Spec 5.18.
    },
    # Spec 5.21.1 requirement 4: the disclosure uses the actual figure and the
    # assessment uses the hypothetical, and the two appear on the same page of
    # the same document.  P18 reads BOTH, explicitly, and the record renderer
    # refuses to emit a consolidation disclosure carrying only one.
    disclosure_requires=("actual", "hypothetical"),
)

# The other eight values with the same property (spec 5.21.1).  Declared with
# the same shape; elided here for length, present in the real file.
#   revolving_utilisation, worst_arrears_months, total_exposure,
#   discretionary_income, max_affordable_instalment,
#   affordability_verdict_code, probability_of_default
# ...and risk_grade above.

max_affordable_instalment = shared_value(
    "max_affordable_instalment",
    dtype="Money",
    produced_by="p10.capacity",
    consumed_by=["p12", "p13", "p14", "p15", "p16", "p17"],
    axes=(BASIS, ADJUSTMENT, PASS),         # all three.  The only value with all three.
    basis_contract={"p17": "basis_of"},
    # Spec 5.11: it runs up to five times in one decision -- once product-neutral
    # before P11, once per routed product after it, and up to three more inside
    # L1.  "The record must say which run supported the offer."  PASS is not a
    # convenience here; without it "the affordability verdict" is ambiguous
    # between four of them and spec 9.2 property 1 fails.
    site_attribution_required=True,
)

instalment = shared_value(
    "instalment",
    dtype="Money",
    produced_by="p12.annuity",
    consumed_by=["p10", "p13", "p14", "p16", "p17", "p18"],
    axes=(BASIS, ADJUSTMENT, PASS),
    # Note the cycle in the consumer list: P12 produces it, P10 reads it, and
    # P10 is upstream of P12 in the phase order.  That edge is legal ONLY
    # because it is carried by L1/L3 and declared in ordering.py.  An
    # undeclared edge of this shape is a build error (see ordering.py).
    back_edges_permitted_via=("L1", "L3"),
)

# ---------------------------------------------------------------------------
# The rest.  Twenty-three values with exactly two consumers, plus the ten
# between three and seven, declared identically.  The full list is what
# `decider2 values --graph` renders, and it is the input to blast radius.
# ---------------------------------------------------------------------------

REGISTER = (
    decision_date, net_monthly_income, risk_grade, adjustment_set_id,
    existing_obligations, max_affordable_instalment, instalment,
    # ... 34 more
)

# Asserted at build against what the flow actually wires.  Spec 5.21: 41.
EXPECTED_SHARED_INTERMEDIATES = 41
EXPECTED_WITH_FIVE_PLUS_CONSUMERS = 12
EXPECTED_WITH_NINE_PLUS_CONSUMERS = 3
