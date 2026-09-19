"""Eight entry points, one flow, no copies.

Spec 4.1.1 states the problem this file exists to solve:

    "Eight copies would work on day one.  Eight copies would also guarantee that
     within two years the tax table is correct in six of them, that a cap rule
     added in one is missing in three, and that the entry-point agreement
     criterion in 10 can never be met."

===========================================================================
THE CENTRAL CLAIM OF THIS PROJECT
===========================================================================
Applicability is DERIVED by default and DECLARED only where it is a policy
decision -- and a declaration must carry a reason, an owner and an approval.

Derivation rule, in full: a phase runs on an entry point iff every input in its
declared interface is satisfiable from that entry point's input schema plus the
outputs of the phases that precede it and are themselves applicable.  A decision
point inside an applicable phase runs iff the same holds for its own inputs.

That one rule buys most of the eight-entry-point problem for free:

  * P05 fraud does not run on entry points 3 and 4 because 188 of its 205
    decision points read device, session and consortium signals that a batch
    cycle does not have.  Nobody wrote "skip fraud on batch".
  * P03 runs 9 of its 62 decision points on entry point 7 because the other 53
    read `client_id`, and entry point 7 has no resolved client.  Spec 4.1's
    "REDUCED: product and channel availability only" is a CONSEQUENCE, not an
    instruction.
  * P17 runs 6 of product 10's 14 assertions on entry point 7 because assertion
    7 reads `max_affordable_instalment`, which P10 did not produce, and 8, 9,
    10, 11, 12, 13 read values that need a client.  Spec 4.1 explicitly leaves
    open whether that is "one phase with a declared applicability per assertion,
    or two phases that will diverge within a year"; derivation answers it
    without choosing, because nobody declares anything.

What derivation does NOT cover, and must not be allowed to fake, is the
genuinely policy-shaped divergence:

  * P15's population-level allocation must not run on entry points 2 and 4.
    Its inputs ARE available on entry point 4.  It is suppressed because an
    entry-point-4 pre-approval is indicative, not applied, and because entry
    point 2 is a client request and is not budget-constrained: "a client who
    asks in a month when she ranked 412 000th gets the increase, and that is
    correct, not a defect" (spec 5.16).  That is a decision by T4 and T8, and
    it is DECLARED, with a reason.
  * P09's twelve regulatory entries on entry point 7.  Derivation would leave
    more of them applicable than policy wants.
  * O-06's ordering inversion, declared per entry point (ordering.py).

===========================================================================
THE METRIC THAT TELLS YOU THE SCALE WALL IS ARRIVING
===========================================================================
Spec 13 Q25 asks what evidence would show the boundary had been crossed, and
lists "the count of reduced-form phases growing".  This file makes that a
number rather than an impression:

    DECLARED_APPLICABILITY_COUNT = 19

Nineteen places where the eight entry points diverge for a reason that is not
data availability.  Every one is a policy decision with an owner and an
approval reference.  A rise past ~35 means the entry points are diverging for
reasons nobody chose, and that -- not decision-point count, not team count -- is
this project's stated tripwire for splitting the flow.  See README 12.

FRAMEWORK-DEMANDS #1, #2, #3.
"""

from __future__ import annotations

from decider2.entrypoint import Declared, EntryPoint, Derived, PhaseSet

# ---------------------------------------------------------------------------
# The eight.  Each is a small declaration; the phase set is NOT in it.
# ---------------------------------------------------------------------------

EP01_NEW_APPLICATION = EntryPoint(
    code=1, name="new_credit_application", mode="realtime",
    input_schema="schemas/ep01_application.json",
    record_shape="record_shapes:FullAssessment",
    budget="budgets/ep01.toml",
    degradation_profile="degradation/modes.toml#interactive",
    volume_per_day=55_000, peak_per_second=9,
    # P14 is conditional, so `phase_set_id` is NOT derivable from
    # `entry_point_code` alone (spec 4.6).  The conditional is a property of
    # L1's entry condition, declared once in loops/l1_consolidation.py, and the
    # two resulting phase sets are enumerated at build.
    conditional_phase_sets=("P14_absent", "P14_present"),
    latency_class_on={"P14_present": "budgets/ep01_loop.toml"},
    manifest="entrypoints/ep01_new_application.toml",
)

EP04_CAMPAIGN = EntryPoint(
    code=4, name="campaign_preapproval", mode="batch",
    input_schema="schemas/ep04_client_mart.json",
    record_shape="record_shapes:CampaignRow",
    budget="budgets/ep04.toml",
    degradation_profile="degradation/modes.toml#batch",
    volume_per_cycle=14_200_000, window="6h", sustained_rate=657,
    # Skips exactly two phases, and both fall out of derivation: identity is
    # pre-resolved in the mart (P02's inputs are already outputs) and P05 has no
    # event to assess.  It runs the other SIXTEEN, which makes it the most
    # complete traversal in the estate.  Nothing in this declaration says so.
    manifest="entrypoints/ep04_campaign_preapproval.toml",
)

EP07_QUOTATION = EntryPoint(
    code=7, name="quotation", mode="realtime",
    input_schema="schemas/ep07_quotation.json",
    record_shape="record_shapes:QuotationRecord",
    budget="budgets/ep07.toml",
    degradation_profile="degradation/modes.toml#quotation",
    volume_per_day=22_000,
    # Spec 4.1: "no credit decision record is created -- and yet every number
    # must be disclosure-accurate, because a quotation is a representation the
    # Bank can be held to, and a quotation that understates the initiation fee
    # is a refundable overcharge across every agreement written from it."
    emits_outcome_code=False,
    emits_decision_record=False,
    emits_quotation_record=True,
    manifest="entrypoints/ep07_quotation.toml",
)

EP08_WHATIF = EntryPoint(
    code=8, name="what_if", mode="on_demand",
    # Its phase set is the phase set of the decision being intervened on.  That
    # is not a ninth configuration; it is `phase_set_id` read from the record.
    phase_set_from="decision_of_record.phase_set_id",
    record_shape="record_shapes:WhatIfComparison",
    # Spec 4.1: non-confusability is a HARD requirement.  These four flags are
    # structural, not conventions: the record shape has no `assessment_id` slot
    # to fill, the emitter for the decision store is not linked into this
    # variant at all, and every rendering carries the marking in its template.
    non_production=True,
    writable_to_decision_store=False,
    issuable_to_client=False,
    satisfies_regulator_request=False,
    manifest="entrypoints/ep08_whatif.toml",
)

# EP02 limit change, EP03 limit programme, EP05 consolidation, EP06 re-price:
# same shape, elided for length.
ENTRY_POINTS = (EP01_NEW_APPLICATION, ..., EP04_CAMPAIGN, ..., EP07_QUOTATION,
                EP08_WHATIF)


# ---------------------------------------------------------------------------
# The nineteen declared divergences.  Every one carries a reason, an owner and
# an approval.  This is the whole list; there is nowhere else to put one.
# ---------------------------------------------------------------------------

DECLARED_APPLICABILITY = (
    Declared(
        target="p15.allocation", off_on=(2, 4),
        because="Entry point 2 is a client request and is not budget-constrained; "
                "entry point 4 produces indicative pre-approvals, not applied limits. "
                "Its inputs ARE available on both, so derivation would run it.",
        declared_by=("T4", "T8"), approved="CC-2027-02", review="2028-06-30",
    ),
    Declared(
        target="p09.class:appetite", off_on=(7,),
        because="A quotation commits the Bank to a price and assesses no client. "
                "Only the twelve regulatory-class entries run.",
        declared_by="T3", approved="COMP-2027-11", review="2029-01-31",
    ),
    Declared(
        target="p12.overlay:rate_add_on", off_on=(8,),
        because="A what-if re-runs the recorded overlay stack, never today's.",
        declared_by="T1", approved="ENG-2027-04", review="never",
    ),
    Declared(
        target="p07.challenger", off_on=(3, 4),
        because="Challenger traffic share is defined over interactive volume. "
                "Running it in batch would multiply the sample by six and "
                "invalidate the design.",
        declared_by="T5", approved="MV-2027-09", review="2028-03-31",
    ),
    Declared(
        target="p04.wave2", off_on=(6,),
        because="Re-price runs on internal state and the bureau view already "
                "held; no fresh external retrieval. Spec 4.1 entry point 6.",
        declared_by="T1", approved="ENG-2027-07", review="2028-12-31",
    ),
    # ... fourteen more, each with a reason, an owner, an approval and a review.
)

DECLARED_APPLICABILITY_COUNT = 19
DECLARED_APPLICABILITY_TRIPWIRE = 35        # README 12


# ---------------------------------------------------------------------------
# What the build does with all of this.
# ---------------------------------------------------------------------------
#
# 1. Derive the phase set and the decision-point set for each entry point, and
#    for each conditional variant.  Eight entry points, eleven phase sets.
# 2. Compare against the ASSERTIONS in each entry point's .toml manifest.  A
#    mismatch names the entry point, the phase, the direction and the most
#    likely cause -- usually a newly added input that quietly made a phase
#    inapplicable somewhere, which is exactly the change that would otherwise be
#    found in production by a client whose second application was declined for a
#    reason the first one did not mention.
# 3. Emit one compiled specialisation per phase set.  Entry point 7 does not
#    carry a branch over `entry_point_code` in every phase; the phases that do
#    not apply are NOT IN ITS KERNEL.  Spec 4.1: "whatever makes entry point 1
#    work must be absent here -- and absent by construction, not by a
#    conditional inside every phase."  That is what buys sub-50 ms.
# 4. Emit the `phase_set_id` registry, archived with the build, so that a 2034
#    replay of a 2027 decision resolves `phase_set_id` 6 to the same list.
#
# Cost, stated: eleven specialisations of an eighteen-phase graph is eleven
# compiles at image build.  See FRAMEWORK-DEMANDS #3.
