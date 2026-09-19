"""The acceptance criteria, as tests. §10, criteria 1-22.

Four of these are the ones that would actually catch a mispriced contract, and
all four are shaped by framework features that doc 03 does not have.  They are
written here to show what the surface has to make possible.
"""

from __future__ import annotations

import decider2
from decider2.testing import assert_modes_agree, assert_paths_agree, corpus

from modules.pricing.probe import PriceCandidate
from modules.solve.search import MaximumAffordableAmount
from modules.validation.independent import FinalValidation
from pipelines.flex_loan_granting import granting


# --- Criterion 6 -----------------------------------------------------------
# "The band-edge non-monotonicity case in §5.8 is a NAMED REGRESSION TEST, with
#  all 41 declared inversions on the current card covered, and it FAILS LOUDLY
#  if a search returns R48 500."

def test_r50000_inversion_returns_50000_not_48500():
    """Grade 9, 60 months, ceiling R1 560.00, domain [R2 000, R95 000].

    Feasible set is {R2 000 .. R48 500} union {R50 000}.  The answer is
    R50 000.  A bisection over the whole domain returns R48 500 and under-lends
    by R1 500; a descending scan from the top returns R50 000 but takes 450
    probes and only works because it started above the answer.

    The declared strategy does it in NINE probes and the nine are recorded:

        #  phase             candidate    rate     instalment   feasible
        -  ----------------  -----------  -------  -----------  --------
        1  bracket           (inverse)    18.25%   --           a_max = R59 500
        2  segment-left      R55 000      18.25%   R1 702.01    no  -> segment dead
        3  segment-left      R50 000      18.25%   R1 558.73    YES -> answer is here
        4  bisect            R52 500      18.25%   R1 630.34    no
        5  bisect            R51 200      18.25%   R1 593.11    no
        6  bisect            R50 600      18.25%   R1 575.93    no
        7  bisect            R50 300      18.25%   R1 567.33    no
        8  bisect            R50 100      18.25%   R1 561.59    no
        9  bisect            R50 000      18.25%   R1 558.73    YES

    Probe 2 is the one that matters: it kills the entire R55 000-R59 999 band
    with ONE evaluation, because within a band the instalment is monotone and
    the left edge is its minimum.
    """
    pass


def test_all_41_declared_inversions():
    """Parameterised over `FlexRateCard.version(d).inversions` -- the declared list,
    read from the manifest, not hard-coded.  58 after the re-band of change
    scenario 1, and this test does not change."""
    pass


# --- Criterion 3 -----------------------------------------------------------
# "For a 10 000-application sample, the solve's answer equals the answer from
#  exhaustively evaluating every R100 candidate at every permitted term.  Zero
#  disagreements.  Re-run on every rate card version."

def test_exhaustive_agreement_on_current_card():
    """`exhaustive()` is a MODE of the same Search object -- same domain, same
    probe, same feasibility test, different strategy.  So the only thing that
    can differ is the strategy.  If it were a second implementation, a shared
    misunderstanding of the domain or the feasibility test would pass this test
    while both were wrong."""
    pass  # assert MaximumAffordableAmount.exhaustive().agrees_with(MaximumAffordableAmount, corpus(10_000))


def test_exhaustive_agreement_runs_on_every_card_version():
    """Wired to card staging, not to a schedule. A card that has not passed this
    cannot be activated."""
    pass


# --- Criterion 2 -----------------------------------------------------------
# "Every offer passes an INDEPENDENT FROM-SCRATCH RE-CHECK over a 250 000-
#  application regression set. ZERO failures. Not waivable."

def test_independence_is_structural_not_procedural():
    """This is the test that makes criterion 2 cheap.

    Independence is a negative lineage assertion evaluated at BUILD time:

        lineage(FinalValidation.outputs)
            is disjoint from
        steps(MaximumAffordableAmount) | steps(PriceCandidate) | steps(CapWaterfall)
            except for the four primitives in `shares_primitives`

    So the 250 000-record run is a regression test over DATA, and the property
    it depends on is checked over STRUCTURE, once, at build.  Without the
    structural half, 250 000 passes prove only that 250 000 records agreed.
    """
    assert FinalValidation.independence_holds()  # raises naming the shared step


def test_no_stage_downstream_of_solve_writes_offered_amount():
    """The silent failure §5.8(7) describes -- an offer affordable when evaluated
    but not at the amount finally written, because a cap moved it into another
    band afterwards -- is UNREACHABLE if nothing downstream of the solve can
    move the amount.  Caps narrow the DOMAIN, never the answer.  One lineage
    query, checked at build, removes a whole class of production incident."""
    assert granting.writers_of("offered_amount", after=MaximumAffordableAmount) == []


# --- Criterion 4 -----------------------------------------------------------

def test_probe_budget_is_proved_not_sampled():
    """'No application performs more than 24 pricing evaluations for any single
    term, EVER, on any input, including adversarial ones.'

    Sampling cannot establish 'ever'.  The worst case is a closed form over the
    card's axis -- 1 bracket + S segments + ceil(log2(W)) -- evaluated for every
    (term, grade) at card validation.  The current card's worst case is 21.
    """
    report = granting.tables["flex_rate_card"].validation["search_budget_is_satisfiable"]
    assert report["worst_case_probes"] <= report["declared_budget"]


# --- Criterion 11 ----------------------------------------------------------

def test_realtime_and_batch_are_the_same_answers():
    """'Real-time and batch produce IDENTICAL outputs on a 100 000-record monthly
    reconciliation sample. Zero differences.'

    `assert_paths_agree` is the fourth rung of the equivalence ladder and doc 02
    §3.1 does not have it: the existing three rungs are interpreted / stepped /
    fused, all within one entry point.  `score()` and `apply()` differ in
    fusion grouping (doc 02 §1.2 gives them opposite defaults) and in the
    boundary they cross, so their agreement is a separate claim and needs a
    separate assertion.  FRAMEWORK-DEMANDS #21.
    """
    pass  # assert_paths_agree(granting, corpus(100_000))


def test_all_four_rungs():
    """interpreted == stepped == fused == score, bit-exact.

    Bit-exact is the right criterion here and is achievable only because
    FLEX_NUMERICS bans fastmath pipeline-wide.  Doc 02 measured fastmath at
    46-73% of rows differing by up to 17 ULP for a 1.09x gain.
    """
    pass  # assert_modes_agree(granting, corpus(50_000), exact=True)


# --- Criterion 5, 15, 17 ---------------------------------------------------

def test_cap_chain_is_retrievable_from_the_record_alone():
    """'For any of the three ceilings, the final value, the rule that bound it, the
    full chain, and the per-rule verdict of all 52 rules are retrievable from
    the STORED RECORD alone.'"""
    pass


def test_adding_a_rule_at_any_sequence_position_is_not_a_release():
    """Criterion 15. Add a rule at seq 27 to the interior, assert:
      * the structure fingerprint is UNCHANGED (generic kernel, no codegen)
      * no compile occurs
      * the chain has one more row for records where it binds
      * the verdict vector has one more entry for every record
    """
    pass


# --- Criteria 18-22, the overlay set ---------------------------------------

def test_stack_disabled_runs_through_the_same_implementation():
    """Criterion 20. `enabled: false` on the set, not a second pipeline."""
    pass  # decider2.impact(granting.with_adjustments(ON), granting.with_adjustments(OFF), sample)


def test_a_loosening_overlay_is_rejected_at_set_definition():
    """Criterion 19, and §13.19's 'what makes it impossible to bypass by defining a
    negative magnitude'. Direction is on the POINT, in code; magnitude is in the
    SET, in config; validation composes them BEFORE anything runs."""
    pass


def test_rate_add_on_cannot_exceed_the_statutory_ceiling():
    """Criterion 22. Tested AT THE ADD-ON, not only at card validation --
    `post_assert` on the RATE_ADD_ON point."""
    pass


def test_every_live_overlay_has_an_unexpired_review_date():
    """Criterion 21, and change scenario 18 -- the three-year-old tightening whose
    author has left."""
    pass


# --- The corpus ------------------------------------------------------------
#
# Doc 03 §1.2: "the corpus must include BOUNDARY VALUES.  The overflow was found
# by binary search, not by sampling."  For this project the boundary corpus is
# generated, not sampled:
#
#   * every one of the 96 amount-band edges, at edge, edge-R100 and edge+R100
#   * the fee kink at R12 700, same three points
#   * every declared inversion, from both sides
#   * an affordability ceiling set to the exact instalment of each band edge,
#     and to that instalment plus and minus one cent  <- the "no, by 4 cents"
#     case, which random draws would never produce
#   * a domain of one candidate, an empty domain, and a domain where nothing is
#     affordable (§5.8.2)
#
# ~2 400 constructed applications, regenerated on every card version.  That is
# the concrete form of "the corpus" that doc 05 §9 depends on and that no
# framework document has yet defined.
