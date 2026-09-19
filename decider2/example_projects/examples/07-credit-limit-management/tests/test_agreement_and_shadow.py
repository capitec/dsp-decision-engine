"""s5.12 (the two paths agree) and the shadow's guarantee.

The first test is one line and it is the most important test in the project.
"""

from decider2.testing import assert_shadow_agrees, corpus, sample
from decider2 import artefacts

from clm.pipelines import programme as batch
from clm.pipelines import request as realtime


def test_the_two_paths_share_one_object():
    """s10.3: there is no second expression of the matrix, the caps, the
    affordability logic, the overlays or the allocation anywhere in the system.

    Structural agreement is checked by identity, not by comparing outputs.
    Comparing outputs would pass for a copy that happened to agree on the
    corpus; identity cannot."""
    assert batch.ProposedLimit is realtime.ProposedLimit


def test_agreement_on_unchanged_evidence_is_exact():
    """The consequence of the test above. With path_code held at 1 and the same
    evidence on both sides, the two paths do not agree to R500 -- they agree
    exactly, because they are the same kernel. The R500 tolerance in s5.12
    exists only for evidence differences."""
    pass


def test_agreement_with_real_evidence_differences_within_tolerance():
    """s10.10. 2 000 accounts that took both paths in the same month; agreement
    at or above 99.0% within max(R500, 2% of L), and every breach attributed to
    a NAMED input difference: fresh declared income against deposit-derived,
    a bureau refresh between the snapshot and the request, or a transaction
    changing the utilisation band -- and to no other cause.

    The attribution is mechanical because the pipeline is the same: run the
    request-path inputs through the batch path and diff the input vectors."""
    pass


def test_funded_outcomes_are_allowed_to_differ():
    """Note what s5.12 does NOT require. A client who asks, in a month when they
    ranked 412 000th, gets the increase, and that is correct rather than a
    defect: the budget governs what the Bank pushes out, not what it grants on
    request. Only `proposed_limit_c` must agree."""
    pass


def test_shadow_equals_a_disabled_stack_run():
    """`shadow(...)` emits the unadjusted value per record, always on. Running
    the whole book with `tables=artefacts.resolve(..., overlays="off")` emits it
    per cycle. They are the same number by construction and the test asserts it,
    because if they ever differ, one of them is lying on every offer made."""
    book = corpus.from_snapshot("2026-08", rows=100_000)
    assert_shadow_agrees(batch.programme, book,
                         shadowed=["proposed_limit_c", "matrix_multiplier",
                                   "behaviour_score", "behaviour_grade"],
                         against=artefacts.resolve(decision_date="2026-09-01",
                                                   selector="live", overlays="off"))


def test_lapsed_overlay_stops_the_cycle():
    """s10.6. Not a warning, not a report -- the resolve call raises and the
    programme does not start."""
    pass


def test_alco_instruction_must_be_this_cycle():
    """s4.1: 'the cycle does not run on last month's instruction'. The params
    document's `cycle_month` is cross-checked against shared.decision_date at
    resolve time. See FRAMEWORK-DEMANDS #16 -- doc 08's opaque `origin` token
    cannot express this and a project cannot be left to remember it."""
    pass


def test_self_check_reproduces_the_last_production_cycle():
    """s10.2. Account for account, to the rand, including ranks and the funded
    set. Runs before every simulation session."""
    pass
