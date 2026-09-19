"""The twelve acceptance criteria, as tests.

Every one of these is a property of the STRUCTURE rather than of an example,
which is what makes them worth writing. A test that asserts one applicant gets
R4 312.18 protects one applicant.
"""

import pytest
from decider2.testing import (
    assert_modes_agree,
    corpus_from_declared_edges,
    kernel_fingerprint,
)

from modules.verdict import Sufficiency, Verdict
from pipelines.affordability import Assessment, evidence
from pipelines.consumers import replay
from policy.modes import ALL_MODES
from policy.overlays import ADMISSIBLE


# --- acceptance 2: the four modes share the arithmetic ---------------------

def test_all_modes_share_one_kernel():
    """A mode that needed different machine code is a different calculation.

    This is the structural enforcement that spec question 2 asks for. It is not
    a review convention and it is not a docstring: four profiles, one kernel
    fingerprint, and a mode that acquires its own arm fails here rather than in
    2028 when three of them still have the 2026 tax table.
    """
    prints = {kernel_fingerprint(Assessment.under(m)) for m in ALL_MODES}
    assert len(prints) == 1, f"modes diverged into {len(prints)} kernels"


def test_a_profile_cannot_reach_a_step():
    from decider2 import profile
    with pytest.raises(ValueError, match="is not a param, a switch or an emit"):
        profile("bad", code=99, params={"deductions.income_tax_cents": 0})


# --- acceptance 3: monotonicity, tested and not asserted -------------------

def test_monotone_in_proposed_instalment_across_declared_edges():
    """Project 03's search has no valid stopping condition without this.

    The corpus is derived from the DECLARED edges of the five named artefacts --
    72 statutory band floors, 96 buffer grid cells, 6 residual floor cells, the
    tolerance band, and the internal norm bands -- crossed with the param
    bounds. Random draws never find the residual-floor/buffer interaction,
    because it is a measure-zero set in the input space and an inevitable one in
    the applicant population.
    """
    corpus = corpus_from_declared_edges(Assessment)
    assert len(corpus) > 5_000
    Assessment.check_declared_properties(corpus)


# --- acceptance 4 and 7 in one: the cut, and the annotation ----------------

def test_hold_and_resume_equals_apply(sample):
    """The 400th call equals the first, and equals a cold call."""
    held = Assessment.hold(sample.frame, upto=evidence)
    for scenario in sample.scenarios:
        assert held.resume(**scenario) == Assessment.apply(sample.frame.with_(**scenario))


def test_nothing_accumulates_between_resumes(sample):
    held = Assessment.hold(sample.frame, upto=evidence)
    first = held.resume(**sample.scenarios[0])
    for scenario in sample.scenarios[1:]:
        held.resume(**scenario)
    assert held.resume(**sample.scenarios[0]) == first


def test_cut_soundness_is_a_build_time_check():
    """The check that makes `hold` safe runs without data.

    If a step upstream of the cut read `accounts`, a held prefix would be stale
    for all 400 scenarios and every one of them would be plausibly wrong. Static
    lineage answers it with no execution, which is the requirement doc 02 5
    states and this is where it earns its keep.
    """
    report = Assessment.check_cut(evidence)
    assert report.upstream_reads_of_resumed == []


def test_scalar_and_annotation_cannot_disagree(sample):
    """Acceptance 10. Not a comparison -- a materialisation-neutrality check.

    The scalar is DEFINED as a reduction over the annotation, so this asserts
    that asking for the annotation does not change the scalars. If it ever did,
    the annotation would be a second computation.
    """
    lean = Assessment.apply(sample.frame)
    rich = Assessment.apply(sample.frame, materialise=["accounts_annotated"])
    assert lean.scalars() == rich.scalars()
    for row in rich.rows():
        assert row.existing_obligations_cents == (
            sum(a.obligation_cents for a in row.accounts_annotated)
            + row.enquiry_velocity_uplift_cents
        )


# --- acceptance 7: indeterminate is never returned as fail -----------------

@pytest.mark.parametrize("code", [c for c in Sufficiency if c != Sufficiency.SUFFICIENT])
def test_each_sufficiency_code_produces_indeterminate(code, fixture_for):
    """A test set built from EACH code, as the spec requires -- so a new code

    added without a corresponding evidence path fails collection rather than
    quietly never firing.
    """
    result = Assessment.apply(fixture_for(code))
    assert result.affordability_verdict_code == Verdict.INDETERMINATE
    assert result.evidence_sufficiency_code == code


def test_verdict_has_no_boolean_context():
    with pytest.raises(TypeError, match="Verdict has no boolean context"):
        bool(Verdict.INDETERMINATE)


def test_verdict_inequality_against_one_member_is_refused():
    with pytest.raises(TypeError, match="folds MARGINAL, FAIL and INDETERMINATE"):
        _ = Verdict.PASS != Verdict.FAIL


def test_every_narrowing_states_its_indeterminate_answer():
    """Five consumers, five written answers, readable in one place."""
    from modules import verdict as v
    narrowings = [n for n in vars(v).values() if getattr(n, "is_narrowing", False)]
    assert len(narrowings) == 5
    assert all(n.treat_indeterminate_as is not None for n in narrowings)


# --- acceptance 8: an overlay that would increase capacity is rejected -----

def test_loosening_overlay_is_rejected_at_definition_time():
    from decider2 import OverlayRegister
    with pytest.raises(ValueError, match="would increase capacity"):
        OverlayRegister.validate({
            "overlays": [{
                "id": "TEST-001", "order": 10,
                "target": "max_affordable_instalment_cents",
                "op": "scale_up", "magnitude": 1.08,
            }]
        })


def test_negative_magnitude_cannot_bypass_the_direction_rule():
    from decider2 import OverlayRegister
    with pytest.raises(ValueError, match="magnitude -3.0 is less than the minimum 0.0"):
        OverlayRegister.validate({
            "overlays": [{
                "id": "TEST-002", "order": 10,
                "target": "retained_pct", "op": "add_pp", "magnitude": -3.0,
            }]
        })


def test_admissible_table_matches_the_composition():
    """`ADMISSIBLE` in policy/overlays.py is a materialised artefact a reviewer

    reads without a tool. This asserts it still matches what composing each
    operator's direction with each target's `tightens_when` produces.
    """
    from decider2 import derive_admissible
    assert ADMISSIBLE == derive_admissible(Assessment)


def test_overlay_outside_its_scope_is_an_error_not_a_no_op(sample):
    """Spec 5.6.2(2). A no-op is indistinguishable from a scope that was

    written wrongly, and the silence is what lets a mis-scoped overlay run for
    three years.
    """
    with pytest.raises(ValueError, match="evaluated outside its declared scope"):
        Assessment.with_overlays(scope_mismatched_register).apply(sample.frame)


# --- acceptance 5 and 9: replay, and the disabled stack --------------------

def test_replay_reproduces_to_the_cent(historical_snapshot_2026):
    result = replay(historical_snapshot_2026)
    assert result == historical_snapshot_2026.recorded_outputs


def test_replay_never_reaches_a_live_system(historical_snapshot_2026):
    with no_network(), no_filesystem_outside("tables/"):
        replay(historical_snapshot_2026)


def test_pinned_and_dated_resolvers_agree(historical_snapshot_2026):
    """Two independent paths to the same table versions.

    A disagreement means a version file was edited in place, which is otherwise
    undetectable and is the one failure effective dating cannot defend itself
    against.
    """
    pinned = replay(historical_snapshot_2026)
    dated = replay(historical_snapshot_2026, resolver="dated")
    assert pinned.resolved_versions == dated.resolved_versions


def test_disabled_overlay_stack_uses_the_same_implementation(sample):
    from decider2 import EMPTY_REGISTER
    off = Assessment.with_overlays(EMPTY_REGISTER).apply(sample.frame)
    on = Assessment.apply(sample.frame)
    assert kernel_fingerprint(off) == kernel_fingerprint(on)
    assert (off.max_affordable_instalment_cents
            == on.max_affordable_instalment_unadjusted_cents)


# --- acceptance 11 and 12: the ladder, and table versions ------------------

def test_every_rung_references_only_values_that_exist():
    """A rung template naming a value no step produces is a build error.

    This is what stops the narration drifting from the calculation, which
    spec 9.1 says is itself the finding.
    """
    assert Assessment.check_ladder().orphan_references == []
    assert Assessment.check_ladder().empty_sections == []


def test_every_table_version_resolved_is_recorded(sample):
    result = Assessment.apply(sample.frame)
    assert set(result.resolved_versions) == set(Assessment.dated_artefacts())


# --- the equivalence ladder, four rungs ------------------------------------

def test_modes_agree(sample):
    assert_modes_agree(Assessment, corpus_from_declared_edges(Assessment))
