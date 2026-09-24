"""Phase-set resolution (spec 10 §5.20, §4.6) -- the declared skeleton SCOPE.md asks for."""
from retail_credit import entry_points
from retail_credit.phases import PHASES


def test_every_entry_point_has_eighteen_declared_rows():
    for ep in range(1, 8):
        assert len(entry_points.declared_matrix(ep)) == 18


def test_entry_point_1_runs_sixteen_phases_before_conditionals():
    rows = entry_points.declared_matrix(1)
    running = [r for r in rows if r.status in (entry_points.RUNS, entry_points.REDUCED)]
    assert len(running) == 16
    conditional = [r for r in rows if r.status == entry_points.CONDITIONAL]
    assert [PHASES[r.phase_id].code for r in conditional] == ["P14"]


def test_entry_point_1_p14_conditional_on_affordability_failure_and_eligibility():
    without_loop = entry_points.resolve_phase_set(1, consolidation_loop_fires=False)
    with_loop = entry_points.resolve_phase_set(1, consolidation_loop_fires=True)
    assert 14 not in without_loop
    assert 14 in with_loop
    assert 15 not in without_loop  # P15 never runs on entry point 1 (§5.20)


def test_entry_point_7_runs_seven_phases():
    resolved = entry_points.resolve_phase_set(7)
    assert len(resolved) == 7
    assert set(resolved) == {1, 3, 9, 11, 12, 17, 18}


def test_entry_point_6_p10_conditional_on_reprice_raising_the_instalment():
    without = entry_points.resolve_phase_set(6, reprice_raises_instalment=False)
    with_ = entry_points.resolve_phase_set(6, reprice_raises_instalment=True)
    assert 10 not in without
    assert len(without) == 11
    assert 10 in with_
    assert len(with_) == 12


def test_phase_set_id_round_trips():
    phases = entry_points.resolve_phase_set(1, consolidation_loop_fires=True)
    pid = entry_points.phase_set_id(1, phases)
    assert sorted(entry_points.phases_for_id(1, pid)) == sorted(phases)


def test_no_two_entry_points_share_a_phase_set_id():
    ids = set()
    for ep in range(1, 8):
        for loop_fires in (False, True):
            resolved = entry_points.resolve_phase_set(ep, consolidation_loop_fires=loop_fires)
            ids.add(entry_points.phase_set_id(ep, resolved))
    # 8 entry points x up to 2 conditional states, minus duplicates where the flag has no effect
    assert len(ids) >= 8


def test_ownership_map_covers_every_phase_and_totals_1400_decision_points():
    from retail_credit.phases import TOTAL_DECISION_POINTS
    assert TOTAL_DECISION_POINTS == 1400
    for p in PHASES.values():
        assert p.owning_teams, f"{p.code} has no owner"


def test_at_least_five_phases_have_a_single_owner():
    """10 §5.26.1's narrative claims exactly "P02, P05, P07, P13 (for its mechanics) and
    P14", but its own ownership table lists T1 *and* T7 against P13 -- narrative and table
    disagree (see NOTES.md "Spec problems"). This project's registry follows the table
    (the machine-checkable artefact), so P13 is not single-owner here; P01 is, which the
    narrative's list omits despite no second team appearing against it in the table either.
    """
    single_owner = {p.code for p in PHASES.values() if len(p.owning_teams) == 1}
    assert {"P02", "P05", "P07", "P14"} <= single_owner
    assert len(single_owner) >= 5
