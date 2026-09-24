"""The full-width skeleton: all 23 phases, all 9 entry points (spec 11 §5.1-§5.2),
declared even where this slice does not flesh them out -- see `phases.py`."""
from __future__ import annotations

from business_credit_e2e import phases, vocab


def test_all_23_phases_are_declared():
    assert set(phases.PHASES) == {f"O{i}" for i in range(1, 18)} | {f"L{i}" for i in range(1, 7)}


def test_decision_point_total_matches_spec_11_5_2():
    assert phases.TOTAL_DECISION_POINTS == 1_900


def test_origination_subtotal_matches_spec():
    origination_total = sum(p.decision_points for pid, p in phases.PHASES.items() if pid.startswith("O"))
    lifecycle_total = sum(p.decision_points for pid, p in phases.PHASES.items() if pid.startswith("L"))
    assert origination_total == 1_280
    assert lifecycle_total == 620


def test_fleshed_out_phases_are_exactly_what_this_project_builds():
    fleshed = {p.phase_id for p in phases.phases_owned_here()}
    # EP-1's origination phases (O1-O17) plus L1/L2 (review.py, covenant.py) are real;
    # O8 (behavioural), O12 (collateral allocation) and L3-L6 are declared stubs only.
    assert fleshed == {f"O{i}" for i in range(1, 18) if i not in (8, 12)} | {"L1", "L2"}
    stubbed = {p.phase_id for p in phases.declared_but_stubbed()}
    assert stubbed == {"O8", "O12", "L3", "L4", "L5", "L6"}


def test_every_entry_point_has_a_declared_phase_row():
    for ep_code in vocab.ENTRY_POINT_NAMES:
        row = phases.phases_for_entry_point(ep_code)
        assert row, f"entry point {ep_code} has no phases"
        assert set(row.values()) <= {phases.RUNS, phases.PARTIAL, phases.SKIPPED}


def test_ep1_runs_the_full_origination_set():
    row = phases.phases_for_entry_point(vocab.EP1_NEW_TO_BANK)
    assert set(row) == {f"O{i}" for i in range(1, 18)}
    assert all(status == phases.RUNS for status in row.values())


def test_ep3_annual_review_runs_l1():
    row = phases.phases_for_entry_point(vocab.EP3_ANNUAL_REVIEW)
    assert row["L1"] == phases.RUNS


def test_unknown_entry_point_raises_rather_than_defaulting():
    import pytest
    with pytest.raises(ValueError):
        phases.phases_for_entry_point(99)
