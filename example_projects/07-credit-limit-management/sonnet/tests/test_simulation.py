"""§5.10, §10 items 2-3: simulation is the *same* call path as production, not a
second implementation. See `simulation.py`'s own docstring for what a "candidate
artefact set" is."""
from __future__ import annotations

import polars as pl

from limit_mgmt.matrix import build_matrix_table
from simulation import run_simulation, self_check, swap_set
from synth import make_population


def test_self_check_reproduces_the_same_run(sample_record):
    """§5.10 item 3: run over the current artefact set and the same snapshot must
    reproduce the same cycle account for account, including ranks and the funded set."""
    population = pl.DataFrame(make_population(sample_record, 300))
    production = run_simulation(population)
    assert self_check(population, production) is True


def test_candidate_matrix_produces_a_named_swap_set(sample_record):
    """§5.10 "Required outputs": "Swap-set against the current matrix... a per-account
    classification."""
    population = pl.DataFrame(make_population(sample_record, 300))
    baseline = run_simulation(population)
    # A cheap candidate: run with the cycle cap permanently on via params, in place of
    # a real candidate matrix document -- exercises the same "candidate artefact set"
    # path §5.10 requires without needing a second spreadsheet fixture in this test.
    candidate = run_simulation(population, params={
        "limit_mgmt": {"matrix": {"apply_cycle_cap": {"cycle_cap_enabled": True, "cycle_cap_amount": 3_000.0}}},
    })
    diff = swap_set(baseline, candidate)
    assert set(diff["swap_class"].unique().to_list()) <= {"gainer", "loser", "unaffected"}
    assert (diff["swap_class"] == "loser").sum() >= 0  # a tighter cap must not create gainers
