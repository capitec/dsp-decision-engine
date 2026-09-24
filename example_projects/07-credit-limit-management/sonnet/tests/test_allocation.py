"""§5.8, the population-level budget allocation -- the reason this project is in the
set. Runs the *same* per-account pipeline `pipeline.build()` produces in batch over a
synthetic population, then `limit_mgmt.allocation.run_allocation()` over the whole
result -- proving the two compose, not asserting the allocation module in isolation
against hand-built rows.
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest
from decider import Engine

from limit_mgmt.allocation import BudgetInstruction, run_allocation
from limit_mgmt.matrix import build_matrix_table
from limit_mgmt.vocab import AllocationOutcome
from synth import make_population

import pipeline as pl_mod

N = int(os.environ.get("LIMIT_MGMT_TEST_POPULATION", "4000"))


@pytest.fixture(scope="module")
def scored_population(sample_record):
    records = make_population(sample_record, N)
    df = pl.DataFrame(records)
    exe = Engine().bind(pl_mod.build(build_matrix_table()))
    return exe.run(df, params={})


def test_batch_run_produces_one_row_per_account(scored_population):
    assert scored_population.height == N
    assert scored_population["account_id"].n_unique() == N


def test_allocation_funds_within_budget(scored_population):
    df, summary = run_allocation(scored_population, BudgetInstruction())
    assert summary["envelope_consumed"] <= BudgetInstruction().applied_limit_budget * BudgetInstruction().over_allocation_factor
    assert summary["funded_count"] > 0
    funded = df.filter(pl.col("allocation_outcome_code") == int(AllocationOutcome.FUNDED))
    total_funded_amount = (funded["final_proposed_limit"] - funded["current_limit"]).sum()
    assert total_funded_amount == pytest.approx(summary["envelope_consumed"], rel=1e-6)


def test_ranking_is_descending_and_ties_break_by_account_id(scored_population):
    """§5.8 rule 7: "Ties are broken deterministically, by account_id ascending." """
    df, _ = run_allocation(scored_population, BudgetInstruction())
    ranked = df.filter(pl.col("allocation_rank") > 0).sort("allocation_rank")
    values = ranked["ranking_value"].to_list()
    for a, b in zip(values, values[1:]):
        assert a >= b - 1e-9
    # within any block of exactly-equal ranking_value, account_id must ascend
    for i in range(len(values) - 1):
        if abs(values[i] - values[i + 1]) < 1e-9:
            assert ranked["account_id"][i] < ranked["account_id"][i + 1]


def test_every_account_has_a_reproducible_outcome(scored_population):
    """§10 item 8 / determinism: same inputs, same funded set, twice."""
    df1, s1 = run_allocation(scored_population, BudgetInstruction())
    df2, s2 = run_allocation(scored_population, BudgetInstruction())
    assert s1["funded_count"] == s2["funded_count"]
    joined = df1.select("account_id", "allocation_outcome_code").join(
        df2.select("account_id", "allocation_outcome_code"), on="account_id", suffix="_2",
    )
    mismatches = joined.filter(pl.col("allocation_outcome_code") != pl.col("allocation_outcome_code_2"))
    assert mismatches.height == 0


def test_non_selected_account_states_its_rank_and_the_line(scored_population):
    """§5.8 "Recorded": a non-funded account's rank and the total ranked must be
    answerable without re-running the cycle."""
    df, summary = run_allocation(scored_population, BudgetInstruction())
    below_line = df.filter(pl.col("allocation_outcome_code") == int(AllocationOutcome.BELOW_LINE))
    if below_line.height:
        row = below_line.row(0, named=True)
        assert row["allocation_rank"] > 0
        assert row["ranking_value"] <= summary["funding_line_ranking_value"] + 1e-9
        assert summary["total_ranked"] >= row["allocation_rank"]


def test_smaller_budget_funds_fewer_accounts(scored_population):
    """Change scenario 2: ALCO halves the budget -> fewer funded, nothing about
    eligibility changes."""
    full = run_allocation(scored_population, BudgetInstruction())[1]
    half = run_allocation(scored_population, BudgetInstruction(applied_limit_budget=1_200_000_000.0))[1]
    assert half["funded_count"] <= full["funded_count"]


def test_tail_skip_does_not_trim_any_offer():
    """§5.8 rule 3: "No increase may be trimmed to fit." Every funded account's amount
    equals its own proposed increase, never a partial amount."""
    df = pl.DataFrame({
        "account_id": [1, 2, 3], "client_id": [1, 2, 3], "product_code": [20, 20, 20],
        "current_limit": [10_000.0, 10_000.0, 10_000.0],
        "final_proposed_limit": [10_600.0, 12_000.0, 11_000.0],
        "behaviour_grade": [2, 2, 2], "mob_band": [4, 4, 4],
        "probability_of_default": [0.02, 0.02, 0.02],
        "is_excluded": [False, False, False],
        "increase_path_code": [1, 1, 1],
        "change_type_code": [1, 1, 1], "decrease_notice_class_code": [4, 4, 4],
    })
    out, summary = run_allocation(df, BudgetInstruction(
        applied_limit_budget=1_500.0 / 1.38, over_allocation_factor=1.38,
        rwa_envelope=1e9, expected_loss_envelope=1e9, fairness_reserve_pct=0.0,
    ))
    funded = out.filter(pl.col("allocation_outcome_code") == int(AllocationOutcome.FUNDED))
    for row in funded.iter_rows(named=True):
        assert row["funded_amount"] in (600.0, 2_000.0, 1_000.0)


def test_client_level_immediate_decrease_suppresses_the_other_account(scored_population):
    """§5.7 "the simultaneous case": an immediate-class decrease on any account of a
    client suppresses every increase for that client this cycle."""
    df = scored_population
    with_decrease = df.filter(pl.col("change_type_code") == 3)  # ChangeType.DECREASE
    if with_decrease.height == 0:
        pytest.skip("no decrease fired in this synthetic draw")
    client = with_decrease["client_id"][0]
    other_accounts = df.filter((pl.col("client_id") == client) & (pl.col("change_type_code") != 3))
    if other_accounts.height == 0:
        pytest.skip("this client only holds the one account")
    out, _ = run_allocation(df, BudgetInstruction())
    other = out.filter((pl.col("client_id") == client) & (pl.col("change_type_code") != 3))
    assert (other["allocation_outcome_code"] == int(AllocationOutcome.OVERLAY_SUPPRESSED)).all()
