"""§5.10 -- simulation. **Not a second implementation** (§10 item 3, README §2 hard
part 2): `run_simulation()` calls exactly the two functions production calls --
`pipeline.build()` bound and run in batch, then `limit_mgmt.allocation.run_allocation()`
-- over a candidate artefact set instead of the live one.

A "candidate artefact set" (§13 Q4) is a `matrix_config` (a `DecisionTableConfig`
built from a candidate spreadsheet, in place of `configs/<version>/matrix.json`), a
`params` override dict (a candidate dial, cap, buffer, threshold -- anything wired as
a `param()`), and/or a `BudgetInstruction` (a candidate ALCO instruction). None of
these are new code paths: `pipeline.build(matrix)` already takes the matrix as an
argument, `Engine.run(df, params=...)` already takes a params override, and
`BudgetInstruction` is already `run_allocation`'s own parameter.

**Self-check (§5.10 item 3, required before every simulation session):** run this
module over the *current production* matrix/params/budget and the last production
snapshot; the result must reproduce the last production cycle account for account,
to the rand, including ranks and the funded set. Because production and simulation
share one call path, the self-check is `run_simulation(...) == the production run
that already happened` -- not a second calculation to keep in sync.

**With the overlay stack disabled** (§5.10 item 2, Model Risk's own view): pass
`params={"...": {"adjustment_stack_enabled": False}}` for every overlay-bearing node
-- the same `param()` flip `tests/test_pipeline.py::test_stack_disabled_is_the_same_apply_stack_call`
proves at the mechanism level, run here over the whole book instead of one account.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from decider import Engine
from decider.steps.tables import DecisionTableConfig

from limit_mgmt.allocation import BudgetInstruction, run_allocation
from limit_mgmt.matrix import build_matrix_table
from pipeline import build


@dataclass(frozen=True)
class SimulationResult:
    scored: pl.DataFrame               # per-account pipeline output, the candidate set applied
    funded: pl.DataFrame               # `scored` plus allocation_rank/outcome/funded_amount
    summary: dict                      # cycle_summary (funded count, envelope consumed, funding line, ...)


def run_simulation(
    population_df: pl.DataFrame,
    matrix_config: DecisionTableConfig | None = None,
    params: dict | None = None,
    budget: BudgetInstruction = BudgetInstruction(),
) -> SimulationResult:
    """The whole cycle, over a candidate artefact set. Omit every argument to reproduce
    the current production run (the self-check)."""
    matrix_config = matrix_config or build_matrix_table()
    exe = Engine().bind(build(matrix_config))
    scored = exe.run(population_df, params=params or {})
    funded, summary = run_allocation(scored, budget)
    return SimulationResult(scored=scored, funded=funded, summary=summary)


def swap_set(baseline: SimulationResult, candidate: SimulationResult) -> pl.DataFrame:
    """§5.10 "Required outputs": gainers, losers and unaffected, a per-account
    classification (not a before/after total) between two simulation runs -- typically
    the current matrix (`baseline`) against a candidate (`candidate`)."""
    joined = (
        baseline.funded.select("account_id", "final_proposed_limit", "allocation_outcome_code")
        .rename({"final_proposed_limit": "baseline_limit", "allocation_outcome_code": "baseline_outcome"})
        .join(
            candidate.funded.select("account_id", "final_proposed_limit", "allocation_outcome_code")
            .rename({"final_proposed_limit": "candidate_limit", "allocation_outcome_code": "candidate_outcome"}),
            on="account_id", how="full", coalesce=True,
        )
    )
    return joined.with_columns(
        pl.when(pl.col("candidate_limit") > pl.col("baseline_limit")).then(pl.lit("gainer"))
        .when(pl.col("candidate_limit") < pl.col("baseline_limit")).then(pl.lit("loser"))
        .otherwise(pl.lit("unaffected")).alias("swap_class"),
        (pl.col("candidate_limit") - pl.col("baseline_limit")).alias("change"),
    )


def self_check(population_df: pl.DataFrame, last_production: SimulationResult) -> bool:
    """§5.10 item 3: simulation over the current production artefact set and the last
    production snapshot must reproduce the last production cycle account for account, to
    the rand, including ranks and the funded set."""
    replay = run_simulation(population_df)
    diff = swap_set(last_production, replay).filter(pl.col("swap_class") != "unaffected")
    rank_diff = (
        last_production.funded.select("account_id", "allocation_rank")
        .join(replay.funded.select("account_id", "allocation_rank"), on="account_id", suffix="_replay")
        .filter(pl.col("allocation_rank") != pl.col("allocation_rank_replay"))
    )
    return diff.height == 0 and rank_diff.height == 0
