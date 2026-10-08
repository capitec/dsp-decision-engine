"""The monthly cycle, in production and in simulation: one code path, so a simulation cannot drift from production.

A candidate artefact set is a matrix, a params document (a dial, a cap, a buffer, the overlay stack switched off) or a
budget instruction. Run the cycle over the same snapshot with and without it and compare with `swap_set`.
"""
from __future__ import annotations

from typing import NamedTuple

import polars as pl
from decider import DecisionTableConfig

from limit_mgmt import overlays
from limit_mgmt.allocation import BudgetInstruction, allocate
from limit_mgmt.matrix import build_matrix_table
from limit_mgmt.pipeline import build


class Cycle(NamedTuple):
    scored: pl.DataFrame     # the per-account pipeline's output
    book: pl.DataFrame       # `scored` plus the allocation's decision on every account
    summary: dict            # `allocate`'s cycle figures


def run_cycle(
    snapshot: pl.DataFrame, matrix: DecisionTableConfig | None = None, params: dict | None = None,
    budget: BudgetInstruction = BudgetInstruction(),
) -> Cycle:
    """Score every account in `snapshot` and allocate the budget. With no candidates it is the production cycle.

    Raises `ValueError` while any overlay is past its review date: an unreviewed dial must not run unnoticed.

    Example::

        production = run_cycle(snapshot)
        tighter = run_cycle(snapshot, params={"limit_mgmt": {"matrix": {"apply_cycle_cap": {"cycle_cap_enabled": True}}}})
        swap_set(production, tighter)["swap_class"].value_counts()
    """
    overdue = overlays.due_for_review(snapshot["decision_date"].max())
    if overdue:
        raise ValueError(f"overlays past their review date: {overdue}")
    scored = build(matrix or build_matrix_table()).compile("fused").run(snapshot, params=params)
    book, summary = allocate(scored, budget)
    return Cycle(scored, book, summary)


def swap_set(baseline: Cycle, candidate: Cycle) -> pl.DataFrame:
    """Per account, whether the candidate cycle's granted limit is a `gainer`, `loser` or `unaffected` against the baseline's.

    An account missing from either cycle is compared as if its limit were null (neither class), so run both over one snapshot.
    """
    def limits(cycle: Cycle, name: str) -> pl.DataFrame:
        return cycle.book.select("account_id", pl.col("granted_limit").alias(f"{name}_limit"),
                                 pl.col("allocation_outcome_code").alias(f"{name}_outcome"))
    swap = limits(baseline, "baseline").join(limits(candidate, "candidate"), on="account_id", how="full", coalesce=True)
    change = pl.col("candidate_limit") - pl.col("baseline_limit")
    return swap.with_columns(
        change.alias("change"),
        pl.when(change > 0).then(pl.lit("gainer")).when(change < 0).then(pl.lit("loser"))
        .otherwise(pl.lit("unaffected")).alias("swap_class"))
