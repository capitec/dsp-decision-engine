"""The solve: the largest amount on the R250 grid whose instalment fits under the instalment ceiling.

The rate is a step function of the amount, so a smaller amount can cost more per month than a larger one just past a
band edge (the card drops 155 basis points at R60 000). Affordability is therefore not monotone in the amount, and the
solve tries every candidate rather than halving an interval.
"""
from __future__ import annotations

import math

from decider import flow, optimise, param, step

from retail_credit import pricing
from retail_credit.vocab import BIND_AFF, BIND_CAP, BIND_MAX, BIND_MIN, BIND_REQ

MAX_CANDIDATES = 2_000  # (R500 000 - R2 000) / R250 + 1 = 1 993


def search_domain(
    requested_amount: float, amount_cap: float,
    product_min: float = param(2_000.0, ge=0.0), product_max: float = param(500_000.0, ge=0.0),
    grid: float = param(250.0, gt=0.0),
) -> tuple[float, float, float]:
    """(search_floor, search_grid, search_ceiling): the grid runs from the product minimum to the largest grid point
    under every ceiling."""
    return product_min, grid, math.floor(min(requested_amount, amount_cap, product_max) / grid) * grid


@step(output="candidate_count")
def candidate_count(search_floor: float, search_grid: float, search_ceiling: float) -> int:
    return int((search_ceiling - search_floor) // search_grid) + 1 if search_ceiling >= search_floor else 0


def offered_amount(index: int, search_floor: float, search_grid: float) -> float:
    return search_floor + (index - 1) * search_grid


@step(output="score")
def affordable_amount(offered_amount: float, instalment: float, instalment_ceiling: float) -> float:
    """The amount itself when its instalment fits, so the largest affordable candidate wins."""
    return offered_amount if instalment <= instalment_ceiling else -math.inf


def unaffordable(score: float) -> bool:
    return score == -math.inf


def solved_amount(best_index: int, search_floor: float, search_grid: float) -> float:
    return 0.0 if best_index < 0 else offered_amount(best_index, search_floor, search_grid)


def solve_binding_constraint(
    solved_amount: float, search_ceiling: float, requested_amount: float, amount_cap: float, search_floor: float,
    search_grid: float,
) -> str:
    """Which limit stopped the amount: the instalment ceiling, the request, the amount cap, the product maximum or
    the product minimum (nothing in range)."""
    if search_ceiling < search_floor:
        return BIND_MIN
    if solved_amount == 0.0 or solved_amount < search_ceiling:
        return BIND_AFF
    top = math.floor(requested_amount / search_grid) * search_grid
    if solved_amount >= top:
        return BIND_REQ
    return BIND_CAP if solved_amount >= math.floor(amount_cap / search_grid) * search_grid else BIND_MAX


def solve_unit():
    """Reads `requested_amount`, `amount_cap`, `instalment_ceiling` and what `pricing.candidate_unit` reads; writes
    `solved_amount`, `solve_binding_constraint` and `record`, the winning `pricing.Quote` (null when nothing fits)."""
    best = optimise(
        candidate_count,
        flow(step(offered_amount), pricing.candidate_unit(), affordable_amount, name="evaluate"),
        disqualify=unaffordable, max_candidates=MAX_CANDIDATES, name="solve", record=pricing.Quote)
    return flow(step(search_domain, outputs=("search_floor", "search_grid", "search_ceiling")), best,
                step(solved_amount), step(solve_binding_constraint), name="solve_offer").emit(
        "solved_amount", "solve_binding_constraint", "record")
