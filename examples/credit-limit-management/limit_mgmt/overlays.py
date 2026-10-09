"""The overlays the programme runs under: the dial's arithmetic and the cycle cap, and the review check.

An overlay tightens a value without editing the table or model behind it, and the unadjusted value stays beside the
adjusted one. The dial is a `multiply` on the multiplier's excess over 1.00 (so a 1.50 cell at an 80% dial is 1.40,
not 1.20). The overlays themselves are `adjustment_stack` documents (`configs/<version>/*_adjustments.json`). The cycle
cap is a ceiling, which `add`/`multiply`/`set` cannot express, so it is a param-switched step.
"""
from __future__ import annotations

from datetime import date

from decider import param, step

from limit_mgmt.documents import shipped

DOCUMENTS = ("behaviour_score_adjustments", "behaviour_pd_adjustments", "matrix_dial_adjustments", "buffer_adjustments")


def due_for_review(on: date, stacks=None) -> list[str]:
    """Ids of the overlays past their review date and not yet lapsed. A cycle should not run while any are."""
    return [a.adjustment_id for stack in stacks or map(shipped, DOCUMENTS) for a in stack.due_for_review(on)]


def matrix_multiplier_excess(matrix_multiplier_unadjusted: float) -> float:
    return round(matrix_multiplier_unadjusted - 1.0, 6)


def matrix_multiplier(matrix_multiplier_excess_adjusted: float) -> float:
    return round(1.0 + matrix_multiplier_excess_adjusted, 4)


def apply_cycle_cap(
    matrix_max_increase_unadjusted: float,
    cycle_cap_enabled: bool = param(False),
    cycle_cap_amount: float = param(5_000.0, ge=0.0),
) -> tuple[float, bool]:
    """The cell's maximum increase, held to `cycle_cap_amount` while the cap is on; and whether the cap bound."""
    if not cycle_cap_enabled:
        return matrix_max_increase_unadjusted, False
    capped = min(matrix_max_increase_unadjusted, cycle_cap_amount)
    return capped, capped < matrix_max_increase_unadjusted


matrix_multiplier_excess_step = step(matrix_multiplier_excess)
matrix_multiplier_step = step(matrix_multiplier)
apply_cycle_cap_step = step(apply_cycle_cap, outputs=("matrix_max_increase", "cycle_cap_bound"))
