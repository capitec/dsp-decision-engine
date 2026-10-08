"""The overlays the programme runs under, all built on `credit_core.adjustments`.

An overlay tightens a value without editing the table or model behind it, and the unadjusted value stays beside the
adjusted one. The dial is a `multiply` on the multiplier's excess over 1.00 (so a 1.50 cell at an 80% dial is 1.40,
not 1.20). The cycle cap is a ceiling, which `add`/`multiply`/`set` cannot express, so it is a param-switched step.
"""
from __future__ import annotations

from datetime import date

from decider import param, step

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister

SET_ID = "AS-07-2026.09"

BEHAVIOUR_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-07-001", kind="score_shift", target="behaviour_score",
        effect=AdjustmentEffect("add", -15.0), scope={"channel_code": 5}, stack_position=1,
        owner="Model Risk", approval_reference="MRC-2026-041",
        rationale="Partner-originated channel running worse than model prediction in early life",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True),
    Adjustment(
        adjustment_id="ADJ-07-002", kind="odds_multiplier", target="probability_of_default",
        effect=AdjustmentEffect("multiply", 1.25), scope={"product_code": 21}, stack_position=1,
        owner="Model Risk", approval_reference="MRC-2026-044",
        rationale="Access Facility observed default running above prediction",
        effective_from=date(2026, 3, 1), effective_to=date(2027, 3, 1), review_date=date(2026, 12, 1),
        tighten_only=True),
])

MATRIX_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-07-DIAL-2026-03", kind="cap_adjustment", target="matrix_multiplier_excess",
        effect=AdjustmentEffect("multiply", 0.80), scope={}, stack_position=1,
        owner="Credit Risk Policy", approval_reference="CRC-2026-088",
        rationale="Funding environment tightened; run the programme at 80% of the matrix this cycle",
        effective_from=date(2026, 9, 1), effective_to=date(2026, 11, 1), review_date=date(2026, 10, 15),
        tighten_only=True),
])

# The programme underwrites against a salary pattern and a bureau file, not evidence collected days earlier, so its
# affordability buffer is wider than origination's. It is an overlay on the shared buffer grid, not a fork of it.
BUFFER_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-07-BUFFER-BASE", kind="buffer_adjustment", target="affordability_buffer_applied",
        effect=AdjustmentEffect("add", 0.06), scope={}, stack_position=1,
        owner="Credit Risk Policy", approval_reference="CRC-2026-091",
        rationale="Programme affordability rests on inferred income and a monthly bureau file",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True),
])

REGISTERS = (BEHAVIOUR_ADJUSTMENTS, MATRIX_ADJUSTMENTS, BUFFER_ADJUSTMENTS)


def due_for_review(on: date) -> list[str]:
    """Ids of the overlays past their review date and not yet lapsed. A cycle should not run while any are."""
    return [a.adjustment_id for register in REGISTERS for a in register.due_for_review(on)]


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

matrix_dial_step = MATRIX_ADJUSTMENTS.apply_stack_step(
    "matrix_multiplier_excess", SET_ID,
    adjusted_output="matrix_multiplier_excess_adjusted", unadjusted_output="matrix_multiplier_excess_unadjusted",
).relabel(writes={"adjustment_set_id": "matrix_adjustment_set_id", "adjustments_applied": "matrix_adjustments_applied"})

buffer_adjustment_step = BUFFER_ADJUSTMENTS.apply_stack_step(
    "affordability_buffer_applied", SET_ID, base_field="affordability_buffer_applied",
    unadjusted_output="affordability_buffer_unadjusted",
).relabel(writes={"adjustment_set_id": "buffer_adjustment_set_id", "adjustments_applied": "buffer_adjustments_applied"})
