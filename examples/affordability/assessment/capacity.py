"""How much of the discretionary income may be committed: a proportional buffer and an absolute floor race, then overlays tighten."""
from __future__ import annotations

from datetime import date

from decider import DecisionTableConfig, flow, step

from credit_core import affordability, rounding
from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.evidence import cell_id

BUFFER_GRID_VERSION = "buffer-grid-2026.09"
RESIDUAL_FLOOR_VERSION = "residual-floor-2026.09"
ADJUSTMENT_SET_ID = "AS-02-2026.09"

# affordability_binding_constraint_code
BUFFER_BOUND = 1
RESIDUAL_FLOOR_BOUND = 2

_PRODUCTS = {10: 0.0, 11: 0.01, 20: 0.0, 21: 0.0, 30: 0.02, 40: 0.05}  # product -> extra buffer for longer terms

# Tighten-only: an overlay that would raise capacity is rejected when it is defined.
CAPACITY_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-02-2026-001", kind="cap_adjustment", target="max_affordable_instalment",
        effect=AdjustmentEffect("multiply", 0.90), scope={"product_code": 10}, stack_position=1,
        owner="Credit Risk Policy", approval_reference="CRC-2026-031",
        rationale="Product 10 early-life delinquency above appetite for two quarters",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True),
])


def build_buffer_grid() -> DecisionTableConfig:
    """The share of discretionary income held back per risk grade and product: 10% at grade 1 rising to 35% at grade 12.

    Writes `affordability_buffer_applied` and `buffer_grid_cell_id`; an unlisted grade or product gets 35%.
    """
    rows = [{"product": product, "grade": grade, "buffer_pct": round(0.10 + (grade - 1) * 0.25 / 11 + extra, 4),
             "cell_id": cell_id("buffer_grid", BUFFER_GRID_VERSION, product, grade)}
            for product, extra in _PRODUCTS.items() for grade in range(1, 13)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "affordability_buffer_grid",
        "columns": {"product": "Int64", "grade": "Int64", "buffer_pct": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "eq", "variable": "risk_grade", "value_column": "grade"}]},
        "outputs": ["buffer_pct", "cell_id"],
        "default": [0.35, None],
    }).relabel(writes={"buffer_pct": "affordability_buffer_applied", "cell_id": "buffer_grid_cell_id"})


def build_residual_floor_table() -> DecisionTableConfig:
    """The rand amount that must remain after the instalment, by `dependants_count` (0 to 5 or more).

    Writes `residual_floor_amount` and `residual_floor_cell_id`.
    """
    amounts = [800.0, 1_100.0, 1_400.0, 1_700.0, 2_000.0, 2_300.0]
    edges = [None, 0.5, 1.5, 2.5, 3.5, 4.5, None]
    rows = [{"lo": edges[i], "hi": edges[i + 1], "amount": amount,
             "cell_id": cell_id("residual_floor", RESIDUAL_FLOOR_VERSION, i)} for i, amount in enumerate(amounts)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "residual_floor",
        "columns": {"lo": "Float64", "hi": "Float64", "amount": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "between", "variable": "dependants_count", "lower_bound_column": "lo",
                       "upper_bound_column": "hi"},
        "outputs": ["amount", "cell_id"],
        "default": [2_300.0, None],
    }).relabel(writes={"amount": "residual_floor_amount", "cell_id": "residual_floor_cell_id"})


# credit_core's capacity step, fed the grid's buffer instead of a flat one.
buffer_capacity = affordability.max_affordable_instalment_step.named("buffer_capacity").relabel(
    writes={"max_affordable_instalment": "buffer_capacity"})


def floor_capacity(discretionary_income: float, residual_floor_amount: float) -> float:
    return max(0.0, round(discretionary_income - residual_floor_amount, 2))


@step(outputs=("max_affordable_instalment_before_overlay", "affordability_binding_constraint_code"))
def binding_capacity(buffer_capacity: float, floor_capacity: float) -> tuple[float, int]:
    """The lower of the two limits, and which one it was (the buffer on a tie).

    It never reads the proposed instalment, so the verdict can only get worse as the proposal grows.
    """
    if buffer_capacity <= floor_capacity:
        return buffer_capacity, BUFFER_BOUND
    return floor_capacity, RESIDUAL_FLOOR_BOUND


def overlay_unit():
    """Applies `CAPACITY_ADJUSTMENTS`; writes `max_affordable_instalment`, its `_unadjusted` twin and the overlays applied."""
    return flow(
        CAPACITY_ADJUSTMENTS.apply_stack_step("max_affordable_instalment", ADJUSTMENT_SET_ID,
                                              base_field="max_affordable_instalment_before_overlay"),
        rounding.round_instalment_step.relabel(
            reads={"amount": "max_affordable_instalment"}, writes={"instalment": "max_affordable_instalment"}),
        name="overlay")
