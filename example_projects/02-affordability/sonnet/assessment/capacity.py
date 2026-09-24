"""Stage 6 -- discretionary income and capacity (spec 02 §5.6).

`core.affordability.discretionary_income` (00) supplies the ladder itself
(gross minus statutory deductions minus living expenses minus court-ordered
deductions minus obligations). What DEPS.md's "Cycles" §2 leaves to 02 is
the *capacity* step on top: two constraints -- a proportional buffer and an
absolute residual floor -- of which "the binding one wins" (§5.6.2), and a
tighten-only overlay stack (`core.adjustments`, reused unmodified) over the
combined result.
"""
from __future__ import annotations

from decider import step
from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id

BUFFER_GRID_VERSION = "buffer-grid-2026.09"
RESIDUAL_FLOOR_VERSION = "residual-floor-2026.09"

BUFFER_BOUND = 1
RESIDUAL_FLOOR_BOUND = 2

# 12 grades x 6 products -- working depth against the spec's 12 x 8 (00 NOTES.md's own
# precedent: only the slice's one dominant-difficulty table is kept at full declared
# size; here that is `core.rate_card`, built in 00, not this grid).
_PRODUCTS = (10, 11, 20, 21, 30, 40)


def _buffer_pct(grade: int, product: int) -> float:
    """10% at grade 1 rising to 35% at grade 12 (§5.6.2), plus a per-product loading for
    longer-term products -- synthetic, the same shape the spec describes."""
    base = 0.10 + (grade - 1) * (0.25 / 11)
    loading = {10: 0.0, 11: 0.01, 20: 0.0, 21: 0.0, 30: 0.02, 40: 0.05}[product]
    return round(min(base + loading, 0.60), 4)


def build_buffer_grid() -> DecisionTableConfig:
    rows = [
        {"product": product, "grade": grade, "buffer_pct": _buffer_pct(grade, product),
         "cell_id": _cell_id("buffer_grid", BUFFER_GRID_VERSION, product, grade)}
        for product in _PRODUCTS for grade in range(1, 13)
    ]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "affordability_buffer_grid",
        "columns": {"product": "Int64", "grade": "Int64", "buffer_pct": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
        ]},
        "outputs": ["buffer_pct", "cell_id"],
        "default": [0.35, None],  # unmatched grade/product: the most conservative band
    }).relabel(writes={"buffer_pct": "affordability_buffer_applied", "cell_id": "buffer_grid_cell_id"})


# dependants_count band -> residual floor amount (§6.1: "6 dependant counts").
_RESIDUAL_BANDS = [
    (float("-inf"), 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, float("inf")),
]
_RESIDUAL_AMOUNTS = [800.0, 1_100.0, 1_400.0, 1_700.0, 2_000.0, 2_300.0]


def build_residual_floor_table() -> DecisionTableConfig:
    rows = [
        {"lo": lo, "hi": hi, "amount": amount, "cell_id": _cell_id("residual_floor", RESIDUAL_FLOOR_VERSION, i)}
        for i, ((lo, hi), amount) in enumerate(zip(_RESIDUAL_BANDS, _RESIDUAL_AMOUNTS))
    ]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "residual_floor",
        "columns": {"lo": "Float64", "hi": "Float64", "amount": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "between", "variable": "dependants_count", "lower_bound_column": "lo",
                        "upper_bound_column": "hi", "allow_gaps": True},
        "outputs": ["amount", "cell_id"],
        "default": [2_300.0, None],
    }).relabel(writes={"amount": "residual_floor_amount", "cell_id": "residual_floor_cell_id"})


def max_affordable_instalment_before_overlay(
    discretionary_income: float, affordability_buffer_applied: float, residual_floor_amount: float,
) -> tuple[float, int]:
    """§5.6.2: "`max_affordable_instalment` is not `discretionary_income`... two constraints
    apply and the binding one wins." `discretionary_income` itself is never floored at zero
    (§5.6.1) but capacity cannot be negative, so it is floored here. Depends only on
    `discretionary_income`, `affordability_buffer_applied` and `residual_floor_amount` --
    never on `proposed_instalment` -- which is what keeps the verdict monotone in
    `proposed_instalment` (§5.7.2(c)): moving the proposed instalment can never move this
    number."""
    di = max(0.0, discretionary_income)
    from_buffer = round(max(0.0, di * (1.0 - affordability_buffer_applied)), 2)
    from_floor = round(max(0.0, di - residual_floor_amount), 2)
    if from_buffer <= from_floor:
        return from_buffer, BUFFER_BOUND
    return from_floor, RESIDUAL_FLOOR_BOUND


max_affordable_instalment_before_overlay_step = step(
    max_affordable_instalment_before_overlay,
    outputs=("max_affordable_instalment_before_overlay", "affordability_binding_constraint_code"),
)
