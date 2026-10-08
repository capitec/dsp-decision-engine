"""Minimum living expenses: the higher of the statutory norm and the Bank's stricter internal norm.

Both tables are effective-dated. The record keeps the cell it read from each, and which table bound.
"""
from __future__ import annotations

import math
from datetime import date

from decider import DecisionTableConfig, allow_fallback, missing_as

from credit_core.dates import EffectiveDatedSet, EffectiveVersion
from credit_core.evidence import cell_id

# expense_basis_code
DECLARED = 1
NORM_FLOOR = 2
STATEMENT_DERIVED = 3

STATUTORY_NORMS = EffectiveDatedSet("expense_norms.statutory", [
    EffectiveVersion("norms.statutory@2025", date(2025, 1, 1), date(2026, 10, 1)),
    EffectiveVersion("norms.statutory@2026", date(2026, 10, 1)),  # gazetted change
])
INTERNAL_NORMS = EffectiveDatedSet("expense_norms.internal", [
    EffectiveVersion("norms.internal@2025", date(2025, 1, 1), date(2026, 4, 1)),
    EffectiveVersion("norms.internal@2026", date(2026, 4, 1)),
])
statutory_version_step = STATUTORY_NORMS.resolver_step(output="expense_norms_statutory_version")
internal_version_step = INTERNAL_NORMS.resolver_step(output="expense_norms_internal_version")

# The grid has two dimensions, so each dimension's bands repeat once per group of rows. A band ladder with
# an open edge (None) is only allowed once per group, so these bands carry explicit infinite edges instead.
_INF = float("inf")
_INCOME_BANDS = [(-_INF, 3_000.0), (3_000.0, 6_000.0), (6_000.0, 12_000.0), (12_000.0, _INF)]
_DEPENDANT_BANDS = [(-_INF, 0.5), (0.5, 2.5), (2.5, _INF)]

# Statutory amount per (income band, dependants band), row-major; the 2026 norms are 5% above 2025.
_STATUTORY_2025 = [1500, 1800, 2200, 1800, 2200, 2700, 2300, 2800, 3400, 2900, 3500, 4200]
_STATUTORY = {"norms.statutory@2025": _STATUTORY_2025,
              "norms.statutory@2026": [round(a * 1.05) for a in _STATUTORY_2025]}
# Internal norm as a multiple of the statutory one, per product.
_INTERNAL_MULTIPLIER = {"norms.internal@2025": {10: 1.10, 20: 1.15}, "norms.internal@2026": {10: 1.12, 20: 1.18}}


def _grid():
    """(cell index, income lo, income hi, dependants lo, dependants hi) for every cell of one version."""
    cells = [(il, ih, dl, dh) for il, ih in _INCOME_BANDS for dl, dh in _DEPENDANT_BANDS]
    return [(i, *cell) for i, cell in enumerate(cells)]


def _band_expression(version_variable: str, *extra_eq: tuple[str, str]) -> dict:
    return {"type": "and", "expressions": [
        {"type": "eq", "variable": version_variable, "value_column": "version"},
        *({"type": "eq", "variable": v, "value_column": c} for v, c in extra_eq),
        {"type": "between", "variable": "gross_monthly_income", "lower_bound_column": "inc_lo",
         "upper_bound_column": "inc_hi", "allow_gaps": True},
        {"type": "between", "variable": "dependants_count", "lower_bound_column": "dep_lo",
         "upper_bound_column": "dep_hi", "allow_gaps": True},
    ]}


def build_statutory_table() -> DecisionTableConfig:
    """Writes `statutory_norm_amount` and `statutory_norm_cell_id`."""
    rows = [{"version": version, "inc_lo": il, "inc_hi": ih, "dep_lo": dl, "dep_hi": dh,
             "amount": float(amounts[i]), "cell_id": cell_id("expense_norms.statutory", version, i)}
            for version, amounts in _STATUTORY.items() for i, il, ih, dl, dh in _grid()]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "statutory_expense_norms",
        "columns": {"version": "String", "inc_lo": "Float64", "inc_hi": "Float64", "dep_lo": "Float64",
                    "dep_hi": "Float64", "amount": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": _band_expression("expense_norms_statutory_version"),
        "outputs": ["amount", "cell_id"],
        "default": [0.0, None],
    }).relabel(writes={"amount": "statutory_norm_amount", "cell_id": "statutory_norm_cell_id"})


def build_internal_table() -> DecisionTableConfig:
    """Writes `internal_norm_multiplier` and `internal_norm_cell_id`, per `product_code`."""
    rows = [{"version": version, "product": product, "inc_lo": il, "inc_hi": ih, "dep_lo": dl, "dep_hi": dh,
             "multiplier": multiplier, "cell_id": cell_id("expense_norms.internal", version, product, i)}
            for version, by_product in _INTERNAL_MULTIPLIER.items() for product, multiplier in by_product.items()
            for i, il, ih, dl, dh in _grid()]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "internal_expense_norms",
        "columns": {"version": "String", "product": "Int64", "inc_lo": "Float64", "inc_hi": "Float64",
                    "dep_lo": "Float64", "dep_hi": "Float64", "multiplier": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": _band_expression("expense_norms_internal_version", ("product_code", "product")),
        "outputs": ["multiplier", "cell_id"],
        "default": [1.0, None],
    }).relabel(writes={"multiplier": "internal_norm_multiplier", "cell_id": "internal_norm_cell_id"})


def internal_norm_amount(statutory_norm_amount: float, internal_norm_multiplier: float) -> float:
    return round(statutory_norm_amount * internal_norm_multiplier, 2)


def norm_floor(statutory_norm_amount: float, internal_norm_amount: float) -> float:
    """The binding floor: the higher of the two norms."""
    return max(statutory_norm_amount, internal_norm_amount)


@allow_fallback
def norm_table_version(
    statutory_norm_amount: float, internal_norm_amount: float,
    expense_norms_statutory_version: str, expense_norms_internal_version: str,
) -> str:
    """The version of the table whose norm bound (the internal one on a tie)."""
    return expense_norms_internal_version if internal_norm_amount >= statutory_norm_amount \
        else expense_norms_statutory_version


def _claimed(declared: float, statement: float) -> tuple[float, int]:
    if not math.isnan(statement):
        return statement, STATEMENT_DERIVED
    return (0.0 if math.isnan(declared) else declared), DECLARED


def expense_basis_code(
    norm_floor: float, declared_living_expenses: float = missing_as(math.nan),
    statement_living_expenses: float = missing_as(math.nan),
) -> int:
    """`STATEMENT_DERIVED` or `DECLARED` for the claimed figure, or `NORM_FLOOR` when the floor is higher."""
    claimed, basis = _claimed(declared_living_expenses, statement_living_expenses)
    return NORM_FLOOR if norm_floor > claimed else basis


def living_expenses(
    norm_floor: float, declared_living_expenses: float = missing_as(math.nan),
    statement_living_expenses: float = missing_as(math.nan),
) -> float:
    """The higher of the claimed expenses (statement over declared) and the norm floor."""
    claimed, _ = _claimed(declared_living_expenses, statement_living_expenses)
    return round(max(claimed, norm_floor), 2)
