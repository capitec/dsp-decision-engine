"""`core.expense_norms` -- minimum living expense floor (spec 00 §6.3).

Two tables, both effective-dated, and the *binding* one is whichever is
higher for that record (00 §6.3 "Hard part") -- the statutory
minimum-expense-norm and the Bank's stricter internal variant per product.
Which table (and which version of it) bound must be in the evidence, so
both lookups carry their own `cell_id` through to the final basis.

# ponytail: 4 income bands x 3 dependant bands, not the spec's 12 x 6 (and
# 2 products for the internal variant, not 8). This table is not this
# slice's dominant-difficulty artefact (that's `core.rate_card`); the
# mechanism -- two effective-dated tables, binding = higher, basis
# recorded -- is what SCOPE.md asks this capability to prove. Widen
# `_STATUTORY_BANDS` / `_INTERNAL_BANDS` to extend.

Change scenario 1 (SCOPE.md): "the regulator gazettes new expense norms
effective the first of next month; decisions before that date must
continue to use the old table forever" is exactly `EffectiveDatedSet` +
per-record version resolution, exercised in `tests/test_expense_norms.py`.
"""
from __future__ import annotations

from datetime import date

from decider.steps.tables import DecisionTableConfig

from credit_core.dates import EffectiveDatedSet, EffectiveVersion
from credit_core.evidence import cell_id as _cell_id

DECLARED = 1
NORM_FLOOR = 2
STATEMENT_DERIVED = 3

STATUTORY_NORMS = EffectiveDatedSet(
    "expense_norms.statutory",
    [
        EffectiveVersion("norms.statutory@2025", date(2025, 1, 1), date(2026, 10, 1)),
        EffectiveVersion("norms.statutory@2026", date(2026, 10, 1), None),  # the gazetted change
    ],
)
INTERNAL_NORMS = EffectiveDatedSet(
    "expense_norms.internal",
    [
        EffectiveVersion("norms.internal@2025", date(2025, 1, 1), date(2026, 4, 1)),
        EffectiveVersion("norms.internal@2026", date(2026, 4, 1), None),
    ],
)
statutory_version_step = STATUTORY_NORMS.resolver_step(output="expense_norms_statutory_version")
internal_version_step = INTERNAL_NORMS.resolver_step(output="expense_norms_internal_version")

# income band index -> (lo, hi); dependants band index -> (lo, hi). +-inf, not None: this
# ladder repeats once per version/product group, and DecisionTableConfig only allows one
# open-lower and one open-upper row in the whole table (see credit_core.rate_card._bands).
_INF = float("inf")
_INCOME_BANDS = [(-_INF, 3_000.0), (3_000.0, 6_000.0), (6_000.0, 12_000.0), (12_000.0, _INF)]
_DEP_BANDS = [(-_INF, 0.5), (0.5, 2.5), (2.5, _INF)]

# base statutory amount by (income_band_idx, dep_band_idx), per version. Rows before
# the gazette are 5% lower than after, a small realistic "the regulator raised the floor".
_STATUTORY_BASE = {"norms.statutory@2025": [1500, 1800, 2200, 1800, 2200, 2700, 2300, 2800, 3400, 2900, 3500, 4200],
                    "norms.statutory@2026": [1575, 1890, 2310, 1890, 2310, 2835, 2415, 2940, 3570, 3045, 3675, 4410]}
# internal multiplier over statutory, per product (10=Flex Loan, 20=Everyday Card), per version.
_INTERNAL_MULTIPLIER = {"norms.internal@2025": {10: 1.10, 20: 1.15},
                         "norms.internal@2026": {10: 1.12, 20: 1.18}}


def build_statutory_table() -> DecisionTableConfig:
    rows = []
    for version, amounts in _STATUTORY_BASE.items():
        i = 0
        for inc_lo, inc_hi in _INCOME_BANDS:
            for dep_lo, dep_hi in _DEP_BANDS:
                rows.append({
                    "version": version, "inc_lo": inc_lo, "inc_hi": inc_hi, "dep_lo": dep_lo, "dep_hi": dep_hi,
                    "amount": float(amounts[i]), "cell_id": _cell_id("expense_norms.statutory", version, i),
                })
                i += 1
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "statutory_expense_norms",
        "columns": {"version": "String", "inc_lo": "Float64", "inc_hi": "Float64", "dep_lo": "Float64",
                    "dep_hi": "Float64", "amount": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "expense_norms_statutory_version", "value_column": "version"},
            {"type": "between", "variable": "gross_monthly_income", "lower_bound_column": "inc_lo",
             "upper_bound_column": "inc_hi", "allow_gaps": True},
            {"type": "between", "variable": "dependants_count", "lower_bound_column": "dep_lo",
             "upper_bound_column": "dep_hi", "allow_gaps": True},
        ]},
        "outputs": ["amount", "cell_id"],
        "default": [0.0, None],
    }).relabel(writes={"cell_id": "statutory_norm_cell_id"})


def build_internal_table() -> DecisionTableConfig:
    rows = []
    for version, by_product in _INTERNAL_MULTIPLIER.items():
        for product, mult in by_product.items():
            i = 0
            for inc_lo, inc_hi in _INCOME_BANDS:
                for dep_lo, dep_hi in _DEP_BANDS:
                    rows.append({
                        "version": version, "product": product, "inc_lo": inc_lo, "inc_hi": inc_hi,
                        "dep_lo": dep_lo, "dep_hi": dep_hi, "multiplier": mult,
                        "cell_id": _cell_id("expense_norms.internal", version, product, i),
                    })
                    i += 1
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "internal_expense_norms",
        "columns": {"version": "String", "product": "Int64", "inc_lo": "Float64", "inc_hi": "Float64",
                    "dep_lo": "Float64", "dep_hi": "Float64", "multiplier": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "expense_norms_internal_version", "value_column": "version"},
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "between", "variable": "gross_monthly_income", "lower_bound_column": "inc_lo",
             "upper_bound_column": "inc_hi", "allow_gaps": True},
            {"type": "between", "variable": "dependants_count", "lower_bound_column": "dep_lo",
             "upper_bound_column": "dep_hi", "allow_gaps": True},
        ]},
        "outputs": ["multiplier", "cell_id"],
        "default": [1.0, None],
    }).relabel(writes={"cell_id": "internal_norm_cell_id"})


def internal_norm_amount(amount: float, multiplier: float) -> float:
    """`amount` here is the statutory table's output, reused as the internal variant's base."""
    return round(amount * multiplier, 2)


def norm_floor(amount: float, internal_norm_amount: float) -> float:
    """The binding floor: whichever table produced the higher figure (00 §6.3 "Hard part")."""
    return max(amount, internal_norm_amount)


def norm_table_version(amount: float, internal_norm_amount: float,
                        expense_norms_statutory_version: str, expense_norms_internal_version: str) -> str:
    return (expense_norms_internal_version if internal_norm_amount >= amount
            else expense_norms_statutory_version)


def expense_basis_code(
    norm_floor: float, declared_living_expenses: float | None = None, statement_living_expenses: float | None = None,
) -> int:
    claimed, basis = _claimed(declared_living_expenses, statement_living_expenses)
    return NORM_FLOOR if norm_floor > claimed else basis


def living_expenses(
    norm_floor: float, declared_living_expenses: float | None = None, statement_living_expenses: float | None = None,
) -> float:
    claimed, _basis = _claimed(declared_living_expenses, statement_living_expenses)
    return round(max(claimed, norm_floor), 2)


def _claimed(declared: float, statement: float) -> tuple[float, int]:
    if statement is not None:
        return statement, STATEMENT_DERIVED
    return (declared or 0.0), DECLARED
