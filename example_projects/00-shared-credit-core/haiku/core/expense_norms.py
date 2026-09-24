from __future__ import annotations
"""Minimum living expense floor (00 §6.3)."""
from dataclasses import dataclass


@dataclass
class ExpenseNormsResult:
    """Result of expense norm calculation."""
    living_expenses: float
    expense_basis_code: int  # 1=declared, 2=norm floor, 3=statement-derived
    norm_table_version: str  # Version ID for audit trail


# Simplified norm table: income bands × dependants
# Real implementation would have full statutory and internal variants
NORM_TABLE = {
    # (income_band, dependants) -> minimum_monthly_expense
    (5000, 0): 1200,
    (5000, 1): 1500,
    (5000, 2): 1800,
    (10000, 0): 1500,
    (10000, 1): 2000,
    (10000, 2): 2500,
    (15000, 0): 2000,
    (15000, 1): 2500,
    (15000, 2): 3000,
    (20000, 0): 2500,
    (20000, 1): 3000,
    (20000, 2): 3500,
}


def apply_expense_norms(
    gross_monthly_income: float,
    dependants_count: int,
    declared_living_expenses: float,
    statement_derived_expenses: float = 0.0
) -> ExpenseNormsResult:
    """
    Apply the statutory minimum-expense-norm mechanism.
    A floor on declared living expenses so clients cannot be lent to on basis
    of implausibly low expense claims.

    Two tables: statutory (regulator-published) and internal (stricter).
    The binding one (higher) applies.
    """
    # Find applicable norm from table
    norm_value = 0.0

    # Simplified: find closest income band
    bands = sorted([b[0] for b in NORM_TABLE.keys() if b[1] == dependants_count])
    if bands:
        for band in bands:
            if gross_monthly_income <= band:
                norm_value = NORM_TABLE.get((band, dependants_count), 0.0)
                break
        if norm_value == 0.0 and bands:
            norm_value = NORM_TABLE.get((bands[-1], dependants_count), 0.0)

    # Determine which expense to use (binding highest)
    candidates = [declared_living_expenses, statement_derived_expenses, norm_value]
    living_expenses = max(c for c in candidates if c > 0)

    if living_expenses == norm_value:
        basis = 2  # Norm floor applied
    elif statement_derived_expenses > 0 and living_expenses == statement_derived_expenses:
        basis = 3  # Statement-derived
    else:
        basis = 1  # Declared

    return ExpenseNormsResult(
        living_expenses=living_expenses,
        expense_basis_code=basis,
        norm_table_version="1.0.0"  # Would come from config
    )
