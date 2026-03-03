"""
Simple Hamilton module used in the getting_started.ipynb demo.

Inputs (injected at execution time):
    income          – pl.Expr  (e.g. pl.col("income"))
    debt_to_income  – pl.Expr  (e.g. pl.col("debt_to_income"))

Computed nodes:
    income_scaled       – income / 100_000  (normalises to [0, 1] range)
    dti_risk            – debt_to_income * 100  (converts ratio to percentage points)
    combined_risk_score – weighted combination of both signals
"""
import polars as pl
from .helper import mul


def income_scaled(income: pl.Expr) -> pl.Expr:
    """Normalise income to a 0-1 scale (capped at 100 000)."""
    return (income / 100_000).clip(0.0, 1.0)


def dti_risk(debt_to_income: pl.Expr) -> pl.Expr:
    """Convert debt-to-income ratio to a percentage-point risk score."""
    return debt_to_income * 100


def combined_risk_score(income_scaled: pl.Expr, dti_risk: pl.Expr) -> pl.Expr:
    """Weighted combination: lower income and higher DTI both increase risk.

    Score is in the range [0, 100] where higher means higher risk.
    """
    # Note the way we are doing imports we cannot do:
    # from helper import mul
    # Because we only load in the path for the duration of the import statement
    return (mul(dti_risk, 0.6) + mul((1 - income_scaled), 40)).round(2)
