"""dti_features – module 2 of the pandas_streaming demo.

Derives two features from the debt_to_income column:

* ``dti_pct``        – DTI expressed as a percentage (0–100 range)
* ``dti_risk_flag``  – boolean: True when DTI > 0.4

Input:
    ``input`` (``pd.DataFrame``) – a batch containing a ``debt_to_income`` column.
"""

import pandas as pd


def debt_to_income(input: pd.DataFrame) -> pd.Series:
    """Extract the debt_to_income column from the input DataFrame."""
    return input["debt_to_income"]

def dti_pct(debt_to_income: pd.Series) -> pd.Series:
    """Convert debt-to-income ratio to a percentage (multiply by 100)."""
    return (debt_to_income * 100).round(2)

def dti_risk_flag(debt_to_income: pd.Series) -> pd.Series:
    """True when debt-to-income is greater than 0.4 (high DTI risk)."""
    return debt_to_income > 0.4
