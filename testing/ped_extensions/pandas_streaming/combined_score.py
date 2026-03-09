"""combined_score – module 3 of the pandas_streaming demo.

Combines the outputs of modules 1 & 2 into a single composite risk score:

* ``credit_score`` – a simple heuristic score in [0, 100]:
      50  × (1 – dti_pct / 100)    (rewards low DTI)
    + 50  × (income_log / log1p(150_000))  (rewards higher income)

  Higher score = lower risk.

Inputs (all ``pd.Series`` produced by upstream modules):
    ``income_log``   – from income_features
    ``dti_pct``      – from dti_features
"""

import pandas as pd
import numpy as np


_MAX_INCOME_LOG = np.log1p(150_000)


def credit_score(income_log: pd.Series, dti_pct: pd.Series) -> pd.Series:
    """Composite credit score in [0, 100]. Higher = lower risk."""
    income_component = (income_log / _MAX_INCOME_LOG).clip(0, 1) * 50
    dti_component = (1 - (dti_pct / 100).clip(0, 1)) * 50
    return (income_component + dti_component).round(2)
