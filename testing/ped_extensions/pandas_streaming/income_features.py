"""income_features – module 1 of the pandas_streaming demo.

Derives two features from the income column:

* ``income_log`` – natural log of income (handles 0 with a +1 guard)
* ``income_band`` – categorical band: "low" / "mid" / "high"

Input:
    ``input`` (``pd.DataFrame``) – a batch containing an ``income`` column.
"""

import pandas as pd
import numpy as np


def income_log(input: pd.DataFrame) -> pd.Series:
    """Natural log of income (log1p to avoid log(0))."""
    return np.log1p(input["income"])


def income_band(input: pd.DataFrame) -> pd.Series:
    """Bucket income into 'low' / 'mid' / 'high' bands."""
    cuts = pd.cut(
        input["income"],
        bins=[0, 40_000, 80_000, float("inf")],
        labels=["low", "mid", "high"],
        right=True,
    )
    return cuts.astype(str)
