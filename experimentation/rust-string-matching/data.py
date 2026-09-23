"""The owner's shape as data. `ft1`, `ft2` uniform in [0, 1); `ft3` an
employer name string column at one of two cardinalities:

  low  -- 12 distinct employers (3 of them start with "dog"), ~25% of rows
          match /^dog/
  high -- ~1 distinct per row (unique suffix), 25% start with "dog"

Selectivity of the numeric AND is set by `c1`: `ft1 > 1 - s` passes a
fraction `s` of rows; `ft2 > 0.0` always passes (so the AND is real but
the second test never filters). s = 0 uses c1 = 2.0.
"""
from __future__ import annotations
import numpy as np, polars as pl

EMPLOYERS_LOW = [
    "dog walkers inc", "dogsbody holdings", "dogma analytics",
    "capitec bank", "shoprite holdings", "sasol limited", "mtn group",
    "vodacom", "standard bank", "naspers", "discovery limited", "bidvest group",
]


def make_data(n: int, cardinality: str, seed: int = 0) -> tuple[pl.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    ft1 = rng.random(n)
    ft2 = rng.random(n)
    if cardinality == "low":
        idx = rng.integers(0, len(EMPLOYERS_LOW), n)
        emp = pl.Series("ft3", [EMPLOYERS_LOW[i] for i in idx])
        employers = EMPLOYERS_LOW
    elif cardinality == "high":
        prefix = np.where(rng.random(n) < 0.25, "dog", "cat")
        emp = pl.Series("ft3", [f"{p} employer #{i:08d}" for p, i in zip(prefix, range(n))])
        employers = None
    else:
        raise ValueError(cardinality)
    df = pl.DataFrame({"ft1": ft1, "ft2": ft2, "ft3": emp})
    return df, employers


def thresholds(selectivity: float) -> tuple[float, float]:
    c1 = 2.0 if selectivity == 0 else 1.0 - selectivity
    return c1, 0.0
