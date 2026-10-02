from __future__ import annotations

import polars as pl
import pytest

from decider import flow, param


def ratio(income: float, debt: float) -> float:
    return debt / income if income > 0 else 1.0


def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit


def tier(approved: bool, income: float) -> float:
    return 1.0 if approved else (0.5 if income > 0 else 0.0)


@pytest.fixture
def pipeline():
    return flow(ratio, approved, tier, name="demo")


@pytest.fixture
def frame():
    # ratios 0.0, 0.2, 0.4, 0.6: a limit of 0.3 rejects record 2 as well as record 3.
    return pl.DataFrame({"income": [1000.0] * 4, "debt": [float(i * 200) for i in range(4)]})

