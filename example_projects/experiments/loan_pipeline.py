"""A small loan-pricing flow with a branch, for the experiment example notebooks — not a product pipeline.

`ratio -> rate -> pricing (secured | unsecured) -> approved, payment`: the
branch gives metrics a step inside an arm to attach to, and `approved` and
`payment` are the flow's output.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from decider import branch, flow, param, step


def ratio(income: float, debt: float) -> float:
    return debt / income if income > 0 else 1.0


def rate(ratio: float, base: float = param(0.10)) -> float:
    return base + 0.05 * ratio


def is_secured(collateral: float) -> bool:
    return collateral > 0


@step(output="rate")
def secured(rate: float, discount: float = param(0.02)) -> float:
    return rate - discount


@step(output="rate")
def unsecured(rate: float, ratio: float, premium: float = param(0.04)) -> float:
    return rate + premium * (1.0 + ratio)


def approved(ratio: float, rate: float, limit: float = param(0.4), max_rate: float = param(0.2)) -> bool:
    return ratio <= limit and rate <= max_rate


def payment(debt: float, rate: float) -> float:
    return debt * (1.0 + rate) / 12.0


def build():
    pricing = branch(is_secured, secured, unsecured, modifies=["rate"], name="pricing")
    return flow(ratio, rate, pricing, approved, payment, name="loans")


def applications(n: int = 100_000, seed: int = 0) -> pl.DataFrame:
    """`n` synthetic applications: income, debt and collateral (zero for about 40% of them)."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(8.2, 0.4, n).round(0)
    return pl.DataFrame({
        "income": income,
        "debt": (income * rng.beta(2, 5, n)).round(0),
        "collateral": np.where(rng.random(n) < 0.4, 0.0, rng.uniform(1_000, 20_000, n)).round(0),
    })
