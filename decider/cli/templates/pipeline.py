"""The {{name}} pipeline: `decider build` and `decider serve` call `build()` for each config version.

Plain arguments (`income`, `debt`, `applied_on`) arrive with each request.
`param()` arguments are policy, read from `configs/<version>/params.json`.
"""
from datetime import date

from decider import flow, missing_as, param


def debt_ratio(income: float, debt: float = missing_as(0.0)) -> float:
    return debt / income


def month_end(applied_on: date) -> bool:
    return applied_on.day >= 25


def approved(debt_ratio: float, month_end: bool,
             limit: float = param(0.4, ge=0.0, le=1.0),
             month_end_limit: float = param(0.3, ge=0.0, le=1.0)) -> bool:
    return debt_ratio <= (month_end_limit if month_end else limit)


def build():
    return flow(debt_ratio, month_end, approved, name="{{name}}")
