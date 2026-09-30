"""A minimal external package that has no durable ids, to prove imported steps
lower with `Origin.id is None` (derived-only identity) and the real import path
shows up in `Origin.source`.
"""


def statutory_deductions(income: float) -> float:
    return income * 0.1


def helper(income: float) -> float:
    return income
