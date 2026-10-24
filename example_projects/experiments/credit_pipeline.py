"""A minimal flow for the experiment example notebooks — not a product pipeline.

Deliberately tiny (ratio -> approved) so the notebooks under
`example_projects/experiments/` can focus on configuring an experiment, not on
the flow itself.
"""
from __future__ import annotations

from decikit import flow, param


def ratio(income: float, debt: float) -> float:
    return (debt + 50.0) / income if income > 0 else 1.0


def approved(ratio: float, limit: float = param(0.35)) -> bool:
    return ratio <= limit


def build():
    return flow(ratio, approved, name="credit")
