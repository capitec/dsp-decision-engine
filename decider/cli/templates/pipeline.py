"""The {{name}} pipeline: `decider serve` calls `build()` for each config version."""
from decider import flow, missing_as, param


def debt_ratio(income: float, debt: float = missing_as(0.0)) -> float:
    return debt / income


def approved(debt_ratio: float, limit: float = param(0.4, ge=0.0, le=1.0)) -> bool:
    return debt_ratio <= limit


def build():
    # Each argument added here is loaded from the config version's document of that name,
    # e.g. `build(tree)` loads `configs/<version>/tree.json` as a ConfigurableStep.
    return flow(debt_ratio, approved, name="{{name}}")
