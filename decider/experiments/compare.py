"""Comparing one scenario's output against the baseline, at the value level.

The comparison policy decides which produced columns are compared and with what
float tolerance; `diff_outputs` returns the changed row indices per column, and
`close` is the single equality rule every summary and visualisation shares.
"""
from __future__ import annotations

import math
from typing import Any

from decider.experiments.model import ComparisonPolicy, Tolerance


def close(a: Any, b: Any, tol: Tolerance) -> bool:
    """Whether `a` and `b` are equal under `tol` (floats with tolerance, everything else by value)."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= tol.atol + tol.rtol * max(abs(a), abs(b))
    return a == b


def diff_outputs(base: dict | None, other: dict | None,
                 comparison: ComparisonPolicy) -> dict[str, list[int]]:
    """The changed row indices per compared column; an empty dict means the outputs agree."""
    if base is None or other is None:
        return {}
    cols = comparison.outputs or tuple(base)
    out: dict[str, list[int]] = {}
    for col in cols:
        left, right = base.get(col), other.get(col)
        if left is None or right is None:
            continue
        changed = [i for i, (a, b) in enumerate(zip(left, right)) if not close(a, b, comparison.tolerance)]
        if changed:
            out[col] = changed
    return out
