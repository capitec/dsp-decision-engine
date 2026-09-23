from __future__ import annotations

from typing import Any

import polars as pl

from decider.engine.ir.decls import NullPolicy, base_annotation
from decider.engine.wiring import resolve

# One past the largest integer float64 holds exactly: where an int -> float cast first loses precision.
INT_NEAR_2_53 = 2**53 + 1

_BASELINE = {bool: True, str: "corpus"}
_ZERO = {bool: False, str: "", int: 0}


def corpus(step: Any, baseline: float = 1.0) -> dict[str, pl.DataFrame]:
    """Boundary-value frames for every input column `step` reads.

    `"boundary"` has a baseline row plus, per input, one row moving only that
    column to zero, to a negative value (not for `bool` or `str`), to null
    (only when no step reading it declares it required, since a required null
    is an error) and, for an `int`, to 2**53 + 1. A `case` column names each
    row, e.g. `"zero:net_income"`. `"single"` is its first row, `"chunked"`
    the same rows in two chunks, `"empty"` no rows.

    Example::

        for name, frame in corpus(pipeline).items():
            assert_equivalent(pipeline, frame)
    """
    plan = resolve(step)
    if not plan.inputs:
        raise ValueError("corpus(): the step reads no input columns")
    required = {v.name for c in plan.calls for i, v in zip(c.node.inputs, c.reads)
                if v.producer is None and i.null_policy is NullPolicy.REQUIRED}
    types = {i.name: _scalar(i.annotation) for i in plan.inputs}
    base = {name: _BASELINE.get(t, t(baseline)) for name, t in types.items()}
    rows = [base | {"case": "baseline"}]
    for name, t in types.items():
        edges = {"zero": _ZERO.get(t, 0.0)}
        if t in (int, float):
            edges["negative"] = -t(baseline)
        if name not in required:
            edges["null"] = None
        if t is int:
            edges["int_near_2**53"] = INT_NEAR_2_53
        rows += [base | {name: value, "case": f"{edge}:{name}"} for edge, value in edges.items()]
    schema = {name: _DTYPES[t] for name, t in types.items()} | {"case": pl.String}
    boundary = pl.DataFrame(rows, schema=schema)
    half = boundary.height // 2
    return {
        "boundary": boundary,
        "single": boundary.head(1),
        "chunked": pl.concat([boundary.head(half), boundary.tail(-half)], rechunk=False),
        "empty": boundary.clear(),
    }


_DTYPES = {bool: pl.Boolean, int: pl.Int64, float: pl.Float64, str: pl.String}


def _scalar(annotation: Any) -> type:
    t = base_annotation(annotation)
    return t if t in _DTYPES else float
