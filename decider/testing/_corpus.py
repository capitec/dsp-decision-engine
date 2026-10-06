from __future__ import annotations

import typing
from typing import Any

import polars as pl

from decider._annotations import Money, metadata_of
from decider._engine.wiring import resolve
from decider.ir import NullPolicy, base_annotation

# One past the largest integer float64 holds exactly: where an int -> float cast first loses precision.
INT_NEAR_2_53 = 2**53 + 1

# Ties and exact-double surprises: half-even and half-up differ on them, and 2.675 and 1.005 sit just below the tie.
TIES = (0.125, 0.5, 1.005, 1.5, 2.5, 2.675, -2.5)
# Principals in cents: large but realistic, and the largest one whose product with 1000 still fits int64.
LARGE_CENTS = {"10**11": 10**11, "10**13": 10**13, "2**62//1000": 2**62 // 1000}

_BASELINE = {bool: True, str: "corpus"}
_ZERO = {bool: False, str: "", int: 0}


def corpus(step: Any, baseline: float = 1.0) -> dict[str, pl.DataFrame]:
    """Boundary-value frames for every input column `step` reads.

    `"boundary"` has a baseline row plus, per input, one row moving only that
    column to zero, to a negative value (not for `bool` or `str`), to null
    (only when no step reading it declares it required, since a required null
    is an error) and, for an `int`, to 2**53 + 1. A `float` also gets the
    rounding ties `0.125`, `2.675`, `-2.5`... (`"tie:2.675:rate"`), and an `int`
    annotated `Money(cents=True)` gets large principals up to one whose
    product with 1000 nears the int64 edge (`"large:10**11:principal"`). A
    `case` column names each row, e.g. `"zero:net_income"`. `"single"` is its
    first row, `"chunked"` the same rows in two chunks, `"empty"` no rows.
    A step with no inputs gets a one-row `"boundary"` holding only `case`.

    Example::

        for name, frame in corpus(pipeline).items():
            assert_equivalent(pipeline, frame)
    """
    plan = resolve(step)
    required = {v.name for c in plan.calls for i, v in zip(c.node.inputs, c.reads)
                if v.producer is None and i.null_policy is NullPolicy.REQUIRED}
    types = {i.name: _scalar(i.annotation) for i in plan.inputs}
    cents = {i.name for i in plan.inputs if _is_cents(i.annotation)}
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
        if t is float:
            edges |= {f"tie:{v}": v for v in TIES}
        if name in cents:
            edges |= {f"large:{k}": v for k, v in LARGE_CENTS.items()}
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


def _is_cents(annotation: Any) -> bool:
    candidates = [annotation, *typing.get_args(annotation)]
    return any((m := metadata_of(a)) is not None and isinstance(m, Money) and m.cents for a in candidates)
