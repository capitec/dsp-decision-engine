"""The Python decision-table matcher: the compiled matcher's twin, reporting every row it tries."""
from __future__ import annotations

import operator
import typing as t

from decider.steps.tables.schema import (
    AndExpression,
    BetweenExpression,
    BoundMode,
    EqExpression,
    InExpression,
    OrExpression,
)


def _keys(e: t.Any) -> list[str]:
    if isinstance(e, (AndExpression, OrExpression)):
        return [k for sub in e.expressions for k in _keys(sub)]
    return [e.value_column] if isinstance(e, EqExpression) else []


def _band(e: BetweenExpression, rows: list[dict], r: int, keys: list[str]) -> tuple[t.Any, t.Any]:
    # A missing bound is the opposite one of the nearest row with the same eq values.
    lower, upper = e.lower_bound_column, e.upper_bound_column
    lo = rows[r][lower] if lower else None
    hi = rows[r][upper] if upper else None
    if lo is None and upper and (q := _nearest(rows, r, -1, keys)) is not None:
        lo = rows[q][upper]
    if hi is None and lower and (q := _nearest(rows, r, 1, keys)) is not None:
        hi = rows[q][lower]
    return lo, hi


def _nearest(rows: list[dict], r: int, step: int, keys: list[str]) -> t.Optional[int]:
    q = r + step
    while 0 <= q < len(rows) and any(rows[q][k] != rows[r][k] for k in keys):
        q += step
    return q if 0 <= q < len(rows) else None


def _holds(e: t.Any, rows: list[dict], r: int, values: dict[str, t.Any], keys: list[str]) -> bool:
    # A comparison with a null is null in decider 0.3, which never matches.
    if isinstance(e, AndExpression):
        return all(_holds(sub, rows, r, values, keys) for sub in e.expressions)
    if isinstance(e, OrExpression):
        return any(_holds(sub, rows, r, values, keys) for sub in e.expressions)
    x = values[e.variable]
    # A null number reaches the matcher as NaN.
    if x is None or x != x:
        return False
    if isinstance(e, BetweenExpression):
        lo, hi = _band(e, rows, r, keys)
        low, high = ((operator.ge, operator.lt) if e.mode is BoundMode.lower_inclusive else (operator.gt, operator.le))
        return (lo is None or low(x, lo)) and (hi is None or high(x, hi))
    if isinstance(e, EqExpression):
        v = rows[r][e.value_column]
        return v is not None and x == v
    if isinstance(e, InExpression):
        return x in [v for v in rows[r][e.values_column] or () if v is not None]
    return bool(x)


def reference(expression: t.Any, names: t.Sequence[str], outputs: t.Sequence[str], default: t.Sequence[t.Any],
              table: t.Optional[str], rows: t.Any) -> t.Callable:
    """`reference(row, params, consts, visit)`: the table matched in Python, calling `visit(str(r))` per row tried.

    `rows` holds the inline rows' `Rows`, or `table` names the param they come
    from. Returns one value per output: the first matching row's, else the default.

    Example::

        match = reference(expr, ["score"], ["band"], ["other"], None, rows)
        match((42.0,), (), (), print)    # prints "0", "1" and returns ("mid",)
    """
    keys = _keys(expression)

    def match(row: tuple, params: t.Any, consts: tuple, visit: t.Callable[[str], None]) -> tuple:
        values = dict(zip(names, row))
        data = (getattr(params, table) if table else rows).data
        for r, found in enumerate(data):
            visit(str(r))
            if _holds(expression, data, r, values, keys):
                return tuple(found[c] for c in outputs)
        return tuple(default)

    return match
