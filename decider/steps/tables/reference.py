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


def _band(e: BetweenExpression, rows: list[dict], r: int) -> tuple[t.Any, t.Any]:
    # A missing bound is the neighbouring row's opposite one.
    lower, upper = e.lower_bound_column, e.upper_bound_column
    lo = rows[r][lower] if lower else None
    hi = rows[r][upper] if upper else None
    if lo is None and r > 0 and upper:
        lo = rows[r - 1][upper]
    if hi is None and r < len(rows) - 1 and lower:
        hi = rows[r + 1][lower]
    return lo, hi


def _holds(e: t.Any, rows: list[dict], r: int, values: dict[str, t.Any]) -> bool:
    # A comparison with a null is null in decider_old, which never matches.
    if isinstance(e, AndExpression):
        return all(_holds(sub, rows, r, values) for sub in e.expressions)
    if isinstance(e, OrExpression):
        return any(_holds(sub, rows, r, values) for sub in e.expressions)
    x = values[e.variable]
    # A null number reaches the matcher as NaN.
    if x is None or x != x:
        return False
    if isinstance(e, BetweenExpression):
        lo, hi = _band(e, rows, r)
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
    def match(row: tuple, params: t.Any, consts: tuple, visit: t.Callable[[str], None]) -> tuple:
        values = dict(zip(names, row))
        data = (getattr(params, table) if table else rows).data
        for r, found in enumerate(data):
            visit(str(r))
            if _holds(expression, data, r, values):
                return tuple(found[c] for c in outputs)
        return tuple(default)

    return match
