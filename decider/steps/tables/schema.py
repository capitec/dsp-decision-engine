"""A decision table's conditions, in decider_old's vocabulary, and the checks its rows must pass."""
from __future__ import annotations

import typing as t

from pydantic import BaseModel, Field, model_validator

from decider.steps.trees.schema import RangeEndLogic

BoundMode = RangeEndLogic
"""`lower_inclusive` is `[lower, upper)`, `upper_inclusive` is `(lower, upper]`."""


class AndExpression(BaseModel):
    """Every sub-expression holds for the row."""

    type: t.Literal["and"] = "and"
    expressions: t.List["Expression"]


class OrExpression(BaseModel):
    """Some sub-expression holds for the row."""

    type: t.Literal["or"] = "or"
    expressions: t.List["Expression"]


class BetweenExpression(BaseModel):
    """`variable` lies in the row's band.

    A row's missing lower bound is the previous row's upper, and a missing
    upper the next row's lower; only the first row may leave its lower edge
    open and only the last its upper. Unless `allow_gaps`, each row's upper
    must equal the next row's lower.
    """

    type: t.Literal["between"] = "between"
    variable: str
    lower_bound_column: t.Optional[str] = None
    upper_bound_column: t.Optional[str] = None
    mode: BoundMode = BoundMode.lower_inclusive
    allow_gaps: bool = False

    @model_validator(mode="after")
    def _a_bound(self) -> BetweenExpression:
        if not self.lower_bound_column and not self.upper_bound_column:
            raise ValueError("At least one of lower_bound_column or upper_bound_column must be specified")
        return self


class InExpression(BaseModel):
    """`variable` is one of the row's list in `values_column`."""

    type: t.Literal["in"] = "in"
    variable: str
    values_column: str


class IsTrueExpression(BaseModel):
    """The boolean `variable` is true; it reads no column, so it holds for every row or none."""

    type: t.Literal["is_true"] = "is_true"
    variable: str


class EqExpression(BaseModel):
    """`variable` equals the row's value in `value_column`.

    `{"type": "eq", "variable": "BureauKey", "value_column": "key"}`
    """

    type: t.Literal["eq"] = "eq"
    variable: str
    value_column: str


Expression = t.Annotated[
    t.Union[AndExpression, OrExpression, BetweenExpression, InExpression, IsTrueExpression, EqExpression],
    Field(discriminator="type"),
]
AndExpression.model_rebuild()
OrExpression.model_rebuild()

Leaf = t.Union[BetweenExpression, InExpression, IsTrueExpression, EqExpression]


def leaves(e: t.Any) -> t.Iterator[Leaf]:
    """Every condition of `e` that reads a variable, depth first."""
    if isinstance(e, (AndExpression, OrExpression)):
        for sub in e.expressions:
            yield from leaves(sub)
    else:
        yield e


def bounds(leaf: BetweenExpression, rows: t.Sequence[t.Mapping[str, t.Any]]) -> list[tuple[t.Any, t.Any]]:
    """Each row's `(lower, upper)`, a missing one taken from the neighbouring row; `None` is an open edge."""
    n = len(rows)
    lower = [r.get(leaf.lower_bound_column) if leaf.lower_bound_column else None for r in rows]
    upper = [r.get(leaf.upper_bound_column) if leaf.upper_bound_column else None for r in rows]
    return [(lower[i] if lower[i] is not None else upper[i - 1] if i > 0 else None,
             upper[i] if upper[i] is not None else lower[i + 1] if i < n - 1 else None) for i in range(n)]


def check_rows(expression: t.Any, rows: t.Sequence[t.Mapping[str, t.Any]]) -> None:
    """Raise `ValueError` if a `between` band of `rows` can't be resolved or, without `allow_gaps`, leaves a gap."""
    n = len(rows)
    for leaf in leaves(expression):
        if not isinstance(leaf, BetweenExpression):
            continue
        resolved = bounds(leaf, rows)
        for i, (lo, hi) in enumerate(resolved):
            if lo is None and hi is None:
                raise ValueError(f"Row {i}: both bounds are unresolvable. "
                                 "Only row 0's lower and the last row's upper may be None (open edges).")
            if lo is None and i > 0:
                raise ValueError(f"Row {i}: lower bound unresolvable — only row 0 may have an open lower edge.")
            if hi is None and i < n - 1:
                raise ValueError(f"Row {i}: upper bound unresolvable — only row {n - 1} may have an open upper edge.")
            if not leaf.allow_gaps and i < n - 1:
                following = rows[i + 1].get(leaf.lower_bound_column) if leaf.lower_bound_column else None
                following = hi if following is None else following
                if hi is not None and hi != following:
                    raise ValueError(f"Row {i} upper ({hi}) != row {i + 1} lower ({following}): "
                                     "ranges are not contiguous. Set allow_gaps=True to permit this.")
