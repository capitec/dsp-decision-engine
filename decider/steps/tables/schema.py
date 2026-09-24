"""A decision table's conditions, in decider 0.3's vocabulary, and the checks its rows must pass."""
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

    Rows that share their `eq` values (say every row of one grade)
    form one ladder of bands, checked on its own. In a ladder, a row's
    missing lower bound is the previous row's upper and a missing upper the
    next row's lower; the first row may leave its lower edge open (`None`)
    and the last its upper. Unless `allow_gaps`, each row's upper must equal
    the next row's lower in its ladder.
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


def groups(expression: t.Any, rows: t.Sequence[t.Mapping[str, t.Any]]) -> list[list[int]]:
    """Row numbers grouped by their `eq` column values, in row order: each group is one band ladder."""
    keys = [leaf.value_column for leaf in leaves(expression) if isinstance(leaf, EqExpression)]
    found: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        found.setdefault(tuple(row.get(k) for k in keys), []).append(i)
    return list(found.values())


def bounds(leaf: BetweenExpression, rows: t.Sequence[t.Mapping[str, t.Any]],
           ladders: t.Sequence[t.Sequence[int]]) -> list[tuple[t.Any, t.Any]]:
    """Each row's `(lower, upper)`, a missing one taken from the neighbouring row of its group; `None` is open."""
    resolved: list[tuple[t.Any, t.Any]] = [(None, None)] * len(rows)
    for ladder in ladders:
        n = len(ladder)
        lower = [rows[i].get(leaf.lower_bound_column) if leaf.lower_bound_column else None for i in ladder]
        upper = [rows[i].get(leaf.upper_bound_column) if leaf.upper_bound_column else None for i in ladder]
        for k, i in enumerate(ladder):
            resolved[i] = (lower[k] if lower[k] is not None else upper[k - 1] if k > 0 else None,
                           upper[k] if upper[k] is not None else lower[k + 1] if k < n - 1 else None)
    return resolved


def check_rows(expression: t.Any, rows: t.Sequence[t.Mapping[str, t.Any]]) -> None:
    """Raise `ValueError` if a `between` band of `rows` can't be resolved or, without `allow_gaps`, leaves a gap.

    Rows with the same `eq` values form one ladder: its first row may
    leave its lower edge open (`None`), its last row its upper.
    """
    ladders = groups(expression, rows)
    for leaf in leaves(expression):
        if not isinstance(leaf, BetweenExpression):
            continue
        resolved = bounds(leaf, rows, ladders)
        for ladder in ladders:
            first, last = ladder[0], ladder[-1]
            of = "" if len(ladders) == 1 else " of its group (rows sharing its eq values)"
            for k, i in enumerate(ladder):
                lo, hi = resolved[i]
                if lo is None and hi is None:
                    raise ValueError(f"Row {i}: both bounds are unresolvable; give it a lower or an upper bound. "
                                     f"Only the first row{of} (row {first}) may leave its lower bound None and "
                                     f"only the last (row {last}) its upper.")
                if lo is None and i != first:
                    raise ValueError(f"Row {i}: lower bound unresolvable; only the first row{of} (row {first}) "
                                     "may have an open lower edge.")
                if hi is None and i != last:
                    raise ValueError(f"Row {i}: upper bound unresolvable; only the last row{of} (row {last}) "
                                     "may have an open upper edge.")
                if not leaf.allow_gaps and i != last:
                    after = ladder[k + 1]
                    following = rows[after].get(leaf.lower_bound_column) if leaf.lower_bound_column else None
                    following = hi if following is None else following
                    if hi is not None and hi != following:
                        raise ValueError(f"Row {i} upper ({hi}) != row {after} lower ({following}): ranges are "
                                         "not contiguous. Set allow_gaps=True to permit this.")
