"""The decision-table document — decider 1's vocabulary, kept as-is.

Every type name, field name and default below is decider 1's, from
`decider/modules/credit/decision_table/config.py` and `module.py`:
`AndExpression`/`OrExpression`/`BetweenExpression`/`InExpression`/
`IsTrueExpression`/`EqExpression`, `variable`, `lower_bound_column`,
`upper_bound_column`, `values_column`, `value_column`, `mode`, `allow_gaps`,
`outputs`, `default`. A `DecisionTableModule` config that decider 1 accepts
parses here with only its wrapper renamed.

Two divergences:

1. **`parameters` is rows + dtypes, not a polars `DataFrame` subclass.**
   decider 1's `ParametersConfig` extends `decider.serializable.dataframe.
   DataFrame`, which drags polars into the document type. Here the document
   is plain data (`rows: list[dict]`, `dtypes: list[tuple[str, str]]`) —
   which is the same JSON decider 1's `DataFrame` serialises to, and keeps
   the schema importable without the heavy stack (doc 02 §2, "the graph is
   data"). The `data=`/`dtypes=` spelling decider 1's tests use is accepted
   verbatim.

2. **`BoundMode` *is* `decider2.trees.schema.RangeEndLogic`.** decider 1 has
   two identical two-value enums under two names — `BoundMode` in the
   decision table, `RangeEndLogic` in the tree — with the same members and
   the same rationale. They are aliased here so a fix to one cannot leave
   the other behind. decider 1's own comment, kept because it is the reason
   the enum has exactly two members:

       ``both_inclusive`` / ``both_exclusive`` are intentionally omitted:
       they create overlaps or gaps at shared boundaries, making contiguous
       range tables ambiguous.
"""
from __future__ import annotations

import typing as t

from pydantic import BaseModel, Field, model_validator

from decider2.trees.schema import RangeEndLogic

__all__ = [
    "BoundMode",
    "ParametersConfig",
    "AndExpression",
    "OrExpression",
    "BetweenExpression",
    "InExpression",
    "IsTrueExpression",
    "EqExpression",
    "Expression",
    "DecisionTable",
]

BoundMode = RangeEndLogic
"""decider 1's `BoundMode`, aliased onto the tree's `RangeEndLogic`.

    lower_inclusive  ->  [lower, upper)   i.e.  lower <= x < upper
    upper_inclusive  ->  (lower, upper]   i.e.  lower <  x <= upper
"""


class ParametersConfig(BaseModel):
    """The table itself: N rows of bound/value/output columns.

    decider 1's `ParametersConfig(data=..., dtypes=...)`, same constructor
    keywords, without the polars `DataFrame` base class.
    """

    data: t.List[t.Dict[str, t.Any]] = Field(default_factory=list)
    dtypes: t.Union[t.Dict[str, str], t.List[t.Tuple[str, t.Any]]] = Field(
        default_factory=dict
    )

    @property
    def columns(self) -> list[str]:
        if isinstance(self.dtypes, dict):
            return list(self.dtypes)
        return [name for name, _ in self.dtypes]

    @property
    def dtype_map(self) -> dict[str, t.Any]:
        if isinstance(self.dtypes, dict):
            return dict(self.dtypes)
        return {name: dt for name, dt in self.dtypes}

    def column(self, name: str) -> list[t.Any]:
        return [row.get(name) for row in self.data]

    def __len__(self) -> int:
        return len(self.data)


class _BaseExpression(BaseModel):
    """decider 1's `BaseExpression` surface, minus the polars half.

    `get_variables()` is kept by name because it is the method decider 1's
    `DecisionTableModule.expand_nodes` calls to discover what the table
    reads; here it is what the generated kernel's signature is built from.
    """

    def get_variables(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def validate_parameters(self, parameters: ParametersConfig) -> None:  # pragma: no cover
        raise NotImplementedError


def _dedup(names: t.Iterable[str]) -> list[str]:
    """Order-preserving de-dup — decider 1's `seen`/`result` walk in
    `AndExpression.get_variables`, and the determinism doc 05 §4.2 needs."""
    return list(dict.fromkeys(names))


class AndExpression(_BaseExpression):
    type: t.Literal["and"]
    expressions: t.List["Expression"]

    def get_variables(self) -> list[str]:
        return _dedup(v for e in self.expressions for v in e.get_variables())

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        for e in self.expressions:
            e.validate_parameters(parameters)


class OrExpression(_BaseExpression):
    type: t.Literal["or"]
    expressions: t.List["Expression"]

    def get_variables(self) -> list[str]:
        return _dedup(v for e in self.expressions for v in e.get_variables())

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        for e in self.expressions:
            e.validate_parameters(parameters)


class BetweenExpression(_BaseExpression):
    """A banded lookup — decider 1's `BetweenExpression`, same fields.

    The bound-resolution rule is decider 1's and is reproduced exactly,
    because it is load-bearing for real contiguous tables: a row's missing
    lower bound is taken from the *previous* row's upper, and a missing
    upper from the *next* row's lower. Only row 0 may have an open lower
    edge and only the last row an open upper edge; `allow_gaps=False` (the
    default) additionally requires each row's upper to equal the next row's
    lower.
    """

    type: t.Literal["between"]
    variable: str
    lower_bound_column: t.Optional[str] = None
    upper_bound_column: t.Optional[str] = None
    mode: BoundMode = BoundMode.lower_inclusive
    allow_gaps: bool = False

    def get_variables(self) -> list[str]:
        return [self.variable]

    def resolved_bounds(
        self, parameters: ParametersConfig
    ) -> list[tuple[t.Optional[float], t.Optional[float]]]:
        """decider 1's `_safe_list_get` neighbour fill, verbatim."""
        n = len(parameters)
        lower = parameters.column(self.lower_bound_column) if self.lower_bound_column else [None] * n
        upper = parameters.column(self.upper_bound_column) if self.upper_bound_column else [None] * n

        def at(lst: list, i: int) -> t.Any:
            return lst[i] if 0 <= i < len(lst) else None

        out = []
        for i in range(n):
            lo = lower[i] if lower[i] is not None else at(upper, i - 1)
            hi = upper[i] if upper[i] is not None else at(lower, i + 1)
            if lo is None and hi is None:
                raise ValueError(f"Row {i} has no lower or upper bound after resolution")
            out.append((lo, hi))
        return out

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        if not self.lower_bound_column and not self.upper_bound_column:
            raise ValueError(
                "At least one of lower_bound_column or upper_bound_column must be specified"
            )
        columns = parameters.columns
        for col, label in ((self.lower_bound_column, "Lower"), (self.upper_bound_column, "Upper")):
            if col is not None and col not in columns:
                raise ValueError(f"{label} bound column '{col}' not found in parameters columns")

        n = len(parameters)
        lower = parameters.column(self.lower_bound_column) if self.lower_bound_column else [None] * n
        upper = parameters.column(self.upper_bound_column) if self.upper_bound_column else [None] * n

        def at(lst: list, i: int) -> t.Any:
            return lst[i] if 0 <= i < len(lst) else None

        for i in range(n):
            lo = lower[i] if lower[i] is not None else at(upper, i - 1)
            hi = upper[i] if upper[i] is not None else at(lower, i + 1)
            if lo is None and hi is None:
                raise ValueError(
                    f"Row {i}: both bounds are unresolvable. Only row 0's lower and the "
                    "last row's upper may be None (open edges)."
                )
            if lo is None and i > 0:
                raise ValueError(
                    f"Row {i}: lower bound unresolvable — only row 0 may have an open lower edge."
                )
            if hi is None and i < n - 1:
                raise ValueError(
                    f"Row {i}: upper bound unresolvable — only row {n - 1} may have an "
                    "open upper edge."
                )
            if not self.allow_gaps and i < n - 1:
                next_lower = lower[i + 1] if lower[i + 1] is not None else hi
                if hi is not None and next_lower is not None and hi != next_lower:
                    raise ValueError(
                        f"Row {i} upper ({hi}) != row {i + 1} lower ({next_lower}): ranges "
                        "are not contiguous. Set allow_gaps=True to permit this."
                    )


class InExpression(_BaseExpression):
    """Set membership against a per-row list column — decider 1's
    `InExpression`."""

    type: t.Literal["in"]
    variable: str
    values_column: str

    def get_variables(self) -> list[str]:
        return [self.variable]

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        if self.values_column not in parameters.columns:
            raise ValueError(
                f"Values column '{self.values_column}' not found in parameters columns"
            )
        for i, row in enumerate(parameters.data):
            value = row.get(self.values_column)
            if value is not None and not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"Values column '{self.values_column}' must be a list type, row {i} "
                    f"holds {type(value).__name__}"
                )


class IsTrueExpression(_BaseExpression):
    """A boolean gate that consults no table column — decider 1's
    `IsTrueExpression` returns `[expr] * len(parameters)`, i.e. the same
    condition for every row."""

    type: t.Literal["is_true"]
    variable: str

    def get_variables(self) -> list[str]:
        return [self.variable]

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        return None


class EqExpression(_BaseExpression):
    """Equality against a per-row scalar column — decider 1's
    `EqExpression`, including its docstring's example config::

        {"type": "eq", "variable": "BureauKey", "value_column": "key"}
    """

    type: t.Literal["eq"]
    variable: str
    value_column: str

    def get_variables(self) -> list[str]:
        return [self.variable]

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        if self.value_column not in parameters.columns:
            raise ValueError(
                f"Value column '{self.value_column}' not found in parameters columns"
            )


Expression = t.Annotated[
    t.Union[
        AndExpression,
        OrExpression,
        BetweenExpression,
        InExpression,
        IsTrueExpression,
        EqExpression,
    ],
    Field(discriminator="type"),
]

AndExpression.model_rebuild()
OrExpression.model_rebuild()


class DecisionTable(BaseModel):
    """decider 1's `DecisionTableModule`, as a document.

    Same fields: `parameters`, `expression`, `outputs`, `default`. The
    `type`/`unnest_output` fields are dropped — `type` discriminated
    decider 1's module registry, which decider2 does not have, and
    `unnest_output` existed because decider 1 returned a struct column;
    decider2's output frame is flat by construction (doc 03 §7), so every
    output column is already a top-level column.

    First match wins, and a row that matches nothing takes `default` —
    decider 1's `calculate_decision_table_output` chains `when/then` in row
    order and closes with `otherwise(default)`.
    """

    name: str = "decision_table"
    parameters: ParametersConfig
    expression: Expression
    outputs: t.List[str]
    default: t.Optional[t.List[t.Any]] = None

    @model_validator(mode="after")
    def validate_config(self) -> "DecisionTable":
        for output in self.outputs:
            if output not in self.parameters.columns:
                raise ValueError(
                    f"Output column '{output}' not found in parameters columns"
                )
        if self.default is not None and len(self.default) != len(self.outputs):
            raise ValueError(
                f"Default values length ({len(self.default)}) must match outputs "
                f"length ({len(self.outputs)})"
            )
        self.expression.validate_parameters(self.parameters)
        return self

    def default_for(self, column: str) -> t.Any:
        if self.default is None:
            return None
        return self.default[self.outputs.index(column)]
