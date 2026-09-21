import typing as t
from enum import Enum
import polars as pl
from pydantic import BaseModel, Field
from decider._ext import TypeDiscriminatedBaseModule
from decider.serializable.dataframe import DataFrame


class ParametersConfig(DataFrame):
    @property
    def columns(self) -> t.List[str]:
        return self.df.columns


def _safe_list_get(lst: t.List, i: int) -> t.Optional[t.Any]:
    """Return lst[i], or None when i is out of range."""
    return lst[i] if 0 <= i < len(lst) else None


class BoundMode(str, Enum):
    """Interval convention for BetweenExpression.

    lower_inclusive  →  [lower, upper)   i.e.  lower <= x < upper
    upper_inclusive  →  (lower, upper]   i.e.  lower <  x <= upper

    ``both_inclusive`` / ``both_exclusive`` are intentionally omitted:
    they create overlaps or gaps at shared boundaries, making contiguous
    range tables ambiguous.
    """
    lower_inclusive = "lower_inclusive"
    upper_inclusive = "upper_inclusive"


class BaseExpression(BaseModel):
    """Base class for all decision table expressions"""

    def __call__(self, parameters: pl.DataFrame, **kwargs: pl.Expr) -> t.List[pl.Expr]:
        """Return one boolean pl.Expr per row of *parameters*.

        Each element is a scalar-broadcast expression that evaluates to True when
        the corresponding parameter row matches the input variables in **kwargs.
        """
        raise NotImplementedError
    
    def validate_parameters(self, parameters: "ParametersConfig") -> None:
        """Validate that required variables exist in parameters with correct types"""
        raise NotImplementedError

    def get_variables(self) -> t.List[str]:
        """Return the list of external input variable names this expression consumes."""
        raise NotImplementedError


class AndExpression(TypeDiscriminatedBaseModule):
    type: t.Literal["and"]
    expressions: t.List["Expression"]
    
    def __call__(self, parameters: pl.DataFrame, **kwargs: pl.Expr) -> t.List[pl.Expr]:
        # Each child returns List[pl.Expr] of length len(parameters).
        # AND: zip-combine per row with all_horizontal.
        per_child = [expr(parameters, **kwargs) for expr in self.expressions]
        return [
            pl.all_horizontal(*row_conds)
            for row_conds in zip(*per_child)
        ]

    def validate_parameters(self, parameters: "ParametersConfig") -> None:
        for expr in self.expressions:
            expr.validate_parameters(parameters)

    def get_variables(self) -> t.List[str]:
        seen = set()
        result = []
        for expr in self.expressions:
            for v in expr.get_variables():
                if v not in seen:
                    seen.add(v)
                    result.append(v)
        return result


class OrExpression(TypeDiscriminatedBaseModule):
    type: t.Literal["or"]
    expressions: t.List["Expression"]
    
    def __call__(self, parameters: pl.DataFrame, **kwargs: pl.Expr) -> t.List[pl.Expr]:
        per_child = [expr(parameters, **kwargs) for expr in self.expressions]
        return [
            pl.any_horizontal(*row_conds)
            for row_conds in zip(*per_child)
        ]

    def validate_parameters(self, parameters: "ParametersConfig") -> None:
        for expr in self.expressions:
            expr.validate_parameters(parameters)

    def get_variables(self) -> t.List[str]:
        seen = set()
        result = []
        for expr in self.expressions:
            for v in expr.get_variables():
                if v not in seen:
                    seen.add(v)
                    result.append(v)
        return result


class BetweenExpression(TypeDiscriminatedBaseModule):
    type: t.Literal["between"]
    variable: str
    lower_bound_column: t.Optional[str] = None
    upper_bound_column: t.Optional[str] = None
    mode: BoundMode = BoundMode.lower_inclusive
    allow_gaps: bool = False

    def __call__(self, parameters: pl.DataFrame, **kwargs: pl.Expr) -> t.List[pl.Expr]:
        if self.variable not in kwargs:
            raise ValueError(f"Variable '{self.variable}' not found in expression arguments")

        var_expr = kwargs[self.variable]
        n = len(parameters)
        lower_vals = parameters[self.lower_bound_column].to_list() if self.lower_bound_column else [None] * n
        upper_vals = parameters[self.upper_bound_column].to_list() if self.upper_bound_column else [None] * n
        row_conditions: t.List[pl.Expr] = []

        for i, (lower, upper) in enumerate(zip(lower_vals, upper_vals)):
            lower = lower if lower is not None else _safe_list_get(upper_vals, i - 1)
            upper = upper if upper is not None else _safe_list_get(lower_vals, i + 1)
            if lower is None and upper is None:
                raise ValueError(f"Row {i} has no lower or upper bound after resolution")
            conditions: t.List[pl.Expr] = []
            if lower is not None:
                lb = pl.lit(lower)
                conditions.append(var_expr >= lb if self.mode == BoundMode.lower_inclusive else var_expr > lb)
            if upper is not None:
                ub = pl.lit(upper)
                conditions.append(var_expr < ub if self.mode == BoundMode.lower_inclusive else var_expr <= ub)
            row_conditions.append(
                pl.all_horizontal(*conditions) if len(conditions) > 1 else conditions[0]
            )

        return row_conditions

    def validate_parameters(self, parameters: "ParametersConfig") -> None:
        if not self.lower_bound_column and not self.upper_bound_column:
            raise ValueError("At least one of lower_bound_column or upper_bound_column must be specified")

        for col_attr, label in (
            (self.lower_bound_column, "Lower"),
            (self.upper_bound_column, "Upper"),
        ):
            if col_attr is None:
                continue
            if col_attr not in parameters.columns:
                raise ValueError(f"{label} bound column '{col_attr}' not found in parameters columns")
            pl_dtype = parameters.df.schema[col_attr]
            if not pl_dtype.is_numeric():
                raise ValueError(f"{label} bound column '{col_attr}' must be numeric, got {pl_dtype}")

        df = parameters.df
        n = len(df)
        lower_vals = df[self.lower_bound_column].to_list() if self.lower_bound_column else [None] * n
        upper_vals = df[self.upper_bound_column].to_list() if self.upper_bound_column else [None] * n

        for i, (lower, upper) in enumerate(zip(lower_vals, upper_vals)):
            lower = lower if lower is not None else _safe_list_get(upper_vals, i - 1)
            upper = upper if upper is not None else _safe_list_get(lower_vals, i + 1)
            if lower is None and upper is None:
                raise ValueError(
                    f"Row {i}: both bounds are unresolvable. "
                    f"Only row 0's lower and the last row's upper may be None (open edges)."
                )
            if lower is None and i > 0:
                raise ValueError(f"Row {i}: lower bound unresolvable — only row 0 may have an open lower edge.")
            if upper is None and i < n - 1:
                raise ValueError(f"Row {i}: upper bound unresolvable — only row {n-1} may have an open upper edge.")
            if not self.allow_gaps and i < n - 1:
                next_lower = lower_vals[i + 1] if lower_vals[i + 1] is not None else upper
                if upper is not None and next_lower is not None and upper != next_lower:
                    raise ValueError(
                        f"Row {i} upper ({upper}) != row {i+1} lower ({next_lower}): "
                        f"ranges are not contiguous. Set allow_gaps=True to permit this."
                    )

    def get_variables(self) -> t.List[str]:
        return [self.variable]


class InExpression(TypeDiscriminatedBaseModule):
    type: t.Literal["in"]
    variable: str
    values_column: str
    
    def __call__(self, parameters: pl.DataFrame, **kwargs: pl.Expr) -> t.List[pl.Expr]:
        if self.variable not in kwargs:
            raise ValueError(f"Variable '{self.variable}' not found in expression arguments")

        var_expr = kwargs[self.variable]
        return [
            var_expr.is_in(pl.lit(row[self.values_column], dtype=pl.List(pl.Utf8)))
            for row in parameters.iter_rows(named=True)
        ]
    
    def validate_parameters(self, parameters: "ParametersConfig") -> None:
        # Note: self.variable is an input variable, not a column in parameters
        # We only need to validate that the values column exists
        
        # Check that values column exists and is a list type
        if self.values_column not in parameters.columns:
            raise ValueError(f"Values column '{self.values_column}' not found in parameters columns")
        
        pl_dtype = parameters.df.schema[self.values_column]
        if not isinstance(pl_dtype, pl.List):
            raise ValueError(f"Values column '{self.values_column}' must be a list type, got {pl_dtype}")

    def get_variables(self) -> t.List[str]:
        return [self.variable]


class IsTrueExpression(TypeDiscriminatedBaseModule):
    type: t.Literal["is_true"]
    variable: str

    def __call__(self, parameters: pl.DataFrame, **kwargs: pl.Expr) -> t.List[pl.Expr]:
        if self.variable not in kwargs:
            raise ValueError(f"Variable '{self.variable}' not found in expression arguments")
        expr = kwargs[self.variable]
        return [expr] * len(parameters)

    def validate_parameters(self, parameters: "ParametersConfig") -> None:  # noqa: ARG002
        pass

    def get_variables(self) -> t.List[str]:
        return [self.variable]


class EqExpression(TypeDiscriminatedBaseModule):
    """Match rows where the input variable equals the value in a parameters column.

    Supports any scalar type (numeric, string). Each row in the parameters table
    provides the value to compare against — the condition is True when the input
    variable equals that row's value exactly.

    Example config::

        {
          "type": "eq",
          "variable": "BureauKey",
          "value_column": "key"
        }
    """

    type: t.Literal["eq"]
    variable: str
    value_column: str

    def __call__(self, parameters: pl.DataFrame, **kwargs: pl.Expr) -> t.List[pl.Expr]:
        if self.variable not in kwargs:
            raise ValueError(f"Variable '{self.variable}' not found in expression arguments")
        var_expr = kwargs[self.variable]
        return [
            var_expr == pl.lit(row[self.value_column])
            for row in parameters.iter_rows(named=True)
        ]

    def validate_parameters(self, parameters: "ParametersConfig") -> None:
        if self.value_column not in parameters.columns:
            raise ValueError(
                f"Value column '{self.value_column}' not found in parameters columns"
            )

    def get_variables(self) -> t.List[str]:
        return [self.variable]


Expression = t.Annotated[
    t.Union[
        AndExpression,
        OrExpression,
        BetweenExpression,
        InExpression,
        IsTrueExpression,
        EqExpression,
    ],
    Field(discriminator="type")
]

# Update forward references
AndExpression.model_rebuild()
OrExpression.model_rebuild()