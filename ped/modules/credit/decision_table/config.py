import ast
import typing as t
import functools
from dataclasses import dataclass
from enum import Enum
import polars as pl
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from ped.serializable.dataframe import DataFrame
from ped.exceptions import wrap_import_errors
from ped._ext import TypeDiscriminatedBaseModule



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
    
    def validate_parameters(self, parameters: DataFrame) -> None:
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

    def validate_parameters(self, parameters: DataFrame) -> None:
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

    def validate_parameters(self, parameters: DataFrame) -> None:
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

    def validate_parameters(self, parameters: DataFrame) -> None:
        if not self.lower_bound_column and not self.upper_bound_column:
            raise ValueError("At least one of lower_bound_column or upper_bound_column must be specified")

        df = parameters.df
        columns = df.columns
        for col_attr, label in (
            (self.lower_bound_column, "Lower"),
            (self.upper_bound_column, "Upper"),
        ):
            if col_attr is None:
                continue
            if col_attr not in columns:
                raise ValueError(f"{label} bound column '{col_attr}' not found in parameters columns")
            dtype = df[col_attr].dtype
            if dtype not in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                raise ValueError(f"{label} bound column '{col_attr}' must be numeric, got {dtype}")

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
    
    def validate_parameters(self, parameters: DataFrame) -> None:
        # Note: self.variable is an input variable, not a column in parameters
        # We only need to validate that the values column exists
        
        # Check that values column exists and is a list type
        df = parameters.df
        if self.values_column not in df.columns:
            raise ValueError(f"Values column '{self.values_column}' not found in parameters columns")
        
        values_dtype = df[self.values_column].dtype
        if not isinstance(values_dtype, pl.List):
            raise ValueError(f"Values column '{self.values_column}' must be a list type, got {values_dtype}")

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
    
    def validate_parameters(self, parameters: DataFrame) -> None:  # noqa: ARG002
        # Note: self.variable is an input variable, not a column in parameters
        # No validation needed for IsTrueExpression beyond basic structure
        pass

    def get_variables(self) -> t.List[str]:
        return [self.variable]


Expression = t.Annotated[
    t.Union[
        AndExpression,
        OrExpression,
        BetweenExpression,
        InExpression,
        IsTrueExpression,
    ],
    Field(discriminator="type")
]

# Update forward references
AndExpression.model_rebuild()
OrExpression.model_rebuild()