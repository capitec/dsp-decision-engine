import ast
import typing as t
import functools
from dataclasses import dataclass
from enum import Enum
import polars as pl
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from ped.exceptions import wrap_import_errors
from ped._ext import TypeDiscriminatedBaseModule


if t.TYPE_CHECKING:
    import pandera.pandas as pa

_ALLOWED_NAMES: t.Dict[str, t.Any] = {
    "Optional": t.Optional,
    "List": t.List,
    "Dict": t.Dict,
    "str": str,
    "string": str,   # pandera-style alias
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
}


class _TypeEvaluator(ast.NodeVisitor):
    def visit_Name(self, node: ast.Name) -> t.Any:
        if node.id not in _ALLOWED_NAMES:
            raise ValueError(f"Disallowed name: {node.id!r}")
        return _ALLOWED_NAMES[node.id]

    def visit_Subscript(self, node: ast.Subscript) -> t.Any:
        base = self.visit(node.value)
        sub = self.visit(node.slice)
        return base[sub]

    def visit_Index(self, node: ast.Index) -> t.Any:  # Python < 3.9
        return self.visit(node.value)

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Constant(self, node: ast.Constant) -> t.Any:
        return node.value

    def generic_visit(self, node: ast.AST) -> t.Any:
        raise ValueError(f"Unsupported syntax: {ast.dump(node)}")


def _safe_eval_type(expr: str) -> t.Any:
    """Parse a type-expression string into a real type using AST (no eval)."""
    tree = ast.parse(expr.strip(), mode="eval")
    return _TypeEvaluator().visit(tree.body)


@dataclass(frozen=True)
class ParsedDtype:
    type: type
    optional: bool
    list_inner_type: t.Optional[type] = None  # set when type is list

    @property
    def is_list(self) -> bool:
        return self.type is list


@functools.lru_cache(maxsize=None)
def _parse_dtype(dtype_str: str) -> ParsedDtype:
    """Parse a dtype string into a ParsedDtype.

    Uses an AST walker with an allowlist of safe names — no eval().
    Supports ``int``, ``float``, ``str`` / ``string``, ``bool``,
    ``list`` / ``List[X]``, and ``Optional[X]`` variants of all the above.
    Results are cached so repeated calls for the same string are free.
    """
    result = _safe_eval_type(dtype_str)
    # Unwrap Optional[X]  ==  Union[X, None]
    optional = False
    if t.get_origin(result) is t.Union and type(None) in t.get_args(result):
        optional = True
        result = next(a for a in t.get_args(result) if a is not type(None))
    # Detect list / List[X]
    if result is list or t.get_origin(result) is list:
        args = t.get_args(result)
        inner = args[0] if args else None
        return ParsedDtype(type=list, optional=optional, list_inner_type=inner)
    return ParsedDtype(type=result, optional=optional)

class ParametersConfig(BaseModel):
    columns: t.List[str]
    values: t.List[t.List[t.Any]]
    dtypes: t.Dict[str, str]
    
    _parameters_df: pl.DataFrame = PrivateAttr()
    _pandera_schema: "pa.DataFrameSchema" = PrivateAttr()
    _parsed_dtypes: t.Dict[str, "ParsedDtype"] = PrivateAttr()
    
    @model_validator(mode='after')
    def validate_and_create_dataframe(self):
        with wrap_import_errors(optional_source="dt"):
            import pandera.pandas as pa
        # TODO we could make the pandera validation optional?
        # Validate that all dtypes correspond to existing columns
        for col in self.dtypes:
            if col not in self.columns:
                raise ValueError(f"Dtype specified for column '{col}' but column not found in parameters columns")
        
        # Validate that values matrix has correct dimensions
        if self.values:
            expected_cols = len(self.columns)
            for i, row in enumerate(self.values):
                if len(row) != expected_cols:
                    raise ValueError(f"Row {i} has {len(row)} values but {expected_cols} columns expected")
        
        # Create DataFrame from values
        df_dict = {}
        for i, col in enumerate(self.columns):
            df_dict[col] = [row[i] for row in self.values]
        
        self._parameters_df = pl.DataFrame(df_dict)
        
        # Parse dtypes once and cache them, then build the pandera schema
        parsed_dtypes: t.Dict[str, ParsedDtype] = {}
        schema_dict = {}
        for col, dtype_str in self.dtypes.items():
            try:
                parsed = _parse_dtype(dtype_str)
            except ValueError:
                raise ValueError(f"Unsupported dtype '{dtype_str}' for column '{col}'")
            parsed_dtypes[col] = parsed
            pa_dtype = pa.PythonList(parsed.list_inner_type) if parsed.is_list else parsed.type
            schema_dict[col] = pa.Column(pa_dtype, nullable=parsed.optional)
        
        self._parsed_dtypes = parsed_dtypes
        self._pandera_schema = pa.DataFrameSchema(schema_dict)
        
        # Convert to pandas for pandera validation, then back to polars
        pandas_df = self._parameters_df.to_pandas()
        try:
            validated_df = self._pandera_schema.validate(pandas_df)
            self._parameters_df = pl.from_pandas(validated_df)
        except pa.errors.SchemaError as e:
            raise ValueError(f"Schema validation failed: {e}")
        
        return self


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
            dtype = parameters._parsed_dtypes[col_attr]
            if dtype.type not in (float, int):
                raise ValueError(f"{label} bound column '{col_attr}' must be numeric, got {dtype.type}")

        df = parameters._parameters_df
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
        
        values_dtype = parameters._parsed_dtypes[self.values_column]
        if not values_dtype.is_list:
            raise ValueError(f"Values column '{self.values_column}' must be a list type, got {values_dtype.type}")

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