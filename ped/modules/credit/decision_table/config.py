import typing as t
import polars as pl
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from ped.exceptions import wrap_import_errors
from ped._ext import TypeDiscriminatedBaseModule


if t.TYPE_CHECKING:
    import pandera.pandas as pa

class ParametersConfig(BaseModel):
    columns: t.List[str]
    values: t.List[t.List[t.Any]]
    dtypes: t.Dict[str, str]
    
    _parameters_df: pl.DataFrame = PrivateAttr()
    _pandera_schema: "pa.DataFrameSchema" = PrivateAttr()
    
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
        
        # Create Pandera schema and validate
        schema_dict = {}
        for col, dtype_str in self.dtypes.items():
            if dtype_str == "float":
                schema_dict[col] = pa.Column(float)
            elif dtype_str == "int":
                schema_dict[col] = pa.Column(int)
            elif dtype_str == "string":
                schema_dict[col] = pa.Column(str)
            elif dtype_str.startswith("list"):
                schema_dict[col] = pa.Column(object)  # Lists stored as objects
            else:
                raise ValueError(f"Unsupported dtype '{dtype_str}' for column '{col}'")
        
        self._pandera_schema = pa.DataFrameSchema(schema_dict)
        
        # Convert to pandas for pandera validation, then back to polars
        pandas_df = self._parameters_df.to_pandas()
        try:
            validated_df = self._pandera_schema.validate(pandas_df)
            self._parameters_df = pl.from_pandas(validated_df)
        except pa.errors.SchemaError as e:
            raise ValueError(f"Schema validation failed: {e}")
        
        return self


class BaseExpression(BaseModel):
    """Base class for all decision table expressions"""
    
    def __call__(self, **kwargs: pl.Expr) -> pl.Expr:
        """Evaluate the expression with given polars expressions"""
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
    
    def __call__(self, **kwargs: pl.Expr) -> pl.Expr:
        expr_results = [expr(**kwargs) for expr in self.expressions]
        return pl.all_horizontal(*expr_results)
    
    def validate_parameters(self, parameters: t.Dict[str, t.Any]) -> None:
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
    
    def __call__(self, **kwargs: pl.Expr) -> pl.Expr:
        expr_results = [expr(**kwargs) for expr in self.expressions]
        return pl.any_horizontal(*expr_results)
    
    def validate_parameters(self, parameters: t.Dict[str, t.Any]) -> None:
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
    mode: t.Literal["lower_inclusive", "upper_inclusive", "both_inclusive", "both_exclusive"] = "both_inclusive"
    
    def __call__(self, **kwargs: pl.Expr) -> pl.Expr:
        if self.variable not in kwargs:
            raise ValueError(f"Variable '{self.variable}' not found in expression arguments")
        
        var_expr = kwargs[self.variable]
        conditions = []
        
        if self.lower_bound_column:
            lower_bound = kwargs.get(self.lower_bound_column)
            if lower_bound is None:
                raise ValueError(f"Lower bound column '{self.lower_bound_column}' not found in expression arguments")
            
            if self.mode in ["lower_inclusive", "both_inclusive"]:
                conditions.append(var_expr >= lower_bound)
            else:
                conditions.append(var_expr > lower_bound)
        
        if self.upper_bound_column:
            upper_bound = kwargs.get(self.upper_bound_column)
            if upper_bound is None:
                raise ValueError(f"Upper bound column '{self.upper_bound_column}' not found in expression arguments")
            
            if self.mode in ["upper_inclusive", "both_inclusive"]:
                conditions.append(var_expr <= upper_bound)
            else:
                conditions.append(var_expr < upper_bound)
        
        if not conditions:
            raise ValueError("At least one of lower_bound_column or upper_bound_column must be specified")
        
        return pl.all_horizontal(*conditions) if len(conditions) > 1 else conditions[0]
    
    def validate_parameters(self, parameters: "ParametersConfig") -> None:
        # Note: self.variable is an input variable, not a column in parameters
        # We only need to validate that the bound columns exist
        
        # Check that bound columns exist and are numeric types
        if self.lower_bound_column:
            if self.lower_bound_column not in parameters.columns:
                raise ValueError(f"Lower bound column '{self.lower_bound_column}' not found in parameters columns")
            
            lower_dtype = parameters.dtypes.get(self.lower_bound_column)
            if lower_dtype not in ["float", "int"]:
                raise ValueError(f"Lower bound column '{self.lower_bound_column}' must be numeric type, got {lower_dtype}")
        
        if self.upper_bound_column:
            if self.upper_bound_column not in parameters.columns:
                raise ValueError(f"Upper bound column '{self.upper_bound_column}' not found in parameters columns")
            
            upper_dtype = parameters.dtypes.get(self.upper_bound_column)
            if upper_dtype not in ["float", "int"]:
                raise ValueError(f"Upper bound column '{self.upper_bound_column}' must be numeric type, got {upper_dtype}")
        
        if not self.lower_bound_column and not self.upper_bound_column:
            raise ValueError("At least one of lower_bound_column or upper_bound_column must be specified")

    def get_variables(self) -> t.List[str]:
        return [self.variable]


class InExpression(TypeDiscriminatedBaseModule):
    type: t.Literal["in"]
    variable: str
    values_column: str
    
    def __call__(self, **kwargs: pl.Expr) -> pl.Expr:
        if self.variable not in kwargs:
            raise ValueError(f"Variable '{self.variable}' not found in expression arguments")
        if self.values_column not in kwargs:
            raise ValueError(f"Values column '{self.values_column}' not found in expression arguments")
        
        var_expr = kwargs[self.variable]
        values_expr = kwargs[self.values_column]
        
        return var_expr.is_in(values_expr)
    
    def validate_parameters(self, parameters: "ParametersConfig") -> None:
        # Note: self.variable is an input variable, not a column in parameters
        # We only need to validate that the values column exists
        
        # Check that values column exists and is a list type
        if self.values_column not in parameters.columns:
            raise ValueError(f"Values column '{self.values_column}' not found in parameters columns")
        
        values_dtype = parameters.dtypes.get(self.values_column)
        if not (values_dtype and values_dtype.startswith("list")):
            raise ValueError(f"Values column '{self.values_column}' must be a list type, got {values_dtype}")

    def get_variables(self) -> t.List[str]:
        return [self.variable]


class IsTrueExpression(TypeDiscriminatedBaseModule):
    type: t.Literal["is_true"]
    variable: str
    
    def __call__(self, **kwargs: pl.Expr) -> pl.Expr:
        if self.variable not in kwargs:
            raise ValueError(f"Variable '{self.variable}' not found in expression arguments")
        
        return kwargs[self.variable]
    
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