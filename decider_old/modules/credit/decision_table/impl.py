import typing as t
import polars as pl
from .config import Expression


def default_form_output_struct_from_row(
    row_values: t.Dict[str, t.Any],
    output_columns: t.List[str]
) -> pl.Expr:
    """
    Create a struct from row values for specified output columns.
    
    Args:
        row_values: Dictionary mapping column names to values for the current row
        output_columns: List of column names to include in the output struct
        
    Returns:
        pl.Expr: A struct expression with the specified output columns
    """
    output_fields = []
    for col in output_columns:
        output_fields.append(pl.lit(row_values[col]).alias(col))
    return pl.struct(*output_fields)


def calculate_decision_table_output(
    parameters: pl.DataFrame,
    expression: Expression,
    output_columns: t.List[str],
    default: t.Optional[t.List[t.Any]] = None,
    output_fn: t.Callable[[t.Dict[str, t.Any], t.List[str]], pl.Expr] = default_form_output_struct_from_row,
    **kwargs: pl.Expr
) -> pl.Expr:
    """
    Calculate decision table output by evaluating expression against each parameter row.
    
    Args:
        parameters: DataFrame containing decision table parameters
        expression: Expression object that can be called with **kwargs
        output_columns: List of columns to include in output
        default: Default values to return if no rows match
        output_fn: Function to create output struct from row values
        **kwargs: Input variable expressions (e.g., v1=pl.col("age"), v2=pl.col("income"))
        
    Returns:
        pl.Expr: Expression that evaluates to output struct for first matching row
    """
    # expression() returns one boolean pl.Expr per parameter row.
    conditions = expression(parameters, **kwargs)

    output_expr = pl
    for condition, row_values in zip(conditions, parameters.iter_rows(named=True)):
        row_output = output_fn(row_values, output_columns)
        output_expr = output_expr.when(condition).then(row_output)

    # Handle default case
    if default is not None:
        default_values = {col: default[i] for i, col in enumerate(output_columns)}
        default_output = output_fn(default_values, output_columns)
    else:
        default_fields = [pl.lit(None).alias(col) for col in output_columns]
        default_output = pl.struct(*default_fields)

    if output_expr is pl:
        return default_output
    return output_expr.otherwise(default_output)
