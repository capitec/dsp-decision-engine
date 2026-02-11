import typing as t
import polars as pl

if t.TYPE_CHECKING:
    from .config import Expression, DecisionTableConfig


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
    expression: "Expression",
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
    output_expr = pl
    
    # Process each row in parameters DataFrame
    for row_idx in range(len(parameters)):
        # Get row values as dict
        row_values = {}
        for col in parameters.columns:
            row_values[col] = parameters[col][row_idx]
        
        # Create kwargs for expression evaluation (row parameters + input variables)
        expr_kwargs = {}
        # Add parameter values as literals
        for col, value in row_values.items():
            expr_kwargs[col] = pl.lit(value)
        # Add input variables
        expr_kwargs.update(kwargs)
        
        # Evaluate expression condition for this row
        condition = expression(**expr_kwargs)
        
        # Create output for this row using output_fn
        row_output = output_fn(row_values, output_columns)
        
        # Add to when/then chain
        output_expr = output_expr.when(condition).then(row_output)
    
    # Handle default case
    if default is not None:
        default_values = {col: default[i] for i, col in enumerate(output_columns)}
        default_output = output_fn(default_values, output_columns)
    else:
        # Create null struct for unmatched cases
        default_fields = [pl.lit(None).alias(col) for col in output_columns]
        default_output = pl.struct(*default_fields)
    
    if output_expr is pl:
        # No conditions, return default
        return default_output
    else:
        return output_expr.otherwise(default_output)


def evaluate_decision_table_from_config(
    config: "DecisionTableConfig",
    **kwargs: pl.Expr
) -> pl.Expr:
    """
    Convenience function to evaluate decision table using a DecisionTableConfig.
    
    Args:
        config: DecisionTableConfig object
        **kwargs: Input variable expressions
        
    Returns:
        pl.Expr: Decision table evaluation result
    """
    return calculate_decision_table_output(
        parameters=config.parameters._parameters_df,
        expression=config.expression,
        output_columns=config.outputs,
        default=config.default,
        **kwargs
    )


def extract_struct_fields(
    df: pl.DataFrame,
    struct_column: str,
    field_names: t.List[str]
) -> pl.DataFrame:
    """
    Extract fields from a struct column into separate columns.
    
    Args:
        df: Input DataFrame
        struct_column: Name of the struct column
        field_names: List of field names to extract
        
    Returns:
        pl.DataFrame: DataFrame with extracted fields as separate columns
    """
    extractions = []
    for field_name in field_names:
        extractions.append(
            pl.col(struct_column).struct.field(field_name).alias(field_name)
        )
    return df.with_columns(extractions)