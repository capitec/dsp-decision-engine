import typing as t
import polars as pl


def extract_struct_field(
    column: pl.Expr,
    field_name: str
) -> pl.Expr:
    """
    Extract a field from a struct column.
    
    Args:
        column: Struct column expression
        field_name: Name of the field to extract
        
    Returns:
        pl.Expr: Expression for the extracted field
    """
    return column.struct.field(field_name)

def extract_struct_fields(
    column: pl.Expr,
    field_names: t.List[str]
) -> t.Dict[str, pl.Expr]:
    """
    Extract fields from a struct column into separate columns.
    
    Args:
        column: Struct column expression
        field_names: List of field names to extract
        
    Returns:
        Dict[str, pl.Expr]: Dictionary mapping field names to expressions
    """
    extractions = {}
    for field_name in field_names:
        extractions[field_name] = extract_struct_field(column, field_name)
    return extractions
