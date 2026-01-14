import typing as t
from dataclasses import dataclass
import polars as pl
from hamilton import node
from .core import DeciderExpandableModule

if t.TYPE_CHECKING:
    from hamilton import node


# TODO Lots of work to do in this file still need to work out what we going to call the inputs and how we going to map it in a way that it can still compile further if we want

def run_expression(input: pl.Expr, expression: pl.Expr) -> pl.Expr:
    """Runs a Polars expression on a DataFrame.
    
    Args:
        df: The input DataFrame
        expression: The Polars expression to run
        
    Returns:
        DataFrame with the expression result
    """
    return df.with_columns(expression)


@dataclass
class PolarsExpressionModule(DeciderExpandableModule):
    """A module that creates a single node to run a Polars expression.
    
    This allows creating nodes dynamically from Polars expressions without
    having to define separate functions for each expression.
    
    Attributes:
        expression: The Polars expression to run
        node_name: The name for the created node
    """
    expression: pl.Expr
    node_name: str
    
    def expand_nodes(self) -> t.Dict[str, "node.Node"]:
        """Creates a single node that runs the Polars expression.
        
        Returns:
            Dictionary with the single node using the specified node_name.
        """
        # Create a function that captures the expression
        def expression_runner(df: pl.DataFrame) -> pl.DataFrame:
            return run_expression(df, self.expression)
        
        # Set the function name for better debugging
        expression_runner.__name__ = self.node_name
        expression_runner.__doc__ = f"Runs Polars expression: {self.expression}"
        
        # Create and return the node
        expression_node = node.Node.from_fn(expression_runner)
        return {self.node_name: expression_node}