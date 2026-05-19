"""Join module for combining dataframes.

This module provides JoinModule which creates frame-level nodes for joining dataframes.
"""

import typing as t
import polars as pl
from decider.modules.core import BaseModule, Node, ExternalInputNode


class JoinModule(BaseModule):
    """Module that joins two dataframes.

    Creates a single frame node that performs a join operation.

    Example:
        >>> join = JoinModule(
        ...     name="user_transactions",
        ...     left="transactions",
        ...     right="users",
        ...     on="user_id",
        ...     how="left",
        ...     output_frame="enriched"
        ... )
        >>> result = join.execute({
        ...     "transactions": txn_df,
        ...     "users": user_df
        ... })
        >>> # result["enriched"] contains the joined data
    """

    type: t.Literal["join"]
    left: str
    """Name of the left dataframe"""

    right: str
    """Name of the right dataframe"""

    on: t.Union[str, t.List[str]]
    """Column(s) to join on"""

    how: t.Literal["inner", "left", "outer", "cross", "semi", "anti"] = "left"
    """Join strategy"""

    output_frame: str = "joined"
    """Name of the output frame (default: 'joined')"""

    suffix: str = "_right"
    """Suffix for duplicate column names from right frame"""

    def expand_nodes(self) -> t.List[Node]:
        """Create a single frame node that performs the join."""

        def join_func(left_df: pl.LazyFrame, right_df: pl.LazyFrame) -> pl.LazyFrame:
            """Execute the join operation."""
            return left_df.join(
                right_df,
                on=self.on,
                how=self.how,
                suffix=self.suffix
            )

        node = Node(
            name=f"{self.name}_join",
            callable=join_func,
            node_type="frame",
            target_frame=self.output_frame,
            input_map={
                "left_df": ExternalInputNode(self.left),
                "right_df": ExternalInputNode(self.right),
            }
        )

        return [node]
