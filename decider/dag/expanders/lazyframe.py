# import typing as t
# from dataclasses import dataclass
# import polars as pl
# from hamilton import node
# from .base import DeciderExpandableModule

# if t.TYPE_CHECKING:
#     from hamilton import node


# def inject_data_into_lazyframe(lazyframe: pl.LazyFrame, data: pl.DataFrame) -> pl.DataFrame:
#     # TODO work out what to do here
#     pass


# @dataclass  
# class LazyFrameModule(DeciderExpandableModule):
#     """A module that creates a single node to run a LazyFrame query with data injection.
    
#     This allows creating nodes dynamically from LazyFrame query plans that can
#     have data injected at runtime.
    
#     Attributes:
#         lazyframe: The LazyFrame query plan to run
#         node_name: The name for the created node
#     """
#     lazyframe: pl.LazyFrame
#     node_name: str
    
#     def expand_nodes(self) -> t.Dict[str, "node.Node"]:
#         """Creates a single node that runs the LazyFrame query with data injection.
        
#         Returns:
#             Dictionary with the single node using the specified node_name.
#         """
#         # Create a function that injects data into the lazyframe
#         def lazyframe_runner(data: pl.DataFrame) -> pl.DataFrame:
#             return inject_data_into_lazyframe(self.lazyframe, data)
        
#         # Set the function name for better debugging
#         lazyframe_runner.__name__ = self.node_name
#         lazyframe_runner.__doc__ = f"Runs LazyFrame query with data injection: {self.lazyframe}"
        
#         # Create and return the node
#         lazyframe_node = node.Node.from_fn(lazyframe_runner)
#         return {self.node_name: lazyframe_node}