import typing as t
from dataclasses import dataclass, field
from typing_extensions import TypedDict
import pandas as pd
import polars as pl
from hamilton import node
from .hamilton import HamiltonModule
from .inject import InjectedModule
from ..lazy_execution import wrap_input_node, wrap_processing_node, wrap_collector_node

if t.TYPE_CHECKING:
    from hamilton import node
    from polars._typing import SchemaDict


class MapBatchesKwargs(TypedDict, total=False):
    """TypedDict for map_batches parameters."""
    validate_output_schema: bool
    projection_pushdown: bool
    predicate_pushdown: bool
    cluster_with_columns: bool
    no_optimizations: bool
    inherit_optimization: bool


@dataclass
class StreamablePandasModule(HamiltonModule):
    """A module that wraps a Hamilton graph with streamable pandas input/output nodes.
    
    This inherits from HamiltonModule to get the core graph, then adds:
    - An input node that converts external input to internal parameter name
    - An output node that collects specified outputs into a DataFrame
    - Lazy execution wrappers for efficient polars map_batches processing
    
    The input/output nodes are injected around the existing Hamilton graph.
    
    Attributes:
        external_input_name: Name of external input parameter (e.g., "df")  
        internal_input_name: Name of internal parameter that functions use (e.g., "input")
        output_node_name: Name for the output collection node
        output_columns: Set of column names to collect in the output DataFrame
        output_schema: Schema dict for the output DataFrame (required for map_batches)
        builder: The Hamilton Builder instance (inherited from HamiltonModule)
        map_batches_kwargs: Additional kwargs to pass to polars map_batches
    """
    # Required fields first
    external_input_name: str
    internal_input_name: str
    output_node_name: str
    output_columns: t.List[str]
    output_schema: "SchemaDict"
    # Optional fields with defaults last
    map_batches_kwargs: MapBatchesKwargs = field(default_factory=dict)
    
    def expand_nodes(self, config: t.Dict[str, t.Any]) -> t.Dict[str, "node.Node"]:
        """Gets Hamilton nodes and injects input/output nodes with lazy execution wrappers.
        
        Returns:
            Dictionary mapping node names to Hamilton nodes with lazy execution wrappers applied.
            
        Raises:
            ValueError: If input/output node names collide with existing Hamilton nodes.
        """
        # Get the base Hamilton graph nodes
        hamilton_nodes = super().expand_nodes(config)
        
        # Check for name collisions
        self._check_name_collisions(hamilton_nodes)
        
        # Create input node with parameter injection and wrap it
        input_node = wrap_input_node(self._create_input_node())
        
        # Create output collection node and wrap it with collector
        output_node = wrap_collector_node(
            self._create_output_node(),
            schema=self.output_schema,
            **self.map_batches_kwargs
        )
        
        # Apply parameter mapping to existing nodes if needed
        # updated_nodes = self._apply_parameter_mapping(hamilton_nodes)
        
        # Wrap all Hamilton processing nodes
        wrapped_hamilton_nodes = {
            name: wrap_processing_node(node_obj) 
            for name, node_obj in hamilton_nodes.items()
        }
        
        # Combine all wrapped nodes
        return {
            **wrapped_hamilton_nodes,
            self.internal_input_name: input_node,
            self.output_node_name: output_node
        }
    
    def _check_name_collisions(self, hamilton_nodes: t.Dict[str, "node.Node"]) -> None:
        """Check if input/output node names collide with existing nodes."""
        existing_names = set(hamilton_nodes.keys())
        
        
        conflicts = []
        if self.internal_input_name not in existing_names:
            raise ValueError(
                f"Internal input name '{self.internal_input_name}' does not exist in Hamilton graph. "
                f"Please ensure it matches the parameter name used in your Hamilton functions."
            )
        if hamilton_nodes[self.internal_input_name].node_role != node.NodeType.EXTERNAL:
            raise ValueError(
                f"Internal input name '{self.internal_input_name}' exists but is not an input node. "
                f"Please ensure it is defined as an input parameter in your Hamilton functions."
            )
        ## TODO check type is pd.DataFrame o
        if self.output_node_name in existing_names:
            conflicts.append(self.output_node_name)
            
        if conflicts:
            raise ValueError(
                f"Node name collision(s): {conflicts}. "
                f"These names already exist in the Hamilton graph. "
                f"Please choose different names for internal_input_name or output_node_name."
            )
    
    def _apply_parameter_mapping(self, hamilton_nodes: t.Dict[str, "node.Node"]) -> t.Dict[str, "node.Node"]:
        """Apply parameter mapping to Hamilton nodes if external/internal names differ."""
        if self.external_input_name == self.internal_input_name:
            # No mapping needed
            return hamilton_nodes
            
        # Use InjectedModule's static method for parameter mapping
        parameter_mapping = {self.external_input_name: self.internal_input_name}
        updated_nodes = {}
        
        for node_name, node_obj in hamilton_nodes.items():
            mapped_node, _ = InjectedModule._map_input_vars(node_obj, parameter_mapping)
            updated_nodes[node_name] = mapped_node
            
        return updated_nodes
    
    def _create_input_node(self) -> "node.Node":
        """Creates input node that maps external parameter to internal parameter name."""
        def input_func(**kwargs) -> pd.DataFrame:
            """Marks the entrypoint for where the batching from a polars lazyframe would begin"""
            df = kwargs[self.external_input_name]
            return df.to_pandas()
        
        input_func.__name__ = self.internal_input_name
        input_func.__doc__ = f"Converts {self.external_input_name} to streamable pandas DataFrame"
        
        # Create node with external parameter name mapped to internal
        input_types = {self.external_input_name: (pl.LazyFrame, None)}
        
        return node.Node(
            name=self.internal_input_name,
            typ=pd.DataFrame,
            doc_string=input_func.__doc__,
            callabl=input_func,
            input_types=input_types,
            tags={"module": "streamable_pandas"},
            node_source=node.NodeType.STANDARD
        )
    
    def _create_output_node(self) -> "node.Node":
        """Creates dynamic output collection node based on output_columns."""
        
        # Create input types dynamically based on output_columns
        input_types = {
            col_name: (pd.Series, None) 
            for col_name in self.output_columns
        }
        
        def output_func(**kwargs) -> pd.DataFrame:
            """Collects specified columns into output DataFrame"""
            # Polars is very order specific on the schema so this way we can standardize order
            # Note in newer python versions default dicts maintain insertion order 
            # We are relying on that below
            ordered_data = {col: kwargs[col] for col in self.output_columns}
                
            return pd.DataFrame(ordered_data)
        
        output_func.__name__ = self.output_node_name
        output_func.__doc__ = f"Collects {', '.join(sorted(self.output_columns))} into output DataFrame"
        
        return node.Node(
            name=self.output_node_name,
            typ=pd.DataFrame,
            doc_string=output_func.__doc__,
            callabl=output_func,
            input_types=input_types,
            tags={"module": "streamable_pandas"},
            node_source=node.NodeType.STANDARD
        )