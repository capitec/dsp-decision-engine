import typing as t
from dataclasses import dataclass, field
from hamilton.driver import Builder
from .base import DeciderExpandableModule

if t.TYPE_CHECKING:
    from hamilton import node


@dataclass
class HamiltonModule(DeciderExpandableModule):
    """A module wrapper that builds a Hamilton graph and extracts its nodes.
    
    This allows integrating existing Hamilton modules and their function graphs
    into the Decider framework by building the graph and extracting all nodes.
    
    """
    
    def __post_init__(self):
        """Initialize with default builder if none provided."""
        self.builder = Builder()

    def use_builder(self, builder: Builder) -> "HamiltonModule":
        """Set a custom Hamilton Builder for this module.
        
        Args:
            builder: The Hamilton Builder instance to use
        Returns:
            The original HamiltonModule instance with updated builder
        """
        self.builder = builder
        return self
    
    def with_modules(self, *modules) -> "HamiltonModule":
        """Create a HamiltonModule with the specified modules.
        
        Args:
            *modules: Hamilton modules to include in the builder
            
        Returns:
            The original HamiltonModule instance with updated builder
        """
        self.builder = self.builder.with_modules(*modules)
        return self
    
    def with_config(self, **config) -> "HamiltonModule":
        """Create a new HamiltonModule with additional config.
        
        Args:
            **config: Configuration key-value pairs to add to the builder
            
        Returns:
            The original HamiltonModule instance with updated config
        """
        self.builder = self.builder.with_config(*config)
        return self
    
    def expand_nodes(self, config: t.Dict[str, t.Any]) -> t.Dict[str, "node.Node"]:
        """Builds the Hamilton graph and extracts all nodes from the functional graph.
        
        Args:
            config: Configuration dictionary passed to the Hamilton builder
            
        Returns:
            A dictionary mapping node names to their corresponding Hamilton nodes.
        """
        # Build the Hamilton driver with the provided config
        driver = self.builder.with_config(config).build()
        
        # Access the functional graph to get all nodes
        function_graph = driver.graph
        
        # Extract all nodes from the graph
        nodes_dict = {}
        for node_name, node_obj in function_graph.nodes.items():
            nodes_dict[node_name] = node_obj
            
        return nodes_dict
