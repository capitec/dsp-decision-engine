import typing as t
import inspect
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

# WIth our approach relying on hamilton nodes makes it really complicated to implement namespaces
# as each layer of namespacing requires a new wrapper function to map inputs to the original function
# This leads to a tonne of overhead just to map. If we use a custom node type with inputs captured we can get around this issue
# It also allows us to separate concerns of graph building vs module building.
# The risks are that hamilton nodes include a tonne of extra features like collect that we wont be implementing
# I think for all of our use-cases this is okay.
# This is just a reference of the hamilton node to help design our node
# class Node(object):
#     """Object representing a node of computation."""
#     name: the name of the node in the graph
#     typ: the type of the output of the node
#     doc_string: the doc string of the node, used for documentation purposes
#     callabl: the callable function that the node represents
#     node_source: the source of the node, used for tracking where the node came from
#     input_types: a dict mapping input names to their types, used for type checking and mapping inputs from other nodes
#     tags: a dict of tags for the node, used for categorization and other metadata
#     namespace: a tuple representing the namespace of the node, used for namespacing nodes to avoid name collisions and organize nodes
#     originating_functions: a tuple of functions that this node originated from, used for tracking the lineage of the node and for debugging purposes
#     optional_values: a dict of optional values that can be used to store extra information about the node, used for flexibility and extensibility of the node

@dataclass
class PEDNode:
    name: str
    callable: t.Callable
    
    namespace: t.Tuple[str, ...] = field(default_factory=tuple)
    # This is if we augment the node at any point we can use this to extact information like
    # docstring, module, and types
    original_callable: t.Optional[t.Callable] = field(default=None)
    input_map: t.Dict[str, str] = field(
        default_factory=dict, 
        metadata={
            "description": "Maps this module's input parameters to external variable names. "
                         "Format: {my_input_param: external_variable_name}. "
                         "Example: {'data': 'user_records'} means this module's 'data' parameter "
                         "will receive the value from the external 'user_records' variable."
        }
    )
    additional_kwargs: t.Dict[str, t.Any] = field(default_factory=dict, metadata={"description": "Additional keyword arguments for the node. can be used to capture additional information needed for the node."})

    @property
    def namespaced_name(self) -> str:
        """Returns the fully qualified name of the node, including its namespace."""
        return ".".join(self.namespace + (self.name,))

    @property
    def type_callable(self) -> t.Callable:
        """Returns the original callable if available, otherwise returns the current callable."""
        return self.original_callable or self.callable

    @classmethod
    def from_callable(
        cls, 
        func: t.Callable,
        name: t.Optional[str] = None,
        input_map: t.Optional[t.Dict[str, str]] = None,
        additional_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> "PEDNode":
        """Create a PEDNode from a callable function.
        
        Args:
            func: The callable function to wrap in a PEDNode
            name: Optional node name. If not provided, uses func.__name__
            input_map: Optional mapping of function parameters to external variable names.
                      Format: {function_param: external_variable_name}.
                      If not provided, all required parameters map to themselves.
            additional_kwargs: Optional additional metadata for the node
            
        Returns:
            A PEDNode instance configured with the function and mappings
            
        Example:
            def process_data(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
                return data[data.value > threshold]
                
            # Create node that maps 'data' param to 'user_records' variable
            node = PEDNode.from_callable(
                process_data,
                name="data_processor", 
                input_map={"data": "user_records"}
            )
        """
        name = name or func.__name__
        function_kwargs = inspect.signature(func).parameters
        required_kwargs = {
            k 
            for k, v in function_kwargs.items() 
            if v.default == inspect.Parameter.empty 
            and v.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        input_map = input_map or {}
        input_map = {k: input_map.get(k, k) for k in required_kwargs}
        additional_kwargs = additional_kwargs or {}
        return cls(
            name=name,
            callable=func,
            original_callable=func,
            input_map=input_map,
            additional_kwargs=additional_kwargs,
        )


class BaseModule(BaseModel, ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'type' not in cls.__annotations__:
            raise TypeError(f"{cls.__name__} must define a 'type' class variable")

    type: str
    input_mapping: t.Dict[str, str] = Field(
        default_factory=dict, 
        description="Maps this module's input parameters to external variable names. "
                   "Format: {my_input_param: external_variable_name}. "
                   "Example: {'data': 'user_records'} means this module's 'data' parameter "
                   "will receive the value from the external 'user_records' variable."
    )
    # Again i think this only really makes sense for referenced modules so leaving out for now until the use becomes apparent
    # version: t.Optional[str] = Field(default=None, description="Module version, latest if not specified")
    # I think that source could be for a specific module type maybe?
    # source: t.Optional[str] = Field(default=None, description="Override source for module discovery")
    # I think this should be on the outer level
    # output_mapping: t.Dict[str, str] = Field(default_factory=dict, description="Map internal outputs to external names")
    # Im wondering if its best to split this from the input mapping or have it similar.
    # internal_overrides: t.Dict[str, str] = Field(default_factory=dict, description="Override internal functions")

    # TODO determine if we need this to first expose a def compile(...) -> NodeExpander
    # and then class NodeExpander(ABC): def expand_nodes(...) -> List[Node]
    # This approach is often times a bit more flexible and was useful in the last implementation to do some heavy work required before returning nodes
    # However for this implementation we assume that the graph is cached at the upper level so the expand nodes can 
    # really be used to execute a node as well.
    @abstractmethod
    def expand_nodes(self) -> t.List[PEDNode]:
        """
        Expands the module into a list of Hamilton nodes. This is where the logic of how the module is represented as a graph goes.
        Note: The use of a list over a dict is deliberate as Node contains a name parameter and using a dict makes it ambiguous as to which name we are using.
        """
        ...


    def module_namespaced_nodes(
        self, 
        module_name: str,
    ) -> t.List[PEDNode]:
        """
        Expand module nodes with namespace and apply input mapping.
        
        This method:
        1. Expands the module into PEDNodes
        2. Adds the module namespace to each node
        3. Applies input mapping to redirect external variables to internal parameters
        
        Args:
            module_name: The namespace to apply to all nodes in this module
            
        Returns:
            List of PEDNodes with namespace and input mapping applied
        """
        # Get the raw nodes from the module
        raw_nodes = self.expand_nodes()
        
        # Track all node names in this module for internal dependency mapping
        node_map = {node.name: node for node in raw_nodes}
        
        namespaced_nodes = []
        for node in raw_nodes:
            # Add module namespace to node
            new_namespace = (module_name,) + node.namespace
            
            # Apply input mapping transformation
            updated_input_map = {}
            for internal_param, current_external_var in node.input_map.items():
                if current_external_var in self.input_mapping:
                    # This parameter should be mapped to a different external variable
                    updated_input_map[internal_param] = self.input_mapping[current_external_var]
                elif current_external_var in node_map:
                    # This is a dependency on another node in this module - namespace it
                    updated_input_map[internal_param] = node_map[current_external_var].namespaced_name
                else:
                    # External dependency - keep as is
                    updated_input_map[internal_param] = current_external_var
            
            # Create the updated node
            namespaced_node = PEDNode(
                name=node.name,
                callable=node.callable,
                namespace=new_namespace,
                original_callable=node.original_callable,
                input_map=updated_input_map,
                additional_kwargs=node.additional_kwargs
            )
            
            namespaced_nodes.append(namespaced_node)
        
        return namespaced_nodes
