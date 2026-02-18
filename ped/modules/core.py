import typing as t
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
    input_map: t.Dict[str, str] = field(default_factory=dict, metadata={"description": "Map from other modules outputs to this modules inputs"})
    additional_kwargs: t.Dict[str, t.Any] = field(default_factory=dict, metadata={"description": "Additional keyword arguments for the node. can be used to capture additional information needed for the node."})

class BaseModule(BaseModel, ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'type' not in cls.__annotations__:
            raise TypeError(f"{cls.__name__} must define a 'type' class variable")

    type: str
    input_mapping: t.Dict[str, str] = Field(default_factory=dict, description="Map from other modules outputs to this modules inputs")
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
