import typing as t
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
# Note: At some point in the future it would be best to have our own Node type
# However we are currently very tied to hamilton and reimplementing the node may come with extra 
# side effects we would have to account for so for now we will just use the hamilton node and abstract it away later if we need to.
from hamilton.node import Node

class BaseModule(BaseModel, ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'type' not in cls.__annotations__:
            raise TypeError(f"{cls.__name__} must define a 'type' class variable")

    type: str
    input_mapping: t.Dict[str, str] = Field(default_factory=dict, description="Map internal names to external names")
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
    def expand_nodes(self) -> t.List[Node]:
        """
        Expands the module into a list of Hamilton nodes. This is where the logic of how the module is represented as a graph goes.
        Note: The use of a list over a dict is deliberate as Node contains a name parameter and using a dict makes it ambiguous as to which name we are using.
        """
        ...
