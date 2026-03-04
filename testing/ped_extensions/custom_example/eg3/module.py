import typing as t
from ped.modules.core import BaseModule, PEDNode
from .impl import a, b, c




class EG3Module(BaseModule):
    """Example module 3 - functions with config parameters."""
    # TODO Not sure if it would be possible to enforce some sort of namespacing for the types
    type: t.Literal["custom_example.eg3"]
    multiplier: int = 0
    constant: float = 10.0
    
    def expand_nodes(self) -> t.List[PEDNode]:
        # Pass config as additional kwargs to each node
        return [
            PEDNode.from_callable(
                a,
                static_kwargs={"constant": self.constant}
            ),
            PEDNode.from_callable(
                b,
            ),
            PEDNode.from_callable(
                c,
                static_kwargs={"multiplier": self.multiplier}
            ),
        ]