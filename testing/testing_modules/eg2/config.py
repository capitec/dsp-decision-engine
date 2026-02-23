import typing as t
from ped.modules.core import BaseModule, PEDNode
from .impl import a, b, c


class EG2Module(BaseModule):
    """Example module 2 - simple chain functions."""
    type: t.Literal["eg2"] = "eg2"
    
    def expand_nodes(self) -> t.List[PEDNode]:
        return [
            PEDNode.from_callable(a),
            PEDNode.from_callable(b),
            PEDNode.from_callable(c),
        ]