import typing as t
from ped.modules.core import BaseModule, PEDNode
from .impl import converted


class EG4Module(BaseModule):
    """Example module 4 - type conversion function."""
    type: str = "eg4"
    
    def expand_nodes(self) -> t.List[PEDNode]:
        return [
            PEDNode.from_callable(converted),
        ]