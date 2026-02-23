import typing as t
from pydantic import BaseModel
from ped.modules.core import BaseModule, PEDNode
from .impl import a, b, c, Eg3Config




class EG3Module(BaseModule):
    """Example module 3 - functions with config parameters."""
    type: t.Literal["eg3"] = "eg3"
    config: Eg3Config = Eg3Config()
    
    def expand_nodes(self) -> t.List[PEDNode]:
        # Pass config as additional kwargs to each node
        return [
            PEDNode.from_callable(
                a,
                static_kwargs={"config": self.config}
            ),
            PEDNode.from_callable(
                b,
                static_kwargs={"config": self.config}
            ),
            PEDNode.from_callable(
                c,
                static_kwargs={"config": self.config}
            ),
        ]