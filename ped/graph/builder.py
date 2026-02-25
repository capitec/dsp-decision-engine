import typing as t
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from .graph import BaseGraph

if t.TYPE_CHECKING:
    from ped.modules import ConstructedGraphModules


TGraph = t.TypeVar("TGraph", bound=BaseGraph)

class BaseBuilder(BaseModel, t.Generic[TGraph], ABC):
    """Base configuration for graph builders"""
    type: str = Field(description="Type of the graph builder")

    @abstractmethod
    def build_graph(
        self, 
        modules: "ConstructedGraphModules", 
        output_nodes: t.List[str],
    ) -> TGraph:
        ...
