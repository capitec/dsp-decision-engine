import typing as t
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from .nodes import NodeData, PositionedNode
from ..v1.variables import VariableMap
from ..v1.edges import MultiSourceEdge
from ..v1.schema import OutputSchema
from logging import getLogger
from ped.modules.tree.shared import WithTreeOutput


logger = getLogger(__name__)


class TreeOutput(BaseModel):
    columns: t.List[str]
    data: t.List[t.List[str]]
    dtype: t.Optional[t.List[str]] = None


class TreeMetadata(BaseModel):
    name: str
    description: str


class Tree(BaseModel,WithTreeOutput):
    metadata: TreeMetadata | None = None
    edges: t.List[MultiSourceEdge]
    nodes: t.List[PositionedNode]
    subtrees: t.List[str] = Field(default_factory=list)

    format_version: t.Literal[2] = Field(alias="formatVersion", default=2)

    def upgrade(self):
        """V2 is the latest version, return self"""
        return self

