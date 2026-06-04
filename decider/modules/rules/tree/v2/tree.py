import typing as t
import re
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from .nodes import NodeData, PositionedNode
from ..v1.edges import MultiSourceEdge
from logging import getLogger
from dspd.components.common.shared import WithTreeOutput, TreeOutput
from dspd.components.serializable.schema import PolarsSchema
from dspd.components.common.parameters import WithParameters

if t.TYPE_CHECKING:
    from dspd.components.tree.v3.tree import Tree as V3Tree


logger = getLogger(__name__)


class TreeMetadata(BaseModel):
    name: t.Optional[str] = None
    description: t.Optional[str] = None


class SubTree(BaseModel):
    id: t.Optional[str] = None
    name: t.Optional[str] = None


class Tree(WithTreeOutput, WithParameters):
    type: t.Literal["ui-tree"] = "ui-tree"
    metadata: TreeMetadata | None = None
    edges: t.List[MultiSourceEdge]
    nodes: t.List[PositionedNode]
    subtrees: t.List[SubTree] = Field(default_factory=list)
    input_schema: t.Optional[PolarsSchema] = Field(
        default=None, description="Input schema for casting inputs at runtime"
    )

    format_version: t.Literal[2] = Field(alias="formatVersion", default=2)

    def get_required_parameters(self) -> t.Set[str]:
        required_parameters = set()
        for node in self.nodes:
            required_parameters.update(node.data.get_required_parameters())
        return required_parameters

    def to_tree_module(self) -> "t.Any":
        """Convert to FlatRuleModule via v3 upgrade.

        This method exists for backward compatibility with tests.
        The conversion path is: v2 → v3 → flat rules → FlatRuleModule.
        """
        v3_tree = self.upgrade()
        return v3_tree.to_tree_module()

    def upgrade(self) -> "V3Tree":
        """Upgrade v2 tree to v3 format.

        V3 is the latest format with unified types shared with flat rules.
        V2 nodes are converted to v3 nodes via their to_v3_node() method.
        """
        from dspd.components.tree.v3.tree import Tree as V3Tree
        from dspd.components.tree.v3.tree import TreeMetadata as V3TreeMetadata
        from dspd.components.tree.v3.tree import SubTree as V3SubTree
        from dspd.components.tree.v3.nodes_ui import PositionedNode as V3PositionedNode
        from dspd.components.tree.v3.nodes_ui import Position as V3Position

        # Convert nodes - v2 nodes need to be converted to v3 node format
        v3_nodes = []
        for node in self.nodes:
            v3_data = node.data.to_v3_node()
            v3_nodes.append(
                V3PositionedNode(
                    id=node.id,
                    position=V3Position(x=node.position.x, y=node.position.y),
                    data=v3_data,
                )
            )

        # Convert metadata
        v3_metadata = None
        if self.metadata:
            v3_metadata = V3TreeMetadata(
                name=self.metadata.name,
                description=self.metadata.description,
            )

        # Convert subtrees
        v3_subtrees = [V3SubTree(id=st.id, name=st.name) for st in self.subtrees]

        # Edges remain the same format (MultiSourceEdge)
        return V3Tree(
            metadata=v3_metadata,
            edges=self.edges,
            nodes=v3_nodes,
            subtrees=v3_subtrees,
            output=self.output,
            parameters=self.parameters,
            parameters_col=self.parameters_col,
            input_schema=self.input_schema,
        )
