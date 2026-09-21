"""V3 Tree structure - manages nodes and edges separately.

Following v2 conventions where tree structure is defined by edges,
not embedded in nodes. This allows the same node type definitions to work
with both inline (for flat rules) and graph-based (for UI) representations.
"""

import typing as t
import polars as pl
from pydantic import BaseModel, Field
from .nodes_ui import NodeData, PositionedNode
from ..v1.edges import MultiSourceEdge
from ...common.shared import WithTreeOutput, TreeOutput
from .....serializable.schema import PolarsSchema
from ...common.parameters import WithParameters
from decider.modules.core import BaseExecuteModule
from decider.types import TInputType, TOutputType

if t.TYPE_CHECKING:
    from ...flat_rules.nodes import RuleType
    from decider.executor import Executor


class TreeMetadata(BaseModel):
    """Metadata for the tree."""

    name: t.Optional[str] = None
    description: t.Optional[str] = None


class SubTree(BaseModel):
    """Subtree identifier."""

    id: t.Optional[str] = None
    name: t.Optional[str] = None


class Tree(WithTreeOutput, BaseExecuteModule, WithParameters):
    """V3 Tree structure with nodes and edges.

    Structure:
    - Nodes contain data (conditions, operators, leaf values)
    - Edges define the tree structure via sourceIndex
    - For UnaryNode: sourceIndex=0 is 'then', sourceIndex=1 is 'otherwise'
    - For CasesNode: sourceIndex maps to branch indices
    - For CompositeNode: sourceIndex=0 is 'then', sourceIndex=1 is 'otherwise'
    """

    type: t.Literal["v3-tree"]
    name: str = "output"
    metadata: t.Optional[TreeMetadata] = None
    edges: t.List[MultiSourceEdge]
    nodes: t.List[PositionedNode]
    subtrees: t.List[SubTree] = Field(default_factory=list)
    input_schema: t.Optional[PolarsSchema] = Field(
        default=None, description="Input schema for casting inputs at runtime"
    )

    format_version: t.Literal[3] = Field(alias="formatVersion", default=3)

    def get_required_parameters(self) -> t.Set[str]:
        """Get all parameters required by nodes in this tree."""
        required_parameters = set()
        for node in self.nodes:
            required_parameters.update(node.data.get_required_parameters())
        return required_parameters

    def to_tree_module(self) -> "t.Any":
        """Convert to FlatRuleModule for backward compatibility.

        This method exists for compatibility with tests that expect to_tree_module().
        The new architecture uses to_flat_rule_tree() → FlatRuleModule.
        """
        from ...flat_rules.nodes import RuleRoot, RuleMeta
        from ...flat_rules.module import FlatRuleModule

        # Convert to flat rule tree
        flat_rule = self.to_flat_rule_tree()

        # Wrap in RuleRoot
        rule_root = RuleRoot(meta=RuleMeta(), rule=flat_rule)

        # Create FlatRuleModule
        return FlatRuleModule(
            rule=rule_root,
            output=self.output,
            parameters=self.parameters if hasattr(self, "parameters") else {},
            parameters_col=(
                self.parameters_col if hasattr(self, "parameters_col") else "parameters"
            ),
        )

    def to_flat_rule_tree(self) -> "RuleType":
        """Convert this v3 UI tree to flat rules.

        This reconstructs the tree structure from edges and converts each node
        to its flat rule equivalent, preserving position metadata.
        """
        from ...flat_rules.nodes import LeafRule
        from ...common.nodetypes import NodeMeta, NodePosition

        keyed_nodes = {n.id: n for n in self.nodes}

        def _meta(node_id: str) -> NodeMeta:
            """Create NodeMeta from positioned node's position."""
            pos = keyed_nodes[node_id].position
            return NodeMeta(position=NodePosition(x=pos.x, y=pos.y))

        # Build adjacency: node_id -> {source_index -> target_node_id}
        children: t.Dict[str, t.Dict[int, str]] = {}
        for edge in self.edges:
            for si in edge.data.sourceIndex:
                children.setdefault(edge.source, {})[si] = edge.target

        def _build(node_id: str) -> "RuleType":
            """Recursively build flat rule tree from node + edges."""
            node = keyed_nodes[node_id]
            data = node.data
            node_children = children.get(node_id, {})
            meta = _meta(node_id)

            def get_child(idx: int) -> "RuleType":
                """Get child at source index, or default leaf if not connected."""
                target_id = node_children.get(idx)
                return (
                    LeafRule(result_idx=-1) if target_id is None else _build(target_id)
                )

            # Convert node to flat rule, passing get_child callback
            # Note: data.meta is set from node.data, but we also pass meta from position
            flat_node = data.to_flat_rule_node(node_id, get_child)
            # Update meta from UI position if not already set
            if flat_node.meta is None:
                flat_node.meta = meta
            return flat_node

        # Find root nodes (nodes with no incoming edges)
        root_keys = {n.id for n in self.nodes} - {e.target for e in self.edges}

        if len(root_keys) == 0:
            raise ValueError("Tree has no root nodes (circular structure)")
        if len(root_keys) > 1:
            # Multiple roots - use first subtree root or first node
            subtree_mapping = {
                st.id: i for i, st in enumerate(self.subtrees) if st.id is not None
            }
            node_mapping = {n.id: i for i, n in enumerate(self.nodes)}

            ordered_roots = sorted(
                root_keys,
                key=lambda rk: (
                    subtree_mapping.get(rk, float("inf")),
                    node_mapping.get(rk, float("inf")),
                ),
            )
            root_id = ordered_roots[0]
        else:
            root_id = next(iter(root_keys))

        return _build(root_id)

    def execute(self, inputs: TInputType, _executor: "Executor") -> TOutputType:
        from ...flat_rules.nodes import RuleRoot, RuleMeta
        from ...flat_rules.module import FlatRuleModule

        flat_rule = self.to_flat_rule_tree()
        module = FlatRuleModule(
            rule=RuleRoot(meta=RuleMeta(), rule=flat_rule),
            output=self.output,
            parameters=self.parameters,
            parameters_col=self.parameters_col,
        )
        frame = inputs["input"]
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()
        compiled = module.build_expression()
        return frame.select(compiled.expr.struct.unnest()).lazy()
