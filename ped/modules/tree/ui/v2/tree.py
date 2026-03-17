import typing as t
import re
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from .nodes import NodeData, PositionedNode
from ..v1.variables import VariableMap
from ..v1.edges import MultiSourceEdge
from ..v1.schema import OutputSchema
from logging import getLogger
from ped.modules.tree.shared import WithTreeOutput

if t.TYPE_CHECKING:
    from ped.modules.tree.module import PrioritizedTreeModule


logger = getLogger(__name__)


class TreeOutput(BaseModel):
    columns: t.List[str]
    data: t.List[t.List[str]]
    dtype: t.Optional[t.List[str]] = None


class TreeMetadata(BaseModel):
    name: str
    description: str

class SubTree(BaseModel):
    id: t.Optional[str] = None
    name: t.Optional[str] = None


class Tree(WithTreeOutput):
    metadata: TreeMetadata | None = None
    edges: t.List[MultiSourceEdge]
    nodes: t.List[PositionedNode]
    subtrees: t.List[SubTree] = Field(default_factory=list)

    format_version: t.Literal[2] = Field(alias="formatVersion", default=2)

    def upgrade(self):
        """V2 is the latest version, return self"""
        return self

    def to_tree_module(
        self,
        name: t.Optional[str] = None,
        parameters_col: str = "parameters",
        default_parameters: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> "PrioritizedTreeModule":
        """Convert this v2 UI tree to a PrioritizedTreeModule ready for execution.

        Each subtree root is converted recursively into an execution NodeType using
        the edge adjacency.  Unconnected child slots become no-match LeafNodeType(-1).
        """
        from ped.modules.tree.module import PrioritizedTreeModule, SubTreeRoot
        from ped.modules.tree.nodes import (
            LeafNodeType, RangesNodeType, NumericalNodeType, CategoricalNodeType,
            StringNodeType, BranchType, MinMaxConditionType, NumericalConditionType,
            CategoricalConditionType, StringPatternConditionType, RangeEndLogic, InputRef,
            NodeMeta, NodePosition,
        )
        from .nodes import LeafNode, NumericalNode, CategoricalNode, RangeNode, StringMatchNode, VariableReference

        keyed_nodes = {n.id: n for n in self.nodes}

        def _meta(node_id: str) -> NodeMeta:
            pos = keyed_nodes[node_id].position
            return NodeMeta(position=NodePosition(x=pos.x, y=pos.y))

        # adjacency: node_id -> {sourceIndex -> target_node_id}
        children: t.Dict[str, t.Dict[int, str]] = {}
        for edge in self.edges:
            for si in edge.data.sourceIndex:
                children.setdefault(edge.source, {})[si] = edge.target

        def _child(node_id: str, idx: int):
            cid = children.get(node_id, {}).get(idx)
            return LeafNodeType() if cid is None else _build(cid)

        def _ref(val):
            """VariableReference → InputRef, float stays float."""
            return InputRef(key=val.name) if isinstance(val, VariableReference) else val

        def _build(node_id: str):
            data = keyed_nodes[node_id].data
            m = _meta(node_id)

            if isinstance(data, LeafNode):
                return LeafNodeType(result_idx=data.leaf_value, meta=m)

            if isinstance(data, NumericalNode):
                return NumericalNodeType(
                    feature=data.feature, meta=m,
                    branches=[BranchType(
                        when=NumericalConditionType(op=data.comparison_op, threshold=_ref(data.threshold)),
                        then=_child(node_id, 0),
                    )],
                    otherwise=_child(node_id, 1),
                )

            if isinstance(data, CategoricalNode):
                if data.category_list and isinstance(data.category_list[0], VariableReference):
                    cats = InputRef(key=data.category_list[0].name)
                else:
                    cats = list(data.category_list)
                # category_list_right_child=True → matches go to sourceIndex 1 (right)
                left, right = (_child(node_id, 0), _child(node_id, 1))
                then_child, otherwise_child = (right, left) if data.category_list_right_child else (left, right)
                return CategoricalNodeType(
                    feature=data.feature, meta=m,
                    branches=[BranchType(when=CategoricalConditionType(categories=cats), then=then_child)],
                    otherwise=otherwise_child,
                )

            if isinstance(data, RangeNode):
                thrs = [v for v in data.thresholds if isinstance(v, (int, float))]
                end_logic = RangeEndLogic.upper_inclusive if data.default_left else RangeEndLogic.lower_inclusive
                N = len(thrs)
                if N == 0:
                    return _child(node_id, 0)
                branches = [
                    BranchType(
                        when=MinMaxConditionType(min=thrs[i - 1] if i > 0 else None, max=thrs[i]),
                        then=_child(node_id, i),
                    )
                    for i in range(N)
                ]
                return RangesNodeType(
                    feature=data.feature, branches=branches, meta=m,
                    otherwise=_child(node_id, N), end_logic=end_logic, strict=False,
                )

            if isinstance(data, StringMatchNode):
                if data.match_any:
                    pat = set(data.patterns) if len(data.patterns) > 1 else (data.patterns[0] if data.patterns else "")
                    return StringNodeType(
                        feature=data.feature, meta=m,
                        branches=[BranchType(
                            when=StringPatternConditionType(pattern=pat, match_type=data.match_type, case_sensitive=data.case_sensitive),
                            then=_child(node_id, 0),
                        )],
                        otherwise=_child(node_id, 1),
                    )
                else:
                    branches = [
                        BranchType(
                            when=StringPatternConditionType(pattern=p, match_type=data.match_type, case_sensitive=data.case_sensitive),
                            then=_child(node_id, i),
                        )
                        for i, p in enumerate(data.patterns)
                    ]
                    return StringNodeType(
                        feature=data.feature, branches=branches, meta=m,
                        otherwise=_child(node_id, len(data.patterns)),
                    )

            raise ValueError(f"Unknown v2 node type: {type(data)}")

        if name is None:
            name = re.sub(r'\W+', '_', self.metadata.name.lower()) if self.metadata and self.metadata.name else "TreeModule"
        roots = [
            SubTreeRoot(node=_build(st.id), meta={"name":st.name})
            for st in self.subtrees
        ]
        return PrioritizedTreeModule(
            roots=roots,
            name=name,
            output=self.output,
            schema=self.schema,
            default=self.default,
            parameters_col=parameters_col,
            default_parameters=default_parameters or {},
        )

