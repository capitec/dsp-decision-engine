"""V3 Tree nodes following flat rules conventions.

Unified type system that shares core types with flat_rules.
Nodes are managed via edges (tree.py), not embedded in node definitions.
"""

from ...common.nodetypes import (
    BaseRule,
    NodeMeta,
    NodePosition,
)

from .nodes_ui import (
    # Node types
    LeafNode,
    UnaryNode,
    CasesRanges,
    CasesStringMatch,
    CasesIsIn,
    CompositeNode,
    NodeData,
    PositionedNode,
    Position,
    # Unary operators (aliased in nodes_ui from common.nodes)
    UnaryLeq,
    UnaryLt,
    UnaryEq,
    UnaryGt,
    UnaryGeq,
    UnaryNeq,
    BetweenOp,
    IsInOp,
    StringMatchOp,
    IsNullOp,
    IsNotNullOp,
    IsTrueOp,
    IsFalseOp,
)

# TUnaryOp and condition types live in common.nodes
from ...common.nodes import (
    TUnaryOp,
    RangeCondition,
    StringMatchCondition,
    IsInCondition,
    CompositeCondition,
)

from .tree import (
    Tree,
    TreeMetadata,
    SubTree,
)

__all__ = [
    # Base types
    "BaseRule",
    "NodeMeta",
    "NodePosition",
    # Nodes
    "LeafNode",
    "UnaryNode",
    "CasesRanges",
    "CasesStringMatch",
    "CasesIsIn",
    "CompositeNode",
    "NodeData",
    "PositionedNode",
    "Position",
    # Unary operators
    "UnaryLeq",
    "UnaryLt",
    "UnaryEq",
    "UnaryGt",
    "UnaryGeq",
    "UnaryNeq",
    "BetweenOp",
    "IsInOp",
    "StringMatchOp",
    "IsNullOp",
    "IsNotNullOp",
    "IsTrueOp",
    "IsFalseOp",
    "TUnaryOp",
    # Conditions
    "RangeCondition",
    "StringMatchCondition",
    "IsInCondition",
    "CompositeCondition",
    # Tree structure
    "Tree",
    "TreeMetadata",
    "SubTree",
]
