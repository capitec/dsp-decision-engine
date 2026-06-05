"""V3 Tree nodes — UI/tree representation.

All structural definitions (operators, conditions, base nodes) live in
dspd.components.common.nodes. This module adds only to_flat_rule_node()
for converting the edge-based tree representation to flat rules.
"""

import typing as t
from pydantic import BaseModel, Field, RootModel

from ...common.nodetypes import (
    LeafNodeCore,
    BaseRule,
    NodeMeta,
    NodePosition,
    TLogicOp,
    RangeEndLogic,
)
from ...common.nodes import (
    TUnaryOp,
    RangeCondition,
    StringMatchCondition,
    IsInCondition,
    CasesBranch,
    TCaseCondition,
    CompositeCondition,
    TCondition,
    BaseUnaryNode,
    BaseCasesRanges,
    BaseCasesStringMatch,
    BaseCasesIsIn,
    BaseCompositeNode,
    # Re-export operators so v3/__init__.py keeps working
    UnaryLessThanEqual as UnaryLeq,
    UnaryLessThan as UnaryLt,
    UnaryEqual as UnaryEq,
    UnaryGreaterThan as UnaryGt,
    UnaryGreaterThanEqual as UnaryGeq,
    UnaryNotEqual as UnaryNeq,
    UnaryBetween as BetweenOp,
    UnaryIsIn as IsInOp,
    UnaryStringMatch as StringMatchOp,
    UnaryIsNull as IsNullOp,
    UnaryIsNotNull as IsNotNullOp,
    UnaryIsTrue as IsTrueOp,
    UnaryIsFalse as IsFalseOp,
)

if t.TYPE_CHECKING:
    from ...flat_rules.nodes import RuleType


# =============================================================================
# Leaf Node
# =============================================================================


class LeafNode(LeafNodeCore):
    """Leaf node for tree v3."""

    NODE_TYPE: t.ClassVar[str] = "leaf"
    id: t.Optional[str] = Field(default=None)

    def to_flat_rule_node(
        self, node_id: str, get_child: t.Callable[[int], "RuleType"]
    ) -> "RuleType":
        from ...flat_rules.nodes import LeafRule

        return LeafRule(result_idx=self.result_idx)

    def get_required_features(self) -> t.Set[str]:
        return set()

    def get_required_parameters(self) -> t.Set[str]:
        return set()


# =============================================================================
# Unary Node
# =============================================================================


class UnaryNode(BaseUnaryNode):
    """Single condition node — children resolved via graph edges.

    Edge mapping:
    - sourceIndex=0 -> 'then' branch (condition true)
    - sourceIndex=1 -> 'otherwise' branch (condition false)
    """

    NODE_TYPE: t.ClassVar[str] = "unary"

    def to_flat_rule_node(
        self, node_id: str, get_child: t.Callable[[int], "RuleType"]
    ) -> "RuleType":
        from ...flat_rules.nodes import UnaryRule

        return UnaryRule(
            id=node_id,
            condition=self.condition,
            then=get_child(0),
            otherwise=get_child(1),
        )


# =============================================================================
# Cases Nodes
# =============================================================================


class CasesRanges(BaseCasesRanges):
    """Multi-way range branching — children resolved via graph edges.

    Edge mapping:
    - sourceIndex=0..N-1 -> branch for conditions[0..N-1]
    - sourceIndex=N -> 'otherwise' branch
    """

    NODE_TYPE: t.ClassVar[str] = "cases"

    def to_flat_rule_node(
        self, node_id: str, get_child: t.Callable[[int], "RuleType"]
    ) -> "RuleType":
        from ...flat_rules.nodes import (
            CasesRanges as FlatCasesRanges,
            CasesBranch as FlatCasesBranch,
        )

        flat_conditions = [
            FlatCasesBranch(when=cond, then=i) for i, cond in enumerate(self.conditions)
        ]
        otherwise_idx = len(self.conditions)
        flat_branches = [get_child(i) for i in range(otherwise_idx)] + [
            get_child(otherwise_idx)
        ]

        return FlatCasesRanges(
            id=node_id,
            feature=self.feature,
            conditions=flat_conditions,
            end_logic=self.end_logic,
            strict=self.strict,
            otherwise=otherwise_idx,
            branches=flat_branches,
        )


class CasesStringMatch(BaseCasesStringMatch):
    """Multi-way string matching — children resolved via graph edges.

    Edge mapping:
    - sourceIndex=0..N-1 -> branch for conditions[0..N-1]
    - sourceIndex=N -> 'otherwise' branch
    """

    NODE_TYPE: t.ClassVar[str] = "cases"

    def to_flat_rule_node(
        self, node_id: str, get_child: t.Callable[[int], "RuleType"]
    ) -> "RuleType":
        from ...flat_rules.nodes import (
            CasesStringMatch as FlatCasesStringMatch,
            CasesBranch as FlatCasesBranch,
        )

        flat_conditions = [
            FlatCasesBranch(when=cond, then=i) for i, cond in enumerate(self.conditions)
        ]
        otherwise_idx = len(self.conditions)
        flat_branches = [get_child(i) for i in range(otherwise_idx)] + [
            get_child(otherwise_idx)
        ]

        return FlatCasesStringMatch(
            id=node_id,
            feature=self.feature,
            match_type=self.match_type,
            case_sensitive=self.case_sensitive,
            trim_whitespace=self.trim_whitespace,
            conditions=flat_conditions,
            otherwise=otherwise_idx,
            branches=flat_branches,
        )


class CasesIsIn(BaseCasesIsIn):
    """Multi-way categorical branching — children resolved via graph edges.

    Edge mapping:
    - sourceIndex=0..N-1 -> branch for conditions[0..N-1]
    - sourceIndex=N -> 'otherwise' branch
    """

    NODE_TYPE: t.ClassVar[str] = "cases"

    def to_flat_rule_node(
        self, node_id: str, get_child: t.Callable[[int], "RuleType"]
    ) -> "RuleType":
        from ...flat_rules.nodes import (
            CasesIsIn as FlatCasesIsIn,
            CasesBranch as FlatCasesBranch,
        )

        flat_conditions = [
            FlatCasesBranch(when=cond, then=i) for i, cond in enumerate(self.conditions)
        ]
        otherwise_idx = len(self.conditions)
        flat_branches = [get_child(i) for i in range(otherwise_idx)] + [
            get_child(otherwise_idx)
        ]

        return FlatCasesIsIn(
            id=node_id,
            feature=self.feature,
            conditions=flat_conditions,
            otherwise=otherwise_idx,
            branches=flat_branches,
        )


_TCasesVariant = t.Annotated[
    t.Union[CasesRanges, CasesStringMatch, CasesIsIn],
    Field(discriminator="op"),
]


class CasesNode(RootModel[_TCasesVariant]):
    """Wrapper for all Cases node variants (discriminated by 'op' field)."""

    root: _TCasesVariant

    @property
    def type(self) -> str:
        return "cases"

    @property
    def id(self) -> t.Optional[str]:
        return self.root.id

    @property
    def feature(self):
        return self.root.feature

    @property
    def op(self) -> str:
        return self.root.op

    def to_flat_rule_node(
        self, node_id: str, get_child: t.Callable[[int], "RuleType"]
    ) -> "RuleType":
        return self.root.to_flat_rule_node(node_id, get_child)

    def get_required_features(self) -> t.Set[str]:
        return self.root.get_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        return self.root.get_required_parameters()


# =============================================================================
# Composite Node
# =============================================================================


class CompositeNode(BaseCompositeNode):
    """Composite AND/OR/NOT node — children resolved via graph edges.

    Edge mapping:
    - sourceIndex=0 -> 'then' branch (composite true)
    - sourceIndex=1 -> 'otherwise' branch (composite false)
    """

    NODE_TYPE: t.ClassVar[str] = "composite"

    def to_flat_rule_node(
        self, node_id: str, get_child: t.Callable[[int], "RuleType"]
    ) -> "RuleType":
        from ...flat_rules.nodes import CompositeRule

        return CompositeRule(
            id=node_id,
            op=self.op,
            conditions=self.conditions,
            then=get_child(0),
            otherwise=get_child(1),
        )


# =============================================================================
# Node Data Union and Positioned Node
# =============================================================================

NodeData = t.Annotated[
    t.Union[LeafNode, UnaryNode, CasesNode, CompositeNode],
    Field(discriminator="type"),
]


class Position(BaseModel):
    x: float
    y: float


class PositionedNode(BaseModel):
    id: str
    position: Position = Field(default_factory=lambda: Position(x=0, y=0))
    data: NodeData
