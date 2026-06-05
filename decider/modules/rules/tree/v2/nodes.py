"""Data layer node types with variable support."""

import typing as t
from uuid import UUID
import typing_extensions as t_ext
from logging import getLogger
from pydantic import BaseModel, Field, model_validator
from ...common.nodetypes import RangeEndLogic
from ...common.shared import InputRef

logger = getLogger(__name__)


class RangeNode(BaseModel):
    """Range test node."""

    NODE_TYPE: t.ClassVar[str] = "numerical_range_test_node"
    node_type: t.Literal[NODE_TYPE] = NODE_TYPE  # type: ignore[valid-type]
    feature: str
    # Heterogeneous list: each threshold can be a literal float or a variable reference
    thresholds: t.Union[InputRef, t.List[t.Union[float, InputRef]]] = Field(
        default_factory=list,
        description="Threshold values (floats or variable references)",
    )
    end_logic: RangeEndLogic = RangeEndLogic.lower_inclusive

    def get_required_parameters(self) -> t.Set[str]:
        required_params = set()
        thresholds = self.thresholds
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
        for threshold in thresholds:
            if isinstance(threshold, InputRef):
                required_params.add(threshold.key)
        return required_params

    def to_v3_node(self) -> "t.Any":
        """Convert v2 RangeNode to v3 CasesRanges."""
        from ..v3.nodes_ui import CasesRanges, RangeCondition

        thrs = (
            self.thresholds if isinstance(self.thresholds, list) else [self.thresholds]
        )

        # Build range conditions from thresholds
        # Range i: [thrs[i-1], thrs[i])
        conditions = [
            RangeCondition(
                min=thrs[i - 1] if i > 0 else None,
                max=thrs[i],
            )
            for i in range(len(thrs))
        ]

        return CasesRanges(
            feature=self.feature,
            conditions=conditions,
            end_logic=self.end_logic,
            strict=False,
        )


class NumericalNode(BaseModel):
    """Numerical test node."""

    NODE_TYPE: t.ClassVar[str] = "numerical_test_node"
    node_type: t.Literal[NODE_TYPE] = NODE_TYPE  # type: ignore[valid-type]
    feature: str
    comparison_op: t.Literal["<=", "<", "==", ">", ">=", "!="] = "<="
    threshold: t.Union[float, InputRef] = Field(
        description="Direct threshold value or a variable reference",
    )

    def get_required_parameters(self) -> t.Set[str]:
        if isinstance(self.threshold, InputRef):
            return {self.threshold.key}
        return set()

    def to_v3_node(self) -> "t.Any":
        """Convert v2 NumericalNode to v3 UnaryNode."""
        from ..v3.nodes_ui import (
            UnaryNode,
            UnaryLeq,
            UnaryLt,
            UnaryEq,
            UnaryGt,
            UnaryGeq,
            UnaryNeq,
        )

        # Map comparison_op to appropriate v3 unary operator
        op_map = {
            "<=": UnaryLeq,
            "<": UnaryLt,
            "==": UnaryEq,
            ">": UnaryGt,
            ">=": UnaryGeq,
            "!=": UnaryNeq,
        }

        op_class = op_map[self.comparison_op]
        condition = op_class(feature=self.feature, threshold=self.threshold)

        return UnaryNode(condition=condition)


class CategoricalNode(BaseModel):
    """Categorical test node."""

    NODE_TYPE: t.ClassVar[str] = "categorical_test_node"
    node_type: t.Literal[NODE_TYPE] = NODE_TYPE  # type: ignore[valid-type]
    feature: str

    category_list: t.Union[InputRef, t.List[t.Union[int, float]]] = Field(
        default_factory=list, description="Direct category values"
    )

    def get_required_parameters(self) -> t.Set[str]:
        required_params = set()
        category_list = self.category_list
        if not isinstance(category_list, list):
            category_list = [category_list]
        for category in category_list:
            if isinstance(category, InputRef):
                required_params.add(category.key)
        return required_params

    def to_v3_node(self) -> "t.Any":
        """Convert v2 CategoricalNode to v3 UnaryNode with IsInOp."""
        from ..v3.nodes_ui import UnaryNode, IsInOp

        # Convert to UnaryNode with IsInOp condition
        condition = IsInOp(feature=self.feature, values=self.category_list)
        return UnaryNode(condition=condition)


class StringMatchNode(BaseModel):
    """String match node."""

    NODE_TYPE: t.ClassVar[str] = "string_match_node"
    node_type: t.Literal[NODE_TYPE] = NODE_TYPE  # type: ignore[valid-type]
    feature: str

    patterns: t.Union[InputRef, t.List[str]] = Field(
        default_factory=list, description="Direct pattern values"
    )
    match_type: t.Literal["exact", "starts_with", "contains", "ends_with", "regex"] = (
        "exact"
    )
    case_sensitive: bool = True
    match_any: bool = True

    def get_required_parameters(self) -> t.Set[str]:
        patterns = self.patterns
        if isinstance(patterns, InputRef):
            return {patterns.key}
        return {}

    def to_v3_node(self) -> "t.Any":
        """Convert v2 StringMatchNode to v3 UnaryNode or CasesStringMatch."""
        from ..v3.nodes_ui import (
            UnaryNode,
            StringMatchOp,
            CasesStringMatch,
            StringMatchCondition,
        )
        from ...common.nodetypes import TStringMatchType

        if self.match_any:
            # match_any=True: Use UnaryNode with StringMatchOp
            # Normalize: old V2 allowed a bare InputRef; new unified schema uses List[Union[str, InputRef]]
            patterns = (
                self.patterns if isinstance(self.patterns, list) else [self.patterns]
            )
            condition = StringMatchOp(
                feature=self.feature,
                patterns=patterns,
                match_type=TStringMatchType(self.match_type),
                case_sensitive=self.case_sensitive,
            )
            return UnaryNode(condition=condition)
        else:
            # match_any=False: Use CasesStringMatch for separate branches per pattern
            pattern_list = (
                self.patterns if isinstance(self.patterns, list) else [self.patterns]
            )
            conditions = [
                StringMatchCondition(patterns=[p] if isinstance(p, str) else p)
                for p in pattern_list
            ]
            return CasesStringMatch(
                feature=self.feature,
                match_type=TStringMatchType(self.match_type),
                case_sensitive=self.case_sensitive,
                conditions=conditions,
            )

class LeafNode(BaseModel):
    """Leaf node."""

    NODE_TYPE: t.ClassVar[str] = "leaf"
    node_type: t.Literal[NODE_TYPE] = NODE_TYPE  # type: ignore[valid-type]
    leaf_value: int = -1

    def get_required_parameters(self) -> t.Set[str]:
        return set()

    def to_v3_node(self) -> "t.Any":
        """Convert v2 LeafNode to v3 LeafNode."""
        from ..v3.nodes_ui import LeafNode as V3LeafNode

        return V3LeafNode(result_idx=self.leaf_value)


# Union type for all data layer nodes
NodeData = t_ext.Annotated[
    t.Union[RangeNode, NumericalNode, CategoricalNode, StringMatchNode, LeafNode],
    Field(discriminator="node_type"),
]


class Position(BaseModel):
    x: float
    y: float


class PositionedNode(BaseModel):
    id: str
    position: Position = Field(default_factory=lambda: Position(x=0, y=0))
    data: NodeData
