"""Data layer node types with variable support."""

import typing as t
import typing_extensions as t_ext
from logging import getLogger
from pydantic import BaseModel, Field, model_validator

logger = getLogger(__name__)


class HasVariablesMixin:
    variables: t.List[str] = Field(
        default_factory=list, description="Variable IDs used by this node"
    )


class RangeNode(BaseModel):
    """Range test node."""

    NODE_TYPE: t.ClassVar[str] = "numerical_range_test_node"
    node_type: t.Literal[NODE_TYPE] = NODE_TYPE  # type: ignore[valid-type]
    split_feature_id: int
    default_left: bool = False
    thresholds: t.List[float] = Field(
        default_factory=list, description="Direct threshold values"
    )


class NumericalNode(BaseModel, HasVariablesMixin):
    """Numerical test node."""

    NODE_TYPE: t.ClassVar[str] = "numerical_test_node"
    node_type: t.Literal[NODE_TYPE] = NODE_TYPE  # type: ignore[valid-type]
    split_feature_id: int
    default_left: bool = False
    comparison_op: t.Literal["<=", "<", "==", ">", ">="] = "<="
    threshold: t.Optional[float] = Field(
        description="Direct threshold value", default=None
    )

    @model_validator(mode="after")
    def validate_threshold_or_variable(self) -> t_ext.Self:
        if self.threshold is None and len(self.variables) != 1:
            raise ValueError(
                "A single variable must be provided if no threshold is given"
            )
        return self


class CategoricalNode(BaseModel, HasVariablesMixin):
    """Categorical test node."""

    NODE_TYPE: t.ClassVar[str] = "categorical_test_node"
    node_type: t.Literal[NODE_TYPE] = NODE_TYPE  # type: ignore[valid-type]
    split_feature_id: int
    default_left: bool = False
    category_list: t.List[int] = Field(
        default_factory=list, description="Direct category values"
    )
    category_list_right_child: bool = False

    @model_validator(mode="after")
    def validate_threshold_or_variable(self) -> t_ext.Self:
        if len(self.category_list) == 0 and len(self.variables) != 1:
            raise ValueError(
                "One or more variables must be given if no categories are provided."
            )
        return self


class StringMatchNode(BaseModel, HasVariablesMixin):
    """String match node."""

    NODE_TYPE: t.ClassVar[str] = "string_match_node"
    node_type: t.Literal[NODE_TYPE] = NODE_TYPE  # type: ignore[valid-type]
    split_feature_id: int
    default_left: bool = True
    patterns: t.List[str] = Field(
        default_factory=list, description="Direct pattern values"
    )
    match_type: t.Literal["exact", "starts_with", "contains", "ends_with", "regex"] = (
        "exact"
    )
    case_sensitive: bool = True
    match_any: bool = True

    # @model_validator(mode="after")
    # def validate_threshold_or_variable(self) -> t_ext.Self:
    #     if not self.match_any and len(self.variables):
    #         raise ValueError("String match node only supports variables on match_any.")
    #     if len(self.patterns) == 0 and len(self.variables) != 1:
    #         raise ValueError(
    #             "One or more variables must be given if no patterns are provided."
    #         )
    #     return self


class LeafNode(BaseModel):
    """Leaf node."""

    NODE_TYPE: t.ClassVar[str] = "leaf"
    node_type: t.Literal[NODE_TYPE] = NODE_TYPE  # type: ignore[valid-type]
    leaf_value: int = -1
    output_data: t.Optional[t.Dict[str, t.Any]] = None


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
