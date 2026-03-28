import typing as t
from pydantic import BaseModel, Field


class TestNodeData(BaseModel):
    """Base class for test node data matching frontend structure."""

    split_feature_id: int
    default_left: bool = False


class RangeNodeCoreData(TestNodeData):
    """Range test node data layer."""

    NODE_TYPE: t.ClassVar[str] = "numerical_range_test_node"
    node_type: t.Literal["numerical_range_test_node"] = NODE_TYPE
    thresholds: t.List[float] = Field(
        default_factory=list, description="Direct threshold values"
    )


class NumericalNodeCoreData(TestNodeData):
    """Numerical test node data layer."""

    NODE_TYPE: t.ClassVar[str] = "numerical_test_node"
    node_type: t.Literal["numerical_test_node"] = NODE_TYPE
    threshold: float = Field(description="Direct threshold value")
    comparison_op: t.Literal["<=", "<", "==", ">", ">="] = "<="


class CategoricalNodeCoreData(TestNodeData):
    """Categorical test node data layer."""

    NODE_TYPE: t.ClassVar[str] = "categorical_test_node"
    node_type: t.Literal["categorical_test_node"] = NODE_TYPE
    category_list: t.List[int] = Field(
        default_factory=list, description="Direct category values"
    )
    # Basically if the value should be in or not in set
    # The ui defaults to left for items in category and then right otherwise
    category_list_right_child: bool = False


class StringMatchNodeCoreData(TestNodeData):
    """String match node data layer."""

    NODE_TYPE: t.ClassVar[str] = "string_match_node"
    node_type: t.Literal["string_match_node"] = NODE_TYPE
    patterns: t.List[str] = Field(
        default_factory=list, description="Direct pattern values"
    )
    match_type: t.Literal["exact", "starts_with", "contains", "ends_with", "regex"] = (
        "exact"
    )
    case_sensitive: bool = True
    # If True: match any pattern (2 branches), If False: individual branches per pattern
    match_any: bool = True
    # For string it makes more sense to set default left to true because its generally more "falsy"
    default_left: bool = True


class LeafNodeCoreData(BaseModel):
    """Leaf node data layer."""

    NODE_TYPE: t.ClassVar[str] = "leaf"
    node_type: t.Literal["leaf"] = NODE_TYPE
    leaf_value: int = -1
