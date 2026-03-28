"""Data layer node types with variable support.

This module contains the data layer representations of tree nodes that match
the frontend structure, including support for variables. These nodes are used
for serialization, storage, and API communication before being compiled into
execution-ready nodes.
"""

import sys
import typing as t
import typing_extensions as t_ext
from logging import getLogger
from pydantic import BaseModel, Field, model_validator


from .core_node_data import (
    RangeNodeCoreData,
    NumericalNodeCoreData,
    CategoricalNodeCoreData,
    StringMatchNodeCoreData,
    LeafNodeCoreData,
)
from .variable_types import VariableType

# newer python order is guaranteed with dict
if sys.version_info > (3, 7):
    OrderedDict = dict
else:
    from collections import OrderedDict

logger = getLogger(__name__)

if t.TYPE_CHECKING:
    from .tree import Tree
    from .compiled_node_types import (
        RangeTestNode,
        NumericalTestNode,
        CategoricalTestNode,
        StringMatchNode as CompiledStringMatchNode,
        LeafNode as CompiledLeafNode,
    )


class HasVariablesMixin:
    variables: t.List[str] = Field(
        default_factory=list, description="Variable IDs used by this node"
    )


# Data layer nodes with compilation methods
class RangeNode(RangeNodeCoreData):
    """Range test node with compilation capability."""

    def compile(self, tree: "Tree", **kwargs) -> "RangeTestNode":
        """Compile this data node into an execution node."""
        from .compiled_node_types import RangeTestNode

        return RangeTestNode.model_validate(
            {
                **self.model_dump(exclude={"variables"}),
                "thresholds": sorted(self.thresholds),
            }
        )


class NumericalNode(HasVariablesMixin, NumericalNodeCoreData):
    """Numerical test node with compilation capability."""

    threshold: t.Optional[float] = Field(
        description="Direct threshold value", default=None
    )

    @model_validator(mode="after")
    def validate_threshold_or_variable(self) -> t_ext.Self:
        if self.threshold is None and len(self.variables) != 1:
            raise ValueError(
                f"A single variable must be provided if no threshold is given"
            )
        return self

    def compile(self, tree: "Tree", **kwargs) -> "NumericalTestNode":
        """Compile this data node into an execution node."""
        if len(self.variables):
            assert (
                len(self.variables) == 1
            ), "Numerical Node expects exactly 0 or 1 variables"
            var_key = self.variables[0]
            var = tree.variables[var_key]
            assert (
                var.var_type == VariableType.NUMERIC
            ), "Only numeric types can be used for numeric test nodes"

            from .compiled_node_types import NumericalTestNodeWithVariables

            return NumericalTestNodeWithVariables(
                **self.model_dump(exclude={"threshold", "variables"}),
                threshold=0,
                variable_key=var.name,
                default_variable_value=var.value,
            )

        from .compiled_node_types import NumericalTestNode

        return NumericalTestNode.model_validate(
            {**self.model_dump(exclude={"variables"})}
        )


class CategoricalNode(HasVariablesMixin, CategoricalNodeCoreData):
    """Categorical test node with compilation capability."""

    @model_validator(mode="after")
    def validate_threshold_or_variable(self) -> t_ext.Self:
        if len(self.category_list) == 0 and len(self.variables) != 1:
            raise ValueError(
                f"One or more variables must be given if no categories are provided."
            )
        return self

    def compile(self, tree: "Tree", **kwargs) -> "CategoricalTestNode":
        """Compile this data node into an execution node."""
        from .compiled_node_types import (
            CategoricalTestNodeWithVariables,
            CategoricalTestNode,
        )

        if len(self.variables):
            variable_values = {}
            for var_id in self.variables:
                var = tree.variables[var_id]
                assert (
                    var.var_type == VariableType.NUMERIC
                ), f"Categorical node only supports numeric var type. However {var.name} is {var.var_type}"
                int_value = int(var.value)
                assert (
                    int_value == var.value
                ), f"Categorical nodes only support integer values. {var.name} has value of {var.value}"
                variable_values[var.name] = int_value

            return CategoricalTestNodeWithVariables.model_validate(
                {
                    **self.model_dump(exclude={"variables"}),
                    "variable_values": variable_values,
                }
            )

        return CategoricalTestNode.model_validate(
            self.model_dump(exclude={"variables"})
        )


class StringMatchNode(HasVariablesMixin, StringMatchNodeCoreData):
    """String match node with compilation capability."""

    @model_validator(mode="after")
    def validate_threshold_or_variable(self) -> t_ext.Self:
        if not self.match_any and len(self.variables):
            raise ValueError(f"String match node only supports variables on match_any.")
        if len(self.patterns) == 0 and len(self.variables) != 1:
            raise ValueError(
                f"One or more variables must be given if no patterns are provided."
            )
        return self

    def compile(self, tree: "Tree", **kwargs) -> "CompiledStringMatchNode":
        """Compile this data node into an execution node."""
        from .compiled_node_types import (
            StringMatchNode as CompiledStringMatchNode,
            StringMatchNodeWithVariables,
        )

        if len(self.variables):
            variable_values = {}
            for var_id in self.variables:
                var = tree.variables[var_id]
                assert (
                    var.var_type == VariableType.STRING
                ), f"String node only supports string var type. However {var.name} is {var.var_type}"
                variable_values[var.name] = var.value

                return StringMatchNodeWithVariables.model_validate(
                    {
                        **self.model_dump(exclude={"variables"}),
                        "variable_values": variable_values,
                    }
                )

        return CompiledStringMatchNode.model_validate(
            self.model_dump(exclude={"variables"})
        )


class LeafNode(LeafNodeCoreData):
    """Leaf node with compilation capability."""

    output_data: t.Optional[t.Dict[str, t.Any]] = None

    def compile(
        self,
        tree: "Tree",
        node_id: str,
        output_leaf_map: t.Dict[str, int],
        node_name_mapping: t.Dict[str, str],
        **kwargs,
    ) -> "CompiledLeafNode":
        """Compile this data node (leaf nodes don't change)."""
        from .compiled_node_types import LeafNode as CompiledLeafNode

        return CompiledLeafNode(
            **self.model_dump(exclude=["leaf_value"]),
            leaf_value=output_leaf_map[node_name_mapping.get(node_id, node_id)],
        )


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
