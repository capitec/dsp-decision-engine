from spockflow.nodes import VariableNode
import pandas as pd
import typing as t
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from .nodes import NodeData, PositionedNode
from .variables import VariableMap, PlaceHolderVariable
from .edges import MultiSourceEdge
from .schema import OutputSchema
from logging import getLogger


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
    order: int  # Frontend uses "order" instead of "priority"
    rootNodeId: str
    isActive: t.Optional[bool] = None
    hidden: t.Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def translate_priority_to_order(cls, values: t.Any) -> t.Any:
        """Convert priority to order for backward compatibility."""
        if isinstance(values, dict):
            # If we have priority but no order, translate it
            if "priority" in values and "order" not in values:
                values = values.copy()  # Don't mutate original
                values["order"] = values["priority"]
                # Remove priority from the dict since we don't have it as a field
                values.pop("priority", None)
        return values

    @property
    def priority(self) -> int:
        """Get priority for compatibility with older versions."""
        return self.order


class Tree(VariableNode):
    features: t.List[str]
    nodes: t.List[PositionedNode]
    metadata: TreeMetadata | None = None
    tree_output: TreeOutput | None = Field(alias="treeOutput", default=None)
    edges: t.List[MultiSourceEdge]
    subtrees: t.List[SubTree] = Field(default_factory=list)
    variables: VariableMap = Field(
        default_factory=dict, description="Variable definitions keyed by ID"
    )
    output_schema: t.Optional[OutputSchema] = Field(
        alias="outputSchema",
        default=None,
        description="Structured output schema with global types support",
    )
    node_output_format_version: int = Field(
        alias="nodeOutputFormatVersion",
        default=-1,
        description="Output format version: 0=legacy table, 1=global types schema -1 Try infer",
    )
    format_version: t.Literal[1] = Field(alias="formatVersion", default=1)
    variable_input_name: str = "variables"
    _node_lookup: t.Dict[str, NodeData] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_output(self) -> "Tree":
        if self.node_output_format_version == -1:
            if self.tree_output is not None:
                self.node_output_format_version = 0
            elif self.output_schema is not None:
                self.node_output_format_version = 1
            else:
                raise ValueError(
                    f"If node_output_format_version is not specified either tree_output or output_schema must be provided"
                )

        if self.node_output_format_version == 0:
            assert (
                self.tree_output is not None
            ), "Tree output must be supplied for v0 node output format"
        elif self.node_output_format_version == 1:
            assert (
                self.output_schema is not None
            ), "Output Schema must be supplied for v1 node output format"
            node_issues = []
            for n in self.nodes:
                if n.data.node_type == "leaf":
                    if n.data.output_data is None:
                        if not self.output_schema.has_default_values():
                            node_issues.append(
                                f"Output data of node: {n.id} is missing and no default value is set."
                            )
                    else:
                        validation_errors = self.output_schema.validate_data(
                            n.data.output_data
                        )
                        node_issues.extend(
                            [f"Node {n.id}: {error}" for error in validation_errors]
                        )

            if node_issues:
                raise ValueError(
                    f"Tree output validation failed:\n" + "\n".join(node_issues)
                )

        return self


    def upgrade(self):
        """V1 is the latest version, return self"""
        return self

    def get_variables(self) -> VariableMap:
        """Get all variables defined in this tree."""
        return self.variables

    def get_variable(self, variable_id: str) -> t.Optional[PlaceHolderVariable]:
        """Get a specific variable by ID."""
        return self.variables.get(variable_id)

    def add_variable(self, variable: PlaceHolderVariable) -> None:
        """Add a variable to the tree."""
        self.variables[variable.id] = variable

    def remove_variable(self, variable_id: str) -> bool:
        """Remove a variable from the tree. Returns True if variable existed."""
        return self.variables.pop(variable_id, None) is not None

    def verify_tree(self, results: "ValidationResults") -> None:
        """
        Perform comprehensive tree-specific verification checks for V1 trees.
        Updates the ValidationResults object with findings.
        """
        from .util import _build_child_mapping

        # Step 1: Check for unused variables
        if self.variables:
            # Collect all variable references from nodes
            used_variables = set()
            for node in self.nodes:
                if hasattr(node.data, "variables") and node.data.variables:
                    used_variables.update(node.data.variables)

            # Find unused variables
            defined_variables = set(self.variables.keys())
            unused_variables = defined_variables - used_variables

            for var in unused_variables:
                results.add_warning(f"Unused variable: {var}")

        # Step 2: Individual node compilation validation
        output_df, output_map = self.output_df_and_map()
        child_map = _build_child_mapping(self.edges)
        all_children = set()
        for targets in child_map.values():
            all_children.update(targets.values())
        all_roots = {n.id for n in self.nodes} - all_children
        self._validate_individual_node_compilation(
            results, output_df, output_map, all_roots
        )

        # Step 3: Check for leaf nodes without outputs
        for node in self.nodes:
            if node.data.node_type == "leaf":
                # For V1 format with output_schema, check if node has output_data
                if self.node_output_format_version == 1:
                    if node.data.output_data is None and (
                        self.output_schema is None
                        or not self.output_schema.has_default_values()
                    ):
                        results.add_node_warning(
                            node.id,
                            "Leaf node has no output data and no default value is set",
                            node.data.node_type,
                        )
                # For V0 format with tree_output, check if node has leaf_value
                elif self.node_output_format_version == 0:
                    if (
                        not hasattr(node.data, "leaf_value")
                        or node.data.leaf_value is None
                    ):
                        results.add_node_warning(
                            node.id, "Leaf node has no leaf_value", node.data.node_type
                        )

        # Step 4: Check for unconnected leaf nodes
        self._check_unconnected_leaf_nodes(results)

    def _validate_individual_node_compilation(
        self, results: "ValidationResults", output_df, output_leaf_map, all_roots
    ) -> None:
        """Validate each node individually like the compiler does."""

        # Create a temporary output map for compilation

        for node in self.nodes:
            try:
                # Try to compile each node individually
                compiled_node = node.data.compile(
                    tree=self,
                    output_leaf_map=output_leaf_map,
                    node_id=node.id,
                    node_name_mapping={node.id: node.id},
                )

                # For decision nodes, check if they have the expected number of edges
                if compiled_node.IS_LEAF and node.id in all_roots:
                    results.add_node_warning(
                        node.id,
                        f"Leaf node is a root node; typically root nodes should be decision nodes",
                        node.data.node_type,
                    )

            except Exception as e:
                results.add_node_error(
                    node.id, f"Node compilation failed: {str(e)}", node.data.node_type
                )

    def _check_unconnected_leaf_nodes(self, results: "ValidationResults") -> None:
        """Check for leaf nodes that have no incoming edges."""
        # Build set of nodes that are targets of edges (have incoming edges)
        nodes_with_incoming_edges = set()
        for edge in self.edges:
            if hasattr(edge, "target"):
                nodes_with_incoming_edges.add(edge.target)
            elif hasattr(edge, "targets"):  # MultiSourceEdge
                nodes_with_incoming_edges.update(edge.targets)

        # Find leaf nodes without incoming edges (except root nodes in subtrees)
        subtree_roots = {subtree.rootNodeId for subtree in self.subtrees}

        for node in self.nodes:
            if (
                node.data.node_type == "leaf"
                and node.id not in nodes_with_incoming_edges
                and node.id not in subtree_roots
            ):
                results.add_node_warning(
                    node.id,
                    "Leaf node is not connected to any decision path",
                    node.data.node_type,
                )
