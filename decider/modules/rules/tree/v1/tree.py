import typing as t
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from .nodes import (
    NodeData,
    PositionedNode,
    LeafNode,
    RangeNode,
    NumericalNode,
    CategoricalNode,
    StringMatchNode,
)
from .variables import VariableMap, PlaceHolderVariable
from .edges import MultiSourceEdge, MultiEdgeData
from .schema import OutputSchema, FieldType
from logging import getLogger

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Output collection helpers
# ---------------------------------------------------------------------------


def _collect_outputs_v0(
    nodes: t.List[PositionedNode],
    tree_output: t.Optional["TreeOutput"],
) -> t.Tuple[t.List[t.Dict], t.Optional[t.Dict], t.Dict[str, int]]:
    """v0: outputs live in a flat table (treeOutput); leaf_value is the row index."""
    if tree_output is None:
        return [], None, {}

    collected_output = [dict(zip(tree_output.columns, row)) for row in tree_output.data]
    node_id_output_map = {
        node.id: node.data.leaf_value
        for node in nodes
        if isinstance(node.data, LeafNode)
    }
    return (
        collected_output[1:],
        collected_output[0] if len(collected_output) > 0 else None,
        node_id_output_map,
    )


def _collect_outputs_v1(
    nodes: t.List[PositionedNode],
    output_schema: t.Optional[OutputSchema],
) -> t.Tuple[t.List[t.Dict], t.Optional[t.Dict], t.Dict[str, int]]:
    """v1: output data is embedded in each leaf via output_data; deduplicate into a list."""
    outputs: t.List[t.Dict] = []
    # Use a stable key (sorted items) to deduplicate identical output dicts
    seen: t.Dict[str, int] = {}
    node_id_output_map: t.Dict[str, int] = {}

    for node in nodes:
        if not isinstance(node.data, LeafNode):
            continue
        if node.data.output_data is None:
            node_id_output_map[node.id] = -1
            continue
        key = str(sorted(node.data.output_data.items()))
        if key not in seen:
            seen[key] = len(outputs)
            outputs.append(node.data.output_data)
        node_id_output_map[node.id] = seen[key]

    default_value = (
        output_schema.default_values
        if output_schema and output_schema.has_default_values()
        else None
    )
    return outputs, default_value, node_id_output_map


# ---------------------------------------------------------------------------
# Schema conversion helper
# ---------------------------------------------------------------------------

_FIELD_TYPE_TO_POLARS: t.Dict[FieldType, str] = {
    FieldType.STRING: "String",
    FieldType.NUMBER: "Float64",
    FieldType.BOOLEAN: "Boolean",
}

_LIST_ITEM_TYPE_TO_POLARS: t.Dict[FieldType, str] = {
    FieldType.STRING: "String",
    FieldType.NUMBER: "Float64",
    FieldType.BOOLEAN: "Boolean",
}


_RECORD_FIELD_TYPE_TO_POLARS: t.Dict[str, str] = {
    "string": "String",
    "number": "Float64",
    "boolean": "Boolean",
}


def _record_to_polars_struct(record_def: "RecordDefinition") -> t.Dict[str, t.Any]:
    """Convert a v1 RecordDefinition to a nested PolarsSchema-compatible struct dict."""
    from .schema import RecordDefinition

    return {
        field_name: _RECORD_FIELD_TYPE_TO_POLARS.get(field_type, "String")
        for field_name, field_type in record_def.field_items()
    }


def _convert_custom_types_to_defs(output_schema: "OutputSchema") -> t.Dict[str, t.Any]:
    """Extract custom types from OutputSchema and convert to StructTypeDef/CategoricalTypeDef.

    Returns a dict mapping custom type ID to its type definition, suitable for the
    v2 TreeOutput.type_defs field.
    """
    from .schema import CustomTypeKind
    from ...serializable.dtypes import StructTypeDef, CategoricalTypeDef

    type_defs: t.Dict[str, t.Any] = {}

    for field in output_schema.fields:
        # Check if field has a custom type (either directly or as list item type)
        custom_type = field.custom_type if field.custom_type else None
        custom_type_id = custom_type.id if custom_type else None

        # For list fields, check if the list item type is custom
        if field.field_type == FieldType.LIST and field.list_type == FieldType.CUSTOM:
            custom_type_id = field.custom_type_id
            # Get the custom type from the field (it should be there based on schema)
            custom_type = field.custom_type

        if not custom_type or not custom_type_id:
            continue

        # Skip if we've already processed this type (multiple fields can reference same type)
        if custom_type_id in type_defs:
            continue

        if custom_type.type_kind == CustomTypeKind.RECORD:
            # Convert record definition to struct type def
            struct_def = _record_to_polars_struct(custom_type.definition)

            # Extract the actual record data
            records_data = custom_type.definition.records

            type_defs[custom_type_id] = StructTypeDef(
                name=custom_type.name,
                type="struct",
                definition={
                    "fields": struct_def,
                    "data": records_data,
                    "display_field": "id",
                },
            )
        elif custom_type.type_kind == CustomTypeKind.ENUM:
            # Convert enum definition to categorical type def
            type_defs[custom_type_id] = CategoricalTypeDef(
                name=custom_type.name,
                type="categorical",
                definition={"categories": custom_type.definition.values},
            )

    return type_defs


def _convert_schema(output_schema: "OutputSchema") -> t.Dict[str, t.Any]:
    """Convert a v1 OutputSchema to the dict format accepted by PolarsSchema.

    Custom types are now converted to type references (ExplicitType with type_id)
    instead of inlining their definitions.
    """
    from .schema import CustomTypeKind
    from ...serializable.schema import ExplicitType

    result: t.Dict[str, t.Any] = {}
    for field in output_schema.fields:
        if field.field_type in _FIELD_TYPE_TO_POLARS:
            result[field.field_name] = _FIELD_TYPE_TO_POLARS[field.field_type]
        elif field.field_type == FieldType.LIST:
            # Handle list of custom types
            if field.list_type == FieldType.CUSTOM and field.custom_type_id:
                item_type = ExplicitType(type="Custom", type_id=field.custom_type_id)
            else:
                item_type = _LIST_ITEM_TYPE_TO_POLARS.get(field.list_type, "String")
            result[field.field_name] = ExplicitType(type="List", inner=item_type)
        elif field.field_type == FieldType.CUSTOM:
            # Custom types are now referenced by ID instead of inlined
            from .schema import CustomTypeKind

            if field.custom_type and field.custom_type.type_kind == CustomTypeKind.ENUM:
                # Reference the categorical type def
                result[field.field_name] = ExplicitType(
                    type="Custom", type_id=field.custom_type.id
                )
            elif (
                field.custom_type
                and field.custom_type.type_kind == CustomTypeKind.RECORD
            ):
                # Reference the struct type def
                result[field.field_name] = ExplicitType(
                    type="Custom", type_id=field.custom_type.id
                )
            else:
                result[field.field_name] = "String"
        else:
            result[field.field_name] = "String"
    return result


# ---------------------------------------------------------------------------
# Node upgrade helper
# ---------------------------------------------------------------------------


def _resolve_variable(var_id: str, variables: VariableMap) -> t.Any:
    """Resolve a variable ID to its value, with a warning if not found."""
    var = variables.get(var_id)
    if var is None:
        logger.warning(f"Variable ID '{var_id}' not found in variable map; using None.")
        return None
    return var.value


def _upgrade_node_data(
    data: NodeData,
    features: t.List[str],
    variables: VariableMap,
    node_id_output_map: t.Dict[str, int],
    node_id: str,
) -> NodeData:
    from ..v2.nodes import (
        RangeNode as V2RangeNode,
        NumericalNode as V2NumericalNode,
        CategoricalNode as V2CategoricalNode,
        StringMatchNode as V2StringMatchNode,
        LeafNode as V2LeafNode,
    )
    from ...common.shared import InputRef

    if isinstance(data, RangeNode):

        return V2RangeNode(
            feature=features[data.split_feature_id],
            default_left=data.default_left,
            thresholds=data.thresholds,  # already floats; no variable support in v1 RangeNode
        )

    if isinstance(data, NumericalNode):
        if data.threshold is not None:
            threshold: t.Union[float, InputRef] = data.threshold
        else:
            var_name = variables[data.variables[0]].name
            threshold = InputRef(key=var_name)
        return V2NumericalNode(
            feature=features[data.split_feature_id],
            default_left=data.default_left,
            comparison_op=data.comparison_op,
            threshold=threshold,
        )

    if isinstance(data, CategoricalNode):
        # Note: V2 CategoricalNode doesn't have category_list_right_child
        # We'll handle edge swapping in the upgrade() method
        if data.variables:
            if len(data.variables) > 1:
                logger.warning(
                    f"CategoricalNode with ID '{node_id}' has multiple variables; only the first will be used."
                )
            category_list: t.Union[InputRef, t.List[int]] = InputRef(
                key=variables[data.variables[0]].name
            )
        else:
            category_list = list(
                data.category_list
            )  # already a list of ints; no variable support in v1 CategoricalNode
        return V2CategoricalNode(
            feature=features[data.split_feature_id],
            category_list=category_list,
        )

    if isinstance(data, StringMatchNode):
        if data.variables:
            if len(data.variables) > 1:
                logger.warning(
                    f"StringMatchNode with ID '{node_id}' has multiple variables; only the first will be used."
                )
            patterns = InputRef(key=variables[data.variables[0]].name)
        else:
            patterns = (
                data.patterns
            )  # already a list of strings; no variable support in v1 StringMatchNode
        return V2StringMatchNode(
            feature=features[data.split_feature_id],
            default_left=data.default_left,
            patterns=patterns,
            match_type=data.match_type,
            case_sensitive=data.case_sensitive,
            match_any=data.match_any,
        )

    if isinstance(data, LeafNode):
        return V2LeafNode(
            leaf_value=node_id_output_map.get(node_id, -1),
        )

    raise ValueError(f"Unknown node type: {type(data)}")


def _upgrade_nodes(
    nodes: t.List[PositionedNode],
    features: t.List[str],
    variables: VariableMap,
    node_id_output_map: t.Dict[str, int],
) -> t.Tuple[t.List[PositionedNode], t.Set[str]]:
    """Upgrade nodes from v1 to v2 format.

    Returns:
        Tuple of (upgraded_nodes, nodes_needing_edge_swap)
        nodes_needing_edge_swap contains IDs of nodes that had category_list_right_child=True
    """
    from ..v2.nodes import PositionedNode as V2PositionedNode

    upgraded_nodes = []
    nodes_needing_edge_swap = set()

    for node in nodes:
        # Track categorical nodes with category_list_right_child=True
        if (
            isinstance(node.data, CategoricalNode)
            and node.data.category_list_right_child
        ):
            nodes_needing_edge_swap.add(node.id)

        upgraded_nodes.append(
            V2PositionedNode(
                id=node.id,
                position=node.position.model_dump(mode="python"),
                data=_upgrade_node_data(
                    node.data, features, variables, node_id_output_map, node.id
                ),
            )
        )

    return upgraded_nodes, nodes_needing_edge_swap


class TreeOutput(BaseModel):
    columns: t.List[str]
    data: t.List[t.List[str]]
    dtype: t.Optional[t.List[str]] = None


class TreeMetadata(BaseModel):
    name: t.Optional[str] = None
    description: t.Optional[str] = None


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


class Tree(BaseModel):
    type: t.Literal["ui-tree"] = "ui-tree"
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

    def upgrade(self):
        """Upgrade to the latest v2 tree format."""
        from ..v2.tree import (
            Tree as V2Tree,
            SubTree as V2SubTree,
            TreeOutput as V2TreeOutput,
        )
        from ...serializable.schema import PolarsSchema

        keyed_nodes = {n.id: n for n in self.nodes}

        # Determine which output collection strategy to use
        use_v0 = self.node_output_format_version == 0 or (
            self.node_output_format_version == -1 and self.tree_output is not None
        )

        if use_v0:
            collected_output, default_value, node_id_output_map = _collect_outputs_v0(
                self.nodes, self.tree_output
            )
            # v0 format: infer schema from collected output data
            # If there's data, infer from first row; otherwise use empty struct
            _schema_collected_output = (
                collected_output + [default_value]
                if default_value
                else collected_output
            )
            if _schema_collected_output and len(_schema_collected_output) > 0:
                from ...serializable.schema import convert_schema
                import polars as pl

                # Infer schema from the first output row
                temp_df = pl.DataFrame([_schema_collected_output[0]])
                dtypes_struct = convert_schema(temp_df.schema)
            else:
                dtypes_struct = {}  # Empty struct
            type_defs = {}
        else:
            from ...serializable.schema import ExplicitType

            collected_output, default_value, node_id_output_map = _collect_outputs_v1(
                self.nodes, self.output_schema
            )
            if self.output_schema:
                dtypes_struct = _convert_schema(self.output_schema)
                # Extract custom type definitions from the output schema
                type_defs = _convert_custom_types_to_defs(self.output_schema)
            else:
                # No schema: infer from collected output data
                if collected_output and len(collected_output) > 0:
                    from ...serializable.schema import convert_schema
                    import polars as pl

                    temp_df = pl.DataFrame([collected_output[0]])
                    dtypes_struct = convert_schema(temp_df.schema)
                else:
                    dtypes_struct = {}  # Empty struct
                type_defs = {}

            for field_key, field_dtype in dtypes_struct.items():
                if (
                    isinstance(field_dtype, ExplicitType)
                    and field_dtype.type == "Custom"
                ):
                    custom_type = type_defs[field_dtype.type_id]
                    if custom_type.type == "struct":
                        for row in collected_output:
                            # Handle case where value is already an index (int) or a dict
                            field_value = row[field_key]
                            if isinstance(field_value, dict):
                                row[field_key] = {
                                    "$key": field_value[
                                        custom_type.definition.display_field
                                    ]
                                }
                            # If it's already an int (index), leave it as is

        # Upgrade nodes and get the set of nodes that need edge swapping
        upgraded_nodes, nodes_needing_edge_swap = _upgrade_nodes(
            self.nodes, self.features, self.variables, node_id_output_map
        )

        # Swap edges for categorical nodes with category_list_right_child=True
        # In v2, matching categories always go to sourceIndex 0, non-matching to sourceIndex 1
        # In v1 with category_list_right_child=True, it was reversed
        upgraded_edges = []
        for edge in self.edges:
            if edge.source in nodes_needing_edge_swap:
                # Swap sourceIndex: 0 <-> 1
                swapped_indices = [1 - idx for idx in edge.data.sourceIndex]
                upgraded_edges.append(
                    MultiSourceEdge(
                        id=edge.id,
                        source=edge.source,
                        target=edge.target,
                        data=MultiEdgeData(sourceIndex=swapped_indices),
                    )
                )
            else:
                upgraded_edges.append(edge)

        input_schema = [[ft, "float32"] for ft in self.features]

        for node in self.nodes:
            if isinstance(node.data, NumericalNode):
                feature_name = self.features[node.data.split_feature_id]
                if feature_name not in input_schema:
                    input_schema[node.data.split_feature_id] = [feature_name, "float32"]
            elif isinstance(node.data, CategoricalNode):
                feature_name = self.features[node.data.split_feature_id]
                if feature_name not in input_schema:
                    input_schema[node.data.split_feature_id] = [feature_name, "string"]
            elif isinstance(node.data, StringMatchNode):
                feature_name = self.features[node.data.split_feature_id]
                if feature_name not in input_schema:
                    input_schema[node.data.split_feature_id] = [feature_name, "string"]

        # Convert v1 variables to v2 parameters
        from ...common.parameters import ParameterInfo
        from ...serializable.schema import PrimitiveSchema

        parameters = {}
        if self.variables:
            for var in self.variables.values():
                # Map v1 variable types to v2 parameter types
                if var.var_type == "numeric":
                    param_type = PrimitiveSchema(type="Float64")
                elif var.var_type == "string":
                    param_type = PrimitiveSchema(type="String")
                else:
                    logger.warning(
                        f"Unsupported variable type '{var.var_type}' for variable '{var.name}'; defaulting to String"
                    )
                    param_type = PrimitiveSchema(type="String")

                parameters[var.name] = ParameterInfo(
                    type=param_type, default_value=var.value
                )

        return V2Tree(
            metadata=self.metadata and self.metadata.model_dump(mode="python"),
            edges=upgraded_edges,
            nodes=upgraded_nodes,
            input_schema=input_schema,
            subtrees=[
                V2SubTree(id=s.rootNodeId, name=s.name)
                for s in (
                    sorted(self.subtrees, key=lambda st: st.order)
                    if self.subtrees
                    else []
                )
                if s.rootNodeId in keyed_nodes
            ],
            output=V2TreeOutput(
                data=collected_output,
                default=default_value,
                dtypes=dtypes_struct,
                type_defs=type_defs,
            ),
            parameters=parameters,
            parameters_col=self.variable_input_name,
        )
