import typing as t
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from .nodes import NodeData, PositionedNode, LeafNode, RangeNode, NumericalNode, CategoricalNode, StringMatchNode
from .variables import VariableMap, PlaceHolderVariable
from .edges import MultiSourceEdge
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

    collected_output = [
        dict(zip(tree_output.columns, row))
        for row in tree_output.data
    ]
    node_id_output_map = {
        node.id: node.data.leaf_value
        for node in nodes
        if isinstance(node.data, LeafNode)
    }
    return collected_output, None, node_id_output_map


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
        if node.data.leaf_value == -1 or node.data.output_data is None:
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


def _convert_schema(output_schema: OutputSchema) -> t.Dict[str, t.Any]:
    """Convert a v1 OutputSchema to the dict format accepted by PolarsSchema."""
    result: t.Dict[str, t.Any] = {}
    for field in output_schema.fields:
        if field.field_type in _FIELD_TYPE_TO_POLARS:
            result[field.field_name] = _FIELD_TYPE_TO_POLARS[field.field_type]
        elif field.field_type == FieldType.LIST:
            item_type = _LIST_ITEM_TYPE_TO_POLARS.get(field.list_type, "String")
            result[field.field_name] = {"List": item_type}
        elif field.field_type == FieldType.CUSTOM:
            # Enum → Categorical, Record → String (complex struct conversion out of scope)
            from .schema import CustomTypeKind
            if field.custom_type and field.custom_type.type_kind == CustomTypeKind.ENUM:
                result[field.field_name] = "Categorical"
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
        VariableReference,
    )

    if isinstance(data, RangeNode):
        return V2RangeNode(
            feature=features[data.split_feature_id],
            default_left=data.default_left,
            thresholds=data.thresholds,  # already floats; no variable support in v1 RangeNode
        )

    if isinstance(data, NumericalNode):
        if data.threshold is not None:
            threshold: t.Union[float, VariableReference] = data.threshold
        else:
            var_name = variables[data.variables[0]].name
            threshold = VariableReference(name=var_name)
        return V2NumericalNode(
            feature=features[data.split_feature_id],
            default_left=data.default_left,
            comparison_op=data.comparison_op,
            threshold=threshold,
        )

    if isinstance(data, CategoricalNode):
        if data.category_list:
            category_list: t.List = list(data.category_list)
        else:
            # Variable replaces the category list
            category_list = [VariableReference(name=variables[v].name) for v in data.variables]
        return V2CategoricalNode(
            feature=features[data.split_feature_id],
            default_left=data.default_left,
            category_list=category_list,
            category_list_right_child=data.category_list_right_child,
        )

    if isinstance(data, StringMatchNode):
        if data.patterns:
            patterns = list(data.patterns)
        else:
            # Variable provides the pattern; materialise its string value
            patterns = [str(_resolve_variable(v, variables)) for v in data.variables]
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
) -> t.List[PositionedNode]:
    from ..v2.nodes import PositionedNode as V2PositionedNode
    return [
        V2PositionedNode(
            id=node.id,
            position=node.position,
            data=_upgrade_node_data(node.data, features, variables, node_id_output_map, node.id),
        )
        for node in nodes
    ]


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


class Tree(BaseModel):
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
        from ..v2.tree import Tree as V2Tree
        from ped.serializable.schema import PolarsSchema

        keyed_nodes = {n.id: n for n in self.nodes}

        # Determine which output collection strategy to use
        use_v0 = (
            self.node_output_format_version == 0
            or (self.node_output_format_version == -1 and self.tree_output is not None)
        )

        if use_v0:
            collected_output, default_value, node_id_output_map = _collect_outputs_v0(
                self.nodes, self.tree_output
            )
            pl_schema = None
        else:
            collected_output, default_value, node_id_output_map = _collect_outputs_v1(
                self.nodes, self.output_schema
            )
            pl_schema = (
                PolarsSchema(root=_convert_schema(self.output_schema))
                if self.output_schema
                else None
            )

        return V2Tree(
            metadata=self.metadata,
            edges=self.edges,
            nodes=_upgrade_nodes(self.nodes, self.features, self.variables, node_id_output_map),
            subtrees=[
                s.rootNodeId
                for s in (sorted(self.subtrees, key=lambda st: st.order) if self.subtrees else [])
                if s.rootNodeId in keyed_nodes
            ],
            output=collected_output,
            schema=pl_schema,
            default=default_value,
        )
            

