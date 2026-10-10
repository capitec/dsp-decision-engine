import typing as t
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class OperationType(str, Enum):
    """Types of operations that can be performed on a project"""

    # Checkpoint operations (not undoable)
    CHECKPOINT_CREATED = "checkpoint_created"

    # Node operations (undoable)
    CREATE_NODE = "create_node"
    DELETE_NODE = "delete_node"
    UPDATE_NODE = "update_node"
    MOVE_NODE = "move_node"

    # Edge operations (undoable)
    CREATE_EDGE = "create_edge"
    DELETE_EDGE = "delete_edge"
    UPDATE_EDGE = "update_edge"

    # Feature operations (undoable)
    ADD_FEATURE = "add_feature"
    REMOVE_FEATURE = "remove_feature"
    RENAME_FEATURE = "rename_feature"

    # Tree output operations (undoable)
    UPDATE_TREE_OUTPUT = "update_tree_output"

    # Bulk operations (creates checkpoint boundary)
    BULK_STATE_UPDATE = "bulk_state_update"  # Legacy - use checkpoint_created instead


class BaseOperation(BaseModel):
    """Base operation model"""

    operation_type: OperationType
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000)
    )
    user_id: str
    is_undoable: bool = True


class CheckpointOperation(BaseOperation):
    """Operation marking a checkpoint creation"""

    operation_type: t.Literal[OperationType.CHECKPOINT_CREATED] = (
        OperationType.CHECKPOINT_CREATED
    )
    checkpoint_id: str
    revision_number: int
    validation_result: t.Dict[str, t.Any] = Field(default_factory=dict)
    is_undoable: bool = False  # Checkpoints are not undoable


class CreateNodeOperation(BaseOperation):
    """Operation for creating a new node"""

    operation_type: t.Literal[OperationType.CREATE_NODE] = OperationType.CREATE_NODE
    node_id: str
    node_data: t.Dict[str, t.Any]
    position: t.Dict[str, float]  # {x, y}


class DeleteNodeOperation(BaseOperation):
    """Operation for deleting a node"""

    operation_type: t.Literal[OperationType.DELETE_NODE] = OperationType.DELETE_NODE
    node_id: str
    # Store node data for undo
    node_data: t.Dict[str, t.Any]
    position: t.Dict[str, float]
    connected_edges: t.List[str]  # Edge IDs that were connected


class UpdateNodeOperation(BaseOperation):
    """Operation for updating node properties"""

    operation_type: t.Literal[OperationType.UPDATE_NODE] = OperationType.UPDATE_NODE
    node_id: str
    field: str  # Which field was updated (e.g., "threshold", "comparison_op")
    old_value: t.Any
    new_value: t.Any


class MoveNodeOperation(BaseOperation):
    """Operation for moving a node"""

    operation_type: t.Literal[OperationType.MOVE_NODE] = OperationType.MOVE_NODE
    node_id: str
    old_position: t.Dict[str, float]  # {x, y}
    new_position: t.Dict[str, float]  # {x, y}


class CreateEdgeOperation(BaseOperation):
    """Operation for creating a new edge"""

    operation_type: t.Literal[OperationType.CREATE_EDGE] = OperationType.CREATE_EDGE
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_data: t.Dict[str, t.Any]


class DeleteEdgeOperation(BaseOperation):
    """Operation for deleting an edge"""

    operation_type: t.Literal[OperationType.DELETE_EDGE] = OperationType.DELETE_EDGE
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_data: t.Dict[str, t.Any]  # Store for undo


class UpdateEdgeOperation(BaseOperation):
    """Operation for updating edge properties"""

    operation_type: t.Literal[OperationType.UPDATE_EDGE] = OperationType.UPDATE_EDGE
    edge_id: str
    field: str
    old_value: t.Any
    new_value: t.Any


class AddFeatureOperation(BaseOperation):
    """Operation for adding a feature"""

    operation_type: t.Literal[OperationType.ADD_FEATURE] = OperationType.ADD_FEATURE
    feature_name: str
    feature_index: int


class RemoveFeatureOperation(BaseOperation):
    """Operation for removing a feature"""

    operation_type: t.Literal[OperationType.REMOVE_FEATURE] = (
        OperationType.REMOVE_FEATURE
    )
    feature_name: str
    feature_index: int  # Store index for undo


class RenameFeatureOperation(BaseOperation):
    """Operation for renaming a feature"""

    operation_type: t.Literal[OperationType.RENAME_FEATURE] = (
        OperationType.RENAME_FEATURE
    )
    feature_index: int
    old_name: str
    new_name: str


class UpdateTreeOutputOperation(BaseOperation):
    """Operation for updating tree output configuration"""

    operation_type: t.Literal[OperationType.UPDATE_TREE_OUTPUT] = (
        OperationType.UPDATE_TREE_OUTPUT
    )
    field: str  # e.g., "columns", "data", "displayColumn"
    old_value: t.Any
    new_value: t.Any


# Union type for all operations
Operation = t.Union[
    CheckpointOperation,
    CreateNodeOperation,
    DeleteNodeOperation,
    UpdateNodeOperation,
    MoveNodeOperation,
    CreateEdgeOperation,
    DeleteEdgeOperation,
    UpdateEdgeOperation,
    AddFeatureOperation,
    RemoveFeatureOperation,
    RenameFeatureOperation,
    UpdateTreeOutputOperation,
]


class OperationInverter:
    """Handles inversion of operations for undo functionality"""

    def invert_operation(self, operation: Operation) -> t.Optional[Operation]:
        """
        Create the inverse operation for undo functionality
        Returns None for operations that cannot be undone
        """
        if not operation.is_undoable:
            return None

        op_type = operation.operation_type

        if op_type == OperationType.CREATE_NODE:
            op = t.cast(CreateNodeOperation, operation)
            return DeleteNodeOperation(
                user_id=op.user_id,
                node_id=op.node_id,
                node_data=op.node_data,
                position=op.position,
                connected_edges=[],  # Would need to be computed
            )

        elif op_type == OperationType.DELETE_NODE:
            op = t.cast(DeleteNodeOperation, operation)
            return CreateNodeOperation(
                user_id=op.user_id,
                node_id=op.node_id,
                node_data=op.node_data,
                position=op.position,
            )

        elif op_type == OperationType.UPDATE_NODE:
            op = t.cast(UpdateNodeOperation, operation)
            return UpdateNodeOperation(
                user_id=op.user_id,
                node_id=op.node_id,
                field=op.field,
                old_value=op.new_value,  # Swap old/new
                new_value=op.old_value,
            )

        elif op_type == OperationType.MOVE_NODE:
            op = t.cast(MoveNodeOperation, operation)
            return MoveNodeOperation(
                user_id=op.user_id,
                node_id=op.node_id,
                old_position=op.new_position,  # Swap old/new
                new_position=op.old_position,
            )

        # TODO: Add more operation inversions as needed
        # For now, return None for unsupported operations
        return None


# Global inverter instance
operation_inverter = OperationInverter()
