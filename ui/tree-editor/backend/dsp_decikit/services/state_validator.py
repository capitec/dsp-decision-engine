import typing as t
import json
import logging
from datetime import datetime, timezone

try:
    import jsonpatch

    JSONPATCH_AVAILABLE = True
except ImportError:
    JSONPATCH_AVAILABLE = False
    jsonpatch = None

logger = logging.getLogger(__name__)


class StateValidator:
    """Validates consistency between operation chains and checkpoint states"""

    def __init__(self):
        if not JSONPATCH_AVAILABLE:
            logger.warning(
                "jsonpatch not available - install with: pip install jsonpatch"
            )

    def reconstruct_state_from_operations(
        self, base_state: t.Dict, operations: t.List[t.Dict]
    ) -> t.Dict:
        """Reconstruct state by applying operations to base state"""
        current_state = self._deep_copy(base_state)

        for operation in operations:
            op_type = operation.get("operation_type")

            if op_type == "bulk_state_update":
                # For bulk updates, replace entire state
                current_state = operation.get("data", {})

            elif op_type == "checkpoint_created":
                # Checkpoint operations don't change state, they're markers
                continue

            # TODO: Add other operation types
            # elif op_type == 'move_node':
            #     self._apply_move_node(current_state, operation)
            # elif op_type == 'update_threshold':
            #     self._apply_threshold_update(current_state, operation)

        return current_state

    def validate_checkpoint_consistency(
        self, base_state: t.Dict, operations: t.List[t.Dict], checkpoint_state: t.Dict
    ) -> t.Dict:
        """
        Validate that operations + base_state equals checkpoint_state
        Returns validation result with diff information
        """
        # Reconstruct state from operations
        reconstructed_state = self.reconstruct_state_from_operations(
            base_state, operations
        )

        # Generate JSON diff
        diff_result = self.generate_json_diff(reconstructed_state, checkpoint_state)

        # Create validation result
        validation_result = {
            "is_consistent": (
                len(diff_result["patch"]) == 0 if JSONPATCH_AVAILABLE else True
            ),
            "reconstructed_state": reconstructed_state,
            "checkpoint_state": checkpoint_state,
            "diff_patch": diff_result["patch"],
            "diff_summary": diff_result["summary"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operations_count": len(operations),
        }

        if not validation_result["is_consistent"]:
            logger.warning(f"State inconsistency detected: {diff_result['summary']}")

        return validation_result

    def generate_json_diff(self, source_state: t.Dict, target_state: t.Dict) -> t.Dict:
        """Generate JSON diff between two states"""
        if not JSONPATCH_AVAILABLE:
            return {
                "patch": [],
                "summary": "jsonpatch not available - cannot generate diff",
                "has_differences": False,
            }

        try:
            # Create JSON patch from source to target
            patch = jsonpatch.make_patch(source_state, target_state)
            patch_ops = list(patch)

            # Create human-readable summary
            summary = self._create_diff_summary(patch_ops)

            return {
                "patch": patch_ops,
                "summary": summary,
                "has_differences": len(patch_ops) > 0,
            }

        except Exception as e:
            logger.error(f"Error generating JSON diff: {e}")
            return {
                "patch": [],
                "summary": f"Error generating diff: {e}",
                "has_differences": False,
            }

    def apply_diff_patch(
        self, source_state: t.Dict, patch_ops: t.List[t.Dict]
    ) -> t.Dict:
        """Apply JSON patch operations to source state"""
        if not JSONPATCH_AVAILABLE or not patch_ops:
            return source_state

        try:
            patch = jsonpatch.JsonPatch(patch_ops)
            return patch.apply(source_state)
        except Exception as e:
            logger.error(f"Error applying JSON patch: {e}")
            return source_state

    def _create_diff_summary(self, patch_ops: t.List[t.Dict]) -> str:
        """Create human-readable summary of patch operations"""
        if not patch_ops:
            return "No differences"

        summary_parts = []
        for op in patch_ops:
            op_type = op.get("op", "unknown")
            path = op.get("path", "")

            if op_type == "add":
                summary_parts.append(f"Added {path}")
            elif op_type == "remove":
                summary_parts.append(f"Removed {path}")
            elif op_type == "replace":
                summary_parts.append(f"Changed {path}")
            elif op_type == "move":
                summary_parts.append(f"Moved {path}")
            elif op_type == "copy":
                summary_parts.append(f"Copied {path}")

        return "; ".join(summary_parts)

    def _deep_copy(self, obj: t.Any) -> t.Any:
        """Deep copy object using JSON serialization"""
        try:
            return json.loads(json.dumps(obj))
        except (TypeError, ValueError):
            # Fallback for non-serializable objects
            return obj

    # TODO: Implement specific operation appliers
    # def _apply_move_node(self, state: t.Dict, operation: t.Dict):
    #     """Apply move_node operation to state"""
    #     pass
    #
    # def _apply_threshold_update(self, state: t.Dict, operation: t.Dict):
    #     """Apply threshold update operation to state"""
    #     pass


# Global instance
state_validator = StateValidator()
