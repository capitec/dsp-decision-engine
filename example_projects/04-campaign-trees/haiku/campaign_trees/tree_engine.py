"""Tree evaluation engine with path capture."""

from typing import Dict, Any, List, Optional, Tuple
from datetime import date
from campaign_trees.types import TreeLeaf, TreeNode, TreeEvaluationResult


class TreeEngine:
    """Evaluates campaign trees with path capture."""

    def evaluate(
        self,
        tree: Dict[str, Any],
        client_id: int,
        campaign_id: int,
        features: Dict[str, Any],
        cycle_date: date,
        decision_id: str,
    ) -> TreeEvaluationResult:
        """
        Evaluate a tree for a client, capturing the path taken.

        Args:
            tree: Tree definition with nodes and edges
            client_id: Client identifier
            campaign_id: Campaign identifier
            features: Client feature values
            cycle_date: Decision date
            decision_id: Decision record ID

        Returns:
            TreeEvaluationResult with path and leaf outcome
        """
        # Start at root node
        current_node_key = tree.get("root_key", "node_1")
        path = []
        path_nodes = []

        # Traverse tree nodes until leaf
        max_depth = tree.get("max_depth", 12)
        for _ in range(max_depth):
            node_data = tree.get("nodes", {}).get(current_node_key)
            if not node_data:
                break

            path.append(current_node_key)
            path_nodes.append(
                TreeNode(
                    key=current_node_key,
                    level=node_data.get("level", 0),
                    condition=node_data.get("condition", ""),
                    features=node_data.get("features", []),
                )
            )

            # If leaf, return result
            if node_data.get("is_leaf", False):
                leaf = self._build_leaf(node_data)
                return TreeEvaluationResult(
                    client_id=client_id,
                    campaign_id=campaign_id,
                    tree_version=tree.get("version", 1),
                    leaf=leaf,
                    path=path,
                    path_nodes=path_nodes,
                    overlay_stack_id=0,
                    decision_id=decision_id,
                    cycle_date=cycle_date,
                )

            # Evaluate condition to determine next node
            condition_held = self._evaluate_condition(
                node_data.get("condition_expr", "true"),
                features,
            )

            # Move to next node based on condition
            if condition_held:
                current_node_key = node_data.get("true_branch", "leaf_default")
            else:
                current_node_key = node_data.get("false_branch", "leaf_default")

        # Fallback: return default leaf (should not happen with valid tree)
        default_leaf = TreeLeaf(
            key="leaf_default",
            outcome_code=3,  # Do not target
            tier_code=0,
            amount_rule="0",
            channels=[],
            priority_weight=0.0,
            reason_label=999,
            reason_text="Tree traversal error",
        )

        return TreeEvaluationResult(
            client_id=client_id,
            campaign_id=campaign_id,
            tree_version=tree.get("version", 1),
            leaf=default_leaf,
            path=path,
            path_nodes=path_nodes,
            overlay_stack_id=0,
            decision_id=decision_id,
            cycle_date=cycle_date,
        )

    def _evaluate_condition(
        self,
        condition_expr: str,
        features: Dict[str, Any],
    ) -> bool:
        """
        Evaluate a condition expression against client features.

        Supports: >, <, >=, <=, ==, in
        Examples: "age >= 25", "income > 5000", "code in [1,2,3]"
        """
        try:
            # Simple condition parsing
            if " in " in condition_expr:
                parts = condition_expr.split(" in ")
                feature_name = parts[0].strip()
                codes_str = parts[1].strip()
                # Parse [1,2,3] format
                codes = [
                    int(x.strip())
                    for x in codes_str.strip("[]").split(",")
                ]
                return features.get(feature_name, 0) in codes

            # Comparison operators
            for op in [">=", "<=", ">", "<", "=="]:
                if op in condition_expr:
                    parts = condition_expr.split(op)
                    feature_name = parts[0].strip()
                    threshold = float(parts[1].strip())
                    value = features.get(feature_name, 0)

                    if op == ">=":
                        return value >= threshold
                    elif op == "<=":
                        return value <= threshold
                    elif op == ">":
                        return value > threshold
                    elif op == "<":
                        return value < threshold
                    elif op == "==":
                        return value == threshold

            # Default: true
            return True
        except Exception:
            # On parse error, return false (conservative)
            return False

    def _build_leaf(self, node_data: Dict[str, Any]) -> TreeLeaf:
        """Build a TreeLeaf from node data."""
        return TreeLeaf(
            key=node_data.get("key", "unknown"),
            outcome_code=node_data.get("outcome_code", 3),
            tier_code=node_data.get("tier_code", 0),
            amount_rule=node_data.get("amount_rule", "0"),
            channels=node_data.get("channels", []),
            priority_weight=node_data.get("priority_weight", 0.5),
            reason_label=node_data.get("reason_label", 0),
            reason_text=node_data.get("reason_text", ""),
        )
