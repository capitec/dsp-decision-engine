"""Tree definition generation."""

from typing import Dict, Any
import random


def generate_campaign_trees(
    campaign_id: int,
    tree_version: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Generate a deterministic campaign tree.

    Args:
        campaign_id: Campaign ID (used for determinism)
        tree_version: Version number of the tree
        seed: Seed for RNG

    Returns:
        Tree definition dict with nodes and structure
    """
    rng = random.Random(seed)

    # Tree depth and width: 4-12 levels, 20-400 nodes
    depth = rng.randint(4, 8)  # 4-8 levels for tractability
    node_count = rng.randint(20, 100)  # 20-100 nodes per tree

    tree = {
        "campaign_id": campaign_id,
        "version": tree_version,
        "root_key": "node_1",
        "max_depth": depth,
        "nodes": {},
        "node_identity_map": {},  # Maps old version keys to new (for version 2+)
    }

    # Generate internal nodes
    features = [
        "age", "tenure_months", "income_band", "behavior_score",
        "flex_balance", "payment_missed", "discretionary_income",
        "flex_term_remaining", "settlement_ratio", "employment_type"
    ]

    outcomes = [
        {"code": 1, "tier": 1, "amount": "min(pre_assessed, 250000)", "channels": ["app", "email"]},
        {"code": 1, "tier": 2, "amount": "min(pre_assessed, 150000)", "channels": ["sms", "app"]},
        {"code": 1, "tier": 3, "amount": "min(pre_assessed, 80000)", "channels": ["sms", "app"]},
        {"code": 3, "tier": 0, "amount": "0", "channels": [], "reason": "Below campaign floor"},
        {"code": 3, "tier": 0, "amount": "0", "channels": [], "reason": "Insufficient headroom"},
    ]

    # Build tree deterministically
    node_queue = [("node_1", 1, [])]  # (key, level, path_to_root)
    node_idx = 1

    while node_queue and node_idx < node_count:
        node_key, level, path_to_root = node_queue.pop(0)

        if level >= depth:
            # Create leaf at max depth
            outcome = outcomes[node_idx % len(outcomes)]
            tree["nodes"][node_key] = {
                "key": node_key,
                "level": level,
                "is_leaf": True,
                "outcome_code": outcome["code"],
                "tier_code": outcome["tier"],
                "amount_rule": outcome["amount"],
                "channels": outcome["channels"],
                "priority_weight": 0.5 + 0.3 * (node_idx % 10) / 10,
                "reason_label": 100 + (node_idx % 100),
                "reason_text": outcome.get("reason", "Targeted"),
            }
            continue

        # Internal node: choose feature and threshold
        feature = features[node_idx % len(features)]
        threshold = rng.randint(20, 80)

        tree["nodes"][node_key] = {
            "key": node_key,
            "level": level,
            "is_leaf": False,
            "features": [feature],
            "condition": f"{feature} > {threshold}",
            "condition_expr": f"{feature} > {threshold}",
            "true_branch": f"node_{node_idx + 1}",
            "false_branch": f"node_{node_idx + 2}",
        }

        # Add child nodes to queue
        node_idx += 1
        node_queue.append((f"node_{node_idx}", level + 1, path_to_root + [node_key]))
        node_idx += 1
        node_queue.append((f"node_{node_idx}", level + 1, path_to_root + [node_key]))

    # Ensure all branches terminate in leaves
    for node_key, node_data in list(tree["nodes"].items()):
        if not node_data.get("is_leaf", False):
            # Create leaves for missing branches
            if node_data.get("true_branch") not in tree["nodes"]:
                outcome = outcomes[hash(node_data["true_branch"]) % len(outcomes)]
                tree["nodes"][node_data["true_branch"]] = {
                    "key": node_data["true_branch"],
                    "level": node_data.get("level", 0) + 1,
                    "is_leaf": True,
                    "outcome_code": outcome["code"],
                    "tier_code": outcome["tier"],
                    "amount_rule": outcome["amount"],
                    "channels": outcome["channels"],
                    "priority_weight": 0.6,
                    "reason_label": 200,
                }
            if node_data.get("false_branch") not in tree["nodes"]:
                outcome = outcomes[(hash(node_data["false_branch"]) + 1) % len(outcomes)]
                tree["nodes"][node_data["false_branch"]] = {
                    "key": node_data["false_branch"],
                    "level": node_data.get("level", 0) + 1,
                    "is_leaf": True,
                    "outcome_code": outcome["code"],
                    "tier_code": outcome["tier"],
                    "amount_rule": outcome["amount"],
                    "channels": outcome["channels"],
                    "priority_weight": 0.4,
                    "reason_label": 201,
                }

    return tree


def generate_node_identity_map(
    old_tree: Dict[str, Any],
    new_tree: Dict[str, Any],
) -> Dict[str, str]:
    """
    Map node keys between tree versions.

    Requirement §5.4.3: node_key must be stable if condition and position unchanged.

    Returns mapping of old_node_key -> new_node_key or "changed"/"removed"
    """
    mapping = {}

    # For each old node, find matching new node
    for old_key, old_node in old_tree.get("nodes", {}).items():
        if old_node.get("is_leaf"):
            # Leaves match on outcome code + tier + amount rule
            matched = False
            for new_key, new_node in new_tree.get("nodes", {}).items():
                if (new_node.get("outcome_code") == old_node.get("outcome_code") and
                    new_node.get("tier_code") == old_node.get("tier_code") and
                    new_node.get("amount_rule") == old_node.get("amount_rule")):
                    mapping[old_key] = new_key
                    matched = True
                    break
            if not matched:
                mapping[old_key] = "removed"
        else:
            # Internal nodes match on condition
            condition = old_node.get("condition", "")
            matched = False
            for new_key, new_node in new_tree.get("nodes", {}).items():
                if new_node.get("condition") == condition:
                    mapping[old_key] = new_key
                    matched = True
                    break
            if not matched:
                mapping[old_key] = "changed"

    # Mark new nodes that were added
    for new_key in new_tree.get("nodes", {}):
        if new_key not in mapping.values():
            mapping[f"new_{new_key}"] = new_key

    return mapping
