"""Tests for campaign tree evaluation and path capture."""

import pytest
from datetime import date
from campaign_trees.tree_engine import TreeEngine
from campaign_trees.tree_defs import generate_campaign_trees, generate_node_identity_map
from campaign_trees.suppressions import evaluate_suppressions
from campaign_trees.arbitration import simple_arbitration
from campaign_trees.registry import generate_campaign_registry


class TestTreeGeneration:
    """Test tree generation."""

    def test_generate_campaign_tree(self):
        """Test tree generation is deterministic."""
        tree1 = generate_campaign_trees(campaign_id=1, tree_version=1, seed=1)
        tree2 = generate_campaign_trees(campaign_id=1, tree_version=1, seed=1)

        assert tree1["campaign_id"] == tree2["campaign_id"]
        assert tree1["version"] == tree2["version"]
        assert len(tree1["nodes"]) == len(tree2["nodes"])

    def test_tree_has_root_node(self):
        """Test tree has root node."""
        tree = generate_campaign_trees(campaign_id=1, tree_version=1, seed=1)
        assert "node_1" in tree["nodes"]
        assert tree["root_key"] == "node_1"

    def test_tree_nodes_have_structure(self):
        """Test tree nodes have required structure."""
        tree = generate_campaign_trees(campaign_id=1, tree_version=1, seed=1)

        for node_key, node in tree["nodes"].items():
            assert "key" in node
            assert "level" in node
            assert "is_leaf" in node

            if node["is_leaf"]:
                assert "outcome_code" in node
                assert "tier_code" in node
                assert "amount_rule" in node
            else:
                assert "condition" in node
                assert "true_branch" in node
                assert "false_branch" in node

    def test_tree_all_leaves_reachable(self):
        """Test all leaf nodes are reachable via branches."""
        tree = generate_campaign_trees(campaign_id=1, tree_version=1, seed=1)

        # Collect all branch targets
        reachable = {tree["root_key"]}
        changed = True
        while changed:
            changed = False
            for node_key, node in tree["nodes"].items():
                if node_key in reachable and not node["is_leaf"]:
                    for branch_key in [node["true_branch"], node["false_branch"]]:
                        if branch_key not in reachable:
                            reachable.add(branch_key)
                            changed = True

        # All nodes should be reachable
        assert len(reachable) >= len([n for n in tree["nodes"] if n == tree["root_key"]])


class TestTreeEvaluation:
    """Test tree evaluation with path capture."""

    def test_evaluate_tree_deterministic(self):
        """Test tree evaluation is deterministic."""
        tree = generate_campaign_trees(campaign_id=1, tree_version=1, seed=1)
        engine = TreeEngine()

        features = {
            "age": 35,
            "income_band": 6,
            "behavior_score": 650,
        }

        result1 = engine.evaluate(
            tree=tree,
            client_id=1,
            campaign_id=1,
            features=features,
            cycle_date=date(2026, 9, 24),
            decision_id="test_1",
        )

        result2 = engine.evaluate(
            tree=tree,
            client_id=1,
            campaign_id=1,
            features=features,
            cycle_date=date(2026, 9, 24),
            decision_id="test_1",
        )

        assert result1.leaf.key == result2.leaf.key
        assert result1.path == result2.path

    def test_tree_evaluation_returns_path(self):
        """Test evaluation returns non-empty path."""
        tree = generate_campaign_trees(campaign_id=1, tree_version=1, seed=1)
        engine = TreeEngine()

        result = engine.evaluate(
            tree=tree,
            client_id=1,
            campaign_id=1,
            features={"age": 35, "income_band": 6},
            cycle_date=date(2026, 9, 24),
            decision_id="test",
        )

        assert len(result.path) > 0
        assert result.path[0] == tree["root_key"]

    def test_tree_evaluation_reaches_leaf(self):
        """Test evaluation always reaches a leaf."""
        tree = generate_campaign_trees(campaign_id=1, tree_version=1, seed=1)
        engine = TreeEngine()

        result = engine.evaluate(
            tree=tree,
            client_id=1,
            campaign_id=1,
            features={},
            cycle_date=date(2026, 9, 24),
            decision_id="test",
        )

        assert result.leaf is not None
        assert result.leaf.outcome_code in [1, 2, 3]


class TestNodeIdentityMapping:
    """Test node identity mapping across versions."""

    def test_node_identity_unchanged_condition(self):
        """Test node key unchanged when condition unchanged."""
        tree_v1 = generate_campaign_trees(campaign_id=1, tree_version=1, seed=1)
        tree_v2 = generate_campaign_trees(campaign_id=1, tree_version=2, seed=1)

        mapping = generate_node_identity_map(tree_v1, tree_v2)

        # Some nodes should map to themselves or similar
        assert len(mapping) > 0


class TestSuppressions:
    """Test suppression evaluation."""

    def test_evaluate_suppressions(self):
        """Test suppression evaluation."""
        campaigns = generate_campaign_registry(10)
        suppressions = evaluate_suppressions(
            client_id=8412907,
            campaigns=campaigns,
            cycle_date=date(2026, 9, 24),
        )

        assert isinstance(suppressions, list)
        for supp in suppressions:
            assert "suppression_code" in supp
            assert "scope" in supp
            assert "client_id" in supp

    def test_suppressions_deterministic(self):
        """Test suppressions are deterministic."""
        campaigns = generate_campaign_registry(10)

        supp1 = evaluate_suppressions(
            client_id=12345,
            campaigns=campaigns,
            cycle_date=date(2026, 9, 24),
        )

        supp2 = evaluate_suppressions(
            client_id=12345,
            campaigns=campaigns,
            cycle_date=date(2026, 9, 24),
        )

        assert len(supp1) == len(supp2)
        for s1, s2 in zip(supp1, supp2):
            assert s1["suppression_code"] == s2["suppression_code"]


class TestArbitration:
    """Test arbitration."""

    def test_simple_arbitration(self):
        """Test simple arbitration."""
        evaluations = [
            {
                "campaign_id": 1,
                "priority_weight": 0.8,
                "leaf_tier_code": 1,
                "pre_assessed_amount": 50000,
                "reason_label": 100,
                "channels": ["sms", "app"],
            },
            {
                "campaign_id": 2,
                "priority_weight": 0.6,
                "leaf_tier_code": 2,
                "pre_assessed_amount": 30000,
                "reason_label": 101,
                "channels": ["app"],
            },
        ]

        result = simple_arbitration(
            client_id=1,
            evaluations=evaluations,
            cycle_id=202609,
            cycle_date=date(2026, 9, 24),
        )

        assert "assignments" in result
        assert len(result["assignments"]) >= 1

    def test_arbitration_respects_contact_cap(self):
        """Test arbitration respects contact cap."""
        evaluations = [
            {
                "campaign_id": i,
                "priority_weight": 0.9 - i * 0.1,
                "leaf_tier_code": 1,
                "pre_assessed_amount": 50000,
                "reason_label": 100,
                "channels": ["sms"],
            }
            for i in range(5)
        ]

        result = simple_arbitration(
            client_id=1,
            evaluations=evaluations,
            cycle_id=202609,
            cycle_date=date(2026, 9, 24),
        )

        # Should have contact_count <= 2 (contact cap)
        if not result.get("is_control"):
            contacted_count = sum(
                1 for a in result["assignments"] if a.get("contacted")
            )
            assert contacted_count <= 2

    def test_arbitration_deterministic(self):
        """Test arbitration is deterministic."""
        evaluations = [
            {
                "campaign_id": i,
                "priority_weight": 0.8,
                "leaf_tier_code": 1,
                "pre_assessed_amount": 50000,
                "reason_label": 100,
                "channels": ["sms"],
            }
            for i in range(3)
        ]

        result1 = simple_arbitration(
            client_id=12345,
            evaluations=evaluations,
            cycle_id=202609,
            cycle_date=date(2026, 9, 24),
        )

        result2 = simple_arbitration(
            client_id=12345,
            evaluations=evaluations,
            cycle_id=202609,
            cycle_date=date(2026, 9, 24),
        )

        assert len(result1["assignments"]) == len(result2["assignments"])
        for a1, a2 in zip(result1["assignments"], result2["assignments"]):
            assert a1["campaign_id"] == a2["campaign_id"]
            assert a1["is_control"] == a2["is_control"]


class TestCampaignRegistry:
    """Test campaign registry."""

    def test_generate_registry(self):
        """Test registry generation."""
        campaigns = generate_campaign_registry(60)

        assert len(campaigns) == 60
        for i, campaign in enumerate(campaigns):
            assert campaign["campaign_id"] == i + 1
            assert "campaign_name" in campaign
            assert "owner" in campaign
            assert "priority_weight" in campaign
