"""Decider pipeline for retail campaign targeting trees."""

from decider import step, flow, missing_as
from datetime import date
from typing import List, Dict, Any
import uuid


@step()
def intake_and_dates(cycle_id: int, cycle_date: date) -> Dict[str, Any]:
    """Intake cycle metadata."""
    return {
        "cycle_id": cycle_id,
        "cycle_date": cycle_date,
        "decision_id": str(uuid.uuid4()),
    }


@step()
def load_campaigns(
    cycle_id: int,
    campaign_count: int = missing_as(60),
) -> Dict[str, Any]:
    """Load campaign registry."""
    from campaign_trees.registry import generate_campaign_registry
    campaigns = generate_campaign_registry(campaign_count)
    return {
        "campaigns": campaigns,
        "campaign_count": len(campaigns),
    }


@step()
def load_trees(campaigns: List[Dict], cycle_date: date) -> Dict[str, Any]:
    """Load and compile campaign trees."""
    from campaign_trees.tree_defs import generate_campaign_trees

    trees = {}
    for campaign in campaigns:
        tree_data = generate_campaign_trees(
            campaign_id=campaign["campaign_id"],
            tree_version=campaign.get("tree_version", 1),
            seed=campaign["campaign_id"],
        )
        trees[campaign["campaign_id"]] = tree_data

    return {
        "tree_registry": trees,
        "tree_count": len(trees),
    }


@step()
def apply_suppressions(
    client_id: int,
    campaigns: List[Dict],
    cycle_date: date,
) -> Dict[str, Any]:
    """Evaluate suppressions for the client across campaigns."""
    from campaign_trees.suppressions import evaluate_suppressions
    suppressions = evaluate_suppressions(client_id, campaigns, cycle_date)
    return {
        "suppression_records": suppressions,
        "suppressed_campaigns": [
            s["campaign_id"] for s in suppressions
            if s.get("suppression_code", 0) < 100
        ],
    }


@step()
def evaluate_campaign_trees(
    client_id: int,
    campaigns: List[Dict],
    tree_registry: Dict,
    cycle_date: date,
    client_features: Dict[str, Any],
    suppressed_campaigns: List[int],
    decision_id: str,
) -> Dict[str, Any]:
    """Evaluate trees for each campaign, capturing paths."""
    from campaign_trees.tree_engine import TreeEngine
    engine = TreeEngine()
    evaluations = []
    paths = []

    for campaign in campaigns:
        campaign_id = campaign["campaign_id"]

        if campaign_id in suppressed_campaigns:
            continue

        tree = tree_registry.get(campaign_id)
        if not tree:
            continue

        result = engine.evaluate(
            tree=tree,
            client_id=client_id,
            campaign_id=campaign_id,
            features=client_features,
            cycle_date=cycle_date,
            decision_id=decision_id,
        )

        evaluations.append(result.to_dict())
        paths.append({
            "campaign_id": campaign_id,
            "client_id": client_id,
            "path": result.path,
            "leaf_key": result.leaf.key,
        })

    return {
        "tree_evaluations": evaluations,
        "path_records": paths,
    }


@step()
def apply_overlays_to_trees(
    tree_evaluations: List[Dict],
    cycle_date: date,
) -> Dict[str, Any]:
    """Apply overlay stack to tree results."""
    from campaign_trees.overlays import load_overlay_stack, apply_tree_overlays

    overlay_stack = load_overlay_stack(cycle_date)
    adjusted_evaluations = []

    for eval_result in tree_evaluations:
        adjusted = apply_tree_overlays(
            tree_evaluation=eval_result,
            overlay_stack=overlay_stack,
            cycle_date=cycle_date,
        )
        adjusted_evaluations.append(adjusted)

    return {
        "adjusted_evaluations": adjusted_evaluations,
        "overlay_stack_id": overlay_stack.get("stack_id", 0),
    }


@step()
def consume_pre_assessments(
    client_id: int,
    adjusted_evaluations: List[Dict],
    cycle_date: date,
) -> Dict[str, Any]:
    """Consume pre-assessed amounts from project 03 batch."""
    from campaign_trees.pre_assessment import stub_pre_assessments

    pre_assessments = stub_pre_assessments(
        client_id=client_id,
        campaign_count=len(adjusted_evaluations),
    )

    for eval_result in adjusted_evaluations:
        campaign_id = eval_result.get("campaign_id")
        pa = pre_assessments.get(campaign_id, {})
        eval_result["pre_assessed_amount"] = pa.get("amount", 0)
        eval_result["pre_assessed_term"] = pa.get("term", 60)
        eval_result["binding_constraint"] = pa.get("binding_constraint", "NONE")
        eval_result["risk_grade"] = pa.get("risk_grade", 7)

    return {
        "evaluations_with_amounts": adjusted_evaluations,
    }


@step()
def run_arbitration(
    client_id: int,
    cycle_id: int,
    evaluations_with_amounts: List[Dict],
    cycle_date: date,
) -> Dict[str, Any]:
    """Simple arbitration: rank campaigns, respect channel caps."""
    from campaign_trees.arbitration import simple_arbitration
    result = simple_arbitration(
        client_id=client_id,
        evaluations=evaluations_with_amounts,
        cycle_id=cycle_id,
        cycle_date=cycle_date,
    )

    return {
        "arbitration_result": result,
        "assignments": result.get("assignments", []),
    }


@step()
def build_output_records(
    client_id: int,
    cycle_id: int,
    cycle_date: date,
    decision_id: str,
    assignments: List[Dict],
    path_records: List[Dict],
    suppression_records: List[Dict],
) -> Dict[str, Any]:
    """Construct final output records for the cycle."""
    return {
        "client_id": client_id,
        "cycle_id": cycle_id,
        "cycle_date": str(cycle_date),
        "decision_id": decision_id,
        "assignment_count": len(assignments),
        "assignments": assignments,
        "paths": path_records,
        "suppressions": suppression_records,
    }


def build():
    """Build the campaign targeting pipeline."""
    return flow(
        intake_and_dates,
        load_campaigns,
        load_trees,
        apply_suppressions,
        evaluate_campaign_trees,
        apply_overlays_to_trees,
        consume_pre_assessments,
        run_arbitration,
        build_output_records,
    )
