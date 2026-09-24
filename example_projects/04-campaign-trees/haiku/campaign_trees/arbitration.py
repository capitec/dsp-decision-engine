"""Simple arbitration: rank campaigns, respect channel caps."""

from typing import Dict, Any, List
from datetime import date
from campaign_trees.types import Assignment, ArbitrationResult
import hashlib


def simple_arbitration(
    client_id: int,
    evaluations: List[Dict[str, Any]],
    cycle_id: int,
    cycle_date: date,
) -> Dict[str, Any]:
    """
    Simple arbitration: rank campaigns by priority_weight and contact cap (2/month).

    Spec §5.6: Cycle cap is 2 contacts per client. Channel capacity is population-level.
    This slice does deterministic ranking without channel optimization.

    Requirements:
    1. Deterministic: same inputs -> same assignments
    2. Re-runnable: cycle re-run -> identical contact list
    3. Reason codes for non-contacts (rank, channel cap, fatigue, control)
    4. Capacity utilization reporting
    """
    # Deterministic holdout / control assignment (spec §5.7)
    variant_hash = int(
        hashlib.md5(
            f"{client_id}_{cycle_id}_variant_v1".encode()
        ).hexdigest(),
        16,
    ) % 100

    control_hash = int(
        hashlib.md5(
            f"{client_id}_{cycle_id}_control_v1".encode()
        ).hexdigest(),
        16,
    ) % 100

    is_control = control_hash < 5  # 5% control per default
    variant = 1 if variant_hash < 90 else 2  # 90% champion, 10% challenger

    # Sort campaigns by priority_weight (descending)
    ranked = sorted(
        evaluations,
        key=lambda e: (
            -e.get("priority_weight", 0),
            e.get("reason_label", 999),  # Break ties on reason
        ),
    )

    # Contact cap: 2 per client per cycle (spec §5.6)
    contact_count = 0
    contact_limit = 2
    assignments = []

    for eval_result in ranked:
        campaign_id = eval_result.get("campaign_id")

        # Determine if contact is made
        contacted = contact_count < contact_limit and not is_control

        # Build assignment record
        assignment = Assignment(
            assignment_id=f"asn_{client_id}_{campaign_id}_{cycle_id}",
            client_id=client_id,
            campaign_id=campaign_id,
            cycle_id=cycle_id,
            cycle_date=cycle_date,
            tree_version=eval_result.get("tree_version", 1),
            variant=variant,
            is_control=is_control,
            leaf_key=eval_result.get("leaf_key", "unknown"),
            offer_tier_code=eval_result.get("leaf_tier_code", 1),
            offered_amount=eval_result.get("pre_assessed_amount", 0),
            term_months=eval_result.get("pre_assessed_term", 60),
            channel=eval_result.get("channels", ["sms"])[0] if eval_result.get("channels") else "sms",
            priority_weight=eval_result.get("priority_weight", 0.5),
            reason_label=eval_result.get("reason_label", 0),
            reason_text=eval_result.get("reason_text", "Targeted"),
            contacted=contacted,
        )

        if not contacted:
            if is_control:
                assignment.not_contacted_reason = "control_group"
            elif contact_count >= contact_limit:
                assignment.not_contacted_reason = "contact_cap_reached"
            else:
                assignment.not_contacted_reason = "ranked_out"
        else:
            contact_count += 1

        assignments.append(assignment)

    return {
        "assignments": [asn.__dict__ for asn in assignments],
        "contact_count": contact_count,
        "is_control": is_control,
        "variant": variant,
    }
