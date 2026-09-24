"""Overlay application to trees."""

from typing import Dict, Any, List
from datetime import date


def load_overlay_stack(cycle_date: date) -> Dict[str, Any]:
    """
    Load overlay stack in force at cycle_date.

    For the 04-slice, we have a simple stub with 3-5 overlays.
    Real spec: 40 overlays across programme at any time.
    """
    # ponytail: overlay stub, fully resolve in production
    return {
        "stack_id": hash((cycle_date.year, cycle_date.month)) % 1000,
        "overlays": [
            {
                "overlay_id": "OV001",
                "kind": "volume_dial",
                "campaign_id": 1,
                "parameter": "discretionary_income_threshold",
                "adjustment": 200,  # Increase threshold by R200
                "effective_from": "2026-09-01",
                "effective_to": "2026-10-31",
                "owner": "Campaign Forum",
            },
            {
                "overlay_id": "OV002",
                "kind": "cut_off_shift",
                "campaign_id": 2,
                "parameter": "min_risk_grade",
                "adjustment": -1,  # Tighten by 1 grade (5 -> 6)
                "effective_from": "2026-09-01",
                "effective_to": "2026-11-30",
                "owner": "Credit Risk Policy",
            },
            {
                "overlay_id": "OV003",
                "kind": "cap_reduction",
                "tier_code": 1,
                "adjustment": 0.9,  # Reduce by 10%
                "effective_from": "2026-09-15",
                "effective_to": "2026-10-31",
                "owner": "Channel Operations",
            },
        ],
    }


def apply_tree_overlays(
    tree_evaluation: Dict[str, Any],
    overlay_stack: Dict[str, Any],
    cycle_date: date,
) -> Dict[str, Any]:
    """
    Apply overlay stack to a tree evaluation result.

    Overlays may change:
    - The leaf reached (volume dial, cut-off shift)
    - The amount offered (cap reduction)
    - The effective risk (score shift, odds multiplier)

    Requirement: unadjusted_leaf is recorded when overlay changes the answer.
    """
    result = tree_evaluation.copy()

    # Store unadjusted values
    unadjusted_amount = result.get("pre_assessed_amount", 0)
    unadjusted_tier = result.get("leaf_tier_code", 1)

    # Apply overlays (simple version: only cap reduction for now)
    for overlay in overlay_stack.get("overlays", []):
        if overlay["kind"] == "cap_reduction":
            campaign_id = result.get("campaign_id")
            tier = result.get("leaf_tier_code")

            # Apply if tier matches
            if overlay.get("tier_code") == tier:
                adjusted_amount = unadjusted_amount * overlay["adjustment"]
                result["pre_assessed_amount"] = adjusted_amount
                result["amount_adjusted"] = True

    # Record unadjusted values per spec §5.3.4(b)
    if result.get("amount_adjusted"):
        result["unadjusted_amount"] = unadjusted_amount
        result["unadjusted_tier_code"] = unadjusted_tier

    result["overlay_stack_id"] = overlay_stack.get("stack_id", 0)

    return result
