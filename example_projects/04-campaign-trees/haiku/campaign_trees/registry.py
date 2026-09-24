"""Campaign registry generation."""

from typing import List, Dict, Any


def generate_campaign_registry(campaign_count: int = 60) -> List[Dict[str, Any]]:
    """
    Generate a campaign registry.

    Args:
        campaign_count: Number of campaigns to generate

    Returns:
        List of campaign definitions
    """
    campaigns = []

    for i in range(1, campaign_count + 1):
        campaign = {
            "campaign_id": i,
            "campaign_name": f"Campaign {i}",
            "product_code": 10 + (i % 5),  # Products 10-14
            "tree_version": 1,
            "status": "active" if i <= int(campaign_count * 0.9) else "paused",
            "owner": f"owner_{i % 11}",  # 11 campaign owners
            "priority_weight": 0.5 + (i % 100) / 200,  # 0.5-1.0
            "tier_a_max_amount": 250000 - (i * 100),
            "tier_b_max_amount": 150000 - (i * 50),
            "tier_c_max_amount": 80000,
            "channel_preference": ["sms", "app", "email", "call"][i % 4:] + ["sms", "app", "email", "call"][:i % 4],
            "holdout_rate": 0.05 if i <= 50 else 0.10,  # 5% or 10%
            "variant_split": [0.9, 0.1] if i % 3 == 0 else [1.0],  # 90/10 or 100%
        }
        campaigns.append(campaign)

    return campaigns
