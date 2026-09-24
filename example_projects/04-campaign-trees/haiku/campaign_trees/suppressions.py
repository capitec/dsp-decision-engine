"""Suppression logic."""

from typing import List, Dict, Any
from datetime import date, timedelta


def evaluate_suppressions(
    client_id: int,
    campaigns: List[Dict[str, Any]],
    cycle_date: date,
) -> List[Dict[str, Any]]:
    """
    Evaluate suppressions for a client across campaigns.

    Suppression registry (simplified for 04-slice):
    - S01: Deceased (absolute, all campaigns)
    - S02: Debt review (absolute, all campaigns)
    - S07: Marketing opt-out (measurement-relevant, all campaigns)
    - S08: No channel consent (absolute, per-channel)
    - S14: Do-not-target list (absolute, all campaigns)
    - S21: Recent decline/cooling-off (absolute, per-campaign)
    - S23: Product already held (absolute, per-campaign)
    - S30: Contact fatigue all-channel (absolute, per-client)
    """
    suppressions = []

    # Absolute suppressions (require no tree evaluation)
    absolute_suppressions = [
        {
            "code": 1,
            "name": "Deceased",
            "condition": lambda cid: cid % 10000 == 0,  # ~0.01%
            "scope": "all_campaigns",
        },
        {
            "code": 2,
            "name": "Debt review",
            "condition": lambda cid: cid % 5000 < 10,  # ~0.2%
            "scope": "all_campaigns",
        },
        {
            "code": 14,
            "name": "Do-not-target list",
            "condition": lambda cid: cid % 4600 < 1,  # ~0.022% (3100 / 14.2M)
            "scope": "all_campaigns",
        },
    ]

    # Measurement-relevant suppressions (require tree evaluation)
    measurement_relevant = [
        {
            "code": 7,
            "name": "Marketing opt-out",
            "condition": lambda cid: cid % 13 == 0,  # ~7.7%
            "scope": "all_campaigns",
        },
    ]

    # Evaluate absolute suppressions
    for supp in absolute_suppressions:
        if supp["condition"](client_id):
            suppressions.append({
                "suppression_code": supp["code"],
                "suppression_name": supp["name"],
                "scope": supp["scope"],
                "campaign_id": None,
                "channel_code": None,
                "client_id": client_id,
                "cycle_date": str(cycle_date),
                "measurement_relevant": False,
            })

    # Evaluate measurement-relevant suppressions
    for supp in measurement_relevant:
        if supp["condition"](client_id):
            suppressions.append({
                "suppression_code": supp["code"],
                "suppression_name": supp["name"],
                "scope": supp["scope"],
                "campaign_id": None,
                "channel_code": None,
                "client_id": client_id,
                "cycle_date": str(cycle_date),
                "measurement_relevant": True,
            })

    # Per-campaign suppressions
    for campaign in campaigns:
        campaign_id = campaign["campaign_id"]

        # S23: Product already held (per-campaign)
        if client_id % (campaign_id + 100) < 20:  # Varies by campaign
            suppressions.append({
                "suppression_code": 23,
                "suppression_name": "Product already held",
                "scope": "per_campaign",
                "campaign_id": campaign_id,
                "channel_code": None,
                "client_id": client_id,
                "cycle_date": str(cycle_date),
                "measurement_relevant": False,
            })

        # S21: Recent decline/cooling-off (per-campaign, per-product)
        if client_id % (campaign_id + 50) < 15:  # ~varies
            last_decline_days = (client_id % campaign_id) % 60
            if last_decline_days < 30:  # Within 30 days
                suppressions.append({
                    "suppression_code": 21,
                    "suppression_name": "Recent decline, cooling-off",
                    "scope": "per_campaign",
                    "campaign_id": campaign_id,
                    "channel_code": None,
                    "client_id": client_id,
                    "cycle_date": str(cycle_date),
                    "measurement_relevant": False,
                })

    return suppressions
