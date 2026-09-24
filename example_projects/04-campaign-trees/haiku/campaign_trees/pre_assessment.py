"""Pre-assessment stub for project 03 batch output."""

from typing import Dict, Any
import random


def stub_pre_assessments(
    client_id: int,
    campaign_count: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Stub pre-assessed amounts from project 03 batch mode.

    Spec §5.5 "Emits": pre_assessed_amount, pre_assessed_term,
    binding_constraint (appetite, affordability, exposure, product_max),
    risk_grade, validity window.

    Real version: Load from project 03's batch output.
    This stub: Generate deterministically based on client_id.
    """
    rng = random.Random(client_id)

    assessments = {}
    for campaign_id in range(1, campaign_count + 1):
        # Deterministic but varies by campaign
        seed_val = client_id * 1000 + campaign_id
        rng_campaign = random.Random(seed_val)

        # Pre-assessed amounts: R2k to R500k in R100 bands (spec §5.5 req 5)
        base_amount = rng_campaign.randint(20, 5000) * 100

        # Risk grade: 1-12
        risk_grade = rng_campaign.randint(1, 12)

        # Term: 6-84 months
        term = rng_campaign.choice([6, 12, 24, 36, 48, 60, 72, 84])

        # Binding constraint (spec §5.8): appetite, affordability, exposure, product_max
        constraints = ["appetite", "affordability", "exposure", "product_max"]
        binding = constraints[seed_val % len(constraints)]

        assessments[campaign_id] = {
            "amount": base_amount,
            "term": term,
            "binding_constraint": binding,
            "risk_grade": risk_grade,
            "validity_days": 35,  # 35 days for monthly cycles
        }

    return assessments
