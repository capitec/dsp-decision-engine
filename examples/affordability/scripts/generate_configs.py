"""Write `configs/<version>/`: the capacity overlays, the reason codes and the policy numbers of the assessment.

    uv run python examples/affordability/scripts/generate_configs.py [version]
"""
import json
import sys
from pathlib import Path

from decider import ConfigurableStep

from assessment.capacity import ADJUSTMENT_SET_ID
from assessment.pipeline import build

# Tighten-only: an overlay that would raise capacity is rejected when the document loads.
CAPACITY_ADJUSTMENTS = {
    "type": "adjustment_stack", "name": "apply_max_affordable_instalment_adjustments",
    "target": "max_affordable_instalment", "set_id": ADJUSTMENT_SET_ID,
    "base_field": "max_affordable_instalment_before_overlay",
    "adjustments": [
        {"adjustment_id": "ADJ-02-2026-001", "kind": "cap_adjustment", "target": "max_affordable_instalment",
         "effect": {"op": "multiply", "value": 0.90}, "scope": {"product_code": 10}, "stack_position": 1,
         "owner": "Credit Risk Policy", "approval_reference": "CRC-2026-031",
         "rationale": "Product 10 early-life delinquency above appetite for two quarters",
         "effective_from": "2026-01-01", "effective_to": None, "review_date": "2026-12-31", "tighten_only": True},
    ],
}

# 200 + the evidence_sufficiency_code, and 208 for a proposal over the maximum.
REASON_CODES = {
    "type": "reason_codes", "name": "rank_reasons", "version": "affordability-reasons-2026.09",
    "codes": [
        {"code": code, "severity_rank": rank, "description": description, "is_regulatory": regulatory}
        for code, rank, description, regulatory in [
            (201, 10, "Applicant 1 income could not be established", True),
            (202, 10, "Applicant 2 income could not be established", True),
            (203, 5, "Neither applicant's income could be established", True),
            (204, 30, "Bureau view older than the permitted window", True),
            (205, 25, "An account type requires manual review", True),
            (206, 20, "Income evidence below the minimum for this assessment", True),
            (207, 15, "No net income established", True),
            (208, 40, "Proposed instalment exceeds the affordable maximum", False),
        ]
    ],
}


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    (out / "capacity_adjustments.json").write_text(json.dumps(CAPACITY_ADJUSTMENTS, indent=2) + "\n")
    (out / "reason_codes.json").write_text(json.dumps(REASON_CODES, indent=2) + "\n")
    pipeline = build(ConfigurableStep.load(CAPACITY_ADJUSTMENTS), ConfigurableStep.load(REASON_CODES))
    (out / "params.json").write_text(json.dumps(pipeline.parameters().defaults(), indent=2) + "\n")
    print(f"wrote the capacity overlays, the reason codes and {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
