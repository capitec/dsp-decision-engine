"""Write `configs/<version>/` for the quote pipeline: the synthetic rate card, the score overlays, the reason codes and the params.

    uv run python examples/shared-credit-core/scripts/generate_configs.py [version]
"""
import json
import sys
from pathlib import Path

from credit_core import eligibility
from credit_core.pipeline import R_AFFORDABILITY_FAIL, build
from credit_core.rate_card import generate_flex_loan_card

from decider import ConfigurableStep, DecisionTableConfig

# Two tighten-only score overlays. An applicant in segment 3 who came through channel 4 gets both, in stack order.
SCORE_ADJUSTMENTS = {
    "type": "adjustment_stack", "name": "apply_score_adjustments", "target": "score", "set_id": "AS-2026.09",
    "base_field": "score_raw",
    "adjustments": [
        {"adjustment_id": "ADJ-2026-001", "kind": "score_shift", "target": "score",
         "effect": {"op": "add", "value": -18.0}, "scope": {"segment_code": 3}, "stack_position": 1,
         "owner": "Credit Risk Policy", "approval_reference": "CRC-2026-014",
         "rationale": "New-to-bank segment defaulting above model prediction for two quarters",
         "effective_from": "2026-01-01", "effective_to": None, "review_date": "2026-12-31", "tighten_only": True},
        {"adjustment_id": "ADJ-2026-002", "kind": "score_shift", "target": "score",
         "effect": {"op": "add", "value": -5.0}, "scope": {"channel_code": 4}, "stack_position": 2,
         "owner": "Credit Risk Policy", "approval_reference": "CRC-2026-019",
         "rationale": "Broker channel 4 showing early-life deterioration",
         "effective_from": "2026-06-01", "effective_to": "2027-06-01", "review_date": "2027-01-01",
         "tighten_only": True},
    ],
}

REASON_CODES = {
    "type": "reason_codes", "name": "rank_reasons", "version": "reasons-2026.09",
    "codes": [
        {"code": code, "severity_rank": rank, "description": description, "is_regulatory": regulatory}
        for code, rank, description, regulatory in [
            (eligibility.R_UNDERAGE, 10, "Applicant below minimum age", True),
            (eligibility.R_NOT_RESIDENT, 15, "Applicant not resident", True),
            (eligibility.R_NO_CAPACITY, 5, "No contractual capacity", True),
            (eligibility.R_PRODUCT_UNAVAILABLE, 40, "Product not available to applicant", False),
            (eligibility.R_EXCLUSION_LIST, 8, "On an internal exclusion list", True),
            (eligibility.R_SANCTIONED, 1, "Sanctioned party", True),
            (eligibility.R_DECEASED, 2, "Deceased on record", True),
            (eligibility.R_DEBT_REVIEW, 12, "Under debt review", True),
            (eligibility.R_ADMINISTRATION, 12, "Under administration", True),
            (R_AFFORDABILITY_FAIL, 20, "Proposed instalment exceeds affordable capacity", True),
        ]
    ],
}


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    card = generate_flex_loan_card("rc-2026.09")
    (out / "rate_card_flex_loan.json").write_text(json.dumps(card, separators=(",", ":")))
    (out / "score_adjustments.json").write_text(json.dumps(SCORE_ADJUSTMENTS, indent=2) + "\n")
    (out / "reason_codes.json").write_text(json.dumps(REASON_CODES, indent=2) + "\n")
    pipeline = build(DecisionTableConfig.load(card), ConfigurableStep.load(SCORE_ADJUSTMENTS), ConfigurableStep.load(REASON_CODES))
    (out / "params.json").write_text(json.dumps(pipeline.parameters().defaults(), indent=2) + "\n")
    print(f"wrote {len(card['rows'])} rate card rows, the score overlays, the reason codes and {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
