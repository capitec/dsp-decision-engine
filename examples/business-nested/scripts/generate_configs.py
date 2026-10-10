"""Write `configs/<version>/`: the overlay stacks, the reason codes and the params document.

    uv run python examples/business-nested/scripts/generate_configs.py [version]
"""
import json
import sys
from pathlib import Path

from credit_core.adverse_events import JUDGMENT
from credit_core.vocab import PRODUCT_BUSINESS_TERM_FACILITY, SEGMENT_SME_PEOPLE

from decikit import ConfigurableStep

from business_nested import outcome
from business_nested.pipeline import build

OWNER = "Business Credit Risk Policy"

# A natural person behind a new-to-bank business scores 12 points lower: defaults above the model's prediction.
ENTITY_SCORE_ADJUSTMENTS = {
    "type": "adjustment_stack", "target": "entity_score", "set_id": "AS-05-ENT-2026.09",
    "base_field": "score_before_overlay",
    "adjustments": [
        {"adjustment_id": "ADJ-05-ENT-001", "kind": "score_shift", "target": "entity_score",
         "effect": {"op": "add", "value": -12.0}, "scope": {"segment_code": SEGMENT_SME_PEOPLE}, "stack_position": 2,
         "owner": OWNER, "approval_reference": "CRC-2026-041",
         "rationale": "Natural-person entities behind new-to-bank businesses defaulting above prediction",
         "effective_from": "2026-01-01", "effective_to": None, "review_date": "2026-12-31", "tighten_only": True},
    ],
}

# Term facilities are 25% more likely to default than the model says, for now. Odds only ever go up.
BUSINESS_PD_ADJUSTMENTS = {
    "type": "adjustment_stack", "target": "probability_of_default", "set_id": "AS-05-BIZ-2026.09",
    "base_field": "probability_of_default_before_overlay",
    "adjustments": [
        {"adjustment_id": "ADJ-05-BIZ-001", "kind": "odds_multiplier", "target": "probability_of_default",
         "effect": {"op": "multiply", "value": 1.25}, "scope": {"product_code": PRODUCT_BUSINESS_TERM_FACILITY},
         "stack_position": 6, "owner": OWNER, "approval_reference": "CRC-2026-055",
         "rationale": "Term facilities defaulting above the model's prediction, two quarters",
         "effective_from": "2026-01-01", "effective_to": "2026-12-31", "review_date": "2026-11-01",
         "tighten_only": True},
    ],
}

# Halve the judgment materiality threshold for construction until the end of 2026: the policy lever the thresholds
# exist for. A materiality threshold is a ceiling that only shrinks, hence `cap_adjustment`.
EVENT_THRESHOLD_OVERLAYS = {
    "type": "event_threshold_overlays", "set_id": "AS-05-EVT-2026.09",
    "adjustments": [
        {"adjustment_id": "ADJ-05-014", "kind": "cap_adjustment", "target": "material_threshold",
         "effect": {"op": "multiply", "value": 0.5}, "scope": {"sector_code": 412, "event_type_code": JUDGMENT},
         "stack_position": 1, "owner": OWNER, "approval_reference": "CRC-2026-071",
         "rationale": "Halve judgment materiality thresholds for construction, two quarters",
         "effective_from": "2026-01-01", "effective_to": "2026-12-31", "review_date": "2026-11-01",
         "tighten_only": True},
    ],
}

REASON_CODES = {
    "type": "reason_codes", "name": "rank_reasons", "version": "business-nested-reasons-2026.09",
    "codes": [
        {"code": code, "severity_rank": rank, "description": description, "is_regulatory": regulatory}
        for code, rank, description, regulatory in [
            (outcome.R_ENTITY_DISQUALIFYING, 1, "An entity behind the business carries a disqualifying adverse record", True),
            (outcome.R_SOLE_PROPRIETOR_STATUTORY_FAIL, 2, "Statutory affordability assessment failed", True),
            (outcome.R_DISPUTED_EVENT, 3, "The decline rests on an adverse event that is disputed", True),
            (outcome.R_STRUCTURE_UNRESOLVED, 5, "Disclosed ownership structure could not be resolved", True),
            (outcome.R_INSUFFICIENT_PEOPLE_COVERAGE, 20, "Scoreable ownership below the coverage requirement", True),
            (outcome.R_SOLE_PROPRIETOR_NOT_ESTABLISHED, 22, "Statutory affordability could not be established", True),
            (outcome.R_NO_SCOREABLE_NATURAL_PERSON, 25, "No scoreable natural person behind the business", True),
            (outcome.R_BUSINESS_GRADE_BELOW_CUTOFF, 30, "Business grade below the minimum acceptable grade", False),
            (outcome.R_BUSINESS_GRADE_IN_REFERRAL_BAND, 35, "Business grade in the referral band", False),
        ]
    ],
}

DOCUMENTS = {
    "entity_score_adjustments": ENTITY_SCORE_ADJUSTMENTS, "business_pd_adjustments": BUSINESS_PD_ADJUSTMENTS,
    "event_threshold_overlays": EVENT_THRESHOLD_OVERLAYS, "reason_codes": REASON_CODES,
}


def main(version: str = "0.2.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    for name, document in DOCUMENTS.items():
        (out / f"{name}.json").write_text(json.dumps(document, indent=2) + "\n")
    pipeline = build(**{name: ConfigurableStep.load(document) for name, document in DOCUMENTS.items()})
    (out / "params.json").write_text(json.dumps(pipeline.parameters().defaults(), indent=2) + "\n")
    print(f"wrote {', '.join(DOCUMENTS)} and {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
