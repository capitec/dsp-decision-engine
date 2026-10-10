"""Write `configs/<version>/`: the synthetic rate card, the cap reductions, and the params document for one cycle.

    uv run python examples/campaign-trees/scripts/generate_configs.py [cycle_date] [version]

The params document holds the tree overlays in force on `cycle_date` (default: the sample request's, 2026-09-24).
The tree document `campaign_23.json` is maintained by hand and is not touched.
"""
import json
import sys
from datetime import date
from pathlib import Path

from credit_core.rate_card import generate_flex_loan_card

from campaign_trees import overlays
from campaign_trees.pipeline import build

from decikit import ConfigurableStep, DecisionTableConfig, TreeConfig

# Tier A is the funding-constrained tier in October.
CAP_REDUCTIONS = {
    "type": "adjustment_stack", "name": "cap_reduction", "target": "tier_ceiling", "set_id": "AS-CAMPAIGN23",
    "adjusted_output": "advertised_tier_ceiling", "unadjusted_output": "tier_ceiling_before_cap_overlay",
    "scope_types": {"campaign_id": "int", "offer_tier_code": "int"},
    "adjustments": [
        {"adjustment_id": "ADJ-C23-CAP-001", "kind": "cap_adjustment", "target": "tier_ceiling",
         "effect": {"op": "multiply", "value": 0.8}, "scope": {"campaign_id": overlays.CAMPAIGN_ID, "offer_tier_code": 1},
         "stack_position": 1, "owner": "Credit Risk Policy", "approval_reference": "CRC-2026-041",
         "rationale": "Tier A funding constrained for the October cycle; 20% off while reviewed",
         "effective_from": "2026-10-01", "effective_to": "2026-11-01", "review_date": "2026-10-25", "tighten_only": True},
    ],
}


def main(cycle_date: str = "2026-09-24", version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    card = generate_flex_loan_card("rc-2026.09")
    (out / "rate_card_flex_loan.json").write_text(json.dumps(card, separators=(",", ":")))
    tree = TreeConfig.load(json.loads((out / "campaign_23.json").read_text()))
    (out / "cap_reductions.json").write_text(json.dumps(CAP_REDUCTIONS, indent=2) + "\n")
    pipeline = build(tree, DecisionTableConfig.load(card), ConfigurableStep.load(CAP_REDUCTIONS))
    params = pipeline.parameters().defaults()
    # The rate card document is the one copy of the card; the params document holds the policy numbers only, and the
    # versions of the card and the credit life rates follow their rows.
    for table in ("flex_loan_rate_card", "flex_loan_rate_card_version", "credit_life_rates", "credit_life_version"):
        del params["campaign_trees"]["evaluate"]["evaluation"]["pre_assessment"][table]
    shifts, stack_id = overlays.resolve(overlays.TREE_OVERLAYS, overlays.CAMPAIGN_ID, date.fromisoformat(cycle_date),
                                        tree.tree.to_tree().required_params())
    params["campaign_trees"]["evaluate"]["evaluation"]["tree_adjusted"].update(shifts)
    params["campaign_trees"]["overlay_stack_id"]["overlay_stack_id"] = stack_id
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"wrote {len(card['rows'])} rate card rows, the cap reductions and {out / 'params.json'} ({stack_id})")


if __name__ == "__main__":
    main(*sys.argv[1:])
