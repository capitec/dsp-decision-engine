"""Write `configs/<version>/`: the synthetic rate card, and the params document for one cycle.

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

from decider import DecisionTableConfig, TreeConfig


def main(cycle_date: str = "2026-09-24", version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    card = generate_flex_loan_card("rc-2026.09")
    (out / "rate_card_flex_loan.json").write_text(json.dumps(card, separators=(",", ":")))
    tree = TreeConfig.load(json.loads((out / "campaign_23.json").read_text()))
    pipeline = build(tree, DecisionTableConfig.load(card))
    params = pipeline.parameters().defaults()
    shifts, stack_id = overlays.resolve(overlays.TREE_OVERLAYS, overlays.CAMPAIGN_ID, date.fromisoformat(cycle_date),
                                        tree.tree.to_tree().required_params())
    params["campaign_trees"]["evaluate"]["evaluation"]["tree_adjusted"].update(shifts)
    params["campaign_trees"]["overlay_stack_id"]["overlay_stack_id"] = stack_id
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"wrote {len(card['rows'])} rate card rows and {out / 'params.json'} ({stack_id})")


if __name__ == "__main__":
    main(*sys.argv[1:])
