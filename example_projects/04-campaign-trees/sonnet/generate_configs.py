"""Regenerate `configs/<version>/`: the rate card (reused unmodified from `credit_core`,
exactly as 00 and 03 generate it -- see their own `generate_configs.py`) and this project's
`params.json`, including the overlay stack resolved for one chosen `cycle_date` (§5.3.4
requirement 7: resolved by `cycle_date`, never by "today" -- see `pipeline.py`'s
`_overlay_stack_id` docstring for why that resolution happens here, once per cycle, rather
than inside the per-record pipeline).

Usage: `python generate_configs.py [version] [configs_dir] [cycle_date]`
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from credit_core.rate_card import generate_flex_loan_card

from campaign_trees.overlays import TREE_OVERLAYS, overlay_stack_id, resolve_tree_params


def build_params(cycle_date: date) -> dict:
    resolved, applied = resolve_tree_params(TREE_OVERLAYS, {"campaign_id": 23}, cycle_date)
    tree_adjusted_params = {k: v for k, v in resolved.items() if v is not None}
    return {
        "campaign_trees": {
            "tree_adjusted": tree_adjusted_params,
            "overlay_stack_id_param": {"overlay_stack_id": overlay_stack_id(applied)},
            "estimate_max_affordable_instalment": {"serviceability_ratio": 0.30},
        }
    }


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "0.1.0"
    configs_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "configs")
    cycle_date = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date.today()
    out_dir = configs_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)

    card = generate_flex_loan_card("rc-2026.09")
    (out_dir / "rate_card_flex_loan.json").write_text(json.dumps(card))
    print(f"wrote {len(card['rows'])} rate card rows to {out_dir / 'rate_card_flex_loan.json'}")

    params = build_params(cycle_date)
    (out_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
    print(f"wrote {out_dir / 'params.json'} for cycle_date={cycle_date} "
          f"(overlay_stack_id={params['campaign_trees']['overlay_stack_id_param']['overlay_stack_id']})")


if __name__ == "__main__":
    main()
