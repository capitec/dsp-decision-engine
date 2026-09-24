"""Generate `configs/<version>/{live_rules,shadow_rules,overlay_base_rules}.json`.

Same pattern as 00's `generate_configs.py`: the rule *shapes* are generated
code (SCOPE.md: "a generated rule set"), written out as `ConfigurableStep`
documents so they can be inspected, diffed and -- in a real deployment --
refreshed without a redeploy (00 §9), exactly like 00's Flex Loan rate
card. `pipeline.build()` regenerates the identical `RuleCatalog` (a fixed
seed, §01 "Framework friction" / 09 §5.15 item 3: deterministic, not
random) to attach governance metadata to whichever document version is
loaded; the two are generated together and must not drift apart --
recorded in NOTES.md as a real coupling this slice accepts.

Usage: `python generate_configs.py 0.1.0`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fraud_interdiction.rules import build_rule_catalog, build_overlay_base_document, build_tree_document


def main(version: str) -> None:
    out_dir = Path(__file__).parent / "configs" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = build_rule_catalog()
    live = catalog.by_status("live")
    shadow = catalog.by_status("shadow")

    (out_dir / "live_rules.json").write_text(json.dumps(build_tree_document(live, "live_rules"), default=str))
    (out_dir / "shadow_rules.json").write_text(json.dumps(build_tree_document(shadow, "shadow_rules"), default=str))
    (out_dir / "overlay_base_rules.json").write_text(
        json.dumps(build_overlay_base_document(live, "overlay_base_rules"), default=str))
    print(f"wrote {len(live)} live, {len(shadow)} shadow, "
          f"{len(catalog.overlay_eligible())} overlay-base rules to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0.1.0")
