"""Regenerate `configs/<version>/params.json` from `pipeline.build()`'s own declared params.

`pipeline.parameters().defaults()` (project 00 NOTES.md "What worked well")
walks every bound step's `param()` declarations and produces a complete,
correct params document -- the only manual edits below are this project's
own calibration overrides (10 §5.13(b): the initiation fee's base/
marginal-rate/threshold/cap, which `core.fees.initiation_fee`'s own
defaults do not match).

Usage: `python generate_configs.py [version] [configs_dir]`
"""
import json
import sys
from pathlib import Path

from pipeline import build

# Overrides by *step name* (the params document's last path segment), not by shape --
# two steps can share a parameter's shape (`credit_life_premium` and `credit_life_cap_applied`
# both take a lone `cap`, just like `initiation_fee_capped` does) and shape-matching silently
# patches the wrong one. This project hit exactly that once: NOTES.md "Framework friction".
STEP_OVERRIDES = {
    "initiation_fee": {"base_fee": 210.0, "marginal_rate": 0.095, "threshold": 1200.0, "cap": 1480.0},
    "initiation_fee_capped": {"cap": 1480.0},
    "credit_life_premium": {"cap": 1000.0},
    "credit_life_cap_applied": {"cap": 1000.0},
}


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "0.1.0"
    configs_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "configs")
    out_dir = configs_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)

    pl = build()
    params = pl.parameters().defaults()

    def _patch(node, key=None):
        if not isinstance(node, dict):
            return
        if key in STEP_OVERRIDES:
            node.update(STEP_OVERRIDES[key])
        for k, v in node.items():
            _patch(v, k)

    _patch(params)

    (out_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
    print(f"wrote {out_dir / 'params.json'}")


if __name__ == "__main__":
    main()
