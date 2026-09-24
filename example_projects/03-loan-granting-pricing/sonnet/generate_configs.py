"""Regenerate `configs/<version>/rate_card_flex_loan.json` from `credit_core.rate_card`
(project 00's generator, reused unchanged -- see NOTES.md).

Usage: `python generate_configs.py [version] [configs_dir]`
"""
import json
import sys
from pathlib import Path

from credit_core.rate_card import generate_flex_loan_card

RATE_CARD_VERSION = "rc-2026.09"


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "0.1.0"
    configs_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "configs")
    out_dir = configs_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = generate_flex_loan_card(RATE_CARD_VERSION)
    (out_dir / "rate_card_flex_loan.json").write_text(json.dumps(doc))
    print(f"wrote {len(doc['rows'])} rows to {out_dir / 'rate_card_flex_loan.json'}")


if __name__ == "__main__":
    main()
