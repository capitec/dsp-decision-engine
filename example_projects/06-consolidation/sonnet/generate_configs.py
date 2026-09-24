"""Regenerate `configs/<version>/rate_card_*.json`.

`rate_card_flex_loan.json` is project 00's original Flex Loan (product 10) card,
reused unchanged -- needed only for the short-circuit hand-off to project 03's
pipeline (`loan_granting.pipeline.build(rate_card_flex_loan)` takes the same
argument). `rate_card_product11.json` is this project's own card (product 11,
Flex Loan Consolidation).

Usage: `python generate_configs.py [version] [configs_dir]`
"""
import json
import sys
from pathlib import Path

from credit_core.rate_card import generate_flex_loan_card

from consolidation.rate_cards import generate_product11_card

FLEX_LOAN_VERSION = "rc-2026.09"


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "0.1.0"
    configs_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "configs")
    out_dir = configs_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)

    flex_loan_doc = generate_flex_loan_card(FLEX_LOAN_VERSION)
    (out_dir / "rate_card_flex_loan.json").write_text(json.dumps(flex_loan_doc))
    print(f"wrote {len(flex_loan_doc['rows'])} rows to {out_dir / 'rate_card_flex_loan.json'}")

    product11_doc = generate_product11_card()
    (out_dir / "rate_card_product11.json").write_text(json.dumps(product11_doc))
    print(f"wrote {len(product11_doc['rows'])} rows to {out_dir / 'rate_card_product11.json'}")


if __name__ == "__main__":
    main()
