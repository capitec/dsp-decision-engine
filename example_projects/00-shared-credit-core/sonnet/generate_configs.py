"""Regenerate `configs/<version>/rate_card_flex_loan.json` from `credit_core.rate_card`.

`core.rate_card`'s table refresh must never require a code deployment (00
§9 NFR): the JSON document is a data artefact, generated here from the
library's own synthetic-but-deterministic surface. Treasury's real
refresh would replace this file's `rows` directly; this script exists only
because this project has no spreadsheet ingestion pipeline to generate one
from (out of scope, 00 §12).

Usage: `python generate_configs.py [version] [configs_dir]`
"""
import json
import sys
from pathlib import Path

from credit_core.rate_card import generate_flex_loan_card


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "0.1.0"
    configs_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "configs")
    out_dir = configs_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = generate_flex_loan_card("rc-2026.09")
    (out_dir / "rate_card_flex_loan.json").write_text(json.dumps(doc))
    print(f"wrote {len(doc['rows'])} rows to {out_dir / 'rate_card_flex_loan.json'}")


if __name__ == "__main__":
    main()
