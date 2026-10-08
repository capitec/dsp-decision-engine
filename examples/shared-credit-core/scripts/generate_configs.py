"""Write `configs/<version>/` for the quote pipeline: the synthetic rate card and the params document.

    uv run python examples/shared-credit-core/scripts/generate_configs.py [version]
"""
import json
import sys
from pathlib import Path

from credit_core.pipeline import build
from credit_core.rate_card import generate_flex_loan_card

from decider import DecisionTableConfig


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    card = generate_flex_loan_card("rc-2026.09")
    (out / "rate_card_flex_loan.json").write_text(json.dumps(card, separators=(",", ":")))
    params = build(DecisionTableConfig.load(card)).parameters().defaults()
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"wrote {len(card['rows'])} rate card rows and {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
