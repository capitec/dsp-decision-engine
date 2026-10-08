"""Write `configs/<version>/params.json`: every policy number at its default, this product's fee and premium terms, and
the Flex Loan rate card.

    uv run python examples/retail-credit-e2e/scripts/generate_configs.py [version]
"""
import json
import sys
from pathlib import Path

from retail_credit.flex_card import card_tables
from retail_credit.pipeline import build

# The shared fee and premium steps default to other products' terms. Overrides are by step name, because two steps can
# share a parameter name and matching on the name alone would patch the wrong one.
STEP_OVERRIDES = {
    "initiation_fee": {"base_fee": 210.0, "marginal_rate": 0.095, "threshold": 1200.0},
    "credit_life_premium": {"cap": 1000.0},
}


def fill(node: dict, tables: dict) -> None:
    """Patch every step of `node` named in `STEP_OVERRIDES`, and put the card's tables where `card_rate` reads them."""
    for name, value in node.items():
        if name in STEP_OVERRIDES:
            value.update(STEP_OVERRIDES[name])
        elif name == "card_rate":
            value.update(tables)
        elif isinstance(value, dict):
            fill(value, tables)


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    params = build().parameters().defaults()
    fill(params, card_tables())
    (out / "params.json").write_text(json.dumps(params, separators=(",", ":")) + "\n")
    print(f"wrote {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
