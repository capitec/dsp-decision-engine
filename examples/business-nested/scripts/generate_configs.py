"""Write `configs/<version>/params.json`: the policy numbers of the decision, at their defaults.

    uv run python examples/business-nested/scripts/generate_configs.py [version]
"""
import json
import sys
from pathlib import Path

from business_nested.pipeline import build


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.json").write_text(json.dumps(build().parameters().defaults(), indent=2) + "\n")
    print(f"wrote {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
