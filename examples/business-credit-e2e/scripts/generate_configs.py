"""Write `configs/<version>/`: the policy numbers of the decision at their defaults, and the DSCR covenant documents.

    uv run python examples/business-credit-e2e/scripts/generate_configs.py [version]
"""
import json
import sys
from pathlib import Path

from business_credit_e2e.pipeline import build

# The 2028 definition changes the denominator (lease liabilities) and keeps the threshold.
DSCR_VERSIONS = {
    "type": "effective_versions", "family": "COV-DSCR", "output": "covenant_definition_version",
    "versions": [
        {"version_id": "dscr-2024.01", "effective_from": "2024-01-01", "effective_to": "2026-03-01"},
        {"version_id": "dscr-2026.03", "effective_from": "2026-03-01", "effective_to": "2028-09-01"},
        {"version_id": "dscr-2028.09", "effective_from": "2028-09-01", "effective_to": None},
    ],
}
DSCR_TERMS = {
    "type": "covenant_terms", "name": "covenant_terms",
    "terms": [
        {"version_id": "dscr-2024.01", "threshold": 1.20, "cure_days": 30},
        {"version_id": "dscr-2026.03", "threshold": 1.25, "cure_days": 30},
        {"version_id": "dscr-2028.09", "threshold": 1.25, "cure_days": 30},
    ],
}


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    for name, doc in (("dscr_versions", DSCR_VERSIONS), ("dscr_terms", DSCR_TERMS)):
        (out / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
    (out / "params.json").write_text(json.dumps(build().parameters().defaults(), indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
