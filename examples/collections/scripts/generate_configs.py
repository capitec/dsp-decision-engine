"""Write `configs/<version>/`: the treatment matrix document and the params document.

    uv run python examples/collections/scripts/generate_configs.py [version]
"""
import json
import sys
from pathlib import Path

from decider import DecisionTableConfig

from collections_treatment import capacity, matrix
from collections_treatment.pipeline import build


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    document = matrix.generate_matrix()
    (out / "treatment_matrix.json").write_text(json.dumps(document, separators=(",", ":")))
    params = build(DecisionTableConfig.load(document)).parameters().defaults()
    params["collections_treatment"]["capacity"]["pools"] = capacity.default_pool_rows()
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"wrote {len(document['rows'])} matrix cells and {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
