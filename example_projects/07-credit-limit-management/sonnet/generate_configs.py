"""Regenerates `configs/<version>/matrix.json` -- the 1 152-cell limit
assignment matrix as the externalised document `pipeline.build(matrix)`
loads (§6.3 item 1: "authored in a spreadsheet... loaded without a
deployment"). Mirrors 00's `generate_configs.py` for the Flex Loan rate
card.

    uv run --project <REPO> python generate_configs.py 0.1.0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from limit_mgmt.matrix import generate_matrix_rows


def main(version: str) -> None:
    out_dir = Path(__file__).parent / "configs" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "type": "decision_table", "name": "limit_assignment_matrix",
        "columns": {"product": "Int64", "grade": "Int64", "util_band": "Int64", "mob_band": "Int64",
                    "multiplier": "Float64", "max_increase": "Float64", "min_increment": "Float64",
                    "cell_id": "String"},
        "rows": generate_matrix_rows(),
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "eq", "variable": "behaviour_grade", "value_column": "grade"},
            {"type": "eq", "variable": "utilisation_band", "value_column": "util_band"},
            {"type": "eq", "variable": "mob_band", "value_column": "mob_band"},
        ]},
        "outputs": ["multiplier", "max_increase", "min_increment", "cell_id"],
        "default": [1.00, 0.0, 2_500.0, None],
    }
    out_path = out_dir / "matrix.json"
    out_path.write_text(json.dumps(doc, indent=2))
    print(f"wrote {out_path} ({len(doc['rows'])} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0.1.0")
