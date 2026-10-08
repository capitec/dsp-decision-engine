"""Write `configs/<version>/`: the 1 152-cell limit matrix and the params document.

    uv run python examples/credit-limit-management/scripts/generate_configs.py [version]

The params document is every default, then the programme's own policy: the income haircuts for deposit evidence
(none on a steady salary, 15% on irregular deposits, none on declared income, which is indexed for inflation
instead) and how old a bureau view may be before the assessment cannot use it at all. Automatic against
conditional is decided by the programme's own, tighter 35 days.
"""
import json
import sys
from pathlib import Path

from limit_mgmt.matrix import build_matrix_table, matrix_document
from limit_mgmt.pipeline import build

_INCOME = "affordability_bridge/evidence/applicant{n}_income/income_haircut_applied"
POLICY = {
    **{f"{_INCOME.format(n=n)}/{name}": value for n in (1, 2) for name, value in (
        ("haircut_statement_high_confidence", 0.0), ("haircut_statement_low_confidence", 0.15), ("haircut_declared", 0.0))},
    "affordability_bridge/evidence/bureau_is_stale/staleness_window_days": 60,
}


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    params = build(build_matrix_table()).parameters().defaults()
    for path, value in POLICY.items():
        *keys, name = ["limit_mgmt", *path.split("/")]
        node = params
        for key in keys:
            node = node[key]
        assert name in node, path
        node[name] = value
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    (out / "matrix.json").write_text(json.dumps(matrix_document(), indent=2) + "\n")
    print(f"wrote {out / 'params.json'} and {out / 'matrix.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
