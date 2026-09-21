"""decider2 Tree -> JDM: build the example tree, convert it, load the
result into a real ZEN engine, and check it against both the independent
oracle and decider2's own compiled answer for the same inputs."""
from __future__ import annotations

import itertools
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from example_tree import build_tree, oracle  # noqa: E402
from decider2_tree_to_jdm import convert_tree  # noqa: E402

import zen  # noqa: E402
from decider2 import flow  # noqa: E402
from decider2.trees import tree_module  # noqa: E402


def main() -> None:
    tree = build_tree()
    doc, report = convert_tree(tree)
    print("=== conversion report ===")
    print(f"  decider2 nodes converted : {report.nodes_converted}")
    print(f"  JDM switchNodes emitted  : {report.switch_nodes}")
    print(f"  JDM expressionNodes      : {report.expression_nodes} (distinct output rows in use)")
    print(f"  lost params              : {report.lost_params or '(none)'}")
    print(f"  warnings                 : {report.warnings or '(none)'}")

    out_path = HERE / "example_tree.jdm.json"
    out_path.write_text(json.dumps(doc, indent=2))
    print(f"  JDM document written to  : {out_path}")

    engine = zen.ZenEngine()
    decision = engine.create_decision(json.dumps(doc))

    with tempfile.TemporaryDirectory() as bd:
        tm = tree_module(tree, build_dir=bd)
        pipeline = flow(tm.module)

        has_defaults_vals = [False, True]
        income_vals = [-100.0, 0.0, 2500.0, 4999.0, 5000.0, 17000.0, 49999.0, 50000.0, 200000.0]
        region_vals = ["GP", "WC", "EC", "NC"]

        n = 0
        mismatches = []
        for hd, inc, reg in itertools.product(has_defaults_vals, income_vals, region_vals):
            rec = {"has_defaults": hd, "income": inc, "region": reg}
            exp = oracle(hd, inc, reg)
            d2 = pipeline.score(rec)
            zctx = {"has_defaults": hd, "income": inc, "region": reg}
            z = decision.evaluate(zctx)["result"]
            n += 1
            ok = (
                d2["tier_code"] == exp["tier_code"] == int(z["tier_code"])
                and d2["limit"] == exp["limit"] == float(z["limit"])
            )
            if not ok:
                mismatches.append((rec, exp, {"tier_code": d2["tier_code"], "limit": d2["limit"]}, z))

        print(f"\n=== identical-answers check: {n} combinations ===")
        if mismatches:
            print(f"  {len(mismatches)} MISMATCHES:")
            for m in mismatches[:20]:
                print(" ", m)
            raise SystemExit(1)
        print(f"  PASS: oracle, decider2 (compiled) and ZEN (JDM conversion) agree on all {n} cases.")


if __name__ == "__main__":
    main()
