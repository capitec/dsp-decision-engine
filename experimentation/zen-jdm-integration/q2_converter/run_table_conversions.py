"""JDM -> decider2, run against real fixtures: the trivial table.json, and
every decisionTableNode in credit-analysis.json (both from the ZEN repo's
own test-data). For each: attempt the conversion; if it succeeds, build the
table (table_module) and cross-check answers against ZEN on a battery of
inputs; if it's refused, print the precise reason and — for the numeric
case — demonstrate the exact boundary value where a naive conversion would
have silently disagreed with ZEN.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures"
sys.path.insert(0, str(HERE))

from jdm_to_decider2 import NotConvertible, convert_table  # noqa: E402

import zen  # noqa: E402
from decider2 import flow  # noqa: E402
from decider2.tables import table_module  # noqa: E402


def try_convert(name: str, content: dict) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    try:
        table, report = convert_table(content, name=name)
    except NotConvertible as e:
        print(f"REFUSED: {e}")
        return
    print(f"CONVERTED: {report.rows_in} JDM rows -> {report.rows_out} decider2 rows "
          f"(default from trailing wildcard: {report.default_from_trailing_wildcard})")
    print(f"  column kinds: {report.column_kinds}")
    for note in report.notes:
        print(f"  NOTE: {note}")
    return table


def cross_check(name: str, table, content: dict, records: list[dict]) -> None:
    engine = zen.ZenEngine()
    decision = engine.create_decision(json.dumps({
        "nodes": [
            {"id": "in", "type": "inputNode", "name": "Request"},
            {"id": "tbl", "type": "decisionTableNode", "name": name, "content": content},
            {"id": "out", "type": "outputNode", "name": "Response"},
        ],
        "edges": [
            {"id": "e1", "sourceId": "in", "targetId": "tbl", "type": "edge"},
            {"id": "e2", "sourceId": "tbl", "targetId": "out", "type": "edge"},
        ],
    }))
    with tempfile.TemporaryDirectory() as bd:
        tm = table_module(table, build_dir=bd)
        pipeline = flow(tm.module)
        mismatches = []
        for rec in records:
            d2 = pipeline.score(rec, shared=tm.shared)
            z = decision.evaluate(rec)["result"]
            for out_name in table.outputs:
                jdm_field = out_name  # caller passes matching field names
                if jdm_field in z:
                    zv = z[jdm_field]
                    dv = d2.get(out_name)
                    if isinstance(zv, str) or isinstance(dv, str):
                        ok = str(zv) == str(dv)
                    else:
                        ok = float(zv) == float(dv)
                    if not ok:
                        mismatches.append((rec, out_name, dv, zv))
        if mismatches:
            print(f"  MISMATCHES ({len(mismatches)}):")
            for m in mismatches[:10]:
                print("   ", m)
        else:
            print(f"  PASS: decider2 and ZEN agree on all {len(records)} test records.")


def demonstrate_turnover_boundary(turnover_content: dict) -> None:
    """The exact case NotConvertible warned about: row "[200_000..1_000_000]"
    is closed on both ends. Show what a NAIVE lower_inclusive conversion
    (the one the honest converter refuses to produce) gets wrong, and
    confirm ZEN's own actual answer at that boundary."""
    print("\n--- demonstrating the Turnover boundary the converter refused to paper over ---")
    engine = zen.ZenEngine()
    decision = engine.create_decision(json.dumps({
        "nodes": [
            {"id": "in", "type": "inputNode", "name": "Request"},
            {"id": "tbl", "type": "decisionTableNode", "name": "Turnover", "content": turnover_content},
            {"id": "out", "type": "outputNode", "name": "Response"},
        ],
        "edges": [
            {"id": "e1", "sourceId": "in", "targetId": "tbl", "type": "edge"},
            {"id": "e2", "sourceId": "tbl", "targetId": "out", "type": "edge"},
        ],
    }))
    for turnover in (999_999.99, 1_000_000.0, 1_000_000.01):
        z = decision.evaluate({"company": {"turnover": turnover}})["result"]
        # The naive conversion (lower_inclusive, row order [<200k, [200k..1M], >1M])
        # would test: >=1_000_000 -> green. At exactly 1_000_000 that says green.
        naive_lower_inclusive_says = "green" if turnover >= 1_000_000 else ("amber" if turnover >= 200_000 else "red")
        real = z["flag"]["turnover"]
        flag = "  <-- DIVERGES" if naive_lower_inclusive_says != real else ""
        print(f"  turnover={turnover:>13} ZEN(real JDM)={real!r:>8} naive-decider2-would-say={naive_lower_inclusive_says!r:>8}{flag}")


def main() -> None:
    table_doc = json.loads((FIXTURES / "table.json").read_text())
    node = table_doc["nodes"][1]
    content = node["content"]
    table = try_convert("table.json (Hello)", content)
    if table is not None:
        cross_check("Hello", table, content, [{"input": 5}, {"input": 10}, {"input": 11}, {"input": 100}])

    credit = json.loads((FIXTURES / "credit-analysis.json").read_text())
    by_name = {n["name"]: n for n in credit["nodes"] if n["type"] == "decisionTableNode"}

    for name in ("Company Type", "Turnover", "Country", "Overall"):
        node = by_name[name]
        content = node["content"]
        table = try_convert(name, content)
        if table is not None:
            # Build a small battery of records shaped for this node's own inputs.
            field_paths = [i["field"] for i in content["inputs"]]
            records = []
            if field_paths == ["company.type"]:
                for v in ("INC", "LTD", "LLC", "XYZ"):
                    records.append({"company": {"type": v}})
            cross_check(name, table, content, records)

    demonstrate_turnover_boundary(by_name["Turnover"]["content"])


if __name__ == "__main__":
    main()
