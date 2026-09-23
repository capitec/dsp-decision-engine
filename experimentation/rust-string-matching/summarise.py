"""results.jsonl -> the markdown tables in RESULTS.md."""
import json, sys
from pathlib import Path
rows = [json.loads(l) for l in (Path(__file__).resolve().parent / "results.jsonl").read_text().splitlines()]
bench = [r for r in rows if r["item"] == "bench"]
COLS = [("frame_tier", "1. frame tier (polars expr + kernel)"),
        ("rust_batch", "1b. Rust batch (match_rows column + kernel)"),
        ("per_category_from_utf8", "2. per-category, Utf8 in (cast + mask + kernel)"),
        ("per_category_from_categorical", "2b. per-category, Categorical in (mask + kernel)"),
        ("lazy_rust_column", "3. lazy Rust over column (buffers + kernel)"),
        ("lazy_rust_dict_from_categorical", "3b. lazy Rust over dict, Categorical in (kernel)")]
for kind in ("regex", "starts_with"):
    for card in ("low", "high"):
        sub = [r for r in bench if r["kind"] == kind and r["cardinality"] == card]
        if not sub: continue
        r0 = sub[0]; p = r0["prep"]
        print(f"\n#### {kind} `{r0['pattern']}` — cardinality **{card}** ({r0['distinct']:,} distinct, n={r0['n']:,})\n")
        print(f"prep, ns/row: polars expr {p['frame_expr']:.1f} · cast(Categorical) {p['cast_categorical']:.1f} · "
              f"Rust mask over dict {p['rust_mask_dict']:.3f} · column buffers {p['column_buffers']:.1f} · Rust match_rows over column {p['rust_match_rows_column']:.1f}\n")
        print("| strategy (end to end, ns/row) | " + " | ".join(f"{r['selectivity']:.0%}" for r in sub) + " |")
        print("|---|" + "---:|" * len(sub))
        for key, label in COLS:
            print(f"| {label} | " + " | ".join(f"{r['end_to_end'][key]:.1f}" for r in sub) + " |")
        print("| *kernel only: lazy Rust* | " + " | ".join(f"*{r['kernel_only']['lazy_rust']:.1f}*" for r in sub) + " |")
        print("| *kernel only: no string node* | " + " | ".join(f"*{r['kernel_only']['baseline_no_string']:.1f}*" for r in sub) + " |")
