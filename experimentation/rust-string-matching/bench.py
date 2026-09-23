"""The measurement: `ft1 > c1 AND ft2 > c2 AND ft3 ~ /^dog/` at four
selectivities (fraction of rows reaching the string node) x the strategies,
on one cardinality per process (see data.py; polars' global string cache
would otherwise let one cardinality's dictionary leak into the other's),
and for TWO forms of the owner's condition:

  regex        `^dog` via the regex crate  (frame tier: `str.contains`)
  starts_with  `dog`  via `starts_with`    (frame tier: `str.starts_with`)

decider2's `TStringMatchType` distinguishes them, and the second is the
honest cost floor for what `^dog` actually asks.

Every strategy is timed END TO END for one batch — its own prep (the
frame-tier expression, the dictionary cast + mask, or the buffer grab)
plus its kernel — and the kernel alone, so the reader can see which part
each strategy is paying for. Median of REPS runs, ns/row. Answers are
asserted identical across strategies at every cell.

    python bench.py low|high [n_rows]
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, polars as pl
from binding import StringBuffers, compile_pattern, make_ptr_table, match_rows_py
from data import make_data, thresholds
import kernels

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
SELECTIVITIES = [0.0, 0.01, 0.5, 1.0]
REPS = 7
FORMS = [  # (kind, pattern, frame-tier expression, python reference)
    ("regex", "^dog", lambda s: s.str.contains("^dog"), lambda x: bool(re.match("^dog", x))),
    ("starts_with", "dog", lambda s: s.str.starts_with("dog"), lambda x: x.startswith("dog")),
]


def med(fn, n, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e9 / n


def main():
    card = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    df, _ = make_data(n, card, seed=42)
    s = df["ft3"]
    ft1, ft2 = df["ft1"].to_numpy(), df["ft2"].to_numpy()
    out = np.empty(n, dtype=np.bool_)
    PT = make_ptr_table()
    print(f"cardinality={card} n={n} polars threads={pl.thread_pool_size()} mean bytes/str={s.str.len_bytes().mean():.1f}")

    # --- strategy-independent preps ---------------------------------------
    def prep_cast():
        cat = s.cast(pl.Categorical)
        return cat.to_physical().to_numpy().view(np.int32), cat.cat.get_categories()
    t_cast = med(prep_cast, n)
    codes, cats = prep_cast()
    dsb = StringBuffers(cats)
    n_distinct = len(cats)
    t_bufs = med(lambda: StringBuffers(s), n)
    sb = StringBuffers(s)
    t_call = None
    print(f"distinct={n_distinct}; prep ns/row: cast(Categorical)={t_cast:.2f}  column buffers via _get_buffers()={t_bufs:.2f}")

    rows = []
    for kind, pattern, frame_expr, pyref in FORMS:
        pid = np.int32(compile_pattern(pattern, kind))
        ref = np.array([pyref(x) for x in s.to_list()])
        t_frame = med(lambda: frame_expr(s).to_numpy(), n)
        t_mask = med(lambda: match_rows_py(pid, dsb), n)
        t_batch = med(lambda: match_rows_py(pid, sb), n)
        fm = frame_expr(s).to_numpy(); mask = match_rows_py(pid, dsb)
        assert np.array_equal(fm, ref) and np.array_equal(match_rows_py(pid, sb).astype(bool), ref) \
            and np.array_equal(mask[codes].astype(bool), ref)
        c1, c2 = thresholds(0.5)  # warm-up / cache load
        kernels.baseline_no_string(ft1, ft2, c1, c2, out)
        kernels.frame_tier(ft1, ft2, fm, c1, c2, out)
        kernels.per_category(ft1, ft2, codes, mask, c1, c2, out)
        kernels.lazy_rust(ft1, ft2, sb.offsets, sb.values, PT, pid, c1, c2, out)
        kernels.lazy_rust_via_dict(ft1, ft2, codes, dsb.offsets, dsb.values, PT, pid, c1, c2, out)
        kernels.call_overhead_probe(PT, 10)
        if t_call is None:
            t_call = med(lambda: kernels.call_overhead_probe(PT, n), n)
            print(f"extern C call through pointer-table argument: {t_call:.2f} ns/call")
        print(f"\n### {kind} {pattern!r}: match rate={ref.mean():.3f}; prep ns/row: polars frame expr={t_frame:.2f}  "
              f"rust mask over dict={t_mask:.3f}  rust match_rows over column={t_batch:.2f}")
        for sel in SELECTIVITIES:
            c1, c2 = thresholds(sel)
            expected = (ft1 > c1) & (ft2 > c2) & ref
            reached = int(((ft1 > c1) & (ft2 > c2)).sum())
            fns = {
                "frame_tier": lambda: kernels.frame_tier(ft1, ft2, fm, c1, c2, out),
                "per_category": lambda: kernels.per_category(ft1, ft2, codes, mask, c1, c2, out),
                "lazy_rust": lambda: kernels.lazy_rust(ft1, ft2, sb.offsets, sb.values, PT, pid, c1, c2, out),
                "lazy_rust_via_dict": lambda: kernels.lazy_rust_via_dict(ft1, ft2, codes, dsb.offsets, dsb.values, PT, pid, c1, c2, out),
            }
            k = {name: med(fn, n) for name, fn in fns.items()}
            k["baseline_no_string"] = med(lambda: kernels.baseline_no_string(ft1, ft2, c1, c2, out), n)
            for fn in fns.values():
                fn(); assert np.array_equal(out, expected)
            e2e = {
                "frame_tier": t_frame + k["frame_tier"],
                "rust_batch": t_bufs + t_batch + k["frame_tier"],  # Rust needs the same materialised buffers the lazy kernel does
                "per_category_from_utf8": t_cast + t_mask + k["per_category"],
                "per_category_from_categorical": t_mask + k["per_category"],
                "lazy_rust_column": t_bufs + k["lazy_rust"],
                "lazy_rust_dict_from_categorical": k["lazy_rust_via_dict"],
            }
            rows.append({"cardinality": card, "n": n, "kind": kind, "pattern": pattern, "selectivity": sel,
                         "reached": reached, "distinct": n_distinct, "kernel_only": k,
                         "prep": {"frame_expr": t_frame, "cast_categorical": t_cast, "rust_mask_dict": t_mask,
                                  "column_buffers": t_bufs, "rust_match_rows_column": t_batch},
                         "end_to_end": e2e, "call_overhead_ns": t_call})
            print(f"-- {sel:.0%} ({reached} rows reach the node) kernel-only: " +
                  " ".join(f"{a}={b:.2f}" for a, b in k.items()))
            print("   e2e: " + "  ".join(f"{a}={b:.2f}" for a, b in e2e.items()))
    with open(RESULTS, "a") as f:
        for r in rows:
            f.write(json.dumps({"experiment": "rust-string-matching", "item": "bench", **r}) + "\n")


if __name__ == "__main__":
    main()
