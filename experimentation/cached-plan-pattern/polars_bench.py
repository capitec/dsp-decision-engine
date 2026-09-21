"""Q1 headline measurement: of polars' per-call cost on a fixed tree-shaped
expression, what fraction is the OPTIMIZER re-running (which a cached plan
removes), measured entirely through polars' OWN supported public API —
no `rtlf` needed for this part.

Method (two independent supported-API probes, cross-checked against each
other):

  1. `lf.collect()` (optimize + build physical plan + execute) vs.
     `lf.collect(optimizations=QueryOptFlags.none())` (build physical plan +
     execute, optimizer passes turned off). The delta is a first estimate
     of "the optimizer's own share".

  2. `lf.explain(optimized=True)` calls the exact same internal
     `to_alp_optimized()` that both `collect()` and rtlf's own
     `RealtimeLazyFrame::new()` call, then formats the IR to a string.
     `lf.explain(optimized=False)` formats the RAW (unoptimized) IR — same
     formatting cost, no optimizer. **The naive read of
     `explain(optimized=True)` alone is a trap** (see `explain_baseline` in
     the results / RESULTS.md "what surprised me": for a deep plan, STRING
     FORMATTING dominates `explain()`, not the optimizer — the two explain
     costs are within a few percent of each other). The corrected estimate
     is the delta `explain(True) - explain(False)`, which cancels the
     formatting cost and leaves (approximately) the optimizer's own time.

Method 1 and the corrected method 2 are reported side by side specifically
because they were found to disagree badly before the formatting-cost
correction, and agree closely after it — that disagreement, and its
resolution, is itself a finding (recorded in RESULTS.md), not swept under
the rug.

Neither method can isolate PHYSICAL PLAN CONSTRUCTION (`create_physical_plan`)
from execution — that is unavoidable through any public `collect()` call, by
design (there is no supported API to hold a compiled physical plan across
calls). That is exactly the gap `rtlf`'s `CompiledRealtimeLazyFrame` fills
via unofficial internal APIs (`rtlf_bench.py`, separate file, only if the
build succeeds).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from polars.lazyframe.opt_flags import QueryOptFlags

sys.path.insert(0, str(Path(__file__).parent.parent / "tree-codegen-vs-interpreted"))
import tree_shapes as ts  # noqa: E402

from tree_to_polars import rows_to_polars_df, to_polars_expr  # noqa: E402

RESULTS = Path(__file__).parent / "results.jsonl"
N_REPS = 15


def append_result(record: dict) -> None:
    record["ts"] = time.time()
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record))


def min_time_s(fn, reps: int = N_REPS) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def probe(lf: pl.LazyFrame, *, reps: int = N_REPS) -> dict:
    """The four supported-API timings for one lazyframe, at whatever batch
    size its underlying DataFrame already has."""
    # Warm every code path once (import/JIT-ish warmup inside polars' Rust
    # side, first-call allocator warmup, etc.) before any timed rep.
    lf.collect()
    lf.collect(optimizations=QueryOptFlags.none())
    lf.explain(optimized=True)
    lf.explain(optimized=False)

    t_collect = min_time_s(lambda: lf.collect(), reps)
    t_collect_noopt = min_time_s(lambda: lf.collect(optimizations=QueryOptFlags.none()), reps)
    t_explain_opt = min_time_s(lambda: lf.explain(optimized=True), reps)
    t_explain_raw = min_time_s(lambda: lf.explain(optimized=False), reps)

    optimizer_only_via_explain = max(0.0, t_explain_opt - t_explain_raw)
    optimizer_delta_via_collect = max(0.0, t_collect - t_collect_noopt)

    return {
        "collect_s": t_collect,
        "collect_no_opt_s": t_collect_noopt,
        "explain_optimized_s": t_explain_opt,
        "explain_raw_s": t_explain_raw,
        "optimizer_only_via_explain_delta_s": optimizer_only_via_explain,
        "optimizer_frac_via_explain_delta": optimizer_only_via_explain / t_collect if t_collect else float("nan"),
        "optimizer_delta_via_collect_s": optimizer_delta_via_collect,
        "optimizer_frac_via_collect_delta": optimizer_delta_via_collect / t_collect if t_collect else float("nan"),
    }


# ---------------------------------------------------------------------------
# Part A — depth sweep, linear `pl.when` chain, batch=1000 (rtlf's own
# benchmark convention: BATCH_SIZE=1000, RUNS_PER_DEPTH=10 — here N_REPS=15,
# minimum not mean, per this repo's own measurement convention).
# ---------------------------------------------------------------------------

def build_linear_expr(depth: int, nfeat: int) -> pl.Expr:
    e = pl.lit(0.0)
    for d in range(depth):
        feat = f"feat_{d % nfeat}"
        e = pl.when(pl.col(feat) < float(d)).then(pl.lit(float(d))).otherwise(e)
    return e.alias("score")


def part_a_depth_sweep(batch_size: int, depths, *, label: str) -> None:
    rng = np.random.default_rng(0)
    for depth in depths:
        nfeat = min(depth, 10) or 1
        schema = {f"feat_{i}": pl.Float64 for i in range(nfeat)}
        df = pl.DataFrame(
            {f"feat_{i}": rng.uniform(0, max(depth, 1), batch_size) for i in range(nfeat)},
            schema=schema,
        )
        expr = build_linear_expr(depth, nfeat)
        lf = df.lazy().select(expr)
        stats = probe(lf)
        append_result({
            "experiment": "polars_depth_sweep",
            "label": label,
            "batch_size": batch_size,
            "depth": depth,
            **stats,
        })


# ---------------------------------------------------------------------------
# Part B — the three realistic decider2-shaped trees, at both 100k rows
# (ns/row, comparable to the numba engines) and a single record (the
# realtime `score()` floor).
# ---------------------------------------------------------------------------

def build_shape_lf(shape, n_rows: int, seed_row: int):
    import zlib
    numeric, string_codes = ts.make_rows(shape, n_rows, seed=zlib.crc32(shape.name.encode()) & 0xFFFF ^ seed_row)
    df = rows_to_polars_df(shape, numeric, string_codes)
    expr = to_polars_expr(shape)
    return df.lazy().select(expr), df, numeric, string_codes


def part_b_realistic_shapes() -> None:
    shapes = [
        ts.build_credit_tree(target_leaves=20, seed=7),
        ts.build_full_binary(5, seed=1),
        ts.build_one_sided_chain(100, seed=3),
    ]
    for shape in shapes:
        for n_rows, tag in [(100_000, "batch_100k"), (1, "single_record")]:
            lf, df, numeric, string_codes = build_shape_lf(shape, n_rows, seed_row=0)
            reps = N_REPS if n_rows > 1 else 200
            stats = probe(lf, reps=reps)
            ns_per_row = stats["collect_s"] / n_rows * 1e9
            append_result({
                "experiment": "polars_realistic_shape",
                "shape": shape.name,
                "n_rows": n_rows,
                "tag": tag,
                "ns_per_row_collect": ns_per_row,
                **stats,
            })


if __name__ == "__main__":
    RESULTS.parent.mkdir(exist_ok=True)
    print("=== Part A: linear-chain depth sweep, batch=1000 ===")
    part_a_depth_sweep(1000, [1, 5, 10, 25, 50, 100, 200, 400], label="linear_b1000")
    print("=== Part A2: linear-chain depth sweep, batch=1 (single record) ===")
    part_a_depth_sweep(1, [1, 5, 10, 25, 50, 100, 200, 400], label="linear_b1")
    print("=== Part B: realistic decider2-shaped trees ===")
    part_b_realistic_shapes()
    print("DONE")
