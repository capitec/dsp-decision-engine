"""Q2: composed, cached njit callables (`composed_callables.py`) vs.
decider2's real codegen path (`decider2.trees.codegen`, driven exactly as
`../tree-codegen-vs-interpreted/codegen_bench.py` drives it) vs. the
sibling experiment's already-measured generic interpreter — cited, not
re-measured, per the brief's instruction not to duplicate that work.

Same three shapes as `polars_bench.py`'s Part B, same row seeds, so all
three engines (four, counting the cited interpreter) answer identically
and are compared on identical data:

  - credit_tree (20 leaves, mixed numeric+string) — decider2's realistic shape
  - full_binary_d5 (32 leaves) — small controlled full-binary
  - one_sided_chain_100 (101 leaves) — deep policy-waterfall shape

Measures, per shape:
  - ns/row at 100,000 rows (minimum of 7, matching this repo's own
    "measure the kernel, not the scheduler" convention)
  - single-record latency (minimum of many repeated 1-row batch calls,
    Python call boundary included — this is what actually reaches a
    `score()` caller)
  - numba `.signatures` length for composed_callables' three kernels,
    checked across ALL shapes in one process, to verify the "compiled once,
    ever" claim the whole pattern rests on.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tree-codegen-vs-interpreted"))
import tree_shapes as ts  # noqa: E402
import codegen_bench  # noqa: E402
from interpreted_kernel import walk_batch as interp_walk_batch  # noqa: E402

import composed_callables as cc  # noqa: E402

RESULTS = Path(__file__).parent / "results.jsonl"
N_TIMING_RUNS = 7
N_SINGLE_REPS = 2000


def append_result(record: dict) -> None:
    record["ts"] = time.time()
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record))


def min_ns_per_row(fn, args, n_rows: int, runs: int = N_TIMING_RUNS) -> float:
    best = float("inf")
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best / n_rows * 1e9


def min_single_call_ns(fn, args, reps: int = N_SINGLE_REPS) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best * 1e9


def matrices_for(shape, flat, n_rows, seed_row):
    """Column order for the INTERPRETER / composed-callables engines:
    `flat.numeric_features`/`flat.string_features` (alphabetically sorted,
    `to_flat_tree`'s convention)."""
    numeric, string_codes = ts.make_rows(
        shape, n_rows, seed=zlib.crc32(shape.name.encode()) & 0xFFFF ^ seed_row
    )
    numeric_matrix = np.column_stack([numeric[f] for f in flat.numeric_features]) if flat.numeric_features else np.zeros((n_rows, 0))
    if flat.string_features:
        string_matrix = np.column_stack([string_codes[f] for f in flat.string_features]).astype(np.int32)
    else:
        string_matrix = np.zeros((n_rows, 0), dtype=np.int32)
    return (
        np.ascontiguousarray(numeric_matrix, dtype=np.float64),
        np.ascontiguousarray(string_matrix, dtype=np.int32),
        numeric,
        string_codes,
    )


def codegen_matrices_for(emitted, numeric: dict, string_codes: dict, n_rows: int):
    """Column order for CODEGEN's row_loop: `emitted.features`/
    `emitted.string_features`, in the order codegen's own emitted signature
    declares them — NOT necessarily the same order as `flat`'s (this is
    exactly `run_experiment.py::build_codegen_matrices`, ported here rather
    than imported, since it is a 6-line function of `run_experiment.py`'s
    own module-level `N_ROWS`, not a reusable export)."""
    numeric_features_in_order = [f for f in emitted.features if f not in emitted.string_features]
    feat_matrix = (
        np.column_stack([numeric[f] for f in numeric_features_in_order])
        if numeric_features_in_order else np.zeros((n_rows, 0))
    )
    if emitted.string_features:
        str_matrix = np.column_stack([string_codes[f] for f in emitted.string_features]).astype(np.int32)
    else:
        str_matrix = np.zeros((n_rows, 0), dtype=np.int32)
    return np.ascontiguousarray(feat_matrix, dtype=np.float64), np.ascontiguousarray(str_matrix, dtype=np.int32)


def process_shape(shape) -> None:
    print(f"--- {shape.name} ---")
    flat = ts.to_flat_tree(shape)

    # -- codegen: real production path, unmodified, via the sibling's harness.
    codegen = codegen_bench.compile_codegen_tree(shape)

    # -- correctness at a small size against the (already-validated) interpreter.
    numeric_matrix, string_matrix, numeric, string_codes = matrices_for(shape, flat, 2000, seed_row=999)
    codegen_feat_matrix, codegen_str_matrix = codegen_matrices_for(codegen.emitted, numeric, string_codes, 2000)
    out_interp = np.zeros(2000)
    interp_walk_batch(
        numeric_matrix, string_matrix, flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
        flat.pat_start, flat.pat_count, flat.patterns, flat.left, flat.right, flat.leaf_value, out_interp,
    )
    out_codegen = np.zeros(2000)
    codegen.row_loop(codegen_feat_matrix, codegen_str_matrix, out_codegen)
    out_composed = cc.run_batch(flat, numeric_matrix, string_matrix)
    assert np.array_equal(out_interp, out_codegen), f"{shape.name}: codegen mismatch"
    assert np.array_equal(out_interp, out_composed), f"{shape.name}: composed mismatch"
    print(f"  correctness OK (codegen, composed both match interpreter), n={2000}")

    for n_rows, tag in [(100_000, "batch_100k"), (1, "single_record")]:
        numeric_matrix, string_matrix, numeric, string_codes = matrices_for(shape, flat, n_rows, seed_row=0)
        codegen_feat_matrix, codegen_str_matrix = codegen_matrices_for(codegen.emitted, numeric, string_codes, n_rows)
        cc_args = cc.build_plan(flat)

        if n_rows > 1:
            codegen_ns = min_ns_per_row(codegen.row_loop, (codegen_feat_matrix, codegen_str_matrix, np.zeros(n_rows)), n_rows)
            composed_out = np.empty(n_rows, dtype=np.float64)
            composed_ns = min_ns_per_row(
                cc.walk_batch_composed,
                (numeric_matrix, string_matrix, *cc_args, composed_out),
                n_rows,
            )
        else:
            out1 = np.zeros(1)
            codegen_ns = min_single_call_ns(codegen.row_loop, (codegen_feat_matrix, codegen_str_matrix, out1))
            out2 = np.empty(1, dtype=np.float64)
            composed_ns = min_single_call_ns(
                cc.walk_batch_composed,
                (numeric_matrix, string_matrix, *cc_args, out2),
            )

        append_result({
            "experiment": "q2_composed_vs_codegen",
            "shape": shape.name,
            "n_rows": n_rows,
            "tag": tag,
            "codegen_ns_per_row_or_call": codegen_ns,
            "composed_ns_per_row_or_call": composed_ns,
            "composed_over_codegen_ratio": composed_ns / codegen_ns if codegen_ns else float("nan"),
            "emitted_lines": codegen.emitted_lines,
            "codegen_timings": codegen.timings,
        })

    append_result({
        "experiment": "q2_signature_check",
        "shape": shape.name,
        "signatures_after_this_shape": cc.signature_counts(),
    })


if __name__ == "__main__":
    shapes = [
        ts.build_credit_tree(target_leaves=20, seed=7),
        ts.build_full_binary(5, seed=1),
        ts.build_one_sided_chain(100, seed=3),
    ]
    print("signatures before any shape:", cc.signature_counts())
    for shape in shapes:
        process_shape(shape)
    print("signatures after ALL shapes (should be unchanged - 1 each):", cc.signature_counts())
    print("DONE")
