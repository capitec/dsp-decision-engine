"""Rust-via-C-ABI-called-from-inside-njit vs the numba array walker vs the
Rust-via-PyO3 baseline (§U) vs codegen (§P/§Q) — the sixth tree-engine
option for decider2.

Reuses `tree-codegen-vs-interpreted/{tree_shapes,interpreted_kernel}.py`
UNCHANGED — identical tree shapes, identical crc32-seeded row generation,
identical struct-of-arrays encoding — exactly as `rust-tree-interpreter/
run_experiment.py` did, so the three engines' numbers sit on the same rows
and the same trees. decider2/src is never imported or modified here.

Measures, per shape:
  * numba array walker (existing `interpreted_kernel.walk_batch`, unchanged)
  * C-ABI per-row  (`walk_batch_per_row_cabi` — the fusion-preserving shape)
  * C-ABI per-batch (`walk_batch_single_call_cabi` — one call, faster, not
    fusable)

Run:
    <repo>/.venv/bin/python experimentation/rust-cabi-in-kernel/bench_perf.py

Writes results.jsonl (flushed after every measurement).
"""
from __future__ import annotations

import gc
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "tree-codegen-vs-interpreted"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SIBLING))

import tree_shapes as ts  # noqa: E402
import interpreted_kernel as ik  # noqa: E402  (the numba walker, unmodified)
import tree_walk_cabi as tw  # noqa: E402

N_ROWS = 100_000
N_TIMING_RUNS = 7
RESULTS_PATH = HERE / "results.jsonl"


def check_memory() -> None:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        info[key] = int(rest.strip().split()[0])
    available_gb = info["MemAvailable"] / (1024 * 1024)
    print(f"[mem] MemAvailable = {available_gb:.1f} GiB")
    if available_gb < 8.0:
        raise SystemExit(
            f"refusing to run: only {available_gb:.1f} GiB available, need >= 8 GiB "
            "(this machine has been OOM-killed before)."
        )


def append_result(record: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def timed_runs_ns_per_row(fn, args, n_rows: int, *, runs: int = N_TIMING_RUNS) -> list[float]:
    """ns/row for EACH of `runs` calls — §Q found the range mattered more
    than the mean, so every run is kept, not just the min."""
    out = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        out.append(dt / n_rows * 1e9)
    return out


def build_matrices(flat: ts.FlatTree, numeric: dict, string_codes: dict, n_rows: int):
    if flat.numeric_features:
        numeric_matrix = np.column_stack([numeric[f] for f in flat.numeric_features])
    else:
        numeric_matrix = np.zeros((n_rows, 0))
    if flat.string_features:
        string_matrix = np.column_stack([string_codes[f] for f in flat.string_features]).astype(np.int32)
    else:
        string_matrix = np.zeros((n_rows, 0), dtype=np.int32)
    return np.ascontiguousarray(numeric_matrix, dtype=np.float64), np.ascontiguousarray(string_matrix, dtype=np.int32)


def warm_up() -> None:
    """Pay numba's process-wide LLVM warm-up and every kernel's one-ever
    compile before any timed measurement — neither engine's cost. Rust has
    no equivalent to pay (compiled ahead of time); the .so is already
    loaded and its symbols resolved at import of `tree_walk_cabi`."""
    dummy = ts.TreeShape(
        name="__warmup__", root=ts.Leaf(0.0),
        numeric_ranges={"x": (0.0, 1.0)}, string_categories={},
        leaf_count=1, node_count=1,
    )
    flat = ts.to_flat_tree(dummy)
    numeric = np.zeros((4, 1))
    strings = np.zeros((4, 0), dtype=np.int32)
    out = np.zeros(4)
    ik.walk_batch(numeric, strings, flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
                  flat.pat_start, flat.pat_count, flat.patterns, flat.left, flat.right,
                  flat.leaf_value, out)
    assert len(ik.walk_batch.signatures) == 1
    tw.walk_batch_per_row_cabi(flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
                                flat.pat_start, flat.pat_count, flat.patterns,
                                flat.left, flat.right, flat.leaf_value, numeric, strings, out)
    status = tw.walk_batch_single_call_cabi(flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
                                             flat.pat_start, flat.pat_count, flat.patterns,
                                             flat.left, flat.right, flat.leaf_value, numeric, strings, out)
    assert status == 0


def process_shape(shape: ts.TreeShape) -> dict:
    print(f"\n=== {shape.name} (leaves={shape.leaf_count}, nodes={shape.node_count}) ===")
    seed = zlib.crc32(shape.name.encode()) & 0xFFFF
    numeric, string_codes = ts.make_rows(shape, N_ROWS, seed=seed)
    flat = ts.to_flat_tree(shape)
    num_matrix, str_matrix = build_matrices(flat, numeric, string_codes, N_ROWS)

    tree_args = (flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
                 flat.pat_start, flat.pat_count, flat.patterns, flat.left, flat.right, flat.leaf_value)

    # ---- numba array walker ----
    numba_out = np.zeros(N_ROWS)
    numba_args = (num_matrix, str_matrix, *tree_args, numba_out)
    ik.walk_batch(*numba_args)  # warm dispatch path
    numba_runs = timed_runs_ns_per_row(ik.walk_batch, numba_args, N_ROWS)

    # ---- C-ABI, per-row (fusion-preserving shape) ----
    cabi_row_out = np.zeros(N_ROWS)
    cabi_row_args = (*tree_args, num_matrix, str_matrix, cabi_row_out)
    tw.walk_batch_per_row_cabi(*cabi_row_args)
    cabi_row_runs = timed_runs_ns_per_row(tw.walk_batch_per_row_cabi, cabi_row_args, N_ROWS)

    # ---- C-ABI, per-batch (one call, faster, not fusable) ----
    cabi_batch_out = np.zeros(N_ROWS)
    cabi_batch_args = (*tree_args, num_matrix, str_matrix, cabi_batch_out)
    tw.walk_batch_single_call_cabi(*cabi_batch_args)

    def batch_call(*args):
        status = tw.walk_batch_single_call_cabi(*args)
        assert status == 0, f"walk_batch_single_call_cabi reported status {status}"

    cabi_batch_runs = timed_runs_ns_per_row(batch_call, cabi_batch_args, N_ROWS)

    # ---- fairness: identical answers, asserted ----
    if not np.array_equal(numba_out, cabi_row_out):
        mismatches = int(np.sum(numba_out != cabi_row_out))
        raise AssertionError(
            f"{shape.name}: {mismatches}/{N_ROWS} rows disagree between numba and C-ABI per-row. "
            f"First few: numba={numba_out[:5]} cabi={cabi_row_out[:5]}"
        )
    if not np.array_equal(numba_out, cabi_batch_out):
        raise AssertionError(f"{shape.name}: C-ABI per-batch disagrees with numba")
    print(f"  answers identical (numba == cabi/row == cabi/batch) across {N_ROWS} rows: OK")
    print(f"  numba        ns/row: min {min(numba_runs):.1f}, range {min(numba_runs):.1f}-{max(numba_runs):.1f}")
    print(f"  cabi/row     ns/row: min {min(cabi_row_runs):.1f}, range {min(cabi_row_runs):.1f}-{max(cabi_row_runs):.1f}")
    print(f"  cabi/batch   ns/row: min {min(cabi_batch_runs):.1f}, range {min(cabi_batch_runs):.1f}-{max(cabi_batch_runs):.1f}")

    record = {
        "shape": shape.name,
        "leaf_count": shape.leaf_count,
        "node_count": shape.node_count,
        "n_rows": N_ROWS,
        "numba_ns_per_row_runs": numba_runs,
        "cabi_row_ns_per_row_runs": cabi_row_runs,
        "cabi_batch_ns_per_row_runs": cabi_batch_runs,
        "answers_identical": True,
    }
    append_result(record)
    return record


def main() -> None:
    check_memory()
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text("")

    print("\n--- warm-up ---")
    t0 = time.perf_counter()
    warm_up()
    warmup_s = time.perf_counter() - t0
    print(f"warm-up: {warmup_s*1e3:.1f} ms")
    append_result({"event": "warmup", "warmup_s": warmup_s})

    shapes = [
        ts.build_credit_tree(target_leaves=20, seed=7),
        ts.build_full_binary(5, seed=1),
        ts.build_full_binary(7, seed=1),
        ts.build_full_binary(9, seed=1),
        ts.build_one_sided_chain(100, seed=3),
    ]

    results = [process_shape(s) for s in shapes]

    print("\n\n=== SUMMARY (ns/row, min-max of 7 runs) ===")
    header = f"{'shape':<20}{'leaves':>8}{'numba':>18}{'cabi/row':>18}{'cabi/batch':>18}"
    print(header)
    for r in results:
        nb = r["numba_ns_per_row_runs"]; cr = r["cabi_row_ns_per_row_runs"]; cb = r["cabi_batch_ns_per_row_runs"]
        print(
            f"{r['shape']:<20}{r['leaf_count']:>8}"
            f"{min(nb):>8.1f}-{max(nb):<9.1f}"
            f"{min(cr):>8.1f}-{max(cr):<9.1f}"
            f"{min(cb):>8.1f}-{max(cb):<9.1f}"
        )

    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
