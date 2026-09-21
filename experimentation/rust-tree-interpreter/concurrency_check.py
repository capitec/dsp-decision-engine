"""Does Rust's GIL release under concurrent load behave like numba's
`nogil=True` (EXPERIMENTS.md §N4: flat p99 from 1-16 threads) or does it
have its own convoy risk?

`walk_batch` releases the GIL for its entire row loop (`py.detach` in
lib.rs) — the same shape numba's `@njit(nogil=True)` gives, but arrived at
by the Rust compiler's ownership rules rather than a boolean flag applied
to arbitrary code. This measures whether that release delivers real
concurrent throughput (multiple OS threads actually running the walk in
parallel, not serialized behind the GIL) the way §N4 needed nogil=True to
prove for numba.

Not a full re-run of §N4 (that measured decider2's whole `score()` path
under concurrent single-record traffic). This isolates the same question
§N4 asked — does releasing the GIL for the compute actually let concurrent
callers run in parallel — for the Rust batch kernel specifically.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SIBLING = HERE.parent / "tree-codegen-vs-interpreted"
sys.path.insert(0, str(SIBLING))

import tree_shapes as ts  # noqa: E402
import rust_tree_interpreter as rti  # noqa: E402

RESULTS_PATH = HERE / "results.jsonl"
BATCH_ROWS = 10_000
DURATION_S = 1.5


def append_result(record: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def main() -> None:
    shape = ts.build_credit_tree(target_leaves=20, seed=7)
    flat = ts.to_flat_tree(shape)
    numeric, string_codes = ts.make_rows(shape, BATCH_ROWS, seed=123)
    num_cols = np.ascontiguousarray(
        np.column_stack([numeric[f] for f in flat.numeric_features]), dtype=np.float64
    )
    str_cols = np.ascontiguousarray(
        np.column_stack([string_codes[f] for f in flat.string_features]).astype(np.int32)
        if flat.string_features else np.zeros((BATCH_ROWS, 0), dtype=np.int32)
    )
    tree_args = (flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
                 flat.pat_start, flat.pat_count, flat.patterns, flat.left, flat.right, flat.leaf_value)

    # single-thread baseline throughput (calls/sec of one walk_batch call)
    out = np.zeros(BATCH_ROWS)
    args = (num_cols, str_cols, *tree_args, out)
    rti.walk_batch(*args)  # warm

    def single_thread_rate(seconds: float) -> float:
        n = 0
        t_end = time.perf_counter() + seconds
        local_out = np.zeros(BATCH_ROWS)
        while time.perf_counter() < t_end:
            rti.walk_batch(num_cols, str_cols, *tree_args, local_out)
            n += 1
        return n / seconds

    baseline_rate = single_thread_rate(DURATION_S)
    print(f"single-thread baseline: {baseline_rate:.1f} batch-calls/s "
          f"({baseline_rate * BATCH_ROWS:.0f} rows/s)")

    results = {"event": "concurrency_check", "batch_rows": BATCH_ROWS,
               "duration_s": DURATION_S, "baseline_calls_per_s": baseline_rate,
               "by_threads": {}}

    for n_threads in (1, 2, 4, 8):
        counts = [0] * n_threads
        max_call_ns = [0.0] * n_threads
        stop_at = None

        def worker(idx: int):
            local_out = np.zeros(BATCH_ROWS)
            n = 0
            worst = 0.0
            while time.perf_counter() < stop_at:
                t0 = time.perf_counter()
                rti.walk_batch(num_cols, str_cols, *tree_args, local_out)
                dt = time.perf_counter() - t0
                if dt > worst:
                    worst = dt
                n += 1
            counts[idx] = n
            max_call_ns[idx] = worst * 1e9

        stop_at = time.perf_counter() + DURATION_S
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        t_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t_start

        total_calls = sum(counts)
        aggregate_rate = total_calls / wall
        per_thread_rate = aggregate_rate / n_threads
        worst_single_call_us = max(max_call_ns) / 1e3
        scaling_vs_1thread = aggregate_rate / baseline_rate

        print(f"  threads={n_threads:<2} aggregate={aggregate_rate:>9.1f} calls/s "
              f"({aggregate_rate*BATCH_ROWS:>12.0f} rows/s, {scaling_vs_1thread:.2f}x vs 1-thread) "
              f"per-thread={per_thread_rate:>8.1f} calls/s  worst single call={worst_single_call_us:.1f} us")

        results["by_threads"][n_threads] = {
            "aggregate_calls_per_s": aggregate_rate,
            "aggregate_rows_per_s": aggregate_rate * BATCH_ROWS,
            "scaling_vs_1_thread": scaling_vs_1thread,
            "per_thread_calls_per_s": per_thread_rate,
            "worst_single_call_us": worst_single_call_us,
        }

    append_result(results)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
