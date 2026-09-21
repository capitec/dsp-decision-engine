"""A Rust (PyO3 + maturin) tree interpreter vs. the numba array walker vs.
codegen's recorded numbers — the fifth option for decider2's tree engine.

Reuses `tree-codegen-vs-interpreted/{tree_shapes,interpreted_kernel}.py`
UNCHANGED: identical tree shapes, identical row generation (same crc32
seeding, same feature ordering), identical struct-of-arrays encoding. That
is what makes "answers identical to the numba walker" a meaningful
assertion rather than an artefact of two different encodings.

decider2/src/ is never imported or modified here.

Run:
    <repo>/.venv/bin/python experimentation/rust-tree-interpreter/run_experiment.py

Writes results.jsonl (flushed after every measurement).
"""
from __future__ import annotations

import gc
import json
import random
import string
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SIBLING = HERE.parent / "tree-codegen-vs-interpreted"
sys.path.insert(0, str(SIBLING))

import tree_shapes as ts  # noqa: E402
import interpreted_kernel as ik  # noqa: E402  (the numba walker, unmodified)

import rust_tree_interpreter as rti  # noqa: E402

N_ROWS = 100_000
N_TIMING_RUNS = 7
N_SINGLE_RECORD_CALLS = 200_000  # per timing trial, for single-record ns/call
RESULTS_PATH = HERE / "results.jsonl"


def check_memory() -> None:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        info[key] = int(rest.strip().split()[0])  # kB
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
    """ns/row for EACH of `runs` calls (not just the min) — §Q found the
    range mattered more than the mean, so this harness keeps every run."""
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


def warm_up_numba() -> None:
    """Pay numba's process-wide LLVM warm-up and the interpreted kernel's
    one-ever compile before any timed measurement, exactly like
    tree-codegen-vs-interpreted/run_experiment.py does — neither engine's
    cost, and Rust has no equivalent to pay at all."""
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
    ik._walk_one(numeric[0], strings[0], flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
                 flat.pat_start, flat.pat_count, flat.patterns, flat.left, flat.right, flat.leaf_value)


def process_shape(shape: ts.TreeShape) -> dict:
    print(f"\n=== {shape.name} (leaves={shape.leaf_count}, nodes={shape.node_count}) ===")
    seed = zlib.crc32(shape.name.encode()) & 0xFFFF
    numeric, string_codes = ts.make_rows(shape, N_ROWS, seed=seed)
    flat = ts.to_flat_tree(shape)
    num_matrix, str_matrix = build_matrices(flat, numeric, string_codes, N_ROWS)

    tree_args = (flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
                 flat.pat_start, flat.pat_count, flat.patterns, flat.left, flat.right, flat.leaf_value)

    # ---- numba (the existing, already-warm array walker) ----
    numba_out = np.zeros(N_ROWS)
    sig_before = len(ik.walk_batch.signatures)
    numba_args = (num_matrix, str_matrix, *tree_args, numba_out)
    ik.walk_batch(*numba_args)  # warm the dispatch path
    sig_after = len(ik.walk_batch.signatures)
    numba_runs = timed_runs_ns_per_row(ik.walk_batch, numba_args, N_ROWS)

    # ---- rust (module-level walk_batch, GIL released for the whole walk) ----
    rust_out = np.zeros(N_ROWS)
    rust_args = (num_matrix, str_matrix, *tree_args, rust_out)
    rti.walk_batch(*rust_args)  # warm (page faults, branch predictor) — same courtesy as numba
    rust_runs = timed_runs_ns_per_row(rti.walk_batch, rust_args, N_ROWS)

    # ---- rust via PyTree (tree resident in Rust; only rows cross per call) ----
    ptree = rti.PyTree(*tree_args)
    rust_ptree_out = np.zeros(N_ROWS)
    ptree.walk_batch(num_matrix, str_matrix, rust_ptree_out)
    rust_ptree_runs = timed_runs_ns_per_row(ptree.walk_batch, (num_matrix, str_matrix, rust_ptree_out), N_ROWS)

    # ---- fairness: identical answers, asserted ----
    if not np.array_equal(numba_out, rust_out):
        mismatches = int(np.sum(numba_out != rust_out))
        raise AssertionError(
            f"{shape.name}: {mismatches}/{N_ROWS} rows disagree between numba and rust. "
            f"First few: numba={numba_out[:5]} rust={rust_out[:5]}"
        )
    if not np.array_equal(numba_out, rust_ptree_out):
        raise AssertionError(f"{shape.name}: PyTree.walk_batch disagrees with numba")
    print(f"  answers identical (numba == rust == rust/PyTree) across {N_ROWS} rows: OK")

    # ---- single-record latency: one row through the FFI, many independent calls ----
    row0 = num_matrix[0]
    strrow0 = str_matrix[0]

    # bind locally so the loop measures CALL overhead, not repeated
    # attribute lookups on `ik`/`rti`/`ptree` (same convention the existing
    # min_ns_per_row helper uses: `fn` is already bound before the loop).
    numba_walk_one = ik._walk_one
    rust_walk_one = rti.walk_one
    ptree_walk_one = ptree.walk_one
    rust_noop = rti.noop
    k, fi, oc, th, ps, pc, pt, lf, rt, lv = tree_args  # unpack once, not per call

    def numba_one_loop(n):
        for _ in range(n):
            numba_walk_one(row0, strrow0, k, fi, oc, th, ps, pc, pt, lf, rt, lv)

    def rust_one_loop(n):
        for _ in range(n):
            rust_walk_one(row0, strrow0, k, fi, oc, th, ps, pc, pt, lf, rt, lv)

    def rust_ptree_one_loop(n):
        for _ in range(n):
            ptree_walk_one(row0, strrow0)

    def noop_loop(n):
        for _ in range(n):
            rust_noop()

    # warm each path once
    ik._walk_one(row0, strrow0, *tree_args)
    rti.walk_one(row0, strrow0, *tree_args)
    ptree.walk_one(row0, strrow0)
    rti.noop()

    single_runs = {}
    for label, fn in [
        ("numba_walk_one", numba_one_loop),
        ("rust_walk_one_full_args", rust_one_loop),
        ("rust_ptree_walk_one", rust_ptree_one_loop),
        ("rust_noop_call", noop_loop),
    ]:
        vals = []
        for _ in range(5):
            gc.collect()
            t0 = time.perf_counter()
            fn(N_SINGLE_RECORD_CALLS)
            dt = time.perf_counter() - t0
            vals.append(dt / N_SINGLE_RECORD_CALLS * 1e9)
        single_runs[label] = vals
        print(f"  single-record {label:<28}: min {min(vals):.1f} ns/call, range {min(vals):.1f}-{max(vals):.1f}")

    print(f"  numba batch ns/row   : min {min(numba_runs):.1f}, range {min(numba_runs):.1f}-{max(numba_runs):.1f}")
    print(f"  rust  batch ns/row   : min {min(rust_runs):.1f}, range {min(rust_runs):.1f}-{max(rust_runs):.1f}")
    print(f"  rust/PyTree ns/row   : min {min(rust_ptree_runs):.1f}, range {min(rust_ptree_runs):.1f}-{max(rust_ptree_runs):.1f}")

    record = {
        "shape": shape.name,
        "leaf_count": shape.leaf_count,
        "node_count": shape.node_count,
        "n_rows": N_ROWS,
        "numba_ns_per_row_runs": numba_runs,
        "rust_ns_per_row_runs": rust_runs,
        "rust_ptree_ns_per_row_runs": rust_ptree_runs,
        "numba_new_compile_triggered": sig_after > sig_before,
        "single_record_ns_per_call": single_runs,
        "answers_identical": True,
    }
    append_result(record)
    return record


def regex_experiment() -> dict:
    """§T's two regex cases, reproduced with Rust's `regex` crate instead
    of numba->libc. Compares against §T's own numbers:
        numba -> libc POSIX regex:      60.3 ns/call  (6.03 ms / 100k rows)
        polars' internal Rust regex:    42.2 ns/call  (4.22 ms / 100k rows)
        per-category mask (numba path): 0.12 ms total for 12 categories
    """
    print("\n=== regex: per-category mask vs. per-row ===")
    rng = random.Random(42)

    # per-category case: the credit tree's own string categories, plus a
    # synthetic 12-category list sized to match §T's own setup exactly.
    categories_12 = [f"category_{i:02d}" for i in range(12)]
    pattern_cat = "^category_0"  # matches categories 0-9 -> 10/12, representative selectivity

    t0 = time.perf_counter()
    matches, pure_ns = rti.regex_bench_categories(pattern_cat, categories_12)
    wall_s_cat = time.perf_counter() - t0
    overall_ns_per_call = wall_s_cat / len(categories_12) * 1e9
    print(f"  per-category (n=12): pure-rust {pure_ns:.1f} ns/call, "
          f"overall (incl. marshalling) {overall_ns_per_call:.1f} ns/call, "
          f"total wall {wall_s_cat*1e6:.1f} us")

    # per-row case: cardinality ~= row count (free text / account numbers),
    # the case where per-category degenerates and lazy in-kernel matching
    # is the right answer per §T's conclusion.
    rng2 = np.random.default_rng(7)
    alphabet = string.ascii_uppercase + string.digits
    rows = [
        "".join(rng.choices(alphabet, k=12))
        for _ in range(N_ROWS)
    ]
    pattern_row = "^A.*7"  # representative selective pattern, not anchored-only

    t0 = time.perf_counter()
    row_matches, row_pure_ns = rti.regex_bench_rows(pattern_row, rows)
    wall_s = time.perf_counter() - t0
    overall_ns_per_row = wall_s / N_ROWS * 1e9
    n_matched = sum(row_matches)
    print(f"  per-row (n={N_ROWS}): pure-rust {row_pure_ns:.2f} ns/call "
          f"({row_pure_ns * N_ROWS / 1e6:.2f} ms total), "
          f"overall (incl. marshalling) {overall_ns_per_row:.2f} ns/call "
          f"({wall_s*1e3:.2f} ms total), matched={n_matched}")

    record = {
        "event": "regex_experiment",
        "per_category": {
            "n_categories": len(categories_12),
            "pure_rust_ns_per_call": pure_ns,
            "overall_ns_per_call_incl_marshalling": overall_ns_per_call,
            "total_wall_us": wall_s_cat * 1e6,
        },
        "per_row": {
            "n_rows": N_ROWS,
            "pure_rust_ns_per_call": row_pure_ns,
            "overall_ns_per_call_incl_marshalling": overall_ns_per_row,
            "total_wall_ms": wall_s * 1e3,
            "n_matched": n_matched,
        },
    }
    append_result(record)
    return record


def main() -> None:
    check_memory()
    RESULTS_PATH.write_text("")  # fresh file this run

    print("\n--- warm-up (numba LLVM + interpreted-kernel compile; rust needs none) ---")
    t0 = time.perf_counter()
    warm_up_numba()
    numba_warmup_s = time.perf_counter() - t0
    print(f"numba warm-up: {numba_warmup_s*1e3:.1f} ms")
    append_result({"event": "numba_warmup", "warmup_s": numba_warmup_s})

    # rust needs no warmup by construction (compiled ahead of time); record
    # a trivial "first call" timing anyway so the claim is measured, not
    # just asserted.
    t0 = time.perf_counter()
    rti.noop()
    rust_first_call_s = time.perf_counter() - t0
    print(f"rust first-ever call (noop): {rust_first_call_s*1e6:.2f} us")
    append_result({"event": "rust_first_call", "first_call_s": rust_first_call_s})

    shapes = [
        ts.build_credit_tree(target_leaves=20, seed=7),
        ts.build_full_binary(5, seed=1),
        ts.build_full_binary(7, seed=1),
        ts.build_full_binary(9, seed=1),
        ts.build_one_sided_chain(100, seed=3),
    ]

    results = [process_shape(s) for s in shapes]
    regex_result = regex_experiment()

    print("\n\n=== SUMMARY (ns/row, min of runs) ===")
    header = f"{'shape':<20}{'leaves':>8}{'numba min':>12}{'rust min':>12}{'rust/PyTree min':>16}"
    print(header)
    for r in results:
        print(
            f"{r['shape']:<20}{r['leaf_count']:>8}"
            f"{min(r['numba_ns_per_row_runs']):>12.1f}"
            f"{min(r['rust_ns_per_row_runs']):>12.1f}"
            f"{min(r['rust_ptree_ns_per_row_runs']):>16.1f}"
        )

    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
