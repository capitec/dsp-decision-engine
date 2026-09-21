"""Codegen vs. interpreted tree-walk: the experiment.

Answers doc 08 §3.4's own test for decider2 TREES specifically: "can one
compiled loop evaluate every instance of this kind, with the instance
supplied as arrays?" decider2's decision-table engine already says yes for
tables. This measures whether the same is true for trees, against
decider2's real, unmodified `trees/codegen.py` path — not a mock of it.

Run:
    <repo>/.venv/bin/python run_experiment.py

Writes results.jsonl (flushed after every measurement) and prints a
summary table. Nothing here imports/modifies decider2/src; it only calls
public functions from the installed `decider2` package.
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import tree_shapes as ts  # noqa: E402
import interpreted_kernel as ik  # noqa: E402
import codegen_bench as cb  # noqa: E402

N_ROWS = 100_000
N_TIMING_RUNS = 7
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


def min_ns_per_row(fn, args, n_rows: int, *, runs: int = N_TIMING_RUNS) -> float:
    """The MINIMUM wall time across `runs` calls, in ns/row — measuring the
    kernel, not the scheduler (a mean would let one scheduling hiccup
    inflate every shape equally)."""
    best = float("inf")
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best / n_rows * 1e9


_COLD_START_KEYS = (
    "emit_source_s", "write_and_import_s", "matcher_compile_s",
    "path_compile_s", "output_compile_s", "wrapper_build_and_compile_s",
)


def cold_start_total(timings: dict) -> float:
    """Sum of the LEAF timing entries only. `timings['tree_compile_s']` is
    itself `matcher + path + output`, kept in the dict for the printed
    per-shape breakdown — summing every value in the dict would double
    count it."""
    return sum(timings[k] for k in _COLD_START_KEYS)


def append_result(record: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def build_codegen_matrices(shape: ts.TreeShape, emitted, numeric: dict, string_codes: dict):
    numeric_features_in_order = [f for f in emitted.features if f not in emitted.string_features]
    feat_matrix = np.column_stack([numeric[f] for f in numeric_features_in_order]) if numeric_features_in_order else np.zeros((N_ROWS, 0))
    if emitted.string_features:
        str_matrix = np.column_stack([string_codes[f] for f in emitted.string_features]).astype(np.int32)
    else:
        str_matrix = np.zeros((N_ROWS, 0), dtype=np.int32)
    return np.ascontiguousarray(feat_matrix, dtype=np.float64), np.ascontiguousarray(str_matrix, dtype=np.int32)


def build_interpreted_matrices(flat: ts.FlatTree, numeric: dict, string_codes: dict):
    if flat.numeric_features:
        numeric_matrix = np.column_stack([numeric[f] for f in flat.numeric_features])
    else:
        numeric_matrix = np.zeros((N_ROWS, 0))
    if flat.string_features:
        string_matrix = np.column_stack([string_codes[f] for f in flat.string_features]).astype(np.int32)
    else:
        string_matrix = np.zeros((N_ROWS, 0), dtype=np.int32)
    return np.ascontiguousarray(numeric_matrix, dtype=np.float64), np.ascontiguousarray(string_matrix, dtype=np.int32)


def warm_up_numba_process() -> float:
    """numba/LLVM pays a one-time, per-PROCESS startup cost the first time
    ANYTHING is njit-compiled (target-machine init, LLVM module setup) —
    measured here at ~0.2s, separate from compiling any particular
    function. decider2's real server pays this once at process start,
    regardless of trees vs. tables vs. anything else, so it is neither
    approach's cost and is paid here, once, before any timed measurement,
    so it doesn't inflate whichever engine happens to compile first."""
    from numba import njit

    t0 = time.perf_counter()

    @njit
    def _warmup(x):
        return x + 1.0

    _warmup(1.0)
    return time.perf_counter() - t0


def warm_up_interpreted_kernel_once() -> float:
    """The ONE compile the interpreted kernel ever pays in this process —
    triggered on a throwaway 1-node tree, timed and reported on its own so
    the per-shape numbers that follow are visibly ~0 marginal cost, not
    just asserted to be."""
    dummy = ts.TreeShape(
        name="__warmup__", root=ts.Leaf(0.0),
        numeric_ranges={"x": (0.0, 1.0)}, string_categories={},
        leaf_count=1, node_count=1,
    )
    flat = ts.to_flat_tree(dummy)
    numeric = np.zeros((4, 1))
    strings = np.zeros((4, 0), dtype=np.int32)
    out = np.zeros(4)
    t0 = time.perf_counter()
    ik.walk_batch(numeric, strings, flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
                  flat.pat_start, flat.pat_count, flat.patterns, flat.left, flat.right,
                  flat.leaf_value, out)
    dt = time.perf_counter() - t0
    assert len(ik.walk_batch.signatures) == 1, "unexpected: more than one signature after first call"
    return dt


def process_shape(shape: ts.TreeShape) -> dict:
    print(f"\n=== {shape.name} (leaves={shape.leaf_count}, nodes={shape.node_count}) ===")

    # NOTE: seeded via zlib.crc32, not the builtin hash() — str hashing is
    # randomised per process (PYTHONHASHSEED unset, PEP 456) unless pinned,
    # so hash(shape.name) silently drew a DIFFERENT row sample every run.
    # Caught by re-running this script 3x for reproducibility and seeing
    # one_sided_chain_100's ns/row swing 27.7 -> 59.5 (2.15x) between runs
    # of the SAME shape with the SAME seed expression: for a tree whose
    # per-row cost is data-dependent (average root-to-leaf walk length,
    # unlike full-binary's fixed depth), a different row sample is a
    # different *workload*, not scheduler noise. crc32 is stable across
    # processes, so re-running this script now draws the same rows every
    # time (verified below).
    numeric, string_codes = ts.make_rows(shape, N_ROWS, seed=zlib.crc32(shape.name.encode()) & 0xFFFF)

    # ---- codegen path (real decider2.trees.codegen + real numba compile) ----
    codegen = cb.compile_codegen_tree(shape)
    line_cap_ok = codegen.emitted_lines <= 500  # decider2.trees.codegen.LINE_CAP
    feat_matrix, str_matrix = build_codegen_matrices(shape, codegen.emitted, numeric, string_codes)
    codegen_out = np.zeros(N_ROWS)
    codegen.row_loop(feat_matrix, str_matrix, codegen_out)  # warm the dispatcher's cache path
    codegen_ns_per_row = min_ns_per_row(codegen.row_loop, (feat_matrix, str_matrix, codegen_out), N_ROWS)

    # ---- interpreted path (one generic kernel, already compiled once) ----
    t0 = time.perf_counter()
    flat = ts.to_flat_tree(shape)
    flatten_s = time.perf_counter() - t0
    num_matrix, str_matrix2 = build_interpreted_matrices(flat, numeric, string_codes)
    interp_out = np.zeros(N_ROWS)
    sig_count_before = len(ik.walk_batch.signatures)
    interp_args = (num_matrix, str_matrix2, flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
                    flat.pat_start, flat.pat_count, flat.patterns, flat.left, flat.right,
                    flat.leaf_value, interp_out)
    ik.walk_batch(*interp_args)
    sig_count_after = len(ik.walk_batch.signatures)
    interp_ns_per_row = min_ns_per_row(ik.walk_batch, interp_args, N_ROWS)

    # ---- fairness: identical answers, asserted, not assumed ----
    if not np.array_equal(codegen_out, interp_out):
        mismatches = np.sum(codegen_out != interp_out)
        raise AssertionError(
            f"{shape.name}: {mismatches}/{N_ROWS} rows disagree between codegen and "
            f"interpreted. First few: codegen={codegen_out[:5]} interp={interp_out[:5]}"
        )
    print(f"  answers identical across {N_ROWS} rows: OK")
    print(f"  emitted lines            : {codegen.emitted_lines} (LINE_CAP=500 ok: {line_cap_ok})")
    print(f"  codegen compile (tree)   : {codegen.timings['tree_compile_s']*1e3:.1f} ms")
    print(f"  codegen wrapper compile  : {codegen.timings['wrapper_build_and_compile_s']*1e3:.1f} ms")
    print(f"  codegen cold start total : {cold_start_total(codegen.timings)*1e3:.1f} ms")
    print(f"  interpreted flatten time : {flatten_s*1e6:.1f} us")
    print(f"  interpreted sig count    : {sig_count_before} -> {sig_count_after} (0 new = 0 compile)")
    print(f"  codegen ns/row (min of {N_TIMING_RUNS})    : {codegen_ns_per_row:.1f}")
    print(f"  interpreted ns/row (min of {N_TIMING_RUNS}): {interp_ns_per_row:.1f}")

    record = {
        "shape": shape.name,
        "leaf_count": shape.leaf_count,
        "node_count": shape.node_count,
        "n_rows": N_ROWS,
        "emitted_lines": codegen.emitted_lines,
        "within_production_line_cap_500": line_cap_ok,
        "codegen": {
            "timings_s": codegen.timings,
            "cold_start_total_s": cold_start_total(codegen.timings),
            "ns_per_row": codegen_ns_per_row,
        },
        "interpreted": {
            "flatten_s": flatten_s,
            "kernel_signature_count_before": sig_count_before,
            "kernel_signature_count_after": sig_count_after,
            "new_compile_triggered": sig_count_after > sig_count_before,
            "ns_per_row": interp_ns_per_row,
        },
        "answers_identical": True,
    }
    append_result(record)
    return record


def main() -> None:
    check_memory()
    RESULTS_PATH.write_text("")  # fresh file this run

    # decider2's content-addressed cache (decider2.compile.cache) and numba's
    # own on-disk cache=True cache both persist across processes by design
    # (that IS the production feature). For "a new tree config arrives" to
    # mean anything, this run's tree shapes must be genuinely unseen —
    # wiped here so every compile below is a true cold miss, not a replay
    # of a previous run of this same script.
    shutil.rmtree(cb.BUILD_DIR, ignore_errors=True)
    print(f"[cache] wiped {cb.BUILD_DIR} — every compile below is a true cold miss")

    print("\n--- one-time numba/LLVM process warm-up (neither approach's cost) ---")
    numba_warmup_s = warm_up_numba_process()
    print(f"numba process warm-up: {numba_warmup_s*1e3:.1f} ms")
    append_result({"event": "numba_process_warmup", "compile_s": numba_warmup_s})

    print("\n--- one-time interpreted-kernel compile (paid once, ever) ---")
    warmup_s = warm_up_interpreted_kernel_once()
    print(f"interpreted kernel first-ever compile: {warmup_s*1e3:.1f} ms "
          f"(walk_batch.signatures = {len(ik.walk_batch.signatures)})")
    append_result({"event": "interpreted_kernel_warmup", "compile_s": warmup_s})

    shapes = [
        ts.build_credit_tree(target_leaves=20, seed=7),
        ts.build_full_binary(5, seed=1),
        ts.build_full_binary(7, seed=1),
        ts.build_full_binary(9, seed=1),
        ts.build_one_sided_chain(100, seed=3),
    ]

    results = [process_shape(s) for s in shapes]

    print("\n--- warm recompile: same trees, on-disk cache now populated ---")
    print("(a real fresh OS process would still pay module-import cost this")
    print(" skips; this isolates numba's own compiled-artifact cache hit,")
    print(" decider2.compile.cache's own stated design target.)")
    for shape in shapes:
        t0 = time.perf_counter()
        warm = cb.compile_codegen_tree(shape)
        warm_s = time.perf_counter() - t0
        cold_s = next(r for r in results if r["shape"] == shape.name)["codegen"]["cold_start_total_s"]
        print(f"  {shape.name:<20} warm recompile total: {warm_s*1e3:.1f} ms "
              f"(cold was {cold_s*1e3:.1f} ms)")
        append_result({"event": "warm_recompile", "shape": shape.name, "warm_total_s": warm_s})

    print("\n\n=== SUMMARY ===")
    header = f"{'shape':<20}{'leaves':>8}{'nodes':>8}{'lines':>8}{'codegen ms':>12}{'interp us':>12}{'cg ns/row':>11}{'interp ns/row':>14}"
    print(header)
    for r in results:
        print(
            f"{r['shape']:<20}{r['leaf_count']:>8}{r['node_count']:>8}{r['emitted_lines']:>8}"
            f"{r['codegen']['cold_start_total_s']*1e3:>12.1f}"
            f"{r['interpreted']['flatten_s']*1e6:>12.1f}"
            f"{r['codegen']['ns_per_row']:>11.1f}"
            f"{r['interpreted']['ns_per_row']:>14.1f}"
        )

    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
