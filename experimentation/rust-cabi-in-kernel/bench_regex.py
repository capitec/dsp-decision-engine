"""Regex via the C ABI (compile once, `(pointer, length)` per call) against
§T's numba->libc POSIX number (60.3 ns/call) and §U's Rust `regex` crate
via PyO3 (25.6 ns/call pure, 104.4 ns/call including `Vec<String>`
marshalling) — both cited, not re-derived.

Two shapes, matching §T/§U exactly:
  * per-category mask: n=12 short category strings (the shape a
    dictionary-encoded string column's distinct-value list has).
  * per-row: n=100,000 fixed-width 12-byte strings (cardinality ~= row
    count — free text, account numbers; the case where per-category
    degenerates, per §T's own conclusion).

Run:
    <repo>/.venv/bin/python experimentation/rust-cabi-in-kernel/bench_regex.py
"""
from __future__ import annotations

import gc
import json
import random
import string
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import regex_walk_cabi as rw  # noqa: E402

RESULTS_PATH = HERE / "results.jsonl"
N_ROWS = 100_000
N_TIMING_RUNS = 7


def append_result(record: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def timed_ns_per_call(fn, args, n_calls: int, *, runs: int = N_TIMING_RUNS) -> list[float]:
    out = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        out.append(dt / n_calls * 1e9)
    return out


def per_category_case() -> dict:
    print("\n=== regex: per-category mask (n=12) ===")
    categories = [f"category_{i:02d}" for i in range(12)]
    pattern = "^category_0"  # matches 0-9 -> 10/12, same selectivity as §U
    handle = rw.compile_pattern(pattern)
    data, offsets = rw.encode_variable_length(categories)
    out = np.zeros(len(categories), dtype=np.int32)

    rw.regex_match_variable(handle, data, offsets, out)  # warm dispatch
    expected = [1 if c.startswith("category_0") else 0 for c in categories]
    matched = out.tolist()
    ok = matched == expected
    print(f"  matches correct: {ok} ({sum(matched)}/{len(categories)} matched)")

    # single-call (one Python->njit dispatch, scanning all 12 categories
    # once) — dominated by dispatch overhead at this n, reported for
    # completeness against the repeated, dispatch-amortised number below.
    single_call_runs = timed_ns_per_call(rw.regex_match_variable, (handle, data, offsets, out), len(categories))
    print(f"  ns/call (single dispatch, n=12)      : min {min(single_call_runs):.1f}, "
          f"range {min(single_call_runs):.1f}-{max(single_call_runs):.1f}")

    # repeated (one dispatch, 5000 internal repeats of the 12-category
    # scan) — the steady-state per-call cost, comparable to §T's ~58
    # ns/category and §U's 25.6 ns/call "pure compute" citations.
    REPEATS = 5000
    rw.regex_match_variable_repeated(handle, data, offsets, out, 5)  # warm
    repeated_runs = timed_ns_per_call(
        rw.regex_match_variable_repeated, (handle, data, offsets, out, REPEATS),
        len(categories) * REPEATS,
    )
    print(f"  ns/call (amortised, {REPEATS}x{len(categories)} calls) : min {min(repeated_runs):.2f}, "
          f"range {min(repeated_runs):.2f}-{max(repeated_runs):.2f}")

    rw.free_pattern(handle)
    return {
        "n": len(categories),
        "single_dispatch_ns_per_call_runs": single_call_runs,
        "amortised_ns_per_call_runs": repeated_runs,
        "matches_correct": ok,
    }


def per_row_case() -> dict:
    print(f"\n=== regex: per-row (n={N_ROWS}) ===")
    rng = random.Random(42)
    alphabet = string.ascii_uppercase + string.digits
    width = 12
    rows = ["".join(rng.choices(alphabet, k=width)) for _ in range(N_ROWS)]
    pattern = "^A.*7"  # same representative pattern as rust-tree-interpreter's regex_experiment

    handle = rw.compile_pattern(pattern)
    data = rw.encode_fixed_width(rows, width)
    out = np.zeros(N_ROWS, dtype=np.int32)

    rw.regex_match_fixed(handle, data, width, out)  # warm dispatch
    runs = timed_ns_per_call(rw.regex_match_fixed, (handle, data, width, out), N_ROWS)

    import re
    py_re = re.compile(pattern)
    expected_matched = sum(1 for r in rows if py_re.match(r))
    matched = int(out.sum())
    ok = matched == expected_matched
    print(f"  matches correct: {ok} ({matched}/{N_ROWS} matched)")
    print(f"  ns/row: min {min(runs):.2f}, range {min(runs):.2f}-{max(runs):.2f} "
          f"({min(runs) * N_ROWS / 1e6:.2f} ms total, best run)")

    rw.free_pattern(handle)
    return {"n": N_ROWS, "ns_per_call_runs": runs, "matches_correct": ok, "n_matched": matched}


def main() -> None:
    cat = per_category_case()
    row = per_row_case()
    append_result({
        "event": "regex_cabi_experiment",
        "per_category": cat,
        "per_row": row,
    })
    print("\n\n=== SUMMARY: regex via C-ABI (compile once, raw pointer per call) ===")
    print(f"per-category (n=12), single dispatch : min {min(cat['single_dispatch_ns_per_call_runs']):.1f} ns/call")
    print(f"per-category (n=12), amortised        : min {min(cat['amortised_ns_per_call_runs']):.2f} ns/call")
    print(f"per-row (n={N_ROWS})                     : min {min(row['ns_per_call_runs']):.2f} ns/call")
    print("\ncf. §T numba->libc POSIX regex: 60.3 ns/call")
    print("cf. §U Rust regex crate, pure compute: 25.6 ns/call")
    print("cf. §U Rust regex crate via naive PyO3 binding (incl. Vec<String> marshalling): 104.4 ns/call")


if __name__ == "__main__":
    main()
