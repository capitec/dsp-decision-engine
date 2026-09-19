"""EXPERIMENT G / doc 06 O16 — how long does a rule set take to compile?

Measures, for realistically-shaped emitted numba rule kernels:
  1. compile time vs rule count
  2. emitted source lines vs compile time (tests doc 01 §4b's ~15 ms/line)
  3. first_match (early exit) vs all (evaluate everything)
  4. execution time at 100k rows
  5. the rule count where compile crosses ~10 s

Usage:
  .venv/bin/python run.py --counts 3,10,30 --modes first_match,all --repeats 3
  .venv/bin/python run.py --counts 100 --modes first_match --repeats 1 --no-exec
Results append to results.json in this directory.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time

import numpy as np
from numba import njit, types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emit import emit_ruleset, count_lines, N_FLOAT, N_CODE, N_BOOL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_TMP = tempfile.mkdtemp(prefix="ruleset_emit_")
_UID = [0]

_BASE_ARGS = (
    [types.float64[::1]] * N_FLOAT
    + [types.int64[::1]] * N_CODE
    + [types.boolean[::1]] * N_BOOL
)
SIG = {
    "first_match": types.void(*(_BASE_ARGS + [types.int64[::1]])),
    "all": types.void(*(_BASE_ARGS + [types.int64[:, ::1]])),
}


def load_source(src, fname="ruleset"):
    """Write emitted source to a real .py file and import it (as a staging
    worker would), so numba sees genuine source, not an exec'd code object."""
    _UID[0] += 1
    path = os.path.join(_TMP, f"gen_{_UID[0]}.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(f"gen_{_UID[0]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fname), path


def make_data(rows, n_rules, mode, seed=7):
    rng = np.random.default_rng(seed)
    floats = []
    for _ in range(N_FLOAT):
        a = rng.random(rows)
        a[rng.random(rows) < 0.02] = np.nan          # ~2% nulls
        floats.append(np.ascontiguousarray(a))
    codes = [np.ascontiguousarray(rng.integers(0, 10, rows).astype(np.int64))
             for _ in range(N_CODE)]
    bools = [np.ascontiguousarray(rng.random(rows) < 0.5) for _ in range(N_BOOL)]
    if mode == "first_match":
        out = np.zeros(rows, dtype=np.int64)
    else:
        out = np.zeros((rows, n_rules), dtype=np.int64)
    return floats + codes + bools + [out]


def measure(n_rules, mode, repeats, rows, seed, do_exec, exec_repeats, p_nomatch=0.0):
    compile_times = []
    line_counts = []
    fn = None
    for rep in range(repeats):
        # fresh source + fresh module every repeat: nothing can be reused
        src = emit_ruleset(n_rules, mode, seed=seed + 1000 * rep, p_nomatch=p_nomatch)
        line_counts.append(count_lines(src))
        py, _ = load_source(src)
        t0 = time.perf_counter()
        jitted = njit(SIG[mode], cache=False, nogil=True)(py)
        compile_times.append(time.perf_counter() - t0)
        if rep == 0:
            fn = jitted
    lines = statistics.median(line_counts)
    rec = {
        "n_rules": n_rules,
        "mode": mode,
        "p_nomatch": p_nomatch,
        "emitted_lines": lines,
        "emitted_lines_all": line_counts,
        "compile_repeats": repeats,
        "compile_s": compile_times,
        "compile_median_s": statistics.median(compile_times),
        "ms_per_line": 1000.0 * statistics.median(compile_times) / lines,
        "rows": rows if do_exec else None,
    }
    if do_exec:
        args = make_data(rows, n_rules, mode)
        fn(*args)                                   # warm up (already compiled)
        ts = []
        for _ in range(exec_repeats):
            t0 = time.perf_counter()
            fn(*args)
            ts.append(time.perf_counter() - t0)
        rec["exec_median_ms"] = 1000.0 * statistics.median(ts)
        rec["exec_all_ms"] = [round(1000 * t, 4) for t in ts]
        rec["match_rate"] = float(
            (args[-1] != -1).mean() if mode == "first_match"
            else (args[-1] != -1).mean()
        )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="3,10,30")
    ap.add_argument("--modes", default="first_match,all")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--exec-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--p-nomatch", type=float, default=0.0,
                    help="probability a leaf emits the no-match sentinel (-1)")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--dump-source", default=None)
    a = ap.parse_args()

    if a.dump_source:
        n, m = a.dump_source.split(":")
        print(emit_ruleset(int(n), m, seed=a.seed, p_nomatch=a.p_nomatch))
        return

    # LLVM / numba warm-up so the first real measurement is not the outlier
    t0 = time.perf_counter()
    njit(types.float64(types.float64), cache=False)(lambda x: x * 2.0)
    warm = time.perf_counter() - t0
    print(f"# numba warm-up compile: {warm*1000:.0f} ms", flush=True)

    rows_out = []
    for mode in a.modes.split(","):
        for n in [int(x) for x in a.counts.split(",")]:
            t0 = time.perf_counter()
            rec = measure(n, mode, a.repeats, a.rows, a.seed,
                          not a.no_exec, a.exec_repeats, a.p_nomatch)
            rec["wall_s"] = time.perf_counter() - t0
            rows_out.append(rec)
            print(
                f"{mode:12s} rules={n:4d} lines={rec['emitted_lines']:6.0f} "
                f"compile={rec['compile_median_s']:8.3f}s "
                f"({rec['ms_per_line']:6.2f} ms/line) "
                f"exec={rec.get('exec_median_ms', float('nan')):8.3f}ms "
                f"[wall {rec['wall_s']:.1f}s]",
                flush=True,
            )

    old = []
    if os.path.exists(a.out):
        with open(a.out) as fh:
            old = json.load(fh)
    with open(a.out, "w") as fh:
        json.dump(old + rows_out, fh, indent=1)
    print(f"# appended {len(rows_out)} records to {a.out}")


if __name__ == "__main__":
    main()
