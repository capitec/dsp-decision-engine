"""Pure-numba fallback vs compiled (nanoarrow shim), same data, same tree,
same process, as shipped in the prototype wheel: identical answers first,
then cost per call at batch sizes 1, 100, 1k, 1M.

Run in a venv where the d2shim wheel is installed:
    .tmp/venv314/bin/python bench_fallback.py
Timings on a shared 28-core box carry a 5-20% band; median and min of
blocks are both printed.
"""
from __future__ import annotations

import gc
import statistics
import sys
import time

import numpy as np
import polars as pl

from d2shim import _arrowc as arrowc
from d2shim import pure
from d2shim.kernel_compiled import run_tree_compiled
from d2shim.pure import CONTAINS, EXACT, LEAF, PREFIX, STR, build_tree

WORDS = ["a much longer merchant descriptor for dog food", "CAPITEC PAY *DOG GROOMING JHB", "hotdog",
         "cat", "", "UBER TRIP HELP.UBER.COM", "dog", "WOOLWORTHS ONLINE 0861 sandton", "PnP dogstuff"]


def make_series(n, seed):
    rng = np.random.default_rng(seed)
    vals = [None if rng.random() < 0.05 else WORDS[rng.integers(len(WORDS))] for _ in range(n)]
    return pl.Series("s", vals, dtype=pl.String)


# a 5-node tree: two STR nodes and a CMP, so the walk is not a single node
TREE = build_tree([
    (STR, 0, CONTAINS, 0, 1, 2, 0),      # 0: contains "dog"?
    (STR, 0, PREFIX, 1, 3, 4, 0),        # 1:   prefix "PnP"?
    (STR, 0, EXACT, 2, 4, 5, 0),         # 2:   exact "cat"?
    (LEAF, 0, 0, 0, 0, 0, 10),           # 3
    (LEAF, 0, 0, 0, 0, 0, 20),           # 4
    (LEAF, 0, 0, 0, 0, 0, 30),           # 5
])
PATTERNS = ["dog", "PnP", "cat"]


def run_pure(s, feats):
    v = arrowc.export(s)
    try:
        return pure.run_tree(feats, [v], TREE, PATTERNS)
    finally:
        v.release()


def run_compiled(s, feats):
    return run_tree_compiled(feats, [s], TREE, PATTERNS, checked=True)


def run_compiled_unchecked(s, feats):
    return run_tree_compiled(feats, [s], TREE, PATTERNS, checked=False)


def timeit(fn, series, feats, k, blocks):
    samples = []
    j = 0
    for _ in range(blocks):
        gc.disable()
        t0 = time.perf_counter_ns()
        for _ in range(k):
            fn(series[j % len(series)], feats); j += 1
        t1 = time.perf_counter_ns()
        gc.enable()
        samples.append((t1 - t0) / k / 1000.0)
    return statistics.median(samples), min(samples)


def main():
    # ---- identical answers, 1M random rows incl. multi-chunk and sliced -----
    big = make_series(1_000_000, 7)
    two = pl.concat([make_series(400_000, 8), make_series(600_000, 9)])
    sl = big.slice(12_345, 500_000)
    for label, s in (("1M single-chunk", big), ("1M two-chunk", two), ("500k sliced", sl)):
        feats = np.empty((len(s), 0))
        a = run_pure(s, feats); b = run_compiled(s, feats); c = run_compiled_unchecked(s, feats)
        same = np.array_equal(a, b) and np.array_equal(a, c)
        print(f"answers identical [{label}]: {same}  (leaf histogram {np.bincount(a).tolist()})")
        assert same
    # warm the JIT (disk cache) before timing
    for n in (1, 100, 1000):
        s = make_series(n, 0); feats = np.empty((n, 0))
        run_pure(s, feats); run_compiled(s, feats); run_compiled_unchecked(s, feats)

    print(f"\n{'rows':>9} | {'pure (median/min)':>24} | {'compiled checked':>24} | {'compiled unchecked':>24} | pure/compiled")
    for n, k, blocks, n_series in ((1, 2000, 21, 1000), (100, 500, 15, 200), (1000, 100, 11, 50), (1_000_000, 1, 5, 2)):
        series = [make_series(n, seed) for seed in range(n_series)]
        feats = np.empty((n, 0))
        p = timeit(run_pure, series, feats, k, blocks)
        c = timeit(run_compiled, series, feats, k, blocks)
        u = timeit(run_compiled_unchecked, series, feats, k, blocks)
        unit = "us" if n < 1_000_000 else "ms"
        f = 1.0 if n < 1_000_000 else 1e-3
        print(f"{n:>9} | {p[0]*f:>10.1f} / {p[1]*f:<9.1f}{unit} | {c[0]*f:>10.1f} / {c[1]*f:<9.1f}{unit} | "
              f"{u[0]*f:>10.1f} / {u[1]*f:<9.1f}{unit} | {p[0]/c[0]:.2f}x")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
