"""Per-CALL cost of each approach at batch size 1, 10, 100, 1000 and 1M,
broken into stages. Batch size 1 is the primary axis (90% of calls).

Every stage is timed in blocks of `k` calls; the block time / k is one
sample; we report the median and min over blocks in microseconds per call.
"""
from __future__ import annotations

import gc
import statistics
import sys
import time

import numpy as np
import polars as pl

import nashim  # noqa: F401  (puts the incumbent on sys.path)
import arrowc
from kernel import CONTAINS, LEAF, STR, build_tree, pattern_table, run_tree, string_tables, walk_chunk
from kernel_b import run_tree_b, walk_chunk_b
from kernel_c import run_tree_c
from nashim import DEFAULT, FULL, GET_STRING_ADDR, MINIMAL, NanoView
from results_io import record

TREE = build_tree([(STR, 0, CONTAINS, 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)])
PATTERNS = ["dog"]
PAT = pattern_table(PATTERNS)
WORDS = ["a much longer merchant descriptor for dog food", "CAPITEC PAY *DOG GROOMING JHB", "hotdog",
         "cat", "", "UBER TRIP HELP.UBER.COM", "dog", "WOOLWORTHS ONLINE 0861 sandton", "PnP dogstuff"]


def make_series(n, seed):
    rng = np.random.default_rng(seed)
    vals = [None if rng.random() < 0.05 else WORDS[rng.integers(len(WORDS))] for _ in range(n)]
    return pl.Series("s", vals, dtype=pl.String)


def timeit(fn, k, blocks, setup=None):
    """fn(j) is one call; k calls per block; returns (median_us, min_us, samples)."""
    samples = []
    j = 0
    for _ in range(blocks):
        if setup:
            setup()
        gc.disable()
        t0 = time.perf_counter_ns()
        for _ in range(k):
            fn(j); j += 1
        t1 = time.perf_counter_ns()
        gc.enable()
        samples.append((t1 - t0) / k / 1000.0)
    return statistics.median(samples), min(samples), len(samples)


def bench_n(n, k, blocks, n_series):
    series = [make_series(n, seed) for seed in range(n_series)]
    feats = np.empty((n, 0))
    out = np.empty(n, np.int64)
    S = lambda j: series[j % n_series]
    res = {}

    def rec(stage, fn, **extra):
        med, mn, cnt = timeit(fn, k, blocks)
        res[stage] = med
        record(probe="bench", n=n, stage=stage, median_us=med, min_us=mn, blocks=cnt, k=k, **extra)
        print(f"n={n:>8} {stage:<34} median {med:10.2f} us   min {mn:10.2f} us", flush=True)

    # --- shared stage: polars builds the Arrow C stream (the PyCapsule) ---
    rec("polars.__arrow_c_stream__", lambda j: S(j).__arrow_c_stream__())

    # --- A: incumbent ------------------------------------------------------
    rec("A.export (arrowc, ctypes loop)", lambda j: arrowc.export(S(j)).release())
    va = arrowc.export(S(0)); tabs = string_tables(va.chunks[:1]); ta = TREE
    walk_chunk(feats, *tabs, *PAT, ta["thresholds"], ta["kind"], ta["feat_idx"], ta["op"], ta["thr_slot"],
               ta["then_"], ta["else_"], ta["leaf_value"], out, 0, n)
    rec("A.kernel (walk_chunk, tables prebuilt)", lambda j: walk_chunk(
        feats, *tabs, *PAT, ta["thresholds"], ta["kind"], ta["feat_idx"], ta["op"], ta["thr_slot"],
        ta["then_"], ta["else_"], ta["leaf_value"], out, 0, n))
    rec("A.dispatch floor (walk_chunk n=0)", lambda j: walk_chunk(
        feats, *tabs, *PAT, ta["thresholds"], ta["kind"], ta["feat_idx"], ta["op"], ta["thr_slot"],
        ta["then_"], ta["else_"], ta["leaf_value"], out, 0, 0))
    rec("A.string_tables (python glue)", lambda j: string_tables(va.chunks[:1]))
    rec("A.pattern_table (python glue)", lambda j: pattern_table(PATTERNS))
    va.release()
    def a_e2e(j):
        v = arrowc.export(S(j)); run_tree(feats, [v], TREE, PATTERNS, out=out); v.release()
    rec("A.end_to_end", a_e2e)

    # --- B: nanoarrow per row -----------------------------------------------
    rec("B.import (NanoView: stream+init+SetArrayMinimal)", lambda j: NanoView(S(j)).release())
    nv = NanoView(S(0)); addrs = nv.view_addrs(); fn = np.uint64(GET_STRING_ADDR)
    walk_chunk_b(feats, fn, addrs, *PAT, ta["thresholds"], ta["kind"], ta["feat_idx"], ta["op"], ta["thr_slot"],
                 ta["then_"], ta["else_"], ta["leaf_value"], out, 0, n)
    rec("B.kernel (walk_chunk_b, prebuilt)", lambda j: walk_chunk_b(
        feats, fn, addrs, *PAT, ta["thresholds"], ta["kind"], ta["feat_idx"], ta["op"], ta["thr_slot"],
        ta["then_"], ta["else_"], ta["leaf_value"], out, 0, n))
    rec("B.dispatch floor (walk_chunk_b n=0)", lambda j: walk_chunk_b(
        feats, fn, addrs, *PAT, ta["thresholds"], ta["kind"], ta["feat_idx"], ta["op"], ta["thr_slot"],
        ta["then_"], ta["else_"], ta["leaf_value"], out, 0, 0))
    for lvl, name in ((MINIMAL, "minimal"), (DEFAULT, "default"), (FULL, "full")):
        rec(f"validate.{name}", lambda j, lvl=lvl: nv.validate(lvl), level=lvl)
    nv.release()
    def b_e2e(j, checked=False):
        v = NanoView(S(j)); run_tree_b(feats, [v], TREE, PATTERNS, out=out, checked=checked, pat=PAT); v.release()
    rec("B.end_to_end", b_e2e)
    rec("B-checked.end_to_end", lambda j: b_e2e(j, True))

    # --- C: nanoarrow validate once, numba per row ---------------------------
    def c_e2e(j, lvl):
        v = NanoView(S(j)); run_tree_c(feats, [v], TREE, PATTERNS, out=out, level=lvl, pat=PAT); v.release()
    rec("C-none.end_to_end (import only, no validate)", lambda j: c_e2e(j, None))
    rec("C-minimal.end_to_end", lambda j: c_e2e(j, MINIMAL))
    rec("C-default.end_to_end", lambda j: c_e2e(j, DEFAULT))
    rec("C-full.end_to_end", lambda j: c_e2e(j, FULL))
    return res


def main():
    plan = {1: (2000, 25, 1000), 10: (1000, 25, 1000), 100: (500, 21, 500), 1000: (100, 21, 100),
            1_000_000: (1, 5, 2)}
    sizes = [int(a) for a in sys.argv[1:]] or list(plan)
    # warm every kernel once (compile or cache load) before any timing
    s = make_series(3, 0); f = np.empty((3, 0))
    v = arrowc.export(s); run_tree(f, [v], TREE, PATTERNS); v.release()
    nv = NanoView(s); run_tree_b(f, [nv], TREE, PATTERNS); run_tree_b(f, [nv], TREE, PATTERNS, checked=True)
    run_tree_c(f, [nv], TREE, PATTERNS); nv.release()
    for n in sizes:
        k, blocks, n_series = plan[n]
        bench_n(n, k, blocks, n_series)


if __name__ == "__main__":
    main()
