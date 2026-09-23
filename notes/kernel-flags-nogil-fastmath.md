# Kernel flags: `nogil=True` for serving, `fastmath` off

**Decision:** Serving kernels compile with `nogil=True`, unconditionally.
`fastmath` is off by default. A kernel that turns it on is excluded from the
exact cross-mode agreement test and must declare a tolerance.

**Why (`nogil`):** In N4, 1–16 threads called the same compiled kernel
concurrently, with a 20 ms budget:

| threads | `nogil=True` p99 | `nogil=False` p99 | `nogil=False` max |
|---|---|---|---|
| 1 | 879 µs | 890 µs | 2.4 ms |
| 2 | 904 µs | 6.2 ms | 7.6 ms |
| 8 | 893 µs | 115.8 ms | 342.5 ms |
| 16 | 883 µs | 254.1 ms (12.7× the budget) | 378.7 ms |

- Throughput is GIL-bound at about 1,020–1,190 calls/s either way.
- `nogil=False` causes a convoy: each call holds the GIL for its whole run.
- `nogil=True` is safe on any nopython kernel, because compiled code touches no
  Python objects.

**Why (`fastmath`):**
- With `fastmath` off, pure Python and njit agreed 100.00% bit-exact over a
  16-term log/exp/sqrt/div chain (20,000 rows). Exact equality across modes is
  achievable.
- With `fastmath` on, 46–73% of rows differed, by up to 17 ULP, for a 1.09×
  speedup on that chain.

**What we tried:**
- decider2 made `nogil` opt-in per step (off by default), and a fused group got
  it only when every step in it had it.
- An earlier finding went the other way: `nogil=True` lost to `nogil=False` (26%
  vs 55% throughput retained) when competing with a background compile thread.
  That finding only applies to compiling during serving. Compilation moved to a
  subprocess (97.9% throughput retained), so it no longer argues against `nogil`.
- GC tuning was ruled out as a tail fix: `gc.disable()`/`gc.freeze()` left
  p50/p95/p99 identical. The single-thread tail (p99.9 at 8.6% of budget) is OS
  scheduler preemption.
- `fastmath` "is noise (±5%)" was true only for branch-dominated logic. It gives
  2–2.5× on an arithmetic-heavy fused driver, so allow it per kernel with a
  declared tolerance.
- The docs' "1-ULP drift from `log`" didn't reproduce: 0 of 2,000,000 values
  differed, even with `fastmath`.

**Source:** `decider2/docs/EXPERIMENTS.md` (§N4, §N5, §H, §I);
`experimentation/n4-tail-concurrency-swap/`; `experimentation/n4-tail-cause/`;
`experimentation/numeric-divergence/`; `decider2/docs/01-motivation-and-evidence.md` (§4c);
`decider2/src/decider2/graph/step.py`
