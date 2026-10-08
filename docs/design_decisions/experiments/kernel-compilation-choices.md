# When to fuse, release the GIL, use fastmath, fall back or cache?

Date: 2026-09-23 · Script: spikes on the previous engine (decider2), not kept

## Purpose
The measurements behind five kernel rules: fixed fusion, `nogil=True`, no `fastmath`, whole-step Python fallback, and no generated source. Read it before relaxing any of them.

## Method
Synthetic pipelines of 3 to 40 modules at 1M rows (fusion), 1 to 16 threads against a 20 ms budget (`nogil`), a 16-term log/exp/sqrt/div chain on 20,000 rows (`fastmath`), 200k rows x 4 steps (fallback), and fresh-process compiles at 200 to 800 arguments (cache). Shared dev box, so ratios matter more than absolutes.

## Result
Fusion, split/fused time (above 1 means fusion wins):

| modules | cheap straight-line body | heavy body |
|---|---|---|
| 3 | 3.26 | 0.76 |
| 20 | 13.00 | 0.22 |
| 40 | 11.20 | 0.18 |

- Branchy steps went 1.33 to 1.61x for one group, then 0.17x at 16 modules. Cause: at 20 modules the fused kernel pins all 16 ymm registers and spills 47 times per iteration. Skipping a heavy arm made fusion win 1071x. At N=1 fusion saves 0.44 us per kernel (13 us for 30 kernels, 0.07% of 20 ms).
- `nogil`, p99: 16 threads 883 us with, 254.1 ms without (max 378.7 ms); 2 threads 904 us against 6.2 ms. Throughput is GIL-bound at about 1,100 calls/s either way; the loss is a convoy.
- `fastmath` off: Python and njit agreed on 100.00% of rows bit-exactly. On: 46 to 73% differed, up to 17 ULP, for 1.09x. It does give 2 to 2.5x on arithmetic-heavy fused drivers.
- Fallback: all-njit 7.13 ms; whole driver in Python 167.3 ms (23.5x); one node via per-row `objmode` 548.4 ms (77x); `objmode` inside `prange` 4380.6 ms (615x). Hoisting to one `objmode` per batch gives 29.7x, no better than computing the column in Python first (31.3x).
- Cache, cold against warm: 4.93 s against 0.177 s (200 args), 14.75 s against 0.179 s (400), 46.99 s against 0.197 s (800). A cache entry needs the same path, basename, line, mtime and size, signature, CPU and module name; `exec`'d code is uncacheable; a constant edited with a restored mtime was served stale (49.39775 for 49.29765).

## Decision
Fixed grouping, `nogil` unconditionally, `fastmath` off, fallback per whole step on compile errors only, and content-keyed kernels with no generated source. See [engine](../decisions/engine.md).
