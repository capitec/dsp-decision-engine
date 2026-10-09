# When to fuse, release the GIL, use fastmath, fall back or cache?

Date: 2026-09-23 · Script: spikes, in decider-experimentation/ `fusion-body-cost`, `n4-tail-concurrency-swap`, `numeric-divergence`, `objmode-fallback-blast-radius`, `numba_cache_survival`, `kl-contradiction-resolved`, `subprocess-handover`, `prange-crossover`

## Purpose
The measurements behind six kernel rules: fixed fusion, `nogil=True`, no `fastmath`, whole-step Python fallback, no generated source, and no `prange`. Read it before relaxing any of them.

## Method
Synthetic pipelines of 3 to 40 modules at 1M rows (fusion), 1 to 16 threads against a 20 ms budget (`nogil`), a 16-term log/exp/sqrt/div chain on 20,000 rows (`fastmath`), 200k rows x 4 steps (fallback), and fresh-process compiles at 200 to 800 arguments (cache), and `prange` against a serial loop on four bodies (a trivial add to a 256-iteration inner loop) over a row grid, three process runs. Shared dev box, so ratios matter more than absolutes.

## Result
Fusion, split/fused time (above 1 means fusion wins):

| modules | cheap straight-line body | heavy body |
|---|---|---|
| 3 | 3.26 | 0.76 |
| 20 | 13.00 | 0.22 |
| 40 | 11.20 | 0.18 |

- Branchy steps went 1.33 to 1.61x for one group, then 0.17x at 16 modules. Cause: at 20 modules the fused kernel pins all 16 ymm registers and spills 47 times per iteration. Skipping a heavy arm made fusion win 1071x. At N=1 fusion saves 0.44 us per kernel (13 us for 30 kernels, 0.07% of 20 ms).
- Why spills decide it: at 20 heavy modules the fused loop pins all 16 ymm registers and unrolls about 2x where the split kernel uses 7 and unrolls about 5x; the vectorised FP op count never reached 0 (652 packed ops, 0 scalar in the worst case), so it cannot be the signal. Splitting pays a floor of 0.44 µs dispatch plus 0.32-0.76 ns per row per kernel, so fusion wins when the body costs less than that and below 1,000 rows (up to 48x at one row and 40 modules). Break-even is not a row count: cheap straight-line bodies win at every size from 1,000 rows.
- `nogil`, p99: 16 threads 883 us with, 254.1 ms without (max 378.7 ms); 2 threads 904 us against 6.2 ms. Throughput is GIL-bound at about 1,100 calls/s either way; the loss is a convoy.
- `fastmath` off: Python and njit agreed on 100.00% of rows bit-exactly. On: 46 to 73% differed, up to 17 ULP, for 1.09x. It does give 2 to 2.5x on arithmetic-heavy fused drivers.
- Fallback: every one of six compile failures was a `TypingError` (a `NumbaError`) that took the whole driver down, never one node. All-njit 7.13 ms; whole driver in Python 167.3 ms (23.5x); one node via per-row `objmode` 548.4 ms (77x); `objmode` inside `prange` 4380.6 ms (615x, 8x slower than serial; the transition is about 2.7 µs a row). Hoisting to one `objmode` per batch gives 29.7x, no better than computing the column in Python first (31.3x).
- Cache, cold against warm: 4.93 s against 0.177 s (200 args), 14.75 s against 0.179 s (400), 46.99 s against 0.197 s (800). A cache entry needs the same path, basename, line, mtime and size, signature, CPU and module name; `exec`'d code is uncacheable; a constant edited with a restored mtime was served stale (49.39775 for 49.29765).
- Stale code has two independent sources. Clearing only CPython's `__pycache__/*.pyc` still served the old answer when the edited constant occurred once in the function (`co_code` unchanged, `co_consts` not in numba's key), and clearing only numba's cache did too; both cleared gave the new one. An edit to one of 16 shared occurrences changed `co_code` and missed correctly. Content-addressed file and module names (`drv_<sha256[:16]>`) were right in all three cases with neither cache cleared. A module not in `sys.modules` under its pickled name fails with a `ModuleNotFoundError` from inside numba.
- Numerics: with `fastmath` on, instalments still agreed to the cent over 20,000 rows (R0.00 difference), so the drift is only a bit-equality problem. A `ZeroDivisionError` in a serial kernel loop became `inf` under `prange` with the same source, so the loop choice silently changed error semantics.
- `prange`: the crossover is serial wall-clock, not row count, about 90 µs against a 65-70 µs fork/join floor; a rule on predicted serial time scored 97.9% out of sample, a row threshold 66.7%, a warmup probe at 10k rows predicting 100k+ 52.8% (0% for medium bodies). A trivial add reaches 2.8-3.1x at 1M rows; a 64-iteration body 11.4-12.0x at 5M. `parallel=True` compiles 1.92-7.92x slower (median 3.15x).

## Decision
Fixed grouping, `nogil` unconditionally, `fastmath` off, fallback per whole step on compile errors only, content-keyed kernels with no generated source, and no `prange` (nothing in `decider/` uses it). See [engine](../decisions/engine.md).
