# E-F — the `prange` crossover (doc 06 O6)

Measures when `prange` beats a serial `range` row loop, whether that choice can be
made by a warmup measurement, and whether the choice is reproducible enough to put
in an audit record (doc 08 §8).

## The doc claims under test

Doc 01 §4 / doc 06 O6:

> `prange` **has no fixed row threshold.** An earlier ~50k figure was measured on
> one synthetic kernel; E7 shows the crossover is set by **per-row work**, not row
> count: a light body never wins (0.99× even at 1 M rows), while a body containing
> a 64-iteration inner loop crosses at ~5k rows and reaches 7.8× at 5 M.

Doc 01 §4b:

> `parallel=True` compile cost is **1.2–2.6×, shape-dependent**

Doc 05 §5.1 proposes measuring both variants per kernel at warmup. Doc 08 §8 wants
the chosen variant recorded in the audit record.

## Files

| file | what it is |
| --- | --- |
| `prange_crossover.py` | the harness — compiles serial and `prange` variants of 4 bodies, sweeps row counts, and runs 5 phases (see below) |
| `analyse.py` | per-run tables: sweep, `prange` fixed-cost floor, in-sample static threshold, within-process flip rate, compile cost |
| `analyse_transfer.py` | the cross-run analysis: probe transfer, cross-**process** pick stability, out-of-sample static threshold, warmup cost |
| `results.json`, `results_run2.json`, `results_run3.json` | three independent process runs |
| `sweep*.log`, `results_tables.txt`, `analysis_transfer.txt` | captured output |

## Bodies

The body is the axis the doc says sets the crossover, so it is the explicit variable:

- `trivial` — `out[i] = x[i] + y[i]*1.5` (memory-bound)
- `medium10` — ~12 straight-line flops
- `inner64` / `inner256` — a serially-dependent K-iteration inner loop. The
  dependence chain means LLVM cannot vectorise or reassociate it away
  (`fastmath=False`), so K really is K units of per-row work.

## Phases

1. **sweep** — serial vs `prange` over a row grid, 5 independent trials × 15 reps,
   median of trial medians. Also asserts the two variants are bitwise identical.
2. **warmup probe** — exactly what a warmup measurer would do: k ∈ {1,3,5} reps of
   both variants on a probe batch, pick the faster. Records the cost.
3. **flip test** — 15 independent cheap picks per cell, within one process.
4. (folded into 1 — the 5 trials are the reproducibility sample)
5. **compile cost** — `parallel=True` vs serial compile time per shape, including
   10/25/50/100-flat-branch bodies.

## How to run

```bash
# one full run, ~172 s
.venv/bin/python prange_crossover.py --budget 500 --out results_run4.json

# smoke test, ~20 s
.venv/bin/python prange_crossover.py --quick --out /tmp/smoke.json

# per-run tables
.venv/bin/python analyse.py results.json results_run2.json results_run3.json

# cross-run analysis (needs >=2 runs; the LAST one is the held-out test set)
.venv/bin/python analyse_transfer.py results.json results_run2.json results_run3.json
```

`--budget` is a soft wall-clock cap in seconds; cells past it are recorded as
`skipped` rather than silently dropped.

## Machine these results came from

Intel i7-14700HX, 20 cores / 28 threads, `NUMBA_NUM_THREADS=28`, OpenMP threading
layer. Python 3.14.5, numba 0.67.0, llvmlite 0.49.0, numpy 2.4.6.

**The absolute numbers are machine-specific.** The `prange` fork/join floor
(~65–70 µs here) is what sets the crossover, and it will differ on another box —
which is the point of shipping the harness: re-run it rather than trusting the table.

## What the three runs found

- **A light body does win.** `trivial` reaches 2.83×/2.88×/3.11× at 1 M rows. The
  "0.99× even at 1 M" claim is refuted in all three runs.
- **`inner64` crosses at 500 rows, not ~5k**, and reaches 11.4–12.0× at 5 M, not
  7.8×. (7.8× is about what it hits at 100k.)
- **The crossover is set by total serial wall-clock, not per-row work and not row
  count.** Every body crosses when its serial time passes ~85–240 µs, i.e. when
  serial work exceeds the `prange` fork/join floor. A single static rule —
  *`prange` iff predicted serial wall-clock > 90 µs* — fitted on runs 1+2 scores
  **97.9% out of sample on run 3**. A best-fit row threshold scores 66.7%.
- **A cheap warmup probe does not transfer.** A probe at n ≤ 10k predicting the
  variant for n ≥ 100k is **52.8%** accurate overall and **0%** for `medium10` —
  because warmup does not know the production batch size, and batch size is the
  thing that decides.
- **Cross-process pick stability is 46/48 (95.8%)**; both disagreements are cells
  whose true speedup sits in 0.51×–1.16×, where the choice does not matter.
- **`parallel=True` compile cost is 1.92×–7.92× (median 3.15×)** over 24
  measurements — the documented 1.2–2.6× range is refuted; only 1 of 24 fell inside it.
