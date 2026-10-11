# Can one scatter/gather primitive replace `each`'s per-row and batch modes?

Date: 2026-09-30 · Script: `benchmarks/each_unified.py` (the earlier per-row and batch baseline script was not kept)

## Purpose
`each` had a per-row mode (good for `score`) and a batch mode (good for big frames), and the crossover was a user decision. Can one primitive win at both ends, and does `prange` or threading help?

## Method
Fused engine; a light child (one scalar step) and a complex child (five chained steps and a 4-iteration loop); 1 to 100 items per record, 1 to 10,000 records. Numbers are us per record, items = 5, light child. Dev box under load; absolute numbers move 20 to 40% between runs, ratios and the crossover did not.

## Result
| records | per-row | batch | scatter/gather, Python driver | scatter/gather, polars driver |
|---|---|---|---|---|
| 1 | 382 | 4,199 | 219 | 215 |
| 100 | 181 | 47 | 25 | 27 |
| 1,000 | 175 | 12.1 | 31 | 12.5 |
| 10,000 | 173 | 7.5 | 25 | 8.4 |

- Per-row was flat (170 to 270 us light, 266 us complex at 5 items) and scaled with items; batch paid about 4 ms fixed (nested `bind` plus explode, group_by, join) for 7.5 to 8.8 us per record at 10k. Crossover 10 to 50 records.
- The primitive drops the fixed cost (219 to 473 us for one record). With the Python driver it was still 3 to 10x slower than batch on wide frames; the polars driver closes that: complex, 20 items, 10k records, 55 us against 21.7 for batch and 957 for per-row.
- Below about 1,000 exploded elements the polars route (about ten lazy collects, each a Python to Rust round trip) loses to the Python loop, so it only runs on a whole-frame `List(Struct)` column of at least 1024 elements.
- A separate vectorised shape was 4 to 6x slower than the old `frame_step` for one record (6.4 ms), crossing over at about 200 rows: two shapes chosen at bind time is the alternative this replaced.
- `prange` cannot help: the child kernel is 0.008 us per record (under 0.1% of the total); the cost is Python driver work and polars conversion. Threading over parent rows would race on the shared child `Executable` and holds the GIL for most of the row.

## Decision
One `ScatterGatherNode`, with the polars driver above 1024 elements; no `prange`, no threading. See [engine](../decisions/engine.md#each-is-one-scatter-gather-primitive-polars-vectorises-it-above-1024-elements).
