# Benchmarks against decider2 (final record)

decider2 was deleted in T7.1, so the comparison scripts lost their decider2
half and were renamed; they now time `decider` alone:

| then | now |
|---|---|
| `benchmarks/modes_vs_decider2.py` | `benchmarks/modes.py` |
| `benchmarks/control_flow_vs_decider2.py` | `benchmarks/control_flow.py` |
| `benchmarks/trees_vs_decider2.py` | `benchmarks/tree_walk.py` |
| `benchmarks/tables_vs_decider2.py` | `benchmarks/table_match.py` |

To rerun a comparison, check out a commit before T7.1 and follow the old
script's docstring (decider2 needs its `_nashim` built for the interpreter).
All numbers below are fused mode unless named, CPython 3.14, the shared dev
box on 2026-09-23 (often under load, so compare rows within a table, not
across tables). Outputs matched decider2's in every run.

## Summary

| workload | decider2 fused | decider fused | where measured |
|---|---|---|---|
| 20 steps, 1M rows, batch | 1.00 | ~1.00 (ratio) | T3.2 |
| 20 steps, one call | 1.00 | 0.73× the time | T3.2 |
| flagship batch | 8.2M rows/s | 12.5M rows/s | T3.3 part A |
| flagship `score()` p50 / p99 | 26.5 / 40.6 µs | 27.7 / 34.7 µs | T3.3 part A |
| flagship `run` 1 row / 100k / 1M | 851 µs / 12.1 ms / 231 ms | 122 µs / 0.73 ms / 5.7 ms | C1 |
| branch, 1M rows | 3.3-3.5M rows/s | 41.6-44.3M rows/s | T5.1 |
| loop (<= 360 iterations), 200k rows | 0.64M rows/s | 6.33M rows/s | T5.1 |
| 127-node tree, ns/row | 748 | 592 (227 ms / 1M after C1) | T4.2, C1 |
| 127-node tree `score()` p50 | 209 µs | 58 µs | T4.2 |
| string-gated tree, ns/row | 614 | 689 (360 ms / 1M after C1) | T4.2, C1 |
| string-gated tree `score()` p50 | 257 µs | 101 µs | T4.2 |
| 12-band table, ns/row | 304 | 164 | T4.3 |
| 12-band table `score()` p50 | 42.9 µs | 27.4 µs | T4.3 |
| table with rows from params, `score()` p50 | (no equivalent) | 35.8 µs (was 88.6) | G1 |

## Flagship (`modes.py`), T3.3 part A

1M rows through `run(df)`, 20k `score(dict)` calls.

| engine | batch rows/s | score p50 | score p99 |
|---|---|---|---|
| decider2 fused | 8.2M | 26.5 µs | 40.6 µs |
| decider fused | 12.5M | 27.7 µs | 34.7 µs |
| decider stepped | 12.9M | 37.1 µs | 47.6 µs |
| decider interpreted | 0.25M | 59.1 µs | 84.5 µs |

Repeated latency-only runs put fused p50 within 1 µs (2-5%) of decider2 and
p99 level with it.

## Frame boundary (`modes.py`, `tree_walk.py`), C1

Before and after the column-at-a-time Arrow boundary. 20k `score` calls, 5k
one-row runs (median), 100k and 1M rows (median of 15 and 7). Load average
~10 with swap full, so p99 and 1M times move ±50% between runs.

| pipeline | engine | score p50 | score p99 | run 1 row | run 100k | run 1M |
|---|---|---|---|---|---|---|
| flagship | decider2 | 26.6 µs | 42.3 µs | 851 µs | 12.1 ms | 231 ms |
| flagship | decider before C1 | 32.2 µs | 94.6 µs | 225 µs | 5.8 ms | 231 ms |
| flagship | decider after C1 | 29.0 µs | 36.1 µs | 122 µs | 0.73 ms | 5.7 ms |
| tree | decider2 | 215 µs | 391 µs | 3388 µs | 45.3 ms | 1216 ms |
| tree | decider before C1 | 60 µs | 148 µs | 480 µs | 38.5 ms | 1140 ms |
| tree | decider after C1 | 66 µs | 257 µs | 239 µs | 22.6 ms | 227 ms |
| string-gated tree | decider2 | 259 µs | 556 µs | 3493 µs | 45.3 ms | 1018 ms |
| string-gated tree | decider before C1 | 106 µs | 217 µs | 531 µs | 51.6 ms | 944 ms |
| string-gated tree | decider after C1 | 104 µs | 271 µs | 292 µs | 32.4 ms | 360 ms |

## Branch and loop (`control_flow.py`), T5.1

| engine | branch, 1M rows | loop (<= 360 iterations), 200k rows |
|---|---|---|
| decider2 fused | 3.3-3.5M rows/s | 0.64M rows/s |
| decider fused (packed) | 41.6-44.3M rows/s | 6.33M rows/s |
| decider stepped | 14.8M rows/s | 0.81M rows/s |

## Trees (`tree_walk.py`), T4.2

A 127-node tree over 18 features with a String, a Float64 and an Int64
output; 1M rows, 20k `score` calls; load average ~15. decider2's batch time
includes its `decode()` of the String column.

| tree | engine | ns/row | score p50 | score p99 |
|---|---|---|---|---|
| numeric | decider2 fused | 748 | 209 µs | 375 µs |
| numeric | decider fused | 592 | 58 µs | 78 µs |
| string-gated | decider2 fused | 614 | 257 µs | 326 µs |
| string-gated | decider fused | 689 | 101 µs | 117 µs |

After T4.3's tree null handling (null numbers as fills, load ~3): decider 490
ns/row, p50 56 µs; decider2 539-611 ns/row, 200-218 µs.

## Decision tables (`table_match.py`), T4.3 and G1

12 bands writing String, Int64 and Float64 outputs; 1M rows, 20k `score`
calls; load average ~4.

| table | engine | ns/row | score p50 | score p99 |
|---|---|---|---|---|
| bands | decider2 fused | 304 | 42.9 µs | 53.1 µs |
| bands | decider fused | 164 | 27.4 µs | 36.1 µs |
| bands | decider fused, rows param | 178 | 95.9 µs | 109.2 µs |
| bands AND string `in` | decider2 fused | 398 | 75.4 µs | 86.8 µs |
| bands AND string `in` | decider fused | 413 | 69.6 µs | 111.6 µs |
| bands AND string `in` | decider fused, rows param | 423 | 137.1 µs | 157.4 µs |

G1 made `ParamsCache.key` hash a reused params document once: rows-param
`score()` p50/p99 went from 88.6/191.9 µs to 35.8/42.9 µs, and the flagship
with a params document from 42 to 30 µs p50.
