# Do the shared tree walker and table matcher match decider2's generated kernels?

Date: 2026-09-23 (rerun 2026-10-08) · Script: `benchmarks/tree_walk.py`, `benchmarks/table_match.py` (the decider2 side was a script that is not kept)

## Purpose
Check that walking packed arrays with one shared function costs no more than decider2's per-tree generated code, for trees and for tables with rows inline or from params.

## Method
Tree: 127 nodes over 18 features with String, Float64 and Int64 outputs, alone and behind a `starts_with` gate. Table: 12 score bands writing String, Int64 and Float64, alone and ANDed with a string `in`. 1M rows through `run(df)`, 20k `score(dict)` calls, fused mode. The 2026-09-23 runs were on a loaded dev box (load 4-15).

## Result
| Case (2026-09-23) | decider2 ns/row, `score` p50 | decikit ns/row, `score` p50 |
|---|---|---|
| numeric tree | 748, 209 us | 592, 58 us |
| string-gated tree | 614, 257 us | 689, 101 us |
| bands | 304, 42.9 us | 164, 27.4 us |
| bands, rows param | | 178, 95.9 us (35.8 after the params-key fix) |
| bands AND string `in` | 398, 75.4 us | 413, 69.6 us |

- Rows-param `score()` paid for `document_key` JSON-encoding the whole params document every call (about 40 us for 12 rows); remembering the last document object took p50/p99 from 88.6/191.9 to 35.8/42.9 us.
- Rerun today, decikit only: numeric tree 147 ns/row, `score` 70.3 us; gated tree 196 ns/row, 85.0 us; bands 92 ns/row, 29.7 us (rows param 41.5 us); bands AND string list 206 ns/row, 47.4 us (rows param 57.8 us). Interpreted mode is 100-300x slower per row.

## Decision
Keep one shared walker and matcher: batch is on par or better, single-record is 2-4x faster than decider2's. See [configurable steps](../decisions/configurable-steps.md).
