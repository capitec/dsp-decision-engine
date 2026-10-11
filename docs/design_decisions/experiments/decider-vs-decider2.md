# How did the decider toolkit compare with decider2 on each workload?

Date: 2026-09-24 · Script: `benchmarks/modes.py`, `benchmarks/control_flow.py` (decikit alone now; decider2 is gone, so the old column is history)

## Purpose
Check the "slight regression at most" performance rule against the previous engine, and keep the last numbers per workload so a regression is recognisable. Read it before judging a change to the runners, packing or the boundary.

## Method
Fused mode unless named, CPython 3.14, shared dev box often under load (compare rows within a table, not across tables), 2026-09-23 and 24. Batch is `run(df)` at 1M rows; `score` is 20k dict calls. Outputs matched decider2's in every run.

## Result
| workload | decider2 fused | decikit fused |
|---|---|---|
| flagship batch, 3 steps | 8.2M rows/s | 12.5M rows/s |
| flagship `score` p50 / p99 | 26.5 / 40.6 us | 27.7 / 34.7 us |
| flagship `run`: 1 row / 100k / 1M | 851 us / 12.1 ms / 231 ms | 122 us / 0.73 ms / 5.7 ms |
| branch, 1M rows | 3.3 to 3.5M rows/s | 41.6 to 44.3M rows/s |
| loop (<= 360 iterations), 200k rows | 0.64M rows/s | 6.33M rows/s |
| 127-node tree, `run` 1M / `score` p50 | 1216 ms / 215 us | 227 ms / 66 us |
| string-gated tree, `run` 1M / `score` p50 | 1018 ms / 259 us | 360 ms / 104 us |
| 12-band table, ns/row / `score` p50 | 304 / 42.9 us | 164 / 27.4 us |
| table with rows from params, `score` p50 | no equivalent | 35.8 us (was 88.6) |

- Other modes on the flagship: stepped 12.9M rows/s and `score` 37.1 us; interpreted 0.25M rows/s and 59.1 us.
- The last three decikit rows came from one change: the column-at-a-time Arrow boundary (one-row `run` 225 to 122 us; 1M rows 231 to 5.7 ms for the flagship) and hashing a reused params document once (`score` with a params document 42 to 30 us).
- Branch and loop gains come from packing (stepped, Python-driven: 14.8M and 0.81M rows/s); `score` through the branch is 23 us fused against 66 us stepped.
- Latency-only reruns put fused `score` p50 within 1 us (2 to 5%) of decider2, with p99 level or better. Variance is large under load: p99 and 1M-row times move up to 50% between runs.
- Re-run on 2026-10-08 (same scripts, current code, loaded box): flagship fused 133M rows/s batch, `score` p50 37.8 / p99 44.1 us; stepped 115M rows/s, 52.1 us; interpreted 0.31M rows/s, 73.1 us. Branch 926M rows/s fused against 28M stepped; loop 9.1M against 1.08M. Batch is about 10x the 2026-09 figure and `score` about 10 us slower; the cause was not isolated, so treat 25 to 40 us as the `score` band.

## Decision
Performance is at or ahead of the previous engine on every workload, which is why the rules in [engine](../decisions/engine.md) (explicit fusion, packed control flow, dict `score`) stand. A change that moves a row here by more than the noise above needs a reason.
