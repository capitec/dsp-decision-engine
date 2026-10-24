# Where do one request's microseconds go inside the process?

Date: 2026-09-26 (rerun 2026-10-08: flagship `score()` 35.6 us, whole `process_fn` 45.4 us, floor 6.0 us, on a busier box) · Script: `benchmarks/serving_latency.py`

## Purpose
Price every term of `RequestHandler.process_fn` for one record, and find the floor with nothing allocated per call.

## Method
Three pipelines in fused mode: flagship (5 float inputs, 3 steps), tree (18 inputs, the 127-node document from `benchmarks/tree_walk.py`) and gated (the same tree behind a `starts_with`). Lowest p50 of 3 reps of 2-20k calls on a contended box (load ~8.5); absolutes move +-30%, ratios do not.

## Result
| Term (us) | flagship | tree | gated |
|---|---|---|---|
| `pydantic_core.from_json` / `json.loads` | 0.90 / 4.08 | 2.59 / 12.6 | 2.69 / 13.0 |
| `score()` | 30.4 | 62.7 | 136.3 |
| whole `process_fn` | 39.4 | 77.2 | 151.8 |
| floor, nothing allocated per call | 6.98 | 18.2 | n/a |

- Inside flagship's `score()`: state 1.17, load 6.68, params lookup 1.54, runner `iterate` 18.9 (around a 2.1 us kernel), output dict 2.28.
- JSON encode + decode is 3-5% of a request; `process_fn` costs 8-12 us more than its parts (allocator churn, not async or GC).
- The gated tree's 104 us `iterate` was the string span builder: 71.7 us for one 3-byte string (`StringCodes.encode` was 9.7 us). `spans()` has since gained a path that points at each value's own bytes for up to 32 values: gated `score()` is now 82 us (`iterate` 45 us), against 136 us.

## Decision
Keep JSON and `score(dict)` for one record; the pre-bound engine floor (5.6x flagship, 4.2x tree) is the largest open win and is not landed. See [serving](../decisions/serving.md).
