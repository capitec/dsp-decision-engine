# The flexible engine rework against its base

Date: 2026-10-10 · Scripts: `.scratch`-only harnesses (equivalence, per-record, throughput) run against both trees; `benchmarks/modes.py`, `benchmarks/polars_engine.py`

## Purpose
Compare branch `engine/flexible-rework` with its base `44232809` on behaviour, speed and API, and check each prediction of the rework's spec (`docs/design_specs/2026-10-09_flexible_engine_rework/`, files 04 and 06) against what was built.

## Method
- Base code extracted with `git archive` and run on `PYTHONPATH` in the branch's venv (checked by printing `decikit.__file__`); the head is the branch's own tree. Each example runs in its own process, from its latest config version.
- Behaviour: for each of the 12 examples, its request set (the sample plus varied requests) as one frame and as single records on every engine the tree ships (base: interpreted, stepped, fused; head: the same plus polars in memory, streamed with a 7-row chunk, and through `sink_parquet`), compared with fused, plus `assert_equivalent`.
- Per record: base `Executable.score(dict)` against head `Executable.run(dict)` and `RequestHandler.process_fn`, N requests per example (300; consolidation 100, transaction-fraud 200, governance-replay 44), repeat 0 a cold warm-up, then 3 repeats alternating the order (AB, BA, AB); bind and first-call times. Noise band, base against base: p50 ±5% (one -10%), `process_fn` p50 ±1%, throughput ±15%.
- Throughput: fused `run(df)` on each request set cycled to 10k and 100k rows, base and head alternating; the head's polars engine in memory, streamed with 1000- and 10000-row chunks, and through a sink; rows/s and peak RSS.
- Box: 28 threads, 31 GB, shared with other sessions.

## Result

### Behaviour
| | base (3 engines) | head (4 engines) |
|---|---|---|
| frames equal to fused, 12 examples | 12/12, but retail-credit-e2e framed 118 of its 138 requests against 131 on the head (a record-merge bug the branch fixed) | 12/12 on all four engines, polars in memory, streamed (7-row chunk) and sunk to parquet (11/12 sunk: see below) |
| records equal to frame rows | 9/12 | 9/12, the same 3 |
| `assert_equivalent` | passes 9/12 | passes 10/12 (collections now compares a record with a one-row frame) |

- No engine answered differently. The 3 record differences are collections (a cross-row frame step: a record is a one-row frame, by nature) and consolidation and loan-granting-pricing (a record omits an optional key a frame row holds as null; see known limitations).
- All 12 run on polars because five examples now declare their outputs (owner's decision: declared outputs are best practice): consolidation, loan-granting-pricing and retail-credit-e2e type their `dict`/`list[dict]` outputs as TypedDicts, and business-nested declares the items its `each` gathers (business-credit-e2e and governance-replay import it). Every engine now types an `each`'s gathered list from those declarations. Per-record served responses are byte-identical to the base on interpreted and fused for all 12 examples.
- Listed exception (owner-accepted: the declared type fixes the batch schema): in consolidation's served batch, declared-optional keys appear as null in every row and a gathered account's fields follow the declared order (`classified_accounts` 68 rows, `chosen_offer` 69, `runners_up` 33, `search_record` 50; e.g. `chosen_offer` gains `"unpriced_reason": null`). No other example's batch changed. A batch of one shape only (e.g. loan-granting's sample alone) gains declared keys as null the same way.
- credit-limit-management streams on polars but cannot `sink_parquet`: three request fields no step reads arrive as `{}`, an empty struct parquet cannot store; the fused output fails `write_parquet` the same way.
- Found and fixed on the way (all on the base too): a `Decimal` item field inside `each` read its whole part as the mantissa (1.25 doubled gave 0.0002, every engine); an `each` inside a branch raised `AttributeError` on the compiled engines; an int in a nested float field of a declared output failed `run(frame)` on every engine.

### Per record and cold start
Pending: the final base-against-head timings wait until the box is no longer shared with the UI workstream (the lead's signal). The tools and request sets are ready (see Method). Measured per phase so far, alternating, 3 repeats: `run`'s type check costs 0.06 µs per record call with the `dict` check first (P1); `benchmarks/modes.py`'s explicit `compile("fused")` record and batch numbers did not move under the fused default (P3).

### Frame throughput
Pending: the final base-against-head timings wait until the box is no longer shared with the UI workstream (the lead's signal). The tools and request sets are ready (see Method). The polars engine's own throughput is in [the polars host experiment](polars-host-vs-native.md).

### The spec against reality (04 and 06)

| Prediction or risk | Held? | Number |
|---|---|---|
| `run`'s type check costs nothing on the record path (06, P1) | Held with the `dict` check first | 0.06 µs per call; the spec's order cost 0.46-0.51 µs |
| The fused default makes the first `run` wait for numba (04: 4.9-47 s cold, 0.18 s warm) | Held | first fused run 0.8-4.8 s warm, 18 s cold (affordability), against 0.07-0.4 s interpreted |
| A repeated `run` under the fused default | Worse than the spec said (not foreseen) | 2.1-2.7x interpreted on small frames (+2-3.5 ms flagship, ~+35 ms affordability): bind and build per call |
| `FallbackWarning`s surface from `run` | Held; fewer than feared | only fused warns; 4 example steps that ran in Python inside split kernels now compile |
| Launch-time splits made visible (04) | Held, differently | a static build-time rule replaced the zero-row probe; 0 launch splits remain in the 12 examples |
| The host is feasible as designed (06, spike) | Held, except fact 3 | untyped Python-built outputs have no polars dtype and are refused; the five examples that had them now declare them, and every engine types an `each`'s gathered list from its declarations: 12/12 run |
| Optimisation barriers cost nothing in v1 (04) | Held, with a trap | polars 1.30 and earlier pushed a filter below a frame step by default (wrong answers); pushdown is now off explicitly |
| Frame steps break streaming (04) | Held | E2: a frame step's peak grew 3.0-4.0x for 5x the rows |
| Host versus native polars decides P6 (06, E2) | No-go | streamed within noise to 26% of native; tree only the host fits at 50M; frame-step rewrite closes 82-93% of the fallback gap |
| P4 exit: 50M rows streamed well under the frame size | Held only with polars configured | 0.68 GB against a 2.0 GB frame at 4 threads and 100k row groups; 2.5-4.0 GB at defaults (the same for hand-written polars) |
| `Engine` as a pydantic model: drift into runtime state (04) | Guarded | the no-per-call-state convention test walks the engines and the polars executable |
| Removing `score` breaks every caller (04) | Held, done | one rename; `RequestHandler.score` kept (the in-process request entry) |
| Fused batch rows/s and record p50/p99 stay in the noise band (06 regression guard) | Held per phase (P1, P3); the final run is pending | |

### Alternatives
No alternative branch was built. The two contested choices were settled inside their tasks by measurement: the order of `run`'s type checks (P1, measured in place) and host-first against native polars (E2 measured hand-written polars as the ceiling a native translator could reach). P6 was a no-go, so the third (`impl` bodies) was never needed.

### API before and after
The spec's `08-api-sketch.md` "before" ran on the base and its "after" on the branch, unchanged except what P6 and P7 would have added: the same answers on dicts and frames; `run(lazy_frame)` returns a `LazyFrame`; `with decikit.engine("interpreted")` switches `compile()`; `session` opens interpreted; a streamed `scan_parquet` → `sink_parquet` with a frame step gives segments `(ratio, affordable) streamable, (rank_income) not, (flag) streamable`; `assert_equivalent` passes on every engine.

### What got worse
- First fused run in a fresh process: seconds instead of tenths (see the table above); `with decikit.engine("interpreted")` while editing.
- Repeated `pipeline.run` on small frames: 2.1-2.7x interpreted; `compile()` once.
- The polars engine refuses untyped outputs (an author must declare them; the examples now do), and warns twice per fallback step.
- Host memory at polars' default threads: 2.5-4.0 GB streaming 50M rows (polars' own buffering, the same for hand-written polars).

## Decision
Recommend merging once the owner confirms the decisions marked pending in the spec's README and the final per-record timings show no regression beyond noise. Behaviour holds: no engine answers differently, and every refusal is an `EngineError` naming what to change. The costs that remain are in "What got worse" and in [known limitations](../known-limitations.md).
