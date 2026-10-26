# Experiments

Measurements that already constrain the design, and the ones to run. New experiments follow the repo's
shape: one question per file in `docs/design_decisions/experiments/`, with purpose, method, result and
decision, backed by a script in `benchmarks/`. Run benchmarks through `.scratch/heavy.sh` so they are
serialised and memory-capped.

Decided 2026-10-09: the polars host is built first and measured against; no experiment runs before it. The
one experiment that gates further work is E2.

## Already measured

| Question | Result | What it decides here | Source |
|---|---|---|---|
| Does hosting the fused executable inside `map_batches` work? | Spike, 2026-10-09, polars 1.41.2: a streamed collect split a 100-row frame into 25 batches per row-local segment while the `streamable=False` frame step ran once; answers equal fused after a final `select`; a sub-IR built from existing children binds; a zero-row run yields the output schema; a sub-plan rejects params entries for other segments; a blind overwrite after a frame step trips the shadowing check, a read-then-write does not | The v1 host is feasible as designed, and its four edge cases are known | [09-polars-host-design.md](09-polars-host-design.md) |
| Can polars expressions or a plugin replace the tree walker? | `pl.when/then` beats a plugin below about 32 leaves and loses 11x at 512; one record through polars costs 236 to 419 µs against about 0.5 µs for the walker and 28 µs for a whole fused `score` | Trees stay in boundaries; polars is batch-only; serving stays numba | `alternatives-to-a-numba-tree-walker.md` |
| What does a Python fallback cost? | 4 steps, 200k rows: all-njit 7.1 ms, Python 167 ms, per-row `objmode` 548 ms | Fallback steps are the only place a polars body could pay; about 40 per pipeline on real projects | `kernel-compilation-choices.md`, `runners.md` |
| When should `each` use polars? | Above 1024 elements; below it the Python loop wins; polars route costs about ten lazy collects of fixed overhead | Polars has a fixed per-call cost; a batch engine, not a record engine | `each-scatter-gather.md` |
| How fast is the numba engine? | fused 12.5M to 133M rows/s batch, 28 to 38 µs `score`; packed branch 926M rows/s; loop 9.1M | Native polars branches and arithmetic have no per-row speed case; eager frames stay on numba | `decider-vs-decider2.md` |
| Does `prange` help? | Crossover about 90 µs serial against a 65 to 70 µs fork/join floor; compiles 1.9 to 7.9x slower; changes `ZeroDivisionError` to `inf` | Multi-core comes from polars threads over `nogil` kernels or from the cluster, not from numba | `kernel-compilation-choices.md` |
| How should a record be passed? | 400 keyword arguments 1190 µs, a dict 60 µs; a one-row frame 192 µs against 81 µs for the record path | `run(Mapping)` keeps the record path | `single-record-calling-convention.md` |
| Should generated code hold thresholds? | Literals recompile per retune (343 ms each); arguments cost +1 to 4.5 ns/row | Params stay arguments in numba; literals are fine in polars plans, which have no compile cache | `runtime-arrays-vs-generated-code.md` |
| What does the cold compile cost? | 4.9 to 47 s cold, 0.18 s warm | The fused default makes a cold first `run` visible; Spark must ship the cache per executor | `kernel-compilation-choices.md` |

## Dropped

**E1, the vectorised-fallback threshold.** It asked above how many rows a polars body beats the Python body
per row inside the numba engine. The numba engine never calls a polars body (choice 4), so there is nothing
to measure.

## To run

### E2. The host against native polars (gates phase P6)

- Purpose: find out whether translating steps into native polars expressions would beat the host (the fused
  pipeline inside `map_batches`), and by how much, on pipelines people actually run.
- Method: three pipelines (the flagship, pure arithmetic; a fallback-heavy one with ten `str`/`date` steps,
  which is where a polars body could pay; a 127-node tree); 10k, 1M and 50M rows, in memory and streamed
  from parquet with `sink_parquet`; four configurations: eager fused `run(df)`, the host, hand-written polars
  expressions for the same logic as the ceiling a native translator could reach, and the host with the
  fallback-heavy steps rewritten as `frame_step`s (what an author can already do today without any new
  machinery). Wall time and peak RSS.
- Decides: whether P6 (polars bodies and native segments) is built at all. If the host is within noise of
  hand-written polars on the flagship and the tree, and the `frame_step` rewrite closes most of the gap on
  the fallback-heavy pipeline, P6 is not built and `impl` is dropped from the spec.
- Script: `benchmarks/polars_engine.py`.
- Result (2026-10-10): no-go for P6; E3 to E5 are not run. See
  `docs/design_decisions/experiments/polars-host-vs-native.md`.

### E3. Boundary count (only if P6 goes ahead)

- Purpose: measure what each boundary costs in a lazy plan: the Arrow round trip and the lost optimisation.
- Method: a pipeline of 20 arithmetic steps with k of them made non-native (k in 0, 1, 2, 5, 10, 20); 1M
  rows; compare with the fully native and fully hosted ends; `explain()` to confirm pushdown stops at the
  first boundary.
- Decides: how hard the segmenter must try to merge, and whether to warn when a plan has more than a few
  boundaries.
- Script: part of `benchmarks/polars_engine.py`.

### E4. Native branches (only if P6 goes ahead)

- Purpose: whether a `when/then/otherwise` branch in a native segment is ever faster than the packed numba
  branch inside a boundary.
- Method: `benchmarks/control_flow.py` extended with the polars engine; 1M rows; the branch with two arms of
  one, five and twenty arithmetic steps.
- Decides: whether the segmenter emits native branches at all or treats every branch as a boundary (simpler
  segmenter, one fewer thing to keep equivalent). Expected: a boundary, given 926M rows/s packed.

### E5. Trees in native segments (only if P6 goes ahead)

- Purpose: confirm the earlier result on the current walker: a tree as a `pl.when` chain against the walker in
  a boundary.
- Method: `benchmarks/tree_walk.py` extended; 20, 128 and 512 leaves; 1M rows.
- Decides: whether a small tree may ever get a native emission (expected: no; keep trees in boundaries).

### E6. Spark boundaries (phase P7 only)

- Purpose: the per-executor cold compile with and without a shipped cache; `mapInArrow` throughput against
  `maxRecordsPerBatch`; the cost of one cross-pipeline frame operation done in the job instead of a
  `frame_step`.
- Method: a local Spark with 4 executors; the flagship and the fallback-heavy pipeline; 10M rows.
- Decides: the recommended job shape in the Spark guide; whether `decikit build` needs a `--spark-cache`
  output.
- Script: `benchmarks/spark_engine.py`, skipped when pyspark is absent.

### Regression guard for every phase

`benchmarks/modes.py` and `benchmarks/serving_latency.py` before and after each phase: `score` (then
`run(record)`) p50 and p99 and fused batch rows/s must stay inside the noise band recorded in
`decider-vs-decider2.md` (25 to 40 µs; batch within 20%). Phase P1 adds a type check to the hot path and must
show it costs nothing; P3 changes the default of `Step.run` and must show `benchmarks/modes.py`'s explicit
`compile("fused")` numbers unchanged.
