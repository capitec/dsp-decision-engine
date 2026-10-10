# Flexible engine rework

Status: implemented on branch `engine/flexible-rework` (not merged) for P0 to P5; P6 dropped on P5's no-go; P7 not
built. Base commit `44232809` (`feature/python-api-redesign` after the decikit rename and the integration merge),
started 2026-10-10. The rationale now lives in `docs/design_decisions/decisions/engine.md` and `serving.md`, the
measurements in `docs/design_decisions/experiments/polars-host-vs-native.md` and `flexible-engine-comparison.md`;
this folder is history.

## What it proposes

The decider toolkit today has one engine with three modes (`interpreted`, `stepped`, `fused`), chosen by a string from a
closed dict, and two entry points (`run` for a frame, `score` for a dict). The rework makes engines a
registry of pydantic settings documents that anyone can extend, adds a lazy polars engine for large batches,
collapses `score` into `run`, and leaves room for optional polars-native step bodies if they prove worth it:

- `Engine` becomes a registry root (`BaseRegistryModule`), like `ConfigStore`. The three modes become
  `InterpretedEngine`, `SteppedEngine` and `FusedEngine` under a `NumbaEngine` base, keeping their string
  aliases. Settings for `decikit serve` name an engine document, not a mode string. The numba engine itself
  does not change.
- `run` dispatches on its input: a dict takes today's `score` path, a `DataFrame` the current engine, a
  `LazyFrame` the polars engine. `score` is removed. `Step.run` uses the default engine set with
  `decikit.engine(...)`; when nothing is set, that default is fused.
- `PolarsEngine` is lazy: `run(LazyFrame) -> LazyFrame`. It is built as a host first: the fused pipeline runs
  inside `map_batches` per batch, split only at frame steps, which gives streaming, larger-than-memory
  batches and composition with the caller's own plan. Native polars expressions come later, only if a
  measured gap justifies them.
- A step may then declare a polars body with `@fn.impl("polars")` as an optional optimisation. The Python
  body is always required; only the polars engine's native segments use the polars one.
- `SparkEngine` is last and must not shape the rest: the same segmenter, `mapInArrow` boundaries,
  record-by-record pipelines only.

## Decided so far

| Date | Decision |
|---|---|
| 2026-10-09 | `Step.run` follows the default engine; the default when nothing is set is fused |
| 2026-10-09 | Polars bodies are used by the polars engine only; the numba engine never calls one |
| 2026-10-09 | The polars engine serves streaming, composition with a caller's lazy plan, faster batches and polars-native bodies, in that order of certainty; the last two are measured before being built |
| 2026-10-09 | Build the host first, measure against it, then decide on native expressions |
| 2026-10-09 | `FallbackWarning`s stay on under the fused default; `@allow_fallback` is the way to silence one |
| 2026-10-09 | `Step.session()` opens interpreted unless given an engine; it does not follow the default |
| 2026-10-09 | `stepped` stays as an engine |
| 2026-10-09 | Loudness is per engine: fused warns about every step that does not run compiled, kernels that split at first launch included (today nothing reports those); stepped and interpreted never warn; `strict_compile=True` still raises wherever it is set; `@allow_fallback` is discouraged in the guide |
| 2026-10-10 | `Executable.run` checks `dict` first, then `LazyFrame`, `DataFrame`, `Mapping` (spec order cost 0.46-0.51 µs per record call against 0.06 µs) (manager, pending owner confirmation) |
| 2026-10-10 | `RequestHandler.score` keeps its name; only `Executable.score` goes (manager, pending owner confirmation) |
| 2026-10-10 | A scalar step writing a container, `bytes` or a time falls back at build, so its kernel splits around it; replaces the zero-row probe (manager, pending owner confirmation) |
| 2026-10-10 | An all-null typed output takes its declared dtype on every numba engine (manager, pending owner confirmation) |
| 2026-10-10 | `RequestHandler(engine=...)` also takes a tag; a strict refusal is `EngineError`; `stage()` builds kernels before warm-up (manager, pending owner confirmation) |
| 2026-10-10 | `Step.run` caches no executable; the repeated small-frame cost is recorded (manager, pending owner confirmation) |
| 2026-10-10 | The polars host refuses an output with no polars dtype crossing a segment or reaching the output (5 of 12 examples), rather than typing it from data (manager, pending owner confirmation) |
| 2026-10-10 | Host segments emit only what later segments read, split at a read-then-rewrite, declare every pipeline name as input (manager, pending owner confirmation) |
| 2026-10-10 | `PolarsEngine.params_validation` must equal its segments'; `run(DataFrame)` keeps plain `collect()`; a `PolarsEngine` default keeps its settings for `run(LazyFrame)` (manager, pending owner confirmation) |
| 2026-10-10 | `assert_equivalent` compares each record with a one-row frame when the pipeline has a frame step (manager, pending owner confirmation) |
| 2026-10-10 | P6 (polars bodies, native segments, `impl`) is not built: E2 no-go (manager, pending owner confirmation) |
| 2026-10-10 | Unannotated/`Any` outputs and untyped container rendering left as they are and documented as known limitations (manager, pending owner confirmation) |

## Deviations from the plan

- P1: the `dict` check comes first (measured above); `RequestHandler.score` stays.
- P2: no zero-row probe (built, verified, then replaced by the static build-time rule; 4 example steps that ran in
  Python inside split kernels now run compiled); `NumbaEngine` has no `mode`, subclasses return `_runner()`;
  `registered()` does not load entry points; `decikit serve` validates the engine document before starting workers.
- P4: segment emission, splitting and declared inputs as in the table; fact 3 of 09 does not hold for untyped
  Python-built outputs or an `each`'s gathered list, so those are refused; no exception unwrap is needed (open
  question 2: polars passes the step's exception through, in memory, streaming and sinks); every `map_batches` node
  turns pushdown off explicitly (older polars pushed filters below frame steps).
- P5/P6: no-go; `impl`, native segments and E3 to E5 are dropped.
- P7: not built; the entry-point engine test is the seam.
- Pre-existing numba-engine divergences found on the way were fixed on the branch: struct values shared by kernels
  and Python steps, all-null typed outputs, numpy-number outputs, temporal and `bytes` outputs, traces of split
  kernels, interpreted `Columnar` outputs, `missing_as` struct fill.

## Files

| File | What it answers |
|---|---|
| [01-current-architecture.md](01-current-architecture.md) | What exists today, where, and which parts the rework keeps |
| [02-target-architecture.md](02-target-architecture.md) | The design: components, flows, feature matrix per engine |
| [03-design-choices.md](03-design-choices.md) | Each choice, the alternatives considered, why |
| [04-tradeoffs-and-risks.md](04-tradeoffs-and-risks.md) | What gets worse or harder, and what to watch |
| [05-semantics-across-engines.md](05-semantics-across-engines.md) | The equivalence contract every engine must meet |
| [06-experiments.md](06-experiments.md) | Measurements already made that constrain the design, and the ones to run |
| [07-implementation-plan.md](07-implementation-plan.md) | Phases, files to change, tests, docs, migration, open questions |
| [08-api-sketch.md](08-api-sketch.md) | What users will write, before and after |
| [09-polars-host-design.md](09-polars-host-design.md) | The polars host in full: verified facts, algorithm, edge cases, errors, tests, the spike |
| [10-handover.md](10-handover.md) | Read order, repo rules, how to run things, definition of done, reviewer checklists, glossary |

## How this folder relates to the rest of the docs

`docs/design_decisions/decisions/engine.md` holds the rules the current engine follows and the reasons; this
spec does not repeat them, it cites them. When a phase lands, its rationale goes there as new bullets and its
measurements into `docs/design_decisions/experiments/`; `docs/architecture/engine/` then describes the code.
