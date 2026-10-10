# Flexible engine rework

Status: proposal, 2026-10-09. Nothing in this folder is built. Read it before starting any of the work it
describes, and update it as decisions change.

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
