# Trade-offs and risks

What gets worse, harder or riskier under the target architecture, with the mitigation for each. The ones
without a full mitigation are the ones to weigh before starting.

## `Step.run` defaulting to fused changes what people see

Today `pipeline.run(df)` is interpreted: instant on a fresh machine, silent about fallbacks, and the mode in
which a session pauses inside trees. Under the new default:

- The first `run` in a fresh environment waits for numba (4.9 to 47 s cold; 0.18 s with a warm disk cache).
- `FallbackWarning`s surface from `run` for every `str`, `date` and `list` step, once per step per
  executable. Interpreted ran them without a word. Pipelines with many such steps get noisy until the author
  adds `@allow_fallback` or switches engine.
- `GUIDE.md` examples and tests that rely on interpreted behaviour (tree `visit` in a session, instant runs)
  must say `engine="interpreted"` or run inside `with decikit.engine("interpreted")`.

Mitigation: the context manager is one line; the guide shows it first. `assert_equivalent` is unaffected
(explicit engines). The warnings stay on by decision (2026-10-09): they are true, once per step per
executable, and `@allow_fallback` silences an accepted one, so the guide shows `@allow_fallback` beside the
first example that warns. `pipeline.session(df)` keeps opening interpreted, so debugging is unaffected by
the default.

## A launch-time kernel split is silent today

`Kernel.run` catches `FALLBACK_ERRORS` at the first launch and runs the kernel's calls one by one in Python
from then on (`decikit/_engine/compile/units.py:102-109`). Nothing warns, `fallbacks()` does not list it and
`strict_compile` does not raise, so a step returning a `dict`, a `TypedDict` or `list[TypedDict]` runs in
Python under the fused engine without anyone knowing: 0.4 to 2.9 s per 100k rows against 0.10 s for a
`Struct[Item]` return that compiles (measured 2026-10-09 on branch `examples/final`; the write-up is pending in
`known-limitations.md` on branch `r2/output-types`). Until the rework lands, "`fallbacks()` is empty" does not
mean "everything compiled".

Mitigation: choice 20 and P2 make the split visible: record it on the compiled snapshot when it happens, warn
under fused, raise under strict, report it from `fallbacks()`. It is deliberately left unfixed on the
examples branches pending this design, so the fix lands once, per engine.

## Equivalence becomes partly the author's job, where an `impl` exists

A step with a polars body has two implementations, and only tests can show they agree. Serving and small
frames run the Python body on numba; a large batch on the polars engine runs the polars body. A divergence
shows up as "serving and the monthly batch disagree", the most expensive kind. Because polars bodies are
optional (choice 5), most pipelines never face this; the ones that do are the ones an author chose to
optimise.

Mitigation: `assert_equivalent` compares the polars engine with fused on the same frame, and the project
template's test calls it. A later lab check could flag a step with an impl that no equivalence test covers;
not in scope.

## Semantic divergence between polars and Python

Division by zero, integer overflow, rounding modes, string and date edge cases, nulls and decimals all differ
between Python/numba and polars/Spark. The classes and the mapping are in
[05-semantics-across-engines.md](05-semantics-across-engines.md). Some are unfixable by the engine (polars
cannot raise per element); they are detectable, not preventable. Only v2 native segments are exposed; the v1
host runs the Python bodies and has none of this.

## A polars body inside a loop is never used

Loops are always boundaries, so a step inside a loop body runs its Python body on every engine, however
good its polars body is. Authors will be surprised. `fallbacks()` on the polars engine must say "inside a
loop" explicitly, and the guide must say it before the example.

## Optimisation barriers and declared schemas

Every boundary and every frame step is a `map_batches` UDF: polars cannot push projections or predicates
through it, cannot reorder around it, and needs its output schema declared. In v1 that is the whole point
(the plan is a UDF with frame steps between), and the loss is nil because there is nothing native to
optimise. In v2 a pipeline that alternates native and non-native steps gets little from the planner; the
segmenter merges adjacent non-native nodes, which is the only lever. The schema passed to each
`map_batches` is derived by running the segment on a zero-row frame, so it is exact by construction; a frame
step whose body cannot run on zero rows is the one case that needs declared dtypes (see 09).

## Frame steps break streaming

A `frame_step` is a pipeline breaker on the polars engine: the whole frame materialises there. Larger-than-
memory runs work only for pipelines without frame steps, or with a lazy-aware body (a later `impl("polars")`
on a frame step that takes and returns a `LazyFrame`; not planned).

## Translator engines lose trace, sessions and row attribution

No `TraceSink`, no `Session`, and in v2 a `REQUIRED` null is reported with a count, not a row index. A native
expression error surfaces from `collect()` with polars' message unless caught at plan build. Acceptable for
batches of 10k+ rows whose debugging happens on the numba engine with the same pipeline, but it is a real
difference in what "the same pipeline" offers per engine; the feature matrix in
[02-target-architecture.md](02-target-architecture.md) is the reference.

## Removing `score` breaks every caller

Pre-release, so acceptable, but the list is long: the serving handler (two sites), `assert_equivalent`,
`GUIDE.md` (about fifteen examples, all run by `tests/test_guide.py`), `docs/`, benchmarks, tests, the
project template, the MCP tools and bridge if they call it. One commit, no alias, per the repo rule.

## `Engine` becomes a pydantic model

Construction costs microseconds and happens once per process or per staged version; `bind` as a method on a
frozen model is fine. The risk is drift: someone adds runtime state to the engine instead of the runner.
`test_runners_and_executable_keep_no_per_call_state` guards runners and `Executable`; add `Engine` subclasses
to its walk.

## The default-engine context var

A context var set in one thread does not reach threads started earlier, and a notebook cell that sets it
without the context manager leaks it for the rest of the kernel. Both are standard `contextvars` behaviour;
the context manager is the documented form. Serving must not read it.

## Spark

Numba's cold compile is paid once per executor process per pipeline unless the salted disk cache is shipped
with the job and the CPU target matches (`decikit build` records it). `mapInArrow` batch size
(`spark.sql.execution.arrow.maxRecordsPerBatch`, default 10k) sets the per-batch overhead. Any `panic`-class
failure inside a UDF kills the executor, not the driver. All of this is the price of "it also runs on Spark",
and none of it may leak into the numba engine or the segmenter.

## Scope creep

Measured already and not worth building unless a new experiment says otherwise: trees as `pl.when` chains
(lose 11x to the walker at 512 leaves), native branch emission (packed numba branches run at 926M rows/s),
loops in polars (no per-row termination). `impl` itself is scope that may never be needed; P5 decides.

## Conventions

New files stay under 500 lines (`tests/test_conventions.py`); public modules define `__all__`; the layer map
decides where Spark may live (not `_engine`); `PolarsEngine` must keep no per-call state (plan caches keyed by
the params document key, built through `once`); the `_tooling.bridge.protocol -> _engine.run.engine`
allowlist entry goes once `capabilities()` reads the public registry.
