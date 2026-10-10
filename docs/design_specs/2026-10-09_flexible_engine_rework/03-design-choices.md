# Design choices

Each choice the target architecture makes, the alternatives that were considered, and why. When a phase
lands, the bullets that survive move to `docs/design_decisions/decisions/engine.md`.

## 1. Engines are a registry root, not an open dict

- Chosen: `class Engine(BaseRegistryModule, root=True, group="engines")`; the mode strings become aliases.
- Alternatives: keep `RUNNERS` and add `register_mode(name, cls)`; a plain `Protocol` with no registry.
- Why: an engine is a settings document, and the registry already gives settings-driven construction
  (`ConfigStore.resolve(type).model_validate(doc)`), entry-point discovery, lazy built-ins and `json_schema()`
  for UI forms. It closes a real gap today: `strict_compile` and `params_validation` cannot be set for
  `decikit serve`. A dict with `register` would give extensibility and nothing else. The cost is pydantic on
  `Engine`, which holds settings only, so nothing awkward moves onto a model.

## 2. Three thin subclasses, not one `NumbaEngine(mode=...)`

- Chosen: `InterpretedEngine`, `SteppedEngine`, `FusedEngine` under `NumbaEngine`, each fixing `mode`.
- Alternative: one `NumbaEngine` with a `mode` field and special handling of `compile("fused")`.
- Why: the registry resolves an alias to a class. Three aliases to three classes is the mechanism working as
  built; one class with a field needs a second lookup table for the strings.
- `SteppedEngine` stays: it is the only way to inspect compiled results one node at a time, and it costs a
  subclass with one `ClassVar`. Decided 2026-10-09.

## 3. Keep the name `interpreted`; no `DebugEngine`

- Why: interpreted is also the path every `str`, `date` and `list` step takes. Sessions exist in every mode
  (`session(mode="fused")` pauses at kernel boundaries). Calling it "debug" would say it is not for
  production, which is false for those steps.

## 4. Polars bodies are used by the polars engine only; the numba engine is untouched

- Chosen: a step's polars body is read by the polars engine's native segments and by nothing else.
- Alternatives: the numba engine uses it as a vectorised fallback above a row threshold (faster `str`/`date`
  steps on eager frames); an eager `PolarsRunner` overriding `_call`.
- Why: the rule is "serving and anything under 10k rows is numba; polars is for large batches", so the numba
  engine has no business reading a polars body. Keeping it out means one body per engine, no threshold to
  tune, no "both bodies in production" surprise, and the numba engine's tests and numbers stand unchanged.
  Decided 2026-10-09.

## 5. The Python body is always required; a polars body is an optional optimisation

- Why: every pipeline must run on the numba engine, which is where serving, debugging, tracing and the
  equivalence baseline live. A polars-only step would have no reference implementation and could not be
  served. An author adds a polars body only where a batch is measurably slow without it.

## 6. `PolarsEngine` is lazy only; eager is `collect()` at the end

- Chosen: `run(LazyFrame) -> LazyFrame`; `run(DataFrame)` is `run(df.lazy()).collect()`.
- Alternatives: an eager node-by-node polars engine; both, as a mode switch or two engines.
- Why: the engine is a plan translator, and a plan is a `LazyFrame`. Polars' own eager API is lazy under the
  hood, so the mirror costs nothing and gives streaming, `sink_*` and composition with the caller's plan for
  free. Checkpoints are impossible inside a plan polars executes, so an eager polars engine would not buy
  debugging either. Two code paths for the same translation would be two places to keep equal.

## 7. Build the host first; native expressions only if a measured gap justifies them

- Chosen: v1 is the fused pipeline inside `map_batches`, split at frame steps. Then measure it against
  hand-written polars on real pipelines. Native segments and `impl` are built only if the gap is worth it.
- Alternatives: build the translator now; run a standalone spike before any engine code.
- Why: the host is small, delivers streaming and composition on its own, and is the baseline any measurement
  needs. The user's own expectation is that polars-native bodies may turn out unnecessary. Decided
  2026-10-09.

## 8. `impl` is a marker on the function, not a second `def` under the same name

- Chosen: `@tenure.impl("polars")` and `decikit.impl(fn, "polars")`, stored on the function object like
  `@allow_fallback`.
- Alternative: `@step(engine="polars") def tenure(...)` twice, the decorator finding the earlier `tenure` by
  `(module, qualname)` in a global table, the way `typing.overload` works.
- Why: a second `def` rebinds the name, so the mechanism depends on invisible global state keyed by name, which
  is exactly what confuses notebook reloads (the registry already special-cases "same import path means
  redefined"). A marker on the function is explicit, survives `.named()` and `.relabel()` (they share `fn`),
  and `singledispatch.register` is the stdlib precedent for the shape.

## 9. Polars bodies are written against `pl.Expr`

- Why: only native segments call them, and a native segment is an expression plan. No Series variant is
  needed once the numba engine stays out (choice 4).

## 10. `run` dispatches on its input; `score` goes

- Chosen: `run(Mapping)` is today's record path, `run(DataFrame)` the frame path, `run(LazyFrame)` the polars
  plan.
- Alternatives: keep both names; make `run` always take a frame.
- Why: one name for one action, and the input already says what the caller wants. The record path is kept
  because it is the serving path (81 against 192 µs for a one-row frame). A `run` that always took a frame
  would lose it. No compatibility alias: the repo rule is that a moved name changes its callers in the same
  commit.

## 11. `Step.run` follows the default engine; the default is fused when nothing is set

- Chosen: a `ContextVar` set with `decikit.engine(...)`, read by `Step.run` and `compile` when
  `engine=None`; `FusedEngine()` when unset. A `LazyFrame` or Spark input picks its engine by type regardless.
- Alternatives: keep `Step.run` interpreted (today); a process-global setter.
- Why: `run` should be fast and behave like serving, and a notebook should be able to switch everything with
  one line. A setter is a footgun under threads; `contextvars` is how polars, numpy and `decimal` scope such
  settings. The cost is a first-call compile wait on a cold cache; `with decikit.engine("interpreted")` is one
  line away. Decided 2026-10-09.
- `Step.session()` does not follow the default: it opens interpreted unless given an engine. A session exists
  to pause at every node and inside trees, which fused cannot do; `session(df, engine="fused")` is explicit.
  Decided 2026-10-09.
- `FallbackWarning`s stay on under the fused default. They are true, already once per step per executable, and
  `@allow_fallback` silences an accepted one; the alternatives (silent `run`, or no warnings at all) hide
  the slow step from the person most likely to fix it. Decided 2026-10-09. Whether the stepped engine also
  warns is choice 20.

## 12. No dispatch of direct calls

- Why: `ratio(1000.0, 200.0)` under a polars engine has no meaning for literal arguments, and "a step is still
  callable as the plain function" is a documented promise unit tests rely on. If a direct call should ever
  dispatch, it dispatches on the argument type (`pl.Expr` → polars body), which needs no global.

## 13. The v1 segmenter splits only at frame steps; v2 uses the packing rule

- Why: a frame step is the one node whose answer depends on seeing the whole frame, so it is the one place a
  batched host must break. Everything else is row-local and runs correctly per batch. In v2 the rule "a
  control-flow node is native only if everything inside it is native; loops never" is what `compile_packed`
  already applies, and it yields the fewest boundaries.

## 14. A boundary runs the `FusedEngine` over a sub-plan through `map_batches(streamable=True)`

- Alternatives: `map_elements` per step (microseconds per row; no); translating loops into repeated
  `with_columns` passes (no per-row termination, plan depth grows per iteration; no).
- Why: boundaries are row-local by construction, so streaming them is always valid; everything inside is
  existing, tested code; the Arrow boundary that reads a polars frame into kernels already exists.

## 15. Frame steps: whole frame on polars, refused on Spark

- Why: a cross-row step computed per batch is silently wrong. On polars, `map_batches(streamable=False)` gives
  it the whole frame at the cost of a pipeline breaker. On Spark there is no in-process whole frame, and the
  rule is "pipelines are record by record; compose pipelines in the job for anything else".

## 16. `REQUIRED` nulls in a native segment are checked by a streamable UDF (v2)

- Alternative: skip the check in native segments.
- Why: nothing in a lazy expression can raise, and dropping the check would change semantics. In v1 the
  boundary's Arrow read raises exactly as today, so nothing is needed.

## 17. Trace and `Session` are not supported on translator engines

- Why: they are features of the driver, and a translator has no driver. Debugging and tracing happen on the
  numba engine with the same pipeline; the equivalence contract is what makes that sufficient. Raising a clear
  error beats a partial trace.

## 18. Spark lives outside `_engine`, ideally out of tree

- Why: pyspark is optional and heavy; the registry's entry-point path exists for exactly this; building the
  first engine out of tree proves the plugin story; and a hard rule that nothing in the numba engine or the
  segmenter may reference Spark keeps "it also runs on Spark" from shaping the design.

## 20. Loudness about non-compiled steps is per engine

- Decided 2026-10-09: the fused engine is loud about anything that does not run as compiled code; the
  stepped and interpreted engines are not. `@allow_fallback` is discouraged, since a fallback runs row by row
  in Python and costs accordingly.
- Chosen: `FusedEngine` warns once per step for every step outside a shared kernel, build-time `Fallback`
  units and kernels that split at first launch alike, and `strict_compile=True` raises on both unless every
  call involved is `@allow_fallback`; its `fallbacks()` lists launch-time splits with the error as the
  reason. `SteppedEngine` and `InterpretedEngine` emit no `FallbackWarning`; `fallbacks()` still reports for
  stepped, and `strict_compile=True` still raises on stepped when set, because asking for it is not noise.
  The guide says what `@allow_fallback` costs before it shows it, and points to `Struct[Item]` and
  `Columnar[Item]` tuple returns, which compile, as the way to return a record.
- To make a launch-time split visible at `compile()` rather than at the first real batch, the build probes
  each kernel once on zero rows (see P2 in the plan).
- Alternatives: today's behaviour (stepped and fused both warn at build, neither at launch); a `quiet` setting
  on every engine (one more knob for a per-class fact).
- Why: fused is the production engine, where a Python fallback is a performance defect worth hearing about;
  stepped exists to inspect one node at a time, and interpreted runs everything in Python by design, so a
  warning there is noise. The launch-time split is the part that matters most: it is the only fallback that
  neither a warning nor strict mode sees today, and it hits every `dict`-returning step
  ([01-current-architecture.md](01-current-architecture.md), "Compile units"). The fix is in the same place the
  registry phase touches (`fallbacks()` and `strict_compile` per engine), and it is the one deliberate
  exception to "the numba engine is untouched": a reporting change, not an execution change.
- This supersedes the "as in every compiled mode today" wording in choice 11 for the stepped engine.

## 19. Params are literals in native polars and Spark plans (v2)

- Why: the numba rule (params as runtime arguments, never baked into compiled code) exists because recompiling
  costs seconds and the disk cache goes stale. A polars or Spark plan has no compile cache; rebuilding it per
  params document costs microseconds. Inside boundaries the numba rule is unchanged.
