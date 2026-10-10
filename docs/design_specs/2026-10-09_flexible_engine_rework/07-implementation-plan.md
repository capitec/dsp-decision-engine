# Implementation plan

Eight phases, each shippable on its own and each with an exit criterion. P1 to P4 are certain; P5 is the
measurement that decides P6; P7 is Spark, last and outside the core. Sizes are relative: S is a day, M a
few days, L a week or more. Each phase states its behaviour precisely enough to implement from; the
reviewer's checklist is in [10-handover.md](10-handover.md).

## P0. Record the decisions (S)

- Write the decision bullets this spec proposes into `docs/design_decisions/decisions/engine.md`, marked
  "proposed", so the rules are in the place maintainers read before changing the engine.
- Exit: the bullets exist and link here.

## P1. `run` absorbs `score` (M)

Goal: one entry point dispatching on input type; the record path unchanged in cost. No `engine` parameter
yet; `Step.run` stays interpreted until P3.

### Behaviour

```python
def run(self, data, params=None, trace=None):
    if isinstance(data, pl.LazyFrame):           # a numba executable has no lazy plan: collect first
        data = data.collect()
    if isinstance(data, pl.DataFrame):
        return self._run_frame(data, params, trace)      # today's run()
    if isinstance(data, Mapping):
        return self._run_record(data, params, trace)     # today's score(), body unchanged
    raise TypeError(f"run() takes a dict record, a polars DataFrame or LazyFrame, not {type(data).__name__}")
```

The order matters: a `DataFrame` is not a `Mapping`, a `dict` is. The record path returns a dict and keeps
`_record_path` (no polars frame when the plan has no frame steps). Two `isinstance` checks are added to the
hot path; the benchmark must show they cost nothing. `run`'s docstring becomes the user documentation for
both shapes.

### Changes

| Change | Where |
|---|---|
| dispatch as above; `score` removed | `decikit/_engine/run/engine.py` |
| `Step.run(data, params=None, trace=None)` unchanged in signature; delegates to `Executable.run` | `decikit/steps/_base.py` |
| `score` → `run` at both call sites | `decikit/serving/_handler.py:218,238` |
| `exe.score(record, params)` → `exe.run(record, params)` | `decikit/lab/testing/_equivalence.py` |
| every `.score(` example; the "Modes" bullet | `decikit/GUIDE.md` (run by `tests/test_guide.py`) |
| `score` references | `docs/architecture/engine/index.md`, `runners.md`, `docs/guides/*.md`, `docs/design_decisions/decisions/engine.md` ("`score` takes a dict" bullet becomes "`run` takes a dict or a frame") |
| `score` calls | `benchmarks/modes.py`, `serving_latency.py`, `serving_transports.py`; the project template under `_tooling/cli/templates/`; MCP tools and bridge if they call it (`grep -rn "\.score(" decikit`) |
| tests | `tests/run/test_run_dispatch.py` (new): dict in → dict out; frame in → frame out; lazy in → collected frame out; a list → `TypeError` naming the accepted types; a dict record equals the matching row of a frame run. Existing `score` tests renamed, assertions unchanged |

Exit: `benchmarks/modes.py` record p50 within noise of the baseline; `uv run pytest` green;
`grep -rn "\.score(" decikit docs tests benchmarks examples` empty.

## P2. Engine registry (M)

Goal: engines are registry documents; serving names one from settings. The numba engine's code is wrapped,
not changed.

### Behaviour

```python
# decikit/_engine/run/engines.py
class Engine(BaseRegistryModule, root=True, group="engines"):
    model_config = ConfigDict(frozen=True, extra="forbid")
    params_validation: Literal["eager", "lazy"] = "eager"

    @abstractmethod
    def bind(self, step_or_ir: Any) -> Executable: ...


class NumbaEngine(Engine):
    strict_compile: bool = False

    @abstractmethod
    def _runner(self) -> Runner: ...                       # abstract, so NumbaEngine itself never registers

    def bind(self, step_or_ir):                              # today's Engine.bind, minus the dict lookup
        return Executable(resolve(step_or_ir), self._runner(), self.params_validation == "lazy", step_or_ir,
                          engine=self)


class InterpretedEngine(NumbaEngine):
    type: Literal["interpreted"] = "interpreted"
    def _runner(self): return InterpretedRunner()

class SteppedEngine(NumbaEngine):
    type: Literal["stepped"] = "stepped"
    def _runner(self): return SteppedRunner(strict=self.strict_compile)

class FusedEngine(NumbaEngine):
    type: Literal["fused"] = "fused"
    def _runner(self): return FusedRunner(strict=self.strict_compile)


Engine.lazy({"polars": "decikit._engine.polars.engine"})   # P4 adds the module; the alias can be declared now
```

- `Engine.resolve("fused")` returns `FusedEngine`; `Engine.resolve("decikit._engine.run.engines:FusedEngine")`
  too. An unknown tag raises `RegistryError` with a did-you-mean, as every root does.
- `Step.compile(engine="interpreted")`: the parameter is renamed from `mode`; a `str` is resolved
  (`Engine.resolve(tag)()`), an `Engine` instance is used as is. `Step.session(data, params=None,
  engine="interpreted")` likewise; `**engine_kwargs` is removed (pass an instance).
- `Executable.engine` holds the document it was built from.
- Settings: `class EngineSettings(BaseModel): model_config = ConfigDict(extra="allow"); type: str = "fused"`
  replaces `APISettings.mode`. It is resolved in `serving/_handler.py` (`construct_handler_from_settings`):
  `Engine.resolve(s.type).model_validate(s.model_dump())`. Not in `_settings.py`, which is L0 and may not
  import `_engine`. Environment: `DECIKIT_API__ENGINE__TYPE`, `DECIKIT_API__ENGINE__STRICT_COMPILE`,
  `DECIKIT_API__ENGINE__PARAMS_VALIDATION`. `decikit serve --engine fused` sets `api__engine__type`.
- `RequestHandler.engine: Engine = field(default_factory=FusedEngine)` replaces `mode: str`; `stage()` calls
  `self.engine.bind(...)`.
- `BaseRegistryModule.registered()` (new classmethod, beside `json_schema`): `{tag: cls}` for every registered
  subclass of `cls`, importing lazy modules first. The bridge's `capabilities().modes` becomes
  `tuple(sorted(tag for tag, cls in Engine.registered().items() if issubclass(cls, NumbaEngine)))`: only
  engines that support sessions.
- `decikit/__init__.py`: `Engine`, `InterpretedEngine`, `SteppedEngine`, `FusedEngine` (and `PolarsEngine`
  from P4) join `_LAZY` and `__all__`; `import decikit` still imports no numba.
- Loudness, per choice 20: no new setting; the policy is by class. `FusedEngine`'s runner warns at build for
  every `Fallback` unit (as today) and, new, for every kernel that splits at launch, with the numba error as
  the reason; `strict_compile=True` raises in both places unless every call involved is `@allow_fallback`.
  `SteppedEngine`'s and `InterpretedEngine`'s runners never warn; `strict_compile=True` on stepped still
  raises. `fallbacks()` includes launch-time splits as `"split at launch: <error>"`.
- Mechanism, in two parts. First, make the split a build-time fact: at the end of `_build`, launch every
  `Kernel` once on zero rows (empty arrays of each read's dtype, the default bundles). A numba dispatcher
  compiles for its argument types at the first call whatever `n` is, so the typing failure that splits a
  dict-returning kernel (`NumbaNotImplementedError: Cannot cast DictType...`) should fire on the probe; a
  kernel that splits there is reported, warned about or raised on exactly like a build-time `Fallback`, and
  `exe.fallbacks()` after `compile()` is complete. Verify this on the dict-returning step before relying on
  it; if the probe does not trigger the failure, fall back to part two alone. The probe costs the compile
  the first real call would have paid, no more. Second, for a kernel that still splits later (another error
  class, or a dtype the probe did not see): `Kernel.run`'s split branch (`units.py:102-109`) is the one place
  a kernel writes to `self`; give `Kernel` an `on_split` callback the runner installs at build, which records
  the split into the snapshot's `fallbacks` under the runner's lock (write-once) and warns or raises per the
  runner's class. A split happens at most once per kernel, so the callback runs at most once per kernel per
  process. This is the one deliberate change inside the numba engine: reporting, not execution.

### Changes

| Change | Where |
|---|---|
| the classes above; `RUNNERS` and the `Mode` literal removed | `decikit/_engine/run/engines.py` (new); `engine.py` keeps `Executable` and gains `engine=` |
| `compile`/`session` parameter `mode` → `engine`, tag or instance | `decikit/steps/_base.py`; every `compile(mode=`/`session(mode=` caller (`grep -rn "mode=" decikit tests`) |
| `EngineSettings`; `mode` removed; `--engine` | `decikit/_settings.py`, `decikit/_tooling/cli/__init__.py`, `decikit/serving/_handler.py` |
| `DECIKIT_API__MODE` → `DECIKIT_API__ENGINE__TYPE` | `_tooling/cli/templates/`, `docs/guides/projects-and-serving.md`, `README.md` |
| `registered()` | `decikit/_registry.py` |
| `capabilities().modes` from the registry; allowlist entry `_tooling.bridge.protocol -> _engine.run.engine` removed | `decikit/_tooling/bridge/protocol.py`, `tests/conventions_allowlists.py` |
| exports | `decikit/__init__.py` |
| zero-row probe of every kernel at the end of `_build`; `Kernel.on_split`; `SteppedRunner` warns only when its class is loud; `FusedRunner` is loud; `fallbacks()` merges launch-time splits | `decikit/_engine/compile/units.py`, `decikit/_engine/run/runners/stepped.py`, `fused.py` |
| tests | `tests/registry/test_engines.py`: resolve by alias and import path; `NumbaEngine` is not registered; settings round trip through `EngineSettings`; unknown tag error with did-you-mean; an engine loaded from an entry point (a test package, as `tests/mcp` does); `tests/test_imports.py` still green; `tests/serving/` builds a handler from env with `strict_compile=true` |
| tests | `tests/compile/test_launch_time_split.py`: a step returning a `TypedDict` under fused warns once at `compile()` (the probe) and appears in `fallbacks()` before any row runs, with the numba error as the reason; raises under `strict_compile=True`; is silent and accepted under `@allow_fallback`; the same step under stepped is silent but listed, and raises under strict; under interpreted nothing is reported; the split kernel's answers equal interpreted; a kernel forced to split after the probe (monkeypatched `run`) is reported through `on_split` once |
| docs | `docs/architecture/engine/index.md` ("Engine and bind", "Extending"), `package-layout.md` (the closed-set sentence at `:381`, the "new execution mode" row, the registry roots table gains `Engine`), `docs/guides/projects-and-serving.md`, `docs/guides/extending.md` |

Exit: `DECIKIT_API__ENGINE__TYPE=fused DECIKIT_API__ENGINE__STRICT_COMPILE=true decikit serve` refuses a
pipeline with an undeclared fallback; a third-party engine from an entry point is reachable by tag.

## P3. The default engine (S)

Goal: `Step.run` and `compile` follow one context-scoped default, fused when nothing is set; `session`
opens interpreted.

### Behaviour

```python
# decikit/_engine/run/engines.py
_FUSED = FusedEngine()
_default: ContextVar[Engine | None] = ContextVar("decikit_default_engine", default=None)

def default_engine() -> Engine:
    return _default.get() or _FUSED

@contextmanager
def engine(which: str | Engine):
    token = _default.set(Engine.resolve(which)() if isinstance(which, str) else which)
    try:
        yield
    finally:
        _default.reset(token)
```

- `Step.run(data, params=None, trace=None, *, engine=None)`: `engine` may be a tag or an instance; `None`
  means `default_engine()`. A `LazyFrame` is collected and run on that engine until P4 makes it pick the
  polars engine. `Step.compile(engine=None)` the same. `Step.session(data, params=None,
  engine="interpreted")` has a fixed default and ignores the context.
- Exported as `decikit.engine`. It shadows a common variable name when imported bare; that is accepted (the
  alternative, `use_engine`, reads worse in the guide). Record the choice.
- Warnings are unchanged: a `str`, `date` or `list` step warns once per step per executable under the fused
  default, as in every compiled mode today.
- Serving does not read the context var.

### Changes

| Change | Where |
|---|---|
| the code above | `decikit/_engine/run/engines.py`, `decikit/__init__.py` |
| `run`, `compile` read the default; `session` fixed to interpreted | `decikit/steps/_base.py` |
| run `uv run pytest tests/test_guide.py -q`; for each failure decide: the example needs `engine="interpreted"` (say why in the PR), or its shown output changes (a `FallbackWarning` now prints); the "Modes" bullet becomes "Engines" and shows the context manager; `@allow_fallback` appears beside the first example that warns | `decikit/GUIDE.md`, `docs/guides/debugging-and-testing.md`, `docs/guides/compile-and-fallback.md` |
| tests that call `Step.run` and assume interpreted (no warning, tree `visit`) get `engine="interpreted"` or the context manager, each with a reason | `tests/` |
| the bridge and MCP pass their engine explicitly (they already pass a mode) | `decikit/_tooling/bridge/`, `_tooling/mcp/` |
| tests | `tests/run/test_default_engine.py`: unset → `FusedEngine`; context manager sets and resets, also on exception; nesting; a thread started inside the context inherits it, one started before does not; an async task inherits a copy; `session` opens interpreted under a fused default; a `str` step warns once under the default |

Exit: a notebook sets nothing and `pipeline.run(df)` is fused; sets `interpreted` once and every `run` and
`compile` below is interpreted; `benchmarks/modes.py` explicit numbers unchanged.

## P4. `PolarsEngine` v1, the host (M)

Goal: `run(LazyFrame) -> LazyFrame` for any pipeline, streaming, with answers equal to fused. The complete
design, verified facts, algorithm, error list and test list are in
[09-polars-host-design.md](09-polars-host-design.md); this table only places the pieces.

| Change | Where |
|---|---|
| `PolarsEngine(Engine)` with `type: Literal["polars"]` and `segments: FusedEngine = FusedEngine()`; `bind` → `PolarsExecutable` | `decikit/_engine/polars/engine.py` (new) |
| segmentation, schema derivation, params prune, the overwrite drop, emission, final select | `decikit/_engine/polars/host.py` (new; split further if it passes 500 lines) |
| `Step.run(LazyFrame)` picks `Engine.resolve("polars")()` when `engine=None` | `decikit/steps/_base.py` |
| `assert_equivalent(engines=...)` with the polars checks from [05-semantics-across-engines.md](05-semantics-across-engines.md); `MODES` removed; the default engine list comes from `Engine.registered()` through the public `decikit.Engine` | `decikit/lab/testing/_equivalence.py` |
| `PolarsExecutable` added to the no-per-call-state convention test | `tests/test_conventions.py` |
| `PolarsEngine` exported lazily | `decikit/__init__.py` |
| tests | `tests/polars_engine/`: the list in 09 |
| docs | `GUIDE.md` section "Large batches: the polars engine"; `docs/guides/polars-engine.md` (new, in the nav); `docs/architecture/engine/polars.md` (new) |

Exit: every `examples/*` flow equal between fused and the host in memory and streamed with a small chunk
size; a 50M-row parquet scores to parquet with peak RSS well under the frame size on a pipeline without
frame steps.

## P5. Measure (S)

- Run E2 from [06-experiments.md](06-experiments.md) against the host; write the experiment file.
- Exit: a written go/no-go for P6, with numbers. No-go means `impl` and native segments leave this spec.

## P6. Polars bodies and native segments (L, only on a go from P5)

Detailed design is written when P5 says go, following the shape of 09. The pieces:

| Change | Where |
|---|---|
| `impl(fn, engine)` and `impls_of(fn)`, storing on the function object beside `_ALLOW_FALLBACK`; exported as `decikit.impl` | `decikit/steps/_helpers.py`, `decikit/__init__.py` |
| `FunctionStep.impl(engine)` sugar; `to_ir` passes `impls=impls_of(self.fn)`; decoration-time checks: same input and param names in order, no `Decimal`/`Currency` anywhere | `decikit/steps/_function.py` |
| `CallNode.impls: tuple[tuple[str, Callable], ...] = ()` | `decikit/_engine/ir/nodes.py` |
| segmenter v2: the packing rule (native iff everything inside is native; loops never); adjacent non-native nodes merge; reasons per non-native step, "inside a loop" included | `decikit/_engine/polars/host.py` |
| emitter: null-policy wrapping, the `REQUIRED` check UDF, params as `pl.lit`, one `with_columns` per native run, schema resolution per native step for error attribution | `decikit/_engine/polars/native.py` (new) |
| tests | `tests/polars_engine/test_native.py`, `tests/steps/test_impl.py`: the v2 list in 05 |
| docs | `GUIDE.md` section "A polars body for a step"; decisions and the E3 to E5 experiment files |

Exit: E2's fallback-heavy pipeline reaches the measured native speed; `assert_equivalent` passes with and
without impls on the example flows.

## P7. `SparkEngine` (L, out of tree first)

- `decikit-spark` package: `SparkEngine(Engine)` with `type: Literal["spark"]`, entry point
  `decikit.engines`, the same segmentation applied to a Spark `DataFrame`, `mapInArrow` boundaries whose
  function builds the sub-executable once per executor process (a module-level `once` cache keyed by the
  plan's content key) and runs `pl.from_arrow(batch)` → `run` → `to_arrow()`, `frame_step` refused with
  "cross-row steps are not supported on Spark; run them in the job between pipelines", a note or script for
  shipping the numba cache (`decikit build` records the CPU target). E6. Guide page in that package.
- Rule enforced by review and a convention test: no `spark` or `pyspark` reference under `decikit/`.
- Exit: the flagship and the fallback-heavy pipeline run on a 4-executor local Spark with answers equal to
  fused, and warm executors pay no compile.

## Conventions and tests to keep green throughout

- Every new file under 500 lines; `__all__` on public modules (`decikit/_engine/polars/` is private, so none).
- `test_runners_and_executable_keep_no_per_call_state` extended to `PolarsExecutable` and the `Engine`
  subclasses.
- `test_imports.py`: `import decikit` imports neither numba nor pyspark.
- Layer map: `_engine/polars/` is core (polars is a hard dependency); Spark is outside `decikit/`;
  `_settings.py` never imports `_engine`.
- `tests/test_conventions.py::test_no_references_to_docs_or_experiments`: no `doc 03`, `§`, `EXPERIMENTS` in
  code; the spec is cited from `docs/design_decisions/`, never from code.

## Open questions

1. Should a `frame_step` get a lazy-aware body (`LazyFrame -> LazyFrame`) so streaming survives it? Only if
   P4's streaming test on a real pipeline needs it.
2. Which exception type reaches the caller when a step raises inside a `map_batches` segment (polars may
   wrap it)? Decided by test in P4; see 09.
3. Does a zero-row launch trigger the typing failure that splits a dict-returning kernel? Expected yes;
   decided by the first test in `tests/compile/test_launch_time_split.py` (P2). Closed 2026-10-10: yes, for a
   `TypedDict`, a `dict` and a `list[TypedDict]` return (`NumbaNotImplementedError: Cannot cast DictType...` in
   native lowering). Then decided (2026-10-10, pending the owner): a function step that writes a `dict`, a
   TypedDict or a `list` is a build-time `Fallback` instead, so its neighbours stay compiled and no launch is
   needed to find it; with that rule a probe found nothing in the 12 examples and was removed, and `on_split`
   reports the launch-time splits that remain (a `dict` param, a float for a `Decimal`). See
   `docs/design_decisions/decisions/engine.md`, "Fall back to Python only on compile failure".

Settled on 2026-10-09 and recorded in [03-design-choices.md](03-design-choices.md): `stepped` stays
(choice 2); `FallbackWarning`s stay on under the fused default and `session` opens interpreted (choice 11);
loudness is per engine, fused loud and stepped/interpreted quiet (choice 20).
