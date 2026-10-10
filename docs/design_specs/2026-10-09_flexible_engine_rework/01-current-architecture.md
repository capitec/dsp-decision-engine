# Current architecture

What the rework touches, where it lives today, and what each piece is for. Line numbers are as of commit
`54ce33f3`; the code is the truth where they drift. The full description is in `docs/architecture/engine/`;
the reasons are in `docs/design_decisions/decisions/engine.md`.

## Entry points

| Call | Does | Where |
|---|---|---|
| `Step.run(df, params, trace)` | `Engine().bind(self).run(...)`, always interpreted | `decider/steps/_base.py:132` |
| `Step.compile(mode)` | `Engine().bind(self, mode)` | `decider/steps/_base.py:118` |
| `Step.session(df, params, mode, **engine_kwargs)` | a debug `Session` in `mode` | `decider/steps/_base.py:145` |
| `Engine(params_validation, strict_compile)` | two settings and nothing else | `decider/_engine/run/engine.py:56` |
| `Engine.bind(step_or_ir, mode)` | looks `mode` up in `RUNNERS`, resolves the plan, returns an `Executable` | `decider/_engine/run/engine.py:62` |
| `RUNNERS` | `{"interpreted", "stepped", "fused"}` to runner class; documented as a closed set | `decider/_engine/run/engine.py:33` |
| `Executable.run(df, params, trace)` | frame in, frame out | `decider/_engine/run/engine.py:194` |
| `Executable.score(record, params, trace)` | dict in, dict out; builds no polars frame when the plan has no frame steps (`_record_path`, `:131`) | `decider/_engine/run/engine.py:210` |
| `Executable.fallbacks()` | step path to reason, for steps outside a shared kernel | `decider/_engine/run/engine.py:251` |
| `Executable.session(df, params)` | a `Session` over the same runner | `decider/_engine/run/engine.py:267` |

`import decider` must not import numba: `Engine` and `Executable` load through `decider/__init__.py`'s
module `__getattr__` (`tests/test_imports.py` checks it in a subprocess).

## One driver, three runners

Every mode is the interpreted driver with compiled calls. `InterpretedRunner` (`decider/_engine/run/runners/interpreted.py`)
walks the resolved plan: `_node` (`:100`) yields a `before` checkpoint, dispatches on the node class, yields
`after`; `_call` (`:122`) runs a scalar step row by row in Python; `_frame` (`:433`) hands a frame step the
frame with every computed name overlaid; `_branch` (`:191`), `_loop` (`:218`) and `_scatter_gather` (`:243`)
split and merge rows with numpy masks.

| Runner | Overrides | Compiled | Pauses at |
|---|---|---|---|
| `InterpretedRunner` | the driver | nothing | every node, and inside trees through `visit` |
| `SteppedRunner` (`stepped.py:131`) | `_call`: runs the call's unit | each scalar or row call as its own kernel | every node |
| `FusedRunner` (`fused.py:39,54,57`) | `_sequence`, `_branch`, `_loop` | innermost runs of scalar calls as one kernel; branches and loops of plain scalar steps packed | kernel boundaries |

The `Runner` protocol is one method, `iterate(plan, state, params, trace=None, table=None) -> Iterator[Checkpoint]`
(`runners/base.py:38`); `CompiledRunner` adds `_compile(plan, lazy)` and `fallbacks()` (`:53`). A compiled runner
builds a `Compiled` snapshot once per plan under a lock and publishes it whole (`stepped.py:30`).

A runner keeps no per-call state on `self`; `tests/test_conventions.py::test_runners_and_executable_keep_no_per_call_state`
fails any method other than `__init__`, `_compile` or `_build` that assigns to `self`. That rule is what makes
one `Executable` safe from many threads.

## State

`State` (`decider/_engine/run/state.py`) holds one numpy array per `NodeVersion` plus a validity mask for
versions that have nulls. `write(version, values, rows, valid)` (`:72`) scatters into a full-length array;
`read(version, rows)` (`:109`) returns values and mask on a row subset; `frame_of(base, names, rows)` (`:207`)
builds the frame a frame step sees; `from_series(s)` (`:297`) turns a polars `Series` into `(values, valid)`.
`_sources` keeps the polars `Series` a version came from, so compiled calls read Arrow buffers in place.

## IR calls

`CallNode` (`decider/_engine/ir/nodes.py:25`) has `kind` in `scalar | row | frame`, one `fn`, `inputs`, `outputs`,
`params`, `consts`, and `reference`: the plain-Python twin of a compiled row node. Trees, tables and scorecards
lower to row nodes whose `fn` is the shared compiled walker and whose `reference` is the Python walker
(`decider/steps/trees/__init__.py:119-124`). So "one node, two bodies kept equal by tests" already exists; the
rework generalises it.

## Compile units

`compile_plan(plan, fuse, python)` (`decider/_engine/compile/units.py:214`) returns call id to unit. `Kernel`
(`:22`) and `Fallback` (`:120`) share one contract, `run(values, valid, bundles, n)`. `Fallback.run` (`:147`)
loops the rows in Python, boxing Decimals and applying `OPTIONAL` masks per row. `compile_call(node)`
(`decider/_engine/compile/njit.py:121`) returns `(key, callable, reason, declared)`; a non-`None` reason makes
the call a `Fallback` and splits its kernel. Reasons include reading or writing a semantic `str`, a `date`, a
`list` or `dict`, calling a function without `@helper`, or a body numba refuses.

A second kind of fallback happens at launch, not at build. `Kernel.run` catches `FALLBACK_ERRORS` on the
first call (`units.py:102-109`), turns the kernel into one `Fallback` per call (`Kernel._python`) and re-runs.
Nothing warns, `SteppedRunner.fallbacks()` returns the build-time dict (`stepped.py:128`), and
`strict_compile` is not consulted. A step whose output is a `dict`, a `TypedDict` or `list[TypedDict]` is
built as a `Kernel` and hits this on its first launch (`NumbaNotImplementedError: Cannot cast DictType...`),
so it runs in Python under every compiled mode without a word; verified 2026-10-09 on branch
`examples/final`, write-up pending in `known-limitations.md` on branch `r2/output-types`. A `Struct[Item]` or
`Columnar[Item]` return of a tuple compiles. The rework closes this gap (choice 20, P2).

`@allow_fallback` (`decider/steps/_helpers.py`) sets a marker attribute on the function object; the compiler
reads it through `allows_fallback` (an allowlisted private import, `_engine.compile.njit -> steps._helpers`).
This marker-on-the-function pattern is what polars bodies reuse.

## Steps

`FunctionStep` (`decider/steps/_function.py:18`) is a frozen, slotted dataclass: `name`, `fn`, `outputs`,
`nogil`, `reads`, `writes`, `bound`, `id`, `suppress_checks`. `__call__` (`:43`) runs the plain function with
param defaults; `to_ir` (`:59`) harvests the signature and builds a scalar `CallNode`; `step()` (`:71`) is the
constructor and decorator. `FrameStep` (`decider/steps/_frame.py:16`) wraps `fn(df, **params) -> df` and is
already polars-native; `inputs`/`outputs` of `None` mean unknown lineage, a wiring barrier.

`harvest(fn, outputs)` (`decider/_engine/params/harvest.py:28`) reads inputs, params and outputs from a
signature: `x: float` required, `x: float | None` optional, `x: float = missing_as(0.0)` filled,
`x: float = param(...)` a param.

## Polars already inside the driver

Two places show the Series-to-State conversions and the whole-frame handling the polars host needs:

- `_scatter_gather_vectorized` (`interpreted.py:290`) explodes, unnests and regroups a `List(Struct)` column in
  polars instead of a Python loop, but only on a whole-frame column with at least `_SCATTER_ELEMENTS = 1024`
  elements, because the polars route costs about ten lazy collects of fixed overhead.
- `_frame` (`interpreted.py:433`) overlays computed names on the frame, calls the step, and reads each written
  column back with `from_series`, keeping the `Series` as the version's source.

## Registry

`BaseRegistryModule` (`decider/_registry.py`): `class Root(BaseRegistryModule, root=True, group="x")` opens a
registry; every concrete subclass registers under its import path and its `type: Literal["alias"]`;
`Root.resolve(tag)` imports lazy aliases and `decider.<group>` entry points on a miss and returns only
classes registered under that root; `ref(Root)` is a pydantic field type that dispatches on `type`. Seven roots
exist (`ConfigurableStep`, `ConfigStore`, `DatasetProvider`, `Check`, `Metric`, `Plot`, `Block`).

`DeciderConfigSettings.get()` (`decider/_settings.py`) builds a `ConfigStore` from a settings document's `type`
plus its extra fields: `ConfigStore.resolve(self.type).model_validate(self.model_dump())`. The engine registry
copies this.

## Serving

`RequestHandler.mode: str = "fused"` (`decider/serving/_handler.py:95`); `stage()` binds
`Engine().bind(self.pipeline_fn(config), mode=self.mode)` (`:128`) and `score`s at `:218` and `:238`. The mode
comes from `APISettings.mode` (`decider/_settings.py`, env `DECIDER_API__MODE`, CLI `decider serve --mode`).
`strict_compile` and `params_validation` cannot be set from settings at all.

## Lab and tooling

- `assert_equivalent(step, frame, params, modes=MODES, script)` (`decider/lab/testing/_equivalence.py:15`)
  with `MODES = ("interpreted", "stepped", "fused")` hard-coded (`:12`). Per mode: `run(frame)` equals the
  first mode's; `score(record)` per row equals that row of `run`; a session resumed to the end equals `run`;
  a scripted session equals across modes.
- `capabilities().modes` lists `RUNNERS` for the editor bridge (`decider/_tooling/bridge/protocol.py:69`,
  allowlisted private import).
- `docs/architecture/package-layout.md:381`: "Execution modes are a closed set (`RUNNERS`), and so are the IR
  node types."

## What trace, checkpoints and sessions are for

These three are features of the numba driver. A translator engine has none of them and needs none, because
the same pipeline debugs on the numba engine and the equivalence contract says the answers match.

| Feature | What it is | Emitted by | Consumed by | Modes |
|---|---|---|---|---|
| Checkpoints | `before`/`after` per node the run reaches, yielded by the driver's generator | every runner | `Executable._iterate` drains them (`engine.py:246`); `Session` advances them one at a time | all; fused at kernel boundaries |
| `TraceSink` | per-record decision trace: which step ran, in which arm and iteration, with a durable step ref; off unless a sink is passed | kernels write envelopes in-kernel; the driver emits `STEP`, `BRANCH_ARM`, `LOOP_ITERATION`, `FRAME` (`interpreted.py:179-228, 468`) | editor bridge (`_tooling/bridge/bridge.py`), MCP tools (`_tooling/mcp/_tools.py`); the serving handler does not trace by default | all |
| `Session` | break at a path, inspect or override a value, edit a step and re-run from the edit (`_engine/debug/session.py`) | built on checkpoints and `State` | VS Code bridge (`_tooling/bridge/runs.py`, `forks.py`), experiments (`lab/experiments/_jobs.py`), `assert_equivalent` | all; fused pauses at kernel boundaries only |

Checkpoints cost little: `iterate` is not itself a generator and `_Context` is a slotted class, both tuned for
the single-record path (`interpreted.py:30-45, 91`). They exist so `run`, `score` and `session` share one
driver, which is the decision "All modes are one interpreted driver with compiled calls".

## Numbers that constrain the design

| Fact | Value | Source |
|---|---|---|
| Fused `score` p50 / p99 | 27.7 / 34.7 µs (2026-09); 37.8 / 44.1 µs on 2026-10-08 | `experiments/decider-vs-decider2.md` |
| `score` against `run` on a one-row frame | 80.7 against 192.4 µs, same record | `architecture/engine/index.md` |
| Fused batch | 12.5M rows/s (2026-09); 133M on 2026-10-08 | `experiments/decider-vs-decider2.md` |
| Interpreted batch | 0.25M rows/s | same |
| Python fallback cost | 4 steps, 200k rows: all-njit 7.1 ms, Python 167 ms, per-row `objmode` 548 ms | `decisions/engine.md` |
| Fallbacks on real projects | about 40 per pipeline on examples 07 and 08; most remaining time is the object boundary, frame steps and fallbacks | `architecture/engine/runners.md` |
| One record through polars | `pl.when/then` 419 µs at 20 leaves, 543 at 32, 7,796 on a 101 chain; a plugin 236 µs (planner 29, call 37, frame 19) | `experiments/alternatives-to-a-numba-tree-walker.md` |
| Polars expressions against the walker | `pl.when/then` beats a plugin below about 32 leaves, loses 11x at 512 | same |
| Packed branch / loop | 926M rows/s fused against 28M stepped; loop 9.1M against 1.08M | `experiments/decider-vs-decider2.md` |
| `each` polars route | only above 1024 elements; per-record Python 170 to 960 µs, batch about 4 ms fixed | `experiments/each-scatter-gather.md` |
| numba compile | 4.9 to 47 s cold, 0.18 s warm from the salted disk cache | `decisions/engine.md` |
| Keyword arguments | 400 of them cost 1190 µs a call; a dict 60 µs | `experiments/single-record-calling-convention.md` |
