# Directory structure and migration

This file is the concrete plan: every current file, where it goes, and the order of the moves. The decisions behind it are in the README.

## Three rules

1. **A leading underscore anywhere in a path means private.** `decider/_engine/run/state.py` and `decider/checks/_run.py` are private. Everything else is public and defines `__all__`. Inside a private package, files keep their plain names, because the package underscore already says it.
2. **Public packages are thin.** A public package is an `__init__.py` (plus public subpackages such as `experiments.metrics`) over `_`-prefixed implementation files. Users import from the package, never from a file.
3. **Layers only import downward.** Within a layer, imports are allowed if they form no cycle. Only the core may import `decider._engine`. A library that needs it goes in an allowlist with a reason, and the goal is an empty allowlist.

There are no compatibility shims; this is a fresh version. Moving a name means updating its callers in the same commit, including `tools/` and the GUIDE.

## The "core library" grouping

There are three groups, and they are kept as **siblings** under `decider/`, not nested.

| Group | What's in it |
|---|---|
| **Core** | `decider/__init__.py`, `steps/`, the facades `ir.py`, `trace.py` and `debug.py`, `contract/`, and the private `_engine/` |
| **Libraries** | `checks/`, `diff/`, `experiments/`, `data/`, `testing/`, `config/` |
| **Tooling** | `serving/`, `mcp/`, `cli/`, `_bridge/`, `_tools/` |

Nesting them as `decider/core/` or `decider/lib/checks` would only make import paths longer. The grouping lives in two places instead:
- the top-level docstring, which gives one line per package;
- the layer map, enforced by `tests/test_conventions.py`.

```
L4 tooling     serving  mcp  cli  _bridge  _tools
L3 libraries   checks  diff  experiments  data  testing
L2 contract    contract  (refs, describe, manifest, jobs, bridge messages, schemas)
L1 core        decider/__init__  steps  ir  trace  debug  _engine  config
L0 foundation  exceptions  registry  _annotations  _settings  _suggest  _polars_docs
```

## Target tree

`←` gives the source of each file. `NEW` marks a file that doesn't exist yet.

```
decider/
├── __init__.py                 ← __init__.py (exports below)
├── __main__.py                 ← __main__.py
├── GUIDE.md
├── exceptions.py               ← exceptions.py, minus the HTTP and Arrow errors
├── ir.py                       NEW facade over _engine.ir: Input Output ParamDecl NullPolicy CallNode BranchNode
│                                   LoopNode SequenceNode IRNode Origin IRContext to_ir iter_nodes step_map
├── trace.py                    NEW facade: TraceSink Kind Adapter PassThroughAdapter TraceEvent TraceLossError
├── debug.py                    NEW facade: Session + event models (Paused NodeFinished ... WarningEvent ErrorEvent)
├── _annotations.py             ← fields.py + public half of types.py (Money Percent Duration FieldMetadata
│                                   metadata_of Columnar Struct Raw raw_str); exported only from `decider`
├── _settings.py                ← settings.py; lazy get_settings(), adds checks_path beside datasets_path
├── _suggest.py                 ← registry/resolve.py (did-you-mean)
├── _polars_docs/               ← serializable/ (DataFrame → FrameDoc)
│   ├── __init__.py
│   ├── frame.py                ← serializable/dataframe.py
│   └── schema.py               ← serializable/schema.py
├── _tools/
│   ├── __init__.py
│   └── ids.py                  ← ids.py
│
├── ops/                        NEW public: functions users call in a step, Python and kernel alike (kernel-functions.md)
│   ├── __init__.py             one line per domain, no re-exports
│   ├── decimal.py              round(x, digits, mode=Rounding.HALF_UP|HALF_EVEN|DOWN) first; scaled fixed point later
│   ├── general.py              shared numeric helpers, grown on demand
│   ├── strings.py              explicit string functions, when needed
│   └── credit.py               per-record credit functions (phase 7, credit-functions.md)
│
├── credit/                     NEW public library (phase 7): fitting from data → scorecard / param-table documents
│
├── registry/                   public: BaseRegistryModule ref import_path
│   ├── __init__.py
│   └── _core.py                ← registry/core.py, + entry-point lookup on a miss
│
├── steps/                      public: step frame_step flow dag branch loop each optimise EachMode Step
│   │                               ConfigurableStep Value ParamRef StepRef allow_fallback helper load_flow
│   ├── __init__.py
│   ├── _base.py  _function.py  _frame.py  _sequential.py  _branch.py  _loop.py  _dag.py  _each.py
│   ├── _optimise.py  _helpers.py  _values.py  _configurable.py     ← steps/<same name>.py
│   ├── expr/                   public: parse Expr
│   │   ├── __init__.py  _postfix.py                               ← steps/expr/
│   ├── trees/                  public: TreeConfig expr_condition
│   │   ├── __init__.py
│   │   ├── _condition.py  _encode.py  _ops.py  _reference.py  _trace.py  _walker.py   ← steps/trees/
│   │   └── schema/             public: the document grammar, for building trees in Python
│   │       ├── __init__.py  conditions.py  document.py  flat.py  nodes.py  tree.py  v3.py
│   ├── tables/                 public: DecisionTableConfig
│   │   ├── __init__.py  _encode.py  _matcher.py  _reference.py
│   │   └── schema.py           public: table expression classes
│   ├── scorecard/              public: ScorecardConfig
│   │   ├── __init__.py
│   │   ├── _config.py          ← scorecard/config.py
│   │   └── _kernels.py         ← scorecard/kernels.py
│   └── flowfile/               public: load_flow read_flow components
│       ├── __init__.py
│       ├── _load.py            ← flowfile/load.py (_parse → read_flow, _components_index → components)
│       └── _schema.py          ← flowfile/schema.py
│
├── _engine/                    ← engine/ (one `git mv`; inner files keep their names)
│   ├── __init__.py             Engine Executable to_ir step_map (used by the facades and steps)
│   ├── concurrency.py          NEW once(cache, key, build) + per-thread pool (from boundary/extract.py); see thread-safety.md
│   ├── boundary/               ← engine/boundary/ (incl. _arrow/ untouched), + errors.py ← Arrow errors
│   ├── compile/                ← engine/compile/ (span.py keeps only the span type and its boxing)
│   │   └── semantics/          NEW: __init__.py (registers + SALT over this package and decider/ops)
│   │                               numbers.py ← compile/cpython.py; strings.py ← the operations half of span.py;
│   │                               money.py: operators on the cents type (after the spike)
│   ├── debug/                  ← engine/debug/ minus controls.py and lineage.py; commands/wire stay here
│   ├── ir/                     ← engine/ir/
│   ├── params/                 ← engine/params/ (Table → ParamTable)
│   ├── run/                    ← engine/run/
│   ├── trace/                  ← engine/trace/ (envelope stays; sink is exported via decider.trace)
│   ├── wiring/                 ← engine/wiring/ (Version → NodeVersion)
│   └── types.py                ← internal half of types.py
│
├── config/                     public: ConfigStore JsonFileStore Version VersionPart
│   ├── __init__.py  _file.py  _store.py  _versions.py              ← config/
│
├── contract/                   public, for integrators: refs, FlowDescription, Capabilities, RunManifest, Job,
│   │                               JobHandle, bridge messages, CONTRACT_VERSION, json_schema, typescript
│   ├── __init__.py
│   ├── _refs.py                ← contract/refs.py
│   ├── _describe.py            ← contract/describe.py (describe capabilities resolve_step source_location)
│   ├── _manifest.py            ← lifecycle/manifest.py (json_schema → manifest_schema)
│   ├── _jobs.py                ← lifecycle/job.py
│   ├── _messages.py            NEW pydantic models for bridge replies (protocol.ts is generated from these)
│   └── _schema.py              ← contract/schema.py (mcp_schema deleted)
│
├── checks/                     public: Check CheckContext Finding Severity Report SkippedCheck run standard
│   │                               catalogue check (decorator)
│   ├── __init__.py
│   ├── _model.py               ← check/models.py, + code fix file line suppressed
│   ├── _base.py                NEW Check registry root, CheckContext, @check
│   ├── _run.py                 ← check/run.py
│   ├── _config.py              NEW checks.yaml / checks.json loading
│   ├── _report.py              NEW text / SARIF
│   ├── _ast.py                 ← checks/_common.py (step_ref goes; contract builds refs)
│   └── _builtin/               one family per file; built-ins registered lazily in _builtin/__init__.py
│       ├── __init__.py
│       ├── determinism.py      ← checks/static.py (wall_clock, impure)
│       ├── numeric.py          ← checks/static.py (numeric), then the DCD1xx catalogue
│       ├── ids.py              ← checks/durable_ids.py
│       ├── structure.py  params.py  data.py  runtime.py           NEW (data.py absorbs data/preflight.py checks;
│                                                                       runtime.py ← checks/equivalence.py probe)
├── diff/                       public: structure outputs revisions prove StructureDiff OutputDiff Proof
│   ├── __init__.py
│   ├── _structure.py           NEW
│   ├── _outputs.py             NEW, absorbs Divergence._changed from experiments/metrics/compare.py
│   ├── _revisions.py           ← experiments/revision.py
│   ├── _hash.py                NEW IR content hash (tier 1), replaces checks/equivalence.py _signature
│   ├── _prove.py               NEW tiers
│   └── _smt.py                 NEW z3 tier, `decider[prove]` extra
│
├── experiments/                public: Experiment ExperimentDef ExperimentResult Scenario ScenarioResult RunStatus
│   │                               BASELINE run start validate
│   ├── __init__.py
│   ├── _definition.py          ← experiments/experiments.py (models)
│   ├── _builder.py             ← experiments/experiments.py (Experiment builder)
│   ├── _runner.py              ← experiments/runner.py (run + run_experiment merged)
│   ├── _entry.py               ← runner.load_flow, renamed import_flow
│   ├── _worker.py              ← experiments/_worker.py
│   ├── _cost.py                ← data/preflight.py CostEstimate
│   ├── callbacks.py            public: Callback Run Batch NodeBatch Progress WriteOutputs (JobProgress private)
│   ├── metrics/                public: Metric Summary MetricRef Counts Describe Divergence Divergences Flows
│   │   │                           BranchFlows Histogram Bins Stats ValueCounts
│   │   ├── __init__.py  _base.py  _compare.py  _outputs.py  _paths.py   ← metrics/
│   ├── plots/                  public: Plot PlotRef Figure Bars Changes Distribution Ranges Sankey require_altair
│   │   ├── __init__.py  _base.py  _bars.py  _distributions.py  _sankey.py  ← plots/
│   └── report/                 public: Report Block BlockRef Overview Text Table Chart MetricSection Row Section
│       ├── __init__.py
│       └── _blocks.py          ← experiments/report.py
│
├── data/                       public: read scan datasets dataset DatasetProvider FileProvider FolderProvider
│   │                               LoadedData DatasetError LoadError RecordKeyError (+ the 3 subclasses)
│   ├── __init__.py
│   ├── _formats.py             ← data/formats.py
│   ├── _load.py                ← data/load.py (load → read; job= dropped)
│   ├── _keys.py                ← data/identity.py minus resolve_key and record_ref
│   └── _providers/             ← data/providers/ (base.py catalog.py files.py; project_root lives here)
│
├── testing/                    public: assert_equivalent corpus no_recompile
│   ├── __init__.py  _corpus.py  _equivalence.py  _recompile.py   ← testing/
│
├── serving/                    public: RequestHandler
│   ├── __init__.py
│   ├── _handler.py  _parse.py  _format.py                          ← serving/
│   ├── _errors.py              ← HTTP errors from exceptions.py
│   └── _servers/               ← serving/servers/ (core sanic starlette)
│
├── mcp/                        public: run_stdio build_server Policy
│   ├── __init__.py  _server.py  _tools.py  _policy.py  _editor.py  ← mcp/
│
├── cli/                        public: cli
│   ├── __init__.py             ← cli/__init__.py (group + guide/ids/build/serve/mcp/template)
│   ├── _check.py  _experiment.py  _diff.py  _describe.py           NEW commands
│   └── templates/              ← cli/templates/
│
└── _bridge/                    ← debug_bridge/, launched as `python -m decider._bridge`
    ├── __init__.py  __main__.py  bridge.py  datasets.py  draft.py  experiment.py  forks.py
    ├── loading.py  replay.py  rows.py  run.py  runs.py  timeline.py              ← debug_bridge/
    ├── describing.py           ← contract/describing.py
    ├── controls.py             ← engine/debug/controls.py
    └── lineage.py              ← engine/debug/lineage.py
```

### Deleted

Before each deletion, a grep for callers is re-run against the tree at the time.

| What | Why | Checked 2026-10-05 |
|---|---|---|
| `data/scope.py` (`check_scope`, `ExecutionScope`, `ScopeCheck`, `ScopeError`, `frame_steps`) | no caller | only GUIDE 815–825 and `tests/data/test_scope.py`; the frame-step fact becomes check DCD206 |
| `data.identity.resolve_key`, `record_ref` | no caller | only `data/__init__.py` and `tests/data/test_identity.py`; `suggest_key` and `check_key` are used by `load.py` and `providers/base.py` and stay |
| `data/preflight.py` | split into checks DCD304/DCD504 and `experiments/_cost.py` | callers: `experiments/runner.py`, `ExperimentResult.preflight` |
| `checks/equivalence.py` | split into `diff` and the `runtime` checks | tests only |
| `check/`, `checks/` (old layout), `lifecycle/`, `flowfile/`, `debug_bridge/`, `serializable/`, `engine/` | moved | |
| `contract.mcp_schema` | identical to `json_schema` | one test |
| `serving/session_ws.py` | only its test uses it; it ties serving to the debugger | `tests/serving/test_session_ws.py` |
| `experiments.scan` re-export | no caller | `notes/datasets.md` gets amended |
| `config.VersionedConfig` | no importer | to confirm at deletion time |

### Top-level `decider`

```python
# build
step frame_step flow dag branch loop each optimise EachMode allow_fallback helper
param param_table missing_as ParamTable
Money Percent Duration FieldMetadata Columnar Struct Raw raw_str
TreeConfig DecisionTableConfig ScorecardConfig        # module __getattr__: imported on first use
# run
Engine Executable load_flow describe
# extend
Step ConfigurableStep Value ParamRef StepRef
# errors
DeciderError FallbackWarning
```

`Session` stays in `decider.debug` with the events it emits, so you write `from decider.debug import Session`. `Step.session()` is typed with it. `describe` is the one deliberate second path: it lives in `decider.contract` for integrators and at top level for users.

## Outside `decider/`

| Where | Change |
|---|---|
| `tools/vscode-decider/src/python.ts:31` | `-m decider.debug_bridge` → `-m decider._bridge` |
| `tools/vscode-decider/src/debugging.ts:46` | `decider.debug_bridge.run` → `decider._bridge.run` |
| `tools/vscode-decider/src/locate.ts:5` | `decider.debug_bridge.run` → `decider._bridge.run` |
| `tools/decider-ui/src/model/protocol.ts` | generated from `contract._messages` (phase 5) |
| `pyproject.toml` | `fastmcp` → `mcp` extra; new `prove` extra (`z3-solver`); entry-point groups documented |
| `tests/` | `check/` → `checks/`, `lifecycle/` → `contract/`, `debug_bridge/` → `bridge/`, new `diff/`; others unchanged; tests may import private modules |
| `CLAUDE.md` | engine rule now names `_engine/`; Layout points at this file; layer rule added |
| `decider/GUIDE.md` | every moved import; `tests/test_guide.py` keeps it honest |
| `notes/` | `spec-amendments.md` (VS Code decision 13's `check.run`), `datasets.md` (`scan`), `debug-bridge-relocation.md` (stale) |

## Migration phases

Each phase ends with `uv run pytest` green. The first run after phase 3 recompiles the numba cache, because module names change but keys are content-based.

| Phase | Work | Done when |
|---|---|---|
| 0. Bugs | The four bugs in the README, each with a regression test; serving-path races #1–#4 and #7 from `thread-safety.md` with `tests/serving/test_concurrency.py` | tests fail before the fix and pass after; the stress test passes repeatedly in every mode |
| 1. Guard rails | Add the convention tests (below), with an **allowlist of today's violations** | the tests pass; the allowlist is the to-do list |
| 2. Deletions | The "Deleted" table, re-grepping each entry first | tests green; entries leave the allowlist |
| 3. Moves | One commit per row, mechanical (`git mv` + import rewrite, no logic change), in this order: (1) `engine` → `_engine`; (2) `lifecycle` → `contract`; (3) `flowfile` → `steps/flowfile`; (4) `check` + `checks` → `checks` (merge only); (5) `debug_bridge` → `_bridge` (+ `describing`, `controls`, `lineage`, TS paths); (6) `ids`, `serializable`, `registry/resolve`, `settings`; (7) `fields` + `types` → `_annotations` / `_engine/types`; (8) split `exceptions`; (9) `Table` → `ParamTable`; (10) underscore the implementation files in public packages | after each: full pytest, plus the VS Code extension tests for (5) (`notes/vscode-extension-testing.md`) |
| 4. Surface | Trim every `__all__` to the lists above; add the facades and top-level names; `Step.bind`, typed `Step.session`, `parameters().validate()`; rewrite GUIDE imports | allowlist empty for the `__all__` and private-import rules; `test_guide` green |
| 5. Features | checks registry redesign + `decider check`; `diff` + `decider diff`; unified `experiments.run` + `decider experiment run`; registry entry points; bridge `hello` + generated `protocol.ts` | user stories in `user-stories.md` run as written (add them to `test_guide` through the GUIDE) |
| 6. Catalogue | DCD101–104 first, then the rest; diff tiers 1 and 3, then the z3 tier | each check has a test that fires and one that stays quiet |

Phase 3 is the big diff. It is mechanical, so it can be split across worktrees with one move per agent, as long as the moves are merged in order. Steps (1) and (10) touch the most files.

## Convention tests

Add these to `tests/test_conventions.py`, next to the line limit:

1. **Public modules define `__all__`.** A module is public when no path segment starts with `_`.
2. **No private imports across packages.** `decider.<a>._x` may be imported only from inside `decider/<a>/`. `_engine` may be imported only from core (`decider/__init__.py`, `steps`, `ir.py`, `trace.py`, `debug.py`, `contract`) and from an allowlist with a reason. The likely first entries are `diff` (the fingerprint), `checks` and `experiments`, until `describe()` covers what they read.
3. **Layer map.** Each top-level package gets a layer. An upward import fails, and so does a cycle within a layer. Lazy imports inside functions count too, unless they are allowlisted.
4. **`__all__` budget.** A public package with more than 25 names needs an allowlist entry.
5. **No per-call state on shared objects.** Runner classes and `Executable` in `_engine/run/` never assign or mutate `self.<attr>` outside `__init__` and the lock-held compile (`thread-safety.md`).
6. **Kernel functions live in the kernel library.** `@overload*` / `@intrinsic` only under `_engine/compile/semantics/` (plus the allowlisted Arrow intrinsics); `SALT` covers every file in `semantics/` and `ops/` (`kernel-functions.md`).
