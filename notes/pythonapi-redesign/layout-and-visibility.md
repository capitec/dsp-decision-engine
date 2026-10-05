# Layout and visibility

The target namespace, what moves, and how to keep internals private.

## Principles

1. **`import decider` is the authoring vocabulary.** It holds what you write in a pipeline file, plus the handful of types you annotate with. No user ever needs anything under `decider._engine`.
2. **One companion package per task.** A task is checking, diffing, experimenting, loading data, testing, serving or debugging. Each package has a small `__all__`. A large package splits into subnamespaces (`experiments.metrics`) rather than one flat list.
3. **One canonical path per name.** A re-export happens at exactly one outer level. Today `Engine` is exported from three places, and `flow` and `ParamsError` from two each.
4. **Extension authors get a documented kit.** That kit is `decider.ir` and `decider.registry`, so that "for extenders" doesn't mean "reach into the engine".
5. **The integration contract has one home: `decider.contract`.** It holds versioned models shared with VS Code and MCP. They are stable, but users don't import them to write pipelines.
6. **Private means a leading underscore on the module or package.** `__all__` alone controls only `import *` and docs. `dir()` and deep imports still work: `engine.params` shows `bundles declare models tables validate` today.

## Layers

An arrow means "may import from".

```
L4 tooling      cli  mcp  serving  _bridge                      (entry points, transports)
                 │
L3 libraries    checks  diff  experiments  data  testing        (one task each)
                 │
L2 session      debug (Session, events)  trace (TraceSink)
                 │
L1 core         steps  _engine  config  ir (facade)  params (facade)
                 │
L0 foundations  exceptions  registry  contract.refs  measures  annotations  _suggest
```

The rules, enforced by a test (see Enforcement):
- A layer imports only from the same layer or lower ones.
- `_engine` keeps its current rule: it imports nothing from `steps/trees|tables|scorecard`, `config/` or `serving/`. It also adds: nothing from L2 to L4 except lazily for `Session`.
- No package imports a `_private` name from another package. There are six such imports today (`extensibility-and-coupling.md`).
- L4 packages use each other only through public names. Today `mcp` imports `debug_bridge.loading` and `debug_bridge.replay`.

`contract` straddles layers today. `refs.py` is a foundation (used by checks, data and lifecycle), but `describe.py` imports `engine.run`. Split it: `contract.refs`, `contract.manifest` and `contract.jobs` sit at L0. `describe` sits at L2 and is re-exported as `decider.describe`.

## Target tree

```
decider/
  __init__.py          CORE (below)
  exceptions.py        user-facing DeciderError family only
  measures.py          was fields.py: Money Percent Duration FieldMetadata
  annotations.py       was types.py public half: Columnar Struct Raw raw_str
  ir.py                facade over _engine.ir: Input Output ParamDecl NullPolicy CallNode Origin IRContext
  params.py            facade: param param_table missing_as ParamTable Validation
  debug/               Session + event models; wire commands private
  trace/               TraceSink Kind Adapter PassThroughAdapter TraceEvent TraceLossError
  registry/            BaseRegistryModule ref import_path (+ entry-point discovery)
  steps/               constructors, Step, ConfigurableStep, Value, ParamRef, StepRef
    trees/ tables/ scorecard/   __all__ = the *Config class; grammar in .schema
    expr/
    flowfile/          was decider/flowfile; load_flow re-exported at top level
  checks/              merged check + checks (checks.md)
  diff/                NEW (diff.md)
  experiments/         Experiment ExperimentDef ExperimentResult Scenario RunStatus run start
    metrics/ plots/ report/ callbacks/
  data/                read scan catalog dataset DatasetProvider FileProvider FolderProvider LoadedData DatasetError
  testing/             assert_equivalent corpus no_recompile
  config/              ConfigStore JsonFileStore Version VersionPart
  serving/             RequestHandler; settings.py (lazy); _servers/ _parse.py _format.py _errors.py
  contract/            refs, describe models, manifest (was lifecycle), jobs, bridge reply models, schemas
  mcp/                 run_stdio build_server Policy; _editor.py _tools.py
  cli/
  _engine/             was engine/: ir params wiring compile boundary run debug-internals trace-envelope
  _bridge/             was debug_bridge/ (incl. contract/describing.py)
  debug_bridge/        two shim modules only: __main__.py and run.py (VS Code spawns these paths)
  _tools/ids.py        was ids.py
  _polars_docs/        was serializable/
  _suggest.py          was registry/resolve.py (did-you-mean helper used by 4 packages)
```

### The core namespace

```python
# build
step frame_step flow dag branch loop each optimise EachMode
param param_table missing_as ParamTable            # ParamTable was Table; keep Table as an alias for a while
Columnar Struct Raw raw_str
Money Percent Duration FieldMetadata
allow_fallback helper
# configurable documents (module __getattr__: imported on first use, keeps numba out of `import decider`)
TreeConfig DecisionTableConfig ScorecardConfig
# run and inspect
Engine Executable Session load_flow describe
# extend
Step ConfigurableStep Value ParamRef StepRef
# errors
DeciderError FallbackWarning
```

These are pure additions to the current 28 names: `Step`, `Executable`, `Session`, `load_flow`, `describe`, `DeciderError`, `StepRef` and the three configs. Two behaviour additions come with them: `Step.bind(mode)` and a typed `Step.session -> Session`.

The top-level docstring then lists the companion packages, one line each. That list is the map users don't have today.

## Moves, renames, removals

| Change | Why | Who has to update |
|---|---|---|
| `check` + `checks` → `decider.checks` | one feature in two packages that import each other | `mcp/tools.py`, GUIDE, `tests/check/`; amend `vscode-redesign/userstories.md` decision 13 through `spec-amendments.md` |
| `checks.equivalence` / `compare` → `decider.diff`; cross-mode probe → a check | not checks; they pull testing and the debugger into checks | `diff.md` |
| `experiments/revision.py` → `decider.diff._revisions` | revision loading is a diff primitive; experiments reuse it | runner, `_worker` |
| `experiments.__all__` 81 → about 12; built-ins under `.metrics`, `.plots`, `.report`, `.callbacks` | flat list hides the 5 names users need | GUIDE, tests |
| delete the `experiments.scan` re-export | no caller; `scan` belongs to `data` | `notes/datasets.md` says it is public; amend |
| `experiments.load_flow` (`runner.py:318`) → private `_entry.import_flow` | clashes with flow-file `load_flow`; it is the revision worker's importer | `_worker`, `mcp/tools.py` |
| `run` + `run_experiment` + `Experiment.load().run()` → one `run(def_or_path, ...)` | three entry points that resolve paths differently | bridge `experiment.py`, MCP |
| `flowfile/` → `steps/flowfile/`, `decider.load_flow` | it builds a `Step`; named after a file format; eagerly imports trees | `serving/handler.py:159`; bridge uses private `_parse` / `_components_index` (make a public `read_flow`) |
| `lifecycle/` → `contract.manifest` + `contract.jobs`; `__all__` trimmed | a wire contract shown as a user package; `data` should not depend on it | `data/load.py`, experiments, bridge |
| `data.load(job=)` dropped; `LoadedData.summary` owned by `data` | data sits below the contract | bridge wraps it |
| `data.preflight` → data-dependent checks in `checks` | imports engine internals and returns free text | `experiments/runner.py:92`, `ExperimentResult.preflight` |
| `data.scope`, most of `data.identity` → delete or private | about 25 names with no caller outside tests | decision 7 in README |
| `contract/describing.py` → `_bridge/` | VS Code panel payload, not the contract | `mcp/tools.py:18` (`find_pipelines` goes to a neutral helper) |
| `contract.mcp_schema` → delete | equals `json_schema` (a test asserts it) | one test |
| `lifecycle.json_schema` → `contract.manifest_schema` | three functions named `json_schema` | |
| `engine/` → `_engine/` after `ir`, `params`, `debug` and `trace` facades exist | nothing users need is left in it | most of `tests/` (they may import privates); `CLAUDE.md` engine rule |
| `engine.debug.Warning` / `Error` events → `WarningEvent` / `ErrorEvent` | shadow builtins | bridge, `protocol.ts` uses the `type` strings, not class names |
| `debug_bridge/` → `_bridge/`, keep `debug_bridge/__main__.py` and `debug_bridge/run.py` as shims | VS Code spawns `-m decider.debug_bridge` and `-m decider.debug_bridge.run` and evaluates `run.node_path()` | none in `tools/` |
| `types.py` split → `annotations.py` (public) + `_engine/_types.py` (helpers) | stdlib name; 15 helpers imported by 16 files | mechanical |
| `fields.py` → `measures.py` | it holds display units, not pydantic fields | 2 files, 1 test |
| `exceptions.py`: HTTP errors → `serving/_errors.py`, Arrow errors → `_engine` | users can see which errors are theirs | serving, boundary |
| `settings.py` → `serving/settings.py`, lazy `get_settings()` | `DeciderSettings()` built at import; env changes library behaviour | `cli`, `data/providers/catalog.py` (gets its own datasets setting) |
| `ids.py` → `_tools/ids.py` | dev tool; CLI and MCP only | `cli`, `mcp/tools.py:187`, GUIDE |
| `serializable/` → `_polars_docs/`; its `DataFrame` → `FrameDoc` | empty `__init__`; shadows `pl.DataFrame` | 5 importers |
| `registry/resolve.py` → `_suggest.py` | a did-you-mean helper, not registry logic; clashes with `wiring.resolve` | 4 importers |
| `steps.__all__`: drop the 8 concrete `*Step` classes, `as_step`, `TableRef` / `TableValue` | only `describing.py` uses them (via isinstance); give `Step` a `kind` property | describing |
| `steps.trees.__all__` 64 → `TreeConfig`, `expr_condition`; same for tables and scorecard | grammar classes are the document, not the API | none outside steps |
| `mcp.__all__` → `run_stdio`, `build_server`, `Policy` | `editor_dir` / `ws_hash` are a socket protocol | cli |
| `fastmcp` → `mcp` extra | every library user installs an MCP stack | `decider mcp` already imports lazily |
| `serving/session_ws.py` → delete or mark experimental | only tests use it; ties serving to the debugger | one test |

### Names that clash

After the moves above, each of these has one meaning per namespace:

| Name | Keep | Rename / hide |
|---|---|---|
| `Table` | `experiments.report.Table` (reached as `report.Table`) | param table → `ParamTable` |
| `Report` | `checks.Report`, `experiments.report.Report` (namespaced) | `ids.Report` private |
| `StepRef` | `decider.StepRef` (nested-step field) | `contract.StepRef` stays namespaced. Both are legitimate, but say so in both docstrings |
| `Version` | `config.Version` | `wiring.Version` → `NodeVersion` (private) |
| `resolve` | `BaseRegistryModule.resolve` | git → `resolve_sha` (private); `registry.resolve` module → `_suggest` |
| `load_flow` | flow file → Step | experiments' entry importer → private `import_flow` |
| `run` | `Step.run`, `checks.run`, `experiments.run` (each namespaced) | `run_experiment` folded in |
| `Summary` | `experiments.metrics.Summary` | debug `Summary` event → `RunSummary` |

## Hiding: the options compared

| Option | Verdict |
|---|---|
| Underscore modules and packages (`_engine`, `_bridge`) | **Use this for internals.** It is robust: IDEs, docs and `help()` skip them, and a test can enforce the rule. The cost is mechanical renames, with shims for fixed external paths. |
| One `decider.tooling` / `decider.protocol` namespace | **Not as a new package.** Wire models are spread across `contract.refs`, `lifecycle`, debug commands and bridge replies. The real fix is making `decider.contract` that home, since it already describes itself that way. |
| `__all__` curation only | **Necessary but not enough.** Use it on every public module. It doesn't stop `dir()` or deep imports. |
| Lazy `__getattr__` with `DeprecationWarning` | **Only for moved names with a documented old path.** If decision 1 is "just move", this is needed only for the lazy `TreeConfig` etc. at top level. |

Tests may import private modules. A test of an internal is supposed to reach into it.

## Fixed paths that must keep working

These are named from outside Python, so a move needs a shim at the old path:
- `python -m decider.debug_bridge` (`tools/vscode-decider/src/python.ts:31`)
- `python -m decider.debug_bridge.run` (`debugging.ts:46`)
- `decider.debug_bridge.run.node_path()`, evaluated by the IDE (`locate.ts:5`)
- the `decider = "decider.cli:cli"` script entry (`pyproject.toml`), and `python -m decider`
- `decider.serving.RequestHandler`, used by the project template
- the string `"decider.serving.servers.starlette:create_app"` (`cli/__init__.py:136`), plus the sanic one

## Enforcement

Add to `tests/test_conventions.py`, in the same style as `ALLOWLIST`:

1. **Every public module defines `__all__`.** A public module is one with no underscore in its path, excluding `__main__`. Exceptions go in a dict, with a reason.
2. **No cross-package private imports.** `from decider.<a>... import _x` or `import decider.<a>._y` is allowed only when the importer is inside `decider/<a>/`.
3. **Layer rules.** Each top-level package maps to a layer, and the test fails on an upward import. Lazy imports inside functions are allowed only where an allowlist names them (for example `Step.session` importing `Session`).
4. **`__all__` size budget.** This is optional: fail when a package's `__all__` passes something like 25 names without an allowlist entry. That is the cheap guard against a second 81-name list.
