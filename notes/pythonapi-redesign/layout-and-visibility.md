# Layout and visibility

This file explains the reasoning: why the namespace is shaped this way, what moves, which names clash, and how internals are hidden. The file-by-file tree, the layer map, the migration phases and the convention tests are in `directory-structure.md`.

## Principles

1. **`import decider` is the authoring vocabulary.** It holds what you write in a pipeline file, plus the handful of types you annotate with. No user ever needs anything under `decider._engine`.
2. **One companion package per task.** A task is checking, diffing, experimenting, loading data, testing, serving or debugging. Each package has a small `__all__`. A large package splits into subnamespaces (`experiments.metrics`) rather than one flat list.
3. **One canonical path per name.** Today `Engine` is exported from three places, and `flow` and `ParamsError` from two each. The one deliberate exception is `describe`: it lives in `decider.contract` for integrators and at top level for users.
4. **Extension authors get a documented kit.** That kit is `decider.ir` and `decider.registry`, so that "for extenders" doesn't mean "reach into the engine".
5. **The integration contract has one home: `decider.contract`.** It holds refs, `FlowDescription`, the run manifest and jobs (formerly `lifecycle`), the bridge message models and the generated schemas. They are stable and versioned, but users don't import them to write pipelines.
6. **Private means a leading underscore on the module or package.** `__all__` alone controls only `import *` and docs. `dir()` and deep imports still work: `engine.params` shows `bundles declare models tables validate` today.
7. **No shims.** This is a fresh version: a moved name moves, and its callers (including `tools/` and the GUIDE) change in the same commit.

## Why `engine` becomes `_engine` in one go

Once `ir`, `trace` and `debug` have public facades, and `Engine` / `Executable` are exported from `decider`, nothing users need is left in `engine/`. The alternative is underscoring its five internal subpackages one at a time. That means five renames that leave a half-public `engine` package behind.

One `git mv` gives a single rule: nothing under `_engine` is API. The steps:
- Rewrite imports mechanically.
- Tests may keep importing internals.
- Update the engine rule in `CLAUDE.md`.

Generated code contains no `decider.engine` strings, and the numba cache is keyed by content (`numba-compile-cache.md`). So the cost is one cold recompile.

## Moves, renames, removals

| Change | Why | Who has to update |
|---|---|---|
| `check` + `checks` → `decider.checks` | one feature in two packages that import each other | `mcp/tools.py`, GUIDE, `tests/check/`; amend VS Code decision 13 through `spec-amendments.md` |
| `checks.equivalence` / `compare` → `decider.diff` and the `runtime` checks | they aren't static checks, and they pull testing and the debugger into checks | `diff.md` |
| `experiments/revision.py` → `diff/_revisions.py` | revision loading is a diff primitive; experiments reuse it | runner, `_worker` |
| `experiments.__all__` 81 → about 12; built-ins under `.metrics`, `.plots`, `.report`, `.callbacks` | the flat list hides the 5 names users need | GUIDE, tests |
| delete the `experiments.scan` re-export | no caller; `scan` belongs to `data` | amend `notes/datasets.md` |
| `experiments.load_flow` (`runner.py:318`) → private `_entry.import_flow` | clashes with the flow-file `load_flow` | `_worker`, `mcp/tools.py` |
| `run` + `run_experiment` + `Experiment.load().run()` → one `run(def_or_path, ...)` | three entry points that resolve paths differently | bridge `experiment.py`, MCP |
| `flowfile/` → `steps/flowfile/`, re-exported as `decider.load_flow` | it builds a `Step`; it is named after a file format; it eagerly imports trees | `serving/handler.py:159`; the bridge's private `_parse` / `_components_index` become public `read_flow` / `components` |
| `lifecycle/` → `contract/_manifest.py` + `_jobs.py` | a wire contract presented as a user package | `data/load.py`, experiments, bridge |
| `data.load` → `data.read`, `job=` dropped | `data` sits below the contract; most callers want a frame | bridge wraps it |
| `data.preflight` → checks DCD304 / DCD504 + `experiments/_cost.py` | it imports engine internals and returns free text | `experiments/runner.py:92`, `ExperimentResult.preflight` |
| delete `data/scope.py`, `identity.resolve_key` / `record_ref` | no callers (re-checked 2026-10-05) | GUIDE 815–825, two test files |
| `contract/describing.py` → `_bridge/describing.py` | VS Code panel payload, not the contract | `mcp/tools.py:18` |
| `engine/debug/controls.py`, `lineage.py` → `_bridge/` | used only by the bridge | `replay.py` |
| `contract.mcp_schema` → delete | it equals `json_schema` (a test asserts it) | one test |
| `lifecycle.json_schema` → `contract.manifest_schema` | three functions named `json_schema` | |
| `engine/` → `_engine/` | see above | most of `tests/`; `CLAUDE.md` |
| debug events `Warning` / `Error` → `WarningEvent` / `ErrorEvent` | they shadow builtins | bridge (`protocol.ts` matches on `type` strings, not class names) |
| `debug_bridge/` → `_bridge/` | integrator-only; VS Code is the only launcher | `python.ts:31`, `debugging.ts:46`, `locate.ts:5` |
| `fields.py` + public half of `types.py` → `_annotations.py`, exported from `decider` | both hold things you put in an annotation; `types` mixes 4 markers with about 15 helpers | 16 internal importers of the helpers move to `_engine/types.py` |
| `exceptions.py`: HTTP errors → `serving/_errors.py`, Arrow errors → `_engine/boundary/errors.py` | users can see which errors are theirs | serving, boundary |
| `settings.py` → `_settings.py`, lazy `get_settings()`, adds `checks_path` | built at import, so env changes library behaviour; it stays at L0 because `data` and `checks` read paths from it | `cli`, `serving`, `data/_providers` |
| `ids.py` → `_tools/ids.py` | a dev tool used only by the CLI and MCP | `cli`, `mcp/tools.py:187`, GUIDE |
| `serializable/` → `_polars_docs/`; `DataFrame` → `FrameDoc` | empty `__init__`; shadows `pl.DataFrame` | 5 importers |
| `registry/resolve.py` → `_suggest.py` | a did-you-mean helper, not registry logic | 4 importers |
| `Table` → `ParamTable` | clashes with the report block and `TableRef` / `TableValue` | GUIDE, 2 tests |
| `steps.__all__`: drop the 8 concrete `*Step` classes, `as_step`, `TableRef` / `TableValue` | only `describing.py` uses them, via isinstance; give `Step` a `kind` property | describing |
| `steps.trees.__all__` 64 → `TreeConfig`, `expr_condition`; the same for tables and scorecard | the grammar classes are the document, not the API | none outside steps |
| `mcp.__all__` → `run_stdio`, `build_server`, `Policy` | `editor_dir` / `ws_hash` are a socket protocol | cli |
| `fastmcp` → `mcp` extra | every library user installs an MCP stack | `decider mcp` already imports lazily |
| delete `serving/session_ws.py` | only its test uses it; it ties serving to the debugger | one test |

## Names that clash

After the moves, each of these has one meaning per namespace:

| Name | Keep | Rename or hide |
|---|---|---|
| `Table` | `experiments.report.Table` | param table → `ParamTable` |
| `Report` | `checks.Report`, `experiments.report.Report` (namespaced) | `ids.Report` is private |
| `StepRef` | `decider.StepRef` (nested-step field) and `contract.StepRef` (reference) | both are legitimate; each docstring names the other |
| `Version` | `config.Version` | `wiring.Version` → `NodeVersion` (private) |
| `resolve` | `BaseRegistryModule.resolve` | git → `resolve_sha` (private); the `registry.resolve` module → `_suggest` |
| `load_flow` | flow file → Step | the experiments entry importer → private `import_flow` |
| `run` | `Step.run`, `checks.run`, `experiments.run` (each namespaced) | `run_experiment` is folded in |
| `Summary` | `experiments.metrics.Summary` | debug `Summary` event → `RunSummary` |

## Hiding: the options compared

| Option | Verdict |
|---|---|
| Underscore modules and packages | **Use this everywhere.** IDEs, docs and `help()` skip them, and a test can enforce the rule. |
| One new `decider.tooling` / `decider.protocol` namespace | **No.** `decider.contract` already describes itself as that home; it absorbs `lifecycle` and the bridge message models. |
| `__all__` curation only | **Necessary, not enough.** It doesn't stop `dir()` or deep imports. |
| Lazy `__getattr__` with deprecation | **Not needed for moves** (no shims). It is used once: to load `TreeConfig` and friends lazily, so `import decider` doesn't pull in numba. |

Tests may import private modules. A test of an internal is supposed to reach into it.
