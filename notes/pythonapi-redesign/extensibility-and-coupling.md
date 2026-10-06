# Extensibility and coupling

## What is already pluggable

`decider/registry/core.py` (`BaseRegistryModule(root=True)`) does several jobs:
- Each root gets its own registry.
- A subclass registers by import path and by its `type: Literal[...]` alias.
- `resolve(tag)` imports a lazy alias or a `"module:Class"` path and returns **only registered subclasses**, so config can't name arbitrary code.
- `ref(root)` is a pydantic field that dispatches on `type` and keeps subclass fields on dump.
- `json_schema()` feeds the VS Code forms.

Six roots use it:

| Root | Where | Named in |
|---|---|---|
| `ConfigurableStep` | `steps/configurable.py` | step documents, flow files |
| `ConfigStore` | `config/store.py` | settings |
| `Metric` | `experiments/metrics/base.py` | `experiment.json` |
| `Plot` | `experiments/plots/base.py` | `experiment.json` |
| `Block` | `experiments/report.py` | `experiment.json` |
| `DatasetProvider` | `data/providers/base.py` | `datasets/config.json` |

Checks become the seventh (`checks.md`). `Callback` stays a plain class. That is on purpose: callbacks aren't saved with an experiment definition. Document that choice in its docstring.

## Gaps in the plugin story

1. **There is no discovery.** A plugin type in an installed package resolves only if some code has already imported its module, or the document spells it `"pkg.mod:Class"`. That is why serving projects import their step types in `pipeline.py`, and why the GUIDE has to tell users about it (today it doesn't).
   - **Fix:** on a registry miss, `resolve` looks in `importlib.metadata.entry_points(group=f"decider.{root}")`. Each root declares its group name. It is one function in `registry/core.py`, and every root benefits.
2. **The lazy tables for built-ins are scattered.**
   - `ConfigurableStep.lazy(...)` is filled at the bottom of `steps/__init__.py`, and `DatasetProvider.lazy(...)` at the bottom of `data/providers/base.py`.
   - Metrics, plots and blocks register because importing `decider.experiments` imports them all.
   - **Fix:** declare one convention: each root's built-in table sits next to the root class. That table is then the documented list.
3. **Several sets are closed.** Each has its own list, so adding a construct means editing several places.

   | Hard-coded list | Where | Effect |
   |---|---|---|
   | constructor names `_SCALAR` / `_COMPOSITE` | `ids.py:28-31` | a new constructor gets no ids |
   | flow-file node kinds: step, flow, branch, loop | `flowfile/schema.py` `NODE_MODELS`, `flowfile/load.py:92-97` | `dag`, `each` and `optimise` can't be written in YAML |
   | execution modes `RUNNERS` | `engine/run/engine.py:28` | extended only by mutating a module dict |
   | `default_suite` tuple | `checks/__init__.py` | replaced by the check registry and `standard` |

   **Fix:**
   - Constructors and flow-file kinds share one source: a `COMPOSITES` mapping next to the constructors, which both `ids` and `flowfile` read.
   - Leave `RUNNERS` closed (modes are engine internals) but stop exporting it.
4. **Settings are a global singleton.** `DeciderSettings` knows about serving, the API, config stores and `datasets_path`. Each new plugin kind that needs a setting means editing `settings.py`. `DeciderConfigSettings(extra="allow", type=...)` already shows the better pattern: a root reads its own config section by its `type`. Generalise that, and make settings lazy (`layout-and-visibility.md`).
5. **Custom steps hit a wall past `ctx.call`.** Row kernels and packed tables mean building `CallNode`s by hand, as the built-in trees, tables and scorecard do in their `encode.py`.
   - **Fix:** publish `decider.ir` as the documented kit, and add one or two `IRContext` helpers (for example `ctx.row_call`) so the built-ins stop building nodes by hand. That follows "extend, not new primitives": relax `ctx.call` before adding a new constructor.

## Coupling

### Cross-package edges that point the wrong way

From an import graph over `decider/` (AST, cross-package edges only):

| Edge | Where | Why it is wrong | Fix |
|---|---|---|---|
| `check` ↔ `checks` | `check/run.py:44`, every `checks/*.py` | a cycle, one feature in two packages | merge |
| `checks` → `testing` → `engine.debug` | `checks/equivalence.py:13` | static diagnostics import the debugger | move equivalence and compare out |
| `data` → `lifecycle` | `data/load.py` (`JobHandle`, `InputFingerprint`) | the lower layer depends on the protocol | drop `job=`; `data` owns the summary model |
| `data` → `engine.run.params`, `engine.wiring`, `engine.ir.decls` | `data/preflight.py` | an engine-aware function living in `data` | preflight becomes data checks |
| `experiments` → `engine.wiring.plan`, `engine.ir.nodes` | `experiments/runner.py:40-41`, `metrics/paths.py` | walks IR internals for branch info | read it from `describe()` |
| `mcp` → `debug_bridge.loading`, `.replay` | `mcp/tools.py:20,23` | tooling depends on another tool's internals, so `_bridge` can't be hidden | move `load_module` and `trace` to a shared public helper |
| `debug_bridge` → `serving.handler`, `serving.parse` | `bridge.py:42-43` | the bridge reuses HTTP request coercion | move record coercion to a neutral module (`data` or `_engine.boundary`) |
| `engine/debug/controls.py` ↔ `contract/describing.py::node_json` | by dict shape | core depends on a UI payload's shape | controls take typed nodes; `describing` moves to the bridge |
| `contract.describe` → `engine.run` (`RUNNERS`) | `contract/describe.py:15` | `contract.refs` is a foundation while `describe` sits above the engine, in one package | split the layers (`layout-and-visibility.md`) |
| `flowfile` → `steps.trees` (eager) | `flowfile/load.py:21` | defeats lazy loading of trees | import `expr_condition` lazily |
| `serving/session_ws.py` → `engine.debug.hot`, `Session` | | serving depends on the debugger | delete or mark experimental |
| `engine.ir` ↔ `steps` | `engine/ir/context.py` (`as_step`, `TableRef`, `ParamRef`), lazy | legal under the current rule, but it is a cycle | move `Step` / `Value` / `TableRef` basics into a leaf both import, or accept it and document it |

The `CLAUDE.md` engine rule holds today. `engine/` imports nothing from `steps/trees|tables|scorecard`, `config/`, `serving/` or any tooling package. No test enforces it yet.

### Private names imported across packages

Each one points at a helper that wants a public home:

| Import | Where | Wants |
|---|---|---|
| `engine.debug.session._cast` | `experiments/runner.py:34` | a public value-coercion helper (`Session.set` uses the same logic) |
| `experiments.runner._batches`, `_Local` | `experiments/_worker.py:19` | fine once both live in a private `experiments/_run.py` |
| `flowfile.load._parse`, `_components_index` | `debug_bridge/loading.py` | a public `read_flow(path)` and `components(module)` |
| `engine.boundary._arrow.intrinsics` | `steps/trees/walker.py`, `steps/tables/matcher.py`, `steps/expr/postfix.py` | a documented `_engine` intrinsics module that steps may use (allowlisted) |
| `data.load._fingerprint` | `data/providers/base.py:9` | same package after the move; fine |
| `debug_bridge/loading.py` re-exports `project_root` (`# noqa: F401`) | | `project_root` belongs in `data` (or `_tools`), not the bridge |

### Wire protocol drift

- The bridge replies with ad-hoc dicts, and `ui/lib/src/model/protocol.ts` (427 lines) is written by hand.
- `contract.typescript()` exists, but only a test calls it.
- **Fix:**
  - Declare the reply models in `decider.contract`.
  - Generate `protocol.ts` from them, with a test that fails on drift.
  - Add a `hello` handshake carrying the protocol version.

This is the one coupling between Python and TS that a type checker can't catch today.
