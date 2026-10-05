# Python API redesign

Status: proposal, 2026-10-05. Nothing here is implemented yet.

Much of the work for the VS Code redesign (`notes/vscode-redesign/`) landed as
public Python API. The result is hard to read. You can't tell the typical
workflows or which names are meant for users. There are near-duplicates
(`decider.check` / `decider.checks`, two `load_flow`s, three `Table`s), and
one-function packages sit at the top level (`decider.flowfile`). This folder
records an audit of the whole surface and a target shape.

| File | What it covers |
|---|---|
| `user-stories.md` | What developers actually do with decider, by role. For each story: the code it takes today, a rating, and the code it should take. |
| `layout-and-visibility.md` | The target namespace in layers (core, companion libraries, extension kit, integration contract, private). Also what moves, what gets renamed, how to hide internals and how to enforce it. |
| `checks.md` | One `decider.checks` package: checks as registry modules, a stable `Finding` format, suppression, config, CLI, and a standard catalogue (overflow, rounding, division, dead code, params...). |
| `diff.md` | New `decider.diff`: comparing two versions of a pipeline by structure and by outputs, and how far "provably identical" can go. |
| `extensibility-and-coupling.md` | What is pluggable today, what is a closed list, cross-package edges that point the wrong way, and private-name imports. |

## The shape in one picture

```
import decider                      CORE: write and run a pipeline
  flow dag branch loop each optimise step frame_step   param param_table missing_as
  Engine Executable Session load_flow describe   Step ConfigurableStep Value ParamRef
  TreeConfig DecisionTableConfig ScorecardConfig (lazy)   Money Percent Duration ...
  DeciderError FallbackWarning

decider.checks  decider.diff  decider.experiments(.metrics .plots .report)
decider.data    decider.testing  decider.serving  decider.config
decider.debug   decider.trace                     COMPANION LIBRARIES: one task each

decider.ir  decider.registry                      EXTENSION KIT: custom steps and plugins

decider.contract                                  INTEGRATION CONTRACT: versioned models
  refs, FlowDescription, RunManifest, Job, generated schemas      for VS Code / MCP

decider._engine  decider._bridge  ...             PRIVATE: no stability promise
```

## Headline findings

1. **Users have to reach deep.**
   - `Step`, `Executable`, `Session`, `DeciderError`, `load_flow`, `TreeConfig`, `DecisionTableConfig` and `ScorecardConfig` all need imports through `decider.engine.*`, `decider.steps.*` or `decider.flowfile`.
   - The GUIDE tells users to run `help(decider.engine.debug.Session)`.
   - `Step.session()` is typed `-> Any`.
2. **Nothing reads as private.**
   - `steps.trees.__all__` has 64 names; outside `steps/` the code imports only `expr_condition`.
   - `experiments.__all__` has 81 names. `engine.debug.__all__` has 30, including `Warning` and `Error`, which shadow the builtins.
   - `engine.params` has no `__all__` at all.
   - `lifecycle` has 30 names and no GUIDE mention.
3. **The same name means different things in different places:**

   | Name | Meanings |
   |---|---|
   | `load_flow` | flow file → Step, vs import `"module:qualname"` |
   | `Table` | param table, vs report block |
   | `Report` | check findings, vs experiment layout, vs `ids.Report` |
   | `StepRef` | nested-step field type, vs contract reference |
   | `Version` | wiring, vs config |
   | `json_schema` | three functions |
   | `resolve` | git SHA, vs registry tag, vs wiring |
   | `run` | `check.run`, `experiments.run`, `run_experiment`, `Step.run` |

4. **One feature is split across two packages.** `check` holds the runner and models, `checks` holds the built-ins. They import each other, and checks pull in `testing` and the debugger only because `checks/equivalence.py` (a cross-mode test and a version diff) lives there.
5. **The integration contract leaks into user signatures.**
   - `data.load(job=)`, `experiments.run(job=)` and `ExperimentResult.manifest` all take or carry `lifecycle` protocol types.
   - Version constants (`EXPERIMENT_VERSION`, `MANIFEST_VERSION`, `check_*_version`) are user exports.
6. **The CLI has gaps.** There is no `decider check`, `decider experiment run`, `decider diff` or `decider describe`. Every CI story means writing Python.
7. **Extensibility is better than it looks.**
   - Six plugin kinds already share `BaseRegistryModule`: `ConfigurableStep`, `ConfigStore`, `Metric`, `Plot`, `Block` and `DatasetProvider`.
   - Checks are the odd one out.
   - Gaps:
     - There's no entry-point discovery.
     - Flow files accept only step/flow/branch/loop.
     - `ids.py` hard-codes the constructor names.
     - `RUNNERS` can only be extended by mutating it.

## Bugs found during the audit

These need fixing regardless of the redesign. Each was confirmed by reading the code, and the first by running it.

| Bug | Where | Effect |
|---|---|---|
| `checks.compare` declares different flows "structurally equivalent" | `checks/equivalence.py:105-113`: `_signature` hashes path, kind, origin and function source, but not `CallNode.consts` or param defaults | Two `ConfigurableStep`s differing only in `factor=2.0` vs `3.0` come back `equivalent=True, method="structure"`. This is the one place the code claims a proof. |
| A revision scenario silently needs `entry=` | `Experiment(..., entry="")` default; `validate` never checks it | Fails inside the worker subprocess instead of at validation |
| `experiments.run(def_)` resolves `input.path` / `input.dataset` against the cwd | `runner.py:66` calls `input_frame(def_)` without the file | `Experiment.load(path).run()` and `run(def_)` read different data for the same JSON |
| `numeric` check gets money wrong in three ways | `checks/static.py` | It raises a false positive on `cents * rate / 12` (rate is float). It misses `a / b` and misses bare `round()` on a `Money(cents=True)` output. `money-rounding-and-overflow.md` specified the `round()` lint, but it was never built. |
| `notes/debug-bridge-relocation.md` is stale | | It says `forks`/`runs` moved to `experiments/` (they didn't). It also names `tools/jupyterlab-decider`, which doesn't exist. |

## Work order

Each step keeps `uv run pytest` green and is worth doing on its own.

1. **Fix the bugs above.** Add a regression test for each.
2. **Add the layering and visibility tests.** These go in `tests/test_conventions.py` (see `layout-and-visibility.md`). Then fix the six cross-package private imports. No renames yet.
3. **Pure additions to the core namespace.**
   - Add the missing top-level names.
   - Type `Step.session -> Session`.
   - Add `Step.bind(mode)`.
   - Add `pipeline.parameters().validate(doc)`.
4. **Trim `__all__`.** This covers `steps`, `steps.trees/tables/scorecard`, `experiments` (into `.metrics` / `.plots` / `.report`), `engine.debug`, `lifecycle`, `mcp` and `data`. Delete stray re-exports such as `experiments.scan`.
5. **Merge `check` and `checks` into a registry-based `decider.checks`.** Give findings codes and add `decider check`. See `checks.md`.
6. **Create `decider.diff`.** Move comparison out of checks and the revision machinery out of experiments. See `diff.md`.
7. **Unify the experiment run.** One `experiments.run(def_or_path, ...)`, plus a `decider experiment run` command.
8. **Moves and renames.** These are the big mechanical diffs:
   - `flowfile` → `decider.load_flow`.
   - `lifecycle` → `decider.contract`.
   - `describing.py` → the bridge.
   - `engine` → `_engine` behind facades.
   - `debug_bridge` → `_bridge`, with shims for the two TS entry modules.
   - Split `types`, `exceptions` and `settings`.
9. **Standard check catalogue and diff tiers.** The money-safety checks come first.

## Decisions needed

These are not guessed. Each changes what gets built.

1. **Break or shim?** HEAD is "out with the old decider", so this looks like a breaking window. Can moved names just move, or do they need `__getattr__` deprecation shims for a release? The recommendation is to move them. Keep shims only for paths outside this repo (the TS entry modules and `decider.serving.RequestHandler` in templates).
2. **Rename `engine` → `_engine` in one go?** The alternative is underscoring its five internal subpackages one by one. The single rename is a larger diff but leaves one rule ("nothing outside `_engine` is engine internals"). It also means updating the engine rule in `CLAUDE.md`.
3. **Check config location.** The options are `checks.json` beside `configs/`, `[tool.decider.checks]` in `pyproject.toml`, or a key in `experiment.json`.
4. **Suppression mechanism.** The options are in-source `# decider: ignore[DCD101] reason` comments, id-based suppressions in config (which survive refactors), or both.
5. **`z3` as an optional extra.** This only matters if the SMT tier of `diff.prove` is ever built. Recommendation: not now.
6. **Rename `Table` → `ParamTable` and `optimise` → `optimize`?** Recommendation: rename `Table` (it clashes); leave `optimise` alone (renaming gains little).
7. **`decider.data.scope` and most of `decider.data.identity`.** About 25 names have no caller outside tests. They were meant for the VS Code "selected record" flow, which was never wired. Delete them, or park them privately until a caller exists?
