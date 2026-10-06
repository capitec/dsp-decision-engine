# Python API redesign

Status: implemented 2026-10-06 on branch `feature/python-api-redesign`, from `b8aa6ed` to `d29244f`. The full list is `git log 78021e4..feature/python-api-redesign`. Decisions were taken 2026-10-05 (below); deviations are in "As built" at the end.

Much of the work for the VS Code redesign (`notes/vscode-redesign/`) landed as
public Python API. The result is hard to read. You can't tell the typical
workflows or which names are meant for users. There are near-duplicates
(`decider.check` / `decider.checks`, two `load_flow`s, three `Table`s), and
one-function packages sit at the top level (`decider.flowfile`). This folder
records an audit of the whole surface and a target shape.

| File | What it covers |
|---|---|
| `HANDOVER.md` | **Start here to implement it.** A brief for an orchestrator agent: what to read, settled decisions, questions to ask first, work packages by phase with model tiers and gates, traps, suggested skills. |
| `HANDOVER_COMPLETED.md` | **Read after implementation.** What was built, design choices made during the run, every deliberate shortcut, the smaller items left alone, and the allowlists awaiting sign-off. |
| `user-stories.md` | What developers actually do with decider, by role. For each story: the code it takes today, a rating, and the code it should take. |
| `directory-structure.md` | **The plan.** Every current file and where it goes, the layer map, what gets deleted, changes outside `decider/`, migration phases and the convention tests. |
| `layout-and-visibility.md` | Why the namespace is shaped this way: principles, the table of moves and renames, name clashes, and how internals are hidden. |
| `checks.md` | One `decider.checks` package: checks as registry modules, a stable `Finding` format, suppression, config, CLI, and a standard catalogue (overflow, rounding, division, dead code, params...). |
| `diff.md` | New `decider.diff`: comparing two versions of a pipeline by structure and by outputs, and how far "provably identical" can go. |
| `thread-safety.md` | The serving-path thread-safety contract, rules, the re-checked race list (3 new), fixes with no API change, and the stress and convention tests that keep it fixed. |
| `kernel-functions.md` | Where functions that run in kernels live: implicit CPython-equal semantics (numbers, strings) in `_engine/compile/semantics/`, explicit `decider.ops` (`decimal` with three rounding modes, `general`, `strings`, `credit`) for users, money operator overloads behind a spike, one contract, and a cache-salt fix. |
| `credit-functions.md` | Where the coming credit and data-science functions fit: per-record in `decider.ops.credit`, fitting in `decider.credit` (`ScorecardConfig.fit`), metrics in `experiments.metrics`; phase 7. |
| `extensibility-and-coupling.md` | What is pluggable today, what is a closed list, cross-package edges that point the wrong way, and private-name imports. |

## The shape in one picture

```
import decider                      CORE: write and run a pipeline
  flow dag branch loop each optimise step frame_step   param param_table missing_as
  Engine Executable load_flow describe   Step ConfigurableStep Value ParamRef StepRef
  TreeConfig DecisionTableConfig ScorecardConfig (lazy)   ParamTable Money Percent Duration ...
  DeciderError FallbackWarning

decider.checks  decider.diff  decider.experiments(.metrics .plots .report)
decider.data    decider.testing  decider.serving  decider.config
decider.debug (Session) decider.trace             COMPANION LIBRARIES: one task each

decider.ir  decider.registry                      EXTENSION KIT: custom steps and plugins

decider.contract                                  INTEGRATION CONTRACT: versioned models
  refs, FlowDescription, RunManifest, Job, bridge messages, schemas   for VS Code / MCP

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
| Serving-path races: a use-after-free of `str` param spans, a cold-start compile race, the trace sink stored on the shared runner, and a last-writer-wins `exe.report` | `stepped.py`, `interpreted.py`, `engine.py`, from `racecondition_investigation_results.md` re-checked | Wrong or garbage results under concurrent calls on one `Executable`. See `thread-safety.md`. |
| `notes/debug-bridge-relocation.md` is stale | | It says `forks`/`runs` moved to `experiments/` (they didn't). It also names `tools/jupyterlab-decider`, which doesn't exist. |

## Work order

The phased plan, with a "done when" for each phase, is in `directory-structure.md`. In short:
0. Fix the bugs, including the serving-path races in `thread-safety.md`, with a concurrency stress test.
1. Add the convention tests, with an allowlist of today's violations.
2. Delete what nothing calls.
3. Mechanical moves, one commit each, with `engine` → `_engine` first.
4. Trim the surfaces and add the facades and top-level names.
5. New features: checks registry, `diff`, a unified experiment run, CLI commands, entry points, the bridge handshake.
6. The check catalogue (money-safety first) and the diff tiers, including z3.

## Decisions (2026-10-05)

1. **No shims.** This is a fresh version: names move, and callers change in the same commit, including the three bridge paths in `tools/vscode-decider`.
2. **`engine` → `_engine` in one `git mv`**, after the `ir` / `trace` / `debug` facades exist. Reasons are in `layout-and-visibility.md`.
3. **The check policy is `checks.yaml` (or `.json`) at the project root**, beside `configs/` and `datasets/`, and is found the way datasets are (setting `checks_path`). It is not in `configs/`, because that is the served, versioned store. See `checks.md`.
4. **Suppression lives on the step:** `suppress_checks=[alias or code]` on every constructor, as a `ConfigurableStep` field and as a flow-file key. There are no inline comments and no config suppressions. It is stored on `Origin`, so it never recompiles.
5. **z3 is in**, as the `decider[prove]` extra. It handles tier 4 of `diff.prove` and the satisfiability checks for trees and tables.
6. **`Table` → `ParamTable`**, with no alias. `optimise` stays.
7. **Delete `data/scope.py` and `identity.resolve_key` / `record_ref`.** On 2026-10-05 a grep found callers only in the GUIDE (815–825) and in `tests/data/test_scope.py` / `test_identity.py`. `suggest_key`, `check_key` and the key errors are used by `load.py` and the providers, so they stay, privately in `data/_keys.py`. Re-grep at deletion time.

## As built

Deviations from the plan:

- `Step.compile(mode)` replaces `bind`; `FunctionStep.bind(**values)` stays.
- `Validation.check()` exists.
- `config.VersionedConfig` stays, because it has callers.
- The contract `__all__` has 28 names, by decision.
- `Cents` is dropped for `Decimal` / `Currency` (`notes/decimal-and-currency.md`).
- DCD304 keeps an alias, `invalid_params_doc`.
- `import_flow` lives in `steps/_entry.py`.
- z3 counterexamples are step-level evidence (`identical=None`).
- Diff tiers 2 and 3 and `first_divergent_step` are deferred.
- `data.preflight` became DCD304 / DCD504.
