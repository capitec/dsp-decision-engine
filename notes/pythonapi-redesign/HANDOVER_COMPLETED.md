# Handover completed: the Python API redesign

Written 2026-10-06, when the orchestration that `HANDOVER.md` describes was finished. This note is for whoever picks the work up next. It records:
- what was built;
- the design choices made along the way that the plan didn't settle;
- every deliberate shortcut;
- the smaller items that were seen and left alone on purpose.

`README.md` → "As built" has the short list of deviations; this file has the detail.

## State at completion

| | |
|---|---|
| Branch | `feature/python-api-redesign`, 57 commits from `b8aa6ed` to `0a7e791`, on top of `78021e4` (the notes). `git log 78021e4..feature/python-api-redesign` lists them. Not pushed. |
| Tests | 3002 passed, 1 xfailed, 0 failed (baseline 2415 passed). The baseline's four macOS-only failures don't occur on this Linux machine. |
| Concurrency stress test | `tests/serving/test_concurrency.py`, 30 cases, passed 5 runs in a row and is part of every full run |
| Extension | `tools/decider-ui` 46/46, `tools/vscode-decider` 28/28, `pnpm -r build` clean. The headless in-editor smoke test passes its describe and datasets steps, then hangs at `debugFlow` on the pre-refactor commit too. |
| Latency (warm `score()`, 3-step float flow, µs) | before `78021e4`: interpreted 59.8–60.3, stepped 45.5–46.4, fused 29.8–33.4. Now: 60.7, 47.5, 30.6. Re-measure with `benchmarks/score_modes.py`. |
| Phase 7 (credit) | not started; the user picks which functions move |

Every phase gate in `HANDOVER.md` was run. A two-axis code review (standards and spec) ran over the whole branch, and its findings are fixed in `485f02b`.

## How the work was run (worth knowing before you run more)

- **Worktrees.** The Agent tool's `isolation: "worktree"` creates worktrees from the stale `main` (`96889ca`), not the current branch. Create them yourself instead: `git worktree add -b wp/<id> .scratch/wt/wp<id> <branch>`. Commit there, then cherry-pick onto the branch. Conflicts were nearly always in `tests/conventions_allowlists.py`, `decider/checks/_base.py` (the lazy table and `STANDARD`) and the GUIDE, and were easy to resolve.
- **Memory.** Several numba-heavy full suites running at once exhausted RAM and swap and killed the orchestrator. `/tmp` is a RAM-backed tmpfs. Every pytest, benchmark and pnpm run therefore goes through:
  - `.scratch/heavy.sh` for the full suite, benchmarks and pnpm (7 GB cap, 45-minute timeout);
  - `.scratch/light.sh` for targeted runs (4 GB cap, 20-minute timeout).

  Both run the command in its own `systemd-run --user --scope` with swap off, so an overrun kills only that command. They queue on a flock, so there is at most one heavy and one light run at a time. They set `TMPDIR` to `/var/tmp/d<hash>`, which is on disk.

  The temp dir has to be short and outside both git and any hidden directory. Inside the repo, `ids` tests (which need a temp dir outside git) and the editor-socket tests (where a long path exceeds the `AF_UNIX` limit) fail. Under `~/.cache` the hidden-directory filter skips files. Both wrappers are gitignored scratch files, so copy them if you need them.
- **Don't touch the checkout a gate is running in.** Two gate failures came from cherry-picking or running `uv sync` mid-run, not from the code.

## Design choices made during the run

These were decided with the user during the run, or were the smallest correct option where the notes were silent. The user's decisions are marked **(user)**.

### Surface and layout
- **`Step.compile(mode)`, not `Step.bind(mode)` (user).** `FunctionStep.bind(**values)`, which re-defaults params, already existed and shadowed `bind(mode)` on single function steps. The user kept `FunctionStep.bind`. `pipeline.compile("fused")` returns an `Executable`.
- **`Validation.check()` is kept**, and no `raise_for_errors()` alias was added. `pipeline.parameters().validate(doc)` returns it.
- **`config.VersionedConfig` stays.** The plan said it had no importer, but `ConfigStore.read` and `create_version` return it.
- **`decider.contract.__all__` has 28 names (user).** It holds the 18 core names plus the 10 nested model types users receive inside `RunManifest` and `Job`. It keeps a `BIG_ALL` entry for that reason.
- **`describe` at top level** is the one deliberate second path. It is an allowlisted upward import (`__init__ -> contract`).
- **`import decider` no longer imports numba.** `Engine`, `Executable` and the three `*Config` classes load through the module's `__getattr__`, and `_engine/__init__` loads `Engine` lazily. `tests/test_imports.py` checks this in a subprocess.
- **`import_flow`** (a `"module:qualname"` entry to a step) lives in `decider/steps/_entry.py` (layer 1), so `diff` and `experiments` (both layer 3) can share it. A step is never called as a factory, even though function steps are callable.
- **Inferring an experiment's `entry` never calls user code.** It matches only a module-level variable that is the pipeline object itself. An earlier version called every zero-argument function in the module, which hung a test and could have run anything.
- **Top-level `_x` modules** (`_annotations`, `_settings`, `_suggest`, `_polars_docs`, `_tools`, `_bridge`) are private to decider but shared across it, so the private-import rule doesn't flag them. Only `_engine` is restricted to core.
- **`struct_item` moved to `_annotations.py`** and the value-coercion helper to `_engine/types.py` (`cast_values`). `decider.ir` gained `base_annotation`, `FiniteStateMachineNode`, `ScatterGatherNode`, `check_id` and `check_name` so that libraries stop reaching into `_engine`.

### Thread safety
- **Compiled state is one immutable snapshot.** It is the `Compiled` NamedTuple, compiled under a lock, double-checked on the plan, and published with one assignment. A dropped packed kernel is swapped copy-on-write.
- **`exe.report` is per thread** (a `threading.local`). It reads the same for any single-threaded caller.
- **Module caches use `once(cache, key, build)`** from `_engine/concurrency.py`. `ParamsCache` publishes with `setdefault`, so validation may run twice but is pure. Registry roots lock registration and listing, never a hit, and never while importing (an import registers classes).
- **A warm `score()` takes no lock.** The cost of the thread-safety work was about 0.5–1.5 µs per call, from the `threading.local` write and the per-call context.

### Numba cache
- **An unreadable cache index is a miss.** numba unpickles the whole index before comparing keys, so an index written before a decider type moved crashed every compiled run with `ModuleNotFoundError`. Any user upgrading with a warm cache would have hit this. `_TolerantIndexFile` in `njit.py` treats any unpickling failure as an empty index, and the next save rewrites it.
- **`SALT` is found by walking files.** It covers every file in `_engine/compile/semantics/`, plus `span.py`, `decimal_type.py` and `decider/ops/`. A missing required file fails at import, and the convention test recomputes the same set.
- **`fingerprint` follows `module.attr` calls** to the function and hashes it by content. Editing a helper module now invalidates the cached kernels of the steps that call it.

### Checks
- **A check is a registry module** (`Check`, group `decider.checks`) with a stable code. The 34 built-ins run from DCD001 to DCD506, plus DCD207/208 (`table_overlap` and `table_gap`, added to `checks.md`).
- **`checks.run`** takes a flat list of checks, aliases, config dicts or sets.
  - With no list, it uses the project's `checks.yaml`, or `standard` when there isn't one.
  - An explicit list still honours the project's `disable` and `fail_on`.
  - `disable` removes any finding by alias or code, whichever check emitted it.
- **Suppression lives on `Origin.suppress_checks` (user: plural spelling).** It is set by every constructor, `ConfigurableStep` and flow files. It covers the node and its ancestors, and suppressed findings stay in the report. It never enters a cache key; a test checks that adding one doesn't recompile.
- **The temporary `numeric` check is gone.** DCD101–110 replace it and fix its three money bugs:
  - the false positive on `cents * rate / 12`;
  - the missed `a / b`;
  - the missed bare `round()` on a money output.
- **DCD304 is `invalid_params_doc`.** It uses `parameters().validate(doc)`, so it reports every unknown key and every rejected value, not only the first.
- **`data.preflight` is gone.** Its checks became DCD504 and DCD304, its cost estimate became `ExperimentResult.cost`, and `ExperimentResult.preflight` is now a checks `Report`.
- **Tree and table reachability, overlap and gap logic** lives entirely in `checks/_builtin/structure.py`, so `steps` doesn't import `checks`. DCD207/208 come from the same z3 pass as DCD201, so they are reported whenever `unreachable_branch` runs, which is in `standard`. Without the `prove` extra, that pass is listed under `skipped`.

### Diff and proof
- **Tier 1 (IR hash) is the only route to `identical=True` without z3.** Tree and table arrays are hashed by content, not by the raw addresses stored in `consts`. Anything `fingerprint` would key by identity becomes a fresh nonce, so two subprocesses can't read different objects as equal. Anything else that can't be hashed by content also hashes unequal, which is the safe direction.
- **The z3 tier proves trees and tables identical when every changed step is unsat.**
  - A `sat` result returns `identical=None` with a step-level counterexample as evidence, not `False`: the flow around the step may never produce that row.
  - The counterexample is run through both real steps before it is reported.
  - Anything that can't be encoded falls back with a reason.
  - A fuzz test matches z3's verdict against brute-force evaluation.
- **`Divergence` reuses `diff._outputs.changed`**, and revisions moved from experiments to `diff/_revisions.py`. An unknown revision now fails to resolve instead of failing at checkout.

### Experiments
- **One `run(def_or_path, ...)` and one `start()`.** `start()` returns a `RunHandle` with `.status`, `.progress`, `.cancel()` and `.result()`, wrapping the contract's `JobHandle`. `job=` left the user API, and the bridge keeps its own handle through a private path.
- **`Experiment.__init__` keeps the user arguments.** The protocol arguments move to `Experiment.from_def`, and `save()` fills in the inferred entry.
- **`decider experiment run` and `decider diff`** exit 1 unless the experiment completed, or unless there are no differences, respectively.

### Kernel functions and money (user decisions throughout)
- **`decider.ops.decimal.round(x, digits, mode)`** supports `HALF_UP`, `HALF_EVEN` and `DOWN` on the exact double, as plain Python and in kernels. The CPython-equal `round` overload now uses it. That also fixed a wrong early return for large values with digits.
- **The money-overload spike said GO, but the user replaced `Cents` with `Decimal` and `Currency`.** The spike is written up in `notes/money-overloads.md`, and its prototype is on branch `spike/money-overloads`. The design is in `notes/decimal-and-currency.md`, whose §10–11 hold the decisions:
  - An int64 mantissa, with the scale in the numba type; precision 19 or more is a build error.
  - Operators round half-even to the larger scale, as polars does. Explicit `decimal.round` keeps every mode.
  - `Decimal ± int` is legal, and comparison with a float is allowed.
  - `Currency` is scale 4, with a units string supplied by the client and used for display only.
  - Construction:
    - `Currency(major, minor)` reads `minor` in units of the scale: `Currency(2, 6)` is 2.0006 and `Currency(2, 6500)` is 2.65.
    - `minor` can't be negative; write `-Currency(0, 5)` instead.
    - Too much precision raises `ValueError`.
    - A float rounds to the scale with no warning.
  - Kernels use limb arithmetic, so no 128-bit divide (`__udivti3`) is ever emitted; an i128 divide aborts the process.
  - `Money(cents=True)` is gone; `Money()` stays as float display metadata.
- **A flow without decimals pays nothing for them.** Boxers and scales are decided per call site when the plan is built (`Call.decimals`). Before that fix, the Decimal wiring cost interpreted mode about 26 µs per record.

### Bridge and contract
- **Bridge replies are pydantic models** in `contract/_messages.py`, with the same wire shape as before. They are not exported from `decider.contract`, because of its name budget.
- **Only the bridge-reply block of `tools/decider-ui/src/model/protocol.ts` is generated**, between the `// <generated>` markers. Regenerate it with `uv run python -m decider.contract._messages`. A drift test fails when the block is stale.
- **`{"cmd":"hello"}`** returns the protocol and contract versions and the capabilities. The extension doesn't call it on connect yet.

## Shortcuts (`# ponytail:` comments added this run)

Each one is a known ceiling, with the upgrade path in the comment. The 39 `ponytail:` comments that were already in `decider/` before this run aren't listed here.

| Where | Shortcut | Ceiling | Upgrade path |
|---|---|---|---|
| `decider/ops/decimal.py:85` | `q + t` takes two roundings | off by one ulp only on a halfway case just past 2⁵³ | exact arithmetic with big integers |
| `decider/ops/decimal.py:137` | beyond 22 digits, `10**n` isn't an exact double, so rounding scales in floats | approximate for `digits > 22` | big integers |
| `decider/_engine/compile/semantics/numbers.py:34` | LLVM folds a literal `x ** 2` into `x * x` | can differ from libm `pow` by an ulp | stop the fold for literal exponents |
| `decider/diff/_smt.py:414` | table overlap is one solve per row pair | tables over 40 rows skip the pairwise check (reported under `skipped`) | one sweep per variable |
| `decider/diff/_hash.py:52` | a tree or table's reference function is skipped when it carries arrays; the arrays stand in for it | assumes the reference was built from the same arrays | hash the reference's document too |
| `decider/checks/_builtin/determinism.py:142` | DCD003: a name rebound anywhere in the body counts as a copy | misses mutation before the rebind, and mutation through aliases (`y = x; y.append(1)`) | flow-sensitive tracking |
| `decider/checks/_builtin/numeric/floats.py:157` | DCD109: `total + step` counts as growth, `i + 1` doesn't | heuristic for int carries | flow-sensitive range check |
| `decider/checks/_builtin/data.py:162` | DCD503: a name assigned inside a branch or loop becomes unknown | stays quiet where a joined range would flag | join ranges across branches |
| `decider/_bridge/bridge.py:172` | the trace sink's status isn't reported | the bridge `status` lacks it | report it once `Session` accepts a sink |

Other heuristics with no `ponytail:` comment, documented in the checks' docstrings and the workers' reports:
- **DCD104 `division_by_variable`:** any enclosing or earlier test that mentions the divisor counts as a guard. It stays quiet rather than proving the guard right.
- **DCD101 `int64_overflow`:** int × int with no range reasoning. `sum()` of ints and running totals outside loops aren't covered.
- **DCD108 `nan_propagation`:** fires on any float compared without `isnan`/`isfinite` in the same function. It is the noisiest check, at info severity.
- **DCD102 `float_money`:** fires on every float `Money()` field. That is correct, but noisy on flows that still keep money in floats.
- **Type inference in checks** needs real annotations. Unannotated names are unknown, so the checks stay quiet on them.
- **Source checks can't see code without a file**, so GUIDE examples (run with `exec`) and notebook cells get no source findings.
- **DCD201 on `branch`** analyses only conditions defined as functions with `return`, not lambdas.
- **Tree reachability** treats every root of a multi-rule tree as reachable. It can miss a dead branch but never invents one.
- **DCD506 `constant_output`** checks outputs only. The "branch arm never taken" half isn't built.
- **DCD502 and DCD506** share one run of the flow. If that run crashes they stay quiet, since DCD501 reports the crash.
- **DCD503 `int64_range`** only sees input columns present in the frame, and it is pessimistic when the extremes never occur on the same row.
- **`checks.yaml` is found from the current directory**, as datasets are, so MCP `check_report(entry)` depends on the server's working directory.

## Smaller items left alone on purpose

Seen during the run and out of scope; none of them blocks anything.

**Notes and docs**
- `notes/jupyterlab-decider.md` describes `tools/jupyterlab-decider`, `decider_jupyter` and `decider-bridge`, none of which exist.
- `notes/datasets.md` line 14 still says `experiments.runner.scan` (lazy), which describes the old state.
- `docs/` (including `docs/examples/03_config_and_persistence.ipynb`, which uses `VersionedConfig`) describes an older version, as `CLAUDE.md` says.
- `example_projects/**/NOTES.md`, `Design.md`, `IR.md` and older notes still name `decider.engine`, `numeric` and other removed names, as history.
- `projects/loan_scoring` imports a `JsonFileConfigManager` class that no longer exists; that was already stale.
- The `tools/decider-ui/demo/vscode/datasets.ts` comment still mentions `decider.data.load()`.

**Code**
- `benchmarks/span_operations.py` imports `python_only` from `decider`, which doesn't exist. It was broken before this run. `decider/_engine/compile/units.py:417` still names `python_only` in an error hint.
- `decider/_engine/compile/units.py` is exactly 500 lines, at the limit. Split it before adding to it.
- `decider/_engine/run/engine.py` has an unused `Call` import.
- `decider/_engine/params/__init__.py` re-exports names that ruff flags as unused (F401). They are intentional.
- `Bridge.describe` and `Bridge.start` set many attributes outside `__init__`, so methods called before `describe` raise `AttributeError` (`status` works around it with `getattr`). `bridge.py` (472 lines) mixes describe, run, session and forks. The bridge is private, so this was left for its own refactor.
- `decider/diff/_worker.py` and `decider/experiments/_worker.py` repeat the same subprocess reply envelope.
- `RegistryError`'s did-you-mean list doesn't include entry-point names.
- `optimise(record=...)` can't be written in a flow file, because it takes a `TypedDict` class. `each(execution_mode=...)` was also left out, since both modes now scatter.
- Hot reload's `_key` (`_engine/debug/hot.py`) includes a composite's whole `Origin`. Changing `suppress_checks` on a composite therefore counts as an edit and swaps the node. It doesn't recompile.

**Decimal and Currency**
- Decimal steps inside trees, tables and other row nodes (`each` scatter paths included) fall back to Python.
- `float / Decimal`, `int / Decimal`, `Decimal ± float`, `//`, `%` and `**` are type errors in both Python and kernels.
- A `Currency(...)` written as a param default is a `ParamSpec` marker, so a direct Python call to the step sees the marker, not the value.
- numba's long `TypingError` text for a Decimal misuse isn't cut to one line in `_why`.
- The bridge's debug-value serialisation uses `default=str`, so a `Currency` with units prints as `R 10.0000` there.
- `run()` over 200k rows: 22.2 M rows/s for Decimal against 30.4 M for plain ints, because of the polars Decimal cast in and out. Single-record `score()`: 28.3 µs against 26.6 µs.
- `Decimal.from_raw` can't be called inside a step, because numba can't type a classmethod on a plain class. Use `Decimal(n, scale)` with a literal scale.
- `min` and `max` of two decimals return the aligned larger scale in kernels, but an operand at its own scale in Python. The values are equal.
- `_engine/types.py` finds the Decimal classes through `sys.modules`, so that `import decider` stays numba-free.

**Not built from the plan**
- `diff` tier 2 (normalised documents), tier 3 (`diff.outputs(..., domain="thresholds")`) and `first_divergent_step`. Recorded in `diff.md` → "Deferred".
- `datasets().add(...)` (user story B4). `datasets()` returns a plain dict built per call.
- Moving scoring off the event loop in `RequestHandler.process_fn`, out of scope per the handover (`thread-safety.md` → the open question).
- Phase 7 credit functions.

## Convention allowlists (awaiting the user's sign-off)

`tests/conventions_allowlists.py` is the to-do list; every entry has a reason. "Done means" asks for the user's sign-off on what remains. Not yet given.

| Allowlist | Entries | Notes |
|---|---|---|
| `NO_ALL` | 0 | every public module defines `__all__` |
| `BIG_ALL` | 3 | top-level `decider` (34 names, "the authoring vocabulary is one namespace by design"), `contract` (28), `steps/trees/schema` (the document grammar) |
| `UPWARD_IMPORT` | 2 | `__init__ -> contract` (`describe`), `_settings -> config` |
| `LAYER_CYCLE` | 2 | `_engine <-> steps`: the engine reaches `as_step`, `TableRef` and helpers lazily |
| `SHARED_STATE` | 1 | `Executable._params` writes the thread-local report |
| `KERNEL_DECORATOR` | 5 | Arrow intrinsics and scorecard (permanent); `kernel.py` (to confirm); tree walker and table matcher (rewrite as `register_jitable`) |
| `PRIVATE_IMPORT` | 49 | mostly param merging (bridge ×3, experiments), the resolved plan (`_engine.wiring.plan` and `.resolve`, 6 entries), `import_flow` (7 importers of `steps._entry`), and data and bridge helpers |

The cheapest next cuts are:
- a public home for param merging;
- a public home for the resolved plan, as a `decider.ir` name;
- a public spelling of `import_flow`, for example `load_flow` accepting `"module:qualname"`;
- rewriting the tree walker and table matcher as `register_jitable`.

## Pointers

- **Spec and decisions:** `README.md` (decisions and "As built"), `directory-structure.md`, `checks.md` (catalogue, including DCD207/208), `diff.md` ("Deferred"), `kernel-functions.md` (Decisions), `../decimal-and-currency.md` (§10–11), `../money-overloads.md`.
- **Contract changes for VS Code:** `notes/spec-amendments.md`, entries 2026-10-05 and 2026-10-06.
- **Benchmarks:**
  - `benchmarks/score_modes.py`: single-record latency per mode, public API only, so it runs on any commit.
  - `benchmarks/serving_latency.py`: fixed in phase 0 to match the current runner internals.
  - `benchmarks/decimal_steps.py`: Decimal against int.
