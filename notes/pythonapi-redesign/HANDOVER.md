Executed 2026-10-05/06; see README.md → As built.

# Handover: orchestrating the Python API redesign

Written 2026-10-05 for an orchestrator agent. You don't write most of the code yourself. You split the work into packages, brief workers, check each phase against its gate, and stop to ask the user when something is undecided.

The design is finished and agreed with the user. Your job is to carry it out, not redesign it. Where these files disagree with each other, the order of authority is: the user's decisions in `README.md`, then `directory-structure.md`, then the other notes. Where the code disagrees with a note, the code shows the current state and the note shows the target.

## Read first, in this order

| File | Why you need it |
|---|---|
| `CLAUDE.md` (repo root) | Project rules. Every worker follows them. |
| `notes/pythonapi-redesign/README.md` | Problem, findings, bugs, **the user's decisions** |
| `notes/pythonapi-redesign/directory-structure.md` | **The plan:** every file's destination, deletions, changes outside `decider/`, phases, convention tests |
| `notes/pythonapi-redesign/thread-safety.md` | The serving-path contract and the fix list (phase 0) |
| `notes/pythonapi-redesign/checks.md` | The `decider.checks` design, `checks.yaml`, `suppress_checks`, the catalogue (phases 5–6) |
| `notes/pythonapi-redesign/diff.md` | The `decider.diff` design and proof tiers, including z3 (phases 5–6) |
| `notes/pythonapi-redesign/layout-and-visibility.md` | Reasons for each move or rename, and name clashes. Use it when a worker asks why. |
| `notes/pythonapi-redesign/kernel-functions.md` | Where kernel functions go (semantics vs `decider.ops`), their contract, the cache-salt fix (phases 1, 3, 5, 6) |
| `notes/pythonapi-redesign/credit-functions.md` | Where credit functions will fit (phase 7). The refactor leaves room for them; they don't move during it. |
| `notes/pythonapi-redesign/extensibility-and-coupling.md` | Entry points, closed lists, wrong-way imports, the private-import list |
| `notes/pythonapi-redesign/user-stories.md` | The "After" code is the acceptance target for the whole refactor |

Background (only when relevant): `notes/racecondition_investigation_results.md`, `notes/money-rounding-and-overflow.md`, `notes/numba-compile-cache.md`, `notes/origins-out-of-cache-keys.md`, `notes/serving-latency.md`, `notes/vscode-extension-testing.md`, `notes/datasets.md`, `notes/spec-amendments.md`.

## State at handover

- **Branch:** `feature/eachIRNode` at `d6a51e4`. The tree is clean apart from these notes, which are uncommitted.
- **Other worktrees exist:** `dsp-decision-engine-spike-experiments`, `.claude/worktrees/agent-a4d4a3a4a41412b14`, and a prunable temp one. **Leave them alone.**
- **`uv`:** it may not be on `PATH` in agent shells. Use `~/.local/bin/uv`.
- **Test baseline:** four failures before any change, all environmental (see Baseline below). Don't let workers "fix" them inside the refactor.
- **Nothing in the redesign is implemented yet.**

### Baseline

Full suite at `d6a51e4`, run on 2026-10-05 with `~/.local/bin/uv run pytest -q`: **2410 passed, 4 failed, 1 skipped, 1 xfailed** in 1m46s (warm numba cache). All four failures were there before any change:

| Test | Cause |
|---|---|
| `tests/compile/test_compile_cache.py::test_the_cpu_target_is_recorded` | LLVM reports an empty CPU feature string on apple-m2 / Darwin 27 |
| `tests/mcp/test_editor.py::test_bridge_routes_each_workspace_to_its_own_window`, `::test_bridge_returns_the_editors_selection`, `::test_bridge_rejects_a_wrong_token` | `OSError: AF_UNIX path too long`. The test's socket lives under the long per-session temp directory, which goes past macOS's 104-byte limit. It may pass with a short `TMPDIR`. |

Treat **these four failures, and only these**, as the baseline. Any other failure in a gate belongs to the change being gated. Raise the four with the user separately; they aren't part of this refactor.

## Decisions: settled, don't reopen

Details are in `README.md` → Decisions.

1. **No compatibility shims.** Names move, and every caller changes in the same commit, including `tools/vscode-decider/src/{python,debugging,locate}.ts`, `GUIDE.md` and `tests/`.
2. **`engine/` → `_engine/` in one `git mv`**, once the `ir` / `trace` / `debug` facades exist (and in move order, step 1 of phase 3).
3. **The check policy file is `checks.yaml` (or `.json`) at the project root**, found the way datasets are (setting `checks_path`). It is never in `configs/`.
4. **Suppression is `suppress_checks=[alias | code]`** on every step constructor, as a `ConfigurableStep` field and as a flow-file key. It is stored on `Origin`. There are no inline comments and no config suppressions.
5. **z3 is in**, as the `decider[prove]` extra.
6. **`Table` → `ParamTable`**, with no alias. `optimise` keeps its name.
7. **Delete `data/scope.py` and `data.identity.resolve_key` / `record_ref`.** Re-grep for callers right before deleting. The user works in `data/`.
8. **One rule for privacy:** a leading underscore in any path segment means private. Every public module defines `__all__`.
9. **Thread safety:** everything on the serving path is safe for concurrent calls, with **no API or usability change**. `exe.report` becomes per thread.

## Ask the user before you start

These weren't decided. Ask them in one batch (`AskUserQuestion`), and don't guess:

1. **Branch and commits.** The recommendation is a new branch `feature/python-api-redesign` off `d6a51e4`, with one commit per work package and a commit per move in phase 3. Agents commit only with permission, so get it for the whole run up front, and ask whether the notes should be committed first.
2. **`suppress_checks` spelling.** The user wrote `suppress_check=`; the plan uses the plural because it takes a list. Confirm which.
3. **Underscore the implementation files inside public packages** (`steps/_base.py`, `checks/_run.py`, …; phase 3 step 10). The user didn't comment on it, and it is the largest churn. Confirm before step 10.
4. **Deleting `serving/session_ws.py` and `config.VersionedConfig`.** Neither has a non-test caller. The user hasn't confirmed either.
5. **The open questions in `kernel-functions.md` and `credit-functions.md`.** Settled already: the `decider.ops` domains are `credit` / `strings` / `decimal` / `general`, and rounding has the modes `HALF_UP` / `HALF_EVEN` / `DOWN`. Still open: decimal scale and overflow, `round(2.675, 2)` semantics, `Cents + int`, and the fitting package's name. `ops.decimal.round` doesn't wait for them; scaled decimals and the money overloads do.
6. **Moving scoring off the event loop** in `RequestHandler.process_fn` (`thread-safety.md`, the open question). **This is out of scope** unless the user asks for it. If they do, measure it with `benchmarks/serving_latency.py` first.

## Rules every worker brief must include

Copy these into every brief:

1. **First instruction:** "Invoke Skill `ponytail:ponytail` with args `ultra` before doing anything else, and apply it throughout." This is a standing user rule for any worker that moves or ports code. If a simplification would change behaviour that a test pins down, keep the behaviour and leave a `# ponytail: <what could be simpler>` comment.
2. **Read `CLAUDE.md`.** In particular:
   - docstrings only on public API, written as user docs;
   - no references in code to docs, notes, section numbers or reviews;
   - rationale goes in `notes/`;
   - files under 500 lines;
   - `uv run`;
   - the numba traps (content-keyed cache, `nogil` serving kernels, no bare `round()`, `SerializeAsAny` for polymorphic fields).
3. **Prefer extending an existing constructor over adding a new primitive** (a standing user preference).
4. **The scope is the files the work package names.** Don't improve adjacent code. Mention unrelated dead code; don't delete it.
5. **Every `GUIDE.md` Python example runs in `tests/test_guide.py`.** If you move a name, fix the GUIDE in the same change.
6. **If the spec is wrong or ambiguous, stop and report.** Don't guess.
7. **The report back lists:**
   - what changed;
   - the tests run, with their results;
   - tests dropped, and why;
   - `# ponytail:` comments added;
   - anything left open.

**Model per work package** (standing user rule: easy → Haiku, moderate → Sonnet, critical → Opus):
- The engine, compiler, runners, debugger, core API and anything that claims correctness, such as proofs, get **Opus**.
- Well-specified features and ports get **Sonnet**.
- Mechanical renames and greps get **Haiku**.

The tiers are set in the table below.

Isolation: use `isolation: "worktree"` for work packages that run in parallel. Each worktree has its own numba cache, so its first test run is slow. That is expected, not a hang.

## Work packages

Phases follow `directory-structure.md` → Migration phases. A phase starts only when the previous phase's gate is green. **Par** means the package can run at the same time as the others marked Par in its phase.

### Phase 0 — bugs and serving-path races (on the current layout, before any move)

| WP | Scope | Spec | Model | Par |
|---|---|---|---|---|
| 0.1 | `checks.compare` ignores consts and param defaults. Minimal fix: include `consts`, param defaults and param-table content in `_signature`, or drop the structure short-circuit. Add a regression test: factor 2.0 vs 3.0 must not be equivalent. The real replacement comes in WP 5.2. | README → Bugs | Sonnet | Par |
| 0.2 | Experiments: `validate` reports a revision scenario with no `entry`; `run(def_)` resolves `input.path` / `input.dataset` against the definition file the way `Experiment.load` does | README → Bugs | Sonnet | Par |
| 0.3 | Serving races #1, #2, #3, #4 and #7, plus the `_engine/concurrency.py` helper (create it at `engine/concurrency.py` for now) and `tests/serving/test_concurrency.py` with cases (a)–(h) | `thread-safety.md` | **Opus** | Par |
| 0.4 | Races #5, #6, #9, #10, using the WP 0.3 helper | `thread-safety.md` | Sonnet | after 0.3 |
| 0.5 | Refresh the stale `notes/debug-bridge-relocation.md` | README → Bugs | Haiku | Par |

Leave the `numeric` check's money bugs alone here. `numeric` is replaced by DCD101–105 in phase 6, so don't fix it twice.

**Gate:**
- Full pytest is green apart from the known baseline failure.
- The concurrency stress test passes on 5 consecutive runs.
- `uv run python benchmarks/serving_latency.py` is within noise of `main`. This is the "no usability or latency change" check for 0.3 and 0.4.

### Phase 1 — guard rails

| WP | Scope | Spec | Model |
|---|---|---|---|
| 1.1 | The six convention tests in `tests/test_conventions.py`: public `__all__`, no cross-package private imports, the layer map plus a cycle check, the `__all__` budget, no per-call `self` mutation in runners, overloads only in the kernel library with `SALT` covering it. Each has an **allowlist of today's violations, with reasons**. The allowlist is the to-do list for later phases. | `directory-structure.md` → Convention tests; `thread-safety.md` → Keeping it fixed | Sonnet |

**Gate:** the tests pass with their allowlists. Report the allowlist sizes to the user, because they are the progress metric from here on.

### Phase 2 — deletions

| WP | Scope | Model |
|---|---|---|
| 2.1 | Every row of `directory-structure.md` → Deleted that can be done without a move: `data/scope.py`, `identity.resolve_key` / `record_ref`, `contract.mcp_schema`, the `experiments.scan` re-export, and `session_ws.py` / `VersionedConfig` only if the user said yes. Re-grep first. Update the GUIDE (815–825), the tests and `notes/datasets.md`. | Haiku |

`data/preflight.py` and `checks/equivalence.py` are **not** deleted here. Their replacements land in phase 5.

**Gate:** full pytest green, and the allowlists shrink.

### Phase 3 — mechanical moves (sequential, one commit each, no logic changes)

Run this phase **in one worktree, in order**. Each move rewrites imports across `decider/`, `tests/`, the GUIDE and benchmarks, so parallel moves would conflict at every step.

| Step | Move | Model | Extra check |
|---|---|---|---|
| 1 | `engine/` → `_engine/`. **First add the facades** `decider/ir.py`, `trace.py` and `debug.py` (re-exports only). Then `git mv`, rewrite imports, and update the engine rule in `CLAUDE.md`. | Sonnet | expect one cold numba recompile; `test_imports.py` updates its pinned paths |
| 2 | `lifecycle/` → `contract/_manifest.py` + `_jobs.py`; rename `json_schema` → `manifest_schema` | Haiku | |
| 3 | `flowfile/` → `steps/flowfile/`; `_parse` / `_components_index` → public `read_flow` / `components`; lazy import of `expr_condition` | Sonnet | `import decider` still doesn't import trees |
| 4 | `check/` + `checks/` → one `checks/` (**merge only**; the redesign is WP 5.1) | Haiku | |
| 5 | `debug_bridge/` → `_bridge/`, plus `contract/describing.py`, `engine/debug/controls.py` and `lineage.py`; the three TS paths; fix `mcp/tools.py` reaching into bridge internals | Sonnet | `pnpm test` in `tools/vscode-decider`, plus the in-editor tests per `notes/vscode-extension-testing.md` |
| 6 | `ids.py` → `_tools/ids.py`; `serializable/` → `_polars_docs/` (`DataFrame` → `FrameDoc`); `registry/resolve.py` → `_suggest.py`; `settings.py` → `_settings.py` with lazy `get_settings()` and a `checks_path` setting | Haiku | `decider serve` / CLI tests |
| 7 | `fields.py` + the public half of `types.py` → `_annotations.py`; the helpers → `_engine/types.py` | Haiku | |
| 8 | Split `exceptions.py`: HTTP errors → `serving/_errors.py`, Arrow errors → `_engine/boundary/errors.py` | Haiku | `tests/test_exceptions.py` |
| 9a | `engine/compile/cpython.py` → `_engine/compile/semantics/numbers.py`; the operations half of `span.py` → `semantics/strings.py`; `SALT` computed by walking `semantics/` (and `decider/ops/` once it exists) | Sonnet | the numba cache rebuilds once; `tests/compile/test_compile_cache.py` |
| 9 | `Table` → `ParamTable`; `wiring.Version` → `NodeVersion`; debug events `Warning` / `Error` / `Summary` → `WarningEvent` / `ErrorEvent` / `RunSummary` | Sonnet | `protocol.ts` matches on `type` strings; confirm |
| 10 | Underscore the implementation files in public packages (**only if the user confirmed question 3**) | Haiku | |

**Gate after every step:**
- full pytest green;
- convention allowlists are not growing;
- `test_guide` green.

**Gate after step 5:** the extension tests pass.

### Phase 4 — public surface

| WP | Scope | Model | Par |
|---|---|---|---|
| 4.1 | Top-level `decider/__init__.py`, exactly as listed in `directory-structure.md` → Top-level: lazy `TreeConfig` / `DecisionTableConfig` / `ScorecardConfig` via module `__getattr__`; `Step.bind(mode)`; `Step.session -> Session` typed; `parameters().validate(doc)`; a top-level docstring listing each companion package in one line | **Opus** | Par |
| 4.2 | Trim `__all__` in `steps` (+ trees, tables, scorecard), `experiments` (into `.metrics`, `.plots`, `.report`, `.callbacks`; `altair` → `require_altair` in plots), `data` (`load` → `read`, drop `job=`), `mcp`, `contract`, `debug`. One worker per package is fine. | Sonnet | Par, but split by package |
| 4.3 | Rewrite the GUIDE's imports and the user stories' "After" code where it already works | Sonnet | after 4.1 and 4.2 |

**Gate:**
- the private-import and `__all__` allowlists are empty, or every remaining entry has a reason the user accepted;
- `test_guide` green;
- `import decider` doesn't import numba.

### Phase 5 — new features (parallel worktrees, with explicit file ownership)

| WP | Scope | Owns | Model |
|---|---|---|---|
| 5.1a | `suppress_checks=` on every constructor, the `ConfigurableStep` field, the flow-file key, stored on `Origin` (must not enter cache keys) | `steps/`, `_engine/ir/origin.py`, `steps/flowfile/` | **Opus** |
| 5.1b | The `decider.checks` redesign: `Check` registry root, `CheckContext`, the `Finding` standard, `checks.yaml` discovery, `@check`, text and SARIF, `decider check`, the MCP `check_report` / `checks_list`, porting the 4 existing checks, the probe → `runtime` checks (DCD501 / DCD505), `preflight` → DCD304 / DCD504 (cost → `experiments/_cost.py`), the `ConfigurableStep.check(ctx)` hook | `checks/`, `cli/_check.py`, `mcp/_tools.py` (check tools only), `data/preflight.py` removal | Sonnet; starts after 5.1a merges |
| 5.2 | `decider.diff`: `structure`, `outputs` (taking over `Divergence._changed`), `revisions` (from `experiments/revision.py`), `prove` tier 1 (IR content hash on `fingerprint`); removing `checks.compare`; `decider diff` | `diff/`, `experiments/metrics/_compare.py`, `experiments/revision.py`, `cli/_diff.py` | **Opus** (proof correctness) |
| 5.3 | Experiments: one `run(def_or_path, ...)`, `start()` returning a handle, `entry` inferred, `Experiment.from_def`, `decider experiment run` with exit codes; rename `load_flow` → private `_entry.import_flow` | `experiments/` except metrics and revision; `cli/_experiment.py` | Sonnet |
| 5.4 | Registry entry-point discovery per root; one lazy-table convention; a `COMPOSITES` list shared by `ids` and flow files (flow files gain `dag`, `each`, `optimise`) | `registry/`, `_tools/ids.py`, `steps/flowfile/` (after 5.1a) | Sonnet |
| 5.5 | Bridge `hello` handshake, `contract/_messages.py` reply models, generated `protocol.ts` plus a drift test | `contract/_messages.py`, `_bridge/`, `tools/decider-ui/src/model/protocol.ts` | Sonnet |
| 5.7 | `decider/ops/` skeleton plus `decimal.round` with `Rounding.HALF_UP` / `HALF_EVEN` / `DOWN` (`@register_jitable`, every-mode test over `.xx5` ties and negatives); the `fingerprint` fix so `module.attr` calls are hashed by content, with a cache-miss test | `decider/ops/`, `_engine/compile/fingerprint.py`, `semantics/__init__.py` (salt) | **Opus** (cache correctness) |
| 5.8 | **Spike:** money overloads (`Cents` / `CentsType`). Measure `score()` cost, fused and packed typing, params bundles, cross-mode ties; write `notes/money-overloads.md` with a go or no-go | spike branch only | **Opus** |
| 5.6 | `fastmcp` → `mcp` extra; `prove` extra (`z3-solver`) | `pyproject.toml` | Haiku |

**Conflict rules:**
- 5.1b and 5.2 both start from `checks/equivalence.py`. 5.2 owns `compare` / `Equivalence`; 5.1b owns the probe.
- Merge 5.2 first, then rebase 5.1b.

**Gate:**
- every "After" snippet in `user-stories.md` for R1–R7, X2–X3, E3–E5 and I1 runs, and is in the GUIDE (and so in `test_guide`);
- the extension tests pass after 5.5.

### Phase 6 — catalogue and proof tiers

| WP | Scope | Model | Par |
|---|---|---|---|
| 6.1 | DCD101–105 (money safety, replacing `numeric`), with fix hints pointing at `decider.ops.decimal` (needs WP 5.7) | Sonnet | first |
| 6.m | Money overloads plus `Money(cents=True, rounding=...)`, **only if** spike 5.8 said go | Opus | after 5.8 |
| 6.d | `decider.ops.decimal`, **only after** the user answers the decimal questions and a short design note exists | Opus | Par |
| 6.2+ | The other families in `checks.md`, one worker per family file (`structure`, `params`, `ids`, `data`, `runtime`) | Sonnet | Par |
| 6.x | z3 tier 4 in `diff/_smt.py`, plus the satisfiability checks for trees and tables (DCD201, table overlap and gaps) | **Opus** | Par |
| 6.y | `testing.corpus`: `.xx5` ties, large-principal rows, an empty result for flows with no inputs | Haiku | Par |

**Gate:** every check has a test where it fires and a test where it stays quiet. `checks.catalogue()` lists them all.

### Phase 7: credit functions (after the refactor; listed so the layout leaves room)

See `credit-functions.md`. Work packages 7.1–7.4 cover `decider.ops.credit`, the credit metrics and drift check, `decider.credit` with `ScorecardConfig.fit`, and the `decider[credit]` extra. **The user picks which functions move**; don't port anything from the credit library unasked, and copy no proprietary details into this repo's notes.

## Done means

- The full suite is green, apart from failures the user has accepted. That includes the concurrency stress test, the extension tests and `test_guide`.
- The convention allowlists are empty, or every remaining entry has a reason the user signed off.
- The "After" code in `user-stories.md` is real and documented in the GUIDE.
- Serving latency is within noise of the baseline.
- These notes are updated:
  - `notes/README.md` index;
  - `CLAUDE.md` Layout and engine rule (pointing at `directory-structure.md`);
  - `notes/spec-amendments.md` (VS Code decision 13 now says `checks.run`);
  - this folder's `README.md` status line, marked "implemented" with the commits.
- A final `ponytail:ponytail-debt` run lists every `# ponytail:` comment added along the way, for the user.

## Traps

- **Numba cache per worktree.** The first run in a fresh worktree is slow, and so is the first run after the `_engine` rename. Neither is a hang.
- **Lazy imports keep `import decider` light.** `TreeConfig` and friends load through module `__getattr__`. Don't import trees eagerly anywhere (the flowfile move, WP 3.3).
- **Tests may import private modules.** The convention tests cover `decider/` code, not `tests/`.
- **Things outside `decider/` that name Python modules:** the three TS files, `pyproject.toml` (`decider.cli:cli`), the CLI string `"decider.serving.servers.starlette:create_app"` (and sanic), and `cli/templates/` (`RequestHandler`).
- **Every GUIDE example runs.** A rename that misses the GUIDE fails `test_guide`, not some docs check.
- **`SerializeAsAny`.** Polymorphic registry fields (Check configs, report blocks) silently drop subclass fields without it.
- **Thread-safety fixes must not add locks to the warm path.** If a worker's fix takes a lock in `score()`, send it back.

## Suggested skills

| Skill | Who | When |
|---|---|---|
| `ponytail:ponytail` (args `ultra`) | **every worker** | first action in every brief |
| `code-review` | orchestrator | at each phase gate, against the merge-base of that phase |
| `simplify` | orchestrator or worker | after phases 4 and 5, on the phase diff |
| `diagnosing-bugs` | the WP 0.3 / 0.4 worker | if a race won't reproduce, or the stress test flakes |
| `codebase-design` | the WP 5.1b and 5.2 workers | when shaping the `Check` / `CheckContext` and `diff` interfaces |
| `ponytail:ponytail-debt` | orchestrator | at the end, to list the `# ponytail:` shortcuts |
| `workflow-authoring` | orchestrator | **only if the user explicitly asks** for a Workflow run; otherwise use the Agent tool |
