# Orchestrator handover

## How to use this file

1. Open a new Claude Code session in the repo root, on branch
   `feature/decider-v2`, using an Opus model.
2. Paste the kickoff block below.
3. Everything after the kickoff block is the handover. The orchestrator reads it
   as its first action.

---

### Kickoff (paste this)

```
You are the orchestrator for consolidating decider_old and decider2 into the new
numba-first `decider` package. Read Prompt.md in the repo root in full before
doing anything else, then IR.md, Design.md and Plan.md in that order. IR.md is
the signed-off contract. Plan.md is your task list.

Your job is to dispatch the tasks in Plan.md to subagents at the model tier each
task names, review and merge their work into feature/decider-v2, keep
notes/progress.md current, and stop at the checkpoints Prompt.md lists. Do not
write large amounts of code yourself; small merge and integration fixes are fine.

Start with Phase 0 (T0.1, T0.2 and T0.4 in parallel). Before dispatching
anything, reply with a short summary: what you understood, the first batch of
tasks you will dispatch, and anything in the documents that looks inconsistent.
```

---

## 1. Mission

Build `decider/` as one package that:

1. **makes steps from plain functions** with almost no ceremony;
2. **runs steps directly on data** in three modes: `interpreted` (plain Python),
   `stepped` (numba per node, Python driver) and `fused` (numba kernels);
3. **supports stepping through a run**: pause, inspect, set a value, resume, and
   stream JSON events that a UI can consume later.

It keeps decider_old's configurable, serialisable design (pydantic
`ConfigurableStep`s such as decision trees) on top of decider2's compiled
engine.

**Done means** every task in `Plan.md` is complete, the full suite passes on
`feature/decider-v2`, and `decider_old/` and `decider2/src` are deleted (T7.1)
with their behaviour covered by tests under `tests/`.

---

## 2. Read these, in this order

| File | What it is | How to treat it |
|---|---|---|
| `IR.md` | Steps (authoring) and IR (execution): classes, constructors, params, validation, paths and origins, a worked example, acceptance tests | **The contract.** Implement it exactly. If it has to change, stop and ask the user |
| `Design.md` | Goals, decisions D1–D13, architecture diagrams, engine and session design, registry, serving, target layout, measured facts, open questions with defaults | Agreed design. Where `Design.md` §10 gives a default, use it |
| `Plan.md` | Conventions, model tiers, tasks T0.1–T7.1 with dependencies and pass conditions, the critical path | Your task list. Keep its task rows as they are, and record progress in `notes/progress.md` |
| `notes/` | Decision records (filled in by T0.4 and later tasks) | Add to it when a decision of the "we tried X, it failed, so Y" kind is made |

---

## 3. The repo as it stands

- **Branch:** `feature/decider-v2`. `main` is the PR base, but nothing is merged
  to `main` in this effort.
- **`decider/`, the target.** It is a skeleton: `registry/` (subclasses register
  on a `Literal` tag), `serializable/`, `serving/`, `settings.py`,
  `exceptions.py`, `modules/core.py`.
  - `import decider` works.
  - `decider.serving` and `decider.settings` import modules that don't exist
    yet (`decider.config`, `GraphModule`, `decider.executor`). T0.2 fixes that.
- **`decider_old/`, the reference for the configurable design.** Its imports
  still say `decider.*`, so it **does not run**. Read it; don't execute it.
  Useful parts:
  - `modules/rules/tree/v3` (tree semantics);
  - `modules/credit/scorecard`;
  - `config/` (the versioned store);
  - `_ext.py` (the old registry);
  - `cli/`.
- **`decider2/`, the reference engine.** It is a separate `src`-layout package.
  - **Run it with `PYTHONPATH=decider2/src`.** Otherwise, from the repo root,
    `import decider2` resolves to an empty namespace package
    (`decider2.__file__ is None`).
  - Its tests are the behavioural oracle, for example:
    `PYTHONPATH=decider2/src uv run pytest decider2/tests -q`.
  - Numba compile time makes the full suite slow, so run subsets per task.
  - The `_arrow` shim needs a C toolchain. decider2 tests skip rather than fail
    without one; check the skips before trusting a green run.
- **decider2's docs overstate what exists.** `decider2/docs/*.md` describe
  several features that were never built: `debug()`, `observe/`, `fuse()`,
  `parallel()`, `Vocabulary`, `round_half_up`, pydantic `Module`, `frame.Join`,
  `breaks_lineage`. When the docs and the code or tests disagree, **the code and
  tests are the truth**. The docs are useful for *why* (their measurements), not
  for *what exists*.
- **Top-level `tests/`** currently holds decider_old tests against modules that
  don't exist. T0.2 moves them to `tests/_legacy/` and excludes them from
  collection.
- `experimentation/` holds decider2's probes. It is reference only; never import
  from it.
- **Tooling:**
  - `uv` for everything (`uv run pytest`, `uv run python`);
  - Python 3.10 or later;
  - `pyproject.toml` builds the `decider` wheel with hatchling. Markdown files
    at the root are not shipped.

---

## 4. Your role and how to dispatch

- **Dispatch each task in `Plan.md` to a subagent** with the Agent tool, setting
  `model` to the task's tier: `haiku`, `sonnet` or `opus`.
- **Parallel tasks run in their own worktree and branch** (`isolation:
  "worktree"`). Tasks on the critical path may run in the main tree if nothing
  else is running.
- **Respect `Depends on`.** The critical path is
  `T0.2 → T1.2 → T1.1 → T1.3 → T1.4 → T1.5 → T3.3 → T4.2 / T5.1`. Keep the
  side tasks moving alongside it: T0.1, T0.4, T2.1, T3.1 and T4.1 are all
  independent early.
- **You may make small fixes yourself:** merge conflicts, a missing import, a
  test path. Anything bigger goes back to a subagent.

### Brief template (every task, every time)

```
FIRST: invoke the Skill tool with skill "ponytail:ponytail" and args "ultra", and
apply it throughout: the simplest code that passes the tests, no speculative
abstractions, the standard library before custom code. Ponytail never overrides
the acceptance tests or IR.md: if simplifying would change tested behaviour, keep
the behaviour and leave a `ponytail:` comment.

Task <ID>: <title from Plan.md>
Branch / worktree: <name>
Read: IR.md §<...>, Design.md §<...>, CLAUDE.md (conventions)
Port from: <exact source files>
Create: <exact target files>
Tests to port: <exact test files>, to tests/<area>/
Done when: <the task's pass condition from Plan.md, verbatim>

Conventions (non-negotiable):
- Docstrings only on public API, written for library users (a summary, arguments
  where not obvious, optionally a short example). No docstrings on private
  helpers unless the behaviour is non-obvious. Comments explain why, never what.
- Never reference docs, sections, experiments, stages or review findings in code
  (no "doc 03", no "§4.2", no "EXPERIMENTS").
- One purpose per file, under 500 lines. Split into a package instead of growing
  a file.
- A critical "we chose X over Y because Y failed" rationale goes in
  notes/<topic>.md, not in code.
- Do not modify IR.md, Design.md or Plan.md. If the spec is wrong or ambiguous,
  stop and report it instead of guessing.

Finish with: the files changed, the test command and its output, anything
skipped or deviating from the spec, and any `ponytail:` comments you left.
Commit once, on your branch, with a message saying what the task delivered.
```

---

## 5. Review and merge loop (per task)

1. **Read the report.** Anything skipped or deviating from the spec is either
   fixed or raised with the user; it is never silently accepted.
2. **Run the task's tests, the conventions test (`tests/test_conventions.py`,
   from T0.1) and the whole `tests/` suite** on the task branch.
3. **Check the conventions directly:**
   - file sizes;
   - `grep -rnE "doc [0-9]{2}|§|EXPERIMENTS" decider/` returns nothing;
   - docstrings are user-facing.
4. **Opus tasks:** also run the `code-review` skill against `IR.md` and
   `Design.md` before merging.
5. **Merge** into `feature/decider-v2`, then run the **post-merge step:** the
   full suite plus the conventions test on the merged result, fixing joins
   between tasks.
6. **Append to `notes/progress.md`:** the date, task, model, commit, outcome,
   test counts, and follow-ups.

---

## 6. Git rules

- **Everything lands on `feature/decider-v2`.** Per-task branches are fine;
  delete them after merging.
- **One commit per task,** plus merge commits. Use the commit attribution lines
  your session provides.
- **Never push, open a PR, force-push or rewrite history without asking the
  user.** Local commits and merges are fine.
- **Don't delete `decider_old/` or `decider2/` before T7.1.** They are the
  reference throughout.
- **Don't edit files under `decider2/` or `decider_old/`.** Port from them into
  `decider/` and `tests/`.

---

## 7. When to stop and ask the user

- After **Phase 0**: a short status update.
- After **Phase 1**: the user reviews the public API in practice. Show:
  - the `IR.md` §7 example running;
  - a session doing break → set → resume;
  - the event log as JSON.
- After **Phase 3**: show the performance numbers against decider2 (batch
  throughput and `score()` latency) and the equivalence results.
- **Whenever `IR.md` would have to change**, or a decider2 test can't be ported
  faithfully under the new design. Explain the conflict and propose options; do
  not decide alone.
- **Before T7.1**, the deletion.
- Before anything outward-facing: pushing, PRs, anything touching `main`.

---

## 8. Decisions already made

Don't reopen these. The full list is `Design.md` §3; defaults for the remaining
open questions are in `Design.md` §10.

- Pipeline structure lives in Python. Config means params documents plus
  `ConfigurableStep` documents.
- The authoring layer is `Step`, with `FunctionStep`, `FrameStep`, `DagStep`,
  `SequentialStep`, `BranchStep`, `LoopStep` and `ConfigurableStep`.
- Every step emits its expanded IR via `to_ir(ctx)`.
- The IR is closed: `CallNode` (`kind` is `scalar`, `row` or `frame`),
  `SequenceNode`, `BranchNode`, `LoopNode`. Nothing is called "module".
- Every IR node has an `Origin`: a path plus an import-path source. Origins
  never enter compile cache keys.
- Numba-first. The polars expression tier is dropped; polars is used only in
  frame steps.
- Missing inputs are an error by default (`missing_as()`, `T | None`).
- Params are nested by path, with `param(shared_key=...)`. Only shared types are
  checked globally. Validation is per node and cached per params document, with
  `eager` or `lazy` mode; lazy validates on first hit. Fused kernels suspend and
  resume to validate; validating everything up front is the accepted fallback.
- Config values are literals or param references (`Value[T]`, `TableValue`).
  Changing table rows never recompiles.
- `ConfigurableStep` type tags default to the import path, with an optional
  alias.
- Runners are generators, so debugging and production share one code path.

---

## 9. Known traps

- **Numba cache:** generated code must live in real files, not `exec`, and be
  byte-identical between runs. Key compiled code by content, never by path or
  name.
- **Serving kernels are `nogil=True`,** and `fastmath` is off by default.
- **`round()` differs** between CPython and numba, and int64 wraps silently.
  Money is int64 cents, and running totals accumulate in float64.
- **Nested polymorphic pydantic fields need `SerializeAsAny`,** or the
  subclass's fields are silently dropped on dump.
- **Single-record calls take a dict,** not keyword arguments. Cache converted
  params bundles, because conversion costs more than validation.
- **Only `NumbaError` triggers the Python fallback.** Runtime errors always
  propagate.
- **Worktrees and numba:** each worktree has its own `__pycache__` and numba
  cache, so the first test run in a fresh worktree is slow. That is expected.
- **decider2 test files are long,** and several test decider2-specific
  behaviour that the new design removes: `not_applicable_as`, pipeline-level
  refer routing, the reserved `shared` bundle, `Interior`-style packed steps.
  Port the *behaviour that still applies*. List every dropped test in the task
  report, with the reason.

---

## 10. Definition of done, per phase

| Phase | Done when |
|---|---|
| 0 | `import decider` and `import decider.serving` work; `uv run pytest` is green with the legacy tests excluded; the conventions test exists; `notes/` has its first records |
| 1 | The `IR.md` §10 acceptance tests pass; the §7 example runs in interpreted mode; sessions work; events round-trip as JSON |
| 2 | Configurable steps compose and round-trip; the registry changes are done; the usability review is triaged |
| 3 | All three modes agree on the ported decider2 suites; `score()` meets decider2's measured latency; editing one step recompiles only its kernel |
| 4 | Trees and tables are configurable steps with Python reference walkers that agree with the compiled ones; the scorecard is ported |
| 5 | Branch and loop step into arms and iterations; multi-step arms and multiple carries work |
| 6 | Config store, serving, CLI, the websocket adapter and `session.replace`/`delete` |
| 7 | The old packages are deleted and the full suite passes |
