# Handover

How to work on this spec if you were not part of the design discussion: what to read first, the repo's
rules, how to run things, what "done" means per phase, what a reviewer checks, and when to stop and ask.

## Read first (about forty minutes)

1. [README.md](README.md): what is proposed and what has been decided.
2. [02-target-architecture.md](02-target-architecture.md): the design. Keep the feature matrix open.
3. [07-implementation-plan.md](07-implementation-plan.md): the phase you are on, with its exact behaviour.
4. For P4, [09-polars-host-design.md](09-polars-host-design.md) end to end, then run its spike once.
5. [05-semantics-across-engines.md](05-semantics-across-engines.md) before writing tests.
6. [01-current-architecture.md](01-current-architecture.md) when you need to know where something lives.
7. [03-design-choices.md](03-design-choices.md) and [04-tradeoffs-and-risks.md](04-tradeoffs-and-risks.md)
   when you find yourself asking "why not just...". The answer is usually there; if it is not, that is a
   question for the owner, not a decision to take alone.

Then, in the repo: `CLAUDE.md`, `docs/design_decisions/decisions/engine.md` and `api-layout.md`, and
`uv run decider guide`.

## Rules of this repo that bite here

- `uv` for everything: `uv run pytest`, `uv run python`. Python 3.10+.
- Tests live in `tests/<area>/`; names state behaviour (`test_name_overwritten_after_a_frame_step_is_dropped_first`,
  never `test_case_3`).
- Docstrings only on public API, written as user docs (summary, non-obvious arguments, a short example).
  Private helpers: no docstring unless the behaviour is non-obvious. Inline comments say why, never what.
- Never reference these docs, section numbers, experiments or review findings in code. Rationale goes into
  `docs/design_decisions/decisions/<topic>.md`; a measurement into `docs/design_decisions/experiments/`.
- One purpose per file, under 500 lines; a file that grows past it becomes a package.
  `uv run pytest tests/test_conventions.py -q` enforces the limit, the banned references, `__all__`, the
  layer map, the no-per-call-state rule for runners and executables, and the allowlists (a stale allowlist
  entry fails the test, so remove entries you make unnecessary).
- Layers import downward only (`docs/architecture/package-layout.md`). `_settings.py` is L0 and must not
  import `_engine`; that is why engine settings are resolved in `serving/`. Spark code never lives under
  `decider/`.
- The simplest code that passes the tests; no speculative abstractions; stdlib first. If a simplification
  would change behaviour a test pins, keep the behaviour and leave a `# ponytail: <what could be simpler>`
  comment.
- Every Python example in `decider/GUIDE.md` runs in `tests/test_guide.py`. Change the guide and the tests
  together.
- If the docs and the code disagree, or the intended behaviour is ambiguous, stop and report. Do not guess.
- A moved or renamed public name changes its callers in the same commit. No compatibility shims.
- Each git worktree has its own numba cache; the first test run in a fresh worktree is slow (minutes). That
  is normal.

## Running things

| What | Command |
|---|---|
| one area's tests | `uv run pytest tests/polars_engine -q` |
| conventions | `uv run pytest tests/test_conventions.py -q` |
| the guide's examples | `uv run pytest tests/test_guide.py -q` |
| everything | `uv run pytest -q` (slow the first time per worktree) |
| the regression guard | `uv run python benchmarks/modes.py` and `benchmarks/serving_latency.py`, before and after |
| import-weight check | `uv run pytest tests/test_imports.py -q` (`import decider` must not import numba) |
| docs site | `uv run --group docs zensical build` (this folder is excluded from the site) |

On the shared development box, `/tmp` is RAM-backed and the box is often loaded: run heavy tests and
benchmarks through the wrappers in `.scratch/` if they exist there (`heavy.sh`, `light.sh`), and compare
benchmark numbers within one session, never across days.

## Definition of done, per phase

A phase is done when all of these hold:

1. Every test the phase lists in [07-implementation-plan.md](07-implementation-plan.md) exists and passes.
2. The phase's exit criterion is met and the evidence (a command and its output, or a benchmark table) is in
   the PR description.
3. `uv run pytest -q`, including conventions, guide and import tests, is green.
4. `benchmarks/modes.py` and `serving_latency.py` are inside the noise band recorded in
   `docs/design_decisions/experiments/decider-vs-decider2.md` (record p50 25 to 40 µs, batch within 20%).
5. The docs the phase lists are updated, and `docs/design_decisions/decisions/engine.md` has the phase's
   rationale bullets (moved from [03-design-choices.md](03-design-choices.md), not duplicated).
6. This spec is updated: decisions taken, questions closed, anything that turned out differently.

## What a reviewer checks

### P1, `run` absorbs `score`

- `grep -rn "\.score(" decider docs tests benchmarks examples` is empty.
- `Executable.run` checks `LazyFrame` before `DataFrame` before `Mapping`, and the record path is the old
  `score` body unchanged (diff it).
- The `TypeError` for a wrong input names the accepted types.
- `benchmarks/modes.py` record p50 unchanged within noise; the PR shows before and after.
- `GUIDE.md` examples run; the "Modes" bullet no longer says `score`.

### P2, the registry

- `Engine` and its subclasses hold settings only: no attribute is assigned outside the pydantic fields;
  locks and snapshots stay on runners and executables.
- `NumbaEngine` is abstract (an abstract `_runner`), so it never registers as a type; the three subclasses
  register under their aliases and import paths.
- `import decider` still imports no numba (`tests/test_imports.py`); the engine classes load through the
  module `__getattr__`.
- `_settings.py` does not import `_engine`; resolution happens in `serving/_handler.py`.
- The `_tooling.bridge.protocol -> _engine.run.engine` allowlist entry is gone and the test is green.
- `DECIDER_API__ENGINE__TYPE=fused DECIDER_API__ENGINE__STRICT_COMPILE=true decider serve` refuses a
  pipeline with an undeclared fallback (the PR shows the error).

### P3, the default engine

- The context manager resets the variable on exit, including on exception (test).
- `Step.run` and `compile` read the default; `Step.session` opens interpreted unless given an engine (test).
- A `str` step warns once under the default (test), and the guide shows `@allow_fallback` beside the first
  example that warns.
- No test or guide example silently changed its meaning: each `engine="interpreted"` added has a reason in
  the PR.

### P4, the polars host

- Every test in [09-polars-host-design.md](09-polars-host-design.md) exists.
- The spike in 09 still passes against the installed polars.
- `PolarsExecutable` is in the no-per-call-state convention test.
- `test_every_example_flow_equals_fused` runs over every `examples/*` flow; skips name their reason.
- The refusals (`trace`, `session`, qualified emit, frame step without a schema) each have a test and a
  message that says what to do instead.
- `assert_equivalent` includes `polars` by default and the lab does not import `_engine` for it.

### P5, measure

- The experiment file follows the repo's shape (purpose, method, result, decision) and names the script.
- The go/no-go for P6 is written as a sentence with the numbers behind it.

## Stop and ask when

- A test pins behaviour this spec contradicts. Do not change the test to match the spec.
- A benchmark moves outside the noise band and you cannot explain it.
- The installed polars behaves differently from a verified fact in 09 (check `pl.__version__` first).
- The design seems to need an import the layer map forbids.
- A guide example's output changes in a way the plan did not anticipate.
- You want to add a setting, an abstraction or a file the plan does not list.

Write the question down with what you tried and what you expected; the owner would rather answer a precise
question than review a guess.

## Glossary

| Term | Meaning |
|---|---|
| engine | a registry document that turns a pipeline into an `Executable`; today's "mode" |
| numba engine | `InterpretedEngine`, `SteppedEngine`, `FusedEngine`: the existing driver, kernels and `State` |
| driver | `InterpretedRunner` and its subclasses: the code that walks the plan and yields checkpoints |
| plan | `resolve(ir)`: the IR with every read bound to a `NodeVersion` |
| record path | `run(dict)`: one record through one-row arrays, never building a polars frame (the old `score`) |
| host | the v1 polars engine: the fused executable run inside `map_batches` |
| segment | a run of the root sequence's top-level children bound as one sub-executable |
| boundary | a segment that runs in the numba engine; in v1, every segment |
| native segment | v2 only: a run of steps emitted as polars expressions from their polars bodies |
| polars body, `impl` | v2 only: an optional second body of a step written against `pl.Expr` |
| pipeline breaker | a plan node that needs the whole frame; a frame step is one |
| equivalence | every engine gives the same output for the same pipeline, params and rows; `assert_equivalent` checks it |
| checkpoint | `before`/`after` per node, yielded by the driver; what a `Session` pauses on |
| `TraceSink` | per-record decision trace; a numba-engine feature |
