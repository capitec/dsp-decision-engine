# Experiment 03 findings — Experiment interface and local scale

**Run during:** task 09, before the public experiment interface or `experiment.yaml` schema is frozen
**Feeds:** tasks 09, 10, 11, 12

## Question and answer in one line

The experiment interface is **one versioned declarative schema** (the
`experiment.yaml` asset) with **two thin facades** — a YAML file and a small
Python builder — both driving **one core runner** that reuses
`decider_bridge.forks.fork`/`sweep` + `pipeline.session(df)` + `s.set(...)` for
execution and `assert_equivalent`/`corpus` for comparison and case generation.
It does not add a new execution path; it layers provenance, comparison,
aggregation, and job semantics over the session/fork primitives that already
exist.

## What exists today (what the interface must build on, not rebuild)

Feedback C1/R3 already located the primitives. This spike confirms each is
sufficient and pins exactly where they fall short:

| Primitive | Location | What it already gives | What is missing |
|---|---|---|---|
| what-if / override | `pipeline.session(df)` + `s.set("ratio", 0.1)` | checkpoint pause, inspect, `set` (recorded as `override@<path>`), resume | a *declared* override point; a public params setter (`fork` pokes `s._params` privately) |
| fork / sweep / replay | `tools/decider-bridge/decider_bridge/forks.py` (`fork`, `sweep`) | a fresh session replayed to a checkpoint, scenario overrides/params/forces applied, per-scenario error captured as a trace | no cancellation, no fingerprint, no manifest, interpreted-only |
| comparison | `decider/testing/equivalence.py` (`assert_equivalent`) | run == score == session across modes, exact (NaN-aware) | float tolerance is a *policy*, not built in |
| case generation | `decider/testing/corpus.py` (`zeros/negatives/empty/chunked`) | boundary-value frames | nothing — reuse as-is |
| trace material | `forks.collect` / `runs.collect` | `{steps: {path: {col: [...]}}, output: {...}, error}` | none for the spike |

The deep module is therefore a ~60-line runner over `fork`, not a new engine.
The two files produced by this spike (`prototype.py`, `benchmark.py`) prove it.

## The two candidate interfaces

### Candidate A — declarative `experiment.yaml` (asset-first)

The experiment *is* a versioned file. A CLI/runner interprets it; Python
appears only in optional `tests/` and `graphs/` hooks. Highest portability:
plain Python, VS Code, and headless MCP all read the same asset without
evaluating code.

### Candidate B — Python API (code-first)

`Experiment(...).scenario(...).run()` returns a `Results` object. Best notebook
ergonomics and composition; but the definition is not a static asset, so VS
Code and MCP cannot discover or reproduce it without executing Python.

## Chosen interface and rejected alternatives

**Chosen: candidate A as the canonical serialization, candidate B as a thin
builder/serializer over the *same* schema, one core `run_experiment`.**

The prototype makes the equivalence literal: `Spec` (candidate A's YAML, parsed
to a dict) and `Experiment` (candidate B's builder) both feed
`run_experiment`, which is the only place execution happens. The Python API
does not have its own runner; it is ~20 lines of sugar that appends to the same
`Spec`.

| Caller | How it consumes the one schema |
|---|---|
| plain Python | `Experiment(...).scenario(...).run()` — the builder |
| VS Code | reads/writes `experiment.yaml`; discovers `experiments/*/experiment.yaml`; persists as YAML |
| headless MCP | the same YAML/JSON declarative subset, no `eval`; run is a confirmation-gated tool |

**Rejected alternative 1 — Python-code-first (the experiment *is* a
`pytest`/notebook file).** Not serialisable to YAML/MCP without `eval`, no
pre-flight declarative validation surface, VS Code cannot introspect or edit
it, and a result cannot be reproduced from a data asset. It throws away the
one-schema convergence the feedback demanded.

**Rejected alternative 2 — YAML-only with no Python surface.** The flow is a
Python import path and custom hooks are Python; a notebook or plain-Python
caller wants an object, not a file. Pure YAML would force every non-trivial
graph or test into an awkward string-embedded DSL.

**Rejected alternative 3 — a new distributed execution path.** The question is
local reproducible Polars execution; `fork` already is that. A distributed
backend is out of scope and would fork the execution semantics the feedback
explicitly warned against (C1).

## Proposed `experiment.yaml` (versioned, authored)

```yaml
version: 1
experiment:
  name: affordability-verdict-drift
  description: optional
flow:
  entry: 02-affordability.sonnet.pipeline:build   # import path "module:factory"
  decider: ">=1.0,<2.0"                            # supported engine window (M6)
revision:
  symbolic: HEAD                                   # authored intent
  resolved: null                                   # filled to a SHA at run time
input:
  path: data/sample.parquet                        # json | csv | parquet
  id_column: decision_id                           # explicit record identity (M4)
  fingerprint: {algorithm: sha256, value: null}    # filled at run time
scenarios:
  - {name: baseline}
  - name: limit_0.3
    params: {affordability_assessment: {capacity: {overlay: {apply_max_affordable_instalment_adjustments: {adjustment_stack_enabled: false}}}}}
  - name: force_income
    overrides: {disposable_income: 1200.0}         # input override (before run)
  - name: force_ratio
    at: affordability_assessment/capacity/discretionary_income   # declared override point
    overrides: {discretionary_income: 800.0}
  - name: prev_revision
    revision: HEAD^
comparison:
  baseline: baseline
  outputs: [affordability_verdict_code, max_affordable_instalment, discretionary_income_after]
  tolerance: {float_rtol: 1e-6, float_atol: 1e-9}
  determinism: rerun_baseline                      # run baseline twice, flag non_reproducible
summaries:
  - {type: counts, output: affordability_verdict_code}
  - {type: first_divergence, against: baseline}
tests: tests/test_affordability.py                 # optional, reference not code
graphs: graphs/verdict_sankey.py                   # optional, reference not code
```

Two fields carry the load the feedback flagged as missing (M6): `flow.decider`
pins the engine window, and `input.fingerprint` + `revision` make the baseline
unambiguous.

## Proposed result-manifest shape (versioned, generated)

```yaml
version: 1
run:
  id: 2026-09-30T14.00-affordability-verdict-drift
  started_at: ... ; finished_at: ...
  status: completed                              # completed | partial | failed | cancelled | non_reproducible
experiment:
  name: affordability-verdict-drift
  flow: {entry: ..., decider: ...}
  revision: {symbolic: HEAD, resolved: 0e32b63...}
  environment: {decider_version: ..., python_version: ...}
  source: {dirty: false, fingerprint: sha256:...}
input:
  path: data/sample.parquet
  fingerprint: 7799a997db5f75de
  row_count: 4
  schema: {income: Float64, debt: Float64}
  id_column: decision_id
nondeterministic: false
scenarios:
  - {name: baseline, status: completed, error: null, output_ref: results/baseline.parquet, divergences: []}
  - {name: force_ratio, status: failed, error: "ValueError: can't set 'ratio' to 'not-a-float': it doesn't cast to float64", divergences: ["a run errored"]}
comparison: {baseline: baseline, first_divergence: [...]}
```

Every state the spec requires is unambiguous: `nondeterministic` is a
first-class flag (not a silent result), `partial`/`cancelled`/`failed` never
look like a completed deterministic run, and `revision.resolved` is always a
SHA while `symbolic` is preserved as intent.

## Portable declarative subset vs optional Python hooks

The **declarative subset** is what MCP and VS Code may author without running
code: flow entry + revision, input reference + fingerprint, scenarios (params,
input/step-output overrides, forces), comparison outputs + tolerance, and the
built-in summaries (`counts`, `first_divergence`). Everything is data.

The **Python hooks** are *references from the YAML, not embedded code*:
`tests/` (data-driven checks, `assert results.tests.all_pass()` in the
illustrative shape) and `graphs/` (custom visualisations). They receive the
same `Results` object a notebook gets; the core never imports them unless the
caller asks. This keeps the "YAML never embeds arbitrary code" rule from the
review (recommendation "Experiment layout") while preserving the
non-standard-test/graph escape hatch.

## Job, progress, cancellation, and result ownership

- **Cancellation** is a `threading.Event` checked *between* scenarios (coarse,
  cheap, correct for a sweep) and `session.pause()` is already safe *inside* a
  running `resume` for mid-scenario stop. The prototype demonstrates the former.
- **Progress** = scenario index over total; the manifest is append/rewrite per
  completed scenario, so a partial manifest is always readable.
- **Per-scenario failure** is free: `fork` already catches each scenario's
  exception into `trace.error`, so one bad scenario never aborts the sweep.
- **Resume** keys on `(scenario name, input fingerprint, resolved revision)`:
  a completed scenario is skipped only when all three still match; any mismatch
  re-runs it (input fingerprint mismatch → stale → rerun).
- **Ownership:** `decider` writes manifests and scenario output references to a
  caller-chosen directory; it does not own bulk storage, Git LFS, or retention.
  This matches decision #14 and task 01a's "small manifest metadata, not bulk".

## Semantics exercised (all green in `prototype.py`)

| Scenario | Result |
|---|---|
| input fingerprint (sha256 over Arrow IPC bytes) | `7799a997db5f75de` for the 4-row frame |
| symbolic + resolved revision | `HEAD` → `0e32b632c02b6b8beaea5159259d5e96b465b43b` |
| declared override validation | unknown name rejected; produced value without `at` rejected; input override accepted |
| float tolerance comparison | `close()` with `rtol/atol`; divergences reported as `col[row]: a != b` |
| step-output override (fork at a checkpoint) | `at: demo/ratio` → replay to `("demo/ratio","after",1)`, `set("ratio", 0.1)` |
| nondeterministic step | rerun baseline twice; a global-counter step flagged `nondeterministic: true` |
| cancellation | `cancel.set()` → `cancelled: true`, 0 scenarios ran |
| per-scenario failure | `bad_cast` fails, other four complete |
| resume | completed `baseline` skipped (`resumed`), others rerun |
| manifest generation | versioned dict with revision, input, status per scenario |

## Key findings the spike surfaced (these change task 09)

1. **Overrides need a declared point, and `fork` already enforces it.** `fork`
   applies `scenario["overrides"]` at a checkpoint (`target`); `None` means
   "before anything runs", so only inputs are settable there. A produced value
   (e.g. `ratio`) must name the step output it overrides
   (`target=("demo/ratio","after",1)`). This *is* userstories decision #15 —
   the interface's `at:` field is the declared override point, and validation
   ("produced, not an input; declare `at: <step path>`") is the authorisation.

2. **Input overrides are invisible in the output frame.** `session.set` writes
   to state, but `output()` passes input columns through from `state.frame`,
   so `forks.collect`'s `output` does not reflect an input override (the
   session source already carries a `# ponytail` note about exactly this).
   Comparison must either (a) compare *produced* outputs only, or (b) overlay
   input overrides onto the collected frame. The prototype's divergences came
   through produced columns (`tier`), which is the right comparison surface.

3. **The fork path is interpreted-only → the local scale ceiling is ~200k
   rows/s, not the fused ~160M rows/s.** See the benchmark below. The runner
   must route by scenario capability: param-only scenarios to fused
   `exe.run(df, params=...)` (~750× faster), override/force scenarios to the
   session/fork path. This single routing decision is the difference between
   "interactive to ~1M rows" and "billion-row param sweeps".

4. **`fork` mutates `s._params` privately** to swap the params document
   (`# ponytail: ask Session for a params setter if this sticks`). R3 is
   confirmed: rehoming `forks`/`sweeps` into core must add a public params
   setter on `Session` as part of the same change.

## Measured local scale (not extrapolated)

`benchmark.py`, best of 3, one representative 3-step `flow` with a param:

| rows | fused `run()` | stepped `run()` | interpreted `run()` |
|---|---|---|---|
| 1 000 | 0.2 ms | 0.5 ms | 4.2 ms |
| 100 000 | 0.5 ms | 0.6 ms | 467 ms |
| 1 000 000 | 2.3 ms | 4.6 ms | 4 559 ms |
| 5 000 000 | 32 ms | 69 ms | 24 054 ms |

Experiment path (`forks.sweep`, 5 scenarios, interpreted session replay):

| rows | total (5 scenarios) | per scenario |
|---|---|---|
| 100 000 | 2.9 s | 584 ms |
| 1 000 000 | 31 s | 6.25 s |

Fast path (param-only sweep via fused `run`, no session): 5 scenarios × 1M rows
= **42 ms total** vs 31 s via fork (~750×).

Real pipeline (`example_projects/02-affordability/sonnet`, 52 steps, nested
`list[dict]`/`dict` columns): interpreted `score()` ≈ **15 ms/record ≈ 67
records/s**.

**Practical local limit, stated from these numbers:**

- **Param sweeps (fused):** 5M rows = 32 ms, so 100M+ rows per sweep is
  comfortable; the wall clock is seconds, memory is tens of MB per frame. The
  limit is memory at ~10⁸–10⁹ rows, not throughput.
- **Override/force scenarios (interpreted fork):** ~200k rows/s ⇒ a single
  scenario over 1M rows is ~6 s and each additional scenario adds the same
  again. Interactive ceiling is **~1–10M rows across a handful of scenarios**
  (minutes), beyond which the runner must route to fused or sample.
- **Nested-column pipelines (Python fallback):** ~67 records/s ⇒ anything past
  ~10⁵ records needs sampling or a fused re-expression. This is the binding
  constraint for the real affordability flow, not kernel throughput.

## How to reproduce

```bash
PYTHONPATH=tools/decider-bridge uv run python \
    notes/vscode-redesign/experimentation/03-experiment-interface/prototype.py

PYTHONPATH=tools/decider-bridge:example_projects/02-affordability/sonnet:example_projects/00-shared-credit-core/sonnet \
    uv run python notes/vscode-redesign/experimentation/03-experiment-interface/benchmark.py
```

Both are throwaway scripts under `notes/`, not part of `decider`.

## Plan-change recommendations for task 09

1. **Rehome `forks`/`sweeps` + add a public `Session.set_params` before
   building the experiment module** (R3): the runner needs `fork` in core and a
   params setter; today it imports from the bridge and pokes `_params`.
2. **Add the `at:` declared-override-point to the scenario schema** — it is not
   optional; it is how `fork` already authorises a produced-value override, and
   validation surfaces it to callers.
3. **Add scenario-capability routing (param-only → fused; override/force →
   session)** as a task-09 acceptance signal, not a later optimisation; the
   spike shows it is the scale lever.
4. **Add the `flow.decider` engine window and `input.fingerprint` fields now**
   (M6) — both are missing from the task's YAML sketch and both are required for
   Story 3's "unambiguous baseline and input provenance".
5. **Decide the comparison surface explicitly:** produced outputs vs input
   overrides (finding 2) so task 10's aggregation reads the same columns.
