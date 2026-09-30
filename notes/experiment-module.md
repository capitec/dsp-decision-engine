# Experiment module

The chosen interface, rejected alternatives, and the schema the module layers on
`decider.experiments` (fork/sweep), `decider.lifecycle` (manifest/job) and
`decider.data` (load/identity/preflight).

## Chosen interface

One versioned declarative schema — `ExperimentDef`, the `experiment.yaml`
asset — with two thin facades that both produce it (a YAML file and the
`Experiment` builder), driving one `run_experiment`, the only place execution
happens.

- Plain Python: `Experiment(...).scenario(...).compare(...).run()`.
- VS Code / MCP: read and write the same `experiment.yaml`, resolved to the
  same `ExperimentDef`, so the asset and the builder tell one story.
- Execution reuses `forks.fork` (a fresh session replayed to a checkpoint and
  overridden); comparison, aggregation and provenance are layered over it, not
  a second execution path.

**Rejected alternative 1 — Python-code-first (the experiment *is* a notebook /
pytest file).** Not serialisable to YAML or MCP without `eval`, no declarative
pre-flight surface, and a result cannot be reproduced from a data asset.

**Rejected alternative 2 — YAML-only with no Python surface.** The flow entry
is an import path and optional `tests/`/`graphs/` are Python; a notebook or
plain-Python caller wants an object, not a file.

**Rejected alternative 3 — a new distributed execution path.** Local
reproducible Polars execution already exists as `fork`; a second backend would
fork the execution semantics the review warned against.

## `experiment.yaml` (authored, versioned)

```yaml
version: 1
name: affordability-verdict-drift
flow:
  entry: 02-affordability.sonnet.pipeline:build   # module:factory import path
  decider: ">=1.0,<2.0"                            # supported engine window
revision: HEAD                                     # authored intent; resolved to a SHA at run time
input:
  path: data/sample.parquet
  id_column: decision_id
scenarios:
  - name: baseline
  - name: limit_0.3
    params: {affordability_assessment: {capacity: {overlay: {adjustment_stack_enabled: false}}}}
  - name: force_income
    overrides: {disposable_income: 1200.0}         # input override, before anything runs
  - name: force_ratio
    at: affordability_assessment/capacity/discretionary_income   # declared override point
    overrides: {discretionary_income: 800.0}
  - name: prev_revision
    revision: HEAD^
comparison:
  baseline: baseline
  outputs: [affordability_verdict_code, max_affordable_instalment]
  tolerance: {rtol: 1e-6, atol: 1e-9}
  determinism: true                                 # rerun baseline; a differing rerun => non_reproducible
summaries:
  - {type: counts, output: affordability_verdict_code}
  - {type: first_divergence, against: baseline}
tests: tests/test_affordability.py                 # reference, not embedded code
graphs: graphs/verdict_sankey.py
```

`revision` is the spike's `symbolic` intent, mapped to the lifecycle
`Revision.authored`; `resolved` is filled into the manifest, not the asset.

## Run manifest (the experiment layer)

The generic `RunManifest` (task 01a) already carries provenance — `flow`,
`revision`, `source`, `environment`, `input`, `overrides`, `outputs`. The
experiment layer (`ExperimentResult`) adds the states that must never look
equivalent to a deterministic completed run:

- `status: RunStatus` — `completed | non_reproducible | partial | failed |
  cancelled | timed_out`. `NON_REPRODUCIBLE` is a first-class result state
  (R5/D2), not a footnote, and `PARTIAL` is distinct from `COMPLETED`.
- `nondeterministic: bool` — detected by rerunning the baseline with exact
  tolerance.
- `scenarios: tuple[ScenarioResult, ...]` — per-scenario `completed | failed |
  resumed | skipped`, error, divergences and an `output_ref`.
- `divergences: tuple[Finding, ...]` — portable findings: `kind`, `location`
  (`col[row]`), `expected`, `actual`, `message`.
- `preflight: PreflightReport` — data compatibility and run-cost captured
  before anything ran (Story 4).
- `job: Job` — the terminal `JobHandle` snapshot (progress, cancel, timeout).

Resolved revision, environment (python/decider versions), input fingerprint and
schema all come from the lifecycle/`LoadedData` primitives, not re-computed
here.

## Declared override points

`validate_scenarios` (returned problems, not raised) checks every scenario
override against `describe(step).value_slots`: an input column is settable
before anything runs (no `at`), a produced value must name the step output it
overrides (`at`), and that path must produce the value. This *is* the
authorisation the spike found `fork` already enforces — unknown names and
produced-without-`at` are rejected, so a scenario cannot reach an arbitrary
internal state.

## Reuse and performance limits

- Execution: `forks.fork` per scenario; per-scenario failure is isolated
  because `fork` catches each scenario's error into the trace.
- Job model: `JobHandle` for progress (scenario index / total), cancellation
  and timeout checked between scenarios, `PARTIAL` for a stopped sweep, and
  resume that skips a completed scenario only when input fingerprint and
  resolved revision still match.
- Comparison reads *produced* outputs (`comparison.outputs`), because input
  overrides are not reflected in the output frame (spike finding 2).
- Pre-flight and cost: `data.preflight` (`CostEstimate` rows × non-frame calls).

The public interface reflects the spike's measured limits: param-only scenarios
route to the fused `run(df, params=...)` path (~750× faster), override/force
scenarios to the interpreted session/fork path. That routing decision is task
10's; the schema already distinguishes the two (a scenario with only `params`
needs no session replay).

## YAML subset

`decider` has no YAML dependency and the schema is data only, so `yaml.py`
loads/dumps a minimal subset (nested maps, `-` lists, scalars) that round-trips
the whole `ExperimentDef`; the canonical wire form is JSON
(`model_validate_json`). Full YAML (anchors, flow style, tags) is out of scope.

## Open decisions for task 10

- Param-only scenario routing to fused `run` (the scale lever) is described but
  not wired here.
- `resolve_revision` shells out to `git`; a `flow.decider` window check against
  the installed engine is not yet enforced.
- `Scenario.revision` (`HEAD^`) is modelled but not executed — running past
  code on today's engine is a hard boundary the schema names without solving.
- Result files are referenced (`output_ref`) but not written; `decider` owns
  metadata, not bulk storage.
