# Experiment authoring

How to write an experiment a notebook, VS Code and MCP can all run.
`notes/experiment-module.md` records the interface design and rejected
alternatives; this is the practical authoring view.

## Project layout

Experiments are project-owned, prescriptive assets:

```text
experiments/
  <experiment-slug>/
    experiment.yaml     # the definition (versioned, data only)
    README.md           # optional purpose and interpretation
    tests/              # optional data-driven checks (Python)
    graphs/             # optional custom visualisations (Python)
```

`experiment.yaml` references `tests/` and `graphs/`; it never embeds arbitrary
code. Generated run results go to a separate, caller-chosen directory — they are
not part of the definition.

## `experiment.yaml`

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

## Scenario overrides

An override applies only at a declared point:

- **input column** (no `at`) — settable before anything runs;
- **produced value** (`at: <step path>`) — the path must produce the value.

`validate_scenarios` rejects an unknown name or a produced value without `at`,
so a scenario cannot reach an arbitrary internal state. A scenario that only
changes `params` routes to the fused `run(frame, params=...)` path (~750×
faster than the session replay a value override needs).

## Running and results

Plain Python (the builder produces the same `ExperimentDef` a YAML file holds):

```python
from decider.experiments import Experiment

result = (Experiment("drift", pipeline, df)
          .scenario("baseline")
          .scenario("limit_0.3", params={"demo": {"approved": {"limit": 0.3}}})
          .compare(outputs=("tier",))
          .run())
result.status  # COMPLETED, NON_REPRODUCIBLE, PARTIAL, FAILED, CANCELLED, TIMED_OUT
```

Headless from the asset (a plain Python caller, no VS Code):

```python
from decider.experiments import run, load_yaml, ExperimentDef
def_ = ExperimentDef.model_validate(load_yaml(open("experiments/drift/experiment.yaml").read()))
result = run(def_, out_dir="results/drift")
```

Results carry the run manifest (flow, resolved revision, environment, input
fingerprint, overrides, output refs), per-scenario status, divergences, a
pre-flight report and a job snapshot. Nondeterminism is a first-class state,
never a footnote; a partial or cancelled run never looks like a completed
deterministic one.
