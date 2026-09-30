# 09 — Design and spike the experiment module

**Depends on:** 01, 01a, 02, 04, 06

**Blocks:** 10, 11, 12

## Outcome

Choose a small, deep `decider` experiment interface that supports notebooks and
VS Code without duplicating execution, comparison, aggregation, or persistence
semantics.

## Work

- Run `experimentation/03-experiment-interface-and-local-scale-spike.md`
  before freezing the public Python interface or `experiment.yaml` schema.
- Research comparable experiment/evaluation systems and inspect the current
  `decider` comparison and execution machinery.
- Design at least two materially different interfaces and compare their depth,
  locality, error modes, notebook ergonomics, VS Code integration, and test
  surface.
- Spike the uncertain semantics: declared overrides, parameter sweeps, relative
  Git revisions such as `HEAD^`, baseline selection, scenario-to-debug
  reproduction, Polars-backed local execution, and large-result aggregation.
- Define the experiment asset model in `experiment.yaml`, including flow,
  revision, input reference, scenarios, requested summaries, built-in plots,
  and references to optional Python tests/graphs.
- Define how override points are explicitly authored on supported step outputs,
  validated against the static flow contract, surfaced to callers, and
  suggested without treating automatic discovery as authorisation.
- Define result ownership: `decider` writes to a caller-selected directory and
  exposes result metadata; it does not own remote storage or Git LFS policy.
- Design the experiment layer of the versioned run manifest: input
  fingerprinting, resolved revision, engine/Python environment, equality or
  tolerance policy, scenario/job status, output references, captured
  descriptive metadata, and portable finding descriptors.
- Reuse the task 01a job model for progress, cancellation, timeout,
  per-scenario failure isolation, partial manifests, and resume only when
  immutable inputs/manifests match.
- Define detected nondeterminism as a first-class non-reproducible result
  state; controlled clocks/seeds remain deferred.
- Add pre-flight compatibility and run-cost estimation to the experiment
  definition/run workflow.

## Important decisions

- Scenario overrides apply only at declared override points or step outputs.
- Design for aggregate and sampled/drill-down results; do not materialise
  population-sized graph state in memory or the editor.
- The public interface must not freeze until the spikes establish viable
  behaviour and performance limits.

## Done when

- A recommended interface is selected with rejected alternatives and evidence.
- A small prototype validates the hardest semantics and local execution path.
- The YAML and Python/notebook interface tell one consistent story.
- Non-reproducible and partially completed runs have unambiguous manifest
  states rather than appearing equivalent to deterministic completed results.
