# 10 — Implement experiments and VS Code workflows

**Depends on:** 05, 06, 07, 09  
**Blocks:** 11, 12

## Outcome

Deliver reproducible project-owned experiments, aggregate comparison results,
and a clear hand-off from an experiment finding into record-level debugging.

## Work

- Implement the selected `decider` experiment interface and YAML-backed
  definitions.
- Support baseline and variant runs, parameters, input changes, declared
  overrides, scenario combinations, and revision comparisons.
- Produce summaries for path/branch counts, first divergence, changed/
  unchanged/unique values, selected-step differences, and sampled drill-down.
- Implement appropriate built-in visualisations, including a Sankey-style path
  view where data and scale support it.
- Build VS Code entry points that distinguish experiments from debugging and
  display run state, results, failures, and links to source/records.
- Allow a selected scenario/record to launch the equivalent debugger run.
- Support caller-selected result directories and test the documented project
  layout with optional `README.md`, tests, and custom graphs.

## Important decisions

- The extension is an adapter for the experiment module, not an independent
  implementation of experiment semantics.
- Results must state baseline, revision, input provenance, scenario settings,
  and execution limitations so they are reproducible.
- Keep large populations aggregated until a user explicitly drills down.

## Done when

- A saved experiment can be re-run from its definition and compared reliably.
- Users can go from aggregate divergence to records, flow, source, and a debug
  session without manually reconstructing context.
- The UI does not merge exploratory scenarios with mutable live-debug state.

