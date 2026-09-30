# 12 — Integrate, validate, and document the redesign

**Depends on:** 03, 04, 05, 06, 07, 08, 10, 11  
**Blocks:** none

## Outcome

Ship the redesigned capabilities as a coherent product with reliable migration,
performance evidence, clear client responsibility boundaries, and end-to-end
coverage of the primary user stories.

## Work

- Validate the three primary modes end to end: understand a flow, investigate
  an execution, and explore a population.
- Add representative tests for discovery, IDs, tracing across execution modes,
  data loading/scope safety, graph interaction, breakpoints/overrides, checks,
  experiments, MCP, and package relocation.
- Measure and document tracing overhead, trace payload characteristics, local
  Polars experiment capacity, UI responsiveness, and large-result aggregation.
- Test migrations for existing extension users, existing launch configurations,
  old bridge packaging, and pipelines without generated IDs.
- Update user documentation, Python examples, VS Code guides, MCP guidance,
  and project experiment authoring guidance.
- Record supported limits and client-owned responsibilities: trace retention,
  privacy handling, CI release policy, results storage, and sampling for data
  beyond local capacity.

## Important decisions

- Do not claim regulatory compliance, equivalence proof, or unlimited scale
  beyond the exact validated scope.
- Preserve existing behaviour where compatibility is promised; surface
  deliberate changes and migrations plainly.
- Keep performance-sensitive functionality opt-in where the measured cost
  warrants it.

## Done when

- The primary stories are demonstrably usable and regression-covered.
- Limits, migrations, and client responsibilities are documented.
- The package, extension, and MCP interfaces release together without hidden
  compatibility breaks.
