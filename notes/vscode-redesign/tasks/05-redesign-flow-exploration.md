# 05 — Redesign flow discovery and exploration

**Depends on:** 01, 02  
**Blocks:** 07, 10, 11, 12

**Experiment gate:** `experimentation/02-flow-scale-and-gesture-spike.md` must
set and validate graph budgets before implementation freezes.

## Outcome

Make a flow easy to discover and understand before it is executed, with stable
navigation between sidebar, graph, values, and source.

## Work

- Run `experimentation/02-flow-scale-and-gesture-spike.md` before rebuilding
  graph interaction or setting rendering budgets.
- Retain and improve custom pipeline discovery for module-level pipeline values
  and zero-argument pipeline factories; show invalid/ambiguous candidates with
  actionable reasons.
- Redesign the graph around progressive disclosure:
  selected steps show reads/writes, selected edges show carried values, and
  selected values show every read/write/touch point.
- Build one contextual inspector with a small overview and optional structure,
  lineage, and runtime sections instead of competing permanent panes.
- Implement conventional canvas interaction: zoom around pointer, pan empty
  canvas, selection, fit-to-flow, and fit-to-selection.
- Preserve selected node and focal viewport when expanding/collapsing nested
  structure.
- Synchronise selection and navigation across Structure tree, graph, inspector,
  source, and later MCP highlighting.
- Extend or replace existing extension analysis, structure, source-map, and
  comparison modules deliberately; do not duplicate their flow/lineage logic
  behind a new graph-only representation.

## Important decisions

- Avoid dense edge labels as the primary representation of data dependencies.
- Do not conflate static reads/writes with runtime values; make the mode
  explicit in the inspector.
- Confirm supported gestures in the selected graph library before creating
  bespoke interaction logic.

## Done when

- A new user can identify execution order and a step's reads/writes without
  reading crowded edges.
- The graph keeps its focal context through structural changes.
- A selected item resolves consistently to source and its durable identity.
