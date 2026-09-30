# 07 — Redesign debugging and runtime inspection

**Depends on:** 03, 04, 05, 06  
**Blocks:** 10, 11, 12

## Outcome

Make the existing debugger capabilities discoverable through the flow, while
preserving normal Python debugging and clearly separating live debugging from
reproducible experiments.

## Work

- Expose all existing breakpoint types on graph nodes and source, with clear
  visual state for enabled, disabled, hit, and current pause.
- Map condition nodes to condition source and step nodes to step source; do not
  create unnecessary loop-container navigation when defining logic is reachable.
- Make pause state, call stack, step context, input/output values, trace
  evidence, and source selection work together in the runtime inspector.
- Support normal Python debug-console and Variables edits for parameters/state,
  while showing scope and lifetime: current pause, session, or rerun.
- Preserve step-into Python and source editing/rerun workflows where the
  debugger supports them.
- Make the transition from a saved What-If scenario to an equivalent debugger
  launch explicit, one-way, and reproducible.

## Important decisions

- Debugging explains one execution; it does not become the scenario-definition
  interface.
- A user must never mistake a session-local live override for a saved experiment
  change.
- Runtime views consume structured trace data where available rather than
  recreating incompatible explanation records.

## Done when

- A user can find/set a breakpoint and explain a pause from the graph.
- Live changes clearly identify their scope and downstream effect.
- A saved scenario opens an equivalent, inspectable debug run for one record.

