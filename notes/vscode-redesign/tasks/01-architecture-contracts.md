# 01 — Establish shared architecture contracts

**Depends on:** none  
**Blocks:** 02, 03, 04, 05, 06, 07, 08, 11

## Outcome

Define the shared model that lets `decider`, the debug bridge, VS Code,
notebooks, checks, experiments, and MCP refer to the same flow, step, value,
run, revision, and selection without inventing incompatible identifiers.

## Work

- Inventory the current extension, bridge, IR, execution, and configuration
  representations and identify which existing concepts can become canonical.
- Define a compact public representation for flow structure and selection:
  flow, step, edge, value, source location, input data identity, run, revision,
  and focused record/frame.
- Specify static versus runtime facts and ensure callers can tell the
  difference.
- Define error and capability reporting for unavailable data, source mappings,
  optional dependencies, and unsupported execution modes.
- Define the project-owned experiment asset convention:
  `experiments/<slug>/experiment.yaml`, with optional `README.md`, `tests/`,
  and `graphs/`.
- Establish compatibility expectations for existing VS Code commands and
  launch configurations.

## Important decisions

- Keep this a deep module: callers should ask for a stable flow description or
  selection context, not assemble it from IR and debugger internals.
- Do not duplicate business execution semantics in VS Code or MCP.
- Decide which identifiers are public/persisted and which may change across a
  process invocation.

## Done when

- A short, versioned contract names the shared entities and their invariants.
- Existing consumers have a migration path without silently changing behaviour.
- Subsequent tasks can depend on the contract instead of internal file paths or
  webview-specific data.

