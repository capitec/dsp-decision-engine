# 01 — Establish shared architecture contracts

**Depends on:** none  
**Blocks:** 01a, 02, 03, 04, 05, 06, 07, 08, 11

## Outcome

Define the small static-flow contract that lets `decider`, VS Code, notebooks,
checks, experiments, and MCP refer to the same flow structure and source
without inventing incompatible identifiers. Execution lifecycle and
reproducibility are deliberately owned by task 01a; adapter selection/focus is
deliberately owned by adapters.

## Work

- Inventory and extend the existing core rather than designing a parallel
  schema: `engine/ir`, `Origin.path`, `step_map`, `engine/run`, `State`,
  `RunReport`, `Version`, `engine/debug.Session`, existing trace/path output,
  equivalence/corpus utilities, and extension adapters.
- Define compact public static references and queries: `FlowRef`, `StepRef`,
  `EdgeRef`, `ValueSlotRef`, and source location. Keep `ValueSlot` distinct
  from runtime observed values and trace evidence.
- Define reference lifecycle rules: flow IDs and step IDs form the durable
  global reference; derived paths are development context; unresolved durable
  references warn and preserve capture-time descriptive metadata rather than
  silently disappearing.
- Define `RecordRef` semantics at the contract level: a durable reference
  requires an explicit/validated record key, while session row ordinals are
  display-only and ephemeral.
- Make Python models in `decider` the contract owner. Generate versioned JSON
  Schema, TypeScript types, and MCP schemas from those models.
- Specify static versus runtime facts and ensure callers can tell the
  difference.
- Define error and capability reporting for unavailable data, source mappings,
  optional dependencies, and unsupported execution modes.
- Establish compatibility expectations for existing VS Code commands and
  launch configurations, including bridge protocol/version capability checks.
- State the multi-root workspace assumption: discovery and interpreter
  configuration are scoped per workspace folder, never implicitly shared.

## Important decisions

- Keep this a deep module: callers ask for stable flow descriptions and
  queries, not assemble them from IR and debugger internals.
- Do not duplicate business execution semantics in VS Code or MCP.
- Selection, focus, highlighting, run identity, and experiment asset shape are
  outside this contract and have their own owners.

## Done when

- A short, versioned static-flow contract names the shared entities and their
  invariants.
- Existing consumers have a migration path without silently changing behaviour.
- Subsequent tasks can depend on the contract instead of internal file paths or
  webview-specific data.
