# 03 — Relocate and harden the debug bridge

**Depends on:** 01  
**Blocks:** 06, 07, 11, 12

## Outcome

Move `tools/decider-bridge` into `decider.debug_bridge` while preserving the
debug protocol and creating a clear execution seam for VS Code and future
scenario-to-debug launches.

## Work

- Map the current bridge package, DAP protocol, launch configuration, imports,
  packaging, and extension process startup.
- Relocate it under `decider.debug_bridge` with a compatibility/migration plan
  for existing clients and development environments.
- Define explicit debug-session capabilities: pause, resume, step, source
  mapping, frame/record scope, variable inspection, and safe live overrides.
- Preserve the rule that compile-time failure does not silently fall back per
  step, while runtime errors propagate clearly.
- Verify debugger operation through the existing extension and Python test
  surfaces before removing the old package path.

## Important decisions

- Prefer `decider.debug_bridge` over an abbreviated name for discoverability.
- The bridge owns debugger transport/adaptation; core execution semantics remain
  in `decider`.
- Keep the launch contract explicit enough that a reproduced What-If scenario
  can start an equivalent debug run later.

## Done when

- Existing debug workflows and automated tests work from the new package.
- Documentation/configuration refers to the new supported package path.
- The old path is removed only after compatibility obligations are satisfied.

