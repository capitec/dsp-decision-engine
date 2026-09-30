# 03 — Relocate and harden debug transports

**Depends on:** 01  
**Blocks:** 06, 07, 11, 12

## Outcome

Relocate the editor/debug transport package without moving or duplicating
`engine.debug.Session`, preserving every supported consumer and transport.

## Work

- Map the current bridge package, `engine.debug.Session`, stdin JSON-lines
  transport, debugpy attach, websocket transport, launch configuration,
  imports, packaging, and extension/JupyterLab process startup.
- Choose and document a transport-oriented package home. If
  `decider.debug_bridge` is retained, document it explicitly as an adapter over
  `engine.debug.Session`, never a second debug-session implementation.
- Preserve all currently supported transports and their clients through the
  move; do not remove JupyterLab compatibility as an incidental consequence.
- Classify bridge helpers before moving them: transport/process lifecycle stays
  in the bridge, session semantics stay in core, lineage/description joins core
  query ownership, and forks/sweeps are evaluated by task 09 as experiment
  primitives.
- Preserve the rule that compile-time failure does not silently fall back per
  step, while runtime errors propagate clearly.
- Identify and test the existing cross-mode semantic parity contract, including
  the documented numeric guarantees that apply to stepped debug runs and
  compiled execution.
- Verify debugger operation through the existing extension and Python test
  surfaces before removing the old package path.

## Important decisions

- The bridge owns debugger transport/adaptation; core execution semantics
  remain in `engine.debug`.
- Keep the launch contract explicit enough that a reproduced What-If scenario
  can start an equivalent debug run later.

## Done when

- Existing VS Code and JupyterLab debug workflows, transports, and automated
  tests work from the new package.
- Documentation/configuration refers to the new supported package path.
- The old path is removed only after compatibility obligations are satisfied.
