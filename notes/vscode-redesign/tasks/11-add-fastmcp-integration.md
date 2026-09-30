# 11 — Add FastMCP integration

**Depends on:** 01, 02, 04, 05, 06, 07, 08, 09, 10  
**Blocks:** 12

## Outcome

Expose enough structured `decider` context and control through FastMCP that an
agent can conduct discovery, analysis, and visual focus without relying on
screen scraping or inventing workflow state.

## Work

- Run `experimentation/04-fastmcp-topology-spike.md` before choosing server
  transport, lifecycle, or editor IPC.
- Define a small MCP interface over stable core concepts rather than mirroring
  every webview action.
- Provide read tools for flow discovery, flow/subgraph description, step/edge/
  value inspection, lineage, source context, selected-record and trace data,
  run summaries, check reports, experiment definitions, and experiment results.
- Provide visual-focus tools to highlight/reveal flow entities in VS Code.
- Define confirmation-gated tools for code execution, debugger startup,
  experiment runs, source generation, and any persistence.
- Implement FastMCP transport, schemas, error reporting, capability discovery,
  and integration tests against VS Code selection/highlighting.
- Ensure returned data preserves stable identities and makes static/runtime
  context explicit.

## Important decisions

- Read access should be broad enough to support agent-led workflows.
- The client environment and trace adapter determine data availability; MCP
  must report unavailable/redacted data plainly rather than fabricate it.
- MCP is an interface over core and editor modules, not another execution
  engine or experiment implementation.

## Done when

- An agent can discover a flow, explain a selected area from structured context,
  retrieve relevant trace/check/experiment evidence, and focus the matching UI.
- Persisting or code-running tools require explicit confirmation.
- MCP behaviour is covered by contract and integration tests.
