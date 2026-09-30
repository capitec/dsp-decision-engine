# MCP guidance

`decider` exposes a FastMCP server for agents. It is a thin interface over the
core and editor modules — not another execution engine or a mirror of every
webview action.

## Running it

```sh
uv run decider mcp            # headless tools only; raw tools redacted
uv run decider mcp --raw      # also enable per-record value/trace tools
```

Declare it in the agent's MCP config as a stdio server (command
`uv run decider mcp`), the same way any stdio server is declared. The agent's
MCP host owns the process; the VS Code extension is a peer, not the parent.

## Tool surface

**Headless reads** (always available): `discover_flows`, `describe_flow`,
`inspect_step`, `lineage`, `source_context`, `run_summary`, `check_report`,
`experiment_definition`, `experiment_results`. These answer from core and
return structural identity (flow/step ids, paths, sources) and safe summaries.

**Editor-bound** (forwarded to the VS Code window that owns a workspace, over an
authenticated per-window socket): `highlight`, `reveal_source`,
`editor_selection`. These return the window's acceptance, not a round-trip of
editor state.

**Confirmation-gated** (mutating / code-running; the client confirms via the
destructive annotation): `run_flow`, `run_experiment`, `start_debug`,
`generate_ids`.

**Raw, opt-in** (`decider.mcp.rawData`, default off): `raw_record`, `raw_trace`.
A disabled raw tool returns a redaction notice
(`"raw data disabled: enable decider.mcp.rawData to return it"`), never a
fabricated value.

## Workflow

An agent can, without screen scraping:

1. `discover_flows` → list pipelines and their import entries.
2. `describe_flow` (optionally a `subgraph`) → static structure, ids, value slots.
3. `inspect_step` / `lineage` / `source_context` → drill into one step or value.
4. `run_summary` / `check_report` / `experiment_definition` / `experiment_results`
   → tallies, diagnostics and experiment evidence.
5. `highlight` / `reveal_source` → focus the matching UI in VS Code.
6. With confirmation, `run_flow` / `run_experiment` / `start_debug`.

## Capability policy

| class | default | contents |
|---|---|---|
| structural | on | flow/step/edge identity, paths, types, reads/writes, source locations |
| summarised | on | run tallies, check reports, aggregate counts |
| raw | off | per-record values, decision-trace events, raw input rows |

The client environment and the chosen trace adapter determine data
availability; MCP reports unavailable or redacted data plainly rather than
fabricating it. Errors surface as tool errors reusing `decider.exceptions`, not
a new MCP vocabulary.

## Editor bridge security

The MCP process reaches the editor only for the three editor-bound tools, over a
per-window Unix-domain socket (named pipe on Windows) named by the workspace
hash, authenticated by a per-window token in an owner-only directory
(`~/.decider/editor/`, mode 0700). The socket ACL blocks other users; the token
blocks other local processes. Wrong token → `unauthorized`; no window →
`no editor window for '<workspace>'`.
