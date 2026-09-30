# Experiment 04 findings — FastMCP topology and editor bridge

**Run during:** task 11, before MCP transport and lifecycle implementation freezes
**Feeds:** tasks 11, 12

## Question and answer in one line

A `decider`-hosted FastMCP process — owned by the agent's MCP host, spawned as
`uv run decider mcp` over stdio — serves headless core tools itself and reaches
the correct VS Code window through a per-window Unix-domain socket authenticated
by a per-window secret token, with raw record/trace data gated off by default.

## 1. MCP process ownership, lifecycle, and transport

**Recommendation:** the **agent's MCP host owns the process**, not the
extension. The extension is a *peer*, not the parent. The server is a new
`decider mcp` command that runs FastMCP over **stdio**; the agent declares it in
its MCP config the way any stdio server is declared.

- **Transport = stdio.** The agent ↔ server channel is the standard MCP stdio
  JSON-RPC. Prototype confirmed it end-to-end against FastMCP 4.0.10:
  `initialize` → `notifications/initialized` → `tools/list` → `tools/call` over
  a real subprocess, with a clean terminate. No localhost listener is needed for
  the *agent* channel at all.
- **Lifecycle = one process per agent session, stateless per call.** Each tool
  call discovers/loads what it needs and returns; nothing is cached across
  calls except the process itself. This is exactly the pattern
  `Bridge.describe`/`start` already use (`tools/decider-bridge/decider_bridge/bridge.py`),
  so a long-lived MCP process does not become a long-lived debug session: a
  debug session is opened and closed by a confirmation-gated tool, not by the
  server's lifetime.
- **Ownership.** The extension must not spawn the MCP server, or the two
  editor-bound consumers (VS Code, JupyterLab) and the agent would each get a
  different process and the workspace routing in §2 would have no single
  authority. The agent's host spawning `uv run decider mcp` gives one process,
  one cwd (= the workspace), one routing source of truth.
- **`fastmcp` is not currently a dependency.** It was added ephemerally for this
  spike via `uv run --with fastmcp` (resolved 4.0.10, protocol version
  `2025-03-26`). Task 11 must add `fastmcp` to `pyproject.toml` and register the
  `decider mcp` subcommand; until then the prototype runs only with `--with`.

## 2. Editor-bridge authentication and workspace/window routing

**Recommendation:** one **Unix-domain socket per VS Code window, named by the
workspace folder, plus a per-window secret token**, both in a machine-local,
owner-only directory. The workspace folder is the routing key; the token is the
credential. No unauthenticated localhost listener.

- **The bridge is a second hop.** The agent talks to the MCP process over stdio;
  the MCP process talks to the extension only for editor-bound actions
  (highlight/reveal/selection). This is the one channel where "read broad by
  default" meets a live editor, so it is the channel that must be authenticated
  (the R4 threat note): never assume the agent is the only local actor.
- **Addressing/routing.** The extension binds
  `~/.decider/editor/<sha256(workspace)>.sock` and writes a fresh random token to
  `~/.decider/editor/<sha256(workspace)>.token` (both `0600`). The MCP process
  knows its workspace from its own cwd, hashes it the same way, and connects.
  Two windows = two sockets = two tokens; a request routes to exactly the window
  for the named workspace. Prototype confirmed `/w/a` and `/w/b` stay isolated.
- **Authentication.** A caller must present the token read from the token file;
  a wrong token is refused (`unauthorized`), a workspace with no window is
  refused with a routing error naming the workspace (`no editor window for
  '<workspace>'`). The socket's `0600` filesystem ACL already blocks other
  *users*; the token blocks other *processes of the same user*. The token is
  rotated on each window open, so a stale credential dies with its window.
- **Windows note.** Unix-domain sockets do not exist on Windows; the extension
  side falls back to a named pipe with the same token handshake. The routing and
  auth contract is identical; only the transport primitive differs.

## 3. The headless/editor-bound tool split

**Recommendation:** split on *"does the answer live in `decider` core alone?"*.

- **Headless** (served by the MCP process from core, no editor):
  `discover_flows`, `describe_flow`, `inspect_step`/`inspect_value`, `lineage`,
  `source_context`, `run_summary`, `check_report`, `experiment_definition`,
  `experiment_results`, and headless debug open/step via the relocated bridge
  session. These bind to `engine/ir` (`to_ir`, `step_map`, `Origin`),
  `engine/run` (`RunReport`), `testing/`, and `engine.debug.Session`.
- **Editor-bound** (forwarded to the resolved window, return an ack):
  `highlight`, `reveal_source`, `editor_selection`. These return the window's
  acceptance (`{ok, ack}`) — never a round-trip of editor state — because the
  MCP process cannot synchronously round-trip the editor's own state through the
  hop without the agent, and nothing in the workflow needs it to.
- **Confirmation-gated** (a third axis, orthogonal to the split): running a
  flow, starting the debugger, running an experiment, generating IDs/source,
  and any persistence. These carry a destructive annotation (FastMCP
  `destructiveHint`); confirmation is a *client-side* gate driven by that
  annotation plus policy — the server does not implement a second approval
  mechanism.

## 4. Capability policy

**Recommendation:** three data classes, gated by a workspace setting.

| Class | Default | Contents | Gate |
|---|---|---|---|
| structural | on (always) | flow/step/edge identity, paths, types, reads/writes, source locations, execution order | none |
| summarised | on | run summaries, check reports, aggregate counts/tallies, path counts | none |
| raw | **off** | per-record values, decision-trace events, raw input rows | `decider.mcp.rawData = true` |

- Raw is the opt-in that encodes B6/R9: the extension's trace values live in
  session memory only, and the workspace setting can disable raw-data MCP tools
  while structural metadata and safe summaries remain available. The default is
  off; the spike's server answered redacted by default and returned values only
  after opting in.
- **Disabled raw tools return a redaction notice, never fabricated data**
  (`"raw data disabled: enable decider.mcp.rawData to return it"`). This is the
  "report unavailable/redacted plainly rather than fabricate" rule from task
  11's important decisions.
- **Error reporting** reuses `decider.exceptions` (`DeciderError` and
  subclasses) surfaced as MCP tool errors with the same codes the serving layer
  uses, rather than a new MCP error vocabulary (feedback M-minor).

## What was run vs. assessed

- **Run:** `uv run --with fastmcp python notes/vscode-redesign/experimentation/04-fastmcp/prototype.py`.
  It exercises capability discovery (`list_tools`), the headless/editor-bound
  split, structural/summarised access, raw gating both ways, two-window routing,
  token rejection, missing-window routing, the destructive annotation, and a real
  stdio subprocess handshake with clean terminate. All pass.
- **Assessed from knowledge (not run):** the real bindings to `engine/ir`,
  `engine/run`, `testing/`, and the debug session — the prototype uses a minimal
  fake core so the spike stays numba-free and reproducible. The Windows named-
  pipe fallback and the extension-side listener implementation are described, not
  built.

## Plan changes

- **Task 11** gains two concrete deliverables it does not currently name:
  (a) add `fastmcp` as a dependency and register a `decider mcp` CLI subcommand
  (stdlib-adjacent stdio, no new transport code); (b) implement the per-window
  socket+token editor bridge in the extension and the MCP-process client, with
  the token-in-owner-only-file auth. This is the R4/R7 fold the feedback asked
  for.
- The capability policy setting (`decider.mcp.rawData`, default off) is task
  11's raw-data-disable deliverable, already implied by B6/R9 but now pinned.
- No change to the dependency order: the spike confirms the direction (F1/F2,
  Q19/Q20) without revising task 11's block on this experiment.
