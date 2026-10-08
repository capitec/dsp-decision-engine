"""The `decider`-hosted FastMCP server over stdio.

`build_server` registers the headless read tools and the confirmation-gated
(and raw-gated) actions as FastMCP tools; when given an editor bridge it also
registers the editor-bound selection, navigation and debug tools, which forward
to the VS Code window that owns a workspace. `run_stdio` serves one over stdio for
the agent's MCP host, as the `decider mcp` command does. Confirmation is
client-side: mutating tools carry a `destructiveHint` annotation; the server
does not re-approve them.
"""
from __future__ import annotations

from pathlib import Path

from decider.exceptions import wrap_import_errors
from decider._tooling.mcp import _tools as tools
from decider._tooling.mcp._editor import EditorBridge
from decider._tooling.mcp._policy import REDACTED, Policy

with wrap_import_errors("mcp"):
    from fastmcp import FastMCP


def build_server(policy: Policy, bridge: EditorBridge | None = None) -> FastMCP:
    mcp = FastMCP(name="decider")

    @mcp.tool
    def discover_flows(root: str = ".") -> dict:
        """List the decider pipelines under a workspace, as `{name, entry, file, line, kind}`."""
        return tools.discover_flows(root)

    @mcp.tool
    def describe_flow(entry: str, subgraph: str | None = None) -> dict:
        """The static structure of a flow (or subtree) as a FlowDescription: nodes, edges, value slots."""
        return tools.describe_flow(entry, subgraph)

    @mcp.tool
    def inspect_step(entry: str, path: str) -> dict:
        """One node of a flow: its stable identity, source location and touching edges."""
        return tools.inspect_step(entry, path)

    @mcp.tool
    def lineage(entry: str, name: str) -> dict:
        """Every slot, writer and reader of a value across a flow."""
        return tools.lineage(entry, name)

    @mcp.tool
    def source_context(entry: str, path: str) -> dict:
        """The file, line and a window of source around a step."""
        return tools.source_context(entry, path)

    @mcp.tool
    def run_summary(entry: str, data: str) -> dict:
        """Aggregate tallies for a run: row count and per-output value counts (no per-record data)."""
        return tools.run_summary(entry, data)

    @mcp.tool
    def check_report(entry: str, config: str | None = None, data: str | None = None) -> dict:
        """The checks over a flow as the structured Report; `config` names a checks file, `data` enables data checks."""
        return tools.check_report(entry, config, data)

    @mcp.tool
    def checks_list() -> list[dict]:
        """Every registered check: alias, code, family, needs_data, executes, severity and config schema."""
        return tools.checks_list()

    @mcp.tool
    def experiment_definition(path: str) -> dict:
        """An experiment definition (experiment.json) read from a path."""
        return tools.experiment_definition(path)

    @mcp.tool
    def experiment_results(path: str) -> dict:
        """A persisted experiment result (JSON) read from a path."""
        return tools.experiment_results(path)

    @mcp.tool
    def raw_record(entry: str, data: str, record: int) -> dict:
        """Per-step values for one record (raw data, off until decider._tooling.mcp.rawData)."""
        return tools.raw_record(policy, entry, data, record)

    @mcp.tool
    def raw_trace(entry: str, data: str, record: int) -> dict:
        """The decision-trace events for one record (raw data, off until decider._tooling.mcp.rawData)."""
        return tools.raw_trace(policy, entry, data, record)

    @mcp.tool(annotations={"destructiveHint": True})
    def run_flow(entry: str, data: str) -> dict:
        """Run a flow end to end; requires client confirmation before it runs."""
        return tools.run_flow(entry, data)

    @mcp.tool(annotations={"destructiveHint": True})
    def run_experiment(definition: str, data: str) -> dict:
        """Run an experiment; requires client confirmation before it runs."""
        return tools.run_experiment(definition, data)

    @mcp.tool(annotations={"destructiveHint": True})
    def start_debug(entry: str, data: str) -> dict:
        """Open a debug session, paused before anything runs; requires client confirmation."""
        return tools.start_debug(entry, data)

    @mcp.tool(annotations={"destructiveHint": True})
    def generate_ids(path: str = ".", check: bool = True) -> dict:
        """Add durable ids to source under a path; requires client confirmation (writes source)."""
        return tools.generate_ids(path, check)

    if bridge is not None:
        def abs_path(workspace: str, p: str | None) -> str | None:
            return None if p is None else str((Path(workspace) / p).resolve())

        @mcp.tool
        def editor_selection(workspace: str = ".") -> dict:
            """What the user is looking at in VS Code: active file, selection, the JSON path and tree node under the cursor, the shown flow, nodes selected in the Structure view and graph, the debugger's pause and breakpoints. `workspace` is any folder the window has open, or a parent of it."""
            return bridge.send(workspace, {"kind": "selection"})

        @mcp.tool
        def highlight(nodes: list[str], flow: str | None = None, workspace: str = ".") -> dict:
            """Select and reveal a flow node (path like `credit/limit`) in the Structure view and graph. They hold one selection, so only the first node the flow has is selected. `flow` is a flow file to show first when it is not the one shown."""
            return bridge.send(workspace, {"kind": "highlight", "nodes": nodes, "flow": abs_path(workspace, flow)})

        @mcp.tool
        def reveal_source(path: str, line: int, workspace: str = ".") -> dict:
            """Open a file at a 1-based line in the VS Code window."""
            return bridge.send(workspace, {"kind": "reveal", "path": abs_path(workspace, path), "line": line})

        @mcp.tool
        def show_flow(file: str, pipeline: str | None = None, workspace: str = ".") -> dict:
            """Show a flow (a flow YAML, or a Python file with `pipeline`) in the Structure view and graph; returns whether it described cleanly."""
            return bridge.send(workspace, {"kind": "show_flow", "file": abs_path(workspace, file), "pipeline": pipeline})

        @mcp.tool
        def set_breakpoints(nodes: list[str], workspace: str = ".") -> dict:
            """Make the breakpoints exactly these flow nodes (empty clears them); reports the nodes that could not take one (trees, tables and other non-Python steps)."""
            return bridge.send(workspace, {"kind": "set_breakpoints", "nodes": nodes})

        @mcp.tool(annotations={"destructiveHint": True})
        def debug_flow(file: str, pipeline: str | None = None, dataset: str | None = None, workspace: str = ".") -> dict:
            """Run a flow under the debugger in VS Code, on a named dataset or else the sample; waits for the first pause. Requires client confirmation."""
            return bridge.send(workspace, {"kind": "debug_flow", "file": abs_path(workspace, file), "pipeline": pipeline, "dataset": dataset})

        @mcp.tool(annotations={"destructiveHint": True})
        def debug_control(action: str, workspace: str = ".") -> dict:
            """Drive the running debug session: continue, step_over, step_in, step_out or stop; waits for the next pause. Requires client confirmation."""
            return bridge.send(workspace, {"kind": "debug_control", "action": action})

        @mcp.tool(annotations={"destructiveHint": True})
        def debug_evaluate(expression: str, workspace: str = ".") -> dict:
            """Evaluate a Python expression in the paused frame (raw data, off until `decider mcp --raw`); requires client confirmation."""
            if not policy.raw:
                return {"redacted": REDACTED}
            return bridge.send(workspace, {"kind": "debug_evaluate", "expression": expression})

    return mcp


def run_stdio(raw: bool = False) -> None:
    """Serve the MCP server over stdio until the host closes it."""
    build_server(Policy(raw=raw), EditorBridge()).run(transport="stdio")
