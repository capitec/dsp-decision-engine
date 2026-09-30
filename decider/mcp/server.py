"""The `decider`-hosted FastMCP server over stdio.

`build_server` registers the headless read tools and the confirmation-gated
(and raw-gated) actions as FastMCP tools. `run_stdio` serves one over stdio for
the agent's MCP host, as the `decider mcp` command does. Confirmation is
client-side: mutating tools carry a `destructiveHint` annotation; the server
does not re-approve them.
"""
from __future__ import annotations

from fastmcp import FastMCP

from decider.mcp import tools
from decider.mcp.policy import Policy


def build_server(policy: Policy) -> FastMCP:
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
    def check_report(entry: str) -> dict:
        """The default check suite over a flow, as the structured Report."""
        return tools.check_report(entry)

    @mcp.tool
    def experiment_definition(path: str) -> dict:
        """An experiment definition (experiment.yaml or JSON) read from a path."""
        return tools.experiment_definition(path)

    @mcp.tool
    def experiment_results(path: str) -> dict:
        """A persisted experiment result (JSON) read from a path."""
        return tools.experiment_results(path)

    @mcp.tool
    def raw_record(entry: str, data: str, record: int) -> dict:
        """Per-step values for one record (raw data, off until decider.mcp.rawData)."""
        return tools.raw_record(policy, entry, data, record)

    @mcp.tool
    def raw_trace(entry: str, data: str, record: int) -> dict:
        """The decision-trace events for one record (raw data, off until decider.mcp.rawData)."""
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

    return mcp


def run_stdio(raw: bool = False) -> None:
    """Serve the headless MCP server over stdio until the host closes it."""
    build_server(Policy(raw=raw)).run(transport="stdio")
