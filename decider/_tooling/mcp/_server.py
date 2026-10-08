"""The `decider`-hosted FastMCP server over stdio.

`build_server` registers the headless read tools and the confirmation-gated
(and raw-gated) actions as FastMCP tools; when given an editor bridge it also
registers the editor-bound highlight/reveal/selection tools, which forward to
the VS Code window that owns a workspace. `run_stdio` serves one over stdio for
the agent's MCP host, as the `decider mcp` command does. Confirmation is
client-side: mutating tools carry a `destructiveHint` annotation; the server
does not re-approve them.
"""
from __future__ import annotations

from decider.exceptions import wrap_import_errors
from decider._tooling.mcp import _tools as tools
from decider._tooling.mcp import _trees as trees
from decider._tooling.mcp._editor import EditorBridge
from decider._tooling.mcp._policy import Policy

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

    @mcp.tool
    def tree_view(path: str, node: str | None = None) -> dict:
        """A text outline of a tree document (v3 or flat rules) plus its nodes as data; `node` is a node id or rule name to show one subtree."""
        return trees.tree_view(path, node)

    @mcp.tool(annotations={"destructiveHint": True})
    def tree_add_node(path: str, parent: str | None, branch: int, data: dict, id: str | None = None) -> dict:
        """Add a v3 node (`data` as in nodes[].data) under `parent` at `branch` (the edge's sourceIndex); parent=None adds the root of an empty tree. Validated before saving."""
        return trees.tree_add_node(path, parent, branch, data, id)

    @mcp.tool(annotations={"destructiveHint": True})
    def tree_set_node(path: str, node_id: str, data: dict) -> dict:
        """Replace a v3 node's data, keeping its children; refused if it would strand children on branches the new type lacks."""
        return trees.tree_set_node(path, node_id, data)

    @mcp.tool(annotations={"destructiveHint": True})
    def tree_remove_node(path: str, node_id: str) -> dict:
        """Remove a v3 node, its incoming edges and the descendants nothing else leads to; reports what went."""
        return trees.tree_remove_node(path, node_id)

    @mcp.tool(annotations={"destructiveHint": True})
    def tree_set_output(path: str, values: dict, index: int | None = None, dtypes: dict[str, str] | None = None) -> dict:
        """Append a result row to a tree's output.data (returns result_idx) or replace row `index` (-1 = default row); `dtypes` declares new columns."""
        return trees.tree_set_output(path, values, index, dtypes)

    @mcp.tool(annotations={"destructiveHint": True})
    def tree_add_rule(path: str, rule: dict, at: int | None = None) -> dict:
        """Insert a flat rule `{"meta": {"name": ...}, "rule": {...}}` at position `at` (default the end); first_match order matters."""
        return trees.tree_add_rule(path, rule, at)

    @mcp.tool(annotations={"destructiveHint": True})
    def tree_remove_rule(path: str, name: str) -> dict:
        """Remove the flat rule with this `meta.name`; the others keep their order."""
        return trees.tree_remove_rule(path, name)

    if bridge is not None:
        @mcp.tool
        def highlight(workspace: str, flow: str, nodes: list[str]) -> dict:
            """Focus these flow entities (node paths) in the VS Code window for `workspace`."""
            return bridge.send(workspace, {"kind": "highlight", "flow": flow, "nodes": nodes})

        @mcp.tool
        def reveal_source(workspace: str, path: str, line: int) -> dict:
            """Navigate the VS Code window for `workspace` to a flow entity's source."""
            return bridge.send(workspace, {"kind": "reveal", "path": path, "line": line})

        @mcp.tool
        def editor_selection(workspace: str) -> dict:
            """The VS Code window's current editor selection (file, position, text)."""
            return bridge.send(workspace, {"kind": "selection"})

    return mcp


def run_stdio(raw: bool = False) -> None:
    """Serve the MCP server over stdio until the host closes it."""
    build_server(Policy(raw=raw), EditorBridge()).run(transport="stdio")
