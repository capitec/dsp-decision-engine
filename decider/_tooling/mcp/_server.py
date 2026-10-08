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

from typing import Any

from decider.exceptions import wrap_import_errors
from decider._tooling.mcp import _configs as configs
from decider._tooling.mcp import _params as params
from decider._tooling.mcp import _scorecards as scorecards
from decider._tooling.mcp import _tables as tables
from decider._tooling.mcp import _tools as tools
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

    # Config documents: every write is validated by building the document, and refused (file untouched) if it breaks.
    @mcp.tool
    def list_configs(root: str = ".") -> dict:
        """Config documents under a directory: path, kind (step type or params), name, version dir and key."""
        return configs.list_configs(root)

    @mcp.tool
    def read_config(path: str) -> dict:
        """A config document and `problems` (why it would not build; empty when it does)."""
        return configs.read_config(path)

    @mcp.tool
    def validate_config(path: str | None = None, doc: dict | None = None) -> dict:
        """`problems` for a config file, or for an unsaved document (give `path` or `doc`)."""
        return configs.validate_config(path, doc)

    @mcp.tool
    def write_config(path: str, doc: dict, overwrite: bool = False) -> dict:
        """Create a config document (validated); an existing file is only replaced with `overwrite`."""
        return configs.write_config(path, doc, overwrite)

    @mcp.tool
    def patch_config(path: str, ops: list[dict], dry_run: bool = False) -> dict:
        """Apply RFC 6902 JSON Patch `ops` (add, remove, replace, move, copy, test) to a config document and save it if it still builds."""
        return configs.patch_config(path, ops, dry_run)

    @mcp.tool
    def try_config(path: str, rows: list[dict], params: dict | None = None) -> dict:
        """Run a config document's step on `rows` (interpreted) and return the output rows; the check after every edit."""
        return configs.try_config(path, rows, params)

    @mcp.tool
    def table_view(path: str, offset: int = 0, limit: int = 200) -> dict:
        """A decision table: columns, outputs, default, how records match, and its rows as a grid with indexes."""
        return tables.table_view(path, offset, limit)

    @mcp.tool
    def table_add_rows(path: str, rows: list[dict], at: int | None = None) -> dict:
        """Insert rows into a decision table at index `at` (default: the end); saved only if the table still builds."""
        return tables.table_add_rows(path, rows, at)

    @mcp.tool
    def table_update_rows(path: str, updates: list[dict]) -> dict:
        """Change table cells: `updates` is `[{"index": 3, "values": {"rate": 0.1}}]`."""
        return tables.table_update_rows(path, updates)

    @mcp.tool
    def table_delete_rows(path: str, indexes: list[int]) -> dict:
        """Delete the table rows at these indexes."""
        return tables.table_delete_rows(path, indexes)

    @mcp.tool
    def table_add_column(path: str, name: str, dtype: str | dict, value: Any = None, output: bool = False) -> dict:
        """Add a column (polars dtype name, e.g. "Float64") holding `value` in every row; `output` also makes it an output."""
        return tables.table_add_column(path, name, dtype, value, output)

    @mcp.tool
    def table_drop_column(path: str, name: str) -> dict:
        """Drop a column from the table and its rows; refused while the match expression uses it."""
        return tables.table_drop_column(path, name)

    @mcp.tool
    def scorecard_view(path: str) -> dict:
        """A scorecard: each variable's bins as a table, the default, and the params the card declares."""
        return scorecards.scorecard_view(path)

    @mcp.tool
    def scorecard_set_bins(path: str, variable: str, bins: list[dict]) -> dict:
        """Replace one variable's bins in a scorecard; saved only if the scorecard still builds."""
        return scorecards.scorecard_set_bins(path, variable, bins)

    @mcp.tool
    def params_view(path: str, entry: str | None = None) -> dict:
        """A params document; with a pipeline `entry` ("pkg.pipeline:build") also its problems and the schema it must meet."""
        return params.params_view(path, entry)

    @mcp.tool
    def params_set(path: str, changes: dict[str, Any], entry: str | None = None) -> dict:
        """Set params by slash path (`{"pricing/rate/limit": 0.3}`); with `entry`, refused if it adds a problem for that pipeline."""
        return params.params_set(path, changes, entry)

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
