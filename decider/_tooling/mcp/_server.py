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

import sys
from pathlib import Path
from typing import Any

from decider.exceptions import wrap_import_errors
from decider._tooling.mcp import _configs as configs
from decider._tooling.mcp import _params as params
from decider._tooling.mcp import _scorecards as scorecards
from decider._tooling.mcp import _tables as tables
from decider._tooling.mcp import _tools as tools
from decider._tooling.mcp import _trees as trees
from decider._tooling.mcp._editor import EditorBridge
from decider._tooling.mcp._nonfinite import spelled
from decider._tooling.mcp._policy import REDACTED, Policy

with wrap_import_errors("mcp"):
    from fastmcp import FastMCP


INSTRUCTIONS = (
    "Tools for decider decision pipelines. The `decider` skill has the workflows; read it first. "
    "Edit tree, decision table, scorecard and params documents with these tools rather than by hand: "
    "each edit is validated by building the document and refused, file untouched, if it breaks. "
    "Check an edit with try_config on a worked example. Editor tools need the decider VS Code extension."
)


def build_server(policy: Policy, bridge: EditorBridge | None = None) -> FastMCP:
    mcp = FastMCP(name="decider", instructions=INSTRUCTIONS)

    def tool(fn=None, **options):
        if fn is None:
            return lambda f: tool(f, **options)
        return mcp.tool(spelled(fn), **options)

    @tool
    def discover_flows(root: str = ".") -> dict:
        """List the decider pipelines under a workspace, as `{name, entry, file, line, kind}`."""
        return tools.discover_flows(root)

    @tool
    def describe_flow(entry: str, subgraph: str | None = None) -> dict:
        """The static structure of a flow (or subtree) as a FlowDescription: nodes, edges, value slots."""
        return tools.describe_flow(entry, subgraph)

    @tool
    def inspect_step(entry: str, path: str) -> dict:
        """One node of a flow: its stable identity, source location and touching edges."""
        return tools.inspect_step(entry, path)

    @tool
    def lineage(entry: str, name: str) -> dict:
        """Every slot, writer and reader of a value across a flow."""
        return tools.lineage(entry, name)

    @tool
    def source_context(entry: str, path: str) -> dict:
        """The file, line and a window of source around a step."""
        return tools.source_context(entry, path)

    @tool
    def run_summary(entry: str, data: str) -> dict:
        """Aggregate tallies for a run: row count and per-output value counts (no per-record data)."""
        return tools.run_summary(entry, data)

    @tool
    def check_report(entry: str, config: str | None = None, data: str | None = None) -> dict:
        """The checks over a flow as the structured Report; `config` names a checks file, `data` enables data checks."""
        return tools.check_report(entry, config, data)

    @tool
    def checks_list() -> list[dict]:
        """Every registered check: alias, code, family, needs_data, executes, severity and config schema."""
        return tools.checks_list()

    @tool
    def experiment_definition(path: str) -> dict:
        """An experiment definition (experiment.json) read from a path."""
        return tools.experiment_definition(path)

    @tool
    def experiment_results(path: str) -> dict:
        """A persisted experiment result (JSON) read from a path."""
        return tools.experiment_results(path)

    @tool
    def raw_record(entry: str, data: str, record: int) -> dict:
        """Per-step values for one record (raw data, off until decider._tooling.mcp.rawData)."""
        return tools.raw_record(policy, entry, data, record)

    @tool
    def raw_trace(entry: str, data: str, record: int) -> dict:
        """The decision-trace events for one record (raw data, off until decider._tooling.mcp.rawData)."""
        return tools.raw_trace(policy, entry, data, record)

    @tool(annotations={"destructiveHint": True})
    def run_flow(entry: str, data: str) -> dict:
        """Run a flow end to end; requires client confirmation before it runs."""
        return tools.run_flow(entry, data)

    @tool(annotations={"destructiveHint": True})
    def run_experiment(definition: str, data: str) -> dict:
        """Run an experiment; requires client confirmation before it runs."""
        return tools.run_experiment(definition, data)

    @tool(annotations={"destructiveHint": True})
    def start_debug(entry: str, data: str) -> dict:
        """Open a debug session, paused before anything runs; requires client confirmation."""
        return tools.start_debug(entry, data)

    @tool(annotations={"destructiveHint": True})
    def generate_ids(path: str = ".", check: bool = True) -> dict:
        """Add durable ids to source under a path; requires client confirmation (writes source)."""
        return tools.generate_ids(path, check)

    @tool
    def tree_view(path: str, node: str | None = None) -> dict:
        """A text outline of a tree document (v3 or flat rules) plus its nodes as data; `node` is a node id or rule name to show one subtree."""
        return trees.tree_view(path, node)

    @tool
    def tree_add_node(path: str, parent: str | None, branch: int, data: dict, id: str | None = None) -> dict:
        """Add a v3 node (`data` as in nodes[].data) under `parent` at `branch` (the edge's sourceIndex); parent=None adds the root of an empty tree. Validated before saving."""
        return trees.tree_add_node(path, parent, branch, data, id)

    @tool
    def tree_set_node(path: str, node_id: str, data: dict) -> dict:
        """Replace a v3 node's data, keeping its children; refused if it would strand children on branches the new type lacks."""
        return trees.tree_set_node(path, node_id, data)

    @tool
    def tree_remove_node(path: str, node_id: str) -> dict:
        """Remove a v3 node, its incoming edges and the descendants nothing else leads to; reports what went."""
        return trees.tree_remove_node(path, node_id)

    @tool
    def tree_set_output(path: str, values: dict, index: int | None = None, dtypes: dict[str, str] | None = None) -> dict:
        """Append a result row to a tree's output.data (returns result_idx) or replace row `index` (-1 = default row); `dtypes` declares new columns."""
        return trees.tree_set_output(path, values, index, dtypes)

    @tool
    def tree_add_rule(path: str, rule: dict, at: int | None = None) -> dict:
        """Insert a flat rule `{"meta": {"name": ...}, "rule": {...}}` at position `at` (default the end); first_match order matters."""
        return trees.tree_add_rule(path, rule, at)

    @tool
    def tree_remove_rule(path: str, name: str) -> dict:
        """Remove the flat rule with this `meta.name`; the others keep their order."""
        return trees.tree_remove_rule(path, name)

    # Config documents: every write is validated by building the document, and refused (file untouched) if it breaks.
    @tool
    def list_configs(root: str = ".") -> dict:
        """Config documents under a directory: path, kind (step type or params), name, version dir and key."""
        return configs.list_configs(root)

    @tool
    def read_config(path: str) -> dict:
        """A config document and `problems` (why it would not build; empty when it does)."""
        return configs.read_config(path)

    @tool
    def validate_config(path: str | None = None, doc: dict | None = None) -> dict:
        """`problems` for a config file, or for an unsaved document (give `path` or `doc`)."""
        return configs.validate_config(path, doc)

    @tool
    def write_config(path: str, doc: dict, overwrite: bool = False) -> dict:
        """Create a config document (validated); an existing file is only replaced with `overwrite`."""
        return configs.write_config(path, doc, overwrite)

    @tool
    def patch_config(path: str, ops: list[dict], dry_run: bool = False) -> dict:
        """Apply RFC 6902 JSON Patch `ops` (add, remove, replace, move, copy, test) to a config document and save it if it still builds."""
        return configs.patch_config(path, ops, dry_run)

    @tool
    def try_config(path: str, rows: list[dict], params: dict | None = None) -> dict:
        """Run a config document's step on `rows` (interpreted) and return the output rows; the check after every edit."""
        return configs.try_config(path, rows, params)

    @tool
    def table_view(path: str, offset: int = 0, limit: int = 200) -> dict:
        """A decision table: columns, outputs, default, how records match, and its rows as a grid with indexes."""
        return tables.table_view(path, offset, limit)

    @tool
    def table_add_rows(path: str, rows: list[dict], at: int | None = None) -> dict:
        """Insert rows into a decision table at index `at` (default: the end); saved only if the table still builds."""
        return tables.table_add_rows(path, rows, at)

    @tool
    def table_update_rows(path: str, updates: list[dict]) -> dict:
        """Change table cells: `updates` is `[{"index": 3, "values": {"rate": 0.1}}]`."""
        return tables.table_update_rows(path, updates)

    @tool
    def table_delete_rows(path: str, indexes: list[int]) -> dict:
        """Delete the table rows at these indexes."""
        return tables.table_delete_rows(path, indexes)

    @tool
    def table_add_column(path: str, name: str, dtype: str | dict, value: Any = None, output: bool = False) -> dict:
        """Add a column (polars dtype name, e.g. "Float64") holding `value` in every row; `output` also makes it an output."""
        return tables.table_add_column(path, name, dtype, value, output)

    @tool
    def table_drop_column(path: str, name: str) -> dict:
        """Drop a column from the table and its rows; refused while the match expression uses it."""
        return tables.table_drop_column(path, name)

    @tool
    def scorecard_view(path: str) -> dict:
        """A scorecard: each variable's bins as a table, the default, and the params the card declares."""
        return scorecards.scorecard_view(path)

    @tool
    def scorecard_set_bins(path: str, variable: str, bins: list[dict]) -> dict:
        """Replace one variable's bins in a scorecard; saved only if the scorecard still builds."""
        return scorecards.scorecard_set_bins(path, variable, bins)

    @tool
    def params_view(path: str, entry: str | None = None) -> dict:
        """A params document; with a pipeline `entry` ("pkg.pipeline:build") also its problems and the schema it must meet."""
        return params.params_view(path, entry)

    @tool
    def params_set(path: str, changes: dict[str, Any], entry: str | None = None) -> dict:
        """Set params by slash path (`{"pricing/rate/limit": 0.3}`); with `entry`, refused if it adds a problem for that pipeline."""
        return params.params_set(path, changes, entry)

    if bridge is not None:
        def abs_path(workspace: str, p: str | None) -> str | None:
            return None if p is None else str((Path(workspace) / p).resolve())

        @tool
        def editor_selection(workspace: str = ".") -> dict:
            """What the user is looking at in VS Code: active file, selection, the JSON path and tree node under the cursor, the shown flow, nodes selected in the Structure view and graph, the debugger's pause and breakpoints. `workspace` is any folder the window has open, or a parent of it."""
            return bridge.send(workspace, {"kind": "selection"})

        @tool
        def highlight(nodes: list[str], flow: str | None = None, workspace: str = ".") -> dict:
            """Select and reveal a flow node (path like `credit/limit`) in the Structure view and graph. They hold one selection, so only the first node the flow has is selected. `flow` is a flow file to show first when it is not the one shown."""
            return bridge.send(workspace, {"kind": "highlight", "nodes": nodes, "flow": abs_path(workspace, flow)})

        @tool
        def reveal_source(path: str, line: int, workspace: str = ".") -> dict:
            """Open a file at a 1-based line in the VS Code window."""
            return bridge.send(workspace, {"kind": "reveal", "path": abs_path(workspace, path), "line": line})

        @tool
        def show_flow(file: str, pipeline: str | None = None, workspace: str = ".") -> dict:
            """Show a flow (a flow YAML, or a Python file with `pipeline`) in the Structure view and graph; returns whether it described cleanly."""
            return bridge.send(workspace, {"kind": "show_flow", "file": abs_path(workspace, file), "pipeline": pipeline})

        @tool
        def set_breakpoints(nodes: list[str], workspace: str = ".") -> dict:
            """Make the breakpoints exactly these flow nodes (empty clears them); reports the nodes that could not take one (trees, tables and other non-Python steps)."""
            return bridge.send(workspace, {"kind": "set_breakpoints", "nodes": nodes})

        @tool(annotations={"destructiveHint": True})
        def debug_flow(file: str, pipeline: str | None = None, dataset: str | None = None, workspace: str = ".") -> dict:
            """Run a flow under the debugger in VS Code, on a named dataset or else the sample; waits for the first pause. Requires client confirmation."""
            return bridge.send(workspace, {"kind": "debug_flow", "file": abs_path(workspace, file), "pipeline": pipeline, "dataset": dataset})

        @tool(annotations={"destructiveHint": True})
        def debug_control(action: str, workspace: str = ".") -> dict:
            """Drive the running debug session: continue, step_over, step_in, step_out or stop; waits for the next pause. Requires client confirmation."""
            return bridge.send(workspace, {"kind": "debug_control", "action": action})

        @tool(annotations={"destructiveHint": True})
        def debug_evaluate(expression: str, workspace: str = ".") -> dict:
            """Evaluate a Python expression in the paused frame (raw data, off until `decider mcp --raw`); requires client confirmation."""
            if not policy.raw:
                return {"redacted": REDACTED}
            return bridge.send(workspace, {"kind": "debug_evaluate", "expression": expression})

    return mcp


def run_stdio(raw: bool = False) -> None:
    """Serve the MCP server over stdio until the host closes it."""
    # Entries name modules of the project the host launched us in; a project directory that is
    # itself a package resolves from above it, as `decider serve` finds it.
    here = root = Path.cwd()
    while (root / "__init__.py").exists():
        root = root.parent
    sys.path[:0] = list(dict.fromkeys([str(root), str(here)]))
    build_server(Policy(raw=raw), EditorBridge()).run(transport="stdio", show_banner=False)
