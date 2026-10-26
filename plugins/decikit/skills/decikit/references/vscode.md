# The VS Code window: what the user sees, navigation, debugging

Use this when the user points at something in their editor ("this node", "the selected rule", "why does this one stop"), when you want to show them a node or file, or when a bug needs a debugger. Every tool here talks to the decikit VS Code extension in the user's open window; none works headless.

## What the extension shows

- **Flows** view: every flow in the workspace (`flow.yaml` files and Python pipelines).
- **Structure** view and the **graph** panel: the shown flow's nodes, addressed by path like `credit/affordable/eligible`. Breakpoint dots and the paused node are marked on both.
- **Datasets** view: the datasets of the shown flow's project.
- Text editors, including config documents (trees, tables, scorecards, params) as JSON or YAML.

## Find out what "this" is before acting

Call `editor_selection()` first whenever the user says "this", "here", "the selected ...". It returns:

- `file`, `selection` (1-based `start`/`end` line and column, and the selected `text`): the active editor.
- `config`: for a `.json`/`.yaml` file, `path` (keys and indexes from the document root to the cursor), `type`, and `nodeId` (a tree's node id, or a flat rule set's rule name) when the cursor is inside a tree node or rule. `null` for other files.
- `flow`: the flow in the Structure view (`file`, `pipeline`, `described`, `error`), or `null`.
- `structure_selection` and `graph_selection`: node paths the user selected in the Structure view and the graph.
- `debug`: `paused` (node path), `frame` (`file`, `line`, `name` of the innermost frame), `breakpoints` (node paths).

Which field answers which question: "this rule/node" in a config file is `config.nodeId` (then edit it through the tree tools by that id); "this step" in a flow is `structure_selection` or `graph_selection`; "where it stopped" is `debug.paused`. If the field you need is empty, ask the user to click the thing; do not guess from another field.

```
editor_selection()
-> {"file": "/p/configs/0.1.0/grade.json", "selection": {"start": {"line": 9, "column": 13}, "end": {"line": 9, "column": 17}, "text": "leaf"},
    "config": {"path": ["tree", "nodes", 1, "id"], "type": "tree", "nodeId": "leaf"},
    "flow": {"file": "/p/flow.yaml", "described": true}, "structure_selection": ["credit/limit"],
    "graph_selection": null, "debug": {"paused": null, "frame": null, "breakpoints": []}}
```

## Navigate

- `show_flow(file, pipeline=None)` opens a flow in the Structure view and graph and reports whether it described (`described`, `error`). A Python file needs `pipeline`.
- `highlight(nodes, flow=None)` selects a node path in the Structure view and graph. They hold one selection, so only the first node the flow has is selected; the rest come back as `skipped`. `flow` is a file to show first.
- `reveal_source(path, line)` opens a file at a 1-based line. Use it for the code behind a node (`source_context` gives the file and line).

Relative file paths resolve against `workspace`. Use `highlight` after you explain a node ("this is the step I mean"), and `reveal_source` after you edit a file the user should look at.

## Debug a flow

Debugging runs the flow's Python under debugpy; a node with a Python function takes a breakpoint, a tree, table or scorecard step does not (`not_set` lists those).

1. `show_flow(file)` and check `described` is true.
2. `set_breakpoints(nodes)` makes the breakpoints exactly these node paths (empty list clears them). Read `not_set`.
3. `debug_flow(file, dataset=None)` starts the run on the named dataset (else the sample) and waits for the first pause: `paused` is the node path, or `null` when the run ended without stopping.
4. Read the situation with `editor_selection()`: `debug.paused` and `debug.frame`.
5. `debug_control(action)` with `continue`, `step_over`, `step_in`, `step_out` or `stop`; each waits for the next pause and returns it as `paused`.
6. `debug_evaluate(expression)` runs a Python expression in the paused frame, for example `income * 2` or a column of the current row. It needs raw data enabled (below).
7. Fix the cause in the source or config, then run again from step 3; stop the session with `debug_control("stop")` when done.

The three debugging tools run code and ask the user to confirm. A `debug_control` that returns `paused: null` means the run finished (or the debugger stopped outside any node: check `debug.frame`).

## Enable raw data

`debug_evaluate` returns record values, so it answers `{"redacted": "raw data disabled: ..."}` until the MCP server starts with `decikit mcp --raw`, or with `DECIKIT_MCP_RAW=1` in the server's environment (the plugin's MCP config can set it). Tell the user to enable it; do not look for a way around it.

## When no window answers

`workspace` defaults to `.`, the MCP server's working directory. The window is the one whose folder is that path or its nearest parent; failing that, the single window open on a folder below it. The error names the paths tried.

- "no editor window ... tried ...": the extension is not running with that folder open. Ask the user to open the project folder in VS Code (with the decikit extension installed), then retry. If they have several windows, pass `workspace=` the exact folder.
- "N VS Code windows are open under ...": pass `workspace=` one of the listed folders.
- "dead socket": the window crashed or closed; ask the user to reopen the folder.
- `unauthorized` or "unreachable": ask them to run "Developer: Reload Window" in VS Code.

Without a window, fall back to the headless tools: `describe_flow`, `source_context`, `run_summary`, and ask the user to tell you the node.

## Pitfalls

- Node paths are the flow's (`credit/limit`), not file paths; take them from `describe_flow` or `editor_selection`.
- `set_breakpoints` replaces the breakpoints the user set by hand; read `debug.breakpoints` first if they may have their own.
- `highlight` before `show_flow` fails with "no flow is shown".
- Stepping is slow when the run is large: debug on a small `dataset`.

## Tools

- `editor_selection(workspace=".")`: what the user is looking at: file, selection, config path and node id, shown flow, Structure/graph selection, debugger pause and breakpoints
- `highlight(nodes, flow=None, workspace=".")`: select and reveal a flow node in the Structure view and graph
- `reveal_source(path, line, workspace=".")`: open a file at a line
- `show_flow(file, pipeline=None, workspace=".")`: show a flow in the Structure view and graph
- `set_breakpoints(nodes, workspace=".")`: make the breakpoints exactly these flow nodes
- `debug_flow(file, pipeline=None, dataset=None, workspace=".")`: run a flow under the debugger, wait for the first pause
- `debug_control(action, workspace=".")`: continue, step_over, step_in, step_out or stop; wait for the next pause
- `debug_evaluate(expression, workspace=".")`: evaluate a Python expression in the paused frame (needs raw data)
- `describe_flow(entry, subgraph=None)`: node paths and structure of a flow, to pick nodes from
- `source_context(entry, path)`: the file and line behind a node, for `reveal_source`
