# decider for VS Code

See, edit and debug decider flows: Python pipelines and YAML flow files.

## Install

```sh
cd ui/vscode          # Node 22 (see .nvmrc)
pnpm install && pnpm package     # writes decider-vscode-0.1.0.vsix
code --install-extension decider-vscode-0.1.0.vsix
```

Or press F5 in this folder to run it in an Extension Development Host on `examples/`.

The extension needs `decider` importable by the project's Python. It runs the
`decider.python` setting (e.g. `["uv", "run", "python"]`), else the ms-python
interpreter, else `python3`. `examples/.vscode/settings.json` sets `uv run python`.

## What it does

**decider** in the activity bar has three views.

- **Flows**: every `flow.yaml` (a top-level `flow:` and `steps:`), and every
  Python file that imports decider and defines `def build()`,
  `pipeline = flow(...)` or a `*_pipeline()` factory. Click one to show it; ▷
  debugs it. CodeLenses above each pipeline do the same.
- **Structure**: the shown flow as a tree, with an icon per node type. Click a
  node to open where it is written (the YAML line, or the Python) and select it in
  the graph. A step has *Go to step code* and *Toggle breakpoint*.
- **Toolbox** (flow files only): Step, Branch, Loop, Flow, Subflow. Drag one onto
  the Structure: onto a flow, loop or arm it goes inside; onto a step, after it;
  onto a branch, you pick the arm. The **+** inserts at the node selected in the
  Structure. The right-click menu in the Structure has *Insert before / after*,
  *Add inside* and *Delete*.

Inserting asks for what the node needs: a step from the flow's `components`
(and its config's documents), a condition (a step, a bool column or an
`expr`), `modifies`/`carries`, `max`. A new Subflow can write
`flows/<name>.yaml`. Edits keep comments and layout and can be undone.

The **graph** (@decider/ui) opens beside the source: reads and writes, groups that
fold, and an inspector per node.

## Debugging

*Debug flow* runs the pipeline once under the normal Python debugger (debugpy),
in plain Python, on the module's `SAMPLE` / `sample_request.json` or a file you
pick. Set ordinary breakpoints in your step functions.

- A breakpoint shows as a dot on every node whose function holds it, in the
  graph and the Structure. Clicking a dot (or *Toggle breakpoint*) adds or
  removes one at the step's first line. In a function used at two nodes the
  breakpoint is conditional, so it stops at that node only.
- When the debugger pauses, the node it is in is highlighted and selected,
  even inside helpers the step calls. Step, continue, Variables and the Debug
  Console are the Python debugger's own.

Your own launch configuration works too:

```json
{ "type": "debugpy", "request": "launch", "name": "decider: credit",
  "module": "decider._tooling.bridge.run", "args": ["${workspaceFolder}/credit_flow/flow.yaml"] }
```

Any Python debug session that pauses inside the shown flow's steps is mirrored,
including your own scripts and tests, as long as the pipeline runs interpreted
(the default for `pipeline.run`).

## Tests

```sh
uv run pytest tests/bridge -q    # describe of flow files, the runner, node_path
pnpm test                              # YAML outline and edits, breakpoint/pause mapping
pnpm test:vscode                       # inside VS Code: show, breakpoint, debug, pause mapping
```

## Not built

- Moving nodes by drag within the Structure.
- Writing a stub function for a new step name into `components`.
- JSON flow files load and edit, but are not discovered or given CodeLenses.
- Per-node stepping, rewind and what-if from a paused session (see
  `docs/design_decisions/decisions/debugger-and-editors.md`).
