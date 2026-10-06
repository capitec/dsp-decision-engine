# The VS Code extension debugs with debugpy, not a decider debug adapter

**Decision:** `ui/vscode` has no debug adapter of its own. "Debug flow"
launches an ordinary `debugpy` session running `python -m decider.debug_bridge.run`,
which runs the pipeline once in interpreted mode. Breakpoints are plain Python
breakpoints in the step functions; the extension only *mirrors* them onto the flow.

The earlier prototype (`tools/vscode-decider-old`) drove a `decider` DAP adapter
over the bridge's `Session`: decider breakpoints on node paths, its own
step/step-into, and a second debugpy attach to get into a step's Python. Two
debuggers, two breakpoint models, and the Python one was the one people reached for.

## How the flow follows the debugger

- **Where it is paused.** At every stop the extension evaluates
  `decider.debug_bridge.run.node_path()` in the paused frame. It walks the stack
  to the interpreted runner's `_call`/`_frame` and returns `node.origin.path`, so a
  function reused at two nodes (a step that is also a branch's condition) still
  maps to the right one. If the evaluate fails, the innermost frame inside a
  step's `[python.line, python.endLine]` is used.
- **Breakpoint dots.** A source breakpoint marks every node whose Python holds
  it. Toggling a dot on the graph or tree adds a breakpoint at the step's
  `bodyLine`; in a reused function it gets the condition
  `node_path() == "<path>"`, so it stops at that node only. Checked against
  debugpy 1.8 over DAP: the conditional breakpoint skips the other use, and
  `evaluate` at the pause returns the path.
- `bodyLine` skips a docstring and `endLine` is the function's last line (it was
  the last statement's first line, used only by the old adapter).
- Tree and table nodes run decider's reference walker, where `justMyCode` never
  stops, so they get no breakpoint actions.

What this gives up: per-node stepping, rewind, value breakpoints, scenarios and
compare from the paused session. They stay in `decider.engine.debug.Session` and
the bridge, for the notebook host and for a future panel action.

## Editing flow files

The Structure tree of a flow file is built from the YAML itself (TypeScript,
`yaml` package), not from `describe`: a flow that doesn't load yet (a new
`step:` naming a function not yet written) still shows and stays editable.
Paths follow the loader's naming (`<branch>_when`, `<loop>_while`, `as:`), so
items join to the described IR by path. Edits go through the `yaml` Document
API, so comments and layout survive, as one `WorkspaceEdit` the user can undo.
