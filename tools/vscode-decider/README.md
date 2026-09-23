# decider for VS Code / VSCodium (prototype)

Visualise, run and step through decider pipelines.

## Try it

```sh
cd tools/vscode-decider
pnpm install && pnpm build
codium --extensionDevelopmentPath="$PWD" examples/
```

Open `examples/loan.py`. Above `pipeline = ...` there are two lenses:

- **Visualise flow** opens the graph panel and fills the *decider → Structure* tree.
  Nothing runs on data, but the file is imported.
- **Run flow** asks for input rows (the module's `SAMPLE`, a JSON file, or pasted
  JSON) and starts a `decider` debug session that stops on entry.

While a session is running:

| You do | Session does |
|---|---|
| F5 / F10 / F11 / Shift+F11 | `resume` / `step` (runs a branch or loop whole) / `step_into` / out to the end of the enclosing node |
| Breakpoint on a step's `def` line, or a function breakpoint `term/cap_by_income` | `break_at(path)`; a prefix such as `term` also works |
| Breakpoint inside a tree's Python, or a function breakpoint `risk_tree#low_score` | stops at the tree; a locator stops just after it if any row reached that tree node |
| Edit a value in Variables | `set(name, value)`, recorded as `override@<path>` |
| *Restart Frame* on a stack entry | `rewind(path)` |
| *Debug this step in Python* (debug toolbar) | attaches debugpy to the bridge and stops on the first line of the current function |
| *Trace a column's lineage* | which recorded versions the value came from, so far |
| *Focus one record*, or the record picker in the graph panel | values, lineage and previews for one row instead of the batch |

Each call node has *Inputs*, *Outputs* and *State* scopes, and a row node that has
run has a *Visited* scope with the tree positions its rows reached. Every column
expands to its values, its version chain and its lineage. The graph's **State**
tab is a searchable table that marks what the selected node reads and writes.
Picking a column shows its lineage in the side panel and outlines its producers
in the graph. With a record focused, a branch's lineage follows the arm that
record took.

## How it fits together

```
extension host ──DAP──▶ adapter.ts ──JSON lines──▶ python/bridge.py ──▶ Session
      │                                    (fd 3)            └─ debugpy.listen (optional)
      └── webview (React + dagre, built by Vite)
```

- The bridge drives the real `decider.engine.debug.Session` in interpreted mode.
  Lineage (`python/lineage.py`) walks the plan's versions: calls, branch merges
  and loop carries.
- Bridge replies use fd 3, so a `print()` inside a step goes to the Debug Console.
- `examples/.vscode/settings.json` sets `decider.python` to `uv run python`, so the
  bridge finds the `decider` package.

## Tests

```sh
uv run pytest tools/vscode-decider/python -q   # bridge and lineage on the real session
pnpm test                                      # adapter over stdio (DebugClient), graph layout
pnpm test:vscode                               # inside a real VSCodium: lenses, panel, session
```

`test:vscode` launches `/usr/share/codium/codium` by default. Set `VSCODE_EXE` to use another editor.

## Not built yet

- Which tree nodes one record visited. `visit()` reports a locator but not the
  row, so the Visited scope counts rows for the whole batch.
- *Debug this step in Python* stops on the first row the step runs, not on the
  focused record.
- `examples/loan.py` defines its own stand-in tree until the tree step lands.
- `session.replace` / `delete` from the editor (T6.5), and the stepped and fused modes.
