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
| F5 / F10 / F11 | `resume` / `step` (over groups) / `step_into` |
| Breakpoint on a step's `def` line, or a function breakpoint `term/cap_by_income` | `break_at(path)`; a prefix such as `term` also works |
| Edit a value in Variables | `set(name, value)`, recorded as `override@<path>` |
| *Restart Frame* on a stack entry | `rewind(path)` |
| *Debug this step in Python* (debug toolbar) | attaches debugpy to the bridge and stops on the first line of the current function |
| *Trace a column's lineage* | static backward slice from the current node |

Each call node has *Inputs*, *Outputs* and *State* scopes. Every column expands to
its values, its version chain and its lineage. The graph's **State** tab is a
searchable table that marks what the selected node reads and writes.

## How it fits together

```
extension host ──DAP──▶ adapter.ts ──JSON lines──▶ python/bridge.py ──▶ Session
      │                                    (fd 3)            └─ debugpy.listen (optional)
      └── webview (React + dagre, built by Vite)
```

- `python/decider_stub.py` stands in for `decider.steps` and `decider.engine.debug`
  until T1.1 to T1.5 land. The bridge only uses `to_ir`, `Session`, `State` and
  origins, so the switch is one import.
- Bridge replies use fd 3, so a `print()` inside a step goes to the Debug Console.

## Tests

```sh
uv run pytest tools/vscode-decider/python -q   # bridge and stub
pnpm test                                      # adapter over stdio (DebugClient), graph layout
pnpm test:vscode                               # inside a real VSCodium: lenses, panel, session
```

`test:vscode` launches `/usr/share/codium/codium` by default. Set `VSCODE_EXE` to use another editor.

## Not built yet

- Frame, loop and row (tree) nodes. Stepping into a tree's reference walker needs
  `visit()` events from the real Session.
- Per-record filtering. The session runs the whole batch, as Design.md Q1 says.
- Runtime lineage. Lineage is static: it lists the last writer of each input.
  Version chains carry the runtime side.
- `session.replace` / `delete` from the editor (T6.5).
