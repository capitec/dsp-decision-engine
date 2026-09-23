# decider for VS Code / VSCodium (prototype)

Visualise, run and step through decider pipelines.

## Try it

```sh
cd tools/vscode-decider
pnpm install && pnpm build
codium --extensionDevelopmentPath="$PWD" examples/
```

Open `examples/loan.py`. Above `pipeline = ...` there are four lenses:

- **Visualise flow** opens the graph panel and fills the *decider → Structure* tree.
  Nothing runs on data, but the file is imported.
- **Run flow** asks for input rows (the module's `SAMPLE`, a JSON file, or pasted
  JSON) and starts a `decider` debug session that stops on entry.
- **What-if** opens the **Params** tab: every param with its type, default and
  bounds, plus input overrides for one record or all of them. *Compare what-if
  with defaults* runs both and opens **Compare**. *Restart session with these
  params* restarts the debug session with them.
- **Compare with…** picks HEAD, a tag, a branch or a recent commit and runs the
  pipeline as it was there against the working tree, on the same rows.

While a session is running:

| You do | Session does |
|---|---|
| F5 / F10 / F11 / Shift+F11 | `resume` / `step` (runs a branch or loop whole) / `step_into` / out to the end of the enclosing node |
| Breakpoint on a step's `def` line, or a function breakpoint `term/cap_by_income` | `break_at(path)`; a prefix such as `term` also works |
| Breakpoint inside a tree's Python, or a function breakpoint `risk_tree#low_score` | stops at the tree; a locator stops just after it if any row reached that tree node |
| Edit a value in Variables | `set(name, value)`, recorded as `override@<path>` |
| *Restart Frame* on a stack entry | `rewind(path)` |
| *Debug this step in Python* (debug toolbar) | attaches debugpy to the bridge and stops on the first line of the current function; with a record focused, only on that record's call |
| *Trace a column's lineage* | which recorded versions the value came from, so far |
| *Focus one record*, or the record picker in the graph panel | values, lineage and previews for one row instead of the batch |

Each call node has *Inputs*, *Outputs* and *State* scopes, and a row node that has
run has a *Visited* scope with the tree positions its rows reached. Every column
expands to its values, its version chain and its lineage. The graph's **State**
tab is a searchable table that marks what the selected node reads and writes.
Picking a column shows its lineage in the side panel and outlines its producers
in the graph, and its history: every version written so far, with a *rewind*
link that re-runs from that step. With a record focused, a branch's lineage
follows the arm that record took, and selecting a tree shows that record's path
through it.

**Compare** lines two runs up step by step in execution order: each step's
status (same, changed, added, removed), what changed in the step itself (code,
params, reads, writes), which of its outputs differ on which rows, and where
values first diverge. The graph colours the same steps (*diff* toggle).
Revisions are extracted once per commit with `git archive` into a cache under
the system temp directory; the working tree is never touched. The old side
imports its own pipeline code but today's `decider` package.

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
uv run pytest tools/vscode-decider/python -q   # bridge, lineage, traces on the real session
pnpm test                                      # adapter over stdio (DebugClient), layout, comparisons, git
pnpm test:vscode                               # extension API inside VSCodium (mocha)
pnpm test:e2e                                  # Playwright drives VSCodium's UI and takes screenshots
```

`pnpm test:e2e` launches the editor through Playwright's Electron support, with
no window on the desktop (`E2E_HEADED=1` shows it). It clicks the lenses and
quick picks, reaches into the webview's frames, and saves a small JPEG per step
in `test/e2e/shots/`. Those screenshots are how the layout was tuned. Both
editor runs launch `/usr/share/codium/codium`; set `VSCODE_EXE` for another.

## Not built yet

- A comparison's old side runs on today's `decider`; a revision that needs an
  older engine API fails with an import error rather than running.
- Stopping at a breakpoint can open the source in a new editor group next to the
  flow panel.
- `examples/loan.py` defines its own stand-in tree until the tree step lands.
- `session.replace` / `delete` from the editor (T6.5), and the stepped and fused modes.
