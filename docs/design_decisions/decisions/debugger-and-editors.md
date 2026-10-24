# Debugger and editors

Read this before changing the debug `Session`, the editor bridge, the VS Code extension or the shared flow UI: it records why each is shaped as it is.

## The session is built only on the runner protocol

- `Session` drives `runner.iterate` one checkpoint at a time and assumes nothing about which checkpoints a runner yields; fused mode yields one pair per kernel.
- A breakpoint on any step of a fused kernel pauses before the whole kernel (`Paused.kernel` lists the steps). Not: calling it unreachable; nothing of the kernel has run and `set` on its inputs works. Setting a value an earlier step of the same kernel reads therefore differs from interpreted.
- See: `decikit/_engine/debug/session.py`, [Debug sessions](../../architecture/engine/debug.md)

## Stepping is over the batch, not per row

- One step is one node over every row that reaches it; a branch is entered arm by arm, a loop once per iteration.
- Why: per-row stepping pauses once per record and gives different checkpoints per mode. `Checkpoint.arm` and `.iteration` say where you are.
- `value(name)` after a loop is the carry, not the last body write (the body only holds rows that iterated).

## Breakpoints fire on entry; positions inside a row node use `path#locator`

- A path or prefix stops when the run enters the subtree, so `"term"` stops once, not at every node inside.
- A tree or table is one call that cannot yield mid-node, so the session counts `visit` calls and emits `NodeVisited` at the node's `after`.
- Stepped and fused run row nodes compiled and never call `visit`.

## `set` overwrites every version of the name, then records `override@<path>`

- Why: wiring binds each read to one version id, so a branch can still read the prior version and a loop its carry. Overwriting only the latest leaves them stale.
- Cost: `name@producer` for earlier producers shows the override; the `Overridden` event keeps the previous summary.

## `rewind` replays silently; edits rebuild, carry over and replay

- Runners cannot seek, so `rewind(path)` starts a fresh run, drains it to `before path`, then restores current values (overrides included). It evaluates no breakpoints.
- `replace`, `delete` and `reload` share one `_apply`: re-resolve, bind a new executable on a runner copy, carry values over by `(name, producer)`, skip finished upstream nodes. Positional version ids rule out patching a plan.
- An edit that would leave a read with no producer raises `WiringError` first (it would silently become an input column). An error ends the generator; only `rewind` is accepted after.

## Reload compares checkpoint content and restarts at the first difference

- `Session.reload` diffs `(when, path, content)` lists (function fingerprint, params with defaults, wiring); one rebuild covers modify, insert, delete, reorder. Not: per-path `replace` calls, which cannot insert or replace an anonymous root.
- `ModuleWatcher` re-imports every module under the target's directory, `.pyc` deleted first. Not: `importlib.reload` of one file (stale `from x import f`); `__qualname__` re-resolution (swaps steps kept on purpose, lambdas have no global); `__code__` patching (breaks the IR cache).
- Serving never hot reloads (a code change is a new deploy; the audit maps to one code version). See: [experiment](../experiments/hot-reload-edit-latency.md), `decikit/_engine/debug/hot.py`, `edit.py`

## JSON exists only at the edge

- Commands and events are frozen dataclasses with a `kind`; a command's `kind` is the `Session` method name so `apply` dispatches by `getattr`. `Replace` carries a Python object and has no JSON form.
- The editor bridge does not use `COMMAND`; it calls `Session` methods and serialises events.
- See: `decikit/_engine/debug/wire.py`

## The bridge is an adapter over `Session`, never a second debugger

- It lives in `decikit/_tooling/bridge/` and runs as `python -m decikit._tooling.bridge`; no separate distribution and no shim for the old `decikit_bridge` import.
- A bridge session runs interpreted; `assert_equivalent` pins it equal to `run()` on a frame and on each record in all modes (`tests/bridge/test_parity.py`).
- See: [Editor bridge](../../architecture/editor-bridge.md)

## One short-lived process per question; replies on fd 3; types generated from Python

- `ask()` spawns the bridge, writes one line, reads one reply from fd 3, ignores stdout so a step's `print()` cannot corrupt the protocol. A fresh process means an edit saved a second ago counts.
- TypeScript protocol types are generated from the pydantic reply models (`python -m decikit._tooling.bridge.protocol`); every load compiles into a fresh `sys.pycache_prefix` because `.pyc` keys on whole-second mtime and size.

## Forces are overrides of a condition's output

- Just after a branch or loop condition, `Controls` replaces its output with `session.set`; the runner then routes as told and lineage shows "set by you". Not: teaching the IR or runner a forced arm (engine change for a debug feature).
- `Controls.check` is a predicate breakpoint registered first via the bridge's `_checkpoint`, so a user breakpoint cannot pause before the force applies.
- Limits: `rewind` replays without checking, so a force set after the branch ran applies only after rewinding to it (`rerun` does); a loop forced past `max_iterations` stops at the maximum.

## Value breakpoints and history read what a step wrote, not `session.value`

- A watch fires after a step writes the name, for records that newly meet the condition (a value over a limit for five iterations pauses once). `session.value` inside a loop lags one iteration (the carry).
- `Timeline` records changes, not writes: a step writing the value a record already had is `kept`, so "why is x 5" names the step that made it 5. `go_to` rewinds to before the outermost enclosing loop, because rewind keeps the carry.
- `state(row)` takes each record's current value from the timeline for the same reason.

## Forks replay a fresh session to the checkpoint

- A running generator cannot be copied, so `fork` binds a new session and steps to `checkpoint_key` `(path, when, n)`, re-applying recorded `set`s and forces. Steps are pure, so state matches.
- A rewound session refuses to fork (its history no longer describes the path).
- `draft` turns structured changes into experiment overrides and lists everything it cannot convert under `dropped`. A paused session never silently becomes an experiment.

## VS Code debugs with debugpy; the extension has no debug adapter

- "Debug flow" launches `python -m decikit._tooling.bridge.run` under debugpy, interpreted; breakpoints are plain Python and the extension mirrors them onto the flow (`ui/vscode/src/debugging.ts`). `node_path()` walks to the runner's `_call`/`_frame` and returns `node.origin.path`, so a function reused at two nodes maps to the right one. A graph breakpoint on a reused function gets the condition `node_path() == "<path>"` at the step's `bodyLine`.
- Why: the earlier DAP adapter meant two debuggers and two breakpoint models; users reached for the Python one. Per-node stepping, rewind, value breakpoints and forks stay in `Session` and the bridge (no UI caller today). Tree and table nodes (`justMyCode` never stops there) get no breakpoint actions.

## Discovery lists invalid pipeline candidates with a reason

- `find_pipelines` finds module-level steps, a top-level `def build`, and zero-argument factories; an unassembled step or a factory that needs arguments is listed `invalid` with why, never guessed.
- With none valid, `describe` fails with a message saying how to define one; with several, the last valid wins unless the host names one.
- See: `decikit/_tooling/bridge/describing.py`, [Editor bridge](../../architecture/editor-bridge.md)

## Flow files are edited from their YAML, not from `describe`

- The Structure tree is built in TypeScript from the YAML (`yaml` package), so a flow whose new `step:` names a function not yet written still shows and stays editable. Paths follow the loader's naming so items join the described IR.
- Edits go through the `yaml` Document API as one `WorkspaceEdit`: comments and layout survive and undo works.
- See: `ui/vscode/src/outline.ts`, `yamlEdit.ts`

## The UI is one component with a host seam

- `<App send listen can>`: `can` lists the editor-only messages the host handles (`reveal`, `run`, `runTo`, `step`, `debugStep`, `rewind`); controls whose message is missing are hidden. One channel, so a host gains a feature by handling a message and naming it.
- VS Code passes `can = {"reveal"}` and hides the structure tree and breakpoint form (it has its own). Messages to the webview queue until `ready`.
- See: `ui/lib/src/App.tsx`, `ui/vscode/webview/main.tsx`, `ui/vscode/src/graphPanel.ts`

## `@decikit/ui` ships built JavaScript and styles only through `--decikit-*`

- `tsc` emits ESM JS; there is no `"type": "module"` (webpack would demand extensioned imports). Why built JS: a JupyterLab prebuilt extension bundles with webpack, which does not compile a dependency's TypeScript.
- React is a peer and the host calls `createRoot`; `src/model/` has no DOM or React.
- Every rule is nested under `.decikit`, a size container that needs a definite height. A host maps its theme onto `--decikit-*` (`ui/vscode/webview/theme.css`); unset ones fall back at the use site. Effects use block bodies (`scrollIntoView` returns a promise in new Chromium).

## The graph uses a dependency-free layout, fold-by-default and viewport culling

- `layout(ir)` walks the flow's own tree (sequence stacks, branch fans out arms, loop stacks body); flows over 80 steps start folded; only on-screen nodes, clusters and edges are rendered.
- Wheel zoom about the pointer, drag pan, fit; a relayout re-anchors the selected node at its old screen position. Per-column edges show only for the selected step, edge or value.
- Why: dagre overflowed the stack near 2,000 calls; the custom layout is about 600x faster than ELK at 1,000 calls. See: [experiment](../experiments/flow-layout-scale.md), `ui/lib/src/layout.ts`, `Graph.tsx`

## A JupyterLab host would run the bridge in the kernel over a comm

- The extension is gone from the tree; what remains is `Bridge(notebook=...)` (the pipeline, `SAMPLE` and `PARAMS` come from a notebook-namespace snapshot) and `Session.watch` for hot reload per cell. `ui/pnpm-workspace.yaml` still lists the missing package.
- Why a comm and not a server extension: live notebook objects are the reason to debug in Jupyter, and a server process cannot see them. Cost: one message at a time, so no `pause` during a long `resume`.

## Editor limits

- Layout is measured to 10,000 calls (20 ms unfolded); the next ceilings are DOM size and the `describe` payload (about 1 KB a node: 5 to 10 MB at 5,000 to 10,000 calls through `postMessage`). Targets: at most 500 unfolded calls, 5,000 per flow, 2 MB payload.
- `dataEdges` is the one superlinear pass (133 ms at 10,000 calls); `layout.ts` carries the `ponytail:` note (index last writer per column).
- Trace and input values stay in session memory only. A global DataFrame or array counts as changed on every reload (identity fingerprint): extra re-runs, never stale values.

## How the editor is tested

- Python: `tests/bridge` (protocol, controls, timeline, parity), `tests/debug` (session, edit, reload). TypeScript: `pnpm test` in `ui/lib` (model, layout) and `ui/vscode` (YAML outline and edits, breakpoint and pause mapping). `pnpm test:vscode` runs inside VS Code: debug a shared step under debugpy and check the pause maps to that node.
- Screenshot-judged UI loops found layout and wording bugs no assertion caught; the harness was removed with the old extension.
- See: [experiment](../experiments/ui-judge-loop.md)
