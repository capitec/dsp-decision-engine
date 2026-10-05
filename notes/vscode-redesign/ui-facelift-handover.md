# UI facelift — handover

**Written for:** a fresh Claude Code agent picking up this redesign inside VS Code, with no
memory of how we got here. If anything below turns out to be wrong or stale, trust the
code over this document and flag the mismatch rather than guessing.

## The job

> "A lot of the design elements are there in `decider-ui` but they aren't nicely integrated
> in vscode." — the product owner (Sholto)

The individual pieces — graph, inspector, structure tree, experiments table, Sankey — already
exist and work. The job is a **facelift**: make them read as one coherent product inside the
actual VS Code extension, not a redesign of what each piece does. Don't re-litigate the
underlying interaction model (selection, progressive disclosure, fold-by-default) unless you
find a concrete reason to — see "Settled decisions" below for what's already been chosen and
why.

This is UI/visual work. If the `impeccable` skill is available to you, it's a good fit for
this kind of iterative frontend design task.

## Orientation: what you're looking at

`decider` is a Python library for building decision pipelines (credit/lending logic, mostly).
The VS Code extension visualises, runs, and lets you step through those pipelines. Two
packages matter here, both under `tools/` (a pnpm workspace):

- **`tools/decider-ui`** — a host-agnostic React component library (`@decider/ui`). This is
  where almost all of the actual UI code and CSS lives. It's also consumed by a JupyterLab
  extension, so nothing here should assume it's running inside VS Code specifically.
- **`tools/vscode-decider`** — the VS Code extension itself: activation, the debug adapter,
  commands, and a thin webview host that renders `@decider/ui`'s `App` component.

Key files in `tools/decider-ui/src/`:

| File | What it is |
|---|---|
| `App.tsx` | Root component. Owns selection/fold state, routes between the Explore layout and the full-screen Experiments view. |
| `Graph.tsx` | The flow canvas: SVG rendering, zoom/pan/fit gestures, node/edge drawing. |
| `Inspector.tsx` | The one contextual side panel — overview + reads/writes + runtime state + breakpoints, depending on what's selected. |
| `Structure.tsx` | The tree view, kept in sync with the graph's selection/fold state. |
| `Experiments.tsx` / `Sankey.tsx` | The experiments results view: scenario table, drill-down, a hand-rolled Sankey diagram. |
| `layout.ts` | Pure layout/graph-shape functions (no React). Just changed — see below. |
| `style.css` | **Every rule for every component, in one file**, all nested under `.decider`. ~670 lines. This is the thing a facelift mostly touches. |
| `model/protocol.ts` | The message/data shapes between host and UI. |

**Theming is live, not hardcoded.** `tools/vscode-decider/webview/theme.css` maps every
`--decider-*` custom property straight to VS Code's own theme variables (e.g.
`--decider-fg: var(--vscode-foreground)`, `--decider-border: var(--vscode-panel-border)`).
Whatever you design has to work against *whatever theme the viewer has installed*, light or
dark — you're not choosing the palette, you're choosing how to use VS Code's palette well.
`tools/decider-ui/demo/theme.css` has a second, hardcoded set of the same token names — that's
only for the standalone dev harness (see below), not what ships.

## What just changed (already in your working tree, uncommitted)

The graph's layout engine was dagre; it's not anymore. dagre crashed around ~1,000 steps on a
real (grouped) flow and was never going to support the "thousands of nodes" the product needs.
It's been replaced with a small dependency-free layout function that just walks the pipeline's
own tree shape (a sequence stacks children, a branch puts its condition over a row of arms, a
loop puts its condition over its body), plus viewport culling in `Graph.tsx` so the renderer
doesn't choke once thousands of nodes are visible. Both are tested (`tools/decider-ui/test/layout.test.ts`,
including a regression test against a ~2,200-node flow) and the full build passes in both
packages. `git status` will show `layout.ts`, `Graph.tsx`, `package.json` (dagre dependency
removed), and the lockfile as modified — that's expected, not something to undo.

Why this matters for a facelift: the graph can now reasonably show far more than the old
~500-node fold-by-default budget implied. Fold-by-default is now a **product/UX choice** about
what's pleasant to look at, not a performance constraint. Don't assume the old small numbers
still apply if you're designing around scale.

Layout timings for the custom layout that replaced dagre are in `notes/flow-layout-scale.md`.
No xyflow/Cytoscape/ELK comparison numbers were recorded anywhere; that comparison was never written up.

## How to run it and see your changes

Two loops, pick based on what you're touching:

**Fast loop (use this for most visual work):**
```
cd tools/decider-ui && pnpm demo
```
A real Vite dev server with HMR, rendering the actual `App` component (not a copy) against
mock data. Edit anything under `src/` and the browser updates instantly. It has buttons to
switch between a small/large flow, pause (to see the Debug state), and show an experiment
result — so all three modes are reachable without VS Code. Caveat: it uses the hardcoded demo
palette, not real VS Code theme variables, so don't judge final color/contrast here.

**Real loop (check against an actual theme and a real pipeline before calling something done):**
Open `tools/vscode-decider/decider-dev.code-workspace` in VS Code, then Run & Debug →
**"Extension + Examples"** (F5). Its `preLaunchTask` runs the full build automatically. An
Extension Host window opens on `tools/vscode-decider/examples/`; open `examples/loan.py` and
click "Visualise flow" above the pipeline definition. There's no webview watch mode — after a
change, stop and re-press F5 (not just reload) to rebuild.

**Environment gotcha:** the Node on PATH is 20.11; Vite 8/Vitest need 20.19+. Use
`nvm use 22` (already installed) if a build/test command fails with a cryptic module error.

## A reference artifact (starting point, not a direction)

<https://claude.ai/artifact/LrpJCc7UYkHQPES8vTm8SF> — "Decider Flow Explorer". A faithful,
static-ish recreation of *today's* look (real tokens pulled from `style.css`, real mock flow
data, all three modes), built so there was something concrete to react to before any redesign
work started. It is **not** a proposed new direction — it's intentionally just today's UI, cleaned
up for the recreation (its graph layout is hand-placed for tidiness, not run through the real
`layout.ts`, so don't read its exact node spacing as meaningful). Useful as a quick visual
reference without needing to launch VS Code, nothing more.

## Settled decisions (don't re-litigate without a concrete reason)

From the original redesign plan and review, still the intended model:

- **One shared Flow canvas, three mode-specific side views** (Explore, Debug, Experiments) —
  not three different panel types, and not the graph "morphing" per mode. The *inspector*
  changes context; the canvas stays the same object.
- **Flow → Debug is an explicit run action**, not implicit. A paused debug session never
  silently becomes an experiment — converting one to a scenario is an explicit, visible step
  that lists what couldn't be converted rather than dropping it quietly (`Inspector.tsx`'s
  "Convert to draft" section).
- **Progressive disclosure for data dependencies, not dense edge labels.** Reads/writes show
  as a short sub-line on every node; full data-edge labels only appear once something is
  selected. Don't default to showing every data edge at once — it was deliberately rejected as
  the primary representation.
- **Static reads/writes vs. runtime values are visually distinct**, never conflated. A value
  set by a live override during debugging is tagged "· set by you" and must never look like a
  saved experiment result.
- No more editor-group juggling during debug (an old mechanism that moved source files out of
  the flow panel's group was deliberately removed) — don't reintroduce panel-shuffling tricks.

## Known rough edges worth knowing about (not necessarily yours to fix)

- The breakpoint UI only has add/remove, no visual "disabled" state, though the original spec
  called for enabled/disabled/hit/paused as distinct states (`model/protocol.ts`'s `Watch` type,
  `Controls.tsx`'s `WatchForm`). Possibly in scope for a facelift if breakpoint affordances come
  up.
- `style.css` has been through several large rewrites as the redesign landed in pieces (task
  05, then 07, 09, 10, 11 layered on top). Some inconsistency is real, not imagined — that's a
  lot of the reason a facelift is warranted in the first place.

## Don't need to read, but here if you want the full history

`notes/vscode-redesign/tasks/05-redesign-flow-exploration.md` (the graph/canvas spec this is
built from), `06`/`07`/`09`/`10`/`11` for the other modes, and `notes/vscode-redesign/background.md`
for how the whole redesign effort started.
