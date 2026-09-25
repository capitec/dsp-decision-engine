# @decider/ui: the flow debugger's UI, for any host

The React UI of the flow debugger and the model code it shares with its hosts live in
`tools/decider-ui`, a pnpm workspace package (`tools/pnpm-workspace.yaml`). The VS Code
extension (`tools/vscode-decider`) is one host; a JupyterLab 4 extension is meant to be the
next.

## Layout

```
tools/
  pnpm-workspace.yaml     decider-ui, vscode-decider, jupyterlab-decider
  decider-bridge/         decider_bridge: the Python bridge both hosts drive (JSON-lines
                          process for VS Code, a kernel comm for JupyterLab)
  decider-ui/
    src/model/            protocol.ts, compare.ts, sweep.ts, events.ts: bridge shapes, UI
                          messages, run comparison, scenario sweeps, folding session events.
                          No DOM, no React.
    src/*.tsx, layout.ts  the components (App is the root), the graph layout
    src/style.css         every rule nested under .decider
    src/index.ts          exports the model and App
    test/                 model and layout tests (they import decider_bridge and use
                          vscode-decider's examples/)
  vscode-decider/
    src/                  extension host and debug adapter; imports the model from @decider/ui
    webview/main.tsx      the host wiring: acquireVsCodeApi, window messages, capabilities
    webview/theme.css     --decider-* from --vscode-*
  jupyterlab-decider/     the JupyterLab host: see notes/jupyterlab-decider.md
```

`model/` is a folder rather than living next to the components because `compare.ts` and
`Compare.tsx` would collide on a case-insensitive file system.

## Consuming it

The package builds to `dist/` with `tsc` (ESM JavaScript and `.d.ts`), and `exports` points
there; the stylesheet is exported from source as `@decider/ui/style.css`. Built JS, not TSX
from source, because a JupyterLab prebuilt extension bundles with webpack through
`@jupyterlab/builder`, which does not compile a dependency's TypeScript. Vite takes the same
output. `prepare` builds it on `pnpm install`, and the extension's `pnpm build` rebuilds it
first.

There is no `"type": "module"`: with it, webpack requires fully specified (extensioned)
relative imports in `.js` files, and tsc emits them extensionless. `sideEffects: ["*.css"]`
lets the extension host's bundle, which imports only the model from the package root, drop
the components, React and dagre.

React is a peer dependency (`^18.2.0 || ^19.0.0`); only `react` is, because the package never
imports `react-dom`; the host calls `createRoot`. JupyterLab 4 shares React 18 as a
singleton, so the workspace develops and tests against React 18.3: the extension moved from
React 19 to 18. The UI used nothing React 19 only.

## The host seam

```tsx
<App send={(m: FromUI) => void} listen={(on: (m: ToUI) => void) => unsubscribe} can={Set<EditorMessage>} />
```

- `send` delivers a `FromUI` message to the host. The first is `{type: "ready"}`; the host
  should hold its messages until then.
- `listen` subscribes to `ToUI` messages and returns the unsubscribe. VS Code adapts window
  `message` events to it; a JupyterLab widget renders in the page, so it would call the
  subscriber directly (a `Signal`, or a plain callback list).
- `can` lists the editor-only messages the host handles:
  `reveal`, `maximise`, `openDiff`, `debugStep`, `runTo`, `step`, `compareRevision`. The UI
  hides the controls whose message is missing: Open source (and double-click on a step),
  the maximise button, view diff links, Step into the Python, Run to / Run through, and
  Compare with a git revision. A set of message types rather than callbacks keeps one
  channel: a host that gains a capability handles the message and adds its name.

`ToUI`/`FromUI` are in `src/model/protocol.ts`. The rest of the messages are the core: every
host has to answer them from the bridge's debug session.

## Theme variables

The stylesheet reads only `--decider-*` variables. A host maps its theme onto them; VS Code's
mapping is `vscode-decider/webview/theme.css`. Colours with a hex fallback in the stylesheet
work unmapped; the rest need a value.

| variable | VS Code source | fallback in the UI |
|---|---|---|
| `--decider-font`, `--decider-font-size` | `font-family`, `font-size` | |
| `--decider-mono` | `editor-font-family` | |
| `--decider-fg`, `--decider-bg` | `foreground`, `editor-background` | |
| `--decider-muted` | `descriptionForeground` | |
| `--decider-error` | `errorForeground` | |
| `--decider-link` | `textLink-foreground` | |
| `--decider-border` | `panel-border` | |
| `--decider-focus` | `focusBorder` | |
| `--decider-widget-bg`, `--decider-widget-border` | `editorWidget-background`, `-border` | border: `--decider-border` |
| `--decider-shadow` | `widget-shadow` | `rgba(0,0,0,.4)` |
| `--decider-code-bg`, `--decider-quote-bg` | `textCodeBlock-background`, `textBlockQuote-background` | code: `rgba(127,127,127,.1)` |
| `--decider-button-bg`, `-fg`, `-border` | `button-background`, `-foreground`, `-border` | border: `--decider-border` |
| `--decider-button2-bg`, `-fg` | `button-secondaryBackground`, `-Foreground` | |
| `--decider-input-bg`, `-fg`, `-border` | `input-background`, `-foreground`, `-border` | border: `--decider-border` or transparent |
| `--decider-invalid-border` | `inputValidation-errorBorder` | `#be1100` |
| `--decider-dropdown-bg`, `-fg`, `-border` | `dropdown-background`, `-foreground`, `-border` | border: transparent |
| `--decider-badge-bg`, `-fg` | `badge-background`, `-foreground` | |
| `--decider-hover-bg` | `list-hoverBackground` | |
| `--decider-selection-bg`, `-fg` | `list-activeSelectionBackground`, `-Foreground` | |
| `--decider-inactive-selection-bg` | `editor-inactiveSelectionBackground` | `rgba(127,127,127,.15)` |
| `--decider-match-bg` | `editor-findMatchHighlightBackground` | orange, translucent |
| `--decider-paused-bg` | `editor-stackFrameHighlightBackground` | `#ffff0033` |
| `--decider-inserted-bg` | `diffEditor-insertedTextBackground` | `#9bb95533` |
| `--decider-added-fg`, `--decider-deleted-fg` | `gitDecoration-added/deletedResourceForeground` | `#81b88b`, `#c74e39` |
| `--decider-passed` | `testing-iconPassed` | `#73c991` |
| `--decider-pause-icon`, `--decider-breakpoint` | `debugIcon-pauseForeground`, `-breakpointForeground` | orange, `#e51400` |
| `--decider-warning` | `editorWarning-foreground` | `#cca700` |
| `--decider-red`, `-orange`, `-yellow`, `-green`, `-blue`, `-purple` | `charts-*` | VS Code's chart colours |

The mapping is one to one: every distinct `--vscode-*` colour the UI used (48 of them, from
about 220 uses) has its own variable, so VS Code looks exactly as before. Merging near
duplicates (dropdown and input, pause icon and yellow) would save a few lines of mapping but
change colours in some themes. The mapping gives no fallbacks: an unset VS Code colour leaves
the `--decider-*` variable invalid, and the UI's own fallback at the use site applies, as it
did before.

## Scoping

Every rule in `style.css` is nested under `.decider`, the element `App` renders around
everything, so nothing styles the rest of a JupyterLab page. Vite lowers the nesting to
`.decider …` selectors. The keyframes are named `decider-found`.

`.decider` is a size container (`container-type: size`): the narrow and wide layouts are
`@container` queries and the fixed heights `cqh`, so they follow the panel's size, not the
window's. It also makes `.decider` the containing block for anything positioned `fixed`. It
needs a definite height: VS Code's `theme.css` sets `html, body, #root` to 100%.

Page-level rules stay with the host: `color-scheme`, the body's margin and background.

## What a JupyterLab host needs to provide

`tools/jupyterlab-decider` provides all of this (`notes/jupyterlab-decider.md`). The UI side of
the last item is done: the empty state says "Loading the flow…", and Ctrl+F only acts while
the focus is in the flow view (or on the page itself).

- A widget that renders `<App>` into its node with a definite size, and unmounts it on dispose.
- `send`/`listen` over whatever reaches the bridge (a comm, a server extension or a kernel);
  hold messages until `ready`, like `graphPanel.ts` does.
- Handle the core `FromUI` messages from the bridge's session, the way `extension.ts` does.
- A `can` set: probably empty at first (or `step`/`runTo` if it drives the session itself).
- A `--jp-*` to `--decider-*` mapping, on `.decider` or the widget node, for the variables
  without fallbacks above.
- `@decider/ui/style.css` in its style entry.
- React from JupyterLab's shared scope: the package's React is a peer, but pnpm links a dev
  copy next to it; module federation's singleton `react` covers imports from it.
- Leftovers that assume an editor: the empty-state text ("choose Visualise flow"), the
  window-wide Ctrl+F listener in `FindStep.tsx` (JupyterLab uses Ctrl+F itself), and the
  default link and code styles VS Code injects into webviews.
