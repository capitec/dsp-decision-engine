# Testing the VS Code extension

The prototype in `tools/vscode-decider/` is tested at four levels. Each one
catches things the others can't.

| Level | Tool | Catches |
|---|---|---|
| Python bridge | pytest | session, lineage, trace and per-record logic |
| Debug adapter | `@vscode/debugadapter-testsupport` `DebugClient` over stdio | the protocol mapping: stops, scopes, set, breakpoints |
| Extension API | `@vscode/test-electron` + mocha inside the editor | activation, CodeLens, commands, debug sessions |
| UI | Playwright `_electron.launch` on the editor binary | what a person clicks and sees, including webview content; screenshots |

Findings that cost time:

- **Vite library builds honour a package's `browser` field.** For
  `@vscode/debugadapter` that swaps the stdio runner for an empty stub, so the
  adapter process exits silently. Build the extension bundle with `ssr: true`.
- **`DebugSession.run` constructs the session with a legacy boolean.** A
  constructor that takes options must accept and ignore it.
- **`/usr/bin/codium` is a wrapper that detaches.** Test runners have to launch
  `/usr/share/codium/codium`, the Electron binary itself.
- **Headless works with `--ozone-platform=headless`.** Screenshots still render,
  so no Xvfb is needed and no window appears.
- **Webview content is two iframes deep:** `iframe.webview.ready`, then
  `#active-frame`. Playwright's `frameLocator` chains through both.
- **A fresh profile needs quiet settings** (no parent-folder git prompt, no
  secondary side bar), or notifications cover the UI in screenshots.

**Usability loop.** `pnpm test:stories` runs five user stories and captions each
screenshot with what the user just did. An agent with no other context scores
each story from the screenshots alone (discoverability, clarity, feedback,
efficiency, goal answered) and lists fixes. A fresh reviewer each round keeps
the scores honest. Story scripts drive the editor the way a person would:
keys pressed while a webview has focus stay in the webview, so a script clicks
the editor tab before using editor shortcuts.

The screenshots showed layout bugs that no assertion caught. Flex children
wouldn't shrink, so the graph was pushed out of view. Data edges in the layout
graph made it sprawl. Both were fixed from the images.

**At scale.** `pnpm test:large` runs seven stories on `examples/bank` (about
1,000 steps, 1,400 params, 14 lookup tables; see `notes/ux-loop-large.md`).
What the small example never showed:

- **Stale bytecode after a quick edit.** Python keys cached bytecode on the
  source file's mtime in whole seconds and its size. An edit saved within the
  same second as the last load, at the same length, ran the old code. The
  bridge now compiles every load into a fresh `sys.pycache_prefix`.
- **`inspect.getsource` reads the file as it is now**, so diffing a step
  before and after an edit needs the text as it was loaded, kept per file.
- **A custom DAP request without arguments arrives with `args` undefined.**
  Default it to `{}`.
- **The debugger opens a paused step's source in the active editor group**,
  which is the flow panel's once the user has clicked in it. The panel moves
  such a tab to the first group while a decider session runs.
- **A test's `locator(..., { hasText })` matches ancestors too**: a nested
  list item's text is in every item above it, so clicking "the first match"
  hit the outermost one.
