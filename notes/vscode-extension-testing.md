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

The screenshots showed layout bugs that no assertion caught. Flex children
wouldn't shrink, so the graph was pushed out of view. Data edges in the layout
graph made it sprawl. Both were fixed from the images.
