# Handoff: decider VS Code UI facelift / debugger-UI prototyping

Written: 2026-10-01. Repo: `/Users/cp371651/Documents/Workspace/SharedLibraries/dsp-decision-engine`.

## Background — read these first, don't re-derive

- `notes/vscode-redesign/ui-facelift-handover.md` — the original brief for this whole effort (facelift the `@decider/ui` panel inside VS Code). Settled decisions live here (one shared Flow canvas, three mode-specific side views, progressive disclosure for data deps, etc.) — don't re-litigate without a concrete reason.
- `notes/recommendations-for-decider-v2.md` — a separate note the user had open; I never read its contents this session, just saw it referenced in `<ide_opened_file>` tags. Worth checking if it's relevant to whatever comes next.
- `notes/experiment-plot-types-handover.md` — untracked, created by someone/something else during this session (not me). Not inspected.

## What this session actually was

Started as "build a disposable VS Code-shell mock in `tools/decider-ui/demo/` so we can iterate on UI/UX fast without launching the real extension." Stayed that way for a while (sidebar, Explorer/Pipelines/Structure panes, breakpoint dots, context menus, free-text conditional breakpoints — all mock-only, in `demo/vscode/`), then pivoted hard: several rounds of fixes against the mock turned up real, wired-but-orphaned backend capability, so the back half of this session made **real, shipped changes** to `tools/decider-ui/src/` and `tools/vscode-decider/src/`, not just the mock.

**Important:** `tools/decider-ui/demo/` grew a lot of scope *I did not build* — a dataset picker, an editor tab stack, a Parameters tab, an Experiments pane (`demo/vscode/DatasetView.tsx`, `Pane.tsx`, `ParamTableView.tsx`, `ParametersView.tsx`, `datasets.ts`, `experiments.ts`). Multiple times this session I got "file changed on disk since you last read it" notices and found substantial unrelated features already there. This is almost certainly the user (or another agent) working directly in the IDE in parallel. **Re-read `demo/main.tsx` and `demo/vscode/Shell.tsx` fresh before touching them** — don't trust this document's description of their exact current shape, only of what I personally changed.

Also unexpected: `git status` shows `decider/debug_bridge/bridge.py` (modified), `decider/debug_bridge/rows.py` (new), `tests/debug_bridge/test_bridge.py` (modified) — **Python backend changes I did not make and have not reviewed.** Check what they are before assuming the backend is in the state described below.

## Real product changes made this session (not mock-only)

All in `tools/decider-ui/src/` unless noted. Verified with `npx tsc --noEmit -p tsconfig.demo-check.json` (the tsconfig that actually covers `demo/` too — plain `tsconfig.json` only covers `src`/`test`) and `npx vitest run` (36 passing) after each batch, plus `tsc --noEmit` in `tools/vscode-decider` and `tools/jupyterlab-decider` to confirm the other two hosts still compile against the rebuilt `@decider/ui` dist (`npm run build` in `decider-ui` — **do this if you change `src/model/protocol.ts` types, or the other two packages typecheck against stale dist**).

1. **`App.tsx`**: added optional, additive, backward-compatible props so a host can opt into behavior without affecting existing hosts (VS Code extension / JupyterLab both still omit all of these, unchanged):
   - `showStructure` (hide the in-panel Structure tree)
   - `showBreakpointForm` (hide the Inspector's old breakpoint form)
   - `onSelectionChange` (host learns the graph's current selection)
   - `onNodeContextMenu` (right-click a graph node → host shows its own menu)
   - New `EditorMessage` value `"rewind"` (gates the new Step-back button; **not added to the real VS Code extension's or JupyterLab's `can` set** — that's a deliberate per-host opt-in decision I left for later)
2. **`Graph.tsx`**: breakpoint dot moved to bottom-left of each step (matches the "ran" indicator's old spot; moved "ran" to bottom-right to avoid collision). Click toggles a plain breakpoint; right-click opens `onNodeContextMenu`. Paused step shows a pause glyph (two bars) in the dot's spot instead of a separate indicator. Conditional breakpoints show a small "=" overlay.
3. **`Inspector.tsx`**: removed "Convert to draft" entirely (dead-end UI — computed data but no save/export action; backend `decider/debug_bridge/draft.py` left alone, just unreachable). Removed the "Source" section + `onReveal` plumbing (redundant with double-click-to-reveal, which already existed on the graph) — then **re-added** `onReveal` later for a different purpose (see next line). Runtime section (while paused) now has: inline value override (✎ button → input → sends new `setValue` message), and a small toolbar (Replace step… / Reload edited code / Restore original / Skip this step) sending `setValue`/`reloadStep`/`restore`/`skip`.
4. **`model/protocol.ts`**: added `Watch.condition?: string` (free-text expression, alongside the existing structured `name`/`op`/`value`) and a new `FromUI` member `{type:"setValue", name, value}`. The `"controls"` `ToUI` message (push controls into `App` from outside) was also added this session for the sidebar/graph breakpoint-sync work.
5. **`model/breakpoints.ts`**: `breakpointNodes()` now also matches a value/condition watch scoped to exactly one step against that step's *inputs*, not just what it writes (so "pause when my input X meets a condition" shows a dot on the right node). Test added in `test/breakpoints.test.ts`.
6. **`Controls.tsx`**: `watchText()` renders a `condition`-based watch as the raw expression instead of trying to force it through name/op/value.
7. **`style.css`**: added/adjusted `.bp-dot`, `.pause-dot`, `.bp-eq`, removed now-orphaned `.source-line`/`.draft-group` rules.
8. **`tools/vscode-decider/src/extension.ts` + `adapter.ts`**: wired the new `setValue` message for real — `decider.setValue` custom request calls the *same* Python bridge `set` command the native VS Code Variables pane's "Edit Value" already uses (`adapter.ts`'s pre-existing `setVariableRequest`). This is a second front door to existing capability, not a new backend mechanism.
9. **Step back / Replace step were *not* new backend work** — `rewind`, `goTo`, `reloadStep`, `restore`, `skip` were already fully wired end-to-end (protocol → `extension.ts` → `adapter.ts` → Python bridge) from a pre-redesign version of the UI (`tools/decider-ui/src/old/App.tsx` + `old/NodePanel.tsx`). They'd just been dropped from the current component tree during the redesign. I restored UI entry points for them; I did not touch the backend.

## Known gaps / honest caveats

- **Sidebar (mock) breakpoints vs. real `Controls.watches`**: these got unified mid-session — the sidebar's dots now read/write the *same* `Controls` state the Graph does, pushed via the new `"controls"` ToUI message. Should be consistent now; wasn't for a few turns, worth a sanity check.
- **Free-text conditional breakpoints (`Watch.condition`) are UI-only.** I confirmed the real Python engine (`decider/engine/debug/controls.py`'s `_watch`) only understands structured `name`/`op`/`value` watches, evaluated strictly *after* a write. There is no expression evaluator on the backend. I was wrong earlier in the conversation when I claimed a VS Code `SourceBreakpoint` `condition` string already provided this — that mechanism is for the unrelated `decider.debugStep` ("attach pdb to one step") feature, not flow-level breakpoints. If this direction sticks, the backend needs real expression-eval work before it's more than a mock.
- **"Replace step" is the lighter of two designs discussed.** The heavier version (an isolated virtual-document editor scoped to just one function, with edits mapped back into the real file) was explicitly *not* built — current implementation reveals the real source file and relies on "Reload edited code" to pick up whatever's saved there.
- **`setValue` → `decider.setValue` is typecheck-verified only.** I have not run a real debug session against the Python bridge to confirm `self.bridge.request("set", {...})` behaves as expected from this new call site (it's the same command `setVariableRequest` already uses successfully, so risk should be low, but it's untested by me).
- **`"rewind"` capability is opt-in per host** and I did not add it to `tools/vscode-decider/webview/main.tsx`'s `can` set (currently just `["reveal"]`, oddly minimal — possibly stale, did not investigate) or to `jupyterlab-decider`'s. The Step-back button only shows in the demo harness right now.
- I did not investigate the unrelated `bridge.py`/`rows.py`/`test_bridge.py` changes `git status` shows — unknown origin, unknown content.

## Suggested skills for the next agent

- **`run`** — before trusting any description in this doc of current UI behavior, launch the demo (`cd tools/decider-ui && pnpm dev`) and actually look. This session had no screenshot/browser tooling available; nothing visual was verified beyond "it compiles and boots."
- **`impeccable`** — if asked to continue the visual/UX design work (the whole point of this thread). There's an `impeccable` hook active in this repo already flagging a pre-existing, unrelated finding in `src/style.css` (`.scenario-error` side-tab border) every turn — not mine to fix, but worth closing out at some point.
- **`code-review`** or **`security-review`** — before this lands anywhere real, given the volume of changes across three packages (`decider-ui`, `vscode-decider`, and apparently `decider` itself per the untracked Python diffs) in one sitting.
- **`diagnosing-bugs`** — if the next session is about verifying `setValue`/`reloadStep`/`rewind` actually work end-to-end against a real paused debug session, since that's the one path in this handoff that's unverified beyond typechecking.
