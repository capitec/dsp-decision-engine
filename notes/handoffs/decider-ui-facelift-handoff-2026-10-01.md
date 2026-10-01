# Handoff: decider VS Code UI facelift — dataset & parameters design

**Date:** 2026-10-01
**Repo:** `dsp-decision-engine`
**Branch:** `feature/eachIRNode`
**Primary reference doc:** `notes/vscode-redesign/ui-facelift-handover.md` — read this first, it's the original brief for the whole facelift effort (what `decider` is, the three-mode model, settled decisions, how to run the demo harness and the real extension).

## Important: a second agent is working on this same project concurrently

Throughout this session, another agent (or the user, via another session) was live-editing several of the same files — `tools/decider-ui/demo/main.tsx`, `tools/decider-ui/src/App.tsx`, `tools/vscode-decider/src/adapter.ts`, `tools/vscode-decider/src/extension.ts`, `tools/decider-ui/demo/vscode/shell.css`, and others. Several "file changed on disk since you last read it" conflicts occurred and were resolved by re-reading before editing. **Before continuing, run `git status` / `git diff` to see the live state — do not assume this document's file descriptions are still 100% current.** The other agent's work (visible in git status as untracked/modified) includes: `tools/decider-ui/demo/vscode/experiments.ts` (a new Experiments & Results sidebar pane), a generalized right-click `ContextMenu` (replacing a narrower `BreakpointMenu`), breakpoint condition editing changes (`Controls.tsx`, `model/breakpoints.ts`, `test/breakpoints.test.ts`), and a `rewind`/`reloadStep`/`restore`/`skip` message-handling in the demo's mock host. None of that is mine to explain in detail — look at their diffs directly.

## What this session did

Two tracks:

### Track 1 — real backend/extension wiring (not demo-only)

Debug mode's data loading bypassed `decider`'s real `load()` machinery (JSON-only, no schema/fingerprint). Fixed to match what Experiments already did:

- `decider/debug_bridge/rows.py` (**new**) — `load_rows(data)`: file path → `decider.data.load()` (now supports CSV/Parquet too, not just JSON); inline rows → `LoadedData.from_frame()`.
- `decider/debug_bridge/bridge.py` — `_rows()` now captures `self.dataset` (an `InputFingerprint`: name, columns, row_count, fingerprint) and `status()` returns it as `"dataset"`.
- `tests/debug_bridge/test_bridge.py` — two new tests: dataset identity on inline `SAMPLE` rows, and loading a real CSV file by path.
- `tools/decider-ui/src/model/protocol.ts` — added `DatasetInfo` (camelCase, UI-facing) and a matching `dataset` field on both the raw `Status` type and `RunStatus`; added `datasetLabel()` helper.
- `tools/decider-ui/src/model/experiment.ts` — `ExperimentResult.manifest.input` typed (was already populated server-side via `RunManifest.input`, just never surfaced).
- `tools/vscode-decider/src/adapter.ts` — threads `status.dataset` → camelCase `DatasetInfo` → the `decider.status` custom event body.
- `tools/vscode-decider/src/extension.ts` — `pickData()`'s file dialog filter widened from JSON-only to JSON/CSV/Parquet.
- `tools/decider-ui/src/App.tsx` — a small dataset badge (`name · N rows`, fingerprint on hover) in the header, shown identically in Debug and Experiments modes. **Note:** this file has since been further edited by the other agent (added `onNodeContextMenu`, removed `onEditBreakpoint`) — the badge should still be there but re-check.

All of this was verified: Python tests pass (`uv run pytest tests/data tests/experiments tests/debug_bridge`), `tsc --noEmit` clean on both TS packages, existing Vitest suite passes.

### Track 2 — demo harness iteration (`tools/decider-ui/demo/`, visual/UX exploration only, not wired to the real extension)

The demo (`pnpm demo` from `tools/decider-ui`, needs `nvm use 22`) mocks a VS Code window (`vscode/Shell.tsx`) so UI ideas can be iterated on without launching the real extension. Built incrementally per the user's steering, in this order:

1. **Sidebar became a real accordion** (`vscode/Pane.tsx` — collapsible section matching VS Code's Outline/Timeline pane markup) with panes: PIPELINES, STRUCTURE, PARAMETERS, DATASETS, EXPERIMENTS & RESULTS (that last one is the other agent's).
2. **Both tab stacks are generic, not hardcoded two-slot toggles.** This was an explicit correction from the user — the first version toggled visibility between two fixed panes; it was rebuilt as true push/pop stacks:
   - `vscode/Shell.tsx` exports `SourceTab` (left stack: `code` | `params`) and `EditorTab` (right stack: `flow` | `dataset` | `paramTable`).
   - `main.tsx` owns `leftTabs`/`activeLeftTabId` and `tabs`/`activeTabId` as arrays, with `openTab`/`closeTab` (and `openLeftTab`/`closeLeftTab`) push/pop helpers. Opening something already open just activates it; closing pops it and falls back to whatever's now on top.
   - The old single "Hide code"/"Show code" button was replaced by a `×` directly on the code tab, matching the right side's tab-close pattern.
3. **Datasets**: `vscode/datasets.ts` (4 mock datasets, one — `applications.json` — with nested `struct`/`list` columns to demonstrate non-flat data) and `vscode/DatasetView.tsx` (per-column sort + filter icons in the header, not one global search box; a left-hand ▶ "Debug this record" icon per row instead of a big button; a ▸/▾ per-row expand into a pretty-printed JSON block). Clicking a dataset in the sidebar opens its own tab; "Debug this record" jumps to/opens the flow tab with a mocked paused session built from that row (`mock.ts`'s `pausedRunFor`).
4. **Parameters**: `vscode/ParametersView.tsx` — lists every declared param (step, name, type, default, validation from `ParamInfo`'s `ge`/`gt`/`le`/`lt`/`required`) with an editable value column. Key behaviours, each from explicit user feedback:
   - **Selection-scoped**: when a node is selected (shared between the graph and the Structure tree via `selectedPath`), the list narrows to that node's own params plus `shared` ones, with a "show all" / "scope to `<node>`" toggle.
   - **Search box** filtering by step or param name.
   - **"shared" is not a step** — rendered as `shared · all steps`, not as if it were a node path (`stepLabel()` helper).
   - **Table-shaped params**: a param can be `type: "table"` with an array `default` (demo example: `shared.risk_bands`, a scorecard-style lookup table). These render as `[{n} rows] ⤢ expand` instead of a text input.
   - **Full-screen table expansion**: `vscode/ParamTableView.tsx` — clicking "expand" opens the table as its own tab on the right (same stack as Datasets/flow), reusing the exact sort/filter/row-to-JSON mechanics from `DatasetView`, but every cell is editable (it's a real value, not a preview).
   - **Debug-mode framing**: while a mocked debug session is "paused" (`isDebugging` in `main.tsx`), the value column reads "Live value" instead of "Value for next run", the Run button is replaced by a note, and edits are explicitly **forward-only** — "applies from `<pausedAt step>` onward; steps already run keep the value they ran with" — matching the real debugger's `override@<path>` semantics, not a full re-run. `main.tsx` tracks `pausedAt` (the step the mocked session is paused at) to make this concrete.
   - Discoverability fix: Parameters was first exposed only as a small gear icon buried in STRUCTURE's pane header — the user couldn't find it. Replaced with its own PARAMETERS sidebar pane (declared-param count + "Open Parameters" row), matching how DATASETS already works, so the pattern "click an item in this list, a tab opens" is consistent everywhere.
5. Added a "Parameters" codelens button (`vscode/loan-source.ts`) alongside Run/Visualise/What-if.

All of Track 2 was verified after every step: `tsc --noEmit` against a temporary tsconfig including `demo/` (see cleanup below), `pnpm demo:build`, and `pnpm test` (36 Vitest tests passing throughout, growing as the other agent added their own).

## Known loose ends / cleanup needed

- **`tools/decider-ui/tsconfig.demo-check.json` is a stray scratch file** (untracked). It's a temporary tsconfig (extends the real one, adds `demo` to `include`) used to typecheck the demo harness since the real `tsconfig.json` excludes it. `rm` was denied by the permission settings in this session every time it was attempted — **delete this file**, or re-derive the same check command if you need it again:
  `npx tsc --noEmit --target ES2022 --module ESNext --moduleResolution Bundler --lib ES2022,DOM --jsx react-jsx --strict --skipLibCheck --esModuleInterop --types node demo/main.tsx demo/vscode/*.tsx demo/mock.ts` (run from `tools/decider-ui`).
- **Not done**: an in-webview dataset *browser*/schema-preview before running (Track 1 only surfaces the dataset identity *after* a file is already chosen via the native VS Code dialog). Flagged to the user as a bigger scope than "facelift" warrants; they haven't asked for it.
- **Not done**: wiring the demo's Parameters/table edits into anything real — it's all local component state in `main.tsx`, purely for visual/UX iteration. If this design is approved, the real work is: (a) extend `decider/debug_bridge/bridge.py`'s `set_controls`/`set` commands (or a new one) to accept a param override and apply it going forward from the current checkpoint, (b) a real "Parameters" view in `tools/decider-ui/src/` (the actual `@decider/ui` package, not the demo harness) wired through `model/protocol.ts` messages.
- The demo's "Parameters" and "Datasets" features exist **only in `tools/decider-ui/demo/`** (the render harness) — nothing in `tools/decider-ui/src/` (the real shipped `@decider/ui` package) or `tools/vscode-decider/` was changed to add these features for real. That's deliberate — the user is still iterating on the design — but don't mistake the demo for the shipped product when picking this up.

## Suggested skills

- **`impeccable`** — this whole session was iterative frontend/UX design work (sidebar accordions, tab stacks, dataset tables, parameter editing). Load it for any further visual/interaction iteration on the demo harness or the real `@decider/ui` package. Note: an `impeccable` design hook ran automatically throughout this session and flagged one pre-existing, unrelated finding (a `.scenario-error` left border in `tools/decider-ui/src/style.css`), which was triaged as a sanctioned exception and suppressed via `ignore-value` — see `tools/decider-ui/.impeccable/config.json` if that needs revisiting.
- **`ponytail`** — was active throughout (user's global CLAUDE.md sets it on by default) for keeping the mock/demo code minimal; keep using it for any further harness work so it doesn't accidentally grow real functionality it doesn't need.
- **`code-review`** — worth running once the demo-harness design settles and the Track 1 backend/extension changes (bridge.py, rows.py, protocol.ts, adapter.ts, extension.ts) are ready to actually land/PR, to check them against the repo's coding standards independent of this session's own back-and-forth.
