# Handover: a record-first flow debugger

The intended redesign of the flow debugger UI (`tools/decider-ui`), its bridge
(`tools/decider-bridge`) and both hosts (`tools/vscode-decider`,
`tools/jupyterlab-decider`). Nothing here is built yet. The design was agreed
with the user on 2026-09-25; the open questions at the end still need answers.

## Why

The debugger today runs a whole batch and asks the user to focus a record
afterwards. Most views are about one record anyway, and batch-wide views
(first values, one "written by" per column) mislead when records take
different branches. What-if and Scenarios are separate tabs that run outside the
debug session, so their results can't be stepped through or seen in the graph.

## The design

### 1. Pick the data first

"Run flow" (the CodeLens, the ▶ Run flow button in the graph toolbar, or the
JupyterLab toolbar) opens a **data step** before anything runs:

- **Source:** the module's `SAMPLE`, the project's `sample_request.json`, or a
  JSON, JSONL, CSV or Parquet file. The user can change the source later
  without leaving the panel.
- **Table:** a searchable, filterable view of the rows. Show a sample, not
  every row: a million-row file must stay responsive. Search matches any
  column's text; a filter is a column, an operator and a value (the operators
  the breakpoint form already uses, `==` to `>=`).
- **Mode**, one of two:
  - **Single-record mode:** pick one record, or up to **10** to compare. The
    run contains only those records.
  - **Batch mode:** the run contains the whole (filtered) table, for pipelines
    whose steps need more than one record (frame steps, group-bys). The views
    show **10** records from it: the first 10, or ones the user picks.

The data step replaces today's quick pick of `SAMPLE` / JSON file / typed rows
(`pickData` in `vscode-decider/src/extension.ts`). JupyterLab starts runs with
`data: null` (`jupyterlab-decider/src/host.ts`), so it needs the same step.

### 2. Every view follows the picked records

- **One record:** the graph, state table, lineage and timeline are all about
  it; there is no "whole batch" state to switch to.
- **Several records** (up to 10, in either mode): each gets a colour, used
  consistently everywhere.
  - Graph: which branch arm and how many loop iterations each record took,
    marked in its colour.
  - State table: one value column per record. The "last changed by" / "last
    written by" toggle (already built, see below) applies per record.
  - A collapsible section per record in the data view, for details that don't
    fit side by side.

### 3. What-if becomes a fork

Changing a param or an input in the Params tab creates a **fork**: the same
records run with the change, shown as another coloured lane. A fork can be
stepped through, paused at breakpoints and inspected like any record.

When creating a fork the user chooses where it starts:

- **from the beginning:** a fresh run with the change;
- **from where the run is paused** (see the open questions): the run replayed
  to the pause point, then continued with the change.

`decider_bridge/forks.py` already does both for sweeps (`fork`, `sweep`, and
`Bridge.sweep(from_here=...)`): a fork is a fresh session replayed to the
checkpoint. What's new is keeping a fork alive as a steppable session instead of
running it to the end and returning a summary.

### 4. Tabs

**Graph · State · Params · Compare**

- **Params:** edit params and inputs; each edit is a fork (section 3).
- **Compare:** every lane side by side, records and forks. It shows where each
  first diverges and how the outcomes differ. It absorbs today's What-if and
  Scenarios tabs (`Scenarios.tsx`, `SweepResults.tsx`) and the existing
  comparison view (`Compare.tsx`).

### 5. Both hosts

Everything here applies to VS Code and JupyterLab alike. The UI is shared
(`decider-ui`); host-specific work is only how a run starts and how data files
are chosen. VS Code can use its file picker; JupyterLab needs a path box or its
file browser.

## Already built that this depends on

- **Per-record attribution:** `Bridge.state(row)` returns `writtenBy` and
  `changedBy` for each column (`lineage.attribution`). The state table has the
  "last changed by" / "last written by" toggle. A branch or loop merge is never
  credited; a step that wrote the same value again is "written" but not
  "changed".
- **Value metadata:** `decider.Money`, `Percent` and `Duration` (subclasses of
  `decider.FieldMetadata`), declared with `Annotated`. The bridge sends them in
  `describe()["fields"]`. The UI's `fieldOf(name, value)` returns the declared
  metadata, or else a guess from the name marked `assumed`. Per-record colours
  and columns should format through `formatValue` as today.
- **`build()` pipelines:** the bridge finds `def build(...)`, loads the latest
  `configs/` version like `decider serve`, and falls back to
  `sample_request.json` for data. The data step should offer that file as a
  source.
- **Breakpoints:** `{path}` watches pause before a step; the value form adds a
  condition. Forks need to honour the same controls.

## Where the work lands

- `decider-ui/src/model/protocol.ts`: `Tab` (today
  `"graph" | "state" | "params" | "scenarios" | "compare"`), `RunStatus.record`
  (one focused row; becomes the picked rows plus lanes), new messages for the
  data step and forks.
- `decider-ui/src/App.tsx`: at 493 lines, just under the 500-line limit. Split
  it before adding the data step and lanes; the per-tab bodies are the natural
  seams.
- `decider-ui/src/StateTable.tsx`: one value column per record or lane.
- `decider-ui/src/Graph.tsx`: per-lane path marks.
- `decider-bridge/decider_bridge/bridge.py` (476 lines, near the limit): a
  `rows`/`sample` command for the data step (search, filter, sample; never the
  whole frame), starting a run on picked rows, and long-lived forks. Put new
  logic in its own module, as `forks.py` and `lineage.py` are.
- `vscode-decider/src/extension.ts` (`pickData`, `runFlow`, `startShown`) and
  `jupyterlab-decider/src/host.ts` (`start`): open the data step instead of
  running straight away.

## Open questions

1. **Where a fork starts.** The user asked for forks "from the beginning or
   end". This note reads "end" as "from where the run is paused", since a fork
   from the end of a finished run has nothing left to run. Confirm.
2. **Batch mode's 10 records:** the first 10 by default, then user-picked? And
   is it 10 records shown, with forks as extra lanes on top, or 10 lanes in
   total?
3. **Filtering:** does it run in the bridge on the whole file (polars, so fast
   enough), with only the sample sent to the UI? That is the assumption above.
4. **Hints:** the user wants more usage hints across the UI (hover text, empty
   states), but hasn't yet named the screens that confused them. Ask before
   adding them to the new screens.
