# VS Code redesign background

This note captures the current `vscode-decider` extension as investigated in
September 2026, and the outcome the redesign should pursue. It is a reference
for the user-story work in [userstories.md](userstories.md).

## Redesign aim

Make `vscode-decider` easy to navigate and useful at three distinct scales:

1. **Understand a flow**: discover pipelines, learn their structure and data
   dependencies, and reach the responsible source.
2. **Investigate an execution**: load data, trace one or more records, inspect
   and debug a concrete run, and understand values at each step.
3. **Explore behaviour**: conduct reproducible comparisons across a population,
   parameters, versions, and controlled conditions.

The extension should share a consistent flow and source model across these
modes, but should not make users infer whether they are debugging a run,
changing a live run, or executing a reproducible experiment. Agentic tools
should be able to inspect and focus the same model through a deliberate MCP
interface.

## Current extension surface

### Entry points

- Python pipeline files expose CodeLenses: **Run flow**, **Visualise flow**,
  **What-if**, and **Compare with…**.
- The Activity Bar contains a **decider** view with a **Structure** tree.
- The Command Palette exposes commands for visualisation, running, What-If,
  revision comparison, record focus, tracing, and opening pipelines.
- The sidebar discovers `pipeline.py` candidates and offers a picker when more
  than one candidate exists.
- The main flow interface is a retained graph webview that opens beside the
  editor.

The present interaction model is therefore source-file and CodeLens led. Users
need a recognised Python pipeline open to see the clearest entry points.

### Flow structure and analysis

Visualising a pipeline imports it and describes its structure without executing
it on user input. Module-level Python import behaviour still applies.

The graph and Structure view expose:

- calls, inputs, outputs, state, and visited tree positions;
- source navigation for selected nodes;
- reads, writes, and parameters in Structure tooltips;
- state search and read/write indicators for the selected node;
- per-column values, versions, history, and lineage;
- record-focused views that narrow values and paths to one row;
- flow comparisons that identify changed, added, and removed steps, parameter
  and read/write changes, output differences, and first divergence.

There is one retained graph panel rather than independent views per flow. The
extension also moves source files between editor groups to keep the graph
visible during debugging.

### Run and debug

**Run flow** accepts sample data, a JSON file, or pasted JSON and starts a
`decider` debug session, normally paused on entry. The extension supports
standard debugger execution controls, source and function breakpoints,
run-to-node, source navigation, rewind/restart from a stack frame, and an
action to debug the selected step with Python/debugpy.

Variables may be edited during a run. These edits are tracked as named
overrides and are session-local rather than source edits. The extension also
supports graph-panel actions to skip, reload or swap, and restore a step, then
compare that changed execution with the original.

The current debugging integration is capable, but its entry points and the
meaning, scope, and lifecycle of live overrides are not sufficiently apparent
in the user interface.

### Records and data

Records use the first `id`, `*_id`, or `id_*` field as their UI identifier.
Users can focus a record through the graph panel or a Command Palette action,
and can emit column lineage to a dedicated output channel.

The current documented run-data path is sample data, JSON file, or pasted JSON.
The redesign must establish the desired support and semantics for CSV and
Parquet, data discovery, schema inspection, search/filtering, and larger data
sets.

### Parameters, scenarios, and What-If

What-If is accessed through its CodeLens or `decider: What-if (params and
inputs)`. It first ensures the active pipeline is visualised, then opens the
graph panel's **Params** tab.

Current functionality includes:

- viewing parameter types, defaults, and bounds;
- overriding parameters and input values for one record or all records;
- running a changed configuration against a default/current baseline;
- restarting a debug session with the entered parameters;
- defining scenario knobs for parameters and input fields;
- evaluating each Cartesian combination as a separate scenario;
- running scenarios from scratch, or from a paused checkpoint when the session
  can be forked;
- showing selected-record outputs and differences for scenario results;
- opening a step-by-step comparison from a scenario result.

Important current semantics:

- A paused-session scenario includes changes already present in the session.
- Parameters affect only subsequent steps from the pause; an input change
  replaces its value at that point.
- A session that was rewound cannot be forked.
- If no suitable paused session exists, a requested fork falls back to a fresh
  run.
- Scenario combinations can be expensive and no clear cancellation interaction
  was found.

This makes What-If powerful but difficult to orient: it shares the graph panel
with navigation and debugging, while mixes fresh comparisons, session-local
changes, and checkpoint-forked scenarios.

### Comparison and revisions

The extension compares current and historical pipeline code using HEAD, tags,
branches, and recent commits. It materialises old code from Git into a
temporary cache and runs it against the same rows; the working tree is not
changed. It can also open a normal VS Code code diff.

Revision comparison uses today's installed `decider`, so historical code that
requires an older engine interface can fail to import. Debugger source paths
from a historical run can also point into temporary extracted code rather than
the active workspace.

### Configuration and prerequisites

The extension requires separately installed `decider` and `decider-bridge`.
`decider.python` accepts a command array, for example `["uv", "run", "python"]`;
otherwise it falls back to the Python extension's interpreter and then
`python3`. When the environment cannot import the required packages, the
Structure tree is empty or shows an error rather than guiding setup.

## Initial usability findings

### Flow orientation

The graph is the right primary artefact, but data-dependency arrows and labels
currently overload it. Reads and writes should be inspectable in a dedicated
step and edge inspector:

- a selected step should state all values read and written;
- a selected edge should state the values it carries and their relevant
  producers/consumers;
- runtime values should be clearly differentiated from static structure;
- the graph should reserve labels for information that remains useful at the
  current zoom level.

Expansion and collapse must preserve the viewport and selection. Direct
zoom/pan gestures and fit-to-selection are important for larger graphs.

### Execution investigation

The extension already has strong underlying debugger support, but graph-visible
breakpoints, pause state, and live override scope need a clearer interaction
model. Record-oriented and frame-oriented work need explicit execution modes:
the tool must never silently run a frame-sensitive step on an insufficient
subset of data.

### Experiments

The current What-If implementation is an initial experiment facility, but
population-level questions need first-class aggregate results: decision-path
counts, divergence locations, changed/unchanged/unique-value summaries, and
visualisations such as Sankey diagrams. Aggregate findings should drill into
records and then the execution-investigation mode.

Experiment definitions should be saved and reproducible, including input
provenance, pipeline revision, controlled changes, and result references.

## Design direction

The initial recommendation is:

1. Keep a shared flow model that provides stable identifiers for flows, steps,
   edges, values, source locations, runs, and selections.
2. Give orientation, execution investigation, and experiments distinct entry
   points and result views.
3. Treat debugging as an execution-inspection workflow and What-If as a
   reproducible-experiment workflow. Permit an experiment result to launch a
   targeted debugger, rather than merging both interactions.
4. Put the core experiment model in `decider` if notebooks and the extension
   must produce the same semantics. The extension and notebook interfaces would
   then be adapters over a shared, deep experiment module.
5. Add a small MCP interface around high-leverage capabilities—flow discovery,
   flow/subgraph description, step/value lineage, run/experiment summaries, and
   controlled highlighting or navigation—instead of exposing every UI action.

## Tracing and governance

The broader `decider` review adds two capabilities that materially strengthen
the extension redesign. Both belong in the `decider` core first; VS Code is an
adapter that renders and navigates their results.

### Decision tracing

Decision tracing is evidence of how an individual decision was made: matched
rules, selected table rows or scorecard bands, applied rounding, reasons, and
the relevant path through the flow. `decider` emits this structured evidence;
it does not retain it or own its privacy policy. A client-selected post-record
adapter handles delivery, redaction, enrichment, persistence, and retention
outside the kernel. It is not interchangeable with operational observability:

- An OpenTelemetry span tree is useful for latency, failures, and runtime
  operations.
- A decision trace must remain compact, decodable, retained according to
  governance requirements, and capable of explaining domain outcomes.

The core should therefore define trace points and compact trace capture,
including trace-point-conservation checks for compiled/optimised execution.
Trace expansion and post-record handling happen outside the request path. The
default adapter may be a pass-through and need not be JIT compiled; it may
operate on another thread or process. OpenTelemetry is a likely adapter and
wire format, but the adapter seam must let clients choose a different sink or
transform events before they are delivered.

For VS Code, a trace becomes the runtime evidence model behind the
execution-investigation experience: selecting a step shows its trace events,
input/output/reason evidence, and source; selecting a reason or value
highlights its path. The extension should consume traces rather than invent its
own per-step debugging record.

### Governance

`decider` needs a governed Python interface that evaluates flows and revisions
and returns structured reports. It should cover:

- whole-path checks, such as required affordability or disclosed-reason
  conditions;
- common-defect diagnostics, including wall-clock reads, numeric
  over/underflow risks, and floating-point sensitivity;
- version comparison, with both structural proof where possible and
  generated boundary/regime cases where proof is not possible;
- stable identifiers and fingerprints for parameters and configurable flow
  assets; and
- client-selected check suites, plus an extension mechanism for clients that
  need their own checks.

VS Code should expose reports and their concrete evidence—a failing business
path, source location, affected value, or revision difference—and provide
navigation to them. It should not embed separate rule implementations.

Stable step IDs are particularly useful for source navigation, debugging,
parameters, trace decoding, and comparisons. A deterministic derived
identifier is useful for initial discovery, but cannot promise stability across
refactors such as extraction or reordering. A command that generates explicit,
committed IDs provides the durable identity needed by production pipelines. It
must run only against tracked, clean source and must add IDs without making
pipeline code materially harder to read; a manually supplied ID is an
override, not the only path to stability.

## How these findings were reached

This was a code-and-test investigation of the current extension, not a
hands-on usability study. The investigation:

1. Mapped the extension manifest to identify registered commands, CodeLenses,
   views, debugger configuration, and settings.
2. Traced each entry-point command through the extension implementation to its
   graph, debugger, comparison, data, and Git behaviour.
3. Read the user documentation to separate intended workflows and documented
   limitations from implementation details.
4. Reviewed the end-to-end and story tests to confirm the exercised flows,
   including screenshots for the visible graph, state, parameter, comparison,
   and revision interfaces.
5. Classified statements in this note as follows:
   - **Current surface** records implementation or documented behaviour.
   - **Initial usability findings** identify likely friction from the exposed
     interaction structure, combined with the user feedback that prompted this
     redesign.
   - **Design direction** is a recommendation, not an implemented capability.

No user observation, telemetry, task-completion study, or accessibility audit
was performed. Usability conclusions should therefore be validated with real
users during the redesign.

## Evidence consulted

- `tools/vscode-decider/src/extension.ts`
- `tools/vscode-decider/src/analysis.ts`
- `tools/vscode-decider/src/structure.ts`
- `tools/vscode-decider/src/graphPanel.ts`
- `tools/vscode-decider/src/adapter.ts`
- `tools/vscode-decider/src/git.ts`
- `tools/vscode-decider/src/launchArgs.ts`
- `tools/vscode-decider/package.json`
- `tools/vscode-decider/README.md`
- `tools/vscode-decider/test/e2e/`
- `notes/recommendations-for-decider-v2-summary.md`
- `notes/recommendations-for-decider-v2-tasks.md`

The README and end-to-end stories cover graph navigation, records and paused
state, parameters and scenarios, comparisons and revisions, value history,
breakpoints, loops, large flows, and edits.
