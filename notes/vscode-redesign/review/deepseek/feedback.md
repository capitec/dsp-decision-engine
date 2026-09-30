# Redesign review — feedback

Reviewed against the actual `decider` core, not just the extension, because the
design's central promise is that VS Code becomes an adapter over core modules.
Most of the problems below trace to one root cause: **the redesign reads the
core as a black box to be extended, but the core already contains large parts of
what tasks 03, 04, 08 and 09 propose to build.** Getting that boundary wrong now
is exactly the kind of core fix the goal says to avoid.

## Critical

### C1. The evidence base omits the core, so tasks 03/04/08/09 risk rebuilding what exists

`background.md` §"Evidence consulted" lists extension files, the README, e2e
stories, and the two `recommendations-for-decider-v2` notes. It never consults
the modules that already implement the proposed capabilities:

| Task | Proposes to build | Already exists in core |
|---|---|---|
| 03 relocate/harden debug bridge | "create a clear execution seam" | `decider/engine/debug/Session` (break/resume/step/set/rewind, events, `Edits`, `swap`), `decider/serving/session_ws.py` (websocket adapter) |
| 04 build decision tracing | "instrument every mode" | `Session.events` (NodeStarted/Finished/Visited/Overridden/…), `steps/trees/trace.py` + `trace_output`/`path_output` |
| 08 build checks | "revision comparison, structural equivalence, generated regime cases" | `decider/testing/equivalence.py` (`assert_equivalent`), `decider/testing/corpus.py` (zeros/negatives/empty/chunked — already the "generated cases"), recommendations #4/#6 already measured |
| 09 experiment module | "scenario sweeps, fork, replay" | `tools/decider-bridge/decider_bridge/forks.py` (`fork`, `sweep`, replay-to-checkpoint), `pipeline.session(df)` + `s.set("ratio", 0.1)` already documents "what-if" |

The redesign also re-derives findings the recommendations already produced
(trace ≈ 26 µs / 10.7 KB; trace-point conservation; regime-not-threshold case
generation). Re-deriving costs time; worse, the "shared flow model" (task 01)
and "decision trace" (task 04) will be designed as if greenfield and then
conflict with `IR`/`Origin`/`step_map` and `Session.events` during integration.

**Fix now:** task 01's "inventory" must start from `engine/ir` (`IRNode`,
`CallNode`, `SequenceNode`, `Origin.path`, `step_map`), `engine/run`
(`State`, `RunReport`, `Version`), `engine/debug` (`Session`, events), and
`testing/` (`equivalence`, `corpus`). Rewrite tasks 03/04/08/09 as *"extend /
expose / reconcile"* against named modules, not *"build"*.

> **Status:** the agent answered all 22 questions. See "Agent answers
> reviewed" at the bottom for the resolution map and the residual risks that
> remain after those answers.

### C2. `decider.debug_bridge` collides with `decider.engine.debug`, and the relocation is not "move a folder"

The bridge (`tools/decider-bridge/decider_bridge/bridge.py`) is already a thin
JSON-lines adapter over `decider.engine.debug.Session` plus helpers
(`describing`, `lineage`, `forks`, `runs`, `timeline`). Task 03 frames the
relocation as "preserve the debug protocol and create an execution seam", but:

- The *session* itself is already `decider.engine.debug`. Putting the transport
  at `decider.debug_bridge` creates a second top-level "debug" namespace right
  beside the real one, and the doc never distinguishes the adapter from the
  session it wraps. The "bridge owns transport/adaptation" sentence is the right
  instinct but the chosen name and home contradict it.
- The bridge serves **two** consumers: its own docstring says "the Python side
  of the VS Code **and JupyterLab** debuggers". The redesign only tracks VS Code.
- There are **three** transports over the same session: the stdin JSON-lines
  bridge, `debugpy` attach (`--debugpy PORT`), and the starlette websocket
  (`session_ws.py`). Task 03 mentions none; "preserve the debug protocol"
  doesn't say which protocols survive.

**Fix now:** name the boundary first (session = core; transport/adapter =
editor-facing), pick a home that doesn't read as a second session module (e.g.
`decider.serving.debug` or keep `decider.debug_bridge` *only* if the doc
explicitly says "adapter, not the session"), and state which of the three
transports VS Code uses and which of the two editor consumers (VS Code,
JupyterLab) the relocation must keep working.

### C3. Three UI surfaces, one redesign

`tools/` contains `vscode-decider`, `jupyterlab-decider`, **and** `decider-ui`.
`background.md` says "a notebook and the extension should be adapters over the
same experiment interface" but never names the notebook surface, and every task
is written for VS Code alone. Task 09's "notebook ergonomics" criterion will be
evaluated against an imaginary consumer unless the second adapter is pinned to
`jupyterlab-decider` (or explicitly descoped). Same for the bridge relocation
(C2): its *other* consumer is the notebook.

## Major

### M1. "Trace" is two different things and the doc never separates them

There is (a) `trace_output`/`path_output` — a per-tree/per-table **result
column** naming the path/leaf, and (b) `Session.events` / the proposed task-04
**event-level decision trace**. Task 04 never mentions either. At population
scale (task 10 aggregation, Sankey path counts) the natural source is the
cheap per-row path column, not a per-record event trace of 10 KB. The design
must state whether "decision trace" subsumes, replaces, or coexists with
`trace_output`, and which mechanism feeds population aggregation vs. per-record
explanation — otherwise debug (07) and experiments (10) will wire to different
"trace" concepts.

### M2. Privacy/redaction vs. trace-point conservation is unresolved

`background.md` and task 04 both require (i) trace events "rich by default,
client owns redaction/deletion/retention via a post-record adapter" and (ii) a
"trace-point-conservation check" so optimisation can't silently drop evidence.
Recommendation #3 already measured **~2/3 of trace events are personal data**.
The conservation check can only run *before* redaction; "rich by default" emits
PII with no retention guard by default; and a deletion request (recommendation
#3's stated goal) requires deleting personal events *without* the conservation
check treating it as evidence loss. Task 04's "adapter failure behaviour" bullet
is the only nod. The seam (capture → conservation check → adapter) and its
defaults need pinning now, because the trace schema is the backbone of both
debug (07) and experiments (10) and it's the hardest thing to retrofit privacy
into later.

### M3. Durable IDs have no source syntax, and the ID command is an unexamined "generated source" exception

Task 02 requires a command that rewrites pipeline source to insert IDs, but
never says **how** an ID attaches to a step in source (decorator arg? rename?
separate manifest?). `Origin.path` already gives derived identity, and the
recommendations already documented the honest limitation: structure lives in
Python, so IDs on *steps* are a new kind of source edit the repo has avoided.
The repo norm ("no code generation", spec-amendments) is about engine kernels,
but the ID command is the first deliberate source-rewriting CLI and needs its
own justification: what guarantees readability, what makes an ID survive
extraction/reorder, and what happens on merge conflict. Without a chosen
syntax, task 02 is underspecified at exactly the point that matters, and the
MCP confirmation policy (task 11) must also classify "generate IDs" as a
persistent action (it currently lists "source generation" under
confirmation-gated, which is right — but task 02 and task 11 don't agree on the
term).

### M4. Record identity is still the old heuristic

`background.md`: records use "the first `id`, `*_id`, or `id_*` field". Task 06
says "stable record identifiers" but never resolves whether the heuristic stays,
becomes explicit, or how Parquet/CSV (no schema inference) declare an ID. For
billions of Redshift rows, the ID is also the partition/join key, and it's what
task 10's "drill from aggregate → record → debug run" depends on. Decide: keep
the heuristic as default + an explicit override field in the data-load surface,
or require an explicit ID for experiment workloads.

### M5. "Execution scope" is the stepped/fused distinction, already modeled, and the doc doesn't connect them

The user's open-question answer #5 and task 06's "hybrid mode" are re-deriving
what the debug session already knows: fused runners pause at kernel boundaries
and can't `set` values computed inside a kernel (see `engine/debug/session.py`
and `notes/debug-session.md`). "Whole frame vs selected record" maps onto
stepped vs fused *and* onto `FrameStep` (`fn(df)->df`, joins/group-bys). Task 06
should be stated in those terms ("frame steps are `FrameStep` nodes; record
scope = stepped/fused checkpoint granularity"), not as a new semantic layer.

### M6. Scale promise has no sampling primitive, no version pin, and `HEAD^` ignores engine-version drift

- "Millions/billions" (userstories #11) is claimed for experiments, but the
  only mechanism offered for exceeding local capacity is "clients must sample"
  (decision #14) — with no sampling primitive named and no decision on whether
  it lives in core, editor, or MCP. Selecting *one* record out of billions to
  debug (task 10) is a load/store problem, not a graph problem, and nothing
  addresses it.
- `experiment.yaml` (task 09) lists flow/revision/input/scenarios/summaries but
  **no `decider` version pin and no input-data fingerprint**. Story 3's
  acceptance ("unambiguous baseline and input provenance") can't hold without
  both. Add them to the asset model now.
- `background.md` already documents the `HEAD^` failure mode: revision
  comparison runs historical code on *today's* `decider`, and old code that
  needs an older engine interface "can fail to import". Task 09 lists `HEAD^`
  as a spike without acknowledging this is a hard boundary: you cannot run
  arbitrary-past code on the current engine. State the supported window
  (e.g. "revisions compatible with the installed engine") or design a
  per-revision interpreter path explicitly.

### M7. Cancellation and progress were a known gap and still are

`background.md` §"What-If" notes "Scenario combinations can be expensive and no
clear cancellation interaction was found." None of tasks 09/10/11 (or the MCP
confirmation model) mentions cancellation, progress, or timeout for
billion-record experiments or scenario sweeps. The bridge already has
`pause`-while-resume semantics over the session; the experiment runner needs an
equivalent. This is a concrete regression risk if left until task 12.

## Minor / worth tightening

- **Task 01 "shared flow model" is a layer over `IR`/`Origin`/`step_map`, not a
  new schema.** It says "inventory … identify which concepts can become
  canonical" but should name them (`IRNode`, `Origin.path`, `step_map`, `State`,
  `RunReport`, `Version`) so downstream tasks depend on the layer, not invent
  parallel identifiers (task 02's IDs already exist as `Origin.path`).
- **"Static vs runtime facts" (task 01) is already the boundary between `IR`
  and `RunReport`/`Session.events`.** Reuse it; don't define a third.
- **Task 05 doesn't say what happens to `analysis.ts`, `structure.ts`,
  `sourceMap.ts`, `compareRuns.ts`.** These already encode flow/lineage/
  comparison logic; the "redesign" must say reuse vs. re-implement or the new
  graph/inspector will duplicate them.
- **`background.md` line 97 says "restart a debug session with entered
  parameters" while Story 2/decision #6 separate live edits from reproducible
  scenarios.** The docs are internally consistent enough, but the word "restart"
  vs. "fork/replay" matters now that `forks.fork` already implements
  replay-to-checkpoint; align the vocabulary.
- **`decider.ui` (third surface) and `tools/decider-ui` are never mentioned**
  beyond the missing-namespace point in C3; even a one-line "out of scope"
  decision prevents scope creep during tasks 05/07.
- **Error/capability reporting (task 01) should reuse `decider.exceptions`**
  (`WiringError`, `IRError`, `ParamsError`, `MissingInputError`, …) rather than
  invent a new error vocabulary, so MCP (task 11) and VS Code surface the same
  codes.

## Where the high-level boundaries must be fixed now (not later)

These are the seams where a wrong guess now forces a core fix later:

1. **Flow/selection contract (task 01)** — a versioned layer over `IR` +
   `Origin.path` + `step_map` + `State`/`RunReport`, with static-vs-runtime
   made explicit. Wrong guess = every consumer invents identifiers.
2. **Trace event contract (task 04)** — reconciled with `Session.events` and
   `trace_output`, with the capture → conservation-check → adapter seam and its
   privacy defaults fixed. This is the backbone of both 07 and 10.
3. **Experiment interface (task 09)** — built on `session` + `forks.fork`/
   `sweep` + `testing.corpus`/`equivalence`, not a new execution path.
4. **Debug adapter boundary (task 03)** — session stays in `engine.debug`; the
   relocated adapter is named and scoped as transport, and its two consumers
   (VS Code + JupyterLab) and three transports are enumerated.

Get these four seams right and the rest refines without core churn. Get them
wrong and tasks 05–11 will rework them.

---

# Agent answers reviewed

The 22 questions were answered. Most answers resolve the findings above; a few
residual risks and new issues remain. This section is the delta you asked for.

## What is now resolved

| Finding | Resolved by | Note |
|---|---|---|
| C1 rebuild-vs-extend | Q1, Q2 | Contract is a *versioned layer over IR/run*, IDs augment `Origin.path`. But see R1. |
| C2 naming / transports | Q4, Q5 | `debug_bridge` kept only if documented as transport; all three transports preserved. |
| C3 second adapter | Q3 | Plain Python caller is the proof; JupyterLab UI + `decider-ui` out of scope. |
| M1 one trace or two | Q7 | Coexist: `trace_output`/`path_output` for aggregation, `Session.events` for debugger, decision trace adds explanation evidence. |
| M2 conservation/redaction order | Q8 | Capture → conservation → adapter; deletion is downstream of conservation. |
| M3 ID syntax | Q10, Q11 | Short opaque ID via step declaration/decorator; v1 detects post-merge duplicates, no semantic merge. |
| M4 record identity | Q12 | Heuristic = loader suggestion; explicit ID/composite key required for experiment drill-down. |
| M5 scope model | Q13 | Not a new layer; hybrid runs full frame, record focus is display/trace/breakpoints only. |
| M6 scale / pin / HEAD^ | Q14, Q15, Q16 | Core owns deterministic filter/sampling spec; run manifest pins env + input fingerprint; supported window = installed-engine-compatible revisions. |
| M7 cancellation | Q17 | Core experiment runner owns cancellable jobs, progress, timeouts, partial manifests, failure isolation. |
| (none) MCP server home | Q19 | `decider`-hosted stdio process; VS Code is an optional local client/bridge. |
| (none) confirmation | Q20 | "Generate IDs" is confirmation-gated; read data classes separately policy-gated. |
| (none) ordering | Q21 | Task 09 gains a dependency on 04; ID syntax freezes in 02 first. |
| (none) out of scope | Q22 | Explicit: regulated classifications, remote execution, trace retention, `decider2`, per-revision environments. |

## Residual risks (answers that defer or leave a seam soft)

- **R1 — "extend, not rebuild" is still only implied, not mandated.** Q2
  classifies the module list as "an implementation-grounding question for task
  01, not a design decision." That is the load-bearing decision of this whole
  review, and it is now the one thing not explicitly committed. Add a clause to
  task 01's "Important decisions": *extend the canonical modules (`engine/ir`,
  `engine/debug`, `testing/`, `steps/trees/trace.py`); do not rebuild them.*

- **R2 — the trace default still emits PII.** Q8 keeps "rich pass-through" as
  the default adapter, with redaction owned by the client. Q20 mitigates MCP
  *read* access via data-class policy, but the *capture default* is still
  PII-rich with no retention guard. For a governed context, state a default of
  "no capture / no-op adapter until the client configures one"; otherwise the
  safe-by-default claim doesn't hold.

- **R3 — the bridge relocation is now a decomposition, and task 09 inherits it.**
  Q6 splits the bridge: transport/process-launch stays in the relocated adapter;
  `lineage`/`describing` move to core flow/runtime queries; `forks`/`sweeps`
  move to core as experiment primitives. This is right, but it means (a) task 03
  must not re-package the helpers wholesale "to preserve the launch protocol,"
  and (b) task 09 now silently depends on the forks/sweeps rehoming, which no
  task owns. Add an explicit deliverable to 01 or 03: "rehome `forks`/`sweeps`
  and `lineage` into core, then task 09 consumes them."

- **R4 — the MCP answer introduces an unstated spike and a new security surface.**
  Q19 adds a "MCP topology spike" and "authenticated IPC" between a
  `decider`-hosted stdio server and a VS Code client/bridge. Neither appears in
  task 11. Fold the spike into task 11, and give the local editor↔Python-server
  channel a one-line threat note (it is where "read broad by default" meets an
  authenticated agent transport).

- **R5 — the task dependency graph is now stale.** Q21 makes task 09 depend on
  04 and on the forks/sweeps rehoming, but 09's header still says "Depends on:
  01, 02, 06," and no task owns the rehoming. Update 09's dependencies and add
  the R3 deliverable.

- **R6 — who materialises beyond-local data is still undefined.** Q14 puts the
  filter/sampling *spec* in core, and Q15 fingerprints input content, but "client
  supplies access to data beyond local capacity" leaves the actual read path
  (who loads Redshift rows into a Polars frame, where the fingerprint is
  computed) unowned. Fine as a boundary, but state it so tasks 06/09 don't assume
  core reads the warehouse.

- **R7 — the hybrid "third mode" is display-only; reconcile the wording.** Q13
  says record focus affects display, tracing, and breakpoints only, with
  dependency-closure optimisation deferred. `userstories.md` Story 2 still
  describes "a third mode that traces the selected record while retaining the
  full frame" as if it might be an execution mode. Rewrite that line to match, or
  someone will build an execution mode that isn't needed.

## New findings

### N1. Graph scale is a first-class requirement, not an optimisation

The examples include flows of **thousands of nodes with dense edges, ~6–10
levels deep** (e.g. `example_projects/10-retail-credit-e2e/sonnet/pipeline.py`,
which assembles a large flow from imported `credit_core` and `retail_credit`
step modules rather than defining it inline). Task 05 currently treats "large
flows" as one bullet. At this scale:

- **Virtualised rendering + progressive disclosure + expansion/collapse are the
  primary strategy, not a fallback.** A few thousand nodes will not render as
  an SVG/DOM webview without windowing. `graphPanel.ts`/`analysis.ts` are
  unproven at this size and need a benchmark against the retail-credit-e2e flow
  before the redesign commits to a rendering approach.
- **The "select a value → all touch points" view is O(edges) by itself.** With
  dense edges, the inverse-dependency view (dead decision #2) can produce a
  node fan-out as large as the graph. It needs its own progressive/limit story,
  or it becomes the slowest interaction in the tool.
- **Flow discovery must find the *assembled* pipeline, not per-file steps.**
  The 1000s of nodes span `credit_core` plus project modules; the custom finder
  (open question #1) has to resolve the entry-point pipeline and its imported
  steps, not just "a module-level pipeline value" in one file. This sharpens
  open question #1: discovery needs to report *which* pipeline (entry point)
  and its assembled size before the user opens the graph.

Add a scale acceptance signal to Story 1 / task 05: the retail-credit-e2e flow
must be navigable (zoom/pan/expand/fit) at interactive frame rates, with the
touch-point view bounded.
