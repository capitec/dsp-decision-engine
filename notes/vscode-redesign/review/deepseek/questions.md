# Redesign review — questions

Companion to `feedback.md`. Each question is a decision the docs leave open
that is cheap to answer now and expensive to answer wrong later. Grouped by the
seam it belongs to; the first block is the one that unblocks everything else.

## A. Grounding against the existing core (do this first)

1. **Canonical flow model.** Is the shared contract (task 01) a *layer over*
   `engine/ir` (`IRNode`, `CallNode`, `SequenceNode`, `Origin.path`,
   `step_map`) and `engine/run` (`State`, `RunReport`, `Version`), or a new
   parallel schema? If parallel, how do stable IDs (task 02) relate to
   `Origin.path`, which already gives derived path identity and is what
   breakpoints (`"term/cap_by_income"`) and `step_map` keys use today?
   >>> It is a versioned layer over the existing IR/run model, not a parallel
   schema. Task 01 must inventory and map `IRNode`, `Origin.path`, `step_map`,
   `State`, `RunReport`, and `Version`; durable IDs augment derived paths rather
   than replace them. <<<

2. **Which existing modules are the canonical ones?** Please confirm the list
   the tasks should extend rather than rebuild: `engine/debug/Session` + events,
   `steps/trees/trace.py` (`trace_output`/`path_output`),
   `testing/equivalence.py` + `testing/corpus.py`,
   `tools/decider-bridge/decider_bridge/forks.py` (`fork`/`sweep`),
   `serving/session_ws.py`. Anything missing from that list?
   >>> Treat this as an implementation-grounding question for task 01, not a
   design decision to answer from review notes alone. The listed modules are the
   required starting inventory; task 01 must also map the existing extension
   adapters and error vocabulary before naming canonical ownership. <<<

3. **Second adapter surface.** Task 09 says "notebook and extension are
   adapters over the same experiment interface." Is the notebook surface
   `jupyterlab-decider` (the bridge already serves it), or is the notebook
   explicitly out of scope for this pass? `tools/decider-ui` — in scope,
   out of scope, or migrate-to-extend?
   >>> The required second adapter proof is a plain Python caller, not a new
   JupyterLab feature. Preserve JupyterLab bridge compatibility during relocation,
   but `jupyterlab-decider` UI changes and `tools/decider-ui` migration are out of
   scope for this pass. <<<

## B. Debug adapter / bridge relocation (task 03)

4. **Naming and home.** `decider.debug_bridge` sits next to the real
   `decider.engine.debug.Session`. Is the relocated package named as an
   *adapter/transport* (e.g. `decider.serving.debug`) to avoid reading as a
   second session, or is `debug_bridge` kept and documented as "transport, not
   session"? Where does it live so `engine/` still imports nothing from it?
   >>> Keep `decider.debug_bridge` only if its documentation and public names say
   unambiguously that it is a transport adapter over `engine.debug.Session`, not
   a second session. It must sit outside `engine/`, and `engine/` imports nothing
   from it. The exact module home remains subject to the task 03 grounding pass. <<<

5. **Which transports survive?** There are three today: stdin JSON-lines
   bridge, `debugpy` attach, and the starlette websocket (`session_ws.py`).
   Which does VS Code use after the relocation, which does the notebook use,
   and do all three need to keep working through the move?
   >>> Preserve all currently supported transports through the relocation. Task 03
   must document their consumers explicitly: VS Code uses the JSON-lines bridge
   and debugpy integration where needed; JupyterLab compatibility includes its
   existing websocket path. No transport is removed as an incidental result of
   the package move. <<<

6. **Session vs. adapter boundary.** The `Session` already lives in core and
   owns breakpoints/events/overrides. What, precisely, moves into
   `decider.debug_bridge` — only `bridge.py` transport, or also `describing`,
   `lineage`, `forks`, `runs`, `timeline`? Some of those (`forks`, `lineage`)
   look like experiment/lineage primitives that belong in core, not transport.
   >>> Move only transport/process-launch concerns into the relocated bridge.
   Task 01/03 must classify existing helpers: session semantics remain in core;
   lineage/description belong with core flow/runtime queries; forks/sweeps are
   evaluated by task 09 as experiment primitives. Do not fossilise them in a
   transport package merely because they are there today. <<<

## C. Tracing (task 04)

7. **One "trace" or two?** Does the task-04 decision trace subsume, replace, or
   coexist with `trace_output`/`path_output` (per-tree/table result columns) and
   `Session.events`? Which mechanism feeds per-record explanation (debug, task
   07) vs. population aggregation (experiments, task 10)?
   >>> They coexist. `trace_output`/`path_output` remain cheap per-row path/leaf
   columns for population aggregation. `Session.events` remain debugger events.
   The decision trace adds structured explanation evidence; it should reconcile
   with—not blindly replace—both. Per-record explanation combines decision trace
   and DAP state; population aggregation prefers cheap path columns and selected
   trace-derived aggregates. <<<

8. **Conservation vs. redaction boundary.** The conservation check can only run
   pre-redaction; recommendation #3 says ~2/3 of events are personal data; and
   a deletion request must remove personal events. What is the exact order
   (capture → conservation check → adapter), and is the default adapter a
   pass-through that emits PII, or does "rich by default" get a default redaction
   mask? Who owns the deletion path, and how does it avoid tripping the
   conservation check?
   >>> The order is capture, conservation verification, then client post-record
   adapter. The default adapter is a rich pass-through; its client owns
   redaction/deletion/retention. Deletion downstream does not affect conservation,
   which validates capture before policy transformation. <<<

9. **Trace identity.** Every event must resolve to durable step identity (task
   04 "Done when"). If committed IDs (task 02) are optional during development,
   what do events carry when no committed ID exists — `Origin.path`? Is that
   stable enough for the "audit-grade" wording, or is the audit contract
   explicitly "only for flows with committed IDs"?
   >>> Events always carry derived `Origin.path` context and carry a committed
   durable ID when available. Audit-grade durable cross-run linkage is promised
   only for flows with committed IDs; development traces remain useful but have
   that limitation clearly stated. <<<

## D. Identity (task 02)

10. **ID syntax in source.** How is an ID attached to a step — a
    `step(..., id="…")`/decorator argument, a rename, or a separate manifest?
    What does the generated code look like, and what's the readability
    contract? How does an ID survive extraction, reordering, and rename?
    >>> The generator adds a short opaque ID through the existing step declaration
    or decorator argument, not a separate manifest or renamed symbol. It survives
    extraction, reordering, and rename because it is explicit source data. Task 02
    must prototype the exact syntax and reject any form that makes ordinary step
    definitions materially harder to read. <<<

11. **Merge/review behaviour.** The ID command requires a tracked, clean tree.
    What happens when two branches both generate IDs and merge — text conflict,
    or a merge-by-meaning rule like the recommendation #8 "data" work? Is the
    ID a source edit the author reviews, or a background convenience?
    >>> It is a deliberate, confirmed source edit reviewed by the author. v1
    detects duplicate IDs after merge and asks for resolution; it does not attempt
    semantic source merging. Opaque generated IDs minimise independent-branch
    collisions. <<<

## E. Data / execution scope (task 06)

12. **Record identity.** Keep the "first `id`/`*_id`/`id_*`" heuristic as
    default with an explicit override, or require an explicit ID for
    experiment workloads? How do Parquet/CSV (no schema inference) declare it?
    >>> Keep the heuristic as a loader suggestion; require an explicit ID or
    composite key when a saved experiment needs record drill-down. CSV/Parquet
    load settings carry the chosen columns and are captured in the run manifest. <<<

13. **Scope model.** Confirm "selected record / whole frame" maps onto the
    existing stepped vs. fused checkpoint granularity plus `FrameStep`
    (`fn(df)->df`) — i.e. frame scope is *not* a new semantic layer. Is the
    "hybrid mode" in scope, and if so, is it just "run stepped while a
    `FrameStep` receives the full frame"?
    >>> This is not a new flow semantic layer. Initial hybrid behaviour executes
    the full frame using existing debug/checkpoint semantics and uses record focus
    only for display, tracing, and breakpoints. Dependency-closure optimisation is
    deferred until it can prove the same result. <<<

14. **Sampling primitive.** For billions of rows, where does "select one record
    to debug" and "sample for experiments" live — core (`decider`), the editor,
    or MCP? Is there a defined partition/key assumption (the record ID), or is
    sampling a client policy only?
    >>> Core provides deterministic filter/sampling specifications so experiments,
    MCP, and VS Code mean the same thing; experiment YAML records them. Client
    policy chooses when/where to use them and supplies access to data beyond local
    capacity. <<<

## F. Experiments (tasks 09/10)

15. **Reproducibility fields.** `experiment.yaml` lists flow/revision/input/
    scenarios/summaries. Must it also pin the `decider` version and an input-data
    fingerprint/hash? Without them, Story 3's "unambiguous baseline and input
    provenance" is not satisfiable. Confirm both are added.
    >>> Confirmed. Authored YAML names the input/revision intent; each run manifest
    records resolved revision SHA, `decider`/Python environment, input content
    fingerprint, schema, row count, filters, sampling, and overrides. <<<

16. **`HEAD^` and engine drift.** `background.md` already notes historical code
    can fail to import on today's `decider`. Is the supported revision window
    "revisions compatible with the installed engine" (with a clear error
    otherwise), or is a per-revision interpreter path in scope? If it's a hard
    boundary, say so now so task 09's spike stops chasing it.
    >>> The supported window is revisions compatible with the installed engine.
    Per-revision interpreters/environments are out of scope; incompatibility fails
    clearly and is recorded in the result. <<<

17. **Cancellation/progress.** Billion-record runs and scenario sweeps need
    cancellation, progress, and timeouts (background.md flagged this gap for
    the old What-If). Does the experiment runner get them, and does the MCP
    "run experiment" tool surface cancellation? Where does it live (core runner
    vs. editor)?
    >>> Yes. The core experiment runner owns cancellable finite jobs, progress,
    timeouts, logs, partial manifests, and per-scenario failure isolation. VS Code
    and MCP expose the same job handle and cancellation operation. <<<

18. **Result ownership.** Results go to a caller-chosen directory (task 09).
    Is there any requirement for `decider` to write result *metadata* (a small
    manifest with baseline/revision/input hash/params), or is provenance the
    caller's responsibility entirely? Story 5 ("share a finding") needs at
    least a linkable metadata record.
    >>> `decider` always writes a small versioned result manifest in the selected
    results directory. It contains provenance, job/scenario status, output
    references, and finding descriptors; payload storage beyond that remains the
    client's responsibility. <<<

## G. MCP (task 11)

19. **Where does the server run?** FastMCP is a Python framework; the editor is
    TypeScript, and `decider` is the Python core. Does the MCP server run as a
    subprocess beside the bridge (like the debug bridge), or is the editor a
    *client* to a `decider`-hosted server? This determines task 11's "transport"
    and is not stated anywhere.
    >>> Run FastMCP as a `decider`-hosted Python process, normally over stdio for
    agent transport. VS Code is an optional local client/bridge for editor-bound
    actions such as highlight/reveal; the exact authenticated IPC is validated by
    the MCP topology spike before task 11 starts. <<<

20. **Confirmation granularity.** "Running a flow, starting a debugger, writing
    files, or other persistent actions require confirmation." Is "generate IDs"
    (task 02) classified as a persistent action requiring confirmation, and does
    "read" access include trace/record data that may be PII (and therefore
    subject to the client's trace adapter and M2's redaction default)?
    >>> Yes, generated IDs are a confirmed persistent action. Read access is broad
    by default but data classes—structure, summaries, raw records, raw trace—are
    separately configurable by workspace/client policy and constrained by what the
    trace adapter makes available. <<<

## H. Scope of the pass

21. **Ordering risk.** Task 02 (IDs) blocks 09/10/11, but the experiment
    interface (09) also depends on the trace contract (04) and the existing
    session/fork machinery. Should the identity *syntax* decision (Q10) be
    resolved before tasks 09/10 start, given experiments must reference
    step-level overrides by stable ID? The current graph leaves 09 gated only on
    01/02/06, not on 04.
    >>> Correct. Task 09 must depend on task 04 as well as 01, 02, and 06, and
    must reuse the existing session/fork machinery after its grounding pass. The
    ID source syntax freezes in task 02 before experiment assets can reference
    step-level override points. <<<

22. **What is intentionally out of scope?** The docs name three UI surfaces, two
    debug transports, three data formats, and "millions/billions" scale, but no
    single sentence states what this pass will *not* ship (e.g. regulated-domain
    classifications, remote compute, `decider-ui` migration). One explicit
    "out of scope" line would prevent tasks 05–11 from expanding into it.
    >>> Explicitly out of scope: regulated-domain policy classifications and
    centrally mandated rules; remote/distributed execution and backend adapters;
    trace retention/storage policy; JupyterLab UI changes and `decider-ui`
    migration; `decider2` support; and per-revision Python environments. <<<
