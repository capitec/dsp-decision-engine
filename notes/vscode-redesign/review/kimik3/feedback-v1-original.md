# Kimi K3 feedback — notes/vscode-redesign

Independent review of `background.md`, `userstories.md`, and `tasks/01`–`12`.
Judged against the stated goal: the high-level components should be designed
well enough that later refinement won't force major changes or core fixes.

## Verdict

The component separation is sound and I would not restructure it: three user
modes, one shared flow/source model, semantics pushed down into `decider`
(tracing, checks, experiments), the extension and notebooks as adapters, and a
deliberately small MCP surface. Those are the right load-bearing walls.

The risk is not the walls — it is a set of **cross-cutting semantics that no
task currently owns**. They are cheap to decide now (a paragraph each in task
01/04/06/09) and expensive to retrofit, because every consumer (extension,
MCP, notebooks, saved experiment assets, retained traces) will have baked in
its own assumption by then. That is precisely the "core fix later" category.

Below: flaws/oversights grouped by theme, each marked **[core]** (decide now,
retrofitting hurts) or **[refine]** (safe to work out during implementation).

## 1. Identity and provenance semantics are under-specified

### 1a. Record identity is still a heuristic [core]

`background.md` keeps "first `id`, `*_id`, or `id_*` field" as the record
identifier. But record IDs become load-bearing in the redesign: search/filter
(story 2), scenario-to-debug reproduction (task 07), experiment drill-down
(task 10), MCP references (task 11), saved findings. Unanswered: duplicates in
one dataset, datasets with no id-like field, composite keys, and whether a
synthetic row-number fallback is stable under filtering/sorting. Task 01
defines "input data identity" but not **record identity within** a dataset.
Decide: explicit ID-column declaration at load time (heuristic stays as a
default guess), plus behaviour on duplicates/missing.

### 1b. Data provenance is named but never defined [core]

Story 3's acceptance signal demands "unambiguous baseline and input
provenance", and `experiment.yaml` is a versioned asset. A file path is not
provenance — the file can change underneath. Define now: content hash (or
schema + row-count + hash), how Redshift-sourced extracts are fingerprinted,
and whether re-run warns or errors on mismatch. One line in task 09's asset
model; very painful to add once result directories exist in the wild.

### 1c. Run identity lifecycle [core]

Task 01 lists "run" as a shared entity but nothing says whether run IDs are
ephemeral (a debug session) or persistent (referenced by experiment results
and traces). If persisted results reference ephemeral run IDs, links rot
silently. State the lifecycle and the persistence boundary.

### 1d. Stable step IDs: the surrounding rules, not the syntax, are the risk [core]

Task 02 covers generation-safety (tracked + clean tree) well. What's missing:

- **Uniqueness scope** — per flow, per workspace, per repo? Collision
  detection when copy-paste duplicates an ID?
- **Rename drift** — an explicit ID survives a step rename (that's the point),
  so IDs and names diverge over time. Is there a check that flags drift, or is
  divergence accepted?
- **Merge ergonomics** — two branches each add a step; sequential or
  name-derived IDs will conflict; opaque IDs won't but hurt readability. The
  syntax choice should be made *for* merge behaviour, not only readability.
- **Unresolved references** — an `experiment.yaml`, saved trace, or MCP query
  references an ID that no longer resolves after a refactor. Error, warn, or
  best-effort? This will happen constantly; pick the behaviour now.
- **Enforcement** — "required where traces/comparisons/experiments need
  durable links" (decision 12) implies a missing-ID diagnostic. That belongs
  naturally in the task 08 check suite; the tasks don't connect them.

## 2. Tracing: ordering, execution modes, and the debugger boundary

### 2a. Ordering guarantees under concurrency are undefined [core]

Task 04's "done when" says "consume a correctly ordered trace". Ordered *how*?
Experiments target millions of records with Polars/compiled execution, and the
engine's own performance work explores parallel kernels. If the trace schema
assumes a total order and execution later goes parallel (even per-record
parallel), the schema needs surgery. Decide now: per-record event streams with
a record key (my recommendation), or a documented total order.

### 2b. Compiled-kernel trace capture is the highest technical risk — spike it first [core]

Serving kernels are `nogil=True`; you cannot allocate Python objects or call
back into Python freely inside them. Task 04 acknowledges "measured spikes
before the public interface freezes" — good — but the spike sits *inside* a
task that blocks 07, 10, 11, and 12. If nogil capture forces a buffer-based,
post-hoc-decoded design, that constraint shapes the entire event schema.
**Pull the compiled-kernel capture spike out of task 04 and run it as early as
possible, alongside 01/02.** A late failure here cascades through the whole
dependency chain.

### 2c. Trace vs. debugger inspection needs one explicit sentence [core]

Background says the extension "should consume traces rather than invent its
own per-step debugging record", and task 07 says runtime views consume trace
data "where available". But a compact decision trace deliberately does not
capture arbitrary locals/frame state — which is exactly what a debugger's
Variables view is for. And with tracing off (the no-op path), the runtime
inspector still needs to work. State the division of labour: **trace =
decision evidence (path, rules, reasons); DAP = arbitrary live state**. Both
feed the inspector; neither replaces the other. Otherwise 07 will either
starve the debugger or balloon the trace schema.

### 2d. The extension is itself a trace client — its own data handling is unstated [refine]

Trace events are "rich by default"; redaction is the client adapter's job. The
VS Code extension is a client: it will render values in webviews and (today)
emits lineage to an output channel. State the extension's own policy:
session-memory only, no disk persistence, what happens on window reload, and
whether MCP-served trace data follows the same rule.

## 3. Execution scope and debugging semantics

### 3a. The hybrid scope has a dependency trap — say what it actually is [core]

Open question 5 / task 06 defer the hybrid mode ("focus a record, keep the
full frame for frame steps") as "if semantically clear". The trap: if a record
step runs on one row, its outputs only exist for one row, so a **downstream**
frame step reading those outputs cannot see the full frame unless the record
step ran on all rows too. So "minimal execution" is determined by the
read/write dependency graph, not by the step's own kind.

The cleanest semantics is probably: **always execute the full frame; the
record focus only scopes tracing, display, and breakpoints.** Record steps are
row-local by construction, so a per-record trace is well-defined under a
full-frame run; frame steps get what they need for free. Cost: you always pay
full-frame execution. If a cheaper mode is wanted, it must be stated as
"execute the upstream dependency closure of the focused record at frame
granularity" — derivable, but it must be the written design, not a deferred
feasibility question. Either way, decide in task 06, not during implementation.

### 3b. Debug-console edits vs. tracked overrides — draw the line [core]

Decision 6 allows live edits "in the style of normal Python debugging", and
story 2's acceptance signal says "every live override states whether it
applies to this pause, this run, or a future rerun." But a Python debug
console permits arbitrary mutation (call functions, rebind globals) whose
downstream effect is unknowable. Scope/lifetime labelling is only achievable
for **structured edits** (params/state through the UI or session API). State
explicitly: console edits are untracked and excluded from any reproducibility
story; only structured edits become named overrides. Otherwise the acceptance
signal is unattainable.

### 3c. The What-If narrowing should be admitted as a capability cut [refine]

Today's What-If can fork a scenario from a paused session *including arbitrary
session state already present*. Decision 15 constrains reproducible overrides
to declared override points/step outputs — cleaner and reproducible, but a
real reduction of current power. The notes never say "we are dropping X".
Call it out in background/design with the migration story (a forked-session
workflow becomes: save scenario → relaunch debugger → apply structured
overrides), so users hitting the missing capability read it as a decision,
not a regression.

## 4. Experiment model

### 4a. Numeric equality semantics for comparison [core]

Experiments promise changed/unchanged/unique summaries and first-divergence
detection. The engine accumulates running totals in float64. "Changed" needs
an equality policy per type — exact for int64 cents, and what for float64:
exact, tolerance, per-field config? This decision shapes the comparison code,
the result schema, and every experiment result's meaning. It belongs in task
09's spike list; retrofitting tolerance semantics invalidates saved results.

### 4b. Nondeterminism: detected for governance, uncontrolled for experiments [core]

Task 08 detects wall-clock reads as a diagnostic. But a baseline-vs-variant
experiment where a step reads `datetime.now()` (or unseeded randomness)
produces spurious divergence. The experiment runner needs at minimum to
surface non-purity warnings at run time; ideally a controllable clock/seed.
Decide which is in scope.

### 4c. Cancellation, progress, partial results [core-ish]

Background flags "no clear cancellation interaction" for scenario sweeps as a
current pain point — and then no task picks it up. Millions-of-records ×
Cartesian scenarios makes cancellation/progress/resumability a first-run
requirement, not a polish item, and it shapes the runner's interface (job
handles, partial-result semantics: one failed combination fails the run or is
reported?). Add to task 09/10.

### 4d. Engine-version pinning [core]

Reproducibility spans code revision *and* engine version — today's revision
comparison already fails when historical code needs an older `decider`
interface. `experiment.yaml` should pin/record the engine version, and results
should state it. Cheap now, unfixable retroactively for old results.

## 5. Contract versioning itself [core]

Task 01 promises "a short, versioned contract" — but nothing anywhere
addresses **evolution** of the versioned artefacts: the trace schema (retained
per governance requirements — old traces must remain decodable),
`experiment.yaml`, result formats, check reports, MCP schemas, and the ID
format's escape hatch. Each format needs a version field from day one and a
stated policy (reader supports N-1? migrate-on-load?). This is the single most
classic "core fix later" trap in the whole plan, and it costs almost nothing
to pre-commit to.

## 6. UI architecture: the modal model is never stated [core]

This is my largest single ambiguity. The redesign's core promise is three
modes with "distinct entry points and result views", where users never infer
which mode they're in. But no task says **how the UI is physically organised**:

- One retained webview that morphs per mode (the state-machine confusion the
  redesign is trying to escape), or three distinct view types?
- What happens to the current single retained graph panel, and to the
  editor-group manipulation that keeps the graph visible during debugging?
- Mode transitions: experiment → debug is specified (explicit, one-way). What
  about flow → debug, debug → experiment? Can you "promote" a paused session
  into a scenario draft?
- Where do the Structure tree, the contextual inspector, and results views
  live relative to the graph?

Every task (05, 07, 10) assumes an answer. If the tasks are implemented
against different implicit answers, merging them *is* the major rewrite.
Half a page in the design direction fixes it.

## 7. MCP: topology is unspecified [refine, but before task 11 starts]

Decision 9/16 cover permissions well. Unanswered: does the FastMCP server live
in the extension host (per window), or a separate process per workspace? How
does a terminal-based agent connect (stdio? HTTP/SSE? port discovery on
localhost, and what stops a random local process from calling it)? How does
"highlight/reveal in VS Code" target the right window? Which capabilities work
**headless** (no VS Code) — decision 16 wants agents to complete workflows
end-to-end, but an editor-hosted server can't serve a notebook-only agent.
State the instance model and the headless/editor-bound split in task 01's
contract or early in 11.

## 8. Orphaned stories [process]

Two candidate stories have no owning task:

- **Story 4 (validate before running)** — data compatibility, parameter
  constraints, version compatibility, **expected run cost**. No task
  implements cost estimation or pre-flight validation. Either accept it
  (fold into 09/10) or explicitly defer.
- **Story 5 (shareable finding)** — a linkable bundle of flow context,
  evidence, config, revision. Nothing builds it, and it cuts across tracing,
  experiments, checks, and IDs. Accept (task 10 or 12) or defer — but say so.

(Stories 6 and 7 are covered by tasks 11 and 04/08 respectively.)

## 9. Smaller, worth fixing in passing

- **Graph library spike [refine]:** decision to "confirm the library's
  interaction contract first" is right; make it a concrete spike in task 05
  that renders the largest known real flow and drives every required gesture
  (viewport-preserving collapse, fit-to-selection, MCP-driven highlight,
  breakpoint decorations) before the panel rebuild. Also set scale budgets:
  max nodes rendered, layout time, webview memory — story 3 caps *record*
  rendering but nothing caps *flow* size.
- **Notebook adapter claim is never verified [core-ish]:** "notebooks and the
  extension are adapters over the same experiment interface" is a central
  architectural bet, but no task builds even a smoke-test notebook consumer.
  Add to task 10's done-when: an experiment run end-to-end from a plain
  Python (headless) caller. It also de-risks the MCP headless question.
- **Extension ↔ decider version skew [core-ish]:** once the bridge ships
  inside `decider` (task 03), the extension's compatibility with the
  installed engine version becomes a real axis (old `decider` + new extension
  = no bridge). Extend task 01's capability reporting to cover a bridge
  protocol version and a minimum-version check with a helpful error — the
  current failure mode (empty Structure tree) is exactly what background
  criticises.
- **Naming collision [refine]:** the repo already has `experimentation/`
  (engine spikes); project assets will be `experiments/`. Fine in user
  projects, confusing in this repo's docs. One disambiguating line, or pick a
  different asset-dir name now while nothing depends on it.
- **Engine targeting [refine]:** the notes never state whether the redesigned
  extension targets `decider` only, or must also drive `decider2` during any
  migration period. One sentence.
- **Multi-root workspaces [refine]:** per-folder interpreter (`decider.python`)
  and discovery scope in monorepos.
- **Multi-pipeline composition [core]:** everything assumes "a pipeline". If
  flows call flows (now or plausibly later), identity uniqueness and trace
  span boundaries must account for it. Even a "single-flow only for v1,
  IDs scoped per-flow" statement protects the seam.

## Recommended edits to the plan

1. **Task 01:** add record identity, run identity lifecycle, ID uniqueness
   scope + unresolved-reference semantics, and the version-evolution policy
   for every persisted/wire format.
2. **New early spike (or task 04 preamble):** compiled/nogil kernel trace
   capture + concurrent-execution ordering. Run in parallel with 01–03.
3. **Task 04:** state trace-vs-DAP division of labour; define ordering as
   per-record streams; add schema version field.
4. **Task 06:** replace "decide whether a hybrid mode is feasible" with the
   chosen semantics (full-frame execution + record-scoped observation, or
   dependency-closure execution — but written down).
5. **Task 07:** state the structured-override vs. arbitrary-console-edit
   boundary.
6. **Task 09:** add equality semantics, nondeterminism handling, engine
   version pin, input fingerprinting, cancellation/progress/partial results
   to the spike list and asset model.
7. **Task 10:** add a headless (non-VS Code) end-to-end experiment run to
   done-when; own or explicitly defer stories 4 and 5.
8. **Design direction (background/userstories):** half a page on the UI modal
   model — panels, mode transitions, fate of the retained panel.
9. **Task 01/11:** MCP instance/transport model and the headless split.
10. **Task 08:** add "missing durable IDs" to the default check suite;
    connects 02 and 08.
