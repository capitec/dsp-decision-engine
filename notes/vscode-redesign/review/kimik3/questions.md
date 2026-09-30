# Kimi K3 questions — notes/vscode-redesign

Questions I'd want answered (or explicitly deferred) before the relevant task
freezes its interface. Grouped by theme; each names the task/note it affects.
Numbered for easy reference in replies — partial answers are fine.

## A. Identity and provenance

**A1. Record identity rules** (task 01, 06) — When loaded data has no
`id`-like field, or has duplicates, what happens? My suggestion: keep the
current heuristic as a default guess, let the data-load surface override the
ID column explicitly, and error (not guess) on duplicates when a run/experiment
needs record references. Acceptable?
>>> Yes. Keep the heuristic for an initial suggestion, allow an explicit ID
column or composite-key selection, and reject duplicate/missing identities when
a saved experiment, debugger reproduction, or drill-down needs one. A
session-only row ordinal may be shown for inspection but is never persisted as
a record reference. <<<

**A2. Input provenance = content hash?** (task 09) — Should `experiment.yaml`
store a content hash (plus schema/row count) of its input data, and should
re-run warn or hard-error on mismatch? Without this, "reproducible" is
aspirational. Is warning-on-mismatch enough, or should it be configurable
(warn vs error)?
>>> Store the input locator in authored YAML and record a content hash, schema,
and row count in the immutable run manifest. A reproducible rerun errors on
mismatch by default; an explicit input-drift option may run with a prominent
warning and writes the new observed fingerprint to a new manifest. <<<

**A3. Run identity lifecycle** (task 01) — Are run IDs ephemeral (die with the
session) or persistent (stored in experiment results / traces)? If both kinds
exist, what distinguishes them in the shared model?
>>> Both exist. A session run handle is ephemeral and supports live debugging;
a persisted run manifest is immutable provenance for an experiment result or
exported trace. The shared model names them separately and never lets a
session-only ID masquerade as reproducible evidence. <<<

**A4. Step-ID syntax and scope** (task 02) — What do generated IDs look like:
human-readable slugs derived from step names (rename drift risk), short opaque
IDs (merge-friendly, less readable), or sequential (conflict-prone)? And is
uniqueness scoped per flow, per workspace, or per repo?
>>> Generate short opaque, reviewable IDs rather than name-derived slugs or
sequences. IDs are unique within a durable flow; flow ID plus step ID is the
global reference. This prevents rename drift and makes independent branch
generation effectively collision-free without cluttering source. <<<

**A5. Unresolvable ID references** (tasks 02, 04, 09, 10) — A saved experiment
or trace references a step ID that no longer exists after a refactor. Error,
warning with partial rendering, or silent skip? My leaning: warn and render
what resolves — experiments shouldn't become unreadable because one step moved.
>>> Warn and render every resolvable reference. Preserve the unresolved ID and
its original descriptive metadata in the result so users can repair the asset;
never silently skip it. <<<

**A6. Missing-ID enforcement** (tasks 02, 08) — Should "pipeline steps lack
committed durable IDs" be a diagnostic in the task 08 default check suite, so
CI can require IDs on production flows without a separate mechanism?
>>> Yes. The default suite reports absent durable IDs as a configurable
diagnostic, allowing clients to promote it to a CI failure for production
flows without making development flows unusable. <<<

**A7. Flow composition** (task 01) — Is "one pipeline per flow" a v1 guarantee,
or can flows call/embed other flows? If composition is plausible later, ID
uniqueness scope and trace span boundaries should assume it now. If it's out
of scope, say so explicitly.
>>> Make composition representable now: each flow has its own ID and a call to
a sub-flow forms an explicit parent/child boundary in graph and trace context.
No new composition runtime feature is required by this redesign, but the
identity/trace contract must not assume a permanently flat flow. <<<

## B. Tracing and debugging

**B1. Trace ordering under parallel execution** (task 04) — Can the trace
contract be **per-record event streams** (each event carries a record key;
order guaranteed within a record, not across records)? That survives parallel
kernels. Is any planned consumer relying on a single global order?
>>> Yes. Order is guaranteed within a record execution stream. Frame-level
events use an explicit frame scope rather than pretending to belong to one
record. Cross-record ordering is deliberately unspecified; aggregate consumers
must not depend on it. <<<

**B2. Spike timing** (task 04) — Agree to pull the compiled/nogil-kernel trace
capture spike out of task 04 and run it alongside tasks 01–03? It blocks 07,
10, 11, 12, and its outcome shapes the event schema.
>>> Agreed. Run experimentation/01-trace-capture-spike.md while tasks 01–03
establish the shared concepts, and do not freeze task 04's event schema until
the spike reports. <<<

**B3. Trace vs. debugger state** (tasks 04, 07) — Confirm the division:
trace = decision evidence (path, matched rules, reasons, declared values);
DAP/Variables = arbitrary live state. The inspector merges both, and with
tracing off the inspector degrades to DAP-only. Is that the intent?
>>> Confirmed. Trace is declared decision evidence; DAP is arbitrary live
debug state. The inspector labels both sources clearly and remains useful with
trace capture disabled. <<<

**B4. Execution mode behind the debugger** (tasks 06, 07) — Which execution
mode backs debugging — always interpreted/stepped? And do we promise *value
parity* between the debugged run and the compiled experiment run (the kernels
already aim for CPython-equal rounding — is that guarantee extended to all
debug-relevant numerics, or documented as best-effort)?
>>> Debugging uses the existing stepped/session execution model. Supported
execution modes must meet the engine's documented semantic parity guarantees;
we will not promise bit-for-bit float equality beyond those guarantees. Task 03
must identify and test the current parity contract rather than call it
best-effort. <<<

**B5. Console edits** (task 07) — Confirm: only structured edits (params/state
via UI or session API) become tracked, scoped overrides; arbitrary
debug-console mutations are explicitly untracked and excluded from
reproducibility?
>>> Confirmed. Structured parameter/state edits become scoped overrides.
Arbitrary console mutation remains available for Python debugging but is
visibly untracked and cannot be saved or replayed as an experiment. <<<

**B6. Extension as trace client** (task 04, 11) — What's the extension's own
data policy for trace values: session memory only, nothing to disk, and the
same rule for data served over MCP? Should there be a workspace setting to
disable data-returning MCP tools entirely?
>>> The extension keeps trace/input values in session memory by default and
does not write them to disk on its own. MCP read access is broad by default,
but a workspace/client setting must be able to disable raw data-returning
tools while preserving structural summaries. <<<

## C. Execution scope

**C1. Hybrid mode semantics** (task 06) — Is the hybrid mode simply "always
run the full frame; record focus only scopes tracing/display/breakpoints"?
That's semantically clean but always pays full-frame cost. Or do you want the
cheaper "execute the upstream dependency closure at frame granularity" (record
steps upstream of a needed frame step still run on all rows)? The first is a
better default mental model; the second is an optimisation that can come later
*without* changing semantics — agree?
>>> Agreed. The initial hybrid semantics are full-frame execution with record
focus limiting display, tracing, and breakpoints. Dependency-closure execution
is a later optimisation only if it proves semantically identical. <<<

## D. Experiments

**D1. Numeric equality** (task 09) — What does "changed value" mean per type?
Exact for int64 cents is obvious; for float64 running totals: exact, absolute/
relative tolerance, or per-experiment config? My leaning: exact by default,
optional tolerance declared in `experiment.yaml`.
>>> Agreed: exact equality by default; optional type-appropriate absolute and
relative tolerances are declared by the experiment and recorded in its run
manifest. <<<

**D2. Nondeterminism** (tasks 08, 09, 10) — Is runtime *warning* about
non-pure steps (wall-clock, randomness) in experiment results sufficient for
v1, with controlled clock/seed explicitly deferred? Or is determinism control
a launch requirement for credible comparisons?
>>> v1 reports detected nondeterminism and marks affected comparisons as
non-reproducible; controlled clocks and seeds are deferred. A comparison may
still run, but its result must not appear equivalent to a deterministic
reproduction. <<<

**D3. Cancellation and partial results** (tasks 09, 10) — Required for the
first experiment release? And when one scenario in a sweep fails: fail the
whole run, or record that scenario as failed and keep the rest? (My strong
leaning: per-scenario failure isolation + resumable runs.)
>>> Required for the first experiment release. The runner has cancellable
finite-job semantics, records partial results in a result manifest, isolates
scenario failures, and can resume unfinished scenarios when their immutable
inputs/manifests still match. The job model is designed in task 09. <<<

**D4. Engine version pin** (task 09) — Should `experiment.yaml` record the
`decider` version, and results record the version they ran with? And what's
the supported-revision window for `HEAD^`-style comparisons — "revisions
compatible with the installed engine" with a clear error otherwise?
>>> Yes. Results record the resolved Git revision, installed `decider` version,
Python version, and relevant environment metadata. Historical revisions are
supported only when compatible with the installed engine; otherwise the run
fails with that explicit explanation. <<<

**D5. Forked-session scenarios** (background, task 07) — Confirm the deliberate
cut: today's "fork a scenario from a paused session including arbitrary session
state" goes away; scenarios can only override declared override points/step
outputs, reproducibly, from a defined baseline. Any workflows you know of that
depend on the old power?
>>> Confirmed as a deliberate cut for scenarios. Ad-hoc paused-session forks
may remain a debugging convenience, but are not experiment assets and cannot
claim reproducibility. <<<

## E. UI architecture

**E1. The modal model** (design direction; tasks 05, 07, 10) — One sentence
each: how many panel/view types exist; does the graph panel morph per mode or
is it shared read-only structure with mode-specific side views; what happens
to the retained-panel behaviour and the editor-group juggling during debug;
which mode transitions are allowed (experiment→debug is one-way — is
flow→debug just "run", and can a paused session be saved as a scenario
draft)?
>>> There is one shared Flow canvas plus mode-specific side views: Explore,
Debug, and Experiments. The canvas retains static structure; its inspector
changes context rather than the entire panel morphing. Debugging no longer
juggles editor groups. Flow to Debug is an explicit run action; a paused
session can create only a draft that is saved after its changes are converted
to declared overrides and a run manifest. Experiment to Debug is an explicit
reproduction action. <<<

**E2. Graph scale budgets** (task 05) — Any hard numbers from the largest real
flows (nodes, nesting depth)? These should become the spike's test fixture and
the layout/interaction budget.
>>> This requires a real-flow fixture inventory, so
experimentation/02-flow-scale-and-gesture-spike.md gathers representative
node/depth data and sets budgets before task 05's implementation freezes. <<<

## F. MCP

**F1. Instance and transport model** (tasks 01, 11) — Does the FastMCP server
run inside the extension host per window, or as a separate process per
workspace? stdio or HTTP on localhost, and how is the port/socket discovered
and protected from arbitrary local processes?
>>> This needs a topology spike rather than a guessed interface:
experimentation/04-fastmcp-topology-spike.md. The current design direction is
a `decider`-hosted Python FastMCP process using stdio for agent transport, with
an authenticated local extension bridge only for editor-bound actions; no
unauthenticated localhost HTTP listener is assumed. <<<

**F2. Headless scope** (task 11) — Which MCP capabilities must work with no
VS Code running (e.g., an agent driving a notebook/CLI workflow): flow
discovery and description presumably yes; highlight/reveal obviously not;
experiment *runs*? Decide the headless/editor-bound split now since it shapes
where each tool is implemented.
>>> Headless tools cover core discovery, description, checks, experiment
definition/result access, and experiment runs. Highlight/reveal and current
editor selection are extension-bound. A debugger may run headlessly through
the bridge, but UI navigation is not implied. <<<

## G. Scope and process

**G1. Orphaned stories** — Story 4 (pre-flight validation and cost estimation)
and story 5 (shareable/linkable finding): accept into the plan (they'd land in
tasks 09/10/12) or explicitly defer? Either is fine; silence isn't.
>>> Accept both. Pre-flight validation/cost estimation belongs to tasks 06 and
09; a shareable finding is a portable result-manifest entry that tasks 10 and
12 implement. <<<

**G2. Notebook adapter verification** (task 10) — Agree that "an experiment
runs end-to-end from a plain Python caller" belongs in task 10's done-when, as
the standing proof that VS Code is really an adapter?
>>> Agreed. Task 10 must prove an experiment runs end to end from a plain
Python caller; notebook UI work is not required for this pass. <<<

**G3. Naming** — Keep `experiments/` for project assets despite this repo's
existing `experimentation/` (engine spikes)? And one line stating the
redesigned extension targets `decider` only (no `decider2` support window)?
>>> Yes. Use `experiments/` for versioned project assets and
`experimentation/` for repository design spikes. This redesign targets
`decider` only; `decider2` compatibility and `decider-ui` migration are out of
scope. <<<
