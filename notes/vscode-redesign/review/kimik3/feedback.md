# Kimi K3 feedback — notes/vscode-redesign (v2, post-answers)

v1 findings are preserved in `feedback-v1-original.md`; the questions and
answers are in `questions.md`. This version reviews the answers and identifies
what they newly imply for the task plan.

## Verdict

The answers are decisive, mutually consistent, and resolve every concern I
raised without weakening the architecture. In particular: the run-manifest
split (ephemeral session handle vs immutable persisted manifest), per-record
trace ordering with explicit frame scope, full-frame execution with
record-scoped observation, and the one-canvas-plus-mode-views UI model are all
the choices I would have argued for.

The answers collectively do one new thing nobody named: **they invent a
run-manifest-centric provenance model** (A2, A3, D1, D3, D4, G1 all reference
an "immutable run manifest"), yet no task owns it. That is now the main gap —
see R1. Everything else below is consequence-tracking, not disagreement.

## What the answers settle

| Q | Decision |
|---|---|
| A1 | Record ID: heuristic default + explicit/composite override; duplicates/missing rejected when references are needed; session row ordinal never persisted |
| A2 | Input locator in authored YAML; content hash/schema/row count in immutable run manifest; mismatch errors by default; opt-in drift mode writes a new manifest |
| A3 | Ephemeral session run handle vs immutable persisted run manifest, named separately in the shared model |
| A4 | Short opaque generated IDs, unique per flow; flow ID + step ID is the global reference |
| A5 | Unresolvable references: warn, render what resolves, preserve the ID with capture-time metadata for repair |
| A6 | Missing-ID diagnostic in the default check suite, client-promotable to CI failure |
| A7 | Composition representable now: per-flow IDs, explicit parent/child boundary in graph and trace; no new composition runtime |
| B1 | Trace order guaranteed per record stream; frame-level events carry an explicit frame scope; cross-record order unspecified |
| B2 | Nogil/compiled capture spike runs parallel to tasks 01–03; task 04's schema freezes only after it reports |
| B3 | Trace = declared decision evidence; DAP = arbitrary live state; inspector merges and degrades to DAP-only |
| B4 | Debugging uses the stepped/session model; parity per the engine's documented guarantees; task 03 identifies and tests that contract |
| B5 | Structured edits become scoped overrides; console mutations are visibly untracked and never replayable |
| B6 | Extension keeps trace values in session memory; a workspace setting can disable raw-data MCP tools while keeping structural summaries |
| C1 | Hybrid scope = full-frame execution, record-scoped observation; dependency-closure execution only later and only if semantically identical |
| D1 | Exact equality by default; per-experiment tolerances declared in YAML and recorded in the manifest |
| D2 | v1 flags nondeterminism-affected comparisons as non-reproducible; controlled clocks/seeds deferred |
| D3 | Cancellation, partial results, scenario-failure isolation, and resume-on-matching-manifest required for v1; job model designed in task 09 |
| D4 | Results record resolved Git SHA, `decider`/Python versions, environment; historical revisions only where engine-compatible, with explicit errors |
| D5 | Forked-session scenarios are a deliberate cut; ad-hoc forks remain a debug-only convenience, never experiment assets |
| E1 | One shared flow canvas + Explore/Debug/Experiments side views; no editor-group juggling; explicit mode transitions; paused session → draft only via override conversion |
| E2 | Real-flow scale fixture + gesture spike (`experimentation/02`) before task 05's implementation freezes |
| F1 | MCP topology spike (`experimentation/04`); direction: `decider`-hosted FastMCP over stdio, authenticated bridge for editor-bound actions, no unauthenticated localhost HTTP |
| F2 | Headless: discovery, description, checks, experiment definitions/results/runs; editor-bound: highlight/reveal/selection; headless debugging via the bridge |
| G1 | Stories 4 and 5 accepted: pre-flight validation/cost into tasks 06/09; shareable finding = portable manifest entry in tasks 10/12 |
| G2 | Task 10 must prove an experiment runs end-to-end from a plain Python caller |
| G3 | `experiments/` = project assets, `experimentation/` = repo spikes; redesign targets `decider` only |

## What the answers newly imply (the residuals)

### R1. The run manifest is now the backbone and no task owns it [core]

Six answers lean on an "immutable run manifest" (provenance fingerprint,
tolerances, partial results, version pins, shareable findings). It is the
single most-referenced artefact in the answers and appears in no task's work
list. If it is left implicit, tasks 04, 09, 10, and 12 will each design their
slice and the slices will not compose — the exact core-fix-later failure mode.
**Owner needed:** task 01 names it a contract entity (identity, immutability,
version field); task 09 designs the schema. G1's shareable finding and D3's
resume both become manifest operations, which simplifies those stories.

### R2. Capture-time denormalized metadata [core]

A5's "preserve the unresolved ID and its original descriptive metadata" means
every persisted reference (trace event, experiment result, check report)
stores the step/flow names and source locations *as of capture time*. This is
a schema requirement on tasks 04, 08, and 09 — cheap now, and the kind of
thing that is nearly impossible to add to already-persisted artefacts later.

### R3. Flow IDs need the same treatment as step IDs [core]

A4 makes flow ID + step ID the global reference, so flow IDs are durable,
committed, generated identities too. Task 02 currently only discusses step
IDs: where does the flow ID live in source, does the same generator command
emit it, and what are its uniqueness/collision rules? Extend task 02 or
explicitly split flow-ID identity into task 01.

### R4. The trace event envelope is now constrained — the spike must validate it [core]

B1 + A7 fix the envelope: every event carries a record key **or** an explicit
frame-scope marker, plus flow context (parent/child), plus a schema version.
The `experimentation/01` spike must prove this exact envelope is encodable in
the nogil/compiled path (presumably via interned IDs), not just demonstrate
capture in general — a smaller envelope proven there and enlarged later would
be a breaking schema change.

### R5. "Non-reproducible" is a first-class result state [core]

D2 requires that a nondeterminism-affected comparison "must not appear
equivalent to a deterministic reproduction." That is both a result-schema flag
(task 09) and a rendering rule (task 10): the UI needs a distinct visual state,
not a footnote. Add to both tasks' done-when.

### R6. Draft conversion must enumerate what doesn't convert [refine]

E1 + D5 + B5 together imply: saving a paused session as a scenario draft
requires converting session changes into declared overrides, and anything that
can't convert (console mutations, unconvertible state) is dropped. The
conversion UI must list what converted and what was dropped — silently
dropping session state would recreate the exact confusion the redesign
exists to remove. One line in task 07 or 10.

### R7. Spike gates need wiring into task dependencies [process]

`experimentation/01` gates task 04's schema freeze; `experimentation/02` gates
task 05's implementation freeze; `experimentation/04` gates task 11. The task
headers should say so explicitly. Also: the spike numbering (01, 02, 04) skips
03 — align numbering when the files are created, or note what 03 is reserved
for (the task 09 experiment-interface spike is the natural candidate).

### R8. Job-model placement has a dependency tension [core-ish]

D3 puts the cancellable/partial/resumable job model in task 09, but task 06's
data loading is also long-running work, and 09 *depends on* 06. Resolve
explicitly: either v1 data loading is synchronous-with-progress and adopts the
job model when it lands, or the job model is extracted early enough for 06 to
use. What must not happen is two job models.

### R9. Data-policy items need homes [refine]

B6's workspace setting (disable raw-data MCP tools, keep structural summaries)
belongs in task 11's work list; the extension's session-memory-only trace
policy belongs in task 12's documented client responsibilities.

### R10. Fork behaviour-change note [refine]

D5 keeps ad-hoc forks as a debug convenience but removes them as experiment
assets. Annotate background's What-If section and task 07 so current users
read the change as a decision, not a regression.

## Still open (minor; not covered by the answers)

- **O1. Extension ↔ `decider` version policy.** Once the bridge ships inside
  `decider` (task 03), the extension needs a minimum-version check and bridge
  protocol negotiation via task 01's capability reporting — with a guided
  error, not today's empty Structure tree.
- **O2. Multi-root workspaces.** Per-folder interpreter (`decider.python`)
  and discovery scope in monorepos. Safe to refine later; worth one stated
  assumption in task 01 or 06.

## Updated recommended edits to the plan

1. **Task 01:** add the run manifest as a contract entity (identity,
   immutability, version field); record-identity rules (A1); flow+step ID
   model including unresolved-reference semantics (A4/A5); version-evolution
   policy for all persisted/wire formats; capability reporting extended to
   bridge protocol version (O1).
2. **Task 02:** cover flow IDs alongside step IDs (R3); require capture-time
   descriptive metadata wherever IDs are persisted (R2).
3. **Task 03:** add "identify and test the current cross-mode semantic parity
   contract" (B4).
4. **Task 04:** fix the event envelope (record-key/frame-scope + flow context
   + schema version, R4); per-record ordering guarantee (B1); trace-vs-DAP
   division (B3); freeze schema only after `experimentation/01` reports (R7).
5. **Task 05:** gate implementation on `experimentation/02` budgets (E2/R7).
6. **Task 06:** record-ID selection UX (A1); write the hybrid scope as
   full-frame execution + record-scoped observation (C1); absorb story 4's
   data-compatibility validation share (G1).
7. **Task 07:** structured-override vs console-edit boundary (B5); draft
   conversion enumerates unconvertible changes (R6); fork behaviour-change
   note (R10).
8. **Task 08:** missing-durable-ID diagnostic in the default suite (A6).
9. **Task 09:** owns run-manifest schema (R1), the job model —
   cancellation/partial/resume (D3/R8), equality/tolerance model (D1),
   nondeterminism flag (D2/R5), version pins (D4), pre-flight cost estimation
   (G1).
10. **Task 10:** headless end-to-end experiment run in done-when (G2);
    shareable finding as manifest entry (G1); distinct rendering for
    non-reproducible results (R5).
11. **Task 11:** gate on `experimentation/04` (R7); encode the
    headless/editor-bound tool split (F2); raw-data-disable setting (B6/R9).
12. **Task 12:** document the extension's session-memory data policy (B6/R9)
    and the What-If fork migration note (R10).

With those edits made, I have no remaining architectural objections — the
plan's load-bearing seams (identity, trace envelope, run manifest, execution
scope, mode separation, MCP topology) would all be pinned at the right level
of detail for refinement without core rework.
