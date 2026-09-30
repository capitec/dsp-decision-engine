# GPT-5.4 feedback

## Overall assessment after the answers

The answers materially improve the redesign. The earlier big ambiguities now
have good high-level decisions behind them: **`decider` owns the shared
contract**, **selection stays in adapters**, **`Value` is split into distinct
static/runtime/evidence concepts**, **reproducibility gets a real manifest**,
**finite long-running work shares a job model**, **MCP read access is divided
by data class rather than only by side effects**, and **v1 scale is explicitly
local rather than pretending to solve remote/distributed execution now**.

That means the redesign now looks much less likely to need a core rethink.
The main remaining risk is not the architecture direction; it is **drift
between the answered questions and the current task plan**. The next step is
to encode these decisions directly into the task notes so implementation does
not quietly re-open them.

## Decisions that now look strong

- **Contract ownership:** Python models in `decider` own the cross-language
  contract, with versioned JSON Schema and generated TypeScript types.
- **Adapter boundary:** core owns refs and context queries; VS Code/MCP own
  selection, focus, and highlighting.
- **Value model:** `ValueSlot`, `ObservedValue`, and `TraceEvidence` are
  distinct concepts.
- **Reproducibility:** persisted reproducible runs require clean committed
  source; dirty-tree sessions are allowed only as explicitly non-reproducible.
- **Revision semantics:** saved experiments keep both authored symbolic refs
  and resolved SHAs.
- **Execution lifecycle:** finite long-running work shares one job model, while
  a live debug session is not itself a job.
- **Scale scope:** v1 is local Polars execution bounded by local disk/RAM.
- **Experiment seam:** the portable core is declarative; Python tests/graphs
  are optional project hooks outside that portable core.
- **Trace ownership:** retained trace storage/query belongs to the
  client-selected post-record adapter, not the extension.
- **Finding sharing:** a portable finding descriptor now has a clear home.

## Remaining high-priority changes

| Priority | Area | Remaining issue | Why it still matters | Suggested update |
| --- | --- | --- | --- | --- |
| P0 | Task 01 scope | `01-architecture-contracts` still reads broader than the answered design. | The answers are good, but task 01 still risks becoming a shallow “god contract” if it owns selection state, final experiment asset layout, and too many mixed concerns. | Narrow task 01 to the stable shared core: Python-owned models, versioning/migration rules, generated schemas/types, refs, `ValueSlot`/`ObservedValue`/`TraceEvidence`, and error/capability reporting. Keep selection/focus out of core and let task 09 finalise the experiment asset shape. |
| P0 | Run manifests and fingerprints | There is still no explicit task for run manifests, source/data fingerprints, revision resolution, and stale-state rules. | Reproducibility, debugger reproduction, finding links, and “stale vs immutable evidence” all depend on this being designed once. | Add a new early task before 07/09/10 for run manifests, flow/source fingerprints, authored-vs-resolved revisions, environment capture, dataset fingerprints, and stale-artifact semantics. |
| P0 | Job model | The answers define a shared finite-job lifecycle, but the task graph does not yet. | Data loading, checks, revision comparison, experiment runs, and debugger launch/setup should not each invent their own progress/cancellation model. | Add a new early task before 06/09/10/11 for job IDs, progress, cancellation, logs, partial results, and terminal status. |
| P0 | Scope messaging | The notes still talk in places as if millions/billions of rows may be a first-pass target. | That now conflicts with the clearer decision that v1 is bounded by local Polars and local resources. | Update the notes so v1 explicitly promises local execution, aggregate/sampled results, and portable dataset references for later backends—without implying remote/distributed support now. |
| P1 | Finding descriptor | The answers place “share a finding” well, but the tasks do not yet encode it. | This is now a deliberate cross-cutting artifact, not an afterthought. | Update task 10 to create the portable finding descriptor and task 12 to cover rendering/deep-link sharing and graceful handling when optional evidence is unavailable. |
| P1 | Override points | The answers now define override points as explicit source-level declarations, but the tasks still under-specify that seam. | Override points affect experiment authoring, runtime validation, debugger reproduction, and the portable experiment definition. | Expand task 09/10 to cover syntax, validation, discovery assistance, and user-facing reporting for invalid or unavailable override points. |
| P1 | MCP capability model | Task 11 still sounds too much like “broad reads are broadly okay”. | The answers improved this: structural metadata, safe summaries, and raw record/trace payloads are distinct capability classes. | Update task 11 to make these capability classes explicit and to require plain unavailable/redacted states rather than implied access. |
| P1 | Trace sequencing | The conceptual trace model is better grounded now, but task 04 is still large and still blocks a lot. | This is now more of a delivery/sequencing risk than a design flaw. | Either split task 04 into “live session evidence” and “retained/exportable trace”, or at least make that phase boundary explicit so runtime UX is not forced to wait for the entire trace system. |

## What I would change in the task plan now

1. **Tighten task 01** so it describes the answered shared contract, not a
   larger adapter-facing umbrella.
2. **Add a run-manifest/fingerprint task** before debugging, experiments, and
   sharing flows are implemented.
3. **Add a job-lifecycle task** before data loading and experiment execution
   work.
4. **Let task 09 own the final experiment asset shape** even if task 01
   reserves the concept and compatibility expectations.
5. **Update task 11** from “broad read access” to explicit capability classes
   and redaction/unavailability behaviour.
6. **Update task 10 and task 12** to include the portable finding descriptor.
7. **Clarify v1 scale limits** everywhere the notes currently imply more than
   local-resource-bounded execution.

## Task-by-task edits I would make

- **Task 01 — architecture contracts**
  - Make Python models in `decider` the explicit source of truth.
  - State that JSON Schema, TypeScript types, and MCP schemas derive from the
    same models.
  - Add format-version and migration-on-read requirements for persisted and
    wire formats.
  - Remove adapter-owned `selection` from the core contract.
  - Split `Value` into `ValueSlot`, `ObservedValue`, and `TraceEvidence`.
  - Reserve experiment-asset conventions here, but do not freeze the final
    `experiment.yaml` shape in this task.

- **New task — job lifecycle**
  - Introduce a shared model for finite long-running work: job ID, status,
    progress, cancellation, logs, and partial results.
  - Cover data load, checks, revision comparison, experiment runs, and debug
    session launch/setup.
  - Explicitly exclude the live debug session itself from being modeled as a
    job.

- **New task — manifests, fingerprints, and staleness**
  - Define run manifests with authored and resolved revisions, environment
    capture, dataset fingerprint/schema, filtering/sampling, record selection,
    and declared overrides.
  - Define flow/source fingerprints and what becomes stale when source changes.
  - State that saved reproducible runs require clean committed source and are
    never silently retargeted.

- **Task 04 — decision tracing**
  - Keep the retained/exportable trace owned by the post-record adapter.
  - Require first-class “trace unavailable / redacted / not retained” states
    for VS Code and MCP consumers.
  - Consider phasing runtime live evidence ahead of the full retained/export
    story if delivery pressure appears.

- **Task 06 — data loading and execution scopes**
  - Add portable dataset references even though v1 remains local.
  - Make the local-resource bound explicit.
  - Encode record-identity rules: heuristic default, explicit/composite ID
    support, and rejection of duplicate/missing IDs when durable record refs
    are required.

- **Task 09 — experiment module**
  - Define the portable declarative core explicitly: flow/revision/input refs,
    filtering/sampling, scenarios, declared overrides, built-in summaries, and
    result-manifest metadata.
  - Treat Python tests and custom graphs as optional project hooks outside the
    portable contract.
  - Own the least-noisy override-point syntax and validation rules here.
  - Own the final `experiment.yaml` shape here.

- **Task 10 — experiments and VS Code workflows**
  - Add creation of the portable finding descriptor.
  - Make scenario-to-debug reproduction use the run manifest directly.
  - Ensure invalid, unavailable, or stale override points fail plainly.

- **Task 11 — FastMCP**
  - Replace “broad reads” with explicit capability classes for structural
    metadata, safe summaries, and raw record/trace payloads.
  - Require plain capability/error reporting when raw evidence is unavailable
    because of adapter policy or environment limits.

- **Task 12 — integration and documentation**
  - Cover schema/type generation workflow and migration testing.
  - Cover stale-state behaviour and manifest compatibility.
  - Cover evidence-unavailable behaviour for traces/findings.
  - Cover the local-scale promise so docs do not imply distributed execution.

## Remaining blindspots and ambiguities

These are smaller than before, but still worth tightening:

- **Dataset fingerprint cost and stability:** for large files, is the dataset
  identity a content hash, metadata plus schema, or something tiered? This
  affects reproducibility cost and cacheability.
- **Schema/type generation workflow:** where are generated TypeScript types and
  JSON Schemas committed, tested, and versioned? The design is good, but the
  maintenance workflow still needs to be explicit.
- **Optional evidence reopening:** if a finding or experiment manifest points
  at client-owned trace storage that is unavailable later, the UI and MCP need
  a first-class “evidence unavailable” state rather than a partial failure that
  looks like missing data.
- **Fingerprint granularity:** clarify whether a flow/source fingerprint covers
  only the pipeline file, the transitive source closure, the full repo state,
  or the full execution environment. Different use cases may need different
  guarantees.

## Bottom line

With the answered questions, the redesign is now in good shape at the
high-level architecture layer. I no longer think the main risk is that the
core module seams are wrong; the main risk is that the **tasks and contracts do
not yet fully reflect the decisions you have now made**. If you update the
task plan to encode those answers, this should be a solid foundation that can
be refined without major structural rework.
