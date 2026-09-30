# GPT-5.4 questions

These are the questions most likely to force later core changes if they stay unresolved. You do **not** need perfect answers now, but I would want clear answers before freezing interfaces or task boundaries.

## P0 — answer before freezing core contracts

1. **What is the source of truth for the shared cross-language contract?**
   - Python models in `decider`?
   - JSON Schema generated from those models?
   - Generated TypeScript types?
   - A separate protocol package?
   - Why this matters: `decider`, VS Code, notebooks, and FastMCP need one contract owner and one versioning story.
>>> Python models in `decider` own the contract. They produce versioned JSON
Schemas and generated TypeScript types for VS Code; MCP schemas derive from the
same models. Do not create a separate protocol package unless a non-Python
consumer proves generation insufficient. Every persisted or wire format carries
a format version and supports explicit migration on read. <<<

2. **Should `selection` be a core concept at all?**
   - Or should core expose refs and query operations, while VS Code/MCP own “what is currently selected/focused/highlighted”?
   - My leaning: keep selection in the adapters.
>>> Agreed. Core owns stable references and context queries; VS Code and MCP
compose current selection, focus, and highlighting from those references. <<<

3. **What does `Value` mean in the shared model?**
   - A static flow-level slot/field?
   - A runtime observed payload?
   - A trace evidence item?
   - A column/field as seen by a record?
   - If this stays overloaded, the inspector, trace model, and MCP tools will probably diverge later.
>>> Split it. A `ValueSlot` is a static flow field/column identity; an
`ObservedValue` is a runtime value at a slot for a record/frame/run; a
`TraceEvidence` item may reference either but is not itself a value. <<<

4. **What is the canonical run manifest for reproducibility?**
   - Exact commit SHA or symbolic ref plus resolved SHA?
   - Dirty working tree allowed or forbidden?
   - `decider` version, Python version, dependency lock state?
   - Dataset identity, schema snapshot, selected rows/filter/sampling rules?
   - Override declarations and actual override values?
>>> A persisted run manifest records both authored and resolved revisions,
engine/Python/dependency environment, input fingerprint/schema/row count,
filters/sampling, record identity selection, declared overrides and values.
Saved reproducible runs require a clean committed source revision; interactive
dirty-tree runs are allowed only as explicitly non-reproducible session runs. <<<

5. **How should symbolic revisions like `HEAD^` behave in saved experiments?**
   - Stored as authored?
   - Resolved to immutable SHAs on save/run?
   - Both?
   - This matters because “rerun later” and “review what was actually compared” are different needs.
>>> Store both: preserve the authored symbolic ref for intent and resolve it to
an immutable SHA for each run. Rerun uses the saved SHA by default; an explicit
refresh re-resolves the symbolic ref and creates a new manifest. <<<

6. **Do all long-running actions share one execution job model?**
   - Data load
   - Experiment run
   - Check suite
   - Revision comparison
   - Trace export / trace expansion
   - Debugger launch/setup
   - If the answer is yes, define job IDs, progress, cancellation, partial results, and logs once.
>>> Finite long-running operations share a core job model with IDs, progress,
cancellation, logs, terminal status, and partial-result metadata. A live debug
session is not itself a job, although launching/setup may be represented by
one. This is a new early design task before tasks 06 and 09 freeze. <<<

7. **If very large data matters, is local Polars the implementation or just one adapter?**
   - If billions of rows are a real target, do you need an `ExecutionBackend` seam now?
   - If not, should the notes explicitly say the first release is bounded by local disk/RAM?
>>> v1 is local Polars execution and is bounded by local disk/RAM. Define
portable dataset references so a later backend can be added, but do not add an
`ExecutionBackend` seam or remote execution to this pass. <<<

8. **What is the smallest portable experiment definition that must work everywhere?**
   - Which fields are purely declarative and must run the same in VS Code and notebooks?
   - Which things are optional Python extensions (`tests/`, `graphs/`) and therefore non-portable?
   - Where is the line between the stable deep module and project-specific custom code?
>>> The portable declarative core is flow/revision/input references, filtering
and sampling, declared scenarios/overrides, built-in summaries, and result
manifest metadata. Python tests and custom graphs are optional project hooks
outside that portable interface. <<<

## P1 — answer before heavy implementation

9. **What is the canonical record identity when the input data is imperfect?**
   - Duplicate IDs?
   - Missing IDs?
   - Composite keys?
   - Stable row hash fallback?
   - Order-dependent fallback?
   - Record drill-down, debugger reproduction, and aggregate-to-record navigation all depend on this.
>>> Keep heuristic selection as a default, allow explicit ID or composite-key
declaration, and reject duplicate/missing identities where persistent
record-level references are required. Session-only row ordinals are display
helpers, never reproducible identities. <<<

10. **How are declared override points authored and validated?**
    - Source annotations?
    - Flow metadata?
    - Explicit exported objects?
    - Automatic discovery plus allowlist?
    - This is a core experiment seam, not just UI detail.
>>> Override points are explicit source-level declarations attached to supported
step outputs. Task 09 must spike the least noisy syntax and validation rules;
automatic discovery may suggest candidates but cannot make an override valid. <<<

11. **Who owns trace storage/query for anything beyond the current process?**
    - The extension?
    - The experiment result bundle?
    - The client-selected post-record adapter?
    - An external sink only?
    - Capture-only is not enough if VS Code, notebooks, or MCP need to reopen evidence later.
>>> The client-selected post-record adapter owns retained trace storage/query.
The extension holds only session-memory evidence by default; experiment result
manifests may link to a client-owned trace location but do not require one. <<<

12. **What can MCP read automatically when data may be sensitive?**
    - Structural metadata only?
    - Safe summaries?
    - Raw record payloads?
    - Raw trace events?
    - Should workspace trust or client policy disable some categories entirely?
>>> Broad reads are the default goal, but structural metadata, safe summaries,
and raw record/trace payloads are separately controllable workspace
capabilities. The trace adapter/client environment determines whether raw
evidence is available. <<<

13. **What becomes stale when source changes mid-session?**
    - If a user edits Python while paused, or changes Git revision while an experiment result is open, which artifacts become invalid?
    - Flow descriptions?
    - Breakpoints?
    - Trace links?
    - Selected nodes?
    - Saved debugger reproductions?
>>> Every session/result binds to a flow/source fingerprint. Source edits mark
live graph context and unresolved breakpoints stale until re-resolved; traces
and manifests remain immutable evidence of the original fingerprint. Saved
reproductions require a committed revision and are never silently retargeted. <<<

14. **Where does “share a finding” land architecturally?**
    - A deep link into VS Code?
    - A result manifest entry?
    - A portable artifact bundle?
    - A lightweight report format over experiments/checks/traces?
    - This story is easy to defer, but hard to bolt on later without another cross-cutting result schema.
>>> Implement a portable finding descriptor: manifest reference, selected
flow/step/value/record ref, evidence location, and optional human summary.
Task 10 owns creation and task 12 owns rendered/deep-link sharing; it does not
require remote storage. <<<

## My strongest current leanings

- **Core owns refs and evidence; adapters own selection and focus.**
- **Reproducibility needs a run manifest, not just a saved YAML definition.**
- **Experiments need a small declarative core plus optional Python hooks around it.**
- **If “billions of rows” is real, backend choice must be a seam now.**
- **MCP permissions should distinguish safe summaries from raw data, not just reads from writes.**
