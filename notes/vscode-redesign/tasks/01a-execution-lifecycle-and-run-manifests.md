# 01a — Define execution lifecycle and run manifests

**Depends on:** 01  
**Blocks:** 06, 07, 09, 10, 11, 12

## Outcome

Define one deep execution-lifecycle module for finite long-running work and one
versioned run-manifest format for reproducible results. This keeps job state,
progress, cancellation, provenance, partial results, and stale-source handling
out of VS Code, MCP, and individual experiment implementations.

## Work

- Define a finite job interface for data loading, check runs, revision
  comparison, experiment execution, and trace export/expansion: job ID,
  progress, cancellation, timeout, logs, terminal status, failure, and
  partial-result metadata.
- Keep an interactive debug `Session` distinct from a finite job; a debugger
  launch/setup may use a job, but a paused session is not one.
- Define a versioned immutable run manifest containing authored and resolved
  revisions, clean/dirty source state, engine/Python/dependency environment,
  input fingerprint/schema/row count, record/filter/sampling selection,
  declared overrides, and observed outputs/result references.
- Define source-staleness behaviour: source edits mark live graph context and
  breakpoints stale until re-resolved, while trace and manifest evidence remain
  bound to their original source fingerprint.
- Define format-version/migration policy for run manifests, experiment assets,
  trace events, check reports, and MCP schemas.
- Reuse existing exception types and capability reporting so adapters surface
  the same actionable errors.

## Important decisions

- Saved reproducible runs require a clean committed revision; interactive
  dirty-tree runs are explicitly session-only and non-reproducible.
- Symbolic revisions are retained as authored intent and resolved to immutable
  SHAs for every actual run.
- `decider` writes small manifest metadata, not client-owned bulk result or
  trace storage.
- Task 09 owns the experiment-specific manifest schema layered on this
  lifecycle contract; this task owns shared manifest identity, immutability,
  versioning, and job semantics.

## Done when

- A finite operation can report and cancel consistently from Python, VS Code,
  and MCP.
- A result can be reopened or shared from its manifest without relying on an
  ephemeral session ID.
- Format evolution and stale-source behaviour are documented and tested.
