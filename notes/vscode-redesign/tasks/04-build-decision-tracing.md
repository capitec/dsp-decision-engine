# 04 — Build structured decision tracing

**Depends on:** 01, 02  
**Blocks:** 07, 10, 11, 12

**Experiment gate:** `experimentation/01-trace-capture-spike.md` must report
before the trace schema freezes.

## Outcome

Provide a compact, structured trace of how a decision was made that execution
can capture efficiently and clients can export, transform, persist, or render.

## Work

- Run `experimentation/01-trace-capture-spike.md` during tasks 01–03 and use
  its results before freezing the event schema.
- Reconcile existing `Session.events` and tree/table `trace_output`/
  `path_output` with the new decision-evidence trace. Preserve cheap path
  columns for population aggregation and use event evidence for explanation.
- Deliver minimal live decision evidence first for debugger inspection, then
  retained/exportable tracing through the compatible full contract.
- Define trace events for step execution, path/branch outcomes, selected rules
  or table rows, scorecard bands, rounding, reasons, and relevant values.
- Freeze an envelope containing schema version, parent/child flow context,
  durable/derived step references with capture-time descriptive metadata, and
  either a record key or explicit frame scope.
- Guarantee event order within one record stream; leave cross-record order
  unspecified.
- Define trace identity, ordering, source/step references, and compact encoding.
- Instrument every supported execution mode, including compiled kernels.
- Add trace-point-conservation verification so optimisation cannot remove
  declared evidence without detection.
- Provide a no-op capture path with negligible overhead when tracing is off.
- Define the post-record adapter seam. A default pass-through adapter must let
  a client transform, redact, enrich, enqueue, or export events outside JIT
  execution, potentially on another thread or process.
- Provide an OpenTelemetry adapter or mapping where it is useful, without
  making OpenTelemetry the required retention or policy layer.

## Important decisions

- `decider` emits evidence; it does not retain traces or impose client privacy,
  deletion, or retention policy.
- The trace contract must be useful for debugging and explanations, not only
  performance observability.
- Event payload size, sensitive values, buffering, backpressure, and adapter
  failure behaviour require measured spikes before the public interface freezes.

## Done when

- A caller can enable/disable tracing and consume a correctly ordered trace.
- Every trace event resolves to durable flow/step identity and source context.
- The runtime inspector can combine declared trace evidence with arbitrary DAP
  state and degrades explicitly to DAP-only when tracing is disabled.
- Tests demonstrate equivalent evidence across supported execution modes and
  detect trace-point loss.
