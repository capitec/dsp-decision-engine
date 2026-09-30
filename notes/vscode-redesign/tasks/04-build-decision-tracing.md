# 04 — Build structured decision tracing

**Depends on:** 01, 02  
**Blocks:** 07, 10, 11, 12

**Experiment gate:** `experimentation/01-trace-capture-findings.md` validates
the in-kernel primitive. Its follow-up must report before trace schema,
delivery, concurrency, or batch-performance guarantees freeze.

## Outcome

Provide a compact, structured trace of how a decision was made that execution
can capture efficiently and clients can export, transform, persist, or render.

## Work

- Apply the validated finding: fixed-width numeric capture in kernels, joined
  to names/metadata in the driver; conservation before post-record adaptation;
  and live-first delivery compatible with retained/exportable tracing.
- Complete `experimentation/01-trace-capture-follow-up.md` before freezing
  record identity mapping, concurrent ordering, event payloads, decode/delivery
  semantics, or batch tracing policy.
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
- Make batch trace capture opt-in and bounded. Define size limits, sampling or
  aggregation behaviour, and reported loss/degradation rather than applying
  the single-record overhead result to high-volume runs.
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
