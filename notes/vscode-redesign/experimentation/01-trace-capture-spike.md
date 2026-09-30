# Experiment 01 — Compiled decision-trace capture

**Run during:** tasks 01–03, before task 04 freezes any trace schema  
**Feeds:** tasks 04, 07, 09, 10, 11

## Question

Can supported compiled/nogil execution modes record compact, ordered decision
evidence without Python allocation/callbacks in the kernel, while preserving
the engine's execution semantics and acceptable overhead?

## Method

- Exercise representative record, frame, branch, loop, table/tree, and
  override paths in interpreted and compiled modes.
- Compare a no-op path with fixed-size or append-only capture buffers and
  post-hoc decoding.
- Measure throughput, memory, event size, ordering, and adapter backpressure.
- Verify the proposed envelope is encodable in the compiled path: schema
  version, parent/child flow context, durable/derived step references,
  capture-time descriptive metadata, and either a record key or explicit frame
  scope.
- Verify per-record ordering and trace-point conservation before a post-record
  adapter transforms events.

## Decision outputs

- The minimal event envelope and what can be captured in-kernel.
- The capture-to-conservation-to-adapter boundary.
- The supported ordering guarantee and no-op overhead budget.
- Whether live runtime evidence can be delivered before retained/exportable
  tracing, without making the interfaces incompatible.
