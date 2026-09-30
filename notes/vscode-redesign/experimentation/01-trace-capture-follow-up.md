# Experiment 01 follow-up — Trace envelope and delivery proof

**Run before:** task 04 freezes the trace schema or delivery guarantees  
**Builds on:** `01-trace-capture-findings.md`

## Confirmed direction

Compiled kernels can emit fixed-width integer trace events with negligible
single-record overhead. Names and other variable-length data remain outside the
kernel; the driver performs conservation and joins kernel-local indices to
stable metadata; adapters run outside the hot path. Live evidence and later
retained/exportable tracing can use the same capture format.

## Remaining validation

1. Test actual concurrent `nogil` execution. Prove per-record event ordering,
   frame-scope behaviour, buffer ownership, and conservation under parallel
   writes; do not infer it from a serial row-major loop.
2. Define durable record mapping. A kernel row index is not a durable record
   identity; prove how driver-side input keys/composite keys join per-row
   offsets to `RecordRef`, including duplicate/missing-ID rejection and
   frame-only events.
3. Measure and improve decode throughput. The measured Python decode rate
   (~1.4 million events/s) is orders of magnitude below capture throughput.
   Compare vectorised/lazy decoding and aggregation against representative
   batch traces, and set a bounded-memory/latency budget.
4. Exercise a real threaded or process adapter with its bounded queue. Define
   backpressure, overflow, adapter failure, and observability semantics:
   blocking is forbidden in kernels, while loss/downgrade must be reported in
   manifest and trace status.
5. Freeze the one-int64 envelope only after exercising every evidence type.
   Show how table rows, tree paths, scorecard bands, reasons, overrides, and
   any declared numeric value evidence fit through constant-table references
   or an explicitly versioned companion payload; do not silently turn one
   structural event into an insufficient explanation record.
6. Measure batch overhead separately from `score()`. One flat trace buffer
   roughly halved the synthetic batch throughput and can consume substantial
   memory (192 MB for 24 million events); tracing needs an explicit
   opt-in/sampling/limit policy for batch workloads.

## Decision gate

Task 04 may use the measured in-kernel primitive now, but it must not freeze
event schema, cross-thread ordering, retained-delivery semantics, or batch
performance guarantees until this follow-up reports.

