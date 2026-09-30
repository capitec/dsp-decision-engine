# Experiment 01 — compiled decision-trace capture: findings

**Question:** can supported compiled/nogil execution modes record compact,
ordered decision evidence without Python allocation/callbacks in the kernel,
while preserving the engine's execution semantics and acceptable overhead?

**Answer:** yes. The kernel can already write fixed-width integers (int64 /
float64 / bool) into pre-allocated arrays and a growable `Sink`; a decision
trace is one more such write. Everything with a name or a string stays out of
the kernel and is joined back in Python. The in-kernel capture write is
sub-nanosecond; the measurable cost is entirely in materialising and decoding
the captured buffer, which happens off the hot path.

## What was measured

Two scripts under `01-trace-capture/` (`envelope.py`, `overhead.py`), run with
`uv run python`. `overhead.py` exercises representative paths through the real
engine in all three modes; `envelope.py` measures the raw in-kernel primitive
and the post-hoc decode. Full output is in `envelope.log` and `overhead.log`.

Modes exercised: `interpreted`, `stepped`, `fused` (all three, `run` batch and
`score` single-record). Constructs exercised: a scalar chain (record path), a
`branch` + `loop` (packed into one kernel in fused mode), a `TreeConfig` with
`path_output` and `trace_output`, a `DecisionTableConfig`, a `frame_step`, and
an override via `Session.set`. I read the `10-retail-credit-e2e` pipeline but
benchmarked synthetic micro-pipelines instead: the e2e project needs
`credit_core` on `PYTHONPATH` and its compile cost would dominate a controlled
measurement without changing the conclusions.

Not exercised: a real multi-threaded concurrent capture under `nogil=True`
(the kernel is compiled `nogil`, but concurrency was not benchmarked), and a
real out-of-process adapter (backpressure was simulated with a bounded queue).

## 1. Minimal event envelope, and what is capturable in-kernel

One event is **one int64 (8 bytes)**. Field widths are a spike choice, not a
contract (20-bit step ref, 8-bit kind, 16-bit arm, 24-bit iteration — all
shiftable):

- **schema version** — a per-run/buffer constant, *not* per event (a run has
  one, stored once).
- **step reference** — the call's index in its kernel's program (`Spec` order,
  `SrcKind.RES` slot `s`). This is a *derived* reference: kernel-local and
  order-dependent. The *durable* path (`"chain/a3"`) is a compile-time
  constant table; the event carries only the index.
- **parent/child flow context** — branch arm index (`Fork`'s arm `k`) and loop
  iteration counter (`Repeat`'s `count`). Both already exist as integers in a
  packed kernel; in stepped/interpreted modes the driver fills the same fields
  from `Checkpoint.arm` / `Checkpoint.iteration`.
- **capture-time metadata** — the trace-point kind (step / branch-arm /
  loop-iteration / table-row / tree-node / override), an 8-bit tag.
- **record key / frame scope** — the row index `i`, implicit in the buffer
  layout (a flat buffer plus per-row offsets, exactly how `Sink` already
  records variable-length output), not a per-event field.

**Capturable in-kernel:** any fixed-width numeric evidence — step index, arm,
iteration, kind, and numeric payloads (the matched row index, the tree path
number, the scorecard band value). The tree walker already computes a whole
path as one int64 "path number" (`decider/steps/trees/walker.py`), which is
the existing precedent.

**Not capturable in-kernel:** anything variable-length or named — rule names,
tree node ids, table row labels, reasons-as-strings. These enter the kernel
as integer indices into a per-plan constant table (as `trace_output` does:
int64 path choice → `">"`-joined node-id string, decoded in Python). Numba
kernels hold no variable-length strings, so the envelope must never grow a
string field.

## 2. Capture → conservation → adapter boundary

- **Capture (in-kernel, `nogil`):** int64 writes into a pre-allocated buffer
  (fixed-size per-record slots, or a `Sink`-style amortised-doubling buffer
  for append-only). No Python allocation, no callback, no GIL. Measured at
  ~1.2 G events/s, i.e. ~0.8 ns/event (24 M events in ~20 ms).
- **Conservation (post-kernel, Python, on the request path):** verify
  count and per-record order (trace-point conservation), join the kernel-local
  step index to the durable path via the constant table, and turn the numeric
  event into a structured one. This is where a `Session.events`-style record
  can be produced; it is also where an override or fallback (a step that ran
  in Python, row by row) emits, so the envelope must be fillable by the driver
  too, not only by a kernel.
- **Adapter (off the request path, another thread/process):** delivery,
  redaction, enrichment, persistence, retention. Backpressure is the
  adapter's problem (bounded queue → drop or block); the kernel never blocks
  or allocates per event. Simulated: a bounded queue that clears on overflow,
  so the kernel's write path is untouched.

The seam is therefore: *the kernel emits only fixed-width integers; every name,
string and policy decision leaves the kernel at the conservation step.*

## 3. Ordering guarantee and no-op overhead budget

**Ordering.** Within one kernel the engine iterates `for i in range(n): body`,
so events are row-major and, per record, in step execution order — guaranteed,
and the spike's conservation check confirms 24 M in == 24 M out, row 7's
events being its own in step order. Across kernels the Python driver runs them
in sequence; a single flat append-only buffer would interleave rows, so either
per-record buffers (offsets) or a monotonically-increasing sequence counter
supplied by the driver is required for a global order. Within a packed branch
or loop the arm/iteration fields order events per row; a fused kernel's
intermediate scalar values still never exist in `State`, so evidence (the
int64s) is orderable even though the values behind them are not inspectable
without `stepped`.

**No-op overhead.** The capture write itself is negligible (~0.8 ns/event).
The engine-level cost comes from *materialising* the trace, not writing it
(`overhead.log`, 10-step scalar chain, fused):

| variant | batch rows/s | score p50 | score p99 |
|---|---|---|---|
| no trace | 888 M | 25.9 us | 33.8 us |
| +1 int64 column (single buffer) | 440 M | 26.6 us | 36.6 us |
| +1 int64 column per step (10) | 137 M | 60.3 us | 72.7 us |

So a single shared int64 trace buffer costs **<3% on `score()`** (25.9 → 26.6 us,
the usual single-record workload) and roughly halves *batch* throughput because
the batch path is output-materialisation-bound. The per-step-column spelling is
the wrong design: it multiplies the kernel's output surface and is 6× slower.
The existing real in-kernel trace (`TreeConfig.trace_output`) shows the same
split: no trace 20 M rows/s vs 2.8 M rows/s with the trace, but that is the
**String** path materialisation, not the numeric walk. Conclusion for the
budget: **one flat int64 buffer per run, drained and discarded by the driver;
`score()` overhead <3%, no per-step columns, never materialise a string in the
kernel.**

## 4. Live evidence vs retained/exportable tracing

They are compatible and can ship in stages. Live runtime evidence means
decoding the buffer *after each kernel returns* and streaming events at
kernel granularity — which is exactly the granularity a fused run already
yields to the driver (one checkpoint per kernel; `stepped` yields per-step).
Retained/exportable tracing decodes the same buffer once at end-of-record and
hands the stream to the adapter. Because the in-kernel format is identical,
live streaming can be delivered first and retained/exportable tracing later
with **no interface change**. The one thing live delivery cannot do is expose a
fused kernel's internal intermediate scalar mid-kernel; inspecting those still
requires `mode="stepped"`, which is already the debug path. The buffer format
must therefore be stable and versioned (`schema_version`) so a later
conservation layer reads what an earlier capture wrote.

## Numbers (interpreted vs compiled)

`overhead.log` (300 k rows batch; score = single record, p50/p99 in us):

| path | fused | stepped | interpreted |
|---|---|---|---|
| scalar chain, no trace | 888 M rows/s, 25.9 us | — | — |
| scalar chain, +1 int64 | 440 M rows/s, 26.6 us | — | — |
| scalar chain, +10 int64 | 137 M rows/s, 60.3 us | 17.1 M, 136.5 us | 79 k, 280 us |
| branch + loop | 17.1 M rows/s, 65.4 us | 1.6 M, 549 us | 31 k, 254 us |
| tree, no trace | 20.0 M rows/s, 20.6 us | — | — |
| tree, `trace_output` | 2.8 M rows/s, 35.6 us | — | — |
| table | 3.0 M rows/s, 29.5 us | — | 183 k, 37.5 us |
| frame step | 91.8 M rows/s, 1158 us | — | 85 M, 1158 us |

`envelope.log`: event size 8 bytes; buffer 8 bytes/event (192 MB for 24 M
events); in-kernel capture ~1.2 G events/s; post-hoc Python decode ~1.4 M
events/s.

The frame step's `score()` latency (~1.16 ms) dwarfs every other path: a
`frame_step` runs in Python and builds a polars frame per record. That is the
real per-record ceiling for any live-trace design on a frame pipeline, not the
trace capture.
