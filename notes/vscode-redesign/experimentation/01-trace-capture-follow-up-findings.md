# Experiment 01 follow-up — Trace envelope and delivery proof: findings

**Question:** do the six remaining validations hold — concurrent `nogil`
ordering/ownership/conservation, durable record mapping, decode throughput,
a real bounded-queue adapter, one-int64 over every evidence type, and a
batch-only overhead policy — such that task 04 may freeze the schema and
delivery guarantees?

**Answer:** yes. Concurrent `nogil` capture conserves per-record order when each
invocation owns its buffer; record identity is a driver-side key join (not the
kernel row index); decode is vectorised/lazy, not per-event Python; a bounded
threaded adapter reports loss/failure without ever blocking the kernel; every
evidence type fits one int64 (a float64 via bit reinterpretation, strings via a
versioned constant table); and batch tracing is opt-in with sampling + an event
cap because batch overhead is materialisation-bound while `score()` overhead is
negligible. Task 04 may freeze the envelope, with the policies below.

Machine note: this worktree ran under heavy host load (load average ~10 on 28
cores), so absolute throughput/latency numbers are noisier run-to-run than the
first spike's. Structural conclusions (the contrasts, not the absolutes) are the
evidence here; the first spike's controlled numbers remain authoritative for the
`score()` overhead budget.

## 1. Concurrent `nogil` execution

**Measured:** the compiled kernel is `@njit(nogil=True)` and iterates
`for i in range(n)` (serial, row-major *within* one kernel); concurrency is
threads running the same kernel at once. `concurrent_nogil.py` compiles the same
capture kernel twice (`nogil=False` vs `nogil=True`), runs 4 threads on each,
and verifies per-record order and conservation under 8 concurrent writers.

**How:** each thread owns its own `events`/`offsets` buffer; a shared-array
variant hands two threads disjoint slices of one array. `check_owned` asserts
row `r`'s events are exactly `offsets[r]:offsets[r+1]`, each carrying `(row,
step)` in order. Frame scope is shown through the real engine: a frame step
yields one `before`/`after` pair over all rows.

**Results:**
- GIL released (the contrast): 4 threads doing 4× the work took **5.7× one
  thread** with `nogil=False` (serialized) vs **2.0×** with `nogil=True`
  (concurrent; the residual is memory-bandwidth + host load, not the GIL).
- Per-invocation buffers: **8 concurrent writers, 12.8 M events, all conserved,
  per-record order intact.**
- Shared array, disjoint regions: both writers verify; ownership is a
  per-invocation *region*, never the same region for two threads.
- Frame scope: `[('before','history'), ('before','history/join_history'),
  ('after','history/join_history'), ('after','history')]` — one frame step runs
  once over all rows; its events are frame-scope, driver-emitted.

**Implication for task 04:** the in-kernel write needs no locking *if and only
if* the driver allocates each kernel invocation its own buffer (or a disjoint
region of a shared one). The event order guarantee is per-record only; cross-
record order is the driver's sequence, so a monotonically increasing run
sequence number is still needed for a global order. Frame-scope events must be a
distinct scope, not a row.

## 2. Durable record mapping

**Measured:** a kernel row index `i` is not a durable identity. `record_mapping.py`
builds the `row index -> RecordRef` join from the input frame's key column(s) and
exercises the two rejection rules and frame-only events.

**How:** `map_rows(frame, key_cols, reject_duplicates)` maps each row's key tuple
to a `RecordRef(keys, row)`; duplicates and null keys are rejected; a
frame-scope event is `RecordRef((), None)`.

**Results:**
- Single key `client_id`: offsets `[0..n)` map unambiguously.
- Composite key `(tenant, client_id)` disambiguates a repeated `client_id`.
- Duplicate key on one column: rejected (`"row 1: duplicate key ('a',) already
  at row 0"`), not silently merged — a bare row index would merge two records.
- Missing (null) key: rejected; the driver falls back to a run-local row id
  (`offsets[r]` still names the row), so the join degrades, not fails.
- Frame-only event: `RecordRef((), None)`; the per-run frame id is a buffer
  constant, not a per-event field.

**Implication for task 04:** `RecordRef` is derived in the driver from declared
key columns, never from the kernel row index. Duplicate/missing keys must be
rejected (or fall back to run-local ids) *explicitly*; frame-only events carry
`row=None`. The durable step path is already a compile-time constant table; the
record key is the one field that must come from the input, so it is supplied by
the driver at conservation, not written by the kernel.

## 3. Decode throughput

**Measured:** the first spike decoded ~1.4 M events/s in per-event Python.
`decode_throughput.py` compares per-event Python, vectorised numpy, lazy, and
aggregation decodes of a 24-step × N-row buffer.

**How:** the int64 envelope is unpacked four ways; aggregation computes a
per-kind histogram directly from the packed int64s. A 1 M-row × 24-step budget
(192 MB) is timed end-to-end.

**Results:**

| decode | events/s | 24 M events |
|---|---|---|
| per-event Python | ~1.2 M | ~20 s |
| vectorised numpy unpack | ~53 M | 0.79 s |
| aggregation (kind histogram) | ~162 M | 0.43 s |
| lazy | 0 up front | a record's slice on demand |

**Implication for task 04:** the conservation step must be vectorised (numpy
shift/mask) or lazy (decode a slice on demand), never a per-event Python loop.
Budget: **~0.8 s to vectorise-unpack 24 M events (192 MB)**; aggregation is
~2× cheaper still when only counts are needed. The 8 bytes/event buffer is the
memory budget; the decode is CPU, off the hot path.

## 4. Real threaded adapter with bounded queue

**Measured:** a real `queue.Queue(maxsize=…)` adapter (`adapter.py`): a producer
`drain` that must never block (drop-on-overflow), a consumer thread that can be
slow or fail, and status counters.

**How:** `drain` uses `put_nowait` and counts `Full` as `dropped`; the consumer
increments `failed`/`errors` on a poison item; `status()` reports
`{queue, dropped, delivered, failed, errors}`.

**Results:**
- Backpressure: capacity 16, 1.6 M events drained → 200 k records delivered,
  **1.60 M events dropped on overflow; the producer never blocked.**
- Overflow semantics: drop + count, never block; the kernel write path is
  untouched.
- Adapter failure: 1 poison delivery → `1 failed, 1 error` reported; the producer
  still delivered its records (failure is the adapter's, not the run's).
- Observability: `status()` is the only place loss shows; a trace manifest must
  carry these counters so loss/degradation is reported, never silent.

**Implication for task 04:** the adapter seam is a non-blocking handoff plus a
manifest of `{delivered, dropped, failed, errors}`. Blocking is forbidden in the
kernel; overflow and failure are *reported*, not hidden. A process adapter is
the same contract over `multiprocessing.Queue` (the counters survive; delivery
is pickled, so only decodable/numeric events should cross that boundary).

## 5. One int64 over every evidence type

**Measured:** each evidence type through the one-int64 envelope
(`evidence_types.py`): table rows, tree paths, scorecard bands, reasons,
overrides, numeric value evidence.

**How:** fixed-width indices ride the envelope's fields; strings are indices into
a per-plan versioned constant table; a float64 is bit-reinterpreted into int64.

**Results:**
- Table row: matched row index (small int); the row's label is a constant table.
- Tree path: the walker already emits one int64 "path number" (`walker.py`); the
  `>`-joined node ids stay a constant table (`trace.py` precedent).
- Scorecard band: band index 0..n-1; band label a constant table.
- Reason: index 2 → `"over_limit"`; the string never enters the kernel.
- Numeric value: `42.125 → int64 4631125384006467584 → 42.125` (bit
  reinterpretation round-trips a float64 losslessly).
- Override: kind tag + a versioned companion payload holding the value (the
  `override@<path>` producer already names it).

**Implication for task 04:** the one-int64 envelope holds every evidence type;
no structural event silently becomes an insufficient record. A float64 value
must be bit-reinterpreted (not field-packed); strings are indices into a
**versioned** constant table, so the companion payload needs its own
`schema_version`, not just the envelope's.

## 6. Batch overhead separate from `score()`

**Measured:** batch vs single-record overhead of one shared int64 trace column,
through the real engine (`batch_overhead.py`), plus the opt-in/sampling/cap
policy that bounds memory.

**How:** a 10-step scalar chain, one shared int64 trace column (the recommended
single-buffer design), measured as batch rows/s and `score()` p50; a `budget()`
computes kept bytes and dropped events under a sampling rate and event cap.

**Results:**
- Batch: no trace ~574 M rows/s vs +1 int64 trace col ~305 M rows/s (**~1.9×
  slower**; the first spike's controlled run was 888 M → 440 M, i.e. halved).
- `score()`: no-trace vs +trace p50 indistinguishable within noise (the first
  spike's controlled number: 25.9 → 26.6 us, **<3%**).
- Memory: 1 M rows × 10 steps = 80 MB at full rate; **1% sampling = 1 MB**; an
  event cap drops the overflow and reports it.

**Implication for task 04:** batch tracing is opt-in and *off by default*, with a
sampling rate and a per-run event cap; both are manifest-reported, never silent.
Do not apply the single-record overhead result to batch: batch is
materialisation-bound, so the buffer is sized/drained for the batch path, and
`score()` uses the same kernel with negligible cost.

## Decision gate

Task 04 may freeze:
- the one-int64 envelope (with a `schema_version` on both the envelope and the
  versioned companion/constant tables);
- per-record ordering only (cross-record order stays unspecified, or uses a
  driver-supplied monotonic sequence);
- the non-blocking adapter contract with a `{delivered, dropped, failed, errors}`
  manifest.

Task 04 must NOT freeze until decided by its own work: the exact bit-widths
(20/8/16/24 are a spike choice), the driver-side key declaration surface for
`RecordRef`, and the batch sampling/limit defaults — these are interface choices,
now proven feasible, but still to be designed, not carried forward as-is.
