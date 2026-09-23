# Stepped and fused runners

`engine/run/runners/stepped.py` and `fused.py` choices that sessions and later
tasks build on.

- **Both are the interpreted driver with compiled calls.** `SteppedRunner`
  subclasses `InterpretedRunner` and replaces only how a scalar or row call runs
  (through its `compile_plan` unit); frame steps, branches, loops, row subsets
  and merges are the interpreted code. `FusedRunner` also replaces the walk of
  a sequence, so a unit covering several calls runs once.
- **A fused kernel's checkpoints carry its first step's origin.** One
  `before`/`after` pair per kernel; the later steps of the kernel yield nothing.
  The enclosing sequence was the alternative, but it already yields its own
  pair, and a sequence may hold several kernels. A session breaking on a later
  step of a kernel needs `runner.units[call.id]` to map it to the kernel.
- **Strings are codes from one table per runner.** A `str` param's literal is
  interned into the runner's table when its bundle is converted (once per
  params document and node); a `str` input column is looked up in that table
  (-1 when absent). decider2 resolved literals through each frame's column
  categories, which made the converted bundle depend on the data, so it could
  not be cached across calls, and needed the sentinel only for absent literals.
  With one table, `score()` reuses the converted bundle and an absent literal
  never matches, as before. The table only grows with param literals, never
  with data. Kept from decider2, as errors naming the step: a compiled step
  reading a `str` input must declare a `str` param (a body literal would
  compare a code against a string and silently never match), may read only one
  `str` input, and may declare a `str` param only if it reads a `str` input.
- **Nulls behave the same in every mode.** A step returning `None` writes a
  null in interpreted mode too (it used to write `None` as a valid value, or
  NaN for a float). Compiled kernels write a validity mask for an output
  declared `T | None`, stored as `T`'s dtype (`int | None` is int64), and a
  kernel is split after a nullable output that a later step of the same run
  reads, so that reader's null policy is applied by the driver, as in
  interpreted mode.
- **`score()` without frame steps skips polars.** The record goes into one
  read-only array per dtype, viewed per input (one `np.array` per input cost
  about 8 µs of a 36 µs call), and the output dict is built from the state. A
  read-only array types like a column read from a frame, so batch and single
  record share a kernel specialisation. A pipeline with frame steps scores
  through a one-row frame.
- **The params key of an empty document is a constant**, not a JSON hash per
  call (3.7 µs of a single-record call).

**Measured** (`benchmarks/modes_vs_decider2.py`, flagship, 1M rows through
`run(df)`, 20k `score(dict)` calls, CPython 3.14, dev box, 2026-09-23):

| engine | batch rows/s | score p50 | score p99 |
|---|---|---|---|
| decider2 fused | 8.2M | 26.5 µs | 40.6 µs |
| decider fused | 12.5M | 27.7 µs | 34.7 µs |
| decider stepped | 12.9M | 37.1 µs | 47.6 µs |
| decider interpreted | 0.25M | 59.1 µs | 84.5 µs |

Repeated latency-only runs put fused p50 within 1 µs (2-5%) of decider2 and
p99 level with it.
