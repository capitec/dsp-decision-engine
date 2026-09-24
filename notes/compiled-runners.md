# Stepped and fused runners

`engine/run/runners/stepped.py` and `fused.py` choices that sessions and later
tasks build on.

- **Both are the interpreted driver with compiled calls.** `SteppedRunner`
  subclasses `InterpretedRunner` and replaces only how a scalar or row call runs
  (through its `compile_plan` unit); frame steps, branches, loops, row subsets
  and merges are the interpreted code. `FusedRunner` also replaces the walk of
  a sequence, so a unit covering several calls runs once, and runs a branch
  or loop of plain scalar steps as one packed kernel when it can
  (`packed-control-flow.md`).
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
  with data. A kernel can run a step on codes only if it reads one `str`
  input, compares it with `str` params, and has no `str` literal (a body
  literal would compare a code against a string and silently never match).
  decider2 raised for any other `str` step. Here it runs through the Python
  `Fallback` instead, with one warning per executable naming the step, why
  and how to make it compile; `SteppedRunner(strict=True)` (and so
  `exe.runner.strict = True` before the first run) raises as decider2 did.
  Raising put every example project on interpreted mode because of one
  inherited step (`norm_table_version`, two `str` inputs), at about 300
  rows/s.
- **Only numbers and strings enter a kernel.** A step reading a `date`,
  `list`, `dict` or other object input is a `Fallback` without asking numba:
  numba types such an input as float64 and may compile a step that never
  touches it, and the object column then fails to convert at the first run.
  A `Fallback` stores any output that isn't `float`/`int`/`bool` or a
  `Literal` code as an object array. An int column that a step reads as
  `float` (another step reads it as `int`) is cast to float64 for that
  kernel; a decision table otherwise reads the int as a missing float slot.
- **Measured on example projects 07 and 08 (Sonnet), 3000 synthetic rows,
  fused mode:** about 3.3x (07) and 4x (08) interpreted throughput. Most of
  what's left is the object boundary (`Series.to_numpy()` on list and struct
  columns, and building object output columns), frame steps and ~40 Python
  fallbacks per pipeline, not kernels (under 0.1 s of about 3 s). Outputs
  match interpreted mode except where a step calls `round(x, 2)`: numba
  rounds some halves the other way (0.01 on up to 10% of rows).
- **Nulls behave the same in every mode.** A step returning `None` writes a
  null in interpreted mode too (it used to write `None` as a valid value, or
  NaN for a float). Compiled kernels write a validity mask for an output
  declared `T | None`, stored as `T`'s dtype (`int | None` is int64), and a
  kernel is split after a nullable output that a later step of the same run
  reads, so that reader's null policy is applied by the driver, as in
  interpreted mode.
- **Python code sees Python scalars.** Interpreted mode and the per-call
  Python fallback pass each column through `.tolist()`, so a step gets
  `float`/`int`/`bool`, not numpy scalars: `x / 0.0` raises
  `ZeroDivisionError` as it does in a kernel, instead of warning and giving
  inf/nan.
- **Errors from step functions carry the step.** The runner adds a note
  (`exc.add_note`, Python 3.11+; nothing on 3.10) naming the step path, and in
  interpreted mode the frame row, keeping the exception's type. A fused
  kernel can't tell which of its steps raised, so the note lists them all
  and suggests stepped mode. Row indices aren't reported from kernels or
  Python fallbacks.
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
