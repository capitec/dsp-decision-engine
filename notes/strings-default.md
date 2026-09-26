# What a semantic `str` step should do by default

`notes/strings.md` measured one shape. This measures eight, plus a numeric
kernel as a reference row in the same process, and settles the default.

## Method

`uv run python benchmarks/strings_default.py`. One process per table. Each
shape runs in `mode="fused"` (the serving default) over the same 200k-row
frame of four sector strings, and `score()` over the same record. Batch is the
best of three timed runs after a warm-up; `score()` is the p50 and p99 of 5000
calls (2000 interpreted) after 500 warm-up calls and a `gc.collect()`.

Three variants, all measured back to back in one process so machine load
cannot favour one:

- **python per row** — the new default: `compile_call` returns
  `dispatcher.py_func`, `stepped._run`'s `python` flag is true, so nothing but
  a `Raw[...]`/`Rows[...]` declaration is converted.
- **njit per row** — the old default: the same `Fallback`, with its `fn` put
  back on the njit dispatcher (the benchmark also restores `_typed`'s
  pass-through, which the shipped code no longer needs).
- **interpreted** — `mode="interpreted"`, for reference.

The reference row is a one-step numeric pipeline, which stays in a kernel.

## Measured (2026-09-26, loaded dev box, base d825f31)

| shape | variant | batch rows/s | p50 µs | p99 µs |
|---|---|---|---|---|
| numeric kernel (reference) | kernel | 232,705,901 | 26.0 | 88.0 |
| numeric kernel (reference) | interpreted | 649,656 | 36.2 | 65.1 |
| `str` input vs `str` param | python per row | 533,287 | 38.6 | 114.8 |
| `str` input vs `str` param | njit per row | 149,743 | 64.8 | 73.1 |
| `str` input vs `str` param | interpreted | 717,127 | 44.0 | 56.5 |
| `str` input vs body literal | python per row | 647,419 | 30.6 | 40.1 |
| `str` input vs body literal | njit per row | 201,659 | 60.9 | 92.6 |
| `str` input vs body literal | interpreted | 893,155 | 35.5 | 45.1 |
| three `str` inputs | python per row | 539,273 | 35.4 | 43.8 |
| three `str` inputs | njit per row | 109,583 | 93.3 | 106.4 |
| three `str` inputs | interpreted | 649,447 | 56.0 | 94.3 |
| `str` output only | python per row | 769,260 | 30.0 | 53.6 |
| `str` output only | njit per row | 676,517 | 32.6 | 41.1 |
| `str` output only | interpreted | 850,863 | 29.3 | 37.2 |
| `str` input and `str` output | python per row | 726,950 | 28.9 | 48.8 |
| `str` input and `str` output | njit per row | 231,252 | 56.6 | 66.8 |
| `str` input and `str` output | interpreted | 768,513 | 33.6 | 40.8 |
| `str` in a branch condition | python per row | 585,347 | 111.9 | 215.5 |
| `str` in a branch condition | njit per row | 215,860 | 153.2 | 279.6 |
| `str` in a branch condition | interpreted | 339,669 | 114.0 | 232.0 |
| `str` in a loop body | python per row | 270,502 | 269.4 | 700.1 |
| `str` in a loop body | njit per row | 102,913 | 293.3 | 751.4 |
| `str` in a loop body | interpreted | 149,829 | 258.2 | 705.9 |
| `.startswith`/`in`/slicing | python per row | 520,263 | 33.7 | 90.0 |
| `.startswith`/`in`/slicing | njit per row | 166,860 | 63.2 | 160.6 |
| `.startswith`/`in`/slicing | interpreted | 668,902 | 38.2 | 110.0 |

A second run of the same script, on a busier box (its interpreted reference p50
went from 36 to 102 µs), gave the same ordering on every shape and a wider gap:
python per row over njit per row was 1.5x-8.9x in batch and 1.1x-2.5x on p50,
including 0.52M vs 0.34M rows/s and 41.8 vs 103.7 µs for the `str`-output
shape that is nearly a tie above. Read the table as the quieter of the two.

## What that says

- **Python per row wins every shape, on both axes.** Batch: 2.6x-4.9x the
  njit-per-row throughput on every shape that reads a `str`. `score()` p50:
  1.1x-2.6x faster. There is no shape where calling the dispatcher once per
  row pays, so there is nothing to keep it for.
- **A `str` output alone is the one near-tie** (0.77M vs 0.68M rows/s, 30.0 vs
  32.6 µs). Returning a string only *boxes* a result; the cost in the other
  shapes is *unboxing* a Python `str` into numba's unicode on every call, once
  per `str` input. Three inputs is the worst case (0.11M rows/s), which is the
  same ranking `notes/strings.md` found.
- **A `str` step costs about 5 µs on a single record** over the numeric
  kernel (30.6 vs 26.0 p50 for the body-literal shape), so the 90% workload
  barely notices. A `str` *param* adds another 8 µs (38.6), which is params
  bundle validation per call, not the string.
- **Interpreted mode is faster in batch on single-step shapes** (0.89M vs
  0.65M) and slower everywhere a driver has real work to do: the branch
  condition (0.34M vs 0.59M) and the loop body (0.15M vs 0.27M). So a
  string-reading step is not a reason to drop the compiled mode; the numeric
  steps around it still get kernels.
- p99 on this box is noise (the numeric reference reads p99 88 µs against a
  p50 of 26). Read the p50 column and the batch column.

## Decisions

1. **The default for a semantic `str` is a genuine Python fallback.** The
   njit-per-row path existed to keep numba's unicode semantics; nothing needed
   them, because the Python body already has exactly those semantics, and it
   is faster. `_typed`'s semantic-`str` pass-through and `_input_type`'s
   `unicode_type` cases are deleted with it — no kernel signature can ever
   name a variable-length string, so nothing reaches them.
2. **One rule, no exemption for `str` under `strict_compile=True`.** It would
   have been tempting to exempt semantic `str`, since no kernel can hold one
   and the step is not a mistake. But the same is true of `list[dict]`, and
   the escape hatch is one decorator. Exempting `str` would turn strict's
   guarantee ("no step runs in Python unless I said so") into a guarantee with
   a footnote. The warning is also *actionable*: the gap between a `str` step
   (0.5M rows/s) and a `bytes` span in a kernel (7.0M in
   `notes/strings.md`) is 14x, which a batch user wants to be told about.
3. **`@allow_fallback`, one decorator, replacing `python_only`.** Named after
   what the code already calls a `Fallback` and what `fallbacks()` already
   returns. "Dismiss warnings" was the other candidate and was rejected: a
   decorator that only silences must not also satisfy `strict_compile=True`,
   or a user quieting noise has switched off the guarantee. `@allow_fallback`
   says *"I accept how this step runs"*, so strict accepting it is correct —
   the author said so — and it covers the `str` case honestly, where
   `python_only` would have been a lie about a step that used to be compiled.
4. **It does not force Python.** A declared step still gets the compile probe,
   so a step that *can* compile still does. Skipping the probe would save
   nothing that matters — every type-based reason (`list`, `date`, `dict`,
   semantic `str`, semantic `bytes`) returns before any probe — and would
   silently downgrade a step whose author only meant "if it falls back, fine".
5. **Bulk silencing is stdlib.** `decider.FallbackWarning` (a `UserWarning`)
   is the category of every fallback warning, so one
   `warnings.filterwarnings("ignore", category=FallbackWarning)` covers a
   whole project. No engine flag, no config surface. Subclassing
   `UserWarning` keeps the tests that match on `UserWarning` working.
6. **Function steps only.** `@allow_fallback` takes a plain function or
   `step()` of one. A tree, table, scorecard, branch or loop is not one
   function — a row node's `fn` is engine-owned and shared between configs —
   so it raises `TypeError` naming the category filter instead of silently
   doing nothing.
7. **`_numba_type`'s `str` case went too**, though this change did not kill it:
   its own error already said "use float, int or bool", and a `@helper`
   declaring a `str` signature could never be reached from a kernel (a
   semantic `str` step falls back, a `Raw[str]` is an int, a `str` param of a
   byte-reading row node is a span). It now raises that error instead of
   compiling a specialisation nothing calls.

## Two defects the `python` flag was hiding

- **`Raw[...]` served Python values gave a wrong answer, silently.** A step
  whose body numba can't compile (say it calls an undeclared helper) became a
  Python fallback, and `_run`'s one-boolean `python` flag skipped the
  representation for *every* input — so a `Raw[str]` input arrived as a real
  string and `sector == raw_str("private")` compared a `str` with an `int`:
  `[False, False]` in `stepped`/`fused` against `[True, False]` interpreted,
  with no error and no warning. `Raw[...]` and `Rows[...]` are a contract
  about the value, not an optimisation, so the flag is now per input
  declaration: `is_raw(decl.annotation)` builds the representation whatever
  the unit does, and everything else takes the Python value. Refusing to run
  such a step was the alternative and was rejected: "the plain type always
  works" has to hold for `Raw[...]` too. `assert_equivalent` catches the shape
  now, for `Raw[str]`, `Raw[bytes]` and `Rows[Item]`.
- **A numba lowering assert escaped as a naked crash.** `[1, 2] if x > 1 else
  [1.5]` makes numba's lowering fail an internal `assert` with an empty
  message, which is not a `NumbaError`, so it came out of `exe.run` as a bare
  `AssertionError()`. Compiling runs no step code, so an `AssertionError`
  *there* is numba's: `_COMPILE_ERRORS` adds it around the two places we
  compile. Runtime keeps the narrower `FALLBACK_ERRORS`, so a step's own
  `assert` still propagates — pinned by a test. An empty message now reads
  "AssertionError: numba could not compile this step".

## Three warning states, and what each means

| state | warns | `strict_compile=True` | `fallbacks()` |
|---|---|---|---|
| declared (`@allow_fallback`) | no | allows | `@allow_fallback: <reason>` |
| runs in Python, undeclared | yes | raises | `<reason>` |
| compiled, outside the shared kernel (`Rows[Item]`) | yes | allows | `<reason>` |

The third row is now only `Rows[Item]`: each parent row owns a different
number of child rows, so it cannot join an array kernel, but it does compile.
Semantic `str` moved out of it into the second row.

## Contradicting `notes/strings.md`

Nothing, but two of its numbers were optimistic on this box and one claim
needs narrowing:

- It reports the numeric ceiling at 173M rows/s and 22.2 µs p50; this run got
  233M and 26.0 µs. Same order, different day.
- "Plain Python per row is ... within 5 µs of a pure numeric kernel on a
  single record" holds for a `str` compared with a body literal (+4.6 µs) but
  not for one compared with a `str` param (+12.6 µs). The extra is the params
  bundle, not the string.
- Its `str` output row (njit 0.66M vs Python 0.78M) reproduces almost exactly
  (0.68M vs 0.77M). That is the shape where the old default cost least, and it
  is still the shape where the change gains least.
