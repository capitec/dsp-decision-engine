# Making a `Raw[bytes]` span behave like a string

Follow-on from `notes/strings.md`, which established that a fused array kernel can
hold a string only as an int32 dictionary code or as a `(address, byte length)`
span into Arrow memory. This note answers the ergonomics question: can an opted-in
step be written almost normally against a span?

## What it looks like now

```python
def is_private(sector: Raw[bytes] | None) -> bool:
    return sector == "private"

def is_cafe(sector: Raw[bytes] | None, prefix: str = param("caf")) -> bool:
    return sector.startswith(prefix) and len(sector) > 4
```

`==`, `!=`, `len`, `startswith`, `endswith` and `in` work against a `str`
literal, a module-level `str` constant, a `param()` or another span, in **every
mode**, and `sector[1]` is still the byte length so the old spelling keeps
working.

## Why today's opt-in had to be fixed, not just documented

On `d825f31`, the two most natural spellings against a `Raw[bytes]` input:

```
sector == "private"    interpreted/stepped/fused -> [False, False, ...]   warnings: 0
len(sector) == 7       interpreted/stepped/fused -> [False, False, ...]   warnings: 0
```

Not a divergence: **every mode agreed on the wrong answer, with no warning and no
fallback.** `decider.testing.assert_equivalent` — the one tool a careful user has
for catching exactly this — passed clean. numba resolves `tuple == unicode` to a
constant False, and folds `len(tuple_arg)` to the tuple's length, and a `Raw[bytes]`
value was a bare `UniTuple(int64, 2)` in the kernel and a `(address, length)` array
row outside it, so neither spelling had anything to disagree with.

A consistent, silent, wrong answer is the worst shape this bug could take, and it
is the whole justification for the type below.

| written | `d825f31` | now |
|---|---|---|
| `sector == "private"` | compiles, **always False**, no warning | correct, in the kernel |
| `len(sector) == 7` | compiles, **always False**, no warning | correct, in the kernel |
| `sector == want` (`str` param) | TypingError, falls back to Python | correct, in the kernel |

## The bare-tuple route is wrong

Overloading the operators on the existing `SPAN = types.UniTuple(types.int64, 2)`
was the cheap option. It is unusable. Measured directly:

```
before registering the overloads:  (1, 2) == (1, 2) -> True   len((a, b)) -> 2
after:                             (1, 2) == (1, 2) -> False  len((a, b)) -> -999
```

An `@overload(operator.eq)` on the bare pair captures **every** unrelated
`(int64, int64)` tuple in a user's step, and `@overload(len)` captures every
`len()` of one. `(year, month) == (2024, 1)` and `len((lo, hi))` are ordinary
things to write in a step, and both would silently change answer. So the
operations need a type of their own.

## The type cannot be a tuple either

`SpanType` started as a subclass of `types.UniTuple` with
`register_model(SpanType)(models.UniTupleModel)`, which keeps `span[0]`,
`span[1]`, unpacking and the existing kernel lowering for free. `==`,
`startswith`, `endswith` and `in` all bind correctly that way. `len` does not:

`numba.core.analysis.rewrite_semantic_constants` rewrites `len(x)` to
`ir.Const(x.count)` whenever `x` is a function **argument** whose type is a
`types.BaseTuple`, and it runs on the untyped IR, before type inference. No
overload can reach it. Measured: `len(span)` answers 2 when the span is an
argument and answers correctly when the same value comes from an intrinsic in the
same process. Only stopping the type being a `BaseTuple` fixes it.

So `SpanType` is a plain `types.Type` carrying `dtype`/`count`/`__len__` so that
`models.UniTupleModel` still applies to it. The LLVM value is the same
`[2 x i64]` the kernel already built, so `kernel.py` needed one line and nothing
downstream changed representation. The cost is that unpacking is gone:
`addr, n = span` now raises `failed to unpack span` at compile time, and the two
in-kernel sites that did it (`steps/trees/walker.py`, `steps/tables/matcher.py`)
became `span[0], span[1]`.

## Every mode, not just the kernel

A representation that only works in one mode is not shippable, and it would leave
this whole class of bug invisible to `assert_equivalent`. So an interpreted run
hands a `Raw[bytes]` step a `Span` object (`engine/run/representations.py`)
answering the same six operations with the same null rule. `assert_equivalent`
over a `Raw[bytes]` step now passes for every spelling above, and
`tests/run/test_span_operations.py` runs all of them through it.

Two things had to move to get there:

- **`len` on a null is 0, not -1.** Python's `__len__` may not return a negative
  number (`len()` raises `ValueError`), so -1 could never be honoured outside a
  kernel. The -1 stays where it always was, in the byte length: `sector[1] < 0`
  is the null test, in every mode.
- **`missing_as` on a `Raw[...]` input was broken and is fixed.** The fill was
  applied to the built representation, so `fill_missing` met an `(n, 2)` span
  array and a `str` fill and raised `operands could not be broadcast`; a
  `Raw[str]` code would have taken a `str` into an int array the same way. The
  fill now goes into the strings *before* the representation is built, which is
  also the only place it can, since the representation is cached per value and
  shared between readers while the fill belongs to one declaration.

`_argument` no longer substitutes Python `None` for a null `Raw[bytes]` value: a
null `Span` carries its own -1, and `None` would lose it.

## UTF-8: what matches CPython and what is not offered

A span holds the UTF-8 bytes. Checked against CPython over ASCII, empty, 2-, 3-
and 4-byte values (`tests/run/test_span_operations.py` pins all of it):

- `==`, `!=`: byte equality is code-point equality, because UTF-8 encoding is
  injective. Exact.
- `startswith`, `endswith`, `in`: UTF-8 is prefix-free and self-synchronising (a
  continuation byte is `10xxxxxx` and a lead byte never is), so a byte-level
  prefix, suffix or substring match can only land on a code-point boundary.
  Exact.
- `len`: **not** the byte length. It counts the bytes that are not continuations,
  which is CPython's code-point count for valid UTF-8. `len("privé")` is 5 where
  the span is 6 bytes. This is O(n) where `span[1]` is O(1); `span[1]` remains
  available when the byte length is what is wanted.
- **Ordering (`<`, `<=`, `>`, `>=`) is not offered.** Byte order *is* code-point
  order for UTF-8, so the non-null case would be exact and cheap. The null case
  has no CPython analogue at all (CPython raises on `None < "x"`), and a step
  cannot know a value is non-null before comparing it, so any answer would be an
  invention. Left out rather than guessed.

A compile-time `str` reaches the overload as `types.StringLiteral` when
`prefer_literal=True` is set — for an inline literal, a module-level constant and
even a sliced literal — so its UTF-8 bytes are encoded once at compile time and
lowered as a constant array. A `str` whose value numba cannot see (a config
`Value[str]` const) raises a `TypingError` naming the problem, which falls the
step back to Python: right answer, loud, never silently False.

## Nulls

A null is a span of length -1. One rule: **a null is not a string, and equals
nothing, including another null** — the same rule a NaN float feature follows.

- `span == anything` is False, `span != anything` is True (as Python's `None` does).
- `startswith`, `endswith`, `in` are False, even against `""`.
- `len(span)` is 0, so it reads as empty; `sector[1] < 0` is the null test, and
  `sector == ""` still tells an empty string from a null (a null equals nothing).

CPython has no answer for any of these (`len(None)` and `None < "x"` raise), so
this is decider's convention, stated here because it is not derivable.

## Numbers

One process, `mode="fused"`, 200k rows and 3000 `score()` calls, one-step
pipeline, `benchmarks/span_operations.py`. Best of three consecutive runs that
agreed closely; the box was loaded (load average ~24 on 28 cores), so the ratios
are the result, not the absolutes.

| variant | batch rows/s | score p50 | p99 |
|---|---|---|---|
| numeric kernel, no strings (the ceiling) | 666M | **19.6 µs** | 26 |
| `str`, Python per row (`@python_only`) | 246k | **52 µs** | 63 |
| `str`, njit per row (the default for a semantic `str`) | 245k | 53 µs | 62 |
| span, `sector == "private"` | 13.7M | 63 µs | 72 |
| span, `sector == want` (`str` param) | 13.5M | 69 µs | 78 |
| span, `sector[1] >= 0` (the old spelling) | 14.9M | 63 µs | 71 |

- **The readability is free.** The readable span form is within 8% of the bare
  `sector[1] >= 0` floor in batch and identical at `score()`. The literal's bytes
  are materialised once at compile time, so a comparison is a memcmp of at most
  7 bytes.
- **Batch: 56x the semantic-`str` path** (13.7M against 245k rows/s).
- **Single record: still the wrong choice.** 63 µs against 52 µs for a plain
  `str`, and 3.2x the numeric ceiling. The ~11 µs is the span buffer built per
  call, not the comparison.
- This run puts njit-per-row and Python-per-row level for a semantic `str`
  (245k vs 246k), where the earlier run in `notes/strings.md` had njit-per-row
  4.5x worse. At 200k rows both are dominated by the per-row Python dispatch;
  the earlier gap does not reproduce on a loaded box.
- Re-measured later under different load, every row moved together and the
  ordering held: the ceiling 388–505M rows/s and 22–28 µs, `str` 174–238k and
  61–79 µs, the readable span 11.0–12.5M and 75–98 µs. The untouched
  `sector[1] >= 0` row swung 89–185 µs across the same runs, which is the size of
  the environmental noise — read the table above, not a single run.

## Limits

- `addr, n = span` no longer unpacks, by construction. It fails at compile time
  with `failed to unpack span`, not silently.
- Ordering is absent, as above.
- A `str` whose value numba cannot see falls the step back to Python rather than
  comparing in the kernel.

## Recommendation

Ship it, but do not change the default guidance.

The type earns its place on correctness alone: `sector == "private"` and
`len(sector)` were silent wrong answers that every mode agreed on, and the `len`
fold cannot be fixed any other way. Once the type exists the six operations are
~100 lines and cost nothing measurable, so refusing them would leave the opt-in
unreadable for no saving. The interpreted `Span` matters as much as the kernel
type: it is what puts `Raw[bytes]` steps inside `assert_equivalent`, so the next
bug of this shape is caught by the tool users already reach for.

It does **not** change the answer from `notes/strings.md`: keep strings in Python
and annotate `Raw[bytes]` only for a hot batch step. A single record is still
~20% slower than a plain `str`, and single records are 90% of the workload. What
changes is that the batch opt-in is now something a reviewer can read, and that
reaching for it by mistake fails loudly instead of quietly returning False.

The ~11 µs single-record span setup is the thing left worth attacking; it is a
buffer build per call, not the string work.

## Draft for `GUIDE.md`

To be folded in by whoever owns the guide. Suggested home: after **Parameter
tables**, as its own short section; the row for the "Which construct" table is
`| a string column compared in a hot batch step | `Raw[bytes]` |`. The example
below was run through `assert_equivalent` and gives `[1.15, 1.05, 1.0, 1.0]` in
every mode, so it is ready for `tests/test_guide.py`.

---

### Strings in a compiled step

A `str` input is the right default. It cannot join a fused kernel — numba has no
variable-length string array — so a step that reads one runs once per row while
the rest of the pipeline stays in the kernel. On a single record (`score`) that
costs a few microseconds; over a large batch it is the slowest thing in the
pipeline.

For a **hot batch step**, annotate the input `Raw[bytes]` and it joins the
kernel, comparing the UTF-8 bytes in place:

```python
from decider import Raw, flow, param

def sector_loading(sector: Raw[bytes] | None, private: float = param(1.15)) -> float:
    if sector == "private":
        return private
    return 1.05 if sector.startswith("gov") else 1.0
```

`==`, `!=`, `len`, `startswith`, `endswith` and `in` all work, against a `str`
literal, a module-level `str` constant, a `param()` or another `Raw[bytes]`
input. Every mode gives the same answer, so `assert_equivalent` covers it.

Three things to know:

- **`len` counts code points, as Python does** — `len("privé")` is 5 though it is
  6 bytes. `sector[1]` is the byte length.
- **A null is not a string.** It equals nothing, including another null; it holds
  no prefix, suffix or part; and `len` reads it as empty. Test one with
  `sector[1] < 0`, or avoid it with `missing_as("")` or a `Raw[bytes]` (not
  `| None`) input, which rejects nulls.
- **Compare against something fixed.** A string built at run time cannot be
  compared in the kernel; the step falls back to Python and says so.

Reach for this when a batch step over a string column is measurably hot, and not
before: `score()` on one record is slightly *slower* with `Raw[bytes]` than with
a plain `str`, because the byte buffer has to be built for that one row.
