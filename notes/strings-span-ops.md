# Making a `Raw[bytes]` span behave like a string in a kernel

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

`==`, `!=`, `len`, `startswith`, `endswith` and `in` all work, against a `str`
literal, a module-level `str` constant, a `param()` or another span. `sector[1]`
is still the byte length, so the old spelling keeps working.

## The bare-tuple route is not merely risky, it is wrong

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
`types.BaseTuple`, and it runs in the untyped IR, before type inference. No
overload can reach it. Measured: `len(span)` answers 2 when the span is an
argument and answers correctly when the same value comes from an intrinsic in
the same process. Only stopping the type being a `BaseTuple` fixes it.

So `SpanType` is a plain `types.Type` carrying `dtype`/`count`/`__len__` so that
`models.UniTupleModel` still applies to it. The LLVM value is the same
`[2 x i64]` the kernel already built, so `kernel.py` needed one line and nothing
downstream changed representation. The cost is that unpacking is gone:
`addr, n = span` now raises `failed to unpack span` at compile time, and the two
in-kernel sites that did it (`steps/trees/walker.py`, `steps/tables/matcher.py`)
became `span[0], span[1]`.

## What today's opt-in actually does (the reason this is worth having)

On the base commit, the three natural spellings against a `Raw[bytes]` input
behave like this:

| written | today | after |
|---|---|---|
| `sector == "private"` | compiles, **always False**, no warning | correct |
| `len(sector) == 7` | compiles, **always False**, no warning | correct |
| `sector == want` (`str` param) | TypingError, falls back to Python | correct, in the kernel |

The first two are silent wrong answers, not just bad ergonomics. numba resolves
`tuple == unicode` to a constant False and folds `len(tuple_arg)` to 2. That is
the strongest argument for the extension type: it is the only thing that removes
them, because the `len` fold is unreachable from an overload.

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
lowered as a constant array. A `str` that numba cannot see the value of (a
config `Value[str]` const) raises a `TypingError` naming the problem, which falls
the step back to Python: right answer, loud, never silently False.

## Nulls

A null is a span of length -1. One rule: **a null is not a string, and equals
nothing, including another null** — the same rule a NaN float feature follows.

- `span == anything` is False, `span != anything` is True (as Python's `None` does).
- `startswith`, `endswith`, `in` are False, even against `""`.
- `len(span)` is -1, so `len(sector) < 0` is the null test. CPython has no answer
  here (`len(None)` raises), so this is decider's convention, not CPython's.

## Numbers

One process, `mode="fused"`, 200k rows and 3000 `score()` calls, one-step
pipeline. Best of three runs; the box was loaded (load average ~24 on 28 cores),
so treat the ratios, not the absolutes, as the result. `benchmarks/span_operations.py`.

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

## Limits

- **Compiled modes only.** `mode="interpreted"` hands a `Raw[bytes]` step the
  span array row itself, so `sector == "private"` is False for every row there.
  That gap pre-dates this change (only `sector[1]` ever meant anything in
  interpreted mode) and the readable spelling inherits it. Making it
  mode-equivalent needs the interpreted runner to hand over an object that
  answers the same operations *and* keeps the -1 span for a null, which
  `_argument` currently replaces with Python `None`. Not done.
- `addr, n = span` no longer unpacks, by construction.
- Ordering is absent, as above.

## Recommendation

Ship it, but do not change the guidance.

The extension type earns its place on correctness alone: `sector == "private"`
and `len(sector)` are silent wrong answers today, and the `len` fold cannot be
fixed any other way. Once the type exists the six operations are ~100 lines and
cost nothing measurable, so refusing them would leave the opt-in unreadable for
no saving.

It does **not** change the answer from `notes/strings.md`: keep strings in
Python and annotate `Raw[bytes]` only for a hot batch step. A single record is
still 20% slower than a plain `str`, and single records are 90% of the workload.
What changes is that the batch opt-in is now something a reviewer can read, and
that reaching for it by mistake fails loudly instead of quietly returning False.

The ~11 µs single-record span setup is the thing left worth attacking; it is a
buffer build per call, not the string work.
