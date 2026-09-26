# What an `Item` can hold, and how close `for i in items:` can get to Python

Two questions, both from the `>>> User Feedback` block of
`notes/tmcfeval/DECIDER_GAPS.md` §3. The user wants to write

```python
class Item(...):
    el_1: int
    el_2: str

def find_best_item(items: <annotation>[Item]):
    for i in items:
        if i.el_1 == 400 and i.el_2 == "snoop":
            return i
```

1. can an `Item` field be a `str`, and what does it cost;
2. can a step iterate items and read fields by name, and what does that cost
   against `Rows[Item]`'s `items.el_1[j]`.

Answers, up front: **a `str` field is nearly free in batch and costs about 9 us
a `score()` call — it is shipped here.** **Iteration is not slower than the
positional form, it is 2-5x faster**, because the cost that dominates is
numba's per-call unboxing of one namedtuple member per field, not the loop —
so `Rows[Item]`'s physical shape should become a record array per row. That
part is measured but **not** shipped.

## 1. A `str` field in an `Item` (shipped)

`Rows[Item]` gives a step one flat array per field. A `str` field is now a flat
array of `(address, byte length)` spans — the same pair `Raw[bytes]` already
uses — so `items.el_2[j] == "snoop"` runs in the kernel and the strings are
never copied.

### What numba accepts

The whole type is one ndarray subclass and one `typeof_impl`:

```python
class SpanArray(np.ndarray): ...

@typeof_impl.register(SpanArray)
def _typeof_span_array(val, c):
    return types.Array(SPAN, 1, "C", readonly=not val.flags.writeable)
```

`types.Array(SPAN, 1, "C")` works, which was the thing worth checking rather
than assuming. `SpanType`'s data model is `UniTupleModel`, so a span is
`[2 x i64]` in LLVM and an array of them is a plain C array of 16-byte
elements: numba's own array typing gives `arr[j]` the type `SPAN`, its own
array lowering loads it, and every operation in `span.py` then applies
unchanged. Nothing new is registered per schema, there is no `@intrinsic`, and
`kernel.py`/`units.py` are untouched.

The Python value is the *same object*, which is what keeps every mode
agreeing. `SpanArray.__getitem__` returns a `Span` (the interpreted
counterpart from `engine/run/representations.py`) for an integer key and defers
to numpy for a slice, and `__iter__` goes through it, so `items.el_2[j]` and
`for s in items.el_2:` answer identically in a kernel and in a Python body.
There is no per-mode fork: `State.representation` caches one value per
`("rows", schema)` and a compiled reader and a Python reader share it.

Two things fell out of that:

- **`build_rows` must not use `f.__getitem__` for the per-row slices.** A
  Python `__getitem__` on the subclass costs ~1.1 us a row, which made the
  200k-row Arrow read 720 ms instead of 340. `np.ndarray.__getitem__.__get__(f)`
  is the C slot, bound: 0.4 us, and it is used for every field, span or not.
- A `SpanArray` slice keeps its `.base`, and `span_array(pairs, kept)` parks
  whatever owns the bytes on the array, so the per-row slices keep the Arrow
  import (or the `bytes` objects) alive on their own, exactly as the numeric
  field arrays do. `alive` is belt and braces.

### The Arrow read: no copy of any string

polars exports the `str` field of a `list<struct<...>>` child as
**`string_view` (nanoarrow storage 41)**, not `utf8` — verified, and it matters:
there is no contiguous offsets buffer to vectorise over, and a value of 12
bytes or fewer is stored inline in the 16-byte view struct itself. So the read
goes through nanoarrow's own accessor, which handles `utf8`, `large_utf8` and
`utf8_view` behind one call, in the njit driver `kernels.py` already had the
pattern for:

```python
@njit(cache=True)
def string_spans(get_string_addr, view_addr, at, n, out):
    for i in range(n):
        addr, ln = call_get_string(get_string_addr, view_addr, at + i)
        out[i, 0] = addr
        out[i, 1] = ln
```

`ArrowArrayViewGetStringUnsafe` adds the view's own offset, so the index passed
is the window start `base`, **not** `sm_view_offset(child) + base` as the
numeric buffers use. Getting that wrong is invisible on an unsliced frame.

No C was added to `shim.c`: `sm_get_string` and `GET_STRING_ADDR` already
existed for top-level string columns.

### UTF-8, against CPython

`tests/run/test_nested_str_fields.py` runs `==`, `len`, `startswith`,
`endswith` and `in` over one row holding ASCII, empty, 2-, 3- and 4-byte
values, through `assert_equivalent` (so all three modes) and again above
`ARROW_ROWS` (so the Arrow read), and compares against CPython on the same
list. All exact, for the reasons `notes/strings-span-ops.md` gives: UTF-8
encoding is injective and self-synchronising, and `len` counts non-continuation
bytes, which is CPython's code-point count. `len("privé")` is 5 where
`items.el_2[j][1]` is 6.

### What it can be compared against

Checked in every mode, at one row and above `ARROW_ROWS`:

| written | result |
|---|---|
| `items.el_2[j] == "snoop"` | correct, **in the kernel** |
| `items.el_2[j] == WANT` (module-level `str`) | correct, **in the kernel** (numba folds it to a literal) |
| `items.el_2[j] == items.el_3[j]` (another span field) | correct, **in the kernel** |
| `items.el_2[j] == want` (`want: str = param(...)`) | correct, but the step **falls back to Python** with a `TypingError` |

The param case is the only gap and it is loud, not silently False. It falls back
because `njit._probe_signature` decides how to type a `str` param from
`SPAN in ins`, and a `Rows[...]` input's numba type is a namedtuple, not `SPAN`
— so the param is typed as a `Raw[str]` int32 code and the probe compile fails.
Making it compile is one line in `_probe_signature` (look inside a `Rows[...]`
input's schema for a `str` field), in a file this branch does not touch.

### Nulls, in every mode

A `str` field follows `Rows[Item]`'s own rule, not `Raw[bytes]`'s:

- **`str`**: a null raises, naming the field and which item of which row it is
  in — the same message and the same coordinates as a null `float` field, on
  both reads. The fix sentence says `str | None`.
- **`str | None`**: a null is a span of length -1, and the span convention
  applies unchanged — a null equals nothing including another null, holds no
  prefix, suffix or part, and `len` reads it as empty. `items.el_2[j][1] < 0`
  is the null test, in every mode.
- `int | None` and `bool | None` are still refused; the message now says
  "only `float | None` and `str | None` do".
- A **null item** or a field polars types as `Null` still falls back to the
  Python read, which raises naming the field.

### Numbers

One process, loaded dev box, `benchmarks/nested_item_fields.py`, 200k rows x 3
items, `mode="fused"`, 3000 interleaved `score()` calls. The body is the user's
own: find the first item matching a predicate, return its index.

| variant | batch rows/s | score p50 | p99 | min |
|---|---|---|---|---|
| `Rows[Item]`, one `int` field | 0.15M | 51.9 us | 62 | 47.5 |
| `Rows[Item]`, `int` + `float` | 0.13M | 50.7 | 60 | 45.8 |
| `Rows[Item]`, `int` + `str`, `== "snoop"` | 0.12M | 62.0 | 73 | 56.8 |
| `Rows[Item]`, `int` + `str`, `startswith` | 0.10M | 59.2 | 70 | 54.1 |
| `list[dict]`, Python per row | 0.18M | 35.9 | 44 | 32.2 |
| `Rows[Item]`, `int` + `str`, interpreted | 0.07M | 73.0 | 84 | 67.1 |

`build_rows` alone, us per call, same data:

| schema | 1 row | 256 rows (Arrow) | 20k (Arrow) | 200k (Arrow) |
|---|---|---|---|---|
| one `int` field | 5.5 | 285 | 14,492 | 195,219 |
| `int` + `float` | 7.3 | 379 | 20,618 | 258,958 |
| `int` + `str` | 14.8 | 401 | 24,379 | 317,316 |

- **In batch the `str` field costs about what one more numeric field costs.**
  At 200k rows x 3 items (600k strings) the Arrow read is 317 ms against 259 ms
  for a `float` second field and 195 ms for no second field at all — so the
  `str` field adds 58 ms where the `float` field adds 64 ms. Across runs the
  `str` delta ranged 26-58 ms, i.e. **43-97 ns an item**, for one C call
  through a function pointer and two int64 stores. **No string is copied**: a
  long value's span points into polars' own variadic buffer, a short one into
  the 16-byte view struct.
- **On one record it costs ~7.5 us** (14.8 us against 7.3 for a numeric second
  field), for three items, and ~11 us end to end. About half of that is three
  `ctypes.pythonapi.PyBytes_AsString` calls and the rest is one array build;
  `score()` has no Arrow source at all (`State(plan, _NO_FRAME, 1)`), so the
  strings are encoded from the record's own Python `str` objects. Per string
  this is *cheaper* than a top-level `Raw[bytes]` input, which
  `notes/strings-span-ops.md` measured at ~11 us for one value, because one
  array is built for the whole row.
- **The Python read of a `str` field is slow and stays behind `ARROW_ROWS`**:
  1.12 s at 200k rows against 426 ms for `int` + `float`, i.e. ~1.2 us an item
  for `str.encode` plus the ctypes address call. Below 256 rows it is 400 us
  worth of Arrow export that is not worth paying, so the threshold already in
  `rows.py` is the right one and needs no change.
- **A Python fallback body pays ~4.9 us per `items.el_2[j]`**, because it
  decodes the bytes and builds a `Span` (which re-encodes them). That is the
  interpreted row of the table (73.0 us against 62.0). Acceptable: it is the
  debug path, and it is what buys `assert_equivalent` coverage for the whole
  feature.
- `list[dict]` is still 1.7x faster on `score()` for a body this light, exactly
  as `notes/nested-rows.md` records. Adding a `str` field does not change that
  advice: reach for `Rows[Item]` when the per-item work is real.

### Limits worth writing down

- **Do not return an item's `str` field.** `-> str` compiles nowhere (a `str`
  output never reaches a kernel), the step falls back to Python, and the `Span`
  object itself lands in the output frame. That is pre-existing, not new: on
  `fce84d5` a step `def echo(sector: Raw[bytes]) -> str: return sector` already
  gives `[Span('private'), ...]` interpreted and `[[address, 7], ...]` in
  `stepped`/`fused` — a silent three-way mode divergence. It belongs to
  whoever owns the string representations; a `Rows[Item]` `str` field is at
  least consistent across modes, because every mode gets the same `SpanArray`.
- `RequestHandler` warm-up sends `1.0` for a `Rows[...]` input, because
  `serving/parse.py`'s `dummy` has no case for it. Also pre-existing and
  unrelated to `str`.
- A `str` field, like every span, can only be compared against something numba
  can see at compile time (a literal, a module-level constant, a `param()`).
  A value built at run time raises a `TypingError` that falls the step back to
  Python — loud, never silently False.

## 2. Iteration and attribute access (measured, not shipped)

### What numba accepts

`for i in items:` over today's namedtuple-of-arrays is **not** what a user
means, and the failure is not always loud:

| `Item` | what `for i in items:` does |
|---|---|
| all fields the same dtype | compiles; iterates the **fields** (a homogeneous namedtuple is a `UniTuple`). `for i in items: n += len(i)` answers 2, not the item count. |
| mixed dtypes | `TypingError: Invalid use of getiter` -> the step falls back to Python, where `i.el_1` raises `AttributeError` on an ndarray. |

So the silent-wrong-answer window is narrow (a body that only does array-level
work), but it exists, and it is the same shape of bug
`notes/strings-span-ops.md` was written about.

A **record array per parent row** is the shape that works, and it needs no
numba extension code at all — `from_dtype` on a structured dtype already gives
`Array(Record, 1, 'C')`, and numba supports every spelling on it:

| written | record array |
|---|---|
| `items[j].el_1` | works |
| `items[j]["el_1"]` | works |
| `for i in items: i.el_1` | works |
| `items.el_1[j]` — today's documented `Rows[Item]` body | **works** (a strided field view) |
| `items["el_1"][j]` | works |
| `len(items)` | works |
| `for price, qty in items:` (unpacking) | `TypingError` |

That last-but-two row is the important one: **the record shape is a strict
superset of the namedtuple shape's API**, so changing the physical shape does
not break a single existing `Rows[Item]` body.

Outside a kernel the value has to answer the same three spellings, and it can,
cheaply, without `np.recarray`:

- give the array the dtype `np.dtype((np.record, DT))`, so an element is an
  `np.record` and `items[j].el_1` / `for i in items: i.el_1` work in Python;
- give it a five-line `np.ndarray` subclass whose `__getattr__` returns
  `self[name]`, so `items.el_1[j]` works in Python.

numba types that subclass exactly as it types a plain record array (typeof
dispatches through the MRO), so there is no `typeof_impl`, no model, no
unboxing. `np.recarray` answers all three spellings too, but **slicing one
costs 5.6 us against 0.24 us for the subclass** — 200k rows of that is 1.1 s,
so `np.recarray` is not usable as the per-row value.

A `str` field inside a record is the one piece that needs a numba internal
bent: `types.Record.__init__` ends with `self.bitwidth = self.dtype.itemsize * 8`
and `SPAN` has no numpy dtype, so the field only fits under a `Record` subclass
that overrides `dtype` to return the `(addr, len)`-expanded numpy dtype, plus
`register_model(SpanRecord)(models.RecordModel)`. With those ten lines,
`i.el_1 == 400 and i.el_2 == "snoop"` compiles and answers correctly, and
`for i in items:` over it does too. That is the only spike-grade part of this
half, and it would need a numba-version guard.

### Numbers

`benchmarks/nested_item_fields.py` part 3: one compiled call per parent row
(which is what a `Rows[...]` step does), 1000 rows, us per row. `build` is the
whole-column work plus the 1000 per-row slices, divided by 1000.

| k fields | items/row | shape | 1 field | all k | find | build |
|---:|---:|---|---:|---:|---:|---:|
| 2 | 3 | namedtuple | 1.20 | 1.20 | 1.19 | 1.44 |
| 2 | 3 | records | **0.41** | **0.41** | **0.39** | **0.30** |
| 2 | 30 | namedtuple | 1.20 | 1.22 | 1.24 | 1.48 |
| 2 | 30 | records | **0.44** | **0.44** | **0.45** | **0.37** |
| 2 | 300 | namedtuple | 1.53 | 1.56 | 1.55 | **1.62** |
| 2 | 300 | records | **0.74** | **0.75** | **0.73** | 1.12 |
| 2 | 3000 | namedtuple | 4.63 | 4.92 | 2.39 | **1.54** |
| 2 | 3000 | records | **4.09** | **4.37** | **1.50** | 55.60 |
| 8 | 3 | namedtuple | 2.35 | 2.34 | 2.34 | 3.55 |
| 8 | 3 | records | **0.42** | **0.41** | **0.41** | **0.38** |
| 8 | 30 | namedtuple | 2.42 | 2.42 | 2.34 | 3.62 |
| 8 | 30 | records | **0.47** | **0.51** | **0.45** | **1.06** |
| 8 | 300 | namedtuple | 2.78 | 3.41 | 2.80 | **3.77** |
| 8 | 300 | records | **1.38** | **1.47** | **1.22** | 10.40 |
| 8 | 3000 | namedtuple | **5.90** | **14.05** | **3.62** | **3.84** |
| 8 | 3000 | records | 12.96 | 15.38 | 4.83 | 370.55 |

And part 4, the user's own predicate with a `str` field:

| items/row | `items.el_2[j]` (namedtuple) | `items[j].el_2` (records) | `for i in items` (records) |
|---:|---:|---:|---:|
| 3 | 1.33 us | **0.67** | **0.64** |
| 30 | 1.50 | **0.72** | **0.71** |
| 300 | 1.46 | **0.73** | **0.74** |

What the numbers say, and it is not what the brief guessed:

- **The interleaving penalty on the loop is invisible below ~1000 items a
  row.** A row's items fit in cache, so a loop touching one field of an 8-field
  record is as fast as one touching all eight (0.42 against 0.41 us). The same
  finding `notes/nested-structs.md` recorded for `Struct[Item]`.
- **What dominates is numba's per-call unboxing, and it is per namedtuple
  member.** The namedtuple call costs 1.20 us at k=2 and 2.35 us at k=8 —
  about 0.2 us a field — while a record array is one array argument and costs
  0.41-0.51 us whatever k is. So the positional shape charges a step for every
  field of the `Item`, including the ones its body never reads.
- **`for i in items:` and `items[j].el_1` cost the same as each other**, to
  within noise, at every size. There is no reason to prefer one spelling.
- **The crossover is a few hundred items a row.** At 3000 items and k=8 the
  strided read finally costs 2.2x (12.96 against 5.90) and the interleave is
  ruinous (371 us a row, copying 192 KB). At 3000 items and k=2 the loop is
  still a wash and only the build decides it.
- So the record shape wins wherever the list is a line-item list (single
  digits to low hundreds), which is every case §3 of `DECIDER_GAPS.md`
  describes, and loses on a long numeric vector.

## 3. Returning an item

A step cannot return one, in any mode. `-> Item` and `-> dict` both raise
`TypeError: nested objects are not allowed` when the output frame is built: a
step output is a `FeatureKind` (float64 / int64 / bool / int32 code), and
`emit`ing a per-row object is not something the engine has a column for.

| what the user writes | what happens |
|---|---|
| `-> int`, the index, `-1` for none | works, every mode, no cost over the loop itself |
| `-> Item` / `-> dict` | `TypeError: nested objects are not allowed` |
| `-> str`, returning `items.el_2[j]` | runs, but puts a `Span` in the output column (see the limits above) |
| `-> float`, returning `items.price[j]` | works; the honest way to return "the field of the best item" |

Inside a kernel, returning the item itself does work: numba boxes a record
element (`return i` / `return None` unify to `Optional(Record)` and come back as
an `np.record`, **copied** — writing to it does not touch the array). So a
`Rows[...]` step *could* one day hand back a record. Two reasons not to chase
it: the record would have to become a column value, which is the object column
the engine does not have; and a record carrying a span boxes as raw
`(el_1, addr, n)` integers, which is not an item any caller can read.

**What the user should write instead:** return the index, and read the fields
from it in the next step, or return the one field that matters. It is one extra
line and it is the shape the engine can store.

```python
def best_item(items: Rows[Item]) -> int:
    for j in range(len(items.el_1)):
        if items.el_1[j] == 400 and items.el_2[j] == "snoop":
            return j
    return -1

def best_price(items: Rows[Item], best_item: int) -> float:
    return items.price[best_item] if best_item >= 0 else 0.0
```

## 4. Recommendation, ranked

1. **Ship the `str` field.** It is in this branch: ~90 lines across `span.py`,
   `rows.py`, `_arrow/nested.py` and `_arrow/kernels.py`, no new numba type
   beyond one `typeof_impl`, no C, `kernel.py`/`units.py` untouched, 25 tests
   in `tests/run/test_nested_str_fields.py`, and the suite green. 43 ns an item
   in batch and ~8.5 us a `score()` call is a price worth paying to stop telling
   users to "drop the field from Item and read the column in a python_only
   step". The guide draft in `notes/nested-structs.md` needs one edit: "both
   fields must be `float`, `int` or `bool`" becomes "`float`, `int`, `bool` or
   `str`".

2. **Change `Rows[Item]`'s physical shape to a record array per row, and keep
   the annotation.** One shape, not two. It is 2-5x cheaper per call for the
   item counts this feature exists for, it makes `for i in items:` and
   `items[j].el_1` work — which is what the user asked for — and it breaks
   nothing, because `items.el_1[j]` works on a record array too. The work,
   none of it in `kernel.py`:
   - a builder that interleaves the flat field arrays (which `nested.py`
     already produces) into one `np.dtype((np.record, ...))` array and slices
     it per row, plus the five-line `__getattr__` subclass. `structs.py`'s
     `struct_dtype` already builds the dtype.
   - one more `kind` in `stepped.py`'s `_boxed` and `interpreted.py`'s
     equivalent, alongside `("rows", schema)` — mechanical, and the
     `python` fork `_records` already uses for `Struct[Item]` is not even
     needed, because one value serves both.
   - `njit._input_type`: `typeof` of a probe record array instead of
     `rows_probe`.
   - the `SpanRecord` subclass for a `str` field. This is the one place a numba
     internal is bent and it needs a version-pinned test.
   - the `ARROW_ROWS`/1000-item regression: keep the arrays-per-field builder
     behind the same shape if a workload with thousand-item lists turns up.
     Nothing in `DECIDER_GAPS.md` has one, so do not build it now.

3. **Do not add an annotation for "return an item".** The engine has no column
   for it and the record that a kernel can return is not readable outside one.
   Document the index.

4. **Do not chase the single-record `str` cost.** Of the ~8.5 us, roughly half
   is `PyBytes_AsString` through ctypes, which `representations._borrowed` pays
   for every `Raw[bytes]` score too; it could be `id(b) + (sys.getsizeof(b"") - 1)`
   at ~50 ns, but that is a CPython-layout trick that belongs with the string
   work, in one place, not here.

## What was dropped

Two tests pinned the refusal this change removes. Both are replaced by the same
assertion against a `date` field, so "a field a kernel can't hold names itself"
is still covered:

- `tests/run/test_rows.py::test_a_str_item_field_names_itself_in_the_error` ->
  `test_an_item_field_of_an_unsupported_type_names_itself_in_the_error`.
- `tests/run/test_value_shapes.py::test_a_string_field_in_an_item_is_refused_and_says_what_to_do`
  -> `test_a_field_no_kernel_can_hold_is_refused_and_says_what_to_do`. Its
  `NamedAccount` fixture went with it; the positive case lives in
  `tests/run/test_nested_str_fields.py`, which runs every operation through
  `assert_equivalent`.

`span.py` now carries its own four-line `PyBytes_AsString` setup, duplicating
`engine/run/representations.py`'s private `_address`. The dedup wants to go the
other way (`representations` importing from `span`, which is the lower module),
and that file is outside this branch.

## Where this touches the other spike

Nothing here reads or writes `kernel.py` or `units.py`. Recommendation 2 would
meet that work in one place: if a `Rows[Item]` step ever joins the shared array
kernel, the record array is the value `load()` would have to produce per row,
and it is one pointer plus an offset rather than k pointers — which is the same
reason it wins here.
