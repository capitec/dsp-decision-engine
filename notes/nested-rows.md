# `Rows[Item]` read straight out of Arrow

**Question.** `Rows[Item]` gives a step one flat numpy array per `Item` field
plus a per-row slice. `build_rows` built those arrays with a Python loop over
every row and every item, copying value by value. Arrow already stores
`list<struct<...>>` in that shape. Can the arrays be views over the Arrow
buffers, and what does that buy?

## The layout, verified (polars 1.41.2, nanoarrow 0.9.0, `.scratch/probe_layout.py`)

`pl.List(pl.Struct({...}))` exports as:

- **`large_list`**, not `list`: **int64 offsets**. (`list`/int32 is handled too,
  in case polars changes; `fixed_size_list` from `pl.Array` is not, and falls
  back to the Python read.)
- Validity is on the **parent list only**. The child struct of a clean column
  has no validity buffer at all (`has_validity=False`, `null_count=0`), even
  when some parent rows are null.
- The child struct's children are the flat per-field arrays: `double`, `int64`,
  and **bit-packed `bool`** (so a bool field costs one `np.unpackbits`, not a
  view).
- A **sliced frame** carries its offset on the parent list only. The offsets in
  the buffer are read at `[parent_offset : parent_offset + n + 1]` and their
  values index the child fields **directly, with no adjustment for the child's
  own offset**. Verified against the Python read at slice offsets 1, 3, 7, 13
  and 64 — this was the one thing worth checking rather than assuming.
- A slice keeps the **whole child**, so the field arrays are windowed to
  `[offsets[0], offsets[n])` and the offsets rebased onto that window. Without
  that, a null in a required field of a row *outside* the slice would raise for
  a slice that is perfectly clean, and the item index in the message would be
  computed in the wrong coordinate system.
- polars infers a struct field's type from the **first row**, so
  `[[{"q": None}], [{"q": 5.0}]]` gives `Struct({'q': Null})`. nanoarrow
  refuses that export ("Expected array with 0 buffer(s) but found 1"), so
  `read_nested` catches `ArrowImportError` and falls back to the Python read.

What the shim already exposed: `sm_view_child`, `sm_view_offset`,
`sm_view_length`, `sm_view_null_count`, `sm_view_storage_type`,
`sm_view_has_validity`. What it did not: the **address** of a buffer. That is
the whole C change — `sm_view_buffer(view, k)`, four lines. Everything else walks the layout in Python through the accessors
that already existed, once per batch, which is free next to the read itself.
A nested column cannot go in a `FramePlan` (`sm_resolve_col` refuses it), so
`read_nested` binds a `FrameView` with a plan that declares no column and walks
the child by hand.

## What it costs (this box is loaded; same-process comparisons only)

`build_rows` alone, 3 items a row, two fields, one process
(`benchmarks/nested_rows.py`; the Arrow column forces the read below the
threshold so the crossover is visible). `from_series` is the `Series.to_list()`
the boundary already paid before `build_rows` is even called.

| rows | v0 (the copy loop) | Python values | **Arrow** | `from_series` for reference |
|---|---|---|---|---|
| 1 | 19 us | **9** | 130 | 14 |
| 8 | 43 | **33** | 146 | 40 |
| 64 | 257 | **165** | 223 | 184 |
| 256 | 996 | 598 | **570** | 802 |
| 4096 | 14,958 | 9,098 | **5,249** | 14,317 |
| 200,000 | 763,575 | 457,653 | **343,381** | 810,933 |

- **The crossover is ~256 rows**, so `ARROW_ROWS = 256`, and the Python read is
  kept for everything below it. The Arrow read has a **fixed cost of about
  130 us** — `Series.to_frame`, `__arrow_c_stream__`, and a fresh `FrameView`
  (five ctypes buffers and six numpy buffers). `extract_frame` pools its views
  per thread for exactly this reason; here the export happens once per column
  per run, so 130 us amortised over 256+ rows is not worth pooling for.
- `score()` never has an Arrow source at all (`State(plan, _NO_FRAME, 1)` loads
  the record's own Python objects and never builds a frame), so the
  single-record path is Python by construction — exactly the lesson
  `notes/strings.md` records for `str`.
- At 200k rows Arrow is **2.2x** the old copy loop and **1.3x** the rewritten
  Python read. The Arrow read itself is only 16-25 ms of that 343 ms; the rest
  is assembling one namedtuple of slices per row, which both paths pay.
- **On a single record `build_rows` went 19 us -> 9 us**, which is the change
  that matters most for the 90% workload, and none of it is Arrow: it is the
  `rows_schema` cache, `np.fromiter` in place of a per-value numpy store, and a
  shortcut for the one-row shape (nothing to slice, so no windowing).

Two changes mattered as much as the Arrow read:

- `rows_schema` called `get_type_hints` on **every** run of every step reading
  `Rows[...]` — tens of microseconds per `score()`. Now `lru_cache`d, and
  `SteppedRunner._external` resolves the `Item` schema once per plan, so no read
  introspects an annotation at all.
- The per-row assembly is `map(slice, ...)`, `map(array.__getitem__, ...)` and
  `np.fromiter`, so every step runs in C. A Python `for` loop over the rows
  costs a third more (383 ms vs 284 ms at 200k). `np.split` is *worse* (627 ms):
  it goes through `array_split` with Python overhead per piece.
- `np.fromiter(items, dtype, n)` beats `np.array(items, dtype)` on a long list
  (19.5 ms vs 22.5 ms for 600k floats) and ties on a short one. Neither rejects
  a `None`: both turn it into NaN for a float dtype, which is why the null check
  is explicit.

## `Rows[Item]` against plain `list[dict]`, which is the real bar

20k rows, `stepped`, one process (`.scratch/sweep_items.py`). "light" is
`total += price * qty` per item; "heavy" adds 20 arithmetic iterations per item.
Ratio > 1 means `Rows[Item]` wins.

| items | body | `list` p50 | `Rows` p50 | ratio | `list` batch | `Rows` batch | ratio |
|---|---|---|---|---|---|---|---|
| 1 | light | 33.5 us | 56.1 | 0.60 | 289,121 | 159,225 | 0.55 |
| 3 | light | 33.8 | 57.2 | 0.59 | 195,949 | 131,304 | 0.67 |
| 10 | light | 33.9 | 59.2 | 0.57 | 98,581 | 84,432 | 0.86 |
| 30 | light | 35.8 | 60.3 | 0.59 | 39,982 | 39,580 | 0.99 |
| 100 | light | 52.9 | 91.2 | 0.58 | 10,093 | 10,573 | 1.05 |
| 400 | light | 87.8 | 208.5 | 0.42 | 2,436 | 1,980 | 0.81 |
| 3 | heavy | 42.6 | 57.0 | 0.75 | 90,768 | 139,679 | **1.54** |
| 10 | heavy | 58.6 | 56.2 | **1.04** | 33,576 | 82,136 | **2.45** |
| 30 | heavy | 99.8 | 67.3 | **1.48** | 12,442 | 41,095 | **3.30** |
| 100 | heavy | 248.9 | 98.4 | **2.53** | 2,789 | 9,491 | **3.40** |
| 400 | heavy | 1450.6 | 345.9 | **4.19** | 627 | 2,059 | **3.29** |

**`Rows[Item]` costs a fixed ~23 us per `score()` call that `list[dict]` does
not**: building two typed arrays out of Python dicts, plus entering a compiled
dispatcher with a namedtuple-of-arrays argument. A light body never earns that
back — a Python loop doing one multiply-add per item is only a few bytecodes,
and `list` p50 stays flat at ~34 us all the way to 30 items because the loop is
free next to the per-call overhead. `Rows[Item]` earns its keep when the work
per item is real: from ~10 items on `score()` and ~3 items in batch, up to
4.2x on `score()` and 3.4x in batch.

So `Rows[Item]` is worth keeping, but it is a **compute** optimisation, not a
"nested data" optimisation, and the docs should say so: reach for it when the
per-item arithmetic is heavy or the lists are long, and leave a short list with
a one-line body as `list[dict]`. It should not be presented as the way to read
a `list[dict]` column.

## Where the remaining batch time goes (200k rows x 3 items)

`.scratch/trace_run.py`, one `run()` split by phase:

| phase | before | after |
|---|---|---|
| `prepare` (`from_series` -> `Series.to_list()`) | 936 ms | 693 ms |
| `build_rows` | 771 ms | 397 ms (of which the Arrow read 16-25 ms) |
| the rest of `iterate` (one dispatcher call per row) | ~800 ms | ~750 ms |

The Arrow read is no longer the cost. The two things left, in order:

1. **`State.from_frame` calls `Series.to_list()` on the nested column before
   anything reads it** (~640-690 ms of a 1.5 s run, 43%). Nothing reads those
   Python objects when every reader declares `Rows[...]` — the Arrow path
   ignores them except for `len()`. Skipping it needs `State` to hold a nested
   input lazily, which touches `read()` (the hottest method on the
   single-record path) and would change what `state.column()` hands a debug
   session. The same waste is already flagged for string columns by the
   `ponytail:` comment in `State.from_frame`, so it is one fix for both, and it
   belongs to whoever takes that comment on rather than to this change.
2. **One namedtuple of slices per row, in Python** (~210 ms at 200k rows, and
   ~17 of the ~23 us fixed cost of a `score()` call). The floor for the current
   `Fallback` design is one Python object per row per field, because the
   dispatcher is called once per row and wants a real namedtuple. Removing it
   means a compiled driver loop that slices the flat arrays and builds the
   namedtuple inside numba — an `@intrinsic` over a fixed arity, no codegen —
   which would also delete the per-row dispatcher entry. That is the only
   change that would make a light body competitive with `list[dict]`, and it is
   a `Fallback`/unit change, not a nested-data one.

## Nulls: what a null means now, and why

A null inside an item field used to become a **silent NaN**: `build_rows`
stored into a float array, so `{"price": None}` summed as `nan` and a served
request got a plausible wrong number, where the plain `list[dict]` path raises.
Now:

- A field declared **`float | None`** reads a null as **NaN**. That is the
  opt-in, and it is the only null a kernel can hold in band.
- A null in **any other field** raises, naming the field and **which item of
  which row** it is in. Both paths agree: Arrow gets the index from the
  validity bitmap for free; the Python read uses `None in items` (0.12 us on
  one row, where `np.isnan(x).any()` costs 3.9 us) and `items.index(None)`.
  A genuine `float("nan")` in the data is *not* treated as a null.
- `int | None` and `bool | None` are refused at declaration time: there is no
  in-band null for an int64 or a bool. The error says to use `float | None`.
- A **null or absent list for a row** takes the ordinary null policy, so the
  `MissingInputError` a REQUIRED input raises keeps telling the truth:
  `missing_as([])` and `Rows[Item] | None` both now work and both mean "no
  items" / `None`, in all three modes. A non-empty `missing_as([...])` fill is
  refused with a message saying why.
- A **null item** (`[{...}, None]`) and a struct field polars typed as `Null`
  fall back to the Python read, which raises naming the field.

## Lifetime

A field array is `np.asarray(_Borrowed(addr, n, dtype, view))`, so the array's
`.base` chain holds the `_Borrowed`, which holds the bound `FrameView`, which
holds the `ArrowArray` until `release()`. A slice of a field array keeps its
base, so the namedtuples a step receives keep the buffers alive on their own —
`state._representations[vid]`'s `alive` list (which holds the `Nested`) is belt
and braces, not the mechanism.
`test_arrow_backed_field_arrays_outlive_the_frame_they_were_read_from` drops
the frame, the series, the values *and* `alive`, collects, allocates 8 x 200k
floats to reuse the released pages, and then reads the arrays: it fails or
faults if anything in that chain is broken.

`State.write` drops a version's representations, so an override, a rewind or a
`Session.set` rebuilds from the new values — and `State.source` returns `None`
once the values are no longer the frame's column, so an overridden nested input
takes the Python read. A branch or loop row subset indexes the full column's
namedtuple array (`full[rows]`), so the views are the same objects.

## Also fixed on the way

A `Rows[...]` step that falls back to **real Python** (it calls an undeclared
helper, say) used to get raw `list[dict]` values in `stepped`/`fused` while the
interpreted runner gave it the namedtuples — the step then died on
`items.price`. The representation is now built whether or not the unit is a
Python fallback, because for a `Rows[...]` input the arrays *are* the declared
value. Plain `list[dict]` steps are untouched and still run in every mode.

## Dependency on the string work

An `Item` field declared `str` is refused with a message naming the field, and
saying no kernel holds a variable-length string and to read the column in a
`python_only` step. Supporting one means choosing a code or a span per
`notes/strings.md` — and the span case is the interesting one here, because a
`list<struct<... utf8 ...>>` child holds its strings in exactly the buffers a
span points into, so an `Item` `str` field could be `(address, length)` pairs
read with no copy at all. That waits on the semantic-`str` work landing, since
it owns the representation choice.
