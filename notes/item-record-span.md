# Can a record carry a span robustly? Yes -- but that isn't the whole question

`notes/nested-data-decisions.md`'s "Open" section and `notes/nested-item-fields.md`
§2 left one thing blocking the record-per-row shape: `types.Record.__init__` ends
with `self.bitwidth = self.dtype.itemsize * 8`, and `SPAN` has no numpy dtype, so
carrying a `str` field needs a `Record` subclass overriding `.dtype`, which the
spike called spike-grade and numba-version-fragile. This note tests that claim
directly against numba 0.67.0 (the version installed here; `pyproject.toml` pins
only `numba>=0.67.0`, no upper bound) and finds it **not fragile** -- but finds a
second, more important reason to hold off that neither note anticipated, because
it postdates them: the shared-array-kernel path landed in
`notes/nested-rows-in-kernel-landed.md` changed what a record shape would cost.

All prototypes are in `.scratch/item-record-span/`, run directly (`uv run python
.scratch/item-record-span/routeN_*.py`). No production file changed.

## Route 1: a `Record` subclass overriding `.dtype`, plus `register_model`

Built `SpanRecord(types.Record)` with two fields, `el_1: int64` and `el_2: SPAN`,
backed by an ordinary numpy struct dtype `[('el_1','i8'),('el_2_addr','i8'),
('el_2_len','i8')]` (24 bytes) -- the same `(address, byte length)` pair
`Raw[bytes]` and `SpanArray` already use, just at a fixed offset inside a row
instead of its own array. `.dtype` is overridden to return that backing dtype;
`register_model(SpanRecord)(models.RecordModel)` gives it numpy's existing
`RecordModel` (a `CompositeModel` over a raw byte buffer -- it never touches
`.dtype`, only `fe_type.members`/`fe_type.size`).

**What it depends on, read straight from the installed numba source
(`.venv/.../numba/core/types/npytypes.py`, `numba/np/numpy_support.py`,
`numba/core/boxing.py`, `numba/np/arrayobj.py`):**

- `Record.__init__`'s last line is exactly `self.bitwidth = self.dtype.itemsize
  * 8`. Without the override, constructing the type raises immediately:

  ```
  NumbaNotImplementedError: span cannot be represented as a NumPy dtype
  ```

  (from `numpy_support.as_dtype`, which `Record.dtype`'s default implementation
  -- `as_struct_dtype` -- calls per field.) Reproduced live in
  `route1_record_subclass.py`.

- That is the **only** place inside `Record.__init__` that reads `.dtype`, and a
  repo-wide grep of `numba/core` and `numba/np` for `.bitwidth` turns up no other
  reader of a `Record`'s `bitwidth` anywhere in the tree -- it is written and
  never read again.

- The only other consumer of `Record.dtype` is `core/boxing.py`'s `box_record`:
  `c.pyapi.recreate_record(ptr, size, typ.dtype, c.env_manager)`, called only
  when a `Record` value is boxed back to a Python object. A step never does
  this for `Rows[Item]`: `notes/nested-item-fields.md` §3 already refuses `->
  Item` and `-> dict` outputs (`TypeError: nested objects are not allowed`), so
  the one call site the override could get wrong is one `Rows[...]` already
  can't reach.

- Attribute access, iteration, and array-level access on a record **do not
  touch `.dtype` at all** -- `numpy/arrayobj.py`'s `record_getattr` and
  `array_record_getattr` read `typ.offset(attr)` / `typ.typeof(attr)`, i.e.
  `Record.fields`, set directly in `__init__` from the constructor arguments,
  not derived from `.dtype`. So the override's blast radius is exactly one
  property, read at exactly two call sites in numba's own source, one of which
  (`box_record`) is unreachable through this API.

- `register_model(SpanRecord)(models.RecordModel)` is not an internals hack:
  it is the documented, public way numba extension code gives a new type (or a
  subclass of a builtin one) a data model. `RecordModel.__init__` builds its
  LLVM byte-array type from `fe_type.members`/`fe_type.size`, never from
  `.dtype`.

**Verified, all in `route1_record_subclass.py`:**

- `typeof()` on the array: `array(Record(el_1[type=int64;offset=0],
  el_2[type=span;offset=8];24;True), 1d, C)`.
- `for i in items: if i.el_1 == 400 and i.el_2 == "snoop": return True` --
  correct.
- `items[j].el_1` / `items[j].el_2 == "snoop"` -- correct (today's positional
  spelling still works against a record).
- **`items.el_2[j] == "snoop"` -- today's exact `Rows[Item]` spelling for a
  `str` field -- keeps working with zero extra code**, because `el_2` is a
  real field of `SpanRecord.fields`, so numba's own `array_record_getattr`
  handles it: no glue needed for the array-level spelling.
- Removing the `.dtype` override reproduces the exact `NumbaNotImplementedError`
  above.

**Full CPython parity, `route2_full_ops.py` (shares the same span machinery, so
this bar applies to route 1 identically -- see below):** `==`, `len`,
`startswith`, `endswith`, `in`, over ASCII / empty / 2- / 3- / 4-byte UTF-8
(café, 日本語) and a null span (length -1), 105 assertions, all exact against
CPython. The runtime-`str`-param comparison raises the same documented
`TypingError` `Rows[Item]`'s shipped `SpanArray` field already raises (loud, not
silently wrong) -- nothing about being inside a record changes that rule,
because it's the same `SpanType` overloads in `span.py`, untouched.

**Verdict on route 1: not fragile.** The note's fear was specific and testable,
and it does not hold up: the override touches one property, two call sites, one
of them dead for this feature, and the mechanism it uses
(`register_model` for a `Type` subclass) is numba's sanctioned extension route,
not a private one.

## Route 2: two plain int64 fields, span assembled on attribute access

Built the backing array with an **ordinary** numpy struct dtype and no
`Record` subclass at all -- `[('el_2_addr','i8'), ('el_2_len','i8')]`. `typeof()`
on a plain `np.ndarray` (or a bare subclass, no `typeof_impl` needed) already
gives a stock `types.Record` with those two fields; no numba subclassing, no
`.dtype` override, ordinary numpy end to end.

To reach `i.el_2`, registered a second `AttributeTemplate` (`@infer_getattr`)
on `types.Record` whose `generic_resolve` returns `None` for a real field name
(deferring to numba's own `RecordAttribute`) and synthesizes `SPAN` when
`f"{attr}_addr"`/`f"{attr}_len"` both exist, plus a matching `lower_getattr`
that loads both int64s and packs them into a span via a small `@intrinsic`.

**This is genuinely fewer internals than route 1** -- no `Record` subclass, no
`.dtype` override, no new numpy dtype shape. It works because `@infer_getattr`
templates are tried in an order (`ctx._get_attribute_templates`) where the
newly-registered template runs **before** numba's builtin `RecordAttribute`
(verified directly: `ctx._get_attribute_templates(rectype)` lists
`SynthSpanAttribute` first), so returning `None` for a real field name is what
lets numba's own resolver take over -- there is no risk of the builtin's
`record.typeof(attr)` raising `KeyError` first. Full CPython parity verified
the same way as route 1 (`route2_full_ops.py`, 105 assertions, null spans,
multi-byte UTF-8, `==`/`len`/`startswith`/`endswith`/`in`).

**But it does not reach the superset property for free.** `items.el_2[j]` --
today's array-level spelling -- fails on route 2 unmodified:

```
TypingError: Unknown attribute 'el_2' of type
array(Record(el_2_addr[...], el_2_len[...];16;True), 1d, C)
```

because `el_2` genuinely isn't a field of the record; only the per-item
`i.el_2` path was given synthesis, and `array_record_getattr` (the array-level
attribute resolver) reads `.fields` directly, same as before. Matching route
1's superset guarantee would need a **second** override, on `types.Array`'s own
`generic_resolve` for record dtypes, mirroring the first -- i.e. the same
technique twice, at two extension points, instead of once. Route 2 is fewer
internals per override but more overrides, and without the second one it is
close to a no-go on its own terms (§4 of the brief): a `str` field would stay
positional while numeric fields become attributes, which is the readability
regression the whole feature exists to avoid.

**Verdict on route 2: viable, not preferred.** It avoids a `Record` subclass,
but reaching full parity with route 1 costs the same class of work twice over.
Route 1 is the smaller diff for the same guarantee.

## Route 3: hybrid (record for numbers, `SpanArray` for `str`)

Not prototyped -- it works by construction, since it needs nothing route 1
hasn't already proven for the numeric side. But the brief already answered its
own question: a hybrid means `i.el_1` (attribute) next to `items.el_2[j]`
(positional) in the *same* step body, which is a worse API than today's fully
positional one, not a better one. **No-go on its own**, independent of any
numba concern.

## Route 4: keep the namedtuple, add only iteration -- tried, more fragile

The brief asks, for the no-go branch, whether iteration alone can be bolted
onto today's physical shape (arrays-of-per-field, i.e. `rows_class(schema)`'s
namedtuple) with no new physical shape at all. Tried directly against
decider's own `rows.py` (`rows_class`, `build_rows` -- real code, not a
reimplementation), in `route4_iterator_view.py`.

The attempt: keep `items` exactly as `build_rows` builds it today; add a
schema-specific "item view" type (`(parent tuple, index)`, zero-copy, `.el_1`/
`.el_2` index the parent's arrays lazily) and give the parent namedtuple type a
custom `getiter`/`iternext` so `for i in items:` yields one view per index.

This needed real internals: a hand-written `IteratorType`/`SimpleIteratorType`
subclass, a `StructModel` with an `EphemeralPointer` loop counter,
`lower_builtin("getiter", ...)`/`lower_builtin("iternext", ...)` with
`@iternext_impl(RefType.BORROWED)`, and manual `StructProxy` construction of
the yielded value inside the lowering code -- markedly more low-level surface
than route 1's single property override, none of it going through a
pre-built, hardened numba model the way `RecordModel` is. First working
attempt (typing and registration both succeeded cleanly, no conflict with
numba's own `getiter` on `types.NamedTuple`) crashed the process on execution:

```
malloc(): unaligned tcache chunk detected
```

-- a native heap corruption from the lowering code (most likely a refcounting
or struct-layout mistake in the yielded `ItemView`, which itself holds array
references that need NRT tracking). Not debugged further: the point this
route needed to prove -- that hand-rolled iteration is the *cheaper*, safer
alternative to a record -- is already disproven by the fact that a first-pass
implementation corrupts memory, where route 1's first-pass implementation
worked correctly and stayed inside numba's own tested `Record`/`RecordModel`
machinery throughout.

**Verdict on route 4: more fragile than route 1, not cheaper.** Answers the
brief's no-go question directly: no, there is no cheaper way to get `for i in
items:` than the record shape; the alternative touches deeper, less-guarded
internals (manual iterator/refcounting lowering) for the same result.

## The finding neither note anticipated: the kernel path changed the cost

`notes/nested-data-decisions.md`'s "Open" section re-measured the record
shape's *performance* case after `notes/nested-rows-in-kernel-landed.md`
landed, and concluded the perf argument no longer holds -- so this brief scoped
the question down to pure robustness. But re-reading `kernel.py` for this note
turned up a scope gap in that re-measurement.

`decider/engine/compile/kernel.py`'s `_row_body`/`rag()` (lines ~152-163)
constructs a compiled step's `items` value **inside the kernel**, once per
parent row, by slicing each flat field array `lo[i]:hi[i]` and packing the
resulting **array views** into `types.NamedTuple(fields, rows_class(schema))`
-- zero copy, exactly the win the landing note describes ("no new numba type
was needed... what disappears is building a namedtuple of views per row").

`decider/engine/run/runners/stepped.py`'s `_boxed` (lines 210-217) shows this
is not a loop-only path: `kernel = not isinstance(unit, Fallback)` -- so
**any** compiled (non-`Fallback`, non-per-row) `Rows[Item]` step, including a
plain top-level `score()` on one row, goes through `Ragged`/`rag()`, not
through `build_rows`'s namedtuple-of-views-per-item. `build_rows`'s `("rows",
schema)` shape is reached only for a genuinely interpreted `Fallback` and for
a `row`-kind per-row dispatcher.

A record-per-item shape needs one interleaved buffer per row (address, offset
in that buffer, and value all fixed relative to each other); the flat
per-field arrays `rag()` slices are separate Arrow-backed buffers with
independent addresses, so there is no strided view that presents them as one
record array -- getting a record inside the kernel means **copying** each
field's value into a freshly built row-sized record buffer, once per parent
row, inside the compiled kernel.

`notes/nested-item-fields.md`'s own numbers already show this copy is not
free at scale: its part-3 "build" column (measured on the *pre-landing*
per-row-dispatcher architecture) shows records winning 3-4x at 3-30 items but
losing badly at 3000 items/row, k=8 (**370.55 us vs 3.84** for the namedtuple).
That table was never re-measured against `rag()`'s construction, because the
record shape was never wired into it -- the "Open" section's re-measurement
checked whether the *old* per-row-dispatcher argument still applied (it
doesn't) but did not check the *new* kernel-inlined construction's cost,
because that construction didn't need to build a record at all until now.

For decider's stated target (`notes/nested-data-decisions.md`: "line-item
lists, single digits to low hundreds"), this copy is almost certainly cheap --
a few dozen bytes moved once per parent row -- but "almost certainly" is not
"measured," and the existing numbers cannot be reused the way the "Open"
section reused them for the positional-argument case, because they describe a
different construction site.

## Go/no-go

**Go on the robustness question the note raised: a numpy/numba record can
carry a `SPAN` field robustly.** Route 1 (a `Record` subclass overriding
`.dtype`, plus the standard `register_model`) is not fragile: it bends exactly
one property, consumed at two call sites in numba's own source (one of them
already unreachable through this API), using numba's sanctioned type-extension
mechanism throughout. `str` field behaviour is identical to today's
`SpanArray` field -- same overloads, same null rule, same UTF-8 exactness,
same loud refusal for a runtime-built comparison string -- because it's
*literally* the same `SpanType`, just at a fixed record offset instead of its
own array. The cheaper alternative (iterate the namedtuple as-is) was tried
and is *more* fragile, not less, so if the record route were somehow blocked,
the namedtuple should stay exactly as it is rather than reach for hand-rolled
iteration.

**Not yet a go on shipping the physical-shape swap for `Rows[Item]`.** Doing
that for real needs to reach the primary compiled path (`kernel.py`'s
`rag()`), which the landed shared-array-kernel design built specifically to
avoid a per-row copy -- and giving it record semantics reintroduces exactly the
copy it removed, at a cost this note cannot respons­ibly claim is "zero" from
existing numbers, because those numbers predate the construction site that
would now pay it. That is a small, well-scoped follow-up (benchmark the
interleave copy inside `rag()`, using decision 2's own methodology --
`Kernel.run` level, loaded box, ratios not absolutes -- at the item counts
`nested-item-fields.md` already flagged as the crossover), not a reason to
distrust the mechanism itself.

**Recommended next step, if this is picked back up:** measure that one number
first. If it comes back near-zero at "single digits to low hundreds" items
(as every other number in this note suggests it will), route 1 is a clean,
low-risk swap: one new module (`SpanRecord` + a record builder mirroring
`build_rows`'s null/Arrow handling), one line in `njit._input_type`, one
`kind` in `stepped.py`/`interpreted.py`, and `Ragged`/`kernel.py`'s `rag()`
updated to build a record buffer instead of a namedtuple-of-views. If it comes
back non-trivial at the low end of that range, keep the namedtuple: the
readability win was already established as the *only* remaining argument for
this change, and it should not cost single-record latency to get it.
