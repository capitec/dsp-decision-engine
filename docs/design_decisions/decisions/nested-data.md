# Nested data

Read this before changing how lists of records, structs or strings inside records reach a compiled step (`Columnar[Item]`, `Struct[Item]`, `Text`).

## An annotation names meaning; the plain type always works

- `list[dict]`, `list[Item]`, `dict` and a `TypedDict` run in Python with the same answers in every mode; `Text`, `Columnar`, `Struct` are the kernel opt-ins.
- A fallback step gets the same value as a compiled one (`Text`, namedtuple of arrays, record), so `assert_equivalent` can compare modes.
- Why: the best storage depends on row count (int codes won 4.7x in batch, lost 2x on one record), and a code-typed step that fell back compared a `str` to an int and silently answered wrong.
- Not: `Raw[T]`, `Rows[Item]`, `raw_str()` (process-order codes), a `@step(arrays=True)` flag, `Enum` codes for speed.
- See: `decikit/ops/nested.py`, `decikit/ops/text.py`, [kernels](../../architecture/engine/kernels.md)

## A `TypedDict` is the schema; a bare `dict` is the opt-out

- `Item` is a `TypedDict` for `Columnar`, `Struct`, `optimise(record=)` and plain nested data; it must be module-level.
- A `TypedDict` is coerced at the JSON boundary (ISO dates inside it become dates); a bare `dict` is passed through untouched.
- Why: four example projects typed items as `TypedDict` only to get dates coerced; one concept should be the whole schema.
- See: [serving](../../architecture/serving.md)

## `Columnar[Item]` is flat field arrays plus per-row `lo`/`hi`

- `Ragged` (`compile/rows.py`) holds the unsliced field arrays and `lo[i]`, `hi[i]`; the step gets a namedtuple of views, `items.price[j]`.
- Why `lo`/`hi`, not `offsets[n+1]`: a branch or loop row subset is two fancy indexes with no child copy, and null-error indexes stay in whole-column coordinates.
- It costs ~20 us a `score()`; a short list with a light body is faster as `list[dict]`.
- See: [experiment](../experiments/columnar-arrow-read-and-shared-kernel.md), [compile](../../architecture/engine/compile.md)

## A `Columnar[Item]` step joins the shared kernel

- The arrays are kernel arguments (`flats`) and `rag()` in `compile/kernel.py` slices the row's namedtuple inside the kernel; no new numba type.
- Why: a `loop` over it packs; a 9-item bundle search fell 60.6 ms to 0.10 ms, 1.07x of hand-written `@njit` at 16 items.
- Per-row dispatcher instead: `Columnar | None` (no in-band `None`) and row-node inputs. Type the views from the arrays, not a probe (Arrow arrays are read-only).

## Arrow read from 256 rows

- `ARROW_ROWS = 256`: a frame-held `list<struct>` column becomes views over Arrow's offsets and child arrays; below that, or for computed values, arrays come from Python values.
- Why: the read has a ~130 us fixed cost; `score()` has no Arrow source so it is always Python.
- A layout it can't read (`Null`-typed field, null item, `pl.Array`) falls back to the Python read, which raises naming the field.
- See: [boundary](../../architecture/engine/boundary.md), `boundary/_arrow/nested.py`

## Item fields and nulls

- Fields: `float`, `int`, `bool`, `str`, `float | None` (null is NaN), `str | None` (null span); `int | None`, `bool | None` and list/struct fields are `TypeError` at bind.
- Any other null raises naming field, item and row; a real NaN is not a null; null or absent list is no items, `missing_as([])` the only fill.
- Why raise: the earlier silent NaN summed into a plausible wrong number on a money path.
- See: `compile/rows.py` (`_field`, `null_item`)

## A `str` item field is an array of spans

- `SpanArray` of `(address, length)` pairs types as `Array(SPAN, 1, "C")` with one `typeof_impl`; elements are `Text` in Python, so one value serves every mode.
- Why: the Arrow child already holds the strings: 43-97 ns an item, nothing copied. polars gives `string_view` (short values inline); index by window start, not `child.offset + start`.
- Comparison with a `str` `param()` needs `_has_span` in the probe and `_reads_span` in the bundle conversion; `build_rows` uses the unbound C `__getitem__` (Python's cost 1.1 us a row).
- See: `compile/span.py`, [experiment](../experiments/item-str-fields-and-record-shape.md)

## No record per item

- `for i in items` iterates the *fields* (silently, if all dtypes match), so index by position; a step returns an index or one field, not an item.
- Why: a record needs one interleaved buffer per row, so `rag()` would copy: 33.9x (151 us) at 3000 items x 8 fields; the per-call win only existed against the old per-row dispatcher.
- If revisited: a `types.Record` subclass overriding `dtype` + `register_model(RecordModel)` carries a span soundly.
- Not: hand-rolled namedtuple iteration (crashed the process), `np.recarray` (5.6 us a slice).

## `Struct[Item]` is a numpy record array

- Aligned `struct_dtype`, read as `applicant["income"]` in Python and kernel; fields `float`/`int`/`bool`; built via `Series.struct.unnest()` + `extract_frame` (1.9 us at any size).
- Why: interleaving costs the kernel nothing (1.3 ns/row) and it beats the Python dict even on one record (52 vs 56 us); a namedtuple of scalars saves 2.7 ns for a new source kind.
- Required only (no per-field validity); a null field raises even on a row the step skips; a null struct follows the null policy; readers share one `Item`.
- Not: a bare `TypedDict` as the opt-in (it means "Python path"; extra keys satisfy it).
- See: [experiment](../experiments/struct-input-as-record-array.md), `compile/structs.py`

## Outputs compile when the step returns tuples

- `Struct[Item]` output: return a tuple of fields; a dict or `Struct | None` runs in Python with a reason. A `Struct` can be a loop carry, which gives `optimise(record=Item)`.
- `Columnar[Item]` output: return `list[tuple]` (fields `float`/`int`/`bool`); it drains into a `Sink`, a doubling float64 buffer shared by all schemas (`SrcKind.SINK`).
- Why tuples: numba can't type a returned dict as a record. Declared dtypes (`pl.Struct`, `pl.List(pl.Struct)`) keep empty frames typed. No `Columnar` carry.
- See: `compile/sink.py`, `steps/_optimise.py`, [compile](../../architecture/engine/compile.md)

## No list fields inside a struct or item

- Refused at bind; use `dict` / `list[dict]` in Python. Why: per-row dispatch of a `{rate, amounts}` dict measured 10x slower than Python in batch.
- Open: `State.from_frame` still turns every nested column into Python objects first (592 ns/row), capping `Struct` batch gains at ~3x (`ponytail:` in `run/state.py`).
