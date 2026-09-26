# Single-record `(address, length)` spans: where the microseconds were

`notes/strings.md` measured a `Raw[bytes]` span as the fastest batch string
representation and the worst single record. Trees read spans, so a string-gated
tree `score()` paid it: 101-104 µs in `notes/benchmarks-vs-decider2.md`.

Two things cost the microseconds, and neither was the Arrow shim.

## Where it went

One process, CPython 3.14, dev box, 20k `score()` calls; a one-step
`Raw[bytes]` pipeline and the `tree_walk.py` string-gated tree.

`spans()` on one row, 19.8 µs in pieces:

| piece | µs |
|---|---|
| `str.encode` per value | 1.0 |
| `b"".join` + `np.frombuffer` | 1.1 |
| `lengths = np.array([...])` | 0.9 |
| `np.cumsum` / `np.maximum` starts | 6.2 |
| `buffer.ctypes.data` | 1.6 |
| `np.stack` | 7.6 |
| **whole** | **19.8** |

Four numpy calls to lay out a single pair of int64s, and `ndarray.ctypes.data`
costs a microsecond a call (`_arrow/plan.py` already knew that and takes its
addresses once).

The bigger half was not in `spans()` at all: the per-read annotation
introspection in `SteppedRunner._run`, which asks `rows_item`,
`base_annotation`, `representation_for` and `is_raw` about the same annotation
on every single call. `typing.get_origin`/`get_args` were the 3rd and 4th
hottest functions in the profile.

| the annotation calls of one read | uncached | cached |
|---|---|---|
| `Raw[bytes]` | 6.5 µs | 1.9 µs |
| `bytes \| None` (a tree's string feature) | 19.4 µs | 0.7 µs |

`bytes | None` is the expensive one: `is_raw` walks the union, and
`base_annotation` and `representation_for` each walk it again.

Not costs: `extract_frame` and the shim never run for a few rows (the `<= 32`
shortcut), `State.representation`/`source` bookkeeping is under 2 µs, and
`_bundle` already caches a node's converted `str` params per params document,
so a span param is built once, not per call.

## What changed

- `lru_cache` on `is_raw`, `raw_base`, `rows_item`, `representation_key`,
  `representation_for` (`decider/types.py`) and `base_annotation`
  (`engine/ir/decls.py`). A process sees a handful of annotations.
- `spans()`'s few-rows path borrows each value's own UTF-8 bytes instead of
  encoding into a fresh buffer: one `str.encode`, `PyBytes_AsString` through
  `ctypes.pythonapi` for its address, and one `np.array(flat).reshape(n, 2)`.
  No `cumsum`, no `stack`, no `ndarray.ctypes`.

## What keeps the borrowed memory alive

`_borrowed` holds every `bytes` it takes an address from in a list and appends
that list to `alive`. `State.representation` stores `alive` beside the span
array and drops both together, so the bytes outlive every kernel that reads the
spans:

- `score()` and `run()`: the `State` lives across the whole run.
- a branch or loop row subset: `full[rows]` copies the int64 pairs, and the
  addresses in them still point into the bytes the `State` holds.
- `State.write` (a loop body, a `Session.set` override) pops the cached
  representation, so the next read rebuilds the spans over the new values; the
  old pairs and the old bytes are dropped together.
- `State.restore` clears every representation, for the same reason.

The rule is the same one the whole-column path already relies on (`alive` holds
`ExtractedFrame.kernel_frame`): a span array is only valid while the `State`
that built it is. Keeping one past its `State` dangles, as it always did.

`tests/run/test_compiled_fallback.py::test_a_span_keeps_the_bytes_it_borrows_alive_while_a_kernel_reads_them`
pins it: it builds spans, drops every other reference, collects, then allocates
small `bytes` over the freed pool blocks and reads the spans back. Without
`alive.append(kept)` it reads `b'\x7f\x7f\x7f...'` instead of `b'priority'`.

## Before and after

`spans()` on one row: **19.8 → 2.6 µs**.

`score()` p50/p99, same process, the annotation cache and the span build
switched in turn (numeric-kernel ceiling 19.0 µs):

| pipeline | before | + annotation cache | + borrowed spans |
|---|---|---|---|
| one `Raw[bytes]` step | 70.6 / 138 | 64.0 / 79 | 31.0 / 53 |
| string-gated tree, fused | 144.8 / 188 | 121.5 / 362 | **91.4 / 131** |
| string-gated tree, stepped | 139.2 / 297 | 126.4 / 219 | **91.8 / 200** |
| numeric tree, fused (control) | 62.8 / 77 | 66.0 / 155 | 62.3 / 72 |

The string gate's surcharge over the same tree without it: 82.0 → 29.1 µs.
p99 moves ±50% run to run on this box, so read the control row first: it is
within 5% across the three variants here. p50 repeats within 2 µs.

## The threshold stays at 32

Borrowing is O(n) Python, so it only wins while n is small. Measured per call,
one string per row, against the old encode-to-one-buffer path and the Arrow
export (`source=None`, so the export includes building the polars Series):

| n | old | borrowed | Arrow |
|---|---|---|---|
| 1 | 19.8 | 2.7 | 49.7 |
| 8 | 21.9 | 9.8 | 48.9 |
| 32 | 27.9 | 30.0 | 50.8 |
| 128 | 51.5 | 114 | 53.0 |
| 1024 | 262 | 841 | 97.8 |

The existing `<= 32` cut is the crossover. Above it nothing changed: batch still
exports the frame's own buffers, 10.3M rows/s on a one-step `Raw[bytes]`
pipeline over 200k rows (`notes/strings.md` measured 7.0M on a busier box).

## `Annotated` metadata: the decl keeps it, type logic never sees it

Caching the annotation helpers raised the question of what an unhashable
annotation would do, and the answer was that it already crashed:
`Annotated[float, {"unit": "kg"}]` raised a bare `TypeError: unhashable type:
'dict'`, naming neither the step nor the annotation, in every mode at `d825f31`
(`declared_dtype` was already an `lru_cache`, and `feature_kind` looks the whole
annotation up in a dict). Two quieter symptoms came with it: *any* `Annotated`
input fell back to Python row by row ("reads 'loan' as
typing.Annotated[float, Money(...)], which no kernel takes"), and
`Annotated[float | None, ...]` was read as REQUIRED, because `nullable()` sees
`Annotated` and not the union. `fields.py` had promised the opposite all along:
"the engine ignores it and runs the plain type".

Stripping it in `harvest()` fixes all three and is wrong: the flow debuggers
read metadata off the **decls**, not off the function signature
(`tools/decider-bridge/.../describing.py` walks `CallNode.inputs`/`outputs`/`params`
and calls `metadata_of(d.annotation)`), so a decl that has been stripped makes
the debugger guess units from names for every input and output. `pytest` does not
catch it, because `testpaths = ["tests"]` skips `tools/`. Run the bridge suite
for anything that touches a declaration:

```sh
cd tools/decider-bridge && uv run --with pytest --with pytest-asyncio pytest -q tests
```

So the decl keeps exactly what the caller declared, and the strip happens at the
type-logic funnels instead: `annotation_cache` in `decider/types.py` wraps each
cached helper in a thin uncached `plain_annotation` (`get_origin(a) is Annotated`
-> `a.__origin__`), so metadata never becomes a cache key, and `feature_kind`
and `nullable` strip before their own dict lookup and union test.

Two things the shape needs:

- **One spelling to look for.** `Optional[Annotated[T, M]]` hides the metadata
  from `plain_annotation` *and* from `metadata_of`, and its union hashes its
  `Annotated` member, so it crashed even with the funnels stripping. Normalising
  it per call costs 12 µs on the string-gated tree's `score()` — the union test
  needs `get_args`, on the hot path, forever. `harvest()` does it once per
  declaration instead (`hoist_metadata`: `Annotated[T | None, M]`), which also
  makes `metadata_of` find metadata in that spelling for the first time.
- **A param keeps the spelling it was given.** Its annotation goes to pydantic,
  and `Annotated[float | None, AfterValidator(f)]` checks something different
  from `Optional[Annotated[float, AfterValidator(f)]]` — the first sees `None`,
  the second does not. Param annotations reach no cache key (measured: an
  unhashable one runs clean in every mode), so they are left alone.

Cost of stripping at the funnels rather than at the decl, per string read of one
`score()` (five helper calls, one `bytes | None` annotation):

| | per read |
|---|---|
| uncached, at `d825f31` | 7.5-19.4 µs |
| cached, annotation already plain | 0.7 µs |
| cached, wrapper strips (shipped) | 3.8 µs |

The 3.1 µs is five `plain_annotation` calls on a cache hit. Carrying the
metadata in a field on `Input`/`Output` would recover it, at the price of an IR
field and a line in the bridge; so would the `_external` precompute below, which
removes the five calls altogether.

## Left on the table

`_run` still asks `rows_item`/`base_annotation`/`representation_for` per call,
now for the price of a cache lookup and a strip (3.8 µs a read). Precomputing the
representation kind per `(unit, decl)` in `_external`, where the rest of the read
plan already lives, would remove it, and `ctypes.pythonapi.PyBytes_AsString`
(0.54 µs) could become `id(b) + bytes.__basicsize__ - 1` (0.21 µs) if 0.3 µs a
value ever justifies betting on the object layout. Neither was worth the diff.
