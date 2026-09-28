# Struct / Columnar as outputs (gap 4, step 2)

Status of making `Struct[Item]` and `Columnar[Item]` writable, so a step returns
a record or a list of records and `optimise` can carry the winner's record back.

## Done

- **`optimise`** (step 1, committed): the fan-out/reduce construct, lowering to a
  loop, with a single scalar `score`, score-then-disqualify, and the
  `best_index`/`best_score`/`evaluated`/`disqualified` outputs.
- **Output dtype declaration** (committed): `declared_dtype` maps `Struct[Item]`
  to `pl.Struct(...)` and `Columnar[Item]` to `pl.List(pl.Struct(...))`, and
  `_series` applies it, so a step returning either keeps its shape on empty
  rows instead of inferring `Null`. A `Struct[Item]`/`Columnar[Item]` *output*
  already runs today — in Python, row by row, with correct results.
- **Compiled struct output** (decision (a), the tuple return): a step
  `return rate, code` for a `Struct[Product]` output now compiles and fuses.
  `output_dtype(Struct[Item])` is a numpy structured dtype; the kernel packs a
  tuple return into a record scalar (`build_record`) and stores it; `_series`
  and `score` materialise the record as a dict. A dict return stays on the
  Python path (a clear fallback reason). `Struct[Item] | None` stays Python
  too (no record-with-validity-mask kernel store yet).
- **Struct loop carry**: a `Struct[Item]` can be a `loop` carry and packs into
  one kernel. The carry slot holds the record's *bytes* (not its pointer), so
  a record kept across iterations stays valid.
- **`optimise(record=Item)`**: carries the winner's record back. The evaluate
  step also writes `record: Struct[Item]`; `optimise` emits `record` as
  `Struct[Item] | None` (`None` when no candidate survived — gap 4 point 6).
  The keep-record step is a separate single-record step (a tuple-with-record
  return is not compilable), and the finalise step runs in Python.

## Columnar output (decision (d), done)

A `Columnar[Item]` output is a variable-length list per row, which no
fixed-stride kernel column stores — the concern that kept this Python-only.
Shipped shape: a **growable buffer, one pass, no double-invoking the step**.

- `decider/engine/compile/sink.py`'s `Sink` is a jitclass wrapping one
  `(rows, n_fields)` float64 buffer (every field widens to float64; a bool or
  int narrows back once its row is sliced out, in Python) and a `length`.
  `push(item)` amortized-doubles it, `literal_unroll` reading `item`'s
  fields positionally since numba has no runtime index into a heterogeneous
  tuple. One `Sink` shape serves every `Item` schema — no per-schema jitclass.
- The compiled spelling mirrors `Struct[Item]`'s: a step returns
  `list[tuple(...)]`, not `list[dict]` (dict stays a `FallbackWarning`, same
  message shape as the struct case: "return a list of tuples to compile it").
  `njit.py`'s `_ragged_reason` rejects `Columnar[Item] | None` and a field
  outside float/int/bool the same way `_struct_reason` does for inputs.
- `njit.py`'s `_ragged_wrapper(dispatcher)` calls the step once and drains its
  returned list into the sink, returning the new cumulative length — an
  ordinary int64 scalar, so `kernel.py`'s existing `store()` needs no new
  code at all. The wrapper's own signature is fixed arity `(packed, sink)`,
  not `*args`: `kernel.py`'s hand-rolled IR calls a dispatcher with one typed
  value per argument, which resolves against a `*args` signature's single
  packed-tuple type instead of the real arg list — a mismatch only visible
  from *inside* a fused kernel, never from ordinary compiled call sites (a
  plain njit function calling a `*args` one works fine). `units.py`'s
  `Layout.spec` packs the step's own args into one `SrcKind.ROW` tuple (the
  same mechanism a row node already uses) so the wrapper sees exactly two
  arguments.
- `kernel.py` gained one new source kind, `SrcKind.SINK`: the sink is built
  once per `Kernel.run` (not once per row) and threaded through like `flats`
  or `params`, never indexed by row.
- `units.py`'s `Kernel.run` reads the wrapper's per-row cumulative length back
  as an ordinary output column, turns it into per-row `[lo, hi)` slices of the
  sink's buffer, and materialises `list[dict]` per row — the exact shape the
  Python-fallback path already produces, so `state.py` needed no changes.
- `_split_at_nulls` treats a `Columnar[Item]` output the same way it already
  treats a `Struct[Item]` one: a reader always starts a new kernel, since
  what the kernel produces (a length) isn't the value a reader wants.

Not done: a `Columnar[Item]` carry (an `optimise`/`loop` carrying a list
across iterations) — the sink is a `Kernel.run`-lifetime object, not a
carry-lifetime one; carrying one across iterations would need it seeded
once outside the loop and never treated as a per-iteration variable.

## The one decision blocking the rest

A `Struct[Item]` output runs in Python because **numba cannot type the value a
step returns**. The natural spelling returns a dict:

```python
@step(output="offer")
def best(rate: float, code: int) -> Struct[Product]:
    return {"rate": rate, "code": code}   # a dict: no kernel can hold it
```

For a compiled (fused) struct output, the step must return something numba can
type — a tuple of the fields, or a namedtuple:

```python
@step(output="offer")
def best(rate: float, code: int) -> Struct[Product]:
    return rate, code                      # a tuple: packs into a record scalar
```

The choice:

- **(a) tuple / namedtuple return** — compilable, so a struct output fuses and a
  struct can be a loop carry (what `optimise` needs to hand the winner's record
  back without re-evaluating). Costs: the user returns a tuple, not a dict, and
  dict-spelled steps stay on the Python path. `Item` for an *output* would
  probably want to be a `NamedTuple` rather than a `TypedDict`.
- **(b) keep dict return** — the user-friendly spelling, but struct outputs stay
  Python fallback, and `optimise` cannot carry the record without forcing the
  whole 2^n loop into Python.

Everything else is mechanical once (a) vs (b) is decided.

## The mechanical work (for the compiled path, decision (a))

1. `output_dtype` in `units.py`: `Struct[Item]` -> `struct_dtype(schema)` (a
   numpy structured dtype), `Columnar[Item]` -> object (or a ragged store; see 4).
2. `_struct_reason` in `njit.py`: stop rejecting a `Struct[...]` *output*; type
   the return as the record. `_probe_signature` returns input/param types, so the
   output type comes from the declared return annotation.
3. `kernel.py` `call`/`store`: a record scalar return, stored into the
   structured output array (the `from_dtype(struct_dtype)` type is an *array*,
   so the return needs the matching *scalar* record type, and `_cast` needs a
   record-scalar -> record-scalar path, not a cast to the array).
4. `Columnar[Item]` output: a variable-length list of records per row. No shared
   kernel can store that; the honest shape is a Python-fallback store (object
   dtype) or a `Ragged`-style offsets+child-buffers output. Likely Python-only
   for the first cut.
5. Loop carries of a record: `packed.py`/`resolve.py` carry machinery types
   carries by `output_dtype(v.annotation)`; a record dtype carries through a
   `Repeat` as a stack slot (a record scalar), which numba already handles.
6. `optimise` then carries `best: Struct[Item] | None`, seeded `None`, so empty
   reads as `None` (answers gap 4's point 6).
