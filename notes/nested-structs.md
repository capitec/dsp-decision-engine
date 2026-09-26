# A `dict` input as an Arrow struct, and a `dict` whose fields are lists

## What the layout actually is (verified, polars 1.x / nanoarrow shim)

- `pl.Struct({...})` exports as one Arrow `struct` child with **one flat child array
  per field**, same shape and dtype as an ordinary column: field `k` of row `i` is
  `child_k[i]`. Confirmed through `sm_view_child` on the frame view's child.
- The struct has **its own validity bitmap**, and polars **also pushes a null struct
  down into every child**: for `[{income: 1.0, age: 30}, {income: 2.0, age: None}, None]`
  the struct view reports 1 null, child `income` 1 and child `age` 2. So reading a
  field's validity is enough; the struct's own bitmap only tells you *why* a field is
  null. Both `Series.struct.unnest()` and `struct.field()` propagate it the same way.
- A `pl.List` field is a `large_list` child (int64 offsets) whose own child holds the
  flat values; the values child holds only the rows' items (a null struct contributes
  none). Same offsets+values shape `rows.py` and `param_table` already use.
- The shim reads **top-level children only**: `FramePlan.child_idx` indexes
  `frame->children`, and `sm_resolve_col` rejects `struct` and `large_list` outright
  (`ArrowKindError`, with the "split the kernel around them" hint). `sm_view_child`
  can already walk into a struct, but nothing resolves a grandchild.
- `Series.struct.unnest()` costs **1.9 µs flat, at n=1 and at n=200k** — it hands back
  the child Series polars already holds, no copy. Everything else is far worse:
  `df.select(pl.col(x).struct.unnest())` is 150-440 µs (expression engine),
  `struct.field()` is ~38 µs per field, `to_list()` is 592 ns **per row**.

So a struct column needs no new C: `Series.struct.unnest()` turns it into a frame of
flat columns, and `extract_frame` reads those children through the existing shim, with
the casts, validity, fills and error messages an ordinary column gets.

## What a struct input does today

`dict` and a bare `TypedDict` both work in all three modes; in `stepped` and `fused`
they fall back to Python per row with `reads 'applicant' as <class 'dict'>, which no
kernel takes`. That behaviour is unchanged.

## The API: `Struct[Item]`, not a bare `TypedDict`

```python
class Applicant(TypedDict):     # module level, like Rows[Item]'s item
    income: float
    age: int

@step
def afford(applicant: Struct[Applicant]) -> float:
    return applicant["income"] * 2.0 + applicant["age"]
```

`Struct[Applicant]` (a marker beside `Raw[T]` and `Rows[Item]`, whose `Item` is
declared the same way `Rows[Item]`'s is) reaches a kernel as one numpy record per row.
A bare `TypedDict` was rejected as the opt-in:

- A `TypedDict` is documentation, not a layout promise. It is not closed at runtime:
  a dict with extra keys satisfies it, and a real frame's struct often carries more
  fields than the step reads. "An Arrow struct with exactly these fields, every one a
  number, never null" is a much stronger claim than the annotation makes.
- It already works today and means "Python path". A body that does `.get()`,
  `len()`, `in`, iterates keys, mutates or passes the dict on would stop compiling and
  start warning, for an annotation added for documentation. Silent-ish and confusing.
- `dict` and `TypedDict` therefore keep exactly today's behaviour, and the opt-in is
  the one word the user writes when they want the kernel.

The record is read with `applicant["income"]` — the same expression the Python body
already uses, so one body serves every mode (`np.void` and numba's `Record` both
support string subscript; attribute access happens to work in compiled modes and is
not the documented form).

## Numbers (2026-09-26, loaded dev box, one process, `benchmarks/structs.py`)

A one-step pipeline `applicant["income"] * 2 + applicant["age"]`, 200k rows, and
`score()` over 6000 single-record calls per variant, **interleaved** — measuring one
variant's 6000 calls in a block gave a different winner run to run on a busy box, so
the benchmark now rotates through the variants inside one loop and every p50 drifts
together. `prepare ms` is `prepare(df)` alone, out of the same 200k-row run.

| variant | batch rows/s | prepare ms | score p50 | p99 | min |
|---|---|---|---|---|---|
| numeric kernel, flat columns (the ceiling) | 181M | 0.1 | 45.1 µs | 119 | **31.9** |
| `dict`, Python per row (today) | 0.46M | 141.7 | 56.4 µs | 143 | 39.9 |
| `dict`, `mode="interpreted"` | 0.35M | 144.8 | 58.0 µs | 136 | 41.3 |
| `Struct[Applicant]` record, in the kernel | 1.30M | 135.0 | 52.0 µs | 129 | 36.2 |
| `Struct[Applicant]` record, `mode="stepped"` | 1.50M | 132.5 | 46.7 µs | 122 | **32.4** |

Absolute latencies move with the load (the ceiling's p50 ranged 24-45 µs across runs);
the ordering did not. Unlike a `str` input there is **no crossover**: the record path is
4-5 µs *faster* than the Python dict path on a single record and within 1-4 µs of a flat
numeric kernel, so reading the Python dict directly never wins and there is one
implementation, not two.

The per-row costs, measured on their own in a quiet moment, are the reliable part:

| | ns/row |
|---|---|
| `State.from_frame`'s `to_list()` for a nested column (paid by every mode) | 592 |
| `build_struct` from Arrow (unnest + `extract_frame` + interleave copy + alloc) | 8.5 |
| the kernel itself over the record array | 1.3 |
| the same kernel over flat columns | 1.3 |

Two things follow. The interleaved record costs the kernel **nothing** (both fields
share a 16-byte cache line), so the zero-copy shape the brief imagined — a namedtuple
of scalars built in the kernel from k separate child arrays — would buy 2.7 ns/row and
cost a new source kind in `kernel.py`'s `load()` plus a version-to-many-columns change
in `Layout`/`Kernel.run`. Not worth it; `kernel.py` is untouched.

And batch stays a 3x win rather than a 50x one because `State.from_frame` builds a
Python dict per row for every nested column before any step runs: 592 ns/row against the
9.8 ns/row everything else costs, ~135 ms of each 200k-row run. Deferring that (the
`ponytail:` note already in `state.py` says the same for string columns) is the only
remaining batch win, it is a shared change to `State`, and it would help strings and
lists too.

The record dtype is built once per schema and is byte-identical run to run, so numba's
disk cache behaves exactly as for flat columns: in a second process the step body is
loaded from cache and only the shared kernel wrapper (a closure, never cached for any
step) recompiles.

## Errors

Each names the field. `input 'applicant.age' ...` is the `.` path into the column.

| case | before | now |
|---|---|---|
| null struct | `MissingInputError: input 'applicant' ... has 1 null row(s) of 2` | unchanged |
| the column is absent | `... is not in the input frame or record` | unchanged (the null policy still owns it) |
| null field (frame) | silently reached the body, then `TypeError` | `MissingInputError: input 'applicant.age' ... Fill the field in the frame, or declare 'applicant' as a plain dict.` |
| null field (record) | `TypeError: unsupported operand` in the body | same, named |
| field missing from the record | `KeyError: 'age'` | `input 'applicant' of step 'p/afford' has no field 'age' on row 0; the value has ['income']` |
| field missing from the column | `KeyError` per row | `declares field(s) ['age'] that column 'applicant' does not have; it has ['income']` |
| field the frame stores as a string | — | `ArrowKindError: column 'applicant.age' is Arrow string_view but is declared I64` |
| the column is not a struct at all | `TypeError: 'float' object is not subscriptable` | `is declared Struct[...], but column 'applicant' holds float64 values, not structs` |
| a `str`/`date`/`list` field declared | — | Python path, warning `reads field 'applicant.name' as <class 'str'>, which no kernel record holds` |
| `missing_as(...)` or `\| None` | — | Python path, warning `which a kernel record can't hold: it has no per-field validity` |

A record has no per-field validity, so a fill or an optional struct keeps the step on
the Python path it has today rather than inventing a per-field mask.

A **null struct** is an ordinary null column value: the reader's null policy owns it, on
the rows that reader actually runs on, so a branch arm that skips the null rows works
and every mode agrees. A **null field** is stricter — it raises if it appears anywhere
in the column, even on a row the step never runs on — because the record array carries
no per-field mask to defer with, and zero-filling a missing income silently is the one
outcome worth refusing. The message says which field and offers the plain `dict`.

Two limits inherited from rules that already exist:

- Two steps reading one struct column must declare the **same** `Item`; the wiring's
  one-type-per-column rule refuses `Struct[Applicant]` against `Struct[JustIncome]`
  with its usual message. Share the `TypedDict`; a step reading fewer fields costs
  nothing extra.
- The `Item` must be importable where the annotation is resolved (module level, not
  inside a function), like any annotation under `from __future__ import annotations`.

## What keeps the buffers alive

Nothing has to: `build_struct` copies each child into the record array, which owns its
memory (`records.base is None`). The borrowed views `extract_frame` hands out (for
n >= 4096) are consumed before it returns, and they carry their `_Export` owner
anyway. `state.representation`'s `alive` list stays empty, `full[rows]` for a branch or
loop subset is a fresh copy, and a `Session` override drops the cached representation
with the values it was built from and rebuilds from the new dicts.
`test_records_stay_valid_after_the_arrow_export_is_released` runs a 5000-row frame with
`gc.collect()` at every checkpoint after dropping the frame reference, and asserts the
records own their memory — it fails if someone later returns a borrowed view.

## `dict` with lists: wait

Measured on 200k rows of `{"rate": float, "amounts": [3 floats]}`, body only:

| shape | rows/s | 1 record |
|---|---|---|
| Python per row (today) | 3.34M | 0.52 µs |
| njit per row, namedtuple built in Python (what `Rows[Item]` does) | 0.35M | 2.84 µs |
| flat values + offsets built with polars (`explode` + `list.len`) | 39M | — |
| kernel with the namedtuple built **inside** the loop | 90M | — |

The `Rows[Item]` shape — one dispatcher call per row, the slice made in Python — is
**10x slower than Python** in batch and 5.5x slower on one record, exactly the lesson
`notes/strings.md` records for per-row dispatch. The only shape that pays builds the
namedtuple inside the kernel, which needs the `kernel.py` source kind this change
avoided, plus the deferred `to_list()` above; with today's `prepare` it would be a
1.4x end-to-end gain for a large chunk of machinery, and it can never help `score()`.

So a list field refuses to compile, keeps today's Python path, and says which field did
it. Revisit only after `State` stops materialising nested columns eagerly.

## Merge notes

- `kernel.py` and `rows.py` are untouched. `njit.py` gains a
  `_struct_reason` guard before the "odd input" check; `stepped.py`'s `_external`
  carries the struct schema so `_run` decides per read with one `is not None`;
  `exceptions.py`'s `MissingInputError` takes an optional `fix` sentence.
- `serving` learned that `Struct[Item]` is `Item`'s dict at the JSON boundary, so
  warm-up builds a real sample record for it and a `date` field inside it is coerced
  from its ISO string, exactly as for a bare `TypedDict` input. Without that, warm-up
  sent `1.0` for the whole struct and a date field raised a pydantic schema error.
- One real bug fixed on the way, in `units.py`: `Fallback.run` fed a step
  `values.tolist()`, which turns a record array into plain tuples, so a `Struct[...]`
  step that runs one compiled call per row (a `str` output, a `Rows[...]` input
  alongside) died in numba's frontend. It now hands out `list(values)` — `np.void`,
  which numba unboxes as a `Record` and Python reads by field name.

  **Why that doesn't break the plain-scalar rule** (a Python fallback must get real
  Python scalars, so `x / 0.0` raises as it does in a kernel instead of returning
  `inf`), because the next reader of that line will ask:

  1. `_per_row` diverges **only** for a record dtype. Every other column still goes
     through `.tolist()`: a float64 column yields `float`, an object column yields the
     `dict`s and `None`s it holds. Verified directly on the three dtypes.
  2. A record array only ever reaches `Fallback.run` when the fallback is backed by a
     **numba dispatcher**. `SteppedRunner._run` sets
     `python = isinstance(unit, Fallback) and not isinstance(unit.fn, Dispatcher)` and
     builds the struct representation only when `python` is false, so a genuinely
     interpreted step reads the object array of dicts and its fields are Python floats.
  3. On that dispatcher path the body is compiled, and numba's default error model is
     `python`, so it raises `ZeroDivisionError` too.

  So the only consumer of an `np.void` is compiled code, never a Python body.
  `test_dividing_by_zero_still_raises_on_the_per_row_path` pins it in all three modes;
  break either 1 or 2 and it fails.
- The "warn unless the step declares itself Python-only" knob belongs to the
  semantic-`str` work that owns `compile_call`'s reasons and `_compile`'s warning loop.
  A `Struct[...]` step that refuses to compile (a `str`/`list` field, a fill) produces
  one of those warnings and would be silenced by the same switch; no second knob here.
- `Struct[Item]` is exported from `decider` and covered by `tests/run/test_struct_columns.py`.

## Draft for `GUIDE.md` (both nested inputs)

`Rows[Item]` was never in the guide either, and `Raw[str]` being in `__all__` and absent
from it is how 24 example projects came to use none of them. This block belongs as its
own section, `## Nested inputs: a struct, a list of structs`, after "Branch, loop, frame
steps" and before "Trees, rule sets, tables, scorecards" — it needs nothing from the
sections after it, and the cost sentence below reads as a continuation of the
compiled-modes story that section starts.

Prose before the block: *A record's nested object becomes a `Struct[Item]` input, one
record per row; a list of objects becomes `Rows[Item]`, one array per field sliced to
that row. Both declare their fields with a `TypedDict`, and both fields must be `float`,
`int` or `bool`. A plain `dict` or `list` input still works and still runs in Python, row
by row — the annotations are how you ask for a kernel.*

```python
from typing import TypedDict

import polars as pl
from decider import Engine, Rows, Struct, flow

class Applicant(TypedDict):      # a struct column: one record per row, fields float/int/bool
    income: float
    dependants: int

class Account(TypedDict):        # one item of a list-of-structs column
    balance: float

def disposable(applicant: Struct[Applicant]) -> float:
    return applicant["income"] - applicant["dependants"] * 250.0

def owed(accounts: Rows[Account]) -> float:
    total = 0.0
    for j in range(len(accounts.balance)):       # arrays per field, this row's slice
        total += accounts.balance[j]
    return total

def approved(disposable: float, owed: float) -> bool:
    return owed < disposable

pipeline = flow(disposable, owed, approved, name="afford").emit("disposable", "owed")
exe = Engine().bind(pipeline, mode="fused")

record = {"applicant": {"income": 4100.0, "dependants": 2},
          "accounts": [{"balance": 300.0}, {"balance": 50.0}]}
assert exe.score(record) == {**record, "disposable": 3600.0, "owed": 350.0, "approved": True}

df = pl.DataFrame({"applicant": [{"income": 4100.0, "dependants": 2}],
                   "accounts": [[{"balance": 300.0}, {"balance": 50.0}]]})
assert exe.run(df)["approved"].to_list() == [True]

# A struct is flat, so it joins the shared array kernel; a list is ragged, so it can't.
assert "afford/disposable" not in exe.fallbacks()
assert "one call per row" in exe.fallbacks()["afford/owed"]
```

Prose after the block: *A struct is flat — field `k` of row `i` is just `child_k[i]` —
so a struct step joins the shared array kernel and costs what a step reading two plain
columns costs. A list is ragged, so a `Rows[Item]` step runs one compiled call per row
instead, which `fallbacks()` reports and which warns once; that is the price of the
loop, not a mistake. A field a kernel can't hold (a `str`, a date, a nested list), a
`missing_as` fill or a `| None` keeps the whole step in Python and says which field did
it.*

Checked: the block runs as `tests/test_guide.py` execs it (`exec(compile(block,
"GUIDE.md", "exec"), {"__name__": "guide"})`) with no fixture beyond the temporary
project, and prints nothing but the expected one-call-per-row `UserWarning` for
`afford/owed`.

That warning is the lesson, so it stays, and the accepted-fallback decorator goes on
`owed` to close the pair: the step still can't join the kernel, and the author says in
writing that the per-row call is the price. Both asserts are meant to survive that.
**The invariant they rest on:** `SteppedRunner._compile` records
`self._fallbacks[path] = unit.reason` for every `Fallback` *before* it decides whether
to warn, so `fallbacks()` reports the step whether or not anything was printed. A
decorator that only skips the `warnings.warn` (and satisfies `strict_compile`) keeps
`assert "one call per row" in exe.fallbacks()["afford/owed"]` true; one that also drops
the entry from `fallbacks()` turns this guide block red, and the assert should then
become `exe.fallbacks() == {}` with the teaching moved into the prose.
