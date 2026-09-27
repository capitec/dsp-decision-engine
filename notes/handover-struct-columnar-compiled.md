# Handover: compiled struct / columnar outputs (gap 4, step 2)

Next session: make `Struct[Item]` and `Columnar[Item]` **writable in the compiled
(fused) path**, then wire `optimise` to carry the winner's record back. Everything
before this is committed and green (full suite 1998 passed + 1 xfail).

## Where we are

Three commits landed, in order:

1. `feeed6b` — **`optimise`** (step 1): the fan-out/reduce construct. Lowered to a
   plain `loop` (no codegen), fuses into one kernel. See `decider/steps/optimise.py`.
   Outputs `best_index` (`-1` when nothing survives), `best_score`, `evaluated`,
   `disqualified`. Score-then-disqualify; `-inf` scores never win.
2. `b42de75` — **struct/columnar output dtypes** (step 2a): `declared_dtype` in
   `decider/engine/run/state.py` now maps `Struct[Item]` -> `pl.Struct(...)` and
   `Columnar[Item]` -> `pl.List(pl.Struct(...))`, and `_series` applies it. So a
   step that *returns* a struct/list-of-records (dict spelling) already works —
   in Python, row by row — and keeps its shape on empty rows.
3. `e5bafdf` — `notes/struct-columnar-outputs.md`, the design note. Read it first;
   it has the return-representation decision and the mechanical checklist.

## Goal (next session, in slices)

A. **Compiled struct output** — a step `return rate, code` (tuple) for a
   `Struct[Product]` output fuses instead of falling back.
B. **Struct loop carry** — a `Struct[Item]` can be a `loop`/`optimise` carry.
C. **`optimise` returns the winner's record** — carry the best record as a struct,
   seeded `None` (answers gap 4 point 6: `None` when nothing survives).
D. **Columnar output** — likely stays Python-only for the first cut: a
   variable-length list per row has no shared-kernel shape.

Decision already made: **"both" return spellings.** A dict return runs in Python
(current behaviour, keep it); a tuple/namedtuple return compiles. numba decides at
compile time by what the function actually returns — so the compiled path is about
*letting* a tuple-returning step compile and teaching the kernel to store it.

## Architecture map (read these before touching anything)

- `decider/engine/compile/structs.py` — `Struct[Item]` **input**: `struct_dtype(schema)`
  (a numpy structured dtype), `struct_schema`, `build_struct`. Fields are
  float/int/bool only (`bad_field`). Input is a record *array* (one record per row).
- `decider/engine/compile/rows.py` — `Columnar[Item]` **input**: `rows_class`,
  `build_rows` (per-row namedtuples), `build_ragged`/`Ragged` (flat arrays +
  lo/hi, the shared-kernel shape).
- `decider/engine/compile/units.py` — the compile unit. `output_dtype(annotation)`
  (line ~231) is the dtype a kernel stores an output in; today float/int/bool/Literal,
  everything else object. `Layout.spec` (line ~338) builds each call's `Spec`.
  `_split_at_nulls` (line ~216) splits a kernel so a nullable value is never written
  *and* read inside one kernel — the precedent to reuse for structs. `Fallback`
  (line ~89) runs a non-compilable call in Python, object dtype.
- `decider/engine/compile/kernel.py` — the fused kernel. `Spec` (line ~20) is one
  call; `fused_kernel`/`_row_body` lower it to LLVM; `call` (line ~192) invokes the
  step and stores results as SSA values; `store` (line ~260) writes the final
  outputs into `outs[k]`; `_cast` (line ~284) casts values to their dtype.
- `decider/engine/compile/njit.py` — `compile_call` (line ~92) decides compile-vs-
  fallback. `_struct_reason` (line ~161) currently **rejects** a `Struct[...]`
  output ("writes … as Struct[…], which no kernel stores") — this is the gate to
  open. `_input_type` (line ~277) types inputs (`Struct` -> `from_dtype(struct_dtype)`,
  a record array). `_probe_signature` (line ~304) types inputs+params only.
- `decider/engine/compile/packed.py` — fused branch/loop. `_pack` builds a `Repeat`
  for a loop; carries become kernel *variables* (`var`, line ~110). **Line 111:
  `if base_annotation(v.annotation) not in TYPED: raise _Unpackable`** is why a
  struct carry unpacks the loop today. `variables`/`value` (line ~115-119) type a
  carry by `output_dtype`.
- `decider/engine/run/state.py` — `declared_dtype`/`_item_dtype` (step 2a),
  `_series` (object -> polars), `State`. A compiled struct output will land here
  as a structured array that `_series` must turn into a `pl.Struct` column.
- `decider/types.py` — `Struct`, `Columnar`, `struct_item`, `columnar_item`,
  `item_schema` (declaration-order fields as `(name, type)`).
- `decider/engine/ir/decls.py` — `Output`, `base_annotation`, `nullable`, `TYPED`.
- `decider/engine/wiring/plan.py` — `Carry`, `Loop`, `Version` (the resolution the
  runners/compiler walk).

Tests to model on / keep green:
`tests/run/test_struct_outputs.py` (step 2a), `tests/run/test_struct_columns.py`,
`tests/run/test_columnar.py`, `tests/compile/test_compile_kernel.py`,
`tests/compile/test_compile_units.py`, `tests/run/test_optimise.py`.

## The critical numba facts (already worked out — don't re-derive)

- `out[i]['rate'] = r; out[i]['code'] = c` (field-by-field into a structured array)
  **works** in njit — this is the store mechanism.
- A namedtuple `out[i] = Product(r, c)` **fails** (no setitem impl for
  record-array <- namedtuple).
- `np.array((r, c), dtype=structured)` inside njit **fails** (numba's `np.dtype`
  can't take a list of 2-tuples in that form).
- A step `return rate, code` types as `Tuple(float64, int64)`.

So: the kernel cannot treat a struct output as one record scalar it casts to a
record array. It must treat it as a **packed multi-output**: the `Spec` marks the
output as a struct, `call` keeps the returned tuple, and `store` writes each tuple
element into the matching field of the structured array.

## Implementation plan (ordered)

1. **Open the gate.** In `njit.py` `_struct_reason`, stop returning a reason for a
   `Struct[...]` *output* (keep the input checks: bad field, optional-null). Then a
   tuple-returning `Struct` step's `dispatcher.compile` is what decides: dict ->
   fallback, tuple -> compiles.
2. **`output_dtype` for structs.** In `units.py`, return `struct_dtype(struct_schema(item))`
   for `Struct[Item]` outputs, so the kernel allocates a structured array. Add the
   structured array to `Kernel.writes`/`State.values` (a structured dtype is a
   normal numpy dtype there).
3. **Split kernels around struct outputs.** Mirror `_split_at_nulls`: a struct
   written by one call and read by another in the same kernel must split, so the
   reader sees the record *array* (its normal struct-input shape) in the next
   kernel. This dodges tuple-vs-record intermediate values entirely.
4. **Kernel store for structs.** In `kernel.py`, extend `Spec` (or add a flag) so a
   struct output's `call` keeps the tuple and `store` writes it field-by-field into
   `outs[k]` (the structured array). `Spec.dtypes` for a struct output is one
   structured dtype; the return is a tuple of the field dtypes.
5. **Materialization.** In `state.py` `_series`, a structured array -> `pl.Series`
   already yields a `Struct`; verify and adjust so a compiled struct output reads
   back as `Struct(...)` in every mode (assert against the Python/dict path).
6. **Carry.** In `packed.py`, allow a `Struct[Item]` carry: line 111's `TYPED` check
   must admit structs, and `var`/`value` must type the record scalar. The `Repeat`
   carries/updates and `store` then move a record value through the loop body. This
   is the fiddly bit — a struct carry is a record scalar slot, not a `from_dtype`
   array.
7. **Wire `optimise`.** In `decider/steps/optimise.py`, add a `record=` argument
   (an `Item` TypedDict) so `optimise` carries `best: Struct[Item] | None`; the
   keep-best copies the record on update and leaves `None` when nothing survives.
   `best_index == -1` already signals empty; the struct's `None` is the point-6
   answer.

Columnar output (D) is separate and lower priority: `Columnar[Item]` output is a
ragged list per row, which no shared kernel stores — plan for a Python-fallback
store (object) unless/until a `Ragged`-style offsets+child-buffers output is wanted.

## Verification

- `uv run pytest tests/run/test_struct_outputs.py tests/run/test_optimise.py -q`
- `uv run pytest tests/run/test_struct_columns.py tests/run/test_columnar.py tests/compile -q`
- Full: `uv run pytest -q` (expect 1998 passed, 1 xfail before this work; the count
  grows as tests are added).
- `uv run pytest tests/test_conventions.py -q` enforces the 500-line limit and the
  banned-reference rules — keep new files under it.

## Constraints (from the user, binding)

- **No codegen.** No `exec`/generated source. All machinery is callbacks/ordinary
  steps and decider's own compiled kernel — exactly how `loop`/`optimise` already
  work. This constraint is about *user step* code, not decider's internal kernel
  lowering in `kernel.py`/`packed.py`.
- **Simplicity over generality** (ponytail). Simplest thing that passes the tests;
  no speculative abstractions; stdlib first. Docstrings only on public API.
