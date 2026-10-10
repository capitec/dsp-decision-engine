# Strings

Read this before changing how a compiled step holds, compares or returns a string: it lists what the code does and why.

## A kernel holds a string only as a span; `str` runs in Python

- `Text` is the one compiled string type: an `(address, byte length)` span into Arrow memory, length -1 for a null.
- Why: numba has no variable-length string array, and a numpy `U` array recompiles per longest value.
- See: [kernels](../../architecture/engine/kernels.md), [Types and money](../../guides/types-and-money.md)

## A semantic `str` step is a plain Python fallback, not njit per row

- `compile_call` returns `dispatcher.py_func` with a `FallbackWarning`; `@allow_fallback` accepts it, and `strict_compile=True` has no `str` exemption.
- Why: unboxing a `str` per row made njit 2.6x-4.9x slower in batch and 1.1x-2.6x on `run(record)`; a Python `str` step costs under 10 µs on one record.
- See: [experiment](../experiments/string-representations.md), `decikit/_engine/compile/njit.py`

## No dictionary codes: spans replaced `Raw[str]`

- Why: spans were 3x faster in batch at 4 distinct values, 6-7x at 100,000; codes cost a dict lookup per row.
- Codes also need a process-wide table that never forgets, number strings in first-seen order per worker, and make outputs undecodable.
- Not: content-hashed codes (31-bit collisions: 12 over 200k strings, and slower).
- Not: a per-category mask for `starts_with`/`contains` (run the pattern over the distinct values, look up by code). It is 0.8-1.8 ns/row only if the column already arrives encoded; encoding it costs 47 ns/row against 34 for polars `str.contains`. A span node on polars' string view is 56-75 ns/row against 178-185 for the cast, and 17-33 against 62-198 when 1% of rows reach it.
- Not: `Series._get_buffers()` to reach the bytes; it copies 17-28 ns/row, where `__arrow_c_stream__` is 1.2 µs at any size. See [string matching beyond equality](../experiments/string-matching-beyond-equality.md).

## `SpanType` is its own numba type, not a tuple

- `==`, `len` and the rest are overloaded on `SpanType` alone.
- Why: overloads on the bare int pair changed `(1, 2) == (1, 2)` to False, and numba folds `len(tuple_arg)` to 2 before inference.
- Before that every mode agreed on an always-False `sector == "private"`, so the interpreted `Text` mirrors the kernel and `assert_equivalent` covers it.

## Operations: `==`, `!=`, `len`, `startswith`, `endswith`, `in`; no ordering

- A literal, module constant or `param()` compares as constant bytes made at compile time; a string built at run time is a `TypingError`, so the step falls back.
- `len` counts code points (O(n)); `span[1]` is the byte length. Byte matching is exact on code-point boundaries because UTF-8 is self-synchronising.
- Not: `<` and `>`. A null has no CPython ordering, so any answer would be invented.

## A null is not a string

- It equals nothing, not even another null; `startswith`, `endswith` and `in` are False; `len` is 0; `span[1] < 0` tests for null.
- Why: the NaN rule. `len` cannot return -1 outside a kernel.

## One-record spans borrow the values' own bytes

- Up to 32 values, `spans` borrows each `bytes` buffer via `PyBytes_AsString` and keeps them in `alive`; above that it exports Arrow buffers.
- Why: building a frame cost 19.8 µs for one row against 2.6 µs borrowed; 32 is the measured crossover. Annotation helpers are `lru_cache`d on the plain type, taking a string-gated tree `run(record)` from 145 to 91 µs.
- See: [experiment](../experiments/string-representations.md), `decikit/_engine/run/representations.py`

## Declarations keep `Annotated` metadata; type logic strips it

- `plain_annotation` runs inside each cached helper, and `hoist_metadata` normalises `Optional[Annotated[T, M]]` once at harvest.
- Why: the debugger reads units off the decls, and an unhashable `Annotated` metadata crashed every cache key.

## A `Literal` output is an index, decoded where it is read

- A tree's string outputs come back as indices into a `CodeTable` built once per kernel; `run(record)` indexes a cached object array (about 0.1 µs a row), `run()` gathers from a cached polars column.
- `lookup(values, dtype)` does the same for lists and records; a struct flattens to `column.field` columns; `decode()` deep-copies mutable values.
- Not: one decode for both paths. An object column into polars grows with rows (2 µs at 1, 2 ms at 100k) while `gather` is flat. See [experiment](../experiments/string-representations.md), `decikit/_engine/compile/outputs.py`
