# How should a kernel read a polars frame: hand-decoded Arrow, nanoarrow per element, or one gather?

Date: 2026-09-23 · Script: spikes, in decider-experimentation/ `nanoarrow-accessors`, `boundary-rework-design`, `dtype-boundary`

## Purpose
Choose how the boundary decodes Arrow (hand-written layout code against a library), how many C calls one frame costs, and which polars dtypes cross without a copy.

## Method
Three readers on identical data, answers asserted equal: (A) hand-decoded Utf8View in numba, (B) vendored nanoarrow 0.9.0 per row through a C shim, (C) `ValidateFull` once then A. Then a 17-column frame (10 Float64, 4 Int64, 2 Boolean, 1 String) and a 17-node tree, read per column with numpy, by generic nanoarrow accessors, or by descriptors resolved once and one gather. Corrupt buffers were each tried in their own subprocess. Shared 28-core box (+-20%).

## Result
- **Safety**: nanoarrow validates string views not at all (`ArrowArrayViewValidateFull` has no `STRING_VIEW` case, in 0.9.0 and `main`), and its accessor is `...Unsafe`. Buffer index 1000 or offset 2**31-1 SIGSEGV; a length of 2**30 or 2 buffers declared for a view column returns a wrong answer. Hand-decoded numba returned an error leaf on 4 of 5 cases. Arrow C++ `validate(full=True)` does catch every one. Option C crashes exactly as B does.
- **Code owned**: A is 42 lines of Python; nanoarrow with its own bounds checks is 37 lines of C, so the library moves the layout code into C rather than removing it. Where nanoarrow wins: a producer switching to `u`/`U` strings needs no change.
- **One record, string column only**: A 42.6-45.1 µs (21.0 trimmed), B 19.3-20.6 (9.3), C 33-36. The time is the import handshake and Python glue, not decoding; at 1M rows A is 6-10% faster because B's per-row call cannot inline.
- **Whole frame, 17 columns** (`n=1` / 1,000 rows / 1M rows): per-column numpy extraction 130-142 µs / 440 µs / 316-332 ms; generic nanoarrow accessors per element 59-90 µs / 455 µs / 370-389 ms; descriptors resolved once, one C gather call per row 43 µs / 357 µs / 318 ms; same descriptors with numba loading from the resolved addresses 42 µs / 341 µs / 304 ms. At n=1: `__arrow_c_stream__` 3.2 µs, polars' own stream walk 27 µs, nanoarrow import 25-27 µs, today's extraction 122 µs, building `pl.DataFrame([record])` 200-260 µs.
- **Export behaviour**: a frame with columns of different chunk layouts exports as one struct chunk and polars rechunks it in place; a sliced single-chunk frame carries `offset` on each child; Categorical exports as a dictionary with uint32 indices, Enum uint8, String as `vu`.
- **Dtypes** (26 dtype x null combinations): only clean Float64, Int64, Int32, UInt8, Datetime and Duration are zero-copy; every null-bearing column copies. Categorical codes are per-series, in first-appearance order, so the same string gets different codes in two frames; Enum codes are stable if the category list is versioned. `_get_buffers()` silently rechunks; only `to_numpy(allow_copy=False)` gates a copy.

## Decision
Adopted nanoarrow with a whole-frame import and one resolve per batch (`sm_columns`); a Categorical is read with its exported dictionary, not its raw codes. The spike's safety finding still holds: `sm_get_string` calls `ArrowArrayViewGetStringUnsafe`, so the shim trusts the frame's string views. See [serving](../decisions/serving.md) and [zero-copy Arrow gather spike](zero-copy-arrow-gather-spike.md).
