# boundary-rework-design — throwaway probes behind `decider2/docs/BOUNDARY-REWORK.md`

Run from this directory with `../../.venv/bin/python <probe>`; `c/build.sh`
builds `c/librowshim.so` against `../nanoarrow-accessors/vendor` (nanoarrow
0.9.0). Numbers below are the runs the design cites (shared 28-core box,
load present; treat as ±20%).

## probe_gather.py / probe_gather2.py — one frame import + one gather per row

17 columns (10 Float64, 4 Int64, 2 Boolean, 1 String), a 17-node chain tree
reading every feature plus an exact string match at the root. Answers
asserted identical across shapes.

| shape | n=1 (µs/call) | n=1000 | n=1M | kernel-only ns/row @1M |
|---|---|---|---|---|
| N  today: per-column numpy extraction (+ `cast(Categorical)` for the string) + numba `_fill` | 130–142 | 440–450 | 316–332 ms | 284 |
| V1 whole-frame nanoarrow import, generic `ArrowArrayViewGet*Unsafe` per element | 59–90 | 454–457 | 370–389 ms | 374 |
| V2 whole-frame import, column descriptors resolved once in C, one C gather call per row | 43 | 357 | 318 ms | 328 |
| V3 same descriptors, numba loads from resolved addresses, no per-row C call | 42 | 341 | 304 ms | 320 |

n=1 breakdown (17 columns): `df.__arrow_c_stream__()` 3.2 µs; polars'
own stream walk (schema + struct chunk + releases, no nanoarrow) 27–28 µs;
nanoarrow import + release (pooled structs) 25–27 µs total; today's 17-column
numpy extraction 122 µs; `pl.DataFrame([record])` for 17 columns ~200–260 µs.

## probe_chunks.py — frame-level chunking, slicing, dictionaries

- A frame whose columns have different chunk layouts (2/2/1) exports as ONE
  struct chunk, and polars rechunks the frame IN PLACE while exporting
  (`n_chunks` 2/2/1 → 1/1/1 afterwards). An aligned 2- or 4-chunk frame also
  exports as one struct chunk (rechunked in place).
- A sliced single-chunk frame exports children with `offset=3` (the struct
  itself has offset 0); a slice of a multi-chunk frame is rechunked to offset 0.
- Categorical → dictionary array, uint32 indices (nanoarrow storage type 7),
  dictionary `vu`; Enum → uint8 indices (3), dictionary `vu`; nulls are null
  indices. String → `vu`, no dictionary.

## probe_pyarrow_validate.py — does Arrow C++ validate string-view elements?

pyarrow 25.0.1 (installed into the venv for this probe only). Each case in a
subprocess; corruption written into polars' own buffer after a zero-copy
`pa.chunked_array(series)` import.

| case | `validate()` | `validate(full=True)` | element read afterwards |
|---|---|---|---|
| clean / sliced | ok | ok | ok |
| buffer_index = 1000 of 1 | ok | `ArrowIndexError: View at slot 1 references buffer 1000 but there are only 1 data buffers` | SIGSEGV |
| offset = 2³¹−1 | ok | `ArrowIndexError: View at slot 1 references range 2147483647-2147483684 of buffer 0 but that buffer is only 80 bytes long` | SIGSEGV |
| length = 2³⁰ | ok | `ArrowIndexError: ... range 0-1073741824 of buffer 0 but that buffer is only 80 bytes long` | SIGSEGV |
| length = −1 | ok | `ArrowInvalid: View at slot 1 has negative size -1` | SIGABRT |

Arrow C++ full validation checks every string-view element; nanoarrow 0.9.0
and `main` (`src/nanoarrow/common/array.c`, fetched 2026-09-23) do not.

## score() profile (main checkout, flagship pipeline, complete record)

`score()` p50 409 µs; `apply(1-row frame)` p50 1352 µs. cProfile over 2000
calls, cumulative: `compile.driver._return_dtype` → `inspect.signature`
(eval_str) 0.85 s of 2.18 s; `Pipeline.interface` → `_walk` → difflib
`suggest_name` 0.61 s; `flatten_for_runtime` 0.25 s; `_build_call_args` +
`kernel_signature` 0.15 s. The kernel is not in the top 30.
