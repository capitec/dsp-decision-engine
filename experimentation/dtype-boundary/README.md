# Experiment A — what dtypes can actually cross the polars→numba boundary

Tests **doc 05 §1 "Extraction"** (`decider2/docs/05-boundary-and-compilation.md`)
against the real environment, for every dtype credit logic actually uses, in a clean
and a null-bearing form. Written because REVIEW.md §4.4 says the admissible-dtype
contract is unwritten and doc 05 §1 is "the first thing to build".

## Run

```
<repo>/.venv/bin/python experimentation/dtype-boundary/dtype_boundary.py   # ~2 min (numba compiles)
<repo>/.venv/bin/python experimentation/dtype-boundary/part2.py            # ~20 s
<repo>/.venv/bin/python experimentation/dtype-boundary/part3_catcodes.py   # ~2 s, run twice
```

`dtype_boundary.py --quick` skips the 100k-row timing sweep.

Polars emits Rust panics (`pyo3_runtime.PanicException`, a `BaseException`, not an
`Exception`) for some dtypes, and prints an unsuppressable `thread '<unnamed>' panicked`
line to stderr. Filter stderr if you want clean output; the probes catch the panic.

## What each file measures

| file | measures |
|---|---|
| `dtype_boundary.py` | per dtype × {clean, null}: does doc 05's `to_arrow().buffers()` 2-tuple unpack work; polars-native buffer parts; `to_numpy()` dtype/zero-copy; whether a trivial `@njit` kernel compiles against the result; extraction cost at 100k rows, median of 15 after warmup. Then the five traps doc 05 §1 asserts. |
| `part2.py` | the four workarounds the tables imply: string→categorical codes, Decimal→scaled int (incl. a raw Int128 `ctypes` read), sliced/multi-chunk zero-copy correctness, harder attempts to observe garbage under a null, exact Int64+null lossiness. |
| `part3_catcodes.py` | isolates the one question that decides the string path: are Categorical/Enum physical codes stable across frames and across processes? |

## Headline results

- **pyarrow is not installed**, so doc 05 §1's recipe (`series.to_arrow()`) raises
  `ModuleNotFoundError` for *every* dtype. The whole spec is unrunnable as written.
  `Series._get_buffers()` is the polars-native equivalent and needs no new dependency.
- Only **6 of 26** dtype/nullability combinations are zero-copy: clean Float64, Int64,
  Int32, UInt8, Datetime, Duration. **Every** null-bearing column copies.
- **Categorical codes are per-series, first-appearance order** — the same string gets a
  different code in a different frame. Unsafe as a boundary representation. **Enum is
  safe**, conditional on versioning the declared category list with the kernel.
- `_get_buffers()` silently rechunks a multi-chunk series. Only `to_numpy(allow_copy=False)`
  is an honest zero-copy gate.

Numbers are from one machine; re-run rather than quoting them.
