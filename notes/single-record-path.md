# Single-record path: dict in, cached params bundles

**Decision:**
- `score()` takes one `dict` (`score(request, *, params)`), not keyword
  arguments.
- Converted params bundles (the namedtuples passed to kernels) are cached by
  params document and node, and are not rebuilt per request.

**Why (calling convention):** N2, 400 inputs, p50, 20 ms budget:

| convention | p50 | % of budget |
|---|---|---|
| kwargs | 1190.2 µs | 5.95% |
| dict | 60.1 µs | 0.30% |
| reused record buffer | 39.8 µs | 0.20% |
| positional | 93.3 µs | 0.47% |

- The cost is CPython matching keyword names against a 400-parameter signature.
  With body=`pass` the call costs 1094 µs; the same function called
  positionally costs 28.8 µs.
- Cost grows close to quadratically with width: 5.0 → 32.6 → 103.4 → 1190 µs at
  10/50/100/400 inputs.
- kwargs alone costs more than N1's whole optimised path, which is about 127 µs.

**Why (bundles):** In N3, turning a validated model into a NamedTuple costs more
than validating it:

| modules | validation | conversion | whole path |
|---|---|---|---|
| 50 | 78.3 µs | 170.6 µs | 256.5 µs |

- A memoised converted bundle costs 0.22–0.24 µs whatever the module count,
  726–765× faster than converting fresh at M=50.

**What we tried:**
- kwargs is kept for small hand-written calls, where it costs nothing.
- The reused record buffer is fastest but brittle: the caller must know field
  order, and it raises thread-safety and swap-staleness issues. It is opt-in
  only.
- Positional calls are rejected: 400 positional floats risk silent
  transposition.
- **Other N1 findings:**
  - Marshal whole rows, not per field. A per-field Python loop was 673 µs, 92% of
    the cost; the kernel was about 1 µs.
  - Pool the output buffer, and compile against a *writeable* array type.
- Validating raw params on every request is affordable (1.28% of budget at 50
  modules). So a rule that requests may only reference a stored bundle by id
  rests on governance, not speed.

**Source:** `decider2/docs/EXPERIMENTS.md` (§N1, §N2, §N3);
`decider2/docs/05-boundary-and-compilation.md` (§3.1b);
`experimentation/n2-calling-convention/`; `experimentation/params-validation-n3/`;
`experimentation/single-record-overhead/`

**The frame boundary** (a one-row `run(df)`, and batches). Fused mode,
`benchmarks/modes_vs_decider2.py` flagship and `trees_vs_decider2.py` trees,
before and after the column-at-a-time boundary (see `arrow-shim-build.md`);
20k `score` calls, 5k one-row runs (median), 100k and 1M rows (median of
15 and 7). Dev box under load average ~10 with swap full, so p99 and 1M
times move ±50% between runs; `score` doesn't touch the boundary, and nine
alternating before/after runs put tree `score` p50 at 59 against 61 µs.

| pipeline | engine | score p50 | score p99 | run 1 row | run 100k | run 1M |
|---|---|---|---|---|---|---|
| flagship | decider2 | 26.6 µs | 42.3 µs | 851 µs | 12.1 ms | 231 ms |
| flagship | decider before | 32.2 µs | 94.6 µs | 225 µs | 5.8 ms | 231 ms |
| flagship | decider after | 29.0 µs | 36.1 µs | 122 µs | 0.73 ms | 5.7 ms |
| tree | decider2 | 215 µs | 391 µs | 3388 µs | 45.3 ms | 1216 ms |
| tree | decider before | 60 µs | 148 µs | 480 µs | 38.5 ms | 1140 ms |
| tree | decider after | 66 µs | 257 µs | 239 µs | 22.6 ms | 227 ms |
| string-gated tree | decider2 | 259 µs | 556 µs | 3493 µs | 45.3 ms | 1018 ms |
| string-gated tree | decider before | 106 µs | 217 µs | 531 µs | 51.6 ms | 944 ms |
| string-gated tree | decider after | 104 µs | 271 µs | 292 µs | 32.4 ms | 360 ms |

Where a one-row flagship run went, before: `State.from_frame` 105 µs (the
Arrow import and release ~30 µs as six ctypes calls, the numba gather loop
~8 µs, a frozen dataclass per column, `frame.columns` rebuilt per input),
the output frame 33 µs (`pl.DataFrame` of every column rather than
`hstack`). Now `from_frame` is about 60 µs, of which the polars export
itself (`get_schema` + `get_next`) is ~10 µs.
