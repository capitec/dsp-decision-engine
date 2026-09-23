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
