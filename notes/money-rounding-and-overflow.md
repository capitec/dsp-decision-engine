# Money, rounding and int64 overflow

**Decision:**
- Money is a scaled `int64` of cents, never `float` and never `Decimal`.
- Running totals that can grow accumulate in `float64`, which is exact to 2⁵³
  (about R90 trillion in cents).
- Steps don't call bare `round()`; rounding goes through one helper that gives
  the same answer in every mode.
- Test corpora include `.xx5` ties and large principals.

**Why:**
- **Rounding differs by mode.** CPython `round()` uses banker's rounding, njit
  follows numpy, and neither matches `Decimal` HALF_UP consistently:

  | x | Python `round(x, 2)` | njit |
  |---|---|---|
  | 2.675 | 2.67 | 2.68 |
  | 2.665 | 2.67 | 2.66 |
  | 1234.565 | 1234.57 | 1234.56 |
  | 0.025 | 0.03 | 0.02 |

  Each row is a cent on an instalment, and the modes disagree.
- **int64 wraps silently.** Fixed-point compound interest (rate scale 1e12)
  diverges from a principal of R27,431. At R100,000, Python gives R336,241.93 and
  njit gives −R90,971.75. numpy wraps the same way, with a warning; Python ints
  don't wrap.
- **Accumulators wrap too.** A sum-of-squares accumulator over loan amounts in
  cents wraps at 2,667 rows, well inside one batch.

**What we tried:**
- `Decimal` in kernels: it can't enter an njit kernel. decider2's boundary
  rejects a Decimal column and says to cast it to a scaled integer in the frame
  tier first.
- Random-sample tests: the overflow was found by binary search, not by sampling;
  random draws would not have surfaced it.
- decider2's docs specify a `round_half_up` helper and a lint against bare
  `round()`. **Neither was built.** The rule still stands; the helper is
  still to be written.

**Source:** `decider2/docs/EXPERIMENTS.md` (§I);
`decider2/docs/03-authoring-api.md` (§1.2);
`experimentation/numeric-divergence/`; `decider2/src/decider2/_arrow/frame.py`
