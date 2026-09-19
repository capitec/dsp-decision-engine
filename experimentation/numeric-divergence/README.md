# numeric-divergence

Measures **where numba and Python disagree numerically**, and what each disagreement
costs in rands and cents.

It exists because doc 02 §3.1's equivalence ladder requires
`interpreted ≡ stepped ≡ fused` and doc 05 §9 demands agreement "exactly" /
"matching a numpy reference exactly", while REVIEW.md §7 argues exact equality is the
wrong test for floats and the right test for decisions, and that the tolerance policy
is unspecified. Nobody had numbers. This produces them.

## Run

```
.venv/bin/python experimentation/numeric-divergence/numeric_divergence.py
```

Full run: ~15 s wall clock on this machine, of which ~2 s is numba compilation.
No network, no installs, no writes unless you pass `--json`.

Flags:

| flag | effect |
|---|---|
| `--quick` | smaller `n` everywhere, same tests (~4 s) |
| `--only E2,E4` | run a subset |
| `--json PATH` | dump every measurement as JSON for diffing across machines |

## What it measures

| section | question | what it compares |
|---|---|---|
| **E1** | integer overflow | Python `int` (arbitrary precision) vs numba `int64` vs numpy `int64`, on a fixed-point compounding loop and on a sum-of-squares accumulator over loan amounts in cents. Binary-searches the exact principal / batch size at which they first disagree. |
| **E2** | rounding | `round(x)` and `round(x, 2)` in CPython vs njit vs `np.round` vs `Decimal`. Includes a 1,000,000-value population scan over amounts that are exact ties at 2 decimal places. |
| **E3** | float drift and fastmath | A 16-term scorecard chain (`log`, `log1p`, `exp`, `sqrt`, division) plus a calibrated logistic plus an annuity instalment, compiled with `fastmath` off and on, against a pure-Python reference. Reports ULP distributions, whether any cutoff flips a decision, the `x/y` vs `x*(1/y)` reassociation, and whether the drift survives rounding to cents. |
| **E4** | money | Can `Decimal` enter an njit kernel? Then: the annuity instalment and a full amortisation schedule computed three ways — float64 in a kernel, scaled int64 in a kernel, and 60-digit `Decimal` as the reference — compared to the cent. Plus fee-split reconciliation and `int64` headroom for a cents representation. |
| **E5** | division and NaN | `0.0/0.0`, `x/0.0`, `1//0`, `1%0` across five "references" (Python float, numpy scalar, numpy array, njit default, njit `error_model="numpy"`), then the same divide inside a serial row loop and inside a `prange` loop, then whether a kernel can raise an error naming the offending row. |

## How to read it

Every printed number is computed in-process by this script. Nothing is copied from
the design docs. `--json` writes the same measurements in machine-readable form so a
later run on different hardware, a different numba, or a different workload shape can
be diffed against this one.

The parts that matter most for the design are E2.c (rounding disagreement rate),
E4.c (the scaled-int rate-representation trap), and E5.b (serial vs `prange` error
semantics). The parts that came out *negative* — E3.f (no `log` drift) and E3.e2
(fastmath drift never reaches a cent) — matter just as much, because they say which
worry to drop.

## Known limits

- One machine, one CPU, one numba build. `fastmath` and `prange` behaviour are
  LLVM-version and target dependent; re-run before trusting the numbers elsewhere.
- E3's pure-Python reference runs on a 20,000-row subsample because it is ~1,000×
  slower than the kernel; the njit-vs-njit comparisons use the full 500,000.
- E4's Decimal reference uses `ROUND_HALF_UP` at the cent. That is a *choice*, not a
  fact; a different regulatory rounding rule would move E4.b and E4.c.
- E4.c covers five loan configurations, not a sweep. It is enough to show the
  rate-representation failure exists and that the rational form fixes it; it is not a
  claim about frequency.
