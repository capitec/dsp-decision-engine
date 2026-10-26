# What does a fixed-point money type cost, and where do rounding and int64 disagree?

Date: 2026-10-05 (rounding and overflow probe 2026-09-23) · Script: `benchmarks/decimal_steps.py` (the cents-type probe was a spike, not kept; the rounding, overflow and float-versus-integer probes are the spike `numeric-divergence`, in decider-experimentation)

## Purpose

Can money be a typed value at no measurable kernel cost, and where do Python, numba and `Decimal` disagree on rounding and int64 arithmetic?

## Method

A three-step loan flow, `score()` over 20,000 calls and a 200k-row batch, against the same arithmetic on plain int cents; micro-timings of a Python object crossing an njit dispatcher and of the 128-bit rescale; fixed `.xx5` ties and an int64 compound-interest loop found by binary search. Loaded box: read ratios.

## Result

| measurement | value |
|---|---|
| cents-type spike, fused `score()` p50 (plain / typed) | 32.6 / 32.4 µs |
| `Decimal[18, 2]` flow, `score()` p50 (int cents / Decimal), 2026-10-08 | 27.5 / 29.3 µs |
| same flow, `run()` | 26.6M / 20.7M rows/s |
| Python object crossing the dispatcher (alone / in a params 2-tuple) | 3.5 µs / 14-21 µs, against 0.3 / 0.4-0.6 for an int |
| 128-bit multiply-rescale, product fits 64 bits / wide | 6.0 ns / 21 ns (int64 multiply and floor divide, wrong: 2.3 ns) |
| `round(2.675, 2)`, CPython / njit | 2.67 / 2.68 (also 2.665, 1234.565, 0.025) |
| fixed-point compound interest at R100,000, Python / njit | R336,241.93 / -R90,971.75 (diverges from R27,431) |
| sum-of-squares accumulator over cents | wraps at 2,667 rows |

- Polars: results are `Decimal(38, max(s1, s2))`, `*` and `/` round half-even, `Decimal op float` is Float64, overflow raises.
- i128 division in numba aborts the process (`__udivti3`); a software 128-by-64 divide matched Python ints on 40,007 cases.
- Float64 against a 60-digit `Decimal` to the cent: an annuity instalment over 20,000 loans (R5k-R500k, 5-31.5%, 6-84 months) differed in 0 cases, worst error 1.7e-8 cents, nearest tie 5.6e-5 cents; with `fastmath` on, 0 of 20,000 instalments differed by a cent although 46% of the scorecard probabilities moved (up to 8 ULP), and no row flipped at seven realistic cutoffs over 500,000 rows (only a cutoff built to sit on one row's value did). `log` differed in 0 of 2,000,000 values across njit, numpy and CPython.
- A rate stored as scaled int (`round(rate * 1e9)`) was wrong by +5 cents of total interest on R250,000 over 60 months and -2 on R1,000,000 over 84; the exact rational `apr_bp / 120000` and float64 were exact. A cents-times-rate product overflows int64 above R92 million at 1e9 scale. Splitting R1,207.99 into 11 float64 parts summed to R1,208.02; largest-remainder integer parts summed exactly.
- Random-sample tests missed the overflow; only the binary search found it.

## Decision

Typed money is free in compiled modes, so `Decimal` stays with the object made at the kernel edge and hand-rolled 128-bit arithmetic. Rounding goes through one shared body and totals accumulate in float64. See [money](../decisions/money.md).
