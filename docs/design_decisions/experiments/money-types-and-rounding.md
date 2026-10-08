# What does a fixed-point money type cost, and where do rounding and int64 disagree?

Date: 2026-10-05 (rounding and overflow probe 2026-09-23) · Script: `benchmarks/decimal_steps.py` (the cents-type and rounding probes were spikes, not kept)

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
- Random-sample tests missed the overflow; only the binary search found it.

## Decision

Typed money is free in compiled modes, so `Decimal` stays with the object made at the kernel edge and hand-rolled 128-bit arithmetic. Rounding goes through one shared body and totals accumulate in float64. See [money](../decisions/money.md).
