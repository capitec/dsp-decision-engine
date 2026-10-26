# How should a compiled step hold a string, and what does it cost on one record?

Date: 2026-09-26 (codes vs spans 2026-10-06, literal decode 2026-10-07) · Script: `benchmarks/span_operations.py` (numeric vs `str` vs span; the njit-per-row, codes and decode scripts were not kept)

## Purpose

Which representation (Python `str`, njit per row, dictionary code, span) should a semantic string use, how much does the single record pay, and how should `Literal` outputs be decoded?

## Method

One-step pipelines in fused mode over 200k rows and `score()` p50/p99 over 3000-5000 calls, one process per table with a numeric kernel as reference. The code/span comparison used 1M rows, Apple M2 Pro. Box load swung absolute numbers 2x; read orderings.

## Result

| variant | batch rows/s | `score()` p50 |
|---|---|---|
| numeric kernel (ceiling) | 173M-449M | 19-26 µs |
| `str`, Python per row | 0.5M | 27-39 µs |
| `str`, njit per row | 0.11-0.20M | 61-86 µs |
| span, `== "private"` | 7M-14M | 63 µs then 31 µs after the changes below |
| code, in kernel | 2.3M-10M | 56 µs then 22 µs |

- Codes against spans, 1M rows: 10.2M vs 29.4M/s at 4 distinct values, 7.4M vs 42.7M at 100,000; one record 0.6-0.9 µs faster for codes.
- A one-row `spans()` took 19.8 µs (four numpy calls); borrowing bytes made it 2.6 µs, and cached annotation helpers saved 6.5-19 µs a read. String-gated tree `score()` fell 145 to 91 µs; crossover with the Arrow export is n = 32.
- Content-hash codes: crc32 1.5 µs and blake2s 2.7 µs a value against 0.95 µs for the counter table, and blake2s collided 12 times in 200k.
- `Literal` decode, 8192 leaves: `score()` 639 to 14 µs once the choices table stopped being rebuilt per call; one-row `run()` pays about 11 µs of gather.

## Decision

`str` runs in Python, `Text` spans are the one kernel string, codes were removed, one-record spans borrow bytes, and `Literal` outputs decode per reader. See [strings](../decisions/strings.md).
