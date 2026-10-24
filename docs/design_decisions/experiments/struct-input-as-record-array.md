# How should a struct column reach a kernel?

Date: 2026-09-26 · Script: `benchmarks/structs.py`

## Purpose
A `dict` input ran in Python per row. Does a record array (one record per row) beat it, and is a zero-copy namedtuple of scalars worth the extra machinery?

## Method
`applicant["income"] * 2 + applicant["age"]`, 200k rows and 6000 interleaved `score()` calls per variant, against flat numeric columns as the ceiling.

## Result
| variant | batch rows/s | score p50 |
|---|---|---|
| flat numeric columns (ceiling) | 181M | 45.1 us |
| `dict`, Python per row | 0.46M | 56.4 us |
| `Struct[Applicant]` in the kernel | 1.30M | 52.0 us |

- No crossover: the record is faster than the Python dict even on one record, so there is one implementation.
- Per row: kernel over records 1.3 ns, over flat columns 1.3 ns; building the record from Arrow 8.5 ns. A namedtuple of scalars would save 2.7 ns for a new kernel source kind.
- Batch stays 3x, not 50x, because `State.from_frame` builds a Python dict per nested row (592 ns/row, about 135 ms per 200k) before any step runs.
- `Series.struct.unnest()` costs 1.9 us at any size; `struct.field()` is 38 us per field.
- A list field (`{rate, amounts}`) with per-row dispatch measured 10x slower than Python in batch (0.35M vs 3.34M rows/s) and was refused.

## Decision
Record array, float/int/bool fields, required input only. See [nested data](../decisions/nested-data.md).
