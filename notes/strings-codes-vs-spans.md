# `Raw[str]` codes against `Raw[bytes]` spans

Should decider keep two compiled string representations? `Raw[str]` is an int32 code
from one process-wide table; `Raw[bytes]` is an `(address, byte length)` span into the
Arrow buffer. This measures the only thing codes do, equality, on both.

## Method

`uv run python benchmarks/strings_codes_vs_spans.py`, Apple M2 Pro, base 4c50d44,
`mode="fused"`, 1M rows. `first` is the first run over strings the code table has never
seen; `batch` is the best of three after it; `score` is the p50 of 5000 calls on one
record. Every shape is measured in one process.

## Measured (2026-10-06)

| shape | distinct | first M/s | batch M/s | score p50 µs |
|---|---|---|---|---|
| code, `== raw_str("private")` | 4 | 8.4 | 10.2 | 7.9 |
| span, `== "private"` | 4 | 12.8 | 29.4 | 8.5 |
| code, `== param` | 4 | 9.3 | 9.7 | 8.4 |
| span, `== param` | 4 | 12.9 | 29.3 | 9.1 |
| code, literal | 1,000 | 9.0 | 9.7 | 7.8 |
| span, literal | 1,000 | 33.9 | 35.3 | 8.4 |
| code, param | 1,000 | 9.5 | 9.7 | 8.4 |
| span, param | 1,000 | 34.2 | 35.1 | 9.1 |
| code, literal | 100,000 | 5.1 | 7.4 | 7.8 |
| span, literal | 100,000 | 39.6 | 42.7 | 8.4 |
| code, param | 100,000 | 5.4 | 5.9 | 8.4 |
| span, param | 100,000 | 36.9 | 41.3 | 9.3 |

## Reading it

- **Batch: spans are 3x faster at 4 distinct values and 6-7x at 100,000.** A code costs a
  dict lookup per row before the kernel starts; a span costs nothing before it. The gap
  widens with cardinality because the lookups stop hitting cache and new strings take
  the table lock.
- **Single record: codes are 0.6-0.9 µs faster.** That is the span buffer for one value,
  about a tenth of the call.
- **Codes have costs the table does not show.** The table never forgets a string, so a
  long-running server reading a high-cardinality column grows it without bound, up to
  2^31 codes. Codes are numbered in first-seen order per process, so a `Raw[str]` output
  is a number no caller can decode and two workers can disagree on.

## Decision

Remove `Raw[str]` codes. Spans become the one compiled string type, `Text`. Sub-µs on a
single record does not pay for a second representation, a global table, and the
always-`False` trap of comparing a code with a `str` literal.
