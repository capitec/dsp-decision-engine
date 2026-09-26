# Strings in compiled modes: what numba can hold, and what it costs

## What numba offers

Probed directly (`uv run python`, numba as installed):

- `unicode_type` exists only as a **scalar**, unboxed from a Python `str` object per
  call. There is no variable-length string array type.
- An object array of `str` cannot be typed at all (`TypingError` in the frontend).
- A numpy `U` array types as `Array(UnicodeCharSeq(n), ...)`, and `n` is part of the
  type. Taking `n` from the batch means a new specialisation per longest value —
  every distinct request width recompiles. Unusable for an API.

So a shared array kernel can hold a string only as an integer (a dictionary code) or
as a `(address, length)` span into Arrow memory. Anything else is one call per row.

## Measured (2026-09-26, loaded dev box, one process per table)

A one-step pipeline `sector == private` over 200k rows, and `score()` p50/p99 over
3000 single-record calls. `score()` is the 90% workload.

| variant | batch rows/s | score p50 | p99 |
|---|---|---|---|
| numeric kernel, no strings (the ceiling) | 173M | 22.2 µs | 34.9 |
| `str`, Python per row | 0.50M | **26.7 µs** | **34.4** |
| `str`, `mode="interpreted"` | 0.67M | 48.3 µs | 133 |
| `Raw[str]` dictionary code, in the kernel | 2.3M | 56.3 µs | 93 |
| `str`, njit per row (`unicode_type`) | 0.11M | 86.2 µs | 230 |
| `Raw[bytes]` span, in the kernel | 7.0M | 104.5 µs | 182 |

A `str` **output** with no `str` input (`label(x: float) -> str`):

| variant | batch rows/s | score p50 | p99 |
|---|---|---|---|
| njit per row, boxed result | 0.66M | 26.8 µs | 33.9 |
| Python per row | 0.78M | 26.4 µs | 33.6 |

## What that says

- Calling a compiled dispatcher once per row is the **worst** option for a `str`
  input: unicode unboxing plus dispatcher entry costs more than the Python body it
  replaces, in batch (0.11M vs 0.50M rows/s) and at `score()` (86 vs 27 µs).
- Plain Python per row is the best default for semantic `str`, and is within 5 µs of
  a pure numeric kernel on a single record.
- The kernel representations win only in batch: spans 14x Python, codes 4.7x. Both
  are *worse* than Python on a single record — the per-call setup (code encoding
  under a lock, or an Arrow export for the span buffer) is not amortised by one row.
  That single-record cost is the thing worth attacking, since trees read spans.
