# What does a `str` item field cost, and should an item be a record?

Date: 2026-09-26 · Script: `benchmarks/nested_item_fields.py`

## Purpose
Can an `Item` field be a `str` without copying, and would a record per item (`for i in items: i.el_1`) beat the namedtuple of per-field arrays?

## Method
200k rows x 3 items, `fused`, 3000 interleaved `score()` calls, the body "first item matching a predicate". Then the per-row call cost of namedtuple versus record shapes at 3 to 3000 items and 2 or 8 fields. Later re-measured in the kernel, and the numba `Record`-with-span route prototyped.

## Result
- A `str` field is an array of `(address, length)` spans over polars' own buffers: 43-97 ns an item in batch (a `float` field costs 64 ns), about +7.5 us on a single record. Comparisons match CPython over ASCII and 2- to 4-byte UTF-8.
- Against per-row dispatch, records were 2-5x cheaper per call (0.41 vs 1.20 us at 2 fields, 0.41 vs 2.35 us at 8) because numba unboxes one namedtuple member at a time. In the kernel that is gone: building the row is flat in item count and the per-field cost is paid once per launch (about 2% of a `score()`).
- A record needs one interleaved buffer per parent row, so the kernel path would copy it: 2.3x slower at 3 items, 33.9x (151 us) at 3000 items x 8 fields.
- A `Record` carrying a span is sound in numba 0.67 (override `dtype`, `register_model(RecordModel)`; `dtype` is read once, for an unused `bitwidth`). Hand-rolled iteration over the namedtuple crashed the process (`malloc(): unaligned tcache chunk`).
- `np.recarray` slices cost 5.6 us against 0.24 us for a plain record view.

## Decision
Ship the `str` field; keep the namedtuple of arrays and document positional access. See [nested data](../decisions/nested-data.md).
