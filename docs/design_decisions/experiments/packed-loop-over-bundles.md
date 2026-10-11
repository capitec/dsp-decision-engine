# Does a loop over candidate bundles pack, and how far is it from hand-written numba?

Date: 2026-09-27 · Script: `benchmarks/bundle_search.py`, `benchmarks/fan_out_reduce.py`

## Purpose
`optimise(...)` lowers to a `loop`. Check that a loop whose body reads `Columnar[Item]` packs into one kernel in fused mode, and what that is worth against stepped mode and a hand-written `@njit` search.

## Method
Each order has items with price and weight; the loop tries every bundle mask (2^n - 1), sums the bundle and keeps the best. Run with 6 items and 10 orders on 2026-10-08; the answer matched the reference in every mode. Shared box, so p50 only.

## Result
| method | `score()` 1 order | `run()` 10 orders |
|---|---|---|
| hand-written `@njit` | 0.001 ms | 0.006 ms |
| fused (loop packed) | 0.069 ms | 0.310 ms |
| stepped (Python-driven) | 3.989 ms | 30.704 ms |

- Fused is about 58x (`score`) and 100x (`run`) faster than stepped, and about 50x off the hand-written ceiling.
- `fan_out_reduce.py` (4 items, 20 orders, with a `param_table` rate card): `optimise` and a hand-written `loop` both pack, `fallbacks()` is empty, and they differ by under 5% (`score` 0.101 against 0.100 ms; `run` 0.266 against 0.254 ms).

## Decision
`optimise` is a thin lowering to `loop` and needs no kernel of its own; packing a loop with a `Columnar[Item]` body is what makes it viable. See [engine](../decisions/engine.md).
