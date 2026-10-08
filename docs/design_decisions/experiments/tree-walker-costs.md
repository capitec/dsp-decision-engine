# What does it cost to call and compile the shared tree walker?

Date: 2026-09-23 · Script: `benchmarks/tree_walk.py` (the other measurements were one-off runs)

## Purpose
Find the cheapest way to pass a tree to a numba kernel, to build outputs for trees with many outputs, and to carry nulls.

## Method
A 127-node tree over 18 features (also behind a `starts_with` string gate); `score(dict)` p50/p99 and 1M-row `run(df)`. Each variant was run back to back on a loaded box (load 3-15), so ratios hold and absolutes do not.

## Result
| Question | Before | After |
|---|---|---|
| Consts as arrays vs addresses (one-node tree) | 525 ns/row | 30 ns/row |
| Output building: type-level recursion (8 outputs) vs intrinsic (521 outputs) | 48 s compile | 25 s compile |
| Numeric null as `T \| None` vs a fill (`score()` p50, 18 features) | 102 us (75 us with a shared mask) | 58-61 us |
| Trace weights: one extra add per row (`score()` p50) | 63.4 us | 62.5 us (noise) |

- Arrays crossing the per-row call were reference-counted about 60 times a row. `forceinline` recursion over outputs raised `RecursionError` on a 100-rule `mode: "all"` set; slicing a 500-field tuple cost about 100 s in LLVM.
- A new walker shape compiles in 0.5-6 s, then comes from the disk cache.

## Decision
Consts are addresses, outputs are picked by an intrinsic, numeric nulls are fills. See [configurable steps](../decisions/configurable-steps.md).
