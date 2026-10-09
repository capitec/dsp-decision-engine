# What does it cost to call and compile the shared tree walker?

Date: 2026-09-23 · Script: `benchmarks/tree_walk.py` (the other measurements were one-off runs; the inlining rows are the spike `verification-probes`, in decider-experimentation)

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
- The same rule was found on decider2's walker: a non-inlined per-row call that takes arrays cost 365 ns/row; inlining it alone took a mixed tree from 247-262 to 125-133 ns/row (1.6-1.9x). Splitting one float64 feature array into float, int, bool and span arrays costs 12-35% of the walk and fixes two distinct int64 above 2**53 comparing equal as float64.
- Inlining: a tree walked through a real call per row ran 247-262 ns/row end to end and 201-217 walking; `inline="always"` on the walker and its path function gave 125-133 and 96 with identical outputs. Typed feature arrays (int64/bool split) with the inlining cost 12-35% more than inlining alone, and the first report of a 1.6-1.8x win for them was the inlining. Cold compile 1.97 to 1.61 s, warm start 0.89 to 1.05 s (the kernel now compiles the walk itself).
- A walker called once per row costs the array bookkeeping only when its arrays are not loop-invariant: decider2's version, with invariant structure arrays, cost 1.04x (45.6 against 43.8 ns/row); a walker that is handed per-row feature arrays pays about 365 ns/row. A table scan (30 rows x 5 conditions) gained 2% from inlining, decider2's branch-and-loop walker 22%.
- A new walker shape compiles in 0.5-6 s, then comes from the disk cache.

## Decision
Consts are addresses, outputs are picked by an intrinsic, numeric nulls are fills, and the walker's helpers are `inline="always"`. See [configurable steps](../decisions/configurable-steps.md).
