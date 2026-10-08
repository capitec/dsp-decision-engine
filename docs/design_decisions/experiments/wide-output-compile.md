# Why does a tree of hundreds of rules take minutes to compile?

Date: 2026-10-08 · Script: `benchmarks/tree_wide_outputs.py`

## Purpose
A `prioritized_flat_rule` tree in `mode: "all"` with 521 rules took 172 s to bind and score, with almost nothing gained from the numba cache.

## Method
N similar rules (three conditions, one boolean output each), `Engine().bind(flow(tree), mode="fused").score(record)`, in a fresh `NUMBA_CACHE_DIR` (cold) and again in the same one (warm). Compile time per function came from wrapping `Dispatcher.compile`; LLVM time from `-time-passes`.

## Result
| Rules | One walker call: cold / warm | Split in 32s: cold / warm |
|---|---|---|
| 100 | 14.6 s / 7.5 s | 14.0 s / 2.4 s |
| 250 | 43.4 s / 31.8 s | 15.0 s / 3.2 s |
| 521 | 195.6 s / 167.4 s | 14.3 s / 2.6 s |

- The walker (`walk`) is a fixed 7 s; the fused kernel that calls it is the rest. Compile time grows with the square of the outputs a single call returns: the kernel took 2.7 s at 50 outputs, 7 s at 150 and 35 s at 300, with the walker replaced by a stub, so it is not the tree. It is LLVM (SROA, InstCombine, ConstraintElimination, instruction selection), on a function whose arguments and result hold hundreds of tuple elements.
- Plain numba does the same: a function returning a tuple of N booleans, stored into N arrays, compiled in 2.6 s at N=100, 8 s at 200 and 17.6 s at 400. Dropping the stores, shrinking the output spec to one scalar and unsigned store indexes each changed nothing about the growth.
- The fused kernel is never disk-cached. Several narrow calls share one compiled kernel and one cached `walk` when their output types match; one wide call is a new kernel per process.

## Decision
`TreeConfig` splits a `mode: "all"` tree into row nodes of at most 32 outputs. See [configurable steps](../decisions/configurable-steps.md).
