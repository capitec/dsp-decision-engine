# Should trees run in Rust, a polars plugin, ZEN or generated code instead of the numba walker?

Date: 2026-09-21 to 2026-09-23 · Script: spikes, in decider-experimentation/ `rust-tree-interpreter`, `rust-cabi-in-kernel`, `cfunc-pointer-interpreter`, `polars-plugin-trees`, `zen-jdm-integration`, `tree-codegen-vs-interpreted`, `cached-plan-pattern`

## Purpose
Settle that one shared njit walker over arrays is the right tree engine, by building each alternative that someone may propose again and measuring what it would cost to own.

## Method
Same trees and rows for every engine (credit tree of 20 leaves, full binary depth 5/7/9/10, 100-leaf chain), 100k rows, answers asserted equal before timing, min and range of 4-7 reruns on a shared 28-core box. Each alternative was built for real: a PyO3 extension, a plain `cdylib` called from inside the kernel, a `pyo3-polars` plugin, `zen-engine` 2.0.2, and `@cfunc` pointers in a table.

## Result
| ns/row, 100k rows | 20 leaves | 128 | 512 | 101 chain |
|---|---|---|---|---|
| numba array walker | 52.5-53.6 | 85-87 | 119-121 | 34-36 |
| Rust PyO3 `walk_batch` | 46-51 | 73-86 | 86-89 | 30-33 |
| Rust C ABI, called per row inside the kernel | 46-54 | 67-74 | 82-95 | 28-35 |
| generated code | 20.7-22.9 | 42-50 | 83-127 | 28.4-28.8 |

- **Rust** is 12-35% faster than the walker and steadier at depth 9 (under 3% spread). PyO3 cannot fuse into a kernel; the C ABI can (2.4-3.84 ns a call) but kernels touching it lose the numba disk cache unless the address is passed as an argument, and one unprotected `panic!` aborts the process. PyO3 wheels are per Python minor; 50 crates, 15-29 s cold build. A fourth implementation to keep equal to the other three modes.
- **cfunc table interpreter** for branch/loop: the call is free (about 7.2 ns for float64, same as a ctypes global; cache survives), but it pays per node: 223 against 25 ns/row on a nested loop-branch-loop, 8.8x generated code. A `typed.List` of function values (the first POC, `jittree`) was 28-41x.
- **polars plugin**: honest types (`[2**53, 2**53+1] == 2**53` gives `[true, false]`), 17 ns/row at 1M rows on 28 threads against 253 for decider2 fused; but one record costs 236 µs (the tree is about 1 µs; planner 29, plugin call 37, frame 19) and about 700 µs with polars' default pool. Build 5 min 38 s, 252 crates, 5.7 MB wheel, and a Decimal column aborts the process. `pl.when/then` beats it below about 32 leaves and loses 11x at 512.
- **ZEN**: `evaluate_batch` 9,585 ns/row against 6,166 for the old engine (1.55x slower); a per-row loop is 226x, nearly all binding overhead, and `evaluate()` costs 1.2 ms whatever the graph (`async_evaluate` in a loop: 95 µs). The JDM converter refused 3 of 5 real decision tables (one bound mode per table, no per-row wildcard on `in`) and a tree exports with its named params baked in.
- **Generated code**: 2.0-2.7x faster per row to depth 7, but compile is 0.4 s at 20 leaves, 1.4 s at 128, 7.3-8.1 s at 512, 20.6 s at 1,024 and the 500-line cap bites at 100-130 leaves. At depth 10 it ranged 134-242 ns/row over identical reruns; the walker 138-141. The spike's walker compiled once, then took 0.2-2.3 ms per new tree (the decider toolkit's grouped walker: 0.5-6 s per new shape, see [tree walker costs](tree-walker-costs.md)).
- **polars expressions with a cached plan** (the `rtlf` idea): the optimizer is 1-3% of a `collect()` at 25+ nodes (up to 15% on trivial trees); building the physical plan is what grows with the tree, and no public polars API caches it. One record through `pl.when/then` costs 419 µs (20 leaves), 543 µs (32) and 7,796 µs (101 chain), 844x to 14,790x a compiled walker's 0.5 µs, whatever the row count (depth 400: 110.6 ms at 1,000 rows, 111.4 ms at 1). `rtlf` pins `polars==1.38.1` and git-pinned internal crates; its cold build passed 17 minutes. Beware `explain()` as an optimizer proxy: formatting a deep tree made it read 32% when the optimizer was under 1%.
- **composed njit callables** (generic kernels per node kind, called through a function-value table): a structural edit costs 56-189 µs against 375 ms-1.4 s for generated code, but the interpreter has the same zero-recompile property and runs 14-20x faster per row.
- Every engine above is under 10 ms for 100k rows, inside a 20-100 ms budget.

## Decision
Keep the shared numba walker. Rust, a plugin and ZEN buy 12-35% batch speed with a second toolchain and a process-abort failure mode; none helps the single record. Ideas worth keeping from ZEN, not built: a per-column bitset index for tables of 8+ rows and a `collect` hit policy. See [configurable steps](../decisions/configurable-steps.md).
