# Should tree and table data be literals in generated code or runtime arrays?

Date: 2026-09-23 (carried over from decider2's experiments) · Script: spikes on decider2, not kept

## Purpose
Settle whether thresholds, table rows and tree shape should be emitted into generated source or passed to one shared kernel as arrays, and what each costs per row and per edit.

## Method
Rule sets of 3-100 rules, tables of 2-1000 bands and trees up to depth 10 (plus a 128-node one-sided chain), each built both ways. Compile events, compile time and ns/row were compared on a loaded dev box.

## Result
- Thresholds as arguments: 8 retunes as literals were 8 recompiles (343 ms each at 5 rules); as arguments 0 compile events, one signature. Runtime cost +1-4.5 ns/row (+3.1% at 10 rules, +11.2% at 30).
- Rule enablement as a mask array: -2.0% at 10 rules (noise), +5.4% at 30; break-even against one recompile is about 427M rows.
- Table compile cost is independent of row count (the same 31-line source for 2, 50 and 1000 bands); the scan is linear, 70 ns/row at 2 bands and 1045 ns/row at 1000.
- Generated trees are 2.0-2.7x faster per row through depth 7; at depth 10 they ranged 134-242 ns/row over identical reruns (interpreted walk under 2% spread) and compile in 20 s. A 128-node chain hits CPython's 100-level indentation limit.
- Generated tree compile: 0.4 s at 20 leaves, 1.4 s at 128, 7.3-8.1 s at 512; a 500-line cap refused trees from 100-130 leaves. `output_fn`, an if/elif per leaf, cost nearly as much as the path function (3.3 s against 4.0 s at 512).
- A same-length literal edit was served stale by numba's own cache with CPython's `.pyc` cleared, not only by the `.pyc`; with thresholds as arguments there is no source constant to go stale.
- Hoisting thresholds did not shorten compile time (emitted lines were identical); generated source grows as lines^1.4, 56.6 s at 100 rules.

## Decision
Trees, tables and rule thresholds are runtime arrays walked by one shared function; edits never recompile. Rust, a polars plugin and ZEN as alternatives: [alternatives to a numba tree walker](alternatives-to-a-numba-tree-walker.md). See [configurable steps](../decisions/configurable-steps.md).
