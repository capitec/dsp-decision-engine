# How should `score` take a record, and what should it cache?

Date: 2026-09-23 · Script: spikes, in decider-experimentation/ `single-record-overhead`, `n2-calling-convention`, `params-validation-n3`, `verification-probes`

## Purpose
Settle `score`'s calling convention and whether converted params bundles are cached, under a 20 ms single-record budget.

## Method
400 inputs, p50 over repeated calls for the calling conventions; a 50-module params document for validation against conversion. The phases around the kernel were timed one by one at 400 inputs and 633 outputs, and a profile run of the simplest one-tree pipeline. A one-row `run(df)` was also timed before and after the column-at-a-time Arrow boundary.

## Result
- Calling convention (400 inputs): kwargs 1190 us (5.95% of budget), dict 60.1 us, reused record buffer 39.8 us, positional 93.3 us. An empty body costs 1094 us with kwargs against 28.8 us positional: CPython matching names against a 400-parameter signature. Cost grows near-quadratically (5.0, 32.6, 103.4, 1190 us at 10, 50, 100, 400 inputs).
- Params at 50 modules: validation 78.3 us, conversion to a namedtuple 170.6 us; a memoised bundle 0.22 to 0.24 us (about 750x faster). Validating raw params on every request is still only 1.28% of budget.
- `**kwargs` catch-all, 17.9 µs, against the explicit 400-name signature, 1120 µs: the cost is CPython matching call-site names against a long signature, not building the dict. Measured on Python 3.14.5 only; a `timeit` of the empty call gave 1.1 ms at 400 keywords against 2.2 µs positional.
- Checking a 400-field request: pydantic `Model(**request)` 55.0 µs, a key-presence check 7.9 µs, a per-field `isinstance` loop 24.0 µs.
- Before any fix the whole `score()` was 971 µs (p50): marshal 220 µs and per-field readback of 633 values 673 µs were 92% of it. The same data moved as one tuple each way was 64.7 and 55.0 µs, 145 µs in all. Allocating the 633-field structured output was 36.9 µs against 1.38 µs for the kernel call itself (3 flat arrays: 1.42 µs), and GC on or off made no difference.
- Profiling the simplest pipeline (one tree, one feature) showed 104.5 µs a call, nearly all schema-invariant Python redone per record: a topological sort (900 `graphlib` calls per 300 records), fuzzy name matching with `difflib`, and a driver rebuild.
- One-row `run` on the flagship pipeline: 225 us before the column-at-a-time boundary, 122 us after (previous engine 851 us); `score` itself was unchanged.

## Decision
`score` takes a dict, per-pipeline Python and bundles are computed once (by document and node), whole rows move in one call rather than per field, no 400-name signature, and a frame step is the only thing that makes `score` build a polars frame. See [engine](../decisions/engine.md).
