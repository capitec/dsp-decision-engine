# How should `score` take a record, and what should it cache?

Date: 2026-09-23 · Script: spikes on the previous engine (decider2), not kept

## Purpose
Settle `score`'s calling convention and whether converted params bundles are cached, under a 20 ms single-record budget.

## Method
400 inputs, p50 over repeated calls for the calling conventions; a 50-module params document for validation against conversion. A one-row `run(df)` was also timed before and after the column-at-a-time Arrow boundary.

## Result
- Calling convention (400 inputs): kwargs 1190 us (5.95% of budget), dict 60.1 us, reused record buffer 39.8 us, positional 93.3 us. An empty body costs 1094 us with kwargs against 28.8 us positional: CPython matching names against a 400-parameter signature. Cost grows near-quadratically (5.0, 32.6, 103.4, 1190 us at 10, 50, 100, 400 inputs).
- Params at 50 modules: validation 78.3 us, conversion to a namedtuple 170.6 us; a memoised bundle 0.22 to 0.24 us (about 750x faster). Validating raw params on every request is still only 1.28% of budget.
- Marshal whole rows: a per-field Python loop was 673 us, 92% of the cost; the kernel was about 1 us.
- One-row `run` on the flagship pipeline: 225 us before the column-at-a-time boundary, 122 us after (previous engine 851 us); `score` itself was unchanged.

## Decision
`score` takes a dict, bundles are cached by document and node, and a frame step is the only thing that makes `score` build a polars frame. See [engine](../decisions/engine.md).
