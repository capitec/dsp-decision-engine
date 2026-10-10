# Can a new config version compile while the old one serves, and how is it swapped?

Date: 2026-09-23 · Script: spikes, in decider-experimentation/ `staged-compile-atomic-swap`, `subprocess-handover`, `n4-tail-concurrency-swap`

## Purpose
Settle how `stage()`, `activate()` and `rollback()` should work: where the compile runs, whether a request can straddle two versions, and what a swap costs a request in flight.

## Method
A 10-rule kernel served 100k-row batches while a 30-rule kernel (about 490 emitted lines, 5 s to compile) was built in a background thread, then in a child process that wrote numba's disk cache for the parent to load. Swap atomicity: 4 threads on 200k-row batches with a swapper flipping the pointer every 0.26 ms (11.6k swaps in 3 s), against a control that re-reads the pointer per 20k-row chunk. Swap latency: single-record dispatch for 3 s, a swap every 100 ms (758,215 calls). Python 3.14 with the GIL, numba 0.67, 28 cores.

## Result
- Pointer read once per invocation: 0 of 1,564 batches straddled two versions; the control straddled 1,576 of 1,578. `activate()` 0.18 µs median; `rollback()` 3.4 µs with 0 compile events (the old version is still in memory). Three resident versions cost 296 MB against 252 MB for one.
- Single-record calls across 30 swaps: p50 3.5 µs, worst call 274.8 µs (1.37% of a 20 ms budget); the first call after a swap 7.3 µs median (2.1x), worst 14.7 µs.
- A compile in a background thread works and its output is bit-identical to a single-thread compile, but serving keeps 26% of its throughput with a `nogil` kernel (6.41 against 1.18 ms per 100k-row call) and 55% without. The `nogil` loss is a fixed +5.2 ms per call (+5.4 ms at 1M rows) while the compiler thread holds the GIL: it tracks `sys.setswitchinterval` (5 ms: +5.24; 0.5 ms: +0.68; 0.05 ms: +0.15), not batch size.
- numba compiles one function at a time: two 10-rule compiles in two threads took 2.43 s against 2.52 s sequential (1.04x), and a 3-rule compile started 0.5 s into a long one took 5.08 s instead of 0.48 s. A request that needs a lazy compile waits for a staged one.
- A child-process compile left serving at 97.9% of throughput (1.154 against 1.129 ms); the parent then loaded the result in 5.7-10.2 ms with 0 compile events and an identical checksum. Config change to serving: 2.56 s for 10 rules, 7.34 s for 30.
- Across the process boundary a wrong cwd or file name failed loudly, a different cache directory or `def` line missed safely, and a module not registered in `sys.modules` under the pickled name raised `ModuleNotFoundError: No module named '<dynamic>'` from inside numba, with no hint about caching.

## Decision
`module_fn` reads the active version once per request; `stage()` builds and warms a version without serving it, `activate()` is a pointer swap under a lock, `rollback()` pops the previous one. Nothing tunes `setswitchinterval`; the thread-compile penalty is why a deployment should run `decikit build` first so `stage()` mostly loads cached user steps. Fused kernels are built from LLVM IR and are not disk-cached, so the child-process handover measured here covers only user steps and the shared walkers today. See [serving](../decisions/serving.md), `decikit/serving/_handler.py`.
