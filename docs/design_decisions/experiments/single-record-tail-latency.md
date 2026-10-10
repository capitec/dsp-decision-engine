# What drives the tail of single-record latency, and do GC tuning, `nogil` or core pinning change it?

Date: 2026-09-23 · Script: spikes, in decider-experimentation/ `n4-tail-concurrency-swap`, `n4-tail-cause`

## Purpose
At a 20 ms limit the tail is the limit, and earlier experiments reported medians. Find what makes the p99.9 and max of `score()` large, and which mitigation (GC settings, releasing the GIL, pinning the thread) is worth a rule.

## Method
The full `score()` path of the previous engine (400 inputs, 633 outputs, a p50 of about 1,040 µs, so absolutes are 30x today's) over 15,000 to 30,000 calls per configuration. Threads: 1 to 16 on one compiled kernel, built `nogil=True` and as a `nogil=False` twin. For the cause, each call's latency was correlated with `getrusage` deltas (page faults, voluntary and involuntary context switches), the CPU it ran on, and allocator and GC counters read outside the timed window, in four configurations: ambient load, pinned to one core, a no-op in place of `score()`, and 14 of 28 cores busy. An allocation sweep from 4 KB to 1 MB bracketed glibc's 128 KB mmap threshold. Box not idle (Firefox at 80% of a core, other sessions).

## Result
- Single thread, 30,000 calls: p50 1037.5 µs, p99.9 1723.6 µs (8.6% of 20 ms), max 3840.5 µs (19.2%). One GC collection in the whole run, and it fell on none of the calls above p99.
- GC on / `gc.disable()` / `gc.freeze()` after warmup: p50 1040.5 / 1040.4 / 1040.7 µs, p99.9 1165 / 1154 / 1166 µs. No tail effect, because `score()` allocates no reference cycles.
- 16 threads on one kernel, p99: 883 µs built `nogil=True`, 254,052 µs without (max 378.7 ms, 19x the budget). Throughput stays GIL-bound near 1,150-1,190 calls/s either way; the tail is the convoy.
- Tail cause: involuntary context switches (the OS preempting the thread) explain 100% of the slowest 0.1% of calls when pinned or contended (r = 0.56 pinned). Page faults were exactly 0 in 48,000+ calls, voluntary switches 0, allocation count and GC count uncorrelated (|r| < 0.02); the alloc sweep had 0 faults at every size and no cliff at the mmap threshold, up to 80x the real per-call volume; the dispatcher held one signature across 5,000 more calls. The no-op control had no tail (max 6 µs).
- Pinning to one core made it worse: p99.9 2004 to 3268 µs, max 3008 to 4959 µs (+65%), because the scheduler can no longer move the thread off a busy core.
- With 14 of 28 cores busy the median doubled (1053 to 2375 µs) while max over median fell from 2.86x to 1.26x: load shifts the whole distribution rather than adding outliers.
- Not explained: about 95% of the slowest 1% of calls at ambient load show no signal in any counter. Not tested: full saturation, cgroup `cpu.max` throttling (a periodic freeze), `perf` tracing.

## Decision
Kernels are `nogil=True` unconditionally; no GC setting and no `sched_setaffinity` call; judge a shared host by its median under load as well as its max. Reserved cores are a deployment concern (cpuset or cgroup, excluding other processes), not something decider sets. See [engine](../decisions/engine.md) and [stage a version while serving](stage-a-version-while-serving.md) for the swap cost.
