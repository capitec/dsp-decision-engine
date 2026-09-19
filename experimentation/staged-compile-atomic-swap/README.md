# Experiment H — staged compilation and atomic swap (doc 08 §4)

## What this measures

Doc 08 §4 proposes that the whole configuration lifecycle rests on compiling a new
pipeline generation **in the background while the old one keeps serving**, then
swapping a pointer:

> `ACTIVE --stage(structure)--> COMPILING (background) --> STAGED --activate()--> ACTIVE`
> "previous generation keeps serving throughout"
> 1. "**`apply` and `score` read the generation pointer exactly once per invocation.**
>    A swap landing mid-call cannot split one record across two generations, and cannot
>    split one batch."
> 5. "**Rollback is free**, because the previous generation is still compiled and in memory."

None of that had a measurement behind it. This harness supplies one, for six questions:

1. **Does numba compile at all in a background thread** while a compiled kernel runs in
   the main thread — and is the result *correct*, not just exception-free?
2. **Interference**: serving throughput alone vs. during a background compile.
3. **Does the GIL serialise it**, even when the serving kernel is `nogil=True`?
4. **Atomic swap**: does a batch in flight during a swap complete on one generation?
5. **Memory**: RSS with 1, 2 and 3 generations resident.
6. **Rollback**: does swapping back compile anything?

Plus the fallback the brief asked for: **compile in a child process** and hand the result
over through numba's file-backed cache.

## How to run

```
.venv/bin/python experimentation/staged-compile-atomic-swap/run.py            # all phases -> results.json
.venv/bin/python run.py --phase swap        # one phase: interference | gilprobe | lockstep | subproc | swap | memory
```
~2.5 min wall on a 28-core box. Writes `results.json`; creates `_gen/` (generated module +
numba cache) at runtime, which is disposable.

## Workload

The staged kernel is a **30-rule flat rule set (~490 emitted lines, ~5 s compile)** and the
serving kernel a **10-rule set (~160 lines, 1.16 ms per 100k-row call)**, both emitted by
`../ruleset-compile-latency/emit.py` — imported, not duplicated, so the emitted shape is
identical to experiment G's. Every background compile uses the **same source**, so repeats
are the same work; `cache=False` means each `exec` produces a fresh dispatcher and a real
compile. Swap-phase generations are two tiny kernels that write their generation id into
the high digits of every output element, so straddling is detectable per row.

## Results (Python 3.14.5 GIL-enabled, numba 0.67.0, 28 cores, switchinterval 5 ms)

**1–3. Background compile vs. serving** (5.00 s compile, alone; medians of 2 runs)

| serving kernel | alone | during bg compile | throughput retained | p95 | max | bg compile |
|---|---|---|---|---|---|---|
| `nogil=True`  | 1.18 ms, 796 call/s | **6.41 ms** | **26 %** | 6.87 ms | 98.6 ms | 5.14 s (1.03×) |
| `nogil=False` | 1.16 ms, 844 call/s | 1.24 ms | **55 %** | 6.56 ms | 99.5 ms | 12.0 s (2.40×) |

Compilation in a background thread **works and is correct**: no exception, and the
background-compiled kernel's output is bit-identical to the same source compiled
single-threaded (`bg_output_matches_singlethreaded: true`, 4/4 runs).

**3b. The nogil penalty is a GIL hand-off, and it is a fixed cost per invocation**

| | baseline | during compile | added per call |
|---|---|---|---|
| 100k rows, switchinterval 5 ms (default) | 1.16 ms | 6.40 ms | **+5.24 ms** |
| 100k rows, switchinterval 0.5 ms | 1.16 ms | 1.84 ms | +0.68 ms |
| 100k rows, switchinterval 0.05 ms | 1.18 ms | 1.33 ms | +0.15 ms |
| **1M rows**, switchinterval 5 ms | 12.52 ms | 17.91 ms | **+5.39 ms** |

The added cost tracks the switch interval, not the batch size: a `nogil` kernel drops the
GIL, the compiler thread grabs it, and the serving thread waits a full switch interval to
get it back for the next dispatch.

**3c. numba's compiler is globally serialised**

| | |
|---|---|
| small (3-rule) compile alone | 0.483 s |
| same compile started 0.5 s into a 5 s background compile | **5.08 s (10.5×)** |
| two 10-rule compiles, sequential | 2.52 s |
| the same two, one thread each | 2.43 s (**speedup 1.04×**) |

**4. Atomic swap** — 4 worker threads, 200k-row batches (0.54 ms), swapper flipping
generations every ~0.26 ms (≈11.6k swaps in 3 s):

| variant | batches | straddled |
|---|---|---|
| pointer read **once** per invocation | 1564 | **0** |
| control: pointer re-read per 20k-row chunk | 1578 | **1576** |

The control is there because a test that cannot fail proves nothing (cf. doc 05 §9's
criterion-8 timing assertion). It straddles 99.9 % of batches, so the clean run's zero is
a real result. `activate()` median **0.18 µs**, p99 0.36 µs.

**5. Memory** (own subprocess per arm, 30-rule generations, single sample)

| generations compiled | RSS, retained | RSS, each dropped after compiling |
|---|---|---|
| baseline (imports + 100k-row data) | 119.9 MB | 119.9 MB |
| 1 | 251.7 MB | 239.9 MB |
| 2 | 294.9 MB | 293.5 MB |
| 3 | 296.5 MB | 294.1 MB |

Retaining all three costs **2.4 MB more** than dropping each one — i.e. the marginal cost of
*holding* a generation is below the noise of *compiling* one. Dropping back to one
generation reclaimed **0.0 MB**: the allocator keeps the high-water mark.

**6. Rollback** — `rt.rollback()` took **3.4 µs**, triggered **0** numba compile events
(`numba.core.event` recorder), and the next batch was served by the previous generation.

**Fallback: compile in a child process, hand over via numba's file cache**

| | |
|---|---|
| child compile (30 rules, `cache=True`, written to a pinned dir) | 5.4 s |
| serving latency in the parent during it | 1.17 ms vs 1.16 ms baseline (**1.002×**), max **1.85 ms** |
| parent adopting the result | **10.2 ms**, `cache_hits=1, cache_misses=0` |
| output checksum, child vs parent | identical |

## Verdict

Doc 08 §4 is implementable, but **not with a compiler thread**. Properties 1, 4 and 5
(read-once pointer, explicit activation, free rollback) hold as written and are cheap.
Property 3 ("compilation happens in a worker") is what breaks: an in-process worker thread
costs 45–74 % of serving throughput, adds ~5 ms to every `nogil` invocation, and blocks any
*other* compile — including the serving path's own lazy compile for a new dtype — for the
full duration behind numba's global compiler lock. A child process removes all of it at the
cost of depending on the file-backed cache, whose survival conditions experiment F
enumerates.

## What this does NOT measure

- Only one background compile shape (30 rules / ~490 lines). Longer compiles change the
  *duration* of the interference, not its per-call size, but that is an inference.
- `prange` / threading-layer kernels. The serving kernel is single-threaded `njit`.
- Free-threaded (3.14t) CPython — this box's interpreter has the GIL enabled.
- Single sample per memory arm; RSS is a coarse instrument and the numbers are dominated by
  the compiler's allocation high-water mark.
