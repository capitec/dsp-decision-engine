# N5 — N4's unexplained tail, chased down

EXPERIMENTS.md §N4/M1 measured single-thread `score()` at p50 1037µs, p99.9
1723.6µs (8.6% of a 20ms budget), max 3840.5µs (19.2%); ruled out GC (1
collection in 30,000 calls, 0 correlation with slow calls); and named its own
gaps explicitly: **"Not tested... core-pinning as a way to reduce M1's
OS-jitter tail; a second, idle-machine run."** This experiment closes both,
plus instruments every call directly instead of guessing.

## Method

Reuses `experimentation/single-record-overhead/n1_overhead.py`'s `score()` /
`DRIVER` / `REQUEST_DICT` / `PARAMS_RAW` verbatim (imported directly — its
heavy sweep code is behind `if __name__`). Per call, `t0`/`t1`
(`time.perf_counter_ns`) wrap **only** the call being measured; the
instrumentation reads happen *outside* that window (before `t0`, after
`t1`), so latency isn't contaminated by the instrumentation cost itself:

- `resource.getrusage(RUSAGE_SELF)` deltas: `ru_minflt`/`ru_majflt` (page
  faults), `ru_nvcsw`/`ru_nivcsw` (voluntary/involuntary context switches)
- current CPU core, from `/proc/self/stat` field 39 ("processor") via a
  held-open fd + `lseek(0)`+`read` (2.6µs/call measured — `os.sched_getcpu()`
  does **not exist** on this Python 3.14 build, confirmed: `AttributeError`)
- `sys.getallocatedblocks()` delta, `gc.get_count()[0]` delta

Four configs (`tail_cause.py`, `contended_run.py`):

| config | n | condition |
|---|---|---|
| A_unpinned | 20,000 | steady state, ambient box load (unpinned) |
| B_pinned | 20,000 | pinned to core 2 via `os.sched_setaffinity` |
| C_noop_unpinned | 20,000 | identical loop/instrumentation, calling a no-op instead of `score()` |
| D_contended_unpinned | 8,000 | 14 of 28 cores held busy by bounded (`timeout`-capped), self-terminating busy-spin shell loops |

Plus `alloc_sweep.py`: pure-Python alloc+touch+free of a single buffer, 8
sizes from 4KB to 1MB (bracketing glibc's 128KB default `M_MMAP_THRESHOLD`),
n=3000/size — isolates the allocator hypothesis from everything numba/pydantic
does. Its own instrument is validated first: touching a fresh 64MiB buffer
moved `ru_minflt` by 16385 (expected 16384) — confirms the rusage reads are
sensitive, so a "0 faults" reading elsewhere is a real null result, not a
broken sensor.

## Results — see the parent repo's `decider2/docs/EXPERIMENTS.md` §N5 for the
full write-up with all four configs' percentiles, correlation coefficients,
and the doc-02/08 recommendation. Summary: **the tail correlates with
`ru_nivcsw` (involuntary context switches — OS scheduler preemption)**, up to
r=0.56 and 100% of the extreme (p99.9+) tail in the pinned config; page
faults, the allocator, and numba's dispatch cache are all directly measured
and ruled out (0 faults across 48,000+ real calls and a 24,000-call
allocation-size sweep; 1 registered kernel signature, unchanged after 5000
more calls). **Pinning to a single core made the tail worse, not better**
(max 4959µs vs 3008µs unpinned) — it removes the OS's ability to route the
thread away from contention on that one core. Under moderate external
contention (14/28 cores), the median roughly doubled but the max/tail-to-
median ratio actually *shrank* — load shifts the whole distribution up
together rather than blowing up the outlier multiplicatively, in the one
contention regime tested.

## Not tested (dropped for the ~12-minute budget)

- Full 28/28-core saturation, or cgroup/CPU-quota throttling (Kubernetes-
  style `cpu.max` exhaustion) — a harder-edged, periodic-freeze mechanism,
  materially different from raw contention, not exercised here.
- Why ~95% of individual top-1%-slow calls in the *unpinned, light-load*
  config (A) show **no** signal on any of the 7 captured metrics — the
  leading guess (hardirq/softirq handling not attributed to the process's
  own `nivcsw`/rusage counters) is reasoned, not measured; would need
  `perf`/`ftrace`-level tracing to confirm.
- `perf stat` / hardware counters (cache misses, TLB misses) as a
  cross-check on the "migration → cold cache" mechanism suggested by the 5
  raw migration events observed in config A.

## Run

```
/path/to/.venv/bin/python tail_cause.py       # configs A, B, C (~65s)
/path/to/.venv/bin/python alloc_sweep.py      # allocation-size sweep (~seconds)
/path/to/.venv/bin/python contended_run.py    # config D, launches/kills its own bounded stressors (~20s)
```

Writes `results.jsonl` (flushed per config), `results_summary.json`, one
`raw_<label>.npz` per config (per-call arrays for offline re-analysis),
`alloc_sweep_results.jsonl`/`alloc_sweep_summary.json`. `*.log` files are the
console transcripts these numbers are drawn from.
