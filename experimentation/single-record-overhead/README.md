# N1 — the single-record (`score()`) overhead budget

Doc 01 §6.1 reweighted the priority: the single-record path is **primary**
(most invocations send one record), its budget is **20–100 ms**, and the
compiled kernel answers one record in ~1 µs — ~0.005% of a 20 ms budget.
"Do not benchmark the kernel." The open question doc 01 §6.1 names but never
measures: **what fixed overhead does the framework add around it?**

Two files:

- `n1_overhead.py` — the main harness. Phase breakdown, total end-to-end
  (p50/p95/p99/max, GC on/off), allocation probe, width sweep.
- `followups.py` — a decomposition run *after* the main sweep showed two
  phases dominating, to find out why. Run `n1_overhead.py` first (it
  triggers the one-time kernel compile and defines everything `followups.py`
  imports); `followups.py` appends to the same `results.jsonl`.

```
/path/to/.venv/bin/python n1_overhead.py       # ~57s, writes results.jsonl + results_summary.json
/path/to/.venv/bin/python followups.py         # ~18s, appends to results.jsonl
```

`--quick` on `n1_overhead.py` runs a small smoke sweep (~15s).

## What decider2 actually costs, at 400 in / 633 out, N=1

Doc 02 §3.5: `Affordability.score(net_income=42000.0, ..., params=p)` — the
realtime path **bypasses polars entirely**, so unlike experiments A/J/J2
there is no polars-boundary phase here. Shape held identical to
experiments J/J2 (300 f8 + 80 i8 + 20 bool in; 380 f8 + 158 i8 + 95 bool
out), Python 3.14.5 / numba 0.67.0 / numpy 2.4.6 / pydantic 2.13.4, single
unloaded 28-core machine, nothing else running.

**Total, end to end, as actually written** (12,000 calls, GC enabled — the
default a service would run under):

| | absolute | % of 20 ms budget | % of 100 ms budget |
|---|---|---|---|
| p50 | 971.4 µs | **4.86%** | 0.97% |
| p95 | 1028.4 µs | 5.14% | 1.03% |
| p99 | 1048.2 µs | 5.24% | 1.05% |
| max | 1131.4 µs | 5.66% | 1.13% |

GC disabled gave the same p50 (969.98 µs) and a *worse* max: **2844.6 µs
(14.2% of the 20 ms floor)**, a single outlier in 12,000 calls with GC
off — so it was not a GC pause; more likely OS scheduling jitter. One
occurrence in one run; flagged, not confirmed as systematic (would need a
repeat run this budget didn't have room for).

**GC never collected during either 12,000-call run** (`gen0/1/2` deltas all
0). Not because allocation is low (see below, ~50 KB/call) but because
nothing allocated here forms a reference cycle — CPython's refcounting frees
it immediately, so the generational collector never has anything to do. GC
on/off makes no measurable difference to this path.

### Phase breakdown (p50, 8,000 reps/phase, as originally written)

| phase | p50 | % of 20 ms | share of total |
|---|---|---|---|
| 1. accept (`**kwargs` unpack, 400 keys) | 19.18 µs | 0.096% | 2.0% |
| 2. validate params → NamedTuple bundle | 5.84 µs | 0.029% | 0.6% |
| 3. **marshal → 1-row record array (400 fields)** | **220.39 µs** | **1.10%** | **22.7%** |
| 4. kernel dispatch (reference only, not the finding) | 39.89 µs | 0.199% | 4.1% |
| 5. **readback → 633 Python scalars** | **673.28 µs** | **3.37%** | **69.3%** |
| 6. assemble response dict | 0.35 µs | 0.0017% | 0.04% |

Phases 3 and 5 are **92% of total framework cost**. Phase 4 ("kernel
dispatch") looked disproportionately large too (39.9 µs, vs the ~1 µs doc 01
attributes to the kernel) — `followups.py` shows why.

### Why marshal and readback dominate — and why "kernel dispatch" looked 40× too high

`followups.py` decomposes both:

| what | p50 | note |
|---|---|---|
| `np.empty(1, dtype=<633-field record>)` — alloc only | 36.90 µs | **this is the entire "kernel dispatch" cost** |
| actual njit call, output buffer **pooled** (no per-call alloc) | 1.38 µs | matches doc 01's ~1 µs / E9's 0.99 µs / J's 0.94 µs — **confirmed** |
| `np.empty` for 3 **flat** 1D arrays (380+158+95 elements) | 1.42 µs | 26× cheaper than the structured-dtype alloc, same element count |
| `np.empty(1, dtype=<400-field record>)` — input alloc only | 21.57 µs | same pattern, scales with field count (~0.055 µs/field either side) |
| readback, 633 fields, one at a time (`row[name].item()`) | 671.04 µs | as originally written |
| readback, whole row at once (`row.item()` → tuple, zip with names) | 54.98 µs | **12.2× faster**, same result (`assert`-verified equal) |
| readback, 3 flat arrays via `.tolist()` + zip | 47.65 µs | 14× faster |
| marshal, 400 fields one at a time | 225.94 µs | as originally written |
| marshal, one tuple assigned to the whole row | 64.68 µs | **3.5× faster**, same result (`assert`-verified equal) |

**The "kernel" was never the cost.** What doc 01 §6.1 calls "~1 µs" is
confirmed exactly (1.38 µs, matching E9's 0.99 µs and J's 942 ns) once the
per-call `np.empty()` for the *output* record array is taken out of the
loop. That allocation call — not numba, not the compiled kernel — was 95%
of what phase 4 measured. And the true dominant costs, phases 3 and 5, are
not inherent to the record-array *convention* either: they are the cost of
**per-field, string-keyed access on a many-named-field numpy structured
scalar**, done 400 and 633 times respectively. The same data, accessed in
bulk (one tuple in, one tuple out), costs 3.5–14× less — with **no loss of
the named-field record type** the rest of the framework (doc 05 §3) is built
around; only the *marshal/readback loop* changes, not the wire shape.

**Reconstructed total if both loops are written the bulk way** (accept,
validate and assemble left exactly as they are — only marshal and readback
change, and output-buffer pooling removes the allocation phase 4 exposed):

```
accept 18.3us + validate 5.5us + marshal(tuple) 64.7us + dispatch(pooled) 1.4us
  + readback(bulk) 55.0us + assemble 0.35us = 145.2 us  ->  0.73% of a 20 ms budget
```

versus **971.4 µs / 4.86%** as originally written — a **6.7× reduction**,
recorded by `followups.py` as `followup_optimized_total_estimate` in
`results.jsonl`.

### Allocation (tracemalloc + `sys.getallocatedblocks()`, 300 single-call samples)

~49,764 bytes and ~215 net allocated blocks per call, as originally written.
That is dominated by the 400+633 = 1033 Python scalar objects the per-field
marshal/readback loops box and unbox one at a time, plus the two structured
records. It did not translate into GC pressure (see above), but it is
consistent with — and further explains — why the bulk variants above are
faster: fewer intermediate Python objects, not just fewer field-name
lookups.

### Width sweep — accept + marshal only, 10/50/100/400 inputs (not recompiled per width — see below)

| inputs | accept p50 | marshal p50 | combined | % of 20 ms |
|---|---|---|---|---|
| 10 | 0.79 µs | 7.12 µs | 7.91 µs | 0.040% |
| 50 | 3.08 µs | 31.72 µs | 34.80 µs | 0.174% |
| 100 | 4.87 µs | 58.49 µs | 63.36 µs | 0.317% |
| 400 | 18.17 µs | 227.86 µs | 246.03 µs | 1.230% |

Roughly linear in width (both phases), as expected for a per-field loop —
consistent with the marshal/readback finding above: the cost is
per-field-touched, not fixed. **Deliberately not recompiled per width**: the
njit driver's line count is dominated by the fixed 633-line output body, not
by input width (it only ever reads `a0..a4, b0` regardless of how many input
fields exist), so a genuine per-width kernel would cost the same ~11 s
compile at every point and add nothing to the answer — this sweep is scoped
to what actually varies with width: the Python-side accept/marshal loops.

## Verdict

**Not "stop, it's under 1%."** As the calling convention doc 02 §3.5 and
doc 03 §4 actually specify would most naturally be implemented — kwargs in,
pydantic-validated NamedTuple params, a structured record array marshalled
and read back one named field at a time — **the framework costs ~1 ms, which
is 4.9–5.7% of the tightest (20 ms) budget and is a whole millisecond**, the
threshold this batch was told to flag. That is a **partial refutation** of
doc 01 §6.1's implicit claim that everything outside the ~1 µs kernel is
negligible.

**But it is refuted for a specific, fixable reason, not an inherent one.**
92% of the cost is two loops doing per-field named access on numpy
structured scalars 400 and 633 times. Replacing each with a single bulk
call — `rec[0] = tuple(...)` in, `row.item()` out — is measured **3.5–12×
faster on the exact same data** (assert-verified equal output), pooling the
output buffer removes the other 36 µs "kernel dispatch" was actually
spending on `np.empty()`, and the reconstructed total drops to **0.73% of
budget**. That bulk form is not obviously less readable than the per-field
version — arguably more so (`dict(zip(names, row.item()))` is one line, one
idea) — so this is not a maintainability-vs-speed tradeoff in the usual
sense: **the faster form is also the simpler one.**

**Recommendation for doc 05 §3 / doc 02 §3.5's realtime path implementation:**
marshal and read back the record by assigning/reading the whole row at once,
never field-by-field in Python, and reuse (pool) the output buffer across
calls rather than allocating a fresh many-field structured array per
request. The record *type* (doc 05 §3, doc 08's named-field convention) does
not need to change for this — only the two loops that cross it need to stop
touching it one field at a time. This is the maintainability-relevant
finding this batch's brief asked for: where two implementations differ,
recommend the one that reads better — here, the two happen to coincide.

Kernel dispatch itself is confirmed cheap (1.38 µs, matching E9/J) once
correctly isolated from the allocation that was hiding inside it — doc 01
§6.1's "~1 µs" figure for the compiled kernel stands.

## What was NOT tested (dropped for time, budget ~12 min)

- **A literal 400+1-named-parameter generated function**, which is what
  doc 02 §3.5's example signature (`score(net_income=..., expenses=..., ...,
  params=p)`) literally shows. Tested instead: `**kwargs`-unpack into a
  generic catcher, called with a 400-key dict — a proxy for the same
  unpack/bind cost without generating and compiling a 401-argument Python
  function. Accept measured cheap either way (19.2 µs / 0.096% of budget),
  so this is very unlikely to change the verdict, but it was not measured
  directly.
- Concurrency / multi-threaded tail behaviour (doc 06's N4) — this harness
  is single-threaded, one process, nothing else running.
- Repeating the GC-off 2.84 ms max spike to see if it reproduces.
- N2 (kwargs vs dict vs pre-built record, doc 06) and N3 (params validation
  cost at realistic param-set size, doc 06) beyond what N1 incidentally
  measured (accept-kwargs-vs-dict is in the phase table above; params
  validation is phase 2). Both remain open per doc 06.

## Reused from prior experiments (not rewritten)

- Shape constants (400 in / 633 out, same f8/i8/bool split) and the
  record-in/record-out njit driver codegen (`_HEADER`,
  `gen_record_out_source`) — verbatim from
  `experimentation/output-writeback-convention/writeback.py`.
- Writing the generated driver to a real `.py` file and importing via
  `spec_from_file_location` (never `exec()`, per doc 05 §4.1) — pattern from
  `ruleset-compile-latency/run.py:load_source`, reused again in
  `writeback.py:load_source`. Applied only to the njit driver; the
  orchestration code (accept/validate/marshal/readback/assemble) has no
  numba caching hazard, so it's defined directly in the script.
- `elapsed()`/`log()` progress-clock pattern —
  `prange-crossover/prange_crossover.py`.
- Peak-RSS-via-`resource.getrusage` pattern, renamed `rss_mb()` per this
  batch's brief — `chunked-writeback-at-scale/measure_variant.py:peak_rss_kb`.

**Not reused:** polars extraction (`Series._get_buffers()`, experiments
A/J/J2's `extract()`) — doc 02 §3.5 states the realtime path bypasses polars
entirely, so there is no polars boundary in this harness by design.

## Environment / memory

`free -g` before starting: 24 GB available (well over the 8 GB floor).
Single N=1 measurements throughout — peak RSS stayed at ~240 MB for the
whole run (numba + numpy + pydantic baseline plus one compiled driver), so
this did not need the `systemd-run`/detached-process treatment the memory
rules reserve for multi-GB sweeps.
