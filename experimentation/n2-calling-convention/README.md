# N2 — the `score()` calling convention at width

Doc 02 §3.5 and doc 03 §6 specify the realtime entry point as keyword
arguments: `Affordability.score(net_income=42000.0, expenses=18000.0,
instalment=3100.0, params=p)`. At 400 inputs (doc 01 §4d's verified
realistic width) that is 400 keyword arguments per request. Never costed
before this experiment — N1 measured everything *around* the calling
convention using a `**kwargs` catch-all as a cheap proxy and explicitly
flagged the literal signature as untested ("very unlikely to change the
verdict"). **It does change the verdict**, by close to an order of
magnitude — see below.

Run:

```
/path/to/.venv/bin/python n2_calling_convention.py            # full run, ~50s
/path/to/.venv/bin/python n2_calling_convention.py --quick    # smoke test, ~3s
```

Writes `results.jsonl` (flushed per record) and `results_summary.json`.
`run.log` in this directory is the full-run console output.

Environment: Python 3.14.5, numpy 2.4.6, pydantic 2.13.4, single unloaded
28-core machine, nothing else running. `free -g` before starting: 24 GB
available (well over the 8 GB floor) — no memory-sensitive sweep here, peak
RSS stayed at 48.4 MB for the whole run, so no `systemd-run` detachment was
needed.

## What decider2 actually costs, calling convention only

Four conventions, compared at 10 / 50 / 100 / 400 inputs, all-`f8` fields,
12,000 calls each (`timed_ns`, `time.perf_counter_ns`, 200-call warmup,
p50/p95/p99/max reported — at a 20 ms SLA the tail *is* the SLA). GC left at
its default (enabled) — N1 already established, on this exact class of
workload (structured records, no reference cycles), that GC on/off makes no
measurable difference, so that sweep was not repeated here to keep this
experiment inside its own time budget.

Each pipeline = **accept** (however the convention receives 400 values) +
**marshal-equivalent** (get those values into the 1-row record array a
kernel would consume). Kernel dispatch, output readback and response
assembly are **out of scope by design** — doc 01 §6.1 says not to benchmark
the kernel, and N1 already measured those phases exhaustively at this exact
width (dispatch pooled 1.38 µs, readback bulk 54.98 µs, assemble 0.35 µs).
"Combining with N1" below adds those back in for a holistic total without
re-running an ~11 s kernel compile.

| width | convention | p50 | p95 | p99 | max | % of 20 ms (p50) |
|---|---|---|---|---|---|---|
| 10 | kwargs (explicit signature) | 5.03 µs | 5.24 µs | 5.42 µs | 61.9 µs | 0.025% |
| 10 | dict | 3.28 µs | 3.55 µs | 3.79 µs | 12.7 µs | 0.016% |
| 10 | record, reused buffer | 1.96 µs | 2.13 µs | 2.24 µs | 55.1 µs | 0.010% |
| 10 | positional | 3.88 µs | 4.14 µs | 4.45 µs | 29.6 µs | 0.019% |
| 50 | kwargs | 32.6 µs | 35.0 µs | 39.3 µs | 97.9 µs | 0.163% |
| 50 | dict | 9.75 µs | 10.2 µs | 11.2 µs | 18.8 µs | 0.049% |
| 50 | record, reused buffer | 6.22 µs | 6.59 µs | 6.84 µs | 16.2 µs | 0.031% |
| 50 | positional | 14.3 µs | 16.7 µs | 20.0 µs | 47.2 µs | 0.072% |
| 100 | kwargs | 103.4 µs | 111.8 µs | 116.3 µs | 165.6 µs | 0.517% |
| 100 | dict | 18.5 µs | 19.7 µs | 24.3 µs | 59.5 µs | 0.092% |
| 100 | record, reused buffer | 12.1 µs | 12.5 µs | 17.2 µs | 64.9 µs | 0.060% |
| 100 | positional | 27.7 µs | 29.7 µs | 34.3 µs | 57.3 µs | 0.138% |
| **400** | **kwargs (as doc 02 §3.5 shows)** | **1190.2 µs** | **1268.4 µs** | **1312.3 µs** | **2816.5 µs** | **5.95%** |
| 400 | dict | 60.1 µs | 65.1 µs | 68.9 µs | 127.8 µs | 0.301% |
| 400 | record, reused buffer | 39.8 µs | 41.7 µs | 46.6 µs | 136.7 µs | 0.199% |
| 400 | positional | 93.3 µs | 99.4 µs | 102.9 µs | 195.9 µs | 0.467% |

**The headline: kwargs, called exactly as doc 02 §3.5 shows it, costs
1.19 ms p50 at 400 inputs — 5.95% of a 20 ms budget, on its own, before any
marshal, validation, kernel or readback work.** That is bigger than N1's
entire measured framework overhead for *everything else combined* (971 µs).
Every other convention is 12.7–29.8× cheaper at the same width (dict
60.1 µs, record 39.8 µs, positional 93.3 µs).

**And the cost is not linear in width — it is closer to quadratic.**
kwargs: 5.03 µs (W=10) → 32.6 µs (W=50, 6.5× for 5× width) → 103.4 µs
(W=100, 3.2× for 2× width) → 1190.2 µs (W=400, 11.5× for 4× width). Fitting
the four points gives an exponent of **~1.48** on the harness numbers; an
independent `timeit`-based cross-check (below) on the pure binding call
alone gives **~1.9**, i.e. close to O(n²). Every other convention scales
sub-linearly to roughly linearly with width (dict and record both show
worse-than-linear too, but far more mildly: dict is 18.3× for 40× width,
record 20.3× for 40× width — both sit around n^0.75, i.e. *below* linear,
likely dict/tuple construction amortising well).

## Isolating the cost: signature binding, not dict construction

The task's own question: *"is Python's signature binding the cost, or is it
dict construction?"* — @ width=400, 12,000 calls:

| case | p50 | % of 20 ms |
|---|---|---|
| explicit signature, called with kwargs, **body=`pass`** (pure binding) | 1094.2 µs | 5.471% |
| explicit signature, called with kwargs, body builds a dict | 1120.3 µs | 5.602% |
| **the same signature, called positionally**, body builds a dict | **28.8 µs** | **0.144%** |
| `**kwargs` catch-all (`def f(**kwargs): return kwargs`) | 17.9 µs | 0.089% |
| dict pass-through (`def f(d): return d`) | 0.159 µs | 0.0008% |
| record (numpy structured scalar) pass-through | 0.163 µs | 0.0008% |

**It is signature binding, not dict construction — decisively.** Body=`pass`
(1094.2 µs) and body=build-a-dict (1120.3 µs) differ by only 26 µs; building
the dict is a rounding error next to the binding cost. The same 400-parameter
function called **positionally** instead of by keyword — doing strictly more
work (bind *and* build a dict) — is **39× cheaper** (28.8 µs). CPython's
keyword-argument binding for a many-parameter function has to match every
call-site keyword name against the function's parameter names; that matching
is what costs ~1.1 ms at 400 keywords, not the values or the dict.

An independent cross-check (`timeit`, not this harness's `timed_ns`, run
directly from the shell, `number=20000`, body=`pass`, no dict at all)
confirms the shape and rules out a harness artifact:

```
width= 10  kwargs=    1.133us/call  positional=    0.116us/call  ratio=   9.7x
width= 50  kwargs=   19.438us/call  positional=    0.213us/call  ratio=  91.3x
width=100  kwargs=   78.172us/call  positional=    0.350us/call  ratio= 223.0x
width=400  kwargs= 1119.093us/call  positional=    2.209us/call  ratio= 506.7x
```

The ratio itself grows with width (9.7× → 506.7×), which is the signature of
super-linear (here, close to quadratic) growth in the keyword-binding path
specifically — on Python 3.14.5, the interpreter this project is pinned to.
No other Python version was available to test against (the memory/interpreter
rules for this batch prohibit installing anything), so this is reported as a
measured fact about the pinned interpreter, not a general CPython claim.

Also notable: `**kwargs` catch-all (17.9 µs) is **63× cheaper** than the
literal explicit-signature kwargs call (1120.3 µs) for equivalent captured
data — so N1's choice to proxy the accept phase with a catch-all, while an
explicit statement that it was a proxy (README said so), turned out to
understate the real cost of doc 02 §3.5's literal example by **~63×**. That
correction is recorded in `EXPERIMENTS.md` §N1 and in doc 01 §6.1.

## Validation cost (@ width=400, 12,000 calls)

The task's second follow-up: *"if `score()` checks that all required inputs
are present and typed, what does that cost per call, and can it be done once
rather than per call?"*

| case | p50 | % of 20 ms |
|---|---|---|
| pydantic dynamic model, 400 required `float` fields, `Model(**request)` | 55.0 µs | 0.275% |
| manual key-presence check (`required_keys <= request.keys()`) | 7.93 µs | 0.040% |
| manual per-field `isinstance` loop (400 checks) | 24.0 µs | 0.120% |
| no validation | 0.137 µs | 0.0007% |

**Cheap relative to the kwargs finding, not cheap in absolute terms** — full
pydantic input validation at 400 fields is 0.275% of budget, a third of the
manual-loop's already-small 0.120%, all well under the 1-whole-millisecond
flag threshold this batch was told to watch for.

**"Can it be done once rather than per call?" — only the schema, not the
data.** The `create_model(...)` call that builds the 400-field pydantic
class runs exactly once, outside the timed loop, in this harness and in any
real service (build it at import/warmup time, like doc 03 §4's params
model). What is measured above is only `Model(**request)` — validating each
record's *actual values* — which is inherently per-record: input data
differs call to call, so there is no "validate once" option for the data
itself, unlike doc 01 §6.1's params bundle (one bundle per config, not per
record). Note `Model(**request)` itself pays the same kwargs-binding tax as
the score() call, at a much smaller scale (55.0 µs vs the 400-field
score-signature's 1120 µs) — because pydantic's `__init__` also does
keyword-argument binding, just over fewer, smaller checks per field than
the raw interpreter loop this harness isolated above; this experiment did
not decompose pydantic's own internals further (out of scope, and small
either way).

## Combining with N1: a full `score()` total, per convention, at width 400

N1 already measured, and found the *fastest correct* form of, everything
downstream of accept (validate 5.5 µs, marshal-the-bulk-way 64.7 µs, pooled
dispatch 1.4 µs, readback-the-bulk-way 55.0 µs, assemble 0.35 µs = 126.9 µs).
Substituting each convention's accept+marshal-equivalent cost from this
experiment's width-400 row for N1's original `**kwargs`-catch-all proxy
(18.3 µs) gives an honest end-to-end estimate per convention — **this row is
arithmetic composition of two experiments' numbers, not independently
re-measured as one call**:

| convention | accept + marshal (N2) | + N1's downstream (126.9 µs) | total | % of 20 ms |
|---|---|---|---|---|
| **kwargs, literal signature (doc 02 §3.5 as written)** | 1190.2 µs | 126.9 µs | **1317.1 µs** | **6.59%** |
| dict | 60.1 µs | 126.9 µs | 187.0 µs | 0.935% |
| record, reused buffer | 39.8 µs | 126.9 µs | 166.7 µs | 0.834% |
| positional | 93.3 µs | 126.9 µs | 220.2 µs | 1.101% |
| *(N1's original proxy, `**kwargs` catch-all)* | *18.3 µs* | *126.9 µs* | *145.2 µs* | *0.726%* |

**The literal kwargs convention alone costs more than N1's entire optimized
total for everything else in the framework, combined.** Every alternative
convention keeps the whole `score()` call under 1% of a 20 ms budget.

## Maintainability — the tradeoff the task asked to weigh

This is the rare case in this batch where speed and the doc's own stated
convention **do not coincide**, so the two criteria have to be weighed
against each other rather than just reported together:

- **kwargs reads best for a small, hand-written call** — `score(net_income=
  42000.0, expenses=18000.0, instalment=3100.0, params=p)` is exactly the
  kind of call a data scientist writes in a test or a notebook, and at 3–10
  named arguments its cost is irrelevant (µs, not ms). Doc 02 §3.5's example
  is written with 3 named args, not 400 — the example itself is not wrong,
  only its generalization to the framework's actual realistic width (doc 01
  §4d: 400 inputs) is.
- **At 400 inputs, nobody hand-types the call anyway.** A caller with 400
  real inputs already has them in a struct, a validated request body, or a
  DataFrame row — i.e., already has a `dict`-shaped or record-shaped object
  before calling `score()`. `score(**request)` at that point is not more
  readable than `score(request)`; it is the same data, spelled with 400
  extra `=` signs, and it is the spelling that costs 39× more to bind.
- **A pre-built, reused record buffer is the cheapest option (39.8 µs) but
  is the worst on the maintainability axis** — the caller must know the
  compiled record's exact field order/dtype and manage a mutable buffer's
  lifetime (thread-safety, staleness between calls, re-priming after a
  config swap per doc 08 §4). That complexity is worth paying only in a
  demonstrated hot loop (e.g., a tight scoring loop inside a batch-adjacent
  service), not as the default single-record entry point.
- **Positional (93.3 µs) is the worst idea on the maintainability axis
  regardless of speed** — 400 positional floats is a silent-transposition
  hazard (swap `net_income` and `instalment` and nothing raises), which
  matters more for a *credit-decision* framework than almost anywhere else.

## Recommendation for doc 02 §3.5 / doc 03 §6

**Specify `score(request: dict, *, params)` as the primary realtime
convention, not literal per-field keyword arguments.** Keep the kwargs
*spelling* available for small, hand-authored calls (tests, docs, a module
with genuinely few inputs) since it costs nothing there and reads best — but
the framework's own generated production signature, at the widths doc 01
§4d shows are realistic, should accept a mapping. Offer the pre-built,
reused record as an explicit **opt-in fast path** (documented separately,
not the default) for callers who have already profiled a hot loop and are
willing to own the buffer-lifetime complexity that goes with it — it is
1.5× cheaper than a dict (39.8 µs vs 60.1 µs) but that gap is 0.1% of budget,
not worth defaulting to for the 20× the caller-side awkwardness costs.
Validation (pydantic, `Model(**request)`, 0.275% of budget) is cheap enough
to keep on the request path unconditionally; it does not need an opt-out.

## What was NOT tested (dropped for the ~12-minute budget)

- **A GC on/off repeat for this specific workload.** N1 already established
  (same 12,000-call class of measurement, structured records, no reference
  cycles) that GC makes no measurable difference; not re-run here to keep
  this experiment inside its own budget rather than repeating a settled
  question.
- **Real kernel dispatch / output readback as part of this sweep.** Doc 01
  §6.1: "do not benchmark the kernel." N1 already measured that
  exhaustively at this exact width; this experiment's "Combining with N1"
  section adds those numbers back in arithmetically rather than
  re-measuring them, to avoid an ~11 s kernel recompile per run.
- **Mixed dtypes (f8/i8/bool) for the convention sweep.** All fields here
  are `f8`. N1 and experiment A already cover dtype-mix cost at the
  polars↔numba boundary and in the record convention; this experiment is
  scoped to the *calling convention*, which is dtype-independent (the
  400-vs-40 quadratic-looking blowup in kwargs binding does not depend on
  what type each keyword's value is).
- **Concurrency / multi-threaded tail behaviour** (doc 06's N4) — this
  harness is single-threaded, one process, nothing else running, same as
  N1.
- **Decomposing pydantic's own `__init__` internals** beyond the headline
  `Model(**request)` number — noted above as small (55 µs) and out of
  scope.
- **A second Python version to confirm the O(n²)-looking kwargs-binding
  cost is a CPython-general fact rather than specific to 3.14.5** — the
  memory/interpreter rules for this batch prohibit installing anything, and
  none was available to test against. Reported as a measured fact about the
  pinned interpreter (Python 3.14.5), not a general CPython claim.

## Reused from prior experiments (not rewritten)

- `elapsed()`/`log()` progress-clock pattern, `timed_ns()`/`pctiles()`/
  `pct_of_budget()` timing helpers, `jsonl_append()`, `rss_mb()` — verbatim
  from `experimentation/single-record-overhead/n1_overhead.py`, which
  itself reused `rss_mb` from
  `chunked-writeback-at-scale/measure_variant.py:peak_rss_kb` and
  `elapsed()`/`log()` from `prange-crossover/prange_crossover.py`.
- The "generate realistic values at a given width" idea, generalized from
  `n1_overhead.py`'s `width_split()`/`measure_width_sweep()` — this harness
  uses all-`f8` fields rather than N1's f8/i8/bool mix, since the question
  here is calling-convention overhead, not dtype handling (see "What was
  NOT tested").
- N1's own downstream-of-accept numbers (validate/marshal/dispatch/
  readback/assemble, bulk form) — reused arithmetically in "Combining with
  N1", not re-measured.

**Not reused:** polars extraction (doc 02 §3.5's realtime path bypasses
polars entirely, same reasoning as N1); the njit driver codegen from
N1/`writeback.py` (out of scope here — no kernel is compiled or dispatched
in this experiment).
