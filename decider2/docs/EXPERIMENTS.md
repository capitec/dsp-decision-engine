# Measured results — 2026-09-18/19

Twenty experiments run against real numba, on the project's own environment:
**Python 3.14.5, numba 0.67.0, llvmlite 0.49.0, numpy 2.4.6, polars 1.41.2,
pydantic 2.13.4** — which matches doc 01's stated benchmark environment on numpy,
polars and pydantic exactly (numba differs: 0.67.0 here vs 0.66.0 there).

Every harness is checked in under `experimentation/`, runnable, with a README.
That is deliberate: doc 01's fusion numbers existed only as a table, so nobody
could re-run them on a different workload shape — which is exactly the objection
that turned out to be right.

**Nine of eleven refuted or partially refuted a doc claim.** Five refutations
change a design decision rather than a number.

| # | experiment | verdict | harness |
|---|---|---|---|
| A | dtypes at the polars→numba boundary | **refuted** | `experimentation/dtype-boundary/` |
| B | per-node fallback inside an njit driver | **refuted** | `experimentation/objmode-fallback-blast-radius/` |
| C | numba cache survival | **partial** | `experimentation/numba_cache_survival/` |
| D | same-name NamedTuple dispatch trap | **refuted** | `experimentation/namedtuple-dispatch-trap/` |
| E | fusion vs body cost | **partial** | `experimentation/` (fusion) |
| I | numeric divergence | **partial** | `experimentation/numeric-divergence/` |
| E11 | `param()` in a step signature | **partial** | `experimentation/e11-param-in-signature/` |
| F | `prange` crossover (O6) | **refuted** | `experimentation/` |
| G | ruleset compile latency (O16) | **partial** | `experimentation/` |
| H | staged compile + atomic swap (doc 08 §4) | **partial** | `experimentation/` |

| J | output write-back convention | **partial** | `experimentation/output-writeback-convention/` |
| K | subprocess compile + cache handover | **partial** | `experimentation/subprocess-handover/` |
| L | rule constants as arguments | **partial** | `experimentation/` (rule-constants) |
| J2 | chunked write-back at scale | **partial** | `experimentation/` (chunked) |
| N1 | single-record (`score()`) overhead budget | **partial** | `experimentation/single-record-overhead/` |
| N2 | `score()` calling convention at width | **refuted** (doc 02 §3.5's literal kwargs example) | `experimentation/n2-calling-convention/` |
| N3 | params validation on the request path | **confirmed** (affordable as written) | `experimentation/params-validation-n3/` |
| N4 | tail latency, concurrency, config-swap impact | **partial** — swap impact confirmed negligible; concurrent-serving `nogil=True` requirement is new | `experimentation/n4-tail-concurrency-swap/` |
| M | the K/L contradiction | **resolved — both correct, different edit shapes** | `experimentation/kl-contradiction-resolved/` |
| N5 | N4's unexplained tail — root cause | **resolved — OS scheduler preemption (`ru_nivcsw`), not GC/allocator/dispatch; single-core pinning makes it worse** | `experimentation/n4-tail-cause/` |

**G, H and K together settle whether a configuration UI is viable: it is.** A rule
change lands in **2.5 s (10 rules) to 7.3 s (30 rules)**, end to end, with the
serving process retaining 97.9% throughput throughout.

> **Read all batch figures against doc 01 §6.1.** The single-record path is the
> primary one, with a 20–100 ms budget, and the compiled path runs at ~1 µs — so
> fusion, output conventions and chunking are **batch** concerns with four orders
> of magnitude of headroom at N=1. They should be chosen for maintainability
> unless a request-path measurement says otherwise.

---

## E — Fusion depends on body cost, and the stated mechanism is wrong

**The headline: a 62× spread from body cost alone**, at fixed module count and
fixed row count. `ratio = t_split / t_fused`, so **>1 means fusion wins**. All at
1 M rows:

| modules | trivial/straight | trivial/branchy | medium/straight | medium/branchy | heavy/straight | heavy/branchy |
|---|---|---|---|---|---|---|
| 1 | 0.99 | 0.98 | 0.99 | 0.89 | 0.98 | 0.94 |
| 2 | 2.07 | 1.34 | 1.90 | 1.01 | 1.02 | 0.82 |
| 3 | 3.26 | 1.30 | 3.06 | 0.93 | 0.76 | 0.56 |
| 5 | 5.18 | 1.15 | 2.20 | 0.76 | 0.46 | 0.49 |
| 10 | 11.18 | 0.80 | 1.93 | 0.63 | 0.42 | 0.37 |
| 20 | **13.00** | 0.62 | 1.42 | 0.56 | 0.22 | 0.34 |
| 40 | 11.20 | 0.55 | 1.02 | 0.43 | **0.18** | 0.31 |

**The sign of the decision flips.** On a cheap straight-line body, fusion wins by
13×. On a heavy one at the same module count, it loses by 5.6×. Doc 01 §4c's
table is a slice through one body-cost class.

### The mechanism claim is refuted — and so is the observable I proposed

Doc 01 §4b states: *"past ~3–4 branch groups the fused body **loses LLVM
auto-vectorisation** (vector IR values drop to 0 at M≥4)"*.

**In 54 fused kernels the packed-FP count never reached zero.** It grows
monotonically with M — in the *opposite* direction to performance. The worst
regression measured (heavy/straight, M=40, ratio 0.18) has a main loop of **791
packed ymm FP ops and zero scalar FP ops**. It is fully vectorised and 5.6× slower.

> **This invalidates a proposal added to doc 02 §1.2 during review.** That section
> recommended `explain_kernels()` report vectorisation status from
> `.inspect_llvm()`, on the grounds that it is "the measured mechanism behind the
> whole effect". It is not the mechanism, and it is not even correlated in the
> right direction. Withdrawn.

**What actually degrades is row-loop unrolling**, from per-loop disassembly:

| kernel | packed FP in main loop | distinct ymm | `%rsp` spills/iter | unroll | ratio @1M |
|---|---|---|---|---|---|
| heavy/straight split (M=1) | 55 | 7 of 16 | 0 | ~5× | — |
| heavy/straight fused M=4 | 71 | 14 | 0 | ~4× | 0.88 |
| heavy/straight fused M=20 | 391 | 16 (all) | 47 | ~2× | 0.22 |

The split kernel keeps ~5 independent row-groups in flight to hide 4-cycle FMA
latency, using 7 of 16 ymm registers and spilling nothing. The fused kernel pins
all 16, unrolls ~2×, and spills 47 times per iteration, so a long dependent FMA
chain goes **latency-bound**. Doc 01 §4c named the right *cause* ("register
pressure"); doc 01 §4b named the wrong *observable*.

### Two more doc figures that did not reproduce

- **"split kernels hold a flat 0.20 ns/step"** — measured 0.82–0.96 (trivial),
  1.56–2.38 (medium), 0.85–3.30 (heavy). Flat *within* a body-cost class, never
  near 0.20. **0.20 ns/step is below the floor of a single kernel call on this
  machine.**
- **"break-even is ~10k rows"** — there is no common break-even. `trivial/straight`
  never crosses (fusion wins 2.1–13× at every n ≥ 1000); `heavy/branchy` crosses
  below 1000 rows. And the bound is wrong in magnitude: at n=1 fusion wins
  **9.4–48×**, not "≤1.4×", because per-call dispatch (0.44 µs/kernel) dominates.

### The boundary cost, measured directly

One extra split kernel, empty body, dispatch subtracted:

| rows | dispatch | ns/row | implied GB/s (16 B/row) |
|---|---|---|---|
| 1,000 | 0.440 µs | 0.376 | 42.6 |
| 100,000 | 0.438 µs | 0.323 | 49.5 |
| 1,000,000 | 0.443 µs | 0.763 | 21.0 |

> **The discriminator is per-step body cost relative to the boundary floor**
> (~0.32–0.76 ns/row/kernel plus 0.44 µs dispatch) — not module count, not row
> count, not branch-group count. When a step's body costs less than the boundary,
> splitting pays that floor M times and fusion wins by roughly M. When bodies are
> expensive *and* form a long dependency chain, fusion loses to register pressure.
>
> Doc 01 §4c's *"boundary stores are genuinely near-free"* is the claim that fails
> first: they are near-free only relative to a body that costs more than they do.

**Consequence for doc 02 §1.2.** Keep split as the `apply()` default — it is the
safe side and the predictable one — but **drop its stated justification**, which
is not reproducible, and stop describing fusion as a loss above 10k rows. The
`fuse()` combinator becomes more important, not less: on a cheap straight-line
group it is worth up to 13×.

---

## A — Half a credit record cannot cross the boundary, and the recipe does not run

**Blocker: doc 05 §1.2's extraction recipe cannot execute.** `series.to_arrow()`
requires pyarrow, which is not a project dependency. All 26 dtype/nullability
combinations raise `ModuleNotFoundError`. This is "the first thing to build".

The polars-native replacement `Series._get_buffers()` needs no new dependency,
returns `{values, validity, offsets}`, and costs the same (~1.9 µs vs ~1.9 µs).
It is strictly better: pre-sliced validity, pre-split offsets, no 40 MB dep.

**Only 6 of 26 combinations are zero-copy** — clean Float64, Int64, Int32, UInt8,
Datetime, Duration. **Every null-bearing column copies** (~550–630 µs at 100k
rows, versus 0.8–1.2 µs clean).

| dtype | njit on `.to_numpy()` | njit on values buffer | note |
|---|---|---|---|
| Float64 / Int64 / Int32 / UInt8 | ok | ok | zero-copy when clean |
| Boolean | ok | ok | not zero-copy; **+null → `object`, njit FAILS** |
| Utf8 | **FAIL** | ok as uint8 bytes | 3 buffers, not 2; `_get_buffers` 750 µs |
| Categorical / Enum | **FAIL** | ok as codes | codes work; cross-frame stability must be checked |
| Date / Datetime | ok* | ok as int | *compares work, **arithmetic fails**: `add(datetime64[us], datetime64[us])` unsupported |
| Duration | ok | ok | add supported |
| **Decimal** | **FAIL** | **Rust PANIC** | see below |
| List(Float64) | **FAIL** | `_get_buffers` raises | `not implemented for dtype list[f64]` |

**The landmine — money.** `Decimal` does not raise a Python exception. It raises
`pyo3_runtime.PanicException`, which inherits **`BaseException`**, so
`except Exception` in an extraction layer will not catch it, and the process
prints an unsuppressable Rust panic to stderr:

```
thread '<unnamed>' panicked at crates/polars-python/src/interop/numpy/to_numpy_series.rs:122:5
```

An extraction layer must use `except BaseException` at the dtype gate. Money must
be a scaled integer, never `Decimal`, at the boundary.

---

## B — Per-node fallback cannot exist inside a fused driver, and `objmode` is worse than Python

Doc 05 §6 and the README promise *"anything numba can't compile degrades to plain
Python **per-node** rather than failing"*. **All six compile-failure modes took
down the entire driver.** Not once did one node degrade while siblings stayed
compiled.

200,000 rows × 4 steps, medians of 5, all variants bit-identical (max abs diff 0):

| variant | median ms | vs all-njit | ns/row |
|---|---|---|---|
| all-njit, serial | 7.13 | 1.0× | 36 |
| all-njit, `prange` | 1.01 | 0.14× | 5 |
| **whole driver in plain Python** | 167.26 | **23.5×** | 836 |
| **one node via `objmode`, per row** | 548.43 | **77.0×** | 2742 |
| one node via `objmode` inside `prange` | 4380.61 | **614.7×** | 21903 |
| regex node via `objmode`, per row | 720.60 | 101.1× | 3603 |
| regex node **hoisted** to one `objmode` call | 211.81 | 29.7× | 1059 |
| Python pre-pass column + pure njit driver | 222.86 | 31.3× | 1114 |

> **Falling back to pure Python is 3× faster than `objmode` per row.** The
> "graceful" mechanism is worse than the ungraceful one. Doc 05 §6 layer 1 must be
> deleted as written.

**And `objmode` inside `prange` compiles silently** and is bit-exact while being
615× slower — against a 7× speedup for the same driver fully njit'd. Any
"compile both variants and pick at warmup" policy (doc 02 §3.3) selects correctly
only if warmup measures the real body.

The correct shape: **hoist the escape out of the row loop** (29.7×) or pre-compute
the column in Python and keep the driver pure (31.3×). Both are ~2.5× better than
per-row escape and belong in the spec instead of layer 1.

---

## C — numba will serve stale compiled code

**The finding not in any doc and not in the review.** Change a numeric constant in
a generated driver, keep the file size identical, restore the mtime: numba reports
a **cache HIT and returns the pre-edit answer** (49.39775 instead of 49.29765).
The index key hashes `co_code`, which excludes `co_consts`.

Doc 05 §4.2 mandates byte-identical deterministic codegen and doc 02 §3.4 stakes
`decider2 build --verify` on "zero runtime compilations". **A build that normalises
mtimes for reproducibility converts a stale-constant bug into a silent
wrong-decision bug, and `--verify` reports success on it.**

Invalidation matrix at 200 args:

| scenario | bytes | stamp | result |
|---|---|---|---|
| untouched | same | same | **HIT** 0.183 s |
| byte-identical rewrite, same path | same | changed | **MISS** 4.67 s |
| identical rewrite + `os.utime` restore | same | same | **HIT** 0.177 s |
| touch only (bytes never changed) | same | changed | **MISS** 4.80 s |
| identical bytes, different dir | same | n/a | **MISS** 4.61 s |
| whole dir copied, mtimes preserved | same | same | **HIT** 0.171 s |
| same content, file **renamed** | same | same | **MISS** 4.80 s |
| comment line prepended (shifts `def` line) | changed | changed | **MISS** |

**Six conditions must all hold** for a build-time cache to survive to runtime:
same absolute directory, same basename, **same `def` line number** (it is in the
`.nbi` filename), exact `(st_mtime, st_size)`, same argument signature, same
`magic_tuple` (LLVM triple + CPU name + CPU features).

> These are the six **this** experiment could reach. §K later adds a seventh — the
> `sys.modules` registration name — found only once a real child/parent process
> boundary was crossed. Doc 05 §4.2 carries the full seven.

Confirmed, with better numbers than the doc's single point:

| n_args | cold (miss) | warm (fresh process) | speedup |
|---|---|---|---|
| 200 | 4.93 s | 0.177 s | 27.8× |
| 400 | 14.75 s | 0.179 s | 82.5× |
| 800 | 46.99 s | 0.197 s | 238.7× |

**Warm cost is flat at ~0.18 s regardless of driver size** (`import numba` alone
is ~0.40 s). And `exec` fails at *decoration*, not first call:
`RuntimeError: cannot cache function 'f': no locator available for file '<string>'`.

---

## D — The dispatch trap is real, but not as a performance problem

Doc 01 §4c calls it *"a silent, permanent performance trap… ~1 µs to **15–24 µs**
— permanently"*. On numba 0.67 it costs **1.03×**:

| scenario | µs/call | vs clean |
|---|---|---|
| clean, single class | 1.006 | 1.00× |
| same `__name__` + same fields (the trap) | 1.039 | 1.03× |
| different field names (the doc's fix) | 1.085 | 1.08× |
| notebook, 9 cell re-runs, id-derived name | 1.084 | 1.08× |

Across 40 independent trials, 39 fell in 0.84–1.08×. It does not persist either:
a clean driver measured 1.02× after a colliding one ran 2000+ times (independently
confirmed by E11).

**But the collision is genuine as a correctness defect** —
`compute_fingerprint` encodes only `__name__` + field names + field types.

> **Doc 05 §9 criterion 8 cannot close itself.** A timing assertion has *zero
> power*: the collision costs 1.03×, so no budget separates pass from fail, and
> shipping that test would mark the bug green. It must become an exact structural
> check, not a timing one.

---

## I — Two numeric divergences that hit money, and one doc claim that did not reproduce

**Integer overflow at realistic scale.** Fixed-point compound interest, rate scale
1e12:

| principal | Python | njit int64 |
|---|---|---|
| R50,000 | R168,120.73 | R92,003.27 |
| R100,000 | R336,241.93 | **−R90,971.75** |
| R500,000 | R1,681,211.86 | −R91,208.37 |

First divergence at **R27,431**. A sum-of-squares accumulator over loan amounts in
cents **wraps at 2,667 rows** — well inside one batch. numpy int64 wraps
identically (with a warning); Python does not.

**Rounding disagrees systematically, and the docs never mention rounding:**

| x | Python `round(x,2)` | njit | `Decimal` HALF_UP |
|---|---|---|---|
| 2.675 | 2.67 | **2.68** | 2.68 |
| 2.665 | 2.67 | **2.66** | 2.67 |
| 1.115 | 1.11 | **1.12** | 1.12 |
| 1234.565 | 1234.57 | **1234.56** | 1234.57 |
| 0.025 | 0.03 | **0.02** | — |

njit follows numpy; Python uses banker's rounding; neither matches `Decimal`
consistently. Each row is a cent on an instalment.

**Doc 02 §3.1's log-ULP claim did not reproduce.** *"1-ULP drift from `log` … was
observed"* — measured **0 of 2,000,000** values differing between njit `math.log`,
njit `np.log`, host `np.log` and CPython `math.log`; max 0 ULP. Still 0 with
`fastmath=True` on the log itself. That claim is cited as evidence for the middle
rung of the equivalence ladder; the rung may be right, but this evidence is not.

**The good news, and it settles the tolerance question:** exact equality across
the ladder **is** achievable. 100.00% bit-exact agreement between pure Python and
njit over a 16-term log/exp/sqrt/div chain, all three outputs, 20,000 rows — with
`fastmath` off. So doc 05 §9.1's "matching exactly" is the right acceptance
criterion, not a naive one.

**`fastmath` is the single flag that breaks it**, and it did not pay here: 1.09×
on that chain while making **46–73% of rows differ, by up to 17 ULP** — not the
1 ULP the docs describe. Doc 02 §3.3's "off by default" should harden: a kernel
that enables `fastmath` is **excluded from the exact-agreement assertion** and
must declare a tolerance.

---

## E11 — `param()` in a signature works

Doc 03 §4.4's mechanism is implementable. Four of five items pass outright.

**The sentinel default does not need stripping.** Emitted with the default, with
it stripped, and passed explicitly through a driver all produce an identical numba
signature `(float64, float64, float64)` and identical results. So doc 03 §4.4's
sentence *"the emitted step carries no default"* is an unnecessary requirement on
codegen and should be deleted.

The only failure mode is calling the dispatcher *without* the param:
`ValueError: Cannot determine Numba type of <class 'ParamSpec'>` — which is the
correct, loud behaviour.

**The risk that would have killed it did not materialise:** 20 steps each
declaring a param named `cap`, each generating its own bundle, did not blow up
dispatch — consistent with D.

---

## What to run next

**G and H first.** They gate the configuration-UI promise in doc 08 §4: whether a
rule set compiles fast enough to stage (O16), and whether numba can compile in a
background thread while another kernel serves. Neither has any measurement behind
it, and doc 08's whole lifecycle assumes both.

**Then F** (`prange` crossover, O6) — and note that E's result changes its shape:
if body cost is the discriminator for fusion, it is likely the discriminator for
`prange` too, and the two questions may share one answer.

**Then the mitigations** (see REVIEW.md): content-addressed driver naming and a
build manifest for C; rule constants as arguments rather than emitted literals,
which is where C's stale-cache hazard actually lands; a scaled-integer money
primitive and boundary-value corpus generation for I.


---

## G — A rule UI can promise ten seconds, up to about 35 rules

O16 resolves toward the good end: 30 rules is **~6 seconds**, not two minutes.

| rules | emitted lines | compile (first_match) | compile (all) | exec @100k |
|---|---|---|---|---|
| 3 | 65 | 0.63 s | 0.64 s | 1.8 / 4.0 ms |
| 10 | 168 | 1.37 s | 1.43 s | 1.8 / 10.1 ms |
| 30 | 517 | 5.96 s | 5.51 s | 1.8 / 38.4 ms |
| 40 | 642 | **10.86 s** | 9.44 s | 1.8 / 52.5 ms |
| 60 | 967 | 20.67 s | 17.82 s | 1.8 / 85.3 ms |
| 100 | 1617 | **56.56 s** | 48.08 s | 1.8 / 126.7 ms |

**The ten-second promise breaks at ~37 rules / ~600 emitted lines.**

### The cost model must change, not just its constant

Doc 01 §4b's *"≈15 ms per emitted source line is the single best predictor"* is a
single point on a rising curve. Compile scales **∝ lines^1.4** (R²=0.97), with the
local exponent reaching **1.96** between 60 and 100 rules. Marginal cost per added
rule: 106 ms at 10 rules, 229 ms at 30, 554 ms at 60, **897 ms at 100**.

A linear predictor tells a scheduler that 100 rules costs 24 s when it costs 57 s.
Doc 01's *"~500 lines ≈ 10 s"* guardrail survives — the real crossover is 600–710
lines, so it is slightly conservative, which is the right direction.

> **The lever is splitting, not shrinking.** Because compile is super-linear
> *within* a unit, four units of 25 rules cost roughly 4 × 4.5 s ≈ 18 s serial, or
> ~4.5 s across four workers — against **48–57 s** for one unit of 100. A `ruleset`
> that grows past ~30 rules should be split into several compilation units, and
> that is a framework decision rather than an authoring one.

### Two findings that change the UI design

**1. `all` compiles *faster* than `first_match`, while emitting *more* code.** At
100 rules it is 6% more source for 15% less compile time. The early-exit guard
chain is a compile-time tax — which inverts the intuition that less runtime work
means less compile work.

**2. Dead rules cost full compile time.** With every leaf disabled, LLVM proves
the whole loop body dead and execution collapses to 0.031 ms — but compile time is
**unchanged** (60 rules: 21.12 s dead vs 21.28 s live).

> A business user who **turns off** fifty rules rather than deleting them pays the
> full fifty-rule compile on every activation, forever, for code that never runs.
> A rules UI must therefore either delete on disable, or exclude disabled rules
> from emission — and doc 08 §3's interior schema needs an `enabled` flag whose
> semantics are "not emitted", not "emitted and skipped".

---

## H — The swap is solid; the "background worker" cannot be a thread

### Confirmed, and cheaply

| property | measurement |
|---|---|
| read pointer **once per invocation** | **0 straddled batches** of 1564, across 11,605 swaps in 3 s |
| control: re-read per 20k chunk | **1576 of 1578 straddled (99.87%)** |
| `activate()` latency | 0.177 µs median, 0.357 µs p99 |
| rollback | 3.36 µs, **0 numba compile events**, previous generation serves the next batch |
| memory, 3 generations resident | 296 MB vs 252 MB for one — 2.4 MB to retain all three |
| correctness of a background-compiled kernel | bit-identical, 4/4 runs |

The zero is a *measurement*, not an assertion — the deliberately-wrong control
fails 99.87% of the time. **That control is the regression test**, and note what it
models: re-reading the pointer inside a chunk loop is exactly what a polars-style
chunked `map_batches` apply would do.

### Refuted: compilation in a background *thread*

| serving kernel | alone | during background compile | throughput retained |
|---|---|---|---|
| `nogil=True` | 1.183 ms | 6.406 ms | **26%** |
| `nogil=False` | 1.163 ms | 1.242 ms | **55%** |

> **Releasing the GIL made serving worse**, which inverts the obvious reasoning. A
> `nogil` kernel must re-acquire the GIL between dispatches, and the compiler
> thread is Python-level and GIL-greedy, so every call waits a full switch
> interval. The GIL-holding kernel instead squats on the GIL and starves the
> compiler — background compile stretched 1.03× for `nogil=True` but **2.40×** for
> `nogil=False`.

The penalty is **a fixed per-invocation cost, not a proportional slowdown**:
+5.24 ms at 100k rows and +5.39 ms at 1M. It tracks `sys.setswitchinterval`, not
batch size — 5 ms → +5.24 ms, 0.5 ms → +0.68 ms, **0.05 ms → +0.15 ms (1.13×)**.
So exposure is set by **call granularity**: few large batches are nearly immune,
many small realtime calls are not.

**And numba's compiler does not parallelise.** Two 10-rule compiles in separate
threads: **1.036× speedup** over sequential. A 3-rule compile started 0.5 s into a
5 s background compile took **5.077 s instead of 0.483 s — 10.5×**.

> **Doc 08 §4's "(background)" must mean a subprocess, not a thread.** Measured
> subprocess compile of a 30-rule kernel: 5.36 s compile, 6.14 s wall, with the
> serving process untouched. The parent then loads from the pinned cache directory
> — which is exactly the six-condition cache contract in doc 05 §4.2, so the two
> mechanisms have to be designed together.


---

## J — Output write-back: column-major wins, but less than it first appears

At 100k rows, 400 inputs / 633 outputs mixed. Shared: extract 6.1 ms + assemble 335.8 ms.

| variant | compile | kernel | write-back | total | wb % | zero-copy |
|---|---|---|---|---|---|---|
| record → record *(doc 05 §3 as written)* | 11.31 s | 94.2 ms | **775.6 ms** | 1211.7 ms | **64.0%** | n/a |
| record → col-major 2D | **0.27 s** | 349.7 ms | **5.7 ms** | **697.3 ms** | 0.8% | **all True** |
| record → row-major 2D *(control)* | 0.25 s | 80.1 ms | 505.8 ms | 927.8 ms | 54.5% | all False |

**1.74× end-to-end**, write-back 775.6 → 5.7 ms. E9's 54.7% figure is confirmed and
is **64.0%** here.

**The control is what makes it interpretable.** Row-major 2D has the *fastest
kernel of the three* (80.1 ms) yet write-back stays at 505.8 ms and is not
zero-copy. So the win is specifically **column-major**, not "2D instead of
records" — and column-major genuinely pays for it in the kernel, 349.7 ms against
80.1 ms, from losing write locality.

Two things that matter more than the runtime:

- **Compile collapses 11.31 s → 0.25–0.27 s, 42×**, for *both* 2D forms. Against
  §G's compile ceiling and §K's config-UI latency, this is the larger result.
- **At N=1 the ranking inverts**: per-row kernel cost is row-major 800.8 ns,
  record 942.2 ns, col-major **3497.1 ns**. Column-major is 3.7× worse per record.
  It optimises the daily batch at the primary path's expense — though at a 20 ms
  budget the whole spread is 2.7 µs, i.e. 0.01%.

**This experiment OOM-killed the machine twice** (18.7 GB, 20.8 GB RSS) by holding
all three output variants *and* three result frames live for an equality check.
The harness was the bug, not the design — but see §J2, because the underlying
memory requirement is real.

---

## K — The subprocess lifecycle works; two new failure modes

**Doc 08 §4's mechanism is confirmed.**

| | thread (§H) | **subprocess (K)** |
|---|---|---|
| serving throughput retained | 26–55% | **97.9%** |
| parent-side numba compile events | — | **0** |
| parent load time | — | 9.3 ms |
| child/parent checksums | — | bit-identical |

**Wall clock, config change to serving:** 10 rules **2.56 s** (compile 1.91 s),
30 rules **7.34 s** (compile 6.66 s). That is the number a configuration UI can
honestly promise, and it corroborates §G's "≤10 s up to ~35 rules".

### Most cache-condition violations fail safely

| child/parent disagree on | result |
|---|---|
| filename | `FileNotFoundError` — **loud** |
| cwd (relative path) | `FileNotFoundError` — **loud** |
| `NUMBA_CACHE_DIR` | silent MISS → recompiles — **safe** |
| `def` line number, mtime/size frozen | silent MISS → recompiles — **safe** |
| **`sys.modules` registration name** | `ModuleNotFoundError('<dynamic>')` from inside `pickle.loads` — **loud but cryptic** |

That last row is a **seventh condition** §C never found, because §C's harness always
registered the module and never varied it. The failure surfaces deep in numba's
`Environment` rebuild with no hint that a caching contract was violated.

**Only the mtime/size coincidence produces a silent wrong answer.** The other
violations degrade to a recompile, which is slow but correct — a materially better
risk profile than §C implied.

---

## L — Rule thresholds should be arguments, for a different reason than assumed

| measure | literal | argument |
|---|---|---|
| runtime, 10 rules @1M | 39.92 ns/row | 41.16 ns/row (**+3.1%**) |
| runtime, 30 rules @100k | 40.05 ns/row | 44.53 ns/row (**+11.2%**) |
| emitted lines, 3/10/30 rules | 11 / 25 / 65 | **identical** |
| compile time | 0.475 / 0.444 / 1.081 s | 0.297 / 0.517 / 1.325 s — **no consistent sign** |
| **8 threshold retunes** | **8 full recompiles, 343 ms each** | **0 compile events, signatures 1→1** |

**Refuted:** the implied compile-time benefit. Hoisting thresholds does *not*
reduce emitted lines (identical at every rule count) or reliably reduce compile
time. Doc 08 §2 should not claim it.

**Confirmed, and it is the whole case:** a threshold retune never recompiles.
343 ms per retune at *five* rules, and §G's lines^1.4 scaling makes that far worse
at thirty.

**The price is 1–4.5 ns/row** — which under doc 01 §6.1's priority is
**~4.5 nanoseconds against a 20 ms single-record budget**. Free at N=1, ~3–11% on
the daily batch. Adopt it.

**Rule enablement as a mask array is also nearly free:** −2.0% at 10 rules (noise,
net faster) and +5.4% at 30. Break-even against a *single* literal-form recompile
is ~427 million rows. So §G's "disabled rules cost full compile time" is solved by
a mask, decisively.

---

## M — The K/L contradiction, resolved: both are right, about different edit shapes

Both ran a controlled A/B on the stale-constant hazard and **got opposite results**:

- **K:** deleting CPython's `__pycache__/*.pyc` alone fixed it — correct value served,
  `STALE_CAUSED_BY_CPYTHON_PYC=True`. Attributes the hazard to CPython's bytecode
  cache, which is *also* keyed on (mtime, size) and is consulted before numba's
  decorator runs.
- **L:** with the `.pyc` cleared and numba's `.nbi`/`.nbc` untouched, it **still
  served the stale value** — `numba_cache_independently_stale=True`. Attributes it
  to numba's own cache, since the edit changes `co_consts` without changing
  `co_code`'s `LOAD_CONST` operand index.

**Settled by `experimentation/kl-contradiction-resolved/`.** Neither harness had a
bug — both cleared exactly what they said they cleared. A clean 2×2, run with real
**subprocesses** (§K's own methodology, matching doc 08 §4's real compile/load
process boundary — not §L's in-process `spec_from_file_location` reload) and
module-name import (doc 05 §4.1's fix), on §L's single-occurrence-threshold driver:

| arm | `.pyc` cleared | numba cache cleared | served | truth | stale? |
|---|---|---|---|---|---|
| both live | no | no | 0.1 | 0.15 | yes |
| **`.pyc` only cleared** | **yes** | no | **0.1** | 0.15 | **yes** |
| numba cache only cleared | no | yes | 0.1 | 0.15 | yes |
| both cleared | yes | yes | 0.15 | 0.15 | no |

The `.pyc`-only-cleared row is decisive: the child process's own dump of
`co_consts` after that run shows `(0.595, ...)` — CPython genuinely re-parsed the
*edited* source — yet the value served was still the pre-edit `0.1`. Only numba's
own on-disk cache (left untouched in that arm) could have produced that.

**Why K never saw this:** K's driver (`gen_driver.py`) shares one literal
(`0.5`) across every branch; its same-byte-length edit changes only *one* of the
16 occurrences, and the value is still needed at the other 15. Direct `co_consts`
comparison (`verify_mechanism.py`) shows this is not a value-for-value swap: it
**inserts** a new slot (`0.7`) while **keeping** the old one (`0.5`), which shifts
every downstream `LOAD_CONST` operand — `co_code` changes, and numba correctly
misses. Reproducing K's *actual* edit confirms it: `co_code` sha256 changes.
Reproducing L's edit (the literal has no other occurrence — the real shape of a
single rule's threshold) on the same machine confirms the opposite: `co_code`
sha256 is byte-identical before and after. **Both attributions are correct; they
are about edits with different `co_consts`-collision shapes, and K's driver
structurally could not produce the shape that exposes numba's own cache.**

**Practical consequence, now settled rather than assumed: content-addressed
filenames kill both layers at once**, confirmed with *neither* cache cleared
(`content_addressed.py`): an unchanged "redeploy" (same hash, same path, write
skipped) still hits honestly and correctly; an edited one gets a structurally
different path and recompiles correctly. This was always the fix either way, and
now rests on a mechanism, not just an agreement between the two attributions.

---

## J2 — Chunking is mandatory, and the 1.74× does not survive it

**Peak RSS, unchunked, one variant per process:**

| rows | record | col-major |
|---|---|---|
| 100k | 1.77 GB | 1.33 GB |
| 250k | 4.13 GB | 3.06 GB |
| **1M (fitted)** | **16.19 GB** | **11.87 GB** |

Largest batch fitting a 6 GB cap: record ~389k rows, col-major ~534k. **So a 1M-row
batch at this width does not fit, for either convention** — chunking is not an
optimisation, it is a requirement, and doc 05 has no chunking section at all.

**Chunked, 1M rows total:** peak RSS is **bounded by chunk size and flat against
total rows** — 0.45/1.05/1.79/4.02 GB (record) at 10k/50k/100k/250k chunks. Column-
major output survives chunking and stays bit-identical: **134/134 chunk checksums
match**.

> **Default chunk size: 100,000 rows** — the low end of the wall-time-optimal
> 50k–100k band for both conventions, with ~4× the memory headroom of a 250k chunk
> on a machine that must serve other work.

**And an honesty correction to J.** End-to-end wall-time ratio under chunking is
**1.03×–1.57×, never 1.74×**, because per-chunk input assembly and output
persistence — costs the convention does not touch — are **58–82% of wall time**.
On the portion the convention *does* control the win actually *grows* with chunk
size, reaching **3.09×** at 250k. Doc 05 must state the measured range, not a flat
1.74×.

### A direct correction to doc 05 §4.1

**The numba on-disk cache does not survive across fresh processes when the
generated driver is imported with `spec_from_file_location`** — the pattern doc 05
§4.1 recommends — even with a byte-identical file and preserved mtime. Importing
**by module name** instead took the driver from **12.1 s cold to ~0.2 s** on every
subsequent process. This compounds §K's seventh condition: both are about module
identity, and doc 05 §4.1's recommended pattern gets it wrong.

---

## N1 — The single-record overhead budget: not negligible as written, but fixable to negligible

Doc 01 §6.1 names the single-record path as primary (20–100 ms budget) and the
compiled kernel as ~1 µs, then states the surrounding framework cost "has not been
measured." This is that measurement: `Affordability.score(...)` at 400 in / 633
out, N=1, doc 02 §3.5's calling convention (kwargs, no polars), six phases each
timed, `experimentation/single-record-overhead/`.

**Total, as the doc's own conventions most naturally imply an implementation**
(kwargs → pydantic-validated NamedTuple params → 1-row record array, field-by-field
in and out — doc 03 §4's params bundle, doc 05 §3's record convention), 12,000
calls, GC enabled:

| | absolute | % of 20 ms | % of 100 ms |
|---|---|---|---|
| p50 | 971.4 µs | **4.86%** | 0.97% |
| p95 | 1028.4 µs | 5.14% | 1.03% |
| p99 | 1048.2 µs | 5.24% | 1.05% |
| max | 1131.4 µs | 5.66% | 1.13% |

**That is a whole millisecond — the threshold this batch was told to flag — and a
partial refutation of the implicit "everything outside the kernel is free"
reading of doc 01 §6.1.** GC made no measurable difference (p50 969.98 µs
disabled; 0 gen0/1/2 collections in either 12,000-call run — nothing here forms a
reference cycle, so refcounting frees it before GC would ever run). GC-off did
produce one 2.84 ms outlier (14.2% of budget) in 12,000 calls — flagged as a
single occurrence, not confirmed systematic.

**Phase breakdown says why: two phases are 92% of the cost, and neither is the
kernel.**

| phase | p50 | % of 20 ms |
|---|---|---|
| 1. accept (`**kwargs`, 400 keys) | 19.18 µs | 0.096% |
| 2. validate params → NamedTuple | 5.84 µs | 0.029% |
| 3. **marshal → 1-row record (400 fields)** | **220.39 µs** | **1.10%** |
| 4. kernel dispatch | 39.89 µs | 0.199% |
| 5. **readback → 633 Python scalars** | **673.28 µs** | **3.37%** |
| 6. assemble response dict | 0.35 µs | 0.0017% |

**And phase 4 itself is a decoy.** Decomposing it: the njit call alone, with the
output buffer pooled instead of freshly allocated, is **1.38 µs** — matching E9's
0.99 µs and J's 942 ns exactly, so doc 01 §6.1's "~1 µs" kernel figure is
confirmed. The other **36.9 µs was `np.empty()`** allocating the 633-*named-field*
output record — not numba, not the kernel. Allocating the same 633 elements as 3
flat positional arrays instead costs **1.42 µs**, 26× less, for the identical
element count.

**Phases 3 and 5 are the same story.** Both loop over field *names* one at a time
(`rec[0][name] = value`, `row[name].item()`). Reading or writing the *same data*
in one bulk call — `rec[0] = tuple(...)`, `dict(zip(names, row.item()))`,
assert-verified to produce identical output — costs **3.5–12× less**:

| operation | per-field (as written) | bulk (whole row) | speedup |
|---|---|---|---|
| marshal (400 fields) | 220.39–225.94 µs | 60.83–64.68 µs | 3.5× |
| readback (633 fields) | 671.04–673.28 µs | 54.98–58.97 µs | 11.4–12.2× |

**Reconstructed total with both loops written the bulk way and the output buffer
pooled** (accept/validate/assemble untouched): **145.2 µs — 0.73% of a 20 ms
budget**, a 6.7× reduction from the as-written 971.4 µs / 4.86%.

> This is the maintainability finding this batch asked for, and unusually the two
> criteria do not trade off: `dict(zip(names, row.item()))` is not just faster
> than the per-field dict comprehension, it reads as one idea instead of a loop.
> **Doc 05 §3's record convention does not need to change** — only the
> marshal/readback loops that cross it, from per-field to whole-row, plus pooling
> the output buffer instead of allocating a fresh many-field record per call.

Allocation, as written: ~49.8 KB and ~215 net blocks per call (tracemalloc +
`sys.getallocatedblocks()`, 300 samples) — consistent with 1,033 Python scalars
boxed/unboxed one at a time across the two per-field loops, but this did not
translate into GC pressure (see above).

**Width sweep** (accept + marshal only, not recompiled per width — the kernel's
line count is fixed by the 633-line *output* body regardless of input width, so a
per-width kernel would add an ~11 s compile at every point for no new information):
roughly linear in width, as expected for a per-field loop — 7.9 µs combined at 10
inputs, 246.0 µs at 400 (0.040% → 1.230% of a 20 ms budget).

**Not tested (dropped for the ~12-minute budget):** a literal 400/401-named-
parameter generated function matching doc 02 §3.5's example signature verbatim
(tested instead: `**kwargs`-unpack into a generic catcher, a proxy for the same
bind cost); concurrency/tail-under-load (doc 06's N4); repeating the GC-off outlier
to see if it reproduces; N2 and N3 beyond what N1 incidentally covered (kwargs vs
dict is in the phase table; params validation is phase 2).

> **§N2 correction: that proxy understated the real cost by ~63×.** The
> `**kwargs`-catch-all measured here (19.18 µs) is a *generic* capture
> (`def f(**kwargs)`), not doc 02 §3.5's literal per-field signature
> (`def score(net_income, expenses, ...)`, 400 named parameters, called with
> keywords). §N2 measured the literal form directly: **1190.2 µs at width
> 400 — 5.95% of a 20 ms budget, on its own** — because CPython's
> keyword-argument *binding* (not dict construction; isolated separately)
> scales close to quadratically with parameter count for a many-parameter
> signature. N1's "very unlikely to change the verdict" note about this gap
> was wrong; see §N2.

---

## N2 — The `score()` calling convention at width: kwargs is refuted, not the doc's example — its generalization to 400 inputs

Doc 02 §3.5 / doc 03 §6 specify the realtime entry point as keyword arguments:
`Affordability.score(net_income=42000.0, expenses=18000.0, instalment=3100.0,
params=p)`. At the realistic width doc 01 §4d already established (400 inputs),
that is 400 keyword arguments per request. `experimentation/n2-calling-convention/`,
12,000 calls per measurement, Python 3.14.5, single unloaded 28-core machine.

**The headline: kwargs, called exactly as doc 02 §3.5 shows it, costs 1.19 ms p50
at 400 inputs — 5.95% of a 20 ms budget, on the calling convention alone, before
marshal, validation, kernel or readback.** That is bigger than §N1's *entire*
measured framework overhead for everything else combined (971 µs). Every other
convention is 12.7–29.8× cheaper at the same width:

| width | kwargs | dict | record (reused buffer) | positional |
|---|---|---|---|---|
| 10 | 5.03 µs (0.025%) | 3.28 µs (0.016%) | 1.96 µs (0.010%) | 3.88 µs (0.019%) |
| 50 | 32.6 µs (0.163%) | 9.75 µs (0.049%) | 6.22 µs (0.031%) | 14.3 µs (0.072%) |
| 100 | 103.4 µs (0.517%) | 18.5 µs (0.092%) | 12.1 µs (0.060%) | 27.7 µs (0.138%) |
| **400** | **1190.2 µs (5.95%)** | 60.1 µs (0.301%) | 39.8 µs (0.199%) | 93.3 µs (0.467%) |

(p50s, % of a 20 ms budget in parens; full p95/p99/max in the harness README.)

**And it is not linear in width — it is close to quadratic.** kwargs: 5.03 µs →
32.6 µs → 103.4 µs → 1190.2 µs at 10/50/100/400 (implied exponent ≈1.48 on these
four points). Every other convention scales sub-linearly to mildly super-linearly
(dict and record both ≈n^0.75) — kwargs is qualitatively different, not just a
worse constant.

### Isolated: it is signature binding, not dict construction

The direct question the task posed — @ width=400, 12,000 calls:

| case | p50 | % of 20 ms |
|---|---|---|
| explicit 400-param signature, kwargs call, body=`pass` (pure binding) | 1094.2 µs | 5.471% |
| explicit 400-param signature, kwargs call, body builds a dict | 1120.3 µs | 5.602% |
| **the same signature, called positionally**, body builds a dict | **28.8 µs** | **0.144%** |
| `**kwargs` catch-all (N1's proxy) | 17.9 µs | 0.089% |
| dict pass-through | 0.159 µs | 0.0008% |
| record pass-through | 0.163 µs | 0.0008% |

Body=`pass` (1094.2 µs) vs body=build-a-dict (1120.3 µs) differ by 26 µs — dict
construction is a rounding error. The **same** 400-parameter function, called
positionally instead of by keyword — strictly more work (bind *and* build a dict)
— is **39× cheaper**. It is CPython's keyword-name matching against a large
parameter list that costs ~1.1 ms at 400 keywords, confirmed independently with
`timeit` outside this harness (body=`pass`, no dict at all):

```
width= 10  kwargs=    1.133us/call  positional=    0.116us/call  ratio=   9.7x
width= 50  kwargs=   19.438us/call  positional=    0.213us/call  ratio=  91.3x
width=100  kwargs=   78.172us/call  positional=    0.350us/call  ratio= 223.0x
width=400  kwargs= 1119.093us/call  positional=    2.209us/call  ratio= 506.7x
```

The ratio growing with width (9.7× → 506.7×) is the signature of super-linear —
here, close to quadratic — cost in keyword-argument binding specifically, on
Python 3.14.5 (this project's pinned interpreter; no second interpreter was
available to test against without violating the "do not install anything" rule,
so this is reported as a measured fact about the pinned version, not a general
CPython claim).

### Validation: cheap, and only the schema amortizes, not the data

| case (@ width=400) | p50 | % of 20 ms |
|---|---|---|
| pydantic dynamic model, 400 required floats, `Model(**request)` | 55.0 µs | 0.275% |
| manual key-presence check | 7.93 µs | 0.040% |
| manual per-field `isinstance` loop | 24.0 µs | 0.120% |
| no validation | 0.137 µs | 0.0007% |

All well under the whole-millisecond flag threshold. **"Once rather than per
call" only applies to the schema** (`create_model(...)`, built once outside the
loop, same as doc 03 §4's params model) — the data itself is per-record by
nature, so `Model(**request)` runs every call. Note it pays a smaller version of
the same kwargs-binding tax (55.0 µs vs the raw signature's 1120 µs) — not
decomposed further here, and small either way.

### Combining with N1: a full `score()` total per convention

Substituting each convention's width-400 accept+marshal cost for N1's original
`**kwargs`-proxy figure (18.3 µs) into N1's own downstream total (validate 5.5 µs
+ marshal-bulk 64.7 µs + dispatch-pooled 1.4 µs + readback-bulk 55.0 µs + assemble
0.35 µs = 126.9 µs) — arithmetic composition of the two experiments, not an
independently re-measured single call:

| convention | total | % of 20 ms |
|---|---|---|
| **kwargs, literal signature (doc 02 §3.5 as written)** | **1317.1 µs** | **6.59%** |
| dict | 187.0 µs | 0.935% |
| record, reused buffer | 166.7 µs | 0.834% |
| positional | 220.2 µs | 1.101% |
| *(N1's original `**kwargs`-proxy estimate)* | *145.2 µs* | *0.726%* |

**The literal kwargs convention alone costs more than N1's entire optimized total
for everything else in the framework, combined.**

### Maintainability — where speed and the doc's example diverge

- kwargs reads best for a **small**, hand-written call — doc 02 §3.5's own
  example uses 3 named args, not 400, and at 3–10 args its cost is irrelevant.
  The example is not wrong; its unstated generalization to the framework's
  actual realistic width (doc 01 §4d: 400) is.
- At 400 real inputs nobody hand-types the call — the caller already has a
  dict-shaped or record-shaped object (a validated request body, a DataFrame
  row). `score(**request)` at that point is the same data as `score(request)`,
  spelled with 400 extra `=` signs, and the spelling costs 39× more to bind.
- The reused-record buffer is cheapest (39.8 µs) but worst on maintainability —
  the caller must know field order/dtype and own the buffer's lifetime
  (thread-safety, staleness across a doc 08 §4 config swap). Worth it only as an
  opt-in fast path for a demonstrated hot loop, not the default.
- Positional (93.3 µs) is the worst idea regardless of speed — 400 positional
  floats is a silent-transposition hazard in a *credit-decision* framework.

### Recommendation for doc 02 §3.5 / doc 03 §6

**Specify `score(request: dict, *, params)` as the primary realtime convention.**
Keep kwargs *syntax* available for small hand-authored calls (tests, a
low-arity module) where it costs nothing and reads best. Offer the pre-built
reused record as a documented, explicit opt-in fast path, not the default.
Validation (`Model(**request)`, 0.275% of budget) is cheap enough to keep
unconditional on the request path.

### What was not tested (dropped for the ~12-minute budget)

GC on/off repeat (N1 already settled this for this workload class — no reference
cycles, no measurable GC effect); real kernel dispatch/readback as part of this
sweep (doc 01 §6.1 says not to benchmark the kernel; N1's numbers are reused
arithmetically instead of re-triggering an ~11 s kernel compile); mixed dtypes for
the convention sweep (all-`f8` here — N1/experiment A already cover dtype mix,
and the binding cost is dtype-independent); concurrency/tail-under-load (N4);
decomposing pydantic's own `__init__` internals beyond the headline number; a
second Python version to confirm the O(n²)-looking binding cost generalizes
beyond 3.14.5.

---

## N3 — params validation on the request path: affordable as written, and the model→NamedTuple conversion — not validation — is where the cost concentrates

Doc 02 §4: *"An invocation is 1 row or N rows, which covers realtime payload
params and batch uniformly."* Doc 01 §6: *"Params may arrive per invocation,
including in a realtime request payload."* E0 (doc 01 §5.7) measured 58.5 µs to
validate-and-bind a 63-node **structure** config — a one-time, build-time cost
through the discriminated union. N3 measures the adjacent, previously-open
question: a **params** document (doc 08 §6.2's `resolve_params`), which the docs
say may legally arrive on every single realtime request — params don't go through
a discriminated union (only structure does, doc 02 §2.1), so this is a distinct
mechanism from E0, not a re-run of it. `experimentation/params-validation-n3/`,
12,000 calls per measurement, p50/p95/p99/max, each as a % of a 20 ms budget.

**The headline: even at 50 module instances (doc 03 §4.1's "a realistic pipeline
has many"), the full `resolve_params(doc, origin=..., complete=True)` path — doc
08 §6.2's exact signature, including the completeness check via
`model_fields_set` — costs 256.5 µs p50, 278.0 µs p99: 1.28–1.39% of a 20 ms
budget.** That is well under the whole-millisecond flag this batch was told to
raise. **Doc 02 §4's "params may arrive per invocation" is affordable as written,
at every scale tested.**

| M (module instances) | full validation p50 | conversion (fresh) p50 | `resolve_params` total p50 | % of 20 ms |
|---|---|---|---|---|
| 1 | 2.99 µs | 4.47 µs | 9.68 µs | 0.048% |
| 10 | 17.4 µs | 34.9 µs | 56.7 µs | 0.283% |
| 50 | 78.3 µs | 170.6 µs | 256.5 µs | 1.283% |

**But the doc's own framing of where the cost lives is backwards.** Doc 03 §4
mentions "the model becomes a `NamedTuple`" as an aside — "under the hood." At
every M measured, **the model→NamedTuple conversion costs more than the
validation it follows** (M=50: 170.6 µs conversion vs 78.3 µs validation, a 2.2×
ratio) — each is a per-module `model_dump()` + constructor call, so validation and
conversion both scale the same way with M, but conversion pays a second full pass
over the same data. **Caching the *converted* bundle by content — not merely
avoiding re-validation — is where the real win is:** a memoized lookup costs
0.22–0.24 µs regardless of M, a **726–765× speedup at M=50** over redoing the
conversion fresh, because it collapses to a single dict lookup independent of
module count.

**`ParamsCell.get()`/`.swap()` confirm doc 08 §4's batch-context numbers hold at
genuine N=1, with one caveat.** `get()`: 0.159 µs p50 / 0.211 µs p99 — a plain
attribute read, 0.0008% of budget. `swap()` at N=1: 0.278 µs p50 / 0.363 µs p99,
against doc 08 §H's batch-context figure of 0.177 µs median / 0.357 µs p99
(11,605 swaps across a 3 s chunked-batch run, EXPERIMENTS.md §H). **The p99s
essentially agree (0.363 vs 0.357 µs) but the N=1 median is 1.6× the
batch-context median** — both are so far below any budget threshold that the gap
is immaterial, but this is an order-of-magnitude confirmation, not an exact
reproduction. Also confirmed directly: a simulated serving call performs
**exactly one `.get()` per invocation**, matching doc 08 property 1 (the
straddling consequence of re-reading mid-batch was already measured in §H and is
not re-run here).

**The design question the task brief posed: should a realtime request be allowed
to carry raw params, or must it reference a pre-validated bundle by id?**
Performance does not force the restrictive answer. At every measured scale the
full validate-convert-check path stays at or under 1.3% of a 20 ms budget — nowhere
near the whole-millisecond flag this batch was told to raise. **So if realtime
payload params are restricted to referencing a pre-validated bundle by id, that
restriction has to be justified by the governance argument alone (doc 04 §2.1's
CODEOWNERS/review-path boundary) — the performance argument does not
independently support it.** Conversely, where the cheapest possible path is wanted,
one already exists for free: `ParamsCell.get()` on an already-`swap()`-ped bundle
is **~1,600× cheaper than `resolve_params` at M=50** (0.159 µs vs 256.5 µs) —
validate-once-and-reuse (doc 08 §4's staged-swap model) remains strictly better
than validating on every request when it's available, which is exactly the
long-lived-endpoint case doc 08 §4 already designs for. The 1.28%-of-budget number
matters specifically for the case doc 08 §4 does *not* cover: a genuinely
per-request params document that changes every call and so cannot be cached by
content.

**Machine state:** load average ~1.5 on a 28-core box, with Firefox pinned near
83% of one core and three other Claude Code sessions each drawing a few % CPU
throughout — not idle, contrary to the timing rules' preference, so noted per
those rules. Not re-run isolated: the measured effect sizes (tens to hundreds of
µs) are two to three orders of magnitude above a plausible noise floor from
background load on an otherwise-unused core, and a `--quick` smoke run (500 calls)
and the full run (12,000 calls) agreed closely (M=50 full-validation p50: 80.3 µs
smoke vs 78.3 µs full).

**Not tested (dropped for the ~12-minute budget):** concurrency/tail-under-load
(N4, same scope cut as N1/N2); GC on/off beyond one check at M=50 (ratio 0.97 — no
measurable effect, consistent with N1/N2); `resolve_params(..., complete=False)`
(the completeness check itself is cheap — a `set` difference over ≤50 field
names — so its absence is unlikely to change the verdict); a second, idle-machine
run.

---

## N4 — tail latency, concurrency, and config-swap impact on serving

Fifteen prior experiments (now eighteen with N1–N3) measured **medians only**.
Doc 01 §6.1 flags this as the open gap: "concurrency/tail-under-load (N4) is not
[measured]." At a 20 ms SLA the tail *is* the SLA. `experimentation/n4-tail-concurrency-swap/`,
p50/p95/p99/p99.9/max throughout, each as a % of a 20 ms budget.

**Machine state (noted per the timing rules, not idle):** Firefox pinned near 80%
of one core and three other Claude Code sessions running throughout, on a 28-core
box. All figures below should be read with that caveat, especially the single-thread
tail (M1) and the high-thread-count configs (M3).

### M1 — single-thread steady state (n=30,000 full `score()` calls)

| stat | value | % of 20 ms |
|---|---|---|
| p50 | 1037.5 µs | 5.19% |
| p95 | 1098.0 µs | 5.49% |
| p99 | 1114.2 µs | 5.57% |
| p99.9 | 1723.6 µs | **8.62%** |
| max | 3840.5 µs | **19.20%** |

**The tail is real and it is not GC.** p99.9 is 1.7× the median and max is 3.7×
it, but only **one** GC generation-0 collection fired during the whole 30,000-call
run, and it coincided with **0 of the 300** calls at or above p99. (N=1 GC event
is too small to rule GC out categorically — this is suggestive, not a large-N
statistical claim — but it does not support the "GC drives the tail" hypothesis
the brief asked to test, at this call rate and this allocation shape.) The more
plausible driver, not isolated further under the time budget, is OS-level jitter —
thread scheduling, page faults — consistent with a shared, non-idle box; M3 below
shows the same p99/max gap growing by orders of magnitude under genuine thread
contention, which points the same direction.

**Even the max (19.2% of budget) stays inside a 20 ms SLA**, though it eats most
of a 100 ms one and would be worth alerting on if the SLA is closer to the 20 ms
end of doc 01 §6's stated 20–100 ms range.

### M2 — GC on vs `gc.disable()` vs `gc.freeze()` (n=15,000 each)

| variant | p50 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| baseline (GC on) | 1040.5 µs (5.20%) | 1101.5 µs | 1109.5 µs | 1165.1 µs | 1681.3 µs (8.41%) |
| `gc.disable()` | 1040.4 µs (5.20%) | 1100.8 µs | 1107.9 µs | 1153.9 µs | 1207.4 µs (6.04%) |
| `gc.freeze()` after warmup | 1040.7 µs (5.20%) | 1101.0 µs | 1108.9 µs | 1166.0 µs | 1181.8 µs (5.91%) |

**Refuted: disabling or freezing GC is not a meaningful tail mitigation at this
shape.** p50/p95/p99 are identical to three significant figures across all three
variants — well inside measurement noise. The baseline's max (1681 µs) is higher
than the other two, but on 15,000 samples one outlier moves the max without
moving anything else, and M1's max (3840 µs, double this run's baseline max) on a
larger n shows this statistic is itself noisy run to run. **This refutes the
brief's proposed mechanism** ("if per-call allocation drives the tail, GC
mitigation is the fix") **at this call rate** — `score()`'s per-call allocation
(one record array, one params NamedTuple, one 633-entry output dict) does not
generate enough garbage per call to make generation-0 collection a visible cost,
let alone a tail driver. Doc 02 does not need a GC recommendation on this
evidence.

### M3 — concurrency: 1/2/4/8/16 threads, `nogil=True` vs `nogil=False`, serving against serving

EXPERIMENTS.md §H measured `nogil=True` losing to `nogil=False` (26% vs 55%
throughput retention) when the *competition* is a background numba **compiler**
thread. That finding is scoped to compile-during-serving, and doc 08 §4 already
acted on it (compilation moved to a subprocess). N4 asks the question doc 08 §4's
subprocess fix does not address: **multiple concurrent request threads calling
the same already-compiled kernel** — no compile involved, purely concurrent
serving. This uses dispatch+readback only (not full `score()`, to isolate the
concurrency effect from N1's already-measured validation cost), 1.2 s/config:

| threads | nogil=True throughput | nogil=True p99 | nogil=True max | nogil=False throughput | nogil=False p99 | nogil=False max |
|---|---|---|---|---|---|---|
| 1 | 1189/s | 879 µs (4.40%) | 2383 µs (11.9%) | 1170/s | 890 µs (4.45%) | 2402 µs (12.0%) |
| 2 | 1168/s | 904 µs (4.52%) | 2352 µs (11.8%) | 1122/s | 6160 µs (**30.8%**) | 7614 µs (**38.1%**) |
| 4 | 1162/s | 891 µs (4.46%) | 2362 µs (11.8%) | 1077/s | 48024 µs (**240%**) | 90015 µs (**450%**) |
| 8 | 1156/s | 893 µs (4.46%) | 2349 µs (11.7%) | 1021/s | 115816 µs (**579%**) | 342539 µs (**1713%**) |
| 16 | 1178/s | 883 µs (4.42%) | 1897 µs (9.5%) | 1017/s | 254052 µs (**1270%**) | 378673 µs (**1893%**) |

**This is the finding that matters most in this batch.** Total throughput is
GIL-bound and roughly flat regardless of thread count or `nogil` setting
(~1150–1190/s for `nogil=True`, a milder ~1170→1017/s decline for `nogil=False`)
— the Python-level readback glue dominates and serializes everything either way,
consistent with N1's finding that 92% of `score()`'s cost is Python loops, not
the kernel. **But the tail diverges catastrophically.** `nogil=True` keeps p99
and max essentially flat from 1 to 16 threads (~0.9–2.4 ms, 4–12% of budget,
*not blowing the 20 ms budget at any thread count tested*). `nogil=False` is
fine at 1 thread but **degrades monotonically and explosively from 2 threads
on**: p99 is 30.8% of budget at 2 threads, 579% at 8, and **1270% — 12.7× the
entire 20 ms budget — at 16 threads**, with a max of 378.7 ms. This is a convoy
effect: a `nogil=False` kernel call holds the GIL for its own duration, so under
concurrent load, calls queue up behind each other in a way `nogil=True` calls do
not (each releases the GIL immediately, letting the scheduler interleave fairly).

**Doc 08 §4's existing "releasing the GIL makes serving worse" is correct only
for its measured scenario (serving vs. one background compile thread) and does
not generalize — for concurrent request serving, the opposite is true and the
effect is two to three orders of magnitude larger.** A production endpoint
handling concurrent single-record requests needs `nogil=True` compiled kernels
to bound tail latency; doc 08 §4's subprocess-compile fix and this finding are
not in tension (compilation is subprocess-isolated so never competes with
serving threads at all now) but doc 02/doc 08 currently state the `nogil`
recommendation in only one direction, and a reader who took §4's table at face
value could reasonably choose `nogil=False` for serving kernels — which this
measurement shows would be a severe regression under concurrent load. **Doc 02
§3.5 or doc 08 §4 needs to say explicitly: serving kernels are `nogil=True`,
unconditionally — that same flag protects concurrent-request tail latency, not
just background-compile throughput.**

### M4 — config swap under continuous single-record traffic (doc 08 §4's `activate()`)

Doc 08 §4 measured `activate()` at 0.177 µs and 0 straddled batches (§H), but not
the latency impact on in-flight serving at the moment a swap lands. N4 runs
continuous single-record dispatch (10-rule ruleset, kernel-call only — no Python
glue, to isolate the swap-pointer-flip effect cleanly) for 3.0 s, swapping the
generation pointer between two independently-compiled generations every 100 ms
(30 swaps total, 758,215 calls served):

| | p50 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| overall (incl. every call across all swaps) | 3.5 µs (0.017%) | 3.7 µs | 5.3 µs (0.027%) | 9.2 µs (0.046%) | 274.8 µs (**1.37%**) |
| steady state (excl. first-call-after-swap) | 3.5 µs | 3.7 µs | 5.3 µs | 9.2 µs | 274.8 µs |
| **first call after swap, only** (n=30) | 7.3 µs (0.037%) | 11.9 µs | 14.1 µs (0.071%) | 14.7 µs | 14.7 µs (0.074%) |

**Confirmed and extended: the swap does not stall serving, and the first call
after a swap pays a small, bounded, measured cost.** The worst latency anywhere
in 758,215 calls across 30 swaps was 274.8 µs — 1.37% of a 20 ms budget — and
that max belongs to the *overall* population, not specifically a post-swap call
(excluding the 30 first-after-swap calls changes nothing at 4 significant
figures, because they are 0.004% of the sample). The dedicated first-call
measurement shows a real, consistent effect — first-after-swap costs **2.1×** the
steady-state median (7.3 µs vs 3.5 µs) — but its absolute size (14.7 µs worst
case, 0.074% of budget) is three orders of magnitude below anything that would
matter at a 20 ms SLA. **A realtime endpoint swapping config generations does
not need to guard against a client-visible stall; doc 08 §4 can state the
latency-impact question as answered, not just the straddle/atomicity question.**

### What doc 02 / doc 08 should say

1. **Doc 08 §4's `nogil` table needs a scope qualifier**: its "releasing the GIL
   makes serving worse" finding is specific to competing against a background
   *compile* thread (a scenario the subprocess fix has since removed from the
   production path). For concurrent **request** serving — the actual production
   shape — `nogil=True` is not just fine, it is required to avoid a tail blowup
   of up to 12.7× the entire 20 ms budget at 16 concurrent threads. **Withdraw
   any reading of §4 as a general anti-`nogil` recommendation.**
2. **Doc 08 §4 can add the missing half of its swap-latency claim**: `activate()`
   costing 0.177 µs and landing atomically (§H) does not by itself say a
   client-visible request is unaffected — N4 now shows the worst call anywhere
   near 30 swaps was 1.37% of budget, and the specific first-call-after-swap
   cost is 0.074% of budget, worst case. Both numbers are safe to publish as a
   closed question.
3. **Doc 02 has no GC guidance and does not need one** on this evidence — M2
   found no measurable tail difference between GC on, disabled, or frozen at
   `score()`'s allocation shape.

**Not tested (dropped for the ~12-minute budget):** `sys.setswitchinterval`
tuning as a `nogil=False` mitigation (doc 08 §4 already found it a partial
mitigation, not a fix, in the compile-interference context; not re-tested here
for the serving-against-serving context); thread counts above 16; `os.sched_getaffinity`/
core-pinning as a way to reduce M1's OS-jitter tail; a second, idle-machine run
to separate the shared-box noise from M1's tail from a framework-inherent one —
the M3 result (same tail-blowup shape, growing with contention) makes a
framework-level and a noise-level explanation both plausible for the size of
M1's p99.9/max specifically, though M3's `nogil=True` numbers (flat tail across
1–16 threads) argue the *framework* is not the driver once the GIL-holding
kernel variant is ruled out.

---

## N5 — N4's unexplained tail, root cause: OS scheduler preemption, not GC/allocator/dispatch

N4/M1 explicitly named its own gaps: **"Not tested... core-pinning as a way
to reduce M1's OS-jitter tail; a second, idle-machine run."** N5 closes both,
by instrumenting every call directly instead of theorizing further.
`experimentation/n4-tail-cause/`. Method: reuse N1's `score()`/`DRIVER`
verbatim; time only the call itself (`t0`/`t1` wrap `score()`, nothing else),
and capture — *outside* that window, so they don't contaminate the
measurement — `resource.getrusage()` deltas (`ru_minflt`/`ru_majflt` page
faults, `ru_nvcsw`/`ru_nivcsw` context switches), current CPU core (from
`/proc/self/stat` field 39 — `os.sched_getcpu()` does not exist on this
Python 3.14 build, confirmed `AttributeError`), and `sys.getallocatedblocks()`/
`gc.get_count()[0]` deltas. Four configs, n=20,000 (A/B/C) or 8,000 (D):

| config | p50 | p99.9 | max | % budget (max) |
|---|---|---|---|---|
| A — unpinned, ambient load | 1053.1µs | 2004.4µs | 3008.1µs | 15.04% |
| B — pinned to one core | 1052.3µs | 3268.3µs | 4959.0µs | **24.80%** |
| C — no-op control (same loop/instrumentation, no `score()`) | 0.281µs | 0.390µs | 6.032µs | 0.03% |
| D — 14/28 cores externally busy (bounded, self-terminating stressors) | 2374.7µs | 2905.1µs | 2995.9µs | 14.98% |

**Config C rules out the harness/timer itself as the explanation** (brief's
hypothesis 5): an identically-instrumented no-op loop shows essentially no
tail at all. Whatever produces A/B/D's multi-millisecond outliers is
intrinsic to running `score()`, not an artifact of the measurement loop.

**Correlation, not theory** — Pearson r between per-call latency and each
signal, across all calls in each config:

| signal | A (r) | B (r) | D (r) | measured behavior |
|---|---|---|---|---|
| `ru_minflt` / `ru_majflt` (page faults) | 0.0 | 0.0 | 0.0 | **exactly zero in every one of 48,000+ calls, all 4 configs.** Instrument validated separately: touching a fresh 64MiB buffer moved `ru_minflt` by 16385 (expected 16384) — the "always zero" reading is real, not a broken sensor. |
| `ru_nvcsw` (voluntary ctx switch) | 0.0 | 0.0 | 0.0 | always zero — `score()` never blocks on I/O/a syscall mid-call |
| `ru_nivcsw` (involuntary ctx switch — OS preempted the thread) | 0.053 | **0.560** | 0.211 | **the finding.** Nonzero in 0.96% (A) / 1.48% (B) / 7.14% (D) of all calls overall, but in 5.5%/25%/98.75% of the top-1%-slowest calls and **100%/100% of the top-0.1%-slowest calls in B and D** — i.e. every single one of the most extreme outliers in the pinned and contended configs coincides with the process being involuntarily preempted mid-call. |
| CPU migration (core changed) | 0.025 (RR₉₉=60×, but only 5 raw events) | 0 (pinned — impossible) | ≈0 | rare (0.025–0.0625% of calls) and enriched when it happens, but explains only a handful of A's ~200-call top-1% bucket — not the primary driver |
| `sys.getallocatedblocks()` delta | 0.006 | -0.002 | -0.014 | no relationship |
| `gc.get_count()[0]` delta | 0.004 | -0.002 | -0.010 | no relationship — consistent with N4/M2's GC-off finding |

**Allocator/mmap-threshold hypothesis, directly refuted.** Doc 01/N1 measured
~49,764 bytes / ~215 blocks per `score()` call. A separate pure-Python sweep
(`alloc_sweep.py`, no numba involved) allocated+touched+freed a single buffer
8 sizes from 4KB to 1MB — bracketing glibc's 128KB default
`M_MMAP_THRESHOLD` — n=3000/size (24,000 calls): `ru_minflt`/`ru_majflt` per
call was **0.000 at every size tested**, and p50/p99.9/max scaled smoothly
with size (p50: 0.53µs at 4KB → 34.41µs at 1MB) with no threshold cliff.
Repeated same-size alloc/free in a steady-state loop is served from
glibc's/pymalloc's already-faulted free lists, never re-faulting — this
holds up to 80× decider2's actual per-call allocation volume.

**numba dispatch-cache hypothesis, directly refuted for this deployment
shape.** `DRIVER.overloads` has exactly 1 registered specialization, before
and after 5000 additional calls. A single-signature dispatcher has nothing to
miss on — this is measured (the count), though the claim that a
single-signature dispatch structurally cannot incur a type-resolution miss is
reasoned from how numba's dispatcher works, not traced through its C source.

**Pinning to one core makes the tail *worse*, not better** (brief's
hypothesis 4). B's p99.9 is 63% higher and max 65% higher than A's, with
`ru_nivcsw` now explaining literally 100% of the extreme tail. Mechanism:
unpinned, the OS scheduler can route the thread onto a *different, quieter*
core when the current one gets momentarily busy; pinned, that escape hatch is
gone and every bit of whatever else lands on that one core (kernel work,
other processes, IRQ handling) queues up against `score()` directly, with
nowhere else to go. **Do not recommend `sched_setaffinity`-style single-core
pinning as a tail mitigation** — it measurably backfires here. Core isolation,
if wanted, needs cgroup/`cpuset`-level exclusion of *other* processes from the
reserved core(s), not merely pinning the serving process to one.

**Does it scale with anything decider2's own design controls? No — measured,
not inferred, on both axes tested.** Allocation volume: refuted above (0
fault effect across an 80× size range). Contention: `contended_run.py` held
14 of 28 cores busy with bounded (`timeout`-capped, self-terminating)
busy-spin loops and re-ran config A. The median roughly **doubled** (1053µs →
2375µs, 5.3% → 11.9% of a 20ms budget) — but the max/tail-to-median ratio
**shrank** (2.86× → 1.26×), and the max itself (2996µs, 14.98% of budget) did
not exceed A's own max. Under this one contention regime, load raises the
*whole distribution* together rather than making rare outliers
multiplicatively worse — a materially different, and materially less alarming,
degradation shape than "19.2% max becomes 120%." **This is scoped to one
moderate contention level (14/28 cores, bare busy-spin) on this box — full
saturation and, especially, cgroup/Kubernetes-style `cpu.max` quota
throttling (a harder-edged, periodic-freeze mechanism, not probabilistic
preemption) were both dropped for the time budget and could behave
differently.**

**What remains genuinely unexplained:** in config A (unpinned, ambient load),
~95% of the individual top-1%-slowest calls show **no** signal on any of the
7 captured metrics — `ru_nivcsw` and migration together account for only a
handful of A's ~200-call top-1% bucket. The leading explanation — brief
interrupt/softirq handling that doesn't register as a `nivcsw`-counted
context switch on this process, or a race in the sub-microsecond gap between
an instrumentation read and the timed call itself — is **reasoned, not
measured**; confirming it would need `perf`/`ftrace`-level tracing, dropped
for the ~12-minute budget. Once contention is either concentrated (pinning,
config B) or increased (config D), `ru_nivcsw` closes that gap almost
completely (100%/98.75% of the extreme tail) — so the "no signal" residue in
A looks like the same mechanism at a rate too low for this sample size to
catch cleanly, not a distinct second cause, but that is inference from the
pattern across configs, not a direct measurement on A alone.

### What doc 02 / doc 08 should say

1. **N4/M1's tail is now explained, not just "not GC."** It correlates with
   `ru_nivcsw` (OS scheduler preemption) — up to r=0.56, and up to 100% of
   the extreme (p99.9+) tail in two of three real-load configs. It is a
   property of running on a shared/contended host, not of decider2's own
   allocation shape, GC behavior, or numba dispatch — all three were directly
   measured and ruled out (0 page faults across 48,000+ calls and a
   24,000-call allocation sweep spanning 80× the real per-call volume; 1
   dispatch signature, unchanged after 5000 more calls).
2. **Do not add a core-pinning recommendation to doc 02/08.** It was tested
   as a candidate mitigation and measured to make the tail *worse*
   (p99.9 +63%, max +65%) by removing the OS's own contention-routing. If
   dedicated capacity is wanted for a serving process, it is a cgroup/cpuset
   deployment concern (excluding other processes from reserved cores), not a
   `sched_setaffinity` call inside decider2.
3. **Doc 02/08's SLA guidance should track median-under-load, not only max,
   on a shared host.** The one contention regime tested here (14/28 cores
   externally busy) roughly doubled the median (5.3% → 11.9% of a 20ms
   budget) while leaving the max essentially flat (~15%) — under load, the
   risk is the whole distribution shifting up, not a runaway outlier. This
   finding is scoped to bare CPU contention on this box; cgroup/Kubernetes
   `cpu.max` throttling was not tested and could behave differently (a
   periodic hard freeze rather than probabilistic preemption), and full
   28/28-core saturation was intentionally not run (bounded to 14/28 out of
   consideration for other users of this shared box).

---

## O — Strings: numba supports them, the boundary does not, and regex belongs elsewhere

The doc said "strings enter as codes" without saying why, and the first
implementation raised `NotImplementedError` on a string input. Both the capability
claim and the cost were unmeasured. Numba 0.67, polars 1.41, 200k rows.

### Numba's unicode support is real and wider than assumed

In nopython mode, all of these compile and run: `==`, `!=`, `len`, `startswith`,
`endswith`, `in`, `find`, `split`, `upper`, and `numba.typed.List[str]`. So
"numba can't do strings" is false, and any design that assumes it is solving the
wrong problem.

**`re` does not compile.** `TypingError` in the nopython frontend. There is no
regex inside a kernel, and no flag that changes that.

### The boundary is what costs, and it costs 31×

| route | boundary | kernel | total |
|---|---|---|---|
| `typed.List[str]` from a polars column | 1340.6 ms | 18.48 ms | **1359.1 ms** |
| dictionary codes (`Categorical.to_physical()`) | 43.4 ms | 0.04 ms | **43.4 ms** |

**31× end-to-end, 438× in the kernel.** Building a `typed.List` means
materialising N Python string objects and boxing each into numba's runtime;
dictionary codes are an `int32` buffer polars already has. This is the whole
justification for doc 05 §1.5's "strings enter as codes" and it was never
recorded.

### Hoisting literals keeps the author's code natural — and makes a literal a *value*

An author writes `sector == "private"`. Codegen emits a kernel over `int32` codes
with each distinct literal hoisted to a kernel argument holding its code:

| | |
|---|---|
| 200k rows, two literals | **0.16 ms** |
| vs `typed.List` route | **115× faster** |
| literal absent from the data | never matches; `-1` sentinel, no crash |
| `len(kernel.signatures)` across three distinct literal sets | **1** |

That last row is the one that matters beyond performance. **A string literal
becomes a kernel argument, so changing it is a value change with no recompile** —
the same guarantee doc 08 §2 gives a numeric threshold. A policy moving from
`"private"` to `"self_employed"` is a params edit, not a deploy.

The transform is bounded: it applies to a comparison between a string-typed input
and a string constant. It is not general AST rewriting, and anything outside that
shape falls to the routes below.

### Regex belongs in the frame tier, and is faster there anyway

| | 200k rows |
|---|---|
| `pl.col(x).str.contains(regex)` | **5.98 ms** |
| `pl.col(x).str.extract(regex)` | 20.91 ms |
| Python `re` loop | 42.29 ms |
| kernel reading the resulting bool | 0.05 ms |

Polars' regex is the Rust `regex` crate and beats Python's `re` by **7×**. So
numba's lack of `re` costs nothing: shape the string in the frame tier, pass the
boolean or the code into the kernel. `str.contains` plus a kernel is 6.03 ms
against 42.29 ms for `re` alone.

### The compiler will never catch a bare literal — found during implementation

`int32 == "private"` **compiles cleanly in nopython and evaluates to `False`
forever.** No `TypingError`, no warning — numba simply follows CPython's own
`int == str` semantics, where the comparison is legal and always false.

```python
@njit
def f(codes):
    return sum(1 for i in range(len(codes)) if codes[i] == "private")
f(np.array([0, 1, 0, 2], dtype=np.int32))    # -> 0, silently
```

This matters more than it first reads. The rule "a string enters as a code"
means an author who writes a bare literal is comparing a code against text that
was never encoded to match it — and **every layer that could plausibly catch
that, doesn't**. Not the type checker, because the types are legal. Not the
kernel, because it runs. Not a test over real data, because the answer is a
plausible-looking zero rather than a crash. It is doc 03 §2.1's worst failure
mode arriving through the one door nobody is watching.

So the guard cannot be inherited from numba and cannot wait for codegen. It is an
explicit **signature-shape check at param resolution**, before anything compiles:
a step reading a `str`-typed input and declaring no `str`-typed `param()` is a
build error naming the column and pointing at the param form. An earlier comment
in the implementation assumed numba's type checking would eventually supply this
for free; it never will.

### What this settles

1. Strings are **not** an unsupported dtype. They are a tier-2 dtype whose
   boundary representation is dictionary codes (doc 05 §1.5).
2. Equality and membership against literals work through hoisting, with the
   author writing ordinary Python.
3. Regex, and any per-row string manipulation, is a **frame operation** — which
   is where it is faster regardless.
4. `typed.List[str]` stays available as a documented escape hatch for genuine
   per-row string work, with its 31× cost stated rather than discovered.

---

## P — Tree and table compile cost, and where the line cap actually bites

Measured during the decider 1 tree/table migration, 100k rows, 6 float features,
cold compile per shape. Recorded here because the numbers were previously only
in a migration report, and §G's ruleset model turns out not to generalise to
trees.

> **These are §P's own trees, not comparable bit-for-bit with §Q's.** §Q builds
> independently-generated trees of the same nominal shape (different feature
> counts, different random structure) and measures 2–4× faster ns/row with
> 1.3–2× more emitted lines. Neither is wrong; they are different trees. Use
> §P for *absolute* cost against emitted lines, and §Q for the *relative*
> codegen-vs-interpreted question, which it controls for by running both engines
> over the identical tree and the identical rows.

### Trees — full-binary (fan-out)

| depth | leaves | emitted lines | compile | ns/row |
|---|---|---|---|---|
| 5 | 32 | 84 | 0.29 s | 45.6 |
| 7 | 128 | 276 | 1.02 s | 123.8 |
| 8 | 256 | **532** | 1.93 s | 197.7 |
| 9 | 512 | 1044 | 4.32 s | 336.9 |

One-sided chain: 128 nodes → 278 lines, 1.39 s, 116.6 ns/row.

**A realistic credit tree** — 5 score bands × affordability gate × sector match ×
AND-composite, 21 leaves, 23 thresholds, 10 string literals — is **120 emitted
lines, 24% of the cap, 0.67 s cold compile, 104 ns/row.** Roughly 4× headroom.

### Where the ~500-line cap bites

| shape | reaches 500 lines at |
|---|---|
| flat ruleset | **~29–30 rules** (§G: 30 rules = 517 lines, ≈16.5 lines/rule) |
| full-binary tree | **depth 8, ~256 leaves** |
| one-sided waterfall | ~230 nodes |
| realistic credit tree | not reached |

A realistic flow is far from it; **30 rules in one fused group is not**, and 30
rules is an ordinary credit rule set. That is what the cap is for.

### §G's model does not generalise to trees

Trees compile **3–5× cheaper per emitted line** than rulesets and scale
≈ lines^1.0–1.2, not §G's ^1.4: 532 lines → 1.93 s here against §G's 517 lines →
5.96 s, and 1044 → 4.32 s against 967 → 20.67 s. So for trees the 500-line cap is
conservative by ~5× and could sit near 1500. It is left at 500 because doc 05 §7
states that number uniformly; moving it is a decision, not a discovery.

### Tables: compile cost is independent of row count

| bands | emitted lines | compile | ns/row | source sha |
|---|---|---|---|---|
| 2 | 31 | 0.01 s | 70 | 93277fad |
| 50 | 31 | 0.00 s | 125 | 93277fad |
| 1000 | 31 | 0.00 s | 1045 | 93277fad |

**Identical source from 2 bands to 1000** — everything after the first is a cache
hit, because a table is a generic kernel over rows held in `shared` arrays. The
trade is answer time: a table is a linear scan, so 1000 rows costs ~1 µs/row
where a tree of equivalent selectivity is ~100 ns/row. **That is the real choice
between the two engines**, and it is why they are two engines rather than one.

### A hard CPython limit, found only by measuring

**CPython refuses more than 100 levels of indentation.** A one-sided chain of 128
nodes — an ordinary policy waterfall, and doc 01 §4b's *cheap* shape — died with
`IndentationError` at import of the generated file, before numba saw it. No
amount of reasoning about numba would have found it.

Fixed by dropping `else:` entirely: every tree path ends in a `return`, so the
otherwise-arm follows at the same indentation. That also **cut emitted lines
~30%** (depth-7 full-binary: 403 → 276), which cut compile time with it.
`_MAX_NESTING = 90` guards the nested *then*-chain case that remains.

> **This is the strongest argument for the interpreted-tree alternative** (§Q,
> pending): an array-walking kernel has no line cap, no indentation limit, no
> fan-out wall and no per-shape compile at all. If its ns/row is within noise of
> codegen at a 20–100 ms budget, this entire class of limit disappears.


---

## Q — Codegen vs an interpreted tree walk: codegen wins small, and becomes *unpredictable* large

Both engines over the **identical tree and identical rows**, `np.array_equal`
asserted on every shape before any timing counted. 100k rows. Minimum of 4–5
independent full reruns, with the observed range quoted — because the range
turned out to be the finding.

| shape | emitted lines | codegen ns/row | interpreted ns/row | codegen compile |
|---|---|---|---|---|
| credit tree (20 leaves) | — | **20.7–22.9** | 56.7–61.2 | 0.39–0.42 s |
| full-binary d5 | — | **24.5–26.8** | 64.8–68.7 | 0.38–0.44 s |
| full-binary d7 | 524 | **42.2–49.7** | 85.5–100.7 | 1.40–1.57 s |
| full-binary d9 | 2,060 | 82.8–**127.0** | 123.0–127.4 | 7.25–8.12 s |
| one-sided chain, 100 | — | **28.4–28.8** | 38.2–38.7 | 1.31–1.37 s |
| full-binary d10 | 4,108 | 134.0–**241.9** | 138.4–141.0 | 20.6–21.2 s |

**Codegen wins 2.0–2.7× through depth 7** — the range every realistic credit
tree lives in. At d9 it is a coin flip (median 124.7 vs 125.5). At d10 the
ranges fully overlap.

### The finding that matters is variance, not the mean

At d10, codegen measured **134.0, 141.3, 240.9, 241.9 ns/row across four
otherwise-identical reruns — an 80% spread** — while the interpreted kernel gave
138.4–141.0 across the same four runs, under 2%. This survived the seed fix
below, so it is not a data artefact; the likely cause is code-layout and
instruction-cache sensitivity in a multi-thousand-line JIT-compiled function.

> **Past ~2,000 emitted lines, codegen's speed stops being predictable** — in
> exactly the region where its compile cost is already becoming unbearable (20 s
> at d10). For a system with latency budgets, an engine whose p99 depends on
> code layout is a worse property than one that is uniformly slower.

### A harness bug that invalidated an earlier pass

The row generator seeded with `hash(shape.name) & 0xFFFF`. **CPython randomises
string hashing per process**, so every run silently drew *different row data*.
For the one-sided chain — whose per-row cost depends on average walk-to-leaf
length — this produced a **2.15× swing between identical reruns** that read as
system noise and was not. Fixed with `zlib.crc32`; post-fix reruns agree to
~1.4%.

Worth recording as a method note: a benchmark seeded from `hash()` of a string
is not reproducible across processes, and the symptom is indistinguishable from
scheduler noise.

### What this does and does not settle

It does **not** say codegen is wrong. For a realistic credit tree, codegen is
genuinely ~2.7× faster per row, and at 100k rows that is ~2 ms against ~5.7 ms —
both comfortably inside a 20–100 ms budget, but the difference is real at batch
scale.

What it buys against that is **zero compile**: one kernel, ever, for any tree.
No ~500-line cap, no fan-out wall, no CPython indentation limit (§P), and a
structural edit becomes a *values* change rather than a staged 0.4–21 s compile.

So the trade is now measured and stateable: **~3 ms per 100k-row batch, against
0.4–21 s per structural edit plus a cap plus unpredictability at scale.** Which
side that favours depends on whether trees are edited by people or fixed at
build time — which is a product decision, not a benchmark result.


---

## R — GoRules ZEN: closer than expected, and it exposes decider2's realtime overhead

`zen-engine==2.0.2`, the identical 12-band income rule set built as a decider2
`DecisionTable` and as a JDM `decisionTableNode`, plus an independent oracle. All
three agree on every answer including boundary values, asserted before any timing.

| | decider2 | ZEN | |
|---|---|---|---|
| batch, 100k rows | **6,166 ns/row** | 9,585 ns/row (`evaluate_batch`) | decider2 1.55× |
| batch, naive per-row loop | — | 1,394,180 ns/row | **226× worse** |
| single record | 352.7 µs (`score()`) | 1,264.3 µs (sync `evaluate`) | decider2 3.6× |
| single record | 352.7 µs | **94.6 µs** (`async_evaluate`) | **ZEN 3.7×** |
| cold start | ~1.01 s (numba JIT) | ~4.8 ms | ZEN 210× |

### Two things this overturns

**"Per-row FFI is fatal" was wrong.** The ~145× gap between ZEN's own two batch
APIs is *Python binding call overhead*, not the FFI boundary. Crossing into Rust
is cheap; crossing through pybind per row is not. That is a different problem
with different fixes, and it means an out-of-process engine is not automatically
disqualified the way §B's `objmode` result suggested.

**decider2's realtime path is its weakest number, and ZEN beat it.** Measured
directly here:

| | score() p50 |
|---|---|
| 3 trivial steps | 243 µs |
| 30 steps | 1,548 µs |

Doc 05 §3.1b budgets the whole single-record path at **~60 µs** — kernel ~1 µs
plus ~58 µs of everything else. We are **~25× over our own spec**. Confirmed
pre-existing, not a regression: measured at 1,752 µs before the polymorphic
refactor and 1,548 µs after, so the refactor slightly *improved* it.

The cause is already documented and simply not built — `boundary/__init__.py`'s
own docstring says whole-row bulk marshalling "is not part of this package's
scope" yet. §N1 measured a per-field Python loop at **92% of request overhead**,
and §3.1b names the three fixes: marshal whole-row not per-field, pool the
output buffer, and convert params once per generation rather than per request.
**This is the highest-value unbuilt item in the system**, and it took an outside
engine beating us to make it visible.

### JDM as an interchange format

Converters both ways, run against real ZEN-repo fixtures rather than invented
ones. Two of `credit-analysis.json`'s four tables converted cleanly and match
live ZEN; two were refused with demonstrated reasons.

> **A real expressiveness gap, proved live.** JDM's "Turnover" table has a row
> closed on *both* ends — `[200_000..1_000_000]` — which **no single decider2
> `BoundMode` can express**, because `BoundMode` deliberately omits
> `both_inclusive`/`both_exclusive` to prevent overlaps and gaps. Evaluated
> against real ZEN at turnover = 1,000,000 exactly: ZEN says **amber**, a naive
> conversion says **green**. A silent converter would have shipped a
> one-value boundary bug.

decider2 trees are strictly single-parent (reusing a node raises "revisits
node"); JDM is a genuine DAG. That is real expressiveness on JDM's side, traded
against decider2's line-cap guarantee (doc 05 §7), which node reuse would break.

Going the other way, the biggest loss is **decider2's params/values split**: in
JDM a threshold is literal text, with no equivalent indirection, so doc 08 §4's
hot-swap-without-recompile has nothing to map onto.

### Verdict

Take **specific ideas**, not the engine and not the format:

1. **ZEN's per-column bitset row index** for large decision tables
   (`core/engine/src/nodes/decision_table/index.rs`, gated at `MIN_INDEX_ROWS=8`).
   It would live in decider2's `shared` arrays, so it does not disturb the
   "editing rows is free" guarantee — and §P measured a 1000-row table at
   ~1 µs/row precisely because it is a linear scan.
2. **A `collect` hit policy** — a genuine feature gap.

Explicitly **do not** take: `functionNode`/QuickJS (decider2 already refused
embedded code, doc 08 §1.1), `customNode` (solved more simply by `step()`), the
async-runtime dependency, `rand()`, or DAG node reuse for trees.

JDM stays a **one-way partial export** for governance and visualisation, with the
two-sided-bound case refused loudly rather than converted.
