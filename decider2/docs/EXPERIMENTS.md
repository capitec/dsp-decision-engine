# Measured results — 2026-09-18/19

Eleven experiments run against real numba, on the project's own environment:
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

All eleven ran. **G and H together settle whether a configuration UI is viable:
it is, with one change to how compilation is hosted.**

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
`decider build --verify` on "zero runtime compilations". **A build that normalises
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
