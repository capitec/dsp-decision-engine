# 01 — Motivation and evidence

Every design decision in `decider2` traces to something measured or observed.
This document records that evidence so later readers can challenge the
conclusions rather than re-derive them.

Benchmarks were run on an M2 Pro (12 cores), Python 3.14.4, polars 1.41.2,
numba 0.66.0, numpy 2.4.6. All figures are medians of warmed repeats;
implementations were asserted equivalent before timing.

---

## 1. The headline: an existing production-shaped workload

An internal workload of realistic size and shape — the kind of combinatorial,
branchy, early-exit logic this engine exists for — was implemented three times:
in plain Python, in numba, and in polars expressions. Per unit of work, measured
by its own harness:

| implementation | per unit | at large problem size |
|---|---|---|
| raw Python | 7.1 ms | — |
| **numba** | **0.82 ms** | **1.5 s** |
| **polars expressions** | **35.2 ms** | **12.1 s** |

Polars is **43× slower than numba and 5× slower than plain Python** on this shape
of work. Critically, the polars implementation was not written badly — it uses
**zero** `map_elements`/`map_batches`/`fold` escape hatches. The cost is
structural: dozens of `pl.when` chains, ~25 join/explode sites, nested cross
joins that materialise the search space (millions of rows for a *single* unit of
work), and ~90 scratch columns that had to be dropped because every downstream
`group_by`/`sort`/`join` is O(rows × columns).

**It also changed the answer**, which is the more important finding. Early exit
was destroyed: a descending `while` loop became
`filter → group_by(max) → join-back`, evaluating every candidate eagerly. A
one-directional search had to be re-encoded as a predicate, and the
implementation notes concede the naive form can find a match the sequential scan
would not have. One phase is duplicated as an all-null column block because a
phase cannot be conditionally skipped. And **one branch of the logic was dropped
entirely** — the Python and numba implementations handle a case the polars one
does not.

> **Conclusion.** For branchy, sequential, early-exit logic, a columnar
> expression engine is the wrong tool — not merely slower, but semantically
> lossy under porting pressure.

---

## 2. Why: columnar engines cannot short-circuit

Decision-ladder benchmark, identical semantics both ways, 1 M rows:

| branch depth | polars | numba serial | numba prange |
|---|---|---|---|
| 5 | 13.9 ms | 7.7 ms | 1.4 ms |
| 20 | 54.8 ms | 11.6 ms | 2.1 ms |
| 50 | 127.2 ms | 16.4 ms | 3.3 ms |

Polars is **linear in depth** because it must evaluate every condition and every
`then` arm, then select. Numba is **sub-linear** (~depth/2 conditions) because
scalar code branches. Break-even is at **depth ≈ 2–3 conditions**.

This is structural, not an implementation artefact — so the conclusion is not a
bet against polars improving.

Realistic mix (15 derived features + 10 branching rules + weighted score, 1 M rows):

| polars eager | polars lazy | numba serial | numba prange |
|---|---|---|---|
| 132 ms | 70 ms | 35.9 ms | **2.3 ms** |

Sequential dependent steps (10 deep, 1 M rows): polars 36 ms (best case, lazy
with CSE) to 127 ms (eager fused) vs numba **3.96 ms** serial / **0.91 ms**
prange. Dependency depth cannot be flattened — the chain *is* the depth.

**Two supporting findings:**

- Polars gets ~zero benefit from its thread pool on single elementwise
  expressions: depth 20 measured 47.2 ms at `POLARS_MAX_THREADS=1` vs 39.7 ms at
  12. So the two tiers do not compete for cores — polars keeps its threads for
  joins and group-bys, numba gets `prange` for the row loop.
- Reusing a polars `Expr` object in two places **recomputes the whole subtree**
  (4–7× penalty) unless lazy CSE or `with_columns` breaks it. `decider` already
  works around this: `compile_expression_graph` counts fan-out
  (`decider/executor.py:143-147`) and materialises a node as a real column when
  `node.name in output_names or fanout[node.name] >= 2` (`:149-153`), inlining it
  as an expression otherwise (branch at `decider/modules/expression.py:119-123`).
  In a row loop the whole problem disappears — a value used twice is a local
  variable used twice — so that heuristic can simply be deleted rather than
  reimplemented.

---

## 3. Expression *form* dominates, and it is a cliff

Same 5000 nodes, same result:

| form | time |
|---|---|
| left-leaning chain `c0+c1+…+c4999` | **15,100 ms** |
| balanced tree | **37 ms** |
| `pl.sum_horizontal` | **8.2 ms** |

~2000× from form alone. Cost is **O(nodes^2.3) in DAG depth**, and it is
*planning*, not computation — a 1-row frame costs the same as a 10k-row frame
(69 ms vs 72 ms at 500 nodes), independent of frame width, and disabling every
optimiser pass saves only ~20%.

**Nothing amortises.** Six consecutive calls at 1000 nodes stayed flat
(~350–365 ms) whether the `Expr` was reused, the `LazyFrame` was reused, or
`explain()` was primed first. Numba compiles 47–264 ms **per signature,
independent of column count**, with no per-call planning term at all.

**Honest counterweight:** in its best form polars is competitive on wide
*independent* arithmetic — `sum_horizontal` at 1000×100k is 13.8 ms vs numba
serial 22.3 ms. Above ~10 M cells both are memory-bandwidth-bound.

> **Conclusion.** The record tier's justification is **depth — dependency depth
> and branch depth** — not column count and not planning overhead in general.
> Wide independent arithmetic is the one shape polars does well, via variadic
> built-ins, and it is rare in decision logic. Two corollaries: anywhere the
> system emits polars expressions it must not emit linear chains; and
> diagnostics are *cheaper* in the record tier, since path tracking adds an
> immediate operand rather than expression-tree depth.

---

## 4. The polars↔numba boundary is cheap — with sharp edges

| fact | measurement |
|---|---|
| `Series.to_numpy()` on numeric, single-chunk, non-null | genuine zero-copy view, **0.57 µs/col, flat from 100 to 10 M rows** |
| `frame.to_numpy()` (2D), 8 cols × 1 M | **6109 µs vs 4.7 µs** per-column — **1300× worse** |
| write-back `pl.Series(name, arr)` | adopts the numpy pointer, no copy |
| `with_columns` | ~24 µs fixed + ~10 µs/col, **independent of row count** |
| `hstack(pl.DataFrame(dict))` | ~2× cheaper than `with_columns` |
| fixed round-trip overhead | ~29 µs, or ~16 µs via `hstack` |
| viability floor | **~10k rows** (50% crossover ~12k; below ~3k rows marshalling is >80%) |

**Never use 2D extraction.** Per-column arrays into the kernel, always.

### Nulls are a correctness trap, not just a perf question

`.to_numpy()` on a nullable column copies *and silently changes dtype*:

| polars | → numpy | risk |
|---|---|---|
| Float64 | float64 (NaN) | acceptable |
| Int64 / Int32 | **float64** | lossy above 2⁵³ |
| UInt8 | **float32** | lossy |
| Boolean | **object** | kernel cannot consume |

Worse: **the value slot under a null holds garbage, not zero** — an observed
left join produced `9.0` where the value was null. And `.offset` must be applied
or sliced frames read the wrong window.

Correct extraction is raw arrow buffers at 2.9–3.3 µs/1 M. Note the buffer order:

```python
a = s.rechunk().to_arrow()
validity_buf, values_buf = a.buffers()          # [validity, values] — in that order
values = np.frombuffer(values_buf, dtype)[a.offset : a.offset + len(a)]
```

**A clean column has `validity_buf is None`** — no bitmap is allocated at all,
which makes it the cheapest possible fast-path gate (total gate + extract
measured **0.375 µs**, fully zero-copy, `to_numpy(allow_copy=False)` sharing the
arrow address). `null_count()` at 0.08 µs is the alternative gate.

**`rechunk()` does not reset `a.offset`.** A `slice(9, 12)` still reports
`offset=9`, and ignoring it produced a verifiably wrong validity window. Values
must be sliced `[offset : offset+n]`, and validity must be unpacked *before*
slicing since bit *i* maps to element `offset+i`. `rechunk()` costs 0.33 µs when
already single-chunk but **585 µs/1 M for two chunks** — so rechunk once at frame
entry, never per column.

### How a compiled step sees a null (E8)

Five mechanisms measured at 1 M rows, 20% random nulls, float64 (floor = 256 µs
with no null check at all):

| mechanism | µs/1 M | step body reads |
|---|---|---|
| `types.Optional(float64)` | 1907 | `if bureau_score is None:` |
| `(value, is_valid)` pair | 746 / **357**† | `if not bureau_score_valid:` |
| NaN sentinel | **256** | `if math.isnan(bureau_score):` |
| **NamedTuple `.value`/`.valid`** | 749 / **358**† | `if not bureau_score.valid:` |
| bitmap + row index in the step | 1746 | `(va[j>>3]>>(j&7))&1` |

† validity pre-expanded to one byte per row via `np.unpackbits` — 55 µs/1 M,
once per column. All five compile in nopython mode. `jitclass` also compiles but
costs 24,217 µs; rejected.

**Correcting an earlier claim recorded here:** the in-kernel bit test is *not*
faster than NaN-checking. A random-versus-blocked mask test at identical null
counts shows why — `Optional` and in-step bitmap indexing are 2.6× and 3.3×
slower on random masks, i.e. **branch misprediction**. The pair, NaN and
NamedTuple forms compile to *branchless selects* (ratio 1.00×) because the value
load is hoisted above the branch — which is safe precisely because the garbage
slot is allocated and readable. So putting the bit test inside the step is the
**floor, not the ceiling**: it traps the load inside the branch. `Optional`'s
penalty is intrinsic, ~1465 µs even on a byte mask, because `None`-versus-value
is an unavoidable control-flow join.

> **Decision — and note it does not follow the fastest number.** Use plain
> **`Optional[float]`** (`float | None`, checked with `if x is None`) for steps
> that must distinguish, plus a declared-fill form (`missing_as(0.0)`) that
> substitutes at extraction so the step sees a plain float at the 256 µs floor.
>
> `Optional` costs **1907 µs/1 M**, roughly 1.5 ns/row more than the
> `.value`/`.valid` NamedTuple at 413 µs. That was measured, and the NamedTuple
> was the original recommendation here. It was reversed deliberately on two
> grounds that outweigh 1.5 ns/row given readability-by-non-programmers is a
> stated first-class requirement (§6):
>
> 1. **`if x is None` introduces no new concept.** A `.value`/`.valid` wrapper
>    does, for every author touching nullable data.
> 2. **`Optional` is safer.** It holds `None` when invalid, so the garbage slot
>    is unreachable. The wrapper exposes `.value` and relies on the author
>    remembering to check `.valid` — and forgetting produces a wrong answer, not
>    a crash.
>
> The data did not change; the weighting did. If a hot path ever needs the
> 1.5 ns/row back, the NamedTuple form is measured and available.
>
> Still rejected on correctness: the NaN sentinel, ~100 µs/1 M faster than the
> NamedTuple but unable to represent `Int64` or `Boolean` nulls, with
> `to_numpy()` silently mapping nullable `Int64→float64` (lossy above 2⁵³) and
> `Boolean→object`. Arrow extraction preserves `int64`/`uint8` exactly — a mixed
> `Int64`+`Boolean` two-column step ran at 431 µs/1 M.
>
> For `required` inputs the fast path stays fully zero-copy, failing with the
> offending column named:
>
> ```
> step argument 'instalment' is declared required (`Float64`, no `| None`) but
> column 'instalment' has 3 null(s) in 1000 rows (first at row 7, e.g. rows
> [7, 400, 913]). Either fix the input or declare `instalment: Float64 | None`.
> ```

### Compilation knobs

`fastmath=True` is noise (±5%) — branches, not FP strictness, are the cost.

`prange` **has no fixed row threshold.** An earlier ~50k figure was measured on
one synthetic kernel; E7 concluded the crossover is set by **per-row work**, not
row count: a light body never wins (0.99× even at 1 M rows), while a body
containing a 64-iteration inner loop crosses at ~5k rows and reaches 7.8× at 5 M.

> **⚠ Every figure in that paragraph was refuted — [EXPERIMENTS.md](EXPERIMENTS.md) §F**, across three
> independent runs:
>
> | claim | measured |
> |---|---|
> | trivial body "never wins, 0.99× at 1 M" | **2.83–3.11× at 1 M** |
> | 64-iteration body "crosses at ~5k rows" | **crosses at 500 rows** — 10× early |
> | "7.8× at 5 M" | 7.5–7.7× at **100k**; at 5 M it is **11.4–12.0×** |
>
> The last row suggests a row count migrated between measurement and doc, so E7's
> raw output is worth re-checking for others.
>
> **And "per-row work, not row count" is itself the wrong frame.** The crossover
> invariant is **total serial wall-clock**: bodies spanning 1500× in per-row cost
> (1.4 → 1267 ns/row) all cross between **85 and 239 µs** of serial time, and the
> measured `prange` fork/join floor is **65–67 µs**. Crossover *row count* spans
> 2000× across those bodies; crossover *time* barely moves. `prange` wins once the
> work exceeds fork/join overhead — obvious in hindsight, and it collapses O6 from
> a cost model into one calibrated constant.

`parallel=True` compile cost is **1.2–2.6×, shape-dependent** — not the flat 3×
recorded earlier. Worst observed was 100 flat branches at 9289 ms serial →
23,829 ms parallel.

---

## 4b. Control flow in a fused kernel (E7)

Branch and loop combinators nest subgraphs inside the row loop. Four findings,
one of which overturns a design claim.

### Short-circuiting is real

A depth-5, 3-way nest compiles and genuinely skips untaken arms — verified two
ways: side-effect probes register **0** on untaken arms, and a deliberately
expensive skipped arm gives a **1071×** timing ratio (2.70 ms vs 2888 ms per
200k). 0/300 mismatches against a Python oracle. `continue` is correct (0/500
mismatches).

### Fusion is **non-monotone** — this contradicts an earlier claim

| shape | fused vs separate kernels |
|---|---|
| `step → Branch → step` (1 group), N=1e3…1e7 | fused **1.33–1.61× faster**, 0 NRT allocations, 16 MB of intermediates avoided at 1 M |
| 4 modules | fused **0.75×** |
| 8 modules | fused **0.47×** |
| 16 modules | fused **0.17×** (58.7 ms vs 9.8 ms) |
| 32 modules | fused **0.11×** |

**Mechanism as originally recorded:** past ~3–4 branch groups the fused body
loses LLVM auto-vectorisation (vector IR values drop to 0 at M≥4) and per-group
cost rises 0.6 → 1.9 ns/row even with perfectly predictable routing.

> **⚠ Refuted by direct measurement — see [EXPERIMENTS.md](EXPERIMENTS.md) §E.**
> Vector IR values **never drop to 0**. Across 54 fused kernels the packed-FP
> count grows monotonically with M, opposite to performance; the worst regression
> (0.18×) is fully vectorised with 791 packed ymm ops and zero scalar ops.
>
> The real mechanism is **row-loop unrolling collapse under register pressure**:
> split unrolls ~5× using 7 of 16 ymm registers and spilling nothing; fused at
> M=20 pins all 16, unrolls ~2× and spills 47×/iteration, making a long dependent
> FMA chain latency-bound. The cause recorded in §4c ("register pressure") was
> right; the observable named here was not.

**But the sign flips with arm cost.** When arms are expensive, short-circuiting
dominates — 1.62× at 3 arms, and 1071× when one heavy arm is skipped. So neither
"always fuse" nor "never fuse" is right; it depends on group count *and* arm cost.

> **Conclusion.** Fusion must be a **bounded-size compiler decision, not a
> promise.** Doc 02 §1 and doc 03 §3.2 previously claimed the small-module style
> "costs nothing at runtime" because adjacent modules fuse. That is false past
> ~4 branch groups with cheap arms, where it is 6–9× *slower* than one kernel per
> module.

### Compile time is driven by emitted code size, not nesting depth

**≈ 15 ms per emitted source line** is the single best predictor.

> **⚠ Superseded — [EXPERIMENTS.md](EXPERIMENTS.md) §G.** Compile is **super-linear in emitted lines**,
> ∝ lines^1.4 (R²=0.97), with the local exponent reaching **1.96** between 60 and
> 100 rules. A linear 15 ms/line model over-states small rule sets by ~1.8×
> (8.1–11.5 ms/line under 30 rules) and **under-states 100 rules by 2.3×** — it
> predicts 24 s where the measurement is 57 s. The "~500 lines ≈ 10 s" guardrail
> survives and is slightly conservative (the real crossover is 600–710 lines); the
> linear constant does not.

| shape | compile time |
|---|---|
| one-sided nest, depth 1/4/8/16/32 | 174 / 290 / 469 / 978 / 1903 ms |
| **full-binary nest, depth 4/6/8/10** | 660 / 2481 / **10,268** / **92,330 ms** |
| n-way arms 2/8/16/32/64 | 192 / 378 / 752 / 1230 / 3647 ms |
| flat 50 / 100 branches | 3859 / 9289 ms |
| nested loops, depth 1–5 | 138–274 ms (flat) |

**Fan-out is the wall, not depth.** Emitted code is `arms^depth` when every arm
recurses: depth 10 full-binary costs **92 seconds** to compile. Depth alone is
cheap — 32 one-sided levels is 1.9 s. So the limit to enforce is **emitted lines
(~500 ≈ 10 s), not nesting depth.**

### `prange` composes with inner loops; `break` pays less than expected

`prange` over rows with a bounded inner loop and a carried accumulator produced
**bitwise-identical** output and iteration counts, including with `break`
(5.3× speedup).

`break` works and is correct, but delivers only **~47% of the theoretical
saving** — 1.54× wall-clock despite cutting iterations 3.31×, because the exit
test roughly doubles per-iteration cost. **Early exit pays only when it cuts
iterations by more than ~2×.**

---

## 4c. Fusion at scale (E5)

E5 independently reproduced E7's fusion result, added the row-count dimension,
and identified the mechanism. Split/fused ratio (>1 means fusion wins):

| modules (steps) | 1k rows | 10k | 100k | 1M | fused ns/step @1M | split ns/step @1M |
|---|---|---|---|---|---|---|
| 1 (1) | 1.28 | 1.03 | 1.02 | 1.02 | 0.29 | 0.29 |
| 2 (3) | 1.36 | 1.11 | 0.99 | 0.99 | 0.22 | 0.22 |
| 3 (6) | 1.35 | 0.99 | 0.87 | 0.90 | 0.22 | 0.20 |
| 5 (9) | 1.28 | 0.85 | 0.72 | 0.76 | 0.27 | 0.20 |
| 10 (19) | 1.10 | 0.61 | 0.49 | 0.51 | 0.39 | 0.20 |
| 20 (39) | 0.84 | 0.39 | 0.32 | 0.33 | 0.61 | 0.20 |

**Break-even is ~10k rows.** Below it fusion wins (≤1.4×, pure per-call overhead
amortisation). Above it fusion is neutral at 2 modules and loses from 5 up — 2×
at 10 modules, 3× at 20, 4.5× at 40.

**Mechanism: register pressure.** Split kernels hold a flat **0.20 ns/step**
because each stays small enough to vectorise; the fused body's per-step cost
climbs 0.22 → 0.61 ns as it grows.

> **⚠ Two figures here did not reproduce — [EXPERIMENTS.md](EXPERIMENTS.md) §E.**
> Split ns/step measured **0.82–3.30 depending on body cost**, flat *within* a
> body-cost class but never near 0.20 — which is below the floor of a single
> kernel call on the test machine. And **there is no common break-even row
> count**: a cheap straight-line body never crosses (fusion wins 2.1–13× at every
> n ≥ 1000), a heavy branchy one crosses below 1000 rows, and at n=1 fusion wins
> 9.4–48× rather than "≤1.4×".
>
> The discriminator is **per-step body cost relative to the boundary floor**
> (~0.32–0.76 ns/row/kernel plus 0.44 µs dispatch), not module count and not row
> count. Measured spread at fixed module count and fixed row count: **62×**,
> 13.00× to 0.18×, with the sign of the decision flipping. So *"boundary stores
> are genuinely near-free"* holds only relative to a body that costs more than
> they do. Crucially this is **not** caused by
per-module params bundles — a single merged bundle degrades identically (0.790
vs 0.799 ns/block at 20 modules), so the params design is exonerated.

### The premise was backwards, not the guidance

**Boundary stores are genuinely near-free.** Even with a *full polars column
round-trip* at every boundary, split still wins at 1 M rows / 10 modules
(4.1 ms vs 7.5 ms). Polars boundaries only flip the answer below ~100k rows
(~14 µs fixed per crossing).

> So "many small modules" is safe — **but not because they fuse.** It is safe
> because crossing a boundary costs almost nothing. Keep the guidance, reverse
> the mechanism.

> **A later correction to this conclusion.** It originally ended "and cap fused
> groups at ~6–9 steps." Read against the table above, 6 steps is 3 modules
> (0.87–0.90) and 9 steps is 5 modules (0.72–0.76) — both already worse than not
> fusing above 10k rows, so the cap named the range this section identifies as
> harmful. More importantly, E7 puts the same decision at 1071× in the other
> direction when a heavy arm is skipped, and no constant spans four orders of
> magnitude. The cap is withdrawn in favour of explicit fusion (doc 02 §1.2).
> The measurements are unchanged; the guidance drawn from them was wrong.

### Everything else about composition holds

| question | result |
|---|---|
| N distinct params NamedTuple types in one driver | **Yes** — exact at 2/5/10 and up to **80 modules / 84 args** (4.5 s compile). Mixed float/int/bool/nested-tuple/ndarray fields fine. Same module type instantiated 4× fine. |
| `shared` threaded to only some modules | **Yes** — correct, `signatures=1`; an unused bundle measured −3.7 µs per 1 M rows, i.e. free |
| a tap mid-chain | **Cheap, and does not split the kernel** — +0.11 ns/row/tap at 1 M (linear to 4 taps), vs +2.0 ns/row for a real split. **~17× cheaper than a split** |
| overlapping value versions (`term_cap` twice) | **Yes** — plain local reassignment, exact against a Python reference |
| retune without recompile | **Yes** — `signatures` stayed 1 across 15 retunes (5 modules × 3 values); negative control: changing a field float→int *does* add a signature |

Two incidental corrections: `inline='always'` is a **no-op** (numba already
inlines these; hand-inlining matches codegen within 3%), and `fastmath=True`
buys **2–2.5× on an arithmetic-heavy fused driver** — so the earlier "fastmath is
noise (±5%)" finding was workload-specific to *branch*-dominated logic, where FP
strictness genuinely isn't the cost. It is worth enabling selectively, not never.

### ⚠ The one real typing wall — a silent, permanent performance trap

Two **distinct** NamedTuple classes sharing the same `__name__` *and* the same
field names blow per-call dispatch from ~1 µs to **15–24 µs — permanently, for
every call involving that name.** The numba types print identically
(`Z(float64 x 2)`) but compare unequal, so the dispatcher's cache thrashes.

> **⚠ The magnitude did not reproduce on numba 0.67 —
> [EXPERIMENTS.md](EXPERIMENTS.md) §D.** Measured **1.03×** (1.006 → 1.039 µs),
> with 39 of 40 independent trials in 0.84–1.08×, and it does **not** persist: a
> clean driver measured 1.02× after a colliding one ran 2000+ times.
>
> **The collision itself is real** — `compute_fingerprint` encodes only
> `__name__` + field names + field types — so this is a **correctness** defect,
> not a performance one. That inverts the guard: doc 05 §9 criterion 8's timing
> assertion has *zero power* and would mark the bug green. It must be an exact
> structural check.
Different field names avoid it entirely.

This silently contaminated E5's own first benchmark run (45 µs of phantom fixed
overhead at 5 modules) before being found.

> **Mandatory implementation rule**, and it is smaller than it first appears:
>
> 1. **Derive the bundle's class name from the module id.** Module ids are
>    already unique, so collisions *between different modules* become impossible
>    for free — no new check needed.
> 2. **Memoise generation on the pydantic model** —
>    `@lru_cache` over `_bundle_for(model)`. Models are defined at import, so
>    bundles are too. This closes the only residual case: the *same* module's
>    bundle being regenerated per pipeline build, which uniqueness checking
>    cannot see because the id is legitimately identical both times.
>
> One known wrinkle: memoising ties bundle identity to the model *object*, so
> redefining the model — which happens on every notebook cell re-run — yields a
> new bundle. That is technically correct, but interactive use is exactly where
> it would bite, so a regression test asserting dispatch stays near 1 µs is worth
> having.

> **Conclusion, superseded.** This originally read "compile serial and `prange`
> variants at warmup, dispatch on row count". Both halves are withdrawn: the
> figures above were refuted ([EXPERIMENTS.md](EXPERIMENTS.md) §F) and every automatic selection rule was
> ruled out — a fixed row threshold scores 66.7%, a warmup probe 52.8%, and the
> one rule that scores well (97.9%) was fitted on uniform bodies that credit
> logic's early exits do not resemble. `prange` is now **authored**, via a
> `parallel(...)` combinator, with serial the default and one variant compiled per
> kernel (doc 05 §5.1). `fastmath` stays off by default.

---

## 4d. The calling convention at width (E9)

The driver's original convention was one argument per input and output column.
The worry was a ceiling, since a realistic wide shape can need several hundred outputs.

**There is no ceiling.** Positional arguments work to at least **1600**, and the
failure mode is compile time rather than an error — compile scales ≈O(W^1.8),
driven by *argument count* rather than body size (1600 args with an 8-statement
body cost 44 s versus 54 s with a full body). Dispatch is linear at ~54 ns/arg.

But a better convention exists. At 200-in/200-out with a heterogeneous branchy
body:

| convention | compile | µs/call | ns/row | assemble |
|---|---|---|---|---|
| positional | 36.7 s | 18.9 | 1826 | – |
| tuple (literal index) | 44.4 s | 26.5 | 1337 | 0.3 µs |
| `typed.List`, indexed in loop | 2.8 s | 7.6 | **9950** | 138 µs |
| `typed.List`, hoisted | 2.8 s | 5.8 | 1311 | 138 µs |
| 2D array | 8.3 s | 0.41 | 242 | 3047 µs |
| **record (structured) array** | 7.4 s | **0.50** | **241** | 3100 µs |

End-to-end at 100k rows, 200-in/200-out: positional **202 ms** total and
20.8 µs/call, versus record/record **101 ms** and **0.54 µs/call** — 2× better
end-to-end and 37× per call.

**Mixed dtypes settle it.** Real columns are a f8/i8/bool mix, and the record
array is the *only* convention that carries one with O(1) dispatch:

| convention | mixed 300-in/300-out |
|---|---|
| positional | works — 9.0 s, 32.8 µs, 1251 ns/row |
| heterogeneous tuple | works, worse |
| tuple + runtime index | **TypingError** |
| `typed.List` | **AssertionError** on construction |
| 2D array | **impossible** — single dtype only |
| **record array** | works — 2.1 s, **0.67 µs**, **121 ns/row** |

**The target shape is verified:** 400-in/633-out mixed over 100k rows compiles,
executes in 72 ms, answers a single record in 0.99 µs, and matches a numpy
reference exactly.

> **Conclusion.** Use a **1D structured (record) array per side**. Not because
> positional breaks — it doesn't — but because it is 2× slower end-to-end, 37×
> per call, and 4.7× slower to compile at the target width, and because it cannot
> carry a dtype mix with O(1) dispatch.
>
> Hybrid threshold **N ≈ 64 columns per side**: below it positional dispatch is
> <4 µs and the assembly copy (~0.93 ns/element) doesn't pay for itself. That
> figure is **interpolated between measurements at 50 and 200**, not pinned — if
> one convention is preferred, record arrays are never much worse.

### Four things to design around

1. **`exec`'d drivers cannot be cached** — `RuntimeError: no locator available
   for file '<string>'`. Writing generated drivers to a real `.py` takes 1200
   arguments from **29.5 s cold to 0.80 s warm**, which converts the entire
   compile blowup into a build-time cost. Do this regardless of convention; it is
   the single highest-leverage implementation decision here (doc 02 §3.4).
2. **Extracted arrays are `writeable=False`**, so numba types them `readonly` —
   **a distinct signature from writeable arrays.** Mixing the two produces two
   compiled specialisations of the same driver. Pick one and be consistent.
3. **Bool columns are not zero-copy** from polars — Arrow stores them bitpacked,
   `allow_copy=False` raises, and the copy costs ~48 µs/col at 100k rows.
4. **Record write-back becomes the new bottleneck** — gathering strided fields
   back out costs 35–158 ms, which is **54.7% of total time at 633 outputs**.
   Still the best configuration measured, but it is where to optimise next.

> **And one honest limit, consistent with §2–§3.** At this width numba *loses* to
> vectorised numpy on simple bodies (101 ms vs 36 ms). The fused row loop's
> advantage is **branchy logic, not throughput** — the same conclusion the depth
> benchmarks reached, arrived at from the opposite direction.

---

## 5. Failure modes observed in `decider`

Framework behaviours that forced workarounds on users, each cited in `decider`'s
own source. Where a consequence is quantified, the counts come from **one large
internal project** built on `decider` — a port of a legacy procedural system,
not an example of ideal authoring, but a genuine stress test of what the
framework's design compels. That project's contents are not described here; only
the framework-ergonomics evidence is.

### 5.1 The root defect: a step cannot overwrite its own input

`decider` forbids a function having a parameter with the same name as itself —
self-reference raises `ValueError: Circular dependency detected`. But updating a
field in place is the natural shape of waterfall logic.

**The ban is emergent, not intended**, which matters because it means it is
fixable by construction rather than being a deliberate constraint. There is no
self-reference check anywhere. The mechanism is:

- `decider/modules/functional.py:181-185` — after building all nodes,
  `expand_nodes` rebinds any input key matching another function's name:
  `node.input_map[k] = internal_nodes[k]`. For `def a(a: pl.Expr)` that rebinds
  the node's input to *the node itself*.
- `decider/graphutil.py:38` — `topological_sort` (defined at `graphutil.py:14`)
  then detects the resulting cycle and raises:
  `raise ValueError(f"Circular dependency detected: {cycle}")`. This is the
  **only** such raise site in the framework.

Consequences, measured across 66 module directories:

- **79 identity-passthrough functions** (`return <param>`), AST-verified, existing
  purely for wiring
- **Five parallel naming conventions** invented to dodge it: 38 `updated_*`,
  8 `*_result`, 4 `*_raw`, `*_pt`, and **~90 `existing_*`**
- **`name_override`** as a dedicated framework escape hatch
  (`decider/modules/functional.py:61`, stored as a ClassVar at `:169`). Note it
  is applied as a **post-hoc column rename** inside `TModule.execute`
  (`functional.py:190-206`, the rename at `:205`) — not at node-naming time,
  which is exactly why it can satisfy a downstream consumer while the graph still
  wires on the internal name
- **Modules whose only job is renaming**, e.g. `AliasCombineModule`
- **Private names leaked into the public column namespace** — `_stage_01`
  and 9 siblings registered as module outputs
- One value threaded through `_stage_00` → `_stage_09` in a
  900-line module containing **313 `when()` calls**, with all **76 functions
  hand-ordered topologically**

It has already caused a **silent** failure: an earlier revision exposed outputs
under `updated_*`, which "left a downstream module unable to see this
step's actual result when the two are composed" — wrong wiring, no error.

> **Design response.** Overwrite semantics in the source; versioning internal;
> the version chain becomes the audit trail. See doc 03.

### 5.2 Composition is not a DAG

`|` (`decider/modules/core.py:35`, overridden at `modules/expression.py:281` and
`modules/primitives/sequential.py:135`) composes modules as sequential
frame-transform stages in list order, not a dependency-resolving DAG.
`SequentialModule` (`modules/primitives/sequential.py:64`) executes them at
`:102` — `for step in self.steps:` — with no topological sort anywhere in that
file.

Only the **`expr`** kind resolves dependencies. `topological_sort`
(`graphutil.py:14`) has exactly two call sites: `executor.py:79` for the frame
graph and `executor.py:134` for the expression graph. The `record` and `frame`
kinds run functions in **declaration order** — `modules/record.py:66-73` and
`modules/functional.py:246` — documented at `modules/record.py:18-20`:
*"functions are called in the order given — there is no topological sort"*.

One logical module is therefore split into **nine** registered stage modules of
which **21 of 22 functions are identity aliases** — "whole module is glue" —
with `a_risk_category` defined twice, the second silently shadowing the first.

Execution order is separately maintained by hand in JSON with typed UUIDs, and is
known to be wrong: `main.json` has 51 steps, generated `flow.json` has 120, the
true C# flow has 192; documented as "**Unresolved**". Elsewhere, ordering is
asserted in a *comment*: `# Order_1, TL:0, AF:0, CC:0, Order_2, CC:1, …`.

> **Design response.** Execution order is always *derived* from the graph. Never
> hand-maintained, never in config, never in a comment.

### 5.3 The governance story never materialised

The framework supports pydantic-typed config injection
(`decider/modules/functional.py:141-162` collects the config class and per-function
injection flags), and its reference projects demonstrate it. In the production
project it is used **zero times**:
0 `BaseModel` subclasses, 0 typed constructor params, **0 validators** anywhere.
Instead there are **546 inline `pl.lit(<number>)` literals**.

And the values that *are* externalised are **duplicated**:
an internal config file defines `"max_term": 84`
while `84.0` is also hardcoded in two modules. A business user can edit the
readable JSON and the behaviour may not change.

The mechanism that replaced it injects tunables as **frame columns** via
`pl.lit()` from `t.Dict[str, t.Any]` — untyped, unvalidated.

> **Design response.** One canonical location per parameter; validated; and the
> tunable form must be *cheaper to write* than the literal, or it will not be
> adopted. Whichever path is cheaper is the path people take.

### 5.4 No observability, and diagnostics were never lifted

`grep -rE "def (debug|explain|inspect)|DEBUG" decider/` → **0 hits**. No logger,
no prints, no debug flag anywhere in the extension code. All 37 debug scripts
live outside the framework, reaching into `benchmark.py` internals.

There *is* a field named `debug` (`decider/executor.py:61`, `debug: bool = False`)
but it is **effectively a no-op** for observability: collection is gated on the
separate `collect: bool = True` (`executor.py:62`) in both the debug and
non-debug branches of `Executor.execute` (`executor.py:81-94`). A flag named
`debug` that does nothing observable is worse than no flag.

Three *inconsistent* per-module diagnostic mechanisms exist for the same need —
`flat_rules`' `output_fn` (`decider/modules/rules/flat_rules/module.py:173-175`),
`ScoreCard.expose_intermediates`
(`decider/modules/credit/scorecard/module.py:260`), and
`ExecutionPlan.execute(audit=True)` (`decider/plan.py:32-37`, the `audit`
parameter at `:35`) — and the tree modules have no hook at all. `output_fn` is
`null` in every production config; only tests exercise it.

The `flat_rules` mechanism did establish one genuinely good idea: path
information is a **compile-time constant per branch** (`pl.lit` resolved along
each branch), so it costs no extra passes. That generalises directly — in numba
it becomes an immediate stored to one `int64` column.

Two flaws to avoid reproducing: `output_fn` is a `{module_name, function_name}`
code pointer stored *in config*, which forfeits config-as-inspectable-data; and
leaf ids are auto-generated and unstable, so path codes are not comparable
across versions.

> **Design response.** Diagnostics are a framework-level concept, declared as
> data (`taps=[...]`), with deterministic node identity.

### 5.5 They were already hand-rolling the record tier

Ten `.collect()` sites exist purely to escape the lazy frame, e.g.
one internal module — `df = input.collect()` then
`results = [_process_row(r) for r in df.to_dicts()]`. That is a per-record
scalar kernel, implemented in the slowest available form because the framework
offered no such tier.

> This is the strongest argument for the record tier: it is not speculative
> demand, it is existing production code working around the framework.

### 5.6 Verification lives outside the framework

One pytest-style test file for 66 module directories, six assertions total — and
`pyproject.toml` sets `testpaths = ["tests"]`, so **it is not collected by a
default pytest run**, including the one test written specifically to cover a gap
the trace harness cannot reach. Real verification is 77 shell harnesses diffing
against a fixed-sample C# oracle corpus, scored as aggregate per-column correctness
percentages, with no checked-in fixtures.

> **Design response.** Correctness is specification-based: rule-level assertions
> expressing intent. Golden-trace comparison remains available as a capability
> but is not the theory of correctness — there is no permanent oracle.

---

### 5.7 Config resolution — measured, and one live bug

Three designs were mocked with ~50 module types and benchmarked on a 63-node,
3-level pipeline config (pydantic 2.13.4):

| design | nested validate | register 175 types | graph as plain data |
|---|---|---|---|
| A — union over generated classes (status quo) | 74.1 µs | 333 ms | needs a *valid instance* |
| B — plain `dict[str, Definition]` registry | 73.5 µs | 44 ms | 1 line |
| **C — hybrid (data graph + thin model in a union)** | **58.5 µs** | **58 ms** | **1 line** |

Findings:

- **The union is not faster than a registry.** A and B are within 1%.
  Discriminator dispatch is a hash lookup; so is a dict lookup. Validation cost
  is flat in N for both (A′ 40→43 µs from N=10→200).
- The available 1.7× win comes from **native pydantic-core recursion**
  (`steps: List[union]`), which the status quo forfeits by validating each step
  from Python. The field is declared
  `decider/modules/primitives/sequential.py:75` — `steps: t.List[t.Any]  #
  BaseModule; use Any to allow discriminated deserialisation` — with a
  `@field_validator("steps", mode="before")` at `:77-89` dispatching each item
  through `GraphModule.model_validate(item).root` (`:86`). The inline comment
  states the tradeoff was deliberate; the measurement shows what it costs.
- **Live bug:** A cannot adopt native recursion as built. Rebuilding the union on
  every registration leaves *stale nested unions*, and `model_validate` then
  fails with ~100 spurious errors, because each rebuild advances freshness only
  one level. Native recursion requires building the union **once, after all
  registrations**.
- Registration is **superlinear today — growth exponent 1.39**.
- A **loses the step index** in nested errors: a bad id three levels deep reports
  loc `('sequential','steps','sequential','steps','sequential','steps')`, so the
  failing step is unidentifiable. C reports exact indices.
- A cannot render a graph without instantiating it — `expand_nodes`
  (`decider/modules/functional.py:171-186`) calls `Node.from_callable`
  (`decider/modules/expression.py:148`) with `{"config": self}`, so drawing a
  diagram would require inventing valid params for every registered type. This is
  the concrete mechanism behind "generated classes are not introspectable".
- Registration itself is `create_extendable_model` (`decider/_ext.py:65-106`),
  reached via `register_graph_module` (`decider/modules/_ext.py:9-13`, a tuple
  unpack from that factory). Extensions are imported at
  `decider/initialization.py:15` (`initialize_decider`), with the
  `importlib.import_module` sites at `:44-47`, `:50-57` and `:60-62` — which is
  where the "all registration before first validation" constraint has to be
  enforced.
- **All three designs silently ignore a misspelled param**, reporting only the
  real field as "missing". Params models need `extra="forbid"`.
- A's `union_tag_invalid` message is 1332 chars and lists every registered tag.
  The registry's 127-char *"unknown module id 'x' (51 registered). Did you mean:
  …"* is materially better for humans.

> **Conclusion.** Adopt **C**: graph as plain data in a registry, plus a thin
> generated model per definition (`type: Literal[…]` + params, no graph)
> participating in a discriminated union purely for config validation. Build the
> union once at finalisation; wrap the tag error with difflib suggestions; set
> `extra="forbid"` on params models.

### 5.8 Concentrated comparison

Every `decider` behaviour referenced above, where it lives, and what `decider2`
does instead. All citations verified against source.

| behaviour | `decider` location | `decider2` |
|---|---|---|
| Kind chosen by type annotation | `modules/functional.py:21` (`_detect_module_kind`) | Two tiers chosen semantically; no annotation sniffing |
| Module is a generated class | `modules/functional.py:167` (`class TModule(...)`), `:397` (`create_model`) | Pydantic *instance* describing the graph (doc 02 §2) |
| Wiring by rebinding `input_map` | `modules/functional.py:181-185` | Declared inputs/outputs; overwrite legal (doc 03 §3) |
| Self-reference → cycle error | `graphutil.py:38`, via `graphutil.py:14` | Internal versioning; `x = f(x)` is the normal case |
| `name_override` post-hoc rename | `modules/functional.py:169`, applied `:190-206` | Unnecessary — output name is declared directly |
| `\|` is list-order, not a DAG | `modules/core.py:35`; `modules/primitives/sequential.py:64`, `:102` | `\|` builds a dependency-resolved graph |
| `expr` sorts, `record`/`frame` don't | `executor.py:134` vs `modules/record.py:66-73`, `modules/functional.py:246` | Always topologically sorted; order matters only where a value is overwritten |
| Fan-out materialise-vs-inline heuristic | `executor.py:143-153`; `modules/expression.py:119-123` | Deleted — a register value reused twice is free |
| Config injection as class fields | `modules/functional.py:141-162` | `params` as a per-invocation validated `NamedTuple` (doc 02 §4) |
| Registration rebuilds the union | `_ext.py:65-106`; `modules/_ext.py:9-13` | Union built **once** at finalisation (doc 02 §2.1) |
| Per-step Python validation | `modules/primitives/sequential.py:75`, `:77-89` | Params inlined → native pydantic-core recursion |
| Extension import/registration | `initialization.py:15`, `:44-62` | Same shape; must complete before first validation |
| `debug` flag that does nothing | `executor.py:61` (vs `collect` at `:62`) | Four execution modes, equivalence-tested (doc 02 §3.1) |
| Three inconsistent diagnostics | `modules/rules/flat_rules/module.py:173-175`; `modules/credit/scorecard/module.py:260`; `plan.py:35` | One framework-level concept: declared `taps` as data |
| Branch merge upcasts to string | `modules/primitives/branching.py:96` (`how="diagonal_relaxed"`) | Declared output schemas; no implicit dtype reconciliation |
| njit + codegen'd fused driver | `modules/record.py:45` (`jit: bool`), `:108-183` (`_build_jit_driver`) | Generalised: the default path, with per-node fallback and four modes |

The last row matters most: **the mechanism `decider2` is built on already exists
in `decider`** — `_build_jit_driver` njits each function and codegens a fused row
loop. It is opt-in behind a `jit` flag, available only for `kind="record"`, has no
fallback, no observability and no topological sort. `decider2` makes it the
default and builds the missing pieces around it.

## 6. Requirements gathered from stakeholders

- Three personas: **data scientists** write most logic and test it; **software
  engineers** supply harder custom logic; **business users** adjust values and add
  rules (AI-assisted) and must be able to verify a change safely.
- Non-programmer readability is a first-class requirement. Note that both current
  representations fail it: the Python (a rule's name in a function name, its
  effect a `min_horizontal`, its waterfall position implicit in a parameter name)
  *and* the declarative alternative (995 lines of JSON AST with UUID node ids and
  `result_idx: -1` indirection for ~30 rules).
- Full tracing desirable; OTel integration welcome **if it does not measurably
  slow production**.
- Must support mid-execution value override to explore "what would happen if".
- Steps should be **pure** — inputs → outputs — and small enough to understand
  without needing their intermediates.
- Deployment is batch and long-lived realtime endpoints only. Compile latency is
  therefore a non-issue and can be paid at warmup.
- **The single-record path is the primary one.** Most invocations send one record
  and need realtime latency. Batch runs at most once a day and may be materially
  slower. This is a priority, not just a list of two modes, and it reweights
  several results — see below.
- Params may arrive per invocation, including in a realtime request payload.
- The framework must not be prescriptive about types: if a step needs regex or
  awkward Python, it should work, just slower.

### 6.1 What "single record first" reweights

Most measurements in this document were taken on batches. Read against the stated
priority, several change in importance or in sign:

| finding | under batch | under N=1 (primary) |
|---|---|---|
| **fusion** (§4b, §4c) | non-monotone; 0.11–13× depending on body cost | fusion wins **9.4–48×** — per-call dispatch dominates, so it is nearly always right |
| **`prange`** (§4) | an opt-in annotation | **irrelevant** — nothing to parallelise across one row |
| **dispatch floor** (0.44 µs/kernel) | a rounding error | still small — see the budget note below |
| **record write-back** (§4d, 54.7–64% of batch total) | the dominant cost | **absent** — there is no bulk write-back |
| **output convention** | column-major 2D wins 1.74× end-to-end | column-major is **3.7× worse per record**; row-major is best |
| **polars boundary** (§4) | ~600 µs per nullable column | **absent** — `score()` bypasses polars (doc 02 §3.5) |
| **staged config swap** (doc 08 §4) | matters daily | matters continuously — the endpoint is long-lived |

**The realtime budget is low single-digit milliseconds per record**, as low as
achievable. That number is the most important one in this document, because it
sets what is worth optimising — and it is ~1000× larger than the compiled path.

| cost | measured | share of a 1 ms budget |
|---|---|---|
| single record through the kernel (400-in/633-out) | 0.99 µs | **0.1%** |
| one kernel-boundary dispatch | 0.44 µs | **0.04%** |
| fusion's n=1 advantage (9.4–48×) | saves ~tens of µs | **a few %** at most |

> **So the compiled path is not the realtime problem, and optimising it further is
> not where the budget goes.** Everything measured so far concerns a layer that
> already fits ~1000× over. What fills a low-ms budget is the surrounding work —
> params validation, request marshalling, the `score()` calling convention at 400
> inputs, allocation, GC, and tail behaviour under concurrency — **all now
> measured (N1–N4, below).**

### N1 update — the surrounding work costs ~1 ms as naturally implemented, not "negligible"

Measured directly (`experimentation/single-record-overhead/`, EXPERIMENTS.md §N1):
the framework overhead around one `score()` call, at exactly this 400-in/633-out
shape, is **971 µs p50 / 1048 µs p99 — 4.9–5.2% of a 20 ms budget** as the doc's
own conventions (doc 03 §4's pydantic→NamedTuple params, doc 05 §3's record
array) most naturally imply implementing it: kwargs in, then a 1-row record
marshalled and read back **one named field at a time**, 400 fields in and 633
out. That contradicts this section's working assumption that only the ~1 µs
kernel matters — **92% of the 971 µs is two per-field Python loops, not the
kernel** (confirmed separately: the njit call itself, isolated from a stray
per-call `np.empty()` that had been hiding inside "kernel dispatch," is 1.38 µs —
this section's 0.99 µs figure stands). Writing those two loops as one bulk
call each (`rec[0] = tuple(...)` in, `row.item()` out — same record type, same
output, assert-verified identical) cuts the total to **145 µs / 0.73% of budget**,
a 6.7× reduction with no loss of readability either way. **Doc 05 §3 should
specify whole-row marshal/readback for the realtime path, not per-field**, and
this is the maintainability-neutral case the review criterion asks for: the
faster form is not a less-readable one.

Two consequences for the design:

1. **Stop tuning the kernel for realtime; measure the request path.** Fusion,
   `prange` and the output convention are *batch* optimisations. They should be
   chosen on batch evidence and simply not regress the single-record path.
2. **The output convention may still differ per entry point**, because the two
   paths optimise opposite things — `score()` pays kernel time and no write-back,
   `apply()` pays write-back and can afford a slower kernel. But at a low-ms budget
   the `score()` side of that choice is worth ~2.7 µs per record, so it should be
   settled on whichever is simpler unless a request-path measurement says otherwise.

### N2 update — doc 02 §3.5's literal kwargs convention costs more than everything else in this section combined

Measured directly (`experimentation/n2-calling-convention/`, EXPERIMENTS.md §N2):
N1's accept phase used a `**kwargs` catch-all as a cheap proxy (19.18 µs) for doc
02 §3.5's literal example — a real 400-named-parameter signature
(`score(net_income=42000.0, expenses=18000.0, ...)`) called with keyword
arguments. That proxy understated the real cost by **~63×**. The literal form
costs **1190 µs p50 at 400 inputs — 5.95% of a 20 ms budget**, on the calling
convention alone, more than N1's entire measured framework overhead for
everything downstream (971 µs). Isolated, the cost is CPython's
keyword-argument **binding**, not dict construction (a body that only binds, no
dict built, costs 1094 µs; the identical signature called *positionally* costs
28.8 µs — 39× less, doing strictly more work), and it scales close to
quadratically with parameter count, not linearly — confirmed independently
outside the harness with `timeit`. Every alternative convention stays under 1.1%
of budget at the same width: dict 60.1 µs (0.30%), a caller-reused record buffer
39.8 µs (0.20%), positional 93.3 µs (0.47% — but a silent-transposition hazard,
rejected on correctness grounds regardless of speed).

**Consequence for doc 02 §3.5 / doc 03 §6: specify `score(request: dict, *,
params)` as the primary realtime convention, not literal per-field keyword
arguments at width.** Doc 02 §3.5's own example (3 named args) is not wrong —
its unstated generalization to this document's own established realistic width
(§4d: 400 inputs) is. At that width no caller hand-types the call anyway; a
dict-shaped payload is what a real request already looks like, and spelling it
as 400 keyword arguments costs 39× more to bind for the same data. kwargs syntax
remains fine, and reads best, for small hand-written calls (tests, a low-arity
module) where its cost is irrelevant. A reused record buffer is the cheapest
option but should stay an explicit opt-in for a demonstrated hot loop, not the
default — it is only 1.5× cheaper than a dict (a 0.1%-of-budget difference) for
20× the caller-side complexity (buffer lifetime, field-order knowledge,
staleness across a doc 08 §4 config swap).

Both the ns/row figures behind row 5 and the dispatch share in row 3 are derived
from batch runs and **need re-measuring at genuine N=1** before doc 05 §3 commits.

### N3 update — params validation is affordable at every scale tested; the model→NamedTuple conversion, not validation, is the larger and cacheable cost

Measured directly (`experimentation/params-validation-n3/`, EXPERIMENTS.md §N3):
doc 02 §4's *"params may arrive per invocation, including in a realtime request
payload"* costs **256.5 µs p50 — 1.28% of a 20 ms budget** for the full
`resolve_params(doc, origin=..., complete=True)` path (doc 08 §6.2) at 50 module
instances (doc 03 §4.1's "a realistic pipeline has many"), and proportionally
less at 1 or 10 modules (9.7 µs / 0.05% and 56.7 µs / 0.28%). That is under the
whole-millisecond flag this batch was told to raise, so the design question the
task brief posed — must a realtime request reference a pre-validated bundle by id
rather than carry raw params — **cannot be settled on performance grounds; it
stands or falls on doc 04 §2.1's governance argument alone.** The model→NamedTuple
conversion doc 03 §4 describes as happening "under the hood" is, at scale, the
*larger* of the two costs (170.6 µs vs 78.3 µs validation at M=50) — and it is
also the one worth caching: a content-memoized lookup is 726–765× cheaper, flat
at ~0.23 µs regardless of module count, because a repeated params document
collapses to one dict lookup. `ParamsCell.get()`/`.swap()` confirm doc 08 §4's
batch-context figures hold at genuine N=1 (`get()` 0.159 µs; `swap()` 0.278 µs
p50 vs the batch figure's 0.177 µs — same order of magnitude, p99s agree almost
exactly at 0.363 vs 0.357 µs). Where the cheapest possible params path matters,
one already exists at ~1,600× less cost than `resolve_params` at M=50:
`ParamsCell.get()` on an already-swapped bundle — doc 08 §4's staged-swap model,
not per-request validation.

### N4 update — the tail is real (19.2% of budget, worst case) but GC is not its cause; concurrency is where the budget actually gets threatened

Measured directly (`experimentation/n4-tail-concurrency-swap/`, EXPERIMENTS.md
§N4): the first tail measurement in this document set. **Single-thread steady
state** (30,000 `score()` calls): p50 1037.5 µs (5.2%), p99 1114.2 µs (5.6%),
p99.9 1723.6 µs (8.6%), **max 3840.5 µs (19.2% of a 20 ms budget)** — a real tail,
but not attributable to GC: the run saw one GC event total, coincident with zero
of the calls at or above p99, and **`gc.disable()`/`gc.freeze()` produced no
measurable tail difference against baseline** (M2) — refuting the brief's
proposed GC-drives-the-tail mechanism at this allocation shape. **Config swap**
(doc 08 §4's `activate()`) is confirmed cheap for in-flight serving, closing the
gap doc 08 §4 left open: worst call anywhere across 758,215 calls and 30 swaps
was 1.37% of budget; the specific first-call-after-swap cost is 2.1× the steady
median but only 0.074% of budget in absolute terms.

**The concurrency result is the one that changes a recommendation.** Serving the
same compiled kernel from 1–16 concurrent threads (no background compile
involved — a different scenario from EXPERIMENTS.md §H, which doc 08 §4 already
acted on) shows `nogil=True` holding p99/max flat (4–12% of budget) across every
thread count tested, while the identical kernel compiled `nogil=False` degrades
from fine-at-1-thread to **p99 = 1270% of budget (12.7× the entire budget) and
max = 378.7 ms at 16 threads** — a convoy effect from the GIL-holding kernel
serializing calls under contention. **Doc 08 §4's existing `nogil` guidance is
scenario-specific (compile-vs-serving) and reads, uncorrected, as a general
recommendation against `nogil=True` — which this shows would be actively
dangerous for concurrent request serving.** Doc 02 §3.5 / doc 08 §4 should state
`nogil=True` as an unconditional requirement for serving kernels.
