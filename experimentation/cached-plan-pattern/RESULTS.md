# The composed/cached-plan pattern for decider2 — measured

Three questions, in the order the brief asked them. The headline number for
Q1 comes first because the coordinator's refinement made it the thing that
matters most regardless of what happens to `rtlf` itself.

Machine: Python 3.14.5, numba 0.67.0, polars 1.41.2 (repo `.venv`), same
shared box as the sibling experiments. `free -h` checked before every run;
nothing here allocates more than a few hundred MB (100k rows × ≤10 float64
columns). Raw measurements: `results.jsonl` (one line per measurement,
flushed immediately). Reproduce with `<repo>/.venv/bin/python polars_bench.py`
and `<repo>/.venv/bin/python q2_bench.py`.

---

## Headline: of polars' per-call cost, what fraction is the optimizer?

**Answer: a small and shrinking fraction — 1–3% at realistic depth, up to
~15% only for the smallest trees where nearly everything is fixed
overhead. The optimizer was never the dominant cost. Physical-plan
construction is**, and no supported public API can cache that across calls
— only `rtlf`'s internal-API slot-injection technique reaches it.

This was measured two independent ways, entirely through polars' own public
API (`polars_bench.py`), on a linear `pl.when` chain (rtlf's own benchmark
shape), batch=1000:

| depth | `collect()` | `collect(optimizations=none())` | optimizer share (collect-delta) | optimizer share (explain-delta, corrected) |
|---:|---:|---:|---:|---:|
| 1 | 32.6 µs | 27.8 µs | 14.7% | 31.2%† |
| 5 | 104.4 µs | 89.1 µs | 14.6% | 19.4%† |
| 10 | 209.4 µs | 185.5 µs | 11.4% | 13.9% |
| 25 | 732.1 µs | 680.9 µs | 7.0% | 7.3% |
| 50 | 2274.4 µs | 2178.7 µs | 4.2% | 6.5% |
| 100 | 7862.4 µs | 7633.0 µs | 2.9% | 0.2%‡ |
| 200 | 28619.5 µs | 28260.7 µs | 1.25% | 0.9% |
| 400 | 110646.8 µs | 109274.7 µs | 1.24% | 0.7% |

† noisy at trivial depth — absolute times are a few tens of µs, close to
timer resolution. ‡ this single low point is the explain-delta method's own
noise floor (see below), not a real dip.

Method 1 (`collect_delta`): `lf.collect()` (optimize + build physical plan +
execute) vs. `lf.collect(optimizations=QueryOptFlags.none())` (skip the
optimizer's rewrite passes, still build the physical plan, still execute).
Method 2 (`explain_delta`): `lf.explain(optimized=True)` calls the exact
same internal `to_alp_optimized()` that both `collect()` and rtlf's
`RealtimeLazyFrame::new()` call; `lf.explain(optimized=False)` formats the
*raw* IR with no optimizer at all. Both format a string, so their **delta**
cancels the (large — see below) formatting cost and should leave
approximately the optimizer's own time. The two methods land within a few
percent of each other at every depth past the noise floor — a genuine
cross-check, not a single number taken on faith.

**What surprised me here, and is worth flagging on its own:** my first
attempt used `explain(optimized=True)` *alone* as an optimizer-cost proxy,
and it was badly wrong — at depth 300 it suggested the optimizer was
**~32% of total cost** (`explain(optimized=True)` = 18,246.7 µs against a
`collect()` of 62,921.7 µs). Comparing against `explain(optimized=False)`
(17,636.4 µs — *no optimizer at all*) showed why: **string-formatting a
deep expression tree, not the optimizer, is what `explain()` mostly
measures.** The optimizer's real share, once formatting is subtracted out
(610.3 µs of that 18,246.7 µs), is under 1%. Anyone using `explain()` timing
as an optimizer-cost proxy anywhere else should know this trap exists.

**Why this matters for the owner's framing:** rtlf's README is correct that
"every `collect()` re-runs the optimizer," but *eliminating just that* is
not where rtlf's big numbers come from. Its own published README shows the
same shape independently: `RealtimeLazyFrame` (optimizer removed, physical
plan still rebuilt every call) gets **1.00–4.12×** across its linear-chain
and decision-tree benchmarks; `CompiledRealtimeLazyFrame` (physical plan
*also* cached, values injected into pre-wired slots) gets **23–54×**. The
optimizer's slice and the physical-plan-build slice are not the same size,
and the second one is much bigger. That is consistent with what I measured
above from the other direction: optimizer share *shrinks* with plan size
(31%→0.7% here), meaning the thing that keeps growing super-linearly with
plan size (`collect_s` grows ~3.5× per depth-doubling — consistent with doc
01 §3's independently-measured `O(nodes^2.3)`) is **not** the optimizer.

**No supported public polars API can isolate or cache physical-plan
construction.** `collect(optimizations=QueryOptFlags.none())` still calls
`create_physical_plan` every time — that is unavoidable through any public
`collect()` call. Reaching the big number requires exactly what rtlf's
`CompiledRealtimeLazyFrame` does: hold the compiled executor tree across
calls and inject data into pre-wired slots, using internal, unstable APIs
(`create_physical_plan` with a custom `StreamingExecutorBuilder` hook,
walking the `IR` arena directly). **This is a real gap in polars' public
API, not a mistake in decider2's doc 01** — a supported reimplementation of
"cache the compiled plan" gets you the small 1.0–4×  optimizer win, not the
large 23–54× one, unless it also reaches into the same internals rtlf does
(or that capability gets upstreamed).

Also independently reproduces a doc 01 §3 finding on a current polars
version: batch=1 (single record) and batch=1000 give **near-identical**
`collect()` times at every depth (e.g. depth=400: 110.65 ms at batch=1000
vs. 111.4 ms at batch=1 — see `results.jsonl`, `label: "linear_b1"`) — cost
here is driven by plan size, not row count, exactly as doc 01 §3 found
("a 1-row frame costs the same as a 10k-row frame"). That finding still
holds five years and several polars major versions later on this box.

---

## `rtlf` build status

Attempted: `cd /tmp/claude-1000/rtlf && uv sync && uv run maturin develop --release`
in a detached `tmux` session (`rtlf-build`), polled rather than blocked on.
**Did not complete inside this experiment's time budget.** Ran for ~17
minutes (past the brief's ~10-minute guardrail, extended because it was
making steady — if very slow — progress, not stuck) and was still compiling
core polars crates one at a time (`polars_core` alone ran single-threaded
for over 5 minutes; `polars_ops`, `polars_lazy`, `polars_mem_engine`,
`polars_expr`, `pyo3_polars` and the final `lto="fat"` link of the whole
dependency graph were all still ahead of it) before this run stopped
waiting and the `tmux` session was killed to free the shared box. This is
**not** the version-mismatch risk stopping it — `cargo`/`rustup` had the
exact pinned nightly toolchain (`nightly-2026-01-09`) already installed,
network access to GitHub worked, and `uv sync` resolved cleanly; it is a
genuine, large, from-source compile of ~20 patched polars crates plus their
own transitive dependency tree (`reqwest`, `rustls`, `object_store`,
tokio, …, pulled in by Cargo features this extension does not even use,
e.g. cloud object-store support), under a release profile
(`codegen-units=1`, `lto="fat"`) that is itself deliberately trading build
time for runtime speed. A rebuild attempt elsewhere should budget 30–60+
minutes for a first cold build on comparable hardware, not the ~10 the
brief allowed.

**The version coupling is a first-class finding, independent of whether the
build finished, and the owner's own read of it is correct:**

- rtlf's `pyproject.toml` pins `polars==1.38.1` exactly.
- Its `Cargo.toml` patches ~20 `polars-*`/`pyo3-polars` crates to the git
  tag `py-1.38.1` — not a published crates.io version, a specific commit
  tied to one Python wheel release.
- decider2's own environment runs polars **1.41.2** — three minor versions
  ahead.
- Its README states plainly: "relies on internal Polars APIs that may
  change without notice" and ships a `DSL_SCHEMA_HASH` version guard
  specifically because the extension will silently produce a hash mismatch
  against any other polars build.
- decider2 is being open-sourced. Depending on rtlf as shipped would pin
  every downstream user of the realtime path to one specific, unofficial,
  internals-coupled polars build — a materially different commitment than
  depending on public polars APIs.

**The tree/scoring subset is genuinely the safe part, though**, per the
owner's read of rtlf's own README: unsafe operations are `concat`/`vstack`,
cross/asof joins, and `maintain_order=True` aggregations; safe operations
explicitly include `select`, `with_columns`, `filter`, and `when/then/otherwise`
chains, which is exactly what a decision tree compiles to (see
`tree_to_polars.py`). None of the trees measured here touch the unsafe list.
So *if* decider2 ever depended on this technique, scoping it to tree/table
scoring only — never a full pipeline with joins/aggregations — is the
correct boundary, not an accident of what this experiment happened to test.

No "rtlf, measured" section follows — the build did not finish in this
run, so every rtlf-specific number in this document is a citation of its
own published README, not a fresh measurement, clearly marked as such
throughout. The headline Q1 answer does not depend on it: it was obtained
entirely through polars' own supported public API and cross-checked two
independent ways.

---

## Full comparison: four engines, three decider2-shaped trees

Same three trees the sibling `tree-codegen-vs-interpreted` experiment used
(same builder calls, same seeds — `build_credit_tree(target_leaves=20,
seed=7)`, `build_full_binary(5, seed=1)`, `build_one_sided_chain(100,
seed=3)`), so every number below is on identical structure and — for the
numba engines — identical row data. Correctness checked and asserted
(`q2_bench.py`): codegen and composed-callables both match the sibling's
interpreted kernel bit-for-bit at every shape.

### ns/row at 100,000 rows (minimum of 7 batch calls)

| shape | codegen (today) | composed njit callables | interpreted kernel (cited, sibling) | polars `collect()` |
|---|---:|---:|---:|---:|
| credit_tree (20 leaves) | 25.4 | **1044.5** | 56.7–61.2 | 66.4 |
| full_binary_d5 (32 leaves) | 30.0 | **1120.9** | 64.8–68.7 | 44.7 |
| one_sided_chain_100 (101 leaves) | 32.0 | **891.2** | 38.2–38.7 | 177.2 |

### Single-record latency (µs; minimum of 200–2000 repeated 1-row calls)

| shape | codegen | composed njit callables | polars `collect()` | polars/codegen ratio |
|---|---:|---:|---:|---:|
| credit_tree | 0.50 | 4.21 | 418.8 | **844×** |
| full_binary_d5 | 0.49 | 4.48 | 542.5 | **1114×** |
| one_sided_chain_100 | 0.53 | 4.39 | 7796.4 | **14,790×** |

Two things worth reading off these tables directly, before Q2/Q3's own
sections unpack them:

1. **At single-record scale, plain polars is not a marginal loser, it is
   categorically disqualified for the realtime path.** 400 µs–7.8 ms *per
   call* against a 20–100 ms budget is 0.4–39% of the entire budget spent
   before a single other framework cost (marshalling, validation,
   readback — see N1 below) is paid, for one row. This is a different and
   much starker number than doc 01's original 43× (batch, not single-record)
   finding, and it is worth restating why: `create_physical_plan` and the
   Python↔Rust FFI/DataFrame-construction cost around it are paid in full
   whether the batch is 1 row or 100,000 — confirmed directly in this
   experiment's own depth-sweep (batch=1 vs. batch=1000 collect times are
   near-identical at every depth, echoing doc 01 §3 again).
2. **At *batch* scale, plain, uncached polars is not obviously bad** — for
   the two shallower trees it beats composed-callables outright (44.7–66.4
   ns/row vs. 1044.5–1120.9 ns/row) and is within 2–3× of codegen. It only
   degrades sharply on the deep one-sided chain (177.2 ns/row — the
   `pl.when` chain there is 101 nodes deep, and doc 01 §2's "polars is
   linear in depth, numba is sub-linear" finding is visible directly:
   one_sided_chain_100 is the one shape where polars' ns/row is *worse*
   relative to codegen than either other shape). This reopens, mildly and
   narrowly, part of decider2's founding premise for the *batch* entry
   point on shallow/moderate trees specifically — see Q3.

---

## Q2 — composed, cached njit callables

**Answer: the risk named in the brief is real and large, not marginal.**
An indirect call through a `numba.types.FunctionType` value, dispatched
from a 3-entry `numba.typed.List` keyed by node kind, costs **28–41× more
per row** than codegen's emitted straight-line source, and **14–20× more
per row than the sibling's plain array-interpreted kernel** — which uses
the *identical* struct-of-arrays node data and the *identical* per-row
work, differing only in dispatch mechanism (an in-loop `if kind==NUM: ...`
direct branch vs. `fn_table[kind[node]](...)` indirect call). That
isolates the mechanism cleanly: this is the cost of losing inlining and the
branch predictor, not the cost of the data layout.

Design (`composed_callables.py`): exactly three `@njit(cache=True)` kernels
— `leaf_eval`, `num_eval`, `str_eval` — compiled once, ever, reused
unchanged across every tree and every structural edit. Verified directly,
not assumed: `leaf_eval.signatures`/`num_eval.signatures`/`str_eval.signatures`
all stay at length 1 across all three shapes measured, run back-to-back in
one process. Node data (threshold, feature index, op code, pattern range,
left/right child, leaf value) is threaded as explicit arguments from the
same `FlatTree` struct-of-arrays the sibling interpreter uses — never
closed over — matching doc 05 §4.2's rule that a decision-relevant constant
is a value, not something that forces a recompile.

### Structural-edit cost: composed-callables' actual selling point

| approach | cost of a structural edit (new tree shape) |
|---|---:|
| codegen (cited, sibling, cold) — credit_tree | 388–423 ms |
| codegen (cited, sibling, cold) — full_binary_d5 | 375–435 ms |
| codegen (cited, sibling, cold) — one_sided_chain_100 | 1306–1367 ms |
| composed-callables — any of the three shapes | **56–189 µs, zero numba compiles** |

This is a ~2,000–24,000× reduction, and it is qualitative, not just
quantitative: codegen recompiles machine code on every structural edit;
composed-callables never does, by construction (the three kernels are
generic over node data, so "recompose the plan" is pure numpy array
assembly). This is the one place composed-callables genuinely wins
something codegen structurally cannot offer at any price.

**But it does not win anything the sibling's plain interpreter doesn't
already offer, for less.** The array-interpreted kernel has the *identical*
zero-recompile property (one kernel, ever, for the whole process — the
sibling's own headline finding) while running 14–20× faster per row than
composed-callables and only 1.35–2.8× slower than codegen (sibling's own
numbers). For the narrow question "walk one tree," composed-callables is
strictly dominated by the interpreter: same structural-edit cost, worse
execution speed, no offsetting advantage. Its only plausible niche is a
*different* problem decider2 has not obviously asked for: composing
genuinely *heterogeneous* interior kinds (tree + table + arbitrary step
DAG) into one polymorphic driver callable, the way polars' own executor
tree composes wildly different operator kinds through one `Executor`
trait — decider2's doc 08 §3.4 test ("can one compiled loop evaluate every
instance of this kind? If yes, generic kernel; if it needs a switch over
node *types*, codegen") already resolves tree-vs-table dispatch a different
way, per-interior-type, and nothing measured here shows that boundary is
costing decider2 anything today.

---

## Q3 — the honest recommendation, per entry point

| # | approach | `apply()` (batch, throughput-bound) | `score()` (realtime, single-record) |
|---|---|---|---|
| 1 | numba codegen (today) | **Best** at every shape tested (25.4–32.0 ns/row) | **Best** (0.49–0.53 µs/call) — but see N1 below |
| 2 | composed njit callables (Q2) | Worst of the numba options (891–1121 ns/row) — real but small in absolute terms below ~10k rows, real in absolute terms above (100k rows × 1000 ns/row ≈ 100 ms, a whole budget) | Negligible in absolute terms (4.2–4.5 µs, ≤0.02% of a 20 ms budget) but offers nothing the interpreter doesn't already offer for less |
| 3 | array-encoded interpreter (sibling experiment) | Close second to codegen (1.35–2.8× slower, sibling's numbers), zero per-shape compile | Same — noise against the budget, per sibling's own conclusion |
| 4 | polars + cached plan (rtlf) | **Competitive to good** on shallow/moderate trees even *without* caching (44.7–66.4 ns/row, cheaper than composed-callables); degrades on deep chains (177.2 ns/row) | **Disqualifying** even with rtlf's *uncached* mode: 400 µs–7.8 ms/call is 0.4–39% of the entire realtime budget for one row, before any framework cost. rtlf's compiled mode targets exactly this gap but cannot plausibly close a 1,000–15,000× ratio to numba's ~0.5 µs floor — FFI + `Arc<Mutex<...>>` slot injection + `DataFrame` construction have their own irreducible per-call floor, and this experiment could not get a real number on it (build did not finish; see below for what an owner attempt should measure directly if this matters enough to pursue). Also add the version-coupling risk: an experimental, unofficial, internals-coupled dependency for a component being open-sourced.

**One measurement that decides each row, stated plainly:**

- **Row 1 (codegen) wins `apply()`** because 25.4–32.0 ns/row beats
  everything else measured at every shape, and decider2's own §P found its
  cold-compile cost (0.67 s for a realistic 21-leaf/120-line tree) is a
  one-time, not per-call, cost.
- **Row 1 (codegen) wins `score()`** on raw numbers, but the number that
  actually decides `score()` is not in this experiment — it is N1's own
  finding (`experimentation/single-record-overhead/`): the compiled kernel
  is ~1.38 µs of a **971.4 µs** (as normally written) to **145.2 µs**
  (marshal/readback rewritten in bulk) total single-record cost. **Every
  numba engine's kernel-only number in this document (0.49–4.5 µs across
  codegen and composed-callables) is inside the noise floor of that
  framework cost.** The real `score()` optimization target, per N1, is the
  marshal/readback loop around whichever kernel runs — not which numba
  engine computes the tree. This is the honest reason row 2 (composed
  callables) "loses" 8–9× at single-record scale and it still does not
  matter: 4.5 µs vs 0.5 µs is invisible next to a 145–971 µs floor.
- **Row 3 (interpreter) is the right default for structural-edit latency**
  on both entry points — it has composed-callables' zero-recompile property
  without composed-callables' 14–20× per-row penalty, which is why the
  sibling experiment's "switch to the interpreted kernel by default"
  recommendation stands independently of anything measured here.
- **Row 4 (polars/rtlf) is disqualified for `score()` by one number**:
  7796.4 µs for one row of a 101-node tree, **even without any optimizer or
  physical-plan cost counted separately** — it is the *total*
  `collect()` time. No caching strategy applied only to optimizer-or-plan
  cost closes a gap that starts at 15,000×. It is worth a second look for
  `apply()` on shallow/moderate trees specifically **if** decider2 ever
  wants polars-expression authoring for that one entry point and can accept
  the version-coupling risk — and if it does, the fully-supported
  `collect(optimizations=QueryOptFlags.none())` result above (competitive
  with plain `collect()`, cheap to obtain) is a strictly safer starting
  point than depending on rtlf's internals, at the cost of forgoing the
  23–54× rtlf claims for very deep/wide plans that are unlikely to occur in
  decider2's realistic tree shapes anyway (this experiment's actual trees
  never got deep enough to need it — the 3-shape sweep in Part A shows the
  super-linear cost only becomes severe past ~100–200 nodes).

**Does this reopen decider2's founding premise (Q1's original ask)?**
Narrowly and partially, for `apply()` on shallow-to-moderate trees, where
plain uncached polars turned out cheaper than expected relative to codegen
(44.7–66.4 ns/row vs. 25.4–30.0 ns/row — a 1.5–2.7× gap, not doc 01's
original 43×). It does **not** reopen it for `score()` — single-record
polars latency (whether cached or not, since the fixed cost is dominated by
physical-plan construction, not the optimizer) is 3–4 orders of magnitude
above numba's floor, and no amount of "the optimizer wasn't the real cost"
changes that, because the optimizer was never more than ~15% of it.

---

## What surprised me

1. **The optimizer is not the story rtlf's own headline framing suggests —
   physical-plan construction is, and rtlf's own README numbers already
   said so once decomposed** (`RealtimeLazyFrame`'s 1.0–4.1× vs.
   `CompiledRealtimeLazyFrame`'s 23–54×), but the owner's original framing
   ("every `collect()` re-runs the optimizer") and my own first instinct
   both under-weighted this. It took a genuinely wrong first measurement
   (`explain()` timing alone) to find the formatting-cost trap and correct
   toward the real number.
2. **Plain, uncached polars beat composed njit callables outright** on two
   of three shapes at batch scale. I expected "a numba approach, however
   awkward, beats polars" as a safe prior from doc 01; it does not hold
   once the numba approach specifically loses inlining.
3. **The magnitude of the indirect-call penalty**: 28–41× per row is far
   past "a measurable but survivable tax" — it is closer in kind to doc 01
   §1's raw-Python-vs-numba gap (7.1 ms vs. 0.82 ms, ~8.7×) than to any
   "compiled but suboptimal" number, despite every operand in the node
   functions being a plain numba-native array/scalar.
4. **Single-record polars latency is worse, and flatter across row count,
   than doc 01 already established** — I expected some batch-count
   sensitivity even at N=1; there essentially is none (depth 400: 110.6 ms
   at 1000 rows vs. 111.4 ms at 1 row), which is doc 01 §3's finding holding
   almost exactly five polars major versions later.

## Files

- `polars_bench.py` — Q1 headline measurement (optimizer-fraction, two
  methods) + the depth sweep + the three realistic-shape polars numbers.
- `tree_to_polars.py` — the missing third converter (`Num`/`Str`/`Leaf` →
  `pl.when`/`then`/`otherwise`), reusing the sibling's canonical tree
  objects unmodified; correctness-checked against the sibling's interpreted
  kernel before any timing.
- `composed_callables.py` — Q2 prototype: three generic njit kernels
  dispatched through `numba.types.FunctionType` values held in a
  `numba.typed.List`, keyed by node kind.
- `q2_bench.py` — drives real `decider2.trees.codegen` (via the sibling's
  `codegen_bench.py`, unmodified) against `composed_callables.py`, on the
  same three shapes and row data as `polars_bench.py`'s Part B.
- `results.jsonl` — every measurement, one JSON object per line, written
  incrementally.
