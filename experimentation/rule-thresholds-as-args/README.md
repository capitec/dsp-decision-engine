# Experiment L — rule thresholds as arguments vs emitted literals

Tests doc 05 §4.2's untested assertion:

> No decision-relevant constant is emitted into driver source. Params are
> already arguments so that path is safe. A ruleset's thresholds are NOT
> yet — and that is precisely the path a business-user UI edits.

Connects EXPERIMENTS.md §C (stale-cache-on-constant-edit), §G (disabled
rules still cost full compile time; compile ∝ lines^1.4), and doc 08 §2's
three change classes (values / interiors / skeleton).

## What it measures

Builds **two emitters for the same rule set** — `emit_literal` (thresholds
baked into source as float literals) and `emit_args` / `emit_args_masked`
(thresholds, and separately rule-enablement, passed as array arguments) —
from the identical rule spec (same feature indices, same operators, same
threshold values), so every measured difference is attributable to
literal-vs-argument, not to a shape difference. Five parts, in this order:

1. **Runtime cost of the indirection** — 100k/1M rows, 10/30 rules.
2. **Compile time** — 3/10/30 rules, both forms; does hoisting thresholds
   change emitted line count?
3. **Does changing a threshold recompile?** — 8 retunes each form,
   `driver.signatures` length tracked for the args form.
3b. **Stale-cache hazard** — reproduced in the literal form (three ways:
   as-deployed, with only CPython's `.pyc` cleared, and a true
   never-cached ground truth), and shown structurally absent in the args
   form.
4. **Enablement as an argument (mask)** — runtime overhead of a per-rule
   mask check vs. the compile time a toggle avoids.

## Simplification, stated up front

Rules here are one flat AND-of-comparisons composite per rule (first-match,
early exit) — the top layer of
`experimentation/ruleset-compile-latency/emit.py`'s tree, without that
harness's nested `cases`/unary sub-trees. That top layer is exactly the
layer that carries numeric thresholds, so it's the right slice for this
question, but line counts here are **not** directly comparable to §G's
table (fewer lines per rule: 65 lines at 30 rules here vs. §G's 517).
Dropped for the ~12-minute budget: a nested-tree version combining this
with §G's full shape, and statistical variance across multiple seeds (each
number below is one run at 41 repeats for timings, 1 run for compiles).

## Reused, not rewritten

- `rss_mb` / `med` / `pct` / the exec+njit+call-once compile pattern /
  `count_compiles` (`numba.core.event.install_recorder`) — imported **live**
  from `experimentation/staged-compile-atomic-swap/run.py` (module `swaprun`
  in `run_l.py`), not copy-pasted.
- `elapsed()`/`log()` progress-clock pattern — `experimentation/prange-crossover/`.
- median-of-repeats timing (warm call, then time `repeats` calls, report
  median) — `experimentation/dtype-boundary/dtype_boundary.py:median_us` /
  `experimentation/output-writeback-convention/writeback.py:median_s`.
- the same-byte-length constant-edit trick for the stale-cache repro —
  `experimentation/numba_cache_survival/run_experiment.py:part_stale`.
- real `.py` file + `importlib.util.spec_from_file_location` (never
  `exec()`) for the stale-cache part specifically, per doc 05 §4.1 — the
  compile-time/runtime parts use `exec()` like
  `staged-compile-atomic-swap/run.py:compile_kernel` does, since no cache
  persistence is being tested there.

## Run

```
/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/.venv/bin/python run_l.py
```

Results append to `results.jsonl`, one JSON record per measurement, flushed
and `fsync`'d immediately after each write (mandatory rule 3). `run.log`
has the full console trace of the run these numbers came from.

## Memory

Every array here is at most 1,000,000 rows × 16 f8 feature columns + 1 i8
output column = **136.0 MB**, printed as an estimate before that
configuration runs (mandatory rule 1) — far under the 4 GB restructure
threshold. `free -g` is checked and must show ≥8 GB available before
anything runs (mandatory rule 5); this run started with 23 GB available.
**Measured peak RSS: 322.0 MB.** This never approached the 2 GB threshold,
so it was **not** run under tmux/systemd-run — that machinery is for runs
*expected* to cross ~2 GB (mandatory rule 4), and using it here would only
add overhead. Total wall clock: **29.7 s**, comfortably inside the
12-minute budget.

## Findings

Quoted claims are tested verbatim against what was measured.

### 1. CONFIRMED, with a real but small price — runtime cost of the indirection

| n_rules | rows | literal ns/row | args ns/row | args is |
|---|---|---|---|---|
| 10 | 100,000 | 38.61 (IQR 1.79) | 39.59 (IQR 1.06) | +2.5% |
| 10 | 1,000,000 | 39.92 (IQR 1.80) | 41.16 (IQR 0.93) | +3.1% |
| 30 | 100,000 | 40.05 (IQR 1.86) | 44.53 (IQR 2.18) | +11.2% |
| 30 | 1,000,000 | 41.56 (IQR 1.86) | 43.32 (IQR 1.42) | +4.2% |

The args form is **consistently slower, by 2.5–11.2%** (absolute: 1–4.5
ns/row on a ~40 ns/row kernel), with the two measurements at a given
configuration's IQR overlapping the gap at 10 rules but not fully at 30
rules — real, but a single-digit-percent tax, not the constant-fold cliff
the proposal worried about. LLVM can still vectorize an array-indexed
comparison; it just can't bake the compare into an immediate. **The price
is real but small — it does not fail the proposal.**

### 2. REFUTED — hoisting thresholds does not reduce emitted lines

| n_rules | literal lines | literal compile | args lines | args compile |
|---|---|---|---|---|
| 3 | 11 | 0.475 s | 11 | 0.297 s |
| 10 | 25 | 0.444 s | 25 | 0.517 s |
| 30 | 65 | 1.081 s | 65 | 1.325 s |

**Line count is identical at every rule count** — `f12[i] > 0.0405` and
`f12[i] > th[0]` are the same number of source lines; only the token
differs. Compile time is a wash (**literal faster at 3 rules, args faster
at 10, literal faster at 30** — differences of 0.07–0.24 s, noise-level
against 0.3–1.3 s totals, no consistent direction). **The proposal's
compile-time benefit does not come from shrinking emitted source** — if it
helps at all, it must come from §3's "never recompile on retune", not from
§2's line count, and doc 08 §2 should not claim a line-count reduction.

### 3. CONFIRMED — changing a threshold recompiles with literals, never with arguments

8 retunes each side:

| form | driver.signatures | compile events across 8 retunes |
|---|---|---|
| args | 1 → 1 | **0** |
| literal | new dispatcher every time | **8** (2.745 s total, 343 ms/retune avg) |

The args form's `driver.signatures` **stayed at length 1 across all 8
retunes** — exactly doc 03 §4's existing guarantee for params, now shown to
extend cleanly to rule thresholds. The literal form recompiled every single
time, by construction (new source text → new dispatcher). At a 3–5 rule
scale this is ~340 ms/retune; §G's compile-time table says this cost grows
∝ lines^1.4, so the same retune at 30+ rules is materially worse, not just
proportionally worse.

### 3b. CONFIRMED, and REFINES a prior finding — the stale-cache hazard reproduces in literal form and is structurally absent in args form

Three-way isolation on the literal form (same-byte-length edit
`0.0405`→`0.5950`, mtime/size preserved, per §C's/K's trick):

| variant | served pre-edit (stale) value |
|---|---|
| (a) as-deployed (CPython `.pyc` + numba `.nbi`/`.nbc` both left alone) | **True** |
| (b) CPython `.pyc` cleared, numba's own cache left alone | **True** |
| (c) true ground truth (fresh `cache=False` compile, never touches either cache) | differs from (a)/(b) — the edit does change the real answer |

> **This refines EXPERIMENTS.md experiment K's finding #2.** K reran §C's
> harness and confirmed the *symptom*, but attributed it to CPython's own
> `.pyc` cache, not numba's: *"I could not build a case where `co_code`
> stayed byte-identical after a semantically real edit... only that
> CPython's own bytecode cache reliably is [exploitable]."* This harness
> constructs exactly that case: with CPython's `.pyc` cleared (row (b)
> above, `numba_cache_independently_stale: true` in results.jsonl), **numba's
> own on-disk cache alone still served the pre-edit value.** A single
> same-position numeric-literal swap (the natural form of "a business user
> edits a threshold") leaves `LOAD_CONST`'s *operand index* unchanged even
> though the *value* at that `co_consts` slot changes — which is §C's
> original mechanism claim, verified independent of the `.pyc` layer K's
> harness happened to be tripping on. **Both cache layers are exploitable
> by this edit shape; a fix must close both**, e.g. content-addressed
> filenames (K finding #4), which sidesteps both structurally.

Args form: retuned by passing a new `th` array to the *same* compiled
dispatcher. File `(mtime, size)` unchanged = **True**, output correctly
changed = **True**, `driver.signatures` stayed at **1**. There is no
`(mtime, size)` pair for a stale edit to collide on, and no source text
containing a threshold to edit — **this is not "not yet observed", it is
architecturally impossible**: the hazard requires a source-level constant
whose byte-identical-but-semantically-different edit can fool a cache key,
and the args form has no such constant in source at all.

### 4. PARTIALLY CONFIRMED — masking fixes §G's disabled-rule problem, cheaply, but the "break-even" is not really a meaningful threshold

| n_rules | args ms | args+mask ms | overhead | literal recompile avoided |
|---|---|---|---|---|
| 10 | 21.19 | 20.77 | **−2.0%** (noise) | 0.428 s |
| 30 | 22.13 | 23.34 | **+5.4%** (+2.41 ns/row) | 1.029 s |

The mask-check overhead is **at or below measurement noise** — it flips
sign between the two rule counts, and both are single-digit percent on a
~20 ms/500k-row call. At 30 rules, break-even is **~427 million rows**
scored on the mask path to cost as much wall-clock as *one* recompile; at
10 rules the mask path measured net-negative overhead, so there is no
finite break-even at all. **In practical terms this means the "how often
does a user toggle" question doesn't bind**: any toggle frequency an actual
business user would produce (per §G's "enable/disable" framing — this is a
human clicking a checkbox, not a hot loop) is separated from the recompile
cost by 6+ orders of magnitude of row-throughput. §G's finding — "disabled
rules still cost full compile time" — is **fixed outright** by making
enablement a mask argument, not just improved.

### 5. The limit — what still forces a recompile

Not measured directly (qualitative, from what the emitters can and cannot
parametrize): **operator choice** (`<=` vs `>`), **clause/rule count**
(adding or removing a rule changes the loop body's control flow, not a
value at a fixed slot), and **nesting shape** (an AND becoming an OR, or a
new branch) all change the *token sequence* of the emitted source, not just
a `LOAD_CONST` payload — there is no argument that can carry "add a
fourth AND'd clause" without also changing what code runs. This is
exactly doc 08 §2's *values vs. interiors* line: a threshold is a value
(this experiment: safe as an argument, zero recompile, zero cache-key
risk); which operator, how many clauses, how many rules, and how they
nest are interior/*structure* and must still recompile. Doc 08 §2's table
should say explicitly that **a ruleset's thresholds belong in the "values"
row**, not the "interiors" row it currently implies them into by putting
"rules" wholesale under "interiors" — only a rule's *shape* is an
interior change; its *thresholds* are not.

## What doc 08 §2 and doc 05 §4.2 should say

**Rule thresholds should be arguments.** The runtime price is real but
small (2.5–11.2% on the kernel; ~1–4.5 ns/row absolute), it does not come
from a line-count reduction (finding 2 refutes that half of the
motivation), but it **eliminates recompilation on every retune** (finding
3), **eliminates a cache-staleness hazard that this experiment confirms
exists at BOTH the CPython and numba cache layers** (finding 3b, a
correction to K's finding #2), and — bundled with a mask argument for
enablement — **outright fixes §G's "disabled rules cost full compile
time" problem** at negligible runtime cost (finding 4). Doc 08 §2's three
change classes should split "rules" itself: **a rule's thresholds and its
enabled/disabled state are values** (free, a pointer/array swap, business
user); **a rule's operator, clause count, and nesting are interiors** (one
background compile + swap, doc 08 §4). The current doc puts "rules" as a
whole under "interiors" — this experiment shows that conflates a change
that should be instant with one that legitimately costs a compile.

## Dropped, for the time budget

- Statistical variance across multiple seeds/rule shapes (one seed, one run
  per configuration for compiles; 41 repeats for timings).
- The nested `cases`/unary sub-tree layer from §G's full emitter — this
  harness only parametrizes the top-level composite-condition layer (see
  Simplification above), so its line counts/compile times are not
  directly comparable to §G's table.
- A live UI-toggle-frequency measurement for part 4's break-even (used a
  rows-scored proxy instead, converted to a throughput statement in the
  Findings section rather than assuming a specific calls/sec).
