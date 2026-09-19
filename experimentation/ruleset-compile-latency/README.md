# Experiment G — rule-set compile latency (doc 06 **O16**)

## What this measures

How long numba takes to compile a **realistically-shaped flat rule set**, so that
doc 08 §4's "business user edits rules → background worker compiles → atomic
activate" can be given a number.

Claims under test:

> doc 01 §4b — "**≈ 15 ms per emitted source line** is the single best predictor."
> doc 01 §4b — "**Fan-out is the wall, not depth.** … the limit to enforce is
> **emitted lines (~500 ≈ 10 s), not nesting depth.**"
> doc 06 O16 — "Thirty rules with composite conditions might be two seconds or two minutes."

## The emitted shape

`emit.py` mirrors the vocabulary in `decider/modules/rules/`:

| real construct | source | emitted as |
|---|---|---|
| 13 unary operators | `common/nodes/operators.py` | `<= < == > >= != between isin string_match is_null is_not_null is_true is_false` |
| 3 `cases` variants | `common/nodes/cases.py` | `ranges` / `string_match` / `isin`, 3–5 branches + `otherwise`, as `if/elif/else` |
| `CompositeRule` | `common/nodes/composite.py` | AND/OR over 2–4 clauses, 15% NOT-wrapped |
| `LeafRule` | `flat_rules/nodes.py:116` | `res = <result_idx>`; `-1` is the no-match sentinel (`WithUnaryBranches._get_then_rule`) |
| `FlatRuleTree` | `flat_rules/nodes.py:552` | N independent rule trees over one row loop |
| `PrioritizationMode` | `flat_rules/module.py:32` | `first_match` → `if res == -1:` guard per rule, one output column; `all` → every rule body runs, `out[i, r]` per rule |

Each rule = composite condition → `then` subtree → `else` subtree, subtrees nested
to depth 2 (leaf / nested unary / cases). Data surface: 16 `float64[::1]`,
6 `int64[::1]` (category codes), 3 `boolean[::1]`, plus the output array.
~16 emitted lines per rule.

**Stated simplification:** string features are int32-style category codes, because
numba nopython has no array-of-strings dtype. `string_match` therefore emits the
same shape as `isin`. This harness does **not** measure numba unicode-op compile cost.

`p_nomatch` controls how often a leaf emits `-1`. It draws from a *separate* RNG
stream, so changing it leaves the emitted tree shape and line count **bit-identical** —
only which leaves decline to match changes. That keeps compile times comparable
across match rates.

## How to run

```bash
V=/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/.venv/bin/python
D=experimentation/ruleset-compile-latency

# the exact sequence that produced results.json (~4 min wall)
rm -f $D/results.json
$V $D/run.py --counts 3,10,30    --modes first_match,all --repeats 3 --exec-repeats 7
$V $D/run.py --counts 40,50,60   --modes first_match,all --repeats 1 --exec-repeats 5
$V $D/run.py --counts 35         --modes first_match,all --repeats 1 --exec-repeats 5
$V $D/run.py --counts 100        --modes first_match,all --repeats 2 --exec-repeats 5
$V $D/run.py --counts 3,10,30,60 --modes first_match,all --repeats 1 --exec-repeats 7 --p-nomatch 0.4
$V $D/run.py --counts 3,10,30,60 --modes first_match     --repeats 1 --exec-repeats 7 --p-nomatch 1.0

$V $D/analyze.py                       # tables + scaling fits
$V $D/run.py --dump-source 2:first_match   # see what is emitted
```

Every repeat regenerates source from a fresh seed, writes a fresh `.py` file, imports
a fresh module and compiles eagerly with an explicit signature and `cache=False`.
Nothing can be reused between repeats. A trivial warm-up compile runs first so the
first real measurement is not the LLVM-startup outlier.

Machine: Intel i7-14700HX, 28 threads, Fedora. Python 3.14.5, numba 0.67.0,
llvmlite 0.49.0, numpy 2.4.6. Single-threaded `njit` (no `prange`).

## Results

Compile = median over N repeats. Exec = median of 5–7 runs over 100k rows.

### first_match (early exit, one output column)

| rules | emitted lines | reps | compile (s) | ms/line | exec @100k (ms) |
|---:|---:|---:|---:|---:|---:|
| 3 | 65 | 3 | 0.630 | 9.70 | 1.76 |
| 10 | 168 | 3 | 1.371 | 8.16 | 1.79 |
| 30 | 517 | 3 | **5.957** | 11.52 | 1.78 |
| 35 | 563 | 1 | 8.893 | 15.80 | 1.82 |
| 40 | 642 | 1 | **10.864** | 16.92 | 1.76 |
| 50 | 788 | 1 | 15.130 | 19.20 | 1.86 |
| 60 | 967 | 1 | 20.670 | 21.37 | 1.78 |
| 100 | 1617 | 2 | **56.564** | 34.98 | 1.76 |

### all (evaluate everything, one output column per rule)

| rules | emitted lines | reps | compile (s) | ms/line | exec @100k (ms) |
|---:|---:|---:|---:|---:|---:|
| 3 | 66 | 3 | 0.640 | 9.70 | 4.00 |
| 10 | 176 | 3 | 1.425 | 8.10 | 10.08 |
| 30 | 545 | 3 | **5.513** | 10.12 | 38.35 |
| 35 | 596 | 1 | 7.909 | 13.27 | 45.26 |
| 40 | 680 | 1 | **9.441** | 13.88 | 52.54 |
| 50 | 836 | 1 | 12.840 | 15.36 | 62.25 |
| 60 | 1025 | 1 | 17.824 | 17.39 | 85.34 |
| 100 | 1715 | 2 | **48.080** | 28.03 | 126.69 |

## Findings

1. **15 ms/line is not a constant — it is a curve.** Measured 8.1–11.5 ms/line
   below 30 rules, 35.0 ms/line at 100 rules. The doc figure over-states small
   rule sets by ~1.8× and under-states 100 rules by 2.3×.
   Compile ∝ `lines^1.40` globally (R²=0.975, first_match) and `lines^1.31`
   (R²=0.973, all); the **local** exponent rises from 0.8 (3→10 rules) to
   **1.96** (60→100 rules). Marginal cost per added rule: 106 ms at 10 rules,
   **897 ms** at 100.

2. **10 s is crossed at ~37 rules (first_match) / ~42 rules (all)** — i.e. around
   **600–700 emitted lines**, not 500. Doc 01's "~500 lines ≈ 10 s" is the right
   order and slightly conservative *for this shape*.

3. **`all` compiles faster than `first_match` while emitting more code.** At 100
   rules: 1715 lines / 48.1 s vs 1617 lines / 56.6 s — 6% more source, 15% less
   compile time. The `if res == -1:` guard chain is a serial dependence on one
   variable through N control-flow merges and costs LLVM more than N independent
   bodies. Short-circuiting is a **compile-time tax and a runtime win**, not free
   on both sides.

4. **Runtime: `first_match` is flat, `all` is linear.** 1.76–1.86 ms at every rule
   count from 3 to 100, versus 4.0 → 126.7 ms for `all` (~1.25 ms per rule per
   100k rows). Caveat: at `p_nomatch=0` and `0.4` the match rate is 1.000 for
   ≥10 rules, so rows exit within the first rules — this is the *favourable* case
   for `first_match`, and this harness did not construct a workload where most
   rows fall through the whole chain.

5. **Compile cost is paid on emitted source, not on surviving code.** At
   `p_nomatch=1.0` every leaf is `-1`, LLVM proves the loop body dead and exec
   drops to 0.031 ms — but compile is unchanged (60 rules: 21.12 s dead vs
   21.28 s live). Dead rules still cost their full compile time.

6. **Exec time is independent of rule count for `first_match` but compile is not**,
   so the activation-latency budget, not the throughput budget, is what bounds
   rule-set size.

## Answer to the deliverable

A config UI can promise **"live in ≤ 10 s" up to about 35 rules** in one
compilation unit, on this hardware, cold (`cache=False`), single kernel.

- 3 rules → 0.63 s
- 10 rules → 1.4 s
- 30 rules → 5.5–6.0 s
- 40 rules → 9.4–10.9 s ← promise breaks here
- 100 rules → 48–57 s

Beyond ~60 rules the local exponent is ~1.95, so the promise degrades
quadratically. **Extrapolated, not measured:** at that local exponent 150 rules
(~2430 lines) is ≈ 126 s.

The lever is **splitting the rule set across compilation units**: compile is
super-linear in one unit, so four units of 25 rules should cost roughly
4 × 4.5 s ≈ 18 s serial (or ~4.5 s on four workers) against the **measured**
48–57 s for one unit of 100 rules. The 4.5 s figure is interpolated from the
measured 30-rule points (5.5–6.0 s), **not measured at 25 rules**; splitting
was not itself measured in this run. Doc 08 §3.4's generic-kernel
boundary does not have to move at 30 rules; it has to move somewhere between
40 and 100.
