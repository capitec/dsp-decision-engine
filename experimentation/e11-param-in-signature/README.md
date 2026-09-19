# E11 — does `param()` in a step signature survive the compiler?

Tests doc 03 §4.4 ("Declaring a param in the signature") against the five items
listed in doc 06 E11. Every number in the summary below came out of `e11.py` on
this machine; nothing here is an estimate.

## Run it

```bash
.venv/bin/python experimentation/e11-param-in-signature/e11.py            # all parts, ~51 s
.venv/bin/python experimentation/e11-param-in-signature/e11.py --part 5   # the dispatch risk only, ~49 s
.venv/bin/python experimentation/e11-param-in-signature/e11.py --child unique_class_name
```

`results.json` in this directory is the full output of one complete run.

Environment measured: Python 3.14.5, numba 0.67.0, llvmlite 0.49.0,
numpy 2.4.6, pydantic 2.13.4, Linux 7.0.12 x86_64. Nothing was installed.

## What it measures

**Part 1 — does numba tolerate a sentinel default it never uses?**
Compiles `cap_by_income_band` four ways: `njit` applied directly to the function
carrying a `param()` sentinel default; the same step re-emitted from a real
generated source file with the default kept; re-emitted with the default
stripped; and called from inside an `njit` driver that passes the param
explicitly. Also calls the sentinel-defaulted dispatcher *without* the param, to
see what numba says when the sentinel does reach it.

**Part 2 — generated model vs hand-written model.**
Harvests the signature with `inspect.signature` into a pydantic model via
`create_model`, and builds the equivalent model by hand. Compares class name,
field list, JSON schema, and accept/reject behaviour on six validation probes
(in range, at bound, below bound, above bound, extra field, default only). Then
converts both to NamedTuple bundles and compares the *numba* types.

**Part 3 — retune vs retype.**
One `njit` driver taking a params bundle. 15 retunes across the declared range,
checking `driver.signatures` stays at 1. Negative control: redeclare the param
as `int` and check a second signature appears.

**Part 4 — direct Python call.**
A `@step` wrapper that substitutes declared defaults so a plain call needs no
bundle, accepts `params=` as a bundle or a dict, and validates dict overrides
through the harvested model. Also calls the *unwrapped* function plainly, to
show the failure `@step` exists to prevent.

**Part 5 — the risk: one params model per function.**
20 steps, each with a param called `cap`, each with its own generated bundle.
Times per-call dispatch of the driver on a 1-row array (so argument typing
dominates), median of 2000 repeats after warmup. Variants:

| variant | what it isolates |
|---|---|
| `naive_same_name` | 20 distinct classes, all `__name__ == "Params"`, all field `cap` — the doc 01 §4c collision |
| `unique_class_name` | class name derived from the step id, field still `cap` — the proposed rule |
| `unique_field_name` | shared class name, distinct field names |
| `unique_both` | both distinct |
| `one_bundle_baseline` | 1 step, 1 bundle |
| `scalar_args_control` | 20 plain `float` args, no bundles — separates "cost of 20 arguments" from "cost of 20 NamedTuples" |
| `contamination` | a clean single-bundle driver timed before *and after* a colliding driver is hammered, testing doc 01 §4c's "permanently, for every call involving that name" |
| `regen_memoised` / `regen_not_memoised` | the same model's bundle regenerated into a second identically-named class, alternated across calls — tests the `lru_cache` mandate |
| `sweep:<variant>:<n>` | n = 2, 5, 10, 40 |
| `bundling:flat` / `bundling:nested_*` | one merged params argument instead of n — flat fields, or nested per-step bundles |

**Each part-5 variant runs in a fresh subprocess.** The failure under test is a
contamination of numba's per-process argument-type handling, so running two
variants in one interpreter would let the first poison the second.

## Headline results

Per-call dispatch, 1-row array, median of 2000 after warmup:

| variant | n | median µs | µs per param |
|---|---:|---:|---:|
| `naive_same_name` | 20 | **281.8** | 14.09 |
| `unique_class_name` | 20 | 18.93 | 0.95 |
| `unique_field_name` | 20 | 19.61 | 0.98 |
| `bundled_flat` | 20 | **3.50** | 0.18 |
| `bundled_nested_unique` | 20 | 18.84 | 0.94 |
| `scalar_args_control` | 20 | 1.64 | 0.08 |
| `one_bundle_baseline` | 1 | 1.57 | 1.57 |

1. The collision is real and **much worse than doc 01 §4c's 15–24 µs** at this
   width: 87 µs at n=2, 117 at n=5, 178 at n=10, 282 at n=20, 492 at n=40.
2. Naming the bundle class after the step id fixes it — 282 → 18.9 µs at n=20,
   a **15× recovery**. Distinct field names fix it equally well. Either alone
   suffices.
3. But the fix does **not** reach 1 µs. A NamedTuple argument costs ~0.95 µs to
   type per call regardless of naming, against 0.08 µs for a scalar. 20 rules
   with their own bundle is ~19 µs of dispatch per invocation.
4. **One flat params argument is 5.4× cheaper than 20**: 3.50 µs vs 18.93, and
   it barely scales (2.00 µs at n=5, 3.50 at 20, 5.74 at 40). Nesting the
   per-step bundles inside one outer tuple does *not* help (18.84 µs) — the cost
   is per NamedTuple object typed, not per driver argument.
5. Two doc 01 §4c claims did not reproduce: the penalty is **not** permanent
   (a clean driver measured 1.403 µs before and 1.428 µs after, ratio 1.02) and
   regenerating a bundle class without memoising cost nothing (2.18 vs 2.48 µs).
   The penalty needs **two or more distinct same-named classes in one call's
   argument list**; nesting them does not trigger it either (19.74 µs).

See the parent report for the verdict and the naming rule.
