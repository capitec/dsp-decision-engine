# Experiment B — can per-node fallback exist inside a fused njit driver?

## The claim under test

decider2 doc 05 §6 (and doc 02 §3.2) promise two independent degradation layers:

> 1. **Per node.** Try the jitted version; the first time numba cannot compile it for the given
>    argument types, permanently fall back to plain Python for that node only. Every other node
>    is unaffected.
> 2. **Per pipeline.** Try to compile the whole fused driver. If any node is not
>    nopython-compatible the fusion attempt fails as a whole...

and doc 05 §6 adds:

> **Catch `numba.core.errors.NumbaError` only — never bare `Exception`.** A real runtime bug
> (`ZeroDivisionError`) must propagate identically in both compiled and fallback paths.

README.md summarises this as "anything numba can't compile degrades to plain Python per-node
rather than failing". REVIEW.md §7 argues layer 1 cannot exist inside a fused kernel.

## What this harness measures

A toy record pipeline: a row loop over four small step functions
(`step_scale`, `step_clip`, `step_logistic`, `step_bucket`), applied to a float64 column.

1. **Part 1** — the all-njit fused driver compiles in nopython mode and matches the Python
   driver bit-for-bit.
2. **Part 2** — break exactly one node, six different ways, and print the exact exception class,
   its MRO, whether `isinstance(exc, NumbaError)`, and the head of the message.
3. **Parts 3+4** — the blast radius, timed. Five driver variants over the same rows:
   - `(a)` all four nodes njit, serial; `(a2)` the same with `prange`
   - `(b)` njit driver, node 1 escapes to Python **per row** via `objmode`
   - `(b3)` the same, inside `prange`
   - `(c)` the whole driver in plain Python
   - `(e)` njit driver, the bad node **hoisted** to a single `objmode` call for the whole batch
   - `(d)` Python pre-pass building the bad node's column, then a pure njit driver
   Node 1's `objmode` replacement computes *exactly the same arithmetic* as the njit version,
   so `(b) - (a)` is the `objmode` transition cost and nothing else. A second set of variants
   uses a realistic non-compilable node (`re.match` over a formatted string).
   All variants are checked for bit-identical output.
4. **Parts 5+6** — whether `except NumbaError` catches compile failures cleanly, and whether a
   genuine `ZeroDivisionError` propagates identically across plain Python, njit serial and
   njit `prange`, for both Python scalars and numpy scalars.

Every timing warms up first (the first call includes compilation), then reports the **median**
of `--repeats` runs; the first-call time and the implied compile time are printed alongside.

## How to run

```sh
.venv/bin/python experimentation/objmode-fallback-blast-radius/run.py
.venv/bin/python experimentation/objmode-fallback-blast-radius/run.py --rows 500000 --repeats 7
```

Defaults: `--rows 200000 --repeats 5`. Whole run takes ~50 s wall clock on 28 threads
(python 3.14.5, numba 0.67.0, numpy 2.4.6). No arguments needed; nothing is installed or
written. Change `--rows` to re-test on a different workload shape.

## Result summary (200k rows, medians; re-run to regenerate)

| variant                                             | median ms | vs (a) | ns/row |
|-----------------------------------------------------|----------:|-------:|-------:|
| (a) all-njit fused driver                            |      7.13 |   1.0x |     36 |
| (a2) all-njit, prange                                |      1.01 |   0.1x |      5 |
| (b) njit driver, 1 node via objmode **per row**      |    548.43 |  77.0x |   2742 |
| (c) whole driver in plain Python                     |    167.26 |  23.5x |    836 |
| (b3) objmode per row inside prange                   |   4380.61 | 614.7x |  21903 |
| (b2) regex node via objmode per row                  |    720.60 | 101.1x |   3603 |
| (e) regex node hoisted to 1 objmode call             |    211.81 |  29.7x |   1059 |
| (d) Python pre-pass column + pure njit driver        |    222.86 |  31.3x |   1114 |
| (c2) whole driver in plain Python (regex node)       |    394.17 |  55.3x |   1971 |

- `objmode` transition cost: **~2.7 µs per row**.
- Per-row `objmode` is **3.3x slower than giving up on the kernel entirely** (b vs c).
- Under `prange`, per-row `objmode` compiles and is correct but runs **8x slower than serial**
  (GIL), versus a 7.0x speedup for the all-njit driver.
- Every compile failure tested raises `numba.core.errors.TypingError`, which **is** a
  `NumbaError` — the whole driver fails, never one node.
- A row loop over numpy columns dividing by zero returns `inf` in plain Python, raises
  `ZeroDivisionError` under njit serial, and returns `inf` again under njit `prange`.
