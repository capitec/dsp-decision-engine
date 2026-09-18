# 05 — Boundary and compilation

Implementation spec for `compile/`. This is the first thing to build, and most of
it is now determined by measurement rather than choice — every number cited is
from doc 01 §4, §4b–4d.

Scope: getting data out of polars, into a compiled kernel, and back.

---

## 1. Extraction

### 1.1 Rechunk once, at frame entry

```python
frame = frame.rechunk()     # 0.33 µs if already single-chunk
```

Never per column: a genuine two-chunk rechunk costs **585 µs/1 M rows**, and
doing it per column multiplies that by the column count.

### 1.2 Per-column, via arrow buffers

```python
a = series.to_arrow()
validity_buf, values_buf = a.buffers()          # [validity, values] — this order
values = np.frombuffer(values_buf, dtype)[a.offset : a.offset + len(a)]
```

Three traps, each of which produced a wrong answer or a silent cost in testing:

1. **`rechunk()` does not reset `a.offset`.** A `slice(9, 12)` still reports
   `offset=9`. Ignoring it reads the wrong window — verified.
2. **Validity must be unpacked *before* slicing**, because bit *i* maps to
   element `offset + i`, not to element *i*.
3. **Never `frame.to_numpy()`.** 2D extraction of 8 columns × 1 M rows costs
   6109 µs against 4.7 µs per-column — **1300× worse**.

Cost when clean: **0.57 µs/col, flat from 100 to 10 M rows.**

### 1.3 Fast path for clean columns

**A column with no nulls has `validity_buf is None`** — no bitmap is allocated at
all. That is the cheapest possible gate; total gate-plus-extract measured
**0.375 µs**, fully zero-copy, with `to_numpy(allow_copy=False)` sharing the
arrow address. (`null_count()` at 0.08 µs is an alternative.)

### 1.4 Two dtype caveats

- **Bool columns are not zero-copy.** Arrow bitpacks them; `allow_copy=False`
  raises. Budget ~48 µs/col at 100k rows.
- **Extracted arrays are `writeable=False`**, so numba types them `readonly` —
  **a distinct signature from a writeable array.** Mixing the two silently
  produces two compiled specialisations of the same driver. Pick one convention
  and hold it; `readonly` is the natural choice since extraction produces it.

Also never use `.to_numpy()` on a nullable column: it copies *and* silently
changes dtype — `Int64→float64` (lossy above 2⁵³), `UInt8→float32`,
`Boolean→object`.

---

## 2. Nulls

**The slot under a null contains leftover garbage, not zero.** A measured left
join produced `9.0` at a position that was genuinely null. Every design decision
below follows from that: a step must never be able to read an unchecked value.

Four tiers, declared in the step signature (doc 03 §1):

| declaration | mechanism | cost /1 M | notes |
|---|---|---|---|
| `x: float` | validated at boundary, rejected | 256 µs (floor) | default |
| `x: float = missing_as(0.0)` | substituted at extraction | 256 µs | step sees a plain float |
| `x: float \| None` | numba `Optional` | 1907 µs | `if x is None` — the default for steps that must distinguish |
| `x: Maybe[float]` | `.value`/`.valid` struct | 413 µs | opt-in, for a hot path |

`Optional` is the recommended explicit form despite being slower: it introduces
no new concept, and it makes the garbage slot **unreachable** rather than relying
on the author remembering a check.

### Required-input validation

Gate on `validity_buf is None` (or `null_count()`), and on failure raise naming
the column, the count, and example rows:

```
step argument 'instalment' is declared required (`Float64`, no `| None`) but
column 'instalment' has 3 null(s) in 1000 rows (first at row 7, e.g. rows
[7, 400, 913]). Either fix the input or declare `instalment: Float64 | None`.
```

---

## 3. Calling convention

**A 1D structured (record) array per side**, above the threshold below.

Positional arguments are *not* broken — they work to at least 1600, and the
failure mode is compile time, not an error. Records are chosen because at
200-in/200-out they are **2× better end-to-end** (101 ms vs 202 ms at 100k rows),
**37× better per call** (0.54 µs vs 20.8 µs), 4.7× faster to compile, and are the
**only** convention that carries a float/int/bool mix with O(1) dispatch.

**Hybrid threshold: ~64 columns per side.** Below it, positional dispatch is
<4 µs and the record assembly copy (~0.93 ns/element) doesn't pay for itself.
The figure is interpolated between measurements at 50 and 200 — **not pinned**,
so make it a constant with a comment, not a magic number. If a single convention
is preferred, records are never much worse.

### Known bottleneck

**Record write-back dominates at high output counts** — gathering strided fields
back out costs 35–158 ms, which is **54.7% of total time at 633 outputs**. It is
still the best configuration measured, but optimisation effort belongs here, not
in the kernel.

Batch write-back via `hstack(pl.DataFrame(dict))` rather than chained
`with_columns` (~2× cheaper; `with_columns` is ~24 µs fixed + ~10 µs/col,
independent of row count).

---

## 4. Codegen

### 4.1 Real files, never `exec`

**The single highest-leverage decision in this layer.** Numba cannot cache code
without a source file (`RuntimeError: no locator available for file '<string>'`).
Writing drivers to a real `.py` took a 1200-argument driver from **29.5 s cold to
0.80 s warm** — which is what makes compile cost a build-time concern rather than
a startup one.

### 4.2 Determinism is a hard requirement

Numba's cache is keyed on source. Generated code must be **byte-identical**
between build and runtime or every entry misses and startup silently recompiles.

- deterministic ordering everywhere (no set/dict iteration order dependence)
- no addresses, timestamps, `id()` values or PIDs in generated names
- stable naming derived from module ids and value versions

### 4.3 Driver shape

Steps are njit'd individually; only the glue is generated. Nested `Branch`/`Loop`
emit recursively (proven in `experimentation/steptree_poc/jit_codegen.py`).

```python
def _driver(inp, params_m1, params_m2, shared, out):
    for i in range(inp.shape[0]):
        v_disposable = _disposable_income(inp.net_income[i], inp.expenses[i])
        v_ratio      = _ratio(v_disposable, inp.instalment[i])
        if _is_private_sector(inp.sector[i]):           # Branch: real branch
            v_term_cap = _cap_private(v_term_cap, params_m1)
        else:
            v_term_cap = _cap_public(v_term_cap, params_m1)
        v_score = _final_score(v_ratio, v_term_cap, params_m2, shared)
        out.score[i] = v_score
        out.term_cap[i] = v_term_cap                    # a tap
```

Note what this gets for free: short-circuiting (only the taken arm runs), value
overwrite as plain local reassignment, and taps as one extra store.

---

## 5. Compiled variants

All generated from one graph, compiled at warmup, dispatched per invocation.

| variant | steps | driver | purpose |
|---|---|---|---|
| `fused` | njit, inlined | njit | production |
| `fused` + taps | njit, inlined | njit | production diagnostics |
| `stepped` | njit | Python, one step at a time | driver-level debugging; **also the fallback path** |
| `interpreted` | Python | Python | reference semantics; full internals |

`stepped` is not a separate build to maintain — it is structurally identical to
the fallback path (§6), so one mechanism serves both.

### 5.1 serial vs `prange`

**There is no fixed row threshold.** Crossover is set by per-row work: a light
body never wins (0.99× even at 1 M rows); a body with a 64-iteration inner loop
crosses at ~5k and reaches 7.8× at 5 M.

So compile both and **measure per kernel at warmup** — affordable, since
processes are long-lived and both variants are being built anyway. Do not
hardcode a constant.

`prange` composes correctly with bounded inner loops, including `break`
(bitwise-identical output verified).

### 5.2 Flags

- **`fastmath`: off by default, enabled selectively.** It is noise (±5%) on
  branch-dominated logic but worth **2–2.5×** on an arithmetic-heavy fused
  driver. Decide per kernel, not globally.
- **`parallel=True` costs 1.2–2.6× compile time**, shape-dependent.
- **`inline='always'` is a no-op** — numba already inlines these (hand-inlining
  matches codegen within 3%).

---

## 6. Fallback

Two independent layers, so nothing forces an author to write numba-friendly code.

1. **Per node.** Try the jitted version; on the first compile failure for the
   given argument types, permanently use plain Python *for that node only*.
2. **Per pipeline.** Try the fused driver; if any node can't compile, fall back
   to a Python driver calling each node through its own best-available form.

**Catch `numba.core.errors.NumbaError` only — never bare `Exception`.** A real
runtime bug (`ZeroDivisionError`) must propagate identically in both compiled and
fallback paths, or it gets silently misreported as "this needed a fallback".

Cache the decision per node so a doomed compile isn't retried every call.

---

## 7. Fusion grouping

Fusion is **non-monotone** (doc 01 §4b–4c): profitable at 1–2 modules
(1.33–1.61×), harmful past ~5 (0.51× at 10 modules, 0.33× at 20) because register
pressure stops the growing body vectorising. The sign flips back when arms are
expensive, since short-circuiting then dominates.

Interim policy, good enough to build against:

- cap a fused group at **~6–9 steps**
- cap **emitted lines at ~500** (compile runs ≈15 ms/line, and a fully-branching
  depth-10 nest costs 92 s)
- split between groups; boundary stores are near-free

Must be **inspectable and overridable**, or "why is this slow" is unanswerable:

```python
pipeline.explain_kernels()                    # groups, and why
pipeline.pin_kernel_boundary(after="sector_cap")    # override
```

The real cost model is O12, unresolved — a crude cap is fine to start.

---

## 8. The build step

```
uv run decider build <pipeline>     # in the Dockerfile
decider build --verify              # asserts a runtime load triggers ZERO compiles
```

Generates driver sources, compiles all variants, and leaves a warm numba cache in
the image. `--verify` in CI is what stops the cache silently ceasing to work and
being discovered later as slow startups.

> **Choose the CPU target deliberately.** Numba's cache keys include CPU
> features, so a cache built on a CI runner with AVX-512 misses on a smaller
> deployment instance. `NUMBA_CPU_NAME=generic` at build *and* run makes entries
> portable but forfeits the vectorisation that §7 shows is what makes small
> kernels fast. Either build generic and accept slower kernels, or build in the
> CPU family you deploy on.

---

## 9. Acceptance criteria

This layer is done when:

1. A kernel runs end-to-end over a polars frame: extract → record assembly →
   driver → write-back, matching a numpy reference exactly.
2. The **same kernel** answers a single record with no polars involvement.
3. All four variants agree — `interpreted ≡ stepped ≡ fused` — over a test
   corpus, and deliberately injected drift at each layer is localised to the
   right rung (E3).
4. A `required` null violation fails fast naming the column, count and example
   rows; a nullable column round-trips correctly for `Float64`, `Int64` **and**
   `Boolean`.
5. Retuning any params bundle leaves `driver.signatures` at length 1; changing a
   field's *type* adds one (negative control).
6. A node containing something numba cannot compile (e.g. `re.match`) falls back
   to Python **without** disabling compilation for any sibling node.
7. `decider build --verify` reports zero runtime compilations.
8. A regression test asserts per-call dispatch stays near 1 µs, guarding the
   same-name-NamedTuple trap (doc 01 §4c) that silently costs 15–24 µs forever.
9. A sliced and a multi-chunk frame both produce correct results — the
   `a.offset` trap.
