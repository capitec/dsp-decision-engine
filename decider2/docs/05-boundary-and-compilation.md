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
bufs = series._get_buffers()          # {"values", "validity", "offsets"} as Series
values = bufs["values"].to_numpy(allow_copy=False)
valid  = bufs["validity"]             # None when the column is clean
```

> **⚠ Replaced, from measurement — [EXPERIMENTS.md](EXPERIMENTS.md) §A.** The original recipe was
> `series.to_arrow()` then a two-tuple `a.buffers()` unpack. It **cannot execute**:
> `to_arrow()` requires pyarrow, which is not a project dependency, and it raised
> `ModuleNotFoundError` for all 26 dtype/nullability combinations tested. The
> two-tuple unpack is also only valid for fixed-width primitives — Utf8 has three
> buffers.
>
> The polars-native path above costs the same (~1.9 µs/col at 100k rows), needs no
> new dependency, and returns validity pre-sliced and offsets pre-split.

Three traps, each of which produced a wrong answer or a silent cost in testing:

1. **`rechunk()` does not reset `a.offset`.** A `slice(9, 12)` still reports
   `offset=9`. Ignoring it reads the wrong window — verified.
2. **Validity must be unpacked *before* slicing**, because bit *i* maps to
   element `offset + i`, not to element *i*.
3. **Never `frame.to_numpy()`.** 2D extraction of 8 columns × 1 M rows costs
   6109 µs against 4.7 µs per-column — **1300× worse**.

Cost when clean: **0.8–1.2 µs/col** via `to_numpy(allow_copy=False)`,
**1.87–2.03 µs/col** via `_get_buffers()`, at 100k rows.

**Only 6 of 26 dtype/nullability combinations are zero-copy** — clean Float64,
Int64, Int32, UInt8, Datetime and Duration. **Every null-bearing column copies**,
at ~550–630 µs per column at 100k rows versus ~1 µs clean. That is a 500× cliff
and it is the single most important number for sizing a boundary.

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

### 1.5 The admissible-dtype contract

Measured, [EXPERIMENTS.md](EXPERIMENTS.md) §A. A dtype is admissible only via the column marked ✅.

| dtype | `.to_numpy()` → njit | values buffer → njit | verdict |
|---|---|---|---|
| Float64, Int64, Int32, UInt8 | ✅ | ✅ | admissible, zero-copy when clean |
| Boolean | ✅ clean / ✗ `object` when null | ✅ | admissible **via buffer only** |
| Date, Datetime | ✅ compare only | ✅ as int | admissible **as integer**; `add(datetime64, datetime64)` does not compile |
| Duration | ✅ | ✅ | admissible |
| Utf8 | ✗ `object` | ✅ as uint8 bytes + offsets | **not admissible as a value**; dictionary-encode first |
| Categorical, Enum | ✗ `object` | ✅ as codes | admissible **as codes**; code stability across frames must be declared |
| Decimal | ✗ | **Rust panic** | **inadmissible** — use scaled int64 |
| List | ✗ | `_get_buffers` raises | **inadmissible** — O5 |

Two rules follow, both non-optional:

- **Gate on dtype before extraction, and gate with `except BaseException`.**
  `Decimal` raises `pyo3_runtime.PanicException`, which inherits `BaseException`,
  not `Exception` — so the obvious `except Exception` does not catch it and the
  process prints an unsuppressable Rust panic to stderr. Money is the dtype most
  likely to hit this.
- **A string never enters a kernel as a string.** It enters as a categorical code
  or not at all. That makes code stability a declared property of the input
  schema, not an accident of the data.

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
| `x: float \| None` | numba `Optional` | 1907 µs | `if x is None` — the form for steps that must distinguish |

**Three tiers, not four.** `Optional` is the explicit form despite being slower:
it introduces no new concept, and it makes the garbage slot **unreachable**
rather than relying on the author remembering a check.

A `.value`/`.valid` struct measures faster (doc 01 §4) and is **rejected as a
public authoring tier** (doc 03 §1) — it exposes an unchecked accessor, and
forgetting `.valid` produces a wrong answer rather than a crash. It stays
measured and available as an *internal* representation if a hot path ever
justifies it, but it is not something an author writes.

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

Byte-identical source is **necessary but not sufficient**, and normalising
timestamps to get reproducibility introduces a silent wrong-answer bug. [EXPERIMENTS.md](EXPERIMENTS.md) §C.

**Six conditions must all hold** for a build-time cache entry to be used at
runtime:

1. same absolute directory (it selects `__pycache__`)
2. same file basename **and same `def` line number** — both are in the `.nbi`
   filename (`drv.driver-5.py314.nbi`, where 5 is the line number)
3. exact `(st_mtime, st_size)` of the `.py` — a mismatch returns an empty index,
   i.e. a 100% miss
4. same argument signature
5. same `magic_tuple` = (LLVM triple, CPU name, CPU feature string)
6. same sha256 of the function's `co_code` and pickled closure

Plus the original requirements, which still apply: deterministic ordering (no
set/dict iteration order dependence), no addresses, timestamps, `id()` values or
PIDs in generated names, and stable naming derived from module ids.

> **⚠ numba will serve stale compiled code.** Condition 6 hashes `co_code`, which
> **excludes `co_consts`**. Change a numeric constant in a generated driver, keep
> the file size identical, restore the mtime — numba reports a cache **hit** and
> returns the pre-edit answer. `decider build --verify` counting zero compiles
> reports success on it.
>
> Two consequences, both mandatory:
>
> - **Never normalise mtimes for reproducibility.** Use content-addressed file
>   names instead: a driver written to `drv_<sha256[:16]>.py` changes name when its
>   content changes, and a renamed file misses by construction.
> - **No decision-relevant constant is emitted into driver source.** Params are
>   already arguments (doc 03 §4), so that path is safe. A `ruleset`'s thresholds
>   (doc 08 §3) are *not* yet — and that is precisely the path a business-user UI
>   edits. They must ride in the params bundle or a table too.

### 4.3 Driver shape

Steps are njit'd individually; only the glue is generated. Nested `Branch`/`Loop`
emit recursively; a prototype emitter on an unpushed branch does this today (see
the README citation note). E2 must re-establish it here.

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

All generated from one graph and compiled at build (§8). **One row-loop variant
per kernel, not two** — serial unless the author wrote `parallel(...)` (§5.1).

| variant | steps | driver | purpose |
|---|---|---|---|
| `fused` | njit, inlined | njit | production |
| `fused` + taps | njit, inlined | njit | production diagnostics |
| `stepped` | njit | Python, one step at a time | driver-level debugging; **also the fallback path** |
| `interpreted` | Python | Python | reference semantics; full internals |

`stepped` is not a separate build to maintain — it is structurally identical to
the fallback path (§6), so one mechanism serves both.

### 5.1 serial vs `prange`

Every figure previously recorded here was refuted (doc 01 §4, [EXPERIMENTS.md](EXPERIMENTS.md) §F), and so
was the remedy. The conclusion below replaces both.

**Settled: `prange` is authored, not inferred** — the same rule as fusion
(doc 02 §1.2), and for the same reasons.

```python
pipeline = Affordability | fuse(PolicyRules) | Scoring          # serial, the default
pipeline = parallel(Affordability | fuse(PolicyRules) | Scoring) # opt in
```

`parallel(...)` is a combinator like `fuse(...)`: graph data, it renders, diffs and
serialises, and it is **semantically transparent** — `testing/equivalence.py`
asserts a pipeline and its `parallel()`-annotated variant agree. It changes
`parallel=True` on the emitted driver and nothing else.

**Four reasons it is an annotation and not a heuristic:**

1. **Early exit breaks the premise.** Real credit logic short-circuits: some
   applications exit at the first rule, others run fifty. Per-row work is a
   distribution, not a constant, so `ns_per_row` is an average that does not
   describe it — and `prange`'s static schedule load-imbalances against exactly
   that shape. The measurements below were taken on uniform synthetic bodies and
   do not transfer to it.
2. **A warmup measurement cannot work anyway**, structurally — [EXPERIMENTS.md](EXPERIMENTS.md) §F. The
   crossover is set by *total serial wall-clock*, which is per-row work
   **multiplied by row count**; warmup knows the first factor and not the second.
   A probe at n ≤ 10k predicting the variant for n ≥ 100k is right **52.8%** of
   the time overall and **0%** for medium-cost bodies.
3. **One variant, not two.** `parallel=True` costs 1.2–2.6× compile time
   (doc 01 §4b). Compiling both variants for every kernel pays that on every
   kernel that will never want it, and doubles the cache entries that §4.2 shows
   are already fragile.
4. **The threshold is machine-specific**, so an automatic choice makes a pipeline
   behave differently on a laptop and in production, and puts a machine property
   into the audit record (doc 08 §8). An annotation is the same everywhere.

**What the measurement is still good for — a diagnostic, not a switch.** The
crossover invariant is total serial wall-clock: bodies spanning 1500× in per-row
cost all cross between **85 and 239 µs** of serial time, against a measured
fork/join floor of **65–67 µs**. So `explain_kernels()` can report *"this kernel
runs ~N µs serial at your batch size; `parallel()` would likely help"* — an
observation the author acts on, in the same demotion applied to fusion's
vectorisation report. Fitted as an automatic rule it scored 97.9% out of sample
on uniform bodies; that is good enough to advise and not good enough to decide
silently.

`prange` composes correctly with bounded inner loops, including `break` —
bitwise-identical output and iteration counts verified. Early exit *within* a
record is unaffected by parallelism *across* records; the two loops are
independent.

### 5.2 Flags

- **`fastmath`: off by default, enabled selectively.** It is noise (±5%) on
  branch-dominated logic but worth **2–2.5×** on an arithmetic-heavy fused
  driver. Decide per kernel, not globally.
- **`parallel=True` costs 1.2–2.6× compile time**, shape-dependent.
- **`inline='always'` is a no-op** — numba already inlines these (hand-inlining
  matches codegen within 3%).

---

## 6. Fallback

**One layer, not two.** [EXPERIMENTS.md](EXPERIMENTS.md) §B: a nopython driver cannot call a Python function,
so there is no per-node fallback *inside* a compiled kernel. All six
compile-failure modes tested took down the whole driver; not once did one node
degrade while its siblings stayed compiled. The blast radius of one un-njit-able
node is **the whole kernel**.

Measured at 200k rows × 4 steps, all variants bit-identical:

| variant | vs all-njit |
|---|---|
| all-njit, serial | 1.0× |
| **whole driver in plain Python** | **23.5×** |
| one node via `objmode`, per row | **77.0×** |
| one node via `objmode` inside `prange` | **614.7×** |
| escape **hoisted** to one `objmode` call per batch | 29.7× |
| Python pre-pass column, pure njit driver | 31.3× |

> **`objmode` per row is 3× *worse* than giving up and running the whole driver in
> Python.** The graceful mechanism is worse than the ungraceful one, so it must
> not be the recommended path. And `objmode` inside `prange` **compiles silently**
> while being 615× slower — against a 7× speedup for the same driver fully njit'd.

The two shapes worth having, both ~2.5× better than a per-row escape:

1. **Hoist the escape out of the row loop** — one `objmode` call per batch.
2. **Pre-compute the awkward column in Python**, then keep the driver pure.

So an un-njit-able step is a **kernel-boundary decision**, not a per-node one: the
compiler splits the group around it and the rest stays compiled.

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

**There is no grouping heuristic to implement** (doc 02 §1.2). An earlier draft
specified a ~6–9 step cap; it named a range its own evidence shows is already
10–28% worse than not fusing, and no constant can span a decision that ranges
from 0.11× to 1071×. Withdrawn.

What `compile/` implements instead:

- **`apply()` emits one kernel per module by default.** Split holds a flat
  0.20 ns/step at every size measured; it is the predictable choice, and boundary
  stores are near-free.
- **`score()` emits one maximally-fused kernel.** N=1 is far below the ~10k
  break-even, and there is no polars boundary on that path at all.
- **A `fuse(...)` combinator in the graph** groups modules into one kernel
  explicitly. It is authored, not inferred.
- **A hard cap on emitted lines (~500) per kernel**, enforced as a build error
  naming the group. Compile runs ≈15 ms/line and a fully-branching depth-10 nest
  costs 92 s, so this is the one bound that must not be advisory.

Two requirements this places on codegen:

1. **`fuse()` must be semantically transparent.** `testing/equivalence.py` asserts
   that a pipeline and its `fuse()`-annotated variant agree, alongside the
   three-mode ladder. Without that assertion a performance annotation can change
   an answer, and the whole construct becomes unsafe to hand to a non-specialist.
2. **The kernel-to-kernel boundary must be specified**, since splitting is now the
   default rather than the exception. Intermediates between two kernels stay in
   numpy — they do not round-trip through polars. §3 specifies only the polars
   boundary; this is the gap E1 must close.

```python
pipeline.explain_kernels()   # kernels, emitted lines, and whether each vectorised
```

Reporting only — it observes, it does not decide. Vectorisation status is read
from `.inspect_llvm()`, which is the measured mechanism behind the entire effect
(vector IR values drop to 0 at M≥4), so it answers "would fusing here help?"
directly rather than by proxy.

An automatic cost model remains O12, explicitly out of scope for v1.

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
8. A regression test guards the same-name-NamedTuple collision (doc 01 §4c) with
   an **exact structural assertion**, not a timing one: two distinct bundle
   classes must never share `__name__` + field names + field types, because that
   tuple is all `compute_fingerprint` encodes. [EXPERIMENTS.md](EXPERIMENTS.md) §D measured the collision at
   **1.03×**, so a timing budget has zero power and would mark the bug green.
9. A sliced and a multi-chunk frame both produce correct results — the
   `a.offset` trap.
10. **The fallback set is reported and assertable.** A build can require it to be
    empty. This closes a hole in the equivalence ladder: when a node falls back,
    `fused` runs *the same Python object* `interpreted` runs, so the ladder agrees
    on that node by construction — precisely the node where the least verification
    happened. A silent fallback is also a type-dependent one, so CI can otherwise
    certify a different program than production runs.
11. **No module under `binding/` or `params/` imports `json`, `yaml`, `tomllib`,
    `os`, `pathlib`, `socket` or an HTTP client** — asserted by grep and by a test
    that monkeypatches `open`/`socket`/`os.environ` around `resolve_params`. This
    is what keeps config sourcing out of the framework (doc 08 §6).
12. **The topological tie-break is stable.** Reordering the arguments to
    `module(...)` — which doc 03 §3.2 says must not change behaviour — produces
    byte-identical generated source. Otherwise a cosmetic refactor silently
    invalidates the entire numba cache (§4.2).
13. **No stale cache.** Editing a constant in a generated driver, with file size
    and mtime held identical, must **not** produce a cache hit. [EXPERIMENTS.md](EXPERIMENTS.md) §C shows numba
    returns the pre-edit answer, because its index hashes `co_code` and not
    `co_consts`. Content-addressed file naming is what closes this; the test is
    that the doctored-constant case misses.
14. **No `objmode` inside a row loop.** A build fails if generated source contains
    an `objmode` block within the row loop — [EXPERIMENTS.md](EXPERIMENTS.md) §B measured it at 77× serial and
    615× under `prange`, both worse than giving up and running the whole driver in
    Python (23.5×).
15. **An inadmissible dtype fails at the gate, named**, before extraction — with
    the gate catching `BaseException`, since `Decimal` raises a Rust panic that
    `except Exception` does not catch (§1.5).
