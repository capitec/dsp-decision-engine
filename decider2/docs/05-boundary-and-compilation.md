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

### 1.5 The dtype ladder — flexible by default, tighten for speed

**Revised.** An earlier draft called this "the admissible-dtype contract" and
marked `Decimal` and `List` **inadmissible**. That is the wrong default: the
framework must not be restrictive about data. Real inputs are a mix of flat and
deeply nested, with mixed types, and **everything should work — some things just
cost more.** Tightening a dtype is then a performance choice an author makes
deliberately, not a precondition for using the library at all.

Measured, [EXPERIMENTS.md](EXPERIMENTS.md) §A and §B. Three tiers, and nothing is forbidden:

**Tier 1 — zero-copy.** Clean `Float64`, `Int64`, `Int32`, `UInt8`, `Datetime`,
`Duration`. 0.8–1.2 µs/column at 100k rows. Six of 26 combinations measured.

**Tier 2 — copies, still compiled.** Any nullable numeric (~550–630 µs/column at
100k — a 500× cliff, and the single most important number for sizing a boundary);
`Boolean` via its buffer; `Categorical`/`Enum` as codes; `Date`/`Datetime` as
integers. All run in the kernel at full speed once converted.

**Tier 3 — works, via a conversion the framework does for you.** Strings, `List`,
`Decimal`. These cannot enter a nopython kernel directly, so the framework
**converts at the boundary or splits the kernel around the step that needs them**
— both measured at ~30× a pure kernel, against 77× for a per-row `objmode` escape,
which is why the escape is never per-row (§B).

| dtype | tier | how it enters |
|---|---|---|
| Float64, Int64, Int32, UInt8 (clean) | **1** | zero-copy |
| the same, nullable | 2 | copy + validity |
| Boolean | 2 | via buffer |
| Date, Datetime, Duration | 2 | as integers — note `add(datetime64, datetime64)` does not compile |
| Categorical, Enum | 2 | as codes; code stability across frames must be declared |
| Utf8 | 3 | dictionary-encoded to codes, or the kernel splits |
| List | 3 | CSR offsets + values (§O5's `grain`), or the kernel splits |
| Decimal | 3 | **converted to scaled `int64` cents** — which is the money answer anyway (doc 03 §1.2) |

**What the framework owes the author, since nothing is rejected:** `explain_boundary()`
reports each column's tier and its measured cost, so "why is my batch slow" is
answerable by reading a table rather than guessing. Tightening a dtype is then an
informed choice.

**Two hard requirements that survive from the stricter draft**, because they are
correctness rather than performance:

- **Gate with `except BaseException`, not `except Exception`.** `Decimal` raises
  `pyo3_runtime.PanicException`, which inherits `BaseException`. Refined by
  §O11: `_get_buffers()` on a `Decimal` column **succeeds** and returns a usable
  Int128 series; the panic fires one call deeper, on `.to_numpy()` of that buffer.
  So the gate belongs at the conversion, not at extraction.
- **A string never enters a kernel as a string.** It enters as a code. That makes
  code stability a declared property of the input schema rather than an accident
  of the data.

#### Strings in detail — measured, §O

**Numba is not the constraint.** In nopython mode `==`, `!=`, `len`,
`startswith`, `endswith`, `in`, `find`, `split`, `upper` and
`numba.typed.List[str]` all compile and run. Any design premised on "numba can't
do strings" is solving the wrong problem. **The boundary is the constraint**:
building a `typed.List` from a polars column means materialising N Python string
objects and boxing each one — 1340 ms against 43 ms for dictionary codes at 200k
rows, and **31× end-to-end, 438× in the kernel**. That is the whole reason for
the rule above.

**The author still writes ordinary Python.** For a comparison between a
string-typed input and a string constant, codegen emits a kernel over `int32`
codes and **hoists each distinct literal to a kernel argument** holding its code:

```python
def rate(sector: str) -> float:              # what the author writes
    return 0.9 if sector == "private" else 1.0
```

```python
def kernel(sector, out, _lit_private):       # what is emitted
    for i in range(len(sector)):
        out[i] = 0.9 if sector[i] == _lit_private else 1.0
```

0.16 ms at 200k rows, **115× faster** than the `typed.List` route. A literal
absent from the data resolves to a `-1` sentinel and simply never matches, rather
than failing. And because the literal is an *argument*:

> **Changing a string literal is a value change, not a recompile.** Measured:
> `len(kernel.signatures)` stays at **1** across three distinct literal sets. A
> policy moving from `"private"` to `"self_employed"` is a params edit with a
> 3.36 µs swap — the same guarantee doc 08 §2 gives a numeric threshold, which
> an earlier draft assumed strings could not have.

The transform is deliberately bounded to that one shape. It is not general AST
rewriting.

**`re` does not compile in nopython, and that costs nothing.** Regex is a
**frame operation** (doc 02 §6's `frame/`), where polars' Rust `regex` crate beats
Python's `re` by **7×** — 5.98 ms against 42.29 ms at 200k rows. Shape the string
in the frame tier, pass the boolean or the code into the kernel; the kernel then
reads it in 0.05 ms. A step that wants a regex is telling you it is a frame
operation wearing a step's clothes.

> **A bare literal must be a build error, because nothing downstream will catch
> it.** `int32 == "private"` compiles in nopython and is silently `False` forever
> — numba follows CPython's `int == str` semantics, so there is no `TypingError`
> to rely on (§O). A step reading a `str` input and declaring no `str` `param()`
> is therefore rejected at **param resolution, before anything compiles**, naming
> the column and pointing at the param form. This guard cannot be inherited from
> the type checker and cannot be deferred to codegen.

**`typed.List[str]` remains available** for genuine per-row string manipulation
that neither hoisting nor the frame tier covers — with its 31× cost stated up
front rather than discovered in production.

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

### 3.1 Output side: dtype-grouped 2D, layout per entry point

> **This is the internal layout. The user-facing contract is doc 03 §7:** the
> returned frame is *additive* — inputs, plus terminal values, plus whatever the
> pipeline `.emit()`s, minus whatever it `.drop()`s. An intermediate that nothing
> downstream reads and nothing emits is **never materialised**, which is where
> most of the saving below actually comes from.

**Measured — [EXPERIMENTS.md](EXPERIMENTS.md) §J.** Record write-back is confirmed as the dominant batch
cost (**64.0%** of total at 633 outputs here, against E9's 54.7%), and **record
output is dominated on both axes** by dtype-grouped 2D arrays — one
`float64[cols, rows]`, one `int64[...]`, one `bool[...]`, so arity stays at three
arguments.

| output convention | ns/row kernel | write-back @100k | compile |
|---|---|---|---|
| dtype-grouped **row-major** 2D | **800.8** | 505.8 ms | 0.25 s |
| record array *(previously specified)* | 942.2 | 775.6 ms | 11.31 s |
| dtype-grouped **column-major** 2D | 3497.1 | **5.7 ms** | 0.27 s |

Column-major reaches polars **zero-copy** (verified); the other two do not. A
row-major 2D control proves the write-back win comes from *column-major*, not from
"2D instead of records".

**Both 2D forms compile 42× faster** (11.31 s → ~0.26 s), which matters more than
the runtime because it feeds §K's config-change latency.

> **Choose by entry point**, as with fusion (doc 02 §1.2):
> `score()` → **row-major** (kernel time is the whole cost, and there is no bulk
> write-back); `apply()` → **column-major** (write-back dominates).
>
> Per doc 01 §6.1 the `score()` side of that choice is worth ~2.7 µs against a
> 20–100 ms budget, so **settle it on whichever is simpler** unless a request-path
> measurement says otherwise. The batch side is where the evidence bites.

### 3.1b The single-record boundary — dict in, dict out

**This is the serving path, and §1's polars specification does not cover it.**
A realtime request arrives as a dict and never touches polars (doc 02 §3.4), so
extraction, marshalling and write-back are all different code. §N1 measured where
the time goes at realistic width (400 in / 633 out), and it is not the kernel:

| stage | cost | share |
|---|---|---|
| per-field Python loop over the request dict | 673 µs | **92%** |
| kernel | ~1 µs | 0.1% |
| everything else | ~58 µs | 8% |

Three requirements follow, and they are the difference between a 0.7 ms and a
20 ms floor:

- **Marshal whole-row, never per-field.** One bulk conversion into a preallocated
  record buffer, not a Python loop assigning 400 attributes. This is the single
  largest win available anywhere on the request path.
- **Pool the output buffer.** Allocating a 633-wide output per request dominates
  what is left. A pooled buffer is reused across requests; because it is written
  rather than read, the kernel must be compiled against a **writeable** array
  type — a readonly specialisation compiled from a frame-backed array will not
  accept it, and the failure is a confusing typing error at first serve rather
  than at build.
- **Validate params once per generation, not per request.** §N3 found the
  model→NamedTuple conversion, not validation, is where the cost concentrates.
  Convert at `activate()` and hold the NamedTuple; the request path reads it.

Row-major output (§3.1) is the right layout here for the same reason column-major
is right for batch: there is no bulk write-back to amortise, so kernel locality is
the whole cost.

### 3.2 Chunking is mandatory, not an optimisation

**Measured — [EXPERIMENTS.md](EXPERIMENTS.md) §J2.** At 400-in/633-out, a 1 M-row batch **does not fit**:
fitted peak RSS is **16.19 GB** (record) and **11.87 GB** (column-major). The
largest batch inside a 6 GB cap is ~389k rows (record) or ~534k (column-major).

Chunked, peak RSS is **bounded by chunk size and flat against total rows**, and
column-major output stays bit-identical across chunks (134/134 checksums).

> **Default chunk size: 100,000 rows** — the low end of the wall-time-optimal
> 50k–100k band for both conventions, with ~4× the memory headroom of a 250k chunk.

Two contract requirements:

- **Resolve the driver, its coefficients and the generation pointer ONCE per
  batch, outside the chunk loop.** Doc 08 §4 measured that re-reading the
  generation pointer per chunk straddles config versions in **99.87%** of batches.
- **Quote the honest ratio.** End-to-end, column-major beats record by
  **1.03×–1.57×** under chunking, not the 1.74× measured unchunked — per-chunk
  input assembly and output persistence are **58–82% of wall time** and the
  convention touches neither. On the portion it does control the win *grows* with
  chunk size, reaching 3.09×.

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
a startup one. Confirmed independently: 4.93 s → 0.177 s at 200 args, 46.99 s →
0.197 s at 800, with **warm cost flat at ~0.18 s regardless of driver size**.

> **⚠ But how you import it decides whether any of that works — [EXPERIMENTS.md](EXPERIMENTS.md) §J2.**
> The cache **does not survive across fresh processes** when the generated driver
> is loaded with `importlib.util.spec_from_file_location`, even with a
> byte-identical file and a preserved mtime. Import it **by module name** (put its
> directory on `sys.path` and use `importlib.import_module`): measured **12.1 s
> cold → ~0.2 s** on every subsequent process.
>
> This is the same root as §K's seventh cache condition — both are about the
> module's identity in `sys.modules`, not about its bytes.

### 4.2 Determinism is a hard requirement

Byte-identical source is **necessary but not sufficient**, and normalising
timestamps to get reproducibility introduces a silent wrong-answer bug. [EXPERIMENTS.md](EXPERIMENTS.md) §C.

**Seven conditions must all hold** for a build-time cache entry to be used at
runtime:

1. same absolute directory (it selects `__pycache__`)
2. same file basename **and same `def` line number** — both are in the `.nbi`
   filename (`drv.driver-5.py314.nbi`, where 5 is the line number)
3. exact `(st_mtime, st_size)` of the `.py` — a mismatch returns an empty index,
   i.e. a 100% miss
4. same argument signature
5. same `magic_tuple` = (LLVM triple, CPU name, CPU feature string)
6. same sha256 of the function's `co_code` and pickled closure
7. **same `sys.modules` registration name** — [EXPERIMENTS.md](EXPERIMENTS.md) §K. Violating it raises
   `ModuleNotFoundError('<dynamic>')` from inside numba's `pickle.loads`, with no
   hint that a caching contract was broken. Derive it from the same content hash
   as the filename.

**Most violations fail safely.** Measured across a real child/parent boundary
(§K): a filename or cwd mismatch raises `FileNotFoundError` loudly; a cache-dir or
`def`-line mismatch silently *misses* and recompiles — slow but correct. **Only
the mtime/size coincidence produces a silent wrong answer.**

Plus the original requirements, which still apply: deterministic ordering (no
set/dict iteration order dependence), no addresses, timestamps, `id()` values or
PIDs in generated names, and stable naming derived from module ids.

> **⚠ Compiled code can be served stale — at either of two independent layers,
> and both are real.** Condition 6 hashes `co_code`, which **excludes
> `co_consts`**. **Settled** ([EXPERIMENTS.md](EXPERIMENTS.md) §M, resolving §K vs
> §L): CPython's `__pycache__/*.pyc` cache and numba's own on-disk cache are
> **each independently sufficient** to serve a stale value from a byte-identical,
> mtime-preserved edit — which one fires depends on whether the edited constant
> collides with another value in the function's `co_consts`, not on which harness
> ran the test. If the edited literal is shared elsewhere in the function
> (unchanged), the edit is a `co_consts` *insertion* that shifts every later
> `LOAD_CONST` operand, so `co_code` changes and numba's cache misses correctly —
> only CPython's `.pyc` can still serve the pre-edit function object in that case.
> If the edited literal has **no other occurrence** — the shape of a single
> rule's threshold, one value used once — the edit is a same-slot *replacement*,
> `co_code` stays byte-identical, and **numba's own cache serves the stale value
> even with `__pycache__/*.pyc` cleared and the source genuinely re-parsed.**
> Change a numeric constant in a generated driver, keep the file size identical,
> restore the mtime — numba reports a cache **hit** and returns the pre-edit
> answer. `decider2 build --verify` counting zero compiles reports success on it.
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
        out.term_cap[i] = v_term_cap                    # an emitted value
```

Note what this gets for free: short-circuiting (only the taken arm runs), value
overwrite as plain local reassignment, and an emitted value as one extra store.

---

## 5. Compiled variants

All generated from one graph and compiled at build (§8). **One row-loop variant
per kernel, not two** — serial unless the author wrote `parallel(...)` (§5.1).

| variant | steps | driver | purpose |
|---|---|---|---|
| `fused` | njit, inlined | njit | production |
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

Fusion is **non-monotone, and its sign depends on body cost, not module count**
([EXPERIMENTS.md](EXPERIMENTS.md) §E). At fixed module count and fixed rows, body cost alone spans
**62×**: at 20 modules a cheap straight-line body gains 13.0× from fusion while a
heavy one loses 4.5×. Doc 01 §4c's table is a slice through one body-cost class.

The mechanism is **register pressure spilling a dependent FMA chain**, not lost
vectorisation. The worst regression measured is *fully* vectorised — 791 packed
ymm FP ops, zero scalar — and 5.6× slower: it pins all 16 ymm registers, unrolls
~2× instead of ~5×, and spills 47 times per iteration, so it goes latency-bound.
Packed-FP count *rises* with module count while performance falls.

**There is no grouping heuristic to implement** (doc 02 §1.2). An earlier draft
specified a ~6–9 step cap; it named a range its own evidence shows is already
10–28% worse than not fusing, and no constant can span a decision that ranges
from 0.11× to 1071×. Withdrawn.

What `compile/` implements instead:

- **`apply()` emits one kernel per module by default.** Split cost is flat
  *within* a body-cost class (0.82–0.96 ns/step trivial, 1.56–2.38 medium,
  0.85–3.30 heavy — not the 0.20 ns/step an earlier draft claimed, which is below
  the floor of a single kernel call). Flat-in-size is what matters here: it is the
  predictable choice, and boundary stores are near-free.
- **`score()` uses the same grouping as `apply()`.** At N=1 fusion wins 9.4–48×
  against per-call dispatch of 0.44 µs/kernel — but that is a multiple of
  microseconds, and 30 unfused kernels cost ~13 µs against a 20 ms budget (0.07%).
  Maximal fusion would also breach the ~500-line cap by construction at 30 rules
  (§G: 517 lines). One grouping, one codegen path, one cache surface. An author who
  measures a case where it matters writes `fuse()`.
- **A `fuse(...)` combinator in the graph** groups modules into one kernel
  explicitly. It is authored, not inferred.
- **A hard cap on emitted lines (~500) per kernel**, enforced as a build error
  naming the group. Compile is **super-linear in emitted lines** — ∝ lines^1.4,
  with the local exponent reaching 1.96 between 60 and 100 rules (§G) — and a
  fully-branching depth-10 nest costs 92 s. Super-linearity is why this bound must
  not be advisory: the cap is where cost is still recoverable. The ~500-line
  guardrail survives and is slightly conservative; the real crossover is 600–710
  lines.

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
pipeline.explain_kernels()   # kernels, emitted lines, measured cost per group
```

Reporting only — it observes, it does not decide. It reports **emitted lines and
measured cost**, not vectorisation status: an earlier draft had it read
`.inspect_llvm()` on the grounds that vectorisation was "the measured mechanism
behind the whole effect", and §E refuted that — packed-FP count is anti-correlated
with performance, so the proxy points the wrong way. Withdrawn. There is no static
observable that answers "would fusing here help?"; measuring the group does.

An automatic cost model remains O12, explicitly out of scope for v1.

---

## 8. The build step

```
uv run decider2 build <pipeline>     # in the Dockerfile
decider2 build --verify              # asserts NO COMPILATION AFTER WARM-UP
```

**Revised.** This used to say "asserts a runtime load triggers ZERO compiles,
ever" — that stopped being literally achievable once a data-shaped interior's
step (a tree/table/Branch/Loop, doc 08 §3.4, `types.Step.packed`) and the
fused driver itself (`decider2.compile.codegen`, still generated source for
ordinary hand-written steps) both compile LAZILY, on their own first real
call, rather than being eagerly probed at build time the way `decider2.
compile.driver._try_njit` forces one plain step to be. The property people
actually rely on is narrower and still exactly true: **no compilation happens
on the request path.** `Pipeline.precompile()` (`decider2.graph.pipeline`)
moves every specialisation to a controlled point — process start, image
build, or `.serve()`'s own warm-up (`ServeHandle.warm()`, called by
`serving/app.py`'s `app()` before it ever returns an app for a server to bind
a socket on; `GET /ping` answers 503 until `handle.is_warm`, a second,
independent guard) — and `decider2.testing.assert_no_compilation_after_
warmup`/`decider2.testing.count_new_compiles` verify it by COUNTING numba's
own `"numba:compile"` events, not by inference: a `precompile()` that misses
a specialisation is worse than none, because it promises a guarantee it does
not deliver, so this is checked directly rather than assumed.

Measured on a realistic 24-step pipeline (one kernel per step, the doc 05 §7
default — this stage's own report has the full numbers): **cold, no disk
cache, ~4.6s** to first answer; **warm disk cache, ~0.4s**; with `precompile()`
called explicitly at start-up, the compile cost is identical (~3.6s, paid
once, before any request) and the first REAL request then answers in
**~3ms**. Numba's on-disk cache still works wherever it worked before this
pass (verified cold/warm, `NUMBA_DEBUG_CACHE=1`, persistent build dir); what
changed is only that `--verify`'s own claim is now "zero after warm-up",
checked directly, rather than "zero, ever", which was never quite true for
the fused kernel itself.

> **Choose the CPU target deliberately.** Numba's cache keys include CPU
> features, so a cache built on a CI runner with AVX-512 misses on a smaller
> deployment instance. `NUMBA_CPU_NAME=generic` at build *and* run makes entries
> portable but forfeits the instruction selection that makes small kernels fast.
> Either build generic and accept slower kernels, or build in the CPU family you
> deploy on.

---

## 9. Acceptance criteria

This layer is done when:

1. A kernel runs end-to-end over a polars frame: extract → record assembly →
   driver → write-back, matching a numpy reference exactly.
2. The **same kernel** answers a single record with no polars involvement.
3. All three modes agree — `interpreted ≡ stepped ≡ fused` — over a test
   corpus, and deliberately injected drift at each layer is localised to the
   right rung (E3).
4. A `required` null violation fails fast naming the column, count and example
   rows; a nullable column round-trips correctly for `Float64`, `Int64` **and**
   `Boolean`.
5. Retuning any params bundle leaves `driver.signatures` at length 1; changing a
   field's *type* adds one (negative control).
6. A node containing something numba cannot compile (e.g. `re.match`) **splits
   the kernel around itself**: the offending node runs in Python, the nodes before
   and after it stay compiled, and the build names the split and its cause. Falling
   back *without* splitting is impossible — a nopython driver cannot call Python,
   and §B measured the alternatives at 77× (`objmode` per row) and 23.5× (whole
   driver in Python). The blast radius of an un-njit-able node is its kernel, and
   the acceptance test is that the radius stops there (§6).
7. `decider2 build --verify` reports zero compilations AFTER `Pipeline.
   precompile()`/`ServeHandle.warm()` has run (revised — §8 above) —
   verified by counting numba's own compile events
   (`decider2.testing.assert_no_compilation_after_warmup`), not assumed.
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
15. **Every dtype is placed on the ladder at the gate, named**, before extraction
    — nothing is rejected (§1.5, doc 00 §2b). A tier-3 dtype is converted or the
    kernel splits around it, and the gate *reports* which tier each column landed
    on so tightening is an informed choice. The gate must catch `BaseException`,
    since `Decimal` raises a Rust panic that `except Exception` does not catch —
    that is a *probe* failure to handle, not a rejection.
