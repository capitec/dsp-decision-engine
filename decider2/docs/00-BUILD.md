# 00 — Build guide

**Read this before docs 01–08.** They are ~7,900 lines written over several
rounds, and several claims in them were later refuted by measurement. The
corrections are recorded *in place*, so reading linearly you will meet a confident
wrong statement before its retraction. §1 lists every one. §3 says what to build,
in what order, and what is settled versus open for each part.

---

## 1. Superseded claims — read this table first

Every one of these appears in docs 01–08 as an assertion, followed (sometimes
pages later) by its correction. `EXPERIMENTS.md` always wins.

| the doc says | actually | where |
|---|---|---|
| fusion loses because the body stops vectorising (vector IR → 0 at M≥4) | **vectorisation never drops to 0**; it *grows* with M while performance falls. The cause is row-loop unrolling collapse under register pressure | 01 §4b → §E |
| split kernels hold a flat **0.20 ns/step** | 0.82–3.30 ns/step depending on body cost; 0.20 is below one kernel call's floor | 01 §4c → §E |
| fusion break-even is **~10k rows** | no common break-even exists; a cheap straight-line body never crosses | 01 §4c → §E |
| cap fused groups at **~6–9 steps** | withdrawn — that range is already 10–28% *worse* than not fusing. Fusion is authored via `fuse()` | 01 §4c, 02 §1.2 → §E |
| `prange`: a light body never wins; crosses at ~5k; 7.8× at 5 M | **all three refuted** — 2.83–3.11× at 1 M; crosses at 500 rows; 7.8× is the *100k* figure | 01 §4 → §F |
| compile serial and `prange` variants at warmup, dispatch on row count | impossible — crossover depends on batch size, which warmup does not know (52.8% accurate). `prange` is authored via `parallel()` | 01 §4, 05 §5.1 → §F |
| ≈15 ms per emitted source line | super-linear, ∝ lines^1.4, exponent 1.96 at 60→100 rules. Linear predicts 24 s where reality is 57 s | 01 §4b → §G |
| same-name NamedTuple costs **15–24 µs, permanently** | **1.03×** on numba 0.67, and it does not persist. Still a *correctness* defect — so the guard must be structural, not a timing assertion | 01 §4c → §D |
| 1-ULP drift from `log` | **0 of 2,000,000** values differed, even with `fastmath` on the log. `fastmath` itself is real and worse than recorded (17 ULP) | 02 §3.1 → §I |
| extract via `series.to_arrow()` then a 2-tuple `buffers()` unpack | **cannot execute** (needs pyarrow, absent) and is wrong for Utf8 (3 buffers). Use polars-native `_get_buffers()` | 05 §1.2 → §A |
| write-back is 54.7% of total; records are the convention both sides | 64.0% here, and record output is beaten on both axes by dtype-grouped 2D arrays | 01 §4d, 02 §1 → §J |
| per-node fallback to Python inside a fused driver | **impossible** — a nopython driver cannot call Python. `objmode` per row is 77×; plain Python is 23.5×. Blast radius is the kernel | 02 §3.2, 05 §6 → §B |
| byte-identical generated source is enough for cache survival | necessary, not sufficient — **seven** conditions, and stale code can be served, **at both the CPython .pyc layer and numba's own cache layer independently, depending on edit shape** | 05 §4.2 → §C, §K, §L, §M |
| compile in a background worker (read as: thread) | must be a **subprocess** — a thread retains 26–55% of serving throughput, a subprocess 97.9% | 08 §4 → §H, §K |
| `score(net_income=…, expenses=…, …)` with keyword arguments | at 400 inputs that binds in **1190 µs — 5.95% of a 20 ms budget**. `score()` takes a dict | 02 §3.5 → §N2 |
| `decider build --verify` enforces config completeness | it does not; it is a compile-count assertion. Completeness is `resolve_params(complete=True)` | 04 §2.1 → REVIEW §3.1 |
| a param change "provably cannot alter control flow" | false as stated — it cannot alter the **graph**; it absolutely can change which arm a record takes | 04 §2 → REVIEW §3 |
| `explain_kernels()` should report vectorisation | withdrawn — anti-correlated with performance. This was a *review* proposal, not an original one | 02 §1.2 → §E |

**Two further cautions.**

Doc 01 §5's citations — `decider/modules/record.py`, `_build_jit_driver`,
`diagonal_relaxed`, `plan.py`, `name_override` — resolve only against an **unpushed
branch**. On `origin/main` they do not exist and nothing under `decider/` imports
numba. Treat doc 01 §5's code claims as unverified (REVIEW §1).

Doc 01's benchmark figures are **batch** figures. The primary path is a single
record with a **20–100 ms budget**, where the compiled kernel costs ~1 µs — four
orders of magnitude of headroom. Doc 01 §6.1 has the reweighting table. Do not
optimise the kernel for realtime.

---

## 2. What is actually settled

Twenty experiments, harnesses in `experimentation/`, results in
`EXPERIMENTS.md`. These are measured on this machine, this environment
(Python 3.14.5, numba 0.67.0, numpy 2.4.6, polars 1.41.2, pydantic 2.13.4):

- **Extraction**: polars-native `_get_buffers()`; 6 of 26 dtype/nullability
  combinations are zero-copy; every nullable column copies (~600 µs/100k).
- **Dtypes**: a *ladder*, not a gate — **nothing is rejected** (§2b). Strings and
  categoricals enter as *codes*; `Decimal` and `List` sit on the slow tier, being
  converted or split around. The gate must catch `BaseException` when it probes,
  because `Decimal` raises a Rust panic that `except Exception` misses.
- **Output**: dtype-grouped 2D arrays, column-major for `apply()`, row-major for
  `score()`. Both compile 42× faster than records.
- **Chunking**: mandatory — 1 M rows at 400-in/633-out needs 16.2 GB unchunked.
  Default 100k.
- **Marshal/readback**: whole-row bulk, never per-field. 6.7× on the request path.
- **Cache**: seven conditions; import by module name, never `spec_from_file_location`.
- **Lifecycle**: subprocess compile, atomic swap (0 straddled batches of 1564),
  rollback free, 2.5–7.3 s change-to-serving.
- **Serving**: `nogil=True` unconditionally — `nogil=False` reaches p99 = 1270% of
  budget at 16 threads.
- **Rule thresholds and enablement**: arguments, not emitted literals. Retunes
  never recompile.
- **Fusion and parallelism**: authored (`fuse()`, `parallel()`), never inferred.
- **`param()` in the signature**: compiles; the sentinel default needs no stripping.
- **Money**: scaled int64; `round_half_up`; never `Decimal`; never bare `round()`.
- **Ruleset compile**: ≤10 s up to ~35 rules; splitting beats shrinking.

---

## 2b. Decisions taken by the project owner

These are facts about the deployment, not design choices, and they are settled:

| question | answer | consequence |
|---|---|---|
| production hardware | Docker images, target normally known and constant | build for the target, but **record it in the manifest and check at startup**, logging loudly on a mismatch (doc 02 §3.4b) — this is being **open-sourced**, so other deployments will not resemble this one |
| serving | decider2 **ships a server**, keeping `decider` 1's SageMaker conventions | `/ping` + `/invocations`, the overridable `Handler` protocol, swappable backends — and **nothing in the core may import it** (doc 02 §3.6) |
| real data | **not available** — it would leak IP into a public repo | every shape assumption stays marked provisional; the input schema is bootstrappable from a sample frame so it is cheap to correct |
| dtype strictness | **flexible by default; tighten for speed** | doc 05 §1.5 is a *ladder*, not a gate. Nothing is rejected; strings, lists and decimals are converted or the kernel splits around them |

> **This repository is going public.** Before publication, scrub the internal
> identifiers the review already flagged — `AliasCombineModule`
> (doc 01 §5.2) and `_stage_00`–`_09` (docs 01, 03 §3.2, 04 §6) — or
> relax the README's claim that none appear. `decider2/example_projects/` is
> shaped on real internal systems and should be read with that in mind before it
> ships.

---

## 3. Build order

Each layer lists what blocks it. **Do not start a layer whose blockers are open** —
that is how the graph model gets built twice.

### Layer 1 — `compile/boundary/` — READY

Everything here is measured. Doc 05 §1–§3 is a real spec.

`dtypes.py` (admissibility gate) → `extract.py` (`_get_buffers`, clean-column fast
path) → `marshal.py` (whole-row bulk) → `writeback.py` (dtype-grouped 2D, layout
per entry point) → `chunk.py` (100k default).

> **UNBLOCKED — O11 answered (06-open-questions-and-experiments.md §O11).** The format now exists, and
> the finding that shapes it is a constraint rather than a choice: **polars cannot
> supply nullability at all.** A clean and a null-bearing column of the same dtype
> produce `==`-equal `Schema` objects, and `pl.DataType` has no `nullable`
> attribute. So `name` and `dtype` bootstrap from `collect_schema()`, and
> **`nullable` must be hand-authored** — a governance fact like the
> params/structure boundary, not a data fact. Measured propagation:
> `inner`/`semi`/`anti` introduce no nulls, `left` nullifies the right side,
> `full` nullifies both sides *including the join keys*.

### Layer 2 — `compile/` codegen + `runtime/` — READY once Layer 1 is

`naming.py` (content-addressed), `manifest.py`, `emit.py` (deterministic, stable
topological tie-break), `numba/` (kernel, variants, fallback-splits-the-kernel),
`lifecycle.py` (subprocess, staged, atomic), `serve.py` (`nogil=True`).

All measured. `money/` can be built in parallel with these and depends on nothing.

### Layer 3 — `graph/` — SPECIFIED BUT UNVALIDATED

This is the core of the library and **has no empirical support at all**. E2 was
never run. Doc 06 says E2 *"would invalidate versioning-under-the-hood… or the
combinator model"*.

Blockers, in order:

1. **O5 — nesting: largely answered (06-open-questions-and-experiments.md §O5), and it is *not* a third
   kind.** The name is `grain` (a level declaration: key, parent, capacity) with
   two combinators over it — `Each(grain, body)` to descend, `Gather(child,
   into=parent, **folds)` to aggregate. Broadcast falls out of descend rather than
   being a third operation; enumerate happens at the **frame** tier before any
   record-tier code runs. Folds are commutative-associative only (no `first`),
   each carrying a free witness bitset up to capacity 64.

   Measured in numba, nopython, no fallback: descend **14 ns/child**, aggregate
   **35–38 ns/parent**, nested two-level CSR (Application→Entity→Event) at
   **163 ns/application** over 1.86 M events.

   **One shape still open:** project 02's `over(group=)` / `@rung` — a fold that
   collapses duplicate children into fewer children *at the same grain*, which is
   neither descend nor aggregate. It no longer blocks the model's shape.
2. **E2 has never been run.** Build the graph model *as* E2 rather than assuming
   it: scopes, the five-scope invariant, `|` as sequence, `Branch`/`Loop`,
   interface inference and materialisation, `Vocabulary`/`.relabel()`, unbound-name
   resolution (O23).
3. **E3 follows immediately** — the equivalence ladder is the framework's core
   correctness claim and nobody has built three modes and compared them.

### Layer 4 — `interiors/` — BLOCKED

**UNBLOCKED — O15 answered (06-open-questions-and-experiments.md §O15)**, derived from `flat_rules`'
real algebra plus five independently-authored rule documents across four projects.

Keep the four-kind algebra, with three changes each backed by a count:

- **Respell comparison operators as words** (`lt`/`le`/`eq`/`ne`/`gt`/`ge`) — four
  of four real projects invented word spellings; none used `flat_rules`' symbols.
- **Add `Always`.** `flat_rules` cannot express an unconditional rule: an empty
  `CompositeRule` evaluates to **FALSE**, and four of eight `cap_register` rules
  need one. A genuine algebra gap, found by reading code.
- **Add `Predicate`** — a named registered boolean check whose arguments are typed
  as values, so they can be `{param: …}` rather than inlined literals. This extends
  §L's arguments-not-literals rule one level deeper than doc 08 §3 states it.

**The `then` side is a list of effects**, not a single value and not a single
struct — none of the five real shapes can express the others, and a list expresses
all five. Both `first_match` and `all` are needed; both already exist as working
code paths in `flat_rules`.

Still open: O17 (a policy call) and O21, though O15's field inventory is O21's
input.

`decision_table.py` and `scorecard.py` are *less* blocked — they are generic
kernels over tabular data and the shape is clearer.

### Layer 5 — `observe/` — the top risk

`emit.py`, `trace.py`, `audit.py` follow from Layer 3.

`record.py` + `render/` is **O3/E4, the highest-ranked risk, and it is unstarted.**
Build the record first and a renderer second — doc 04 §6.5 guarantees the data, not
the format. The
cold-read study found **9 of 11** independent designers produced no reviewer-facing
artefact when left to themselves. Start from
`example_projects/examples/01-transaction-fraud-interdiction/artefacts/rule-sheet-MS-0208.md`,
which is the best artefact anywhere in that exercise — and add the task that just
failed: **ask a reviewer to reconcile a decision record against the sheet**, not
merely to read it.

`consistency.py` and `blast_radius.py` are new and undesigned, but both are
evidence-backed: six of eleven cold-read divergences were two artefacts asserting
the same fact and disagreeing, and the maintainer persona could not determine a
change's blast radius.

---

## 4. Open questions, by whether they block

**Block a layer — settle before building it:**

| | blocks | why |
|---|---|---|
| input-frame schema format + **O11** (nullability) | Layer 1 | cannot construct a record dtype or pin a signature |
| **O5** nesting | Layer 3 | changes the graph model |
| **O15** interior schema | Layer 4 | `ruleset.py` is unwritable |
| **O21** per-field change class | Layer 4 | the framework must refuse an unclassified field |

**Do not block — resolve when convenient:**

O17 (approval granularity — policy, not design), O19/O20 (need a realistic
pipeline that does not exist yet), O12 (fusion cost model — explicitly out of
scope now that fusion is authored), O9 (stepping UX), and project 02's same-grain fold (§O5's one unresolved shape).

**N4's tail is no longer unexplained** — it is OS scheduler preemption
(`ru_nivcsw`), nonzero in ~1% of calls but **100% of the top-0.1% slowest**. GC,
allocator, page faults and dispatch misses were each measured and refuted
(`ru_minflt`/`majflt`/`nvcsw` correlate at exactly 0.0 across 48,000 calls). It
scales with the *environment*, not with anything the design controls — an 80×
allocation-size range moved nothing. **Pinning to one core makes it worse**
(p99.9 +63%), so `sched_setaffinity` is explicitly not the mitigation;
cgroup/cpuset isolation is.

The K/L attribution contradiction is **no longer open** — settled by
`experimentation/kl-contradiction-resolved/` (EXPERIMENTS.md §M): both caches are
independently vulnerable, for different edit shapes, and content-addressing was
never conditional on which one — see EXPERIMENTS.md §M and doc 05 §4.2.

**Cannot be settled without a human:** O3/E4.

---

## 5. Two things that will bite

**Doc 04 §6 is the design's stated top risk and it is unstarted, not weak.** Nine
of eleven independent designers skipped it. It is people-blocked, so it should
start in parallel with Layer 1 rather than waiting for Layer 5.

**Cross-artefact agreement is the measured defect, not rule authoring.** The mock
projects implemented single-file rule sets essentially perfectly and produced six
contradictions between artefacts that assert the same fact. Whatever else gets
built, `observe/consistency.py` earns its place.

---

## 6. Do not "simplify" these — each looks redundant and is not

A ceremony audit (four independent auditors over the spec and all eleven mock
projects) found ~175 sites of genuine redundancy, now closed as lint rules in
doc 07 §6. It also found five constructs that **read as boilerplate and are
load-bearing**. Each was proposed for deletion by at least one auditor and each
deletion would remove a named guarantee. If you are tempted to cut one, the
burden is to replace the guarantee, not to argue the syntax is noisy.

| construct | why it survives |
|---|---|
| `reads=` on a shadow-isolated node (project 01's `Resolve`; not a decider2 symbol) | It is a **negative** declaration. It says the resolver does *not* contain `shadow_fire_bits`, and that is the entire shadow-isolation guarantee. Deleting it deletes the only statement of what shadow mode may not see. |
| `reads=` on data-shaped interiors | An **upper bound checked at validation** (41 of 43 sites), not documentation. It is what makes a business-user edit checkable without running it. |
| choosing what `.emit()`s | Editorial, not mechanical. In a verified case 4 of 15 steps were worth emitting; inferring "emit everything" would both cost more and say less. The *keyword* `taps=` is gone — an emitted value is just a column (doc 03 §7) — but the judgement it encoded is not automatable. |
| `fuse()` *and* `parallel()` as separate combinators | Orthogonal, and composed as `parallel(fuse(...))`. Collapsing them into one annotation loses the composition. |
| the rule id in the function name | The join key between code, the params document and the policy extract. It is what `implements="§7.4.2"` joins *to*. |

Two more, for the same reason but from the opposite direction — these are absent
and should stay absent:

- **`outputs=` on a module.** 0 of 204 module definitions in the corpus pass it.
  The inferred interface (doc 03 §5.1) is working; do not add a way to restate it.
- **`@step` as a requirement.** It is optional by design (doc 03 §1) and carries
  exactly one job: overriding a default, which 74 genuine renames need. Nothing
  substitutes `param()` defaults for direct calls, because nothing needs to:
  `param()` returns a `float`/`int`/`str` subclass that already *is* its default
  (doc 03 §4.4). A step is directly callable with no decorator, no pipeline and no
  import order.
