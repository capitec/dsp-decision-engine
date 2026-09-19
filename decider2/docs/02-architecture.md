# 02 — Architecture

Evidence for every claim here is in [01-motivation-and-evidence.md](01-motivation-and-evidence.md).

---

## 1. Two tiers, one boundary

`decider` has three co-equal kinds (`expr` / `frame` / `record`) selected
implicitly by *type annotation* (`decider/modules/functional.py:21`,
`_detect_module_kind`). `decider2` has two tiers selected by a *semantic*
question — "is this logic about one record, or about many rows?"

### Record tier (numba)

All decision logic: derived features, scorecards, policy rules, waterfalls,
trees, affordability, offer calculation. Plain pure scalar Python functions,
compiled once into a fused kernel.

### Frame tier (polars)

Only what is inherently about *sets* of rows: joins, group-by aggregations,
filters. Stage one ships `join`, `aggregate`, `filter`. `sort` and `union`
follow; `window` is deferred (schema inference is hardest there, and most window
use in decision logic is aggregation-plus-join).

### The boundary

A kernel consumes named columns and produces named columns. That is the entire
contact surface. Per-column zero-copy extraction in, batched `hstack` write-back
out.

**Calling convention: a 1D structured (record) array per side** (doc 01 §4d).
Positional arguments work — to at least 1600, so 633 outputs was never a
ceiling — but records are 2× better end-to-end, 37× per call, and are the *only*
convention that carries a f8/i8/bool dtype mix with O(1) dispatch. Below ~64
columns per side positional is fine and avoids an assembly copy; above it records
win decisively. The 400-in/633-out target shape is verified working.

Two consequences to hold onto: extracted arrays are `writeable=False` so numba
types them `readonly` — mixing readonly and writeable arrays silently produces
two compiled specialisations — and record **write-back** becomes the dominant
cost at high output counts.

> **Superseded on the output side — doc 05 §3.1, [EXPERIMENTS.md](EXPERIMENTS.md) §J.** Write-back
> measured at **64.0%** of batch total (E9 said 54.7%), and record output is
> beaten on both axes by **dtype-grouped 2D arrays**: column-major for `apply()`
> (write-back 775.6 → 5.7 ms, zero-copy to polars), row-major for `score()`
> (fastest kernel). Both compile **42× faster** than a record. Doc 05 §3.1 is the
> current spec; the input side is unchanged.
>
> And it is no longer "the next optimisation target" in general — that framing
> was batch-only. On the primary single-record path, §N1 found **92% of framework
> overhead is two per-field Python loops**, not write-back and not the kernel.

**You cross the boundary once per kernel, not once per step.** This is why
"all per-record logic in the record tier" holds even for plain arithmetic where
polars is competitive: arithmetic riding inside a kernel that is already running
costs register operations, not a boundary crossing.

### 1.1 Fusion is non-monotone

An earlier version of this document said "make kernels few and large" and that
adjacent modules always fuse profitably. **E7 and E5 independently disproved
that** (doc 01 §4b, §4c): fusion is **non-monotone**, and beyond a small size it
is actively harmful.

| scale | fused vs separate kernels |
|---|---|
| 1–2 modules, >10k rows | neutral to **1.6× faster** |
| 5 modules | **0.76×** |
| 10 modules | **0.51×** |
| 20 modules | **0.33×** |
| 40 modules | **~0.22×** |

Below ~10k rows fusion always wins (≤1.4×, per-call overhead amortisation).
Above it, **register pressure** stops the growing fused body vectorising — split
kernels hold a flat 0.20 ns/step while the fused body climbs 0.22 → 0.61 ns.
The sign flips back when arms are *expensive*, because short-circuiting then
dominates (1071× when a heavy arm is skipped).

**The premise was backwards, not the guidance.** Boundary stores are genuinely
near-free: even with a full polars column round-trip at every boundary, split
still wins at 1 M rows / 10 modules (4.1 ms vs 7.5 ms). So many small modules is
a *safe* style — not because they fuse, but because **crossing a boundary costs
almost nothing.**

Two design claims this exonerates: per-module params bundles are *not* the cause
(one merged bundle degrades identically), and an emitted value does *not* split a kernel
(+0.11 ns/row versus +2.0 ns/row for a real split — ~17× cheaper).

Consequence for the architecture — a clean separation rather than a compromise:

> **The authoring unit (module) and the compilation unit (kernel) are
> decoupled.** Authors write small modules for audit, reuse and readability.
> Kernel boundaries are a separate, explicit concern (§1.2).

Only a genuine frame-tier operation forces a *mandatory* crossing.

### 1.2 Fusion is explicit, not inferred

**Settled.** An earlier draft said "cap fused groups at ~6–9 steps". That number
does not survive its own evidence and has been withdrawn:

- **It is in the wrong unit and on the wrong side of the line.** The E5 table's
  step counts are 3 modules = 6 steps and 5 modules = 9 steps, where the ratios
  are 0.87–0.90 and 0.72–0.76 — fusion is already 10–28% *worse*. The cap named
  the range the same paragraph identifies as harmful.
- **No constant can be right.** The same experiments put fusion at **0.11×** (32
  cheap modules) and **1071×** (one heavy arm skipped). Four orders of magnitude
  on one decision. Doc 01 §4b says so directly — *"it depends on group count
  **and** arm cost"* — and a constant in one variable cannot express that.
- **It is second-order anyway.** The worst case in the table is
  (0.61 − 0.20) ns/step × 39 steps × 1 M rows ≈ **16 ms**, against a record
  write-back of **35–158 ms** at 633 outputs that doc 01 §4d already names as the
  next optimisation target.

So there is no fusion optimiser and no cost model. Instead:

**One grouping, both entry points: never fuse across module boundaries unless an
author asks.** Split cost is flat *within* a body-cost class, a wrong split is
recoverable with `fuse()`, and a wrong fusion is not visible at all.

> **An earlier draft had `score()` "fuse maximally".** It cannot coexist with the
> ~500-line cap (§1.2): §G measures **30 rules at 517 emitted lines**, and a
> realistic realtime pipeline is ~30 rules (doc 04 §6) over a 400-in/633-out
> boundary (doc 01 §4d). So the auto-fused kernel breaches the cap by
> construction — leaving a choice between "nothing realistic builds" and waiving
> the cap on the one kernel that **every interior edit recompiles in `live`
> mode**, at ∝ lines^1.4.
>
> And it buys nothing. Dispatch is 0.44 µs/kernel, so 30 unfused kernels cost
> ~13 µs — **0.07% of a 20 ms budget**. The 9.4–48× at N=1 is a large multiple of
> a microsecond. Against a second codegen path, a second cache-condition surface
> and an unbounded compile, it is not worth it. The measurement stands; the
> conclusion drawn from it did not. Withdrawn.

**Fusion is then something an author asks for**, with a combinator that composes
like any other:

```python
pipeline = Affordability | fuse(ApplyIncomeCap | ApplySectorCap | TermCapBySector) | Scoring
```

Three properties make this safe:

1. **`fuse()` is semantically transparent.** It changes codegen and nothing else.
   This is assertable on the existing equivalence ladder (§3.1): a pipeline and
   its `fuse()`-annotated variant must agree, and `testing/equivalence.py` checks
   it. Performance annotation can therefore never change an answer — which is
   what makes it safe for an agent or a non-specialist to add, move or remove.
2. **It is a combinator, not a flag**, so it is data in the graph like `Branch`
   and `Loop` — it renders, diffs and serialises, and the graph model gains no
   new concept.
3. **Nothing happens implicitly**, so "why is this slow" is answerable by reading
   the pipeline. There is no optimiser whose choices need explaining.

`explain_kernels()` survives as a *reporting* tool — which modules are in which
kernel, and the emitted line count.

> **Correction, from measurement.** A review draft of this section proposed that
> `explain_kernels()` also report **whether the body vectorised**, read from
> `.inspect_llvm()`, on the grounds that vectorisation loss was "the measured
> mechanism behind the whole effect". **That is wrong and the proposal is
> withdrawn.** Across 54 fused kernels the packed-FP count never reached zero; it
> grows monotonically with M, in the *opposite* direction to performance. The
> worst regression measured (ratio 0.18×) has a main loop of 791 packed ymm ops
> and zero scalar ops — fully vectorised and 5.6× slower.
>
> What actually degrades is **row-loop unrolling**. A split kernel unrolls ~5× and
> spills nothing, using 7 of 16 ymm registers; the fused M=20 kernel pins all 16,
> unrolls ~2×, and spills 47 times per iteration, so a long dependent FMA chain
> goes latency-bound. Doc 01 §4c named the right cause ("register pressure");
> doc 01 §4b named the wrong observable. See [EXPERIMENTS.md](EXPERIMENTS.md).

The residual question — a cost model good enough to fuse *automatically* — stays
open as doc 06 O12, now explicitly out of scope for v1.

One guard rail regardless of grouping: **emitted code size**, since compile time
is super-linear in emitted lines — ∝ lines^1.4 ([EXPERIMENTS.md](EXPERIMENTS.md) §G) — and a
fully-branching depth-10 nest costs 92 s. A `fuse()` group whose emitted body
exceeds ~500 lines is a build-time error naming the group, not a silent
90-second compile.

### Why the expression tier is gone

`kind="expr"` existed because chained `pl.when/then` was the only fast path for
per-record logic. With numba as that path, the expression tier collapses into the
record tier. It is **not shipped in stage one** but the seam is designed for it
(§3.3): a node declares *what* it computes, a **strategy** compiles it, so adding
expressions later adds a strategy rather than reworking the graph.

Weird-Python cases (regex, string munging) are served by the **fallback**
(§3.2) — a different mechanism from an expression tier, and already sufficient.

---

## 2. Structure is data

The single most consequential departure from `decider`.

`decider` *dynamically generates classes* — `class TModule(ExpressionModule,
config_class)` at `decider/modules/functional.py:167`, plus `create_model(...)`
at `:397` for the `module_constructor` config mixin. A generated class cannot be
rendered as a diagram, diffed between versions, or safely edited by a tool.

The concrete blocker is that rendering requires a *valid instance*:
`expand_nodes` (`modules/functional.py:171-186`) calls `Node.from_callable`
(`modules/expression.py:148`) with `{"config": self}`, so drawing the graph would
mean inventing valid params for every registered type.

In `decider2` a module is a **pydantic instance describing the graph**: nodes,
declared inputs and outputs, params schema, emitted values. Authoring ergonomics are
unchanged — you still write `module(fn1, fn2)` — but the result is inspectable
data.

This buys three stated requirements directly:

| requirement | mechanism |
|---|---|
| visual verification by a non-programmer | render the data structure |
| "audit step x changed y from 1 to 2" | diff two module/params objects |
| AI-assisted rule editing | mutate validated data, not generate code |

Explicitly **not** class-with-methods: `self` in a step breaks the purity
requirement and blocks numba compilation.

### 2.1 Config resolution — the hybrid

**Settled**, measured in doc 01 §5.7. The graph is plain data in a registry;
*additionally*, each registered definition gets a **thin generated pydantic
model** (`type: Literal[<id>]` + its params fields, no graph) which participates
in a discriminated union used purely for config validation.

```python
# 1. registration: stage a thin generated model whose params are INLINED and
#    which back-references the plain-data definition
definition = ModuleDefinition(steps=[...], params=AffordabilityParams)

class ConfigModel_credit_affordability(BaseModel):     # generated
    model_config = ConfigDict(extra="forbid")
    type: Literal["credit:affordability"]
    min_ratio: float = Field(0.3, ge=0, le=1)          # params inlined, not a nested dict
    definition: ClassVar[ModuleDefinition] = ...       # back-reference, set at generation
```

**Why this is one native pass — four phases, and the lookup is not in the hot
one:**

| phase | when | cost |
|---|---|---|
| 1. registration | import time | builds definition + generated model |
| 2. finalisation | **once**, after all registrations | assembles the union |
| 3. validation | per config | **single pydantic-core pass** — tag resolution, params validation, nested recursion, error localisation. No Python, no registry access. |
| 4. binding | per config, after validation | walk validated tree, read `type(node).definition`. **Cannot fail** — validation already proved every tag is registered. |

The critical property is that params are **inlined into the generated model**, so
pydantic knows the whole schema statically and never needs to ask "which model
applies here?" mid-traversal. Binding is an attribute access afterwards, on a
tree that is already known-valid. Measured at ~0.22 µs/node (44.6 µs validate →
58.5 µs validate-and-bind, 63 nodes).

A plain registry (design B) cannot do this: with config shaped
`{"use": id, "params": {…}}`, params are an untyped dict, so the definition must
be looked up *inside* validation to know what to validate against — which forces
the whole traversal into Python and measured 73.5 µs, no better than the status
quo.

Net: 1.27× faster end-to-end than the status quo, *strictly better* at locating
errors in nested configs, and the graph never lives inside a class.

### 2.2 There is no runtime registry

Worth stating plainly, because "registry" invites the wrong mental model: once
each generated model carries its `definition` as a `ClassVar`, **the union is the
index**. Every id, every param schema and every definition is reachable by
walking its members — which is why E0 measured "list all types and their param
schemas" at 2 lines for this design.

What remains is a **build-time staging collection** only:

1. registration accumulates generated models (a union can't be assembled until
   all members exist);
2. finalisation consumes that collection once to build the union;
3. afterwards it is dead weight.

Dict vs list is an implementation detail; a dict buys free duplicate-id
detection.

The two ways a definition is reached:

| path | mechanism | staging collection involved? |
|---|---|---|
| **Python** | `Affordability = module(fn1, fn2)` — the object holds its own graph | **no, ever** |
| **Config** | `use` id → union member → `ClassVar` definition | no, after finalisation |

So "registering" a module does not mean adding it to a lookup table for later —
it means **making it reachable from config** by generating its config model. Code
that composes pipelines in Python never touches any of this machinery.

Three constraints follow, all non-optional:

1. **Build the union exactly once, at finalisation** — never per registration.
   Per-registration rebuilds leave stale nested unions that silently mis-validate
   (doc 01 §5.7), and they make registration superlinear (exponent 1.39;
   333 ms → 58 ms at 175 types). Operationally this means **all registration must
   complete before the first config is validated** — extension/plugin loading is
   a startup activity. Registering afterwards requires re-finalising, and
   silently skipping that re-finalisation *is* the stale-union bug.
2. **Wrap `union_tag_invalid`.** The inherited message is 1332 chars listing every
   tag. Catch it and re-raise with difflib "did you mean" suggestions from the
   registry.
3. **`extra="forbid"` on params models.** Otherwise a misspelled param is
   silently ignored and surfaces only as the real field being "missing".

---

## 3. Execution

### 3.1 Three modes, one definition

| mode | steps | driver | use |
|---|---|---|---|
| `fused` | njit, inlined | njit | production, batch and realtime |
| `stepped` | **njit (real compiled code)** | Python, one step at a time | driver-level step-through; production numerics |
| `interpreted` | Python | Python | full internals inspection; reference semantics |

All three are generated from the same declared graph, so they cannot drift
semantically. **Taps are orthogonal to the mode**, not a fourth one — any mode may
carry them, at ~0.11 ns/row (§7), which is what makes always-on production
diagnostics affordable.

**The equivalence ladder.** Three-way agreement
(`interpreted ≡ stepped ≡ fused`) is the framework's core correctness test, and
disagreement localises to a layer:

- `interpreted` ≠ `stepped` → **numba changed a step's semantics** (float
  precision, integer overflow, division)
- `stepped` ≠ `fused` → **fusion or inlining changed something**

Without the middle rung you would see "compiled disagrees with Python" and have
no way to tell which layer caused it.

> **Measured, [EXPERIMENTS.md](EXPERIMENTS.md) §I.** The `log` half of this did **not** reproduce: 0 of
> 2,000,000 values differed between njit `math.log`, njit `np.log`, host `np.log`
> and CPython `math.log` — 0 ULP, even with `fastmath=True` on the log itself.
>
> The `fastmath` half is real and larger than recorded: on a 16-term chain it
> made **46–73% of rows differ, by up to 17 ULP**, while buying only 1.09×. With
> `fastmath` off, agreement was **100.00% bit-exact** over 20,000 rows.
>
> So exact equality is the right acceptance criterion (doc 05 §9.1), and
> `fastmath` is the one thing that breaks it: a kernel enabling it is **excluded
> from the exact-agreement assertion** and must declare a tolerance. The
> divergences that actually threaten the ladder are **integer overflow** and
> **rounding**, not `log` — see doc 03 §1.2.

`interpreted` is the reference implementation. A large internal workload arrived
at the same pattern independently — its raw-Python path is described there as
"the correctness oracle the JIT backend is checked against".

`stepped` is *also* the fallback path (§3.2), so it is not a separate debug build
to maintain. It additionally provides batch-wide full tracing if ever wanted, at
the measured ~4× materialisation cost.

### 3.2 Graceful degradation, at the kernel boundary

The compile mechanism has a precedent rather than a production pedigree. An
unreleased branch of `decider` njits each function and codegens a fused row loop
(`decider/modules/record.py:108-183`, `_build_jit_driver`), gated behind
`jit: bool = False` (`modules/record.py:45`); it is opt-in, available only for
`kind="record"`, and has no fallback, no observability and no topological sort
(`modules/record.py:18-20`). Two prototypes in this repo cover the rest:
`experimentation/graph_poc/fallback_pipeline.py` implements exactly the two-layer
degradation below, and `experimentation/jittree/test.py` benchmarks four
compilation strategies for a rule tree.

**Read that as a feasibility signal, not as de-risking.** `decider2` is a
ground-up build: nothing in the released `decider` imports numba, none of this
has run in production, and no part of the old implementation is being carried
forward. The prototypes show the mechanism works; they do not show it works at
the scale, dtype coverage or governance surface this design requires.

Nothing requires the author to write numba-friendly code.

> **⚠ Superseded — [EXPERIMENTS.md](EXPERIMENTS.md) §B.** An earlier draft specified
> *two* layers, the first being "per node: fall back to plain Python for that node
> only, every other node unaffected." **That layer cannot exist.** A nopython
> driver cannot call back into Python, so there is no per-node escape hatch inside
> a compiled kernel. The measured alternatives are `objmode` per row at **77×**
> (615× under `prange`) and running the whole driver in Python at **23.5×** — both
> worse than the thing they were meant to rescue. Doc 05 §6 carries the
> specification; this section is the one a `fallback.py` builder opens, so the
> correction belongs here too.

**One layer.** An un-njit-able node is a **kernel-boundary decision**: the
compiler splits the group around it, that node runs in Python, and the nodes
before and after it stay compiled. The blast radius is the kernel, not the
pipeline and not the node. The decision is cached per node so a doomed compile is
not retried every call, and the build names the split and its cause.

Only `numba.core.errors.NumbaError` triggers fallback. A genuine runtime bug
(e.g. `ZeroDivisionError`) always propagates, compiled or not, so it can never be
silently masked as "this needed a fallback".

### 3.3 Compiled variants, decided by the author

Compile latency is a non-issue (long-lived processes only), so every variant is
**compiled at image build** (§3.4) and *selected* at warmup by measurement. Note
"warmup" here means measuring already-compiled variants, never compiling — §3.4
requires a runtime load to trigger zero compilations.

- **serial vs `prange`** — **authored, not inferred.** Serial is the default; an
  author writes `parallel(...)` around a region, exactly as with `fuse(...)`
  (§1.2). One variant is compiled per kernel, not two. Rationale and the refuted
  alternatives are in doc 05 §5.1; the short version is that real credit logic
  short-circuits, so per-row work is a distribution rather than a constant, and
  every automatic rule tested was fitted on uniform bodies that do not resemble
  it. `parallel=True` costs 1.2–2.6× compile time, so compiling both variants
  everywhere pays that on kernels that will never want it.
- **`fastmath` off by default, enabled selectively.** Noise (±5%) on
  branch-dominated logic, but worth **2–2.5×** on an arithmetic-heavy fused
  driver (doc 01 §4c). Decide per kernel, not globally — and note it is a source
  of 1-ULP drift, so an enabled kernel needs a tolerance policy in the
  equivalence ladder (§3.1).

### 3.4 Compilation happens at image build, not at startup

Compile cost is real — 200 ms to 2 s for realistic shapes, and up to 92 s for a
pathological fully-branching one (doc 01 §4b). Since deployment is Docker images
for long-lived batch jobs and endpoints, that cost belongs in the **build**:

```
uv run decider build <pipeline>     # in the Dockerfile
```

It generates the driver source files, compiles them, and leaves a warm numba
cache in the image. Two requirements, both of which are cheap now and expensive
to retrofit:

- **Codegen must be deterministic.** Numba's on-disk cache is keyed on source, so
  generated code must be byte-identical between build and runtime or every entry
  misses and you silently recompile at startup. No dict-iteration-order
  dependence, no addresses or timestamps in generated names.
- **Generated code must live in real files, not `exec`.** Numba cannot cache code
  with no source file — an earlier prototype hit exactly this
  (`cannot cache function: no locator available for file '<string>'`). This is
  the highest-leverage implementation decision in the compile layer: writing
  drivers to real `.py` files took a 1200-argument driver from **29.5 s cold to
  0.80 s warm** (doc 01 §4d), converting the whole compile blowup into a
  build-time cost.

`decider build --verify` should assert a runtime load triggers **zero**
compilations, so the cache silently ceasing to work is caught in CI rather than
discovered as slow startups.

> **One deliberate tradeoff.** Numba's cache keys include CPU features, so a
> cache built on a CI runner with AVX-512 misses on a smaller deployment
> instance. Setting `NUMBA_CPU_NAME=generic` at build and run makes entries
> portable but forfeits CPU-specific vectorisation — which §1.1 shows is exactly
> what makes small kernels fast. Either build for a generic target and accept
> slower kernels, or build in the CPU family you deploy on. Choose explicitly;
> don't discover it.

### 3.4b The build target is checked at startup, never assumed

Numba's cache keys include CPU features, so a cache built on one CPU family is
silently discarded on another and every process start recompiles — 5 to 50 s,
appearing as "slow startups" rather than as an error.

Deployment is Docker images, so the target is normally known and constant. The
framework still **records the build target in the manifest and compares it at
startup**, logging loudly on a mismatch rather than silently recompiling. That
costs nothing, decides nothing, and turns the failure mode from invisible into
obvious — which matters more here than in a closed system, because **this library
is intended to be open-sourced** and other people's deployment targets will not
resemble this one.

`decider build --cpu generic` remains available for a genuinely mixed fleet. It
forfeits CPU-specific vectorisation, which is what makes small kernels fast — but
at a 20–100 ms single-record budget against a ~1 µs kernel (doc 01 §6.1), that is
very unlikely to be the constraint.

### 3.5 Two entry points, one kernel

```python
Affordability.apply(frame, params=p)              # batch: polars in, polars out
Affordability.score({"net_income": 42000.0,        # realtime: a dict, no polars
                      "expenses": 18000.0,
                      "instalment": 3100.0, …}, params=p)
```

The realtime path **bypasses polars entirely**. A single-record request is deep
inside the regime where marshalling costs more than the work (viability floor
~10k rows), and polars' per-call dispatch floor alone would exceed the request
budget.

> **`score()` takes a dict, not per-field keyword arguments — measured, not a
> style choice.** At the width this document's own §4d establishes as realistic
> (400 inputs), a literal keyword-argument call (`score(net_income=…,
> expenses=…, …)` with all 400 named) costs **1190 µs p50 — 5.95% of a 20 ms
> budget — on the calling convention alone**, because CPython's keyword-argument
> *binding* for a many-parameter signature scales close to quadratically with
> parameter count. A dict costs **60.1 µs (0.30%)** for the same data — 20× less
> — and is what a real caller already has (a validated request body, a
> DataFrame row) rather than 400 hand-typed `name=value` pairs. kwargs syntax is
> still fine, and reads best, for a small, hand-written call — a module with a
> handful of inputs, a test, a doc example — where the cost is irrelevant either
> way. A caller that has already profiled a hot loop may instead pass a
> pre-built, reused numpy record buffer as an explicit opt-in fast path
> (39.8 µs, 0.20%) — cheaper still, but the buffer's lifetime (field order,
> dtype, staleness across a doc 08 §4 config swap) is the caller's to manage, so
> this is not the default. `experimentation/n2-calling-convention/`,
> EXPERIMENTS.md §N2.

> **Serving kernels compile `nogil=True`, unconditionally — measured, not a
> default left to the author.** A production endpoint serves concurrent
> single-record requests through the same compiled kernel. At 1–16 concurrent
> threads calling it, `nogil=True` holds tail latency flat (p99/max 4–12% of a
> 20 ms budget) at every thread count; the identical kernel compiled
> `nogil=False` degrades to **p99 = 1270% of budget (12.7× the entire budget)
> at 16 threads** — a GIL convoy effect, calls queuing behind each other because
> a `nogil=False` call holds the GIL for its own duration. Total throughput is
> unaffected either way (Python-level glue dominates and serializes it
> regardless), so this is purely a tail-latency requirement, easy to miss under
> a median-only benchmark. Doc 08 §4 separately measured `nogil=True` losing to
> `nogil=False` when competing against a background *compile* thread — a
> different scenario, now moot in production since doc 08 §4's compile
> subprocess never competes with a serving thread at all. `experimentation/n4-tail-concurrency-swap/`,
> EXPERIMENTS.md §N4.

---

### 3.6 Serving is a package, not a design constraint

**decider2 ships a server** — but the design must not depend on it, and anyone
should be able to replace it in an afternoon. The existing `decider/serving/`
already has the right shape and **decider2 keeps its conventions**:

- **SageMaker routes**: `GET /ping` (health) and `POST /invocations` (inference).
  These are a platform contract, not a preference.
- **The handler protocol is the seam**: `init_fn`, `module_fn`, `input_fn`,
  `process_fn`, `output_fn`, `shutdown_fn`. A user drops an `inference.py` with a
  `Handler` class into the working directory and overrides only what they need —
  most often `input_fn` to change request preprocessing.
- **Server backends are swappable** behind that protocol. `decider/serving/servers/`
  ships starlette and sanic today; the protocol is what makes that possible.

Three rules keep it decoupled, and they are what makes "write your own serving
layer" real rather than nominal:

1. **Nothing in `graph/`, `compile/`, `params/` or `interiors/` may import from
   `serving/`.** A lint enforces it. The core must be usable as a library with no
   server present.
2. **The core's entry point is `score(dict) -> dict`** (doc 02 §3.5). Everything
   the server does — content negotiation, parsing, formatting, health — sits
   *above* that line and is replaceable without touching it.
3. **Serving owns no state that the core needs.** The params cell and the
   generation pointer live in `runtime/`, not in the handler, so a custom serving
   layer inherits config swapping and atomic activation for free rather than
   reimplementing them.

> **One thing to carry forward from `decider` and one to leave behind.** Keep the
> overridable-`Handler`-from-the-working-directory pattern — it is the mechanism
> that lets a team change preprocessing without forking. Leave behind
> `subscribe_version_updates`, which polls every 10 s and hot-swaps credit logic
> inside `except Exception: pass` (doc 08 §6). Config arrives through
> `ParamsCell.swap` and `Runtime.activate`, which are explicit and measured.

---

## 4. Values, params, and tables

Three distinct kinds, deliberately not conflated.

**Values** flow through the graph, governed by one invariant: **every scope has a
declared interface, names are distinct within a scope, and versioning happens only
at scope boundaries.** The five scopes are step, module, branch arm, loop body and
**pipeline**. A module's interface is *inferred* from its steps and then
materialised into the module data, so it costs no ceremony to author and is still
real enough to render, diff, freeze and rebind (doc 03 §5.1).

So a module's interior is a pure order-independent DAG (no overwrite, distinct
names), while **`|` is a sequence** — written order is execution order — and
overwrite at a *boundary* is the normal case and is how a waterfall
is expressed — `SeedTermCap | ApplyIncomeCap | ApplySectorCap`, each rule an independently
auditable unit. Ordering lives in `|`, never in source declaration order and never
in an argument list. Each version has exactly one producer, so attribution is
exact and a version chain is just a list of boundary crossings. Full treatment in
doc 03 §3.

There is **one type** — Module — and three combinators: `|` (sequence), `Branch`
(conditional, n-way for routing) and `Loop` (bounded iteration with declared
`carries`). Branches and loops are not separate node kinds, which is what keeps
the graph model, compiler, lineage, trace and audit machinery each handling a
single concept. Doc 03 §8.

**Params** are per-invocation tunables in a validated bundle. Fixed type,
free-changing values — so retuning never triggers recompilation. An invocation is
1 row or N rows, which covers realtime payload params and batch alike. Business
users tune params; engineers change structure.

✅ **Measured — [EXPERIMENTS.md](EXPERIMENTS.md) §N3: affordable as written.**
Full `resolve_params(doc, origin=..., complete=True)` validation of a realtime
payload costs 9.7 µs at 1 module instance, 56.7 µs at 10, and 256.5 µs at 50
(doc 03 §4.1's "a realistic pipeline has many") — 0.05–1.28% of a 20 ms budget,
under the whole-millisecond flag. So a realtime request *may* carry raw params
per this section, on performance grounds; whether it *should* is a governance
question (doc 04 §2.1), not a performance one. The model→`NamedTuple` conversion
below costs more than validation itself at scale (2.2× at 50 modules) and is the
part worth caching by content if the same document recurs across requests.

Params are **namespaced per module instance**, so composing `mod1 | mod2 | mod3`
exposes the union without collisions, lets the same module appear twice with
different values, and bounds a param change's blast radius to one module.
Genuinely global values live in one **shared** bundle reached by a second
reserved parameter name (`shared.base_rate`), declared once in a `SharedParams`
model and passed by reference to the steps that ask for it — never copied into
per-module bundles. Modules may also have params **bound** at composition time to
remove them from the caller-facing interface, without making them compile-time
constants. Doc 03 §4.1–4.3.

Note that anything a compiled step reads must arrive as an **argument**: a njit'd
function has no access to Python runtime state, which rules out ambient supply
via `ContextVar` at execution time. It would also make execution modes differ in
mechanism, undermining the §3.1 equivalence ladder.

> The performance boundary (type fixed → no recompile) and the governance
> boundary (business user vs engineer) are **the same boundary**. One mechanism
> enforces both.

**Two refinements, both in doc 08.** First, params/structure is too coarse a
binary — it has no row for a table and no row for a rule, and it puts "add a
rule" in the same bucket as "write new Python". Doc 08 §2 replaces it with three
change classes: **values** (free), **module interiors** (one background compile
and a staged swap), **skeleton** (rebuild). Second, "type fixed" needs
enumerating: a `float | None` field toggling between set and unset flips
`float64` ↔ `Optional(float64)` and *does* recompile. Doc 08 §4.2 lists what
forces one.

**Tables** are lookups keyed by a record attribute — segment-varying cutoffs,
term tables. Data, indexed, separately validated. Not params (params don't vary
by record) and not values (they aren't produced by a step).

---

## 5. Lineage requires declarative frame operations

Static lineage — "which inputs and steps can affect output *z*", answerable
**without executing anything** — is a hard requirement. It falls out of the
record-tier graph for free.

It does **not** survive arbitrary polars. An opaque `.join()` or ad-hoc lazyframe
manipulation makes column lineage unanswerable. Hence frame operations must be
**declarative wrappers with known schema transforms**.

A related failure mode to avoid: `BranchModule` reconciles a cross-branch dtype
mismatch with `pl.concat(..., how="diagonal_relaxed")`
(`decider/modules/primitives/branching.py:96`), which upcasts the whole column to
string and has corrupted genuinely-computed boolean values. Declared output
schemas make that class of implicit reconciliation impossible.

The escape hatch still exists, because blocking people is worse — but it is an
explicit **lineage boundary**:

```python
@breaks_lineage(outputs=["risk_flag"])
def custom_window_thing(frame: pl.LazyFrame) -> pl.LazyFrame: ...
```

The name is chosen for greppability: `grep -r "@breaks_lineage"` yields the
complete list of lineage gaps in a codebase, which makes it a governance feature
rather than just a warning. Lineage queries crossing one report `unknown` rather
than guessing. Outputs are still declared so **schema** propagation survives even
where lineage does not.

Tracking schema through the pipeline also gives free wiring validation, which
kills the silent-composition-failure class recorded in doc 01 §5.2.

---

## 6. Package layout

Reconciled against every settled decision and measured result. §6.1 is the
traceability table — if a decision has no home here, the layout is stale.

```
decider2/
  graph/         values.py       # value identity, boundary versioning, overwrite chains
                 step.py         # a Step: declared inputs, output, params
                 module.py       # Module = pydantic INSTANCE describing the graph
                 interface.py    # interface INFERRED from steps, materialised as data;
                                 #   contract= snapshot + breaking-change check (doc 03 §5.1)
                 combinators.py  # | (sequence), Branch, Loop, fuse, parallel, Map (doc 03 §8)
                 vocabulary.py   # project name map + .relabel() instance relabel (doc 03 §5.2)
                 scope.py        # the five scopes; pipeline precedence; shadowing error (§2.1)
                 resolve.py      # name binding, and the did-you-mean for an unbound one (O23)
                 schema.py       # declared in/out schemas and propagation
                 lineage.py      # static "what can affect z" — MUST descend into interiors (O22)
  interiors/     kind.py         # the data-shaped module protocol: reads/writes declared in
                                 #   code, body from a validated document (doc 08 §3)
                 ruleset.py      # codegen kind — heterogeneous predicates (doc 08 §3.4)
                 decision_table.py, scorecard.py   # generic-kernel kinds, zero recompile
                 schema.py       # the closed node vocabulary (O15)
                 fields.py       # per-FIELD change class; refuses an unclassified field (O21)
  params/        model.py        # pydantic model -> NamedTuple; param() harvested from the
                                 #   signature (doc 03 §4.4); conversion CACHED (§N3)
                 tables.py       # keyed lookups — ONE spelling (O4; six were invented)
                 cell.py         # ParamsCell: read once per invocation, atomic swap
  money/         scaled.py       # int64 cents; Decimal never crosses the boundary
                 rounding.py     # round_half_up, identical in every execution mode (doc 03 §1.2)
  compile/       strategy.py     # THE SEAM: a strategy compiles a node
                 emit.py         # deterministic codegen; stable topological tie-break
                 naming.py       # content-addressed driver + sys.modules name (doc 05 §4.2)
                 manifest.py     # build manifest; --verify checks hashes, not just compiles
                 numba/          kernel.py, variants.py, fallback.py  # fallback splits the
                                 #   KERNEL, never a node (§B)
                 boundary/       dtypes.py    # the dtype LADDER — nothing rejected (§1.5)
                                 extract.py   # polars-native _get_buffers, never to_arrow
                                 marshal.py   # WHOLE-ROW bulk, never per-field (§N1)
                                 writeback.py # dtype-grouped 2D; layout per entry point (§3.1)
                                 chunk.py     # mandatory above ~400k rows; default 100k (§3.2)
  frame/         join.py, aggregate.py, filter.py, opaque.py
  observe/       emit.py, trace.py, audit.py, otel.py
                 record.py       # the structured decision record — the ONLY guaranteed
                                 #   output; every rendering is built from it (doc 04 §6.5)
                 render/         # DEFAULT renderers over that record, replaceable without
                                 #   forking. rule_sheet.py is one renderer, not the artefact
                 blast_radius.py # what a governance-sensitive edit touches, BEFORE the edit
                 consistency.py  # cross-artefact agreement as a build step
  runtime/       invoke.py       # apply() batch + score() single record (dict, not kwargs)
                 plan.py         # ordering DERIVED from the graph, never declared
                 lifecycle.py    # generations: stage -> compile (SUBPROCESS) -> activate
                 serve.py        # nogil=True unconditionally for serving kernels (§N4)
  serving/       handler.py      # init/module/input/process/output/shutdown — THE SEAM
                 servers/        starlette.py, sanic.py — swappable behind the protocol
                 parse.py, format.py, media_types.py   # content negotiation
                 # nothing in graph/ compile/ params/ interiors/ may import this
  binding/       register.py, finalise.py, admit.py, fingerprint.py, errors.py
  testing/       assertions.py, equivalence.py, golden.py, impact.py
                 corpus.py       # boundary-value generation; sampling misses overflow (§I)

decider2_credit/     scorecard/, tree/, rule_table/, waterfall/, affordability/
<client extensions>  same surface, registers into registry
```

### 6.1 Where each settled decision lives

| decision | evidence | home |
|---|---|---|
| fusion and parallelism are **authored**, not inferred | §E, §F | `graph/combinators.py` |
| per-node fallback is impossible; splitting the kernel is the mechanism | §B | `compile/numba/fallback.py` |
| cache survival needs seven conditions + content addressing | §C, §K | `compile/naming.py`, `manifest.py` |
| import the driver **by module name**, never `spec_from_file_location` | §J2 | `compile/naming.py` |
| dtype-grouped 2D output; layout per entry point | §J | `boundary/writeback.py` |
| whole-row bulk marshal/readback, never per-field | §N1 | `boundary/marshal.py` |
| chunking is mandatory; default 100k | §J2 | `boundary/chunk.py` |
| `score()` takes a dict, not 400 kwargs | §N2 | `runtime/invoke.py` |
| serving kernels compile `nogil=True` | §N4 | `runtime/serve.py` |
| compile in a **subprocess**, not a thread | §H, §K | `runtime/lifecycle.py` |
| rule thresholds and enablement are **arguments** | §L | `interiors/ruleset.py` |
| change class per **field** | O21 | `interiors/fields.py` |
| lineage descends into interiors | O22 | `graph/lineage.py` + each kind |
| an unbound name is a typo, everywhere | O23 | `graph/resolve.py` |
| money is scaled int64; `round_half_up` | §I | `money/` |
| corpus must include boundary values | §I | `testing/corpus.py` |
| the decision record is data; renderings are replaceable | O3, cold-read §5.1, doc 04 §6.5 | `observe/record.py` + `observe/render/` |
| flexible dtypes; tighten for speed, reject nothing | §A, §B, §O11 | `boundary/dtypes.py` |
| serving is replaceable; SageMaker `/ping` + `/invocations` | decider 1 convention | `serving/` |
| build target checked at startup, never assumed | §C | `compile/manifest.py` |
| blast radius is visible **before** an edit | cold-read §5.2 | `observe/blast_radius.py` |
| cross-artefact agreement is checked | cold-read §2 | `observe/consistency.py` |
| interface inferred, materialised, freezable | REVIEW §5 | `graph/interface.py` |

Placements that are deliberate rather than incidental:

- **`observe/record.py` is not a documentation tool.** It emits the structured
  decision record that doc 04 §6.5 makes the framework's *only* guarantee here, and
  the cold-read study found **9 of 11** independent designers produced nothing like
  it. Giving it a home in the core library — rather than leaving it to each
  project's README — is the structural answer to that. `render/` ships defaults
  over it; a team that wants a different shape writes a renderer, not a fork.
- **`observe/consistency.py` exists because six of eleven divergences had one
  shape**: two artefacts asserting the same fact and disagreeing. Rule authoring
  was near-perfect; cross-artefact agreement was not checked by anything.
- **`interiors/` is a top-level package, not a corner of `graph/`.** It is where a
  business user's edits land, so it carries the same obligations as the graph:
  lineage, render, a closed vocabulary, and a declared change class per field.
- **`money/` is its own package** because the failure it prevents is a wrong
  answer to the cent, not a performance problem — and because `Decimal` cannot
  cross the boundary at all.
- **`binding/`, not `config/`.** This package validates and binds documents it is
  *handed*; it never fetches one. `decider/config/` accreted 553 lines of storage,
  semver and polling machinery once already (doc 08 §6). There must be no
  `decider2/config/` package and no symbol named `ConfigManager`, and a lint rule
  forbids importing `json`, `os`, `pathlib`, `socket` or an HTTP client anywhere
  under `binding/` or `params/`.
- `compile/strategy.py` remains the expression-tier seam (§1); `runtime/plan.py`
  derives ordering from the graph so it can never be hand-maintained (doc 01 §5.2);
  `testing/equivalence.py` enforces the ladder in §3.1.

Three layers, as in `decider` today: core library → shared credit-granting
modules → client-supplied extensions.

**Still unhoused, deliberately:** nesting/grain (O5) has no package yet because it
has no agreed name — nine were invented across six projects. It is the most-invented
gap in the set and the next thing to design; when it lands it is a peer of
`interiors/`, not a corner of it.

---

## 7. Observability summary

| need | mechanism | cost |
|---|---|---|
| which branch fired, in production | `.emit("<branch>_path")` → one `int64` column, a compile-time immediate | effectively free |
| specific intermediate values in production | `.emit(...)` → extra output columns | ~10 µs/col, row-count-independent |
| step through one record | `stepped` (compiled steps, Python driver) | irrelevant at one record |
| inspect step internals | `interpreted` | irrelevant at one record |
| what-if override mid-flow | `stepped` / `interpreted` | irrelevant at one record |
| "what can affect z" | static lineage from the graph | zero — no execution |
| "step x changed y from 1 to 2" | value version chain + producer attribution | recorded in trace modes |
| pipeline latency/throughput | OTel spans at **stage** granularity | negligible |

**OTel measures the pipeline; emitted columns and traces explain the records.** Per-record
spans would be millions per batch and are never emitted — per-record diagnostics
travel as columns.

Diagnostics are declared as **data** (`pipeline.emit("term_cap")`), never as a code
pointer in config, and node identity is deterministic so path codes remain
comparable across versions (doc 01 §5.4).

---

## 8. Correctness

There is no migration and no permanent oracle, so correctness is
**specification-based**: rule-level assertions expressing intent, which are also
what a compliance reviewer can read. Golden-trace comparison ships as a
*capability* for regression baselines, not as the theory of correctness.

The framework's own test suite must include the §3.1 equivalence ladder.
