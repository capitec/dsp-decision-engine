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
cost at high output counts (54.7% of total at 633 outputs), so that is the next
optimisation target rather than the kernel itself.

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
(one merged bundle degrades identically), and a tap does *not* split a kernel
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

| entry point | default | why |
|---|---|---|
| `apply()` — batch | **never fuse across module boundaries** | the predictable side: split cost is flat *within* a body-cost class, and a wrong split is recoverable with `fuse()` while a wrong fusion is not visible at all |
| `score()` — single record | **fuse maximally** | N=1 is far below the ~10k break-even, where fusion wins purely by amortising per-call overhead (1.28–1.36× already at 1k rows) and `score()` has no polars boundary at all |

Both variants are compiled at build (§3.4) regardless, so carrying two costs
nothing new.

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
runs ≈15 ms/line and a fully-branching depth-10 nest costs 92 s (doc 01 §4b). A
`fuse()` group whose emitted body exceeds ~500 lines is a build-time error naming
the group, not a silent 90-second compile.

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
declared inputs and outputs, params schema, taps. Authoring ergonomics are
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

### 3.1 Four modes, one definition

| mode | steps | driver | use |
|---|---|---|---|
| `fused` | njit, inlined | njit | production, batch and realtime |
| `fused` + taps | njit, inlined | njit | always-on production diagnostics |
| `stepped` | **njit (real compiled code)** | Python, one step at a time | driver-level step-through; production numerics |
| `interpreted` | Python | Python | full internals inspection; reference semantics |

All four are generated from the same declared graph, so they cannot drift
semantically.

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
to maintain — one mechanism, two purposes. It additionally provides batch-wide
full tracing if ever wanted, at the measured ~4× materialisation cost.

### 3.2 Graceful degradation, per node

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

Nothing requires the author to write numba-friendly code. Two independent layers:

1. **Per node.** Try the jitted version; the first time numba cannot compile it
   for the given argument types, permanently fall back to plain Python for that
   node only. Every other node is unaffected.
2. **Per pipeline.** Try to compile the whole fused driver. If any node is not
   nopython-compatible the fusion attempt fails as a whole; fall back to a Python
   driver calling each node through its own best-available form.

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

### 3.5 Two entry points, one kernel

```python
Affordability.apply(frame, params=p)      # batch: polars in, polars out
Affordability.score(net_income=…, params=p)   # realtime: scalars, no polars
```

The realtime path **bypasses polars entirely**. A single-record request is deep
inside the regime where marshalling costs more than the work (viability floor
~10k rows), and polars' per-call dispatch floor alone would exceed the request
budget.

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

```
decider2/
  graph/         values.py    # value identity, boundary versioning, overwrite chains
                 step.py      # a Step: declared inputs, output, params, taps
                 module.py    # Module = pydantic INSTANCE describing the graph
                 combinators.py  # | (sequence), Branch (n-way), Loop (bounded)
                 schema.py    # declared in/out schemas and propagation
                 lineage.py   # static "what can affect z" queries
  params/        model.py     # pydantic model -> NamedTuple, validation
                 tables.py    # keyed lookup tables as a declared kind
  compile/       strategy.py  # THE SEAM: a strategy compiles a node
                 numba/       kernel.py, fallback.py, variants.py
                 boundary/    extract.py (zero-copy, arrow validity), writeback.py
  frame/         join.py, aggregate.py, filter.py, opaque.py
  observe/       taps.py, trace.py, audit.py, otel.py
  runtime/       invoke.py    # batch apply + single-record scalar path
                 plan.py      # ordering DERIVED from the graph, never declared
                 lifecycle.py # generations: stage -> compile -> activate -> rollback (doc 08 §4)
  binding/       register.py  # generate a config model, stage it
                 finalise.py  # assemble the union ONCE; after this the union is the index
                 admit.py     # what a config document may contain (doc 08 §7)
                 fingerprint.py  # content hash of structure + interiors
                 errors.py    # wrap union_tag_invalid with difflib suggestions
  testing/       assertions.py, equivalence.py, golden.py, impact.py

decider2_credit/     scorecard/, tree/, rule_table/, waterfall/, affordability/
<client extensions>  same surface, registers into registry
```

Five placements are deliberate:

- `compile/strategy.py` is the expression-tier seam (§1).
- `runtime/plan.py` derives ordering from the graph, so it can never be
  hand-maintained in JSON or asserted in a comment (doc 01 §5.2).
- `testing/equivalence.py` enforces the ladder in §3.1 — what makes debugging in
  `stepped` or `interpreted` trustworthy.
- **`binding/`, not `config/`.** This package validates and binds *documents it is
  handed*; it never fetches one. The name matters because `decider/config/`
  accreted 553 lines of storage, semver and polling machinery once already
  (doc 08 §6). There must be no `decider2/config/` package and no symbol named
  `ConfigManager`, and a lint rule forbids importing `json`, `os`, `pathlib`,
  `socket` or an HTTP client anywhere under `binding/` or `params/`.
- `runtime/lifecycle.py` is where a new pipeline generation is compiled off the
  request path and swapped atomically — the mechanism that lets config change
  structure without putting a compile in a request (doc 08 §4).

Three layers, as in `decider` today: core library → shared credit-granting
modules → client-supplied extensions.

---

## 7. Observability summary

| need | mechanism | cost |
|---|---|---|
| which branch fired, in production | tap → one `int64` column; branch code is a compile-time immediate | effectively free |
| specific intermediate values in production | declared taps → extra output columns | ~10 µs/col, row-count-independent |
| step through one record | `stepped` (compiled steps, Python driver) | irrelevant at one record |
| inspect step internals | `interpreted` | irrelevant at one record |
| what-if override mid-flow | `stepped` / `interpreted` | irrelevant at one record |
| "what can affect z" | static lineage from the graph | zero — no execution |
| "step x changed y from 1 to 2" | value version chain + producer attribution | recorded in trace modes |
| pipeline latency/throughput | OTel spans at **stage** granularity | negligible |

**OTel measures the pipeline; taps and traces explain the records.** Per-record
spans would be millions per batch and are never emitted — per-record diagnostics
travel as columns.

Diagnostics are declared as **data** (`taps=["term_cap"]`), never as a code
pointer in config, and node identity is deterministic so path codes remain
comparable across versions (doc 01 §5.4).

---

## 8. Correctness

There is no migration and no permanent oracle, so correctness is
**specification-based**: rule-level assertions expressing intent, which are also
what a compliance reviewer can read. Golden-trace comparison ships as a
*capability* for regression baselines, not as the theory of correctness.

The framework's own test suite must include the §3.1 equivalence ladder.
