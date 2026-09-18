# 06 — Open questions and planned experiments

What is genuinely undecided, and the experiment that would settle each. The
principle: **anything that could invalidate a structural decision gets an
experiment before code, not after.**

---

## Open questions

### ~~O1 — Config resolution~~ — **SETTLED by E0**

The hybrid wins with no tradeoff. Full measurements in doc 01 §5.7; design in
doc 02 §2.1. Headline: the union is *not* faster than a registry (74.1 vs
73.5 µs); the real 1.7× win is native pydantic-core recursion, which the status
quo forfeits and cannot adopt without deferring the union build to a single
finalisation step. Three non-optional constraints came out of it — build the
union once, wrap `union_tag_invalid` with difflib suggestions, and set
`extra="forbid"` on params models.

### O2 — In-place param promotion
Can a literal become a tunable *where it sits* (`p("income_cap", 48.0)`), harvested
into the params model? The human-factors case is strong — whichever path is
cheaper is the path taken, and 546 inline literals versus zero config uses is the
evidence (doc 01 §5.3). The cost: a params model derived from source must stay
stable across edits and remain diffable for audit. Also unclear how a business
user browses knobs that live inside function bodies.

### O3 — The reviewable artefact
The weakest part of the design. Both existing representations fail
non-programmer readability (doc 01 §6). The proposal — a generated view of the
module data with descriptions, declared inputs/outputs, param bounds and version
chains — is untested. **This is a requirement, not a nice-to-have**, and if it
cannot be met the governance story collapses.

### O4 — Table/keyed-lookup interface
Sketched provisionally in doc 03 §4. Dense array + present mask works in numba
(the a large internal workload port does exactly that), but the *authoring* surface is unclear, as
is validation and whether tables are per-invocation like params or bound at build.

### O5 — Ragged per-record collections
Deprioritised as workload-specific, but real: a large internal workload enumerates subsets of a
variable-length candidate list per application. CSR offsets, polars `List`
columns, or leave it to the author with graceful degradation (the stated
preference). Deferred until a second workload needs it.

### ~~O14 — Does the calling convention survive width?~~ — **SETTLED by E9**

Raised because the driver passed one argument per column and realistic wide shapes
can need several hundred outputs. **No ceiling exists** — positional works to at least 1600
arguments, with compile time rather than an error as the failure mode.

But E9 changed the convention anyway: **a 1D structured (record) array per
side**, which is 2× better end-to-end, 37× per call, and the only option
carrying a f8/i8/bool mix with O(1) dispatch. Hybrid threshold ~64 columns per
side (interpolated). Target shape 400-in/633-out verified working. Doc 01 §4d.

Also produced the most valuable implementation finding so far: writing generated
drivers to **real files instead of `exec`** takes a 1200-argument driver from
29.5 s cold to **0.80 s warm**, turning compile cost into a build-time cost.

### O6 — `prange` dispatch threshold — **reframed by E7**
There is **no fixed row threshold.** The earlier ~50k figure was one synthetic
kernel; E7 shows the crossover is set by **per-row work**: a light body never wins
(0.99× at 1 M rows) while a body with a 64-iteration inner loop crosses at ~5k
and reaches 7.8× at 5 M.

So a constant is wrong. Options: measure both variants per kernel at warmup
(affordable, since processes are long-lived and both are compiled anyway), or
derive a heuristic from estimated body cost — which is the same estimation
problem as O12, and probably should share its solution.

### ~~O13 — Taps must name a version~~ — **SETTLED**
Default to the **final** version; qualify by **producing module name**
(`term_cap@sector_cap`); `term_cap@*` yields every version as its own column. Doc 03
§7.

Positional qualification (`term_cap@5`) was rejected: inserting a rule earlier
renumbers everything after it, so a tap silently repoints at a different rule's
output — the same fragility as unstable node ids (doc 01 §5.4). A module name is
stable under insertion and reads meaningfully in an audit report.

### ~~O7 — Fusion boundaries in practice~~ — **SETTLED by E5 + E7**

Fusion is non-monotone and harmful past ~5 modules (0.51× at 10, 0.33× at 20);
break-even is ~10k rows; the cause is register pressure defeating vectorisation.
**The guidance survives, the premise was wrong** — small modules are cheap
because boundary stores are near-free, not because they fuse. Cap fused groups
at ~6–9 steps. Doc 01 §4c.

Exonerated by explicit controls: per-module params bundles are not the cause (a
merged bundle degrades identically), and taps do not split kernels.

### ~~O8 — Nested control flow inside a fused kernel~~ — **SETTLED by E7**

Compiles, genuinely short-circuits (1071× on a skipped heavy arm, side-effect
probes confirm), `prange` composes with bounded inner loops bitwise-identically,
and the compile-time limit is **emitted code size (~15 ms/line), not nesting
depth** — fan-out is the wall (`arms^depth`; depth-10 full-binary = 92 s).

But it **overturned the fusion claim**, which is now O12. Full results doc 01 §4b.

### O12 — The fusion cost model
**New, from E7.** Fusion is non-monotone: profitable at one branch group
(1.33–1.61×), harmful past ~4 with cheap arms (0.17× at 16 modules), and
strongly profitable again when arms are expensive (1071× on a skipped heavy arm).
So the compiler must *decide*, and the decision needs a cost model over:

- branch-group count (vectorisation is lost at ≥4)
- estimated arm cost (cheap → split and vectorise; expensive → fuse and
  short-circuit)
- emitted code size (~500 lines ≈ 10 s compile is a reasonable cap)

Estimating arm cost statically is the hard part and may not be reliable.
Fallbacks: measure candidate groupings at warmup (affordable — long-lived
processes), or require authors to hint expensive arms.

Whatever the model, it must be **inspectable and overridable**
(`explain_kernels()`, `pin_kernel_boundary()`), or "why is this slow" becomes
unanswerable. Doc 02 §1.1.

### O9 — Stepping through control flow
`stepped` and `interpreted` advance one step at a time, which is well defined for
a sequence. Open: what "step" means at a `Branch` (step over it, or into the taken
arm?) and at a `Loop` (one iteration, or one step within an iteration?). This is a
debugger-UX question, but it shapes the trace data model, so it needs answering
before `observe/trace.py`.

### ~~O10 — How a compiled step observes a null~~ — **SETTLED**

**Plain `Optional[float]`** (`float | None`, checked with `if x is None`) for
steps that must distinguish, plus a **declared-fill** form (`missing_as(0.0)`)
that substitutes at extraction so most steps see a plain float and never mention
nulls. Doc 01 §4, doc 03 §1.

Note this **does not follow the fastest measurement.** E8 found the
`.value`/`.valid` NamedTuple at 413 µs/1 M versus `Optional` at 1907 µs — about
1.5 ns/row — and the NamedTuple was the original recommendation. Reversed
deliberately: `if x is None` introduces no new concept, and `Optional` is
*safer* because it makes the garbage slot unreachable rather than relying on the
author remembering to check. The data didn't change; the weighting did, in favour
of the stated readability requirement. The faster form stays measured and
available if a hot path ever needs it.

Three findings worth carrying forward:

- **An earlier claim in these docs was wrong.** The in-kernel bit test is *not*
  faster than NaN-checking; it is 3.3× slower on random masks due to branch
  misprediction, making it the floor rather than the ceiling.
- `types.Optional` is intrinsically expensive (1907 µs) — `None`-versus-value is
  an unavoidable control-flow join. `jitclass` is catastrophic (24,217 µs).
- A clean column has **`validity_buf is None`**, making the required-path gate
  essentially free (0.375 µs total, fully zero-copy).

### O11 — Schema propagation must model nullability
Doc 02 §5 requires frame operations to declare schema transforms so static
lineage survives. Nullability is the part that bites: `join(how="left")` makes
every right-side column nullable, `aggregate` changes cardinality, and a filter
can empty a frame. If propagation tracks names and types but not **nullability**,
a downstream `required` input silently receives nulls — and per doc 01 §4 the
value slot under a null holds *garbage, not zero*, so that is a wrong-answer bug
rather than a crash.

Needs deciding alongside O10, since the two together define the null contract end
to end: which columns *can* be null (schema) and how a step *sees* it (step API).

---

## Experiments

Ordered by how much a bad answer would cost.

### ~~E0 — Registry vs union~~ — **DONE**
Settled O1. See doc 01 §5.7. Also uncovered two defects in the status quo worth
carrying forward as regression tests: stale nested unions under per-registration
rebuilds, and the loss of the step index in nested validation errors.

### ~~E9 — Calling convention at width~~ — **DONE**
Settled O14. Record arrays per side; no positional ceiling; real-file codegen is
mandatory. Doc 01 §4d.

Four design constraints it produced: `readonly` vs writeable arrays are distinct
numba signatures; bool columns are not zero-copy from polars (~48 µs/col at
100k); `typed.List` indexed inside a row loop costs 7.6× unless hoisted; and
record write-back is the new dominant cost at high output counts.

It also confirmed, from the opposite direction, the conclusion the depth
benchmarks reached: at 200+ columns of *simple* arithmetic numba loses to
vectorised numpy (101 ms vs 36 ms). The row loop's advantage is branchy logic,
not throughput.

### E1 — The polars↔numba kernel interface
**The one to build first.** Doc 02 asserts the boundary shape; this makes it real
and settles what `compile/boundary/` and `compile/numba/` must look like.

Much of this is now pre-specified by E8 (nulls) and E9 (calling convention), so
it is closer to an assembly task than an open experiment:

1. Per-column zero-copy extraction, including the arrow values+validity path for
   a nullable column, with `rechunk` once at frame entry and `a.offset` respected
2. Record-array assembly per side, with the ~64-column hybrid threshold
3. Generated fused row-loop driver **written to a real `.py` file**, not `exec`
4. Batched write-back
5. Same kernel invoked via the single-record scalar path, no polars
6. A `required` null violation failing fast and naming the column

**Would invalidate:** the two-tier boundary, if real end-to-end cost is far above
the component figures. Less likely now — E9 already measured a 400-in/633-out
kernel end-to-end at 100k rows.

### E2 — Graph model: scopes, combinators and versioning
Settles the core of `graph/`. Prove: name-based wiring with topological sort
inside a module; the scope invariant enforced (duplicate output within a module
is a build error); overwrite at module boundaries expressing a waterfall via `|`;
`Branch` with per-arm scopes, declared `modifies`, pass-through of untouched
values, and build-time rejection of an arm that omits a declared output or
disagrees on its type; `Loop` with declared `carries` and a required bound;
version chain with exact producer attribution across all three combinators;
static lineage answering "what can affect z" without execution; and a diff of two
module versions producing a readable "step x changed y from 1 to 2".

**Would invalidate:** versioning-under-the-hood, if the chain reads as confusing
rather than clarifying; or the combinator model, if `Branch`/`Loop` turn out to
need machinery a plain Module cannot carry.

### E3 — The equivalence ladder
Settles whether the four-mode model is trustworthy. Implement `interpreted`,
`stepped` and `fused` for one module; assert three-way agreement; then
*deliberately inject drift* at each layer (a `fastmath` difference, an integer
overflow, a `log` ULP difference) and confirm the ladder localises it to the
right rung.

**Would invalidate:** `stepped` as a distinct mode, if it never catches anything
`interpreted` does not.

### E4 — The reviewable artefact
Settles **O3**, and is the highest-risk-of-silent-failure item. Render a
realistic waterfall module — ordered steps, descriptions, declared inputs and
outputs, params with bounds, value-version chain — and **put it in front of an
actual credit-risk or compliance reviewer** with a policy document, then see
whether they can verify it unaided. No amount of internal opinion substitutes.

### ~~E5 — Cross-module fusion~~ — **DONE**
Settled O7. Every composition mechanism in doc 03 §4 verified: N distinct params
NamedTuple types fuse fine (up to 80 modules / 84 args), `shared` threads to a
subset for free, taps don't split kernels, value versions are plain local
reassignment, and retuning never recompiles. Full results doc 01 §4c.

Two things it produced beyond its brief:

- **O13** — a tap named `"term_cap"` fires on the *first* version, so tap syntax
  needs version qualification.
- **A silent, permanent performance trap** — two distinct NamedTuple classes
  sharing a `__name__` *and* field names blow per-call dispatch from ~1 µs to
  15–24 µs forever, because the numba types print identically but compare
  unequal and the dispatcher's cache thrashes. It contaminated E5's own first
  run before being caught. The mandatory rule (build each params class once at
  module-definition time, never per pipeline build) and a regression test are
  recorded in doc 01 §4c.

### ~~E7 — Nested control flow in a fused kernel~~ — **DONE**
Settled O8; the combinator model survives. Full results doc 01 §4b.

The valuable part was the **negative** result: it falsified the claim that
adjacent modules always fuse profitably, which two documents were asserting as a
reason the small-module style is free. That claim was load-bearing on an
authoring recommendation, and it was wrong. It is now O12, and the architecture
gained a real separation as a result — authoring unit decoupled from compilation
unit (doc 02 §1.1).

Three secondary corrections it produced: `prange` has no fixed row threshold
(O6), `parallel=True` costs 1.2–2.6× compile rather than a flat 3×, and `break`
delivers only ~47% of its theoretical saving because the exit test roughly
doubles per-iteration cost.

### ~~E8 — Null mechanism for compiled steps~~ — **DONE**
Settled O10. All five candidates compile; the NamedTuple form wins on both axes
at once. See doc 01 §4.

It also produced two boundary rules that belong in `compile/boundary/`:

- **`rechunk()` does not reset `a.offset`** — a `slice(9,12)` still reports
  `offset=9`, and ignoring it produced a verifiably wrong validity window.
  Values slice `[offset:offset+n]`; validity must be unpacked *before* slicing,
  since bit *i* maps to element `offset+i`.
- **Rechunk once at frame entry, never per column** — 0.33 µs when already
  single-chunk, but 585 µs/1 M for two chunks.

And it corrected a fact previously recorded in doc 01 §4, which is a useful
reminder that a plausible mechanism ("bit tests are cheap") can be exactly
backwards once branch prediction is involved.

### E6 — Param ergonomics and composition semantics
Settles **O2**, and validates the composition rules in doc 03 §4.1–4.3, which are
designed but unexercised.

Part one: implement declared and in-place forms; write the same realistic module
both ways; compare edit cost to promote one literal, diff readability, and
whether the knob index remains discoverable.

Part two — the composition mechanics, all of which have a plausible failure mode:

- **Namespacing:** the same module instantiated twice with different params;
  confirm no collision and that each instance's steps see only their own bundle.
- **`shared` as a reserved parameter:** a step declaring `shared` receives the
  one pipeline-level bundle, passed by reference rather than copied. Verify the
  fused driver threads a single `shared` argument to exactly the steps that asked
  for it, and that a module whose required-fields contract isn't satisfied by the
  pipeline's `SharedParams` fails at composition with a message naming both the
  module and the missing field.
- **Module isolation:** a module using `shared` must remain testable alone by
  passing a stub bundle. Confirm nothing about composition is required to
  exercise one module.
- **`.bind()`:** a bound value must leave the caller-facing interface while
  remaining a runtime value — verify it does not become a compile-time constant
  and therefore does not trigger recompilation.

---

## Sequencing

**E0, E5, E7 and E8 all done.** Promoting them ahead of schedule paid off: two of
the four returned negative results that changed the design rather than confirming
it. E5+E7 falsified the fusion premise (the guidance survived, the stated reason
did not), and E8 corrected a performance claim already written into doc 01.

**E1 is next** — it gates the whole `compile/` subtree, and E8 has already
specified much of its null handling. E2 follows, since everything sits on the
graph model, then E3 (needs a graph to compile three ways). E4 should start early
*in parallel*, because it depends on a human reviewer's availability rather than
on code. E6's part two is now largely answered by E5, leaving only the param
ergonomics half.

Open and unscheduled: **O2** (in-place param promotion), **O3/E4** (the
reviewable artefact — still the weakest part of the design), **O4** (tables),
**O5** (ragged collections — deferred by preference), **O6** (`prange`
dispatch, now known to need a per-kernel measurement rather than a constant),
**O9** (stepping UX, informed by E7), **O11** (nullability in schema
propagation — decide alongside the settled O10), **O12** (the fusion cost
model), **O13** (tap version qualification).

Twelve of the original questions plus four new ones; five settled, all five by
measurement rather than argument.

A vertical slice through E1 + E2 + E3 — one realistic waterfall, declared once,
compiled three ways, applied to a batch and a single record, with a param retune
proving no recompile and a tap proving cheap diagnostics — is the first
implementation milestone. It needs no credit modules and no frame operations.
