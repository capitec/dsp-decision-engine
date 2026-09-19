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

### ~~O2 — In-place param promotion~~ — **SETTLED, in the signature not the body**

The question was whether a literal could become a tunable *where it sits*
(`p("income_cap", 48.0)` inline), harvested into the params model. The
human-factors case was always strong — whichever path is cheaper is the path
taken, and 546 inline literals against zero config uses is the evidence
(doc 01 §5.3).

**Answer: yes, but in the signature.** `cap: float = param(48.0, ge=6, le=60)`
(doc 03 §4.4). That takes both of O2's stated objections off the table rather
than accepting them:

- *"A params model derived from source must stay stable across edits and remain
  diffable."* A signature is a declaration; harvesting one is stable in a way
  harvesting statements is not.
- *"Unclear how a business user browses knobs that live inside function bodies."*
  They do not live in bodies. `params_schema()` finds them by inspection, and a
  reviewer reads them where they already read the null policy.

It also avoids an AST rewrite that would have made step bodies stop being plain
Python and forced `interpreted` mode to replicate the substitution — the exact
kind of divergence between modes the equivalence ladder exists to catch.

Together with bare functions as pipeline elements (doc 03 §5.3) and generated
config (doc 07 §5), this takes one policy rule from **six artefacts across four
files to one**. Doc 03 §1.1.

Residual, and it is small: two ways to declare a param (lint forbids both in one
module), direct-call substitution for tests, and confirming numba tolerates the
generated signature — E11.

### O3 — The reviewable artefact
The weakest part of the design. Both existing representations fail
non-programmer readability (doc 01 §6). The proposal — a generated view of the
module data with descriptions, declared inputs/outputs, param bounds and version
chains — is untested. **This is a requirement, not a nice-to-have**, and if it
cannot be met the governance story collapses.

### O4 — Table/keyed-lookup interface — **partly answered by doc 08**
Sketched provisionally in doc 03 §4. Dense array + present mask works in numba
(the port of a large internal workload does exactly that), but the *authoring*
surface is unclear, as is validation.

The build-vs-invocation half is now decided: **table contents are values** and
ride in the params document, free of recompilation, because a `decision_table`
and a `scorecard` are evaluated by a generic kernel rather than codegen
(doc 08 §3.4). What stays open is the authoring and validation surface, and how a
UI edits a table cell-wise.

**The cost of leaving it "provisional" is now measured.** Seven of eleven mock
projects needed a keyed lookup and spelled it **six different ways** — `Table`,
`table`, `dated_table`, `versioned_table`, `lookup_table`, `bound_table`, plus
`temporal_table`/`IntervalStore` ([COLD-READ.md](../example_projects/COLD-READ.md) §3.2). Tables are the second
most-invented gap after nesting, and the one business users most want to edit.

### O15 — The interior document schema
**New, from doc 08 §3.** The `when`/`then` sketch there is illustrative. The real
schema is `flat_rules`' closed algebra — `LeafRule`, `UnaryRule`, `CasesRule`
(ranges / string-match / is-in), `CompositeRule` — plus a required id, a required
description, and param references instead of inline literals. Needs writing
before `ruleset` is built, because it is simultaneously the UI contract, the
codegen input and the reviewable artefact's source.

Two things are already decided by doc 08 §1, and constrain it: a rule's leaves are
declared features or **registered** feature ids, never expression strings
(`_ComputedFeature` goes — §3.2); and no node may carry a `DefinedFunction`-style
`{module_name, function_name}` pointer (`output_fn` goes — doc 08 §1.1). What remains
open is the shape of the `then` side, how `first_match` versus `all`
prioritisation is expressed, and whether a rule carries an approval field (O17).

### ~~O16 — Does a `ruleset` compile fast enough to stage?~~ — **SETTLED, [EXPERIMENTS.md](EXPERIMENTS.md) §G**

**Yes, up to about 35 rules.** 3 rules = 0.63 s, 10 = 1.4 s, 30 = ~6 s, 40 = 9.4–10.9 s,
100 = 48–57 s. **A UI can honestly promise "live in ≤10 seconds" below ~37 rules
/ ~600 emitted lines.** Execution is flat at 1.8 ms per 100k rows for `first_match`
regardless of rule count.

Two consequences the question did not anticipate:

- **The lever is splitting, not shrinking.** Compile is super-linear *within* a
  unit (∝ lines^1.4, local exponent 1.96 at 60→100 rules), so four units of 25
  rules cost ~18 s serial or ~4.5 s across four workers, against 48–57 s for one
  unit of 100. Splitting a large `ruleset` into several compilation units is a
  framework decision, not an authoring one.
- **Disabled rules cost full compile time.** With every leaf disabled LLVM proves
  the body dead and execution collapses to 0.031 ms, but compile is unchanged
  (21.12 s vs 21.28 s at 60 rules). So O15's interior schema needs `enabled` to
  mean **not emitted**, not "emitted and skipped" — otherwise a user who turns off
  fifty rules pays for them on every activation, forever.

The generic-kernel boundary in doc 08 §3.4 does **not** have to move.

### O19 — Interface freezing and the vocabulary map
**New, from doc 03 §5.1–5.2.** A module's interface is inferred from its steps and
materialised into the module data — settled, because it costs no ceremony and is
what makes an edit checkable without executing. Two parts are not settled:

- **What exactly a frozen contract covers.** External vs wired inputs, per-input
  null policy, output types, params schema, required `shared` fields, tap names —
  and whether a step *body* change with an unchanged interface is breaking. It
  can be, via numerics; the contract cannot see it.
- **Whether the vocabulary map earns its place.** It exists because a project with
  hundreds of values cannot carry a relabel at every call site, and because a
  frame-boundary rename cannot address values produced mid-pipeline. But it is a
  second place where a name can come from, which cuts against "one canonical
  location". Measure it on a real project's naming spread before committing.

### O20 — Does shadowing-as-an-error fire too often?
**New, from doc 03 §2.1.** A module output shadowing an input column is a build
error, because that is the silent-rebind bug. The rule is deliberately narrow —
overwrite *between* modules stays silent, since that is the waterfall. The risk is
that a real pipeline trips it constantly and authors learn to qualify reflexively,
which would make the error noise rather than signal. Count the occurrences on a
realistic pipeline before shipping it as an error rather than a warning.

### O21 — Change class per FIELD, not per document
**New, from [COLD-READ.md](../example_projects/COLD-READ.md) §6 Tier 1.** Doc 08 §2's three classes classify *documents*.
The maintainer persona showed the work needs them to classify **fields**: a rule's
`applies_to.segments` decides *who a control applies to* — at least as
governance-sensitive as a threshold — and it falls through the gap, appearing in
neither `shape_projection` nor the dated params entries. The framework should
**refuse a rule document containing a field with no declared change class.**
Corroborated independently by five sketches in `examples/FINDINGS.md` §2.1.

### O22 — Lineage must descend into interiors
**New, from [COLD-READ.md](../example_projects/COLD-READ.md) §1.1.** Four of six traces died at exactly the same
architectural joint: `pipeline.lineage("action_code")` stops at the `ruleset(...)`
box. Doc 04 §3 promises lineage answers questions "without running anything"; doc
08 §3 puts the rules a business user edits *inside* that box. The two meet at a
wall. Every interior kind needs to supply `lineage()` and `render()` over its rows,
or the governance promise stops exactly where the governed content lives.

### O23 — An unbound name is invisible to every check that currently runs
**New, from [COLD-READ.md](../example_projects/COLD-READ.md) §1.1.** Five of six readers found a name with no producer
anywhere in the tree — `overlay_scope_restriction_bits`, `rate_card.Lookup`,
`FallbackGrade`, `ExecutionPackage`, `shipped_offer` — and **every one passed
`ast.parse`**. Doc 03 §2.2 already treats an unbound *step parameter* as a typo;
the evidence says that check must extend to `Branch` arms, `writes=` targets,
`isolates(may_read=…)` names, interior file references and cross-phase seed
strings. Name-based wiring (doc 03 §2) is the design's best asset precisely
because names are greppable — and its failure mode is that an unbound name looks
exactly like a bound one.

### O17 — Approval granularity
**New.** Does activating a staged generation need per-rule approval, or is
document-level enough? Framework-neutral — but it decides whether a rule carries
an approval field, which is a schema question (O15).

### O18 — Interiors in a `sealed` deployment
**New, from doc 08 §4.1.** Can a `sealed` deployment consume an interior document
at *build* time only? Probably yes and probably the common case: UI-authored
rules, compiled in CI, shipped in the image, with no compiler in production. If
so it should be the default and `live` the exception.

### O5 — Ragged per-record collections — **UNDEPRIORITISED: the most-invented gap**

Previously deferred "until a second workload needs it". Eleven independent mock
projects needed it, and invented **nine different names** for it — `grain`,
`Gather`, `Each`, `Enumerate`, `Map`, `Collection`, `Fanout`, `Explode`, `Cross` —
across **six** projects ([COLD-READ.md](../example_projects/COLD-READ.md) §3.2). It is the single most-invented gap in
the set by a wide margin, and a second study using an entirely different method
(`examples/FINDINGS.md`) found `grain` independently.

A business of entities, each with nested adverse events; a client's candidate
consolidations; a campaign's tree nodes — nesting is not workload-specific, it is
the shape of the domain. **Deferring it did not avoid the cost; it distributed
it.** Give it one name and a lineage story before anything is built on top of it.

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

### ~~O6 — `prange` dispatch threshold~~ — **SETTLED by measurement, [EXPERIMENTS.md](EXPERIMENTS.md) §F**

Both the numbers and the proposed remedy were wrong.

**Resolution: there is no automatic rule, because `prange` is authored.** Serial
by default; `parallel(...)` opts in, mirroring `fuse(...)` (doc 05 §5.1).

The measurements still settled the question, by ruling out every automatic option:

- A fixed **row** threshold scores 66.7% — confirming doc 01's rejection of "~50k".
- A **warmup measurement** is structurally impossible: the crossover depends on
  batch size, which warmup does not know. A probe at n ≤ 10k predicting n ≥ 100k
  was right 52.8% overall, 0% for medium bodies.
- A **dispatch-time rule** (serial wall-clock > ~90 µs) scores 97.9% out of
  sample — but only on uniform synthetic bodies. Credit logic short-circuits, so
  per-row work is a distribution and `prange`'s static schedule load-imbalances
  against it. Good enough to advise, not to decide silently.

So the rule survives as a **diagnostic** in `explain_kernels()`, not a switch.

**The warmup-measurement plan cannot work**, structurally: the crossover is set by
total serial wall-clock = per-row work × row count, and warmup knows only the
first factor. A probe at n ≤ 10k predicting n ≥ 100k was right **52.8%** of the
time overall and **0%** for medium bodies.

~90 µs tracks the fork/join floor (measured 65–67 µs) and is machine-specific, so
it is a **startup calibration**, recorded in the audit record with the chosen
variant (doc 08 §8) — which also makes the choice reproducible.

**O6 no longer shares O12's problem.** It needs one scalar per kernel
(`ns_per_row`), not a body-cost model.

Three refuted figures are recorded in doc 01 §4, including a probable row-count
transcription error: the doc's "7.8× at 5 M" matches the measured **100k** figure
(7.5–7.7×), while 5 M measures 11.4–12.0×.

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
because boundary stores are near-free, not because they fuse. Doc 01 §4c.

**The ~6–9 step cap that was written here has been withdrawn.** It sat inside the
region the same table measures as 10–28% worse than not fusing, and no constant
can span a decision ranging from 0.11× to 1071×. Fusion is now explicit:
one kernel per module for `apply()`, maximal fusion for `score()`, and a
`fuse(...)` combinator for anything else. Doc 02 §1.2.

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

**Out of scope for v1.** Doc 02 §1.2 removes the need for a model by making
fusion explicit: `apply()` emits one kernel per module, `score()` fuses
maximally, and a `fuse(...)` combinator is how an author asks for more. Nothing
is inferred, so nothing needs explaining. O12 is now the question of whether an
automatic model would ever beat an explicit annotation by enough to be worth the
unpredictability — and the honest answer may be no, given the effect is worth
~16 ms against a 35–158 ms write-back cost.

If it is revisited, the useful lead is that the mechanism is **directly
observable** rather than needing estimation: vector IR values drop to 0 at M≥4,
readable from `.inspect_llvm()`. That turns a static cost estimate into a
build-time measurement.

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

### E4 — The reviewable artefact — **now has a starting point and a second task**
Settles **O3**, and is the highest-risk-of-silent-failure item. Put a rendered
artefact in front of an **actual credit-risk or compliance reviewer** with a policy
document and see whether they can verify it unaided. No internal opinion substitutes.

**Two changes from the cold-read study ([COLD-READ.md](../example_projects/COLD-READ.md) §5.1).**

**Start from `examples/01-transaction-fraud-interdiction/artefacts/rule-sheet-MS-0208.md`.**
It is the best artefact produced anywhere in that exercise and is closer to solved
than doc 04 §6 assumes: it renders authored value *and* in-force value side by side
with the reason they differ, names the overlay, its approvers and its expiry,
distinguishes a threshold change from a shape change with different approval
routes, states per-feature missing-input behaviour, and is **sized for 635 rules**
— a one-page sheet per rule with a table as the index, rather than a waterfall
diagram that does not scale.

**Add a second task: ask the reviewer to check a decision record *against* the
sheet.** That is the test that just failed. The same project's
`decision-record.example.json` claims an overlay caused a rule to fire; checking
all four predicates against all three threshold vintages by hand shows the rule
fires identically either way, and the file's own prose contradicts its own field.
Reading the sheet is necessary; **reconciling it against an actual decision is what
governance requires**, and nothing in the current brief tests it.

> **The starkest number in the study: 9 of 11 independent designers, each given a
> spec demanding reviewability and told to design the surface they wished existed,
> produced no reviewer-facing artefact at all.** Treat O3 as *unstarted* rather
> than as weak — non-production is invisible to a satisfaction score and obvious to
> a reader who goes looking for it.

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
Settled O10. All five candidates compile. The `.value`/`.valid` NamedTuple is the
fastest explicit form, and **the recommendation was reversed against it** in
favour of plain `Optional` on readability and safety grounds — see O10 above and
doc 01 §4 for the reasoning. Do not read this entry as endorsing the NamedTuple.

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

### ~~E10 — Configuration lifecycle~~ — **DONE (as H and K)**

Doc 08 §4's staged lifecycle is confirmed as a mechanism, with one change: the
background worker must be a **subprocess**, not a thread. Measured, [EXPERIMENTS.md](EXPERIMENTS.md) §H, §K:

| | thread | subprocess |
|---|---|---|
| serving throughput retained | 26–55% | **97.9%** |
| parent-side compile events | — | **0** |

Atomic swap: **0 straddled batches** of 1564 across 11,605 swaps, with a
deliberately-wrong per-chunk control straddling 99.87%. `activate()` 0.177 µs,
rollback 3.36 µs and zero compiles, three generations resident for +2.4 MB.

**Config change to serving: 2.56 s (10 rules), 7.34 s (30 rules).**

Two new cache requirements came out of it (doc 05 §4): import the generated driver
**by module name** — `spec_from_file_location` loses the cache across processes —
and derive the **`sys.modules` registration name** identically in child and parent,
a seventh condition whose violation is a cryptic `ModuleNotFoundError('<dynamic>')`.

### ~~E10b — the original brief, superseded above~~
**New, from doc 08 §9.** Two to three days on top of the E1+E2+E3 vertical slice.
Settles the staged-compile lifecycle and answers **O16**.

One `ruleset` with three rules over the vertical slice's waterfall. Then: swap a
params bundle mid-batch and assert no record straddles two bundles; add a fourth
rule via an interior document and assert `stage` compiles in a worker while the
active generation keeps serving, `activate` is atomic and `rollback` needs no
compile; measure stage-to-active latency at 3, 10 and 30 rules; assert a failed
compile leaves the active generation untouched; and confirm a params document
carrying a `steps` key is rejected.

**Would invalidate:** doc 08 §3, if declaring `reads`/`writes` in code constrains
real rule sets unacceptably; or doc 08 §4, if staging a realistic `ruleset` is too
slow for a UI to promise a bounded wait — in which case doc 08 §1.3's rejection of
an interpreter gets revisited with a measurement instead of an argument.

### E11 — `param()` in the signature survives the compiler
**New, from doc 03 §4.4.** Small and gating, because doc 03 §1.1's whole
one-artefact story rests on it.

1. A step whose signature carries `param()` defaults compiles under `njit` once
   the driver passes params explicitly (the emitted step should carry no default
   at all — confirm that is what happens rather than assuming it).
2. Harvesting the signature produces a model identical to the hand-written one:
   same fields, same validators, same NamedTuple type, same namespace.
3. Retuning leaves `driver.signatures` at length 1; changing a declared param's
   *type* adds one (negative control).
4. A direct Python call to the step works without a params bundle, and a test can
   override with `params=`.
5. The same-name NamedTuple trap (doc 01 §4c) is not reintroduced by generating a
   model per function — per-call dispatch stays near 1 µs.

**Would invalidate:** doc 03 §4.4, and with it §1.1's artefact count, if numba
rejects the generated signature or if per-function model generation trips the
dispatch trap.

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

### ~~J — Output write-back convention~~ — **DONE**
Record output is dominated on both axes by dtype-grouped 2D arrays. Write-back is
**64.0%** of batch total (E9 said 54.7%). Column-major reaches polars zero-copy and
wins 1.74× unchunked; a row-major control proves the win is *column-major*
specifically. Both 2D forms compile **42× faster**. At N=1 the ranking inverts, so
doc 05 §3.1 chooses layout per entry point. [EXPERIMENTS.md](EXPERIMENTS.md) §J.

### ~~L — Rule thresholds as arguments~~ — **DONE**
Adopt, but not for the stated reason. Hoisting constants does **not** reduce
emitted lines or compile time — that claim is withdrawn. The case is that a retune
never recompiles (0 events vs 343 ms each at five rules), for 1–4.5 ns/row. Rule
enablement as a mask array fixes §G's disabled-rule compile tax. [EXPERIMENTS.md](EXPERIMENTS.md) §L.

### ~~J2 — Chunked write-back~~ — **DONE**
Chunking is **mandatory**: 1 M rows at 400-in/633-out needs 16.2 GB (record) or
11.9 GB (column-major). Default chunk **100k rows**. The 1.74× becomes
**1.03–1.57×** end-to-end once per-chunk assembly and persistence are counted.
[EXPERIMENTS.md](EXPERIMENTS.md) §J2.

### ~~N1 — the single-record overhead budget~~ — **DONE**
**Partial refutation of "stop optimising, it's negligible."** As doc 03 §4 /
doc 05 §3's own conventions most naturally imply implementing it (kwargs → pydantic
NamedTuple params → 1-row record marshalled/read back one named field at a time),
total framework cost is **971 µs p50 / 1048 µs p99 — 4.9–5.2% of a 20 ms budget**,
a whole millisecond. But 92% of it is two per-field Python loops (marshal 220 µs,
readback 673 µs), not the kernel — the isolated njit call is **1.38 µs**, matching
this doc's own §6.1 figure exactly. Writing both loops as one bulk call each
(same record type, assert-verified identical output) cuts the total to **145 µs /
0.73% of budget**, 6.7× less, and is not less readable. **Doc 05 §3 should specify
whole-row marshal/readback, not per-field.** [EXPERIMENTS.md](EXPERIMENTS.md) §N1.

### ~~N2 — the `score()` calling convention at width~~ — **DONE**
**Refutes doc 02 §3.5's literal kwargs example, generalized to 400 inputs — not
the example's 3-arg illustration.** kwargs, called exactly as shown (400 named
parameters, keyword call), costs **1190 µs p50 at width 400 — 5.95% of a 20 ms
budget**, bigger than N1's *entire* measured overhead for everything else
combined. Isolated: it is CPython's keyword-argument **binding**, not dict
construction (body=`pass` costs 1094 µs; the same signature called positionally
costs 28.8 µs — 39× less, for strictly more work) — and it scales close to
quadratically with width, confirmed independently via `timeit`. Every
alternative (dict 60.1 µs, reused record 39.8 µs, positional 93.3 µs) is
12.7–29.8× cheaper. **Recommendation: `score(request: dict, *, params)` as the
primary convention**, kwargs kept as syntax for small hand-written calls, a
reused record offered as an explicit opt-in fast path.
[EXPERIMENTS.md](EXPERIMENTS.md) §N2.

### ~~N3 — Params validation per request~~ — **DONE**
**Affordable as written — settled on performance grounds.**
`resolve_params(doc, origin=..., complete=True)` (doc 08 §6.2) costs 256.5 µs at
50 module instances — 1.28% of a 20 ms budget, under the whole-millisecond flag
— so doc 02 §4's "params may arrive per invocation" does not need restricting on
performance grounds; that question now rests entirely on doc 04 §2.1's
governance argument. The model→`NamedTuple` conversion (doc 03 §4), not
validation, turned out to be the larger and more cacheable of the two costs
(2.2× validation's cost at M=50; 726–765× cheaper when memoized by content).
`ParamsCell.get()`/`.swap()` (doc 08 §4) confirmed at genuine N=1, same order of
magnitude as the batch-context figures. [EXPERIMENTS.md](EXPERIMENTS.md) §N3.

### ~~N4 — Tail latency, concurrency, config-swap impact~~ — **DONE**
**Partial — swap impact confirmed negligible; a new `nogil` requirement surfaced
that doc 08 §4 did not state.** p50/p95/p99/p99.9/max measured for single-thread
steady state, GC on/off/frozen, 1–16-thread concurrency at `nogil=True` vs
`nogil=False`, and repeated config-generation swaps under continuous traffic.
GC on vs off/frozen showed no measurable tail difference (refutes the
GC-drives-the-tail hypothesis at this allocation shape). The config swap itself
is confirmed cheap for serving (worst call in 758k calls across 30 swaps: 1.37%
of a 20 ms budget; first-call-after-swap: 2.1× steady median but only 0.074% of
budget in absolute terms). **The headline is concurrency: `nogil=False` kernels
serving concurrent single-record requests hit a convoy effect that blows the
tail to 12.7× the entire 20 ms budget at 16 threads (p99), while `nogil=True`
stays flat at 4–12% of budget from 1 to 16 threads.** This does not contradict
doc 08 §4's existing `nogil` table (§H) — that table measured serving against a
background *compile* thread, a scenario the subprocess fix has since removed
from production — but doc 08 §4 currently reads as a general anti-`nogil`
recommendation, which this shows is only true in the compile-interference case
and is backwards for concurrent request serving. [EXPERIMENTS.md](EXPERIMENTS.md) §N4.

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

**E10 rides on the same vertical slice** and should follow E3 immediately, because
doc 08's lifecycle is what three of the four stated product goals depend on —
UI-driven configuration, pluggable config sources and versioned config all resolve
to "stage, activate, roll back" plus a JSON Schema export.

**Promoted: O2 (in-place param promotion) is no longer unscheduled.** With config
sourcing out of the framework (doc 08 §6), authoring cost is the *only* remaining
defence against repeating the outcome in doc 01 §5.3 — 546 inline literals against
zero config uses, because the tunable form was more expensive to write than the
literal. Under the style doc 03 §3.2 recommends, one policy rule costs a function,
a `module(...)` call, an instance name, a params model, a pipeline entry and a
config entry. E6's param-ergonomics half is the measurement that says whether that
is survivable.

Open and unscheduled: **O3/E4** (the reviewable artefact — still the weakest part
of the design), **O4** (tables — build-vs-invocation now answered by doc 08 §3.4,
authoring surface still open), **O5** (ragged collections — deferred by
preference, but note doc 04 §4.1's reason-code taxonomy needs it), **O6**
(`prange` dispatch, now known to need a per-kernel measurement rather than a
constant), **O9** (stepping UX, informed by E7), **O11** (nullability in schema
propagation — and `build` cannot construct a signature without it, so it gates E1
rather than merely accompanying O10), **O12** (the fusion cost model), **O13**
(tap version qualification), **O15–O18** (the interior document schema, ruleset
compile latency, approval granularity, and interiors in a sealed deployment).

Twelve of the original questions plus eight new ones; five settled, all five by
measurement rather than argument.

A vertical slice through E1 + E2 + E3 — one realistic waterfall, declared once,
compiled three ways, applied to a batch and a single record, with a param retune
proving no recompile and a tap proving cheap diagnostics — is the first
implementation milestone. It needs no credit modules and no frame operations.

**Three additions to that milestone**, each the smallest thing that would falsify
a decision made without evidence:

- **A second project reusing the same module under a different value
  vocabulary** — settling O19. Compose the waterfall twice: once where names
  match, once through a `Vocabulary`. Count how many relabels a realistic naming
  spread actually needs; if it is more than a handful, the three-layer design in
  doc 03 §5.2 has not solved the problem it exists for.
- **Shadowing counted, not assumed** — settling O20. Build a realistic pipeline
  and count how often a module output shadows an input column. If it is frequent,
  doc 03 §2.1's build error is noise and has to become a warning plus a lint.
- **E10's rule-set stage-and-activate**, which answers O16 and decides whether a
  configuration UI can promise a bounded wait.

All three are cheap, and all three are about **guardrails firing at the right
rate** — which is the property that decides whether the framework guides people
toward good practice or trains them to route around it. That is the same failure
doc 01 §5.3 records: a mechanism that costs more than the shortcut is not used.
