# FRAMEWORK-DEMANDS — 10 Retail credit, end to end

Twenty-six numbered demands this project makes of `decider2`. Each names the
spec section that forced it and is marked:

- **[SATISFIED]** — doc 03 (or 02/04/07/08) already does this. Four of twenty-six.
- **[EXTENSION]** — the shape is right, something is missing. Fourteen.
- **[UGLY]** — doc 03/07's answer would produce something I would refuse to
  ship at this size. What I did instead is stated. Eight.

The scale demands (A) come first, per instruction, because they are the
reason this document is longer than 03's or 07's own: every other project in
this set stresses ONE mechanism against ONE difficulty. This one stresses
twelve mechanisms simultaneously, in one deployed artefact, and the
interaction between them — not any one of them alone — is what breaks doc 07's
proposed layout. Read this alongside [`README.md`](README.md), which is the
walkthrough; this is the ledger.

---

## A. Scale — the reason this document is longer than the others'

### 1. Eight entry points sharing most of a flow but not all of it, without eight near-copies. [EXTENSION]

**Forced by:** §4.1, §4.1.1, §5.20, Q6.

**Doc 03's position:** none, directly. Doc 03 is a single-pipeline authoring
API; it has no notion of "entry point," "phase set," or "applicability" at
all — `pipeline = A | B | C` composes ONE graph for ONE caller. Doc 07 §4's
`decider run pipelines/term_loan.py` likewise assumes one pipeline file per
deployable thing.

**What doc 03's shape would produce:** eight pipeline files, each a `|`
sequence over the phases that entry point applies to, OR one pipeline file
with `if entry_point_code == N` scattered through eighteen phases. Spec
4.1.1 names the outcome of the first directly ("eight copies would guarantee
that within two years the tax table is correct in six of them") and O-06 in
`ordering.py` names the outcome of the second (a per-entry-point ordering
inversion expressed as a literal `if` cannot be checked for completeness).

**What this project did:** `flow.py` is ONE graph. Applicability is DERIVED
(`entrypoints/manifest.py`'s `Derived` rule: a phase runs on an entry point
iff every input in its declared interface is satisfiable from that entry
point's schema plus upstream outputs) and DECLARED only where it is a policy
decision (`DECLARED_APPLICABILITY`, 19 entries, each with an owner and an
approval). The build emits eleven specialisations — one compiled kernel set
per `phase_set_id` — from the one graph. Entry point 7's compiled variant
literally does not contain P02, P04, P05, P06, P07, P08, P10, P13, P14, P15,
P16: "whatever makes entry point 1 work must be absent here — and absent by
construction, not by a conditional inside every phase" (`ep07_quotation.toml`).

**Cost, stated:** this requires `EntryPoint`, `Derived`, `Declared`,
`PhaseSet` (`decider2.entrypoint`, imported in `entrypoints/manifest.py`) —
none of which exist in docs 02–08. Derivation also requires the build to
compute, per phase, a satisfiability closure over eight different input
schemas — an eleven-way compile at image build (flow.py's closing comment),
which is the direct, accepted cost of not compiling per-request.

### 2. Eighteen phases and twelve owning teams in one artefact — what makes team A deployable without team B. [EXTENSION]

**Forced by:** §5.26, §5.26.5, §3.1, Q1, Q3, Q4.

**Doc 04's position:** one boundary — params vs structure — with two owners
(business user, engineer). Doc 08 refines this to three change classes
(values / interiors / skeleton, §2). Neither has a notion of TEAM as a
first-class boundary, and neither has a notion of PHASE as a unit distinct
from module: doc 07's layout is `modules/` (unit of reuse and audit) and
`pipelines/` (composition) — flat, with ownership implied by which directory
a file sits in.

**What doc 07's layout would produce:** `phases/p09_policy_gates/` as a
directory suggests a phase has one owner, because doc 07 has no mechanism for
a directory to have five. Spec §5.10 states P09's 196 decision points split
five ways (T4 89, three product teams 41, Compliance 34, Decision Platform
21, Financial Crime 11) — a fact no directory listing can carry. Worse: if
ownership were a path segment (`phases/p09_policy_gates/t4_credit_risk/...`),
§11 scenario 4 (T4 splits into two teams) and scenario 5 (T7/T8 merge) become
repository-wide file moves, and every archived reviewable rendering stops
resolving — "a re-org would be indistinguishable, in the record, from a logic
change" (`OWNERS.toml`'s own docstring).

**What this project did:** ownership is DATA (`OWNERS.toml`), carried as an
`owners=` tuple on the phase envelope (`phases/__init__.py`) and — at
decision-point granularity — checked against the built registry. The
directory tree (`phases/p09_policy_gates/register.py`) is organised by PHASE,
because phase is the unit of ordering, budget, degradation and blast radius;
ownership crosses it thirteen times out of eighteen and is tracked
separately. `.github/CODEOWNERS` is GENERATED from `OWNERS.toml` plus the
registry (see its own header comment), so file-level review routing follows
data, never a directory a re-org would have to rename.

Spec 5.26.5's five deployability conditions are checkable exactly to the
extent the mechanisms above exist: condition 1 (blast radius computable and
disjoint) needs `navigability/index.py`; condition 2 (certification derived
from blast radius, not curated) needs `tests/golden/manifest.toml`'s
stratification to be blast-radius-addressable, which doc 03 §11's
`assert_modes_agree(pipeline, corpus)` — a whole-pipeline call — does not
give for free. **This is the one condition doc 03 does not currently satisfy
even in principle**, because its testing surface has no notion of "run only
the tests whose stratum intersects phases {P12, P13}."

### 3. 41 shared intermediates, one with up to 251 simultaneous versions inside a single consolidation execution, plus a third unadjusted axis. [UGLY]

**Forced by:** §5.21, §5.21.1, Q11.

**Doc 03's position:** one axis. §3.3's version chain is scope-boundary
versioning — `term_cap: 60.0 (Seed) -> 48.0 (IncomeCap) -> 36.0 (SectorCap)`
— and a tap qualifies by producing module (`term_cap@sector_cap`). This is
exactly right for a waterfall and has no notion of a SECOND simultaneous
version of the same name, because doc 03's model is that a name has one
current value per point in the pipeline.

**What doc 03's shape would produce:** the "handles two axes and not the
third" failure spec 5.21.1 names directly. A naming convention —
`existing_obligations_hypothetical_s37_unadjusted` — reproduces doc 01's own
documented failure mode (naming conventions invented under pressure) and
still cannot express "251 live at once," because doc 03's frame has no
notion of an unbounded family of simultaneous versions under one name at
all; `@sector_cap` qualifies ONE more version, not 250.

**What this project did:** `values/bases.py` declares three orthogonal axes
(`BASIS`, `ADJUSTMENT`, `PASS`) as data, ambient inside a declared region
(`basis="inherited"` on a phase envelope) and explicit at scope edges
(`basis="explicit"`, forcing every multi-basis read to carry a marker —
`actual()`, `hypothetical()`, or `basis_of("instalment")`). The build-time
rule (`BARE_READ_OF_MULTI_BASIS_VALUE = "build_error"`) makes an unmarked
read of `existing_obligations` inside `basis="explicit"` code not compile,
naming the value, its axes, and the three markers available. This is the
sketch's most invented mechanism, and it exists because doc 03's answer,
applied honestly, produces the exact adjudicator-discovered failure spec
5.21.1 warns about.

### 4. 24 ordering constraints including two genuine cycles broken by declaration, caught before running rather than after. [EXTENSION]

**Forced by:** §5.22, Q9.

**Doc 03's position:** one mechanism, §8.1: "`|` is a sequence: written
order is execution order." Complete for a flow where every constraint is
"A before B" and none of them is entry-point-conditional or circular.

**What doc 03's shape would produce:** for O-06 (the fraud/bureau ordering
INVERTS by entry point), a single `|` order cannot state it at all — eight
`|` orders would state it eight times, and spec 5.22 is explicit that this is
"the constraint that a single `\|` order cannot express and that eight
copies of the flow would express eight times, wrongly, within two years." For
O-09 and O-10 (genuine cycles), `\|` cannot express a cycle at all — a reader
sees `P10 \| P11 \| P10` with no way to know the second P10 is a declared
break of a real circularity rather than someone running affordability twice
by mistake.

**What this project did:** `ordering.py` makes constraints DATA, checked
against `flow.py` at build, in both directions — every declared constraint
holds in the graph, AND every cycle the graph contains is named by a
`cycle_break`, and every `cycle_break` still names a real cycle. `cycle_break`
(`O-06`, `O-09`, `O-10`) carries `broken_by=TwoPass(...)`, an approval
reference, a review date, and — for O-09 — `residual_recorded_as=
"routing_provisional_delta"` with `residual_monitored=True`: the compromise
is not just declared once, it is measured on an ongoing basis so the
declaration cannot rot into folklore. `isolates` (O-18) and
`ConcurrencyConstraint` (O-24) are two further primitives §5.22 needed that
`precedes`/`cycle_break` alone do not cover — see demand 7.

**A nuance worth stating plainly, because it does not resolve cleanly:**
`ordering.py`'s own `CONSTRAINTS` tuple declares twenty of the twenty-four
constraints its docstring says spec 5.22 tabulates (O-15, O-16, O-17 and O-21
are absent). Having read the graph they describe, this is largely correct
rather than a bug: O-15 (P14 sits inside the P10 loop) is visible directly
from `loops/l1_consolidation.py`'s body, O-16 (P15's per-account work before
its allocation) is an ordinary name-wired data dependency
(`phases/p15_limit_assignment/assignment.py`'s `Allocation` module reads
`caps` by name, which only `Matrix`'s `caps` step produces — doc 03 §2's
wiring rule makes the order provable from the dependency itself, whether
that is `Matrix`'s internal DAG, doc 03 §3.1, or the `\|` sequence the two
modules are composed with in the P15 envelope, doc 03 §8.1 — no separate
`precedes()` needed either way), and O-17 (the product fan-out before P16)
is directly visible in `flow.py`'s `\|` sequence. **O-21 ("record emission
last, and off the critical path") is not covered, and is a genuine gap
rather than an omission**: "last" is visible from sequence, but "off the
critical path" is
an execution-scheduling property with no declarative primitive anywhere in
doc 03, doc 08, or this project's own extensions. It is currently expressed
only as a comment on the P18 envelope (`phases/__init__.py`'s degradation
note: "the record write is asynchronous") — data an operator must trust
rather than a build-checked property. **Framework demand: an
`off_critical_path()` ordering primitive, checked the same way `isolates()`
is** — the build should be able to assert that a phase's completion does not
gate the caller's return, the same way it asserts a phase cannot read a
value.

### 5. A latency budget summing to 111.5 ms with 8.5 ms slack, declared rather than commented, with abandonable and non-abandonable phases. [EXTENSION]

**Forced by:** §5.24, §5.24.1–3, Q19.

**Doc 02's position:** kernel-level performance (fusion, boundary cost,
vectorisation) — real, measured, and entirely about THROUGHPUT of one
kernel. No document defines a per-phase LATENCY BUDGET as a governance
artefact, no document defines an overrun policy, and no document
distinguishes an abandonable phase from a non-abandonable one.

**What doc 02's shape would produce:** a latency target that is a comment
("P13 should be fast") rather than a number an overrun policy can read. Spec
5.24.2 names exactly the failure this produces: "excluding waiting from a
latency target is exactly how a latency target becomes unfalsifiable."

**What this project did:** `budgets/phases.toml` is a declared artefact —
`phase.budget = budget_ref("P01")` (`phases/__init__.py`) resolves to a row
in it, and `budgets/overrun.py` (referenced from `entrypoints/
ep04_campaign_preapproval.toml`'s comment) is what makes a 120 ms p99 budget
and a 6-hour window budget comparable at all: both are declared quantities
against which "did this phase overrun, on this decision" is a checkable
per-decision fact, not a daily average. `abandonable=False` (`P17`'s
envelope) and `not_abandonable` rows in `budgets/phases.toml` make §5.24.3's
overrun policy table (which phases may be dropped, which never can be)
data the build enforces, not a runbook a human follows under pressure.

**What's still missing:** doc 02 has a cost model for THROUGHPUT (fusion,
§1.1–1.2) but nothing analogous for LATENCY BUDGET ALLOCATION across phases
that share one process — P13's 38 ms exists because five OTHER phases this
project added (routing, identity, orchestration, arbitration, a second
affordability pass) together cost 13 ms a standalone flow would not spend.
Framework demand: a way to declare that one phase's budget is a residual of
others', checked at build so a new phase cannot silently eat into P13's
allocation without the negotiation §5.26.5 requires becoming visible.

### 6. Thirteen degraded sources under "degrade toward refer, never toward decline; mark it; re-assess it." [EXTENSION]

**Forced by:** §5.25, §5.25.1, Q21.

**Doc 04's position:** none. `grep`-level search of docs 02–08 for
degradation returns nothing; the closest concept is doc 04 §2.1's "not
enforced by the framework" list, which is silent on external-source failure
entirely.

**What this project did:** `degradation/sources.py` declares a
`DegradedSource` per source (thirteen), each with `entry_points=`,
`continues=`, and a `behaviour=` string that is rendered verbatim into the
reviewable artefact — the same pattern `ordering.py`'s `because=` uses for
the same reason (spec 9.3's governance gap is that nobody owns the
composition; prose in code is the only place that survives a refactor).
Crucially, `degradation="REFUSED"` on P10's envelope is a THIRD value,
distinct from `degradation=None` (P01: nothing can degrade it) and from a
populated tuple (P03: three sources, three different behaviours) —
`degraded_evidence_permitted=True` sits beside it, because "no degraded
affordability mode" and "degraded evidence, a different thing" are one word
apart and are opposite statements (spec 5.11). Doc 04 has no vocabulary for
any of REFUSED, None, or a tuple; this project invented a three-state
degradation posture where doc 04 has zero states.

### 7. Five measurable navigability tests, and what in the layout makes them pass. [EXTENSION]

**Forced by:** §5.27, Q13, Q14, Q15.

**Doc 04's position:** static lineage (§3) — `pipeline.lineage("z")` answers
"what can affect z," free, no execution. This satisfies HALF of N1 (locating
consumers) but not the harder half spec 5.27 states explicitly: "the test is
not 'find where the value is set'; it is 'find EVERY place it can be set'."
`lineage()` answers a downstream question; N1 is an upstream enumeration
question doc 04 does not have a mechanism for at all.

**What this project did:** `values/ceilings.py`'s `ceiling()` declares
`narrows=` as the SAME edge doc 04's lineage would want to walk backwards,
and `navigability/index.py`'s `where()` walks it: `amount_cap` resolves to
132 sites (118 register entries + 6 overlays + 6 product maxima + 1
regulatory maximum + 1 uplift entry) because the structure enumerates them
via declarations that already exist for OTHER reasons (direction-checking,
coincidence resolution), not because a second index was hand-maintained.
N3 (`resolve()`), N4 (`invert()`) and N5 (`trace_loop()`) are declared
alongside it in the same file because they share one property: none of them
executes anything, all of them read declarations or a build-time registry.

### 8. Change collision and blast radius answerable before a change ships. [UGLY]

**Forced by:** §5.28, §5.26.4, §5.26.5, Q5.

**Doc 04's position:** static lineage again, and it is explicitly NOT
sufficient once a loop exists: "naive reachability over a graph containing
L1 returns 'everything can affect everything', because the loop makes P16
reachable from P10 and P10 reachable from P16" (spec 5.28). Doc 04 §3 says a
query crossing a `@breaks_lineage` gap "reports `unknown` rather than
guessing" — the right instinct, applied to the wrong failure: a LOOP is not
a lineage gap, it is a real cycle the lineage mechanism has no way to bound.

**What I did instead of doc 04's lineage as specified:** blast radius has to
be CONDITIONED — on `entry_point_code` (so a quotation-only change to the
anti-harm threshold correctly returns "no effect on entry point 7"), on
`product_code`, and on `loop_pass_index` (so a query can ask "reachable
within one pass" without the loop's own back-edge collapsing the answer to
"everything"). `navigability/index.py` is declared as the home for this
(`phase_set_bitmap()` gives the entry-point conditioning cheaply, because it
is a precomputed bit pattern, not a live traversal); the loop-pass
conditioning is the harder of the two and is the concrete, unresolved
half of this demand — this sketch names where it goes, not the algorithm
that makes it under-60-seconds over 1 400 nodes.

### 9. 109 dead-logic candidates split correct-rare / shadowed / dead / unknown, denominators normalised by reachable population. [EXTENSION]

**Forced by:** §5.29, §5.29.1, Q18.

**Doc 04's position:** taps and OTel spans — what a value WAS, and how long a
kernel took. Neither counts REACHED / FIRED / BOUND per decision point, and
neither has a notion of a per-decision-point denominator that is anything
other than total volume.

**What this project did (declared, not built here — see `README.md`'s tree
for where `deadlogic/candidates.py` sits):** the three-count model (reached,
fired, bound) and the population-normalisation rule fall directly out of
values already declared for other reasons — `values/ceilings.py`'s
`evaluated_did_not_bind` distinction is EXACTLY the fired-but-not-bound
count spec 5.29.1 needs, invented for attribution (demand in `ceilings.py`'s
own docstring) and reused here for free. This is the strongest evidence in
the sketch that declaring a fact once for governance reasons and reusing it
for dead-logic measurement is cheaper than building a separate
instrumentation layer.

### 10. 5 650 test states, with certification cost proportional to blast radius. [EXTENSION]

**Forced by:** §5.30, §5.30.3, Q23.

**Doc 03's position:** `assert_modes_agree(pipeline, corpus)` and
`golden.record(pipeline, corpus)` (§11) — both take the WHOLE pipeline. There
is no notion of "run only the golden cases whose stratum intersects the
phases a change touched," which is exactly what spec 5.30.3 requires ("a
change to one phase must not require re-certifying all eighteen").

**What this project did:** `tests/golden/manifest.toml` (see `README.md`'s
tree) declares the 197-stratum construction as DATA — strata keyed by
(entry point, product, segment), each stratum tagged with the phases it
exercises. A certification run for a change with blast radius {P12, P13,
P14, P17} (a rate-card patch) filters the manifest to strata whose tag set
intersects {P12, P13, P14, P17}, which is a data query, not a re-run of
140 000 decisions. Doc 03's `corpus` argument would need to grow a
`strata=` filter for this to be native rather than bolted on — currently
it is bolted on, hence [EXTENSION] rather than [SATISFIED].

---

## B. Doc 07 tested at 1 400 decision points, 47 tables, 340 parameters, 12 teams

### 11. `modules/` + `pipelines/`, flat, has no unit between "module" and "everything." [UGLY]

**Forced by:** §5.1, §2 (Q6, the dominant question).

Doc 07 §1 is written for, and works well for, a project the size of `03
unsecured loan granting and pricing` — one product, ~30 modules, one
pipeline file. Applied here without modification: `pipelines/retail_credit.py`
would be an eighteen-module `\|` sequence with no place to hang a budget, an
owner set, a degradation posture, or an isolation constraint, because doc 07
§1's four principles talk about modules and pipelines and never mention
either of those things.

**What I did:** `phases/` sits BETWEEN `modules/` (doc 07's unit) and the
one pipeline file (`flow.py`). A phase is "a module with a declared
envelope" (`phases/__init__.py`'s own framing) — the envelope carries eight
fields (id, owners, decision_points, budget, degradation, basis, isolation,
emits) that doc 07's `module(...)` call has no slot for, because every one
of them is read by something OTHER than the compiler: the budget by the
overrun policy, the owners by CODEOWNERS generation, the decision-point
count by a registry assertion. This is doc 07's most direct scale failure:
its layout has nowhere to put governance metadata that is a property of a
COMPOSED THING, because at 30 modules nobody needed one.

### 12. `config/<pipeline>/<env>.json` mirrors the pipeline; at 340 parameters across four cadences that collapses into one file four teams edit. [UGLY]

**Forced by:** §6.2, §5.26.2, §5.26.3.

Doc 07 §1's `config/term_loan/production.json` is one file per pipeline per
environment. At 340 parameters split across five approval routes
(statutory/gazette, policy/quarterly, product/weekly, pricing/monthly-plus-
patch, overlay/ad-hoc) this is one file five different approval processes
must all touch without colliding — exactly the Tuesday/Wednesday/Thursday
scenario spec 5.26.2 works through, where three teams change the SAME
artefact class in one week under three different approval routes.

**What I did:** config is split by ARTEFACT OWNERSHIP CLASS, not by
pipeline: `budgets/`, `degradation/`, `tables/cap_register/`, `tables/
rate_cards/` are each a top-level directory with their own cadence and their
own CODEOWNERS row (see `.github/CODEOWNERS`), rather than sub-paths inside
one `config/retail_credit/` tree doc 07's convention would suggest. A
Treasury patch to `tables/rate_cards/flex_loan/` cannot collide, in the
filesystem sense, with Credit Risk Policy's quarterly release to `tables/
cap_register/cap_register.toml`, even though both artefacts are read by the
same phase (P09, P12).

### 13. "Tests mirror modules" says nothing about tests that must mirror BLAST RADIUS instead. [EXTENSION]

**Forced by:** §5.30.2.

Doc 07 §1 principle 4: "tests mirror modules... module with no test is a
detectable gap." True and necessary (`tests/phases/` does this — see demand
10's `test_p13_solve.py`), but five of the eight test layers spec 5.30.2
requires (golden set, degraded-mode set, re-check regression, entry-point
agreement, replay, swap set) are not module-shaped at all — they are
population-shaped, keyed by (entry point, product, segment, degradation
state), and doc 07 has no directory convention for them. `tests/golden/`,
`tests/degraded/`, `tests/recheck/`, `tests/replay/` sit alongside `tests/
phases/` as siblings with a different addressing scheme, which doc 07's
"mirror the modules" principle does not anticipate and does not forbid —
it simply has nothing to say about the other half of the testing surface
this project needs.

### 14. Doc 07 has no ownership file at all. [EXTENSION]

**Forced by:** §5.26.1, §11 scenarios 4–5.

`OWNERS.toml` and `.github/CODEOWNERS` (generated) are wholly new relative
to doc 07's layout, which assumes one engineer or one small team per
project. At twelve teams, "who reviews this PR" cannot be answered by
directory location (demand 2), and doc 07's silence on the question is not
a gap that matters at 30 modules — it is the first thing that breaks at 340.

### 15. `contracts/` (frozen module interfaces) doesn't carry a BUDGET contract, and this project needs one. [EXTENSION]

**Forced by:** §5.13, §5.26.4.

Doc 07 §5's `contracts/affordability.json` freezes an interface: inputs,
outputs, params. `phases/p12_pricing/rate.py` (etc.) is called AS A BODY by
two phases that do not own it (P13 up to 152 times, `loops/
l1_consolidation.py`'s P14 invocation up to 400 times), each under its own
performance budget (`per_call_budget_us=18.0` on the P12 envelope). An
interface freeze says nothing about whether a change to P12 that adds one
more decision point silently pushes P13's 38 ms allocation over budget. A
contract file that only freezes shape, not cost, misses exactly the
negotiation spec 5.26.4 says must be visible ("T6 cannot be expected to know
that a card cell participates in a consolidation objective" — the analogous
statement for P12/P13/P14 is that neither caller can be expected to know a
P12 change moved their own budget).

---

## C. Authoring-API gaps this project's own extensions expose

### 16. A ceiling is a monotone accumulator, not a value that happens to get overwritten. [UGLY]

**Forced by:** §5.10, §5.27 N1.

Doc 03 §3.2's waterfall (`SeedTermCap \| ApplyIncomeCap \| ApplySectorCap`)
gives auditable overwrite with no notion of DIRECTION, no notion of
"evaluated and did not bind" (a step returning its input unchanged is
indistinguishable from a step that did not run), and no way to say that
exactly one entry, out of 118, may move the value the other way. Applied
honestly to the cap register, doc 03's shape produces 118 authors each
trusted, by convention, not to write a raising rule — a convention spec
5.10's uplift-entry requirement makes explicitly unsafe. `values/
ceilings.py`'s `ceiling(direction=REDUCE_ONLY, may_raise=["CAP-0118"], ...)`
makes direction a property CHECKED AT RUNTIME against the ceiling, not a
discipline 118 rule authors keep.

### 17. A mode may change evidence and parameters, never arithmetic — and nothing currently checks that at build time. [EXTENSION]

**Forced by:** §5.11.

`modes_share_arithmetic=True` (P10's envelope) is a promise, not a checked
property, in this sketch. The spec's own bar — "the four modes share one
step set and differ only in `param()` bindings and `missing_as` policy" — is
exactly checkable by comparing each mode's compiled step DAG for structural
equality, which doc 03 has all the pieces for (structure is data, §2) but no
existing entry point calls this comparison automatically. Framework demand:
`assert_arithmetic_shared(modes=[...])`, analogous to `assert_modes_agree`
for the interpreted/stepped/fused ladder, but across PARAMETER-BOUND
variants of one module rather than across execution modes of one pipeline.

### 18. A `ruleset`'s pass membership can be derived from an attribute of the rule, not declared per rule. [EXTENSION]

**Forced by:** §5.10 (O-10, the register's two-pass split).

Doc 08 §3's `ruleset` declares `reads`/`writes` for the WHOLE ruleset;
individual rules inside it have no declared attribute doc 08 defines. O-10
requires that which of the register's two passes an entry belongs to be
DERIVED from which ceiling it narrows (`pass_derived_from="narrows"` in
`values/ceilings.py`), specifically so "the entries do not know which pass
they are in" (spec 5.10) — the split must stay invisible to the 118 authors.
Doc 08's `ruleset` has no mechanism for a rule-level attribute to route rules
into an implicit partition at all; this project's `pass_derived_from=`
is new.

---

## D. The scale-wall question itself

### 19. Spec 13 Q17 — is a campaign tree leaf a decision point?

**This project's position, taken in `entrypoints/
ep04_campaign_preapproval.toml`: no.** A tree leaf is not one of the 1 400
(§2.1's own stated exclusion), but it IS counted — separately, honestly, as
2 900 "citable decision points" specific to entry point 4
(`citable_decision_points = 2900` in that file), because a client asking
"why this offer, eighteen months ago" is asking about a leaf, and the
record's `path_capture = "first_class_output"` makes that answerable without
promoting the leaf into the 1 400 baseline every other number in this
document is stated against.

**What this costs, stated:** every navigability target (N1's 10-minute
locate, N3's zero-unresolvable-citations, N4's 60-second inversion) is
measured against 1 400 nodes in this document's own numbers (§2.1, §5.27).
Taking the OTHER position — a leaf is a decision point — does not just add
2 900 more of the SAME KIND of node; it adds a node kind with a different
owner (T11 alone, weekly cadence, campaign-forum approval) sitting on TOP of
1 400 nodes with twelve owners and four cadences, and `navigability/
index.py`'s `where()` and `invert()` would need to know that a leaf's
"place it can be set" is a tree-authoring tool's row, not a Python
declaration or a TOML entry — a THIRD storage shape (`campaign_trees.py`'s
`Table`-like tree structure) on top of the two this sketch already has
(code declarations for structure, TOML/CSV for register/rate-card data).
Blast radius (`navigability/index.py`) would need a fourth conditioning
dimension (tree id) beside entry point, product and loop pass. Dead-logic
measurement (demand 9) would need a denominator per LEAF, not per phase,
and a leaf reached zero times in ninety days is a materially different
finding from a cap register entry reached zero times, because a leaf's
absence is a marketing decision (suppress this branch) as often as it is a
defect. **The honest cost of "yes" is not 5.86x more nodes to navigate — it
is a second navigability system, because the existing one is built around
code-and-TOML declarations a tree editor does not produce.**

---

## Tally

| Mark | Count |
|---|---|
| [SATISFIED] | 4 |
| [EXTENSION] | 14 |
| [UGLY] | 8 |
| **Total** | **26** |

Four items are [SATISFIED] and are folded into the numbered demands above
rather than listed separately, because none of them stands alone at this
project's scale: doc 03 §2's structure-as-data (used by every phase
envelope and by `values/ceilings.py`'s runtime direction check), §3.1's
intra-module topological sort (used to justify why O-16 needs no
declaration), §3.3's version chains (the SPINE `values/register.py`'s three
axes extend rather than replace), and doc 04 §3's static lineage
(the STARTING POINT `navigability/index.py` builds on, even where it is not
sufficient alone — demand 8). Every one of the fourteen [EXTENSION] items is
a case where the doc 03/07/08 mechanism is the right shape and stops one
property short of what this project's scale requires. Every one of the eight
[UGLY] items is a case where following doc 03/07 literally would produce
something this project's own worked examples (the R60 000 band-edge
inversion, the 251-version obligations figure, the Tuesday/Wednesday/Thursday
collision) show to be actively wrong, not merely inconvenient.
