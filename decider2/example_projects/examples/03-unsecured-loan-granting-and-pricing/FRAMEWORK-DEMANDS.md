# FRAMEWORK-DEMANDS — 03 Unsecured loan granting and pricing

Thirty numbered demands this project makes of `decider2`. Each names the spec
section that forced it and is marked:

- **[SATISFIED]** — doc 03 (or 02/04/07/08) already does this. Eight of thirty.
- **[EXTENSION]** — the shape is right, something is missing. Eleven.
- **[UGLY]** — doc 03's answer would produce something I would refuse to review.
  What I did instead is stated. Eleven.

The [UGLY] rows are the point of this document.

---

## A. The overlay stack

### 1. An overlay is two artefacts, not one, and the split is at the point. [UGLY]

**Forced by:** §5.4.1, §5.5 "overlays on the ceilings", §5.7(e), §6.3, §13.17,
core-library §6.22. Five stages, six overlay kinds, six-to-twenty live.

**Doc 03's position:** none. Overlays appear in no framework document. Doc 02 §4
offers values / params / tables; an overlay is none of them, which §13.17 states
as the problem ("it changes values, which makes it look like a parameter; it
composes in a declared order, which makes it look like logic; it is separately
approved and separately versioned, which makes it look like neither").

**What doc 03 would make you write:** a module per overlay kind, wired into the
pipeline conditionally, with the scope test as an `if` in a step body, the
composition order as the author's `|` ordering, the unadjusted value as a
manually-added second output, and the disable switch as a param the author
remembers to check in five places. Every one of §5.4.1's six requirements
becomes a convention. The one that fails first is "the order is declared, not
emergent": with `|` the order is in Python, so a Credit Committee reordering two
overlays is a release, and §6.3 says the register changes "sometimes twice in a
month".

**What I did:** `OverlayPoint` (code, skeleton class, engineer-owned) declares
*what may be moved and in which direction*; `adjustment_set` (data, values
class, Credit-Committee-owned) declares *whether, by how much, for whom, in what
order, until when*. `adjustments/points.py`. The six §5.4.1 requirements then
fall out with no additional mechanism — see that file's docstring. The important
consequence is that `enabled: false` gives acceptance criterion 20 ("runs with
the stack disabled through the same implementation") for free, because the set
is a runtime array and the kernel is byte-identical either way.

### 2. An overlay point must be able to target a value, a param, or a table cell. [EXTENSION]

**Forced by:** §5.4.1 overlay 2 (scaling change targets the score-to-odds
*relationship* — a calibration param) and overlay 4 (boundary shift targets
*grade boundary table cells*, not the grade).

Three of the five points target a value. One targets a param. One targets table
cells. Params in doc 03 §4 have no notion of an effective-dated, scoped,
separately-approved modification; tables have no notion of an overlay at all.
Resolving a param overlay at `params_binding` time keeps the bundle's type
fixed, so nothing recompiles — which is the property that makes this cheap.

### 3. Direction belongs on the point, in code; magnitude belongs in the set, in config. [EXTENSION]

**Forced by:** §13.19 — "where is the conservative-only asymmetry enforced? …
what makes it impossible to bypass by defining a negative magnitude?"

Answering "a runtime check" fails: a runtime check that rejects a magnitude has
already let the artefact be approved. Answering "a property of the kind" fails:
a score shift is legitimately bidirectional and a cap reduction is not.
`direction=Direction.REDUCE_ONLY` on the *point*, reviewed under a release,
composed with the set's magnitude at set-validation time, before anything runs.
Acceptance criterion 19.

### 4. An overlay's scope decision is a three-valued recorded fact. [EXTENSION]

**Forced by:** §5.4.1 — "an overlay evaluated outside its scope is an error, not
a silent no-op."

APPLIED / IN_FORCE_BUT_OUT_OF_SCOPE / NOT_IN_FORCE. Two of the three are
currently one thing, and a silent no-op is unrepresentable once they are three.

Worth recording that the spec's own §5.4.1 worked example applies a
channel-4-scoped score shift and a channel-4-scoped PD multiplier to client W,
who is channel 2. Either the example is wrong or the scope is wider than stated.
That ambiguity survived a careful spec into a careful reading, which is the
argument for making the scope outcome a recorded value rather than a condition
inside a rule.

---

## B. The bounded solve

### 5. Non-monotonicity is a property of the artefacts, not of the search. [UGLY]

**Forced by:** §5.8, §13.5 — "how does non-monotonicity get expressed as a
requirement the implementation must satisfy, rather than as a property somebody
hopes holds? What does a specification of 'correct' look like that is checkable
without evaluating all 4 981 candidates?"

**Doc 03's answer:** `Loop(should_continue, Body, carries=[...],
max_iterations=N)`, and you write the search in the body. Correctness is then
unverifiable by any framework mechanism, and the only available check is the
10 000-application exhaustive comparison — which tells you *that* the search
disagreed, never *which artefact* changed under it.

**What I did:** `Partition` over declared breakpoints (`modules/solve/partition.py`).
The instalment is not non-monotone; it is *piecewise* monotone, and every
breakpoint is already known to the artefact that creates it — the rate card's 96
band edges, the fee schedule's one kink, a rate add-on's scope edges. Given the
partition, "instalment is monotone within a segment" is a theorem, so "the
search returns the true maximum" is a proof. Acceptance criterion 3 then changes
meaning: it becomes a regression test on the *breakpoint declarations*, and it
fails naming an artefact that forgot to declare something.

### 6. `@breakpoints` and `@monotone_in` are declarations, not comments. [EXTENSION]

**Forced by:** §5.7(b) — the fee ceiling binds at R12 700 exactly, which "sits
*inside* the rate band R12 000–R12 999, so within a single rate band the
instalment's dependence on the amount changes shape."

A search that partitions on rate-band edges alone is wrong, silently, for
amounts in one band out of 96. Change scenario 10 moves that kink to R13 300 and
into a different band; with the declaration the search stays correct with no
code change. Used three times: `pricing/fees.py`, `pricing/rate_card.py`,
`pricing/credit_life.py`, plus `contributes_breakpoints` on the rate add-on
point — which §5.8.1(2) explicitly demands ("the declared inversion list in §6.1
is incomplete without it").

### 7. The budget counts probes, not iterations. [UGLY]

**Forced by:** §5.8.1, acceptance criterion 4 — "no application performs more
than 24 pricing evaluations for any single term, ever, on any input, including
adversarial ones."

`max_iterations` is the wrong unit. One iteration of a bracket-scan-bisect
strategy performs zero, one or several probes. Counting iterations gives a bound
that is either loose (and violates the criterion) or requires the author to
hand-maintain a probe counter inside a loop body — which is exactly the
hand-written invariant this framework exists to remove.

`Budget(probes=24, on_exhausted=...)` counts executions of the declared probe
module, which is why `PriceCandidate` must be one composite (§13.9) rather than
five library calls a consumer assembles.

### 8. The worst-case probe count must be computable in closed form, at table-validation time. [EXTENSION]

**Forced by:** acceptance criterion 4's word *ever*.

Sampling cannot establish "ever". `1 + S + ceil(log2(W))` is a closed form over
the card's axis, evaluated for every (term, grade) when the card is staged:
`rate_card.search_budget_is_satisfiable`. A re-banded card (change scenario 1,
96 → 120 bands) that would blow the budget never reaches production. The current
card's worst case is 21.

This is the demand I would most want the framework to take seriously, because it
converts the single hardest runtime guarantee in the spec into a data-validation
rule.

### 9. A bounded search must be able to say it failed. [EXTENSION]

**Forced by:** §5.8.1, §13.6 — "is a bounded search that can fail to find an
answer expressible at all, or does every search have to be assumed to succeed?"

`on_exhausted="no_offer"` with an outcome, a queue and a reason code. Not a
truncated answer, not a best-effort, not an exception. Change scenario 13 (0.3%
hit 24; Product wants 32, Credit Systems wants the search improved, "both must
be testable cheaply") is then one param and one strategy knob, each priced by
`decider2.impact()`.

### 10. The binding constraint must be assigned by the framework, not written by the author. [EXTENSION]

**Forced by:** §5.8.6 — nine codes, one per constraint.

The author writing a nine-branch `if` to decide which constraint bound is a
wrong-answer generator, and it is unreviewable. The domain was built from named
constraints (`amount_cap`, `requested_amount`, the product minimum and maximum),
so the framework knows which named edge the answer landed on:
`attribute_from="domain_provenance"`.

### 11. The inverse is published as a bound, not as a result. [EXTENSION]

**Forced by:** §13.10 — "can `core.instalment` be run in reverse, and does using
the inverse as a starting point for the search, rather than as the answer,
change what the library needs to publish?"

Yes. `advance_for_instalment_relaxed` deliberately over-estimates, by using the
cheapest rate in the domain and minimal loadings — which makes it a *provable*
upper bound rather than a good guess, and therefore admissible in the probe-count
proof. `invertible_pair(inverse=..., agree_to=..., corpus=...)` ties forward and
reverse together so their agreement is a generated test, which §5.4's
calibration inverse ("the two directions must agree to within 0.01 of a point")
needs identically.

### 12. Evidence is a write-only sink, and its level cannot change the answer. [EXTENSION]

**Forced by:** §13.7 — "is evidence capture a parameter, and if so, does turning
it down change the answer?"

Eleven fields × 216 probes × 14.2 M applications is 34 billion values a month,
so it has to be a parameter. The guarantee that it is answer-neutral has to be
structural: no step may read from the evidence sink, so it is in no output's
lineage, and that is statically checkable. `assert_modes_agree` additionally
runs the corpus at every level and requires identical decisions.

`Evidence.sampled(rate, keyed_on="client_id")` — keyed on the client, not on a
counter, because a sample whose membership a replay cannot reproduce is not
evidence.

---

## C. The cap waterfall

### 13. A waterfall over 52 config-ordered rules is a kind, not a `|` chain. [UGLY]

**Forced by:** §5.5, acceptance criterion 15, change scenario 3.

**Doc 03's answer:** §3.2 — "a waterfall is expressed as
`SeedTermCap | ApplyIncomeCap | ApplySectorCap`, each rule its own auditable
unit, with the version chain giving attribution." That is genuinely good for
three rules and it fails here for three independent reasons:

1. **Order is config and `|` is code.** Criterion 15: "adding one rule to the
   cap register, in any sequence position, is a configuration change reviewed by
   its owner — not a release." A `|` expression is a Python edit, a review, a
   build, a compile and a deploy, four times a year, with four teams editing the
   same expression concurrently.
2. **The version chain records changes; the spec requires verdicts.** §5.5
   demands the distinction between "did not apply" and "applied and did not
   bind" for all 52 rules. A rule that applies and does not bind produces *no
   version*, so the version chain cannot represent it. Client W's chain has four
   bound rows and six non-bound rows, and "is my rule doing anything?" is a
   question about the six.
3. **Fifty-two modules is fifty-two names in one pipeline expression**, which is
   a merge-conflict surface, not a governance boundary.

**What I did:** `Waterfall` (`modules/caps/register.py`) — tabular, so by doc 08
§3.4's own test it runs on a **generic kernel**, and an interior change is free:
no codegen, no compile, no stage, no release. Criterion 15 becomes true by
construction. Sequence is a `seq` *field*, not array position, so a reorder is a
diff on 52 integers.

### 14. A rule verdict is five-valued. [EXTENSION]

NOT_APPLICABLE (with the predicate that failed) / EVALUATED_NOT_BINDING / BOUND
/ COINCIDENT / DECLINED. §5.5's attribution requirement names the first four
explicitly; COINCIDENT exists because two rules reducing to the same value need a
stated tie-break (earliest in sequence binds) and the later one is a distinct
fact from both "bound" and "did not bind".

### 15. Reduce-only is declared on the ceiling; the one exception is named in code. [EXTENSION]

**Forced by:** §5.5's uplift rule, CAP-0420, "the single exception to reduce-only".

`Ceiling(direction=REDUCE_ONLY)` plus `raise_permitted_by={"CAP-0420": ...}`.
The register cannot grant itself the right to raise because the right is not in
the register. `not_above_class="regulatory"` is evaluated against the chain
rather than against sequence position, so it holds however the register is
reordered — which matters, because the people reordering it quarterly do not
know the constraint exists.

### 16. A cap overlay enters the chain as its own row, attributed to the set. [EXTENSION]

**Forced by:** §5.5 — "'your cap was R95 000, reduced to R76 000 by a policy
overlay approved under CC-2026-31, expiring 2027-01-31' is a different answer to
the client and to the regulator than 'a rule bound it'."

`joins_chain=` and `chain_attribution="adjustment_set"` on the CAP_REDUCTION
point. The chain is one structure with two kinds of producer.

### 17. Whether sixteen predicates cover eighty rules is unresolved. [UGLY — unresolved]

Doc 08 §3.2 is right that an expression string in config is code in config, and
right to remove it. The price it names — "adding a derived value now requires a
code change" — lands harder here than the doc anticipates: four owners, quarterly
cadence, and if every new rule needs a new predicate then "configuration change,
not a release" is a fiction.

My mitigation is that predicates are generic over their thresholds
(`months_employed_below(threshold)` with the threshold as a `{"param": ...}`
reference), so sixteen cover fifty-two today. I have no evidence it holds at
eighty, and I am not going to pretend otherwise. This is the demand most likely
to be wrong.

---

## D. Tables

### 18. A 63 360-cell banded grid is a kind. [UGLY]

**Forced by:** §5.7(a), §6.1, §13.3, §13.4.

**Doc 03's answer:** §4.4's provisional `Table` — `tables.term.max_loan[term]`,
a one-dimensional keyed lookup, explicitly the document's "lowest-confidence
part". It cannot express a banded axis with edges in a companion artefact,
cell-level attribution, effective dating, pre-live validation, cell-level
diffing, or spreadsheet ingestion. Every one of those is a hard requirement.

**What I did:** `grid()` (`modules/pricing/rate_card.py`) with `Axis.banded` /
`Axis.dense`, `emits=[..., "rate_cell_id", ...]` so attribution is free rather
than logged, `@validates` rules attached to the *definition* and run against the
*data*, and a version manifest carrying the validation report and the cell-level
diff with re-pricing impact. §6.1: "'Treasury sent a new spreadsheet' is not a
reviewable change, and at 63 360 cells nobody is going to eyeball it."

### 19. Table *values* are free; a table *axis* is staged. [EXTENSION]

Doc 08 §4.2 says "table contents, for a generic-kernel kind — free" and stops
there. It needs a second row: an axis change alters the cell array's shape,
which is part of the table's type. Change scenario 1 (96 → 120 amount bands) and
change scenario 8 (672 → 896 credit-life cells) are both axis changes and both
must go through `stage`/`activate`, not `swap`.

### 20. The lifecycle must be generic over artefacts, not over pipelines. [EXTENSION]

Doc 08 §4's state machine is over *pipeline* generations. §6.1 and criteria 9
and 10 need it over *table* generations: a mid-month patch applied within two
hours, without a deploy, without interrupting serving, with in-flight
applications completing on the card they started with. That last guarantee falls
straight out of doc 08 §4 property 1 ("the generation pointer is read exactly
once per invocation") — but only if the pointer covers tables. It currently does
not.

### 21. `resident=True` must be asserted at build, not hoped for. [EXTENSION]

**Forced by:** §8 — "a 63 360-cell card must be resident and indexed before the
first request, not on it."

Doc 02 §3.4's `decider build --verify` asserts zero *compilations* on a runtime
load. It says nothing about data. The same flag should assert zero *table loads*.
(The rate card at int32 basis-points × 100 is 253 KB and L2-resident, which is
why §13.4's "140 000 lookups per second" is not the interesting part of that
question. The interesting part is attribution, and the cell id is one int32.)

### 22. Effective dating must be impossible to forget, not merely mandatory. [UGLY]

**Forced by:** §13.15 — "what is the unit of effective dating? … What makes
forgetting it impossible rather than merely discouraged?" Twenty-three tables,
five owners, five cadences.

**Doc 03's answer:** none. `tables` is a reserved parameter name in a one-line
sketch, with no date anywhere near it.

**What I did:** `tables` is a bundle that **cannot be constructed without a
`decision_date`** — `Tables.resolve(registry, decision_date)` — and every step
reading a table asks for `tables`, never for a table directly. Forgetting the
date is not a mistake you can make; it is a constructor you cannot call. Every
resolved version is stamped into the record automatically, which is core-library
change scenario 9 ("an internal audit finding requires that every application
record the version of every table it touched, where today only some are").

The residue: a batch with heterogeneous `decision_date` needs one invocation per
date, so the frame tier needs a `PartitionBy`. Real-time and the monthly
pre-assessment are each single-dated, so this only bites on replay — which is
exactly when it must not bite. Unresolved.

---

## E. Collections, nulls, and the record/frame boundary

### 23. A ragged collection with a declared capacity is a first-class value. [UGLY]

**Forced by:** §4.3 (0..80 bureau accounts, 0..120 enquiries, 0..25 public
records), §4.2 (0..40 internal accounts, 0..6 in-flight, 1..12 consents), §5.9
("the output is ragged, between 0 and 9 offers"), §4.5 (0..80 obligations).

**Doc 03's answer:** everything is a scalar. `Loop` carries scalars. There is no
map, no filter over a collection, no reduce, and no way for a module to produce
between zero and nine of anything.

**What I did:** `Collection[T]` / `Ragged[T, capacity]` with `Map`, `MapFilter`,
`Collapse`, `Rank`. Capacity is required for the same reason `max_iterations` is
required — the record tier has no heap — and declaring it is how an 81st bureau
account becomes a *data-quality value* (DQ-2, refer) rather than an exception.
§5.3 lists "an account list truncated by the bureau" as a DQ-2 condition, so a
framework that can only raise on overflow makes that rule unwritable.

### 24. This pipeline has zero frame-tier nodes, and that is what makes criterion 11 checkable. [UGLY]

**Forced by:** §8 — "real-time / batch identity … stated as an acceptance
criterion, not an aspiration", and §13.12 — "what structural property makes that
checkable rather than hoped for?"

Doc 02 §3.5 says `score()` "bypasses polars entirely". A pipeline containing a
`Join` or an `Aggregate` therefore has no scalar equivalent — so a
bureau-aggregating pipeline would have to run as *two* pipelines, and criterion
11 would be a claim about two implementations agreeing. That is the same
failure mode as a second exhaustive verifier and a second validation stage.

So every ragged input arrives as a per-record collection and every aggregation
over it is record-tier arithmetic. The frame tier is used for nothing here. The
one genuinely set-shaped input — the related-party exposure graph — is resolved
upstream and arrives as a scalar `group_exposure_limit`, which is what §4.2 says
anyway.

### 25. A null needs a reason attached to it. [UGLY]

**Forced by:** §5.1 (pass / fail / **not_evaluated, with the reason it could not
be evaluated** — "a third state, distinct from pass and fail, and it must not
collapse into either"), §5.3 ("no bureau record, a bureau record showing no
accounts, and a bureau enquiry that failed are three different applicants and
must remain three different values through the whole flow"), core-library §7.4
(three null situations, distinct).

Three independent sections demanding the same mechanism.

**Doc 03's answer:** §1 tier 3, `float | None`. One `None`. By the time three
distinct facts reach the scorecard they are one fact, and §5.4's thin-file
routing rule is unimplementable.

**What I did:** `Maybe[T]` — a tagged null carrying an `AbsenceReason`
(`pipelines/reasons.py`). Cost is one int8 per nullable value.

Doc 03 §1 explicitly rejected a `.value`/`.valid` wrapper "because it exposes an
unchecked accessor and adds a concept Python developers don't already have". The
first objection is real and I take it: `Maybe` has no `.value`. It has
`.or_else(default)` and `.absent_reason`, so there is no unchecked read. The
second objection I am overruling — three spec sections independently demand the
concept, so the project has it whether or not the framework does, and having it
in the framework is the only way it stays uniform.

---

## F. Governance, identity and correctness

### 26. `independent_of=` is a negative lineage assertion. [EXTENSION]

**Forced by:** §5.10, §13.14 — "how is the final validation stage expressed so
that it is genuinely independent of the stages it validates, rather than
re-running the same code and agreeing with itself?"

Doc 03 §9 has only positive lineage queries. The negative form is the same
machinery with the result inverted, checked at build, and it is the cheapest
governance feature in this sketch. Two tiers, because the honest version admits
that writing the annuity twice is worse than sharing it:
`independent_of=[...]` plus `shares_primitives={name: corpus}`, where an
unlisted shared step is a build error. See `modules/validation/independent.py`,
including what this does *not* catch.

The same assertion discharges acceptance criterion 2: if nothing downstream of
the solve may write `offered_amount`, the silent failure §5.8(7) describes is
unreachable, and "nothing downstream writes it" is one lineage query.

### 27. A pipeline-level numerics contract, and a fourth rung on the ladder. [UGLY]

**Forced by:** criteria 11 and 14, §8's determinism row.

Doc 02 §3.3 makes `fastmath` a per-kernel author choice. Doc 02 §3.1 then
*excludes* a fastmath kernel from the exact-agreement assertion. So one author
enabling it on one hot kernel four files away breaks bit-exact real-time/batch
identity, and the ladder is designed not to notice. Measured: 46–73% of rows
differing by up to 17 ULP, for 1.09×.

`with_numerics(FLEX_NUMERICS)` (`pipelines/numerics.py`) sets money, rate,
rounding, accumulator width and the fastmath ban for everything inside it, and
is part of the structure fingerprint. It also carries
`parallel_requires_no_reduction=True`, checked statically — because a `prange`
region containing a float accumulation changes summation order between paths.

And the ladder needs `assert_paths_agree`: `interpreted ≡ stepped ≡ fused` are
all within one entry point, while `score()` and `apply()` differ in fusion
grouping (doc 02 §1.2 gives them opposite defaults) and in the boundary they
cross. Their agreement is a separate claim.

### 28. Money is int64 cents, and the core library's published vocabulary contradicts that. [UGLY]

Doc 03 §1.2 is unambiguous: "money is a scaled int64 of cents, never float and
never Decimal", with a measured int64 overflow at R27 431 and a float rounding
divergence worth one cent per row. Core-library §4 publishes `offered_amount`,
`instalment`, `initiation_fee` and `total_cost_of_credit` as `float64`.

One of those is wrong, and this project cannot be the place it gets resolved by
convention. The spec's own worked failure turns on four cents (R1 560.04 against
R1 560.00, "no, by 4 cents"), which is not reliably representable in float64
rands. This project uses int64 cents throughout and converts at the frame
boundary, and raises the contradiction rather than papering over it. Six
consumers converting at every seam is the worst of both outcomes.

### 29. A stable hash must be frozen across releases. [EXTENSION]

**Forced by:** §5.4's challenger — a deterministic 10% of `client_id`, "not of
`application_id`, so a client receives consistent treatment across repeat
applications, and not by a random draw, so a replay reproduces the selection."

A hash whose value changes when the interpreter or the library changes silently
re-randomises the parallel run and destroys the comparison it exists to produce.
`stable_hash_u64` must be specified and frozen, with a corpus, like any other
statutory calculation.

### 30. A consumed capability's table versions are part of this project's declared interface. [EXTENSION]

**Forced by:** §5.6 — "those version identifiers travel into this project's
decision record unchanged — a replay of a granting decision must pin the
affordability tables as tightly as its own."

`consumes(..., carries_versions=[...])` puts them in the wiring, the schema, the
frozen contract and the audit record. A consumer that forgets fails the build,
rather than failing Internal Audit's next sample — which core-library change
scenario 9 records as how the requirement arose.

---

## Marked [SATISFIED] — eight things doc 03 already does correctly

Listed because a document of complaints is not an assessment.

1. **The recommendation objective.** §13.13 asks whether a three-valued
   objective is configuration or three implementations selected by
   configuration. Doc 03 §8.2's routing `Branch` with a param-reading router is
   the complete answer: all three arms are in `lineage()` and in the rendered
   artefact, `branch_path` records which fired as one int64 compile-time
   immediate, and switching it is a params swap. Change scenario 4 (switched
   Tuesday, reverted Thursday) costs microseconds. Nothing needs adding.
2. **The effective-rate IRR.** `Loop(max_iterations=32)` with a convergence
   tolerance is exactly right, and is the one place in this project where doc 03
   §8.3's loop needs no extension.
3. **Reason codes need no new machinery** (doc 04 §4.1). A reason code is a step
   output; which rule fired is `branch_path`. This project has ~96 of them and
   none of them needed a framework feature.
4. **`param()` in the signature** (doc 03 §4.4, confirmed by E11). One rule, one
   artefact, one file. The alternative — a pydantic model per rule — is exactly
   the shape doc 01 §5.3's law says does not get used, and this project has
   about forty single-knob rules.
5. **Small modules are safe** (doc 02 §1.1). Boundary stores are near-free, so
   the audit-shaped decomposition costs nothing. This project uses `fuse()` once,
   around the probe, for the one region executed 216 times per application.
6. **`stage` / `activate` / `rollback`** (doc 08 §4). The mid-month card patch,
   the quarterly register reorder and the monthly model release all want exactly
   this state machine. It needs generalising to artefacts (#20), not replacing.
7. **`decider2.impact(active, candidate, sample)`** (doc 08 §5) is the
   counterfactual §5.8.1(3) asks for and change scenarios 16 and 18 depend on.
   "Re-running the whole search with the stack disabled is too expensive at
   55 000 a day" is answered by "run it on the sample", and the framework
   already has that call.
8. **Config may not contain code** (doc 08 §1). The cap register would have
   grown an expression language within a year without this rule, and four teams
   would each have written arithmetic in it differently. The price is #17 and I
   still think the rule is right.
