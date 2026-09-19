# Framework demands — 07 credit limit management

Numbered, traced, marked. **SAT** = satisfied by doc 03. **EXT** = doc 03 lacks
it but nothing in doc 03 fights it. **UGLY** = doc 03 *has* a way and the way is
bad enough that I did something else — these are the ones worth arguing about.
**AWK** = genuine awkwardness I could not resolve and am not hiding.

---

## The population stage

**1. An ordered fold over a frame with carried scalar state — `Sweep`.**
*s5.8, s13.1, s13.2. UGLY.*
Doc 03 offers `|`, `Branch`, `Loop` and a frame tier of join/aggregate/filter.
None holds 708 000 results and produces one account's answer from the other
707 999. What doc 03 pushes you toward is polars — sort by rank, `cum_sum` the
additional limit, cut at the budget. That gets 80% of the way and dies, because
the remaining 20% is what an auditor tests: the tail rule (skip an account that
does not fit, continue, stop after 2 000 skips), three envelopes binding
independently, and *which one bound*. As window functions that is a chain nobody
can read, lineage cannot follow, and whose numerics depend on partitioning.
Instead: `Sweep(body, carries=[...], requires_order=[...])` — `Loop` with the
rows as the iteration space, body is ordinary scalar Python, carries follow the
loop-body scope rule doc 03 s3 already states. One combinator, not a third tier.
Three consequences are the actual deliverable: the rank is an ordinary output
with a producer and a version chain rather than something attached afterwards;
the carried state *at the moment of consideration* is a per-account column, so
"when we reached you, R0 of R2.4 bn remained" needs no re-run; determinism is
structural, because the carry at row *i* is a pure function of rows 0..*i*-1 in
the declared order.

**2. Ordering is part of the declared frame schema.** *s5.8 c7, s8. EXT.*
`Sweep` is correct only under a declared order. Doc 02 s5 propagates column
schema; it does not propagate ordering, and a `Join` between the `Sort` and the
`Sweep` destroys it silently. `requires_order=["rank_key asc", "account_id
asc"]` must fail the **build** — not the run, and not the numbers.

**3. Exact-ordered sort keys and integer carries.** *s5.8 c7, s8. EXT.*
A float64 ranking value is a fine number and a terrible sort key: two accounts
differing in the last ULP sort differently under a different partitioning, which
moves the funding line, which changes 380 000 answers. Needs
`quantised(name, source=, scale=, mode=, ties=[...])`, and a refusal to take a
raw float as a `Sweep` order key. Same argument for the accumulator: **the
budget carry must be integer**, because float addition is not associative and a
re-partitioned run would fund a different set. Asserted in
`test_budget_carry_is_integer`; should be a framework check, not a project's
diligence.

**4. Fixed-length array carries.** *s5.8 c5 (144 fairness sub-budgets). EXT.*
144 reserved purses is a vector and `Sweep` carries scalars. This project dodges
it by ordering the reserved pass by segment and resetting a two-scalar purse at
the boundary (`allocation/fairness.py`), which works because the partition is a
single key. It will not work for reserves by segment *and* region. Not needed
for v1; needed before someone asks.

---

## Attribution over sets of rules

**5. `panel(...)` — a reduction over named alternatives, all evaluated, with
attribution.** *s5.2 (16 exclusions), s5.5 (7 caps), s5.7 (14 triggers). UGLY.*
Doc 03's answer is a waterfall of modules each overwriting `proposed_limit` with
a `min`, with the version chain recording the sequence. Three requirements break
it. s5.5 wants **every** cap's value on the record for all 4.1 M accounts, and a
waterfall discards a cap the moment a lower one overwrites it — the version chain
only shows values that *moved* the number. s5.5 wants ties recorded and broken by
a declared order, and in a waterfall a tie is invisible: the second `min` is a
no-op indistinguishable from a cap that did not apply. And the version chain is a
trace-mode artefact (doc 04 s4.2) where this is a production column for 4.1 M
rows.
`panel(members, reduce=, codes=, writes=, evidence=["*"])` with reducers
`min | max | any | all | min_headroom`. Members are scopes like branch arms, so
seven caps may all write `cap_value_c`. Closed vocabulary, so codegen is total.
The payoff is change scenario 7: a new deposit-balance cap is one module, one
list entry, one code — and it appears in `binding_cap_code`, in the simulation's
cap incidence and in every audit record, because all three read declared
membership rather than a hand-kept list. Four uses: exclusions (`any`), caps
(`min`), decrease triggers (`any` + severity rank), decrease floors (`max`).

**6. Codes are a registered family joined to panel membership.** *s5.2, s5.5,
s5.7, doc 04 s4.1. EXT.*
`codes.family(...)` maps member name to code and carries the tie order, severity
rank and client wording. A member without a code, or a code without a member, is
a build error. That is how a taxonomy stops being droppable — doc 04 s4.1 records
that a previous port dropped one entirely.

**7. `evidence=[...]` — declared, non-droppable outputs.** *s5.3 ("required
output, not a debugging convenience"), s5.6 ("answerable from this record
alone"), s9.1, s9.2. EXT.*
Taps are optional diagnostics a consumer may omit; evidence is not. A module's
evidence list is part of its frozen contract, a consumer that drops an evidence
column fails the build, and the reviewable artefact renders it. Doc 03 has taps
and nothing else, and "required output" expressed as a tap is a convention that
survives until the first person optimising a wide frame.

---

## Overlays

**8. `overlay_point(...)` — a generic-kernel module over a closed algebra.**
*s5.4, s6.6, s13.5. EXT.*
s13.5 asks the question: overlays change values (params), their composition order
is real logic (structure), the stack in force is evidence — and an implementation
answering that three ways in three places is unmaintainable. This sketch's single
answer: the **register is data** with a closed vocabulary (eight kinds, a
fixed-width conjunctive scope predicate, an apply order), which passes doc 08
s3.4's own test and therefore runs on a generic kernel — so overlays are a
*values* change: free, no compile, mid-cycle. The **point of application is
structure**, declared in Python as a module in the pipeline, so it renders,
diffs and appears in `lineage("proposed_limit_c")`. The **stack in force is
evidence**, as `adjustments_applied` / `adjustment_set_id` columns. Six uses
here: matrix, score and PD, grade boundaries, caps, the affordability buffer,
decrease cut-offs.

**9. Expiry is enforced at resolution, not reported.** *s4.1, s10.6, s13.5
("how is expiry enforced rather than reported?"). EXT.*
`artefacts.resolve(decision_date=...)` raises `OverlayLapsed` before the pipeline
is invoked. Not a warning column, not a monthly report — the cycle does not
start. `clm overlays expiring --within 30d` is the separate, advisory thing.
`artefacts/overlay_register.json` records a real instance firing and being
resolved in 4.5 hours.

**10. `shadow(...)` — one authored sub-graph, two evaluations.** *s5.4, s5.9
("the unadjusted matrix value alongside the adjusted one on every offer made" —
mandatory), s10.4, s10.5. UGLY.*
The requirement is not "each step emits its own unadjusted value". It is that
`proposed_limit_unadjusted_c` — what the *entire* matrix-and-cap chain would have
produced with the stack disabled — is on every record. Doc 03's two options are
both bad: thread a shadow value through every step by hand (doubles the authoring
and diverges at the first careless edit), or run the whole pipeline twice (2x,
and the allocation is then meaningless because the budget is shared).
`shadow(subgraph, neutralise={...}, suffix=, keep=[...])` duplicates the
sub-graph with one input neutralised. The structural guarantee is the point: the
shadow's graph *is* the primary's graph by construction, so they cannot drift.
Cost stated — about 1.9x on the shadowed region, bounded by `keep`. It composes
with the cycle-level form (`tables=resolve(..., overlays="off")`), and
`test_shadow_equals_a_disabled_stack_run` asserts they agree — because if they
ever differ, one is lying on every offer the Bank makes. Three uses: matrix and
caps, the scorecard, decrease thresholds.

---

## Artefacts and tables

**11. Tables need versions, states, effective dates, units, cell identity,
warn-level validators and cell diffs.** *All of s6.3, s13.9. EXT — the largest
gap.*
Doc 03's `Table` sketch is nine lines and labelled "provisional; lowest-
confidence part of this document". The 1 152-cell matrix is "the single artefact
that most determines the portfolio's shape". All five of s6.3's requirements are
missing. (a) **Cell identity as a value** — the lookup must emit `matrix_cell_id`
into the graph, because s6.3.4 requires every decision to name the cell it read;
a tap cannot do it, because a tap is optional and this is not. (b) **Units and
scale on the schema** — authored in rand, stored in int64 cents; a spreadsheet
and an engine disagreeing by 100x is silent and catastrophic, so the conversion
must be the loader's job. (c) **Validators that warn** — s6.3.2's monotonicity
check must flag, not forbid, because the matrix is deliberately non-monotonic in
places; accepted warnings need an acceptor and a date. (d) **Candidate state** —
s13.4's "neither live nor absent"; becoming live is an edit to the manifest, not
a deployment. (e) **Cell-by-cell diff with rand impact** — "policy sent a new
spreadsheet" is not a reviewable change.

**12. The key domain belongs to the version, not the schema.** *Change scenario
5. EXT.*
If the domain lives on the schema, adding a ninth utilisation band invalidates
every historical lookup. On the version, the 8-band cell ids stay resolvable for
as long as decisions made under them are auditable.

**13. `matrix_cell_id` alone is not an identifier — the spec's vocabulary is
wrong here.** *s4.3 against change scenario 5. AWK.*
An int32 cell id means nothing without the table version that defines the key
domain. This sketch records the int *and* `artefact_version` as evidence on every
lookup, so the pair is the real identifier. I did not invent a `CellRef` type
because s4.3 declares an int and the vocabulary is shared — but a 2031 replay
reading `matrix_cell_id = 814` without the version reads a different cell. Worth
fixing in the spec rather than papering over here.

**14. Bands are artefacts with declared closure.** *s5.4 ("without exception ...
'whichever the code does' is not an answer"). EXT.*
`bands(id, edges=[...], lower_closed=True, effective_dated=True)` with `.index()`
and `.near_edge()`. An `if` chain is readable by an engineer and not by an
auditor, and band-edge disputes are the recurring audit finding s5.4 names.

**15. Date arithmetic needs a calendar artefact.** *s6.7 (20 *business* days home
market; 30 and 45 *calendar* days elsewhere). EXT.*
`numpy.busday_offset` is not compilable in a kernel and carries no
effective-dated holiday calendar. A dense table keyed on jurisdiction makes "20
business days" two array lookups. Small, boring, and otherwise it becomes a
Python callback that breaks `score()`'s 200 ms budget.

---

## Effective dating and replay

**16. `decision_date` is a shared param, and "today" must be unreachable.**
*00 s7.3, s10.13, s8. EXT — and a departure from the spec's s4.3.*
One run, one decision date — true for a cycle, a request and a backtest alike.
The payoff is that a step *physically cannot* read a per-record date that drifts
toward "today", and artefact resolution becomes per-invocation, which makes
effective dating structural rather than conventional. Needs a lint to finish:
no `date.today()`, `datetime.now()` or `time.time()` under `modules/` or
`pipelines/`. Doc 07 s6 has the right lint table and not this row.

**17. The artefact set is an invocation argument.** *s5.10 r1-r2, s10.1, s10.3.
EXT.*
`apply(frame, params=, shared=, tables=)`. Doc 03 has `params=` and `shared=`,
and mentions `tables` only as a step parameter name inside the provisional Table
sketch. Making the artefact set an *argument* is the single decision that makes
simulation one implementation rather than two: `apply(..., tables=live)` against
`apply(..., tables=candidate)` and a join. If the matrix were imported, or read
from a baked-in path, or resolved from ambient state, a candidate run would need
a deployment and a second implementation would appear within a quarter — which
is exactly the history s5.10 records ("differed from production by 4.1
percentage points on funded counts").

**18. Params documents need effective dating and a cross-check against the
run.** *s4.1 ("the cycle does not run on last month's instruction"). EXT, and it
contradicts doc 08 s6.2.*
Doc 08 is emphatic that `origin` is "opaque ... never parsed". Right for
provenance, wrong here: the ALCO instruction must *fail to resolve* against a
`decision_date` it does not govern. The binding belongs in a validated field
(`cycle_month`), and `resolve_params` needs to cross-check a params field against
a shared value. It currently cannot see one.

**19. Replay is a chain, not a point.** *s5.8 c6, s6.5, s10.13. AWK.*
This pipeline's own output is an input to its next run, so replaying 2026-09 in
2031 requires the 2026-08 record, which required 2026-07. The framework cannot
fix this — it is a property of the business rule — but it must not pretend
otherwise: the audit record has to name the *prior cycle record id* alongside the
artefact versions, and s8's 7-year retention is a retention requirement on a
chain.

**20. Stage checkpointing keyed on the fingerprints doc 08 s8 already defines.**
*s8 restartability. EXT.*
Every `|` boundary is a checkpointable stage keyed on
`(snapshot_id, structure_fingerprint, artefact_set_fingerprint, params_digest,
stage_name)` — all five already computed for the audit record. Exposing them as a
checkpoint key is nearly free and is the difference between a three-hour restart
and a fifteen-minute resume.

---

## Contracts and the seam to project 02

**21. A frozen contract must cover `evidence` and `tables_required`.** *s10.3,
s10.14, s5.12. EXT.*
Doc 03 s5.1's `contract=` snapshots inputs and outputs. For `ProposedLimit`,
which both pipelines compose, it must also freeze the evidence set (or a consumer
drops the provenance) and the artefact dependencies (or a refactor quietly starts
reading a different table). See `contracts/proposed_limit.json`.

**22. Structural sharing must be assertable by identity.** *s10.3, s5.12. SAT,
and worth naming.*
Because modules are plain composable objects, the strongest possible test is one
line: `assert batch.ProposedLimit is realtime.ProposedLimit`. Comparing outputs
would pass for a copy that happened to agree on the corpus; identity cannot. This
is doc 02 s2 paying for itself, and it is why the s5.12 tolerance is about
*evidence* rather than about code.

**23. A shared capability's degraded mode is expressible in inputs alone.**
*s5.6, s10.14. SAT.*
`core.affordability.at(inputs={"instalment": "notional_instalment_c"})
.bind(affordability_buffer=0.18)` plus a project-owned evidence waterfall
upstream is the whole difference between origination and programme assessment.
Doc 03 s4.3 and s5.2 already give this. The only thing needed on top is #7 —
without non-droppable evidence the degradation becomes invisible behind the
shared interface, which is the failure s5.6 names explicitly.

---

## Gates, postconditions and smaller edges

**24. `Partition(...)` — a frame-tier split with a declared rejoin.** *s5.2's
stated tension. EXT.*
"Complete attribution, and no scoring, affordability or bureau work performed for
an excluded account." Doc 03's `Branch` short-circuits *inside a record*, which
is the wrong granularity — what needs skipping is a bureau join and an
affordability stage. Split across tiers: attribution is record tier (16 compares
over 4.1 M rows, about 40 ms — nothing to save by skipping), work avoidance is
frame tier (`Partition` on `is_considered`, expensive stages on the taken side
only, schema-completed union back so all 4.1 M get a record). `Filter` plus a
manual `Union` works and is not visible in the pipeline expression as a policy
decision. Three uses: exclusions, decrease-required, conditional path.

**25. `ensures=[...]` — postconditions in the graph.** *s9.2 ("structurally
impossible rather than merely tested for"), s10.11, s5.7 rules 1-3. EXT.*
Doc 03 has assertions only in tests. A test over a corpus is not "structurally
impossible"; a postcondition that fails the record is closer.

**26. And the honest limit of #25.** *AWK.*
`ensures` makes the *decision record* unconstructable without consent and an
assessment. It does not police the downstream ledger, which is out of scope (s12)
and is where the limit actually changes. The strongest true claim is "every limit
change this system authorises carries a consent record that pre-dates it". If the
ledger accepts a change from another source, no framework feature helps. Say that
to Compliance rather than implying otherwise in an acceptance criterion.

**27. Money is int64 cents; the spec's vocabulary says float64.** *doc 03 s1.2
against s4.3. AWK.*
Doc 03 s1.2 is unambiguous ("never float"); s4.3 declares `current_limit`,
`proposed_limit` and `applied_limit` as float64. I followed doc 03 and suffixed
them `_c`, breaking the spec's vocabulary in its most visible place. Two things
make it worth it: R500 downward rounding is exact in integers and approximate in
floats, and the budget carry must be integer anyway (#3). Needs a lint: `*_c` is
int64, and a money value in a signature not named `*_c` is an error.

**28. `fuse()` and `parallel()` must be build errors on a `Sweep`.** *EXT,
trivial.*
A sweep is inherently serial and order-dependent. `parallel(Allocate)` is not
slower, it is wrong — and doc 02 s3.3's promise that performance annotation "can
never change an answer" stops being true the moment `Sweep` exists, unless the
combinator refuses it.

**29. `impact()` should take two artefact sets, not two generations.** *s5.10,
s11.1. EXT.*
Doc 08 s5 defines `impact(active, candidate, sample)` over two compiled
generations. The change a policy analyst makes 90% of the time is an artefact
swap, which produces no new generation at all. `impact(pipeline, sample,
tables_a, tables_b)` is the same report with no compile — and it is then the same
call the swap-set is built on.

**30. The reviewable artefact must render panels, tables and overlays — not just
steps.** *s10.1 ("without an engineer at any point"), s6.3, doc 04 s6. EXT, and
it is doc 04's own top risk made harder.*
Doc 04 s6 proposes generating the view from module data: ordered steps with
descriptions, inputs, outputs, params. This project puts four things in front of
a reviewer that are not steps — a 1 152-cell table, a panel whose *membership* is
the policy, an overlay stack with an application order, and a swap-set. A view
that renders steps beautifully and shows the matrix as "a table lookup" has
failed the persona it was built for. E4 should use this project's artefacts,
because they are the hard case.

**31. `Aggregate` needs a windowed form over prior cycles.** *s5.8 c5 ("three
consecutive cycles"). AWK.*
Doc 02 s1 defers `window` from the frame tier. The fairness floor is a
three-cycle look-back over 144 segments, which is exactly window work. I wrote it
as `Aggregate(..., windows=3)` over the cycle-record history and it is the least
convincing frame expression in the sketch.

**32. Scorecard null bins are not `missing_as`.** *s5.3. EXT.*
Doc 03 s1's three null tiers are about a *step's* view of a missing value. A
scorecard needs the null to be a **bin with its own weight**, and needs two
distinct nulls ("insufficient history", "no bureau match") to bin differently.
`missing_as(0.0)` would silently score a 7-month-old account as a 24-month-old
one with a perfect record.
