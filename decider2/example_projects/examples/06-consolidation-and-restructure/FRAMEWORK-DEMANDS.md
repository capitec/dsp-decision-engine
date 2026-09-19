# Framework demands — 06 consolidation and restructure

Twenty-four demands this sketch makes of `decider2`, each traced to the spec
section that forced it and the file where the sketch's answer lives. Marked:

- **SATISFIED BY DOC 03/04** — the proposed mechanism covers this, unmodified.
- **NEEDS EXTENSION** — the right shape, a specific addition is missing.
- **DOC 03 WOULD MAKE THIS UGLY** — following doc 03 as written is worse than
  what's below; stated whether that's a library problem, a framework gap, or
  unresolved.

---

### D1. `Search` must be a fourth combinator, not an encoding of `Loop`

**Spec:** §5.5–5.8; §13 Q1. **Status:** NEEDS EXTENSION.

Doc 03 §8 fixes one type (Module) and three combinators (sequence, `Branch`,
`Loop`). `pipelines/consolidation.py` argues a bounded search is not a `Loop`:
encoding it as one would hide the candidate plan inside a step body (so Credit
Risk Policy's ordering rules stop being editable and become code), force
per-candidate evaluation row-by-row (forfeiting the one vectorised kernel call
400 scenarios need to fit 900 ms), put a variable-length rejection record in a
loop carry (not expressible in compiled code), and collapse the budget to
`max_iterations` (conflating the count bound with the latency bound, which
must never decide the answer). Ask: `Search(plan=, evaluate=, admit=, select=,
budget=, invariant=, records=, evidence=)` as a fourth combinator — one
frame-tier stage, one whole-frame kernel call per tier, one frame-tier
reduction — with its own entry in doc 03 §8.

---

### D2. A frame-level invariant assertion, checkable without re-deriving anything

**Spec:** §5.6.1; AC 8. **Status:** NEEDS EXTENSION.

`search/plan.py` makes income and seven other assessment-level values constant
across the candidate frame by construction (one `Cross(source=
"assessment_invariants", broadcast=True)`, so nothing downstream *can* vary
them), then `Search(invariant=[...])` asserts it anyway with one `n_unique()`
per column, because AC 8 demands the property be provable, not merely true by
inspection. Neither doc 02's frame tier nor doc 03 has a declared
invariant-assertion primitive; it belongs beside `Search`'s other
declarations rather than as a hand-written check a project must remember.

---

### D3. The overlay stack is a fourth change class

**Spec:** §5.5 point 6; §5.8 point 2; §6.3; §9. **Status:** DOC 03 WOULD MAKE
THIS UGLY — here is what the sketch did instead.

The largest demand here. `config/overlays/register.json`: "AN OVERLAY IS NOT A
PARAMS DOCUMENT, AN INTERIOR OR A SKELETON CHANGE... It changes values like a
params document, composes in a DECLARED ORDER like logic, and is separately
approved and separately EXPIRING like neither." Up to four stacks apply in one
assessment (one per product); composition order is per-stack and declared,
never emergent; two overlays targeting one field at equal `order` must be a
validation error, never last-writer-wins; and the whole assessment must stay
runnable with the stack disabled as the **same compiled kernel**, since a
compiled step can only read what arrives as an argument.

What the sketch does: `overlayable(value, scope=[...])` marks a field as an
overlay target in the signature. Resolution produces both
`params.rate_add_on_bps` (resolved) and `params.base.rate_add_on_bps`
(unadjusted), both recorded; the stack-disabled run is `resolved.base()`
through the identical kernel. Ask: `overlayable()` as a sibling of `param()`
(doc 03 §4.4), `.base` guaranteed on any bundle with an overlayable field, and
a fourth doc 08 §2 change class — **adjustments**: free like a value, ordered
and collision-checked like an interior, separately approved and expiring like
neither.

---

### D4. A declared NOT-APPLICABLE state for outputs, distinct from a null input

**Spec:** §5.6.8; §5.7 requirement 6; §13 Q6. **Status:** SATISFIED BY DOC 03
WITH EXTENSION.

Doc 03 §1's three null tiers describe *input* missingness. This project needs
a third state for **outputs**: `card_term_months() -> Maybe[int]` returns
`na()` because product 20 has no term — a sentinel `0` would compute `0-31 =
-31`, which *passes* CON-INT-05's `+24` ceiling, silently approving every
transfer. The framework must refuse to compile a step reading a may-be-na
column without declaring `Maybe[T]`. The same `na()` serves "not-applicable is
distinct from passed" for intervention verdicts, and `Branch.modifies` needs a
`(name, "may_be_na")` tuple form so one arm can legitimately omit a value the
others produce (`search/evaluate.py`'s `PriceByProduct`). Ask: document
`na()`/`Maybe[T]` in doc 03 §1 as a fourth tier, and extend `Branch.modifies`.

---

### D5. A table lookup must be able to return a declared Row, not only a scalar

**Spec:** §13 Q6 verbatim; §5.6.4–§5.6.7; §6.1. **Status:** NEEDS EXTENSION.

Doc 03 §4's provisional `Table` sketch returns a scalar
(`tables.term.max_loan[term]`). Four arms falsify that: `flex_11`'s
`rate_cell` needs `.nominal_annual_rate`, `.cell_id`, `.card_version`,
`.band_edge`; `card_20` needs two cards with different key sets resolved
together (`promo_cell`, `reversion_cell`); `bond_40` needs a *margin*, not a
rate, normalised against a separately-supplied `shared.reference_rate`;
`drive_30` needs the lookup keyed on a value (`ltv_pct`) that is itself an
output of the same circular solve the lookup feeds. Ask: `tables.X.at(...)`
returns a declared `Row` (named fields, minimum `cell_id` plus a version), and
the *meaning* of what a card returns is explicitly product logic, never a
library normalisation.

---

### D6. Tiered evaluation re-invokes the kernel once per tier — the cost model needs to say so

**Spec:** §5.5 points 1–3; §5.6.2's cost paragraph. **Status:** DOC 03 WOULD
MAKE THIS UGLY — here is what the sketch did instead.

The spec's own arithmetic: 400 scenarios × 8–13 library invocations at ~200 µs
each inside 900 ms, against a stated whole-library NFR of <15 ms p99 *per
application* — "this project consumes that budget two hundred times over...
which way it resolves is diagnostic." `search/budget.py` tiers evaluation
into three declared rungs, each a whole frame through one kernel call, so
fixed per-call overhead is paid three times, not 400 — but still three times,
not once, which the per-application framing doesn't model: "at 400 candidates
that overhead is noise; at 40 (batch) it is not." Ask: state a per-tier fixed
cost alongside the per-row marginal cost, so a project can compute tier
affordability rather than discover it.

---

### D7. A new data-shaped module kind is needed: `ordering`

**Spec:** §13 Q2 verbatim; §5.5's H1–H8 table; AC 14. **Status:** NEEDS
EXTENSION.

Doc 08 §3.4 enumerates exactly three interior kinds: `ruleset` (codegen),
`decision_table` (generic kernel), `scorecard` (generic kernel). A
business-authored total order over a candidate set is none of them.
`search/orderings.py`'s `ordering(...)` needs: a closed vocabulary of
`{key, direction, filter, prefix}`; **frame-tier**, not record-tier, execution
(ranking is a sort); derived sort keys as registered features, reusing doc 08
§3.2's "a derived value is a step" mechanism directly; and — the part with no
analogue in the other three kinds — a framework-enforced **totality** check:
the declared tie-break is appended to every rule's key and an ordering whose
final key is not unique over the account frame is refused at config-validation
time, before any compile, with the offending rule named.

---

### D8. `Loop` needs a declared non-convergence outcome

**Spec:** §5.6.2 point 3; §13 Q12. **Status:** SATISFIED BY DOC 03 WITH
EXTENSION.

Doc 03 §8.3 requires `max_iterations` but is silent on what happens at the
bound — the implication is a silent stop. `SolveAdvance` (`search/
evaluate.py`) needs the opposite: hitting the bound in one row of 400 must not
abort the other 399, and must become an ordinary recorded rejection
(`RJ-SOLVE-01`), not a thrown exception. Three additions to `Loop`:
`exhausted="carry"` (publish the carry's last state rather than raise),
`writes_exhausted_flag=`, `writes_iteration_count=` — mechanical additions to
the existing signature, not a new construct.

---

### D9. Per-record diagnostics assume one row is one subject; this project's rows are hypotheses

**Spec:** §13 Q11 verbatim; §5.7 requirements 2–3; §9.5. **Status:** DOC 03/04
WOULD MAKE THIS UGLY — here is what the sketch did instead.

Doc 04 §4.1's tap model and §5.3's PII framing assume a record is the
applicant. This project evaluates up to 400 rows *about* one applicant, each a
hypothesis that may never become a decision: "here a record is a HYPOTHESIS
ABOUT A CLIENT, and there are 400 of them. The framework does not currently
distinguish those." The sketch stores the fourteen verdict columns plus
actual/threshold pairs as ordinary frame columns — not taps — one parquet file
keyed on `application_id`, with retention differentiated by tier (7 years
decision, 3 years per-scenario detail, then counts by reason). Ask: doc 04
needs a second retention/PII tier for evaluated-but-not-decided rows, distinct
from the decided-record tier §5.3 covers.

---

### D10. Selection over an evaluated frame must be a declared reduction

**Spec:** §5.8 (all of it). **Status:** NEEDS EXTENSION.

`search/select.py`: "an ad-hoc `.sort().head(1)` makes column lineage
unanswerable... the chosen scenario's instalment is THE output of the whole
flow and a lineage gap at the last step makes every lineage query return
`unknown`." `Select(by=, among=, top_n=, distinct_on=, tie_break=,
indifference_band_pct=, shadow=)` needs a declared schema transform — every
output column is the same column of one input row — so `lineage()` composes
across it. Doc 02 §1's shipped frame operators (join, aggregate, filter; sort,
union to follow) has no reduction-with-provenance primitive; needed alongside
D18's Fanout/Cross/Explode.

---

### D11. `contract=` needs a form checked across several implementing modules

**Spec:** §13 Q4, Q5; AC 15. **Status:** NEEDS EXTENSION.

Doc 03 §5.1 freezes one module's interface against its own history.
`contracts/product_offer.json` is one contract, four modules, each of which
must satisfy it, checked together at composition — "one contract per ARM
POSITION, checked across four modules." Its `implemented_by` list and
`invariants_asserted_at_composition` block (e.g. "an arm that tests
affordability against a different figure from the one it publishes here is a
defect") is the mechanism doc 03 §5.1 needs to add: a contract type validating
N declared implementers together at build time, not one module's `.interface`
against a snapshot of itself.

---

### D12. `fresh_until` needs multiple declarations per signature with input-dependent remedies

**Spec:** §4.8; §5.3. **Status:** SATISFIED BY DOC 03 WITH EXTENSION.

Doc 03 §1's null tiers cover missingness, not staleness. `bond_40`'s
`property_value` needs two freshness declarations with *different* stale
behaviours, where the remedy for one input's staleness depends on another
input's value: AVM sufficient under 24 months and below R300k; otherwise fall
back to physical valuation. `settlement_amount.py`'s `quotation_state` names
the vocabulary needed: `"degrade"` (weaker basis, assessment continues),
`"hard_stop"` (withdraw the product, not the assessment), `"withdraw"`
(product leaves routing, recorded). Ask: document `fresh_until(path,
on_stale=...)` with this three-value enum as doc 03 §1 vocabulary, distinct
from the null tiers.

---

### D13. Frame-tier operations need a stated determinism contract

**Spec:** §5.5 point 3; AC 2. **Status:** NEEDS EXTENSION.

`search/orderings.py`: "polars' sort must be declared STABLE, and the join
that broadcasts `provider_relief_total` back must have a declared row order.
An unstable sort produces a different plan, which produces a different
winner, from identical inputs... the frame tier currently does not [protect
against this]." The sketch threads `stable=True` through every `Join`,
`Aggregate` and `Plan.interleave` by hand. Ask: doc 02 §1's frame operators
need a documented default (stable or not) and a `stable=` parameter where it
isn't the default.

---

### D14. A new data-shaped module kind is needed: `objective`, with signature-shape validation at registration

**Spec:** §13 Q7 verbatim; §5.8 point 3. **Status:** NEEDS EXTENSION.

`search/measures.py`: "'scale between the best and worst evaluated' is
exactly what a data scientist writes when asked to make five measures
comparable" — review does not catch this. Enforced structurally instead: an
`@measure` step's declared inputs are checked against the frame schema at
registration, and a set-shaped input is refused unless the measure declares
`set_relative=True` (which then forces the evaluated set into the objective's
own recorded inputs — an explicit, audited exception, not a silent one). Doc
08 §3's `ruleset` validates a closed node vocabulary; nothing validates a
*referenced step's own signature shape* as an admission condition. Otherwise
`objective(...)` is `ruleset`'s sibling exactly as described — this one
property needs new machinery.

---

### D15. `core.obligations`'s "one capability, two shapes of answer" needs a seam, not a flag

**Spec:** §13 Q10 verbatim; §5.6.2 point 2. **Status:** DOC 03 WOULD MAKE THIS
UGLY — a library problem the framework should still name.

`search/evaluate.py`: "the library got the seam wrong." Projects 03/07 want
the scalar aggregate; this project wants the per-element annotation *and* the
aggregate, 400 times over. The fix is a library decision, not a framework
mechanism — ordinary `|` composition already makes "compose both, discard
one's output" free: two modules sharing steps, `obligations.treat` (record
tier, reusable at any cardinality) and `obligations.total` (frame tier,
`group_by`), rather than one capability whose output shape varies by a flag.
The framework demand: doc 08 has no admission check that would catch "this
capability's contract changes shape based on a parameter" before it ships, the
way `resolve_params` catches a composition key. Worth having.

---

### D16. No place for a project-owned library that other projects import

**Spec:** §13 Q4, Q15. **Status:** DOC 03/07 WOULD MAKE THIS UGLY —
unaddressed.

Doc 07 states two tiers: the Bank's core library, and one project's own
modules. `inventory/settleability.py` needs a third: settleability
classification and settlement amount derivation are wanted, unmodified, by
sibling projects 07 and 08 — owned by *this* project, not Credit Systems. The
sketch's answer is a `consol_core/` package with a frozen `contract=`,
imported by 07 and 08 — but ownership, release cadence, and whether a
`consol_core/` interior change needs the consuming projects' sign-off, are
unspecified anywhere in doc 07. Stated honestly rather than resolved: the
structural answer prevents the duplication doc 01 §5 records, but the
framework has no governance vocabulary for the tier it creates.

---

### D17. Interior documents and the skeleton need referential integrity, checked at stage time

**Spec:** §11 change scenario 1; §13 Q4, Q5. **Status:** NEEDS EXTENSION.

`products/routing.py`: adding a fifth product splits into an interior change
(a routing table row — free) and a skeleton change (a new pricing arm — an
engineer, a redeploy). "The routing row must not activate before the arm
exists, and nothing in the framework stops it: a routing table naming product
21 with no arm registered is a config that validates and then fails at the
Branch." Each half validates alone; the missing arm is not a schema
violation, it's a missing runtime binding. Ask: `stage()` (doc 08 §4) needs a
referential-integrity pass checking every value an interior document can
*produce* against every value the compiled skeleton can *consume* — spanning
one document and the active generation's skeleton, a check doc 08 §4 does not
currently perform.

---

### D18. The frame tier needs Fanout, Cross and Explode as declared operators

**Spec:** §5.5; §5.6.3. **Status:** NEEDS EXTENSION.

Doc 02 §1 ships join/aggregate/filter, with sort/union to follow. `search/
plan.py` uses three more: `Fanout` (one-to-many against a declared table —
one settlement set becomes up to four priced candidates), `Cross` (broadcast
a one-row invariant frame across N rows, D2's dependency), and `Explode`
(list-to-rows with a declared `when_absent=na()` policy for product 20's
termless axis). All three need declared schema transforms for lineage (doc 02
§5). `products/routing.py` argues the fan-out/branch *split* is "two existing
constructs used correctly," which is true architecturally — but the `Fanout`
class itself, with its own schema-transform contract, is not in doc 02's list.

---

### D19. The audit record needs a plan digest and a table-version *set*

**Spec:** §6.1's cross-table consistency requirement; AC 3. **Status:** NEEDS
EXTENSION.

Doc 08 §8 lists eight audit fields. `pipelines/consolidation.py` adds two more
this project needs: a **plan digest** — a content hash of the *ordered*
candidate list, so "was the search the same?" is answerable without re-running
the plan — and a **table version set**, not a single version, because
"September's vehicle valuation guide against August's Drive Finance rate card
is not wrong so much as unattributable," and only the resolved *set* makes an
incompatible combination detectable. (The overlay stack is the third addition;
that's D3.)

---

### D20. The audit record needs an acting-identity field, distinct from params origin

**Spec:** §5.9 ("a file where the approving authority is recorded as THE
SYSTEM is a finding"). **Status:** NEEDS EXTENSION.

Doc 08 §6.2's `origin=` is explicitly config provenance, not actor identity.
`restructure/concessions.py`'s `authority_outcome`: where a consultant's own
authority covers a level-1 concession, "the record carries the consultant's
identity and their held authority — not 'auto-approved'." Neither doc 04 §5.2
nor doc 08 §8 has a slot for who exercised discretion, as opposed to which
params bundle or generation served the request. Ask: an `actor=` field,
parallel to `origin=` (required, opaque, verbatim, never parsed), recorded
whenever a human — not policy alone — resolved a decision.

---

### D21. Re-derivation needs to be a first-class relationship between two decision records

**Spec:** §5.10; §13 Q13 verbatim. **Status:** DOC 03/04 WOULD MAKE THIS UGLY
— flagged honestly as unresolved by this sketch.

An offer accepted after `offer_valid_until` must be re-derived, with the
difference from the original attributable: which accounts' amounts moved,
whether a rate card version or overlay changed, whether the winner itself
changed. Doc 04 §5.2 describes one decision; doc 08 §4 describes one
pipeline's lifecycle. Neither has a concept for "this decision supersedes
that one, with a structured diff." The sketch does not build this — `fresh_
until` (D12) supplies the expiry *declaration*, and `bond_40/pricing.py`'s
closing note says `output/execution_package.py` (outside this file set) "is
written on that assumption" — but no file here links a re-derived record to
its predecessor. This is a real gap, not a solved problem wearing a citation.

---

### D22. `.at()` needs a conditional form: rebind only for rows matching a predicate

**Spec:** §5.6.6. **Status:** NEEDS EXTENSION.

Doc 03 §5.2's relabel layer rebinds a name for a whole module instance,
unconditionally. `policy/interventions.py`'s `rate_ceiling` needs to read
`nominal_annual_rate` for three products and `reversion_rate` for the fourth,
within **one shared instance** evaluated over a frame mixing all four:
`rate_ceiling.at(inputs={"nominal_annual_rate": "reversion_rate"},
when="product_code == 20")`. Without `when=`, the only doc-03-shape answer is
four copies of the intervention — the exact duplication `.at()` exists to
prevent everywhere else.

---

### D23. Satisfied outright: `.at()`'s relabel layer reuses `core.affordability` unforked

**Spec:** §13 Q9 verbatim; §5.9. **Status:** SATISFIED BY DOC 03.

Stated for balance — no extension needed. `AffordScenario = affordability.
Assess.at(inputs={"existing_obligations": "existing_obligations_after",
"instalment": "committed_monthly"})` and `AffordScenarioStressed`'s second
instance are doc 03 §5.2's third layer exactly as documented: "most projects
never reach the third layer; this project reaches it on page one." Nothing is
wrapped, copied or forked, and the invariant requirement (D2) is solved
upstream of this mechanism, not by it. The demand the mechanism was built for,
working.

---

### D24. Satisfied outright: per-instance params namespacing lets four product teams tune independently

**Spec:** §6.2. **Status:** SATISFIED BY DOC 03.

`config/policy/interventions.json`'s four sibling objects
(`interventions_flex_11`, `interventions_card_20`, `interventions_drive_30`,
`interventions_bond_40`) plus each product's own params namespace are doc 03
§4.1 exactly as specified: a param change's blast radius is bounded to one
module by construction. Four owners, four cadences, one shared intervention
set, zero collisions — delivered with no addition required.
