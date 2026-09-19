# 06 — Consolidation and restructure: what the sketch is and why it is shaped this way

This directory is a **shape**, not an implementation. Every `pass  # ...` body is a
placeholder; nothing here runs. What is real is the *authoring surface* — the
directory layout, the module boundaries, the config/interior split, and the
invented constructs (`Search`, `ordering`, `objective`, `overlayable()`, `na()`,
`fresh_until()` with multiple stale behaviours) that this project needed and that
`decider2`'s current docs either don't have or only gesture at.

The companion file, `FRAMEWORK-DEMANDS.md`, is the more important of the two: a
numbered list of concrete things this sketch asked of the framework, each traced
to the spec paragraph that forced it. This file explains the *shape itself* — why
the tree looks the way it does, how a client's eleven accounts become a chosen
scenario, what a branch consultant is shown, and how "why scenario B and not
scenario D" gets answered from the record rather than from memory.

Read alongside `06-consolidation-and-restructure.md` (the spec) and
`decider2/docs/03-authoring-api.md` / `04-observability-and-governance.md` (the
framework proposal this sketch is stress-testing).

---

## 1. The directory layout, and why

```
config/
  objective/objectives.json        # Credit Committee — which objective, blended how
  overlays/register.json           # Credit Risk Policy / Credit Committee — the adjustment stack
  policy/interventions.json        # Credit Risk Policy — the 14 CON-INT thresholds
  rate_cards/{bond_40,card_20}.json, {drive_30,flex_11}.csv   # Treasury — 4 cards, 3 representations
  search/{budget,ordering_rules}.json   # Credit Risk Policy — the search's own tunables
  settleability/provider_rules.csv      # Credit Risk Policy + Settlements — per-provider facts
contracts/
  product_offer.json               # the frozen shape all four product arms must produce
inventory/
  baseline.py, settleability.py, settlement_amount.py
pipelines/
  consolidation.py                 # composition only — the whole flow in one file
policy/
  interventions.py                 # the 14 @intervention steps (code) over config/policy/interventions.json (values)
products/
  bond_40/, card_20/, drive_30/, flex_11/pricing.py   # four owners, four cadences
  offer_contract.py, routing.py
restructure/
  concessions.py                   # the forbearance variant
search/
  budget.py, evaluate.py, measures.py, orderings.py, plan.py, select.py
```

**The first thing to notice is what `config/` is organised by.** Doc 07 §1's
principle 3 says "config mirrors *pipelines*, not modules" — because a module
reused across several pipelines should get one config entry per pipeline. This
project has effectively one pipeline (`pipelines/consolidation.py`; restructure is
a `Branch` inside it, not a second flow), so that axis collapses, and the six
subdirectories under `config/` are organised by **owner and cadence** instead:
`objective/` is Credit Committee's, quarterly; `overlays/` is a *stack* that spans
several owners and expires; `policy/` is Credit Risk Policy's fourteen thresholds;
`rate_cards/` is Treasury's, monthly, patched mid-month; `search/` is Credit Risk
Policy's again, but a different cadence from `policy/`; `settleability/` is
jointly Credit Risk Policy's and Settlements'. Six real committees and teams sit
behind those six folders, and the layout makes that visible without opening a
file. Product-local config (`config/products/*.json`, `config/restructure/*.json`)
and per-module `contracts/*.json` beyond `product_offer.json` are referenced from
the code (`params="config/products/flex_11.json"`,
`interior="config/products/routing_table.json"`) but not included among the 29
files read for this sketch — a representative excerpt, not the whole tree.

**`contracts/` holds one file for a reason worth stating precisely.** Doc 03 §5.1
describes `contract=` as one module freezing its own interface against its own
history (semver). `contracts/product_offer.json` does a different job: it is
**one contract four separate modules must each satisfy**, checked together at
composition, not four modules each checked against themselves. That distinction
is FRAMEWORK-DEMANDS D11, and it is why the file's own header calls itself a
"FROZEN INTERFACE... for the four product arms" rather than a module's contract.

**`inventory/` is upstream of the search and does the shrinking that makes the
search affordable.** `inventory/settleability.py`'s own docstring says it
plainly: "Shrinking 61 accounts to 14 settleable ones is worth more than any
cleverness downstream — it takes 2^61 off the table before a single scenario is
generated." `settlement_amount.py` then prices each settleable account and
timestamps when that price stops being true (`fresh_until`, §3 below).
`baseline.py` computes the do-nothing position and the six-condition
short-circuit — deliberately wasteful (all six conditions evaluated, never the
first that settles it) because "we could only lend you R14 000" is a complaint
if a consolidation would have released R60 000, and the answer must be on file.

**`policy/` (code) and `config/policy/` (values) are two different things with
the same name on purpose.** `policy/interventions.py` declares fourteen
`@intervention` steps — arithmetic, so it is code, per doc 08 §1's rule that
config is a superset boundary, not a place to write comparisons. What lives in
`config/policy/interventions.json` is the *threshold*, its *range*, and its
*scope* — four per-product sub-objects rather than a single `scope=` field,
because doc 03 §4.1 namespaces params per module *instance*, and the four
product arms are four instances. A product team edits its own object on a
Wednesday without a Credit Risk Policy cycle (AC 9) — see
`interventions_flex_11`, `interventions_card_20`, `interventions_drive_30`,
`interventions_bond_40` side by side in that one file.

**`products/` is four owners in four directories plus two files that are not a
product.** `offer_contract.py` is `abstract=True` — "a module that declares
reads/writes and has no steps, existing so the contract has a Python object to
hang off... Nothing resolves through it at runtime and removing it would change
no behaviour — which is the property that distinguishes a schema from a
superclass." `routing.py` is the other non-product file, and it is where the
sketch answers spec question 4 explicitly: routing is *two* constructs (a
frame-tier `Fanout` for "which products may carry this set", one-to-many; a
record-tier `Branch` for "which arm prices this row", one-to-one), meeting at
`contracts/product_offer.json` and nowhere else. "There is no base product
module, no template method, no 'common pricing' the four arms specialise."

**`search/` is the centre of the project and the one directory doc 03 has no
vocabulary for.** `plan.py` (frame tier — candidate generation is a cross
product, sort, dedupe, truncation, all set-shaped), `evaluate.py` (record tier —
per-scenario affordability and pricing, one kernel invocation over the whole
tiered frame), `orderings.py` and `select.py` (two new data-shaped module kinds,
`ordering` and `objective`), and `budget.py` (the two-bound, tiered-degradation
declaration). `pipelines/consolidation.py` composes all five into one
`Search(...)` object and states outright: *"`Search` is a FOURTH COMBINATOR."*
That single sentence is FRAMEWORK-DEMANDS D1 and the reason `search/` exists as
a top-level directory rather than living inside `pipelines/`.

**`restructure/` is one file, not a parallel tree**, and that is the whole point
of it: "the machinery of 5.5 to 5.8 is reused unchanged... What actually changes
is ONE STAGE: what a candidate is." If restructure needed its own `search/`,
its own `products/`, or its own `policy/`, the machinery would not have been
general — it would have been one flow with a search-shaped hole in it.
`restructure/concessions.py` supplies a different `Plan` (concession
combinations instead of settlement subsets) and a different `Objective`
(sustainability, via `search/measures.py`'s `sustainability_12m` and
`expected_loss_vs_do_nothing`); everything downstream of "what is a candidate"
— evaluation, interventions-as-authority-checks, selection, the record — is the
same code path.

---

## 2. A walkthrough: from eleven accounts to a chosen scenario

`pipelines/consolidation.py` is deliberately readable as a single screen:

```python
Intake = (
    eligibility.Gates
    | module("con_elig_gates", name="consolidation_eligibility")
    | Branch("eligibility_route", {0: proceed, 1: debt_counsellor_route, 2: terminal, 3: refer}, ...)
)

Assessment = bureau.Normalise | Settleability | SettlementAmounts | income.Determine
           | deductions.Statutory | expense_norms.Apply | Baseline

ScenarioSearch = Search(
    name="consolidation_search",
    plan=ConsolidationPlan,       # 5.5 — frame tier
    evaluate=EvaluateScenario,    # 5.6 — record tier
    admit=PolicyInterventions,    # 5.7 — record tier, 14 verdict columns
    select=SelectWinner,          # 5.8 — frame tier reduction
    budget=InteractiveBudget,
    invariant=[...],
    records=[...],
    evidence=SearchEvidence,
)

Output = BeforeAndAfter | ExecutionPackage | RejectionExplanation | Selection
pipeline = Intake | Assessment | ScenarioSearch | Output
```

**Intake (5.1).** `CON-ELIG-01` (debt review) is a `Branch` arm in the
composition, not an early return inside a step — "a Branch, here, in the
composition — not an early return inside an eligibility step, where it would be
invisible in the rendered artefact." That single design choice is what makes a
debt-review client visibly *routed*, never silently *declined*, in
`pipeline.render()`.

**Obligation inventory and settleability (5.2).** `inventory/settleability.py`
is an ordered nine-class `ruleset`, `first_match`, with the class that
classified each account recorded as an output — "the contact centre question
'why was my store card not even CONSIDERED' is a different question from 'why
was it not settled'." Client nominations and exclusions are applied *here*, not
in the search, because an excluded account shrinks the bitmask index for free;
applying it downstream would be "the same answer at several hundred times the
cost."

**Settlement amount (5.3).** Six components, six separately-testable steps in
`inventory/settlement_amount.py`, assembled in one final step — "a single
`compute_settlement_amount` would be the thing a reviewer cannot check." The
closing third of the file is `fresh_until` and the quotation-expiry state
machine (`quotation_state`, `offer_valid_until`) — see §3's "quotation expiry"
entry below.

**Baseline and short-circuit (5.4).** `inventory/baseline.py`'s
`short_circuit_verdict` returns a three-way `NOT_NEEDED /
PREFERABLE_SHOW_BOTH / SEARCH_REQUIRED` — not a boolean, because the spec's two
condition lists are not complements: a client can fail "unnecessary" without
meeting "preferable", and encoding that as one boolean loses the third state.

**Candidate generation (5.5).** `search/plan.py` states the project's central
claim before any code: *"The plan is FRAME-SHAPED. The evaluation is
RECORD-SHAPED. The batch is not clients. The batch is hypotheses."* Settlement
sets are indexed as a `uint64` bitmask (`index_settleable`), four seeds are
always present (`SEED-EMPTY`, `SEED-NOM`, `SEED-FULL`, and each ordering's
prefixes), routing fans one set out to up to four priced outcomes
(`RouteToProducts = Fanout(...)`), terms are bracketed to four per (set,
product) by a closed-form estimate, and the whole plan is
`Plan.interleave(...) | Plan.truncate(...) | Plan.tier(...)` before a single
`Cross(source="assessment_invariants", broadcast=True)` makes income constant
across the frame *by construction* — "there is no operation left that could
vary it."

**Per-scenario evaluation (5.6).** `search/evaluate.py`'s `EvaluateScenario` is
one pipeline expression — `ReDeriveObligations | buffer | fuse(SolveAdvance) |
AffordScenario | PriceByProduct | Measures` — run once, over the whole tiered
frame, not once per candidate. `PriceByProduct` is an ordinary `Branch` over
`product_code`; only the taken arm executes per row, and the four arms
(`FlexConsolidation`, `CardBalanceTransfer`, `DriveRefinance`,
`BondFurtherAdvance`) share nothing but `contracts/product_offer.json`.

**Policy interventions (5.7).** `policy/interventions.py`'s fourteen
`@intervention` steps run as one `module(...)`, composition order irrelevant
because "a module interior is a pure DAG and nothing here reads anything else
here" — which is what makes "how many assessments would change if CON-INT-04
moved to 12%" answerable in isolation (spec requirement 5). The one visible
`.at()` rebind — `rate_ceiling.at(inputs={"nominal_annual_rate":
"reversion_rate"}, when="product_code == 20")` — is "the single most subtle
thing in this file, and it is VISIBLE."

**Objective and selection (5.8).** `search/select.py`'s `SelectWinner = Select(
by=Objective, among="viable", top_n=3, ...)` is a **declared reduction**, not an
ad-hoc sort-and-head, so lineage survives the one place doc 02 §5 warns it is
most at risk. `explain_pairwise` and `shadow_divergence` are the two steps that
turn "scenario 44 won" into "scenario 44 was chosen over scenario 17 because it
releases R412 more per month, at a total cost R9 100 higher, under an objective
weighting instalment relief at 0.6" — see §5 below.

**Output (5.10).** Consumed but not included in this 29-file excerpt:
`output/comparison.py` (`BeforeAndAfter`), `output/execution_package.py`, and
`output/explain.py` (`RejectionExplanation`). Their shape is fully constrained
by what upstream modules publish — the before-and-after table's columns are
literally the five columns `contracts/product_offer.json` calls "the
commensurable surface" — so their absence from the file list does not leave the
walkthrough incomplete; it means their content is already determined.

---

## 3. The hard parts, as the tree expresses them

### A bounded search over an exponential space, under a declared budget

Eighteen settleable accounts is 2¹⁸−1 = 262 143 non-empty subsets; ×4 products
×9 terms is 9 437 148 scenarios; the budget is ≤400 evaluations and ≤900 ms.
The tree's answer has four moving parts, and none of them is "try harder,
faster":

1. **Shrink the space before searching it** — `inventory/settleability.py`
   removes unsettleable accounts (61 → ~14) before a single scenario exists.
2. **Order, then truncate, never the reverse** — `search/orderings.py`'s eight
   `ordering` rules (H1–H8) are `config/search/ordering_rules.json`, an
   *interior* document Credit Risk Policy edits directly: `"id": "H9",
   "enabled": false, "enabled_from": "2026-06-01"` is change scenario 6 staged
   and dated, no engineer. `search/plan.py`'s `OrderAndTruncate` round-robins
   across enabled rules by `sequence` before truncating — "taking the first 400
   rows of 'all of H1's candidates, then all of H2's' means H8 never gets
   evaluated for a client with many accounts."
3. **Two bounds doing two different jobs** — `config/search/budget.json`:
   `candidates` (400) "decides the ANSWER. Must be reproducible";  `wall_ms`
   (900) "decides only whether we DEGRADE." Evaluation proceeds in three tiers
   `[120, 280, 400]`, the clock is read *between* tiers only, and a tier that
   starts is a tier that finishes — `on_wall_exhausted: "stop_at_completed_tier"`.
   Guaranteed termination is therefore structural: the plan is generated once,
   truncated to a fixed count, and evaluation can only stop at one of three
   pre-declared points.
4. **Record which scenarios were evaluated and why the winner won** —
   `pipelines/consolidation.py`'s `records=[...]` list on the `Search(...)`
   declaration names seven things a replay must pin, three of them (`overlay_stack`,
   the effective plan digest, `table_versions` as a *set*) beyond doc 08 §8's
   existing audit-field list. `search/budget.py`'s `classify_termination` gives
   the client a four-way answer — `SPACE_EXHAUSTED`, `COUNT_BOUND`,
   `TIER_DEGRADED`, `LATENCY_BREACH` — because "a search that returns three
   candidates is worse than a search that returns late," and both are different
   from "there was nothing more to find."

### Four heterogeneous product subflows, one flow, no copies

`products/flex_11/pricing.py`'s own header states the test directly: "Nothing
in this file is imported by products 20, 30 or 40, and nothing in this file
imports them. That is the test of whether 'four subflows in one flow' is real
or whether it is four copies with a shared header." What the four *do* share is
exactly two things: `contracts/product_offer.json` (five commensurable
outputs — `committed_monthly`, `total_cost_of_credit`, `horizon_months`,
`advance_or_limit`, `nominal_annual_rate` — "getting these five right is what
lets a 60-month unsecured loan, a revolving limit with a promotional rate, a
balloon-structured vehicle refinance and an 84-month sub-term inside a
240-month bond be RANKED AGAINST EACH OTHER") and the `credit_core` library
capabilities. Nothing else. Each arm carries its own rate representation
(absolute in `flex_11`; promo-then-reversion pair-plus-duration in `card_20`;
margin-over-reference in `bond_40`; a double-channel LTV/amount solve in
`drive_30`), its own `products/<name>.json` params namespace, and its own
`product_rejection_codes` distinct from the fourteen shared policy
interventions — "a product fact a product team can change on Wednesday" versus
"a credit policy decision with a Credit Committee behind it." Several products
can serve the same settlement set at once: `products/routing.py`'s
`ProductRouting` is `prioritisation="all"`, "ALL matching products, not the
first. The whole point," fanned out in the frame tier by `RouteToProducts =
Fanout(on="settlement_set_mask", table=ProductRouting, ...)` in
`search/plan.py`, then priced by an ordinary record-tier `Branch` in
`search/evaluate.py`'s `PriceByProduct` — the fan-out and the branch are
deliberately two different constructs at two different tiers, not one
mechanism doing both jobs.

### An objective that is itself configuration

`config/objective/objectives.json` declares five measures (`OBJ-01`…`OBJ-05`)
plus two restructure-only ones, and blends them by channel/mode in a `weights`
vector that sums to 1.0 — the *branch* blend weights instalment relief at 0.60,
the *contact centre* blend at 0.40, the *restructure* blend drops new-money
entirely and adds `consol:sustainability_12m` at 0.45 under a hard `constraint`
(expected loss must not exceed the do-nothing path — "a constraint invalidates,
a weight trades off, and trading this one off is how a restructure book is
destroyed"). `search/measures.py` explains why this is safe: every `@measure`
step is `absolute=True`, expressed as a ratio to the *baseline*, never
"scaled between the best and worst evaluated" scenario — because a
set-relative measure "makes the winner depend on which losers happened to be
evaluated," which means the budget tier would change the answer. The weight
vector itself is an overlay target (`ADJ-OBJ-011` re-weights instalment relief
from 0.40 to 0.60 for Q3, branch channel only) — "an overlay that changes the
winner with no logic change... every rule, table and line of logic stays
identical and a different scenario is selected." See §5 for how that decision
is defended after the fact.

### Rejection reasons for scenarios never chosen

`policy/interventions.py` is explicit that this is a different problem from a
declined application: "the evaluated scenarios are a 400-row frame, so the
rejections are COLUMNS OF THAT FRAME: fourteen verdict columns (int8:
PASS/FAIL/NA), and for each intervention an actual and a threshold column."
That is 42 columns × up to 400 rows per assessment — "about 130 KB per
assessment before compression, 780 MB a day at 6 000 assessments" — written as
one parquet file keyed on `application_id`, so "why did you not settle my
furniture account" becomes a millisecond filter rather than a data-team
request. `viability_verdict` keeps the asymmetry the spec demands: the
*verdict* is one value, the *reasons* are all of them, because a scenario
rejected by four interventions is a different escalation from one rejected by
a single marginal breach.

### Affordability recomputed up to 400 times, the expensive part held constant

`search/evaluate.py` states outright: "Nothing here knows the search exists.
That is the point: if this pipeline could tell it was inside a search, it
would be a second entry point into affordability and it would drift from the
first one." `AffordScenario = affordability.Assess.at(inputs={...})` is the
*unforked* library module, rebound only where a name genuinely differs (doc 03
§5.2's third layer). Income, expenses, statutory deductions and the seven other
names in the `Search(invariant=[...])` list are attached to the candidate frame
exactly once, by the `Cross(source="assessment_invariants", broadcast=True)` at
the very end of `search/plan.py` — "there is nothing in the plan that could
vary them," so the invariant is enforced by the *absence* of a mechanism that
could break it, checked anyway at runtime with one `n_unique()` per column
because "acceptance criterion 8 demands the property be PROVABLE, not merely
true." `AffordScenarioStressed` is the same module, a second instance, three
rebound inputs — doc 03 §5.2's observation that "most projects never reach the
third layer" is met on page one here.

### Determinism and replayability of a search, including the overlay stack

Three mechanisms, stacked: (1) `search/orderings.py`'s `RankAccounts` declares
every `Join`/`Aggregate` `stable=True`, because "an unstable sort produces a
different plan, which produces a different winner, from identical inputs" — a
requirement `search/plan.py`'s `Plan.interleave(..., stable=True)` repeats at
the truncation step; (2) `config/search/budget.json`'s tiering makes the wall
clock unable to decide the answer, only whether to degrade to a declared rung;
(3) `config/overlays/register.json`'s stack is pinned into the record —
`pipelines/consolidation.py`'s closing comment lists `THE OVERLAY STACK` and
`THE PLAN DIGEST` as two of the three fields it adds to doc 08 §8's list,
because "a replay under a different stack produces a different winner and it
reads as a defect rather than an explanation" without them. `ADJ-LTV-004` in
the overlay register is the concrete illustration of why this matters: applied
October 2025, review date March 2026, **still in force**, tagged
`"STATUS": "PAST REVIEW DATE AND STILL IN FORCE"` — an overlay that outlived
its own review date changes which scenario wins for every Drive Finance
refinance client until someone notices, and the register makes that fact
undeniable rather than requiring someone to remember it.

### The restructure variant: forbearance and authority limits

`restructure/concessions.py` reuses `Plan`'s vocabulary unchanged — `Plan.seed`,
`Plan.combinations(of="permitted_concessions", depth=param(2, ...))`,
`Plan.order_by`, `Plan.distinct`, `Plan.truncate`, `Plan.tier` — over a
different candidate element (a concession *combination* instead of a
settlement subset), which is the evidence the plan machinery is general rather
than consolidation-shaped. Every concession's `npv_cost` decides its
`authority_level_code` (1 consultant → 4 Credit Committee), and
`authority_outcome` is explicit that "a file where the approving authority is
recorded as THE SYSTEM is a finding" — the record must carry the *identity* of
whoever exercised discretion, not just which code ran (FRAMEWORK-DEMANDS D20).
`distressed_classification` runs *before* the concession is granted, not after
a provisioning report finds it, because sequencing — not computation — is the
requirement. `stress_acknowledgement_required` prices "passing standard,
failing stressed" as an input to the *authority* determination rather than a
flat prohibition — "a restructure that only works if nothing else goes wrong
is not a restructure."

### Settlement quotations expiring

`inventory/settlement_amount.py`'s `quotation_state` and `offer_valid_until`
resolve `min(earliest quotation expiry, 21 days, valuation validity)`, with
`fresh_until("quotation.expiry_date", on_stale="degrade")` meaning an expired
quotation does not fail the assessment — the account degrades to "quotable but
not quoted," the outcome is marked conditional, and the search continues. On
product 40 this is not the exception path: registration takes 6–10 weeks, so
"every settlement quotation in the set will expire before disbursement" and
`products/bond_40/pricing.py`'s closing note says `output/execution_package.py`
"is written on that assumption" of re-derivation being the normal path, not an
edge case. What the sketch does **not** resolve — see FRAMEWORK-DEMANDS D21 —
is what structurally links a re-derived decision to the one it supersedes; the
requirement is stated and the freshness *declaration* mechanism exists, but
the *relationship between two decision records* is not built here.

---

## 4. What a branch consultant sees

Not this codebase — a rendered artefact, per doc 04 §6, generated from module
data rather than read as Python or as 1000-line rule JSON. Concretely, for one
assessment, the consultant sees: the top three scenarios (spec 5.8 point 6,
`SelectWinner`'s `top_n=3`, `distinct_on=["settlement_set_mask",
"product_code"]` so three terms of the same product over the same accounts
never count as three alternatives); a before-and-after table whose six rows are
fixed by spec 5.10 and whose columns are exactly `contracts/product_offer.json`'s
"commensurable surface"; where the total cost rises, that fact stated "in rands
and as a percentage, with the reason" rather than buried under a prominent
instalment reduction; the execution package per settled account
(`settlement_amount` with its six-component breakdown, `quotation_expiry_date`,
the payment instruction); and — bulky, and required — every scenario evaluated
but rejected, with which of the fourteen interventions failed and the actual
value against the threshold. On product 40, the security warning
(`WARN-HL-SEC`, versioned) sits beside the comparison and the flow structurally
cannot select that scenario without a captured acknowledgement, because
`products/routing.py` gates the *routing* on `warning_acknowledged`, not the
selection — "an unacknowledged product 40 scenario is never GENERATED, so
there is no path by which one can be selected." Where the top two scenarios are
within the 2% indifference band, the consultant sees both presented as
materially equivalent, not one artificially ranked above the other — because
"a 0.4% difference in a modelled expected value is not a difference a
consultant should defend to a client."

---

## 5. How "why scenario B and not scenario D" is answered

Not by re-running anything. `search/select.py`'s `explain_pairwise` decomposes
the score gap algebraically, because every measure is absolute:

```
score       = Σ w_i · m_i
delta_score = Σ w_i · (m_i(winner) − m_i(other))
```

The step returns, per measure, the two raw values, the weight, the weighted
delta, and the rank of that weighted delta by absolute value — "the consultant
sees the top one; the ombud file gets all five." This is arithmetic, not a
sensitivity analysis, *because* `search/measures.py` structurally forbids
set-relative measures: if any component were scaled against the evaluated set,
the decomposition would stop being additive and "why did B win" would become a
data-science request instead of a record lookup. Two more facts are attached
to every such comparison: `objective_weights_used` **and**
`objective_weights_base` are both recorded, so a comparison made under
`ADJ-OBJ-011`'s re-weighted 0.60 is still explicable after the overlay lapses
and the base 0.40 is back in force; and `shadow_divergence` runs the *same*
comparison against `OBJ-05` (client outcome) regardless of which objective was
actually in force, so "what was the best outcome available to this client" has
an answer on file even when the objective in force that day was OBJ-01. "Why B
and not D" is therefore always one of two answers: either D failed one or more
of the fourteen interventions (and the rejection record names which, with
actual and threshold), or D was viable and lost on points — in which case
`explain_pairwise` names the point.

---

## 6. What changes when a value moves, versus when structure moves

The tree makes doc 08 §2's three change classes concrete, and then adds a
fourth the docs don't have a row for.

**A value moves — free, no compile.** Raising `CON-INT-04`'s anti-harm ceiling
from 15% to 18% for product 30 is one number in
`config/policy/interventions.json`'s `interventions_drive_30` object. Nothing
recompiles; nothing else in the other three products' objects is touched,
because doc 03 §4.1's per-instance namespacing bounds the blast radius to that
one object by construction.

**An interior moves — one background compile, staged, no redeploy, no
engineer.** Adding `H9` to `config/search/ordering_rules.json` (already staged
there, `"enabled": false, "enabled_from": "2026-06-01"`) or adding a row to
`config/products/routing_table.json` for a fifth product are interior changes:
Credit Risk Policy authors them directly, the vocabulary is closed
({key, direction, filter, prefix} for an ordering; N-rows-×-M-conditions for a
routing `decision_table`), and doc 08 §3's totality/emitter guarantees mean the
document either validates and compiles, or fails at validation with a named
offending rule — never at runtime.

**Structure moves — an engineer, a redeploy, a review.** Adding product 21's
pricing arm is a new Python module, a new `Branch` arm in
`search/evaluate.py`'s `PriceByProduct`, and — critically — the routing table
row from the paragraph above **must not activate before the arm exists**.
Nothing in doc 08's three classes currently stops a business-authored interior
row from outrunning the skeleton change it depends on; `products/routing.py`
names this explicitly and it is FRAMEWORK-DEMANDS D17.

**The fourth thing — an overlay moves.** `ADJ-OBJ-011` changes which scenario
wins by editing `search.objective.weights` for one quarter, for one channel,
with an approval reference and a mandatory expiry — and it is not a values
change (it composes in a *declared order* against other overlays, and it must
remain reversible: "the whole assessment must be runnable with the stack
disabled"), not an interior change (it isn't a rule document, it's a value
resolved through a *stack*), and not a skeleton change (nothing recompiles).
`config/overlays/register.json`'s own header says this outright: "Doc 08 2's
three change classes have no row for it, and this file is the evidence." What
the sketch does instead — `overlayable(...)` marking a field as an overlay
target, `params.base` always retained beside the resolved value, the stack
resolved into params *before* any kernel runs so "running with the stack
disabled is the SAME compiled kernel over `resolved.base()`" — is
FRAMEWORK-DEMANDS D3, the largest single demand in the companion file.

---

## Fix made to the existing tree

`config/overlays/register.json`'s `ADJ-LTV-004` scoped itself to
`vehicle_age_band: ["7_9"]` with `base_value: 100.0`. That base value belongs to
the `4_6` band: `config/rate_cards/drive_30.csv`'s `ltv_cap` rows give `4_6` a
cap of `100` and `7_9` a cap of `85`, and `products/drive_30/pricing.py`'s own
docstring narrates the same overlay as tightening "the 4-6 band from 100% to
90%." The `scope` field was corrected to `["4_6"]`; nothing else in the overlay
was touched.
