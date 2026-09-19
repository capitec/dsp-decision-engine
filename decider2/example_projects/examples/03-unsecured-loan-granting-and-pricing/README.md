# 03 — Unsecured loan granting and pricing

This is a sketch. Nothing here imports and nothing runs; every function body is
`pass` with a one-line comment saying what it would do. What is real is the
*shape*: the directories, the artefact kinds, the way a value moves from an
application to an offer, and the boundary between what an engineer changes,
what a policy owner changes, and what a product manager changes. If you could
have any authoring surface `decider2` offered, this is the one this spec would
force you to build.

`FRAMEWORK-DEMANDS.md` is the companion document and it is the more important
one to read first if you only read one: thirty numbered claims about where
doc 03's provisional answer — a scalar `Table`, a `|` chain of one-in-one-out
modules, `Loop(should_continue, ...)` — would produce something nobody could
review at this project's scale. This document does not re-argue those thirty
points; it shows what the tree they produced looks like, walks one application
through it, and says which construct answers which requirement.

Client W runs through the whole of §5 of the spec, and runs through the whole
of this document too: existing client, 38, permanent employment, 41 months'
tenure, channel 2, requesting R120 000 over 60 months, on campaign 4471. The
numbers below are theirs unless stated otherwise.

---

## 1. The layout, and why it is shaped this way

```
adjustments/
  points.py                          # the 5 overlay POINTS — code, engineer-owned
config/flex_loan/
  adjustments/set-0041.json          # the overlay SET — data, Credit-Committee-owned
  interiors/
    cap_register.json                # 52-rule waterfall body
    minimum_viable_offers.json       # 5 suppression rules
  production.params.json             # every param, explicit, nothing inherited
contracts/
  flex_loan_granting.json            # the frozen pipeline interface
modules/
  affordability.py                   # consumed whole from project 02
  bureau_quality.py
  caps/{predicates,register,uplift}.py
  disclosure.py
  eligibility.py
  fraud_handoff.py
  offers/{objectives,viability}.py
  pricing/{credit_life,fees,instalment,probe,rate_card}.py
  scoring/{calibration_and_grading,scorecard,segment}.py
  solve/{partition,search}.py
  validation/independent.py
pipelines/
  flex_loan_granting.py              # the ONLY place execution order lives
  numerics.py
  reasons.py
schemas/
  application_input.json
tables/
  credit_life_rates.sample.csv
  flex_rate_card/{amount_bands.csv,cells.sample.csv,manifest.json,2026-09-B.patch.json}
tests/
  test_acceptance.py
```

Doc 07 §1 gives the canonical `decider2` project layout — `modules/`,
`pipelines/`, `config/`, `contracts/`, `schemas/`, `tests/` — and this project
follows it. Two directories exist that doc 07 does not name, both because
doc 07's two-class view of config (a *params* document, an *interior*
document) is one class short of what this spec needs.

**`tables/`, as a first-class sibling of `config/`, not a subfolder of it.**
Doc 03 §4's "Tables (keyed lookups)" is explicitly provisional and doc 08's
change classes put "table contents, for a generic-kernel kind" inside the same
row as params — free, no compile. That is true of *contents*. It says nothing
about an *axis*: `tables/flex_rate_card/amount_bands.csv` is 96 rows that
define the shape of the 63 360-cell array, and re-banding it (change scenario
1, 96 → 120 bands) changes the cell array's type, which has to be a staged
change, not a values-document swap. Splitting `tables/` out from `config/`
makes that visible before you open either file — see FRAMEWORK-DEMANDS #18–20.

**`adjustments/`, split across two directories, not one.** This is the
project's central structural decision — FRAMEWORK-DEMANDS #1 calls it "the
largest deviation from doc 03 in this project." An overlay is two artefacts of
different classes, changed by different people on different cadences:
`adjustments/points.py` is **code** — engineer-owned, reviewed under a
release, declaring *what may move and in which direction*, living beside
`modules/` because it is structure — while `config/flex_loan/adjustments/set-0041.json`
is **data** — Credit-Committee owned, changed "sometimes twice in a month"
(§6.3), declaring *whether, by how much, for whom, in what order, until when*.

Everything else follows the same three-class shape doc 08 §2 describes for
params/interiors/skeleton, and the tree makes the mapping explicit:

| doc 08 change class | this project | who, cadence |
|---|---|---|
| **values** | `config/flex_loan/production.params.json` | business users, any time, free |
| **interiors** | `config/flex_loan/interiors/cap_register.json`, `.../minimum_viable_offers.json`, `tables/flex_rate_card/*` | policy/product/Treasury owners, staged compile |
| **skeleton** | `pipelines/flex_loan_granting.py`, `modules/**/*.py`, `adjustments/points.py` | engineers, release |
| **adjustments** (a fourth class §6.3 names explicitly) | `config/flex_loan/adjustments/set-0041.json` | Credit Committee, ad hoc |

`contracts/flex_loan_granting.json` and `schemas/application_input.json`
complete doc 07's picture: the frozen interface project 04's campaign batch
depends on, and the input shape `decider build` needs to pick dtypes,
nullability and `Maybe`-vs-plain per column. Both fail the build on drift, and
neither is Python.

---

## 2. From application to offer set

Read `pipelines/flex_loan_granting.py` top to bottom and you have read the
flow — that is deliberate; the file's own docstring says "an engineer changes
this file under a release; nobody else can change it at all." Eleven stages,
five overlay points, one bounded search, one waterfall, in one pipeline
expression:

```python
granting = (
    eligibility.Gates
    | Halt.when("is_eligible", is_=False, reason="primary_reason_code",
                absent_because=ABSENT_GATE_DECLINE)
    | fraud_handoff.ConsentCheck
    | fraud_handoff.FraudDisposition
    | Halt.when("fraud_verdict_code", is_=FRAUD_DECLINE, reason=1210,
                absent_because=ABSENT_FRAUD_DECLINE)
    | bureau_quality.DataQuality
    | segment.AssignSegment
    | scorecard.Score
    | overlay(SCORE_SHIFT)
    | overlay(SCALING_CHANGE)
    | cg.Calibrate
    | overlay(ODDS_MULTIPLIER)
    | cg.Grade
    | overlay(BOUNDARY_SHIFT)
    | scorecard.Challenger.when(scorecard.in_challenger_sample)
    | CapWaterfall
    | overlay(CAP_REDUCTION)
    | affordability.Consume
    | Halt.when("affordability_verdict_code", is_=AFFORDABILITY_FAIL, reason=1310,
                absent_because=ABSENT_AFFORDABILITY_FAIL)
    | parallel(Map(over=permitted_terms, as_="term_months",
                   body=fuse(SolveOneTerm), capacity=9, ...))
    | viability.MinimumViableOffers
    | viability.Deduplicate
    | objectives.Recommend
    | FinalValidation
    | disclosure.Quotation
    | disclosure.Reasons
).with_numerics(FLEX_NUMERICS).named("flex_loan_granting")
```

Walking it for client W:

1. **`eligibility.Gates`** (`modules/eligibility.py`) evaluates all 14 gates —
   never short-circuiting each other, only the flow — and client W passes all
   14. A `GateSet`, not a `Branch` chain or a `|` chain of one-in-one-out
   modules, because §5.1 needs every gate's verdict, not just the first
   failure's.
2. **Consent, then fraud, then bureau data quality**
   (`modules/fraud_handoff.py`, `modules/bureau_quality.py`): a bureau
   enquiry is unreachable without valid consent —
   `requires=["consent_valid_for_bureau_enquiry"]` on
   `bureau_quality.DataQuality` makes skipping it a build error, not a
   runtime check. Client W's fraud verdict is `1 approve`, bureau DQ-0.
3. **Segment, score, calibrate, grade, with four overlay points threaded
   through.** `scorecard_id` 1011, score 618. Under adjustment set 41, both
   `overlay(SCORE_SHIFT)` and `overlay(ODDS_MULTIPLIER)` are scoped to
   `channel_code: [4]`, and client W is channel 2 — yet §5.4.1's own worked
   example applies both anyway. FRAMEWORK-DEMANDS #4 names this rather than
   resolving it ("either the example is wrong or the scope is wider than
   stated"), the argument for a recorded three-valued scope outcome instead
   of a boolean nobody can audit after the fact.
4. **`CapWaterfall`** (`modules/caps/register.py`) plus `overlay(CAP_REDUCTION)`
   — client W's `amount_cap` chain runs 500 000 → 150 000 (CAP-0100) →
   120 000 (CAP-0210) → 76 000 (CAP-0361) → 95 000 (CAP-0420, the one
   permitted raise); `term_cap` 72, bound by CAP-0118.
5. **`affordability.Consume`** (`modules/affordability.py`) takes project 02's
   assessment whole: `max_affordable_instalment` R4 350.00, verdict pass.
6. **The solve** — `parallel(Map(over=permitted_terms, ..., body=fuse(SolveOneTerm)))`
   runs `modules/solve/search.py`'s `MaximumAffordableAmount` once per
   permitted term (nine, minus 84 which `term_cap` 72 already removed). Client
   W's grade-9 twin at 60 months is the R50 000-not-R48 500 case
   `tests/test_acceptance.py` names explicitly.
7. **Offer set construction** (`modules/offers/`) — `viability.MinimumViableOffers`
   drops 72 months (already removed by `term_within_cap` before pricing, since
   `term_cap` is 72), and would have suppressed it again on total cost ratio
   and in duplum; `viability.Deduplicate` finds the 36/48/60-month offers at
   R95 000 too far apart on instalment to collapse; `objectives.Recommend`
   flags 36 months under `recommendation_objective=1` (`largest_amount`).
8. **`FinalValidation`** (`modules/validation/independent.py`) re-derives the
   recommended offer from nothing and asserts fourteen things about it, then
   **`disclosure.Quotation` / `disclosure.Reasons`** produce the five-way cost
   breakdown that must sum to the total, and the ranked reason set.

Client W receives **six offers** — 12, 18, 24, 36, 48 and 60 months — one
flagged recommended, with 72 months absent and its reason recorded, and 84
months never priced at all because `term_cap` removed it before the `Map`
ran.

Two functions at the bottom of `pipelines/flex_loan_granting.py` are the two
entry points, both closing over the same pipeline object — `invoke_batch`
(`pass  # resolve params + tables + adjustments as-of, then granting.apply(...)`),
run once a month over 14.2M clients, and `invoke_realtime`
(`pass  # runtime.generation().score(**application)`), run 55 000 times a
day. There is no third pipeline anywhere in this project — see §3.7 below for
why that has to be true rather than merely likely.

---

## 3. The hard parts, as the tree expresses them

Six constructs earn their own file or their own kind rather than being
expressed with `Loop`, `Branch`, `|` and a scalar `Table`. Each is named in
`FRAMEWORK-DEMANDS.md`; this section says what the construct actually looks
like on disk.

### 3.1 The bounded non-monotone solve

`modules/solve/search.py` and `modules/solve/partition.py`. This is what §5.8
calls "the heart of the specification," and `search.py` opens by tabulating
exactly where `Loop` stops being enough: `max_iterations` counts iterations,
where one iteration may probe zero, one or several times, so it cannot bound
"24 evaluations" without the author hand-maintaining a counter; and
"correct — the true maximum" is hand-written and unverified under `Loop`,
where under `Search` it is a theorem from the partition.

The theorem is `modules/solve/partition.py`'s whole content: the instalment is
not non-monotone, it is **piecewise monotone**, and every breakpoint is
already known to some artefact in the probe —

```python
AMOUNT_PARTITION = Partition(
    variable="offered_amount",
    unit=Money("100.00"),
    sources=[
        breakpoints_of("flex_rate_card.amount"),           # 96 band edges
        breakpoints_of("fees.initiation_fee"),              # 1 derived kink, R12 700
        breakpoints_of("credit_life.credit_life_premium"),  # inherited from the fee
        breakpoints_of("pricing.rate"),                     # the RATE_ADD_ON's scope edges
    ],
    require_monotone_within_segment=PriceCandidate.outputs("instalment"),
)
```

Given that partition, "the search returns the true maximum" stops being a
property somebody hopes holds and becomes a proof — exactly what §13.5 asks
for: a specification of "correct" checkable without evaluating all 4 981
candidates. The 10 000-application exhaustive comparison in
`tests/test_acceptance.py::test_exhaustive_agreement_on_current_card` then
changes what it means: instead of testing the search it becomes "a
**regression test on the breakpoint declarations**," per `partition.py`'s own
docstring — a failure names an artefact that forgot to declare a kink, not
"the search is wrong somewhere."

`modules/pricing/fees.py`'s `@breakpoints("offered_amount", at=["initiation_fee_ceiling_binds_at"])`
is the concrete case §5.7(b) forces: the fee ceiling binds at R12 700 exactly,
which sits *inside* rate band R12 000–R12 999, so a search that partitions on
rate-band edges alone is silently wrong for one band in 96. The kink is
derived (`ceiling_binds_at`), not written down as a literal, so change
scenario 10 (moving it to R13 300) needs no second edit anywhere.

**Termination and the tie-break** are declared, not implicit, in
`modules/solve/search.py`'s `Search(...)`:

```python
strategy=Strategy.bracket_scan_bisect(
    bracket=InverseBracket(inverse="instalment.advance_upper_bound",
                           rate="cheapest_in_domain", loadings="minimal",
                           tighten=param(2, ge=1, le=4, owner="credit_systems")),
    scan="segments_descending", within="bisect"),
budget=Budget(probes=param(24, ge=8, le=64, owner="credit_systems"),
              on_exhausted="no_offer", exhausted_outcome="refer",
              exhausted_queue=6, exhausted_reason=1420),
tie_break=(desc("offered_amount"), asc("total_cost_of_credit"), asc("term_months")),
binding=Binding(output="binding_constraint_code", vocabulary=[...9 codes...],
                attribute_from="domain_provenance"),
```

Bracket first — one provable upper bound from the cheapest rate anywhere in
the domain (`modules/pricing/instalment.py`'s `advance_for_instalment_relaxed`
deliberately over-estimates, so it is a *bound*, not a guess) — then a
descending segment scan (a segment's left edge is its minimum, so one
infeasible probe kills the whole segment), then bisection where monotonicity
is a theorem. Worst case is `1 + S + ceil(log2(W))`, a closed form over the
card's axis; `rate_card.py`'s `search_budget_is_satisfiable` validator
evaluates it for every `(term, grade)` **at card-staging time** against the
budget of 24 — `tables/flex_rate_card/manifest.json` records today's worst
case as 21. Acceptance criterion 4's *ever* is discharged before the card goes
live, not hoped for at runtime. The exhaustive check itself,
`MaximumAffordableAmount.exhaustive()`, exists as a **mode** of the same
object rather than a second implementation — same domain, probe and
feasibility test, only the strategy swapped for "evaluate every candidate" —
because a hand-written second search would let a shared misunderstanding of
the domain pass both.

### 3.2 The 52-rule cap waterfall

`modules/caps/register.py`, with its body in
`config/flex_loan/interiors/cap_register.json`. Not `SeedTermCap | ApplyIncomeCap
| ApplySectorCap` — doc 03 §3.2's own worked example — because that shape
fails three ways the module's docstring states plainly: order is config here
and `|` is code; a rule that applies and does not bind produces no version, so
the version chain (doc 03 §3.3) cannot represent §5.5's required distinction
between "did not apply" and "applied and did not bind"; and fifty-two modules
is fifty-two names in one pipeline expression, edited concurrently by four
teams — a merge-conflict surface, not a governance boundary.

Instead, a `Waterfall`:

```python
CapWaterfall = Waterfall(
    name="cap_register",
    ceilings={
        "amount_cap": Ceiling(seed=Money("500000.00"), direction=Direction.REDUCE_ONLY),
        "term_cap": Ceiling(seed=84, direction=Direction.REDUCE_ONLY),
        "worst_acceptable_grade": Ceiling(seed=12, direction=Direction.TIGHTEN_ONLY),
    },
    raise_permitted_by={"CAP-0420": uplift.CAMPAIGN_UPLIFT_AUTHORITY},
    effects=[Effect.REDUCE_TO, Effect.SCALE_BY, Effect.REDUCE_BY, Effect.TIGHTEN_TO,
             Effect.DECLINE, Effect.RAISE_TO_BOUNDED],
    predicates=predicates.REGISTRY, reads=[...15 fields...], capacity=64,
    tie_break=TieBreak.EARLIEST_IN_SEQUENCE,
    writes=["amount_cap", "term_cap", "worst_acceptable_grade",
            "amount_cap_bound_by", ..., "amount_cap_chain", ...,
            "cap_rule_verdicts", "cap_decline_reason_codes"],
    verdict_kind="five_valued",
    chain_kind="ragged",
    interior="config/flex_loan/interiors/cap_register.json",
    review_artefact="review/cap_register.md",
)
```

Because a rule is "N rules × fixed attributes with a closed effect vocabulary,"
doc 08 §3.4's own test — *"can one compiled loop evaluate every instance of
this kind, with the instance supplied as arrays?"* — puts it on a **generic
kernel**, which is what makes criterion 15 ("adding one rule, in any sequence
position, is a configuration change reviewed by its owner — not a release")
true by construction: sequence is a `seq` field, not array order, so a
reorder is a diff on 52 integers rather than a 52-element array move. A rule
row:

```json
{"rule_id": "CAP-0210", "seq": 17, "owner": "credit_risk_policy", "class": "policy",
 "acts_on": ["amount_cap", "worst_acceptable_grade"],
 "tiers": [{"when": {"predicate": "arrears_in_window",
                     "args": {"min_arrears_months": 3, "window_months": 12}},
            "effect": {"amount_cap": {"reduce_to": {"param": "severe_arrears_cap"}},
                       "worst_acceptable_grade": {"tighten_to": 8}}}, ...],
 "approval_reference": "CC-2026-04"}
```

The predicate is a registered id, `arrears_in_window`, resolved through
`modules/caps/predicates.py` — not an expression string — because doc 08 §3.2
settles that a derived value is a step referenced by id, never arithmetic in a
config document. `predicates.py`'s own docstring is honest about the cost:
sixteen predicates cover 52 rules today by being generic over their thresholds
(`months_employed_below(threshold)`, with `threshold` a `{"param": ...}`
reference rather than a constant), and whether that still holds at 80 rules is
FRAMEWORK-DEMANDS #17, marked **[UGLY — unresolved]**, not assumed fine.

The required output is not the final value but the whole chain plus the rule
that bound it — five-valued verdicts (`NOT_APPLICABLE` / `EVALUATED_NOT_BINDING`
/ `BOUND` / `COINCIDENT` / `DECLINED`), because a boolean cannot distinguish
"did not apply" from "applied and did not bind," and the fifth exists because
two rules reducing to the same number are neither of the other four —
`tie_break=TieBreak.EARLIEST_IN_SEQUENCE` says the earlier binds and the later
is `COINCIDENT`. CAP-0420, the one rule permitted to raise a ceiling, lives
in `modules/caps/uplift.py` as fifteen lines of code rather than a register
row, because the register's own `direction=REDUCE_ONLY` forbids raising — the
exception must be visible under a release, not editable by whoever touches
the register that quarter:

```python
CAMPAIGN_UPLIFT_AUTHORITY = raise_authority(
    rule_id="CAP-0420", ceiling="amount_cap",
    max_ratio=param(1.25, ge=1.0, le=1.5, owner="credit_committee"),
    absolute_ceiling=param(Money("250000.00"), owner="credit_committee"),
    not_above_class="regulatory",
    requires=["campaign_authorised"], may_not_affect=["term_cap", "worst_acceptable_grade"],
    records=["authority_reference", "authorised_value", "achieved_value", "restrained_by_rule_id"],
)
```

`not_above_class="regulatory"` is evaluated against the chain, not sequence
position, so it holds however the register is reordered — which matters,
because the four teams reordering it quarterly do not know it exists.

### 3.3 The 63 360-cell rate card

`modules/pricing/rate_card.py`, with its data in `tables/flex_rate_card/`. Doc
03 §4's provisional `Table` is a one-dimensional keyed lookup —
`tables.term.max_loan[term]` — and cannot express a banded axis with edges
authored in a companion file, cell-level attribution, pre-live validation,
cell-level diffing, or spreadsheet ingestion. All five are hard requirements
here (§5.7(a), §6.1). So instead, a `grid()`:

```python
FlexRateCard = grid(
    name="flex_rate_card", owner="treasury",
    cadence="monthly, plus mid-month patches on repo moves",
    source="tables/flex_rate_card/",
    effective_dated=True,
    axes=[
        Axis.banded("amount", edges="amount_bands.csv", unit=Money, count=96),
        Axis.dense("term_months", low=6, high=60, count=55),
        Axis.dense("risk_grade", low=1, high=12, count=12),
    ],
    cell=Axis.value("nominal_annual_rate", dtype="int32_bp100"),
    resident=True,
    emits=["nominal_annual_rate", "rate_cell_id", "rate_card_version",
           "amount_band_index", "amount_band_low", "amount_band_high"],
)
```

`tables/flex_rate_card/amount_bands.csv` is Treasury's own artefact, 96 rows
authored beside the cells — `Axis.banded(..., edges="amount_bands.csv")` reads
band edges from it rather than assuming them, which is what lets the solve's
partition (§3.1) stay correct after a re-band with no code change.
`rate_cell_id` on `emits=` is one `int32` (band index × 660 + term × 12 +
grade), so cell-level attribution is free rather than something a regulator's
question depends on somebody having remembered to log.

Five `@validates(FlexRateCard, ...)` functions in the same file are §6.1's
card-validation list, run against every version before it may serve — e.g.
`search_budget_is_satisfiable(card, tables)`, evaluating the closed form
above for every `(term, grade)`. The result is *part of the version*, not a
side effect of loading it: `tables/flex_rate_card/manifest.json` carries the
whole validation report alongside `dimensions`, `cells: 63360`, and a
`diff_against_2026_08_B` block with `repricing_impact_last_30_days` —
"Treasury sent a new spreadsheet" stops being an unreviewable change at
63 360 cells, because the artefact names which cells changed, by how much,
and what it would have done to the last 30 days' applications.
`band_edge_inversions_are_declared` is `severity="block_unless_declared"`
rather than a flat block, since §6.1 rule 4 requires the current card's 41
inversions to be *permitted but declared* — Treasury deliberately rewards
crossing R50 000 with 150 basis points, and an **undeclared** one blocks the
card.

`tables/flex_rate_card/2026-09-B.patch.json` is the mid-month path in the same
shape: a patch is a **version**, not an edit — `"patches": "2026-09-A"` — so
`2026-09-A` stays exactly as issued and an application in flight at the
generation swap completes on it (criterion 10). Its `remediation` block
records zero applications priced from the 1 104 cells that briefly breached
the statutory ceiling, because the ceiling artefact is itself effective-dated
from the gazette timestamp: the flow failed loudly rather than lending
unlawfully for the 2h56m it took to patch.

### 3.4 Overlays at five points, composing in a declared order

`adjustments/points.py` and `config/flex_loan/adjustments/set-0041.json`.
Covered in the layout section above as the project's central split; the point
worth repeating here is what that split buys concretely, because §5.4.1 states
six requirements and the file answers all six without new mechanism:

```python
CAP_REDUCTION = OverlayPoint(
    id="policy.ceilings",
    target=["amount_cap", "term_cap"],
    kind=OverlayKind.MULTIPLICATIVE | OverlayKind.ABSOLUTE,
    direction=Direction.REDUCE_ONLY,
    scope_axes=["channel_code", "segment_code", "grade_range", "product_code"],
    set=FLEX_ADJUSTMENTS,
    joins_chain=["amount_cap_chain", "term_cap_chain"],
    chain_attribution="adjustment_set",
    writes_unadjusted=["amount_cap_unadjusted", "term_cap_unadjusted"],
)
```

`direction` sits on the point, in code, reviewed under a release —
`REDUCE_ONLY` here, `INCREASE_ONLY` on the PD multiplier, `EITHER` on the
score shift, since a score shift is legitimately bidirectional and a cap
reduction is not. That is FRAMEWORK-DEMANDS #3's answer to §13.19 ("what
makes it impossible to bypass by defining a negative magnitude?"): a runtime
check would already have let the wrong artefact be approved, so the asymmetry
must be a property of the *point*, checked against the *set*'s magnitude at
set-validation time, before anything runs. `joins_chain=` is why "your cap was
R95 000, reduced to R76 000 by a policy overlay approved under CC-2026-31" is
answerable at all — the overlay enters the waterfall's own chain, attributed
to the set rather than to a rule. The set itself,
`config/flex_loan/adjustments/set-0041.json`, is the readable half and is
genuinely legible to a non-engineer:

```json
{"overlay_id": "ADJ-0327", "position": 5, "point": "policy.ceilings",
 "kind": "multiplicative", "magnitude": 0.80, "target": "amount_cap",
 "scope": {"channel_code": [4]},
 "description": "Amount cap x 0.80 on the call centre channel for the quarter.",
 "rationale": "Loss-given-default on channel 4 originations above R100 000 is 14pp worse than the book...",
 "owner": "S. Naidoo, Credit Risk Policy", "approval_reference": "CC-2026-31",
 "effective_from": "2026-08-01", "effective_to": "2026-10-31",
 "review_date": "2026-10-01", "enabled": true}
```

`position` is a column, sorted before application, so "the order is declared,
not emergent" is a property of the data, not of Python statement order — a
reorder is a diff on two integers, not a release. `enabled: false` at the top
of the file switches the whole stack off through the same kernel (criterion
20), because the set is a runtime array, not a compile-time constant — the
identity `x == x_unadjusted` holds with an empty set, no second implementation
needed.

### 3.5 The ragged 0..9 offer set

`modules/offers/viability.py` and `modules/offers/objectives.py`. Doc 03 has
no vocabulary for a module producing between zero and nine of anything —
`Loop` carries scalars — so this project needs `Collection[T]`, `MapFilter`,
`Collapse` and `Rank`:

```python
MinimumViableOffers = MapFilter(
    name="minimum_viable_offers", over="offers",
    rules=SuppressionRules(interior="config/flex_loan/interiors/minimum_viable_offers.json",
                           capacity=8, collect_all_reasons=True,
                           rank_by="core.reason_codes.severity"),
    writes=["offers", "suppression_reason_codes", "primary_suppression_reason_code"],
)

Deduplicate = Collapse(
    name="deduplicate", over="offers", key=["offered_amount"],
    within={"instalment": pct(param(0.02, ge=0, le=0.25, owner="product"))},
    keep="min_by:total_cost_of_credit",
    writes=["offers", "deduplication_decisions"],
)
```

`collect_all_reasons=True` matters because client W's 72-month offer is
suppressed twice over — total cost ratio 2.02 against 1.85, *and* scheduled
charges of R96 759.76 against a R95 000 advance — and a filter returning a
single bool would record one of those arbitrarily. `Collapse`'s key is
`["offered_amount"]` alone, not amount-and-instalment, which is why client
W's 36/48/60-month offers all survive at R95 000: same key, but instalments
far enough apart on the 2% `within` tolerance that none collapse.
`objectives.py`'s `Recommend` is the one part of this spec doc 03 answers with
no extension at all — a `Branch` reading a param picks one of three
fully-materialised implementations, all visible in `lineage()` whether or not
selected, with `branch_path` recording which fired as a compile-time
immediate. Change scenario 4 — switched Tuesday, reverted Thursday — costs a
params swap, no compile.

### 3.6 Final validation, genuinely independent

`modules/validation/independent.py`. §13.14 is direct about the risk: "how is
the final validation stage expressed so that it is genuinely independent of
the stages it validates, rather than re-running the same code and agreeing
with itself?" The answer is a negative lineage assertion rather than a second
hand-written implementation:

```python
FinalValidation = validator(
    name="final_validation",
    independent_of=[MaximumAffordableAmount, PriceCandidate, CapWaterfall],
    reads=["offered_amount", "term_months", "risk_grade", "decision_date", ...],
    shares_primitives={"core.instalment.annuity": "tests/corpora/annuity_compliance.csv",
                       "core.fees.piecewise": "tests/corpora/initiation_fee_compliance.csv",
                       "core.rounding.round_half_up": "tests/corpora/rounding.csv",
                       "core.dates.resolve_asof": "tests/corpora/effective_dating.csv"},
    assertions=[...14 assertions...],
    on_failure="withdraw_offer", failure_outcome="refer", failure_queue=7,
    raise_incident=True,
)
```

`independent_of=` is checked at build: `lineage(FinalValidation.outputs)` must
be disjoint from `steps(MaximumAffordableAmount) | steps(PriceCandidate) |
steps(CapWaterfall)`, and a single shared step fails the build naming it —
`tests/test_acceptance.py::test_independence_is_structural_not_procedural`'s
whole assertion is `assert FinalValidation.independence_holds()`. The
module's docstring names its own two tiers rather than pretending the
guarantee is absolute: Tier A (independent derivation — every value here is
re-read or recomputed from the chosen amount, term and grade alone) and Tier
B (four shared primitives, each with a Compliance-owned corpus in
`shares_primitives=`) — and it says plainly what Tier B does **not** catch: "a
bug inside a shared primitive is invisible to this stage." That residual is
stated rather than hidden, which is the difference between this and a
validator that re-runs 140 steps and agrees with all of them. The same
negative-lineage mechanism discharges criterion 2 a second way, in
`partition.py`'s `search_domain` step: caps narrow the *domain* the search
explores, never the *answer* it returns, so
`granting.writers_of("offered_amount", after=MaximumAffordableAmount) == []`
is one build-time query making §5.8(7)'s silent failure — an offer affordable
when evaluated but not at the amount finally written — unreachable.

### 3.7 Real-time and batch, identical

`pipelines/numerics.py` and `pipelines/flex_loan_granting.py`'s
`.with_numerics(FLEX_NUMERICS)`. This is the one place the sketch corrects a
contradiction across two framework documents rather than filling a gap: doc 03
§1.2 requires money as scaled int64 cents and measures `fastmath` divergence
of up to 17 ULP on 46–73% of rows, while doc 02 §3.3 makes `fastmath` a
per-kernel author choice and doc 02 §3.1 *excludes* a fastmath kernel from the
exact-agreement assertion by design. One author enabling it on one hot kernel
four files away would silently break criterion 11 ("real-time and batch
produce identical outputs... zero differences") without the equivalence
ladder ever noticing, because the ladder is built to look past exactly that
kernel.

```python
FLEX_NUMERICS = Numerics(
    money="int64_cents", accumulate="float64", rate="int32_bp100",
    rounding=Rounding.HALF_UP,
    fastmath=False, allow_decimal=False, allow_float_money=False,
    parallel_requires_no_reduction=True,
)
```

Declared at the **pipeline** level, so it overrides any per-kernel choice
inside it, and it is part of the structure fingerprint — changing it is a
release with a new audit identity, not a local tweak. `parallel_requires_no_reduction=True`
is checked statically: `parallel(Map(over=permitted_terms, ...))` around the
nine-term solve is legal only because the `Map` body has no cross-row
reduction, and a `prange` region containing a float accumulation would
otherwise change summation order between the batch and real-time paths in a
way no test would catch until a reconciliation run disagreed by a cent. This
is also the concrete reason money is int64 cents rather than float64 rands:
the spec's own worked failure turns on **four cents** (R48 600 → R1 560.04
against a R1 560.00 ceiling), exact in int64 cents and not reliable in
float64.

`tests/test_acceptance.py::test_realtime_and_batch_are_the_same_answers` and
`::test_all_four_rungs` are the fourth rung `assert_paths_agree` adds beside
doc 02 §3.1's three (`interpreted ≡ stepped ≡ fused`): `score()` and `apply()`
differ in fusion grouping and in the boundary they cross, so their agreement
is a claim the other three rungs do not make on their own — and this project
runs both ways for real, with no `if batch:` anywhere and no second pipeline
object.

---

## 4. What a non-engineer sees

Three surfaces, matched to the three roles doc 04 §1 names, none of which is
Python:

- **A Credit Risk Policy analyst** reads `review/cap_register.md` —
  `CapWaterfall`'s `review_artefact=` — generated from the executing interior,
  not a second description that can drift from it. This is acceptance
  criterion 17: confirming the register matches approved policy "without
  reading code."
- **Credit Committee** reads `config/flex_loan/adjustments/set-0041.json`
  directly — every field a required `rationale` (enforced by
  `requires=["rationale", "owner", "approval_reference", ...]` on
  `adjustment_set(...)` in `adjustments/points.py`), an `approval_reference`,
  and both an `effective_to` and a `review_date`, so an overlay nearing
  expiry without renewal is visible in the file itself.
- **Anyone asking "what ran on 3 March"** reads
  `config/flex_loan/production.params.json`, explicit and complete by its own
  header comment, so the question answers from one file rather than
  cross-referencing a commit hash against a config diff.

---

## 5. Explaining one decision

**To a declined applicant.** `modules/disclosure.py`'s `assemble_reasons`
folds every reason vector a stage already produced — `gate_verdicts`,
`cap_rule_verdicts`, `score_reason_codes`, `suppression_reason_codes` — ranked
by the reason registry's severity, capped at `reasons_communicated` (4) but
with all of them recorded. This is a fold over sources that already exist,
not a re-derivation at display time, because §9.1 requires the reasons to be
retrievable months later and a reason assembled from a record that never
stored its inputs cannot be. For client W's grade-9 twin asking "why is there
no 72-month option?": `viability.MinimumViableOffers` already wrote both
applicable reasons — total cost ratio 2.02 against 1.85, and the in-duplum
breach of R96 759.76 against a R95 000 advance — into
`suppression_reason_codes`, so a branch consultant reads the answer rather
than deriving it.

**To a regulator.** The question is "does that rate match the card you
published?" and the chain runs through `rate_cell_id` in
`modules/pricing/rate_card.py`'s `emits=` (attribution travels with the read,
never logged separately), `rate_card_version` (resolved against
`decision_date`, never "today" — a lint forbids `datetime.now()` anywhere
under `modules/` or `pipelines/`), and `tables/flex_rate_card/manifest.json`'s
validation block naming the statutory ceiling in force and the headroom
against it. Where an overlay moved the rate, `adjustments/points.py`'s
`RATE_ADD_ON` point keeps the card's own cell value and the add-on recorded
separately (`writes_unadjusted="nominal_annual_rate_card_value"`) — the literal
difference between "cell 18.50% plus a 75 basis point overlay approved under
CC-2026-22" and one unexplained 19.25%. `FinalValidation`'s
`records="validation_assertions"` (§3.6) stores all fourteen assertions'
computed and expected values on *every* application, so Internal Audit's
quarterly re-derivation of 200 approvals reads them rather than recomputing
them.

---

## 6. What changes when a value moves, versus when structure moves

| what moves | example in this tree | artefact | cost |
|---|---|---|---|
| a **value** | `total_cost_ratio_threshold` in `config/flex_loan/production.params.json` | params document | free — a bundle swap, no compile, `decider2.impact()` prices the effect first |
| an **interior** | a 53rd row in `config/flex_loan/interiors/cap_register.json`; a 6th rule in `.../minimum_viable_offers.json`; a re-banded `tables/flex_rate_card/amount_bands.csv` | interior / table document | one background compile and a staged swap — reviewed by the row's owner, not released |
| an **adjustment** | a new overlay appended to `config/flex_loan/adjustments/set-0041.json` | adjustment set | Credit Committee approval, ad hoc cadence — no compile, because the set is a runtime array the points already read |
| **skeleton** | adding a twelfth pipeline stage to `pipelines/flex_loan_granting.py`; declaring a sixth overlay point in `adjustments/points.py`; changing `FLEX_NUMERICS` | Python | rebuild, redeploy, new structure fingerprint, engineer review |

The four rows are not the same thing wearing different names. A values change
and an adjustment-set change are both "free" in compile terms, but approved
by different people under different authorities — and the adjustment set is
explicitly forbidden from being expressed as either a params edit or a
rate-card edit (§6.3: "an overlay may not be expressed as an edit to a rate
card cell, a grade boundary or a characteristic's points"). `CapWaterfall`'s
`reads=`/`writes=` are fixed in `modules/caps/register.py`, so adding rule 53
to `cap_register.json` cannot alter what `lineage()` reports: the interior is
bounded by an interface declared in code, exactly as doc 08 §3's "the
interface is code" property requires.

---

## A note on this pass

No outright inconsistency in the tree required a fix. The one place worth
flagging as deliberate, not a defect: §5.4.1's Client W applies a
channel-4-scoped overlay to a channel-2 client, and `FRAMEWORK-DEMANDS.md` #4
already names this as an open question about the spec rather than resolving
it — §2 above repeats that rather than picking a side.
