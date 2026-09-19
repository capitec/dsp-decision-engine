# 07 — Credit limit management

This is a sketch: an authoring surface for `decider2`, worked through on one
of the hardest projects in the set. Nothing here runs — function bodies are
`pass  # one line saying what it does` — and the value of the tree is in what
it is *shaped* like: the layout, the names, the combinators chosen, the
things declared as data rather than buried in a function body. Read
[`FRAMEWORK-DEMANDS.md`](FRAMEWORK-DEMANDS.md) alongside this file: that
document argues for the choices; this one shows them assembled into a
project and follows data through it.

The spec is
[`../../07-credit-limit-management.md`](../../07-credit-limit-management.md);
section numbers below (`s5.8`, `s13.2`, …) refer to it. Doc numbers (`doc 03`,
`doc 04`, …) refer to `decider2/docs/`.

---

## The directory layout, and why

```
clm/
  vocabulary.py              # project-owned names, code families, the exact sort key
  sources/book.py            # frame tier: joins, roll-ups, the ragged-history reduction
  features/
    banding.py                # utilisation/mob bands as artefacts, not `if` chains
    temporal.py                # decision_date as a shared param; cooling-off; the business calendar
  eligibility/panel.py        # the 16 hard exclusions, attributed completely
  scoring/behavioural.py      # the 32-characteristic scorecard, its overlays, its shadow
  matrix/assignment.py        # the 1,152-cell matrix, the cycle dial, its shadow
  caps/
    exposure_and_income.py    # C2/C3/C4/C7 -- one module each, one shared interface
    observed_spend.py          # C5, shown in full as the model for a policy analyst's rule
    panel.py                    # the seven-cap reduction, the binding cap named
  affordability/
    evidence.py                # the evidence waterfall -- tiers A-E, haircuts, staleness
    assess.py                  # core.affordability reused, with a wider buffer and a staleness gate
  allocation/
    ranking.py                  # the per-account ranking value; the exact sort key
    sweep.py                    # the population stage -- Sweep, the ordered fold
    fairness.py                  # 144 segment reserves and the conditional sub-envelope
  decrease/
    triggers.py                  # the 14 decrease triggers, their own shadow
    reduction.py                  # per-trigger targets, the floor panel, the closure gate
    precedence.py                 # the client-level increase/decrease precedence
  offer/construct.py           # the minimum meaningful increase, consent as a precondition
  pipelines/
    programme.py                  # the monthly cycle -- and ProposedLimit, shared with realtime
    request.py                     # the event-driven path, same object, different inputs
    decrease_daily.py               # the decrease path, its own cadence, its own governance
  simulate.py                  # not a pipeline -- two invocations of the real one, and a diff
  artefacts/
    manifest.yaml                   # which version of which table was in force, when, by whose approval
    overlay_register.json            # the live dials, their scope, their review dates
    limit_matrix/{schema.json,2026Q3.csv}
    decrease_triggers.csv
    alco_instruction_2026-09.json    # a PARAMS document -- the monthly instruction, not code
  config/programme.production.json  # every other tunable, explicit and complete
contracts/proposed_limit.json  # the frozen interface both pipelines compose
tests/
  test_agreement_and_shadow.py       # the one-line identity test, and the shadow's guarantee
  test_allocation_properties.py       # the Sweep's acceptance criteria, as properties of the outcome
```

Three things about this layout are decisions, not defaults:

**It is organised by domain, not by doc 07's literal `modules/` skeleton.**
Doc 07 §1 says a module directory is "the unit of reuse and audit." This
project drops the `modules/` wrapper entirely: `clm/matrix/`, `clm/caps/`,
`clm/allocation/` *are* module directories, each one a reviewable unit. A
Credit Risk Policy analyst asking "where is the observed-spend cap" gets
`clm/caps/observed_spend.py`, not
`clm/modules/caps/observed_spend/{steps,params,__init__}.py`. The four-file
form doc 07 §3 reserves for a directory "big enough to want one" is never
reached here — every rule fits doc 03 §1.1's one-artefact target, a function
with a `param()` default and a docstring, in one file.

**`clm/vocabulary.py` registers what this project invents, not what it
renames.** Doc 03 §5.2 treats a vocabulary map as the layer for *systematic*
renames against a shared library; this file is different — it declares `*_c`
for money, `*_code` for a registered taxonomy member,
`EXCLUSIONS`/`CAPS`/`DECREASE_TRIGGERS`/`ALLOCATION_OUTCOMES`/`ENVELOPES` as
`codes.family(...)` instances, and the one exact sort key
(`RANK_KEY = quantised(...)`) every ranking and allocation step reads. It also
declares `distinct_from={"behaviour_grade": "risk_grade"}` — the library's
application grade and this project's behavioural grade are different things,
and a shared module must not be allowed to bind one to the other by
name-matching alone (doc 03 §2.1's "ambiguous" build error is for exactly
this case).

**`clm/artefacts/` and `clm/config/` hold two different kinds of document.**
`manifest.yaml` and `overlay_register.json` are *artefacts* — data with a
version, a state (`candidate | live | superseded`), an approval reference, an
effective range. `alco_instruction_2026-09.json` and
`programme.production.json` are *params documents* — "values behind fixed
types" (doc 08 §2), applied by a bundle swap, no compile. Conflating them is
the mistake FRAMEWORK-DEMANDS #18 catches doc 08 §6.2 making: an artefact
needs a validated, cross-checked field (`cycle_month`); a params document's
`origin` is deliberately opaque. Two directories, two lifecycles, two owners
— and `contracts/proposed_limit.json` sits outside `clm/` entirely because it
is the seam to project 02: doc 03 §5.1's `contract=` plus the `evidence` set
and `tables_required` (FRAMEWORK-DEMANDS #21), frozen so the two pipelines
that compose it can never silently drift apart.

---

## A monthly programme run, stage by stage

Read [`clm/pipelines/programme.py`](clm/pipelines/programme.py) top to bottom
and you have read the whole of spec §5 — every stage from §5.1 to §5.9 is one
line of the pipeline expression, in the order it runs:

```python
programme = pipeline(
    Book
    | Exclusions                     # all sixteen, complete attribution, ~40 ms
    | Eligibility
    | Partition(
        on="is_considered",
        taken=ProposedLimit | Ranking,
        skipped="passthrough",
        rejoin=Union(schema="schemas/cycle_record.json"),
    )
    | AttachClientState
    | Precedence
    | Construct                      # minimum meaningful increase, BEFORE ranking
    | FairnessReserve                # 15% of the envelope, by segment
    | Allocation                     # Sort | Sweep -- the population stage
    | ConditionalPass                # the 213 000 on the conditional path
    | ApplyIncrease
).with_vocabulary(vocabulary)
```

Follow one account (grade 3, 42% six-month utilisation, 40 months on book,
Everyday Card) through it:

1. **`Book`** ([`clm/sources/book.py`](clm/sources/book.py)) joins the ragged
   cycle history, the client-level exposure roll-up, and — critically — the
   prior cycle's own output (`PriorCycle`), because hysteresis needs "what did
   we decide last time" as an ordinary effective-dated input.
2. **`Exclusions`** ([`clm/eligibility/panel.py`](clm/eligibility/panel.py))
   evaluates all sixteen `X01`–`X16` predicates regardless of whether one
   already fired, because s5.2 requires the *complete* exclusion set, not the
   first one found. None fire; `is_excluded = False`.
3. **`Eligibility`** computes `is_considered = not is_excluded` — `True` — so
   this account rides the `taken` side of the `Partition`. 1.43 M excluded
   accounts take the `skipped` side and are schema-completed back in with
   their exclusion codes intact, never touching scoring, the bureau join, or
   affordability.
4. Inside `ProposedLimit` (defined once, reused verbatim by the realtime path
   below): `Banding` places it in band 5/5 (40–55% utilisation, 36–59 months);
   `Scoring` produces `behaviour_grade = 3`; `Matrix` reads cell `(3, 5, 5,
   20)` — `multiplier=1.45` on the 2026Q3 sheet
   (`clm/artefacts/limit_matrix/2026Q3.csv` line 10) — giving an uncapped
   target of `current_limit_c * 1.45`; `IncomeEvidence`/`ExpenseBasis`/
   `Obligations`/`Assess` run the degraded-evidence affordability assessment;
   `CapChain` reduces seven caps to one `proposed_limit_c`, rounded down to
   the nearest R500, naming the binding one.
5. **`Precedence`** checks whether a decrease fired on this client's *other*
   account today (that path already ran — see below); it did not, so the
   increase is not suppressed. **`Construct`** then checks the minimum
   meaningful increase *before* ranking (deliberate — see §7 below).
6. **`FairnessReserve` → `Allocation` → `ConditionalPass`**
   ([`clm/allocation/`](clm/allocation/)) is the population stage: this
   account gets a `rank_key`, and whether it is funded depends on 707,999
   other accounts, not on anything about this one.
7. **`ApplyIncrease`** cannot construct a decision record without a consent
   record predating it and an affordability assessment — `ensures=[...]`
   makes that structurally true, not merely tested.

Nothing in that list is a mode switch or a duplicated function — one pipeline
expression, and every stage traces to one named requirement in the spec.

---

## A real-time client request — and why it must agree with the programme

[`clm/pipelines/request.py`](clm/pipelines/request.py) is short, and its
header states the whole design in one line: **"THE ONLY DIFFERENCES FROM THE
PROGRAMME ARE DECLARED INPUTS."**

```python
from clm.pipelines.programme import ProposedLimit

request = pipeline(
    Exclusions | Eligibility | RequestHead | ProposedLimit | RequestTail
    | Construct | ApplyIncrease
).with_vocabulary(vocabulary)
```

`ProposedLimit` here is not a copy, not a re-export, not a module with the
same name in a different file — it is the identical Python object `import`ed
from `pipelines/programme.py`. This is what makes
[`tests/test_agreement_and_shadow.py`](tests/test_agreement_and_shadow.py)'s
first test a one-liner:

```python
def test_the_two_paths_share_one_object():
    assert batch.ProposedLimit is realtime.ProposedLimit
```

That is checked by **identity**, not by comparing outputs on a corpus — a copy
that happens to agree today is exactly the failure mode s10.3 warns about, and
identity is the only test a copy cannot pass by accident.

The differences between the two paths are exactly three, and every one of them
is an input, never a fork:

| | programme | request |
|---|---|---|
| `path_code` | 1 | 2 — the exclusion panel's `applies_on` drops X11/X12; `cooling_off_windows` gives X09 a shorter window, keyed on the same `path_code` column |
| income basis | deposit-derived, tiers A–E | fresh declared income, via `RequestHead`'s `declared_income_requires_verification` |
| after `Ranking` | subject to `Allocation` (the budget) | nothing — a client who asks is served (s5.11) |

Two things make this survivable rather than fragile:

- **`path_code` is a table key, not a branch.** `CoolingOffWindows` in
  [`clm/features/temporal.py`](clm/features/temporal.py) is keyed on
  `(change_type_code, product_code, path_code)`, and the exclusion panel's
  `applies_on="path_code"` lets individual members declare which paths they
  apply on. `temporal.py` says it plainly: *"The two paths therefore share
  one graph and differ only in a declared input — which is what makes the
  s5.12 agreement test a statement about evidence rather than about code."*
- **The realtime entry point bypasses polars entirely** (doc 02 §3.4). The
  artefact set is "resolved once per generation and held, not per request —
  resolving 1,152 matrix cells and a 60-row overlay register per request
  would be the whole latency budget." That is doc 08 §4.1's `sealed`
  deployment mode: the first request after a deploy is no slower than the
  thousandth, because nothing compiles on the request path.

s5.12's tolerance — `|L′ − L| ≤ max(R500, 2% of L)` — exists only because the
*evidence* can differ between a monthly snapshot and a same-day request. Where
evidence is unchanged, `test_agreement_on_unchanged_evidence_is_exact` asserts
the two paths agree **exactly**, because they are the same compiled kernel.
What is explicitly *not* required — and `test_funded_outcomes_are_allowed_to_
differ` says so — is that the funded outcome agrees: the programme is
budget-constrained and a client request is not.

---

## The seven hard parts, as this tree expresses them

### 1. A population-level exposure budget constraining per-account decisions

Every stage before `Allocation` is per-account and embarrassingly parallel.
This one is not, and [`clm/allocation/sweep.py`](clm/allocation/sweep.py)'s
own docstring calls it "the centre of the project and the reason it is in the
set." The invention is `Sweep` — a `Loop` (doc 03 §8.3) with the *rows of a
sorted frame* as the iteration space instead of a counter:

```python
Allocate = Sweep(
    AllocateOne,
    seed=BudgetSeed,
    carries=["limit_budget_remaining_c", "rwa_remaining_c", "el_remaining_c",
             "funded_count", "tail_skips_used"],
    requires_order=["rank_key asc", "account_id asc"],   # checked at build, not at run
    emits_position="allocation_rank",
    halts_when="tail_skips_used >= params.max_tail_skips",
    name="allocation",
)

Allocation = Sort(by=["rank_key", "account_id"], descending=[False, False]) | Allocate
```

Three properties fall out of that shape, and they are the actual payoff:

- **The rank is an ordinary output**, with a producer and a version chain —
  not something joined on afterwards. Direct answer to spec question 13.2
  ("is the rank an output of the account's own evaluation, or something
  attached to it afterwards?").
- **The carried state at the moment of consideration is a column.**
  `evidence=[..., "limit_budget_remaining_c@entry", "funded_count@entry"]`
  means "when we reached you, R0 of R2.4bn remained" is on every one of
  708,000 records without re-running the cycle — what
  `test_non_selection_is_answerable_from_the_record_alone` checks.
- **It is deterministic by construction**, not by discipline: the carry at
  row *i* is a pure function of rows `0..i-1` in a build-time-enforced order
  (`requires_order`), and `RANK_KEY` in
  [`clm/vocabulary.py`](clm/vocabulary.py) is a `quantised(...)` int64, never
  a raw float — a float sort key would let two accounts differing in the last
  ULP fund a different 380,000th account under a different partitioning.
  `test_budget_carry_is_integer` and
  `test_ties_break_by_account_id_and_the_funded_set_is_bit_identical` assert
  exactly this.

**Ranking** ([`clm/allocation/ranking.py`](clm/allocation/ranking.py)) makes
ALCO's monthly choice of objective a `Branch` over all three arms — risk-
adjusted return, expected value, policy priority — so `lineage("ranking_value")`
correctly reports all three as live, and swapping objectives is a params
change, not a deploy. `incumbency_bonus` implements the anti-oscillation
requirement (s5.8 constraint 6) as a **per-record adjustment to the ranking
value**, not a post-hoc correction to the funded set — the docstring is blunt
about why: "a post-hoc swap would be an amount no cell of the matrix produced
and no audit could reproduce."

**Fairness** ([`clm/allocation/fairness.py`](clm/allocation/fairness.py)) is
a second, earlier `Sweep` (`ReservedPass`) over 144 segments (product × grade
× mob band), ordered by segment then rank so two scalar carries
(`reserve_remaining_c`, `current_segment_id`) stand in for what would
otherwise be a 144-wide vector carry — a trick that works, the file admits,
*because the partition is a single key*, and would not generalise to reserves
by segment *and* region (FRAMEWORK-DEMANDS #4). The **conditional list**
(213,000 accounts on the income-confirmation path) gets a *third* instance of
the same combinator, `ConditionalPass`, differing only in seed and
`charge_rate` — one combinator, reused, for three structurally different
allocation problems.

**"Eligible but ranked below the line" is distinguishable from "suppressed by
an overlay."** [`clm/vocabulary.py`](clm/vocabulary.py)'s
`ALLOCATION_OUTCOMES` family registers `below_line`, `fairness_capped`,
`tail_skipped`, `overlay_suppressed`, and `below_minimum` as distinct codes
because s5.8's own table shows an 80% cycle dial moves 41,000 accounts from
"would have qualified" to "below the minimum by the dial" while 96,000
*more* clients get funded at the same money — different answers to a client,
and only the dial-caused one is reversible. `suppression_reason_code` in
[`clm/offer/construct.py`](clm/offer/construct.py) is the one place that can
tell them apart, because it is the one place `additional_limit_c` *and*
`additional_limit_unadjusted_c` are both in scope — the direct payoff of
`shadow` (below).

### 2. Simulation sharing one implementation with production

[`clm/simulate.py`](clm/simulate.py) opens with a claim worth taking
literally: **"There is no simulation pipeline."** The entire mechanism is
three lines:

```python
a = programme.apply(book, params=p, shared=s, tables=live,      origin=...)
b = programme.apply(book, params=p, shared=s, tables=candidate, origin=...)
report = swap_set(a, b, on="account_id")
```

Everything else in the file — `resolve_candidate`, `self_check`, `REPORTS`,
`backtest` — is reporting *around* those three lines; none of it contains a
threshold, a cap, a multiplier, or a rank. That is possible for exactly one
structural reason (FRAMEWORK-DEMANDS #17): **the artefact set is an argument
to `apply()`**, never an import or a baked-in path. `tables=live` and
`tables=candidate` are the same call with a different fourth argument.

Two requirements a "simulation mode" flag could not satisfy:

- **The overlay stack is simulated, not bypassed.**
  `resolve_candidate(..., overlays="off")` runs the *whole book* with the
  stack disabled — the same mechanism Model Risk uses to see the scorecard's
  own behaviour, and the one that answers "what does this month's dial cost
  us," which the file's closing comment names as the question a policy
  analyst actually asks most often, ahead of "what does the new matrix do."
- **`self_check` is not optional ceremony.** Simulation, run over the
  *current production* artefact set against the *last production* snapshot,
  must reproduce that cycle "account for account, to the rand, including
  ranks and the funded set" (s10.2) — "a simulation that cannot reproduce
  production is not evidence about production," so this runs before every
  session, not once at release.

The 20-minute ceiling over 4.1 M accounts is explicitly an I/O budget, not a
compute one — "the pipeline over 4.1 M accounts is minutes of compute. The
cost is reading the book" — which is why the snapshot is a column-pruned
Parquet artefact both production and simulation read.

### 3. A 1,152-cell assignment matrix with overlay dials on a faster approval path

[`clm/matrix/assignment.py`](clm/matrix/assignment.py) is the artefact s6.3
calls "the single artefact that most determines the portfolio's shape," and
the file insists on five things doc 03's nine-line, "provisional,
lowest-confidence" `Table` sketch does not provide:

```python
LimitMatrix = table(
    "clm.limit_matrix",
    keys=("behaviour_grade", "utilisation_band", "mob_band", "product_code"),
    values={"multiplier": float, "max_increase_c": int, "min_increment_c": int},
    unit={"max_increase_c": "ZAR", "min_increment_c": "ZAR"},
    scale={"max_increase_c": 100, "min_increment_c": 100},
    validate=[dense(), bounds("multiplier", ge=1.00, le=2.00),
              warn(monotone_in("behaviour_grade", "multiplier",
                               direction="non_increasing",
                               holding=("utilisation_band", "mob_band", "product_code")),
                   code="MTX-W01", ...)],
    effective_dated=True, states=("candidate", "live", "superseded"),
    cell_id="matrix_cell_id", ...
)
```

- **Cell identity is a value in the graph**, not a side effect: `matrix_cell_
  id` is computed and carried forward as an ordinary output, because s6.3.4
  requires every decision to *name* the one cell it read, and a `tap` is
  optional where this is not.
- **Units and scale are declared on the schema**: authored in rand, scaled to
  int64 cents at load — see the identical unit/scale pair in
  [`clm/artefacts/limit_matrix/schema.json`](clm/artefacts/limit_matrix/schema.json):
  "a spreadsheet in rand and an engine in cents that disagree by a factor of
  100 is a silent 100x error."
- **`warn(...)` flags rather than forbids.** The matrix is deliberately not
  monotonic — a grade-2 account at 5% utilisation and 60 months gets a
  multiplier of 1.00 (spec §5.4) — so a validator that can only reject cannot
  express "flag this and let Credit Risk Policy decide." `manifest.yaml`
  records the acceptance: `MTX-W01: 11 warnings accepted`, `accepted_by:
  h.mokoena`, `accepted_on: 2026-06-18`.
- **A `candidate` state is neither live nor absent** (s13.4). The manifest's
  third matrix version has `state: candidate`, an author, a simulation id
  and no `effective_from` — "a candidate has no dates until Credit Committee
  approves it, which is an edit to this file and nothing else." Becoming live
  is that one edit, never a deployment — the whole answer to spec question
  13.4.

**The cycle dial** — "run the programme at 80% this month" — is
`ApplyMatrixOverlays`, an `overlay_point(...)` scoped to product, grade,
utilisation band, mob band, channel and `path_code`. `path_code` as a
declared scope key is what lets the same dial reach, or not reach, the
"faster approval path" — getting it wrong, the request-path header warns, "is
how the two paths start disagreeing." `ADJ-2026-114` in
[`clm/artefacts/overlay_register.json`](clm/artefacts/overlay_register.json)
is exactly this dial, live now, scoped to `{"path_code": [1, 2],
"product_code": [20, 21]}` — reaching the request path too, on purpose,
recorded as data rather than assumed. The register's `note` states the
failure mode the mechanism exists to prevent: *"a 70% dial applied during one
bad quarter, never reviewed, still quietly suppressing every increase in the
book three years later with nobody able to say who set it or why."*
`ADJ-2026-061` in the register's `lapsed` list is the mechanism firing for
real: a lapsed overlay halted the 2026-09 cycle at 03:14 until withdrawn.

### 4. Affordability in a degraded-evidence mode

[`clm/affordability/evidence.py`](clm/affordability/evidence.py) is explicit
about what it is *not* allowed to do: grow an `if is_programme:` inside
`core.affordability`. Instead, the whole difference between origination and
this programme is expressed as **inputs to the same shared object**:

```python
Affordability = core.affordability.at(
    inputs={"instalment": "notional_instalment_c"},
).bind(affordability_buffer=0.18)
```

— the identical module `core.affordability` project 02 composes, at 12%
buffer there, 18% here, because "origination underwrites against verified
evidence collected days earlier and this programme underwrites against a
salary pattern and a bureau file." The evidence waterfall — tiers A through
E, `EvidenceTiers` table, haircuts, staleness caps — sits entirely upstream
of that call, in this project's own code, never inside the shared capability.

**Degraded evidence routes to a conditional offer, not a failure.** The
staleness gate is its own `panel`, `reduce="all"`, over `income_evidence_
fresh`, `bureau_view_fresh`, `expenses_acceptable` and `obligations_stable`.
`affordability_route_code` in
[`clm/affordability/assess.py`](clm/affordability/assess.py) makes the
three-way split explicit in its own docstring: *"An account failing
affordability on the evidence available is not offered an increase at all. An
account passing on the evidence available but failing a staleness condition
is offered one CONDITIONAL on confirmation."* Tier D/E income, a stale bureau
view, or obligations grown more than 25% since the last assessment each push
an otherwise-affordable account to the conditional list rather than declining
it — which feeds [`clm/allocation/fairness.py`](clm/allocation/fairness.py)'s
`ConditionalPass` downstream, ranked and charged against the budget at its
22% expected conversion rather than at face value.

`IncomeEvidence`'s `evidence=[...]` list (`evidence_tier_code`,
`income_staleness_days`, `income_haircut_applied`, `deposit_account_id`, ...)
is frozen exactly like `contracts/proposed_limit.json`'s `evidence` block —
non-droppable, so a consumer cannot take the income figure and leave the
provenance behind (FRAMEWORK-DEMANDS #7).

### 5. Increase and decrease paths co-existing

These are **two pipelines** — `clm/pipelines/programme.py` (monthly,
consent-gated, budget-constrained) and `clm/pipelines/decrease_daily.py`
(daily, unilateral, notice-gated) — not a `Branch` inside one, because the
two directions differ in governance, not just in sign.
`decrease_daily.py`'s own comment states the reasoning: "Decreases are not
subject to the s5.2 exclusions and not subject to the consent requirement,
which is why this is a pipeline and not a `Branch` inside the programme. Two
directions with different governance are two pipelines." They share `Book`,
`Scoring`, and `Temporal` by import, not by copy — the same sharing
discipline as `ProposedLimit` between the batch and realtime paths.

**Different `ensures=[...]` gate each direction**, in the graph rather than
in a test (FRAMEWORK-DEMANDS #25):

```python
# clm/offer/construct.py — ApplyIncrease
ensures=["affordability_assessment_id != 0", "consent_record_id != 0",
         "consent_timestamp < applied_timestamp",
         "affordability_assessed_day <= applied_day",
         "applied_limit_c <= proposed_limit_c"]

# clm/decrease/reduction.py — ApplyDecrease
ensures=["notice_despatched_day <= notice_effective_day",
         "decrease_target_limit_c >= money_owed_floor_c",
         "is_closure -> head_of_credit_risk_authority_id != 0"]
```

An increase cannot be constructed without consent predating it; a decrease
cannot be constructed below the money already owed, or to zero without the
Head of Credit Risk's authority except on fraud/deceased grounds. Both
become structurally impossible in the record rather than merely tested for —
with the honest limit stated right there in `offer/construct.py`'s comment:
this constrains the *decision record*, not whatever a downstream ledger does
with it afterwards (FRAMEWORK-DEMANDS #26).

**The two directions meet at the client level, in
[`clm/decrease/precedence.py`](clm/decrease/precedence.py).** 480,000 clients
hold both products; ~2,900 per cycle qualify for an increase on one account
and a decrease on the other. `ClientDecreaseState` rolls the decrease path's
output up to `client_id` — an `Aggregate`, because a per-record rule cannot
see a sibling row — and `increase_suppressed_by_client` applies s5.7's exact
precedence: an immediate-class decrease anywhere suppresses every increase
for that client; a notice-class decrease does too, *except* where the sole
trigger is dormancy (D12), which is not a risk signal. The file's own close
is the point: *"You were not offered an increase on your card because your
facility went two cycles past due" is reconstructable from those three
columns and nothing else.*

### 6. Cooling-off and eligibility windows as temporal state

[`clm/features/temporal.py`](clm/features/temporal.py) makes two structural
choices that keep effective dating from being a convention someone has to
remember:

- **`decision_date` is a shared param, never a column.** *"A step physically
  cannot read a per-record date that drifts toward 'today'"* — which is what
  makes replay in 2031 unforgeable rather than merely disciplined
  (FRAMEWORK-DEMANDS #16).
- **`path_code` is a table key inside `CoolingOffWindows`, not a branch**:
  `table("clm.cooling_off_windows", keys=("change_type_code", "product_code",
  "path_code"), values={"window_months": int}, dense=True,
  effective_dated=True)`. s5.11's three carve-outs — X09 at a shorter window
  for client-initiated requests, X11 and X12 not applying at all — are rows in
  this table and in the exclusion panel's `applies_on` column, not an
  `if path_code == 2` inside either.

`cooling_off` in [`clm/eligibility/panel.py`](clm/eligibility/panel.py) is one
comparison against this table's output —
`months_since_last_limit_change < cooling_off_window_months` — and a member of
the same `Exclusions` panel as every other exclusion, evaluated completely for
every account, not short-circuited. `BusinessCalendar` exists in the same
file because "20 business days in the home market" is not arithmetic once a
holiday calendar is involved, and `numpy.busday_offset` is neither compilable
inside a kernel nor effective-dated — so it becomes a dense table keyed on
jurisdiction, two array lookups, rather than a Python callback that would
blow `score()`'s 200ms budget.

### 7. Layered caps where the binding one must be recorded

[`clm/caps/panel.py`](clm/caps/panel.py) opens with the rejected alternative
named explicitly — a waterfall of `min()` calls, exactly what doc 03 §3.2
would produce by default — and three reasons it fails s5.5:

```python
LimitCaps = panel(
    "limit_caps",
    members=[ProductMaximumCap, IncomeMultipleCap, TotalUnsecuredCap,
             GroupExposureCap, ObservedSpendCap, MatrixMaxIncreaseCap,
             AffordabilityCap],
    reduce="min", over="matrix_target_limit_c", codes=CAPS,
    tie_break="declared_order",
    writes={"value": "capped_limit_c", "binding": "binding_cap_code",
            "ties": "binding_cap_ties"},
    evidence=["*"],
)
```

`evidence=["*"]` is why all seven caps' computed values survive to the record
for all 4.1 M accounts, not just the one that bound — a waterfall's version
chain would only show values that *moved* the number, and a tie between two
caps would be invisible (the second `min` is a no-op indistinguishable from a
cap that never applied). `CAPS.tie_order` in
[`clm/vocabulary.py`](clm/vocabulary.py) — `["product_maximum",
"income_multiple", "total_unsecured", "group_exposure", "observed_spend",
"matrix_max_increase", "affordability"]` — is what makes tie-breaking
deterministic and declared rather than an accident of Python dict order.

[`clm/caps/observed_spend.py`](clm/caps/observed_spend.py) — C5, the most
frequently binding cap (384,000 accounts) — is shown in the tree in full as
the shape a policy analyst must be able to read: two bounded `param()`s, one
table lookup by spend band, one floor, one docstring that *is* the
reviewable prose ("a client whose highest month in six was R400 does not
get a R30,000 limit"). Adding an eighth cap (change scenario 7) is one
module, one `CAPS` member, one code, per `panel.py`'s own closing note —
because `binding_cap_code`, the simulation's cap incidence report, and every
audit record all read the panel's declared membership, never a hand-kept
list.

`panel(...)` is reused with a **different reducer** three more times:
`reduce="any"` for exclusions and decrease triggers; `reduce="max"` for
decrease floors ([`clm/decrease/reduction.py`](clm/decrease/reduction.py)),
where the *highest* of `money_owed_floor_c`, `regulatory_floor_c` and
`closure_floor_c` binds. One combinator, four uses, four reducers —
`reduction.py`'s own docstring: "this is the third use of `panel` and it is
where the shape earns its generality."

---

## What a policy analyst sees and edits

Nowhere in this project does retuning a value touch Python. The objects a
Credit Risk Policy analyst — explicitly "not an engineer" per spec §3 —
actually opens are `clm/artefacts/limit_matrix/2026Q3.csv` (the matrix
itself, one row per cell, authored in rand — a quarterly retune is a new
CSV, registered as a `candidate` version in `manifest.yaml`, simulated,
reviewed as a swap-set, and only then given `state: live`, an edit to one
YAML file, never a deployment), `clm/artefacts/overlay_register.json` (every
live dial, cap, shift and buffer adjustment, each with an id, owner,
rationale, approval reference, scope and mandatory review date — "run the
programme at 80% this month" is `ADJ-2026-114`, seven fields, no code), and
`clm/artefacts/alco_instruction_2026-09.json` (the monthly budget, envelopes,
ranking objective and over-allocation factor, a params document overriding
the fields listed under `owned_elsewhere` in
`clm/config/programme.production.json`).

The command line the analyst actually runs is in
[`clm/simulate.py`](clm/simulate.py)'s closing comment — `clm simulate
--candidate clm.limit_matrix=2026Q4-candidate-c --snapshot 2026-08
--decision-date 2026-09-01 --against live --report swap-set,cap-incidence,
funding-line`, plus `clm simulate --overlays off` for the unadjusted book,
`clm matrix diff 2026Q3 2026Q4-candidate-c` for a cell-by-cell rand impact,
and `clm overlays expiring --within 30d` as the monthly control report. No
step in that list requires an engineer or a deployment — exactly spec
§5.10's requirement — and it is only possible because `tables=` is an
argument to the one real pipeline (§2 above), not a path resolved from
ambient state.

---

## How "why was my limit reduced" is answered

[`clm/pipelines/decrease_daily.py`](clm/pipelines/decrease_daily.py) closes
with the actual answer, as a single row of its own output — not a query, not
an analyst's reconstruction:

```
trigger_codes            {5}
primary_trigger_code     5  "over-limit persistence"
d05.observed_value       3          (cycles over limit in the last 4)
d05.threshold_applied    3
d05.observed_day         2026-08-31
adjustments_applied      []          (no cut-off shift was in force)
notice_class_code        1  immediate
jurisdiction_code        1  home market
notice_despatched_day    2026-09-01
decrease_target_limit_c  1_842_000   R18 420
binding_floor_code       1  money owed
money_owed.cap_value_c   1_842_000   R18 420 = balance + authorisations + interest
```

Every one of those columns exists because
[`clm/decrease/triggers.py`](clm/decrease/triggers.py) declares
`evidence=["*", "observed_value", "threshold_applied"]` on the trigger panel —
not the fired trigger alone, but its *observed value against its threshold as
applied* — which is what lets a branch consultant answer "why" in one
conversation with no analyst and no query, exactly as s9.1 demands. The same
pattern answers "why was I not offered an increase" for the
simultaneous-decrease case: `clm/decrease/precedence.py`'s three evidence
columns (`increase_suppressed_by_client`, `suppressing_account_id`,
`suppressing_trigger_code`) are the whole of that answer.

The harder version — "why was I not offered an increase, when I qualified?"
— is answered by `Sweep`'s `@entry` evidence in
[`clm/allocation/sweep.py`](clm/allocation/sweep.py):
`limit_budget_remaining_c@entry`, `funded_count@entry` and `allocation_rank`
give exactly the number spec §5.8 says the system must produce two years
later: *"your account was eligible and affordable; a monthly limit on new
lending meant we could fund 380,000 of the 708,000 accounts that qualified,
and yours ranked 412,000th."* Where the answer is instead "an overlay pushed
you below the minimum," `suppression_reason_code` in
[`clm/offer/construct.py`](clm/offer/construct.py) — reading both
`meets_minimum` and `meets_minimum_unadjusted` — is the one place that can
say so, and say whether it is reversible.

---

## What changes when a value moves, versus when structure moves

This project's artefacts sort cleanly into doc 08 §2's three change classes,
and the sort is visible in the layout, not just in a policy document:

| change | example in this project | artefact | cost | who |
|---|---|---|---|---|
| **value** | the ALCO budget, the over-allocation factor, a cap multiple, an overlay's scope or review date | `alco_instruction_2026-09.json`, `overlay_register.json`, a params document | free — a bundle swap, no compile | ALCO, Credit Risk Policy |
| **interior** (generic-kernel kind) | the matrix's 1,152 cells, the behavioural scorecard's bins, a table's rows | `limit_matrix/2026Q3.csv`, `clm.behavioural` | swap only — both are `table`/`scorecard` kinds (doc 08 §3.4), so a new version loads without a compile | Credit Risk Policy / Model Risk, simulated, Credit Committee-approved |
| **skeleton** | a new cap module, a new decrease trigger, a new pipeline stage | a new file under `clm/caps/`, a new `panel` member, a new line in a `\|` sequence | code review, rebuild, redeploy | engineering |

The interesting row is the middle one: this project's own governance is
*stricter* than the framework's technical cost. A matrix swap is technically
free — no different in kind from any other generic-kernel table reload — but
`manifest.yaml` gates it behind simulation and Credit Committee approval
anyway, because compile cost and governance cost are different axes: doc 08
§5's `impact(...)` review exists precisely because "a param change cannot
alter the graph" is not the same claim as "a param change is safe." A cycle
dial moving from 100% to 80% recompiles nothing and changes which 380,000 of
708,000 accounts get funded (FRAMEWORK-DEMANDS #8) — exactly why
`overlay_register.json` carries an `approval` and a mandatory `review_date`
on every entry despite being, mechanically, the cheapest kind of change the
system has.

The skeleton row is rare on purpose. The seven overlay points in this
project exist so the recurring policy asks — tighten a cut-off, shift a
score, cap an increase — never need a skeleton change. The skeleton only
moves for genuinely new *logic*: an eighth cap, a fifteenth exclusion, a new
trigger. Change scenario 7 (a new deposit-balance cap) costs one file, by
analogy with [`clm/caps/observed_spend.py`](clm/caps/observed_spend.py), one
`CAPS` member, and one line in `LimitCaps.members`
([`clm/caps/panel.py`](clm/caps/panel.py)) — reviewed, but small and legible,
because the panel's declared membership is the only place a new cap has to
be named.

---

## A fix noted

`clm/matrix/assignment.py` imported `shadow` but never called it, even
though `clm/offer/construct.py`'s own docstring claims the unadjusted matrix
value is "the payoff from `shadow` in matrix/assignment.py," and
`meets_minimum_unadjusted` reads `matrix_min_increment_unadjusted_c` and
`matrix_multiplier_unadjusted` — names nothing in the tree produced. That is
an outright inconsistency between two files' stated intent, so `Matrix` now
wraps `MatrixLookup | ApplyMatrixOverlays | Target` in a `shadow(...)` over
`MatrixLookup | Target` with `neutralise={"overlay_set": "off"}`, producing
`matrix_multiplier_unadjusted` and the cell's unadjusted increment ceiling —
the same mechanism `clm/scoring/behavioural.py` and
`clm/decrease/triggers.py` already demonstrate, so this project now has all
three uses FRAMEWORK-DEMANDS #10 claims ("matrix and caps, the scorecard,
decrease thresholds") rather than two.

One gap remains open rather than papered over: FRAMEWORK-DEMANDS #10 and spec
§5.9 require `proposed_limit_unadjusted_c` to reflect what the **entire**
matrix-and-cap chain would produce with the stack disabled, not the matrix
alone. `clm/caps/panel.py`'s `LimitCaps` has no `shadow` of its own — its
overlay points (`ApplyCapOverlays`, `ApplySpendCapOverlays`) are not yet
neutralised anywhere — so `proposed_limit_unadjusted_c`, required as
non-droppable evidence on `Construct`, has no complete producer in this
sketch as it stands. Closing it means composing a second `shadow` over
`LimitCaps | RoundDown` where `clm/pipelines/programme.py` already joins
`Matrix` and `CapChain` — a real design decision about where two independent
overlay registers get jointly neutralised, not a one-line fix — so it is
left as a stated gap rather than guessed at, in the same spirit as
FRAMEWORK-DEMANDS' own AWK markers.
