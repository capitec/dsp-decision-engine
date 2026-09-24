# 07 — Credit limit management

> Fictional. The Bank, its products, thresholds, table dimensions and volumetrics
> are invented for this repository. Regulatory mechanisms referred to are the
> published public ones; every number attached to them is illustrative.

---

## 1. What this is

The Bank holds **4.1 M open revolving accounts** across two products — **Everyday
Card** (`product_code` 20, 2.35 M accounts) and **Access Facility**
(`product_code` 21, 1.75 M accounts) — carrying R84.6 bn of granted limit against
R31.2 bn of drawn balance. Every month it must decide, for each of those
accounts, whether the limit should go up, go down, or stay where it is.

Limits go up because a client who has run a R15 000 card cleanly for three years
is worth more to the Bank at R28 000, and because if the Bank does not offer it a
competitor will. Limits go down because behaviour deteriorates before arrears
appear, and a R60 000 limit on an account that is about to default is R60 000 of
exposure the Bank chose to keep. Both directions are regulated: an increase
requires an affordability assessment and the client's consent, a reduction
requires a reason and — in most cases — prescribed notice.

Two things make this different from every other project in the set.

The first is that **the increase decision is not free**. The aggregate of all
proposed increases lands on the balance sheet, and the Asset and Liability
Committee sets a monthly ceiling on how much additional limit the Bank is willing
to grant — currently **R2.4 bn of applied limit per month**, with an associated
risk-weighted-asset and expected-loss envelope. In a typical cycle the programme
identifies **708 000 accounts** that are eligible, affordable and worth
increasing, whose proposed increases total **R5.1 bn**. Roughly **380 000** of
them can be funded. Which 380 000 is a decision about the population, not about
any one account, and no amount of per-account reasoning produces it. An account
that is declined because it fell below the funding line has a decline reason that
depends on 707 999 other accounts.

The second is that **the same affordability question must be answered with far
worse evidence**. Project 02's assessment runs against a payslip, a bank
statement and a fresh bureau record collected in a journey where the client is
present and motivated. Here there is no journey. The client last spoke to the
Bank 26 months ago (median), may have changed jobs, may have had a child, may
have taken on R4 000 a month of external debt that the bureau file will show and
may have taken on R4 000 a month that it will not. The Bank must still assess
affordability before it increases the limit, and must be honest in its evidence
record about what it actually knew.

Around those two problems sit the rest: a **1 152-cell assignment matrix** that
Credit Risk Policy re-tunes quarterly, a **simulation capability** that must show
the portfolio impact of a candidate matrix over the full book in under twenty
minutes and must not be a second implementation of the same logic, a **decrease
path** with its own triggers and its own governance, and a **real-time path**
serving 14 000 client-requested increases a day that must not disagree with the
batch programme about the same client in the same month.

---

## 2. Why it is in this set

| Question | How this project stresses it |
|---|---|
| **Q2 tables** ● | The limit assignment matrix is 12 × 8 × 6 × 2 with three values per cell — 1 152 cells, 3 456 values — and it is the single artefact that most determines the portfolio's shape. It is authored in a spreadsheet, changed quarterly, must be diffable cell by cell, must be validatable before it is trusted, must be attributable after the fact, and must be capable of existing in a candidate state that has been simulated but not deployed. Eight further tables sit around it, and a register of live overlays sits above it. |
| **Q3 parameters** ● | A quarterly matrix change is made by a Credit Risk Policy analyst who is not an engineer, and cannot be deployed until its portfolio impact has been simulated and the swap-set approved by the Credit Committee. The budget, the ranking weights and the over-allocation factor change monthly and are owned by a different committee. The cooling-off windows change annually. And between quarterly matrix releases, policy needs a dial — "run the programme at 80% of the matrix this month", "cap every increase at R5 000 this cycle" — applied as a named, approved, expiring overlay rather than as an edit to 1 152 cells. None of these may require a code release, and all of them must be resolvable as of a `decision_date` two years later. |
| **Q1 reuse** ● | `core.affordability`, `core.income`, `core.obligations` and `core.expense_norms` are consumed here in a **degraded-evidence mode** that project 02 was not written for: no payslip, stale declared income, deposit-derived salary for only 74% of the book. The same capabilities must serve both, the difference must be a stated input rather than a fork, and the evidence record must make the degradation visible instead of hiding it behind a shared interface. |
| **Q7 audit** ● | Four distinct interrogations, from four parties, listed in §9. The hardest is not the increase — it is explaining a **non-selection**: an account that was eligible, affordable and desirable and did not get an increase because it ranked 412 000th in a list where 380 000 were funded. Running a close second is separating what the matrix said from what this month's dial did to it. |
| Q4 custom modules ○ | The behavioural scorecard, the assignment matrix lookup and the ranking function are project-specific and used by both the batch and the real-time paths, and by simulation. Three consumers, one implementation, no copies. |
| Q5 core kinds ○ | Population-level allocation under a constraint is not a scorecard, a table lookup or a tree. Neither is a swap-set. |
| Q6 codebase ○ | Two products, an increase path, a decrease path, a batch path, a real-time path and a simulation path over the same logic. |

### The hard parts, named in advance

So that an implementation can be judged against them:

1. **A population-level budget constraint sitting on top of per-account
   decisions.** The per-account work is independent and embarrassingly
   parallel; the allocation is not, and it needs every account's result before
   it can produce any account's answer.
2. **Simulation sharing one implementation with production.** Not "kept in
   sync". One.
3. **Affordability with degraded evidence.** Reusing a regulated calculation in
   a mode with less evidence, without either forking it or quietly pretending
   the evidence is better than it is.
4. **Increase and decrease co-existing.** Opposite directions, different
   triggers, different governance, different notice rules, on the same account
   and sometimes on the same client in the same cycle.
5. **Oscillation at thresholds.** An account sitting on a band edge or on the
   funding line must not be granted in March, reduced in April and granted
   again in May because its utilisation moved by one percentage point.
6. **Two paths that must agree.** A batch programme with deposit-derived income
   and a real-time journey with fresh declared income must not give the same
   client materially different answers in the same month, and every difference
   must be attributable to a named input.
7. **Overlays over an approved matrix, interacting with the budget.** A dial
   that shrinks every increase does not shrink the programme — it changes how
   far down the ranking a fixed budget reaches, so more clients get less. The
   unadjusted matrix answer must survive beside the adjusted one everywhere,
   including in simulation, and every overlay must expire.

---

## 3. Actors

| Actor | Responsibility |
|---|---|
| **Credit Risk Policy** | Owns the limit assignment matrix, the decrease trigger thresholds, the cooling-off and suppression windows, the spend-cap parameters, and the register of live overlays over all of them. Re-tunes the matrix quarterly. Not engineers. Must be able to author a candidate matrix, simulate it, and present the swap-set without assistance. |
| **Credit Committee** | Approves each matrix version on the evidence of its simulation. Owns the income-multiple caps and the exposure caps. |
| **Asset and Liability Committee (ALCO)** | Sets the monthly additional-limit budget, the risk-weighted-asset envelope, the expected-loss envelope, the over-allocation factor and the choice of ranking objective. Receives the monthly reconciliation of budget granted against budget consumed. |
| **Portfolio Management** | Runs the monthly cycle. Owns the operational calendar, the acceptance of the run, and the decision to re-run. |
| **Collections and Recoveries** | Owns the risk-based decrease triggers jointly with Credit Risk Policy. Escalates individual accounts outside the monthly cycle. Consumes project 08's treatment state. |
| **Head of Credit Risk** | The authority required for a reduction to zero (facility closure) other than on fraud or deceased grounds. |
| **Regulatory Compliance** | Owns the notice requirements by change type and jurisdiction, the offer and notice wording, the consent taxonomy, and the sign-off that an affordability assessment precedes every increase. |
| **Client Operations** | Answers "why was my limit reduced" at the counter and on the phone, within one call, without an analyst. |
| **Model Risk** | Owns the behavioural scorecards and their calibration; monitors the funding line for drift. Must be able to see each scorecard's own output separately from any policy overlay sitting on it, and to run the book with the overlay stack disabled. |
| **Internal Audit** | Samples increases and re-derives them independently. Tests that consent preceded every limit change. |
| **Product engineering** | Builds and runs it. Does not own any threshold in this document. |

---

## 4. Inputs

### 4.1 Sources and volumes

| Source | Shape | Volume | Freshness | Nullability |
|---|---|---|---|---|
| Revolving account master | One row per open account | 4.10 M | Month-end snapshot, available 03:00 on day 1 of the cycle | Complete by construction |
| Cycle history | 6–24 monthly buckets per account, **ragged** | 78.4 M buckets | Month-end | Accounts opened < 24 months ago have fewer buckets; 6 is the minimum for eligibility |
| Spend category aggregates | 18 merchant category groups × 6 months per account | 443 M values | Month-end | Absent for accounts with no purchase activity in the window |
| Arrears and delinquency history | 24 monthly delinquency states per account | 98.4 M | Month-end | Same raggedness as cycle history |
| Limit change history | 0–40 events per account | 11.8 M events | Real-time, as at snapshot | Empty for 1.34 M accounts that have never had a limit change |
| Offer history | 0–24 offers per account | 22.1 M offers | Real-time | Empty for 0.9 M accounts |
| Bureau refresh | One record per client | 3.62 M clients | Monthly file, as-of 5 working days before the run | **2.4% of clients have no bureau match** |
| Transactional account credits | Up to 36 months of credit transactions | 2.71 M clients with a Bank transactional account | Daily | **26% of revolving account holders bank elsewhere and have no deposit evidence** |
| Declared income and expenses | Last captured or refreshed values | 4.07 M accounts | Age 1–96 months, **median 26 months** | Missing for 0.8% |
| Consent records | Per client, per consent type, with timestamp, channel and wording version | 14.6 M records | Real-time | Absence is meaningful: no automatic-increase consent on file is an exclusion, not a default |
| Client holdings | 1–14 accounts per client across all Bank products | 21.3 M | Daily | — |
| Markers | Fraud, deceased, estate, dispute, debt review, insolvency, staff | 4.10 M flags | Daily | — |
| Project 08 treatment state | Current collections treatment and arrangement state | 0.31 M accounts | Daily | Absent for performing accounts |
| Overlay register | Live overlays over the matrix, the scorecards, the calibration, the grade boundaries and the caps | 20–60 live at any time | Ad hoc, sometimes within the cycle month | An empty register is valid and means the matrix runs unadjusted; a **lapsed** overlay is not valid and must stop the cycle |
| ALCO budget instruction | Budget, envelopes, ranking selection, over-allocation factor, and the cycle dial if one is set | 1 document | Monthly, by the 25th of the prior month | Must be present; the cycle does not run on last month's instruction |

**Client cardinality.** 4.10 M accounts belong to **3.62 M clients**; **480 000
clients hold both products**, which is where the simultaneous increase-and-
decrease case arises.

### 4.2 Per-account state

Per account, at the snapshot:

- `current_limit`, statement balance, current balance, unsettled authorisations,
  accrued unbilled interest, `revolving_utilisation`, `months_on_book`.
- 6–24 cycle buckets, each carrying statement balance, minimum due, payment
  made, days past due at cycle end, purchase value, cash withdrawal value, fee
  and interest charged, and an over-limit flag.
- Arrears history: worst delinquency in 3, 6, 12 and 24 months; count of cycles
  ever past due; months since last arrears.
- Over-limit events: count and consecutive run length over 12 cycles.
- Cash-withdrawal ratio: cash value ÷ total drawn value, over 3 and 12 months.
- Spend categories: 18 groups over 6 months, and the derived trailing
  90th-percentile monthly spend used by the spend cap (§5.5).
- Prior limit changes: date, old limit, new limit, source, whether client-
  initiated, whether accepted.
- Prior offers: date, amount, channel, and response — accepted, declined,
  expired, or the client opted out of automatic increases entirely.
- The client's other holdings, internal exposure and pending applications.

### 4.3 Names declared locally

The library does not publish these; this project declares them and no other
project may redefine them:

| Name | Type | Meaning |
|---|---|---|
| `account_id` | int64 | One revolving account. Stable for life. |
| `current_limit` | float64 | Granted limit at the snapshot. |
| `proposed_limit` | float64 | The limit after the matrix and all caps, before allocation. |
| `applied_limit` | float64 | The limit actually in force after acceptance or notice. |
| `months_on_book` | int16 | Whole months since account opening at `decision_date`. |
| `utilisation_band` | int8 | 1..8, see §6.2. |
| `mob_band` | int8 | 1..6. |
| `behaviour_score` | float64 | Scaled behavioural score. |
| `behaviour_grade` | int8 | 1 (best) .. 12 (worst). Distinct from `risk_grade`. |
| `matrix_cell_id` | int32 | The single cell of the assignment matrix that was read. |
| `matrix_multiplier_unadjusted` | float64 | The cell's multiplier as authored, before any overlay. |
| `proposed_limit_unadjusted` | float64 | The limit the matrix and caps produce with the overlay stack disabled. |
| `cycle_dial_id` | int16 | The overlay, if any, applied to every account in this cycle. |
| `binding_cap_code` | int8 | Which cap produced `proposed_limit`. |
| `evidence_tier_code` | int8 | Which affordability evidence tier was available. |
| `income_staleness_days` | int32 | Age of the income basis at `decision_date`. |
| `allocation_rank` | int32 | Position in the ranked list. |
| `allocation_outcome_code` | int8 | Funded, below the line, fairness-capped, tail-skipped, overlay-suppressed, not ranked. |
| `change_type_code` | int8 | Which of the nine change types this is (§6.5). |
| `notice_class_code` | int8 | Immediate, prescribed notice, or consent-required. |
| `offer_id` | int64 | One offer to one client for one account. |
| `consent_record_id` | int64 | The consent relied on for this change. |

`decision_date`, `product_code`, `client_id`, `gross_monthly_income`,
`net_monthly_income`, `living_expenses`, `existing_obligations`,
`discretionary_income`, `max_affordable_instalment`, `affordability_verdict_code`,
`total_exposure`, `probability_of_default`, `probability_of_default_unadjusted`,
`score_unadjusted`, `adjustment_set_id`, `adjustments_applied`,
`decline_reason_codes` and `primary_reason_code` are the library's and are used
unchanged. In particular this project does not invent a second vocabulary for
overlays: the overlays it applies to its own matrix are the library's mechanism
(`core.adjustments`, 00 §6.22) pointed at a project-owned artefact.

---

## 5. The flow

The monthly programme runs over three hours on the first working day after the
snapshot. Stages 5.1 to 5.6 and 5.9 are per-account and independent. Stage 5.7
is per-account but resolved per client. **Stage 5.8 is not per-account at all**,
and is the reason this project is in the set. Stage 5.10 runs the same logic
under a different question, and stages 5.11 and 5.12 run it one account at a
time, in 200 milliseconds, all day.

### A representative cycle

| Point | Accounts | Additional limit |
|---|---|---|
| Open revolving accounts in scope | 4 100 000 | — |
| After hard exclusions (§5.2) | 2 670 000 | — |
| Scored (§5.3) | 2 670 000 | — |
| Matrix proposes a non-zero increase (§5.4) | 1 420 000 | R11.7 bn |
| Still positive after caps (§5.5) | 1 110 000 | R6.9 bn |
| Affordability: automatic path (§5.6) | 796 000 | R5.7 bn |
| Affordability: conditional on income confirmation | 213 000 | R0.92 bn |
| Affordability: fails | 101 000 | — |
| Above the minimum meaningful increase (§5.9) | 708 000 | R5.11 bn |
| **Funded within the budget (§5.8)** | **380 000** | **R2.39 bn** |
| Below the funding line | 328 000 | R2.72 bn |
| Offers accepted and applied (following 30 days) | 271 700 | R1.71 bn |
| Decreases proposed in the same cycle (§5.7) | 96 000 | −R1.34 bn |

---

### 5.1 Population and account state

**Precondition.** A complete month-end snapshot of the revolving book, a bureau
file no older than five working days, the daily marker feeds, and the ALCO
instruction for the cycle. A missing or partial source stops the cycle; it does
not run on stale inputs and silently label them fresh.

**Determines.** The account state on which everything downstream depends: for
each of 4.1 M accounts, its current limit, balance, utilisation, months on book,
6 to 24 cycles of payment and delinquency history, arrears and over-limit
history, cash-withdrawal ratio, spend by category, the full history of prior
limit changes and prior offers with their outcomes, and the client's other
holdings across the Bank.

Two derived quantities matter enough to name here because later stages key off
them and because they are the usual source of band-edge disputes:

- `revolving_utilisation` — drawn balance ÷ granted limit at the snapshot, and
  separately its 3-, 6- and 12-month means. The matrix keys off the **6-month
  mean**, not the spot value, precisely to damp the oscillation in §5.8.
- **Observed spend** — the trailing 6-month 90th-percentile monthly purchase
  value, which drives the spend cap in §5.5. A client whose highest month in six
  was R400 does not get a R30 000 limit.

**Emits.** A per-account state record, and a per-client roll-up of holdings and
internal exposure.

**Recorded.** The snapshot identifier and its as-of date; the bureau file
identifier and its as-of date; the count of accounts in scope and its variance
against the prior cycle; and, per account, the number of cycle buckets actually
present. Raggedness is data, not a defect: an account with 7 buckets is scored
on 7, and the fact that it had 7 must be visible afterwards.

---

### 5.2 Hard exclusions

**Precondition.** Account state and markers, as of the snapshot.

**Determines.** Whether the account is considered at all. Sixteen exclusions, any
one of which is sufficient.

| Code | Exclusion | Threshold | Accounts |
|---|---|---|---|
| X01 | In arrears now | ≥ 1 cycle past due at the snapshot | 214 000 |
| X02 | Arrears within 6 months | any cycle ≥ 1 past due in the last 6 | 311 000 |
| X03 | Debt review or administration order | any status other than terminated | 96 000 |
| X04 | Insolvency or sequestration | any active status | 4 100 |
| X05 | Deceased or estate | marker present | 12 400 |
| X06 | Fraud marker | on the account or the client | 8 900 |
| X07 | Account under dispute | an open dispute on any transaction | 21 000 |
| X08 | Dormant | zero balance and no transaction for 6 months | 187 000 |
| X09 | Within a cooling-off period | see §6.5 | 402 000 |
| X10 | In-flight application elsewhere in the Bank | undecided credit application within 30 days | 63 000 |
| X11 | Prior declined offer within the suppression window | 4 months | 141 000 |
| X12 | **No consent for automatic increases** | no consent record, or consent withdrawn | 508 000 |
| X13 | Staff account | staff or related-party marker | 9 600 |
| X14 | Too young | `months_on_book` < 6 | 128 000 |
| X15 | Already at the product maximum | limit = R300 000 (card) or R150 000 (facility) | 44 000 |
| X16 | Closed, blocked, or under a payment arrangement | any | 31 000 |

**1 430 000 distinct accounts** are excluded; the codes sum to more because
accounts carry several.

X12 deserves its own note. The Bank may not raise a limit unilaterally without
the client's prior agreement to receive such increases, and the client may
withdraw that agreement at any time. **12.4% of the book has no such agreement on
file.** Those accounts are not offers-that-were-declined and must not be counted
as suppressed marketing; they are outside the automatic programme entirely, and
their only route to a higher limit is to ask for one (§5.11).

**Emits.** An exclusion verdict per account, and the complete set of exclusion
codes that applied.

**Recorded.** Every exclusion that applied, not merely the first one found. This
is in tension with the obvious efficiency — there is no point evaluating a
cooling-off window on a deceased client's account — and the tension is real: an
implementation that short-circuits loses the reason set, and one that evaluates
everything spends 4.1 M × 16 evaluations to produce information that is discarded
for 65% of the book. The requirement is both: complete attribution, and no
scoring, affordability or bureau work performed for an excluded account.

---

### 5.3 Behavioural scoring

**Precondition.** The account survived §5.2 and has at least 6 cycle buckets.

**Determines.** `behaviour_score`, `probability_of_default` and
`behaviour_grade` (1 best .. 12 worst) from a **behavioural scorecard of 32
characteristics** evaluated over a 24-month performance window, one scorecard per
product.

Representative characteristics, by family:

| Family | Characteristics |
|---|---|
| Delinquency | Worst delinquency in 3 / 6 / 12 / 24 months; count of cycles ever past due; months since last arrears; count of returned payment instructions in 12 months |
| Payment behaviour | Payment-to-balance ratio over 3 / 6 / 12 months; count of minimum-payment-only cycles in 6 and 12; count of full-settlement cycles in 12; days since last payment |
| Utilisation | Mean utilisation over 3 / 6 / 12; utilisation trend over 6; count of cycles above 90%; count of over-limit cycles in 12; longest consecutive over-limit run |
| Transacting | Purchase velocity; cash-withdrawal ratio over 3 and 12; merchant category concentration; count of active months in 6; credit turnover ratio |
| Tenure and change | `months_on_book`; count of prior limit increases; months since last limit change; count of declined offers |
| External | Bureau enquiry velocity over 3 and 6 months; worst external delinquency; count of external accounts opened in 6 months; external revolving utilisation |

**Nulls are a scoring bin, not an error**, and the raggedness makes them common:
an account 7 months on book has no 12-month payment ratio and no 24-month worst
delinquency. Those characteristics bin to "insufficient history", which is a
weight in the scorecard, not a zero and not a failure. The same is true of the
2.4% of clients with no bureau match: the six external characteristics bin to "no
bureau record", which is materially different from "bureau record showing
nothing".

**What this is not.** It is essential to state, in the requirements rather than
in a comment, that this is not an application scorecard and the difference cuts
both ways:

- **Richer.** Twelve to twenty-four months of observed repayment behaviour on the
  Bank's own account beats anything an application ever sees. The model separates
  well: the Gini on the card scorecard is 0.71 against 0.58 for the card
  application scorecard.
- **Blinder.** It sees behaviour, not circumstances. The client may have changed
  jobs, lost a second income, had a child, or taken on a private loan the bureau
  will not show for another two months. **The Bank will not know.** A perfect
  payment record on a R3 000 balance tells you nothing about capacity at R25 000.

That asymmetry is the whole reason §5.6 exists and is not optional.

**Overlays.** `behaviour_score` and `probability_of_default` are subject to the
overlay mechanism the library publishes. In a limit programme the recurring uses
are a **score shift** by product or segment (−15 points on accounts opened
through the partner channel, where early-life performance has run worse than the
model predicted) and a **PD multiplier** (× 1.25 on the Access Facility book
during a period when observed default is running above prediction), and
occasionally a **grade-boundary shift** where policy wants a notch of tightening
without touching the model. Because `behaviour_grade` keys the matrix in §5.4, a
one-notch grade shift moves an account to a different cell, a different
multiplier and a different maximum increase — so a score overlay is not a
cosmetic adjustment to a reported number, it is a change to the limit offered.

**Emits.** `behaviour_score`, `score_unadjusted`, `probability_of_default`,
`probability_of_default_unadjusted`, `behaviour_grade`, the unadjusted grade,
per-characteristic point contributions, and the scorecard identifier and version.

**Recorded.** The characteristic vector as evaluated, including which
characteristics were null and why; the per-characteristic contributions (required
output, not a debugging convenience — they are the raw material for the reason
given to a client under §5.7); the scorecard version resolved against
`decision_date`; and the ordered list of overlays that applied with the effect of
each. The unadjusted score and PD survive to the decision record so that Model
Risk can monitor the scorecard's own performance without the overlay in the way.

---

### 5.4 The limit assignment matrix

**Precondition.** The account has a `behaviour_grade`, a 6-month mean
utilisation, a `months_on_book` and a `product_code`.

**Determines.** The starting point for every increase in the programme: a limit
multiplier, a maximum absolute increase, and a minimum increment, read from a
single cell of the assignment matrix.

The matrix is keyed on four dimensions:

| Dimension | Levels | Values |
|---|---|---|
| `behaviour_grade` | 12 | 1 (best) .. 12 (worst) |
| `utilisation_band` | 8 | 0%; 0–10%; 10–25%; 25–40%; 40–55%; 55–70%; 70–90%; above 90% |
| `mob_band` | 6 | 6–11; 12–17; 18–23; 24–35; 36–59; 60+ months |
| `product_code` | 2 | 20 Everyday Card; 21 Access Facility |

**12 × 8 × 6 × 2 = 1 152 cells**, each carrying three values — **3 456 values in
total**, every one of them authored by a human being at Credit Risk Policy in a
spreadsheet, every quarter.

Each cell holds:

| Value | Range across the matrix | Meaning |
|---|---|---|
| Limit multiplier | 1.00 – 1.75 | `current_limit` × multiplier is the uncapped target |
| Maximum absolute increase | R0 – R60 000 | Ceiling on the rand increase from this cell |
| Minimum increment | R500 – R2 500 | Below this the cell yields no increase at all |

The shape of the matrix is policy, and it is not monotonic in the obvious
direction. A grade-2 account at 5% utilisation and 60 months on book gets a
multiplier of 1.00 — it does not need more limit and granting it adds exposure
that will never earn. A grade-2 account at 75% utilisation and 36 months gets
1.55. A grade-9 account gets 1.00 everywhere. Grade-4 accounts in the 6–11 month
band are capped at R3 000 of absolute increase regardless of multiplier, because
nine months of good behaviour is nine months of evidence.

**Exactly one cell is read.** Band edges are closed at the lower bound and open at
the upper, without exception, and an account whose utilisation is exactly 40.0%
falls in the 40–55% band. This is stated as a requirement because band-edge
disputes are a recurring audit finding and "whichever the code does" is not an
answer.

**The cycle dial.** Between quarterly matrix releases, policy and ALCO need to
move the programme without reissuing 1 152 cells and without a quarterly approval
cycle. Two forms recur and both must be expressible as named, approved,
effective-dated overlays over the matrix rather than as edits to it:

- **A global multiplier over the matrix output.** "Run the programme at 80% of
  the matrix this month" — because the funding environment tightened, or because
  the previous cycle's take-up ran hot. Applied to the multiplier's excess over
  1.00, so a cell multiplier of 1.50 becomes 1.40 at an 80% dial, not 1.20.
- **A cycle cap.** "No increase above R5 000 this cycle, whatever the cell says."

Overlays may also be scoped rather than global — at 60% on the Access Facility
book only, or restricted to grades 7–9, or to one channel of origination — and
scope is declared, so an overlay applied outside its scope is an error rather
than a silent no-op. Overlays stack, and the order in which a scoped overlay and
a global dial compose is part of each overlay's definition, not an accident of
evaluation.

Every overlay carries an id, an owner, a rationale, an approval reference, an
effective-from date and a **mandatory review date**. A limit programme is where
the standing failure mode bites hardest: a 70% dial applied during one bad
quarter, never reviewed, still quietly suppressing every increase in the book
three years later, with nobody able to say who set it or why. A dial that reaches
its review date without renewal must surface before the cycle runs, and a lapsed
overlay stops the cycle.

**Emits.** `matrix_cell_id`, the three cell values as authored, the three values
after the overlay stack, the uncapped target limit, and the uncapped target limit
that would have obtained with the overlay stack disabled.

**Recorded.** The matrix version resolved against `decision_date`; the single
cell identifier; the values as authored (`matrix_multiplier_unadjusted`) beside
the values as applied; the overlay stack in force, in order, with each overlay's
id and effect; and the band-edge flag where any of the three banded inputs fell
within 1% of a band boundary.

---

### 5.5 Caps

**Precondition.** An uncapped target limit from §5.4.

**Determines.** `proposed_limit` — the most the Bank is willing to grant this
account on policy grounds, before affordability and before the budget. Six caps
apply; the lowest wins.

| Code | Cap | Definition | Binds for |
|---|---|---|---|
| C1 | Product maximum | R300 000 card, R150 000 facility | 61 000 |
| C2 | Income-multiple | limit ≤ *k* × `gross_monthly_income`, *k* from §6.4 (2.5 – 5.0 by product and segment) | 221 000 |
| C3 | Total unsecured exposure | the client's aggregate unsecured limit across the Bank ≤ the lower of R450 000 and 8 × `net_monthly_income` | 94 000 |
| C4 | Group exposure | from `core.exposure`, over the client and related parties | 18 000 |
| C5 | **Observed spend** | limit ≤ max(`current_limit`, 3.5 × trailing 6-month 90th-percentile monthly spend), floored at R2 000 | 384 000 |
| C6 | Matrix maximum absolute increase | from the cell, after overlays | 156 000 |
| C7 | Affordability | §5.6 | 176 000 |

C5 is the cap that stops the programme doing something indefensible. A client
with a R6 000 card limit, a clean record and 40 months on book sits in a cell
with a 1.6 multiplier, and the matrix would take them to R9 600. If their highest
spending month in the last six was R400, R9 600 is exposure the Bank is creating
for nobody's benefit, and it is exactly the kind of grant that looks negligent in
hindsight. C5 holds them at their current limit. It is the most frequently
binding cap in the programme and Credit Risk Policy tunes its multiple quarterly
by spend band.

Caps are themselves subject to overlay: a **cap adjustment** reducing the
income-multiple by 20% for the self-employed segment for two quarters, or a
tightening of C5's multiple from 3.5 to 2.5 across the book, must be expressible
as an approved, expiring overlay with the unadjusted cap still visible — for the
same reason as everything else here, which is that a Credit Committee asks "what
would we have done without it" at every sitting.

**Rounding.** The proposed limit is rounded **down** to the nearest R500 —
downward without exception, because rounding up can breach a cap that was
computed exactly, and a cap that can be breached by rounding is not a cap. The
increase amount is then `proposed_limit` − `current_limit`.

**Emits.** `proposed_limit`, `proposed_limit_unadjusted`, and `binding_cap_code`.

**Recorded.** **The binding cap, always.** Every cap's computed value must be
available, and the one that bound must be named. "The client was offered R9 600
instead of R14 000" is not an answer; "the observed-spend cap bound at R9 600
against a matrix target of R14 000, an income-multiple cap of R21 500 and an
affordability cap of R18 200" is. Where two caps bind at the same value, both are
recorded and the tie is broken by the order in the table above, deterministically.

---

### 5.6 Affordability reassessment, with degraded evidence

**Precondition.** A positive `proposed_limit` after §5.5.

The Bank may not increase a credit limit without assessing whether the client can
afford the increased commitment. The assessment is the same statutory assessment
project 02 implements; what differs is that here there is **no journey, no
payslip and no client in the room.**

**Determines.** Whether the increase may be offered automatically, must be made
conditional on the client confirming their income, or must not be offered.

**The notional instalment.** A revolving facility has no instalment, so one is
imputed: the affordability test is run against the commitment the client would
carry if the facility were drawn to the new limit and serviced at the contractual
minimum payment. `notional_instalment` = `proposed_limit` × the minimum payment
rate for the product and balance band (§6.8; 3.5% for Everyday Card, 5.0% for
Access Facility at the relevant bands). A R25 000 card limit therefore carries a
R875 notional instalment into `core.affordability`.

**Evidence tiers.** In place of a payslip, an ordered waterfall:

| Tier | Evidence | Coverage | Haircut | Maximum staleness |
|---|---|---|---|---|
| A | **Verified salary deposits** — a recurring credit into a Bank transactional account matching a salary pattern in each of the last 3 months, month-to-month variability ≤ 15% | 74% of accounts | 0% | 45 days |
| B | **Irregular deposit evidence** — recurring credits present but variable or with a gap in the last 3 months | 9% | 15% on the trailing 6-month mean | 45 days |
| C | **Declared income, refreshed** — captured or re-confirmed within 12 months | 6% | 0%, but indexed forward at CPI (5.2% p.a.) only to a maximum of 12 months | 365 days |
| D | **Declared income, stale** — captured more than 12 months ago; median age across the book is 26 months | 9% | Not usable for an automatic increase | — |
| E | **No usable income evidence** — no Bank transactional account and no declared income within 96 months | 2% | Not usable | — |

**The obligations side** is refreshed from the monthly bureau file through
`core.obligations`, giving `existing_obligations` and the change in external
obligations since origination. This is genuinely better evidence than the income
side, and asymmetrically so: the Bank sees the client's new external debt but not
their new salary. Where the bureau shows obligations that have grown by more than
25% since the last assessment, the account is pushed to the conditional path
regardless of income tier.

**The expense side** uses `core.expense_norms`. Declared living expenses older
than 12 months are indexed at CPI and then floored at the statutory minimum norm,
and where the declared figure after indexation is below the norm, the norm binds
and the basis is recorded as such. A declared expense figure older than 36 months
is discarded entirely in favour of the norm.

**The staleness rule.** This is the requirement that makes the degradation
honest rather than hidden:

> An increase may be applied automatically **only** on tier A or tier B income
> evidence no older than 45 days, with a bureau view no older than 35 days, and
> with declared expenses either refreshed within 12 months or replaced by the
> norm floor. Any account failing any of those conditions, but otherwise passing
> affordability on the evidence available, may be offered an increase
> **conditional on the client confirming their income**, and the limit may not
> change until that confirmation is received. Any account failing affordability
> on the evidence available is not offered an increase at all.

Of the 1 110 000 accounts reaching this stage: **796 000** take the automatic
path, **213 000** the conditional path, and **101 000** fail.

**The affordability buffer** applied here is *not* the same value project 03
applies at origination. Origination underwrites against verified evidence
collected days earlier; this programme underwrites against a salary pattern and a
bureau file. The buffer is accordingly wider — 18% against origination's 12% —
and it is a parameter of the assessment, supplied by this project, **not a fork
of the shared capability.** The shared capability must be able to serve both
callers with different buffers, different evidence tiers and different staleness
tolerances, and the buffer is itself overlay-adjustable (a buffer adjustment of
+3 percentage points for a quarter is a routine ask).

**What must be stated as a requirement**, because it is the seam that a careless
implementation will break: the difference between origination affordability and
programme affordability must be visible in the **inputs and the evidence record**,
not in the code. Two implementations of a statutory calculation is a regulatory
finding waiting to happen, and the version that is not in the origination journey
is the one that will fall behind when the regulations change.

**Emits.** `discretionary_income`, `max_affordable_instalment`,
`affordability_verdict_code`, `evidence_tier_code`, `income_staleness_days`, the
affordability-derived limit cap (C7), and the automatic / conditional / fail
verdict.

**Recorded.** The evidence tier and the exact evidence relied on — which account
the deposits were observed in, over which months, at what values; the bureau file
identifier and as-of date; the age of the declared figures and the indexation
applied; the expense basis; the buffer and its overlay stack; the norm table
version; and the assessment's own identifier, which the consent record in §5.9
must reference. A regulator asking "was affordability assessed before this
increase" must be answerable from this record alone, for any increase, without
reconstruction.

---

### 5.7 The decrease path

**Precondition.** An open account, evaluated in every cycle and additionally on
any day a risk trigger fires. **Decreases are not subject to the exclusions in
§5.2** — an account in arrears is excluded from increases and is a prime
candidate for a decrease — nor to the consent requirement, because a reduction is
a unilateral act the Bank is entitled to take with reasons and notice.

**Determines.** Whether the limit should be reduced, by how much, and under what
notice class.

| Code | Trigger | Threshold | Class |
|---|---|---|---|
| D01 | Behaviour score deterioration | drop ≥ 60 points over 3 months, landing in grade 8 or worse | Notice |
| D02 | Grade migration | worsens by ≥ 3 notches into grades 9–12 | Notice |
| D03 | Emerging arrears | 1 cycle past due in the current month | Immediate |
| D04 | Repeat arrears | ≥ 2 cycles past due within 6 months | Immediate |
| D05 | Over-limit persistence | over limit at cycle end in 3 of the last 4 cycles | Immediate |
| D06 | Minimum-payment persistence | payment ≤ 105% of minimum due in 6 of 6 cycles, with utilisation ≥ 85% | Notice |
| D07 | Bureau-detected distress | new default listing, judgment, administration order or debt review at any provider | Immediate |
| D08 | External arrears | any external account ≥ 2 cycles past due with balance ≥ R2 000 | Notice |
| D09 | Enquiry velocity | ≥ 6 credit enquiries in 60 days | Notice |
| D10 | Utilisation spike | 6-month mean ≤ 40% rising to ≥ 90% within 2 cycles | Notice |
| D11 | Cash-withdrawal escalation | cash ≥ 45% of drawn value in 2 consecutive cycles | Notice |
| D12 | Dormancy | zero balance and no transaction for 9 months | Notice |
| D13 | Returned payment instructions | ≥ 3 returned in 6 months | Notice |
| D14 | Income disappearance | verified salary deposit absent for 2 consecutive months where previously present for 6 | Notice |

Each trigger carries its own reduction rule, not merely a flag: a target limit
expressed as a multiple of current balance, a percentage of current limit, or a
fixed step. D05 reduces to the greater of the current balance and 90% of the
current limit; D07 reduces to the current balance; D12 reduces to R2 000 or
proposes closure. Decrease thresholds are overlay-adjustable in the same way as
everything else — a **cut-off shift** tightening D01 from 60 points to 45 for two
quarters during a deteriorating cycle is the archetypal use, and it must expire.

**Four rules constrain every decrease:**

1. **The new limit may not go below the money already owed.** Specifically, the
   new limit must be at least the statement balance **plus unsettled
   authorisations plus accrued unbilled interest**. Reducing a limit to a figure
   that a transaction already authorised will exceed puts the client over limit
   through the Bank's own action, and generates a fee the Bank will have to
   refund.
2. **Notice class governs timing.** Immediate-class reductions take effect on the
   day, with written reasons despatched within 1 business day. Notice-class
   reductions take effect only after the prescribed notice period has run — 20
   business days in the home market, 30 calendar days and 45 calendar days in the
   two other jurisdictions in which the Bank holds these products under local
   licence (§6.7). The account keeps its old limit for the whole notice period.
3. **A reduction to zero is a facility closure** and requires the Head of Credit
   Risk's authority, recorded against the specific account, except where it
   follows a fraud marker or a deceased marker. It is never an automatic
   consequence of a trigger.
4. **The client may ask why**, and must receive the specific trigger, the
   threshold, the observed value and the date the observation was made — not a
   generic statement that the Bank reviews limits periodically.

**The simultaneous case.** 480 000 clients hold both products, and in a typical
cycle **2 900 of them qualify for an increase on one account and a decrease on
the other** — a card used cleanly for four years and a facility drawn to the
limit and paying minimums. This is not a contradiction to be suppressed; it is
two accounts behaving differently. The requirement is an explicit precedence,
recorded:

- An **immediate-class** decrease on any account of the client suppresses every
  increase for that client in the cycle. Deterioration is a client-level signal.
- A **notice-class** decrease on one account suppresses increases on the client's
  other accounts, **except** where the sole trigger is D12 (dormancy), which is
  not a risk signal.
- Where an increase is suppressed for this reason, the suppression, the
  triggering account and the trigger code are recorded against the suppressed
  account. "You were not offered an increase on your card because your facility
  went two cycles past due" must be reconstructable.

**Emits.** A decrease proposal per account: the trigger set, the target limit,
the floor imposed by rule 1, the notice class, the effective date, and the
authority required.

**Recorded.** Every trigger that fired with its observed value against its
threshold, the overlay stack in force on those thresholds, the reduction rule
applied, the binding floor, the notice class and jurisdiction, the notice
despatch record, and — for closures — the authorising individual and the
timestamp of their authorisation.

---

### 5.8 Portfolio budget allocation

**Precondition.** Every account in the population has been through §5.1–§5.7.
This stage cannot start until the last one is done, and it cannot produce any
account's answer until it has seen all of them.

**Determines.** Which of the eligible accounts actually receive an offer.

The aggregate of all proposed increases in a typical cycle is **R5.11 bn across
708 000 accounts**. ALCO's instruction for the cycle permits **R2.4 bn of applied
additional limit**, inside a risk-weighted-asset envelope of **R1.15 bn** and an
incremental 12-month expected-loss envelope of **R48 m**. Roughly half the
programme is fundable. Something must choose.

**This is a population-level decision and it cannot be made per account.** No
property of an account determines whether it is funded; what determines it is
where the account sits relative to 707 999 others and where the money runs out.
Two accounts with identical state, identical grade, identical proposed increase
and identical affordability get different answers in different months because the
population around them moved. An implementation that tries to make this decision
one account at a time will end up with a hard-coded score threshold standing in
for the budget, which will be wrong every month and silently so.

**Ranking.** Each eligible account is given a ranking value under an objective
selected by ALCO in the cycle instruction. Three must be supported:

| Objective | Ranking value | Used when |
|---|---|---|
| Risk-adjusted return per rand of budget | (expected 12-month incremental revenue − expected incremental loss) ÷ additional limit | Default |
| Expected value | expected incremental revenue − expected incremental loss, absolute | When ALCO wants balance growth over efficiency |
| Policy priority | a weighted score over grade, tenure and relationship depth | When retention or a segment strategy dominates |

The default objective's components, all of which are parameters rather than
constants: expected incremental drawn balance = additional limit × a credit
conversion factor (0.42 card, 0.55 facility, varying by utilisation band);
revenue = incremental balance × (net interest margin + interchange + fee yield);
loss = incremental exposure at default × `probability_of_default` × loss given
default (0.74 card, 0.79 facility).

**Constraints the result must satisfy.** Stated as properties of the outcome,
because they are what an auditor will test:

1. Total applied additional limit, in expectation, ≤ the ALCO budget. Offers are
   despatched at an **over-allocation factor** (currently 1.38, derived from the
   trailing 12-month take-up rate of 72.4%), so the offer envelope is R3.31 bn
   against a R2.4 bn applied budget. The factor is ALCO's parameter, not a
   derived quantity the programme may adjust itself.
2. Incremental risk-weighted assets ≤ R1.15 bn and incremental expected loss ≤
   R48 m. Either envelope may bind before the limit budget does, and which one
   bound must be recorded for the cycle.
3. **No increase may be trimmed to fit.** An account is funded at the amount
   §5.5 produced or not funded at all. A part-funded increase is an amount no
   cell of the matrix produced and no audit could re-derive.
4. **The budget tail is explicit.** At the point where the remaining budget is
   smaller than the next account's increase, the rule is: skip that account, mark
   it tail-skipped, and continue down the ranking taking any account whose
   increase fits, up to a maximum of 2 000 tail-skips, after which the cycle
   stops. Back-filling and hard-stopping are both defensible; picking one
   silently is not.
5. **Fairness floors.** No segment — defined as product × grade × mob band, 144
   segments — may have a funded proportion below 40% of the population-wide
   funded proportion in three consecutive cycles. 15% of the envelope is held as
   reserved sub-budgets allocated to satisfy these floors before the main pool is
   allocated by rank. A segment that is starved must be visible as a starved
   segment, not as 6 000 individually unlucky accounts.
6. **Hysteresis at the line.** An account funded in the prior cycle that remains
   eligible, whose ranking value has moved by less than 5%, must not be displaced
   by an account that has never been funded and whose ranking value is within 5%
   of it. Measured requirement: **holding inputs constant between two cycles,
   fewer than 2% of accounts may change funded status.**
7. **Ties are broken deterministically**, by `account_id` ascending. A
   floating-point tie in a ranking value must not make the funded set irre-
   producible on re-run.

**The conditional list.** The 213 000 accounts on the conditional-income path
(§5.6) are ranked separately against a reserved sub-envelope, and charged against
the budget at their expected conversion rate (22%) rather than at face value,
because most will never confirm their income. The conditional list was not
budget-bound in the reported cycle — all 213 000 were offered, at R0.92 bn — but
it must be capable of binding, and the reconciliation in §9 must show provisioned
against realised for the conditional path separately.

**How overlays change this stage, which is the non-obvious part.** A dial that
shrinks every proposed increase does not shrink the programme — it changes how
far down the ranking a fixed budget reaches. Run the same cycle at an 80% dial:

| | No dial | 80% dial |
|---|---|---|
| Accounts with a proposed increase above the minimum | 708 000 | 667 000 |
| Dropped below the minimum meaningful increase by the dial | — | **41 000** |
| Mean funded increase | R6 290 | R5 030 |
| **Accounts funded within the same R2.39 bn envelope** | **380 000** | **476 000** |
| Accounts below the funding line | 328 000 | 191 000 |

96 000 more clients receive an increase, each of them smaller, for the same
money. That is a materially different programme, produced by one overlay, and it
is why the **non-selection reason must distinguish "ranked below the funding
line" from "reduced below the minimum meaningful increase by overlay
ADJ-2026-114"**. They are different answers to the client, different answers to
Credit Committee, and the second is reversible by withdrawing an overlay while
the first is not.

**Emits.** Per account: `allocation_rank`, the ranking value, the
`allocation_outcome_code`, and — for every funded account — the amount funded.
Per cycle: the funding line, the funded count, the envelope consumed, which
envelope bound, and the fairness-floor allocations.

**Recorded.** Enough to answer a non-selection without re-running the cycle: the
account's ranking value and rank, the total ranked, the funded count, the ranking
value at the line, the objective in force, the budget instruction identifier, and
the overlay stack. The intended client-facing answer is *"your account was
eligible and affordable; a monthly limit on new lending meant we could fund
380 000 of the 708 000 accounts that qualified, and yours ranked 412 000th"* —
and a system that cannot produce the number 412 000 two years later cannot say
that.

---

### 5.9 Offer construction and consent

**Precondition.** A funded increase from §5.8, or a decrease proposal from §5.7.

**Determines.** What the client is actually told, through what channel, and what
must be true before the limit changes.

| Rule | Value |
|---|---|
| Rounding | New limit down to the nearest R500 (§5.5) |
| Minimum meaningful increase | R1 000 Everyday Card, R500 Access Facility. Below this, no offer is made — a R400 increase costs more in notice and consent handling than it earns, and it consumes a cooling-off window |
| Offer expiry | 30 calendar days from despatch |
| Channel | In-app message, SMS, email or statement insert, by `core.consent` channel permissions. A regulated notice of a reduction is **not** marketing and is not suppressed by a marketing preference |
| Wording | From the Compliance-owned wording set, by change type, language (3) and channel. The offer states the new limit, the increase, the expiry date, the right to decline, and the right to opt out of future automatic increases |
| Take-up | 71.5% accepted within 30 days; 24.6% expired; 3.9% declined |

**Consent.** A limit does not change until a consent record exists that
references (a) this offer, (b) the affordability assessment identifier from
§5.6, and (c) the wording version the client saw, with a timestamp **strictly
before** the limit change. Where the client holds a standing agreement to receive
automatic annual increases, the record relied on is that agreement, and the
notice period before the change takes effect is 5 business days, during which the
client may decline. A client declining a specific offer is suppressed for 4
months; a client opting out of automatic increases is suppressed **permanently**,
the opt-out is honoured within 1 business day, it is a client-level flag not an
account-level one, and it may only be reversed by the client.

**Recorded.** `offer_id`, `consent_record_id`, the despatch channel and
timestamp, the wording version, the response and its timestamp, and — mandatory —
**the unadjusted matrix value alongside the adjusted one on every offer made**.
The record must be able to say "the matrix said R14 000; this cycle's dial took
it to R11 200; the observed-spend cap took it to R9 600; we offered R9 500 after
rounding".

---

### 5.10 Simulation

**Precondition.** A candidate artefact set — a candidate matrix, or a candidate
overlay stack, or candidate caps, thresholds, budget or ranking objective — and a
book snapshot.

**Determines.** What would happen if that artefact set were deployed. Credit
Committee approves a matrix version **on the evidence of its simulation**; the
simulation output is the governance artefact.

Credit Risk Policy must be able to run it themselves, against a candidate
spreadsheet, without an engineer and without a deployment, over the full 4.1 M
book, in **under 20 minutes** — because a matrix is tuned by trying eight or ten
candidates in an afternoon, and a simulation that takes overnight is a simulation
that gets run once on the version that was going to be approved anyway.

**Required outputs:**

| Output | Detail |
|---|---|
| Bucket migration | How many accounts move cell, and the 1 152 × 1 152 flow reduced to the ~40 material transitions |
| Total additional limit | Proposed, offered and — under the budget — funded |
| Expected additional exposure | Additional limit × credit conversion factor, by product |
| Expected loss impact | Incremental 12-month expected loss, and the RWA consumed |
| Distribution of increases | By grade, utilisation band, mob band, product, income decile, tenure and region — the last three because a distribution that looks fine by grade can be indefensible by region |
| **Swap-set against the current matrix** | Gainers, losers and unaffected, with counts, rand amounts and mean change per segment. Not a before-total and an after-total: a per-account classification |
| Cap incidence | How many accounts each cap binds for, before and after |
| The funding line | Where the budget runs out under the candidate, and the funded count |
| Overlay attribution | The effect of each overlay in the stack, separately from the matrix's own effect |

**Four requirements that are the point of the stage:**

1. **Simulation uses the same logic as production. Not a second
   implementation.** A separate simulation implementation is unacceptable, and
   the reasons are specific, not aesthetic. First, the swap-set is what Credit
   Committee approves — if it comes from different code, the committee approved
   something that will not run. Second, drift is silent and one-directional: the
   previous generation maintained a simulation model in a spreadsheet, and by the
   time anyone checked it differed from production by 4.1 percentage points on
   funded counts, so the matrix change approved in that cycle produced a
   portfolio nobody had modelled. Third, two implementations means two change
   budgets, and the simulation one always loses.
2. **Simulation covers the overlay stack, not just the matrix.** The most common
   question a policy analyst asks is not "what does the new matrix do" but
   **"what does this month's dial cost us"** — and answering it with a separate
   calculation is unacceptable for exactly the same reasons as (1), with the
   added hazard that a dial is set in days rather than quarters and will never
   get a spreadsheet of its own. Simulation must therefore be runnable with the
   stack as it stands, with a candidate stack, and **with the stack disabled**,
   the last being how Model Risk observes the scorecards' own behaviour.
3. **Self-check.** Simulation run over the *current production* artefact set and
   the last production snapshot must reproduce the last production cycle
   **account for account, to the rand**, including ranks and the funded set. A
   simulation that cannot reproduce production is not evidence about production.
   This check runs before every simulation session and its result is part of the
   simulation output.
4. **Backtest.** A candidate set must be runnable over a snapshot 12 months old,
   with the subsequent 12 months of realised performance attached, so that
   "accounts this matrix would have funded" can be scored against what those
   accounts actually did.

**Recorded.** The candidate artefact set, its identifier, the snapshot, the
self-check result, every output above, and the identity of the analyst who ran
it. The simulation that supported an approved matrix version must be retrievable
for as long as decisions made under that version are.

---

### 5.11 The event-driven path

**Precondition.** A client asks for a higher limit — 14 000 times a day, 81%
through the app, 14% in branch, 5% through the call centre.

**Determines.** An immediate answer, using the same matrix, the same caps and the
same affordability logic as the programme, with three differences:

| | Monthly programme | Client request |
|---|---|---|
| Income evidence | Deposit-derived, tiers A–E | **Fresh declared income**, captured in the journey, plus deposit evidence where it exists |
| Budget | Subject to §5.8 | **Not subject to the portfolio budget** — a client who asks is served |
| Consent | Standing agreement or offer acceptance | Given in the journey, before the change |
| Answer | Offer despatched | Immediate, p99 under 200 ms |
| Amount | The matrix-and-cap maximum | min(requested, matrix-and-cap maximum) |

The exclusions in §5.2 apply with three carve-outs: X12 (no automatic-increase
consent) does not apply, because the client is asking; X11 (declined-offer
suppression) does not apply for the same reason; X09 (cooling-off) applies at a
shorter window for client-initiated requests (§6.5). Everything else — arrears,
debt review, fraud, dispute, in-flight application — applies unchanged.

Where the request exceeds the maximum, the Bank counter-offers at the maximum if
that clears the minimum meaningful increase, and otherwise declines with a
`primary_reason_code` from the shared taxonomy. Where fresh declared income is
materially higher than deposit-derived income, the declared figure is subject to
verification before the limit changes, and the answer is "approved, subject to
confirming your income", not a silent decline.

The same overlay stack applies. A cycle dial set for the monthly programme
applies to client requests made in the same period unless the overlay's declared
scope excludes them — and whether a dial is scoped to the programme or to the
product is a decision the overlay must state, because getting it wrong is how the
two paths start disagreeing.

---

### 5.12 Agreement between the two paths

The two paths must give the same client the same answer. Stated testably:

> For an account whose monthly programme run in month M produced a
> `proposed_limit` of L, a client request in month M on unchanged evidence must
> produce a limit L′ with |L′ − L| ≤ max(R500, 2% of L). Where the difference
> exceeds that, it must be attributable to a **named input difference** — fresh
> declared income differing from deposit-derived income, a bureau refresh between
> the snapshot and the request, or a transaction changing the utilisation band —
> and to no other cause.

A monthly reconciliation samples **2 000 accounts that took both paths** and
reports the agreement rate, which must be at or above **99.0%** within tolerance,
with every breach attributed. Note what this does not require: the *funded*
outcomes need not agree, because the programme is budget-constrained and the
request path is not. It is the **proposed limit** that must agree. A client who
asks, in a month when they ranked 412 000th, gets the increase — and that is
correct, not a defect, because the budget governs what the Bank pushes out, not
what it grants on request.

---

## 6. Parameters and tables

### 6.1 The register

| Table | Dimensions | Values | Owner | Cadence | Source |
|---|---|---|---|---|---|
| **Limit assignment matrix** | **12 grades × 8 utilisation × 6 mob × 2 products × 3 values** | **3 456** (1 152 cells) | Credit Risk Policy | Quarterly | Spreadsheet |
| Behavioural scorecards | 2 scorecards × 32 characteristics × ~9 bins | ~576 | Model Risk | On model release, 12–18 months | Model documentation |
| Behaviour grade boundaries | 2 products × 12 grade edges | 24 | Model Risk | Semi-annual | Model team |
| Decrease trigger thresholds | 14 triggers × 6 attributes × 2 products | 168 | Credit Risk Policy + Collections | Quarterly | Internal |
| Income-multiple caps | 2 products × 6 segments × 3 values | 36 | Credit Committee | Semi-annual | Internal |
| Exposure caps | 5 cap kinds × 6 segments | 30 | Credit Committee | Quarterly | Internal |
| Spend cap parameters | 2 products × 5 spend bands × 2 values | 20 | Credit Risk Policy | Quarterly | Internal |
| Cooling-off and suppression windows | 9 change types × 2 products | 18 | Credit Risk Policy | Annual | Internal |
| Notice requirements | 7 change types × 3 jurisdictions × 4 attributes | 84 | Compliance | On regulation | Regulator |
| Minimum payment rates | 2 products × 4 balance bands | 8 | Product | Annual | Contract terms |
| Ranking coefficients | 18 (CCF, margin, interchange, fee yield, LGD, by product and band) | 18 | ALCO + Model Risk | Monthly | Internal |
| Budget and fairness settings | 24 (budget, RWA and EL envelopes, over-allocation factor, objective, floors, hysteresis, tail rule) | 24 | ALCO | Monthly | ALCO instruction |
| Affordability settings | Buffer, evidence tiers, haircuts, staleness tolerances, indexation rate | 22 | Credit Risk Policy + Compliance | Quarterly | Internal |
| Offer and notice wording | 7 change types × 3 languages × 2 channels | 42 | Compliance | On change | Internal |
| **Overlay register** | 20–60 live overlays × 11 attributes | ~500 | Credit Risk Policy, ALCO | **Ad hoc, sometimes mid-cycle** | Internal |

Consumed unchanged from the library: expense norms (statutory and internal), the
obligation treatment matrix, tax tables, risk grade boundaries, the reason code
registry and the adjustment mechanism itself.

### 6.2 Bands

Utilisation (8): 0%; 0–10%; 10–25%; 25–40%; 40–55%; 55–70%; 70–90%; above 90%,
on the 6-month mean, lower bound closed. Months on book (6): 6–11; 12–17; 18–23;
24–35; 36–59; 60+.

### 6.3 What the matrix must support

1. **Authored in a spreadsheet** by a person who does not write code, loaded
   without a deployment, and rejected with a usable message when malformed.
2. **Validated before trust** — every cell present, multipliers within
   [1.00, 2.00], maxima non-negative, minimum increments positive, and a
   monotonicity check that flags (rather than forbids) any cell where a worse
   grade receives a larger multiplier than a better grade at the same
   utilisation and tenure.
3. **Diffable** — a candidate against the incumbent, cell by cell, with the
   count and rand impact of each change. "Policy sent a new spreadsheet" is not
   a reviewable change.
4. **Attributable** — every decision names the one cell it read, in the version
   in force at its `decision_date`, for as long as that decision is auditable.
5. **Candidate-capable** — a version may exist, be simulated and be reviewed
   without being live, and the approved candidate becomes live on a date.

### 6.4 Income-multiple caps (illustrative)

| Segment | Everyday Card | Access Facility |
|---|---|---|
| Permanent, ≥ 24 months employed | 5.0 × | 3.0 × |
| Permanent, < 24 months | 4.0 × | 2.5 × |
| Contract | 3.5 × | 2.5 × |
| Self-employed | 3.0 × | 2.0 × |
| Pensioner | 2.5 × | 2.0 × |
| Income tier C or D (§5.6) | 2.5 × | 2.0 × |

### 6.5 Cooling-off and suppression windows

| Change type | Window before any further increase |
|---|---|
| Programme increase accepted | 6 months |
| Programme increase declined | 4 months |
| Programme offer expired | 3 months |
| Client-requested increase granted | 3 months |
| Client-requested increase declined | 2 months, or immediately on new evidence |
| Notice-class decrease | 9 months |
| Immediate-class decrease | 12 months |
| Facility closure | No further increase |
| Opt-out of automatic increases | Permanent, for the programme path only |

### 6.6 Overlays specific to this project

Every overlay carries an id, owner, rationale, approval reference, declared
scope, effective-from, effective-to and a mandatory review date, and stacks in a
declared order. The recurring kinds here:

| Kind | Over | Typical use |
|---|---|---|
| Matrix multiplier ("the cycle dial") | The matrix multiplier's excess over 1.00 | "Run at 80% this month" |
| Cycle cap | Every increase | "Nothing above R5 000 this cycle" |
| Score shift | `behaviour_score` | −15 points on the partner-originated segment |
| PD multiplier | `probability_of_default` | × 1.25 on the Access Facility book |
| Grade boundary shift | `behaviour_grade` | Tighten the 6/7 edge by 0.3 percentage points of PD |
| Cap adjustment | C2, C3, C5 | Income multiple −20% for self-employed, two quarters |
| Buffer adjustment | The affordability buffer | +3 percentage points |
| Cut-off shift | Decrease thresholds | D01 from 60 points to 45 |

### 6.7 Notice periods by change type and jurisdiction (illustrative)

| Change | Home market | Market B | Market C |
|---|---|---|---|
| Increase, standing agreement | 5 business days before effect | 10 business days | 14 calendar days |
| Increase, offer accepted | Immediate | Immediate | Immediate |
| Decrease, notice class | 20 business days | 30 calendar days | 45 calendar days |
| Decrease, immediate class | Effective on the day, reasons within 1 business day | Same, within 2 | Same, within 2 |
| Closure | 20 business days and authority | 30 calendar days and authority | 45 calendar days and authority |

### 6.8 Minimum payment rates

Everyday Card: 3.5% of balance, minimum R100. Access Facility: 5.0% of balance,
minimum R50. Used only to impute the notional instalment in §5.6.

---

## 7. Outputs

**Per account, every cycle** — for all 4.1 M, including those excluded at §5.2,
because "we did consider your account" is itself an answer:

`account_id`, `decision_date`, the exclusion set, `behaviour_score` and
`score_unadjusted`, `probability_of_default` and its unadjusted counterpart,
`behaviour_grade`, `matrix_cell_id`, the cell values as authored and as adjusted,
`proposed_limit` and `proposed_limit_unadjusted`, `binding_cap_code` and every
cap's computed value, `evidence_tier_code`, `income_staleness_days`, the
affordability verdict and assessment identifier, `allocation_rank`, the ranking
value, `allocation_outcome_code`, `change_type_code`, `notice_class_code`,
`adjustment_set_id` and `adjustments_applied`, and every artefact version
resolved.

**Per offer**: `offer_id`, amount, unadjusted matrix amount, channel, wording
version, despatch and expiry timestamps, response, `consent_record_id`.

**Per decrease**: trigger set with observed values, target, binding floor, notice
class, jurisdiction, effective date, notice despatch record, authority where
required.

**Per cycle**: the funnel in §5; the funding line and funded count; envelope
consumed against each of the three envelopes and which bound; fairness-floor
allocations; the ALCO reconciliation; the overlay stack in force with each
overlay's aggregate effect; and the list of overlays within 30 days of their
review date.

**Per real-time decision**: the same per-account record, plus the requested
amount, the journey identifier and the latency achieved.

---

## 8. Non-functional requirements

| Requirement | Value |
|---|---|
| Monthly programme | 4.1 M accounts complete within a **3-hour window**, including allocation, offer construction and despatch handover |
| Allocation stage | The population-level stage within 25 minutes of that window |
| Simulation | Full 4.1 M book, full overlay stack, all §5.10 outputs, **under 20 minutes**, initiated by a non-engineer |
| Real-time path | **p99 under 200 ms**, p50 under 60 ms, for the decision excluding client network |
| Real-time volume | 14 000 requests/day, sustained 25/second at peak, no degradation on programme-run days |
| Decrease path | Daily, over 4.1 M accounts, within 45 minutes |
| Determinism | Same snapshot, same artefact versions, same overlay stack, same `decision_date` ⇒ identical outputs **including ranks and the funded set**, bit for bit |
| Restartability | The 3-hour window has no room for a full restart; a failed cycle resumes rather than repeats, and a resumed cycle produces the same result as an uninterrupted one |
| Cold start | No per-request compilation; the first real-time request after a deployment is not materially slower than the thousandth |
| Artefact refresh | A matrix version, an overlay or an ALCO instruction takes effect without a code deployment |
| Retention | Per-account cycle records for 7 years; artefact versions and simulations for as long as any decision made under them remains auditable |

---

## 9. Audit, evidence and explainability

Four interrogations, each of which has happened:

**1. "Why was my limit reduced?"** — at a branch counter, answered in one
conversation by a consultant with no analytical tooling. Must yield: the trigger,
its threshold, the observed value, the date of observation, the notice sent and
when, the new limit and the floor that set it. A generic "periodic review" answer
is a complaint the Bank will lose.

**2. The regulator tests affordability and consent.** A sample of 50 000
increases over 18 months. For each: the affordability assessment, its evidence
tier and date, the notional instalment and how it was derived, the income
evidence actually relied on, the expense basis, the norm table version — and a
consent record whose timestamp **precedes** the limit change. Two failure modes
must be structurally impossible rather than merely tested for: a limit change
with no assessment, and a limit change whose consent record post-dates it.

**3. Internal audit re-derives 500 increases.** Given only the recorded inputs
and the artefact versions, auditors reproduce the offered amount independently,
to the rand. This requires that the record contains the cell read, the cell's
authored values, the overlay stack and its order, every cap's computed value, the
binding cap, the rounding, and the allocation outcome. It also requires that
running the same account through the current implementation at its original
`decision_date` returns the original answer — which is project 09's replay
guarantee, and this project must not be the one that breaks it.

**4. ALCO reconciles budget granted against budget consumed.** Monthly:

| Line | Cycle |
|---|---|
| Budget instructed (applied limit) | R2.400 bn |
| Over-allocation factor | 1.38 |
| Offer envelope | R3.312 bn |
| Offers despatched — automatic path | R2.394 bn (380 000 accounts) |
| Offers despatched — conditional path | R0.918 bn (213 000 accounts) |
| Applied within 30 days | R2.310 bn |
| Consumption against budget | 96.3% |
| Expired or declined | R1.002 bn |
| Conditional path: provisioned vs realised | R0.202 bn vs R0.178 bn |
| RWA consumed / envelope | R1.09 bn / R1.15 bn |
| Incremental expected loss / envelope | R44.1 m / R48.0 m |

The 3.7% shortfall must be explained, not merely reported, and the take-up rate
feeds the next cycle's over-allocation factor — which ALCO sets, not the
programme.

**Overlay evidence.** For any past date the Bank must be able to state which
overlays were in force over the matrix, the scorecards, the calibration and the
caps, in what order, over what scope, who approved each and when each expires;
what each one's aggregate effect on that cycle was; and what the cycle would have
produced with the stack disabled. An overlay past its review date and still in
force is a reportable control failure, and the report is monthly.

---

## 10. Acceptance criteria

1. A Credit Risk Policy analyst authors a candidate matrix in a spreadsheet,
   loads it, simulates it over 4.1 M accounts in under 20 minutes, and presents
   the swap-set to Credit Committee, **without an engineer at any point**.
2. The simulation, run over the current artefact set and the last production
   snapshot, reproduces the last production cycle account for account, to the
   rand, including ranks and the funded set.
3. Simulation and production share one implementation. There is no second
   expression of the matrix, the caps, the affordability logic, the overlays or
   the allocation anywhere in the system.
4. A cycle dial is set, simulated and applied without any cell of the matrix
   being edited, and the unadjusted matrix value is present on every offer made
   under it.
5. The cycle can be run with the overlay stack disabled, producing the
   unadjusted answer for the whole book, without a separate implementation.
6. Every overlay in force has an owner, an approval reference, a declared scope
   and a review date, and a lapsed overlay stops the cycle.
7. A non-selected account's answer states its rank, the ranked total, the funded
   count and the value at the line, and distinguishes "below the line" from
   "reduced below the minimum by an overlay".
8. Holding inputs constant between two cycles, fewer than 2% of accounts change
   funded status, and no account receives an increase and a decrease within
   6 months.
9. The monthly programme completes within 3 hours; the real-time path holds p99
   under 200 ms on a programme-run day.
10. The agreement test in §5.12 passes at 99.0% or better, every breach
    attributed to a named input.
11. Affordability is assessed before every increase, with consent recorded
    before every limit change, demonstrable over a 50 000-account sample with
    zero exceptions.
12. A decrease is explainable at a counter in one conversation.
13. A 2026 cycle is re-derivable in 2031 from its recorded inputs and the
    artefact and overlay versions in force at its `decision_date`.
14. The affordability capability is shared with project 02 and not forked; the
    difference between origination and programme assessment is visible entirely
    in inputs and evidence.

---

## 11. Change scenarios

1. **Credit Risk Policy re-tunes the matrix**, changing 340 of 1 152 cells, and
   must show Credit Committee the swap-set and the portfolio impact before
   deployment. This happens every quarter, forever.
2. **ALCO halves the budget** to R1.2 bn for three months. Nothing about any
   account's eligibility changes; the funding line moves and 190 000 accounts
   that would have been funded are not.
3. **A cycle dial is set at 70%** three days before the run, by a policy analyst,
   with a two-month expiry — no matrix reissue, no release, and the simulation of
   its cost produced the same afternoon. Two months later it lapses, and the
   programme refuses to run until it is renewed or withdrawn.
4. **Model Risk replaces the card behavioural scorecard** with a 38-characteristic
   version, and needs both live for a three-month parallel run on a 10% holdout,
   with a PD multiplier overlay on the new one while it beds in.
5. **A fifth utilisation band is added**, splitting 70–90% into 70–80% and
   80–90%, taking the matrix from 1 152 to 1 296 cells. Historical decisions must
   continue to resolve against the 8-band version.
6. **The regulator shortens the notice period** for decreases in one jurisdiction
   from 30 to 21 days, effective on a date, with decisions before that date
   governed by the old period.
7. **A new cap is added** — maximum limit as a multiple of the client's total
   Bank deposit balances — binding for 40 000 accounts, and it must appear in
   `binding_cap_code`, in the simulation's cap incidence and in every audit
   record from that date.
8. **ALCO changes the ranking objective** from risk-adjusted return to policy
   priority for two cycles during a retention campaign, and wants the swap-set
   between the two objectives before deciding.
9. **A second conditional path is required**: offers conditional on the client
   accepting a debit order, not only on confirming income.
10. **The decrease path must run daily** rather than monthly, on a bureau
    distress feed that arrives every morning, while the increase path stays
    monthly.
11. **An audit finding** requires the unadjusted matrix value on every historical
    offer, where today it is recorded only from the date overlays were
    introduced.
12. **A third product is added** — a revolving facility for the small-business
    book — needing the same programme with its own matrix, its own scorecard and
    its own budget line, without disturbing the two existing products.

---

## 12. Out of scope

- Origination of new revolving accounts. This programme changes limits on
  accounts that exist.
- The behavioural model's development and validation. This project consumes
  scorecards; it does not fit them.
- Collections treatment and arrangements, which are project 08's. This project
  consumes their state as an exclusion and as a decrease trigger.
- Message rendering, despatch infrastructure and delivery tracking. The
  programme produces the offer and the notice; it does not send them.
- Ledger posting of the limit change, statement production and fee assessment.
- The replay harness, the regulator pack and cross-project certification, which
  are project 09's.
- Pricing. Interest rates on revolving products are not a function of the limit
  and do not change with it.

---

## 13. Questions the implementation must answer

1. **How is a population-level constraint expressed** over decisions that are
   otherwise per-account? The per-account work is independent and the allocation
   is not; where does the boundary go, and what does the thing on the far side
   of it look like?
2. **How does an account learn its own non-selection?** The answer depends on
   707 999 other accounts. Is the rank an output of the account's own
   evaluation, or something attached to it afterwards, and what does that do to
   the shape of a decision record?
3. **How does one implementation serve production, simulation and a 200 ms
   real-time path** without becoming three implementations with shared
   constants? Production needs 4.1 M accounts in 3 hours, simulation needs the
   same book in 20 minutes initiated by a non-engineer, and the real-time path
   needs one account in 200 milliseconds.
4. **What is a candidate artefact?** A matrix that has been authored, validated
   and simulated but not deployed is neither live nor absent. How does a version
   exist in that state, get exercised over the full book, and then become live on
   a date?
5. **Where do overlays live, structurally?** They change values, which makes
   them parameters; their composition order is real logic, which makes them
   structure; the stack in force on a date is part of the decision record, which
   makes them evidence. An implementation that answers this three different ways
   in three places will be unmaintainable, and a limit programme with a
   never-reviewed cycle dial is the failure this mechanism exists to prevent —
   so how is expiry enforced rather than reported?
6. **How is a shared capability consumed in a degraded-evidence mode** without
   forking it and without the degradation becoming invisible? The affordability
   calculation is the same; the evidence, the buffer and the staleness tolerances
   are not.
7. **How do two opposite decisions co-exist** on one client — an increase on one
   account and a decrease on another — with different triggers, different
   governance, different notice classes and an explicit precedence between them?
8. **What prevents oscillation** at a band edge and at the funding line?
   Hysteresis is a requirement on the result; is it a property of the logic, a
   parameter, or something applied afterwards — and can it be, if the result must
   be re-derivable?
9. **How is a 1 152-cell artefact owned by a non-engineer** loaded, validated,
   diffed, attributed cell by cell and effective-dated, with the same mechanism
   serving eight further tables of wildly different shape?
10. **How is the run restarted** halfway through a 3-hour window such that the
    resumed cycle is identical to an uninterrupted one, given that the allocation
    depends on the whole population?
11. **What is the unit of reuse between the batch and real-time paths**, given
    that they must agree to R500 on the same client, and that agreement is a
    monthly measured acceptance criterion rather than an aspiration?
12. **How much of this is expressible in scorecards, tables and trees**, and what
    is left over? Candidates for the leftover pile: the ranking, the allocation,
    the swap-set, the evidence waterfall, the cap set with its binding-cap
    attribution, and the overlay stack.
