# 06 — Consolidation and restructure

> *Fictional. The Bank, its products, thresholds, table dimensions and volumetrics
> are invented for this repository. Regulatory mechanisms referred to are the
> published public ones; every number attached to them is illustrative.*

---

## 1. What this is

A client of eleven years walks into a branch and asks for R60 000 she cannot
afford. She is servicing eleven credit accounts — two of the Bank's, nine
elsewhere — at a balance-weighted rate of 24.8%, for a combined monthly
instalment of R9 340 against a discretionary income of R1 120. Two of those
accounts are one month in arrears. Declining her is correct and useless. Her
problem is not the R60 000; it is the shape of what she already owes.

Somewhere among the 2 047 non-empty subsets of those eleven accounts there may be
a set the Bank can settle such that her instalment falls, the R60 000 becomes
affordable, the Bank's exposure improves rather than worsens, and — the part that
gets forgotten — the total amount she will pay before she is debt-free does not
rise so far that she has been harmed by being helped.

This project finds that set, prices it across four products that could each
carry it, proves that the one chosen beats the ones that did not, and hands the
back office the instructions to execute it. Or it declines, with a reason that
survives being read aloud two years later.

The same machinery is then run defensively. When the client is not asking for
money but is visibly failing — three months in arrears, a bounced debit order
pattern, a hardship call — the Bank restructures rather than collects. The
question changes from "what can she afford to borrow" to "what can she afford to
keep paying", the option set grows to include forbearance the Bank grants rather
than credit it advances, and the answer is tested against a stressed view of her
income rather than her stated one.

Two properties make this the hardest flow in the set.

The first is that the search space is exponential and the answer must arrive in
2.5 seconds while the client sits at a desk. Eighteen settleable accounts is
262 143 non-empty subsets; multiplied by four candidate products and nine
candidate terms that is 9 437 148 priced combinations. Fewer than four hundred of
them will ever be evaluated. Which four hundred is a business decision, not an
engineering one, and it must be the *same* four hundred every time the same
client is assessed with the same inputs.

The second is that helping is not the same as lowering the instalment.
Lengthening a term always lowers an instalment and almost always raises the total
cost of credit. R120 000 of card debt at 22% over 36 months costs the client
about R4 583 a month and about R45 000 in interest. The same R120 000 secured
against her home at 12% over 240 months costs about R1 321 a month — a 71%
reduction, and a very easy sale — and about R197 000 in interest, which is more
than four times as much, and now her house is the security. A consolidation flow
that optimises the instalment alone is a machine for manufacturing ombud
complaints. The anti-harm rule in §5.7 exists because of this arithmetic, and
most of §9 exists because somebody will eventually allege that it was not
applied.

---

## 2. Why it is in this set

| Question | How this project stresses it |
|---|---|
| **Q5 core component set** | This is the sharpest Q5 test in the slate. A bounded search over a combinatorial space with a declared budget, a business-authored ordering, a per-candidate evaluation and a tunable objective is not a scorecard, not a decision table and not a decision tree. Neither is "route this candidate to the product logic that can carry it, and let several of them answer". If the core set stays at three kinds, both of these become escape hatches in the first real project — which is exactly the failure doc 01 §5 records. |
| **Q6 codebase organisation** | Four product subflows live inside one flow. Each has its own rate card, its own minimum and maximum, its own eligibility gates, its own fee structure, its own collateral requirements and its own policy owner. They are edited by different teams on different cadences and they must not be four copies of each other. There is no other spec here where product divergence and flow unity collide this directly. |
| **Q2 tables** | Four rate cards with four different key dimensions and, worse, three different *representations*: absolute rates, a promotional-plus-reversion pair, and a margin over a reference rate. Plus a 60 000-row vehicle valuation table, a 200-provider settleability matrix, early settlement charge rules and conveyancing tariffs. |
| **Q3 parameters** | The objective function is configuration. Credit Risk Policy switches between five objectives and blends them by weight, per channel, effective-dated, without a release. So is the evaluation budget, so are fourteen policy interventions, and the same pricing capability runs four times in one assessment with four products' settings. Every one of those is also an overlay target: `core.adjustments` can add basis points to one product, tighten a cap on another and re-weight the objective itself, per product, per channel, for a quarter, without a single artefact being edited or a line of logic changing. |
| **Q1 reuse** | Granting and pricing per product is project 03's logic, reused four times over inputs project 03 never contemplated. Affordability is project 02's, invoked hundreds of times per assessment instead of once. If either has to be forked to serve this, the reuse seam is in the wrong place. |
| **Q4 custom modules** | Settleability classification, settlement amount derivation and the before-and-after comparison are this project's own, but collections (08) and limit management (07) both want the first two. They are written here and consumed there, or they are written twice. |
| **Q7 audit** | Secondary only because project 09 owns the replay machinery — but the evidential demand here is unusual: the record must justify decisions that were *not* taken, about accounts that were *not* settled, under an objective that may since have changed. |

---

## 3. Actors

| Actor | Responsibility |
|---|---|
| **Branch consultant** | Runs the assessment with the client present. Discusses the top three outcomes. Cannot alter the search or the objective; can nominate accounts the client insists on settling and accounts the client refuses to settle. |
| **Contact centre agent** | Runs the same assessment inbound, and fields "why did you not settle my store card" the following week. Depends entirely on recorded rejection reasons. |
| **Collections and Recoveries** | Runs the restructure variant. Grants concessions within an authority limit and escalates above it. |
| **Credit Risk Policy** | Owns the fourteen policy interventions, the evaluation budget, the heuristic ordering set, the settleability rules and the anti-harm thresholds. Not engineers. Changes things quarterly, and faster when a portfolio metric moves. |
| **Product teams** (×4) | Own the per-product caps, minimums, maximums, eligibility gates and policy sets for products 11, 20, 30 and 40. Four separate owners, four separate release cadences, one flow. |
| **Treasury / Pricing** | Owns four rate cards and the reference rate. Refreshes monthly, patches mid-month after a repo change. |
| **Credit Committee** | Owns the objective in force and its weights. Approves concession authority bands. |
| **Regulatory Compliance** | Owns the reckless lending position, the disclosure and comparison requirements, the further advance warning wording, and the debt review interaction. |
| **Back office (Settlements)** | Executes the settlement instructions. Discovers, three days later, that a quotation expired. |
| **Internal Audit** | Samples restructures and asks who authorised what. |
| **Office of the Ombud** | Arrives eighteen months later with a complaint that a client was made worse off. |

---

## 4. Inputs

### 4.1 The request

| Name | Type | Null? | Source | Notes |
|---|---|---|---|---|
| `client_id` | int64 | no | Core banking | Existing client only. This flow does not serve new-to-bank. |
| `application_id` | int64 | no | Origination | |
| `decision_date` | date | no | Caller | Governs every table, rate card, norm and cap resolved in this assessment. |
| `channel_code` | int8 | no | Caller | Branch 45%, app 20%, contact centre 25%, outbound campaign 10%. |
| `requested_amount` | float64 | yes | Caller | New money the client wants *in addition to* settlements. Null in the restructure variant and in batch identification. |
| `assessment_mode_code` | int8 | no | Caller | Consolidation-on-request / consolidation-offered / restructure / batch-identification. Locally declared (§4.7). |
| `client_nominated_settle` | list[int64] | yes | Consultant | Accounts the client insists are settled. 0..8 entries. |
| `client_excluded_settle` | list[int64] | yes | Consultant | Accounts the client refuses to have settled — a bond with a preferential rate, an employer loan, a family arrangement. 0..8 entries. |
| `hardship_declared` | bool | no | Caller | Routes toward the restructure variant even where the client is current. |

### 4.2 The obligation inventory

The client's accounts, assembled from internal systems and from up to three
bureaux, normalised through `core.bureau` and de-duplicated. **0..80 accounts,
ragged.** The median assessed client has 9; the 95th percentile has 23; the
observed maximum in the last two years is 61.

Per account:

| Name | Type | Null? | Source | Notes |
|---|---|---|---|---|
| `account_ref` | int64 | no | Derived | Stable within an assessment and across re-assessments of the same client. Internal accounts key on the core banking identifier; external accounts key on a deterministic hash of provider, account type and opened date, because bureau account numbers are masked. |
| `account_type_code` | int8 | no | `core.bureau` | One of the ~45 types in the library's obligation treatment matrix. |
| `provider_code` | int16 | no | `core.bureau` | ~200 known providers plus an unknown-provider sentinel. |
| `is_internal` | bool | no | Derived | The Bank's own accounts. ~28% of assessed accounts. |
| `balance` | float64 | yes | Bureau / core | Null for accounts reported without balance — 4% of external accounts. |
| `credit_limit` | float64 | yes | Bureau / core | Populated for revolving only. |
| `instalment` | float64 | yes | Bureau / core | Stated instalment. Null or zero for 11% of external accounts, which is why `core.obligations` imputes. |
| `nominal_annual_rate` | float64 | yes | Core / bureau / inferred | Known exactly for internal accounts. Reported by the bureau for 38% of external accounts; inferred from balance, instalment and remaining term otherwise, with an inference-confidence flag. |
| `remaining_term_months` | int16 | yes | Core / derived | Null for revolving and for 9% of term accounts. |
| `months_in_arrears` | int8 | no | Bureau / core | 0..9+. |
| `opened_date` | date | yes | Bureau / core | Drives the recently-opened policy gate. |
| `is_secured` | bool | no | Derived from type | |
| `security_type_code` | int8 | yes | Derived | Vehicle, residential property, commercial property, cession, surety, none. |
| `account_status_code` | int8 | no | Bureau / core | Open, closed, in dispute, under debt review, handed over, written off, ceded. |
| `is_disputed` | bool | no | Bureau | A disputed account may not be settled while the dispute is live. |
| `obligation_treatment_code` | int8 | no | `core.obligations` | Per-element annotation: stated / imputed from limit / imputed from balance / excluded. **This project needs the per-element output that projects 03 and 07 discard** (library §6.4). |

### 4.3 Settleability

Not an input — a classification this flow must produce for every account before
any search can begin (§5.2). Listed here because its *inputs* are inputs.

| Name | Type | Source | Notes |
|---|---|---|---|
| `provider_settlement_policy` | lookup | Settleability table | Does this provider issue settlement quotations to a third party at all? 17 of the ~200 do not. |
| `quotation_turnaround_days` | int8 | Settleability table | Internal accounts: instant. External accounts: 5 working days typical, 2 for the eleven providers on the electronic exchange, 10 for the three that still post. |
| `early_settlement_rule_code` | int8 | Early settlement rules | Whether an early settlement charge applies, on what basis, and its cap. |
| `security_release_days` | int8 | Settleability table | Working days to obtain release of a security interest. Vehicle: 10. Residential bond cancellation: 45–90. |
| `existing_quotation` | record | Quotation store | An already-obtained quotation with its amount, its reference and its expiry. Present for 22% of external accounts at assessment time, because the client was assessed before. |
| `debt_review_status` | enum | Bureau / NCR feed | At client level and at account level. |

### 4.4 Income, expenses and affordability

Supplied by **project 02** through the library capabilities `core.income`,
`core.deductions`, `core.expense_norms`, `core.obligations` and
`core.affordability`. This flow does not re-derive income. It re-derives
*obligations* and therefore affordability, hundreds of times per assessment
(§5.6), which is the single most demanding reuse claim in the slate.

### 4.5 Collateral inputs

Only fetched when a secured product is a live candidate, which is itself a
decision the flow must make before it has evaluated anything.

| Name | Type | Null? | Source | Freshness |
|---|---|---|---|---|
| `vehicle_vin` | string | yes | Client / existing finance record | |
| `vehicle_make_model_year_code` | int32 | yes | Vehicle table | Keys the 60 000-row valuation table. |
| `vehicle_mileage_km` | int32 | yes | Client declaration / service record | Declared at assessment; verified before disbursement. |
| `vehicle_condition_code` | int8 | yes | Inspection | Null until inspected; a null forces a condition haircut at the worst band. |
| `property_valuation_amount` | float64 | yes | Valuation system | Physical valuation, with its date. |
| `property_avm_amount` | float64 | yes | AVM provider | Refreshed monthly for the bonded book. |
| `property_valuation_date` | date | yes | Valuation system | A physical valuation older than 24 months does not support an advance above R300 000. |
| `bond_registered_amount` | float64 | yes | Core banking | The registered bond amount caps a further advance without a new registration. |
| `prior_encumbrances` | float64 | yes | Deeds feed | Second bonds, builders' liens. |

### 4.6 Distress signals (restructure variant)

Arrears ageing over 12 months, debit order failure count over 6 months,
month-on-month salary deposit variance, hardship flags, prior arrangement
history and prior concessions granted with their dates and amounts, and the
client's current provisioning classification. Prior concessions matter twice
over: a second concession within 12 months is a different authority level and a
different reporting consequence.

### 4.7 Names this project declares locally

The library does not publish these. They are declared here, once, and used
unchanged wherever this flow is consumed. A future revision of the library that
publishes any of them supersedes this section.

| Name | Type | Meaning |
|---|---|---|
| `settleability_code` | int8 | §5.2. Nine values. |
| `settlement_amount` | float64 | What it costs to close the account on the assumed settlement date — **not** the balance. |
| `quotation_reference` | string | The provider's reference for a settlement quotation. |
| `quotation_expiry_date` | date | After which the amount is not binding. |
| `settlement_set` | list[int64] | A subset of `account_ref`. |
| `scenario_id` | int32 | A (`settlement_set`, `product_code`, `term_months`, new money) combination, within one assessment. |
| `scenario_rank` | int16 | Position in the final ordering. 1 is the chosen outcome. |
| `objective_id` | int16 | Which objective was in force. |
| `objective_score` | float64 | The scenario's value under it. |
| `instalment_relief` | float64 | Sum of settled instalments less the new instalment, in rands per month. |
| `total_cost_delta` | float64 | New total cost of credit less the settled accounts' remaining cost, in rands. |
| `client_outcome_score` | float64 | The blended client-impact measure (§5.8). |
| `rejection_reason_code` | int16 | Why a scenario was not viable. **Distinct from `decline_reason_codes`**, which are about the client. |
| `concession_code` | int16 | Which forbearance was granted (restructure variant). |
| `authority_level_code` | int8 | The authority that a concession required. |
| `assessment_mode_code` | int8 | §4.1. |

Fifteen locally declared names against a library of twenty-one capabilities is
high. The README says the count is worth watching; this is the project that makes
the count uncomfortable.

### 4.8 Freshness and volumes

| Input | Freshness requirement | Behaviour when stale |
|---|---|---|
| Bureau view | ≤ 14 days at `decision_date` | `bureau_is_stale`; consolidation may proceed but the settlement set may contain only internally-verified accounts |
| Internal balances | Same day | Hard stop |
| Settlement quotations | Unexpired at `decision_date` | Account falls to "quotable but not quoted"; its settlement amount is estimated and the outcome is conditional |
| Vehicle valuation table | Current month's edition | Hard stop for product 30 |
| Property AVM | ≤ 90 days | Falls back to physical valuation requirement |
| Rate cards | Version in force at `decision_date` | Hard stop |

Volumes: **6 000 consolidation assessments per day**, peaking at 900 in the
hour after month-end pay dates. **800 000 clients per month** in the batch
identification run. **1 400 restructure assessments per day**, rising to 4 000 in
the month after a rate increase.

---

## 5. The flow

Eleven stages. Stages 5.5 to 5.8 are the project; the rest exist to make them
possible or defensible.

| # | Stage | Produces |
|---|---|---|
| 5.1 | Intake and eligibility | Whether this client may be assessed at all |
| 5.2 | Obligation inventory and settleability | Which accounts are even candidates |
| 5.3 | Settlement amount derivation | What settling each candidate costs, and until when |
| 5.4 | Baseline assessment | The do-nothing position, and whether the search is needed |
| 5.5 | Candidate scenario generation | Which ≤400 of 9.4 million combinations get evaluated, in what order |
| 5.6 | Per-scenario evaluation | Each candidate re-affordability-tested and priced by product |
| 5.7 | Policy interventions | Which candidates are invalid, and why |
| 5.8 | Objective and selection | The winner, the runners-up, and the demonstration |
| 5.9 | Restructure variant | The same machinery run defensively |
| 5.10 | Output and execution package | What the client is shown and what the back office does |
| 5.11 | Batch identification | The same flow at 800 000 clients and a smaller budget |

### 5.1 Intake and eligibility

**Preconditions.** A `client_id` with at least one internal relationship, a
`decision_date`, an `assessment_mode_code`.

**Determines.** Whether the assessment proceeds, and in which mode.

The library's `core.eligibility` gates apply unchanged — age, capacity,
residency, deceased and estate status, sanctions, exclusion lists. This flow adds
its own gates, and the ordering matters because several of them are cheap and
terminal:

| Gate | Rule | Outcome |
|---|---|---|
| CON-ELIG-01 | Client under debt review at `decision_date` | No new credit may be extended. Consolidation is unavailable. Route to the restructure variant under the debt counsellor path; do **not** emit a plain decline, because a debt review decline reads as a credit refusal and is not one. |
| CON-ELIG-02 | Client under administration or sequestration | Terminal decline. |
| CON-ELIG-03 | A consolidation concluded within the last 6 months | Refer. Serial consolidation is the strongest predictor of the next default and the loudest signal in an ombud file. |
| CON-ELIG-04 | Fewer than 2 settleable accounts after §5.2 | Consolidation is not applicable. Route to plain granting if new money was requested. |
| CON-ELIG-05 | No verified income at `decision_date` | Terminal. |
| CON-ELIG-06 | Bureau view unobtainable (not merely stale) | Terminal for consolidation; the settlement set cannot be established. |
| CON-ELIG-07 | An active reckless lending allegation or complaint against an account in the inventory | Refer to Compliance. |

**Records.** Every gate evaluated and its verdict — not only the failing one.
CON-ELIG-01 in particular must be recorded with the source and date of the debt
review status, because the bureau and the regulator's register disagree often
enough that the discrepancy is itself a finding.

### 5.2 Obligation inventory and settleability

**Preconditions.** Eligibility passed. Bureau normalised, internal accounts
loaded, duplicates resolved.

**Determines.** For each of 0..80 accounts, a `settleability_code` and the
evidence behind it.

**This stage is where the search space is set.** An account that is not settleable
never enters a scenario, and shrinking 61 accounts to 14 settleable ones is worth
more than any cleverness downstream.

Classification, evaluated per account, in the stated order, first match wins:

| `settleability_code` | Name | Condition |
|---|---|---|
| 7 | **Unknown** | Provider unknown, or the attributes needed to classify are absent. Distinct from "not settleable" — library §7.4 applies. These accounts are excluded from settlement sets but **remain in the obligation figure**, and their count is reported, because a scenario built on an inventory with six unknowns is a weaker scenario. |
| 6 | **Blocked by status** | Disputed, under debt review at account level, handed over, in legal process, ceded to a third party, or written off. |
| 5 | **Blocked by policy** | Opened within the last 3 months (CON-INT-02); account type on the non-consolidatable list — court-ordered maintenance, emoluments attachment orders, tax debt, municipal accounts, student loans on concessionary terms. |
| 4 | **Blocked by provider** | The provider does not issue settlement quotations to third parties, or does not accept third-party settlement. 17 providers; 2.3% of external accounts. |
| 3 | **Settleable with security release** | Secured. Settleable only if the security is released, or transfers to the new facility. Carries `security_release_days` and a release cost. |
| 8 | **Partially settleable** | Revolving. Can be paid down and the limit reduced or the facility closed, but "settlement" is a paydown plus a limit action, and the amount is the balance at a future date, which the client can change by spending. |
| 2 | **Settleable, quotation obtainable** | External, provider quotes, no quotation in hand. `settlement_amount` is **estimated**; the outcome is conditional on the actual quotation. |
| 1 | **Settleable, quotation held** | An unexpired quotation exists with a firm amount and expiry. |
| 0 | **Settleable, internal** | The Bank's own account. Amount derivable instantly and exactly. |

**Emits.** Per account: `settleability_code`, the rule that classified it, the
quotation turnaround, the release requirement and its duration, and the early
settlement rule that will apply. At inventory level: counts by class, the
proportion of the client's total obligation that is settleable, and the count of
unknowns.

**Records.** For every account classified 4, 5, 6 or 7 — the reason. The contact
centre question is not only "why did you not settle my store card"; it is
frequently "why was my store card not even considered", and those are different
answers with different remedies.

**A constraint that surprises people.** Client nominations and exclusions are
applied here, not in the search. An account the client refuses to have settled is
removed from candidacy (and the refusal recorded); an account the client insists
on is marked mandatory, and a scenario that omits a mandatory account is not
generated. A mandatory account that is not settleable is a hard conflict that
must be surfaced to the consultant, not silently dropped.

### 5.3 Settlement amount derivation

**Preconditions.** Settleability classified.

**Determines.** For each settleable account, what it costs to close it, and the
date on which that figure stops being true.

**The settlement amount is not the balance.** Stating this in the specification
because it has been got wrong in production by every lender that has ever built
this, and the failure mode is a shortfall discovered by the back office after
disbursement, which is an unsecured unauthorised advance.

```
settlement_amount =
      outstanding capital at the assumed settlement date
    + interest accrued from the last statement to that date (per diem × days)
    + fees and charges unpaid at that date
    + early settlement charge, where permitted and applicable
    + security release, cancellation or de-registration cost, where secured
    − unearned service fees and premiums refunded on early settlement, where the
      product provides for a rebate
```

Each term has requirements of its own:

- **The assumed settlement date** is `decision_date` + the provider's payment
  turnaround + the Bank's own disbursement lag, per provider, from the
  settleability table. It ranges from 1 to 12 working days. The amount depends on
  it, so it must be recorded alongside the amount.
- **Per diem accrual** means the true amount is only knowable on the day the
  money arrives. The flow therefore computes the settlement amount at the assumed
  date and adds a **settlement buffer** — 1.5% of the settlement total, capped at
  R2 500 — to the advance. The buffer is a tunable. Where the actual settlement
  is lower, the residue is applied to the new facility; where it is higher, the
  shortfall is the client's, which must be disclosed.
- **Early settlement charges** follow the published mechanism: below the
  large-agreement threshold no charge arises; at or above it, a fixed-rate
  agreement may attract a charge not exceeding a stated number of months'
  interest where the required notice is not given. Bonds are the sharp case — the
  cancellation notice period is long enough (illustratively 90 days) that the
  charge is routinely incurred and routinely forgotten.
- **Rebates** cut the other way and are also forgotten. Some products refund
  unearned monthly service fees and unearned credit life premium on early
  settlement. Omitting the rebate overstates the required advance, which the
  client pays interest on for the next five years.

**Emits.** Per settleable account: `settlement_amount`, its component breakdown,
the assumed settlement date, `quotation_reference` and `quotation_expiry_date`
where a quotation is held, an `amount_basis_code` (quoted / derived-internal /
estimated), and an estimation tolerance where estimated.

**Records.** The full component breakdown per account. When the back office
settles R41 218.66 against a figure of R40 903.12 shown to the client, the
difference must be attributable to a named component.

### 5.4 Baseline assessment

**Preconditions.** Settlement amounts derived. Income established once for the
assessment (§5.6 requires that it never varies).

**Determines.** The do-nothing position, and whether the search is warranted.

Compute, with no consolidation:

| Measure | Definition |
|---|---|
| `existing_obligations` | `core.obligations` over the full inventory |
| Current total instalment | Sum of treated instalments |
| Weighted average rate | Balance-weighted `nominal_annual_rate` across the inventory, with the inference-confidence flags propagated |
| Total remaining cost | Sum over term accounts of (remaining instalments × instalment), plus, for revolving accounts, the cost of amortising the balance at its current rate over the policy paydown horizon (36 months). **Revolving debt has no natural term, so the comparison needs an assumed one, and the assumption must be a declared parameter rather than a number in someone's head.** |
| Longest remaining term | The horizon the client is currently committed to |
| Debt service ratio | `existing_obligations` / `net_monthly_income` |
| Baseline affordability | `core.affordability` for `requested_amount` as a plain advance, no consolidation |

**The short-circuit.** If the request passes affordability as made, consolidation
may be unnecessary. It is unnecessary — and the flow should complete as plain
granting under project 03 — when **all** of:

- the request is affordable as made, with the post-advance instalment leaving
  discretionary income above the product's buffer; and
- post-advance debt service ratio ≤ 40%; and
- no account is in arrears; and
- the inventory's weighted average rate does not exceed the rate the client would
  achieve on a consolidation by more than 300 basis points; and
- fewer than 6 active credit accounts; and
- no settleable account carries a rate at or above 28%.

Consolidation remains preferable, and must be evaluated and **presented
alongside** the plain offer even when the plain offer passes, when any of:

- an account is in arrears; or
- the weighted average rate exceeds the achievable consolidation rate by more
  than 300 basis points; or
- debt service ratio exceeds 40%; or
- there are 6 or more active accounts, on the grounds that the failure rate of a
  household managing nine debit orders is materially higher than one managing
  three; or
- a settleable account carries a rate at or above 28%.

Where the plain offer fails affordability, the search is not optional.

**Records.** The baseline measures, the short-circuit evaluation with each
condition's verdict, and — where the flow short-circuited — the fact that
consolidation was considered and why it was not pursued. A client told "we can
only lend you R14 000" who later discovers a consolidation would have released
R60 000 is a complaint, and the answer must be on file.

### 5.5 Candidate scenario generation

**Preconditions.** Settleability and settlement amounts established. Baseline
computed. Mandatory and excluded accounts applied.

**Determines.** Which scenarios are evaluated, and in what order.

**The size of the problem.** For a client with 18 settleable accounts there are
2¹⁸ − 1 = **262 143** non-empty settlement sets. Each may be carried by up to
four products, at up to nine candidate terms each — **9 437 148** scenarios. The
median client, at 9 settleable accounts, still presents 511 sets and about 18 400
scenarios. Neither is evaluable inside 900 milliseconds. A design that attempts
exhaustive evaluation fails on the 95th-percentile client, which is precisely the
client consolidation exists for.

**The requirement is therefore a bounded search**, with these properties:

1. **A declared evaluation budget.** At most **400 scenario evaluations** and at
   most **900 ms** of search time in interactive modes; at most **40** and
   **60 ms** in batch identification. Both are parameters owned by Credit Risk
   Policy, not constants.
2. **Budget exhaustion is a recorded fact, not a silent truncation.** The output
   must state the budget, the consumption, and whether the search terminated
   because it ran out of candidates or ran out of budget. A client whose search
   was truncated is in a different position from one whose space was exhausted,
   and the contact centre needs to know which.
3. **Determinism.** The same client, the same inputs, the same `decision_date`,
   the same parameter versions and the same adjustment stack ⇒ the same scenarios,
   in the same order, with the same winner. No dependence on wall-clock time, iteration order over an
   unordered collection, floating-point accumulation order, or the number of
   available workers. Where the budget is expressed in milliseconds, the flow
   must also carry a deterministic candidate-count bound, because a time-based
   cut-off is not reproducible.
4. **A total ordering.** Every ordering rule below must resolve to a total order;
   ties break on `account_ref` ascending, then `product_code` ascending, then
   `term_months` ascending.
5. **A recorded trace.** Which scenarios were generated, in what order, which
   were evaluated, which were rejected and why, and where the budget stopped.
6. **The overlay stack is an input, not an ambient condition.** Adjustments
   (library §6.22) can add basis points to one product's rate, tighten another's
   loan-to-value cap, multiply a probability of default, and re-weight the
   objective — all within one assessment. Two assessments of the same client on
   the same inputs and the same code, under different overlay stacks, are
   *expected* to select different winners. The stack in force at `decision_date`
   — its members, their scopes, their composition order — must therefore be
   pinned into the search record beside the parameter and table versions. A
   replay that does not pin it is not a replay, and the difference it produces
   will be reported as a defect.

**The ordering rules.** These are business rules. They are owned and edited by
Credit Risk Policy, they are effective-dated, and the set in force at
`decision_date` is part of the evidence.

| # | Rule | Rationale |
|---|---|---|
| H1 | Settle the highest effective rate first | The cheapest source of client benefit. An account at 31% is worth settling before one at 14%. |
| H2 | Settle the highest instalment relief per rand of settlement first | R800 a month released for a R14 000 settlement beats R900 released for a R210 000 settlement, when the constraint is affordability. |
| H3 | Settle the shortest remaining term last | An account with four instalments left releases its instalment in four months for free. Settling it consumes advance and releases almost nothing. |
| H4 | Never include a secured account unless its security releases, or transfers to the new facility | A settled vehicle finance whose security does not move leaves the Bank unsecured on a depreciating asset, and the client uninsurable. |
| H5 | Prefer accounts with a provider the client is in arrears with | Arrears are contagious across a provider relationship; settling one account of three at a provider in arrears removes a debit order but not the relationship. |
| H6 | Prefer accounts whose quotation is in hand and unexpired | A scenario built on held quotations is executable this week. A scenario built on estimates is conditional. |
| H7 | Prefer settling a provider relationship in full over settling part of it | Leaving a R380 balance open at a provider the client has otherwise exited produces a forgotten account and a default listing. |
| H8 | Prefer accounts with the highest re-accumulation risk | A store card settled and left open will be re-used. Where the facility cannot be closed, prefer settling it early and requiring closure as a condition. |

**What the search must generate.** The specification does not say how the
candidates are enumerated, but it constrains what must be present among them:

- the **empty set** — the baseline, so the chosen outcome is always compared to
  doing nothing;
- for each ordering rule, the **prefixes** of that ordering — settle the top 1,
  top 2, … top *k* by that rule — because a business-authored ordering that never
  produces a candidate is a rule nobody can validate;
- the **client's nominated set**, and that set extended by each ordering;
- the **full settleable set**, which is frequently unaffordable but is the
  client's mental model and must be priced so it can be shown to have been;
- for each generated settlement set, the **product routings** that can carry it
  (§5.6) and, within each, a bounded set of candidate terms — the product
  minimum, the maximum permitted by CON-INT-05, and the terms that bracket the
  affordability constraint;
- and, where a mandatory account exists, only sets that contain it.

**What the search must not do.** It must not stop at the first affordable
scenario. The first affordable scenario under H1 is rarely the best under the
objective in force, and an implementation that treats affordability as a stopping
condition rather than a constraint silently redefines the objective.

**Records.** The ordering rule set and its version, the generated candidate list
with generation order, the budget, the consumption, the termination cause.

---

### 5.6 Per-scenario evaluation

**Preconditions.** A generated scenario: a settlement set, a product, a term, a
new money amount. Baseline established. Budget not exhausted.

**Determines.** Whether this scenario is viable, what it costs, and what it does
for the client and for the Bank.

**Terminology, because the distinction matters downstream.** A **settlement set**
is what the client cares about — which debts go away. A **scenario** is a
settlement set carried by a specific product over a specific term with a specific
amount of new money riding along. One settlement set commonly produces six to
twelve scenarios, because up to four products can carry it and each offers
several terms. The evaluation budget is counted in scenarios. The outputs shown
to the client are grouped by settlement set, because "we could settle these five
accounts, and here are three ways of doing it" is the conversation a consultant
actually has.

#### 5.6.1 What varies and what must not

This is the reuse requirement that will hurt.

**Invariant across every scenario in an assessment**, established exactly once:
`gross_monthly_income`, `income_source_code`, `income_haircut_applied`,
`statutory_deductions`, `net_monthly_income`, `living_expenses`,
`expense_basis_code`, `dependants_count`, and the bureau-derived risk
characteristics that do not depend on the settlement set.

Two scenarios in one assessment that disagree about the client's income are a
defect, not a difference of opinion. The requirement is stated positively: the
income and expense determination is performed once per assessment, and every
scenario evaluation is required to consume the same result. An implementation
that re-derives income per scenario is wrong even when it happens to agree, and
an implementation that re-derives it per scenario and disagrees by a cent has
produced an assessment nobody can defend.

**Varies per scenario**: the obligation set, and everything downstream of it.

#### 5.6.2 The recomputation

For each scenario:

1. **Reduce the inventory.** Remove the settled accounts. For partially
   settleable revolving accounts, reduce the balance to zero and — where the
   product policy requires closure or limit reduction — reduce or remove the
   limit, because a settled card with its limit intact is a re-accumulation
   waiting to happen and `core.obligations` must see the post-action state.
2. **Re-derive obligations.** `core.obligations` over the reduced inventory,
   producing a new `existing_obligations`, a new `worst_arrears_months`, a new
   `revolving_utilisation` and fresh per-account treatments. The per-element
   output is required here — library §6.4's "one capability, two shapes of
   answer" problem is this project's problem.
3. **Compute the required advance:**

   ```
   required_advance =
         Σ settlement_amount over the settlement set
       + settlement buffer (1.5%, capped R2 500)
       + new money requested
       + capitalised initiation fee
       + capitalised product-specific costs (valuation, registration,
         de-registration, transfer duty where applicable)
   ```

   Capitalised fees depend on the advance, and the advance depends on the fees.
   The rate depends on the amount band, and the amount depends on the rate
   through the capitalised credit life premium. This is project 03's circular
   solve, appearing here once per scenario rather than once per application. It
   must converge, it must have a declared iteration bound — six iterations, R1
   tolerance — and non-convergence must be a recorded scenario rejection rather
   than a thrown error.
4. **Re-derive affordability.** `core.affordability` over the new obligation
   figure and the scenario's own instalment, producing a new
   `discretionary_income`, `max_affordable_instalment` and
   `affordability_verdict_code`.
5. **Price it** through the routed product's logic (§5.6.4 to §5.6.7).
6. **Measure it** — `instalment_relief`, `total_cost_delta`, the new weighted
   average rate, the new debt service ratio, the Bank's expected value, and
   `client_outcome_score`.

**The cost of this.** Each scenario evaluation invokes, at minimum,
`core.obligations`, `core.affordability`, `core.rate_card`, `core.fees`,
`core.credit_life`, `core.instalment` (one to six times inside the solve),
`core.adjustments` and `core.rounding` — eight to thirteen library invocations.
At 400 scenarios that is **3 200 to 5 200 library invocations inside a 900 ms
search budget**, or roughly 200 microseconds each. The library's stated
non-functional requirement is a whole-library contribution under 15 ms at p99
*per application* (library §9). This project consumes that budget two hundred
times over in a single assessment. Either the library's envelope means something
different for iterative consumers, or one of the two numbers is wrong. The
specification states the requirement and leaves the resolution to the
implementation, because which way it resolves is diagnostic.

#### 5.6.3 Product routing

A scenario is routed to exactly one product. Which products *can* carry a given
settlement set is a rule, not a preference:

| Product | May carry a settlement set when |
|---|---|
| **11 Flex Loan Consolidation** | Any settleable set of 2 or more accounts, provided no secured account remains secured. The universal fallback: if any product can carry a set, this one usually can. |
| **30 Drive Finance refinance** | The set contains the client's existing vehicle finance account, or the client offers an unencumbered vehicle that passes the age and mileage gates. The vehicle finance account, where it exists, is mandatory in the set — a vehicle cannot be refinanced without settling what is currently secured on it. |
| **20 Everyday Card balance transfer** | Every account in the set is a card or revolving facility on the transferable account type list. A set containing one term loan cannot be routed here. |
| **40 Home Loan Further Advance** | The client holds a bond with the Bank, available equity covers the advance, and the client has acknowledged the security warning (§5.6.7). |

Where two or more products can carry a set, all of them are candidate scenarios
and all are evaluated, subject to budget. A set that only one product can carry
is not thereby preferred — it is simply cheaper to evaluate.

#### 5.6.4 Flex Loan Consolidation — product 11

Unsecured, settlement-linked, the default route. Proceeds are paid directly to
the creditors; only new money reaches the client.

| Aspect | Requirement |
|---|---|
| Amount | R10 000 – R500 000 |
| Term | 12 – 84 months |
| Rate | Own rate card: 72 amount bands × 73 terms × 12 grades = 63 072 cells, absolute annual rates. Plus a margin adjustment grid keyed on the proportion of the advance paid to *external* creditors (4 bands × 12 grades = 48 cells), because refinancing the Bank's own book at a consolidation rate is not the same transaction as taking a client's debt off a competitor. |
| Grade | A consolidation-specific scorecard, with characteristics plain granting does not have: the count of accounts being settled, the proportion of income currently servicing debt, the number of providers exited, and whether the client has consolidated before. |
| Fees | `core.fees` initiation and monthly service; `core.credit_life` premium. |
| Own policy set | Minimum 2 accounts settled. At least 60% of the advance must go to external creditors, so that the product is not a disguised cash loan. Maximum 8 accounts settled (CON-INT-01). New money ≤ 25% of the settlement total, capped at R50 000. Minimum instalment relief 10%. Maximum term: the lesser of 84 months and the longest settled remaining term + 24. |
| Affordability treatment | Standard: the contractual instalment is the tested figure. |
| Overlay targets | Rate add-on in basis points, scoped to product 11 and optionally to channel; maximum amount reduction; minimum instalment relief raised for a quarter. |
| Its own rejections | Fewer than 2 accounts; external proportion below 60%; advance below R10 000 or above R500 000; instalment relief below the floor. |

#### 5.6.5 Drive Finance refinance — product 30

Secured against a vehicle. Structurally the most different from the others,
because the collateral has to be valued, its condition affects the price, and the
security has to move from one provider to another while the client keeps driving
the car.

| Aspect | Requirement |
|---|---|
| Amount | R30 000 – R1 500 000 |
| Term | 12 – 60 months for a refinance, against 12 – 72 for new business. A refinance may not extend the total financed life of the vehicle beyond 84 months from original registration. |
| Valuation | Trade guide lookup on make/model/year/derivative (~60 000 rows), adjusted for mileage against a kilometre-band table and for condition against a condition haircut table. A null condition code forces the worst band, which is a deliberate incentive to inspect. |
| Loan-to-value | `(advance + balloon) / adjusted retail value`, against a cap by vehicle age: ≤3 years 110%, 4–6 years 100%, 7–9 years 85%, ≥10 years ineligible. **LTV is a key dimension of the rate card**, which means the price depends on the amount through two channels — the amount band and the LTV band — and the circular solve in §5.6.2 has to survive both. |
| Vehicle gates | Age at end of term ≤ 12 years. Mileage ≤ 220 000 km at inception, and implied annual mileage ≤ 40 000 km. |
| Rate | The library's Drive Finance card: 60 amount bands × 61 terms × 12 grades × 5 LTV bands = 219 600 cells, plus a refinance-specific vehicle age adjustment (8 age bands × 5 LTV bands = 40 cells). |
| Balloon / residual | Permitted to 30% for terms ≤ 48 months, 20% at 60 months, 0% above. A balloon lowers the instalment and is therefore attractive to the objective — and it leaves a lump sum at the end. **The total cost of credit used by the anti-harm rule must include the balloon**, or the rule can be defeated by structure rather than by argument. |
| Security | The existing provider's interest must be released and the Bank's registered. 10 working days typical. The settlement cannot be executed until the release is arranged, and the quotation may expire in the interim (§5.10). |
| Fees | Initiation, monthly service, valuation fee, registration fee, and comprehensive insurance as a condition precedent — an uninsured vehicle is not security. |
| Affordability treatment | The contractual instalment, plus a required insurance premium estimate where the client does not already hold cover. Omitting the insurance premium overstates affordability by R700 to R1 400 a month on a typical vehicle. |
| Overlay targets | LTV cap reduction (a tightening applied without reissuing the rate card); vehicle age gate tightened; rate add-on by LTV band. |
| Its own rejections | Vehicle not identifiable; valuation unavailable; LTV above cap; vehicle age or mileage gate; existing finance account not settleable; insurance refused. |

#### 5.6.6 Everyday Card balance transfer — product 20

Revolving. The odd one out, because it has no term and no instalment, and both
the objective and the anti-harm rule are defined in terms of things it does not
have.

| Aspect | Requirement |
|---|---|
| Limit | R1 000 – R300 000. The output is a **limit**, not an amount. |
| Transferred balance | ≤ 80% of the approved limit, so the client is not immediately at the ceiling. |
| Rate | A promotional rate for a fixed period, then a reversion rate. Promotional card: 12 grades × 8 limit bands × 5 promotional durations (0, 6, 12, 18, 24 months) = 480 cells. Reversion: the library's revolving card, 12 grades × 20 limit bands = 240 cells. **This is a rate card with two rates and a date**, which no other product in this flow has, and it is the reason a rate card cannot be assumed to return a scalar. |
| Payment | A minimum payment formula, not an instalment: the greater of R50 and (3.0% of balance + interest + fees). |
| Mandatory paydown | The transferred portion carries a contractual schedule: at least 1/36 of the transferred amount per month, in addition to the minimum payment, so the transferred balance amortises within 36 months. Without this, a balance transfer is an indefinite extension at a reverted rate and the anti-harm rule has nothing to measure. |
| Transfer fee | 2.5% of the transferred amount, added to the utilised balance. |
| Affordability treatment | **Not the promotional minimum payment.** Affordability is tested against a stressed payment: the reversion rate applied to the full approved limit, amortised over 36 months. A client who can afford the R310 promotional minimum but not the R1 240 stressed payment has been sold a cliff. |
| Own policy set | Card and revolving accounts only. Settled revolving accounts must be closed or limit-reduced by the transferred amount as a condition; the condition must appear in the execution package. Unavailable where more than one revolving account is in arrears. Unavailable where the client has taken a balance transfer within 12 months. |
| Total cost comparison | Computed over the mandatory paydown schedule at the promotional rate for the promotional period and the reversion rate thereafter — not at the promotional rate throughout, which is how this product is mis-sold. |
| Overlay targets | Promotional duration capped (24 months withdrawn for a quarter); transfer fee waived by segment; reversion rate add-on. |
| Its own rejections | A non-revolving account in the set; transferred balance above 80% of limit; more than one revolving account in arrears; stressed payment unaffordable. |

#### 5.6.7 Home Loan Further Advance — product 40

Secured against the client's home. The most powerful instalment reduction
available, and the one most likely to end up in front of the ombud.

| Aspect | Requirement |
|---|---|
| Precondition | An existing bond with the Bank, in good standing, not in arrears, not in legal process. |
| Valuation | An automated valuation is sufficient where the last physical valuation is under 24 months old and the advance is below R300 000. Otherwise a physical valuation is a condition precedent, adding 10 to 15 working days. |
| Available equity | `min(property value × LTV cap, bond registered amount) − outstanding home loan balance − prior encumbrances`. Where the advance exceeds the registered bond amount, a new registration is required, which changes the cost and the timeline materially. |
| LTV cap | 90% owner-occupied, 75% investment, 80% where the bond is under 12 months old. |
| Term | The lesser of 240 months and the remaining bond term, and it may not extend beyond the client's 70th birthday. |
| Rate | **A margin over the reference rate, not an absolute rate**: 12 grades × 8 LTV bands × 6 term bands × 2 occupancy types = 1 152 cells of basis points. A fourth rate card with a fourth key set and a third representation. |
| Fees | Initiation fee, bond registration cost by a conveyancing tariff table (~24 amount bands), valuation fee, and — where a new registration is needed — deeds office and attorney costs. |
| Mandatory sub-term structure | **The consolidated portion must be contractually amortised over at most 84 months**, by way of a required additional payment above the bond instalment, and the anti-harm rule is evaluated against that structure rather than against the bond's remaining term. Without this, the product's total cost is indefensible: R120 000 of card debt moved to a 240-month bond at 12% costs the client roughly R197 000 in interest against roughly R45 000 where it stood, a 337% increase, for a 71% instalment reduction. The instalment reduction is real, the client will want it, and it is still the wrong answer for most clients. |
| Warning requirement | An explicit, versioned warning that unsecured debt is being secured against the client's home, and that the consequence of default changes from a judgment to the loss of the property. The warning must be displayed, its version recorded, and the client's acknowledgement captured with timestamp, channel and the identity of the person who presented it. **No scenario on product 40 may be selected without a recorded acknowledgement.** |
| Timeline | Registration takes 6 to 10 weeks. Every settlement quotation in the set will expire before disbursement. Product 40 scenarios are therefore always conditional and always require re-derivation (§5.10). |
| Overlay targets | LTV cap reduction; probability-of-default multiplier feeding `risk_grade` and therefore the margin; the anti-harm threshold itself. |
| Its own rejections | No bond, or bond in arrears; insufficient equity; LTV above cap; valuation unavailable or stale; term beyond age limit; warning not acknowledged; sub-term structure unaffordable. |

#### 5.6.8 What each evaluation emits and records

**Emits** per scenario: `scenario_id`, the settlement set, `product_code`,
`term_months`, `offered_amount`, `nominal_annual_rate` with its unadjusted
counterpart and the overlays that applied, `instalment`, `total_cost_of_credit`,
`effective_annual_rate`, the fee breakdown, the new `existing_obligations`,
`discretionary_income`, `affordability_verdict_code`, `instalment_relief`,
`total_cost_delta`, the new weighted average rate, the new debt service ratio,
the Bank's expected value, `client_outcome_score`, conditions precedent, and a
viability verdict.

**Records**: the rate cell identifier and rate card version, the overlay
identifiers and their effects in composition order, the table versions resolved,
the solve iteration count, and — for every non-viable scenario — the
`rejection_reason_code` set.

### 5.7 Policy interventions inside the search

**Preconditions.** A priced scenario.

**Determines.** Whether the scenario is permitted, irrespective of how good it
looks under the objective.

These fire during evaluation and constrain it. They are owned by Credit Risk
Policy, effective-dated, tunable within declared ranges, and several of them are
overlay targets — meaning a scenario can be rejected by an overlaid threshold
rather than a base one, and the record must say which.

| # | Intervention | Default | Tunable range | Scope |
|---|---|---|---|---|
| CON-INT-01 | Maximum accounts settled in one consolidation | 8 | 3 – 15 | Per product |
| CON-INT-02 | No settlement of an account opened within N months | 3 | 0 – 12 | Global |
| CON-INT-03 | New instalment at least X% below the sum of settled instalments | 10% | 0 – 30% | Per product, per channel |
| CON-INT-04 | **Anti-harm**: new total cost of credit may not exceed the settled accounts' remaining cost by more than Y% | 15% | 0 – 60% | Per product |
| CON-INT-05 | Maximum term extension over the longest settled account's remaining term | +24 months | 0 – +60 | Per product |
| CON-INT-06 | Rate ceiling: new rate may not exceed the balance-weighted average rate of the settled accounts | binding | on / off / +N bps | Per product |
| CON-INT-07 | No consolidation for a client under debt review | binding | not tunable | Global |
| CON-INT-08 | New money ≤ Z% of the settlement total, capped at an absolute amount | 25%, R50 000 | 0 – 50%, R0 – R150 000 | Per product, per channel |
| CON-INT-09 | Post-consolidation debt service ratio ceiling | 45% | 30 – 60% | Per product, per grade |
| CON-INT-10 | Minimum proportion of the advance paid to external creditors | 60% | 0 – 100% | Per product |
| CON-INT-11 | No settlement of an account in dispute | binding | not tunable | Global |
| CON-INT-12 | Secured accounts may only be settled where the security releases or transfers | binding | not tunable | Global |
| CON-INT-13 | Maximum consolidations per client per rolling 24 months | 2 | 1 – 3 | Global |
| CON-INT-14 | Minimum post-consolidation discretionary income after the new instalment | R850 | R500 – R2 500 | Per product, per household size |

**Requirements on the mechanism:**

1. **A violated intervention invalidates the scenario.** It does not adjust it,
   score it down, or refer it. The scenario is removed from selection.
2. **Every violation is recorded, per scenario, with its `rejection_reason_code`
   and the actual and threshold values.** "Rejected by CON-INT-04: total cost
   +23.4% against a threshold of 15%" is an answer. "Rejected by policy" is not,
   and the contact centre will ask, because the client will ask.
3. **All violations are evaluated, not only the first.** A scenario rejected by
   four interventions is a different conversation from one rejected by a single
   marginal breach, and the second kind is the one a consultant may usefully
   escalate.
4. **Where an overlay changed a threshold, the record must name the overlay**,
   its identity, its approval reference, its effective window and the base value
   it replaced. A scenario that would have passed at the base threshold and
   failed at the overlaid one is the single most likely subject of a later
   query, from Credit Committee as often as from a client.
5. **Interventions must be evaluable in isolation**, so that Credit Risk Policy
   can ask "how many of last month's assessments would have changed if CON-INT-04
   moved to 12%" without re-running the world by hand. Project 09 owns the
   harness; this flow owns being harness-able.
6. **Not-applicable is distinct from passed.** CON-INT-10 does not apply to
   product 20. A record showing it as passed is misleading.

---

### 5.8 Objective and selection

**Preconditions.** A set of evaluated scenarios, each viable or rejected with
reasons.

**Determines.** Which scenario is offered, which are shown as alternatives, and
why.

**The objective is configuration.** The Bank does not have one objective; it has
five, it switches between them, and it blends them. It does this because the
business reason for consolidating changes — in a growth quarter the point is new
money, after a rate increase the point is instalment relief, and under
supervisory attention the point is client outcome.

| `objective_id` | Objective | Measure |
|---|---|---|
| OBJ-01 | Maximise new money released | New money in the scenario, subject to every constraint |
| OBJ-02 | Minimise the client's monthly commitment | New instalment plus retained obligations |
| OBJ-03 | Minimise total cost of credit | New total cost of credit plus the remaining cost of retained accounts |
| OBJ-04 | Maximise the Bank's expected value | Incremental margin over cost of funds across the expected life, less expected loss (PD × LGD × exposure), **less the margin forgone on the Bank's own accounts that the scenario settles**. A consolidation that refinances the Bank's own 26% loan at 19% destroys value, and an objective that measures gross rather than incremental margin will happily recommend it. |
| OBJ-05 | Maximise client outcome | A weighted score over the instalment relief ratio, the total cost increase ratio (negatively), the proportion of accounts exited, the reduction in weighted average rate, and the reduction in the number of providers |

**Requirements:**

1. **The objective in force is a parameter**, resolved by `decision_date`,
   `channel_code`, segment and `assessment_mode_code`. Changing it — including
   changing a blend's weights — must not require a code change or a release.
2. **Blends are permitted**: a weight vector over the five measures summing to
   1. The weight vector is itself an overlay target. "For this quarter, weight
   instalment relief at 0.6 and new money at 0.1 in the branch channel" is an
   approved, effective-dated, expiring overlay, and it changes which scenario
   wins without a single rule changing. The overlay's identity must appear in the
   selection record, or the decision becomes inexplicable the moment the overlay
   lapses.
3. **Every component measure must be absolute, not set-relative.** Expressing a
   component as "this scenario's instalment relief, scaled between the best and
   worst relief among the evaluated scenarios" makes the winner depend on which
   losers happened to be evaluated — which means the budget cut-off changes the
   answer, and a replay with a larger budget produces a different winner from the
   same inputs. Components are ratios to the baseline. Where a set-relative
   measure is genuinely unavoidable, the evaluated set becomes part of the
   objective's recorded inputs.
4. **Tie-breaking is declared and deterministic**: lowest total cost of credit,
   then fewest accounts settled, then lowest `scenario_id`.
5. **An indifference band.** Where the winner's objective score exceeds the
   runner-up's by less than 2%, both must be presented as materially equivalent
   rather than ranked, because a 0.4% difference in a modelled expected value is
   not a difference a consultant should defend to a client.
6. **The top three are returned**, each independently viable, each having passed
   every intervention. A rejected scenario is never a runner-up. The three must
   be **distinct in substance** — at most one per (settlement set, product)
   combination — because three terms of the same product over the same accounts
   is one alternative presented three times.
7. **Superiority must be demonstrable.** For the winner against each runner-up:
   the objective score of each, the per-component decomposition, and which
   component accounts for the difference. "Scenario 44 was chosen over scenario
   17 because it releases R412 more per month, at a total cost R9 100 higher,
   under an objective weighting instalment relief at 0.6" is the sentence the
   record must be able to produce.
8. **The client-outcome shadow.** Whatever objective is in force, the best
   scenario under OBJ-05 must also be identified and recorded. When the objective
   in force was OBJ-01 and the client later complains, the question asked will be
   "what was the best outcome available to this client", and the only defensible
   position is to have known the answer at the time.

**Records.** `objective_id`, the weight vector and its source, the overlay stack
affecting it, every scenario's objective score with component decomposition, the
final ranking, the indifference band evaluation, and the shadow result.

### 5.9 The restructure variant

**Preconditions.** `assessment_mode_code` is restructure, or eligibility routed
here from CON-ELIG-01. The client is in arrears, at risk, or has declared
hardship.

**Determines.** What the Bank can offer a client who cannot pay, such that the
client stays in the book and the Bank's loss is smaller than under the do-nothing
path.

The machinery of §5.5 to §5.8 is reused unchanged. What changes:

**The objective.** Sustainability, not new money: maximise the probability that
the client remains current for 12 months, subject to the Bank's expected loss not
exceeding the expected loss under the do-nothing path. New money is unavailable
except for the capitalisation of arrears and fees.

**The option set grows.** Forbearance options are things the Bank grants on
existing agreements rather than credit it advances, and they combine:

| `concession_code` | Concession | Bounds |
|---|---|---|
| CNC-01 | Payment holiday | 1 – 3 months, interest continuing to accrue and capitalising |
| CNC-02 | Term extension on an existing internal agreement | Up to +36 months, not beyond the product maximum |
| CNC-03 | Rate concession | Up to −400 basis points, for 6, 12 or 24 months, or for the remaining term |
| CNC-04 | Arrears capitalisation | Up to 6 months of arrears folded into the balance |
| CNC-05 | Temporary instalment reduction with step-up | 50 – 80% of contractual for 6 or 12 months, then contractual or above |
| CNC-06 | Fee and interest waiver | Up to R15 000 |
| CNC-07 | Partial capital forgiveness | Highest authority only |
| CNC-08 | Consolidation into product 11 at a concessionary rate | Where affordability supports it |

Combinations are options. A payment holiday followed by a term extension is a
distinct option from either alone, and the search must be able to consider it —
which enlarges the space again, since the option set multiplies the settlement
set space.

**Authority limits.** Every concession carries a cost, measured as the net
present value forgone against the contractual position, and that cost determines
who may approve it:

| `authority_level_code` | Authority | NPV cost of concession |
|---|---|---|
| 1 | Collections consultant | ≤ R5 000 |
| 2 | Team leader | R5 000 – R25 000 |
| 3 | Credit Risk manager | R25 000 – R150 000 |
| 4 | Credit Committee | Above R150 000, and any CNC-07 |

The flow must compute the NPV cost, determine the required authority, and either
apply the concession (where the running actor holds it) or produce a referral
carrying everything the approver needs. A second concession to the same client
within 12 months raises the required authority by one level.

**Reporting consequences.** A concession granted because of financial distress
may constitute a distressed restructure, with consequences for provisioning
classification and for the client's bureau status. The flow must classify each
option as distressed or commercial, must record the classification and its basis,
and must surface it before the concession is granted — not after, when the
provisioning report finds it.

**Stressed affordability.** A restructure that only works if nothing else goes
wrong is not a restructure. Every option is tested against a stressed view:
income reduced by 10%, living expenses increased by 8%, and retained
variable-rate obligations repriced 200 basis points higher. Both verdicts —
standard and stressed — are recorded. An option passing standard and failing
stressed may be offered only with a recorded acknowledgement that it was, and
only up to authority level 2.

**Debt review interaction.** A client under debt review may not take new credit,
but their existing agreements can be restructured through the debt counsellor or
court-ordered path. The flow must detect this, route to that path, and produce an
output the debt counsellor can use. Emitting a decline is wrong, and it is what a
flow that treats debt review as a decline gate will do.

**Records.** Every option considered, its NPV cost, its authority requirement,
both affordability verdicts, its distressed classification, the concession
granted, the authority that granted it, and the options rejected with reasons.

### 5.10 Output and execution package

**Preconditions.** A selected scenario, or a decline.

**Determines.** What the client is shown, what the client agrees to, and what the
back office does on Monday.

**The before-and-after comparison.** Mandatory, and its content is prescribed
because this is the artefact an ombud will read first:

| | Before | After |
|---|---|---|
| Monthly instalment | Sum of treated instalments across all accounts | New instalment plus retained obligations |
| Weighted average rate | Balance-weighted across the inventory | The new facility's rate, blended with retained accounts |
| Longest remaining term | From the inventory | The new facility's term |
| Total cost to settle everything | Sum of remaining costs | New total cost of credit plus retained remaining cost |
| Number of accounts | Count active | Count after settlement |
| Number of providers | Count distinct | Count after settlement |

Where the total cost rises, the comparison must say so **explicitly, in rands and
as a percentage, with the reason** — "you will pay R41 300 more in total because
the term extends from 31 months to 60 months". A comparison that shows the
instalment reduction prominently and the total cost increase in a footnote is the
thing this requirement exists to prevent. Where product 40 was selected, the
security warning and its acknowledgement accompany the comparison.

**The execution package**, per settled account: `account_ref`, provider,
`settlement_amount` with its component breakdown, `quotation_reference`,
`quotation_expiry_date`, the assumed settlement date, the payment instruction
(beneficiary, reference, method), any security release or transfer action with
its expected duration, and any required account closure or limit reduction.

**The new facility**: `product_code`, `offered_amount`, `term_months`,
`nominal_annual_rate` with its unadjusted value and the overlays applied,
`instalment`, `total_cost_of_credit`, `effective_annual_rate`, the fee breakdown,
security requirements, and conditions precedent with their owners.

**The alternatives**: scenarios ranked 2 and 3 in full, with their own
comparisons, so the consultant can have the conversation rather than read out a
verdict.

**The rejections**: every scenario evaluated but not viable, with its
`rejection_reason_code` set and the actual-versus-threshold values. This is bulky
and it is required. "Why did you not settle my furniture account" has an answer,
and the answer is in this list.

**The expiry problem.** Every quotation has an expiry. The offer's validity is
`min(earliest quotation expiry, the Bank's offer validity of 21 days, any
valuation validity)`. An acceptance after that date is not executable on the
quoted figures, and the flow must:

- state `offer_valid_until` prominently on the output;
- on acceptance after expiry, **re-derive** — new quotations, new settlement
  amounts, a re-run search — rather than adjusting the old figures;
- link the re-derivation to the original assessment, and produce the difference
  between them attributably: which accounts' amounts moved, whether the rate card
  version changed, whether an overlay came into or went out of force, and whether
  the winning scenario changed;
- treat a changed winner as a disclosure event, because the client agreed to
  something that is no longer on offer.

On a product 40 scenario, where registration takes 6 to 10 weeks, re-derivation
is not an edge case; it is the normal path, and it must be designed for rather
than handled.

### 5.11 Batch identification

**Preconditions.** Monthly. 800 000 clients with 2 or more credit accounts and a
servicing relationship.

The same flow, with a smaller budget: at most 40 scenario evaluations per client
and 60 ms, over a reduced candidate set — products 11 and 20 only, since 30 and
40 need collateral data that cannot be assembled at this scale. The objective is
fixed per run, not per channel.

Output is a ranked list for campaign use: clients whose best available scenario
clears a materiality floor (instalment relief ≥ R400 per month **and**
`total_cost_delta` within the anti-harm threshold), with the indicative outcome.
Subject to `core.consent` suppressions.

**The requirement that matters here**: a client identified in the batch and then
assessed in a branch two days later must get a consistent story. The batch
outcome is indicative, the interactive outcome is binding, and the difference —
larger budget, more products, live quotations, collateral data — must be
explainable to the client standing at the desk holding a letter. A batch that
promises R1 200 of relief where the branch delivers R600 is a complaint
generator, so the batch materiality floor must be set with the interactive
distribution in mind, and the divergence must be measured monthly.

---

## 6. Parameters and tables

### 6.1 Tables

| Table | Dimensions | Cells | Owner | Cadence | Source |
|---|---|---|---|---|---|
| **Rate card, Flex Loan Consolidation (11)** | 72 amount bands × 73 terms × 12 grades | **63 072** | Treasury | Monthly, patched mid-month | Spreadsheet |
| Consolidation margin adjustment | 4 external-proportion bands × 12 grades | 48 | Treasury | Monthly | Spreadsheet |
| **Rate card, Drive Finance (30)** | 60 amount bands × 61 terms × 12 grades × 5 LTV bands | **219 600** | Treasury | Monthly | Library §8 |
| Refinance vehicle age adjustment | 8 age bands × 5 LTV bands | 40 | Credit Risk Policy | Semi-annual | Internal |
| **Rate card, balance transfer promotional (20)** | 12 grades × 8 limit bands × 5 promotional durations | **480** | Treasury | Monthly | Spreadsheet |
| Rate card, revolving reversion (20) | 12 grades × 20 limit bands | 240 | Treasury | Monthly | Library §8 |
| **Rate card, further advance margins (40)** | 12 grades × 8 LTV bands × 6 term bands × 2 occupancy types | **1 152** | Treasury | Monthly | Spreadsheet, in basis points over reference |
| Settleability rules | ~200 providers × 6 attributes | 1 200 | Credit Risk Policy | Monthly, ad hoc on provider change | Internal + Settlements |
| Provider payment instructions | ~200 providers × 7 attributes | 1 400 | Settlements | Ad hoc | Operations |
| Early settlement charge rules | 6 agreement classes × 5 notice bands × 4 product classes | 120 | Compliance | On legislative change | Statute |
| Non-consolidatable account types | 45 account types × 4 products | 180 | Credit Risk Policy | Quarterly | Internal |
| **Vehicle valuation guide** | ~60 000 make/model/year/derivative rows × 6 values | **360 000** | Vendor, loaded by Treasury | Monthly | Trade guide |
| Mileage adjustment | 14 kilometre bands × 8 age bands | 112 | Credit Risk Policy | Annual | Internal |
| Vehicle condition haircuts | 5 conditions × 8 age bands | 40 | Credit Risk Policy | Annual | Internal |
| Conveyancing and registration tariff | 24 amount bands × 3 components | 72 | Compliance | Annual | Published tariff |
| Product policy limits | 4 products × ~28 limits | 112 | Product teams (×4) | Weekly | Internal |
| Policy interventions | 14 interventions × up to 6 scope dimensions | ~60 | Credit Risk Policy | Quarterly | Internal |
| Objective definitions and weights | 5 objectives × 6 channels × 4 modes | 120 | Credit Committee | Quarterly, sometimes monthly | Internal |
| Ordering rule set | 8 rules × 4 attributes | 32 | Credit Risk Policy | Semi-annual | Internal |
| Rejection reason registry | ~60 codes × 9 attributes | 540 | Compliance | Monthly | Internal |
| Concession catalogue and authority bands | 8 concessions × 4 levels × 5 attributes | 160 | Credit Committee | Annual | Internal |
| **Adjustment register, this flow's scope** | 15–40 live overlays × 11 attributes | ~350 | Credit Risk Policy | Ad hoc, sometimes weekly | Library §6.22 |

Both library properties apply to every row: cell-level attribution, and a
reviewable diff on every new version. A fifth requirement appears here that the
library does not state — **cross-table consistency**. The vehicle valuation guide
and the Drive Finance rate card are refreshed by different people on different
days, and an assessment that reads September's valuation against August's rate
card is not wrong so much as unattributable. The version set resolved for an
assessment must be recorded as a set, and incompatible combinations must be
detectable.

### 6.2 Parameters, by owner

| Class | Parameters | Owner | Cadence |
|---|---|---|---|
| Library-global | Reference rate, statutory fee caps, tax tables, expense norms | Credit Systems / Compliance | As gazetted |
| Library-policy | Affordability buffer, obligation treatment matrix | Credit Risk Policy | Quarterly |
| **Search** | Evaluation budget (400 interactive / 40 batch), time budget (900 ms / 60 ms), ordering rules in force and their sequence, candidate term set per product | Credit Risk Policy | Quarterly, faster under load |
| **Interventions** | All fourteen of §5.7, within declared ranges, scoped per product, channel, grade and household size | Credit Risk Policy | Quarterly |
| **Objective** | `objective_id` per channel, segment and mode; blend weights; the 2% indifference band; the shadow objective | Credit Committee | Quarterly, sometimes monthly |
| **Settlement mechanics** | Settlement buffer (1.5%, R2 500 cap), assumed settlement date offsets per provider, revolving paydown horizon (36 months) | Credit Risk Policy / Settlements | Quarterly |
| **Stress** | Income haircut (10%), expense uplift (8%), rate stress (200 bp) | Credit Risk Policy | Semi-annual |
| **Product-local** (×4 owners) | Minimum and maximum amounts, term bounds, LTV caps, promotional durations, balloon caps, vehicle age and mileage gates, the 84-month sub-term on product 40, transfer fee | Product teams | Weekly |
| **Compliance** | Security warning wording and version, prescribed comparison content, reason and rejection registries | Regulatory Compliance | On change |
| **Batch** | Materiality floor (R400 relief), product subset, run objective | Credit Risk Policy | Monthly |

The requirement that makes this awkward: **the same pricing capability runs four
times in one assessment with four products' settings**, and a product team must
be able to change its own maximum amount on a Wednesday without touching the
other three or waiting for Credit Risk Policy's quarterly cycle. Library §7.2
asserts this is possible; this project is where the assertion is tested.

### 6.3 Overlays in this flow

Overlays are not a separate parameter class; they sit over the artefacts above,
under a different approval path, with their own expiry. The ones this flow must
support, because Credit Risk Policy has asked for each of them at some point:

| Overlay kind | Target here | Effect |
|---|---|---|
| Rate add-on | Product 11 rate card, branch channel | +75 bp for a quarter without reissuing 63 072 cells |
| Cap adjustment | Product 30 LTV cap | 100% → 90% for vehicles over 6 years, pending a loss review |
| Odds multiplier | Probability of default feeding `risk_grade` on product 40 | PD × 1.25, which moves the grade, which moves the margin, which moves which scenarios pass CON-INT-06 |
| Cut-off shift | CON-INT-09 debt service ratio ceiling | 45% → 40% for grades 8–12 |
| Buffer adjustment | The affordability buffer inside `core.affordability` | +3 percentage points, which changes how many scenarios survive at all |
| Objective re-weight | The blend weights of §5.8 | Instalment relief 0.4 → 0.6 for a quarter |
| Threshold adjustment | CON-INT-04 anti-harm ceiling | 15% → 12% on product 11 only |

Three consequences follow, and all three are requirements:

1. **Four stacks in one assessment.** Overlays are scoped by product, so a single
   scenario set is priced under as many as four different overlay stacks — a rate
   add-on on product 11, a tightened LTV cap on product 30, a PD multiplier on
   product 40, nothing at all on product 20. Composition order is per stack and
   is part of each overlay set's definition, not emergent from evaluation order.
2. **An overlay can change the winner with no logic change.** Re-weighting the
   objective, or tightening the anti-harm ceiling, changes which scenario is
   selected while every rule, table and line of logic stays identical. This is
   the mechanism working as intended, and it is also the reason a replay must pin
   the stack: the same client, the same accounts, the same code and a different
   overlay stack must reproduce a *different* winner, and that must read as an
   explanation rather than a discrepancy.
3. **The unadjusted answer must remain available.** Every scenario carries its
   unadjusted rate, cap and threshold alongside the adjusted ones, and the whole
   assessment must be runnable with the stack disabled — which is how Credit Risk
   Policy measures what an overlay is actually doing to approval rates and to the
   mix of winning products.

---

## 7. Outputs

**To the caller:**

| Output | Notes |
|---|---|
| `outcome_code` | Approve / approve with conditions / refer / decline |
| `primary_reason_code`, `decline_reason_codes` | Library taxonomy, client-facing |
| Chosen scenario | Full pricing, settlement set, comparison, conditions precedent |
| Runners-up (2) | Same detail, materially distinct |
| Before-and-after comparison | §5.10, prescribed content |
| Execution package | Per-account settlement instructions, expiries, security actions, closure conditions |
| `offer_valid_until` | The binding minimum of quotation, valuation and offer validity |
| Rejected scenarios | With `rejection_reason_code` sets and actual-versus-threshold values |
| Search evidence | Budget, consumption, termination cause, ordering rules in force, objective in force, overlay stack in force |
| Referral package | Where an authority level is required (restructure) or a conflict must be resolved |

**Persisted**, for project 09 to assemble into the decision record: everything
above, plus the full inventory as classified, every settlement amount with its
components, the invariant income and expense determination, every table and rate
card version resolved, every overlay with its identity, scope, order, effect and
approval reference, the unadjusted values beside the adjusted ones, both
affordability verdicts in the restructure variant, and the shadow objective
result.

The record is large: 400 scenarios × roughly 60 recorded values is 24 000 values
per assessment, at 6 000 assessments a day. Whether every evaluated scenario is
persisted in full, or a viable subset in full and the remainder in summary, is an
open question in §13 — but "we did not keep it" is not available as an answer for
the rejected scenarios, because those are the ones people ask about.

---

## 8. Non-functional requirements

| Requirement | Value |
|---|---|
| Interactive latency | 2.5 s p95, 4.0 s p99, end to end, branch and contact centre, including up to 400 scenario evaluations |
| Search budget | 900 ms of that, with a deterministic candidate-count bound as well as a time bound |
| App channel latency | 3.5 s p95 — a slower budget, a larger evaluation allowance where quotations are already held |
| Throughput | 6 000 assessments/day, 900/hour peak, 2.4 M scenario evaluations/day |
| Batch | 800 000 clients monthly within a 6-hour window, ≤40 scenarios each — 32 M scenario evaluations per run |
| Restructure | 1 400/day, 4 000/day after a rate increase |
| Determinism | Identical inputs, versions, `decision_date` and overlay stack ⇒ identical scenario set, identical order, identical winner, identical pricing to the cent |
| Reproducibility of truncation | A budget-truncated search must truncate identically on replay. A time-based cut-off alone cannot satisfy this. |
| Degradation | Where quotations cannot be retrieved, the assessment proceeds on estimates with the outcome marked conditional. Where the vehicle guide is unavailable, product 30 is withdrawn from routing and the withdrawal is recorded — the assessment does not fail. |
| Cold start | No per-request compilation; the first assessment after a rate card refresh is not materially slower than the thousandth |
| Table refresh | Rate cards, settleability rules, valuation guides and overlays all refresh without a deployment |
| Availability | 99.5% in branch hours; the batch has a 24-hour recovery window |

---

## 9. Audit, evidence and explainability

Five demands, each of which has been made of a real consolidation book.

**1. The ombud complaint, eighteen months later.** "My client consolidated and is
worse off." Answerable requires, for that `application_id`: the comparison
actually presented and its wording version; the alternatives considered and
rejected, with reasons; the anti-harm rule's evaluation with its actual and
threshold values; **the objective in force on that day, including any overlay
re-weighting it**; the shadow client-outcome result; the security warning and its
acknowledgement where product 40 was selected; and the settlement quotations used
with their expiries. Where an overlay caused the threshold that admitted the
scenario to be looser than the base — or the objective to favour instalment
relief over total cost — that overlay's identity, approval reference and
effective window are part of the answer, not context for it.

**2. The internal audit sample of restructures.** Forty files. Each must show the
concession granted, its NPV cost, the authority level required, the authority
that actually approved it, whether it was a repeat concession, both affordability
verdicts, and the distressed-restructure classification with its basis. A file
where the approving authority is recorded as the system is a finding.

**3. The reckless lending review.** The regulator's question is whether
affordability was assessed on the client's position *after* the consolidation, on
the full new commitment, with expenses at or above the norm floor. The evidence
must show the post-consolidation obligation set, not the pre-consolidation one,
and must show which accounts were assumed settled — because an affordability
assessment that removes an obligation the Bank did not in fact settle is the
precise failure the prohibition exists to catch.

**4. The Credit Committee overlay review.** Quarterly: every overlay affecting
this flow, its age, its approval, its expiry, its scope, its composition position,
its measured effect on approval rates and on the mix of winning products, and
what unwinding it would do. An overlay past its review date must surface as an
exception. The comparison against the stack-disabled run is the evidence.

**5. The contact centre query, the same week.** "Why did you not settle my
furniture account?" The answer is one of: it was not settleable (and why), it was
never generated as a candidate (and the budget ran out at candidate 400), it was
generated and the scenario containing it was rejected by a named intervention
(with values), or it was viable but lost to the chosen scenario by a stated
margin on a stated component. All four are different, all four are legitimate,
and the record must distinguish them within seconds, not on request to a data
team.

**Retention.** Seven years for the decision record; the rejected-scenario detail
for three years, at which point it may be summarised to counts by rejection
reason — a retention rule that must be a parameter, because Compliance will
change it.

---

## 10. Acceptance criteria

1. An assessment of a client with 18 settleable accounts completes within 2.5 s
   p95, evaluating no more than the configured budget, and produces the same
   winner on ten consecutive runs.
2. The same assessment replayed six months later, with the artefacts and overlay
   stack in force at its `decision_date`, reproduces the winner, the runners-up,
   the rejection set and the pricing to the cent.
3. The same assessment replayed under a *different* overlay stack reproduces a
   different winner where the overlays warrant it, and the output explains the
   difference by naming the overlays responsible.
4. Every scenario evaluated carries a viability verdict; every non-viable
   scenario carries at least one `rejection_reason_code` with actual and
   threshold values; no scenario is silently dropped.
5. A settlement set that can be carried by three products produces three priced
   scenarios per term, each priced from its own rate card with its own fee
   structure and its own policy set.
6. Credit Risk Policy changes the anti-harm threshold on product 11 only, by
   overlay, and the change takes effect on the stated date without a deployment,
   without touching the other three products, and with the base value still
   visible in the record.
7. Credit Committee changes the objective from OBJ-01 to a 0.6/0.4 blend for the
   branch channel only, without a deployment, and the selection record names the
   objective and the weights used.
8. The income and expense determination is provably identical across all
   evaluated scenarios in an assessment.
9. A product team raises product 20's maximum limit without any change to
   products 11, 30 or 40 and without a Credit Risk Policy cycle.
10. Running the whole assessment with the overlay stack disabled produces the
    unadjusted result alongside the adjusted one, from the same implementation.
11. A client under debt review is routed to the restructure path and never
    receives a credit decline.
12. A product 40 scenario cannot be selected without a recorded warning
    acknowledgement, demonstrated by a test that attempts it.
13. The batch run completes 800 000 clients in under 6 hours and its outcomes for
    a sampled 1 000 clients reconcile, within a declared tolerance, with
    interactive assessments of the same clients on the same date.
14. A Credit Risk Policy analyst reads the generated description of the ordering
    rules, the interventions and the objective, and confirms it matches policy,
    without reading code.
15. Adding a fifth product to the routing set does not require changes to the
    search, the objective, the interventions or the other four products.

---

## 11. Change scenarios

1. **A fifth product joins the search.** The Access Facility (21) becomes a
   consolidation destination, with its own limit-based logic and its own rate
   card.
2. **Credit Risk Policy tightens the anti-harm ceiling** from 15% to 12%, by
   overlay, on product 11 only, in the branch channel only, for one quarter, with
   a mandatory expiry — and needs the approval-rate impact modelled before it
   goes live and measured afterwards.
3. **An overlay is found still in force three years on.** A tightened LTV cap
   applied during one bad quarter has never been unwound. Unwinding it changes
   which scenarios win for tens of thousands of clients, and someone must explain
   why last year's declines would now pass. The flow must be able to show the
   stack-disabled comparison and the population affected.
4. **Two overlays collide**: a channel-scoped rate add-on and a segment-scoped PD
   multiplier both apply to product 11, and the Credit Committee wants the net
   effect and the order in which they applied.
5. **The evaluation budget changes** from 400 to 700 interactive and from 40 to
   25 in batch, and the latency and outcome impact must be measurable before the
   change ships.
6. **A ninth ordering rule is added** — prefer settling accounts whose provider
   has the shortest quotation turnaround — authored by Credit Risk Policy, live
   the following Monday.
7. **A provider stops issuing settlement quotations.** 3 400 accounts change
   settleability overnight; assessments in flight must not silently include them.
8. **The regulator prescribes the comparison**, adding the effective annual rate
   and a five-year total cost projection, applicable to new assessments only,
   with the old wording preserved for everything before.
9. **The balance transfer promotional period moves** from 12 to 18 months and the
   reversion card is re-keyed by a sixth limit band.
10. **Drive Finance adds a used-vehicle rate card** with an extra key dimension,
    without disturbing the new-vehicle card or the other three products.
11. **The further advance warning wording changes.** Decisions before the change
    keep the old wording and version in evidence forever.
12. **The restructure variant must run nightly** over the whole arrears book —
    240 000 clients — rather than on demand, to pre-compute offers for outbound
    collections.
13. **The Bank acquires a book.** 1.1 M accounts arrive with provider codes absent
    from the settleability table, and must classify as unknown rather than as
    settleable.
14. **Credit Committee wants blended objectives per segment as well as per
    channel**, where today the blend varies only by channel and mode.
15. **Contact centre asks for the rejection reasons in client-facing language**,
    which means ~60 internal rejection codes need Compliance-owned wording in
    three languages, versioned like the decline taxonomy.

---

## 12. Out of scope

- Retrieving settlement quotations from providers. This flow consumes them and
  states what it needs; the integration is somebody else's.
- Executing settlements, disbursement, payment instruction transmission and
  reconciliation.
- Bond registration, conveyancing, vehicle registration and de-registration.
- Negotiation with debt counsellors and the court-ordered restructure process
  itself. The flow produces what that process needs.
- Contract generation and the customer journey.
- Collections treatment and arrangement management — project 08.
- Portfolio-level budget constraints on consolidation volumes — project 07 owns
  that shape of problem.
- Replay, diff and certification machinery — project 09. This flow produces
  evidence; it does not store or replay it.
- Model development. The consolidation scorecard is consumed, not fitted.

---

## 13. Questions the implementation must answer

1. **What is a bounded search, structurally?** It generates candidates from
   business-authored orderings, evaluates them under a budget, records what it
   did, and stops. It is not a scorecard, a decision table or a decision tree. Is
   it a core component kind, a composition of existing kinds, or an escape hatch
   — and if it is an escape hatch, how many projects need it before it stops
   being one?
2. **How is the candidate ordering expressed** so that Credit Risk Policy can
   author, read, reorder and retire the eight rules without an engineer, while
   the ordering stays total and the search stays deterministic?
3. **How is a budget expressed and enforced** such that truncation is
   reproducible? A wall-clock cut-off is not, and a candidate-count bound alone
   does not protect the latency envelope. Are both required, and which binds?
4. **What is product routing, structurally?** One scenario reaches four
   heterogeneous product subflows with different inputs, different tables,
   different policy sets, different owners and different outputs, and several can
   answer the same question. Is that one construct or four, and where does the
   shared part end?
5. **How do four product subflows live in one flow** without becoming four copies
   of it, given that four separate teams edit them on four cadences?
6. **What does a rate card return?** Product 11 returns an absolute rate, product
   20 returns a promotional rate and a reversion rate and a duration, product 40
   returns a margin over a reference. Is `core.rate_card` one capability with a
   variable shape of answer, or is the assumption that a rate lookup returns a
   scalar wrong?
7. **How is an objective function configuration?** It is a weighted combination
   over five measures, tunable per channel without a release, overlayable, and it
   changes which scenario wins. Is that a parameter, a table, or logic?
8. **Where does an overlay sit relative to a flow?** It changes values like a
   parameter, composes in a declared order like logic, and is separately approved
   and separately expiring like neither. When four different overlay stacks apply
   to four product subflows inside one assessment, what is the thing that holds
   them, and how does the flow run with the stack disabled without becoming a
   second implementation of itself?
9. **How is affordability recomputed 400 times** without re-deriving income 400
   times, without forking `core.affordability`, and without the invariant part
   being something the implementer has to remember to hold still?
10. **What does `core.obligations` look like** when its input is a ragged
    collection minus a subset, 400 times over, and both its scalar aggregate and
    its per-element annotation are needed?
11. **How are rejection reasons recorded for things that did not happen?** A
    rejected scenario is not a declined client, its reasons are a different
    taxonomy, and there are up to 400 of them per assessment. Where do they live,
    for how long, and how does a contact centre agent reach one in seconds?
12. **How does the circular solve compose with the search?** Each of 400
    scenarios contains a converging solve of up to six iterations. Does the budget
    count scenarios or arithmetic, and what happens to determinism when the solve
    does not converge?
13. **How is quotation expiry represented** such that an offer knows when it
    stops being true, a re-derivation is a first-class relationship rather than a
    new assessment, and the difference between the two is attributable?
14. **What is the unit of reuse between this flow and project 03?** Granting and
    pricing per product is project 03's logic applied to inputs it never
    contemplated, four times, inside a search. If it has to be forked to serve
    this, where should the seam have been?
15. **Which of settleability classification, settlement amount derivation and the
    before-and-after comparison belong in the library?** Projects 07 and 08 want
    the first two. They are written here first, which is how the previous
    generation ended up with the same logic in three places.
16. **What is persisted, of 24 000 values per assessment, 6 000 times a day?**
    And how is the decision made, given that the values most likely to be asked
    about are the ones describing outcomes that never happened?
