# 02 — Affordability and obligations assessment

> Fictional. The Bank, its products, thresholds, table dimensions and volumetrics
> are invented for this repository. Regulatory mechanisms referred to are the
> published public ones; every number attached to them is illustrative.

The mechanism here is South Africa's National Credit Act affordability regime —
the published Regulation 23A: ascertain gross income from prescribed evidence,
deduct statutory deductions, apply a minimum expense norms table broken down by
monthly gross income, take into account all debt repayment obligations reflected
on the consumer's credit bureau profile, and arrive at a discretionary income
that caps what the new instalment may be. That mechanism is real and public.
**Every band edge, fixed component, marginal percentage, haircut, buffer and
tolerance below is invented.** The gazetted norms table is keyed on gross monthly
income alone and has a handful of bands; the twelve-band, dependant-varying form
used here is the Bank's fictional rendering of the same arithmetic shape.

---

## 1. What this is

Before the Bank may advance credit it must satisfy itself, on evidence, that the
applicant can afford the instalment — and show its working years later to a
regulator, an ombud or a court. Wrong in the optimistic direction is reckless
lending: the agreement set aside, the debt written off, the Bank penalised. Wrong
in the pessimistic direction declines people who could have paid.

This project is that calculation. It takes an applicant — or two, on a joint
application — establishes what they earn and how well that is evidenced, deducts
what the state and the courts take first, establishes what it costs them to live
using the higher of what they declare and what the statutory norm requires,
converts a ragged list of up to eighty existing credit accounts into a monthly
obligation figure, and arrives at a discretionary income, a maximum affordable
instalment, and a verdict.

It is the most-reused component in the set, and projects 03, 05, 06, 07 and 08
consume it **differently**: a pass or fail against a known instalment; the
largest instalment the applicant could carry; four hundred runs inside one search
with debts hypothetically removed; fourteen million records overnight; and
somebody already in arrears whose circumstances are the opposite of an
applicant's. One regulated calculation underneath, five shapes of answer on top —
and five standing incentives to fork it. The previous generation's evidence says
where forking leads: a decline-reason taxonomy dropped entirely in a port, 546
inline literals where a parameter should have been, 79 passthrough steps written
around a missing rename (doc 01 §5). A forked affordability calculation is worse
than any of those, because four of the five forks will be wrong about tax in 2028
and nobody notices until the ombud asks.

The parts known to be hard, named up front so the implementation cannot claim
surprise:

1. **One calculation, five consumers, five evidence sets, five required outputs.**
   Nothing may be duplicated to serve them.
2. **Effective-dated statutory tables.** Tax brackets, expense norms and fee caps
   resolve by `decision_date`, never by today; a 2030 replay of a 2026 assessment
   must reproduce it to the cent.
3. **A ragged collection producing both a scalar and a per-element annotation.**
   Both are required output; neither is a debugging convenience.
4. **Called hundreds of times inside one search.** Per-call cost is a design
   constraint, not an optimisation for later.
5. **Regulated evidence.** Which income tier, which haircut, which expense basis,
   which norm version, which accounts and how each was treated — all of it must
   survive four years and be legible to a non-technical adjudicator.
6. **Policy overlays on top of a statutory calculation.** The Bank tightens by
   overlay rather than by rebuilding, so the regulator must always see the
   statutory answer underneath the Bank's chosen conservatism. Overlays may only
   move the answer in the conservative direction, and that asymmetry must be
   enforced rather than trusted.

---

## 2. Why it is in this set

| Question | How this project stresses it |
|---|---|
| **Q1 reusable components** ● | This is the reuse test case. Five consumers, one regulated calculation, and a strong incentive on each consumer to fork. If reuse works anywhere it must work here; if it fails here the library is decoration. |
| **Q2 tables** ● | Six substantial tables, three statutory and effective-dated, one a 45-row treatment matrix edited quarterly by non-engineers whose cells select *behaviour* rather than values. |
| **Q4 custom modules across flows** ● | One implementation runs in four assessment modes with different evidence and different expense treatment. Modes must not become copies. |
| **Q7 audit** ● | A legal obligation with a named adjudicator, a four-year horizon and a prescribed arithmetic chain. Separating the statutory answer from the Bank's overlays is part of it, not decoration. |
| **Q3 parameters** ● | Statutory values the Bank may not change, policy values changed quarterly, consumer-local values five projects set differently, and an overlay stack with its own approval path — in one calculation, with boundaries that must hold. |
| **Q5 core component set** ○ | Band lookup with a fixed-plus-marginal formula, per-element treatment selection over a ragged list, highest-of-several-bases resolution, ordered overlay stack. None is a scorecard, a decision table or a decision tree. |
| **Q6 codebase organisation** ○ | Owned centrally, consumed by five product teams who cannot change it but must parameterise it. |

---

## 3. Actors

| Actor | Responsibility |
|---|---|
| **Credit Systems** | Owns the implementation. The only party that may change the arithmetic. |
| **Regulatory Compliance** | Owns the statutory norms table, the statutory evidence requirements and the minimum evidence tiers. Signs the calculation off against the regulation. Does not write code and must not have to read any. |
| **Credit Risk Policy** | Owns the internal norms, the obligation treatment matrix, the income haircut matrix, the buffer grid and the verdict thresholds. Quarterly. Not engineers. |
| **Credit Committee** | Approves the adjustment sets — overlays that tighten the answer for a named scope over a declared period — and reviews each at its expiry date. |
| **Tax and Payroll Reference** | Supplies effective-dated tax brackets, rebates and the unemployment insurance ceiling as gazetted. Annual, occasionally mid-year. |
| **Product teams** (×5) | Consume the assessment. Set consumer-local values only. May not alter evidence rules, norms, treatments or overlays. |
| **Collections operations** | Consumes it in arrangement mode (project 08), where the client is under stress and the question is sustainability, not appetite. |
| **Internal Audit** | Samples assessments and re-derives them. |
| **The National Credit Regulator, the Credit Ombud, the Tribunal, a court** | May require the full assessment for one application years after the fact. Not hypothetical: the regulation provides the complaint path and the turnaround times. |
| **The applicant** | Declares income and expenses, and may complete the exceptional-basis questionnaire where declared expenses fall below the norm. |

---

## 4. Inputs

### 4.1 Sources and shapes

| Input | Source | Shape / cardinality | Nullability | Freshness |
|---|---|---|---|---|
| Application header | Origination | scalar | `decision_date`, `product_code`, `channel_code`, `assessment_mode_code` never null | at request |
| Applicants | Origination | 1..2 records | second present only when `is_joint_application` | at request |
| Declared income | Applicant | 0..6 sources per applicant, ragged | frequently null | at request |
| Payslip set | Document capture | 0..3 per source | null for non-salaried | ≤ 90 days |
| Employer confirmation | Employer register / HR letter | 0..1 per source | usually null | ≤ 60 days |
| Internal transaction view | Bank core systems | 0..6 monthly aggregates | null if not a client of the Bank | ≤ 2 days |
| External statement analysis | Statement aggregator | 0..12 monthly aggregates + `statement_confidence` | null when not obtained | ≤ 35 days |
| Bureau income estimate | Credit bureau | 0..1 | often null | with bureau pull |
| Variable pay history | Payslip or statement derivation | 0..12 monthly values per component, ragged | gaps are meaningful | ≤ 12 months |
| Declared living expenses | Applicant | 8..14 categories | zero ≠ absent | at request |
| Statement-derived expenses | Statement aggregator | same categories, may be partial | per-category nullable | ≤ 35 days |
| Bureau account list | `core.bureau` | 0..80, ragged | many per-account fields nullable | `bureau_as_of_date` |
| Internal account list | Bank core systems | 0..25, ragged | balances never null | ≤ 1 day |
| Court-ordered deductions | Payslip, public records, declaration | 0..6, ragged | frequently null | mixed |
| Settlement quotes | Project 06 / settlement desk | 0..80 | null outside consolidation | ≤ 7 days |
| `risk_grade` | `core.risk_grade` | scalar | null in some modes | at request |
| Adjustment set in force | `core.adjustments` | ordered overlay list, 0..12 | empty is normal | resolved at `decision_date` |
| Proposed instalment | Caller | 0..1 | null when the caller wants the maximum instead | at request |

### 4.2 Volumes and regulatory freshness

| Context | Volume | Shape |
|---|---|---|
| New application, real time | ~38 000/day, peak 140/s | single record |
| Consolidation search (project 06) | up to 400 assessments per application | single record, repeated |
| Limit management batch (project 07) | 14 M records monthly | batch |
| Collections arrangement (project 08) | ~26 000/day | single record |
| Portfolio simulation (project 09) | full book replay on demand | batch |

The debt repayment history underpinning the obligations figure must have been
obtained within a short window before approval — illustratively seven business
days for non-mortgage credit and fourteen for mortgage-secured products. If
`bureau_as_of_date` falls outside that window at `decision_date`,
`bureau_is_stale` is true and the assessment cannot conclude on that bureau view.
That produces `indeterminate`, not `fail`.

### 4.3 Names this project declares locally

The library does not publish these. Each is declared once and used unchanged;
each is a candidate for promotion into `core.*`, and needing nine is itself a
finding about the library's vocabulary. `adjustment_set_id` and
`adjustments_applied` are the library's names and are used as published.

| Name | Type | Meaning |
|---|---|---|
| `assessment_mode_code` | int8 | New application, limit increase, arrangement, scenario. |
| `court_ordered_deductions` | float64 | Emoluments attachment, maintenance, administration order. Deducted before discretionary income but outside `statutory_deductions` as the library defines it. |
| `income_evidence_month_count` | int8 | Qualifying months actually used for averaging, per source. |
| `statement_confidence` | float64 | 0..1, from the statement aggregator. |
| `obligation_dedup_group_id` | int32 | Links an internal account to its bureau reflection. |
| `discretionary_income_after` | float64 | Discretionary income once the proposed instalment is committed. |
| `evidence_sufficiency_code` | int8 | Why an assessment is `indeterminate`. Zero when it is not. |
| `max_affordable_instalment_unadjusted` | float64 | Capacity before any overlay (§5.6.2). |
| `living_expenses_unadjusted` | float64 | The applied expense figure before any stringency overlay. |

---

## 5. The flow

Seven stages, in order. Every stage records what it used, what it chose and why.

### 5.1 Stage 1 — Applicant and household framing

**Determines.** Who is assessed, as what, and whose money and whose debts are in
scope.

- A single application assesses one applicant. A joint application assesses two
  **as one household**: incomes are summed after their own individual haircuts,
  expenses are consolidated rather than added, and the expense norm is looked up
  on the **combined** gross monthly income. That is how the norms table is
  prescribed to apply to joint consumers; applying it per applicant understates
  the floor by roughly the fixed component.
- `dependants_count` is a household figure, declared once. Where two applicants
  declare different counts the **higher** is used and the discrepancy recorded. A
  dependant who is also an applicant is not a dependant — checked by identity,
  not by name.
- Household expense consolidation: per category, the **higher** of the two
  declarations for shared categories (accommodation, water and electricity, food,
  insurance) and the **sum** for personal ones (transport, medical,
  communication, education, maintenance paid). The classification is a parameter,
  because Credit Risk Policy will argue about food.
- Obligations: every account on which either applicant is a principal debtor
  counts **once**. A joint account reported on both bureau profiles and counted
  twice is the commonest joint-application defect, inflating obligations by the
  household's largest debt. An account where one applicant is surety and the other
  is principal debtor collapses to the principal treatment.
- Where one applicant's income cannot be established to the product's minimum
  evidence tier, the household is **not** assessed on the other's income alone.
  The verdict is `indeterminate`, naming the applicant. Silently dropping an
  applicant is the worst available bug here, because it produces a plausible
  number.

**Emits.** Household composition, per-applicant scope of income and obligations,
the shared/personal classification applied, and any declaration discrepancies.

### 5.2 Stage 2 — Income determination

**Determines.** `gross_monthly_income`, `income_source_code`,
`income_verification_tier`, `income_haircut_applied`, `income_variability_ratio`.

#### 5.2.1 The evidence waterfall

Six tiers, strongest first. Per income source independently, the strongest tier
for which evidence exists is used; weaker contradictory evidence is recorded but
does not change the tier.

| Tier | Evidence | What establishes it |
|---|---|---|
| 1 | Employer-confirmed | Written confirmation from an employer on the verified register, stating amount and frequency. |
| 2 | Payslip set | The latest three consecutive payslips. Two where `months_employed` < 3, with the employment letter. |
| 3 | Internal salary-deposit observation | Six months of credits into an account at the Bank, classified as salary, from a consistent originator. |
| 4 | External statement inflow analysis | Up to twelve months of aggregated inflows, with `statement_confidence`. |
| 5 | Bureau-estimated income | A modelled estimate from the bureau. |
| 6 | Client-declared | Unverified declaration. |

The statutory floor is that income is **validated**, not merely declared: three
payslips, or bank statements showing three salary deposits; for the non-salaried
three months of documented proof of income or three months of statements; for the
self-employed and informally employed three months of statements or the latest
financial statements. Tier 6 alone therefore cannot support a grant on most
products. The weakest permitted tier is a product parameter: 4 for products 10,
11 and 20; 5 for 21; 3 for 30 and for the proprietor's personal income on 50/51;
2 for 40.

#### 5.2.2 Haircuts

Every tier carries a haircut varying by `employment_type_code`, as a percentage
reduction of that source's figure. `X` means the combination is not permitted and
forces a fall-through to a weaker tier.

| Tier | Permanent | Contract | Self-employed | Pensioner | Social grant | Informal |
|---|---|---|---|---|---|---|
| 1 employer-confirmed | 0% | 2% | X | X | X | X |
| 2 payslip | 0% | 3% | X | 0% | X | X |
| 3 internal deposits | 2% | 5% | 12% | 2% | 0% | 15% |
| 4 statement inflow | 5% | 8% | 18% | 5% | 0% | 22% |
| 5 bureau estimate | 15% | 20% | 30% | 15% | X | X |
| 6 declared | 25% | 30% | 40% | 20% | 0% | 35% |

Three modifiers apply on top, additively, capped at 55% in total: for tier 4 add
`(1 − statement_confidence) × 20` percentage points, with the tier unusable below
a confidence of 0.55; add 5 points where `months_employed` < 6 on a contract and
3 points where < 3 on permanent employment; add 5 points where
`income_variability_ratio` > 0.35 and 10 points where > 0.60. Social grant income
takes a zero haircut at every permitted tier because the amount is a published
statutory figure — and is **capped** at that figure, with a declared amount above
it reduced and flagged.

#### 5.2.3 Variable pay

Base pay comes from the latest period. Commission, overtime, shift allowance and
bonus are averaged over a window of up to twelve months, and which months count
is where this goes wrong.

- A month in which the component was payable and was zero **counts**, as zero. A
  month for which no record exists — not yet employed, statement gap, payslip not
  supplied — **does not count**, and shrinks the denominator.
- A month exceeding three times the window median is excluded as a once-off
  (back-pay, a settlement), recorded as excluded, denominator reduced.
- The statutory floor is an average over not less than three pay periods where
  income shows material variance. The Bank's floor is higher for
  commission-dependent employment.
- With fewer than the required months the component is included at a reduced rate,
  not assumed away: 0% for 0–2 qualifying months, 50% for 3–5, 75% for 6–8, 90%
  for 9–11, 100% for 12.
- An annual or thirteenth-cheque bonus is divided by twelve and included at 60%,
  and only where twelve months of history exist.
- Where `income_variability_ratio` exceeds 0.60 the variable component is capped
  at the **lowest** observed qualifying month rather than the average.

#### 5.2.4 Evidence rules by employment type

| Type | Required evidence | Particulars |
|---|---|---|
| Permanent | Three payslips or three salary deposits | The straightforward case. |
| Contract | Payslips plus the contract | Income beyond the contract end date is excluded; a contract expiring inside the proposed term is recorded and passed to the caller — an appetite question, not an affordability one. |
| Self-employed | Three months' statements or the latest financial statements | Drawings, not turnover. Where only business statements exist an owner's drawings ratio is applied and recorded. Never tier 1 or 2. |
| Pensioner | Pension advice or three deposits | No unemployment insurance deduction. Age rebates apply. A term extending beyond an age ceiling is recorded, not enforced here. |
| Social grant | Grant register or three deposits | Capped at the published figure, zero haircut. Grant types differ in amount and duration; a child-linked grant ends on a determinable date, which is recorded. |
| Informal | Three months' statements | Never above tier 3. Cash income with no banking footprint cannot be established and produces `indeterminate` for that source, not zero. |

#### 5.2.5 Combining sources

An applicant may have 0..6 sources, each established, haircut and averaged
independently, then summed.

- `gross_monthly_income` is the sum of post-haircut source figures, for the
  household on a joint application.
- `income_verification_tier` reported is the **weakest** tier among sources
  contributing more than 10% of the household total. A 4% side income at tier 6
  does not drag a salaried applicant to tier 6, and it must be visible that it
  did not. `income_source_code` records the tier of the **largest** source.
- `income_haircut_applied` is the effective blended reduction — one minus
  post-haircut over pre-haircut, to four decimal places.
- Tier 5 and 6 sources combined may not exceed a product-set proportion of the
  household total — illustratively 15% on products 10, 11, 30 and 40, 30% on 20
  and 21. The excess is discarded, not the source.

**Must be recorded.** Per source: the tier, the evidence identifier, the haircut
components, months used and months excluded with reasons, the averaging window,
the inclusion rate, and the figures before and after. The tier and the haircut
actually applied are what an adjudicator asks for first, because they are the
difference between a validated income and a declared one.

### 5.3 Stage 3 — Statutory deductions

**Determines.** `statutory_deductions`, `net_monthly_income`,
`court_ordered_deductions`.

- **Income tax.** Gross is annualised, the effective-dated bracket table applied,
  age-dependent rebates subtracted, the result divided by twelve. The rebate class
  depends on `applicant_age_years` **at `decision_date`**: primary below 65,
  primary plus secondary from 65, plus tertiary from 75. An applicant who turns 65
  in the month of assessment is a real case and the boundary must be stated, not
  discovered.
- **Unemployment insurance contribution.** A fixed percentage of gross subject to
  a monthly remuneration ceiling — illustratively 1% capped at R177.12. Not
  payable on pension or social grant income. With two employers the ceiling
  applies per employer, so the household contribution can legitimately exceed the
  single-employer maximum. A small number most naive implementations get wrong.
- **Compulsory retirement contributions.** From the payslip where present.
  Otherwise, where the employer is on the register of funds with a known mandatory
  rate, that rate is applied and the fact that it was imputed rather than observed
  is recorded.
- **Court-ordered deductions.** Emoluments attachment and garnishee orders,
  maintenance orders, administration order payments. Carried as
  `court_ordered_deductions`, deducted before discretionary income, and — the trap
  — **excluded from obligations in stage 5 wherever they match a bureau account.**
  An attachment order in respect of a credit agreement appears both as a payslip
  deduction and as a bureau account; counting both double-charges the applicant by
  the full instalment.
- **Not statutory, and not to be treated as such**: medical scheme contributions,
  union dues, group life cover, staff loan repayments, savings club deductions.
  Payslips lump these into one deductions total. They belong in living expenses or
  in obligations. Where the payslip cannot be decomposed, the whole non-statutory
  remainder goes into living expenses and the failure to decompose is recorded.

Tax tables, rebates and the insurance ceiling resolve against `decision_date`,
not today. A replay in 2030 of a 2026 assessment uses the 2026 tables, including
a mid-year change if one occurred. Selecting by today is the defect class that
produces a plausible wrong answer on every historical record at once and stays
invisible until someone reconciles.

**Emits.** `net_monthly_income` = `gross_monthly_income` − `statutory_deductions`,
with each deduction component recorded separately alongside the bracket, the
rebate class, the table version, and whether retirement was observed or imputed.

### 5.4 Stage 4 — Living expenses

**Determines.** `living_expenses`, `living_expenses_unadjusted`,
`expense_basis_code`, `norm_table_version`.

Four candidate bases are computed and **the highest binds**. That is the point of
the regulation: a client cannot be lent to on an implausibly low expense claim.

**Basis A — declared expenses.** Between 8 and 14 categories by product and
channel, matching the prescribed questionnaire: accommodation, transport, food,
education, medical, water and electricity, maintenance paid, insurance premiums,
communication, domestic help, childcare, credit life and funeral cover, savings
commitments, other. A category declared as zero and a category not asked are
different and must not be conflated — "no accommodation expense because I live
with family" is a valid declaration; a blank field is missing data. Zero across
all categories is not a low-expense applicant but a non-declaration, and it sets
an evidence flag.

**Basis B — statement-derived expenses.** Outflows categorised into the same set.
Debt service outflows are **excluded** — that is stage 5's business, and
including them double-counts. Partial coverage is normal and is recorded.

**Basis C — the statutory minimum expense norm.** The gazetted mechanism: a table
keyed on monthly gross income, with a **fixed component** per band plus a
**marginal percentage of income above the band floor**. The Bank's twelve-band
rendering at zero dependants — illustrative, and continuous at every band edge,
which the real one also is:

| Band | From | To | Fixed component | Marginal % above band floor |
|---|---|---|---|---|
| 1 | R0.00 | R800.00 | R0.00 | 100.00% |
| 2 | R800.01 | R3 000.00 | R800.00 | 6.75% |
| 3 | R3 000.01 | R6 250.00 | R948.50 | 6.75% |
| 4 | R6 250.01 | R10 000.00 | R1 167.88 | 9.00% |
| 5 | R10 000.01 | R15 000.00 | R1 505.38 | 9.00% |
| 6 | R15 000.01 | R25 000.00 | R1 955.38 | 9.00% |
| 7 | R25 000.01 | R35 000.00 | R2 855.38 | 8.20% |
| 8 | R35 000.01 | R50 000.00 | R3 675.38 | 8.20% |
| 9 | R50 000.01 | R75 000.00 | R4 905.38 | 6.75% |
| 10 | R75 000.01 | R110 000.00 | R6 592.88 | 6.75% |
| 11 | R110 000.01 | R160 000.00 | R8 955.38 | 6.75% |
| 12 | R160 000.01 | unlimited | R12 330.38 | 6.75% |

Dependant adjustment applied to the selected cell, giving the 12 × 6 × 2 shape:
fixed-component multipliers of 1.00 / 1.18 / 1.33 / 1.45 / 1.55 / 1.62 and
marginal uplifts of 0.00 / 0.40 / 0.75 / 1.05 / 1.30 / 1.50 percentage points for
0, 1, 2, 3, 4 and 5-or-more dependants. The table is **stored fully expanded** —
144 values, not a base table times a factor vector — because a gazette amendment
can change one cell without changing its neighbours and a stored formula cannot
represent that.

Edge cases, stated rather than inferred:

- **Income below the lowest band.** Band 1 takes 100% of income as the norm. An
  applicant earning R400 has a R400 floor and necessarily no discretionary income.
  Correct, and not to be special-cased into a pass.
- **Income above the highest band.** Band 12 is unbounded and its marginal rate
  continues indefinitely; at R3 000 000 a month the norm exceeds R203 000, which
  is absurd and is nonetheless what the table says. The Bank applies an absolute
  ceiling above which the norm stops growing — a policy parameter, recorded when
  it binds, because it is the Bank departing upward from the table's literal
  reading in the applicant's favour.
- **Band edges.** Bands include their upper bound; the next band's floor is one
  cent higher, so income of exactly R25 000.00 is band 6. The arithmetic is
  continuous across the edge by construction, so the floor never jumps — but the
  band **identity** changes, and the recorded identity is what an adjudicator
  reads. Joint applications select the band on combined household gross.
- **Zero or absent declared expenses.** The norm binds. Where the applicant wishes
  to claim expenses **below** the norm, the regulation permits it on an
  exceptional basis where justified, provided the prescribed questionnaire is
  completed; without one the claim is unavailable and the norm stands. With one,
  the lower figure may be used, and the questionnaire reference, the approver and
  the justification join the record. Bank policy makes this route available on
  products 20 and 21 only.

**Basis D — the Bank's internal norm.** Stricter than statutory, per product, and
frequently the binding basis: product 10 at fixed × 1.10 and marginal +0.5 pp; 11
at × 1.15 and +0.5 pp; 20 and 21 at × 1.20 and +1.0 pp; 30 at × 1.12 plus a
running-cost addendum per financed vehicle; 40 at × 1.25 and +1.5 pp plus a
rates-and-taxes addendum. Statutory norms do not apply to a juristic person, so on
products 50 and 51 the proprietor's personal norm applies where a personal surety
is assessed. Stored expanded — 12 × 6 × 2 × 8 = 1 152 values — and effective-dated
independently of the statutory table.

**Resolution.** `living_expenses_unadjusted` = max(A, B, C, D), subject to the
questionnaire exception. A **stringency overlay** may then apply (§5.6.2): an
approved, scoped, expiring adjustment that changes which basis binds — typically
forcing the internal norm to bind on a named channel, or uplifting it by a
percentage for a period. An overlay may only raise the applied figure. **It may
never reduce it below basis C, because the statutory floor is not adjustable.**
That asymmetry is a validation rule on the overlay definition, not a runtime hope.

`expense_basis_code` names which basis bound and `norm_table_version` names the
statutory version resolved at `decision_date`; the internal version is recorded
whenever the internal norm bound. Where two bases tie to the cent the statutory
basis is reported as binding, because that is the one the adjudicator will ask
about.

**Must be recorded.** All four candidate figures — not only the winner — the band
and dependant cell, both table versions, statement coverage, any questionnaire
reference, whether the absolute ceiling bound, and both `living_expenses` and
`living_expenses_unadjusted` where an overlay applied.

### 5.5 Stage 5 — Existing obligations

**Determines.** `existing_obligations`, `obligations_internal`,
`obligations_external`, `worst_arrears_months`, `accounts_in_arrears_count`,
`total_exposure`, `revolving_utilisation`, and one
`obligation_treatment_code` per account.

The statutory requirement is to take into account all monthly debt repayment
obligations in terms of credit agreements as reflected on the applicant's credit
bureau profile. "As reflected" is doing a great deal of work: bureau profiles lag,
disagree with internal records, report instalments that are stale, and omit
accounts opened last week. This stage is where a ragged list of up to eighty
heterogeneous accounts becomes one number — and where most of the per-element
evidence the ombud later asks for is produced.

#### 5.5.1 Assembling one account list from two sources

The bureau list (0..80) and the internal list (0..25) overlap, because the Bank's
own facilities are also reported to the bureau — with a different identifier, a
different account type code, and a lag of up to 60 days.

- Accounts are matched into `obligation_dedup_group_id` groups on provider
  identity, account type family, opened date within 45 days, and original advance
  within 5% or R500, whichever is larger. A group with an internal member takes
  the **internal** balance, instalment and arrears figures, because they are
  current and the bureau's are not.
- An unmatched internal account is included. An internal account the bureau does
  not yet reflect is not a discrepancy, it is a lag.
- A bureau account the Bank believes is closed is **not** dropped on the internal
  view alone. Closure requires either an internal settled status or a bureau
  closed status; one source alone downgrades it to a recorded discrepancy and the
  account stays in, at its stated instalment. Dropping obligations on thin
  evidence is the optimistic direction, which is the direction that is reckless.
- Accounts opened since `bureau_as_of_date` are invisible here by construction.
  Where bureau enquiry velocity shows credit-seeking in the intervening window —
  illustratively two or more credit enquiries in 30 days — a policy uplift is
  added to `existing_obligations` as a named line, never silently. The uplift is
  a parameter and is recorded as its own treatment so it can be excluded from
  analysis.

#### 5.5.2 Choosing a treatment per account

Each account resolves to exactly one treatment from the matrix keyed on
`account_type_code` (~45 types). The matrix cell selects a **behaviour**, not a
value — which is what makes this table different from every other table in the
library and worth watching:

| Treatment | Applies to | Figure used |
|---|---|---|
| `STATED` | Instalment credit with a fixed instalment — personal loans, vehicle finance, mortgages | The reported instalment. |
| `PCT_LIMIT` | Revolving facilities where the limit is the exposure — credit cards, store cards, overdrafts | A percentage of the limit, illustratively 5% for cards and 3% for overdrafts. |
| `PCT_BALANCE` | Revolving facilities where the Bank prices off drawn balance | A percentage of the balance, floored at an absolute minimum. |
| `GREATER_OF` | Revolving where both are meaningful | The greater of stated instalment and the imputed figure. |
| `MIN_PAYMENT` | Products with a contractual minimum-payment formula | The formula: a percentage of balance plus interest and fees, floored at an absolute amount. |
| `EXCLUDE_CLOSED` | Settled, closed, written-off, paid-up accounts | Zero, subject to §5.5.1's two-source rule. |
| `EXCLUDE_ON_QUOTE` | Any account with a live settlement quotation | Zero — **only in scenario mode** (§5.8). Never in new-application mode. |
| `CONTINGENT` | Surety, guarantor, co-signatory where another party is principal debtor | A percentage of the principal's instalment, illustratively 20%, rising to 100% where the principal account is in arrears. |
| `TERM_AWARE` | Accounts maturing inside the proposed term | The instalment, plus a recorded note of the maturity date — see below. |
| `REFER` | Account types the matrix cannot treat mechanically | No figure; forces `indeterminate`. |

Four points about the matrix that decide whether it is one table or several:

- **A cell carries a behaviour and its coefficients together.** `PCT_LIMIT` at 5%
  and `PCT_LIMIT` at 3% are the same behaviour with different parameters; a
  quarterly retune changes the coefficient, not the behaviour. Behaviour changes
  are rarer and are approved differently.
- **`TERM_AWARE` deliberately does not reduce the obligation.** An account with
  four months left is a real obligation today. Whether its expiry may be
  anticipated is an appetite question belonging to the caller, so the maturity
  date is *emitted*, not acted on. Anticipating expiry here would be lending
  against money the applicant does not yet have.
- **Arrears do not change the treatment**, but they are recorded. Catch-up amounts
  are not added to the obligation — the contractual instalment is the obligation —
  but `worst_arrears_months` and `accounts_in_arrears_count` travel to the caller,
  who may decline on them. Project 08 needs the opposite: there, the arrears *are*
  the subject.
- **The applicant's own new facility is never in this list.** Where the new
  facility replaces an existing one, the replaced account is removed by the
  caller's scenario construction (§5.8), not by an inference here.

#### 5.5.3 Aggregation and the per-account annotation

Both outputs are required, and neither is a convenience:

- **The scalar.** `existing_obligations`, split internal and external because the
  Bank's exposure to itself is treated differently by appetite;
  `total_exposure`; `revolving_utilisation` as drawn over limit across revolving
  accounts; `worst_arrears_months`; `accounts_in_arrears_count`.
- **The per-account annotation.** For every account in the input list: the
  treatment code, the figure used, the basis that produced it (stated, imputed,
  formula, contingent percentage), the source that won the dedup, whether a
  discrepancy was recorded, and the reason for any exclusion. Project 06 orders
  its consolidation search on this annotation and cannot run without it. Projects
  03 and 07 consume only the scalar. The annotation must therefore be producible
  without being forced on callers who do not want eighty rows back, and the
  scalar must never be computed by a second code path that could disagree with
  the annotation it summarises.

**Must be recorded.** The assembled list with its provenance, every dedup group
and which source won, every discrepancy, each account's treatment and figure, the
matrix version, and any enquiry-velocity uplift as its own line.

### 5.6 Stage 6 — Discretionary income and capacity

**Determines.** `discretionary_income`, `max_affordable_instalment`,
`max_affordable_instalment_unadjusted`, `affordability_buffer_applied`.

#### 5.6.1 The chain

The prescribed arithmetic, in the prescribed order, with every intermediate
retained because the adjudicator asks for the ladder and not the answer:

```
gross_monthly_income
  − statutory_deductions                 (§5.3)
  = net_monthly_income
  − living_expenses                      (§5.4, the higher-of resolution)
  − court_ordered_deductions             (§5.3)
  − existing_obligations                 (§5.5)
  = discretionary_income
```

Order matters for presentation, not arithmetic, and presentation is the point: a
regulator reads this ladder against the regulation line by line. A discretionary
income that is correct but assembled in a different order is harder to defend
than it needs to be.

`discretionary_income` may be negative, and a negative value is meaningful — it
is the finding that the applicant is already over-committed, and its magnitude
tells project 06 how much relief a consolidation must produce. It is never
floored at zero.

#### 5.6.2 The buffer, and overlays on it

`max_affordable_instalment` is not `discretionary_income`. Lending an applicant
their entire discretionary income leaves nothing for the variance the evidence
cannot see. Two constraints apply and the binding one wins:

- a **proportional buffer** — a percentage of discretionary income retained,
  from a grid keyed on `risk_grade` × `product_code`, illustratively 10% at grade
  1 rising to 35% at grade 12, and higher on longer-term products;
- an **absolute residual floor** — a minimum rand amount that must remain after
  the proposed instalment, keyed on `dependants_count`, which protects
  low-income applicants for whom a percentage is too small to matter.

`max_affordable_instalment_unadjusted` is the result of those two. The overlay
stack (`core.adjustments`) then applies, in its declared order, producing
`max_affordable_instalment`. Both survive, and the difference between them is the
Bank's chosen conservatism as distinct from the statutory calculation.

Five requirements on overlays here, and the first is the one that matters:

1. **Overlays may only move the answer in the conservative direction.** An
   overlay may raise the buffer, raise the residual floor, raise the expense
   stringency or lower the maximum instalment. An overlay that would increase
   capacity is **invalid at definition time**, not rejected at runtime. The
   statutory calculation is a floor on conservatism, and a mechanism that can
   breach it is a reckless-lending mechanism regardless of intent.
2. Scope is declared — product, segment, channel, grade range — and an overlay
   evaluated outside its scope is an error, not a no-op.
3. The stack's order is declared. A grade-scoped buffer overlay and a
   channel-scoped one both applying to one assessment compose in a stated order,
   and the composed effect is recorded per overlay, not only in total.
4. Every overlay carries an expiry. The failure mode is a tightening applied in
   one bad quarter, still in force four years later, with nobody able to say what
   unwinding it would do — and here it is worse than elsewhere, because a stale
   overlay declines affordable applicants invisibly and produces no complaint to
   detect it by.
5. The assessment must be runnable with the stack disabled, through the same
   implementation, because that is how the statutory calculation's own behaviour
   is monitored.

**Must be recorded.** Every rung of the ladder, the buffer grid cell, which of
the two constraints bound, `adjustment_set_id`, `adjustments_applied` with each
overlay's effect, and both the adjusted and unadjusted capacity.

### 5.7 Stage 7 — Verdict, and the three shapes of answer

**Determines.** `affordability_verdict_code`, `discretionary_income_after`,
`evidence_sufficiency_code`.

#### 5.7.1 The verdict

| Verdict | Condition | Meaning |
|---|---|---|
| `pass` | The proposed instalment is at or below `max_affordable_instalment` | Affordable on the evidence. |
| `marginal` | Within a tolerance band above it, illustratively 5% | Affordable only on assumptions the evidence does not support. The caller decides; this project does not. |
| `fail` | Above the tolerance band | Not affordable on the evidence. |
| `indeterminate` | The evidence does not support any of the above | **Not a failure of affordability.** |

`indeterminate` is a distinct outcome and conflating it with `fail` is the most
consequential error available in this project. It means the Bank could not
establish something — income below the product's minimum tier, a stale bureau
view, a `REFER` account type, a statement confidence below threshold, an
applicant on a joint application whose income could not be established. The
remedy is evidence, not decline, and the client-facing communication differs
accordingly. `evidence_sufficiency_code` names which of these applied; it is zero
when the verdict is not `indeterminate`.

#### 5.7.2 Three consumers, three shapes

The same seven stages serve three questions, and the difference between them is
entirely in what the caller asks for at the end:

**(a) Pass or fail against a known instalment.** Projects 05 and 08 arrive with
an instalment in hand and want a verdict. `discretionary_income_after` is emitted
so the caller can see the margin.

**(b) The largest instalment the applicant could carry.** Projects 07 and 04
arrive with no instalment and want capacity. `max_affordable_instalment` is the
answer; no proposed instalment is supplied and the verdict is `pass` or
`indeterminate` only.

**(c) The largest *amount* the applicant could borrow.** Project 03 wants this
and **this project cannot answer it**. The amount depends on the instalment,
which depends on the rate and the fees, which depend on the amount. The loop
closes in project 03, not here.

The boundary in (c) is a requirement, not a disclaimer. For project 03's search
to terminate and be defensible, this project must guarantee three properties:

1. **Monotonicity in the proposed instalment.** If instalment *I* passes, every
   instalment below *I* passes on the same inputs. Without it project 03's search
   has no valid stopping condition and could return an offer that fails a
   re-check. Monotonicity must be tested, not assumed — the residual floor and the
   buffer interact, and a naive implementation can produce a non-monotone step.
2. **Stability under repetition.** The same inputs give the same answer on the
   four-hundredth call as on the first. Nothing accumulates between calls.
3. **Separability of the expensive part.** Within one project-06 assessment the
   income, deductions and expenses do not change across scenarios; only the
   obligation set does. The assessment must expose a way to vary the obligations
   without redoing the evidence waterfall four hundred times. This is stated as a
   requirement because the alternative — project 06 reimplementing the cheap part
   itself — is exactly the fork this project exists to prevent.

### 5.8 The four assessment modes

One implementation, four callers, four evidence situations. `assessment_mode_code`
selects the mode; the stages are the same stages.

| | **New application** | **Limit increase** (07) | **Arrangement** (08) | **Scenario** (06) |
|---|---|---|---|---|
| Income evidence | Full waterfall, tiers 1–4 typical | No fresh payslip; tier 3 internal deposits dominant; declared income refreshed on a cadence | Often unverifiable; client under stress and disengaged | Inherited from the parent assessment, computed once |
| Stale income evidence | `indeterminate` | Routes to a *conditional* outcome — the increase may be offered subject to the client confirming income — rather than failing | Permits a weaker tier with the outcome reinterpreted as sustainability, not appetite | n/a |
| Expenses | Declared vs norm floor vs statement | As new application, with declared expenses often months old | Statement-derived preferred; the norm floor still binds, because an unsustainable arrangement is not a kindness | Inherited |
| Obligations | Full list, `EXCLUDE_ON_QUOTE` never applies | Full list; the facility being increased is included at its *current* limit | Full list; the account in arrears is the subject and is handled by the caller | Hypothetical — quoted accounts removed, the new facility's instalment added |
| Buffer | Standard grid | Standard grid; a distinct overlay scope in practice | Different grid: sustainability rather than appetite, with a larger residual floor | Inherited from new application |
| Typical verdict use | pass / fail / indeterminate | capacity (shape b) | pass / fail on a proposed arrangement | pass / fail, hundreds of times |

Two requirements on the modes themselves:

- **A mode may not change the arithmetic.** It selects evidence rules, parameter
  sets and which outputs are produced. A mode that needed different arithmetic
  would be a different calculation and would have to be justified as such.
- **Modes must not become copies.** The observable failure is four
  near-identical assessments diverging over two years, with the tax table fixed
  in three of them. Any structure in which a mode can be updated without the
  others is a structure that will produce that.

---

## 6. Parameters and tables

### 6.1 Tables

| Table | Dimensions | Cells | Owner | Cadence | Effective-dated |
|---|---|---|---|---|---|
| Statutory expense norms | 12 income bands × 6 dependant counts × 2 components | 144 | Compliance | On gazette | **Yes** |
| Internal expense norms | 12 × 6 × 8 products | 1 152 | Credit Risk Policy | Quarterly | Yes |
| Obligation treatment matrix | 45 account types × behaviour + 5 coefficients | 270 | Credit Risk Policy | Quarterly | Yes |
| Income haircut matrix | 6 tiers × 6 employment types | 36 | Credit Risk Policy | Quarterly | Yes |
| Haircut modifiers | 3 modifiers × thresholds and increments | 12 | Credit Risk Policy | Quarterly | Yes |
| Tax brackets and rebates | 8 brackets × 4 rebate classes | ~50/year | Credit Systems | Annual, occasionally mid-year | **Yes** |
| Unemployment insurance | rate + ceiling | 2 | Credit Systems | Annual | **Yes** |
| Buffer grid | 12 grades × 8 products | 96 | Credit Risk Policy | Quarterly | Yes |
| Residual floor | 6 dependant counts | 6 | Credit Risk Policy | Quarterly | Yes |
| Minimum evidence tier | 8 products | 8 | Compliance | Rarely | Yes |
| Expense category classification | 14 categories × shared/personal | 14 | Credit Risk Policy | Rarely | Yes |
| Adjustment register | 0..12 live overlays × 11 attributes | ~130 | Credit Committee | Ad hoc | **Yes** |

Two properties are required of all of them: **cell-level attribution** — an
outcome names the cell of the version of the table that produced it — and
**diffability**, because a quarterly treatment-matrix change is reviewed by
somebody who needs to see what moved, not a replacement file.

### 6.2 Parameter ownership

Four classes, and the boundaries between them are the governance model:

| Class | Examples | Owner | Who may not change it |
|---|---|---|---|
| **Statutory** | Norms table, tax brackets, insurance ceiling, minimum evidence tiers, the averaging floor | Compliance, transcribing the gazette | Everyone else, including Credit Risk Policy |
| **Policy** | Internal norms, treatment matrix, haircuts, buffer grid, residual floor, verdict tolerance | Credit Risk Policy | Product teams |
| **Consumer-local** | Which mode, which product's minimum tier applies, whether the annotation is wanted, the proposed instalment | The calling project | — |
| **Overlay** | Buffer and stringency overlays with scope, order and expiry | Credit Committee | Product teams; and no overlay may loosen |

A product team must be able to set consumer-local values without reaching policy
values, and no party may reach statutory ones through this project at all. The
statutory class is not merely owned by Compliance — it is unreachable from the
consuming projects, which is a stronger statement than an access convention.

---

## 7. Outputs

**Always.** `gross_monthly_income`, `income_source_code`,
`income_verification_tier`, `income_haircut_applied`, `income_variability_ratio`,
`statutory_deductions`, `court_ordered_deductions`, `net_monthly_income`,
`living_expenses`, `living_expenses_unadjusted`, `expense_basis_code`,
`norm_table_version`, `existing_obligations`, `obligations_internal`,
`obligations_external`, `total_exposure`, `revolving_utilisation`,
`worst_arrears_months`, `accounts_in_arrears_count`, `discretionary_income`,
`max_affordable_instalment`, `max_affordable_instalment_unadjusted`,
`affordability_buffer_applied`, `affordability_verdict_code`,
`evidence_sufficiency_code`, `adjustment_set_id`, `adjustments_applied`.

**On request.** The per-account annotation (0..80 rows); the full evidence ladder
in rendered form; `discretionary_income_after` where a proposed instalment was
supplied.

**Persisted, for every assessment.** The complete input snapshot, every table
version resolved, every intermediate in the ladder, the per-account annotation,
the overlay stack, and `decision_date`. Seven years. Project 09 consumes it.

---

## 8. Non-functional requirements

| Requirement | Value |
|---|---|
| Arithmetic core, single assessment | Under 1 ms, excluding evidence retrieval |
| Repeated call inside project 06's search | Under 1.5 ms with the evidence portion held constant, so 400 scenarios fit a 900 ms budget with room for pricing |
| New application, end to end | Contribution under 8 ms at p99, excluding external data calls |
| Batch, project 07 | 14 M records within a 3-hour window |
| Determinism | Identical inputs, versions and `decision_date` produce identical outputs to the cent |
| Replay | A 2026 assessment re-derived in 2033 reproduces exactly, using 2026 tables |
| Availability | The real-time path inherits origination's availability target; there is no degraded affordability mode, because a guessed affordability answer is worse than no answer |

There is deliberately no "fast approximate" mode. A cheaper approximation would
be used by project 06 inside its search and would then disagree with the
authoritative assessment on the offer that search produced — which is the exact
defect class that makes an offer fail its own re-check.

---

## 9. Audit, evidence and explainability

### 9.1 The adjudicator's question

An ombud adjudicator, four years later, with one application and no technical
background, asks: *on what basis did you conclude this person could afford this?*
The answer must be a single artefact showing, in the regulation's own order:

1. Each income source, its evidence tier, the document or observation that
   established it, the haircut and why, the months averaged and the months
   excluded with reasons, and the figures before and after.
2. Tax, insurance and retirement deductions with the table version and the rebate
   class, and the age at `decision_date` that selected it.
3. Court-ordered deductions and their instruments.
4. All four expense bases — declared, statement-derived, statutory norm,
   internal norm — which bound, the norm table version, and the band and
   dependant cell.
5. Every account considered, its treatment, the figure used, and every account
   excluded with its reason.
6. The ladder to discretionary income.
7. The buffer, the residual floor, which bound, and — separately — every overlay
   applied, with the unadjusted figure beside the adjusted one.
8. The verdict, and where `indeterminate`, what was missing.

It must be generated from what actually ran. A hand-maintained description will
diverge, and a divergence between the documented calculation and the deployed one
is itself the finding.

### 9.2 The statutory answer and the Bank's conservatism

The adjudicator's question is frequently *sharper* than affordability: it is
whether the Bank declined someone the statutory calculation would have approved,
and on what authority. Answering requires the unadjusted figures beside the
adjusted ones, the overlay that made the difference, its approval reference and
its scope. An assessment that records only the final number cannot answer it, and
"the system said no" is not an answer to a regulator.

### 9.3 Reproduction

A sampled re-derivation must reproduce to the cent from the persisted snapshot
without reaching any live system. Reaching a live bureau or tax table during a
replay produces today's answer to a historical question, which is the failure
mode effective dating exists to prevent — and it fails silently, returning a
plausible number.

### 9.4 Diagnosis

When an assessment is wrong, the investigation needs: the ladder with every
intermediate; the ability to change one input — one account's treatment, one
income source's tier, the overlay stack — and re-run without deploying anything;
and the changed run marked so it can never be confused with the original. The
common real defects are a dedup that merged two genuinely different accounts, an
income month silently excluded, and an overlay applied outside its scope.

### 9.5 Sensitivity

Every artefact here is applicant financial data: income, employer, expenses,
debts, arrears. A rendered evidence ladder is among the most sensitive records
the Bank holds about a person. Access is controlled, non-production use is
masked, and a debugging session on a real assessment is production data access.

---

## 10. Acceptance criteria

1. Five consuming projects use one implementation. No fork, no near-copy, no
   consumer-side reimplementation of any stage.
2. The four modes share the arithmetic; a change to the tax calculation reaches
   all four at once, demonstrably.
3. Monotonicity in the proposed instalment holds, and is tested across band
   edges, buffer boundaries and the residual floor — not asserted.
4. Four hundred scenario assessments complete within project 06's budget with the
   evidence portion computed once.
5. A 2026 assessment replayed in 2033 reproduces to the cent.
6. Compliance can supply a spreadsheet of inputs and expected outputs and have it
   run as a test suite, without an engineer.
7. `indeterminate` is never returned as `fail`, demonstrated by a test set built
   from each `evidence_sufficiency_code`.
8. An overlay that would increase capacity is rejected when defined, not when
   run.
9. Every assessment can be run with the overlay stack disabled, through the same
   implementation.
10. The scalar obligation figure and the per-account annotation are produced by
    one computation and cannot disagree.
11. The rendered evidence ladder is legible to a non-technical adjudicator,
    judged by one.
12. Every table version resolved is recorded on every assessment.

---

## 11. Change scenarios

1. **The regulator gazettes a new norms table** effective the first of next
   month. Every assessment before that date must use the old table forever, and
   both must be live simultaneously during the transition.
2. **The tax tables change mid-year** for the first time, rather than at year end.
3. **Credit Risk Policy wants the buffer to vary by channel**, where today it
   varies by grade and product — broker-originated business having deteriorated.
4. **A new account type appears** at the bureau and needs a treatment. Until it
   has one it must produce `indeterminate`, not silently score zero.
5. **The treatment for credit cards changes** from 5% of limit to the greater of
   5% of limit and the stated instalment — a behaviour change, not a coefficient
   change, and approved differently.
6. **Project 06 needs the three most expensive obligations by instalment**
   returned, which no other consumer wants.
7. **The unemployment insurance ceiling rises**, and an applicant with two
   employers is found to have been under-deducted for eight months.
8. **Collections wants a distinct buffer grid** for arrangement mode, with a
   larger residual floor and a different tolerance for `marginal`.
9. **A court finds** that a particular expense category was consolidated when it
   should have been summed on joint applications, requiring re-derivation of two
   years of assessments under the corrected rule — while the original assessments
   remain reproducible as they were.
10. **A fifth consumer arrives** — a new product team — wanting shape (b) with a
    product minimum evidence tier that does not yet exist.
11. **Observed early arrears rise** on one channel and the Credit Committee
    applies a buffer overlay for two quarters; the approval-rate impact must be
    estimated before it goes live and measured after.
12. **An overlay from three years ago is discovered still in force.** It must be
    possible to list every live overlay with its age, approval and expiry, and to
    show what unwinding it would do to today's approval rate.
13. **The statement aggregator changes its confidence scale** from 0..1 to
    0..100 without telling anyone. Every affected assessment is plausible and
    wrong.

---

## 12. Out of scope

- Obtaining the evidence: bureau calls, statement aggregation, document capture
  and OCR, employer register maintenance.
- Deciding what to do with the verdict. This project says what is affordable; it
  does not approve, decline, price or size anything.
- Computing a maximum loan amount (§5.7.2(c)) — project 03.
- Choosing which accounts to settle — project 06.
- Fraud and identity assessment of the evidence supplied.
- Storage and retrieval of the persisted assessment record — project 09.
- Fitting or validating any model.

---

## 13. Questions the implementation must answer

1. **How does one calculation serve five consumers with five evidence sets and
   three shapes of answer** without either a fork or a parameter surface so wide
   that every consumer configures a different calculation?
2. **What is a mode?** It selects evidence rules, parameter sets and outputs but
   may not change the arithmetic. What structure enforces that boundary, rather
   than documenting it?
3. **How is the expensive part separated from the cheap part** so that four
   hundred scenario assessments recompute only the obligations, without the
   separation becoming a second entry point that can drift from the first?
4. **How is monotonicity in the proposed instalment guaranteed**, given that the
   buffer and the residual floor interact, and how is it tested?
5. **What is the obligation treatment matrix, structurally?** Its cells select
   behaviours with coefficients, not values. Is that a table, a rule set, or
   something the core component vocabulary does not yet have?
6. **How does one computation produce both a scalar and a per-element annotation**
   over a ragged collection, with callers able to take either, and with no
   possibility of the two disagreeing?
7. **Where does the highest-of-four-bases resolution live?** It recurs — expense
   bases here, the cap waterfall in project 03, roll-up in project 05. Is it a
   component or a shape that keeps being rewritten?
8. **How is effective dating made unforgettable**, given that forgetting it
   produces a plausible wrong answer rather than a failure, and that four
   different tables need it?
9. **How is the statutory class made unreachable** from five consuming projects —
   not by convention, but structurally?
10. **What is an overlay here**, given that it must be conservative-only,
    scoped, ordered, expiring, separately approved, and removable at runtime
    through the same implementation?
11. **How is the evidence ladder generated** such that it is simultaneously the
    execution record and a document an ombud adjudicator can read? Doc 04 §6
    ranks this as the top open risk; this project is where a real adjudicator can
    judge it.
12. **How is `indeterminate` kept distinct from `fail`** through five consumers,
    four modes and every intermediate, when the cheapest implementation of every
    one of them is a boolean?
13. **How is a spreadsheet of expected inputs and outputs, supplied by
    Compliance, made runnable as a test** without an engineer translating it?
