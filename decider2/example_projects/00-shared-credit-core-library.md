# 00 — Shared credit core library

> Fictional. The Bank, its products, thresholds, table dimensions and volumetrics
> are invented for this repository. Regulatory mechanisms referred to are the
> published public ones; every number attached to them is illustrative.

---

## 1. What this is

Every other project in this set needs to work out what somebody earns, what they
already owe, what an instalment would be, what the fees are, what a score means
as a probability, and why an application was declined. Nine projects each
answering those questions separately is the failure this library exists to
prevent.

`credit-core` is the Bank's shared decision library. It publishes a small number
of **capabilities** — named, versioned, independently testable units of credit
logic — which consuming projects assemble into flows. It is owned by a central
Credit Systems team, consumed by six product teams, and changed under a release
process that consumers can survive.

It is also the project that most directly tests whether reuse in `decider2` is
real. A library that consumers have to fork, wrap, or copy-paste out of is not a
library, and the design docs record exactly that outcome in the previous
generation: 79 identity-passthrough steps existed because a consumer could not
rename a value a shared unit produced (doc 01 §5.1).

---

## 2. Why it is in this set

| Question | How this project stresses it |
|---|---|
| **Q1 reusable components** | This *is* Q1. The library's contents are the hypothesis; the other nine specs are the test. |
| **Q3 parameters** | Shared tunables (prime rate, appetite settings) and consumer-local tunables must coexist without either side owning the other. |
| **Q4 custom modules across flows** | Six consumers, one implementation, no forks. |
| **Q5 core component set** | The library is where "this shape recurs" becomes visible. Anything appearing in three capabilities is a candidate core component. |
| **Q6 codebase organisation** | A library, six consumers, a vocabulary, contracts, and a release train. |
| **Q2 tables** | Several capabilities are mostly table. Who owns the table, the library or the consumer? |
| **Q7 audit** | Adjustments (§6.22) are deliberate, approved changes to an approved artefact's answer. Separating model output from policy overlay is a governance requirement, not a preference. |

---

## 3. Actors

| Actor | Responsibility |
|---|---|
| **Credit Systems** | Owns the library. Approves interface changes. Runs the release train. |
| **Product engineering teams** (×6) | Consume capabilities. May not modify them. May parameterise them. |
| **Credit Risk Policy** | Owns parameter values and table contents for policy-bearing capabilities. Not engineers. |
| **Regulatory Compliance** | Signs off the capabilities implementing statutory calculations. Requires evidence, not code. |
| **Treasury / Pricing** | Owns rate cards and the cost-of-funds inputs. |
| **Internal Audit** | Asks, years later, what a capability did on a specific date. |

---

## 4. Canonical vocabulary

The library owns these names. Consuming projects use them unchanged; a project
that needs a different name for the same concept declares the mapping in one
place and does not rename per use site.

### Client and application

| Name | Type | Meaning |
|---|---|---|
| `client_id` | int64 | Internal client identifier. Stable for life. |
| `application_id` | int64 | One credit request. |
| `decision_date` | date | The date whose rules, rates and norms govern this assessment. **Not** "today" — see §7.3. |
| `product_code` | int16 | See §5. |
| `channel_code` | int8 | Branch, app, web, call centre, broker, partner. |
| `applicant_age_years` | float64 | At `decision_date`. |
| `dependants_count` | int8 | Financially dependent persons. |
| `employment_type_code` | int8 | Permanent, contract, self-employed, pensioner, social grant, informal. |
| `months_employed` | float64 | Current employer tenure. |
| `is_joint_application` | bool | Two applicants assessed as one household. |

### Income and affordability

| Name | Type | Meaning |
|---|---|---|
| `gross_monthly_income` | float64 | Before deductions, after verification and haircuts. |
| `income_source_code` | int8 | Which evidence tier established the income (§6.1). |
| `statutory_deductions` | float64 | Tax, unemployment insurance, compulsory retirement. |
| `net_monthly_income` | float64 | Gross less statutory deductions. |
| `living_expenses` | float64 | The applied figure, after the norm floor is enforced. |
| `existing_obligations` | float64 | Monthly cost of debt already held. |
| `discretionary_income` | float64 | Net less living expenses less existing obligations. |
| `max_affordable_instalment` | float64 | The most the applicant may be committed to, after appetite haircut. |
| `affordability_verdict_code` | int8 | Pass, marginal, fail, indeterminate. |

### Risk

| Name | Type | Meaning |
|---|---|---|
| `scorecard_id` | int16 | Which scorecard produced `score`. |
| `score` | float64 | Scaled score. |
| `probability_of_default` | float64 | Calibrated 12-month PD, 0..1. |
| `risk_grade` | int8 | 1 (best) .. 12 (worst). The pricing and appetite key. |
| `bureau_as_of_date` | date | When the bureau view was taken. |
| `bureau_is_stale` | bool | Older than the product's tolerance at `decision_date`. |
| `score_unadjusted` | float64 | The scorecard's own output, before any overlay (§6.22). |
| `probability_of_default_unadjusted` | float64 | Calibrated PD before any overlay. |
| `adjustment_set_id` | int16 | Which set of overlays was in force. |
| `adjustments_applied` | list[int16] | Every overlay that altered a value, in application order. |

### Offer and pricing

| Name | Type | Meaning |
|---|---|---|
| `requested_amount` | float64 | What the client asked for. |
| `offered_amount` | float64 | What the Bank will advance. |
| `term_months` | int16 | Contractual term. |
| `nominal_annual_rate` | float64 | Annual interest rate before fees. |
| `initiation_fee` | float64 | Once-off, statutorily capped. |
| `monthly_service_fee` | float64 | Statutorily capped. |
| `credit_life_premium` | float64 | Monthly, statutorily capped. |
| `instalment` | float64 | Total monthly payment including fees and premium. |
| `total_cost_of_credit` | float64 | Sum of all payments over the term. |
| `effective_annual_rate` | float64 | All-in cost expressed annually. |

### Outcome

| Name | Type | Meaning |
|---|---|---|
| `outcome_code` | int8 | Approve, approve with conditions, refer, decline. |
| `decline_reason_codes` | list[int16] | Every reason that applied, most severe first. |
| `primary_reason_code` | int16 | The one communicated to the client. |
| `referral_queue_code` | int8 | Where a referral goes. |

---

## 5. Product catalogue

Used consistently across all ten specs.

| `product_code` | Product | Type | Term range | Amount range |
|---|---|---|---|---|
| 10 | **Flex Loan** | Unsecured term loan | 6–84 months | R2 000 – R500 000 |
| 11 | **Flex Loan Consolidation** | Unsecured, settlement-linked | 12–84 months | R10 000 – R500 000 |
| 20 | **Everyday Card** | Revolving credit card | n/a | R1 000 – R300 000 limit |
| 21 | **Access Facility** | Revolving credit facility | n/a | R500 – R150 000 limit |
| 30 | **Drive Finance** | Vehicle asset finance, secured | 12–72 months | R30 000 – R1 500 000 |
| 40 | **Home Loan Further Advance** | Secured against property | 12–240 months | R50 000 – R2 000 000 |
| 50 | **Business Term Facility** | SME, secured or unsecured | 6–60 months | R50 000 – R10 000 000 |
| 51 | **Business Revolving Facility** | SME revolving | n/a | R50 000 – R5 000 000 |

---

## 6. Published capabilities

Each capability is a named unit with a declared interface. The library publishes
**twenty-two**. They are listed with what they take, what they produce, and the
thing about them that makes reuse hard — because the hard part is the point.

### 6.1 `core.income` — income determination

Establishes `gross_monthly_income` and `income_source_code` from whichever
evidence exists, applying a verification haircut per evidence tier.

- **Takes**: declared income, payslip-derived income, bank-statement-derived
  income and its confidence, employer-confirmed income, bureau-estimated income,
  `employment_type_code`, `months_employed`, variable-pay history (up to 12
  monthly values, ragged).
- **Produces**: `gross_monthly_income`, `income_source_code`,
  `income_verification_tier`, `income_haircut_applied`,
  `income_variability_ratio`.
- **Hard part**: the evidence waterfall is a policy artefact that Credit Risk
  Policy changes without an engineer, but the arithmetic per tier is not. Where
  the line falls decides whether this is one capability or two.

### 6.2 `core.deductions` — statutory deductions

- **Takes**: `gross_monthly_income`, `employment_type_code`, tax year context.
- **Produces**: `statutory_deductions`, `net_monthly_income`.
- **Hard part**: tax tables are effective-dated and must be selected by
  `decision_date`, not by today's date. A replay in 2028 of a 2026 decision must
  use 2026 tables.

### 6.3 `core.expense_norms` — minimum living expense floor

Implements the statutory minimum-expense-norm mechanism: a floor on declared
living expenses, by income band, so that a client cannot be lent to on the basis
of an implausibly low expense claim.

- **Takes**: `gross_monthly_income`, `dependants_count`, declared living
  expenses, statement-derived living expenses.
- **Produces**: `living_expenses`, `expense_basis_code` (declared / norm floor /
  statement-derived), `norm_table_version`.
- **Table**: income bands × dependant counts, roughly 12 × 6, plus a fixed
  component and a marginal rate per band. Regulator-published; the Bank
  additionally maintains a stricter internal variant per product.
- **Hard part**: two tables, one statutory and one internal, both
  effective-dated, with the binding one being the higher. Which table applied
  must be in the evidence.

### 6.4 `core.obligations` — existing debt obligations

Converts a variable-length list of the client's existing credit accounts into a
monthly obligation figure and supporting detail.

- **Takes**: an account list (0..80 accounts, ragged) with per-account
  `account_type_code`, balance, limit, instalment, months in arrears, opened
  date, closed flag, is-internal flag, settlement quote where available.
- **Produces**: `existing_obligations`, `obligations_internal`,
  `obligations_external`, `worst_arrears_months`, `accounts_in_arrears_count`,
  `total_exposure`, `revolving_utilisation`, and a per-account
  `obligation_treatment_code`.
- **Table**: treatment matrix, ~45 account types × treatment rule — use stated
  instalment, impute as a percentage of limit, impute as a percentage of balance,
  exclude, or exclude only when a settlement is in flight.
- **Hard part**: it consumes a ragged collection and emits both a scalar
  aggregate and a per-element annotation. Project 06 needs the per-element
  detail; projects 03 and 07 need only the scalar. One capability, two shapes of
  answer.

### 6.5 `core.affordability` — discretionary income and capacity

- **Takes**: `net_monthly_income`, `living_expenses`, `existing_obligations`,
  `risk_grade`, `product_code`, household composition.
- **Produces**: `discretionary_income`, `max_affordable_instalment`,
  `affordability_verdict_code`, `affordability_buffer_applied`.
- **Hard part**: three consumers want three different things from it — a
  pass/fail, a maximum instalment, and a maximum loan amount. The third is not
  computable here, because it depends on the rate, which depends on the amount.
  See project 03.

Full flow in [project 02](02-affordability-assessment.md).

### 6.6 `core.instalment` — amortisation

- **Takes**: `offered_amount`, `term_months`, `nominal_annual_rate`, fee
  components, balloon or residual amount where applicable, payment timing
  convention.
- **Produces**: `instalment`, `total_cost_of_credit`,
  `effective_annual_rate`, `total_interest`, first and final payment amounts.
- **Hard part**: it is called inside iterative solves (projects 03, 05, 06) tens
  of times per application, and its inverse — solve the advance that produces a
  given instalment — is needed just as often. Whether the inverse is a second
  capability or the same one run differently is an open question the specs do not
  answer.

### 6.7 `core.fees` — statutory fee schedule

- **Takes**: `offered_amount`, `product_code`, `decision_date`.
- **Produces**: `initiation_fee`, `monthly_service_fee`, indirect tax on each,
  and the capped-versus-calculated flags.
- **Hard part**: caps are legislated, inflation-adjusted annually, and
  effective-dated. The calculation is a small piecewise function of amount with
  a ceiling. Getting it wrong is a regulatory finding, so it needs its own
  evidence trail even though it is ten lines of arithmetic.

### 6.8 `core.credit_life` — credit life premium

- **Takes**: `offered_amount`, outstanding balance basis, `term_months`,
  `applicant_age_years`, `employment_type_code`, cover type (single or joint),
  whether the client has substituted their own policy.
- **Produces**: `credit_life_premium`, `credit_life_cap_applied`,
  `cover_type_code`.
- **Table**: premium rate per R1 000 of cover, by age band × term band ×
  employment type — about 14 × 8 × 6. Capped by statute.

### 6.9 `core.rate_card` — priced rate lookup

A generic keyed lookup over a priced grid, used by every lending project with a
different grid.

- **Takes**: `product_code`, `offered_amount`, `term_months`, `risk_grade`,
  plus product-specific extra keys (loan-to-value for Drive Finance, security
  type for business facilities).
- **Produces**: `nominal_annual_rate`, `rate_card_version`,
  `rate_cell_id`, and the flags for band-edge and out-of-range lookups.
- **Table**: see §8. This is the largest artefact in the library.
- **Hard part**: the grid is authored in a spreadsheet by Treasury, refreshed
  monthly, occasionally patched mid-month, and has to be diffable, validatable
  and attributable cell by cell.

### 6.10 `core.scorecard` — scorecard evaluation

- **Takes**: a characteristic vector (20–60 values, mixed numeric and
  categorical, frequently null).
- **Produces**: `score`, per-characteristic points contribution,
  `scorecard_id`, `scorecard_version`, and the reason codes for the largest
  negative contributions.
- **Hard part**: the per-characteristic contributions are required output, not
  a debugging nicety — adverse-action explanation depends on them. Nulls are a
  scoring bin, not an error.

### 6.11 `core.calibration` — score to probability, and back

- **Takes**: `score`, `scorecard_id`, calibration segment.
- **Produces**: `probability_of_default`, and the inverse — the score
  corresponding to a target probability.
- **Hard part**: the inverse direction is used in appetite logic ("what score
  would we need for this to be worth doing at this price?"), so the relationship
  must be invertible and both directions must agree to tolerance.

### 6.12 `core.risk_grade` — grade assignment

- **Takes**: `probability_of_default`, `product_code`, segment.
- **Produces**: `risk_grade`, and the probability band edges that produced it.
- **Table**: 12 grade boundaries per product per segment; roughly 8 × 5 × 12.

### 6.13 `core.bureau` — bureau view normalisation

- **Takes**: the raw bureau response — header, account list, enquiry list,
  public-record list, all ragged.
- **Produces**: normalised counts, worst-status indicators, ages of adverse
  items, enquiry velocity over several windows, `bureau_as_of_date`,
  `bureau_is_stale`, and a data-quality verdict.
- **Hard part**: three bureaux with three schemas, and a normalisation layer
  that must not silently lose an adverse item. Consumed by every lending
  project and by project 05 at entity level.

### 6.14 `core.adverse_events` — adverse event classification

Classifies a single adverse event (judgment, default listing, administration
order, debt review, tax non-compliance, insolvency, litigation) into a severity.

- **Takes**: event type, amount, date, status, disputed flag, satisfied flag,
  satisfied date.
- **Produces**: `event_severity_code` (immaterial / minor / material /
  disqualifying), `event_age_months`, and the rule that classified it.
- **Hard part**: it operates on one element of a ragged collection, and its
  results must be rolled up — by count, by worst, by weighted amount —
  differently by each consumer. Project 05 nests it two levels deep.

### 6.15 `core.eligibility` — hard eligibility gates

Age, residency, capacity to contract, product availability, existing
relationship status, exclusion lists, sanctioned-status, deceased and estate
flags, debt review and administration status.

- **Produces**: `is_eligible`, `decline_reason_codes`.
- **Hard part**: these gates must short-circuit — there is no point scoring a
  deceased applicant — while still producing a complete reason set.

### 6.16 `core.exposure` — group exposure aggregation

- **Takes**: client and related-party identifiers, existing internal facilities,
  pending applications.
- **Produces**: `total_exposure`, `exposure_headroom`, exposure by product
  family, and the binding cap.
- **Hard part**: "related party" is a graph query, which is set-shaped work, but
  the cap check is record-shaped.

### 6.17 `core.appetite` — risk appetite limits

- **Takes**: `risk_grade`, `product_code`, segment, campaign or channel context.
- **Produces**: maximum amount, maximum term, maximum instalment-to-income
  ratio, minimum price, and which limit bound.
- **Table**: appetite grid, grade × product × segment, ~12 × 8 × 6, with five
  values per cell.

### 6.18 `core.reason_codes` — the decline reason taxonomy

A registry, not a calculation: every reason code the Bank may communicate, with
its severity rank, its client-facing wording in three languages, its regulatory
classification, and its effective dates.

- **Produces**: given a set of fired reasons, the ranked list and the
  `primary_reason_code`.
- **Hard part**: ~380 codes, owned by Compliance, referenced from every project,
  and — in the previous generation — dropped entirely during a port. The
  registry must make that impossible to repeat.

### 6.19 `core.dates` — effective-dated selection

Given `decision_date` and a versioned artefact family, select the version in
force. Used by tax tables, expense norms, fee caps, rate cards, scorecards,
appetite grids and the reason taxonomy.

- **Hard part**: it is the mechanism on which every replay guarantee depends,
  and it is boring enough to be got wrong.

### 6.20 `core.rounding` — money conventions

Rounding of advances to the nearest R100, of instalments to the cent, of rates
to four decimal places, and the direction of each. Stated as a capability
because inconsistent rounding between the pricing step and the contract step is
a real defect class and produces cent-level mismatches that fail reconciliation.

### 6.21 `core.consent` — consent and disclosure state

- **Takes**: client consent records, marketing preferences, bureau-enquiry
  consent, data-sharing consent, all with timestamps and channels.
- **Produces**: which downstream actions are permitted at `decision_date`.
- **Hard part**: consumed by project 04 as a suppression, by projects 03 and 07
  as a precondition, and by project 08 as a channel restriction.

### 6.22 `core.adjustments` — post-model overlays

A scorecard is validated, signed off and then left alone for twelve to
eighteen months. Reality moves faster. When observed default rates drift from
predicted, when the Bank wants to tighten one channel for a quarter, or when a
macroeconomic view says the book is riskier than the model believes, the answer
is **not** to rebuild the model — it is to apply an adjustment over it.

This capability applies named, approved, effective-dated overlays to values
produced elsewhere in the library, and keeps the pre-adjustment value visible
beside the post-adjustment one.

- **Takes**: a base value, its provenance (which scorecard, which grade, which
  segment, which product, which channel), `decision_date`, and the adjustment
  set in force.
- **Produces**: the adjusted value, the unadjusted value, and the ordered list of
  overlays that applied with the effect of each.

**Adjustment kinds that must be expressible:**

| Kind | Applies to | Example |
|---|---|---|
| Score shift | `score` | −18 points on the new-to-bank segment |
| Scaling change | the score-to-odds relationship | points-to-double-the-odds moved from 20 to 22 for one scorecard |
| Odds multiplier | `probability_of_default` | PD × 1.35 on a channel showing early-life deterioration |
| Calibration re-anchor | the calibration relationship | shift the anchor point so a given score maps to a higher PD |
| Grade boundary shift | `risk_grade` | move the grade 6/7 boundary by 0.4 percentage points of PD |
| Cut-off shift | a decision threshold | tighten the minimum acceptable grade by one notch |
| Rate add-on | `nominal_annual_rate` | +75 basis points on a rate card cell range without reissuing the card |
| Cap adjustment | a policy cap | reduce the maximum amount by 20% for one sector |
| Buffer adjustment | the affordability buffer | raise the buffer by 3 percentage points |

**Six properties, each load-bearing:**

1. **An adjustment is an overlay, not an edit.** The base artefact — the
   scorecard, the calibration, the rate card — stays exactly as validated and
   approved. This is not fastidiousness: a model validator must be able to see
   the model's own output and the policy overlay separately, and a regulator
   will ask which is which.
2. **The unadjusted value survives.** Every adjusted output carries its
   unadjusted counterpart. "What would we have done without the overlay?" is
   asked at every Credit Committee.
3. **Adjustments stack, and order is part of the definition.** A segment shift
   and a channel multiplier may both apply to one application; the composition
   order changes the answer and must be declared, not emergent.
4. **Each adjustment carries an identity and a justification** — an id, a
   description, an owner, an approval reference, a rationale, an effective-from
   date, and an effective-to date.
5. **Expiry is mandatory, not optional.** Every adjustment declares a review
   date. An adjustment reaching it without renewal must surface, because the
   real failure mode is a temporary tightening applied during one bad quarter
   that is still silently in force four years later, and which nobody can now
   explain.
6. **Scope is declared.** An adjustment states exactly which populations it
   applies to — product, segment, channel, grade range, scorecard, campaign —
   and applying it outside that scope is an error, not a silent no-op.

**Hard part**: this is a mechanism that deliberately changes the answer of an
already-approved artefact, at runtime, under a different approval path from the
thing it modifies. It is simultaneously a parameter change (values move without
new logic), a structural concern (the composition order is real logic), and an
audit artefact (the overlay stack in force on a date is part of the decision
record). Which of those it most resembles is the question, and the library must
not answer it three different ways in three places.

Consumed by projects 03, 04, 05, 06, 07 and 08. Governed and reported by
project 09.

---

## 7. Library-wide requirements

### 7.1 Interface stability

A published capability's interface is frozen at release. Adding an output is a
minor release; changing the meaning of an existing output, removing one, or
changing a required input is a major release requiring consumer sign-off. Every
consumer must be able to state which version of each capability it is on, and
the library must be able to list which consumers are on which version.

At least two capabilities must be simultaneously live in two major versions
during a migration window, in different consuming projects. A design that cannot
do this forces a flag-day upgrade across six teams.

### 7.2 Parameter ownership

Three ownership classes must be distinguishable, because different people change
them on different cadences under different approvals:

| Class | Example | Owner | Cadence | Approval |
|---|---|---|---|---|
| **Library-global** | Prime rate, tax tables, statutory fee caps | Credit Systems | As gazetted | Compliance |
| **Library-policy** | Expense norm variant, affordability buffer, obligation treatment matrix | Credit Risk Policy | Quarterly | Credit Committee |
| **Consumer-local** | Product minimum amount, campaign thresholds | Product team | Weekly | Product owner |

A consuming project must be able to set consumer-local values without being able
to set library-global ones, and the same capability used twice in one flow with
different consumer-local values must be possible — project 06 runs the same
pricing capability four times with four products' settings inside one flow.

### 7.3 Effective dating is not optional

Every assessment names a `decision_date`, and every versioned artefact — table,
scorecard, norm, cap, rate card, reason code — resolves against it. "Today" never
appears in credit logic. A 2029 replay of a 2026 decision, on 2026 inputs, must
reproduce the 2026 output to the cent.

### 7.4 Nulls are meaningful

Across the library, three null situations are distinct and must not be conflated:
a value that was not collected, a value that was collected as zero, and a value
that could not be established. "No bureau record" and "bureau record showing no
accounts" are different applicants. The library must express the difference, and
a consumer must not be able to lose it accidentally.

### 7.5 Testability

Every capability is testable standalone, without a pipeline, without a database,
and without constructing a full application. Credit Risk Policy must be able to
supply a spreadsheet of expected inputs and outputs for a policy-bearing
capability and have it run as a test.

### 7.6 Adjustments are layered, never merged

An adjustment set is a first-class artefact with its own version, its own
approval and its own lifetime, separate from the artefacts it modifies. Merging
an adjustment into a base table "to keep things simple" is prohibited, because it
destroys the distinction between what the model said and what policy decided,
which is the distinction the whole mechanism exists to preserve.

The library must be able to answer, for any date: which adjustments were in
force, over what scope, in what order, who approved each, when each expires, and
what the aggregate effect on the portfolio was. It must also be able to run a
flow with the adjustment stack disabled, which is how the base artefact's own
performance is monitored.

### 7.7 Evidence

Each capability declares what it contributes to the decision record: which
inputs it used, which table versions it resolved, which branches it took, which
reason codes it raised. The assembly of those contributions into a decision
record is project 09's job; producing them is the library's.

---

## 8. Tables owned by the library

| Table | Dimensions | Cells | Owner | Change cadence | Source |
|---|---|---|---|---|---|
| Expense norms (statutory) | 12 income bands × 6 dependant counts × 2 components | 144 | Compliance | On gazette, ~2 yearly | Regulator publication |
| Expense norms (internal) | same, × 8 products | 1 152 | Credit Risk Policy | Quarterly | Internal |
| Obligation treatment matrix | 45 account types × 6 attributes | 270 | Credit Risk Policy | Quarterly | Internal |
| Tax tables | 8 brackets × 4 rebate classes, effective-dated | ~50 per year | Credit Systems | Annual | Revenue authority |
| Statutory fee caps | 4 product classes × 3 components | 12 | Compliance | Annual | Regulator |
| Credit life premium rates | 14 age bands × 8 term bands × 6 employment types | 672 | Credit Risk Policy | Annual | Insurer |
| **Rate card, Flex Loan** | **96 amount bands × 55 terms × 12 grades** | **63 360** | Treasury | Monthly | Spreadsheet |
| Rate card, Drive Finance | 60 amount bands × 61 terms × 12 grades × 5 LTV bands | 219 600 | Treasury | Monthly | Spreadsheet |
| Rate card, revolving | 12 grades × 20 limit bands | 240 | Treasury | Monthly | Spreadsheet |
| Rate card, business | 40 amount bands × 55 terms × 12 grades × 4 security types | 105 600 | Treasury | Monthly | Spreadsheet |
| Risk grade boundaries | 8 products × 5 segments × 12 grades | 480 | Credit Risk Policy | Semi-annual | Model team |
| Appetite grid | 12 grades × 8 products × 6 segments × 5 values | 2 880 | Credit Committee | Quarterly | Internal |
| Scorecard definitions | 9 scorecards × ~45 characteristics × ~8 bins | ~3 200 | Model team | On model release | Model documentation |
| Reason code registry | 380 codes × 9 attributes | 3 420 | Compliance | Monthly | Internal |
| Sector risk table | 420 industry codes × 6 attributes | 2 520 | Credit Risk Policy | Annual | Internal |
| Adjustment register | 40–120 live overlays × 11 attributes | ~1 000 | Credit Risk Policy | Ad hoc, sometimes weekly | Internal |

Two properties are required of every table in this list:

1. **Cell-level attribution.** Given an outcome, it must be possible to say which
   cell of which version of which table was read.
2. **Diffability.** A new version of a rate card must produce a review artefact
   showing which cells changed, by how much, and what the aggregate impact is.
   "Treasury sent a new spreadsheet" is not a reviewable change.

---

## 9. Non-functional requirements

| Requirement | Value |
|---|---|
| Real-time capability latency budget | Whole-library contribution under 15 ms at p99 for a single application |
| Batch throughput | 14 M client records through the affordability and scoring capabilities within a 4-hour window |
| Determinism | Identical inputs, identical versions, identical `decision_date` ⇒ identical outputs, bit for bit |
| Cold start | No per-request compilation; first request after deployment is not materially slower than the thousandth |
| Table refresh | A rate card refresh must not require a code deployment |

---

## 10. Acceptance criteria

1. Six consuming projects use the library without forking any capability.
2. Every capability has a frozen, published interface and a test suite owned by
   the capability, not by a consumer.
3. Policy-owned tables are editable by their owners without an engineer, and
   every edit produces a reviewable diff.
4. A capability can be used twice in one flow with different settings.
5. Two major versions of a capability can be live simultaneously across
   different consumers.
6. Any historical decision can be re-derived to the cent with the artefacts in
   force at its `decision_date`.
7. A Credit Risk Policy analyst can read the generated description of a
   capability and confirm it matches policy, without reading code.
8. Any flow can be run with its adjustment stack disabled, producing the
   unadjusted answer alongside the adjusted one, without a separate
   implementation.
9. The adjustments in force on any past date are recoverable, with their scope,
   order, approval and expiry.

---

## 11. Change scenarios

Each of these has been requested of a real credit library. A good structure makes
them cheap.

1. **The regulator gazettes new expense norms** effective the first of next
   month. Decisions before that date must continue to use the old table forever.
2. **Treasury moves the Flex Loan rate card** from 96 amount bands to 120, and
   changes three terms' worth of cells mid-month after a repo rate change.
3. **A new product launches** — Drive Finance for used vehicles — needing the
   existing rate card capability with an extra key dimension, without disturbing
   the other seven products.
4. **Compliance adds 40 reason codes** and reclassifies 12 existing ones as
   regulatory-mandated rather than discretionary.
5. **The model team replaces one scorecard** with a version having 52
   characteristics instead of 45, and needs both live for a three-month parallel
   run in different segments.
6. **Credit Risk Policy wants the affordability buffer to vary by channel**,
   where today it varies only by product and grade.
7. **A consuming team needs the obligations capability to also return** the three
   most expensive accounts by instalment, which no other consumer wants.
8. **The tax table changes mid-year** rather than at year-end, for the first
   time.
9. **An internal audit finding** requires that every application record the
   version of every table it touched, where today only some are recorded.
10. **The bank acquires a book** with a fourth bureau format, and it must flow
    through the same normalisation.
11. **Observed defaults run 30% above prediction** on one scorecard for two
    consecutive quarters. Credit Risk Policy applies a probability-of-default
    multiplier for six months while the model is rebuilt, and must show the
    Credit Committee the approval rate impact before it goes live and the
    realised impact afterwards.
12. **A temporary tightening from three years ago** is discovered still in force.
    The library must be able to list every live adjustment, its age, its
    approval and its expiry, and to show what unwinding it would do.
13. **Two adjustments collide**: a segment-level score shift and a
    channel-level odds multiplier both apply to the same application, and the
    Credit Committee needs the net effect and the order in which they applied.

---

## 12. Out of scope

- Data acquisition: bureau calls, statement retrieval, document OCR.
- The customer-facing application journey.
- Contract generation, disbursement and servicing.
- Model development. The library consumes scorecards; it does not fit them.
- Storage of decision records. The library produces evidence; project 09
  consumes it.

---

## 13. Questions the implementation must answer

1. **What is a capability, structurally?** Twenty-one capabilities differ wildly
   in shape — some are arithmetic, some are table lookups, some are rule sets,
   some consume ragged collections. Is there one unit of reuse, or several?
2. **Where does a table live** — inside the capability that reads it, beside it,
   or in a shared table store? Who owns it when two capabilities read the same
   table?
3. **How does a consumer parameterise a shared capability** without the library
   having to know about the consumer, and without the consumer being able to
   reach settings it should not?
4. **How is an interface frozen**, and what exactly does freezing cover — names,
   types, nullability, semantics?
5. **How do two major versions coexist** in one deployed system?
6. **What does a capability that operates on one element of a ragged collection
   look like**, and how do its results roll up, when three consumers roll them up
   differently?
7. **Can `core.instalment` be run in reverse**, and if so is that the same unit
   or a second one?
8. **How is effective dating expressed** so it cannot be forgotten, given that
   forgetting it silently produces plausible wrong answers?
9. **What does the generated description of a capability look like** such that a
   policy analyst can sign it off? This is doc 04 §6's open risk, met first here
   because the library is where non-engineers arrive.
10. **What is the smallest set of core component kinds** that covers all
    twenty-two capabilities without a general-purpose escape hatch appearing in
    more than two of them?
11. **What is an adjustment, structurally?** It changes values, which makes it
    look like a parameter; it composes in a declared order, which makes it look
    like logic; it is separately approved and separately versioned, which makes
    it look like neither. Is an overlay a core component kind of its own?
12. **How does a flow run with adjustments disabled** without becoming a second
    implementation of itself?
