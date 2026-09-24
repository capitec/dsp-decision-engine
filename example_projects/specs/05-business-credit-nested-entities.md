# 05 — Business credit granting with nested entities

> Fictional. The Bank, its products, thresholds, table dimensions and volumetrics
> are invented for this repository. Regulatory mechanisms referred to are the
> published public ones; every number attached to them is illustrative.

---

## 1. What this is

A small or medium business applies for a **Business Term Facility**
(`product_code` 50) or a **Business Revolving Facility** (`product_code` 51). The
Bank must decide whether to lend, how much, over what term, at what price,
against what security, and on what conditions.

The difficulty is not the arithmetic. It is that a business is not one thing. It
is a registered entity, plus the people and companies standing behind it:
directors, members, trustees, shareholders above the disclosure threshold, the
ultimate beneficial owners those shareholdings resolve to, the sureties and
guarantors the Bank will require, and the other businesses under common control
whose exposure aggregates with this one. Each of those related entities carries
its own credit history, and inside that history is a variable-length list of
adverse events — judgments, listings, administration orders, tax notices —
each of which has to be classified on its own merits before any of them can be
summed into a view of the entity, and before those entity views can be summed
into a view of the business.

That is two levels of ragged nesting, both variable in length, with the outer
length between 1 and 40 and the inner length between 0 and 60. Everything the
Bank does downstream — the score, the grade, the appetite, the price, the
conditions — sits on top of that structure. And when the answer is no, the
business, the regulator and the Bank's own credit committee all want the same
thing: **which entity, and which event**.

A decline that can only say "adverse credit history" is not an answer. A decline
that says "entity 7, Ms M. Dlamini, 32% effective ownership and a required
surety, unsatisfied civil judgment of R184 000 dated 14 March 2025, classified
disqualifying under AE-C-03" is. Producing the second, through two levels of
nesting, on 900 applications a day and again on 180 000 existing clients every
month, is the project.

---

## 2. Why it is in this set

| Question | How this project stresses it |
|---|---|
| **Q5 core component set** | This is the project's centre of gravity. The flow's dominant shape is *a collection of records, each carrying a collection of records, each classified, each rolled up under rules that are neither worst-of nor average*. Scorecards, decision tables and decision trees do not express it. If a nesting-and-roll-up construct is a missing core component kind, this spec is where it becomes undeniable — and if it is not, this spec is where the general-purpose escape hatch will appear, and be counted. |
| **Q7 audit and debugging** | Attribution has to survive two levels. The business decline reason must name an entity and an event; the entity verdict must name the rule and the events that bound it; a roll-up rule that fires on a *count* attributes to a *set* of events, not one. A dispute over a single judgment must be answerable with a counterfactual: what the entity verdict, the business grade and the offer would have been without it. And because policy overlays sit on top of the models at four levels, the attribution must separate three causes that look identical from the outside: the entity's own data, the model's own answer, and an approved adjustment to it. |
| **Q1 reuse** | Six library capabilities are consumed inside the outer collection rather than once per application: `core.bureau`, `core.adverse_events`, `core.scorecard`, `core.calibration`, `core.risk_grade` and, for sole proprietors, `core.affordability` by way of [project 02](02-affordability-assessment.md). These were written for one applicant. Here they are applied to forty. Whether that is the same capability used many times, or a second thing, is the reuse question in its sharpest form. |
| **Q4 custom modules across flows** | The entity structure resolution, the adverse roll-up and the financial spread are project-specific and are used at least three times each: in granting, in the monthly re-assessment batch, and in [project 07](07-credit-limit-management.md)'s business limit review. They must be written once. The previous generation's answer to this was copy-paste, and the copies diverged. |
| **Q6 codebase organisation** | 420 sector codes with two tables hanging off them, two scorecard families with different characteristic sets, roughly 140 new reason codes, fourteen business disqualification rules, twelve entity disqualification rules, thirty-four event classification rules, twelve roll-up rules, ten people-blend rules and fifteen validation rules — owned by four different teams, changing on four different cadences. Organising that so a policy analyst can find the rule they want to change is the test. |
| **Q2 tables** (secondary) | Sector benchmarks, collateral advance rates, the surety cover table and the 105 600-cell business rate card. |
| **Q3 parameters** (secondary) | Amount thresholds that vary by entity role; coverage ratios; blend weights that vary by segment and by whether financials exist. |

### The parts already known to be hard

These are named here because a design that only discovers them during
implementation has discovered them too late.

| # | Difficulty |
|---|---|
| H1 | **Two levels of ragged nesting with variable fan-out.** 1..40 entities, each with 0..60 events. Median 4 and 6; worst observed 40 entities and 600 total events. The outer and inner lengths are independent. |
| H2 | **Roll-up semantics that are neither worst-of nor average.** Counts, aggregate amounts, recency, velocity and trend all bind, and they bind differently by entity role. |
| H3 | **Attribution through two levels.** A business outcome must resolve to (entity, event, rule). Count rules attribute to sets. Blend rules attribute to weights. |
| H4 | **The same person in two places.** One natural person may reach the applicant through two ownership paths, and may appear on two unrelated applications on the same day with different roles. |
| H5 | **Partial re-runs.** When one entity's bureau view refreshes, the Bank wants that entity re-evaluated and the business re-rolled-up without re-running everything — and the difference explained. |
| H6 | **The performance profile of variable fan-out.** A 3-second budget on an application whose entity count varies by a factor of forty, and an 8-hour batch over 180 000 of them. |
| H7 | **Overlays inside the nesting.** Post-model adjustments (`core.adjustments`, [00](00-shared-credit-core-library.md) §6.22) apply at four different levels of this flow, including on the event thresholds at the innermost level. An overlay applied to one entity propagates upward through two roll-ups, and the attribution must still separate what the entity's data did from what the model did from what policy decided. |

---

## 3. Actors

| Actor | Responsibility |
|---|---|
| **Business Credit Risk Policy** | Owns the disqualification rule sets, the event classification thresholds, the roll-up rules, the coverage requirements and the blend weights. Not engineers. Change quarterly, and out of cycle after a loss event. |
| **Business Credit Analysts** (≈70) | Run applications, request the missing document, apply qualitative overrides inside their authority, and are the first people to read an attribution and say it is wrong. |
| **Credit Committee** | Approves above the delegated limits, approves restricted sectors and unresolved structures, and reads the pack this flow produces. Meets twice weekly. |
| **Financial Crime / Compliance** | Owns the screening outcome feed, the match-confidence bands and the disposition rules for probable matches. Owns the reason-code wording, including what may lawfully be said to the business about a third party. |
| **Treasury / Pricing** | Owns the business rate card and the rate floor components. |
| **Sector Analytics** | Owns the 420-code sector table, its exclusion and appetite attributes, and the sector ratio benchmarks. Annual refresh, with mid-year patches after a sector shock. |
| **Model Team** | Owns the two scorecard families and their calibrations. |
| **Portfolio Management** | Owns group exposure caps, single-name concentration limits and sector portfolio caps. Runs the monthly re-assessment and consumes its output. |
| **Internal Audit and the regulator** | Ask, years later, why a specific business was declined and what a specific judgment contributed. |
| **The business, and the individual** | Two different claimants with two different entitlements: the business is entitled to the reason for its decline; the individual whose record caused it is entitled to their own record's reasons, and the business is not entitled to those. §9.2. |

---

## 4. Inputs

### 4.1 The applicant business

| Name | Type | Cardinality | Notes |
|---|---|---|---|
| `application_id` | int64 | 1 | |
| `client_id` | int64 | 0..1 | Null for a new-to-bank business. |
| `decision_date` | date | 1 | Governs every table version. Never "today". |
| `product_code` | int16 | 1 | 50 or 51. |
| `requested_amount` | float64 | 1 | R50 000 – R10 000 000. |
| `requested_term_months` | int16 | 0..1 | Null for product 51. |
| `registration_number` | string | 0..1 | Null for sole proprietors and unregistered partnerships. |
| `legal_form_code` | int8 | 1 | 1 sole proprietor, 2 partnership, 3 private company, 4 close corporation, 5 trust, 6 non-profit company, 7 co-operative, 8 personal liability company, 9 external company. |
| `registration_status_code` | int8 | 1 | Registered, in deregistration, deregistered, business rescue, provisional liquidation, final liquidation, converted, unknown. |
| `registration_date` | date | 0..1 | |
| `trading_since_date` | date | 0..1 | May precede registration. |
| `sector_code` | int16 | 1 | One of 420. Self-declared, then verified against transaction behaviour. |
| `jurisdiction_code` | int8 | 1 | |
| `tax_reference` | string | 0..1 | |
| `tax_compliance_status_code` | int8 | 1 | Compliant, non-compliant, unavailable, not registered. Freshness: must be ≤ 30 days old at `decision_date`. |
| `vat_registration_number` | string | 0..1 | |
| `declared_annual_turnover` | float64 | 1 | |
| `facility_purpose_code` | int8 | 1 | One of 24. |
| `security_offered` | list[record] | 0..12 | §4.5. |

### 4.2 The entity list — the outer collection

Between **1 and 40** entities after structure resolution (§5.1). Before
resolution, the disclosed structure is a graph of unbounded size.

| Field | Type | Null? | Notes |
|---|---|---|---|
| `entity_id` | int64 | no | Stable within the application. |
| `entity_key` | string | no | The identity this entity de-duplicates on: verified identity number for a natural person, registration number for a juristic one. |
| `is_natural_person` | bool | no | |
| `relationship_type_code` | int8 | no | §4.3. |
| `parent_entity_key` | string | yes | Null where the entity attaches directly to the applicant. |
| `path` | list[string] | no | The ownership route from the applicant to this entity. Length ≤ 3. |
| `depth` | int8 | no | 1..3. |
| `direct_ownership_pct` | float64 | yes | Of the immediate parent. Null for non-owning roles. |
| `effective_ownership_pct` | float64 | no | Derived: the product of `direct_ownership_pct` along `path`. |
| `is_controlling` | bool | no | Disclosed control, which may exist without ownership (a trust's founder, a shareholders'-agreement veto, a sole director). |
| `appointment_date` | date | yes | |
| `resignation_date` | date | yes | Non-null entities are retained and evaluated where resignation is within 12 months of `decision_date`. |
| `is_required_surety` | bool | no | Set by policy at §5.13, not by the applicant. |
| `residency_code` | int8 | yes | |
| `date_of_birth` | date | yes | Natural persons. |
| `deceased_flag` | bool | yes | Three-valued: confirmed alive, confirmed deceased, not established. |
| `identity_verification_code` | int8 | no | Verified, partially verified, failed, not attempted. |
| `screening_outcome_code` | int8 | no | Clear, possible, probable, confirmed, pending. |
| `screening_confidence` | float64 | yes | 0..100. Null where `clear`. |
| `screening_list_codes` | list[int8] | yes | Sanctions, domestic PEP, foreign PEP, adverse media, internal restricted. |

**Volumes**: median 4 entities, mean 6.2, p95 14, p99 27, maximum 40.
Distribution is heavily bimodal — sole proprietors have 1, family companies have
3 to 6, and anything involving a trust has 9 or more.

### 4.3 Entity kinds

| `relationship_type_code` | Kind | Owns? | Typically controls? |
|---|---|---|---|
| 1 | Director | sometimes | often |
| 2 | Member (close corporation) | yes | yes |
| 3 | Trustee | no | yes |
| 4 | Trust beneficiary | indirect | no |
| 5 | Shareholder at or above the 5% disclosure threshold | yes | sometimes |
| 6 | Ultimate beneficial owner | indirect | sometimes |
| 7 | Personal surety | no | — |
| 8 | Corporate guarantor | no | — |
| 9 | Group company under common control | no | — |
| 10 | Partner | yes | yes |
| 11 | Sole proprietor principal | yes | yes |
| 12 | Authorised signatory without ownership | no | no |

The disclosure threshold of 5% and the beneficial-ownership concept follow the
public company-registry and financial-intelligence mechanisms; the threshold
value used here is illustrative and is a parameter (§6.1).

### 4.4 Adverse events — the inner collection

Between **0 and 60** per entity. Median 6, p95 40, p99 52. The worst observed
application carried 600 events across 40 entities, with 187 of them on one
entity.

| Field | Type | Null? | Notes |
|---|---|---|---|
| `event_id` | int64 | no | |
| `entity_key` | string | no | The entity it attaches to, not `entity_id` — an event follows the person, not the path. |
| `event_type_code` | int8 | no | §5.5, fourteen types. |
| `amount` | float64 | yes | Null is meaningful: a judgment with no recorded amount is not a judgment for R0. |
| `event_date` | date | no | |
| `status_code` | int8 | no | Active, paid, rescinded, under appeal, abandoned, unknown. |
| `is_disputed` | bool | no | Dispute lodged and unresolved. |
| `is_satisfied` | bool | no | |
| `satisfaction_date` | date | yes | Required non-null where `is_satisfied`; a satisfied event with no date is a data-quality failure, not an old event. |
| `source_code` | int8 | no | Bureau A / B / C, internal, court roll, revenue authority, client-declared. |
| `duplicate_group_id` | int64 | yes | Set where the same underlying event is reported by two bureaux. |

### 4.5 Financial information

| Source | Shape | Cardinality | Freshness |
|---|---|---|---|
| Annual financial statements | ~48 standard lines after mapping | 1..3 periods, ragged; periods may be of unequal length | Most recent year-end ≤ 21 months before `decision_date` |
| Management accounts | same 48 lines, reduced completeness | 0..1 | ≤ 3 months |
| Bank account turnover analysis | monthly credit turnover, transfers excluded | 6..24 months | ≤ 1 month |
| Debtors and creditors ageing | 5 buckets each | 0..1 | ≤ 1 month |
| Audit level | audited / independently reviewed / compiled / client-prepared / none | 1 per period | — |
| Audit opinion | unqualified / qualified / adverse / disclaimer / n/a | 1 per audited period | — |
| Security offered | collateral class, declared value, valuation date, valuer, existing encumbrances | 0..12 | Valuation ≤ 24 months for property, ≤ 12 months for plant |

Roughly 38% of applications arrive with no financial statements at all and must
be assessed on bank turnover. 22% arrive with one period only. Unequal period
lengths — a nine-month first period after incorporation — occur on about 6% and
must be annualised with a flag, not silently compared against twelve-month
benchmarks.

### 4.6 Internal and external context

- Existing Bank facilities held by the applicant, and by every entity and every
  business sharing an entity with it (`core.exposure`).
- 24 months of internal conduct on the applicant's transactional account, where
  one exists: average balance, excess days, returned debits, deposit
  concentration.
- Bureau views: a commercial bureau view per juristic entity and a consumer
  bureau view per natural-person entity, both normalised through `core.bureau`.
  Freshness tolerance 30 days at `decision_date`; older is `bureau_is_stale` and
  forces a refresh or a referral.
- Screening outcomes from Financial Crime, per entity, with confidence.
- The sector tables, the rate card, the appetite grid and the reason registry
  from [00](00-shared-credit-core-library.md).
- **The adjustment set in force at `decision_date`** — `adjustment_set_id` and
  the overlays it contains, from `core.adjustments`. Between 6 and 25 of the
  Bank's live overlays are typically in scope for this flow at any time. This is
  an input to the assessment, not a property of the deployment: a replay resolves
  the set that was in force then, not the set in force now.

---

## 5. The flow

Thirteen stages. Stages 5.5 to 5.8 are the nested core and are the reason this
document exists.

### 5.1 Stage — application intake and structure resolution

**Preconditions**: an application record, a disclosed structure, and a
`decision_date`.

**Determines**: the bounded, de-duplicated entity list that every later stage
operates on.

The disclosed structure is a **graph**, not a list and not a tree. A company is
held by a trust and a holding company; the holding company is held by the same
two people who are trustees of the trust; one of those people also holds 6%
directly. Cross-holdings and cycles occur, most often where two group companies
hold shares in each other.

The Bank collapses that graph to a bounded structure under declared rules:

| Rule | Value |
|---|---|
| Maximum expansion depth | **3 levels** below the applicant |
| Maximum resolved entity count | **40** |
| Expansion materiality floor | a holding is expanded only where `effective_ownership_pct` ≥ **5.0** |
| Effective ownership | the product of direct percentages along the path, to 4 decimal places |
| Cycle handling | a path that revisits an `entity_key` already on that path terminates at the revisit and records `cycle_truncated` |
| Non-owning roles | directors, trustees and signatories are attached at their own level and are not expanded through |

**When the bound is exceeded.** Depth exceeded, or resolved count above 40, or
disclosed ownership resolving to less than 90% of issued equity, does **not**
decline the application. It produces `structure_unresolved` with the specific
cause, suspends the automated outcome, and routes to credit committee with the
partial structure and the unexpanded remainder listed. An application the Bank
cannot see through is a referral, not a rejection — but it may never be an
approval.

**De-duplication.** The same natural person reached by two paths is **one
entity** with:

- ownership = the **sum** of effective ownership across all paths;
- role = the **most senior** role held, under the ordering
  11 > 2 > 10 > 3 > 1 > 5 > 6 > 7 > 8 > 4 > 12;
- control = the **disjunction** of the control flags on every path;
- adverse events attached **once**, never once per path;
- **every** path retained and recorded, because the committee pack must show how
  the person reaches the business.

Failing to de-duplicate double-counts a person's judgments and double-counts
their ownership; both have happened, and the second produced an ownership total
of 143%.

**Emits**: the resolved entity list; per entity its path set, effective
ownership, control flag and criticality class (§5.4); ownership reconciliation
totals; `structure_unresolved` and `cycle_truncated` flags with causes.

**Records**: the disclosed graph as received, the resolved list, and the
transformation between them — which entities were dropped below the materiality
floor, which paths were truncated, which keys were merged.

### 5.2 Stage — regulatory regime determination

**Determines**: whether this facility is a regulated credit agreement, which
changes which rules apply for the remainder of the flow.

The consumer credit statute covers juristic persons below a size threshold and
covers natural persons without qualification. So the same product, to two
different applicants, sits in two different regimes:

| Condition | Regime |
|---|---|
| Sole proprietor or partnership of natural persons | **Regulated**, without size test |
| Juristic person, annual turnover **and** asset value both below **R1 000 000**, and facility ≤ **R250 000** | **Regulated** |
| Juristic person, turnover or asset value at or above R1 000 000 | **Unregulated** |
| Facility above R250 000 to a small juristic person | **Unregulated** (large agreement exemption) |

Consequences of the regulated regime, all of which change later stages:

- statutory fee caps apply through `core.fees` rather than the Bank's business
  fee schedule;
- a statutory affordability assessment is required on the natural person behind
  the facility, through [project 02](02-affordability-assessment.md), in addition
  to the Bank's debt service coverage test;
- the statutory decline-reason disclosure obligations apply, which changes what
  §9.2 must produce;
- credit life cover rules apply where a surety is a natural person.

**Records**: the regime, the test values that determined it, and — where the
regime is borderline, within 10% of the thresholds — a flag, because turnover
restatement later can move an agreement across the line.

### 5.3 Stage — business-level absolute rules of disqualification

**Preconditions**: the resolved structure and the business data.

**Determines**: whether the business itself disqualifies.

Fourteen rules. Every one is evaluated even after the first fails — the outcome
short-circuits the *remainder of the flow*, not the *rule set*. A business that
fails four of these must be told all four, because fixing one and reapplying to
be declined on the next is the failure mode this requirement exists to prevent.

| Rule | Test | Outcome | Reason |
|---|---|---|---|
| B-AROD-01 | `registration_status_code` in {deregistered, final liquidation} | Decline | 5101 |
| B-AROD-02 | in deregistration, or annual returns outstanding > 24 months | Decline, curable | 5102 |
| B-AROD-03 | business rescue in force | Refer to committee; approval only with a practitioner-endorsed plan and post-commencement ranking | 5103 |
| B-AROD-04 | provisional liquidation | Decline | 5104 |
| B-AROD-05 | months trading at `decision_date` < 12 | Decline | 5105 |
| B-AROD-06 | months trading 12..23 | Permitted only where grade ≤ 6 **and** personal surety cover ≥ 100% of the facility; otherwise refer | 5106 |
| B-AROD-07 | `sector_code` carries `exclusion_flag` (31 of 420 codes) | Decline | 5107 |
| B-AROD-08 | `sector_code` carries `restricted_flag` (46 of 420 codes) | Refer to committee | 5108 |
| B-AROD-09 | `tax_compliance_status_code` = non-compliant | Decline, curable | 5109 |
| B-AROD-10 | tax status unavailable or older than 30 days | Condition precedent, and refer above R2 000 000 | 5110 |
| B-AROD-11 | trailing 12-month turnover ≥ R1 000 000 and no VAT registration | Decline, curable | 5111 |
| B-AROD-12 | screening outcome at business level = confirmed; or probable with confidence ≥ 80 | Decline (confirmed) / refer (probable) | 5112 / 5113 |
| B-AROD-13 | prior write-off with the Bank within 120 months, unrecovered | Decline | 5114 |
| B-AROD-14 | prior write-off subsequently settled in full | Refer to committee | 5115 |
| B-AROD-15 | audit opinion adverse or disclaimer on the most recent audited period | Decline | 5116 |
| B-AROD-16 | audit opinion qualified | EBITDA haircut 10% and refer above R2 000 000 | 5117 |
| B-AROD-17 | `jurisdiction_code` non-domestic | Refer to committee | 5118 |
| B-AROD-18 | trailing 12-month turnover below the product minimum — R1 000 000 (product 50), R500 000 (product 51) | Decline | 5119 |
| B-AROD-19 | `facility_purpose_code` on the excluded-purpose list (7 of 24) | Decline | 5120 |
| B-AROD-20 | existing Bank facility in legal collection or ≥ 90 days in arrears | Decline | 5121 |

(The table names twenty rules; the rule family is referred to as fourteen
*classes* of disqualification in policy documents, and the mapping between
policy classes and executable rules is itself something the implementation must
keep visible. Policy writes classes; the flow evaluates rules.)

**Emits**: `business_arod_verdict` (clear / refer / decline), the complete set of
fired reason codes ranked through `core.reason_codes`, and the curability flag
per fired reason.

**Records**: every rule's inputs and outcome, including the rules that passed —
a later dispute about B-AROD-11 needs the turnover figure that was used.

### 5.4 Stage — entity criticality and entity-level disqualification

**Preconditions**: the resolved entity list, screening outcomes, bureau views.

**Determines**: for each entity, whether it disqualifies itself, and — the point
of the stage — **what that means for the business**.

An entity failing a disqualification rule does not automatically fail the
business. A 3% shareholder who is under debt review is not the same event as the
sole director and 100% owner being under debt review. The dependency is
explicit, through a criticality class computed first:

| Class | Definition |
|---|---|
| **Critical** | `is_controlling`, or `effective_ownership_pct` ≥ 25.0, or `is_required_surety`, or the sole director / sole trustee / sole member, or a corporate guarantor providing more than 20% of required cover |
| **Significant** | `effective_ownership_pct` ≥ 10.0 and < 25.0, or a director or trustee without control |
| **Peripheral** | everything else: ownership below 10%, no control, not a surety, not a director |

Twelve rules, each with three outcomes by class. The disposition matrix is 12 × 3
= 36 cells and is a policy artefact.

| Rule | Test | Critical | Significant | Peripheral |
|---|---|---|---|---|
| E-AROD-01 | identity verification failed | Business decline (5201) | Business refer | Exclude entity, condition precedent |
| E-AROD-02 | screening confirmed (confidence ≥ 95) | Business decline (5202) | Business decline | Business decline |
| E-AROD-03 | screening probable (80–94) | Business refer, mandatory committee | Business refer | Record, committee note |
| E-AROD-04 | screening possible (60–79) | Business refer | Record | Record |
| E-AROD-05 | deceased confirmed | Business decline unless a substitution is in place (5203) | Exclude entity, condition precedent | Exclude entity |
| E-AROD-06 | unrehabilitated insolvency or sequestration | Business decline (5204) | Business decline | Material adverse event |
| E-AROD-07 | disqualified from directorship, and acting as a director | Business decline (5205) | Business decline | n/a |
| E-AROD-08 | under debt review, not cleared | Business decline (5206) | Material adverse event, refer | Material adverse event |
| E-AROD-09 | under 18, or lacking contractual capacity | Business decline if surety; else exclude from surety and from scoring | Exclude from surety | Record |
| E-AROD-10 | age at final instalment date > 75 and is a surety | Require a co-surety; surety cover halved | n/a | n/a |
| E-AROD-11 | non-resident | Refer to committee; surety cover not counted | Refer | Record |
| E-AROD-12 | named on a Bank write-off, any capacity, within 120 months | Business decline (5207) | Business decline | Material adverse event |
| E-AROD-13 | juristic entity deregistered or in liquidation, and is a guarantor | Guarantee not counted; refer | Guarantee not counted | Record |
| E-AROD-14 | confirmed fraud marker | Business decline (5208) | Business decline | Business decline |

Three rules — E-AROD-02, E-AROD-12 and E-AROD-14 — decline regardless of class.
That asymmetry is deliberate and must be readable as deliberate, not as an
oversight in a matrix.

Where the disposition is "material adverse event", the finding does not decline
here; it is injected into that entity's event list at §5.5 with a synthetic
event so that it participates in the count-based and amount-based roll-up rules
rather than sitting outside them. Synthetic events are marked as such.

**Emits**: per entity, a disqualification verdict, the fired rules, the
criticality class and the class's inputs; at business level, the aggregated
disposition and the reason set.

**Records**: the criticality class **and the values that produced it**, because
"critical because 25.4% effective ownership across two paths" is a different
audit answer from "critical because sole director", and both must be
reproducible against a structure that may since have changed.

### 5.5 Stage — adverse event classification

**Preconditions**: the entity list with its attached events, de-duplicated
across bureaux by `duplicate_group_id`.

**Determines**: one severity for each event, independently.

This is the innermost work: it runs on every event of every entity, 0..60 by
1..40, worst case 600 per application, worst case ~4 300 000 per monthly batch.
It consumes `core.adverse_events` but cannot consume it unchanged — the library
capability classifies on type, amount, date and satisfaction, and here the
amount thresholds **vary by the criticality class of the entity the event sits
on**. The same R80 000 unsatisfied judgment is material on a peripheral
shareholder and disqualifying on a controlling one. Whether that is the library
capability parameterised, or a project capability that wraps it, is question 3
in §13.

**The fourteen event types**

| Code | Type | Amount meaningful? | Ages out? |
|---|---|---|---|
| 1 | Civil judgment | yes | yes, slowly |
| 2 | Default listing | yes | yes |
| 3 | Administration order | no | yes |
| 4 | Debt review flag | no | on clearance |
| 5 | Tax non-compliance notice | yes | yes |
| 6 | Dishonoured payment | no | yes, quickly |
| 7 | Litigation or summons served | yes | on resolution |
| 8 | Prior Bank write-off | yes | **no** |
| 9 | Sequestration or liquidation application | yes | on rehabilitation |
| 10 | Trace alert | no | yes |
| 11 | Rehabilitation order | n/a | mitigating, not adverse |
| 12 | Municipal or utility default | yes | yes |
| 13 | Rental default listing | yes | yes |
| 14 | Confirmed fraud marker | no | **no** |

**Amount thresholds by criticality class** (the material / disqualifying pair,
in rand):

| Event type | Critical | Significant | Peripheral |
|---|---|---|---|
| Civil judgment | 10 000 / 50 000 | 25 000 / 100 000 | 50 000 / 250 000 |
| Default listing | 5 000 / 40 000 | 7 500 / 75 000 | 15 000 / 150 000 |
| Tax non-compliance | 25 000 / 150 000 | 50 000 / 300 000 | 100 000 / 500 000 |
| Litigation | 250 000 / 1 000 000 | 500 000 / 2 000 000 | 750 000 / 3 000 000 |
| Municipal / rental | 10 000 / 60 000 | 20 000 / 120 000 | 40 000 / 200 000 |

Litigation additionally tests against turnover: an amount above 25% of trailing
turnover is disqualifying irrespective of the rand thresholds.

**These thresholds are overlay targets.** The commonest policy intervention in
business credit is not a model change; it is "halve the judgment materiality
threshold for construction for two quarters". That arrives through
`core.adjustments` as a scoped overlay on this table — position 1 of the declared
stack (§5.10) — and it is applied **before** any event is classified, because it
changes the classification itself rather than a value derived from it.

Three requirements follow, and the third is the awkward one:

1. The base threshold table is **not edited**. The overlay is layered over it,
   and both values are visible: the base R50 000 and the overlaid R25 000.
2. Every classified event records **which threshold value was used, and whether
   it was the base or an overlaid one**, with the overlay's identifier.
3. **The chain stays attributable.** An overlaid threshold can reclassify an
   event from material to disqualifying, which changes the entity verdict at
   §5.6, which can decline the business at §5.8 PP-06. The decline reason must
   therefore read as a complete chain — *entity 7, event 41, classified
   disqualifying under AE-C-03 against a threshold of R25 000, which is the base
   threshold of R50 000 halved by overlay ADJ-05-014 (sector 412, approved by
   Credit Committee on 2026-02-11, expiring 2026-08-31)* — and not merely as
   "disqualifying event". An overlay that can decline a business without saying
   so is the failure mode this requirement exists to prevent.

**Classification rules** — thirty-four, of which the load-bearing ones are:

| Rule | Condition | Severity |
|---|---|---|
| AE-C-01 | Judgment, satisfied, `satisfaction_date` more than 24 months before `decision_date` | Immaterial |
| AE-C-02 | Judgment, satisfied, satisfaction within 24 months, amount ≤ material threshold | Minor |
| AE-C-03 | Judgment, unsatisfied, amount > disqualifying threshold | **Disqualifying** |
| AE-C-04 | Judgment, unsatisfied, amount between the thresholds, event age ≤ 36 months | Material |
| AE-C-05 | Judgment, unsatisfied, amount between the thresholds, age 36–60 months | Minor |
| AE-C-06 | Judgment, unsatisfied, age > 60 months, amount ≤ disqualifying threshold | Minor — an unsatisfied judgment never reaches immaterial |
| AE-C-07 | Judgment with null amount, unsatisfied | Material, and `amount_unknown` flag |
| AE-C-08 | Default listing active, amount ≥ material threshold | Material |
| AE-C-09 | Default listing paid, cleared ≥ 12 months | Immaterial |
| AE-C-10 | Administration order in force | Disqualifying (critical, significant) / Material (peripheral) |
| AE-C-11 | Administration order rescinded ≥ 12 months | Minor |
| AE-C-12 | Debt review active | Disqualifying (critical) / Material (other) |
| AE-C-13 | Debt review, clearance certificate ≥ 6 months old | Immaterial |
| AE-C-14 | Tax notice open | Material, escalating to disqualifying above threshold |
| AE-C-15 | Tax notice resolved < 6 months | Minor |
| AE-C-16 | Dishonoured payment, single occurrence | Immaterial (count handles the rest, at §5.6) |
| AE-C-17 | Litigation, summons served, defended with a filed plea | one severity class lower than the amount implies |
| AE-C-18 | Prior Bank write-off, any age, any amount | Disqualifying (critical, significant) / Material (peripheral) |
| AE-C-19 | Sequestration application withdrawn | Minor |
| AE-C-20 | Confirmed fraud marker | Disqualifying, all classes |
| AE-C-21 | Rehabilitation order present | Downgrades the linked sequestration event by two classes; never below minor within 12 months of the order |
| AE-C-22 | **Any event with `is_disputed`** | Classified **one class lower**, never below minor, flagged `classification_provisional` |

AE-C-22 carries a further consequence that reaches all the way to the outcome: a
**disputed event may never be the sole cause of a decline**. Where removing every
provisional classification would change the outcome from decline to anything
else, the outcome becomes a referral with the dispute identified. The Bank does
not decline a business on a record the individual is formally contesting.

**Ageing.** Severity decay is a table, not an expression: event type × age band
(0–6, 6–12, 12–24, 24–36, 36–60, 60+ months) × a decay in classes, with a floor
per type. 14 × 6 = 84 cells. Types 8 and 14 have a floor equal to their initial
severity and never decay.

**Emits**, per event: `event_severity_code`, the rule that classified it, the
threshold values used **and their unoverlaid counterparts**, the class of the
entity that supplied those thresholds, any overlay identifiers that altered them,
`event_age_months`, and the provisional flag.

**Records**: all of the above, for every event, including the immaterial ones. An
event classified immaterial is still an event a committee will ask about.

### 5.6 Stage — entity adverse verdict

**Preconditions**: every event on the entity classified.

**Determines**: one verdict per entity — clear / minor / material /
disqualifying — **and the events that justify it**.

This roll-up is explicitly **not** worst-of and explicitly not an average.
Twelve rules apply; where several fire, the most severe verdict wins, but *every*
fired rule is recorded because the attribution differs by rule.

| Rule | Condition | Verdict |
|---|---|---|
| AE-R-01 | Any event disqualifying | Disqualifying |
| AE-R-02 | ≥ 3 minor events with `event_date` within 12 months of `decision_date` | Material |
| AE-R-03 | ≥ 5 minor events of any age | Material |
| AE-R-04 | ≥ 2 material events within 24 months, entity critical | Disqualifying |
| AE-R-05 | ≥ 2 material events within 24 months, entity significant or peripheral | Material, and the business outcome is capped at refer |
| AE-R-06 | Aggregate unsatisfied amount > max(R150 000, 15% of `requested_amount`) | Material |
| AE-R-07 | Aggregate unsatisfied amount > max(R500 000, 50% of `requested_amount`) | Disqualifying |
| AE-R-08 | Any material event within 6 months | Verdict floored at material; `recent_adverse` set, which blocks entity grades 1–3 |
| AE-R-09 | ≥ 8 dishonoured payments within 12 months | Material |
| AE-R-10 | Count in the trailing 12 months exceeds the count in the preceding 12 months by ≥ 3 | Escalate the otherwise-computed verdict by one class |
| AE-R-11 | All events immaterial and count ≤ 2 | Clear |
| AE-R-12 | Entity peripheral | Verdict capped at material, **except** where AE-C-18 or AE-C-20 fired |

Three properties are required of this roll-up and are the reason it cannot be
expressed as worst-of:

1. **Counts create severity that no individual event has.** Three minor events
   produce a material verdict. Attribution therefore names three events, not one.
2. **Aggregates create severity that no individual event has.** Six unsatisfied
   judgments of R30 000 each are a R180 000 problem and none of them is
   individually material.
3. **Trend and recency modify what counts and aggregates would give.** The same
   six judgments spread over five years and concentrated in the last eight months
   are different entities.

**Attribution is mandatory and structured.** Each entity verdict carries:

- the binding rule identifier;
- the **set** of `event_id` values that satisfied it — one for AE-R-01, three or
  more for AE-R-02, the full contributing set for AE-R-06;
- the computed quantity where the rule is quantitative (the count, the aggregate,
  the trailing-window delta);
- the next-most-severe rule that fired, and its events, because a committee
  routinely asks "and if that one were removed?".

**Emits**: `entity_adverse_verdict`, the binding rule, the attributing event set,
the provisional flag where any contributing event is disputed, and a per-entity
adverse summary (counts by severity, aggregate amounts, worst event, most recent
event).

Where any contributing event was classified against an overlaid threshold, the
verdict additionally carries `verdict_overlay_sensitive` and **the verdict that
the unoverlaid thresholds would have produced**. A verdict of material that would
have been minor without an overlay, and a verdict of material that would have
been material anyway, are different facts, and the committee that reads them
treats them differently.

### 5.7 Stage — entity scoring

**Preconditions**: entities not excluded at §5.4; bureau views normalised.

**Determines**: a score, a probability of default and a grade per entity.

Two scorecard families, applied by entity nature:

| | Natural persons | Juristic entities |
|---|---|---|
| Scorecard | `BUS-PERS-01` | `BUS-COMM-01` |
| Characteristics | 38 | 29 |
| Source | consumer bureau via `core.bureau`, plus internal conduct where the person banks with the Bank | commercial bureau, trade payment index, registry filings, sector, age of entity, internal conduct |
| Distinctive characteristics | worst arrears in 24 months, enquiry velocity over 3/6/12 months, revolving utilisation, number of active accounts, months since worst delinquency, age, residential stability | trade payment index, days beyond terms, supplier count, filing punctuality, sector default rate, years registered, directors' average tenure |
| Calibration segment | `person-behind-business` | `commercial-sme` |
| Thin-file variant | `BUS-PERS-01T`, 17 characteristics | `BUS-COMM-01T`, 12 characteristics |

Both run through `core.scorecard` → `core.calibration` → `core.risk_grade`, and
both must emit per-characteristic point contributions, because the entity's
contribution to a business decline has to be explicable at characteristic level
and not only at grade level.

**Thin file and no hit** — four distinct situations, which must not collapse into
one:

| Situation | Definition | Treatment |
|---|---|---|
| **Scored** | ≥ 3 accounts and ≥ 12 months of history | Full scorecard |
| **Thin file** | 1–2 accounts, or < 12 months of history | Thin-file variant; grade floored at 7 (cannot be better); counts toward coverage at half weight |
| **No hit** | bureau returns a valid response with no record | No score; grade assigned from a fallback table on role × age × tenure; does **not** count toward coverage |
| **No enquiry possible** | consent absent, identity unverified, or the bureau errored | No score, no grade, `entity_unscoreable`; does not count toward coverage; above 10% ownership this forces a referral |

The difference between "no hit" and "no enquiry possible" is exactly the
distinction `00` §7.4 requires and is exactly the one that gets lost. A juristic
entity registered less than 12 months before `decision_date` is floored at
grade 8 regardless of its commercial bureau result.

**Overlays at entity level.** Both scorecard families carry their own post-model
adjustments, and they are not the same adjustments. The personal scorecard is
shared in substance with the Bank's consumer lending and drifts with the consumer
cycle; the commercial scorecard drifts with the business cycle and with sector
shocks. So a typical live set contains, for instance, a −12 point score shift on
`BUS-PERS-01` for the new-to-bank segment, a PD multiplier of 1.25 on
`BUS-COMM-01` for entities in the construction sector grouping, and a grade
boundary shift moving the 6/7 boundary for juristic entities only.

The requirements are those of `core.adjustments` §6.22, applied **per entity**
rather than per application, which is where this project is unusual:

- the scorecard and calibration artefacts are untouched; the overlay is layered;
- each entity carries `score_unadjusted` and
  `probability_of_default_unadjusted` **alongside** the adjusted values, and the
  grade it would have received unadjusted;
- `adjustments_applied` is recorded **per entity**, in application order, with
  the effect of each — so an application with fourteen entities may carry
  fourteen different overlay lists, because scope depends on the entity's nature,
  sector and segment, not on the application's;
- applying a `BUS-PERS-01`-scoped overlay to a juristic entity is an **error**,
  not a silent no-op, and the flow must fail loudly rather than quietly scoring
  an entity with the wrong stack.

**Emits**: per entity — `scorecard_id`, `score`, `score_unadjusted`,
per-characteristic contributions, `probability_of_default`,
`probability_of_default_unadjusted`, `risk_grade`, the unadjusted grade,
`adjustments_applied`, the floor applied if any, the scoring situation code, and
the top three negative contributions as reason codes.

### 5.8 Stage — the people component

**Preconditions**: entity verdicts (§5.6) and entity grades (§5.7).

**Determines**: one probability of default and one grade representing everybody
standing behind the business, plus the caps they impose.

This is the stage with the most rules per line of output, and the one policy
changes most often. Ten rules, in a declared order of application.

**PP-01 — inclusion.** An entity enters the blend if it owns ≥ 5.0% effective,
or holds control, or is a director, member, trustee or partner. Sureties and
guarantors are excluded from the blend and handled by PP-08. Entities excluded at
§5.4 are out of the blend but remain in the record.

**PP-02 — coverage.** The scoreable share of ownership is
Σ(effective ownership of included, scoreable owner-entities) ÷ Σ(effective
ownership of all owner-entities). It must be ≥ **75%**, and for sole proprietors
and single-owner companies the single owner must be scoreable. Below 75%: refer,
reason 5401, `insufficient_people_coverage`. Thin-file entities count at half
their ownership toward the numerator.

**PP-03 — base weights.** Owner-entity weight = effective ownership. Non-owning
controllers (directors, trustees) receive a notional 10 percentage points each,
collectively capped at 30 points, added before normalisation. Weights are then
normalised to sum to 1.

**PP-04 — control weighting.** Where any single entity holds control or ≥ 50%
effective ownership, its weight is set to `max(effective ownership, 0.60)` and
the remaining weights are renormalised across the rest. A business run by one
person is that person's risk, whatever the shareholder register says.

**PP-05 — blend.** The blend is over **probabilities of default on the log-odds
scale**, not over grades. Averaging grade numbers is arithmetically wrong because
grades are not linear in risk, and doing it was a real defect in the previous
generation. The blended PD is converted back to a grade through
`core.risk_grade` with the `sme-people` segment.

**PP-06 — worst-of overrides.** Applied after the blend, and they dominate it:

| Condition | Effect |
|---|---|
| Any included entity with `entity_adverse_verdict` = disqualifying | **Business decline**, reason 5402, attributed through §5.6 to the entity and events |
| Any included entity with grade 11 or 12 and effective ownership ≥ 20% | People grade capped at 10 |
| Any included entity with grade ≥ 9 holding control | People grade capped at that entity's grade |
| Two or more included entities at grade ≥ 9 together holding ≥ 40% | People grade worsened by 2 |
| `recent_adverse` on any critical entity | People grade may not be better than 5 |

**PP-07 — materiality exclusion is not an amnesty.** An entity excluded from the
blend by PP-01 is still subject to §5.4 and §5.6. A 2% shareholder with a
confirmed fraud marker still declines the business. Exclusion is from the
*average*, not from the *rules*.

**PP-08 — sureties.** A surety's grade does not contribute to the people PD. It
determines how much cover that surety provides, which caps the facility:

| Surety grade | Cover multiple of assessed net worth | Absolute ceiling |
|---|---|---|
| 1–3 | 1.00 | R10 000 000 |
| 4–6 | 0.75 | R5 000 000 |
| 7–8 | 0.50 | R2 000 000 |
| 9 | 0.25 | R750 000 |
| 10–12 | 0.00 | R0 |

Multiple sureties: cover is the **sum** of individual covers, capped at 1.5× the
largest single cover, because correlated sureties from one household are not
independent. A surety who is also an owner contributes to the blend as an owner
*and* provides cover as a surety, and the two must not be conflated in the
record.

**PP-09 — corporate guarantors.** Cover = min(the guaranteed amount, the
guarantor's own unused appetite at its own grade under `core.appetite`, 40% of
the guarantor's tangible net worth). A guarantor is scored as a juristic entity
in its own right at §5.7, including its own adverse events.

**PP-10 — minimum composition.** At least one scoreable natural-person entity is
required for micro and small segments. Zero scoreable natural persons is a
referral, reason 5403, regardless of how good the commercial view is.

**PP-11 — overlays propagate, and must remain separable.** The blend consumes
entity PDs that have already been overlaid at §5.7. It therefore produces a
people PD that carries policy adjustments made three levels down, on a subset of
the entities, with different overlays on different entities. Three outputs are
required, not one:

| Output | Meaning |
|---|---|
| `people_pd` | The blend of adjusted entity PDs. The operative figure. |
| `people_pd_unadjusted` | The same blend over `probability_of_default_unadjusted`, with identical weights. What the models alone said. |
| `people_pd_overlay_contribution` | The difference, decomposed **by entity and by overlay**: which entity's overlay moved the people PD, by how much, at what weight. |

The decomposition matters because the question a committee actually asks is
*"why is this business a grade 7 this quarter when it was a grade 6 last
quarter?"*, and the answer is one of exactly three things, which must not be
confusable:

1. **the entity's own data changed** — a new judgment, a changed ownership
   percentage, a refreshed bureau view;
2. **the model changed** — a scorecard release between the two assessments;
3. **an overlay changed** — a sector multiplier that came into force, or one that
   expired.

An implementation that can only report the blended result reports all three as
the same event, and the Bank then rebuilds a model to solve a problem that was a
policy overlay reaching its review date.

No overlay may be applied at this stage that has already been applied at §5.7 to
the same entity. A sector PD multiplier scoped to juristic entities and a sector
PD multiplier scoped to the business are two different overlays with two
different scopes, and stacking both over a group company that is *also* in that
sector double-counts. The declared stack order (§5.10) and the declared scopes
exist to make that detectable rather than discovered in a portfolio review.

**Emits**: `people_pd`, `people_pd_unadjusted`, `people_grade`, the unadjusted
people grade, the overlay contribution decomposition, the weight applied to every
included entity, the binding cap rule where a cap applied, total surety cover,
total guarantee cover, the coverage ratio, and the excluded-entity list with
reasons.

**Records**: the full weight vector. "Why did this business grade move from 6 to
7 when nothing changed?" is answered by the weight vector and the overlay
decomposition together, and by nothing else.

### 5.9 Stage — business financial assessment

**Preconditions**: at least one of financial statements, management accounts, or
6+ months of bank turnover.

**Determines**: a financial component PD, a confidence level, and the cash-flow
figures the pricing stage tests against.

**Spreading.** Submitted statements are mapped to a standard form of 48 lines
through a mapping table of roughly 600 source labels. Unmapped labels above 2% of
turnover force a manual mapping and a referral; below 2% they aggregate into a
residual line. 1..3 periods, ragged, most recent first. Periods shorter than 10
or longer than 14 months are annualised and flagged `period_annualised`.

**Derived measures** — eleven, each with a defined formula and a defined
treatment when its inputs are absent:

| Measure | Definition |
|---|---|
| Turnover | Revenue, net of indirect tax |
| Turnover trend | Period-on-period growth; CAGR where 3 periods exist; single-period businesses get `trend_unavailable`, not zero |
| Gross margin | Gross profit ÷ turnover |
| EBITDA | Operating profit + depreciation + amortisation, adjusted for declared non-recurring items above R50 000 |
| Interest cover | EBITDA ÷ finance charges |
| Debt service coverage | (EBITDA − tax paid − maintenance capex + subordinated director loan movements) ÷ (interest + scheduled principal + proposed annual debt service) |
| Current ratio | Current assets ÷ current liabilities |
| Gearing | Interest-bearing debt ÷ tangible net worth |
| Tangible net worth | Equity − goodwill − intangibles − related-party receivables − debit director loans |
| Director loan accounts | Credit balances quasi-equity **only where formally subordinated**; debit balances treated as distributions and deducted from cash available |
| Working capital cycle | Debtor days + stock days − creditor days |

**Sector benchmarking.** Each of nine ratios is compared against its sector's
median, 25th and 75th percentiles from a table of 420 sectors × 9 ratios × 3
statistics. The output per ratio is a percentile band (1..5) and a points
contribution. Sectors with fewer than 30 observations in the benchmark fall back
to their parent sector grouping, and that fallback is recorded.

**Bank-statement alternative.** Where statements are absent or unusable:

- 6..24 months of credit turnover, excluding inter-account transfers, reversals,
  loan advances and own-account deposits (a classification rule set of its own,
  ~20 rules);
- turnover volatility (coefficient of variation) and seasonality;
- a net margin proxy from the sector table applied to turnover to estimate
  EBITDA, with a confidence band;
- excess days, returned debits, and lowest monthly balance as conduct measures.

A statement-based assessment **caps the facility at R1 500 000** and **caps the
business grade at 6**. It cannot produce a top grade, ever.

**Haircuts and confidence.**

| Audit level | EBITDA haircut | Grade cap |
|---|---|---|
| Audited, unqualified | 0% | none |
| Audited, qualified | 10% | 7 |
| Independently reviewed | 5% | none |
| Compiled / accountant-prepared | 15% | 7 |
| Client-prepared, unsigned | 25% | 8 |

| Age of most recent year-end at `decision_date` | Treatment |
|---|---|
| ≤ 9 months | No haircut |
| 9–15 months | 5% EBITDA haircut |
| 15–21 months | 15% haircut, and management accounts required |
| > 21 months | Financials unusable; fall back to bank turnover |

The statutory audit requirement itself follows the public
public-interest-score mechanism; whether a business *should* have been audited
and was not is recorded as a qualitative factor.

**Overlays on the financial component.** The financial score is an approved
artefact like any other, and carries its own overlays: a score shift where
observed defaults in a segment outrun the financial model's prediction, and a PD
multiplier where a benchmark set is known to be stale between annual refreshes.
These are **distinct from the haircuts above**, and confusing the two is a real
risk: a haircut is a data-quality adjustment to an *input* (unaudited EBITDA is
worth less), applied inside the assessment by rule; an overlay is a policy
adjustment to the *output*, applied on top of a validated model under a separate
approval. Both may apply to one application; each must be separately visible;
neither may be implemented as the other.

**Emits**: the 48-line spread, the eleven measures with their inputs, the nine
benchmark bands, `financial_pd`, `financial_pd_unadjusted`,
`financial_confidence_code` (high / medium / low / none), every haircut applied
with its cause, every overlay applied with its identifier and effect, and the
cash-available figure the pricing stage will test.

### 5.10 Stage — the combined business grade

**Determines**: `risk_grade` and `probability_of_default` for the business.

Four components:

| Component | Source | Available when |
|---|---|---|
| Financial | §5.9 | Always, at some confidence |
| People | §5.8 | Always |
| Behavioural | 24 months of internal conduct, scorecard `BUS-BEH-01`, 21 characteristics | Existing clients with ≥ 6 months of conduct |
| Qualitative | Analyst override, bounded | On request |

**Weights** vary by segment and by financial confidence. Segments by trailing
turnover: micro below R3 000 000; small R3 000 000 to R30 000 000; medium
R30 000 000 to R150 000 000. Above R150 000 000 is out of scope (§12).

| Segment | Financial confidence | Financial | People | Behavioural |
|---|---|---|---|---|
| Micro | high / medium | 0.30 | 0.55 | 0.15 |
| Micro | low / none | 0.15 | 0.70 | 0.15 |
| Small | high / medium | 0.45 | 0.40 | 0.15 |
| Small | low / none | 0.25 | 0.60 | 0.15 |
| Medium | high / medium | 0.60 | 0.25 | 0.15 |
| Medium | low / none | 0.40 | 0.45 | 0.15 |

Where the behavioural component is absent, its weight redistributes pro rata
across the other two. Blending is on log-odds, as at PP-05.

**The qualitative override** adjusts the resulting grade by at most ±2, with a
rationale chosen from a fixed list of 18 override reasons and free text. Its
authority is bounded:

| Facility amount | Who may override | Range |
|---|---|---|
| ≤ R500 000 | Senior analyst | ±1 |
| ≤ R2 000 000 | Credit manager | ±2 |
| > R2 000 000 | Credit committee only | ±2 |

Improving overrides require a second signature. Worsening overrides never do. An
override may not reverse a disqualification at §5.3, §5.4 or §5.6 — those are
absolute, and an override that attempts it is rejected with a reason, not
silently ignored.

**An override is not an overlay.** They are adjacent enough to be confused and
must not be: an override is one analyst's judgement about one application,
recorded against that application; an overlay is an approved policy instrument
applying to a declared population until it expires. They have different
authorities, different lifetimes, different scopes and different evidence
obligations. Both may act on the same grade, and the record must show which did
what.

**The declared overlay stack.** Overlays compose, and the composition order
changes the answer, so the order is part of the definition rather than an
emergent property of where the code happens to call `core.adjustments`. Eleven
positions apply to this flow:

| Order | Overlay target | Level | Typical scope |
|---|---|---|---|
| 1 | Adverse-event materiality thresholds (§5.5) | event | sector, entity criticality class |
| 2 | Score shift or scaling change on `BUS-PERS-01` / `BUS-COMM-01` | entity | scorecard, segment, new-to-bank |
| 3 | Calibration re-anchor or odds multiplier on entity PD | entity | scorecard, sector grouping |
| 4 | Entity grade boundary shift | entity | entity nature |
| 5 | Score shift or PD multiplier on the financial component (§5.9) | business | segment, financial confidence |
| 6 | **Sector PD multiplier** over the 420-sector table | business | `sector_code` or sector grouping |
| 7 | Business-level PD multiplier | business | segment, channel, product |
| 8 | Business grade boundary shift | business | product, segment |
| 9 | Cut-off shift — the minimum acceptable grade | decision threshold | product, sector |
| 10 | Cap adjustment — appetite maximum, surety cover, group cap (§5.11) | appetite | grade, sector |
| 11 | Rate add-on in basis points (§5.12) | pricing | rate cell range, grade, security type |

Position 6 is the one Business Credit Risk Policy uses most: *"tighten
construction by 25% for two quarters"* is a PD multiplier of 1.25 scoped to a
sector grouping, with an approval reference and an expiry date, and it is applied
without reissuing a scorecard, a sector table or a rate card. It is also the one
most likely to collide with position 3, where a sector-scoped multiplier has
already been applied to juristic entities in the same sector — see PP-11.

Two overlays occupying the same position with overlapping scope is a **conflict**
and must be detected when the adjustment set is published, not when an
application encounters it.

**Emits**: `risk_grade`, `probability_of_default`, `risk_grade_unadjusted`,
`probability_of_default_unadjusted`, `adjustment_set_id`,
`adjustments_applied` with each overlay's effect in application order, every
component's PD and weight in both adjusted and unadjusted form, the segment and
how it was determined, and the override with its rationale, author and
authority — separately from the overlays.

### 5.11 Stage — appetite, security and group exposure

**Determines**: the ceilings the offer must respect, before any pricing happens.

**Maximum facility** by grade × sector appetite class × security type: 12 × 5 × 4
= 240 cells, from `core.appetite` extended with the sector dimension.

**Collateral treatment** — declared value × advance rate, less prior
encumbrances:

| Collateral class | Advance rate |
|---|---|
| Residential or commercial property, first bond | 70% |
| Property, second bond | 45% |
| Debtors book, under cession, aged < 90 days | 40% |
| Plant and equipment, under notarial bond | 25% |
| Stock, under general notarial bond | 20% |
| Listed securities | 60% |
| Cash or Bank deposit under pledge | 100% |
| Guarantees | per §5.8 PP-08 and PP-09 |

**Security type**, which is a key into the rate card, derives from the cover
ratio = total adjusted cover ÷ offered amount:

| Cover ratio | `security_type` |
|---|---|
| ≥ 1.00 | 1 — fully secured |
| 0.50 – 0.99 | 2 — partially secured |
| < 0.50 with surety cover ≥ 25% of the amount | 3 — surety-backed |
| otherwise | 4 — unsecured |

Note what this creates: **security type depends on the offered amount**, and the
rate depends on the security type, and the affordable amount depends on the
rate. Reducing an offer can improve the cover ratio, change the security type,
and lower the rate. That is a second source of non-monotonicity in §5.12, on top
of the amount bands.

**Group exposure** through `core.exposure`. The group is every business sharing
a critical entity with the applicant — common control, or ≥ 25% common
ownership. Aggregate all facilities, drawn and undrawn, plus pending
applications. Caps:

| Grade | Group exposure cap |
|---|---|
| 1–3 | R25 000 000 |
| 4–6 | R15 000 000 |
| 7–8 | R7 500 000 |
| 9–10 | R2 500 000 |
| 11–12 | nil new exposure |

**Single-name concentration**: no group above R25 000 000 without committee, and
no group above the portfolio single-name limit at any authority.

**Sector concentration**: where the sector is at or above its portfolio cap,
every new application in it is referred, whatever its grade.

**Cap overlays.** Position 10 of the stack applies here: "reduce the maximum
facility by 20% for sector 412 until the end of the third quarter" is a cap
adjustment, not a new appetite grid. The appetite grid, the surety cover table
and the group exposure caps are all overlay targets, and every ceiling this stage
emits therefore carries both its base value and its overlaid value. Where an
overlaid cap is the binding constraint at §5.12, the binding constraint record
names the overlay — a client told "we can only offer R1 600 000" is being told
the result of a policy decision with an expiry date, and the analyst must be able
to see that it expires.

**Emits**: every ceiling with its source, its base value, its overlaid value and
the overlay identifier where one applied, and `appetite_headroom`.

### 5.12 Stage — structuring and pricing

**Preconditions**: a grade, a cash-available figure, the ceilings from §5.11, the
security schedule, and a requested amount.

**Determines**: the **largest amount the Bank will offer**, and its complete
structure. This is the stage that has to reconcile a client's request with four
constraints that each depend on the answer.

**The circularity.** The rate depends on the amount band, the term, the grade and
the security type. The instalment depends on the amount, term, rate and fees. The
coverage test depends on the instalment. The security type depends on the amount.
So the amount depends on itself, twice over, through two non-monotone
relationships.

**The candidate space.** Amount bands: 40 (the rate card's own banding, from
R50 000 to R10 000 000). Terms: 55 permitted values for product 50 (6 to 60
months, and the sector's maximum term may be lower — asset life caps apply to
equipment purposes). For product 51 the term is notional: the coverage test uses
an amortisation of the full limit over 36 months. The candidate space is
therefore up to **2 200 (amount, term) pairs**, of which typically 200 to 600 are
admissible after the ceilings at §5.11.

**Per-candidate evaluation**, in order:

1. **Cap the amount** by the least of: `requested_amount`, the appetite maximum
   (§5.11), adjusted security cover ÷ the minimum cover ratio for the grade,
   group exposure headroom, the product maximum (R10 000 000 for 50, R5 000 000
   for 51), the statement-based cap where §5.9 applied it, and total surety
   cover where the facility is surety-backed.
2. **Determine the security type** at this amount (§5.11) — it may differ from
   the type at the requested amount.
3. **Look up the rate**: `core.rate_card` on the business grid, 40 amount bands ×
   55 terms × 12 grades × 4 security types = **105 600 cells**. Add the sector
   premium (0–150 bps from the sector table), apply any **rate add-on overlay**
   (position 11 — a basis-point add-on over a declared cell range, which is how
   Treasury reprices a grade band mid-month without reissuing 105 600 cells),
   subtract the relationship discount (0–75 bps, authority-limited), and apply
   the **rate floor** — cost of funds + capital charge + expected loss at the
   grade's PD + operating cost. Where the floor binds, it is recorded as the
   binding price constraint. The rate record carries the cell identifier, the
   cell's own value, every add-on with its overlay identifier, and the final
   rate, so that a repricing overlay is never mistaken for a rate card change.
4. **Fees**: initiation fee and monthly service fee through `core.fees` under the
   regulated regime, or the business fee schedule otherwise; an assessment fee;
   and, for secured facilities, bond registration and valuation costs, which are
   a cost of the deal and are capitalised only on request.
5. **Instalment** via `core.instalment`.
6. **Coverage test**: debt service coverage ratio at the resulting instalment,
   against the grade's threshold.

| Grade | Minimum DSCR | Minimum interest cover | Maximum post-facility gearing |
|---|---|---|---|
| 1–4 | 1.20 | 2.00 | 3.0 (micro) / 2.5 (small) / 2.0 (medium) |
| 5–7 | 1.30 | 2.25 | as above, less 0.25 |
| 8–9 | 1.45 | 2.50 | as above, less 0.50 |
| 10–12 | 1.60 | 3.00 | committee only |

7. **Owner affordability**, where the owner's income *is* the business's income —
   sole proprietors, partnerships, and single-owner micro companies where
   drawings exceed 60% of EBITDA. Two different tests, and which applies is
   determined at §5.2:
   - **Regulated regime**: the statutory affordability assessment on the natural
     person, through [project 02](02-affordability-assessment.md), with the
     proposed instalment as a new obligation and business net profit after
     drawings as income. A statutory fail is a hard fail.
   - **Unregulated regime**: drawings are deducted from EBITDA before the
     coverage test, and the owner's personal affordability is tested only where a
     personal surety is required and the surety cap binds.
8. **Record the binding constraint** for this candidate — one of twelve codes:
   requested amount, appetite grade cap, sector cap, security cover, surety
   cover, group headroom, product maximum, statement-based cap, DSCR, interest
   cover, gearing, owner affordability, rate floor.

**The requirement on the search.** The Bank must offer the **largest admissible
amount**. Ties are broken by the shortest term, then by the lowest
`total_cost_of_credit`. Because the instalment is not monotone in the amount —
a smaller amount can fall into a worse-priced band, or cross a security-type
boundary in either direction — **bisection on the amount is invalid**, and an
implementation that assumes monotonicity will quietly return a smaller offer than
the Bank would have made.

The requirement is stated as a property rather than as a method:

> Whatever search is used must return the same answer as exhaustive evaluation of
> the admissible candidate set, and there must be a test that demonstrates this
> over a sample of at least 5 000 historical applications.

It must also be **bounded and terminating**: a declared maximum number of
candidate evaluations, a declared ordering, and a declared outcome when the
maximum is reached (the best admissible candidate found so far, flagged
`search_truncated`, referred if the truncation occurred before any admissible
candidate was found).

**Outcomes of the stage:**

| Situation | Outcome |
|---|---|
| Largest admissible amount ≥ `requested_amount` | Approve at the requested amount |
| Between 60% and 100% of requested, and ≥ R50 000 | **Reduced offer**, requiring client acceptance; the shortfall reason is the binding constraint |
| Below 60% of requested but ≥ R50 000 | Reduced offer, and refer where the binding constraint is DSCR or owner affordability |
| Below R50 000 (the product minimum) | Decline, reason = the binding constraint's reason code |
| No admissible candidate at any amount or term | Decline, reason 5501 |

**Emits**: `offered_amount`, `term_months`, `nominal_annual_rate`,
`rate_cell_id`, every fee, `instalment`, `total_cost_of_credit`,
`effective_annual_rate`, the DSCR at the offer, the security schedule required,
the security type achieved, and `binding_constraint_code`.

**Records**: every candidate evaluated, in order, with its rate cell, instalment,
DSCR and rejection reason. A client asking "why not R2 000 000?" gets the row for
R2 000 000.

### 5.13 Stage — final validation, conditions and referral

**Determines**: whether the assembled offer is internally consistent, properly
authorised, and complete — and where it goes next.

**Consistency rules** (FV-01..08). Each recomputes something the flow already
produced and requires agreement:

| Rule | Check |
|---|---|
| FV-01 | The instalment recomputes from amount, term, rate and fees to the cent |
| FV-02 | The DSCR recomputes from the spread and the instalment |
| FV-03 | The grade used to price equals the final grade after any override |
| FV-04 | The security type used for the rate lookup equals the security schedule's achieved class |
| FV-05 | The offered amount is at or below every ceiling from §5.11, individually |
| FV-06 | Effective ownership across the resolved structure reconciles to 100% ± 0.5%, or `structure_unresolved` is set |
| FV-07 | Every fired reason code exists in the registry version in force at `decision_date` |
| FV-08 | No approval carries an unresolved disqualification at §5.3, §5.4 or §5.6 |

**Required sureties** (FV-09..11):

- every entity with effective ownership > **25%** provides a personal surety,
  unless waived by committee with a recorded rationale;
- every group company with cross-exposure > R1 000 000 provides a cross-guarantee;
- a sole director provides a surety irrespective of ownership.

An offer that relies on surety cover the entity has not agreed to provide is not
an offer; the surety requirement becomes a condition precedent and the facility
does not fund without it.

**Conditions precedent** — drawn from a catalogue of 34, attached by rule:

| Condition | Attached when |
|---|---|
| Cession of debtors | Debtors book contributes to cover |
| First covering bond registered | Property contributes to cover |
| Notarial bond over plant | Plant contributes to cover |
| Personal surety executed | Per FV-09 |
| Updated tax clearance, ≤ 3 months old at drawdown | Always |
| Management accounts ≤ 3 months old | Most recent year-end older than 9 months |
| Subordination of director loan accounts | Gearing > 2.5, or loan credit balances counted as quasi-equity |
| Key-person insurance ceded | Facility > R2 000 000 and a single critical natural person |
| Primary transaction account with the Bank | Product 51, always; product 50 above R1 000 000 |
| Confirmation of no material change | Always |
| Valuation not older than 12 months | Plant or stock cover |

**Covenants** attached to the facility (monitored outside this flow, but set
here): DSCR ≥ the grade threshold tested annually; a gearing ceiling; a minimum
proportion of turnover — 70% — through the Bank's account; a capex limit; no
further borrowing above R250 000 without consent; annual financial statements
within 6 months of year-end.

**Authority and referral.** Five authority levels:

| Level | Limit |
|---|---|
| Analyst | R500 000 |
| Credit manager | R2 000 000 |
| Senior credit | R7 500 000 |
| Credit committee | R25 000 000 |
| Board credit committee | above R25 000 000 |

Mandatory committee triggers, irrespective of amount: a probable screening match;
`structure_unresolved`; `insufficient_people_coverage`; a restricted sector; an
approve recommendation at grade 10–12; a disputed event that would otherwise be
the binding decline; an override request beyond delegated authority; foreign
control; any business-rescue history; an exposure above the sector portfolio cap.

**The committee pack**, produced by this stage, is a defined artefact and is
specified in §9.3.

**Emits**: `outcome_code`, `decline_reason_codes`, `primary_reason_code`,
`referral_queue_code`, the authority level required, the conditions and covenants
attached, and the complete offer.

---

## 6. Parameters and tables

### 6.1 Parameters

| Parameter | Value | Owner | Cadence |
|---|---|---|---|
| Structure depth bound | 3 | Business Credit Risk Policy | Rare |
| Resolved entity count bound | 40 | Policy | Rare |
| Expansion materiality floor | 5.0% | Policy, tracking the disclosure mechanism | On regulation |
| Critical ownership threshold | 25.0% | Policy | Quarterly |
| Significant ownership threshold | 10.0% | Policy | Quarterly |
| Blend inclusion floor | 5.0% | Policy | Quarterly |
| Coverage requirement | 75% | Policy | Quarterly |
| Controlling entity weight floor | 0.60 | Policy | Quarterly |
| Non-owning controller notional weight | 10 points, capped at 30 | Policy | Quarterly |
| Minor-event count window | 3 events in 12 months | Policy | Quarterly, and after loss events |
| Aggregate unsatisfied thresholds | R150 000 / 15%; R500 000 / 50% | Policy | Quarterly |
| Recency window | 6 months | Policy | Quarterly |
| Velocity threshold | 8 dishonoured payments in 12 months | Policy | Quarterly |
| DSCR thresholds by grade band | 1.20 / 1.30 / 1.45 / 1.60 | Credit Committee | Semi-annual |
| Interest cover minimum | 2.00 / 2.25 / 2.50 / 3.00 | Credit Committee | Semi-annual |
| Gearing ceilings by segment | 3.0 / 2.5 / 2.0 | Credit Committee | Semi-annual |
| Segment turnover boundaries | R3m / R30m / R150m | Policy | Annual |
| Blend weights | 18 combinations | Policy | Quarterly |
| Override authority limits | 4 levels | Credit Committee | Annual |
| Search evaluation bound | declared per deployment | Engineering, with Policy sign-off | Rare |
| Regulated-regime thresholds | R1 000 000 turnover / assets; R250 000 facility | Compliance | On legislation |
| Reduced-offer floor | 60% of requested | Product | Quarterly |

Roughly **48** parameters in total. Of these, 31 are owned by people who are not
engineers and expect to change them without a release.

### 6.2 Tables

| Table | Dimensions | Cells | Owner | Cadence |
|---|---|---|---|---|
| Sector risk table | 420 sectors × 6 attributes (exclusion, restriction, appetite class, default rate, net margin proxy, maximum term) | 2 520 | Sector Analytics (library table, §8 of 00) | Annual + shocks |
| Sector ratio benchmarks | 420 sectors × 9 ratios × 3 statistics | 11 340 | Sector Analytics | Annual |
| Business rate card | 40 amount bands × 55 terms × 12 grades × 4 security types | 105 600 | Treasury (library table) | Monthly |
| Entity disqualification disposition | 14 rules × 3 criticality classes | 42 | Policy | Quarterly |
| Event amount thresholds | 14 event types × 3 classes × 2 severities | 84 | Policy | Quarterly |
| Event ageing decay | 14 types × 6 age bands | 84 | Policy | Quarterly |
| Collateral advance rates | 8 classes × 3 attributes | 24 | Credit Committee | Annual |
| Surety cover table | 12 grades × 2 values | 24 | Credit Committee | Annual |
| Appetite maximum facility | 12 grades × 5 appetite classes × 4 security types | 240 | Credit Committee | Quarterly |
| Group exposure caps | 12 grades | 12 | Portfolio Management | Quarterly |
| Statement line mapping | ~600 source labels → 48 standard lines | 600 | Business Credit Risk Policy | Continuous |
| Transaction classification for turnover | ~20 rules × 6 attributes | 120 | Policy | Semi-annual |
| Scorecard `BUS-PERS-01` | 38 characteristics × ~8 bins | ~300 | Model Team | On release |
| Scorecard `BUS-COMM-01` | 29 characteristics × ~8 bins | ~230 | Model Team | On release |
| Thin-file variants | 17 + 12 characteristics × ~6 bins | ~175 | Model Team | On release |
| Behavioural scorecard `BUS-BEH-01` | 21 characteristics × ~8 bins | ~170 | Model Team | On release |
| Fallback grades for no-hit entities | 5 roles × 6 age bands × 4 tenure bands | 120 | Policy | Annual |
| Conditions precedent catalogue | 34 conditions × 5 attributes | 170 | Legal and Policy | Semi-annual |
| Override reason list | 18 reasons | 18 | Policy | Annual |
| Reason codes contributed to the registry | ~140 codes, ranges 5100–5599 | — | Compliance | Monthly |

The 140 new reason codes are a material addition to the library's 380-code
registry and are themselves a change scenario for
[project 00](00-shared-credit-core-library.md).

---

## 7. Outputs

### 7.1 To the caller

| Output | Notes |
|---|---|
| `outcome_code` | Approve / approve with conditions / refer / decline |
| `offered_amount`, `term_months`, `nominal_annual_rate`, fee components, `instalment`, `total_cost_of_credit`, `effective_annual_rate` | The offer |
| `binding_constraint_code` | Why the offer is not larger |
| `risk_grade`, `probability_of_default` | Business level |
| `people_grade`, `financial_pd`, behavioural PD, component weights | The composition of the grade |
| `decline_reason_codes`, `primary_reason_code` | Ranked through `core.reason_codes` |
| `attributing_entity_id`, `attributing_event_ids` | Non-null on any decline attributable to an entity. **This is the output the project exists for.** |
| Per-entity summary | Role, effective ownership, criticality class, adverse verdict, binding roll-up rule, grade, scoring situation, weight in the blend, inclusion or exclusion with reason |
| Security schedule | Collateral required, class, value, advance rate, cover achieved |
| Conditions precedent and covenants | With their triggering rules |
| `referral_queue_code`, authority level required | |
| Data-quality flags | `structure_unresolved`, `cycle_truncated`, `search_truncated`, `bureau_is_stale` per entity, `amount_unknown` per event, `period_annualised`, `classification_provisional` |

### 7.2 Persisted

The complete nested record, at all three levels:

- the disclosed structure as received and the resolved structure, with the
  transformation between them;
- per entity: every input used, every rule evaluated, the criticality class and
  its inputs, the scoring result with per-characteristic contributions, the
  adverse verdict with its attributing event set;
- per event: the classification, the rule, the thresholds used and where they
  came from;
- the financial spread, the derived measures, the benchmark bands, every haircut;
- the blend weights;
- every pricing candidate evaluated, with its rate cell identifier;
- every table version resolved against `decision_date`.

Retention: **7 years after the facility closes**, or 7 years after the decline.

---

## 8. Non-functional requirements

| Requirement | Value |
|---|---|
| Volume | 900 applications/day, concentrated 09:00–16:00, peak 4× mean in the last two business days of the month |
| Entity fan-out | median 4, mean 6.2, p95 14, p99 27, maximum 40 |
| Event fan-out | median 6 per entity, p95 40, maximum 60; worst observed application 600 events total |
| End-to-end latency | **under 3 seconds** at p95, excluding external data acquisition; under 6 seconds at p99 |
| Worst-case latency | the 40-entity, 600-event application must complete inside 8 seconds, and must not be the cause of a timeout on any other application |
| Latency variance | The p99 must not be dominated by fan-out. A 40-entity application may take longer than a 1-entity one; it may not take forty times longer end to end, because the per-application fixed work dominates at the median |
| Re-assessment batch | 180 000 existing business clients with full entity structures — approximately 1 120 000 entities and 6 700 000 events — within an **8-hour window**, monthly |
| Partial re-assessment | A single entity's refresh must re-derive that entity and the business roll-up without re-running the whole application, in under 500 ms |
| Determinism | Identical inputs, versions and `decision_date` produce identical outputs, bit for bit, including the order of `decline_reason_codes` and the order of the pricing candidate record |
| Ordering independence | The outcome must not depend on the order in which entities or events arrive. A structure supplied in a different order produces the same verdict and the same attribution |
| Table refresh | A sector table or rate card refresh must not require a deployment |
| Availability | The flow degrades to referral, never to approval, when an external view is unavailable |

The ordering-independence requirement deserves emphasis: it is easy to satisfy
for worst-of roll-ups and easy to violate for count-based and first-match ones.
It must be tested by shuffling.

---

## 9. Audit, evidence and explainability

### 9.1 The five questions that must be answerable

| Question | Asked by | How long after |
|---|---|---|
| Which entity and which event declined this business? | The business, the analyst, the committee | Immediately, and for 7 years |
| What did each level contribute to the grade? | Credit committee, Internal Audit | At decision, and on replay |
| How was this specific judgment classified, and what difference did it make? | The individual, the Ombud, the regulator | Up to 7 years |
| This entity's data was wrong and is now corrected — what changes? | The business, the analyst | Days to months |
| Reproduce this decision exactly on the artefacts in force at the time | Internal Audit, the regulator | Up to 7 years |

### 9.2 Attribution, and the limit on disclosure

Internally, a decline must resolve to a triple: **(entity, event set, rule)**.
Where the binding rule is quantitative the computed quantity accompanies it. Where
the binding rule is a blend cap, the attribution is the weight vector and the
entity whose grade bound the cap.

Externally, the position is different and must be built in rather than bolted on.
The business is entitled to the reason for *its* decline. It is generally **not**
entitled to the details of a third party's credit record — a director's personal
judgment is that director's information. So the flow produces **two** reason
sets:

- the **internal attribution**, complete, naming the entity and the events;
- the **communicable reason**, drawn from the registry's client-facing wording,
  which may say "an individual associated with the business has adverse credit
  information" and must carry the route by which that individual can obtain their
  own reasons directly.

Which reason codes are communicable at business level, and which are not, is an
attribute in the registry owned by Compliance. Getting this backwards is a
privacy breach in one direction and a regulatory failure to give reasons in the
other.

### 9.3 The committee pack

A defined artefact, produced for every referred application, showing **every
level's contribution**:

1. The business summary, the decision, the offer and the binding constraint.
2. The resolved structure, as a diagrammable table: every entity, its path or
   paths, effective ownership, control, criticality class, and for de-duplicated
   entities every path by which they reach the applicant.
3. Per entity: the adverse event list with each event's classification, the rule
   that classified it and the thresholds used; the adverse verdict with its
   binding rule and attributing event set; the score, grade and top three
   negative characteristic contributions.
4. The people blend: every weight, the control adjustment if applied, every cap
   that fired.
5. The financial spread, the eleven measures, the nine sector benchmark bands
   with the sector's median, and every haircut with its cause.
6. The grade composition: each component's PD, its weight, the segment, and the
   override with its author and rationale.
7. The pricing record: the candidates evaluated, the rate cells read, and the
   constraint that bound.
8. The conditions and covenants with their triggering rules.

The pack is generated, not assembled by hand. An analyst assembling it by hand is
the state this requirement exists to prevent.

### 9.4 Single-event counterfactuals

A dispute about one judgment requires four answers:

1. how it was classified — rule, thresholds, the entity class that supplied them;
2. what the **entity verdict** would have been without it;
3. what the **people grade and business grade** would have been without it;
4. what the **offer** would have been without it.

That is a re-derivation with one event excluded, on the artefacts in force at the
original `decision_date`. It must be available for any event, not only the
binding one, because a dispute usually concerns the event the individual cares
about rather than the event the rule fired on.

### 9.5 Corrected data and re-runs

When an entity's data is corrected — an identity verified, a judgment rescinded,
an ownership percentage restated — the re-run's difference from the original must
be **explainable at every level**: which inputs changed, which event
classifications changed, which entity verdicts changed, which weights changed,
which grade changed, and what that did to the offer. A re-run that produces a
different answer without an attributable cause is a defect, and finding those is
[project 09](09-governance-and-replay-harness.md)'s job.

### 9.6 The same person on two applications

A natural person appearing on two applications on the same day must be evaluated
consistently: the same event list, the same classifications where the criticality
class is the same, and **different** classifications where it is not — because
the thresholds legitimately differ by role. Both applications' records must be
able to show the other's existence and the role difference, because "you declined
them there and approved them here" is a question that gets asked.

---

## 10. Acceptance criteria

1. Every decline attributable to an entity carries a non-null entity identifier
   and a non-empty attributing event set, and both resolve to records in the
   decision evidence. Zero exceptions across a 5 000-application regression set.
2. A count-based roll-up rule attributes to **all** the events that satisfied it,
   not to one of them.
3. The outcome and the attribution are invariant under shuffling of the entity
   list and of each entity's event list. Demonstrated by a randomised test over
   at least 1 000 applications, 20 shuffles each.
4. A natural person reached by two ownership paths is evaluated once, with summed
   ownership, the most senior role, and both paths recorded.
5. Structure bounds are enforced, and exceeding them produces a referral with the
   unexpanded remainder listed — never a decline and never an approval.
6. The largest-admissible-amount property (§5.12) is demonstrated against
   exhaustive evaluation over at least 5 000 historical applications, with zero
   disagreements.
7. The pricing search is bounded: a declared maximum number of evaluations, never
   exceeded, and a declared behaviour on truncation.
8. Business and entity disqualification rule sets produce their **complete** fired
   set, verified by a test that plants multiple simultaneous failures.
9. A disputed event never causes a decline on its own; the outcome becomes a
   referral naming the dispute.
10. A single-event counterfactual is producible for any event on any entity,
    for any decision within the retention period, on the artefacts in force at
    its `decision_date`.
11. A partial re-assessment of one entity produces the same business outcome as a
    full re-run with the same data, within the 500 ms budget.
12. The monthly batch completes within the 8-hour window at the stated
    volumetrics, on the p99 fan-out distribution, not the median.
13. A Business Credit Risk Policy analyst can change the minor-event count
    window, the coverage requirement or an amount threshold without an engineer,
    and the change produces a reviewable diff and a re-scored sample.
14. Every table read during an assessment records its version and the cell read,
    including the rate card cell and the sector benchmark row.
15. The client-facing reason set never discloses a third party's event detail,
    and the internal attribution always contains it.

---

## 11. Change scenarios

Each has been requested of a real business-lending assessment. A good structure
makes them cheap; the previous generation made several of them a rewrite.

1. **The beneficial-ownership disclosure threshold drops from 5% to 3%.** Median
   resolved entity count rises from 4 to 6, p95 from 14 to 19, and roughly 8% of
   applications newly exceed the count bound. The expansion floor, the blend
   inclusion floor and the criticality thresholds are three different parameters
   that all happen to be 5% today, and only one of them moves.
2. **A new entity kind** — an employee share trust — which owns 12%, controls
   nothing, has no bureau record, and must not drag the coverage ratio below 75%.
3. **The depth bound rises from 3 to 4** for applications above R5 000 000 only.
4. **The three-minor-events-in-12-months rule becomes four-in-18**, for
   peripheral entities only, leaving critical and significant unchanged.
5. **A new adverse event type** arrives: a national municipal-arrears register,
   with amounts, no satisfaction concept, and its own ageing behaviour. It needs
   thresholds in three criticality classes and a row in the ageing table.
6. **The commercial scorecard is replaced** with a 34-characteristic version, and
   both must run in parallel for three months — the old one for existing clients
   and the new one for new-to-bank, in the same flow, on the same application
   where a group company is an existing client and the applicant is not.
7. **The sector table grows from 420 to 500 codes** with a remapping of 60
   existing codes. Historical decisions must continue to resolve their old codes.
8. **DSCR thresholds become sector-specific** as well as grade-specific, turning a
   12-row table into a 420 × 12 one.
9. **Treasury adds a fifth security type** to the rate card, taking it from
   105 600 to 131 400 cells, and the security-type derivation rules change with
   it.
10. **A third product launches** — Business Asset Finance, secured on the financed
    asset — reusing every stage but the security treatment and adding an asset-life
    term cap.
11. **Sureties at grade 8 stop counting**, mid-quarter, after a loss review. Every
    application in the pipeline must be re-assessed under the new table, and the
    ones already offered must be identified.
12. **The regulator requires the beneficial-ownership chain to appear in the
    decline record**, including the intermediate entities that were dropped below
    the materiality floor.
13. **Group exposure must include entities related through a common trust
    beneficiary**, not only through control or ownership — widening the graph
    query without widening the assessment structure.
14. **Policy decides that a disputed event should suspend the application for 21
    days** rather than force a referral, with automatic re-assessment on
    resolution.
15. **Credit committee wants the pack to show, for every declined application,
    the smallest change that would have made it approvable** — which amount,
    which term, which entity's event removed, or which additional security.

---

## 12. Out of scope

- **Data acquisition**: bureau calls, screening enquiries, company-registry
  retrieval, document collection, and the optical extraction of figures from a
  PDF financial statement. The flow consumes a spread; it does not perform the
  extraction. The **mapping** of extracted labels to standard lines is in scope.
- **Screening itself.** The flow consumes a screening outcome and a confidence
  from Financial Crime; it does not match names against lists.
- **Post-approval servicing**: facility documentation, bond registration,
  drawdown, covenant monitoring, annual review triggers and limit management —
  the last of which is [project 07](07-credit-limit-management.md).
- **Collections and workout** — [project 08](08-collections-treatment.md).
- **Corporate and large-corporate lending** above R150 000 000 turnover, which is
  individually underwritten and not a rules-driven assessment.
- **Specialised lending**: property development, project finance, invoice
  discounting mechanics, trade finance instruments, asset-based lending against a
  revolving borrowing base.
- **Model development.** The flow consumes scorecards and calibrations; it does
  not fit them.
- **Decision record storage and replay tooling** — [project
  09](09-governance-and-replay-harness.md).
- **The case-management workflow** around referrals: queues, allocation,
  service levels and the analyst's screen.

---

## 13. Questions the implementation must answer

1. **What is the unit that operates on one element of a collection, when the
   element itself contains a collection?** Event classification runs inside entity
   evaluation, which runs inside application evaluation. Is that one construct
   applied twice, two constructs, or a general capability with a depth
   parameter — and does the answer survive a third level appearing (change
   scenario 13)?

2. **How is a roll-up expressed** when it is neither worst-of nor an average, but
   a rule set over counts, aggregates, recency, velocity and trend, where several
   rules can fire and the most severe wins but all must be recorded? Is roll-up a
   core component kind in its own right, or is it a rule set that happens to run
   over a collection?

3. **Where does the parameterisation of a shared capability by its context
   live?** `core.adverse_events` classifies an event; here the thresholds it uses
   depend on the criticality of the entity the event hangs off, which is computed
   two levels up. Is that a parameter passed inward, a wrapped capability, a
   second capability, or a fork — and which of those can the library survive six
   consumers doing?

4. **How does attribution propagate upward** without every level having to
   hand-assemble it? A business decline names an entity and a set of events. Does
   the framework carry that, or does each project rebuild it — and the previous
   generation's answer was that each project rebuilt it, badly, and one of them
   dropped it entirely.

5. **What identifies an element of a collection** stably enough to attribute to
   it, across a re-run, across a data correction that changes the collection's
   length, and across a version change? Positional identity fails the moment an
   event is removed.

6. **How is a count-based attribution represented** — a rule that fires on three
   events attributes to three events, and a rule that fires on an aggregate
   attributes to every contributing event. Is the attribution shape uniform, or
   does it vary by rule kind?

7. **How is ordering independence guaranteed** rather than hoped for, given that
   several roll-up rules are count-based and one is a first-match?

8. **What does a partial re-derivation look like?** One entity refreshes. Its
   events reclassify, its verdict changes, the blend weights are unchanged but
   the blend result is not, and the price may move. What is the smallest re-run
   that is provably equivalent to a full one, and how is that proof obtained?

9. **How are two scorecard families with different characteristic sets applied
   to a single heterogeneous collection**, where the choice depends on an
   attribute of the element, and both must contribute to one blended result?

10. **How is a non-monotone bounded search expressed and proven?** The pricing
    stage cannot bisect, must terminate, must return the exhaustive answer, and
    must record every candidate. Is that a general searching construct, a
    project-specific one, or the same shape as [project
    03](03-unsecured-loan-granting-and-pricing.md)'s and [project
    06](06-consolidation-and-restructure.md)'s?

11. **Where does the circularity between the security type and the offered
    amount get resolved**, given that the security type is a key into the rate
    card and a function of the amount the rate card prices?

12. **How is a graph collapsed to a bounded structure**, and where does that work
    live — it is set-shaped work feeding record-shaped work, the same seam
    `core.exposure` has, and this project needs both on the same application.

13. **How is a rule set of 34 event classification rules, 12 roll-up rules, 20
    disqualification rules, 14 entity rules, 10 blend rules and 15 validation
    rules organised** so that a policy analyst can find the one they were told to
    change, and an engineer can see what else reads the same threshold?

14. **How do two reason sets — internal attribution and communicable reason —
    coexist** without the communicable one being derived by hand, and without the
    internal one leaking?

15. **What does the fan-out cost?** A 40-entity, 600-event application against a
    1-entity, 0-event one, in the same deployment, under one latency budget. Is
    the cost structure predictable enough to size, and what happens to the batch
    when the p99 tail is what dominates?

16. **Can the same assessment run record-at-a-time for granting and
    set-at-a-time for the monthly batch** and produce identical results, given
    that the batch's 6 700 000 events are the same shape as one application's 600?

17. **What is the smallest change that would have made this approvable?**
    (Change scenario 15.) It is a counterfactual search over the same
    non-monotone space, and whether the design can answer it cheaply is a good
    test of whether the pricing stage was expressed as data or as control flow.
