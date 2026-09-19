# 03 — Unsecured loan granting and pricing

> Fictional. The Bank, its products, thresholds, table dimensions and volumetrics
> are invented for this repository. Regulatory mechanisms referred to are the
> published public ones; every number attached to them is illustrative.

---

## 1. What this is

A client asks the Bank for a **Flex Loan** — `product_code` 10, unsecured, R2 000
to R500 000, 6 to 84 months — naming an amount and a term. The Bank must answer
within a fifth of a second, and the answer is one of two things: a decline
carrying every reason that applied, or a **set of offers**, one flagged as
recommended, each stating an amount, a term, an instalment, a total cost and an
effective rate the client can be held to.

The requested amount is frequently not the amount the client can have. It may
exceed what policy permits at their risk grade, what the Bank's appetite allows,
what their group exposure leaves room for, or — most often — what they can
afford. So the real question is not "may they have R120 000?" but "**what is the
largest amount they may have, at each term the Bank will write?**"

That question has no closed form. The interest rate is read from a grid keyed on
the amount, so changing the amount changes the rate. The initiation fee is a
piecewise function of the amount and is capitalised into the advance, so the
amount funded and the amount financed are different numbers. The credit life
premium is charged per R1 000 of the amount financed, which includes the
capitalised fee. The instalment depends on all three. Affordability is a test on
the instalment. Advances round to R100, and rounding down can move the advance
into a *worse* rate band, so the instalment is **not monotone in the amount**.
The answer, in other words, is found by a search, and the search has to be
bounded, deterministic, reproducible years later, and correct — because an
approved offer is a contract, and a mispriced contract is both a regulatory
finding and a direct financial loss.

Around that search sits the rest of a granting flow: eligibility gates, a fraud
handoff, bureau retrieval and data-quality assessment, a 45-characteristic
application scorecard with calibration and grading, a stack of approved
post-model overlays that policy applies over the model's own answers and over
the published rate card, a waterfall of 52 independently-owned policy caps, the
affordability capability published by project 02, pricing, offer-set
construction, and a final validation that re-derives the chosen offer from
nothing and refuses to ship it if anything disagrees.

This project is the primary home of those overlays. A scorecard is validated,
signed off and then left alone for twelve to eighteen months; a rate card is
reissued monthly but not on a Wednesday afternoon because one channel's early
vintages turned. When reality moves faster than the artefact, policy does not
rebuild the model — it applies a named, approved, effective-dated adjustment on
top of it, and the adjustment is a different artefact under a different approval
path from the thing it modifies. Five of this project's stages are adjustable,
and the order in which the adjustments compose changes the price.

The same logic runs twice: 55 000 times a day in real time, and once a month over
14 million clients to produce the pre-assessed offers project 04 puts into
campaigns. The two paths must agree exactly.

---

## 2. Why it is in this set

| Question | How this project stresses it |
|---|---|
| **Q2 tables** | The Flex Loan rate card is 96 amount bands × 55 terms × 12 grades — **63 360 cells** — and it is not read once per application but tens of times, inside the search, at amounts the search itself chooses. The credit life table is 672 cells, the scorecard definitions ~1 440 rows across four segments, the cap register 52 rules × 11 attributes. Cell-level attribution is required for every read. A grid this size authored in a spreadsheet by people who are not engineers is the central integration problem, not a footnote. |
| **Q5 core component set** | Three shapes in this spec fit none of scorecard / decision table / decision tree. The first is the **bounded solve**: a deterministic search over a discrete money grid, with a non-monotone objective, a hard evaluation ceiling, a defined tie-break and a recorded binding constraint. The second is the **cap waterfall**: 52 independently-owned rules, each narrowing one of three ceilings, where the required output is not only the final value but the whole chain and the identity of the rule that bound it. The third is the **overlay stack**: an ordered, scoped, effective-dated composition applied over five different values in five different stages, which must keep the unadjusted value alive beside the adjusted one and must be runnable switched off. All three recur — 06 solves, 07 waterfalls, and six of the ten projects adjust — which is the test of whether they are core. |
| **Q3 parameters** | The rate card is externalised, refreshed monthly by Treasury and patched mid-month after a repo move, without a code deployment. The cap register is reordered quarterly by four different owners. The recommendation objective is switched by a product manager. The statutory ceilings are changed by Compliance on gazette. Four cadences, four approval routes, one flow. Sitting over all of them is the **adjustment set** — an artefact with its own version, its own approval and its own expiry, which changes the answers of already-approved artefacts at runtime without editing them. It is simultaneously a parameter change, a piece of real composition logic and an audit artefact, and it must not be answered three different ways. |
| **Q1 reuse** | This project consumes ten capabilities from project 00 — `core.eligibility`, `core.bureau`, `core.scorecard`, `core.calibration`, `core.risk_grade`, `core.appetite`, `core.rate_card`, `core.fees`, `core.credit_life`, `core.instalment`, `core.rounding`, `core.reason_codes`, `core.dates` — and the whole of project 02's affordability assessment, and calls several of them repeatedly inside the search with different arguments each time. If the library's seams are wrong, they are wrong here first. |
| **Q4 custom modules** | The solve, the cap waterfall and the offer-set construction are project-specific today and wanted by project 06 within two quarters. Whether they can move without a copy-paste is the measurement. |
| **Q7 audit** | A declined applicant may demand reasons. A regulator may test whether pricing matched the published card — and, where an overlay moved the rate, which part came from the card and which from policy. Model Risk must see the model's own answer separately from what policy did to it. Internal Audit samples 200 approvals a quarter and re-derives each. A disputed instalment surfaces four years later and must reproduce to the cent, with the overlay stack that was in force on the day. |
| **Q6 codebase** | Secondary. One product, but 52 policy rules with four owners and a quarterly reordering cadence. |

---

## 3. Actors

| Actor | Responsibility |
|---|---|
| **Unsecured Lending Product** | Owns the product. Owns the permitted term list, the minimum viable offer rules, the deduplication tolerance and the recommendation objective. Changes things weekly. |
| **Credit Risk Policy** | Owns the cap waterfall's policy-class rules (31 of 52), the grade boundaries, the appetite grid slice and the credit life table. Quarterly cadence, Credit Committee approval. |
| **Treasury / Pricing** | Owns the rate card and the long-term loading. Monthly refresh, mid-month patches on repo moves. Authors in a spreadsheet. |
| **Regulatory Compliance** | Owns the statutory rate ceiling, the initiation and service fee caps, the credit life premium cap, the in duplum test, the pre-agreement disclosure content and the reason code registry. Signs off that pricing is lawful. Requires evidence, not code. |
| **Model Risk / Model team** | Owns the four scorecards, their calibrations, the challenger and the parallel-run traffic share. |
| **Financial Crime** | Owns the application-fraud verdict consumed from project 01 and the exclusion lists. |
| **Credit Committee** | Grants the authority limit under which the campaign uplift rule may raise a cap. |
| **Branch and call-centre consultants** | Present the offer set to a client and must be able to explain why the recommended offer is recommended and why a term is missing. |
| **Campaign Marketing (project 04)** | Consumes the monthly batch pre-assessment. Cannot change the logic that produces it. |
| **Internal Audit** | Samples approvals and re-derives them. Raises findings when a table version was not recorded. |
| **Client** | Entitled to reasons on refusal, to a pre-agreement quotation, to the cost breakdown, and to substitute their own credit life policy. |

---

## 4. Inputs

### 4.1 The application

| Field | Type | Card. | Null? | Notes |
|---|---|---|---|---|
| `application_id` | int64 | 1 | no | |
| `client_id` | int64 | 1 | no | Resolved before this flow. |
| `decision_date` | date | 1 | no | Governs every table version. Never "today". |
| `product_code` | int16 | 1 | no | Always 10 here. |
| `channel_code` | int8 | 1 | no | 1 branch, 2 app, 3 web, 4 call centre, 5 partner, 6 broker. |
| `requested_amount` | float64 | 1 | yes | Null on pre-assessment and on "tell me what I qualify for" journeys — 31% of app-channel volume. |
| `requested_term_months` | int16 | 1 | yes | Null in the same cases. |
| `purpose_code` | int8 | 1 | yes | 14 values. Informational except for two exclusions. |
| `is_joint_application` | bool | 1 | no | 4% of volume. Two applicants, one household assessment. |
| `campaign_id` | int32 | 1 | yes | Present on 22% of volume; drives the uplift rule. |
| `applicant_age_years` | float64 | 1 | no | At `decision_date`. |
| `employment_type_code` | int8 | 1 | no | 1 permanent, 2 contract, 3 self-employed, 4 pensioner, 5 social grant, 6 informal. |
| `months_employed` | float64 | 1 | yes | Null for 9% of self-employed applicants. |
| `employer_id` | int32 | 1 | yes | Matched against the exposure watchlist. |
| `residency_code` | int8 | 1 | no | 6 values including three non-resident classes. |
| `credit_life_substitution_declared` | bool | 1 | no | The client's statutory right to use their own policy. |

### 4.2 Internal client state

| Field | Type | Card. | Null? | Notes |
|---|---|---|---|---|
| `internal_tenure_months` | float64 | 1 | yes | Null for new-to-bank (28% of volume). |
| `internal_accounts` | list | 0..40 | — | Product, balance, limit, instalment, arrears months, opened date, status. |
| `internal_exposure_total` | float64 | 1 | no | Zero for new-to-bank. |
| `group_exposure_limit` | float64 | 1 | no | From `core.exposure`; includes related parties. |
| `in_flight_applications` | list | 0..6 | — | Other Flex Loan or consolidation applications not yet concluded. Median 0, p99 2. |
| `behaviour_summary` | struct | 1 | yes | 11 derived behaviour characteristics; null for new-to-bank. |
| `debt_review_status_code` | int8 | 1 | no | None, application lodged, under review, rescinded, cleared. |
| `administration_order_flag` | bool | 1 | no | |
| `insolvency_status_code` | int8 | 1 | no | None, sequestrated, rehabilitated. |
| `deceased_flag` / `estate_flag` | bool | 1 | no | |
| `exclusion_list_hits` | list | 0..5 | — | Sanctions, internal fraud, staff restriction, litigation. |
| `consent_records` | list | 1..12 | — | From `core.consent`. Bureau-enquiry consent is mandatory. |

### 4.3 Bureau

Retrieved from one of three bureaux and normalised through `core.bureau`.

| Field | Card. | Notes |
|---|---|---|
| Header, identity block | 1 | May return **more than one subject** for a single identity number — 0.4% of enquiries. |
| Account list | 0..80 | Median 7, p95 23, p99 41. Ragged. |
| Enquiry list | 0..120 | Windows of 30, 60, 90 and 365 days are derived. |
| Public record list | 0..25 | Judgments, defaults, administration orders, notices. |
| `bureau_as_of_date` | 1 | Staleness tolerance for Flex Loan is **45 days** at `decision_date`. |

No-hit rate is 6.2% of applications. Thin file — fewer than three accounts ever
opened, or under 12 months of bureau history — is a further 11.4%.

### 4.4 Fraud verdict

Produced by project 01's application-fraud engine, called synchronously with an
**800 ms** budget.

| Field | Values |
|---|---|
| `fraud_verdict_code` | 1 approve, 2 refer, 3 decline, 4 unavailable |
| `fraud_reason_codes` | 0..8 codes |
| `fraud_response_ms` | observed latency |

### 4.5 Affordability inputs

Everything project 02 requires: declared and verified income, payslip and
statement derivations, dependants, declared living expenses, and the client's
existing obligations list (0..80 accounts, from `core.obligations`). This project
does not re-derive any of it; it consumes `max_affordable_instalment`,
`discretionary_income`, `affordability_verdict_code` and the version identifiers
of the norm and buffer tables that produced them.

### 4.6 Volumes

| | Value |
|---|---|
| Real-time applications | 55 000 per day |
| Peak arrival rate | 9 per second, sustained for up to 20 minutes on payday Fridays |
| Monthly batch pre-assessment | 14 000 000 clients, 6-hour window, 648 records/second sustained |
| Distinct applicants per month | ~1 180 000 |
| Approval rate | 41% of applications produce at least one offer |

---

## 5. The flow

Eleven stages. Preconditions, what each determines, what it emits, what must be
recorded. A single worked applicant — **client W** — runs through the whole
document so the numbers join up: existing client, age 38, permanent employment,
41 months' tenure with the employer, channel 2, requesting **R120 000 over 60
months**, on campaign 4471.

### 5.1 Intake and eligibility gates

**Preconditions.** An application record exists and `decision_date` is set.

**Determines.** Whether the Bank may lend to this person at all, before spending
money on a bureau enquiry or a fraud check.

Gates applied, each producing its own reason code from the registry:

| Gate | Test | Reason |
|---|---|---|
| Product availability by channel | Flex Loan is not written on channel 5 below R10 000 or on channel 6 at all after 2026-03-01 | 1102 |
| Minimum age | `applicant_age_years` ≥ 18.0 at `decision_date` | 1110 |
| Maximum age at maturity | age at `decision_date` + `term_months`/12 ≤ 75.0 | 1111 |
| Capacity to contract | Not under curatorship; not a minor | 1112 |
| Residency | `residency_code` in the permitted set; two non-resident classes excluded | 1120 |
| Employment type | Social grant (5) and informal (6) excluded for Flex Loan; pensioner (4) permitted to R80 000 only | 1130, 1131 |
| Debt review | Any status other than "none" or "cleared" declines | 1140 |
| Administration order | Active order declines | 1141 |
| Insolvency | Sequestrated declines; rehabilitated permitted with a 24-month seasoning | 1142 |
| Deceased / estate | Either flag declines and raises an operational alert | 1150 |
| Exclusion lists | Any hit declines; sanctions hits additionally route to Financial Crime | 1160–1164 |
| Duplicate detection | An identical application (same client, amount within R500, same day) concluded in the last 24 hours | 1170 |
| In-flight detection | An unconcluded Flex Loan or consolidation application exists | 1171 |

**The short-circuit requirement.** There is no point scoring a deceased
applicant, and a bureau enquiry on an excluded person is a cost and a consent
problem. So these gates stop the flow. But a client who fails four gates is
entitled to know they failed four gates, and a decline letter naming one reason
when four applied has been the subject of a complaint. **Every gate that is
determinable from data already in hand must be evaluated and recorded even after
the first failure.** Gates that require data not yet retrieved are recorded as
`not_evaluated`, with the reason they could not be evaluated — this is a third
state, distinct from pass and fail, and it must not collapse into either.

**Emits.** `is_eligible`, `decline_reason_codes` (ranked), `primary_reason_code`,
and a per-gate verdict vector of 14 entries.

**Recorded.** All 14 gate verdicts, the input value each tested, the version of
the channel availability matrix and the exclusion list snapshot identifier.

*Client W passes all 14.*

### 5.2 Identity, consent and fraud handoff

**Preconditions.** Eligible.

**Determines.** Whether a bureau enquiry is lawful, and whether the application
is fraudulent.

Bureau-enquiry consent must be present, valid at `decision_date`, and obtained
through a channel permitted for that consent type. Absent or expired consent is
not a decline — it is a **referral** to the originating channel to re-obtain
consent, reason 1201, queue 1. Proceeding without it is a statutory breach and
the flow must make it structurally impossible.

The application-fraud verdict is requested from project 01's engine. Handling:

| `fraud_verdict_code` | Action |
|---|---|
| 1 approve | Continue |
| 2 refer | Continue the assessment to completion, but the final outcome is forced to `refer`, queue 3, and the offer set is computed and held rather than presented |
| 3 decline | Decline, reason 1210, no further assessment |
| 4 unavailable | See below |

**Timeout.** The 800 ms budget is exceeded on roughly 0.9% of calls. On timeout
the verdict is `4 unavailable` — which is **not** an approval, and the
distinction must survive into the decision record. The default treatment is
`refer`, queue 3. A single documented exception applies: applications below
R15 000 from clients with at least 24 months' internal tenure and no adverse
internal history proceed under a low-risk bypass, with the bypass, its authority
reference and the observed latency recorded. The bypass share of volume is
monitored; exceeding 2% of daily volume raises an operational alert.

**Emits.** `consent_verdict`, `fraud_verdict_code`, `fraud_reason_codes`,
`fraud_bypass_applied`.

**Recorded.** Consent record identifiers and their timestamps, the fraud engine's
response identifier, its latency, and which of the four handling paths was taken.

### 5.3 Bureau retrieval and data quality

**Preconditions.** Consent valid, fraud verdict not `decline`.

**Determines.** The normalised bureau view, and whether it is fit to decide on.

Normalisation is `core.bureau`'s job. This stage owns the **verdict on the
result**:

| Verdict | Meaning | Treatment |
|---|---|---|
| DQ-0 | Clean | Continue |
| DQ-1 | Minor defects — one unparseable account of many, a missing opened date | Continue, record |
| DQ-2 | Material defects — conflicting identity subjects, an account list truncated by the bureau, a status code outside the published domain | **Refer**, queue 2 |
| DQ-3 | Unusable — response malformed, header identity does not match the applicant | **Refer**, queue 2, and re-enquire |

The important requirement is the negative one: **a data-quality problem may never
become a decline.** A client whose bureau record could not be parsed has not been
shown to be a bad risk, and declining them on that basis is both wrong and
indefensible under a reasons request. DQ-2 and DQ-3 refer.

**Staleness.** If `bureau_as_of_date` is more than 45 days before
`decision_date`, `bureau_is_stale` is set. A stale view may be used for the
monthly batch pre-assessment but never for a real-time offer that will be
presented; in real time it triggers a fresh enquiry. The tolerance is a
per-product parameter (Flex Loan 45 days, Consolidation 30, Everyday Card 60) and
must be settable per product without touching the others.

**No-hit and thin file.** A no-hit (6.2%) routes to the thin-file scorecard if
any internal behaviour exists, and otherwise to manual underwriting, queue 4 —
never to an automatic decline. The three states — no bureau record, a bureau
record showing no accounts, and a bureau enquiry that failed — are three
different applicants and must remain three different values through the whole
flow.

**Conflicting identity records.** Where the bureau returns more than one subject
for the identity number, the verdict is DQ-2 regardless of how similar the
subjects are. Automatic merging is prohibited.

**Emits.** The normalised bureau view, `bureau_as_of_date`, `bureau_is_stale`,
`data_quality_verdict`, `bureau_source_code`, enquiry velocity over four windows.

**Recorded.** The raw response identifier, the normalisation version, every
defect found with its class, and the verdict.

### 5.4 Application scorecard, calibration and grading

**Preconditions.** Bureau view available or explicitly absent with a segment that
tolerates it.

**Determines.** `score`, `probability_of_default`, `risk_grade`, and the reasons
a score is low.

**Segment.** Four scorecards, assigned by a precedence that must be stated and
stable:

| Priority | Segment | `scorecard_id` | Condition | Share |
|---|---|---|---|---|
| 1 | Thin file | 1012 | Fewer than 3 bureau accounts ever, or under 12 months' bureau history | 11.4% |
| 2 | New to bank | 1010 | No internal account ever held | 22.1% |
| 3 | Existing client | 1011 | At least 3 months of internal behaviour | 66.5% |

**Characteristics.** 45 per scorecard: 22 bureau-derived, 11 internal behaviour,
8 application-form, 4 demographic. Each is binned into at most 8 bins. **Null is
its own bin, always**, with its own points — a missing value is information, not
an error, and for the thin-file scorecard 14 of the 45 characteristics are null
for more than a third of applicants.

**Output required, not optional.** The stage emits the score *and the point
contribution of every one of the 45 characteristics*, because the adverse-action
explanation is derived from them and a decline letter that cannot name its
reasons is a compliance failure. Contributions are expressed relative to the
population-neutral points for that characteristic, so that "you scored below
average on this" is meaningful. The **top four negative contributions** become
reason codes.

**Calibration.** Score to `probability_of_default` by the standard log-odds
scaling: 20 points double the odds, so the factor is 20 / ln 2 = 28.8539; the
offset is anchored at a score of 600 corresponding to 30:1 good:bad odds, giving
501.84. Scores are scaled to the range 350–900 and clipped. The inverse must also
be available — appetite logic asks "what score would we need for this to be worth
writing at this price?" — and the two directions must agree to within 0.01 of a
point.

**Grading.** `probability_of_default` to `risk_grade` 1..12:

| Grade | PD from | PD to | | Grade | PD from | PD to |
|---|---|---|---|---|---|---|
| 1 | 0.00% | 0.45% | | 7 | 4.60% | 7.00% |
| 2 | 0.45% | 0.75% | | 8 | 7.00% | 10.50% |
| 3 | 0.75% | 1.20% | | 9 | 10.50% | 15.50% |
| 4 | 1.20% | 1.90% | | 10 | 15.50% | 22.50% |
| 5 | 1.90% | 3.00% | | 11 | 22.50% | 33.00% |
| 6 | 3.00% | 4.60% | | 12 | 33.00% | 100.00% |

Boundaries are per segment, so 48 boundary values for this product.

**Challenger.** A fifth scorecard, `scorecard_id` 1013, runs on a deterministic
**10%** of traffic, selected by a stable hash of `client_id` — not of
`application_id`, so a client receives consistent treatment across repeat
applications, and not by a random draw, so a replay reproduces the selection. The
challenger scores, calibrates and grades in full and records what it *would* have
decided, but does not decide. Both scores, both PDs and both grades are recorded
for every application in the parallel run. The traffic share is a parameter the
Model team changes; it has been moved from 5% to 10% to 20% and back within one
year.

**Emits.** `scorecard_id`, `scorecard_version`, `score`, 45 point contributions,
`probability_of_default`, `risk_grade`, `score_reason_codes` (top 4 negative),
and the challenger's parallel set where applicable.

**Recorded.** Every characteristic's raw value, its bin, its points; the
calibration parameters; the grade boundary set version.

*Client W: `scorecard_id` 1011, score 618, PD 5.80%, `risk_grade` 7. Top negative
contributions: revolving utilisation (−14 points), months since most recent
arrears (−11), number of active unsecured accounts (−9), enquiry count in 90 days
(−6).*

#### 5.4.1 Overlays on the model's answers

A scorecard is validated, signed off and then left alone for twelve to eighteen
months. Reality does not wait. When observed defaults run above prediction, or
one channel deteriorates, or a macroeconomic view says the book is riskier than
the model believes, policy does not rebuild the model — it layers a named,
approved, effective-dated **overlay** over the model's output. Four of them apply
in this stage, and they apply in a declared order because the order changes the
grade:

| # | Overlay | Applies to | Illustrative live example |
|---|---|---|---|
| 1 | Score shift | `score` | −18 points on the new-to-bank segment, scorecard 1012, expires 2027-03-31 |
| 2 | Scaling change | the score-to-odds relationship | points-to-double-the-odds moved from 20 to 22 on scorecard 1011, flattening the curve at the tails |
| 3 | Odds multiplier | `probability_of_default` | PD × 1.35 on channel 4, applied after calibration and before grading |
| 4 | Boundary shift | `risk_grade` | the grade 6/7 boundary moved from 4.60% to 4.20% PD, tightening the top of the acceptable range |

Six requirements, each of which the stage must satisfy rather than assume:

- **The scorecard is never edited.** A score shift is not a change to a
  characteristic's points, and a scaling change is not a re-fit. Model Risk must
  be able to see the model's own answer and the policy overlay as two separate
  things, because the validation opinion covers only the first.
- **The unadjusted values survive.** `score_unadjusted`,
  `probability_of_default_unadjusted` and the unadjusted grade are recorded on
  every application alongside the adjusted ones, whatever the outcome.
- **The order is declared, not emergent.** A score shift then a PD multiplier is
  not the same as a PD multiplier then a score shift, because calibration sits
  between them. The composed order is part of the adjustment set's definition and
  is recorded per application.
- **Scope is declared** — scorecard, segment, channel, product, grade range — and
  an overlay evaluated outside its scope is an error, not a silent no-op.
- **Every overlay expires.** A review date is mandatory. The failure mode is a
  tightening applied in one bad quarter, still in force four years later, priced
  into every offer, with nobody able to say what unwinding it would do.
- **The flow runs with the stack disabled**, through this same implementation and
  not a second one, because comparing the base scorecard's predictions against
  realised outcomes is how the model is monitored — and an overlay in the
  measurement makes the model look accurate when it is not.

The challenger scorecard is overlaid **separately or not at all**, declared per
adjustment set. A challenger carrying the champion's overlays measures the
overlay, not the challenger.

*Client W under adjustment set 41: W originated through channel 2, so overlay 3
— the ×1.35 odds multiplier scoped to channel 4 — is **evaluated and does not
apply**. Neither does overlay 1, scoped to new-to-bank, W being an existing
client. `score` 618 and `probability_of_default` 5.80% therefore equal their
unadjusted counterparts and `risk_grade` stays 7. The record still names set 41
and lists both overlays as evaluated-not-applied with the scope test that
excluded each — because "no overlay applied" and "no overlay was in force" are
different facts, and only the first is true here.*

*The counterfactual must be answerable: had W originated through channel 4, the
multiplier would have taken PD to 7.83% and `risk_grade` to 8, moving the rate
cell and the whole offer set with it. That movement would have come entirely
from policy rather than from anything about W — which is why §5.4.1 requires the
unadjusted values on every application rather than only on the ones an overlay
touched.*

### 5.5 Policy caps — the waterfall

**Preconditions.** `risk_grade` assigned.

**Determines.** Three ceilings that constrain everything downstream:

| Ceiling | Seed value | Direction |
|---|---|---|
| `amount_cap` | R500 000 (product maximum) | May only be reduced |
| `term_cap` | 84 months | May only be reduced |
| `worst_acceptable_grade` | 12 | May only be reduced (tightened) |

**The register.** There are currently **52 rules**. The count has varied between
44 and 58 over the last two years and the register is explicitly sized for 40–60.
Ownership is split four ways: Credit Risk Policy (31), Unsecured Lending Product
(12), Credit Systems (6), Financial Crime (3). Each rule has an identifier, an
owner, a class, a sequence position, an applicability condition, the ceiling it
acts on, its value expression, its effective dates and its approval reference.
**The register is reordered and added to quarterly.** Reordering matters, because
a rule that reduces to a percentage of the current value gives a different answer
depending on what ran before it.

Each rule may do one of three things: reduce one ceiling, decline outright, or do
nothing. Representative rules:

| `rule_id` | Owner | Class | Acts on | Example condition and effect |
|---|---|---|---|---|
| CAP-0010 | Product | product | amount | Product maximum R500 000 |
| CAP-0025 | Product | product | — | Channel availability; declines on a closed channel |
| CAP-0100 | Credit Risk Policy | appetite | amount | Amount by grade, from the appetite grid: grade 1 → R500 000, grade 7 → R150 000, grade 10 → R45 000, grade 12 → decline |
| CAP-0118 | Credit Risk Policy | appetite | term | Term by grade: grades 1–4 → 84, 5–7 → 72, 8–9 → 60, 10–11 → 36 |
| CAP-0140 | Product | policy | amount | First loan for a new-to-bank client → R60 000 |
| CAP-0175 | Credit Risk Policy | policy | amount | Employment tenure: under 24 months → R70 000; under 6 months → R25 000; under 3 months → decline |
| CAP-0210 | Credit Risk Policy | policy | amount, grade | Arrears history: any account 3+ months in arrears in the last 12 → R40 000 and tighten to grade 8; any 2-month arrears in the last 24 → R120 000 and tighten to grade 9 |
| CAP-0240 | Credit Risk Policy | policy | amount | Enquiry velocity: 6+ credit enquiries in 60 days → R30 000; 10+ → decline |
| CAP-0305 | Credit Systems | exposure | amount | Employer or sector on the exposure watchlist → R50 000 |
| CAP-0330 | Product | policy | amount, term | Channel cap: broker → R150 000 and 60 months; partner → R100 000 |
| CAP-0361 | Credit Systems | exposure | amount | Group exposure headroom: group limit less existing internal exposure |
| CAP-0420 | Credit Committee | campaign | amount | **The one rule permitted to raise a cap** |

**The uplift rule.** CAP-0420 is the single exception to "reduce only". Where a
valid `campaign_id` is present and the campaign is a Credit-Committee-authorised
pre-approved campaign, the rule may raise `amount_cap` by up to **25%**, subject
to an absolute authority ceiling of **R250 000**, and subject to never exceeding
a cap set by any rule of class `regulatory`. It cannot raise `term_cap` or relax
`worst_acceptable_grade`. Its authority reference is recorded on every
application where it fires. If the uplift is *restrained* by a ceiling — that is,
if it was authorised to raise to a value it was not permitted to reach — both the
authorised value and the restraining rule must be recorded.

**The attribution requirement.** This is the hard part, and it is a requirement,
not a nicety. For each of the three ceilings, after the whole register has run,
it must be answerable:

1. **What is the final value?**
2. **Which rule set it?** Exactly one answer, unambiguous, including in the case
   where two rules reduced to the same value — in which case the earlier one in
   sequence is the binder and the later one is recorded as *coincident*.
3. **What was the full chain?** Every successive value the ceiling took, with the
   rule that produced it, in order.
4. **Which rules were evaluated and did not bind**, and which were **not
   applicable**, and why. A rule that did not apply and a rule that applied but
   did not bind are different facts, and a policy owner asking "is my rule doing
   anything?" needs both.

*Client W's `amount_cap` chain:*

| Seq | `rule_id` | Applicable | Before | After | Status |
|---|---|---|---|---|---|
| 1 | CAP-0010 | yes | — | R500 000 | seed |
| 3 | CAP-0100 | yes | R500 000 | R150 000 | **bound** |
| 9 | CAP-0140 | no (existing client) | R150 000 | R150 000 | not applicable |
| 12 | CAP-0175 | yes (41 months) | R150 000 | R150 000 | evaluated, did not bind |
| 17 | CAP-0210 | yes (2-month arrears, 14 months ago) | R150 000 | R120 000 | **bound** |
| 21 | CAP-0240 | yes (3 enquiries in 60 days) | R120 000 | R120 000 | evaluated, did not bind |
| 26 | CAP-0305 | no (employer not listed) | R120 000 | R120 000 | not applicable |
| 33 | CAP-0330 | yes (channel 2) | R120 000 | R120 000 | evaluated, did not bind |
| 38 | CAP-0361 | yes (limit R180 000, exposure R104 000) | R120 000 | R76 000 | **bound** |
| 52 | CAP-0420 | yes (campaign 4471, authority CC-2026-14) | R76 000 | **R95 000** | **raised** |

Final `amount_cap` **R95 000**, bound by CAP-0420. Chain: 500 000 → 150 000 →
120 000 → 76 000 → 95 000. `term_cap` 72, bound by CAP-0118.
`worst_acceptable_grade` 9, bound by CAP-0210; grade 7 passes.

**Emits.** Three ceilings, three chains, 52 per-rule verdicts, and any outright
decline reasons.

**Overlays on the ceilings.** The register itself is not the only way a ceiling
moves. A cap overlay reduces `amount_cap` or `term_cap` by a percentage or an
absolute amount over a declared scope — illustratively "amount cap × 0.80 on
channel 4 for the quarter" — and enters the chain as its own row, attributed to
the adjustment set rather than to a rule. Two constraints: a cap overlay may only
**reduce** a ceiling, so it can never do what CAP-0420's authority-bounded uplift
does; and the chain must remain complete, because "your cap was R95 000, reduced
to R76 000 by a policy overlay approved under CC-2026-31, expiring 2027-01-31" is
a different answer to the client and to the regulator than "a rule bound it".

### 5.6 Affordability

**Preconditions.** `risk_grade` assigned (the affordability buffer varies by
grade).

This project does not compute affordability. It consumes project 02's published
assessment and takes from it `max_affordable_instalment`, `discretionary_income`,
`affordability_verdict_code`, `living_expenses`, `existing_obligations` and the
version identifiers of every table project 02 resolved. Those version identifiers
travel into this project's decision record unchanged — a replay of a granting
decision must pin the affordability tables as tightly as its own.

An `affordability_verdict_code` of *fail* declines with reason 1310. *Marginal*
continues but forces `outcome_code` to "approve with conditions" and reduces the
offer set to terms of 24 months or less. *Indeterminate* refers, queue 5.

*Client W: net income R24 800.00, living expenses R9 450.00 (norm floor applied),
existing obligations R6 120.00, discretionary income R9 230.00, appetite haircut
12% at grade 7, **`max_affordable_instalment` R4 350.00**, verdict pass.*

### 5.7 Pricing

**Preconditions.** A candidate amount, a candidate term, a risk grade.

This stage is the arithmetic the search calls repeatedly. It takes an amount and
a term and returns an instalment. It has four parts, and their order is fixed
because each depends on the last.

**(a) The rate.** Read from the Flex Loan rate card: **96 amount bands × 55 terms
× 12 grades = 63 360 cells.**

Amount bands are not uniform, because pricing granularity matters most at small
amounts:

| Range | Band width | Bands |
|---|---|---|
| R2 000 – R4 999 | R500 | 6 |
| R5 000 – R49 999 | R1 000 | 45 |
| R50 000 – R149 999 | R5 000 | 20 |
| R150 000 – R349 999 | R10 000 | 20 |
| R350 000 – R500 000 | R30 000 | 5 |

Term columns cover every month from 6 to 60 — 55 columns. Terms of 61 to 84
months are priced from the 60-month column plus a **long-term loading** held in a
separate 2 × 12 table (61–72 and 73–84, by grade). That this is a second table
rather than 24 more columns is a Treasury decision, not a technical one, and it
must remain visible as such.

Rates in force at `decision_date` range from 12.90% (grade 1, top amount band,
60 months) to 27.50% (grade 12, R2 000–R2 499, 6 months).

**The statutory ceiling.** The National Credit Act prescribes a maximum interest
rate for unsecured credit transactions expressed as a margin over the repo rate.
At a repo rate of 7.00% the ceiling is 28.00% nominal annually. The worst cells
on the card therefore sit 50 basis points inside the ceiling. **A repo cut lowers
the ceiling and can put cells out of compliance without anybody touching the
card.** A 25 basis point cut moves the ceiling to 27.45% and immediately puts
roughly 1 100 cells in breach. The ceiling must be re-evaluated against the whole
card on every repo move and on every card version, and no offer may be priced
from a cell above the ceiling in force at `decision_date` — the flow must fail
loudly rather than lend at an unlawful rate.

**(b) Fees.** `initiation_fee` is a piecewise function of the advance:

> R180 plus 10.00% of the advance in excess of R1 000, subject to a statutory
> ceiling of R1 350, both excluding indirect tax at 15%.

The ceiling binds at an advance of **R12 700** exactly. Below that the fee rises
with the amount; above it the fee is flat. That kink sits *inside* the rate band
R12 000–R12 999, so within a single rate band the instalment's dependence on the
amount changes shape. The fee is **capitalised into the advance** — the client
receives `offered_amount` and finances `offered_amount + initiation_fee
including tax`. Those two numbers are different and must never be confused; the
disclosure names both.

`monthly_service_fee` is R72.00 excluding tax, R82.80 including, statutorily
capped. Both fees are effective-dated and adjusted annually by Compliance.

**(c) Credit life premium.** Charged monthly, per R1 000 of the amount financed —
which includes the capitalised initiation fee, so the premium depends on the fee,
which depends on the amount. The rate is read from a **14 age bands × 8 term
bands × 6 employment types = 672 cell** table, with values from R1.85 to R4.50
per R1 000. R4.50 is the statutory ceiling for unsecured credit; any cell
exceeding it is a card defect, not a price.

Term band rates for a 38-year-old on permanent employment: 6–12 months R1.95,
13–18 R2.20, 19–24 R2.55, 25–36 R2.85, 37–48 R3.00, 49–60 R3.10, 61–72 R3.35,
73–84 R3.60.

The client's statutory right to **substitute their own policy** must be handled:
where `credit_life_substitution_declared` is true, the premium is zero, the
instalment drops accordingly, and the offer carries a condition that proof of
cover is produced before disbursement. The affordability test is performed
against the *lower* instalment, and the record states that it was.

**(d) The instalment.** Ordinary monthly annuity on the amount financed at the
nominal rate over the term, plus the monthly service fee, plus the credit life
premium, rounded to the cent. Total cost of credit is the instalment times the
term. Effective annual rate is the annualised internal rate of return on the
amount *advanced* against the full instalment stream — including fees and
premium, which is why it exceeds the nominal rate substantially at short terms.

**(e) The rate add-on.** A rate overlay adds basis points over a declared range
of the card — a grade range, a term range, an amount-band range, a channel — and
is applied to the cell's rate *after* lookup and *before* the annuity. It exists
because reissuing a 63 360-cell card to move one grade's pricing by 50 basis
points is a Treasury release, a validation cycle and a review artefact, whereas a
repricing decision is frequently made in a week. Four requirements: the add-on
may not push the resulting rate above the statutory ceiling in force, and the
check happens after the add-on rather than only at card validation; the card's own
cell value and the add-on are recorded separately, so a regulator asking whether
pricing matched the published card gets "cell 18.50% plus a 75 basis point
overlay approved under CC-2026-22" rather than a single unexplained 19.25%; the
add-on participates in the solve like any other rate movement, including its
effect on band-edge non-monotonicity; and an add-on may not be merged into the
card, ever, because that destroys the distinction it exists to preserve.

**Emits, per evaluation.** `nominal_annual_rate`, `rate_cell_id`,
`rate_card_version`, `initiation_fee`, `monthly_service_fee`,
`credit_life_premium`, `instalment`, `total_cost_of_credit`,
`effective_annual_rate`, and flags for every cap that bound.

### 5.8 The solve

**This is the heart of the specification.**

**Preconditions.** `amount_cap`, `term_cap`, `worst_acceptable_grade`,
`max_affordable_instalment`, `risk_grade`, and pricing available.

**Determines.** For each permitted term, the **largest amount the client can be
offered that they can afford** — or the fact that no such amount exists.

**The domain.** Advances round to **R100**. The candidate amounts run from the
product minimum R2 000 to `min(requested_amount, amount_cap, R500 000)`. Over the
full range that is 4 981 candidates; across 9 terms, 44 829 candidate
evaluations if every one were tried. It is not division.

**Why it is not division.** Five reasons, each independently sufficient:

1. **The rate is a step function of the amount.** Reducing the amount can move it
   into a band priced *worse*, so the instalment goes **up** when the amount goes
   **down**. The instalment is not monotone in the amount.
2. **The initiation fee is capitalised and piecewise.** The amount financed is
   not the amount advanced, and the gap between them is itself a function of the
   amount, with a kink at R12 700.
3. **The credit life premium is charged on the amount financed**, so it inherits
   both the fee's piecewise shape and the term band's step.
4. **Rounding to R100 interacts with band selection.** Rounding a candidate down
   can cross a band edge; rounding it up can breach a cap.
5. **Caps apply to the result**, and a capped result may land in a different band
   again.

**The worked failure.** Client W's grade-7 twin at grade 9, 60 months, with
`max_affordable_instalment` **R1 560.00**. The rate card prices R50 000–R54 999
at 18.25% and R49 000–R49 999 at 19.75% — Treasury rewards crossing R50 000 with
150 basis points:

| Amount | Band | Rate | Financed | Instalment | Affordable? |
|---|---|---|---|---|---|
| R51 000 | 50 000–54 999 | 18.25% | R52 552.50 | R1 587.36 | no |
| R50 100 | 50 000–54 999 | 18.25% | R51 652.50 | R1 561.59 | no |
| **R50 000** | 50 000–54 999 | 18.25% | R51 552.50 | **R1 558.73** | **yes** |
| R49 900 | 49 000–49 999 | 19.75% | R51 452.50 | R1 598.33 | no |
| R49 000 | 49 000–49 999 | 19.75% | R50 552.50 | R1 571.82 | no |
| R48 600 | 48 000–48 999 | 19.75% | R50 152.50 | R1 560.04 | no, by 4 cents |
| R48 500 | 48 000–48 999 | 19.75% | R50 052.50 | R1 557.09 | yes |

The feasible set is **{R2 000 … R48 500} ∪ {R50 000}**. The correct answer is
**R50 000**. A search that halves the interval on the assumption that
affordability is monotone returns R48 500 and under-lends by R1 500. A search
that scans downward from the requested amount and stops at the first affordable
point returns R50 000 — but only because it started above it. Neither property is
safe to assume; both must be *demonstrated*.

**Requirements on the search.** Not how it works — what must be true of it.

1. **Bounded.** A hard ceiling of **24 pricing evaluations per term**, 216 per
   application. Exceeding it is not permitted. If the ceiling is reached without
   a proven maximum, the term yields no offer and the application's outcome
   becomes `refer`, queue 6, reason 1420 — an explicit, recorded, monitored
   outcome, not a silently truncated answer.
2. **Terminating.** It must terminate on every input, including a domain of one
   candidate, an empty domain, and a domain where nothing is affordable.
3. **Deterministic.** No randomness, no dependence on wall-clock time, no
   dependence on iteration order over an unordered collection, no dependence on
   floating-point accumulation order that could differ between the real-time and
   batch paths. Same inputs, same versions, same `decision_date` ⇒ same amount,
   to the rand.
4. **Correct.** The returned amount must be the **true maximum** over the R100
   grid — the largest feasible candidate, not the largest candidate below the
   first infeasible one.
5. **Tie-broken.** Where the objective admits more than one answer, the rule is
   stated and fixed: prefer the larger amount; among equal amounts, prefer the
   lower `total_cost_of_credit`; among equal costs, prefer the shorter term. The
   tie-break must be a parameter of the specification, not an accident of
   evaluation order.
6. **Attributed.** Each term's result records the number of evaluations
   performed, the amounts evaluated, and the **binding constraint**:

   | Code | Constraint |
   |---|---|
   | BIND-AFF | Affordability — the instalment ceiling |
   | BIND-CAP | The policy `amount_cap` |
   | BIND-REQ | The requested amount |
   | BIND-MIN | The product minimum, R2 000 |
   | BIND-MAX | The product maximum, R500 000 |
   | BIND-TCR | The total cost ratio threshold |
   | BIND-DUP | The scheduled in duplum test |
   | BIND-CEIL | No cell at or below the statutory rate ceiling |
   | BIND-EXH | Evaluation ceiling reached |

7. **Re-checkable.** **The offer must never fail affordability when re-checked
   from scratch.** An independent re-derivation — read the rate for the final
   amount and term, compute the fee, compute the premium, compute the instalment,
   compare to `max_affordable_instalment` — must pass, carrying no state from the
   search. This is a **mandatory acceptance test**, run over a 250 000-application
   regression set, with **zero** tolerated failures. It is mandatory because the
   failure mode it catches is silent: an offer that was affordable at the moment
   the search evaluated it but is not affordable at the amount that was finally
   written, because a cap moved it into another band afterwards.
8. **Exhaustively verified, periodically.** For a 10 000-application sample, the
   search's answer is compared against evaluating **every** R100 candidate in the
   domain at every permitted term. Zero disagreements permitted. This runs on
   every rate card version, because a new card can introduce a new band-edge
   inversion that no previous card had.

**Emits.** Per permitted term: the maximum feasible amount or an explicit
"no feasible amount" with its reason, the evaluation count, the binding
constraint code, and the full pricing set for the winning amount.

#### 5.8.1 The solve under overlays

The search runs on **adjusted** values throughout — the adjusted grade selects
the rate row, the rate add-on moves the rate, the cap overlay narrows the domain,
and project 02's buffer overlay moves `max_affordable_instalment`. Three
consequences the implementation must handle rather than discover:

1. **The overlay stack is part of the search's identity.** The same client, the
   same inputs and the same code produce a different winning amount under a
   different stack. A replay that does not pin the stack reproduces a plausible
   wrong answer, which is worse than failing. `adjustment_set_id` is recorded on
   every offer, and the effect of each overlay on the chosen offer is recorded
   individually.
2. **An overlay can change the shape of the problem, not only its answer.** A
   rate add-on scoped to a term range alters where band-edge inversions fall, so
   the non-monotonicity the search must tolerate is a property of the card *plus*
   the stack, and the declared inversion list in §6.1 is incomplete without it.
3. **The unadjusted offer is not computed.** Recording `score_unadjusted` is
   cheap; re-running the whole search with the stack disabled to find out what
   the offer would have been is not, at 55 000 applications a day. The
   requirement is therefore narrower and must be stated as such: the search runs
   once, on adjusted values, and the counterfactual offer is produced on demand
   for sampling, investigation and impact estimation — by the same
   implementation, never a second one.

### 5.9 Offer set construction

**Preconditions.** The solve has run for every permitted term.

**Determines.** What the client is actually shown.

**Permitted terms.** 6, 12, 18, 24, 36, 48, 60, 72, 84 — nine values, filtered by
`term_cap` and by a per-segment permitted term list (new-to-bank clients are not
offered 84 months). Client W's `term_cap` of 72 removes 84 before any pricing
happens, and that removal is recorded with its reason.

**Client W's candidate set**, at `max_affordable_instalment` R4 350.00 and
`amount_cap` R95 000:

| Term | Amount | Rate | Instalment | Total cost | Ratio | EAR | Bound by |
|---|---|---|---|---|---|---|---|
| 6 | R22 100 | 22.50% | R4 333.71 | R26 002.26 | 1.18 | 76.6% | BIND-AFF |
| 12 | R43 200 | 21.50% | R4 347.89 | R52 174.68 | 1.21 | 43.1% | BIND-AFF |
| 18 | R61 900 | 20.50% | R4 347.05 | R78 246.90 | 1.26 | 36.0% | BIND-AFF |
| 24 | R77 900 | 20.50% | R4 348.64 | R104 367.36 | 1.34 | 34.3% | BIND-AFF |
| 36 | R95 000 | 19.00% | R3 897.20 | R140 299.20 | 1.48 | 31.1% | BIND-CAP |
| 48 | R95 000 | 18.75% | R3 246.67 | R155 840.16 | 1.64 | 30.4% | BIND-CAP |
| 60 | R95 000 | 18.50% | R2 860.25 | R171 615.00 | 1.81 | 29.7% | BIND-CAP |
| 72 | R95 000 | 19.00% | R2 663.33 | R191 759.76 | 2.02 | 30.3% | BIND-CAP |
| 84 | — | — | — | — | — | — | term above cap |

**Minimum viable offer rules.** Applied after the solve, each with its own reason
code, each independently tunable by Unsecured Lending Product:

| Rule | Threshold | Effect on client W |
|---|---|---|
| Minimum amount | No offer below **R2 000** | none |
| Minimum instalment | No offer whose instalment is below **R150.00** — uneconomic to service | none |
| Total cost ratio | No offer whose `total_cost_of_credit / offered_amount` exceeds **1.85** | **suppresses 72 months** (2.02) |
| Scheduled in duplum | No offer whose total interest, fees and premiums over the term exceed the amount advanced — a conservative pre-agreement application of the in duplum rule | **also suppresses 72 months** (charges R96 759.76 against an advance of R95 000) |
| Effective rate ceiling | No offer whose `effective_annual_rate` exceeds **60.00%** | **suppresses 6 months** (76.6%) |

The 6-month suppression deserves a note, because it looks like a defect and is
not. A once-off R1 552.50 fee recovered over six months is, annualised, enormous;
the total cost ratio of 1.18 is the lowest in the set and would never catch it.
The two rules measure different things and both are needed. Where an offer is
suppressed by more than one rule — as 72 months is — **all** applicable reasons
are recorded, ranked, and the most severe is the one reported.

**Deduplication.** Two surviving offers whose amounts are equal and whose
instalments differ by less than 2% collapse to the one with the lower total cost.
Client W's 36, 48 and 60 month offers are all at R95 000 but their instalments
differ by far more than 2%, so all three survive. The tolerance is a parameter.

**Ranking and recommendation.** The surviving offers are ranked and one is
flagged `is_recommended`. The objective is a **tunable parameter with three
values**, and the business has changed it twice since go-live:

| Objective | Definition | Client W's recommendation |
|---|---|---|
| `largest_amount` | Largest `offered_amount`; ties broken by lowest total cost | **36 months, R95 000** (ties with 48 and 60 at R95 000; 36 has the lowest total cost) |
| `lowest_total_cost` | Lowest `total_cost_of_credit` per rand advanced | **12 months, R43 200** (ratio 1.21) |
| `best_expected_value` | Highest expected margin: (total cost − advance − funding cost) × (1 − PD over the term) − LGD × EAD × PD over the term, with PD over the term derived from the 12-month PD, LGD 72%, average EAD 55% of the advance, funding cost 8.75% | **60 months, R95 000** (R30 157, against R25 509 at 48 months, R20 227 at 36, R13 358 at 24, R8 759 at 18, R5 504 at 12) |

Three objectives, three different recommendations, from one offer set. The
objective must be changeable without a code release, and which objective was in
force must be recorded on every application — a client asking "why did you
recommend 60 months?" in 2029 is asking about a parameter value in 2026.

**The output is ragged.** Between **0 and 9 offers** per application. Client W
receives **six**: 12, 18, 24, 36, 48 and 60 months. Zero offers with no policy
decline is a distinct outcome — "declined on affordability" — and must not be
reported as an approval with an empty set.

**Emits.** The ordered offer set, per-offer full pricing, `is_recommended`, the
objective in force, and a suppression record for every term that produced no
offer, with its reason.

### 5.10 Final validation

**Preconditions.** A recommended offer exists.

**Determines.** Whether the Bank is willing to be bound by it.

The chosen offer is **re-derived end to end from its own amount and term**,
carrying nothing forward from the solve, and every derived value is asserted:

| # | Assertion |
|---|---|
| 1 | The rate equals the rate card cell for the final amount band, the final term and the risk grade, in the card version in force at `decision_date` |
| 2 | The rate is at or below the statutory ceiling in force at `decision_date` |
| 3 | The initiation fee equals the piecewise calculation and is at or below the statutory ceiling |
| 4 | The monthly service fee equals the capped value |
| 5 | The credit life premium equals the table lookup for the age band, term band and employment type, and is at or below R4.50 per R1 000 |
| 6 | The instalment recomputes **to the cent** |
| 7 | The instalment is at or below `max_affordable_instalment` |
| 8 | The total cost satisfies the scheduled in duplum test |
| 9 | The total cost ratio is at or below the threshold |
| 10 | The amount is at or below every ceiling that bound, and at or above the product minimum |
| 11 | The term is at or below `term_cap` and in the permitted list for the segment |
| 12 | The grade is at or better than `worst_acceptable_grade` |
| 13 | The amount is a multiple of R100 |
| 14 | Every table version referenced is the one `decision_date` resolves to |

**Any mismatch is a hard failure.** Not a warning, not a logged anomaly, not a
value quietly corrected. The offer is withdrawn, the application becomes `refer`
to queue 7, and an incident is raised. The reason is not fastidiousness: an offer
presented to a client is an offer the Bank may be held to. A cent of disagreement
between the pricing and the contract is a reconciliation break; a rate above the
ceiling is an unlawful agreement; a fee above the cap is a refundable overcharge
across every account written since the defect appeared. The cost of shipping a
wrong offer is unbounded and the cost of referring one is a few minutes of an
underwriter's time.

The failure rate of this stage is itself a monitored metric. It should be zero.
Any non-zero rate is a defect in an upstream stage, and the validation exists to
find it before a client does.

### 5.11 Disclosure outputs

**Preconditions.** A validated offer, or a decline.

**On approval**, the pre-agreement quotation values, each disclosed separately
and each traceable to the calculation that produced it:

- `offered_amount` — what the client receives
- `initiation_fee` and its indirect tax, and the statement that it is capitalised
- the amount financed — advance plus capitalised fee — stated explicitly as a
  different number from the advance
- `nominal_annual_rate`, and whether it is fixed for the term
- `monthly_service_fee` including tax
- `credit_life_premium`, the cover it buys, and **the client's right to
  substitute their own policy**, stated in the disclosure and not only in the
  terms
- `instalment`, the number of instalments, the first and final payment amounts
  and dates
- `total_cost_of_credit`, broken into capital, interest, initiation fee, service
  fees and premiums — five figures that must sum to the total
- `effective_annual_rate`
- the validity period of the quotation

**On decline**, the reasons. The client is entitled to be told why credit was
refused. The reason set is assembled from every stage that contributed — gate
failures, fraud decline, policy rule declines, affordability failure, and the
scorecard's top negative contributions where the decline was score-driven —
ranked by the registry's severity order, with `primary_reason_code` being the one
communicated. Up to four reasons are communicated; all are recorded. Where the
decline was score-driven, the communicated reasons must be the characteristic
contributions, phrased in the registry's client-facing wording, in the client's
language of record.

**On referral**, the queue, the reason, and the full assessment to date so that
the underwriter does not start from nothing.

---

## 6. Parameters and tables

### 6.1 Tables

| Table | Dimensions | Cells | Owner | Cadence | Source |
|---|---|---|---|---|---|
| **Flex Loan rate card** | **96 amount bands × 55 terms × 12 grades** | **63 360** | Treasury | Monthly, plus mid-month patches on repo moves | Spreadsheet |
| Long-term loading | 2 term bands × 12 grades | 24 | Treasury | Monthly | Spreadsheet |
| Amount band definitions | 96 bands × 3 attributes | 288 | Treasury | With the card | Spreadsheet |
| Statutory rate ceiling | 1 margin over repo, effective-dated | ~4/year | Compliance | On repo move | Regulator |
| Initiation fee schedule | base, marginal rate, ceiling, tax rate | 4 | Compliance | Annual | Regulator |
| Service fee cap | 1 value, effective-dated | 1 | Compliance | Annual | Regulator |
| Credit life premium rates | 14 age × 8 term × 6 employment | **672** | Credit Risk Policy | Annual | Insurer |
| Credit life statutory cap | 1 value per R1 000 | 1 | Compliance | On gazette | Regulator |
| Scorecard definitions | 4 segments × 45 characteristics × 8 bins | ~1 440 | Model team | On model release | Model documentation |
| Challenger scorecard | 1 × 45 × 8 | ~360 | Model team | On model release | Model documentation |
| Calibration parameters | 5 scorecards × (factor, offset, anchor) | 15 | Model team | On model release | Model documentation |
| Grade boundaries | 4 segments × 12 grades | 48 | Credit Risk Policy | Semi-annual | Model team |
| Appetite grid slice | 12 grades × 6 segments × 5 values | 360 | Credit Committee | Quarterly | Internal |
| Cap register | 52 rules × 11 attributes | 572 | Four owners | Quarterly | Internal |
| Permitted term list | 4 segments × 9 terms | 36 | Product | Weekly | Internal |
| Channel availability | 6 channels × 4 segments × 3 attributes | 72 | Product | Weekly | Internal |
| Minimum viable offer thresholds | 5 rules × 2 attributes | 10 | Product | Weekly | Internal |
| Exposure watchlist | ~2 400 employer and sector identifiers | 2 400 | Credit Systems | Monthly | Internal |
| Reason codes referenced | 96 of the library's 380 | — | Compliance | Monthly | Library |

**Two properties are required of every table here**, inherited from the library:
cell-level attribution — given an outcome, which cell of which version was read —
and diffability — a new rate card version produces a review artefact naming the
cells that changed, by how much, and the aggregate impact on the last 30 days'
applications re-priced. "Treasury sent a new spreadsheet" is not a reviewable
change, and at 63 360 cells nobody is going to eyeball it.

**Rate card validation** runs on every version before it is permitted to go live:

1. Every cell is populated — a null rate is a defect, never a "use the neighbour".
2. Every cell is at or below the statutory ceiling in force on its effective date.
3. Rates are monotone non-increasing across grades within an amount-band/term
   cell — a better grade is never priced worse. Violations are blocked.
4. Band-edge inversions — where the adjacent lower band is priced higher, as at
   R50 000 — are **permitted but must be declared**, listed in the review
   artefact, and counted. The current card has 41 of them, deliberately, and the
   solve must handle every one. An undeclared inversion blocks the card.
5. The mid-month patch path produces the same artefact as a full refresh.

### 6.2 Parameters

| Parameter | Value | Owner | Cadence |
|---|---|---|---|
| Bureau staleness tolerance | 45 days | Credit Risk Policy | Rare |
| Fraud call budget | 800 ms | Financial Crime | Rare |
| Fraud bypass amount ceiling | R15 000 | Financial Crime | Annual |
| Fraud bypass tenure floor | 24 months | Financial Crime | Annual |
| Challenger traffic share | 10% | Model team | Frequently |
| Uplift maximum | 25% | Credit Committee | Quarterly |
| Uplift authority ceiling | R250 000 | Credit Committee | Quarterly |
| Rounding unit for advances | R100 | Credit Systems | Never |
| Rounding direction | Down to the nearest R100 | Credit Systems | Never |
| Evaluation ceiling per term | 24 | Credit Systems | On evidence |
| Tie-break order | amount desc, total cost asc, term asc | Product | Rare |
| Minimum offer amount | R2 000 | Product | Annual |
| Minimum instalment | R150.00 | Product | Annual |
| Total cost ratio threshold | 1.85 | Product | Quarterly |
| Effective rate suppression ceiling | 60.00% | Product | Quarterly |
| Deduplication instalment tolerance | 2% | Product | Rare |
| Recommendation objective | one of three | Product | Changed twice since go-live |
| LGD, average EAD, funding cost | 72%, 55%, 8.75% | Treasury | Quarterly |
| Reasons communicated | 4 | Compliance | Rare |

Three ownership classes must remain distinguishable, because they are changed by
different people on different cadences under different approvals: statutory
values that Compliance changes on gazette, policy values that Credit Risk Policy
changes quarterly under Credit Committee approval, and product values that
Unsecured Lending changes weekly under a product owner's signature. A product
manager must be able to move the total cost ratio threshold and must **not** be
able to move the statutory fee cap.

### 6.3 The adjustment set

The overlays of §5.4.1, §5.5, §5.7(e) and project 02's buffer are one artefact
with one version, and it does not belong to any of the tables it modifies.

| Attribute | Notes |
|---|---|
| Overlay id, description, rationale | The rationale is a required field, not a courtesy |
| Kind | Score shift, scaling change, odds multiplier, boundary shift, cap reduction, rate add-on |
| Scope | Scorecard, segment, channel, product, grade range, amount range, term range |
| Magnitude | Points, ratio, percentage points, basis points |
| Position in the stack | An integer; composition order is declared |
| Owner and approval reference | A named person and a committee minute |
| Effective from, effective to | Both required |
| Review date | Required, and enforced — an overlay reaching it without renewal surfaces |
| Enabled | So the stack can be run off without deleting its definition |

Live count runs 6 to 20 across this product. The register is changed under Credit
Committee approval, on no fixed cadence — sometimes twice in a month — which is
precisely why it cannot ride inside the artefacts it modifies, each of which has
its own slower cadence and its own different approver.

Two prohibitions worth stating as requirements. An overlay is **never merged**
into the base artefact "to simplify", because merging destroys the separation
between what the model said and what policy decided. And an overlay may not be
expressed as an edit to a rate card cell, a grade boundary or a characteristic's
points, for the same reason.

**A fourth ownership class**, then, alongside §6.2's three: overlays, owned by
the Credit Committee, changed ad hoc, approved individually, expiring
individually. A product manager may not create one; Treasury may not create one
over the card it owns.

---

## 7. Outputs

### 7.1 To the caller

| Field | Notes |
|---|---|
| `outcome_code` | Approve, approve with conditions, refer, decline |
| `offers` | 0..9, each with amount, term, rate, fees, premium, instalment, total cost, effective rate, `is_recommended`, and the constraint that bound it |
| `decline_reason_codes`, `primary_reason_code` | Ranked; client-facing wording resolved |
| `referral_queue_code` | Where a referral goes, with the reason |
| `risk_grade`, `probability_of_default` | For downstream use |
| `max_affordable_instalment` | Echoed, with its source version |
| Disclosure block | §5.11 |
| `assessment_id` | The key to the full record |

### 7.2 Persisted

Everything needed to re-derive the decision without the originating systems:

- All 14 eligibility gate verdicts and the values they tested
- Consent identifiers; the fraud verdict, its latency and its handling path
- The raw bureau response identifier, the normalisation version, every defect,
  the data-quality verdict
- The segment, `scorecard_id`, `scorecard_version`, all 45 characteristic values,
  bins and point contributions, the score, the calibration parameters, the PD,
  the grade boundary version and the grade — and the same set for the challenger
  where it ran
- All three cap chains in full, all 52 per-rule verdicts, the uplift authority
  reference, and any restraint
- Project 02's affordability outputs and every table version it resolved
- **Every pricing evaluation the search performed**, in order: the candidate
  amount, the rate cell read, the fee, the premium, the instalment and the
  feasibility verdict — because "why R50 000 and not R49 900?" is a question that
  is only answerable from the evaluations
- Per term: the evaluation count, the binding constraint, the result
- The full candidate offer set before suppression, every suppression with its
  reason, the deduplication decisions, the objective in force, the ranking
- All 14 final validation assertions with their computed and expected values
- The version identifier of every table touched, without exception
- Timings per stage

---

## 8. Non-functional requirements

| Requirement | Value |
|---|---|
| Real-time throughput | 55 000 applications/day, peak 9/second sustained 20 minutes |
| Real-time latency | **p99 under 120 ms end to end**, excluding external calls (bureau, fraud, affordability's own external retrievals) |
| Latency budget | eligibility 3 ms · bureau normalisation 12 ms · scorecard, calibration, grading 8 ms · cap waterfall 6 ms · affordability 10 ms · **solve 55 ms** · offer set 8 ms · validation 12 ms · disclosure 6 ms |
| Batch throughput | 14 000 000 clients in a 6-hour window — 648 records/second sustained, up to 140 000 pricing evaluations/second |
| **Real-time / batch identity** | Given the same inputs, the same parameter set and the same `decision_date`, the two paths produce **identical** outputs. Stated as an acceptance criterion, not an aspiration. Verified monthly by replaying a 100 000-record sample of batch output through the real-time path with zero differences permitted. |
| Determinism | Identical inputs, versions and `decision_date` ⇒ identical outputs, bit for bit, on any host, in any order, at any concurrency |
| Cold start | No per-request compilation. The first request after a deployment is not materially slower than the thousandth. A 63 360-cell card must be resident and indexed before the first request, not on it. |
| Table refresh | A rate card refresh must not require a code deployment, and must not interrupt serving. A mid-month patch must be applicable within 2 hours of a repo announcement. |
| Version pinning | An application in flight when a card changes completes on the card it started with |
| Availability | 99.95% during business hours; a degraded mode that refers rather than declines when a dependency is unavailable |

---

## 9. Audit, evidence and explainability

Four demands, all of which have been made of real granting systems.

**1. A declined applicant demands reasons.** Within 20 business days, in the
client's language of record, the Bank must state why credit was refused. If the
decline was score-driven, the reasons are the scorecard's top negative
contributions, which means the per-characteristic points must exist and be
retrievable for an application that was declined months ago. If the decline came
from a policy rule, the rule must be nameable in client-facing terms. The
requirement that the reasons be *complete* — that a client who failed four gates
is told about four gates — is why §5.1 evaluates gates past the first failure.

**1a. Model Risk asks what the model said, and what policy did to it.** The
validation opinion covers the scorecard, not the overlays. A model monitoring
cycle compares predicted against realised default rates for the *base* model, so
it needs `score_unadjusted` and `probability_of_default_unadjusted` on every
application, and it needs them for declined applications too — where available —
because an overlay that declines the top of a segment changes the population the
model is measured on. Separately, the Credit Committee asks the opposite
question: of this quarter's decline rate movement, how much was the client
population, how much was the model, and how much was the overlay set the
committee itself approved. Answering requires the adjusted and unadjusted values
side by side on every record, not a reconstruction.

**2. A regulator tests whether pricing matched the published card.** A sample of
concluded agreements is selected and, for each, the regulator asks: what amount,
what term, what grade, what rate — and does that rate match the card the Bank
published for that date? Answering requires the rate card version, the specific
cell, the amount band that amount fell into, and the band definitions in force.
"The system computed 18.50%" is not an answer; "cell [band 68, term 60, grade 7]
of card version 2026-09-A, which the effective-dated resolution for 2026-09-14
selects, contains 18.50%" is.

**3. Internal Audit samples 200 approvals a quarter and re-derives each.** From
the stored record alone, with no access to the originating systems, each of the
200 must reproduce: the same grade, the same caps, the same solve result, the
same offer set, the same recommendation, the same instalment to the cent. A
finding is raised for any record where a table version was not captured, which is
how the library's requirement that *every* table version be recorded came about.

**4. A disputed instalment surfaces four years later.** A client in 2030 disputes
the instalment on an agreement written in 2026. The Bank must reproduce the 2026
calculation using the 2026 rate card, the 2026 fee caps, the 2026 credit life
table, the 2026 expense norms and the 2026 tax tables — **to the cent**. Every
table this project touches resolves against `decision_date`, and "today" never
appears anywhere in the logic.

**Two further explainability requirements** that are harder than they look:

- **"Why this amount?"** The answer is not a rule identifier; it is a chain. For
  client W at 24 months: the requested R120 000 was capped to R95 000 by the
  waterfall, and then affordability bound at R77 900 because R78 000 would have
  cost R4 354.01 against a ceiling of R4 350.00. Both halves must be
  reconstructable, which means the evaluations that proved R78 000 infeasible
  must be stored, not just the answer.
- **"Why is there no 72-month option?"** Because the total cost ratio would be
  2.02 against a threshold of 1.85, and because scheduled charges of R96 759.76
  would exceed the R95 000 advance. A branch consultant must be able to say this
  to a client's face without phoning anyone.

---

## 10. Acceptance criteria

1. Given an application and a `decision_date`, the flow produces either a decline
   with a complete, ranked reason set, or an offer set of 0 to 9 offers with
   exactly one flagged as recommended.
2. Every offer in every produced offer set passes an **independent from-scratch
   re-check** of affordability, pricing and every cap, over a 250 000-application
   regression set. **Zero** failures. This is the test named in §5.8 and it is not
   waivable.
3. For a 10 000-application sample, the solve's answer equals the answer from
   exhaustively evaluating every R100 candidate at every permitted term. Zero
   disagreements. Re-run on every rate card version.
4. No application performs more than 24 pricing evaluations for any single term,
   ever, on any input, including adversarial ones.
5. For any of the three ceilings on any application, the final value, the rule
   that bound it, the full chain of successive values, and the per-rule verdict of
   all 52 rules are retrievable from the stored record alone.
6. The band-edge non-monotonicity case in §5.8 is a named regression test, with
   all 41 declared inversions on the current card covered, and it fails loudly if
   a search returns R48 500.
7. The recommendation objective can be switched between its three values by a
   product manager without a code deployment, and the objective in force is
   recorded on every application.
8. A rate card refresh — 63 360 cells — is applied without a code deployment and
   produces a cell-level diff artefact with the aggregate impact of re-pricing the
   last 30 days' applications.
9. A mid-month patch touching 1 100 cells after a repo move is applied within two
   hours and validated against the new statutory ceiling.
10. Applications in flight at the moment of a card change complete on the card
    they started with.
11. Real-time and batch produce identical outputs on a 100 000-record monthly
    reconciliation sample. Zero differences.
12. p99 end-to-end latency is under 120 ms excluding external calls, measured
    over a full peak day.
13. 14 M clients complete the batch pre-assessment within 6 hours.
14. A 2026 decision replayed in 2030 reproduces to the cent, using only the
    stored record and the effective-dated artefacts.
15. Adding one rule to the cap register, in any sequence position, is a
    configuration change reviewed by its owner — not a release.
16. The final validation stage's failure rate is zero; any non-zero rate is
    treated as a production incident.
17. A Credit Risk Policy analyst can read a generated description of the cap
    waterfall and confirm it matches the approved policy, without reading code.

18. The adjusted and unadjusted score, PD and grade are recorded on every
    application, approved or declined.
19. An overlay that would *increase* capacity, raise a ceiling or reduce a rate
    below the card is rejected when the adjustment set is defined — with the
    single exception of the authority-bounded uplift rule, which is a register
    rule and not an overlay.
20. The flow runs with the overlay stack disabled through the same
    implementation, and a full population can be scored both ways to produce the
    impact estimate that precedes a committee decision.
21. Every live overlay has an unexpired review date, and the set can be listed
    with each overlay's age, approval and expiry.
22. A rate add-on cannot produce a rate above the statutory ceiling, tested at
    the add-on rather than only at card validation.

---

## 11. Change scenarios

Each of these will be asked for within two years of go-live. A good structure
makes them cheap.

1. **Treasury re-bands the rate card** from 96 amount bands to 120, splitting the
   R50 000–R149 999 range more finely. The solve's correctness assumptions must
   survive band edges moving, and the 41 declared inversions become 58.
2. **The repo rate moves mid-month**, lowering the statutory ceiling below 1 100
   cells on the live card. Those cells must be corrected and re-validated within
   two hours, and every application priced from them in the interim identified.
3. **Credit Risk Policy adds three rules to the cap register** and reorders eleven
   existing ones, because a percentage-based rule was running before an absolute
   one and giving the wrong answer.
4. **Product changes the recommendation objective** from `largest_amount` to
   `best_expected_value` on a Tuesday, then back on the Thursday after the
   conversion numbers come in.
5. **The definition of `lowest_total_cost` is disputed.** It currently picks the
   smallest loan, which nobody wanted. It is re-specified twice.
6. **A fourth scorecard segment appears** — self-employed — with 52
   characteristics instead of 45, running in parallel with the existing thin-file
   scorecard in an overlapping population for three months.
7. **Compliance adds a sixth minimum viable offer rule**: no offer whose
   instalment exceeds 35% of net monthly income, regardless of the affordability
   assessment.
8. **The credit life table gains a seventh employment type** and the age bands go
   from 14 to 16, taking the table from 672 to 896 cells.
9. **A new term appears.** Product wants 30 months added to the permitted list for
   existing clients only. The rate card already has a column for it.
10. **The initiation fee ceiling is adjusted** for inflation, moving the kink from
    R12 700 to R13 300 and changing which band it sits in.
11. **Project 06 wants the solve** for consolidation, where the objective is to
    settle a chosen subset of existing debts rather than to maximise an amount,
    across four products with four separate rate cards.
12. **Project 07 wants the cap waterfall** for credit limit management, where the
    ceilings are limits rather than amounts and a portfolio budget constrains the
    result after the waterfall has run.
13. **The evaluation ceiling is questioned.** Analysis shows 0.3% of applications
    hit 24 evaluations and refer. Product wants 32; Credit Systems wants the
    search improved instead. Both must be testable cheaply.
14. **Joint applications** move from a single blended assessment to two graded
    applicants with a worse-of rule, changing what the scorecard consumes without
    changing what the solve does.
15. **The bureau adds a field** that four of the 45 characteristics should use,
    and the Model team wants it live in the challenger before the champion.

16. **Observed defaults run 30% above prediction** on scorecard 1011 for two
    consecutive quarters. Credit Risk Policy applies a PD multiplier for six
    months while the model is rebuilt, and must show the Credit Committee the
    approval-rate and volume impact before it goes live and the realised impact
    after.
17. **Treasury wants 50 basis points on grades 9 to 12** across all terms, live
    next week. Reissuing the card takes three weeks.
18. **A tightening overlay from three years ago is found still in force**, its
    author having left. The Bank must be able to list every live overlay with its
    age, approval and expiry, and to show what unwinding this one would do to
    today's approval rate and margin.
19. **Two overlays collide** — a segment score shift and a channel odds
    multiplier both applying to one application — and the committee wants the net
    effect, the order they composed in, and the count of applications where the
    interaction changed the grade.
20. **The model team wants the challenger run clean** while the champion carries
    three overlays, so that the parallel run measures the challenger rather than
    the overlay set.

---

## 12. Out of scope

- Bureau retrieval itself, statement retrieval, document capture and OCR.
- Income verification and obligation derivation — project 02 owns both.
- The fraud decision itself — project 01 owns it; this project consumes a verdict.
- Campaign selection and targeting — project 04 consumes this project's batch
  output; it does not influence it beyond `campaign_id`.
- Consolidation and settlement search — project 06.
- Contract generation, disbursement, the debit order mandate and servicing.
- Collections, arrears treatment and restructure — project 08.
- Model development. This project consumes scorecards; it does not fit them.
- Storage, replay and diff infrastructure — project 09 owns them; this project
  produces the evidence they consume.
- The client-facing journey, its wording and its layout. This project produces the
  values that journey displays.

---

## 13. Questions the implementation must answer

1. **Is a bounded solve a core component?** The requirement recurs — 03, 05 and 06
   all need a deterministic, bounded, attributed search over a discrete domain
   with a non-monotone feasibility test. Is that one reusable kind, or is each
   search bespoke? If it is a kind, what does it take as its domain, its
   feasibility test and its objective, and how does it record what bound it
   without the caller writing that recording by hand?
2. **Is a cap waterfall a core component?** 52 independently-owned rules, three
   ceilings, reduce-only with one authorised exception, and a mandatory
   attribution chain. Is this a decision table with extra requirements, or a
   distinct kind? What happens when project 07 wants the same shape over limits
   and a portfolio budget?
3. **How does a 63 360-cell grid get integrated?** Where does it live, how is it
   validated before going live, how is it versioned, how is a single cell read
   attributed, and how is it refreshed monthly and patched mid-month without a
   deployment and without interrupting serving?
4. **How is a grid read tens of times per application, inside a search, at
   amounts the search chooses**, within a 55 ms budget and a 140 000-lookup/second
   batch rate — while still recording which cell each read hit?
5. **How does non-monotonicity get expressed as a requirement** that the
   implementation must satisfy, rather than as a property somebody hopes holds?
   What does a specification of "correct" look like here that is checkable without
   evaluating all 4 981 candidates?
6. **Where does the evaluation ceiling live**, and what is the defined behaviour
   when it is reached? Is a bounded search that can fail to find an answer
   expressible at all, or does every search have to be assumed to succeed?
7. **How are the search's intermediate evaluations captured?** They are required
   evidence — "why R50 000 and not R49 900?" is only answerable from them — but
   there are up to 216 of them per application and 14 M applications a month in
   batch. Is evidence capture a parameter, and if so, does turning it down change
   the answer?
8. **How is a ragged output expressed** — 0 to 9 offers, each with 12 fields,
   plus a suppression record for every term that produced nothing? And how does
   the downstream ranking and selection over that collection stay explainable?
9. **How is the shared library consumed inside a search?** `core.rate_card`,
   `core.fees`, `core.credit_life`, `core.instalment` and `core.rounding` are
   called together, repeatedly, with different arguments each time. Is that one
   composite the library should publish, or five calls the consumer assembles —
   and if the latter, what stops six consumers assembling them differently?
10. **Can `core.instalment` be run in reverse** — solve the advance that produces a
    given instalment — and does using the inverse as a starting point for the
    search, rather than as the answer, change what the library needs to publish?
11. **How do 52 rules owned by four teams live in one register** that is
    reordered quarterly, where sequence position is semantically significant, and
    where a policy owner must be able to add one without a release and see whether
    their rule is doing anything?
12. **How is the same logic run in real time and in batch with an identity
    guarantee?** Not "the same code" — the same *answers*, verified. What
    structural property makes that checkable rather than hoped for?
13. **Where does the recommendation objective live?** It is a parameter with three
    values that changes a scoring computation over a variable-length collection. Is
    that configuration, or is it three implementations selected by configuration,
    and what does the evidence record look like either way?
14. **How is the final validation stage expressed** so that it is genuinely
    independent of the stages it validates, rather than re-running the same code
    and agreeing with itself?
15. **What is the unit of effective dating?** Fourteen tables, five owners, five
    cadences, and a replay guarantee that depends on every one of them resolving
    against `decision_date`. What makes forgetting it impossible rather than
    merely discouraged?
16. **How does a policy analyst read this?** The cap waterfall, the minimum viable
    offer rules and the grade boundaries are all policy artefacts that Credit Risk
    Policy signs off. What does the generated description look like, and is it
    generated from the thing that executes or from a second description that can
    drift?
17. **What is an overlay, structurally?** It changes values without changing
    logic, which makes it look like a parameter. It composes in a declared order
    that changes the answer, which makes it look like logic. It is separately
    owned, separately approved, separately versioned and separately expiring,
    which makes it look like neither. Five stages here are overlaid and six of
    the ten projects use them — so answering this three different ways in three
    places is the failure to avoid.
18. **How does the flow run with its overlay stack disabled** without becoming a
    second implementation of itself, given that the counterfactual is needed for
    sampling and impact estimation but is far too expensive to compute on every
    one of 55 000 daily applications?
19. **Where is the conservative-only asymmetry enforced?** An overlay may tighten
    and may not loosen, except for one authority-bounded register rule that may
    raise a cap. Is that a validation on the adjustment set's definition, a
    property of the kinds themselves, or a runtime check — and what makes it
    impossible to bypass by defining a negative magnitude?
