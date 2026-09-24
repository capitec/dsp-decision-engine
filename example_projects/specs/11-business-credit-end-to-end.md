# 11 — Business credit, end to end

> Fictional. The Bank, its products, thresholds, table dimensions and volumetrics
> are invented for this repository. Regulatory mechanisms referred to are the
> published public ones; every number attached to them is illustrative.

---

## 1. What this is

The Bank's single business credit decision service. One deployed artefact,
covering every credit decision the Bank makes about a business — from the day a
new client walks in, through five years of annual reviews, quarterly covenant
tests, daily early-warning passes, limit amendments and security substitutions,
and, for the tenth of the book where it goes wrong, restructure, forbearance and
exit.

The scale is the first problem:

| | |
|---|---|
| Decision points | **≈ 1 900** — rules, gates, caps, thresholds, branch conditions and cell selections that can change an outcome |
| Tables | **63**, owned by eleven different functions on six different cadences |
| Facility types | **9** (§4.2), each with its own term range, security expectations, rate table and covenant conventions |
| Sector codes | **420**, with two tables hanging off each |
| Scorecard families | **2** (personal-behind-business and commercial), plus thin-file and behavioural variants |
| Entry points | **9** (§5.1), with latency budgets four orders of magnitude apart |
| Owning teams | **14** |
| Business clients | **180 000** |
| Live facilities | **240 000** |
| Re-decisions retained | **five years' worth per facility**, every one reproducible, every one comparable with every other |

That is project 10's problem, and if it were only that, this document would be
project 10 with different products.

The second problem is the one that makes this a separate project: **time**.

A retail application is decided once. The client applies, the Bank answers, and
the decision is over. Everything afterwards is servicing.

A business facility is *never* decided once. It is decided at origination, and
then re-decided at its annual review, every year, for as long as it exists. Its
covenants are tested on their own schedules — monthly, quarterly, annually — for
the whole of its life, against definitions agreed on the day it was written. It
is re-assessed every night against a book-wide early-warning pass. Its group is
re-assessed whenever any entity in any related business changes anywhere. It is
amended when the client wants more, restructured when the client cannot pay, and
handed to Recoveries when that fails.

So the subject of a decision here is not an application. It is a **facility with
a history**, and the Bank's relationship with it is a sequence of decisions about
the same thing, made by different people, under different policy, using different
tables, against different models, executed by a flow that has itself been changed
forty times in between.

The requirement that follows is the spine of this document:

> The flow that decides this facility in 2031 is not the flow that decided it in
> 2026. The two must still be comparable, and the system must know exactly in
> what sense they are comparable and in what sense they are not.

A grade migration report that puts a 2026 grade 6 next to a 2031 grade 7 and
calls it a one-notch deterioration is either true, or an artefact of two
scorecard recalibrations and a master-scale collapse that happened in between.
Both possibilities look identical in a spreadsheet. Producing a system that can
tell the Bank which one it is looking at — for 240 000 facilities, five years
deep, under a flow that never stops changing — is the project.

**How this project is built, and why that is the experiment.** There is a third
thing being tested here, and it is a property of the *construction* rather than
of the business problem.

**This project is deliberately assembled out of components that already exist.**
It consumes the `core.*` capabilities published by
[project 00](00-shared-credit-core-library.md), the regulated affordability
assessment built by [project 02](02-affordability-assessment.md), the whole of
[project 05](05-business-credit-nested-entities.md)'s entity-structure, adverse
event and people-blend machinery, and
[project 06](06-consolidation-and-restructure.md)'s concession search and
authority model. Its origination phases (§5.3) are almost entirely project 05's
flow, referenced by section and not restated. Where a number is already fixed in
another spec — 40 entities, 60 events, 34 classification rules, 12 roll-up rules,
105 600 rate card cells, a 75% coverage requirement, an eight-concession
catalogue — this document uses that number and does not invent a new one.

That is the opposite of how [project 10](10-retail-credit-end-to-end.md) is
built. Project 10 is constructed standalone, from nothing, with its own numbers,
and it tests how a very large flow gets **structured from scratch**. Project 11
is constructed by reference, and it tests how easy it is to **reuse components
that already exist**, at scale, inside something much larger than the thing they
were written for. The two together are an A/B on the same question, and the
difference between them is the measurement.

The reuse is therefore not an editorial convenience to keep this document short.
It is the point. §4.9 inventories every component consumed and how many times per
assessment. §5.17 develops what that reuse actually costs — the vocabulary
mismatches, the missing 10%, the semantics that differ by consumer, the version
coupling across four release cadences, the performance profile nobody designed
for, and the specific places where forking would be the cheapest thing the team
could do. That section is the most valuable material in this document, because no
isolated spec in the set can produce it: you only find out what a component costs
to reuse by reusing it somewhere it was not designed for.

---

## 2. Why it is in this set

| Question | How this project stresses it |
|---|---|
| **Q6 codebase organisation at scale, plus time** | This is the centre of gravity and it is Q6 in a form no other spec has. 1 900 decision points, 63 tables and 14 owning teams is the static problem, and project 10 has it too. The addition here is that the same artefact must serve nine entry points *across five years*, and every version of it must remain resolvable, renderable and re-runnable for the whole retention period. Organising a codebase so a policy analyst can find a rule is hard. Organising it so they can find the rule **as it was in March 2028**, and see what has happened to it since, is the test. |
| **Q1 reuse and Q4 custom modules across flows** | This project is **built by reuse on purpose** (§2.2), which makes it the set's primary Q1 instrument. Twenty-two library capabilities and the whole of [project 05](05-business-credit-nested-entities.md)'s entity machinery are consumed here, but not once per application — they are consumed at origination, again at each of five annual reviews, again inside every amendment, again on a daily early-warning pass over 240 000 facilities, and again inside a group cascade triggered by something that happened at another client. Person-level capabilities written for one applicant (`core.income`, `core.affordability`, `core.adverse_events`, `core.scorecard`) run inside a business lifecycle, on entities that change between runs. The reuse question is not "can two flows share this" but "can the same unit produce comparable answers about the same subject five years apart, in a flow forty times its size, without being forked". |
| **Q7 audit and explainability** | Project 05 must explain one decision. This project must explain a **sequence** of decisions about one subject, made under a changing flow. Five re-decisions, 40 flow versions, two scorecard recalibrations, a master-scale change, 11 covenant tests, 3 waivers and a restructure — and someone in 2031 who was not there asking why the 2027 review graded it 7. Plus the cascade case: explaining to the client of business B why their facility was reduced because of something that happened at business A, without disclosing business A. |
| **Q3 parameters** (secondary) | Roughly 190 parameters, of which about 130 are owned by people who are not engineers, across five functions with different approval routes. Several of them — covenant thresholds in particular — are not policy parameters at all but **contractual** ones, fixed per facility at documentation and immune to policy change thereafter. Distinguishing the two is not a nicety; getting it wrong tests a 2029 covenant against 2031's standard wording. |
| **Q2 tables** (secondary) | 63 tables, the largest being a 105 600-cell business rate card, a 23 040-cell asset finance card, a 3 888-cell delegated authority matrix and a covenant definition library carrying roughly 1 900 live definition versions. |
| **Q5 core component set** (secondary) | Inherits project 05's nesting and roll-up question, and adds three shapes no earlier spec produces: a **schedule** (a covenant instance that fires on its own calendar for years), a **cross-record constraint** (one collateral item securing three facilities that are assessed separately), and a **cascade** (a change to one subject re-opening decisions about others). |

### 2.1 The stress, stated plainly

Three things are being tested, in this order.

**First — Q6 with a time axis.** A large codebase is hard. A large codebase
whose every past state must remain executable, renderable and attributable for
five years, while fourteen teams change it weekly, is a different thing. The
failure mode is not that the code becomes unmaintainable; it is that the code
becomes unmaintainable *and* the Bank loses the ability to explain what it used
to do, which is a regulatory finding rather than an engineering inconvenience.

**Second — Q1 and Q4, reuse of person-level capabilities inside a business
lifecycle.** The library's capabilities were written to answer a question once
about one person. Here, `core.adverse_events` classifies an event on a director
who sits on six businesses, on a Tuesday night, as part of a cascade, with the
answer needing to be consistent with the classification the same event received
in that director's own personal loan application that morning — and legitimately
*inconsistent* with it where the criticality class differs. Reuse across flows is
the easy case. Reuse across flows, across subjects and across years is the real
one.

**Third — Q7 over a five-year trail of repeated decisions about one subject.**
Project 09 can replay one decision. This project needs the sequence: what the
grade was, what it is, what moved it, and how much of the movement is real. The
honest answer is sometimes "we cannot compare these two numbers", and the
requirement is that the system be **able to say so** rather than quietly
producing a difference.

### 2.2 Built by reuse — the deliberate A/B with project 10

Projects 10 and 11 are the same experiment run two ways.

| | **Project 10 — retail, end to end** | **Project 11 — business, end to end** |
|---|---|---|
| Construction | Standalone. Rebuilt from nothing, with its own numbers. | By reference. Assembled from components that already exist. |
| What it measures | How a very large flow gets **structured from scratch** — where the seams land when nothing constrains them. | How easy it is to **reuse components that already exist**, at scale, inside something far larger than they were written for. |
| The failure it can expose | A structure that works at six modules and collapses at six hundred. | A library that consumers have to fork, wrap or copy-paste out of — which [project 00](00-shared-credit-core-library.md) §1 names as the outcome the previous generation actually had. |
| Its own numbers | Sets them. | Inherits them wherever a consumed spec already fixed one. |

Four consequences, and they are requirements on the implementation rather than
notes on this document:

1. **No consumed component may be re-specified here.** Where this flow needs
   project 05's adverse event roll-up, it names §5.6 of that document and states
   only what changes because the roll-up now runs inside a lifecycle. An
   implementation that reimplements the roll-up because reimplementing was
   cheaper has produced the finding, and that finding is the result of the
   experiment rather than a defect in it.
2. **Inherited numbers are inherited.** 1..40 entities, 0..60 events, 34
   classification rules, 12 roll-up rules, the 75% coverage requirement, the
   105 600-cell rate card, the eight-concession catalogue, the four affordability
   assessment modes. Where this document restates one it is quoting, not
   redefining.
3. **New material is new.** The nine facility types, the covenant definition
   library, the 186 early-warning signals, the six watchlist grades, the
   seven-level authority matrix, the amendment re-open matrix and the cascade
   are this project's, and this project sets their numbers.
4. **The cost of reuse must be measured, not assumed.** §5.17 is that
   measurement, and §13's sharpest question — at what point does reuse cost more
   than rebuilding, and what evidence would show that line had been crossed — is
   the one this project exists to force into the open.

### 2.3 What is new here that project 05 does not have

Project 05 is the origination deep-dive and this document does not repeat it. Its
thirteen stages are this project's origination phases, referenced and not
restated. What is new:

- the facility exists afterwards, and everything that happens to it afterwards is
  also a decision made by this flow;
- eight more entry points, seven of which project 05 does not have;
- seven more facility types, with security and covenant conventions project 05's
  two do not need;
- covenants, which project 05 sets in one line of §5.13 and never tests;
- a group that changes while nobody is looking at it;
- collateral shared between facilities that are assessed months apart;
- and the requirement that a decision be comparable with its own predecessor.

### 2.4 What is new here that project 10 does not have

Project 10 shares the scale problem — one flow, many entry points, many teams,
one decision record. The difference is that project 10's subject is an
application, which is decided and finished. This project's subject persists,
accumulates state, and is re-decided. Everything in §5.10 to §5.16 exists because
of that and has no counterpart in project 10.

### 2.5 The parts already known to be hard

Named here because a design that discovers them during implementation has
discovered them too late.

| # | Difficulty |
|---|---|
| H1 | **Comparability of re-decisions under a changing flow.** Five decisions about one facility, under five flow versions, two scorecard recalibrations and a master-scale change. Stating what "comparable" means, and detecting when it fails. |
| H2 | **Two different version-resolution rules in one flow.** Policy artefacts resolve by `decision_date`. Contractual artefacts — covenant definitions, security terms, pricing margins — resolve by the version agreed **when the facility was written**, and never move. A flow that applies one rule to both is wrong in a way that produces plausible answers. |
| H3 | **Bi-temporality of the entity structure.** Facts have an effective date (when the change happened) and a knowledge date (when the Bank learned of it). Replay needs the knowledge-date view; a covenant on ownership change needs the effective-date view. Both must be available and they must not be confusable. |
| H4 | **Cross-facility constraints.** One collateral item securing three facilities. The position cannot be decided facility by facility, but the facilities are decided facility by facility, months apart, by different people. |
| H5 | **The group cascade.** A change to one entity propagates to every business it touches and every facility of those businesses. Unbounded propagation is a book-wide re-assessment triggered by a director's parking fine; bounded propagation must still be complete enough to be correct. |
| H6 | **Five volume profiles in one artefact.** 2-second interactive; 4-second interactive; a monthly batch of 15 000 full reviews; 2.4 M covenant tests a year with a 13× seasonal peak; a daily pass over 240 000 facilities with variable-length entity structures. |
| H7 | **Authority that moves during the assessment.** The approving authority is a function of the proposed structure, and the proposed structure changes while it is being assessed. The authority at routing is not the authority at approval. |
| H8 | **A covenant is a schedule, not a rule.** It is set once, tested on its own calendar for years, may step over time, may be cured, may be waived for a stated period with an expiry, and is bound to a definition version that policy cannot touch. |
| H9 | **Degraded operation is the normal case, not the exception.** 30% of annual reviews run on stale or incomplete financials. A design in which the complete-evidence path is the main path and everything else is an error branch will have its main path used 70% of the time. |
| H10 | **Forbearance must be classified at the moment of decision.** Not derived later by Finance from a payment pattern. And a restructure that is *not* forbearance must carry the evidence for the negative classification. |
| H11 | **Fourteen teams, one artefact, continuous change.** Including three that only ever read it (Provisioning, Recoveries, Audit) and whose requirements nonetheless constrain what the writers may do. |
| H12 | **Navigability after five years.** Forty flow versions since the decision being questioned. The person asking was not there and cannot read code. |
| H13 | **The cost of reuse, at a scale nobody designed the components for.** Person-level vocabulary inside a business flow; components that produce 90% of what a consumer needs; semantics that differ by consumer; four release cadences coupled into one artefact; and a daily pass calling capabilities sized for 900 applications a day. §5.17. |

---

## 3. Actors

Fourteen owning teams, plus the consumers who change nothing and constrain
everything.

| # | Team | Owns | Cadence | Blast radius |
|---|---|---|---|---|
| 1 | **Business Origination Engineering** | The origination phases O1–O17, the intake contracts, the pre-assessment path | Weekly releases | Entry points 1, 2, 9; indirectly all |
| 2 | **Business Credit Risk Policy** | Disqualification rules, event classification thresholds, roll-up rules, blend weights, coverage requirements, DSCR and gearing thresholds, degraded-review rules | Quarterly, and out of cycle after a loss event | Every entry point |
| 3 | **Portfolio Management** | Group exposure caps, single-name and sector concentration limits, the review cohort calendar, watchlist mandated actions, exit triggers | Quarterly | Entry points 3, 5, 8 |
| 4 | **Covenant and Documentation Operations** | Covenant instances, test schedules, certificate receipt, the covenant definition library's instance layer | Continuous — every facility documented | Entry point 4, and every review |
| 5 | **Early Warning** | The 186-signal catalogue, signal weights and decay, watchlist grade boundaries | Monthly | Entry point 5, and authority modifiers everywhere |
| 6 | **Credit Risk Modelling** | The two scorecard families, their thin-file and behavioural variants, calibrations, the master scale and its restatement maps | On model release, 2–4 times a year | Every entry point that grades |
| 7 | **Sector Analytics** | The 420-code sector table, sector ratio benchmarks, sector early-warning overlays | Annual, with mid-year patches after a sector shock | O7, O11, L3 |
| 8 | **Treasury and Pricing** | Six rate tables, the rate floor components, repricing add-on overlays | Monthly, occasionally mid-month | O13, L1's repricing decision |
| 9 | **Financial Crime and Compliance** | Screening dispositions, match-confidence bands, reason-code wording, what may lawfully be said about a third party | Monthly | O3, and every communicable reason |
| 10 | **Provisioning and Regulatory Reporting** | Staging and significant-increase-in-credit-risk triggers, forbearance probation and cure rules, the reporting classifications | On accounting-standard change, and semi-annually | L5, L6, and the classification produced at L2 and L3 |
| 11 | **Legal and Security Documentation** | Covenant standard wording and its versions, cure periods, cross-default thresholds, negative pledge exceptions, collateral revaluation and allocation rules | Annual, and on precedent change | L2, O12, O16 |
| 12 | **Business Recoveries and Workout** | The handoff contract, workout decision inputs, the concession catalogue jointly with Policy | Semi-annual | L5, L6 |
| 13 | **Business Banking Channel (Relationship Management)** | The pre-assessment experience, the indicative offer rules, what an RM may commit to | Monthly | Entry point 9 |
| 14 | **Credit Governance** | The delegated authority matrix, pack contents per level, the review scope matrix, the amendment re-open matrix, committee secretariat | Annual, and out of cycle on a mandate change | Every entry point |

| Consumer | What they ask for, and when |
|---|---|
| **Credit Committee and Board Credit Committee** | The pack, twice weekly and monthly respectively. Read the flow's output; approve above delegated authority; approve every policy change at items 2, 3, 11 and 14. |
| **Internal Audit** | Whether the decision that was made is the decision the approved policy required, up to seven years later. |
| **The regulator** | Whether forbearance was reported as forbearance; whether grade migration is real; whether the policy in force was the policy approved. |
| **Finance** | The staging classification and the forborne flag, monthly, at reporting dates that are not the Bank's decision dates. |
| **The business client** | Why the answer was no, why the limit went down, and why the price changed. |
| **The individual entity** | Their own record's reasons — which the business is not entitled to. [Project 05](05-business-credit-nested-entities.md) §9.2 governs, unchanged. |

---

## 4. Inputs

### 4.1 The subject

Unlike every earlier project, the subject of an assessment here is not always an
application. Four subject kinds, and the flow must be able to say which it has:

| Subject | Identified by | Used by entry points |
|---|---|---|
| **Application** | `application_id` | 1, 2, 9 |
| **Facility** | `facility_id`, stable for the life of the facility across every amendment | 3, 4, 6, 7 |
| **Client** | `client_id`, stable for life | 3, 5, 8 |
| **Group** | `group_id`, **not** stable — a group's composition is derived, and its identity changes when its composition does (§5.12) | 8 |

The instability of `group_id` is deliberate and is a stated requirement rather
than a defect: a group is a derived set, and pretending it has a stable identity
is how a 2031 group exposure report silently includes a business that left the
group in 2028. A group is identified by its composition as at a date, and every
group-level output names that date.

### 4.2 The facility catalogue

Products 50 and 51 are [project 00](00-shared-credit-core-library.md) §5's. The
remaining seven are introduced here and are used consistently from this point on.

| `product_code` | Facility | Type | Term range | Amount / limit range |
|---|---|---|---|---|
| 50 | **Business Term Facility** | Amortising term loan | 6–60 months | R50 000 – R10 000 000 |
| 51 | **Business Revolving Facility** | Revolving, annual review | n/a | R50 000 – R5 000 000 |
| 52 | **Commercial Asset Finance** | Instalment sale or lease over the financed asset | 12–84 months, capped at 80% of assessed asset life | R100 000 – R25 000 000 |
| 53 | **Commercial Property Finance** | Amortising, secured by a first covering bond | 60–240 months | R500 000 – R80 000 000 |
| 54 | **Invoice and Debtor Finance** | Revolving against a borrowing base | n/a; 12-month facility review, 90-day notice | R250 000 – R30 000 000 |
| 55 | **Trade Finance Facility** | Import letters of credit, documentary collections, import loans | 30–180 days per transaction; 12-month facility | R250 000 – R50 000 000 |
| 56 | **Guarantee Facility** | Performance, payment, retention and advance-payment guarantees | Up to 60 months per instrument | R100 000 – R40 000 000 |
| 57 | **Business Overdraft** | Revolving on the transactional account, repayable on demand | n/a; annual review | R25 000 – R7 500 000 |
| 58 | **Bridging Facility** | Single-purpose, repaid from a named identified source | 1–12 months, one 3-month extension permitted | R250 000 – R25 000 000 |

| `product_code` | Security expectations | Rate table | Covenant conventions |
|---|---|---|---|
| 50 | Surety from every holder above 25%; collateral per §5.11 | Business rate card, 105 600 cells | DSCR tested annually; gearing ceiling; 70% of turnover through the Bank; capex limit; no further borrowing above R250 000 without consent; annual financial statements within 6 months of year-end |
| 51 | As 50 | Business rate card | As 50, plus a 30-consecutive-day clean-down below 20% of limit once in each 12 months |
| 52 | The financed asset, title retained; notarial bond over related plant; residual or balloon to 30% | Asset finance rate card, 23 040 cells | DSCR; asset insurance and maintenance undertakings; no disposal or relocation without consent; residual value test at 70% of elapsed term |
| 53 | First covering bond; LTV ≤ 70% investment, ≤ 75% owner-occupied; cession of rentals | Property rate card, 17 280 cells | LTV covenant tested annually on revaluation; interest cover ≥ 1.30 on net rental; tenancy schedule and weighted average lease expiry undertaking; no further encumbrance |
| 54 | Cession of debtors; 75% advance on eligible debtors; personal surety | Debtor finance grid, 576 cells | Borrowing base certificate monthly; dilution ≤ 5%; no single debtor above 20% of the book; nothing over 90 days eligible; verification audit twice yearly |
| 55 | Pledge and cession of goods and documents; 20–30% cash cover by grade | Trade fee table, 504 cells | Tenor limit per instrument; single-transaction sub-limit; marine and goods insurance ceded; no open-account extension beyond stated terms |
| 56 | Counter-indemnity; cash or ceded deposit cover 0–100% by grade | Guarantee fee table, part of the trade table | Tangible net worth maintenance; no further guarantees without consent; contingent exposure counted at 100% of face for group exposure and at a 50% conversion factor for appetite |
| 57 | Surety plus general notarial bond, typically | Overdraft rate table, 720 cells | Clean-down; excess rules; 70% of turnover through the account; management accounts monthly above R2 000 000 |
| 58 | Cession of the identified repayment source plus collateral cover ≥ 1.30 | Bridging table, 360 cells | Exit source verified at origination and re-verified monthly; milestone undertakings; no competing charge over the proceeds; progress certificates |

Facility type is a first-class key, not a label: it keys the rate table, the
covenant default set, the collateral expectations, the authority matrix, the
review scope and the amendment re-open matrix. Adding a tenth is change scenario
2 in §11 and is the single most informative test of the structure.

### 4.3 Facility state

Every re-decision reads state the previous decision did not have. This is the
input class project 05 has none of.

| Input | Shape | Freshness | Notes |
|---|---|---|---|
| Facility register | One record per facility; 240 000 live | Real time | Type, limit, drawn balance, contractual rate and margin basis, start date, maturity, review date, arrears state, status |
| Facility decision history | 1..~40 decisions of record per facility | Immutable | Origination, each review, each amendment, each covenant-driven decision, each restructure. §5.10 |
| Covenant instances | 0..14 per facility; mean 4.0; 960 000 live instances | Immutable once documented | Each bound to a definition version (§6.3) |
| Covenant test history | 0..~60 tests per facility over five years | Immutable | Result, measured value, headroom, classification, cure, waiver |
| Waiver register | 0..8 per facility | Immutable, with expiry | §5.5 |
| Watchlist state | One grade plus signal set per facility, daily | Daily | Current grade, grade history, contributing signals, mandated actions and their completion |
| Collateral register | 0..20 items per client, shared across facilities | Valuations dated; revaluation cadence by class | §5.11 |
| Collateral allocation | Per (collateral item, facility) pair, as at a date | Immutable per decision | §5.11 |
| Conduct | 24–60 months of transactional behaviour | Daily | Excesses, excess days, returned debits, turnover, deposit concentration, lowest balance |
| Arrears and payment history | Per facility, monthly buckets | Daily | Including the payment history of every other provider from the bureau |
| Forbearance state | Flag, measure, grant date, probation clock | On grant, then daily | §5.8 |
| Staging classification | Stage 1 / 2 / 3, with trigger | Monthly, at Finance's reporting date | §5.8 |
| Group membership as at a date | Derived set | Recomputed on cascade | §5.12 |

### 4.4 The entity structure, as a time series

Project 05 consumes an entity structure. This project consumes the **history** of
one. Every structural fact carries two dates, and the distinction is load-bearing
(H3):

| Field | Meaning |
|---|---|
| `effective_from` / `effective_to` | When the fact was true in the world |
| `known_from` | When the Bank learned it |
| `source_code` | Registry filing, client disclosure, bureau, screening feed, RM note |
| `fact_kind` | Appointment, resignation, share transfer, allotment, trust deed amendment, beneficial ownership restatement, address, status |

Volumes: 180 000 businesses × mean 6.2 entities = **≈ 1 116 000 entity
attachments**. Structural change events arrive at roughly **34 000 per month**
across the book, of which about 11 000 are share transfers or allotments and the
remainder appointments and resignations. Median lag between `effective_from` and
`known_from` is **47 days**; p95 is **310 days**. Roughly 4% of facts arrive with
an effective date more than two years in the past, which means they land behind
decisions already made.

### 4.5 Financial information over time

Project 05 §4.5's shapes, with one addition: the flow now holds **1..6 periods**
per client rather than 1..3, because five years of reviews accumulate them, and
the comparability requirement means the earlier periods cannot be discarded when
the accounting policy changes.

| Situation | Share of annual reviews |
|---|---|
| Statements with most recent year-end ≤ 9 months old | 44% |
| Year-end 9–15 months old | 26% |
| Year-end 15–21 months old | 18% |
| Year-end > 21 months old, or absent | 12% |

The last two rows are the **30% degraded case** of H9 and are developed at §5.4.

### 4.6 Signals

The early-warning feed, consumed daily across the whole book.

| Family | Signals | Source | Cadence | Subject |
|---|---|---|---|---|
| Business bureau | 34 | Commercial bureaux | Daily delta | The business |
| Entity bureau | 28 | Consumer and commercial bureaux | Daily delta | Any entity |
| Internal conduct | 41 | Transactional systems | Daily | The client's accounts |
| Covenant proximity and breach | 17 | This flow, §5.5 | On test | The facility |
| Other-provider behaviour | 22 | Bureau payment profiles | Monthly | Business and entities |
| Sector and macro | 19 | Sector Analytics | Monthly, and on shock | Sector code |
| Structural and adverse media | 25 | Registry, screening, media feed | Daily | Any entity |
| | **186** | | | |

### 4.7 Volumes and freshness

| Input | Volume | Tolerance at `decision_date` |
|---|---|---|
| Commercial bureau view, per business | 180 000 refreshed on review, 240 000 read daily as deltas | 30 days for a decision; deltas for signals |
| Consumer bureau view, per natural-person entity | ≈ 780 000 natural persons across the book | 30 days for a decision |
| Screening outcomes, per entity | ≈ 1 116 000 attachments | 90 days, or on status change |
| Tax compliance status | Per business | 30 days |
| Registry filing status | Per juristic entity | 30 days |
| Financial statements | 1..6 periods | §4.5 |
| Management accounts | 0..1 | 3 months |
| Bank turnover | 6..60 months | 1 month |
| Borrowing base certificate (product 54) | Monthly | 15 days |
| Collateral valuations | 0..20 items | Property 24 months, plant 12 months, debtors 1 month, listed securities daily |
| Covenant compliance certificates | ≈ 2.4 M tests/year | Per covenant instance's delivery terms |

### 4.8 Names this project declares locally

The library does not publish these and this project declares them once.

| Name | Type | Meaning |
|---|---|---|
| `facility_id` | int64 | Stable for the life of the facility, across every amendment and restructure. |
| `decision_of_record_id` | int64 | One entry in a facility's decision history. §5.10. |
| `assessment_kind_code` | int8 | Which of the nine entry points produced this decision. |
| `review_date` | date | The date a facility's review is due. Distinct from the date it was run. |
| `review_basis_code` | int8 | Complete / late / stale / turnover-only / not performed. §5.4. |
| `covenant_instance_id` | int64 | One covenant on one facility. |
| `covenant_definition_version` | string | The version of the definition **agreed for that instance**, not the current standard. §6.3. |
| `covenant_test_id` | int64 | One test of one instance on one test date. |
| `breach_class_code` | int8 | None / technical / material / severe. |
| `waiver_id` | int64 | With scope, period and expiry. |
| `watchlist_grade` | int8 | W0..W5. |
| `signal_set` | list[record] | The signals contributing to a watchlist grade, with values and weights. |
| `cascade_id` | int64 | One propagation event. §5.12. |
| `allocation_id` | int64 | One (collateral item → facility) allocation as at a date. §5.11. |
| `authority_level_code` | int8 | 1..7. §5.14. |
| `master_scale_version` | string | Which grade scale a grade is expressed on. §5.10. |
| `comparison_basis_code` | int8 | As-graded / restated / not comparable. §5.10. |
| `forbearance_flag` | bool | With measure, grant date and probation clock. |
| `staging_code` | int8 | 1 / 2 / 3. |
| `knowledge_date` | date | The date the Bank's view of the world is taken as at. Distinct from `decision_date`. §5.13. |

The count matters: **20 locally declared names** against the library's canonical
set. Per the README §7, a high count means the library's vocabulary is wrong.
Fifteen of these twenty are about *time* or *state over time*, and none of the
ten earlier specs needed them. That is either a gap in the library or evidence
that lifecycle vocabulary belongs somewhere else, and it is question 21 in §13.

### 4.9 The reuse inventory

Everything this project consumes rather than builds. This is an input inventory
in the same sense as §4.1 to §4.7: these components are supplied to the flow, and
the flow's correctness depends on them exactly as it depends on a bureau feed.

**Per-assessment invocation counts** are for one full origination or annual review
at the mean entity fan-out of 6.2 entities and 38 events
([05](05-business-credit-nested-entities.md) §8). Where the p99 differs by an
order of magnitude it is given, because the p99 is what sizes the batch.

#### Library capabilities — [project 00](00-shared-credit-core-library.md)

| Capability | Consumed by phases | Per full assessment | p99 | Per day, across all entry points |
|---|---|---|---|---|
| `core.dates` §6.19 | Every phase | ≈ 180 resolutions | 340 | ≈ 62 M |
| `core.bureau` §6.13 | O3, O5, L3 | 6.2 (one per entity) | 27 | ≈ 48 000 entity deltas + review batch |
| `core.adverse_events` §6.14 | O5, L3 | 38 (one per event) | 520 | ≈ 9 400 new events, plus 570 000 in the monthly batch |
| `core.scorecard` §6.10 | O5, O7, O8 | 7.4 | 29 | ≈ 112 000 |
| `core.calibration` §6.11 | O5, O6, O7, O9 | 9.6 | 32 | ≈ 140 000 |
| `core.risk_grade` §6.12 | O5, O6, O9 | 9.0 | 30 | ≈ 132 000 |
| `core.eligibility` §6.15 | O3, O4, O5 | 7.2 | 28 | ≈ 105 000 |
| `core.exposure` §6.16 | O10, L3, EP-8 | 1, over a group of mean 2.3 businesses | 41 businesses | ≈ 26 000, and every cascade |
| `core.appetite` §6.17 | O11 | 1 per facility type in scope, mean 1.4 | 9 | ≈ 14 000 |
| `core.rate_card` §6.9 | O13, L1 | **200–600** (one per pricing candidate) | 2 200 | ≈ 9 M |
| `core.instalment` §6.6 | O13, O14, L1, L5 | 200–600 | 2 200 | ≈ 9 M |
| `core.fees` §6.7 | O13 | 200–600 | 2 200 | ≈ 9 M |
| `core.rounding` §6.20 | O13, O14, L1, L5 | ≈ 900 | 3 000 | ≈ 14 M |
| `core.reason_codes` §6.18 | O4, O5, O17, L1, L2, L3 | 1 ranking over 0..24 fired codes | 41 codes | ≈ 26 000 rankings |
| `core.adjustments` §6.22 | O5, O7, O9, O11, O13 — the 11-position stack at [05](05-business-credit-nested-entities.md) §5.10 | ≈ 26 overlay evaluations | 118 | ≈ 380 000 |
| `core.consent` §6.21 | O1, O3, O5, L3 | 7.2 | 28 | ≈ 105 000 |
| `core.income` §6.1 | O14, via project 02 | 0..1 | 1 | ≈ 620 |
| `core.deductions` §6.2 | O14, via project 02 | 0..1 | 1 | ≈ 620 |
| `core.expense_norms` §6.3 | O14, via project 02 | 0..1 | 1 | ≈ 620 |
| `core.obligations` §6.4 | O14, via project 02 | 0..1, over 0..80 accounts | 80 accounts | ≈ 620 |
| `core.affordability` §6.5 | O14, via project 02 | 0..1 | 1 | ≈ 620 |
| `core.credit_life` §6.8 | O13, regulated regime with a natural-person surety | 0..1 | 1 | ≈ 180 |

All **22** published capabilities are consumed. Fifteen of them are consumed
*inside a collection* rather than once, which is project 05's H-list and is not a
new finding; what is new is that eleven of them are additionally consumed *again*
at every review, every amendment and every cascade, on a subject whose data has
moved since.

#### Project-level components

| Component | Published by | Consumed by | Per full assessment | Notes on what changes here |
|---|---|---|---|---|
| Entity structure resolution — graph collapse, de-duplication, bounds | [05](05-business-credit-nested-entities.md) §5.1 | O2, EP-8, L3 | 1 | Must now run **as at a date** against a bi-temporal fact store (§5.13) |
| Regulatory regime determination | [05](05-business-credit-nested-entities.md) §5.2 | O3 | 1 | Can change between reviews when turnover crosses R1 000 000 — §5.4 |
| Business absolute rules (20 rules) | [05](05-business-credit-nested-entities.md) §5.3 | O4 | 1 | Seven of the twenty become *review* findings rather than decline gates — §5.3 |
| Entity criticality and disqualification (14 rules × 3 classes) | [05](05-business-credit-nested-entities.md) §5.4 | O5 | 6.2 | Criticality can change when the register changes, with no new event |
| Adverse event classification (34 rules, 84-cell threshold table) | [05](05-business-credit-nested-entities.md) §5.5 | O5, L3 | 38 | Must be callable for **one event on one entity** for the daily pass |
| Entity adverse verdict roll-up (12 rules) | [05](05-business-credit-nested-entities.md) §5.6 | O5, L3 | 6.2 | Same |
| Entity scoring, two families + thin-file | [05](05-business-credit-nested-entities.md) §5.7 | O5 | 6.2 | Scorecard version must be pinnable to a past assessment |
| People component blend (11 rules, PP-01..PP-11) | [05](05-business-credit-nested-entities.md) §5.8 | O6, L3 | 1 | Needs a partial re-blend over one changed entity (§5.17) |
| Financial spreading, haircuts, sector benchmarking | [05](05-business-credit-nested-entities.md) §5.9 | O7, L1, L2 | 1 over 1..6 periods | Now 1..6 periods, not 1..3, and periods may be on two accounting bases |
| Combined grade and the 11-position overlay stack | [05](05-business-credit-nested-entities.md) §5.10 | O9 | 1 | Must emit the master scale version (§5.10) |
| Appetite, collateral treatment, group exposure | [05](05-business-credit-nested-entities.md) §5.11 | O10, O11, O12 | 1 | Collateral treatment becomes cross-facility (§5.11) |
| Non-monotone bounded structuring search | [05](05-business-credit-nested-entities.md) §5.12, shape shared with [03](03-unsecured-loan-granting-and-pricing.md) §5 and [06](06-consolidation-and-restructure.md) §5.5 | O13 | 1 search, 200–600 candidates | Seven more facility types, each with its own candidate space |
| Final validation, conditions, committee pack | [05](05-business-credit-nested-entities.md) §5.13, §9.3 | O16, O17 | 1 | Five authority levels become seven; the pack becomes level-dependent |
| Regulated affordability, four assessment modes | [02](02-affordability-assessment.md) §5, §5.8 | O14 | 0..1 | A **fifth mode** is needed — §5.17 |
| Obligation inventory and settleability | [06](06-consolidation-and-restructure.md) §5.2, §5.3 | L5 | 0..1 | Over facilities, not retail accounts |
| Concession catalogue and NPV authority model | [06](06-consolidation-and-restructure.md) §5.9 | L5 | 0..1 | Eight concessions become 21 across nine facility types — §5.8 |
| Stressed affordability and distressed classification | [06](06-consolidation-and-restructure.md) §5.9 | L5 | 0..1 | Becomes the forbearance classification of record — §5.8 |
| Portfolio-budget limit decisioning | [07](07-credit-limit-management.md) | L1's limit decision on revolving facilities | 1 per revolving facility reviewed | A portfolio constraint over record-level decisions, now inside a review |
| The 23-item evidence contract | [09](09-governance-and-replay-harness.md) §5.15 | Every phase | — | Consumed as an obligation, not a component. §9 |
| Replay, diff, swap-set, certification, the reviewable artefact | [09](09-governance-and-replay-harness.md) §5.1–5.7 | Governance, not runtime | — | This project is its largest and hardest subject |

#### What this project builds itself

Everything else, and the list is short by design: the nine facility types and
their tables (§4.2), the covenant definition library and its instance, schedule,
breach, cure and waiver machinery (§5.5, §6.3), the 186-signal early-warning
catalogue and the six-grade watchlist (§5.6), the amendment re-open matrix
(§5.7), the seven-level delegated authority matrix (§5.14), the cascade (§5.12),
the cross-facility collateral allocation (§5.11), and the comparability and
restatement machinery (§5.10).

Roughly **1 280 of the 1 900 decision points are consumed rather than written**
(§5.2). That ratio is the experiment's headline number, and the second headline
number is whatever fraction of those 1 280 ends up forked. §5.17.

---

## 5. The flow

One artefact. Nine ways in. Seventeen origination phases, six lifecycle phases,
and seven cross-cutting requirements that belong to no single phase and are the
reason this document is not project 05 with more products.

### 5.1 The nine entry points

The entry points are a structural concern, not an interface detail. They share
most of the flow and differ in which phases run, on what data, against what
latency budget, producing what output, for whom. A design in which an entry point
is a wrapper that calls "the flow" and then post-processes will discover that
four of the nine do not run most of the flow at all.

| # | Entry point | Trigger | Volume | Window / latency | Subject |
|---|---|---|---|---|---|
| **EP-1** | **New-to-bank business application** | Client applies | **900/day**, 09:00–16:00, 4× peak in the last two business days of the month | p95 **4 s**, p99 8 s, excluding data acquisition | Application |
| **EP-2** | **Additional facility for an existing client** | Client applies, or RM originates | **1 400/day** | p95 **4 s** | Application, against an existing client and facility set |
| **EP-3** | **Annual review** | `review_date` falls in the cohort month | **180 000/year**, in monthly batches of ≈ 15 000 by review-date cohort | **8-hour** window, overnight, one cohort per month | Facility, and the client's whole facility set |
| **EP-4** | **Covenant test** | A covenant instance's test date, or receipt of a compliance certificate | **≈ 2.4 M tests/year**; peak month ≈ 310 000 | Scheduled nightly; certificate-driven within **2 business days** of delivery | Covenant instance |
| **EP-5** | **Early warning evaluation** | Daily, whole book | **240 000 facilities/day** | **3-hour** window, complete by 05:00 | Facility, via client and entity |
| **EP-6** | **Limit or facility amendment** | Client request, RM request, or a covenant or cascade consequence | **600/day** | p95 **4 s** interactive; batch for consequence-driven amendments | Facility |
| **EP-7** | **Restructure or forbearance** | Arrears, hardship declaration, watchlist escalation, covenant breach | **120/day** | p95 **20 s** — a search, not a lookup | Client's whole facility set |
| **EP-8** | **Group exposure re-assessment** | Any entity change anywhere in any group (§5.12) | Event-driven; **≈ 1 900/day** cascades touching ≈ 14 000 facilities | p95 **15 min** from the triggering event; bounded | Group, as at the event date |
| **EP-9** | **Relationship-manager pre-assessment** | RM asks "what could I offer this client" | **3 000/day** | p95 **under 2 s** | Client, on partial data, committing nothing |

#### What each entry point runs, produces and feeds

Phases are named by the identifiers in §5.2: **O1–O17** origination,
**L1–L6** lifecycle.

| | Phases run in full | Phases run partially | Not run | Output | Consumed by |
|---|---|---|---|---|---|
| **EP-1** | O1–O17 | — | L1–L6 | Decision, offer, covenant set, conditions, authority routing, committee pack | RM, credit analyst, committee, Documentation |
| **EP-2** | O1, O4, O6–O17 | O2, O3 (refresh only), O5 (changed entities only) | L1–L6 | As EP-1, plus a restated group and security position across the **existing** facilities (§5.11) | RM, analyst, committee, and the owners of the facilities whose position moved |
| **EP-3** | O1, O3–O17, **L1** | O2 (as-at-date resolution) | L2–L6 | Grade migration, re-pricing decision, limit decision, covenant reset proposal, exit recommendation, next review date | Portfolio Management, RM, committee, Provisioning |
| **EP-4** | **L2** | O7 (spreading of the delivered figures only, on the covenant's own definitions) | O1–O6, O8–O17, L1, L3–L6 | Test result, measured value, headroom, breach classification, cure clock, waiver requirement | Covenant Operations, RM, Early Warning, Provisioning |
| **EP-5** | **L3** | O5 (event classification for new events only), O6 (partial re-blend) | O1–O4, O7–O17, L1, L2, L4–L6 | Watchlist grade, signal set, grade-change events, mandated actions | Early Warning, Portfolio Management, RM, Provisioning |
| **EP-6** | **L4**, plus whichever of O1–O17 the amendment re-open matrix names | — | The rest of O1–O17, L1–L3, L5, L6 | Amended facility terms, re-tested security position across the shared pool, new authority level, covenant consequences | RM, Documentation, the owners of affected facilities |
| **EP-7** | **L5**, O7, O10, O12, O14, O15, O17 | O5, O6 (refresh) | O1–O4, O8, O9, O11, O13, O16, L1–L4, L6 | Concession package across the facility set, forbearance classification, staging consequence, NPV cost, authority routing | Workout, Provisioning, committee, the client |
| **EP-8** | O2 (as at the event date), O10 | O11, O12 (allocation re-test), L3 | O1, O3–O9, O13–O17, L1, L2, L4–L6 | Revised group composition and headroom per business; the list of facilities whose position changed; cascade record | Portfolio Management, RM of every affected business, Credit Governance |
| **EP-9** | O11 | O1, O4 (subset), O7 (last known), O9 (last grade), O13 (indicative only) | O2, O3, O5, O6, O8, O10, O12, O14–O17, L1–L6 | **Indicative, non-binding** capacity and price range, with its own confidence statement | The RM, and nobody else |

#### EP-9 deserves its own rules

A pre-assessment is the entry point most likely to be got wrong, because it looks
like a cheap version of EP-1 and is not.

1. **It commits nothing, and must be incapable of appearing to.** Its output is
   marked indicative, carries an explicit confidence band, and states the three
   things it did not do: no bureau enquiry, no screening refresh, no entity
   re-assessment.
2. **It must not create a footprint.** No credit enquiry is lodged. Consent for
   an enquiry has usually not been given at this point, so `core.consent` gates
   the entry point itself, not just its data use.
3. **It is not a decision of record**, and must not enter a facility's decision
   history — but it **is** recorded as an interaction, because the question that
   gets asked three months later is *"you told me R3 000 000 in March and offered
   R1 800 000 in June"*.
4. **Reconcilability is a hard requirement.** Where an EP-9 answer and a later
   EP-1 or EP-2 answer differ by more than the stated confidence band, the
   difference must be attributable to named causes: data acquired since, entities
   assessed since, a policy change, or the pre-assessment's own declared
   assumptions proving wrong. An unattributable divergence is a defect.
5. **Under 2 seconds, on partial data, with no external calls.** Which means it
   runs on the last-known state of everything and says so.

### 5.2 The phase inventory, and how a phase is read

Twenty-three phases. Each is stated in the same four terms:

- **Determines** — what the phase establishes.
- **Owned by** — which spec owns its detail. For O2–O16, this is nearly always
  [project 05](05-business-credit-nested-entities.md) or
  [project 00](00-shared-credit-core-library.md), and **the detail is not
  restated here**.
- **Entry points** — which of the nine run it.
- **What changes because it is inside a lifecycle** — the only genuinely new
  content for an origination phase, and the reason the phase appears at all.

#### The decision point budget

The 1 900 figure is not decoration; it is what §5.16's navigability requirement
and §9.3's reviewable artefact are measured against. A **decision point** is a
rule, gate, cap, threshold comparison, branch condition or cell selection that
can change an outcome.

| Phase | Decision points | Consumed or written |
|---|---|---|
| O1 Request and relationship resolution | 34 | Written |
| O2 Group and entity structure resolution | 68 | Consumed — [05](05-business-credit-nested-entities.md) §5.1 |
| O3 KYB, screening and registration status | 76 | Consumed — [05](05-business-credit-nested-entities.md) §5.2, §5.4 (E-AROD-01..04) |
| O4 Business absolute rules of disqualification | 88 | Consumed — [05](05-business-credit-nested-entities.md) §5.3 |
| O5 Per-entity rules, event classification, entity scoring | 196 | Consumed — [05](05-business-credit-nested-entities.md) §5.4–5.7 |
| O6 Roll-up to the people component | 62 | Consumed — [05](05-business-credit-nested-entities.md) §5.8 |
| O7 Financial spreading, haircuts, sector benchmarking | 132 | Consumed — [05](05-business-credit-nested-entities.md) §5.9 |
| O8 Behavioural assessment | 38 | Consumed — [05](05-business-credit-nested-entities.md) §5.10 |
| O9 Combined grade and adjustments | 54 | Consumed — [05](05-business-credit-nested-entities.md) §5.10, [00](00-shared-credit-core-library.md) §6.22 |
| O10 Group exposure aggregation and concentration | 58 | Consumed — [05](05-business-credit-nested-entities.md) §5.11, [00](00-shared-credit-core-library.md) §6.16 |
| O11 Appetite and facility eligibility | 96 | Partly written — nine facility types |
| O12 Security and collateral assessment | 88 | Partly written — cross-facility allocation (§5.11) |
| O13 Pricing and structure negotiation | 72 | Consumed — [05](05-business-credit-nested-entities.md) §5.12 |
| O14 Debt service coverage and personal affordability | 64 | Consumed — [05](05-business-credit-nested-entities.md) §5.12, [02](02-affordability-assessment.md) |
| O15 Covenant setting | 74 | **Written** |
| O16 Conditions precedent and subsequent | 42 | Partly consumed — [05](05-business-credit-nested-entities.md) §5.13 |
| O17 Authority routing and decision record | 38 | Partly written — seven levels (§5.14) |
| **Origination subtotal** | **1 280** | ≈ 78% consumed |
| L1 Annual review | 164 | **Written** |
| L2 Covenant monitoring | 148 | **Written** |
| L3 Early warning and watchlist | 142 | **Written** |
| L4 Amendment | 58 | **Written** |
| L5 Restructure and forbearance | 76 | Consumed — [06](06-consolidation-and-restructure.md) §5.9, extended |
| L6 Exit and handoff | 32 | **Written** |
| **Lifecycle subtotal** | **620** | ≈ 12% consumed |
| **Total** | **1 900** | |

Every one of the 1 900 must resolve to exactly one owning team (§5.15), a
plain-language description, its effective dates, the reason codes it can raise,
and the tables it reads. A decision point with no owner, or two, is a defect
found by a check rather than by an incident.

### 5.3 The origination phases

Seventeen phases, run in full by EP-1 and in declared subsets by six of the other
eight. This is [project 05](05-business-credit-nested-entities.md)'s flow. It is
referenced, not restated; what follows for each phase is only what the lifecycle
changes.

#### O1 — Request and relationship resolution

**Determines**: what is being asked for, by whom, against what existing
relationship, on what `decision_date` and at what `knowledge_date`.

**Owned by**: this project. It is the one origination phase project 05 does not
have in this form, because project 05's applicant either exists or does not,
whereas here the client may have eleven facilities, three of them on watchlist,
one forborne and one in a cross-default event nobody has called.

**Entry points**: all nine, in nine different forms.

**What changes inside a lifecycle**:

- The request may be against a facility rather than for one, and the phase must
  resolve `facility_id` and load its decision history (§5.10) before anything
  else runs.
- `decision_date` and `knowledge_date` are set here and are separate (§5.13).
  For EP-3 the `decision_date` is the review date, which may precede the run date
  by up to 30 days; for EP-4 it is the covenant's test date, which may precede
  the run by 90 days because the certificate arrived late. **A phase that reads
  the run date is wrong in both cases and will replay successfully with the wrong
  answer** ([09](09-governance-and-replay-harness.md) §5.15 item 4).
- Existing state is loaded and made available to every later phase: watchlist
  grade, forbearance flag and probation clock, staging code, cross-default
  state, live waivers, review status. Four of these are **authority modifiers**
  (§5.14) and one — the forbearance flag — changes what L5 is permitted to do.

**Records**: the request, both dates, the facility set as at `knowledge_date`,
and the four state flags with their as-at dates.

#### O2 — Group and entity structure resolution

**Determines**: the bounded, de-duplicated entity list, and the group.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.1 —
unchanged: 3 levels, 40 entities, a 5.0% materiality floor, path products to 4
decimal places, cycle truncation, de-duplication by `entity_key` with summed
ownership and most-senior role.

**Entry points**: EP-1, EP-2 (refresh), EP-3, EP-8; EP-5 and EP-9 read a cached
resolution rather than recomputing.

**What changes inside a lifecycle**: **the resolution is as at a date, over a
bi-temporal fact store** (§5.13). Project 05 resolves a structure supplied with
the application. Here the structure is derived from a history, and two different
questions are asked of it:

| Question | Resolution |
|---|---|
| What did the Bank know when it decided? | `known_from ≤ knowledge_date`, and `effective_from ≤ decision_date` |
| What was actually true then? | `effective_from ≤ decision_date < effective_to`, regardless of `known_from` |

The first is what replay needs. The second is what an ownership-change covenant
needs. They give different answers for about **9% of assessments** and the
difference is not detectable from the output, which is why it must be a declared
property of the phase rather than an implementation choice.

**Records**: the resolution, both date parameters, and — new here — the
**structure delta against the previous decision of record**: entities added,
removed, ownership moved, roles changed, criticality classes changed. That delta
is an input to L1's comparability statement and to L3's signals.

#### O3 — KYB, screening and registration status

**Determines**: whether the Bank may do business with this business and these
entities at all.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.2 (regime)
and §5.4 (E-AROD-01 to E-AROD-04), plus `core.eligibility` and `core.consent`.

**Entry points**: EP-1, EP-2, EP-3; EP-8 for a changed entity only.

**What changes inside a lifecycle**:

- Screening is not a one-off. An entity clear in 2026 can be a confirmed match in
  2029, and the consequence is not "decline the application" — there is no
  application — but a **cascade** (§5.12) plus a facility-level consequence that
  ranges from a note to immediate freeze of undrawn amounts. The disposition
  matrix of project 05 §5.4 gains a fourth column: *what this does to a facility
  that already exists*.
- The regulated/unregulated regime determination can **change between reviews**
  when turnover crosses R1 000 000 or the facility crosses R250 000. A facility
  originated unregulated and reviewed regulated does not retrospectively become
  regulated — the agreement's regime is fixed at contract — but the *next*
  facility does, and the review must say so. Confusing the regime of the contract
  with the regime of the assessment is a real defect class.
- Tax compliance and registry filing status must be re-checked at every review,
  and an information undertaking exists on both, so a failure here is
  simultaneously a policy finding (O4) and a **covenant breach** (L2). One fact,
  two consequences, two owners, and they must not double-count.

#### O4 — Business absolute rules of disqualification

**Determines**: whether the business itself disqualifies.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.3 — twenty
rules, B-AROD-01 to B-AROD-20, complete fired set, curability flag per reason.

**Entry points**: EP-1, EP-2, EP-3, EP-7; EP-9 runs a declared subset of eleven.

**What changes inside a lifecycle**: **seven of the twenty rules mean something
different when the facility already exists.** A rule that declines a new
application cannot decline a facility the Bank has already written — it produces
a finding, and the finding has a defined consequence.

| Rule | At origination | At review, on an existing facility |
|---|---|---|
| B-AROD-01 deregistered / final liquidation | Decline | Immediate exit trigger; facility frozen; L6 |
| B-AROD-02 in deregistration | Decline, curable | Condition subsequent with a 60-day clock; watchlist W2; covenant breach on the information undertaking |
| B-AROD-03 business rescue | Refer | Mandatory L5 assessment and a post-commencement finance decision; Stage 3 candidate |
| B-AROD-07 excluded sector | Decline | **Run-off**: no new lending, no limit increase, no term extension; existing facility honoured to maturity |
| B-AROD-09 tax non-compliant | Decline, curable | Covenant breach, 30-day cure, drawstop on undrawn amounts |
| B-AROD-13 prior write-off | Decline | Not applicable — the write-off may be on this very client |
| B-AROD-20 existing facility ≥ 90 days in arrears | Decline | The subject *is* the facility; routes to L5 or L6 |

The requirement is that this mapping be **declared data**, in one place, and not
re-derived by each entry point. Project 05's rule set is consumed unchanged; what
is added is a per-rule, per-entry-point disposition — which is the same shape as
project 05's own 14 × 3 criticality disposition matrix, and probably the same
construct.

#### O5 — Per-entity rules, adverse event classification and entity scoring

**Determines**: for each entity, its criticality class, its disqualification
verdict, the classification of each of its adverse events, its adverse verdict,
and its score, PD and grade.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.4, §5.5,
§5.6 and §5.7 in their entirety. Fourteen entity rules over three criticality
classes; the 84-cell threshold table; thirty-four classification rules; the
84-cell ageing decay; twelve roll-up rules; two scorecard families with
thin-file variants; four distinct thin-file/no-hit situations. **None of it is
restated here.**

**Entry points**: EP-1, EP-2 (changed entities), EP-3, EP-5 (new events only),
EP-8 (the triggering entity).

**What changes inside a lifecycle** — three things, and the third is the hardest:

1. **The unit of work shrinks.** The daily pass and the cascade need *one event
   on one entity* classified and *one entity's verdict* re-rolled, not a whole
   application assessed. Project 05 specifies this as its partial re-assessment
   requirement (§8, 500 ms) but specifies it as an optimisation of a whole
   assessment. Here it is the **primary** call pattern: 9 400 event
   classifications a day arrive as singletons, not as applications. §5.17.
2. **Entity criticality can change with no new data about the entity.** A
   director's ownership rises from 22% to 27% because somebody else sold shares.
   The entity's own record is unchanged; its criticality class moves from
   significant to critical; every one of its events reclassifies against
   different thresholds; its verdict may change; the people blend changes. The
   cause is *another entity's* change, which is exactly the cascade shape
   (§5.12).
3. **The same entity is assessed for several businesses at once.** Project 05
   §9.6 requires consistency across two applications on one day. Here, one
   director may sit on six businesses with eighteen facilities, and the daily
   pass assesses all of them. The classifications must be identical where the
   criticality class is identical and must legitimately differ where it is not —
   and the record of each must be able to point at the others.

#### O6 — Roll-up to the people component

**Determines**: one PD and grade for everybody standing behind the business, plus
the caps they impose and the surety and guarantee cover they provide.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.8 — PP-01
to PP-11, log-odds blending, the 75% coverage requirement, the control weight
floor of 0.60, the surety cover table, the three required outputs `people_pd`,
`people_pd_unadjusted` and `people_pd_overlay_contribution`.

**Entry points**: EP-1, EP-2, EP-3, EP-5 (partial), EP-7 (refresh), EP-8.

**What changes inside a lifecycle**:

- The weight vector becomes a **time series**, and project 05 §5.8's closing
  requirement — that "why did this grade move from 6 to 7 when nothing changed"
  is answered by the weight vector and the overlay decomposition together — now
  spans years rather than quarters. The causes list grows from three to six:
  the entity's own data changed; the model changed; an overlay changed; **the
  structure changed**; **the master scale changed**; **the blend rules
  themselves changed**. §5.10.
- A **partial re-blend** is needed: one entity's PD moves, the weights are
  unchanged, the blended result is not. Project 05 does not publish this as a
  separate answer shape. §5.17.
- Sureties expire, die, resign and are substituted. PP-08's cover calculation
  runs on a surety set that changes between reviews, and a facility whose surety
  cover has fallen below what it was priced on is a finding at L1 and a signal at
  L3 — not silently re-priced.

#### O7 — Financial statement spreading, quality haircuts and sector benchmarking

**Determines**: the 48-line spread, eleven derived measures, nine sector
benchmark bands, `financial_pd` and `financial_confidence_code`.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.9 — the
~600-label mapping table, the annualisation rules, the audit-level haircuts, the
statement-age haircuts, the bank-turnover alternative with its R1 500 000 cap and
grade cap of 6, and the distinction between a data-quality haircut and a policy
overlay.

**Entry points**: EP-1, EP-2, EP-3, EP-4 (the covenant's own definitions only),
EP-7; EP-9 uses the last spread on file.

**What changes inside a lifecycle** — four things:

1. **1..6 periods, not 1..3.** Five years of reviews accumulate statements, and
   the comparability requirement means the early ones cannot be discarded.
2. **Periods may be on different accounting bases.** A lease accounting change
   between 2027 and 2028 moves operating lease commitments onto the balance
   sheet, which changes gearing, interest cover and EBITDA — for the *same
   business*, with no change in its economics. The spread must carry the basis
   per period, and any measure computed across periods of different bases must
   either restate or refuse. This is change scenario 4 in §11 and it is not
   hypothetical.
3. **Covenant definitions freeze their own spreading rules.** A DSCR covenant
   agreed in 2026 defines its numerator and denominator by reference to specific
   standard lines, on a stated basis, over a stated reference period. When the
   mapping table changes in 2029, the covenant test must still compute the 2026
   definition. The review's DSCR and the covenant's DSCR are then **two different
   numbers for the same business in the same year**, both correct. §5.5.
4. **The confidence code drives the degraded review ladder** (§5.4) rather than
   just a grade cap.

#### O8 — Behavioural assessment

**Determines**: a behavioural PD from internal conduct.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.10's
behavioural component and scorecard `BUS-BEH-01`, 21 characteristics.

**Entry points**: EP-2, EP-3, EP-5 (as signal inputs), EP-7.

**What changes inside a lifecycle**: it becomes the **most** informative
component rather than the least. At origination for a new-to-bank client it is
absent and its 0.15 weight redistributes. At the third annual review there are 36
months of conduct, and the same 0.15 weight is now carrying the only genuinely
current information in the assessment — the financials are 14 months old and the
bureau view is 30 days old, but the excess days are from yesterday.

The requirement that follows: **the weight table must be able to vary by
relationship tenure**, not only by segment and financial confidence. Project 05's
18-combination weight table becomes 18 × 4 tenure bands = 72. That is change
scenario 6 in §11, and whether it is a table edit or a re-release is the test.

#### O9 — Combined business grade and adjustments

**Determines**: `risk_grade` and `probability_of_default` for the business, with
their unadjusted counterparts and the overlay decomposition.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.10 — the
component weights, log-odds blending, the bounded qualitative override, the
eleven-position declared overlay stack, and the override-is-not-an-overlay
distinction. `core.adjustments` ([00](00-shared-credit-core-library.md) §6.22)
governs the overlays.

**Entry points**: EP-1, EP-2, EP-3, EP-7; EP-9 reads the last grade.

**What changes inside a lifecycle** — one thing, and it is the document's spine:

**The grade must carry the scale it is expressed on.** `master_scale_version` is
a required output of this phase, alongside `risk_grade`. A grade without its
scale is a number that cannot be compared with any other grade, and five years
of reviews will span at least one recalibration. §5.10 develops what follows.

Second, smaller but sharp: **an overlay that expired between two assessments is a
cause of grade movement**. Project 05 already requires the overlay decomposition;
here the decomposition must be **differenced** against the previous decision of
record, so that "the construction sector multiplier of 1.25 expired on
2028-09-30" appears as a named cause of a grade improvement rather than as an
unexplained one.

#### O10 — Group exposure aggregation and concentration limits

**Determines**: aggregate exposure across the group, headroom, and which
concentration limit binds.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.11's group
exposure section and `core.exposure`
([00](00-shared-credit-core-library.md) §6.16). The group is every business
sharing a critical entity — common control or ≥ 25% common ownership. Caps by
grade: R25 000 000 down to nil at grades 11–12.

**Entry points**: EP-1, EP-2, EP-3, EP-7, EP-8.

**What changes inside a lifecycle** — three things:

1. **Contingent exposure.** Products 55 and 56 create exposure that is not drawn
   and may never be. Guarantees count at 100% of face value for group exposure
   and at a 50% conversion factor for appetite; trade instruments count at their
   face value until the underlying transaction settles. A guarantee issued in
   2027 and expiring in 2030 occupies headroom for three years and then releases
   it, with no decision taken — which means **headroom changes without a
   decision**, and the phase must be runnable on a schedule as well as on a
   request.
2. **Economic interdependence widens the group beyond control.** Two businesses
   with no common shareholder, where one derives 60% of its turnover from the
   other and occupies premises it owns, are a single risk. Project 05's group
   definition is control-and-ownership only; this project's is control,
   ownership **and** declared economic dependence, and widening it is change
   scenario 5 in §11 and project 05's change scenario 13.
3. **The group is a moving set.** §4.1's rule — a group is identified by its
   composition as at a date — is enforced here. Every group-level output carries
   its composition date and the member list that produced it.

#### O11 — Appetite and facility eligibility

**Determines**: which of the nine facility types this client may have, and the
ceilings each is subject to.

**Owned by**: `core.appetite` ([00](00-shared-credit-core-library.md) §6.17) and
[project 05](05-business-credit-nested-entities.md) §5.11's appetite grid,
**extended** by this project to nine facility types.

**Entry points**: all nine. It is the only phase every entry point runs, and it
is what EP-9 exists to answer.

**What changes**: this is the first origination phase with substantial new
material. Project 05's appetite maximum table is 12 grades × 5 sector appetite
classes × 4 security types = 240 cells for two products. Nine products make it
**2 160 cells**, and the facility types are not interchangeable — a grade 8
client may have an overdraft and may not have a bridging facility at any amount.

| Constraint | Source |
|---|---|
| Maximum facility amount | Appetite grid, grade × sector class × security type × facility type |
| Eligible facility types at this grade | Facility eligibility matrix, 12 grades × 9 types, with reasons |
| Maximum term | The lesser of the product maximum, the sector maximum term, and — for product 52 — 80% of assessed asset life |
| Group headroom | O10 |
| Sector portfolio cap | Portfolio Management |
| Single-name concentration | Portfolio Management |
| Watchlist restriction | W2: no increase; W3: no new facility; W4+: no new exposure of any kind |
| Forbearance restriction | Live forbearance: no new facility without L5-level authority |

The last two are new and are the lifecycle's contribution: **appetite is a
function of facility state, not only of client risk.** A grade 5 client on
watchlist W3 has less appetite than a grade 7 client at W0, and the reason is not
in any scorecard.

#### O12 — Security and collateral assessment

**Determines**: the adjusted cover available, its allocation across facilities,
and the resulting `security_type` per facility.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.11's
collateral advance rates and security type derivation — 70% first bond, 45%
second bond, 40% debtors under 90 days, 25% plant, 20% stock, 60% listed
securities, 100% cash; cover ratio → security type 1..4.

**Entry points**: EP-1, EP-2, EP-3, EP-6, EP-7, EP-8 (re-test only).

**What changes inside a lifecycle**: **everything about allocation**, which
project 05 does not need because it has one facility. This is developed properly
at §5.11 and is one of the two hardest requirements in the document.

Three smaller changes: the advance rate table extends from 8 collateral classes
to 14 (adding the financed asset under instalment sale, commercial property by
type, eligible debtors under a borrowing base, guarantee counter-indemnity cover,
ceded deposits and ceded proceeds); valuations age and must be refreshed on a
cadence by class; and **a revaluation is an event with consequences** rather than
a field update.

#### O13 — Pricing and structure negotiation

**Determines**: the largest admissible amount and its complete structure.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.12 in full —
the circularity between amount, security type and rate; the non-monotonicity that
makes bisection invalid; the requirement to match exhaustive evaluation over
5 000 historical applications; the bounded, terminating search with a declared
candidate ordering and a declared truncation behaviour; the twelve binding
constraint codes; the rate floor.

**Entry points**: EP-1, EP-2, EP-6, EP-3 (repricing only), EP-9 (indicative
only).

**What changes inside a lifecycle**:

- **Six more rate tables and six more candidate spaces.** Product 53's space is
  24 amount bands × 10 LTV bands × 181 terms; product 52's is bounded by asset
  life; product 54's is a borrowing base rather than an amount; products 55 and
  56 price per instrument, not per facility. The search shape is project 05's;
  the spaces are not.
- **Re-pricing is not pricing.** At L1 the facility exists, and the question is
  not "what would we charge" but "what do we charge, what would we charge, and
  what may we do about the difference". §5.4.
- **The authority level moves while this phase runs** (§5.14). Each candidate
  carries an authority level, and the search is therefore over a space in which
  the approver changes.

#### O14 — Debt service coverage and, where the owner is the business, personal affordability

**Determines**: whether the proposed structure is serviceable.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.12 items 6
and 7 — the DSCR, interest cover and gearing thresholds by grade band — and
[project 02](02-affordability-assessment.md) in full for the statutory
assessment on the natural person behind a regulated agreement.

**Entry points**: EP-1, EP-2, EP-3, EP-6, EP-7.

**What changes inside a lifecycle**:

- Project 02 publishes four assessment modes (§5.8: new application, limit
  increase, arrangement, scenario). **A fifth is needed** — a periodic
  re-assessment on a sole proprietor whose evidence is a year old and whose
  business income is the same money as their personal income, counted once.
  §5.17 develops what a consuming project does when a component gives it 90% of
  what it needs.
- The DSCR computed here and the DSCR tested by a covenant (L2) are different
  numbers on different definitions, and the review pack must show both without
  implying that one is wrong.

#### O15 — Covenant setting

**Determines**: which covenants attach to this facility, at what levels, on what
schedules, under which definition versions.

**Owned by**: **this project.** Project 05 §5.13 lists six covenants in four
lines and monitors none of them. This is where the lifecycle actually begins, and
it is the phase whose output is consumed for the next five years by an entry
point (EP-4) that runs 2.4 M times a year.

**Entry points**: EP-1, EP-2, EP-3 (reset proposal), EP-6 (reset), EP-7 (reset).

**What it produces**: a set of **covenant instances**, each of which is:

| Attribute | Notes |
|---|---|
| `covenant_instance_id` | Unique, stable for the life of the covenant |
| Template | One of 148 in the definition library (§6.3) |
| `covenant_definition_version` | **The version agreed now, frozen for the life of this instance** |
| Level and step schedule | A covenant is not one number: gearing ≤ 3.0 falling to 2.75 at month 18 and 2.50 at month 36 |
| Reference period | Most recent audited period / rolling 12 months to the test date / annualised interim |
| Test dates | Derived from the schedule rule and the client's financial year end |
| Delivery terms | What must be delivered, by when, in what form, signed by whom |
| Cure rights | Whether an equity cure is available, how it applies, and its limits |
| Headroom basis | The base case against which headroom is measured |
| Tested entity | Solo, consolidated, or a named obligor group |

**How the set is chosen**: a default set by facility type × grade band × amount
band (180 rows, §6), then rule-driven additions — a director loan subordination
undertaking where gearing exceeds 2.5; a clean-down where the facility is
revolving; a borrowing base certificate for product 54; an LTV covenant for 53;
tenancy undertakings where rental income services the debt; ownership-change
consent above R2 000 000.

**The two hard requirements set here and enforced at L2**:

1. **Covenant levels must be set with declared headroom to the base case**, and
   the headroom must be recorded. A DSCR covenant at 1.30 against a base case of
   1.31 is not a covenant, it is a trap; the Bank's convention is a minimum 20%
   cushion on cover ratios and 0.25× on leverage-style covenants, and a set
   outside that convention is an exception requiring the next authority level up.
2. **The definition version is frozen here and never moves.** §5.5, and H2.

#### O16 — Conditions precedent and conditions subsequent

**Determines**: what must be true before the facility funds, and what must become
true afterwards.

**Owned by**: [project 05](05-business-credit-nested-entities.md) §5.13's
34-condition catalogue, **extended to 61** by separating precedent from
subsequent and adding the seven new facility types' requirements.

**Entry points**: EP-1, EP-2, EP-6, EP-7.

**What changes inside a lifecycle**: conditions **subsequent** are a lifecycle
object and project 05 has none. A condition subsequent has a deadline, an owner,
an escalation ladder and a consequence on failure — and failure is a covenant
breach, not a re-decision. The commonest are bond registration and delivery of
the registration certificate (30 business days), notarial bond registration (30
days), accession of a guarantor to meet a cover test (20 days), insurance
endorsement and cession (15 days), and account-control arrangements for products
54 and 58 (10 days).

The requirement: **an extension of a condition subsequent is an authority act
with a limit**, and repeated extensions are an early warning signal in their own
right (§5.6). A condition subsequent extended three times is a condition that was
never going to be met.

#### O17 — Authority routing and the decision record

**Determines**: who must approve this, what they are given, and what is written
down.

**Owned by**: partly [project 05](05-business-credit-nested-entities.md) §5.13's
mandatory committee triggers and §9.3's committee pack; the authority matrix
itself is this project's and is developed at §5.14.

**Entry points**: all nine, including EP-9 — which routes to nobody and must
record that it routed to nobody.

**What changes inside a lifecycle**: five authority levels become seven; the
matrix gains facility type as a dimension; state flags become authority
modifiers; the pack becomes level-dependent; and the authority recomputes as the
proposed structure moves (§5.14, H7).

And one requirement that belongs to no other phase: **the decision of record is
written here**, appended to the facility's decision history, linked to its
predecessor, carrying its comparison basis (§5.10). Every entry point writes one
except EP-9, and EP-9 must be structurally incapable of writing one.

### 5.4 L1 — Annual review

**Trigger**: `review_date` falls in the cohort month. 180 000 reviews a year, run
in monthly batches of roughly 15 000 by review-date cohort, in an 8-hour
overnight window.

**Determines**: whether the Bank still wants this facility on these terms, and
what has changed since it last asked.

A review re-runs most of the origination assessment — O1, O3 to O17 — on fresh
financials, a refreshed structure, a refreshed bureau view and a year of conduct.
What makes it a distinct phase rather than a re-run is that it produces **four
outputs origination does not produce**, and all four are comparisons.

#### 5.4.1 The four outputs

**1. Grade migration.** Three grades, not one: the grade at origination, the
grade at the last review, and the grade now — each with its
`master_scale_version`, its component decomposition, and its overlay
decomposition. Plus a **cause decomposition** of the movement since the last
review, apportioned across six named causes:

| Cause | Example |
|---|---|
| The business's own data | Turnover fell 14%; DSCR fell from 1.64 to 1.28 |
| The entity structure | A 30% shareholder resigned as director; criticality class of two entities changed |
| An entity's own data | A R240 000 judgment against the controlling member |
| The model | `BUS-COMM-01` replaced by `BUS-COMM-02` in March |
| An overlay | The construction sector PD multiplier of 1.25 expired on 30 September |
| The scheme | Master scale v3 → v4; the 6/7 boundary moved by 0.35 percentage points of PD |

The requirement: **the six must sum to the observed movement**, and a residual
that cannot be apportioned is a defect, reported as one. This is the same
requirement as [project 09](09-governance-and-replay-harness.md) §5.5's swap-set
attribution, applied to one subject over a year rather than to a population over
a release.

**2. A re-pricing decision.** Three numbers and an action:

| | |
|---|---|
| Contractual rate | What the client pays, and on what basis — fixed, or a margin over a reference rate |
| Indicated rate | What O13 would price today at the current grade, security type and facility type, including the rate floor |
| Margin shortfall | The difference, in basis points and in rand per year |
| Action | None / reset the margin on notice / reprice at the next roll or drawdown / flag for renegotiation / decline to renew |

Most facilities cannot be repriced mid-term: a product 50 term facility at a
fixed margin is fixed. Products 51, 54, 57 and the trade and guarantee lines can
be repriced on notice. The requirement is that **the shortfall be reported
whether or not it can be acted on**, because the aggregate of unactionable
shortfalls across the book is a portfolio fact the Credit Committee needs, and a
review that silently drops it produces a book that is systematically underpriced
for reasons nobody recorded.

**3. A limit decision.** For revolving facilities — 51, 54, 57 and the trade and
guarantee lines, about 96 000 of the 240 000 — the review decides whether the
limit increases, holds, reduces or is cancelled. This consumes
[project 07](07-credit-limit-management.md)'s machinery: a record-level decision
inside a portfolio-level budget, simulated before deployment. What is new is
that the record here is a facility with covenants and shared collateral, so a
limit change re-opens O12's allocation across every facility sharing that
collateral (§5.11), and a limit reduction on an undrawn facility is a decision
the client must be given notice of.

**4. An exit recommendation, where appropriate.** Produced against eighteen
declared exit triggers (§6): grade at or worse than 10 for two consecutive
reviews; sector in run-off; three or more covenant waivers in 24 months;
forbearance granted twice; watchlist W3 or worse for 180 days; security cover
below 0.40 with no remediation plan; relationship no longer economic after the
rate floor. A recommendation is not a decision — L6 is — but it must be produced,
and a review that produces no recommendation on a facility meeting four triggers
is a finding.

#### 5.4.2 Comparability — the hard requirement

> A review in 2031 must be comparable with the origination in 2026, even though
> the scorecards, the sector table, the appetite grid and the flow itself have
> all changed.

**What comparability means.** Two assessments of the same facility are comparable
when all five hold:

1. **Same subject.** `facility_id` and `client_id` resolve to the same thing at
   both points, and where the structure changed, the change is enumerated rather
   than absorbed.
2. **Same quantities, or a declared mapping.** A DSCR at t₀ and a DSCR at t₁ are
   comparable when they were computed on the same definition, or when a mapping
   between the two definitions is published. Two numbers with the same name and
   different definitions are not comparable and must not be put in the same
   column.
3. **Same scale.** Both grades expressed on one `master_scale_version`, either
   because it did not change or because a restatement exists (§5.10).
4. **Decomposable difference.** Every unit of movement attributable to one of the
   six causes above.
5. **Both reproducible.** Each assessment re-derivable on the artefacts in force
   at its own `decision_date` and `knowledge_date`, with no external calls
   ([09](09-governance-and-replay-harness.md) §5.15).

**What comparability requires.** Each of the five imposes an obligation on every
decision of record:

| Requirement | Obligation |
|---|---|
| Same subject | Stable `facility_id` across amendments and restructures; the structure delta from O2 recorded |
| Same quantities | Every derived measure records the definition version that produced it, not only its value |
| Same scale | `master_scale_version` on every grade; restatement maps published, versioned and dated (§5.10) |
| Decomposable difference | The previous decision of record is an **input** to this one, and the cause decomposition is a required output |
| Reproducible | The 23-item contract of [project 09](09-governance-and-replay-harness.md) §5.15, in full, on every one of the 1 900 decision points |

And the requirement that makes it enforceable rather than aspirational:

> Every decision of record carries a `comparison_basis_code` against its
> predecessor — **as-graded**, **restated**, or **not comparable** — with the
> reason. A report that places two grades side by side must read that code, and a
> report mixing bases is invalid and must be rejected rather than footnoted.

#### 5.4.3 The degraded review

**30% of reviews run on stale or incomplete financials.** This is the normal
case, not the exception (H9), and there must be a defined degraded review rather
than an indefinite deferral. Five bases, and the review must land on exactly one:

| `review_basis_code` | Condition | Treatment |
|---|---|---|
| **1 — complete** | Most recent year-end ≤ 9 months, management accounts ≤ 3 months | Full review. No caps. Next review in 12 months. |
| **2 — late** | Year-end 9–15 months | 5% EBITDA haircut ([05](05-business-credit-nested-entities.md) §5.9). Full review. Information undertaking breach recorded if the covenant deadline has passed. Next review in 12 months. |
| **3 — stale** | Year-end 15–21 months, management accounts mandatory and present | 15% haircut. Grade capped at 7. **No limit increase, no term extension, no new facility.** Next review brought forward to **6 months**. Watchlist W1 minimum. |
| **4 — turnover-only** | Year-end > 21 months, or absent; ≥ 6 months of turnover through the Bank | [Project 05](05-business-credit-nested-entities.md) §5.9's bank-statement alternative: grade capped at 6, facility capped at R1 500 000 or the existing limit if higher. Undrawn amounts frozen. Next review in **3 months**. Watchlist W2 minimum. Covenant breach on the information undertaking, with its cure clock running. |
| **5 — not performed** | No financials, no turnover (the client banks elsewhere), no response | **The review still completes.** Outcome: grade unchanged but flagged `grade_stale`, facility frozen, watchlist W2, exit assessment mandatory within 60 days, and the review recorded as *performed, evidence absent* — which is a different fact from *not performed*, and the second must be impossible. |

Three rules govern the ladder:

1. **A review is never deferred.** Every facility in the cohort produces a
   decision of record by its review date plus 30 days. "Awaiting financials" is
   not an outcome; basis 4 or 5 is.
2. **The basis is an output, and it drives the next review date.** A degraded
   review buys less time than a complete one, which is the mechanism by which the
   book does not silently drift onto stale information.
3. **Degradation is not a decline.** Basis 4 and 5 restrict, they do not
   terminate. A facility is only exited through L6.

**Why this is a structural requirement rather than an error-handling one**: if
the complete-evidence path is the main path and the other four are branches off
it, the main path is used 44% of the time and four-fifths of the logic lives in
exception handling that nobody reviews. The five bases must be five declared
modes of one assessment, in the same sense as
[project 02](02-affordability-assessment.md) §5.8's four modes — a mode selects
evidence rules, parameter sets and which outputs are produced, and **a mode may
not change the arithmetic**.

#### 5.4.4 The review's own record

Additional to the standard decision record: the review basis, the cause
decomposition, all three grades with their scales, the re-pricing numbers and
action, the limit decision with its portfolio context, the exit triggers met, the
covenant compliance history for the year, the waiver history, the watchlist
history, the conditions subsequent outstanding, and the next review date with
the rule that set it.

### 5.5 L2 — Covenant monitoring

**Trigger**: a covenant instance's test date, or receipt of a compliance
certificate. **≈ 2.4 M tests a year** across ≈ 960 000 live covenant instances on
240 000 facilities, mean 4.0 per facility, range 0–14.

| Test frequency | Share of instances | Tests/year |
|---|---|---|
| Annual | 68% | ≈ 653 000 |
| Quarterly | 26% | ≈ 998 000 |
| Monthly | 6% | ≈ 691 000 |
| | | **≈ 2 342 000** |

Seasonality is severe: 62% of annual tests attach to the two commonest financial
year ends, so the peak month carries **≈ 310 000 tests** against a mean month of
195 000. §8.

**Determines**: for each test — whether the covenant was met, by how much, what
class of breach if not, what cure is available, what waiver is required, and what
the consequences are for this facility, the client's other facilities and the
group.

#### 5.5.1 The covenant definition as a versioned artefact

This is H2 and it is the requirement most likely to be implemented wrongly,
because the wrong implementation is the one every other artefact in the estate
uses.

> Policy artefacts resolve by `decision_date`
> ([00](00-shared-credit-core-library.md) §7.3). **Contractual artefacts resolve
> by the version agreed when the instance was created, and never move.**

A DSCR covenant written into a facility agreement on 2026-03-14 is a term of a
contract. The Bank's standard DSCR wording changing in 2028 — a new treatment of
lease liabilities in the denominator, say — does not change that contract. A test
run in 2029 must use **definition version 2026-03-14**, not the current standard,
not the version in force at the test date, and not the version in force at
`decision_date`.

The failure mode is precise: an effective-dated resolution by `decision_date`
picks the 2029 standard, computes a plausible number, and breaches a covenant the
client did not breach. It will not error. It will not look wrong. It will be
wrong.

**The definition library** (§6.3) holds **148 templates** and roughly **1 900
live definition versions** — because a template that has been revised four times
has four versions, all of which are still in force somewhere in the book. A
version is retired only when the last instance bound to it closes, which for
product 53 can be twenty years.

**A definition version carries** (22 attributes, §6.3): the defined term; the
measure with its numerator and denominator by reference to the 48 standard lines;
the accounting basis and whether it is frozen GAAP; the reference period
convention; the test date rule; the delivery obligation and its deadline; the
form of certification and who signs; whether a grace period runs from the test
date or from delivery or from the Bank's knowledge; the threshold and its step
schedule; cure rights and their limits; the entity or obligor group tested;
permitted exclusions and add-backs; the headroom basis; and its own effective
dates as a *standard*, distinct from the dates it is in force as an *instance*.

#### 5.5.2 Three dates, never conflated

Every test has three dates and collapsing any two of them is a defect class:

| Date | Meaning | Example |
|---|---|---|
| **Test date** | The date as at which the ratio is measured | 2029-06-30, the client's quarter end |
| **Delivery date** | The date the certificate and accounts are due | 2029-08-29, 60 days later |
| **Determination date** | The date the Bank records the result | 2029-09-04, when Covenant Operations processed it |

Three consequences follow, and each is a requirement:

1. The ratio is computed on data as at the **test date**, on the definition bound
   to the instance, regardless of what is known later.
2. The breach, if any, **occurs on the test date** — which means a cure period
   measured from the test date can be part-spent before the client knows. The
   definition states which convention applies, and this project's standard is
   that grace runs from the earlier of delivery and the Bank's knowledge, with
   the older facilities on the older convention.
3. The **determination date** is when the flow ran and is what
   [project 09](09-governance-and-replay-harness.md) replays against. A test run
   three months late must reproduce as the test it was, not as a test of today.

And a fourth state which is neither pass nor breach: **not tested**, because the
certificate has not arrived. That is a breach of the *information undertaking*,
which is a different covenant instance with its own definition, its own cure and
its own waiver. One late certificate therefore produces one certain breach and
one unknown result, and a design that collapses them will report a financial
covenant breach the client has not committed or a pass the Bank has not
established.

#### 5.5.3 Breach classification

Four classes, from a 592-cell matrix of covenant template × severity band.

| `breach_class_code` | Class | Typical cause | Consequence |
|---|---|---|---|
| 0 | **None** | Met, with headroom | Headroom recorded; headroom below 10% is a signal (§5.6) |
| 1 | **Technical** | Certificate late, wrong form, unsigned, an administrative information undertaking missed | Cure period; RM contact; no drawstop; three technical breaches in 12 months escalate to class 2 |
| 2 | **Material** | A financial covenant missed by less than 20% of the threshold's headroom convention; an undertaking breach with no loss | Cure period; **drawstop on undrawn amounts**; waiver required to restore availability; watchlist W1 minimum; SICR assessment |
| 3 | **Severe** | A financial covenant missed by more than that; negative pledge breach; disposal of secured assets; ownership change without consent; a breach that is also an insolvency indicator | No cure. Immediate committee referral, watchlist W3 minimum, security review, staging assessment, cross-default assessment across the client and the group |

The classification consumes the *measured* value and the *threshold*, so it is
computable — but the band edges are policy, owned by Business Credit Risk Policy,
and the mapping from template to band structure is owned by Legal. Two owners,
one matrix, and §5.15's problem in miniature.

#### 5.5.4 Cure

| Covenant family | Cure available | Period |
|---|---|---|
| Information undertakings | Deliver the thing | 15 business days from the earlier of notice and knowledge |
| Financial ratio covenants | **Equity cure**, where the definition grants it | 30 days from the delivery date |
| Negative pledge, disposals, ownership change | None — consent or waiver only | — |
| Positive undertakings (insurance, tax, registration) | Remedy | 20 business days |
| Borrowing base and clean-down | Remedy by repayment | 10 business days |

**Equity cure rules**, where granted:

- The shareholder injects new equity or formally subordinated shareholder debt,
  and the covenant is recomputed on a pro-forma basis **for the test period in
  which the breach occurred**.
- The definition states whether the injection is applied to the cashflow measure,
  to EBITDA, or to reduce borrowings. The Bank's standard applies it to reduce
  borrowings; older instances apply it to cashflow. **It is never applied to
  EBITDA**, because that masks a profitability problem rather than solving a
  liquidity one, and because an EBITDA cure inflates the *next* three rolling
  tests as well.
- **No double-dip**: an injection applied to reduce borrowings may not also be
  counted as cashflow, and an injection counted in one test period may not be
  counted in another.
- Limits: **no more than two cures in any four consecutive test periods**, never
  in consecutive periods, and no more than four over the life of the facility. A
  cure request beyond these limits is not a cure; it is an amendment (L4) or a
  restructure (L5).
- A cured breach is recorded as a **breach that was cured**, not as a pass.
  The distinction survives into the review pack, the watchlist signal set and the
  staging assessment, and a design that writes "pass" after a cure has destroyed
  the fact that matters.

#### 5.5.5 Waivers

A waiver is the Bank agreeing not to enforce a breach. It is a decision, with an
authority, a scope and an end.

**Six requirements, each load-bearing:**

1. **A waiver names what it waives.** A specific `covenant_instance_id`, at a
   specific `covenant_definition_version`, for a specific test date or a specific
   stated period. "Covenants waived" is not a waiver.
2. **A waiver has an expiry, and the expiry is mandatory.** Exactly as an
   adjustment does ([00](00-shared-credit-core-library.md) §6.22, property 5).
   The failure mode is identical and just as common: a waiver granted for one bad
   quarter in 2027, still silently in force in 2031, with nobody able to say what
   testing the covenant again would show.
3. **A waiver is recorded against the covenant, not against the facility.** A
   facility with four covenants and one waived covenant is not "waived".
4. **The authority is a function of the breach class, the grade, the exposure and
   the facility type** — a 432-cell matrix (§6) — and is never below level 3. No
   waiver is automatic, at any amount, at any grade. This is the one place in the
   flow where the automated mandate is explicitly unavailable.
5. **A waiver may carry conditions**, and the conditions are themselves tracked:
   a tightened level going forward, a fee, monthly rather than quarterly
   reporting, a dividend standstill, a partial repayment. A conditional waiver
   whose conditions are not met is not a waiver.
6. **At expiry the covenant tests again, automatically.** The system does not
   wait to be asked.

**Repeated waivers are a signal in their own right**, and are treated as one:

| Pattern | Consequence |
|---|---|
| Two consecutive waivers on one covenant | The covenant may not be waived a third time. It is reset through L4 or the facility goes to L5. Watchlist W2. |
| Three waivers on any covenants of one facility in 24 months | Mandatory L5 forbearance assessment, because a pattern of concessions granted to a business in difficulty **is** forbearance regardless of what each was called. Authority +1 level. |
| Any waiver where the same covenant was cured in the preceding four test periods | Committee referral; the structure is presumed unsustainable until shown otherwise |

The last three rows are the mechanism that stops a slow restructure from
happening one waiver at a time, unreported, which is the failure mode the
requirement exists to prevent.

#### 5.5.6 Cross-default

A material or severe breach on one facility puts the client's other facilities
into a **cross-default state**. Within a group, it reaches other members where
cross-guarantees exist or the documentation says so.

| Element | Rule |
|---|---|
| Scope | All facilities of the same client; group members where a cross-guarantee or composite security exists |
| De minimis | No cross-default where the defaulted amount is below the greater of **R250 000** and **1% of group exposure** |
| Effect | The state is **recorded** on every affected facility; undrawn amounts are stopped; no new lending; no limit increase |
| Calling it | Acceleration is an authority act at level 6, and is almost never exercised |
| The requirement | **An uncalled cross-default is a state, not a non-event.** Provisioning needs it, the next review needs it, and a facility in an uncalled cross-default is not a facility in good standing. A design that only records called cross-defaults records almost nothing. |

Cross-default is the clearest case in the document of a decision that cannot be
made facility by facility. The breach happens on one facility; the state lands on
all of them; the authority to act sits above all of them; and the three are
assessed by three different teams.

#### 5.5.7 What a test records

Every one of the 2.4 M annual tests records: the instance, the definition
version, all three dates, the inputs by standard line with their source period,
the computed value, the threshold in force on the step schedule, the headroom in
both ratio and percentage terms, the result, the breach class, the cure
availability and clock, any waiver applied with its identifier and expiry, the
cross-default consequences, and the signals emitted to L3.

Seven years' retention on each. That is roughly **16.4 M test records over the
five-year window**, and their storage profile is §8's problem.

### 5.6 L3 — Early warning and watchlist

**Trigger**: daily, over the full book. **240 000 facilities**, ≈ 180 000
clients, ≈ 1 116 000 entity attachments, in a 3-hour window completing by 05:00.

**Determines**: a watchlist grade per facility, the signals that produced it, and
the actions the grade mandates.

#### 5.6.1 The signals

186 signals across seven families (§4.6). Each carries: an identifier, its
family, its source, its computation window, its trigger level, its weight, its
decay profile, whether it is a **trigger** (mandatory escalation regardless of
the aggregate) or a **contributor**, and its owner.

Representative signals, to fix the shape:

| Family | Signal | Level |
|---|---|---|
| Business bureau | Commercial grade deterioration ≥ 2 notches in 90 days | Contributor, weight 8 |
| Business bureau | New judgment against the business, any amount | Trigger → W2 |
| Entity bureau | Any critical entity's personal grade deteriorates ≥ 3 notches | Contributor, weight 6 |
| Entity bureau | Any critical entity enters debt review | Trigger → W3 |
| Internal conduct | Excess days ≥ 10 in a calendar month | Contributor, weight 5 |
| Internal conduct | Returned debits ≥ 3 in 30 days | Contributor, weight 7 |
| Internal conduct | Credit turnover down ≥ 25% against the trailing 12-month mean | Contributor, weight 9 |
| Internal conduct | Clean-down not achieved in the covenant year | Trigger → W1 |
| Covenant | Headroom below 10% at two consecutive tests | Contributor, weight 7 |
| Covenant | Material breach | Trigger → W1 |
| Covenant | Severe breach | Trigger → W3 |
| Covenant | Waiver requested | Contributor, weight 8 |
| Covenant | Certificate more than 30 days late | Contributor, weight 4 |
| Other providers | Any facility with another provider ≥ 30 days past due | Contributor, weight 8 |
| Other providers | Enquiry velocity ≥ 4 commercial enquiries in 90 days | Contributor, weight 5 |
| Sector | Sector default rate up ≥ 40% year on year | Contributor, weight 4, applied book-wide by sector |
| Structural | Auditor resigned, or audit opinion moved to qualified | Contributor, weight 9 |
| Structural | Any critical entity's screening status moves to probable or confirmed | Trigger → W3 |
| Structural | Adverse media on any entity, above a confidence threshold | Contributor, weight 6 |
| Structural | Business rescue or liquidation application filed | Trigger → W5 |

**Decay.** A signal is not permanent. Each has a decay profile over six age bands
(0–30, 30–60, 60–90, 90–180, 180–365, 365+ days) with a floor. A returned debit
from fourteen months ago carries a fraction of its weight; a confirmed fraud
marker carries all of it, forever. The 1 116-cell decay table is the mechanism
by which a facility comes *off* the watchlist without anybody doing anything —
and the requirement that a grade improvement be attributable to decay rather than
to a new fact is why §5.6.3 exists.

#### 5.6.2 The watchlist grades and their mandated actions

| Grade | Name | Entry | Mandated actions | Who |
|---|---|---|---|---|
| **W0** | Standard | Aggregate score < 15, no triggers | None | — |
| **W1** | Monitor | Score 15–29, or a W1 trigger | Conduct reviewed monthly; RM notified; next review date brought forward to 6 months | RM |
| **W2** | Watch | Score 30–49, or a W2 trigger | RM client contact within **10 business days**; **site visit within 30 days**; quarterly full review; monthly management accounts required; **no limit increase**; valuations refreshed | RM + Credit |
| **W3** | Special mention | Score 50–74, or a W3 trigger | **Undrawn amounts frozen**; no new facility; monthly reporting; security position reviewed and re-registered where needed; credit committee note; **SICR assessment mandatory**; joint management with Credit Risk | Credit Risk |
| **W4** | Substandard | Score ≥ 75, or a W4 trigger | **Exit plan within 30 days** with milestones and an owner; independent viability assessment; provisioning referral; forbearance assessment; no concession without level 6 | Credit Risk + Provisioning |
| **W5** | Workout | A W5 trigger, or W4 unresolved for 90 days | Transfer to Business Recoveries; L6 | Recoveries |

Three rules on the grades themselves:

1. **Escalation is automatic; de-escalation above W2 is not.** A facility moves
   up on the daily pass. Moving down from W3, W4 or W5 requires a recorded human
   decision with a rationale and an authority, because the decay table will
   otherwise quietly rehabilitate a business that has not recovered.
2. **A mandated action is tracked to completion.** An overdue site visit is
   itself a signal, and an action list with a 40% completion rate is a watchlist
   that exists on paper.
3. **The grade is per facility, the signals are per subject.** A signal on an
   entity applies to every facility of every business that entity touches; a
   signal on the client applies to all of that client's facilities; a signal on a
   facility applies to one. Three scopes, one grade, and the aggregation across
   scopes is where the fan-out lives.

#### 5.6.3 The hard part — 240 000 facilities, daily, with fan-out

The daily pass is the largest single workload in the estate and has a shape no
other project in this set produces.

**The fan-out.** A signal on one entity propagates to every facility of every
business that entity touches. Concretely: a director sitting on 6 businesses
averaging 1.3 facilities each is 8 facilities from one judgment. The distribution
is heavily skewed — the p99 entity touches **31 facilities**, and 340 entities in
the book touch more than 100.

| Quantity | Value |
|---|---|
| Facilities graded daily | 240 000 |
| Entity attachments traversed | ≈ 1 116 000 |
| Signals evaluated per facility | 186, of which ≈ 22 have any data |
| New events arriving daily | ≈ 9 400 |
| Entities with a changed signal set daily | ≈ 48 000 |
| Facilities whose grade could change daily | ≈ 31 000 |
| Facilities whose grade actually changes daily | ≈ 2 400 |
| Window | 3 hours, complete by 05:00 |

**The requirement, in three parts:**

1. **The pass is incremental.** Only subjects with a changed signal set, or with
   a signal crossing a decay band boundary, are re-evaluated. Evaluating 240 000
   facilities × 186 signals from scratch nightly is roughly 45 M signal
   evaluations and does not fit the window at the fan-out involved.
2. **The incremental pass is provably equivalent to a full one.** A full pass
   runs weekly as a reconciliation, and any facility whose grade differs between
   the incremental and full results is a defect, reported and investigated. This
   is [project 05](05-business-credit-nested-entities.md) §8's partial
   re-assessment requirement at book scale, and the proof obligation is the same.
3. **The cause of every grade change is recorded**, and distinguishes the two
   cases that look identical: a grade moved because **a new signal fired**, or a
   grade moved because **an old signal decayed out**. The first is news; the
   second is arithmetic; and a watchlist committee told they are the same thing
   will stop reading the report.

**The inherited-component tension.** The daily pass calls
`core.adverse_events`, project 05 §5.5's classification and project 05 §5.6's
roll-up — components written for a 900-a-day origination path — roughly 9 400
times a day as singletons, plus 48 000 entity re-rolls. §5.17 develops what that
costs and what it demands of the components.

### 5.7 L4 — Amendment

**Trigger**: a client or RM request, or a consequence of L2, L3 or §5.12.
**600/day.**

**Determines**: whether the change is acceptable, on what terms, approved by
whom, and what else it re-opens.

An amendment re-opens **part** of the origination assessment and not all of it.
Which part is the whole of the requirement:

> Which phases an amendment re-opens must be **declared data**, reviewed and
> approved as policy, and not left to the judgement of whoever is handling it.

That artefact is the **amendment re-open matrix**: 14 amendment kinds × 28
phase-parts = 392 cells, owned by Credit Governance, approved by Credit
Committee, versioned and effective-dated like any other table.

| # | Amendment | Re-opens | Notes |
|---|---|---|---|
| AM-01 | Limit increase ≤ 20% and ≤ R500 000, grade ≤ 6, no watchlist | O10, O11, O12, O13, O14, O17 | The cheap path. Does **not** re-open O5 or O7 |
| AM-02 | Limit increase beyond AM-01's bounds | O1–O17 in full | It is a new credit decision wearing an amendment's clothes |
| AM-03 | Term extension | O11, O13, O14, O15 (the step schedule shifts), O17 | Asset-life cap re-tested for product 52 |
| AM-04 | Security substitution | O12 **for every facility sharing either item**, O11, O13 (security type may move, which moves the rate), O17 | §5.11 |
| AM-05 | Security release | As AM-04, plus a mandatory re-test of every facility relying on the released item, plus a cover-ratio floor condition | The authority is that of the **resulting** position, not of the release |
| AM-06 | Covenant reset | O15, O7 (the new level's base case), L2 (a new definition version from the reset date), O17 | Prior tests stay bound to the prior version |
| AM-07 | Covenant removal | O15, O17, mandatory escalation +1 level | Removing a covenant is an authority act, never an administrative one |
| AM-08 | Repricing | O13, O17 | Including a repricing arising from L1 |
| AM-09 | Drawdown or repayment schedule change | O14, O15, O17 | A change to debt service changes every DSCR covenant's denominator |
| AM-10 | Surety or guarantor substitution | O5 (the new entity), O6, O11, O12, O17 | The outgoing surety's release is a security release — AM-05 applies |
| AM-11 | Ownership change consent | O2, O3, O5, O6, O9, O10, **and a cascade** (§5.12) | The single most expensive amendment in the matrix |
| AM-12 | Facility type conversion (e.g. 57 → 50) | O11, O12, O13, O14, O15, O16, O17 | A new covenant set, a new rate table, a new authority cell |
| AM-13 | Borrower added to or removed from a group facility | O2, O10, O12, O16, O17 | Composite security and cross-guarantees must be re-papered |
| AM-14 | Trade instrument tenor or currency change | O11, O13, O17 | Products 55 and 56 only |

Four requirements on the matrix itself:

1. **An amendment that re-opens nothing is not an amendment**; it is a record
   change, handled elsewhere, with no decision of record.
2. **An amendment inherits the facility's covenant definitions** unless it
   explicitly replaces them. A replacement creates a new
   `covenant_definition_version` dated at the amendment, leaving every prior test
   bound to the prior version. §5.5.1.
3. **The matrix is a governance artefact, not a routing table.** A change to it
   changes what the Bank re-assesses, which is a policy decision, so it carries
   an approval, an effective date and a diff exactly as the appetite grid does.
4. **The re-opened phases run on today's artefacts, against a record produced on
   older ones.** The amendment's decision of record therefore carries a
   `comparison_basis_code` against the origination, like any other re-decision
   (§5.10).

### 5.8 L5 — Restructure and forbearance

**Trigger**: arrears, declared hardship, a severe covenant breach, watchlist W4,
or a third waiver. **120/day.**

**Determines**: what the Bank can offer a client who cannot pay on current terms,
how that is classified, and what it costs.

The machinery is
[project 06](06-consolidation-and-restructure.md)'s and is consumed rather than
restated: the option search, the bounded scenario generation, the NPV cost of
concession, the authority ladder driven by that cost, the stressed affordability
test, the before-and-after comparison, and the rule that a second concession
within 12 months raises the required authority by one level.

**What is different in a business setting** — six things:

#### 5.8.1 The subject is a facility set, not a debt list

Project 06 searches over which of a client's retail accounts to settle. Here the
subject is a business with up to nine facility types, some drawn, some undrawn,
some contingent, some self-secured, and a restructure is a **package across all
of them**. The concession catalogue grows from project 06's eight to **21**,
because a payment holiday means something different on a term facility, an
overdraft, an invoice finance line and a guarantee facility, and because three
concessions exist here that have no retail counterpart: conversion of a hardcore
overdraft to an amortising term facility, reduction of a borrowing base advance
rate in exchange for a longer runway, and release of a contingent facility in
exchange for cash cover.

#### 5.8.2 Security reallocation

A restructure almost always moves security: releasing an asset to be sold,
substituting cover, or taking additional security as a condition of the
concession. Every such move re-opens the cross-facility position (§5.11) — for
facilities that are **not part of the restructure**, owned by nobody in the
workout conversation. The requirement is that they be re-tested and their
position reported, and that the authority for the restructure be that of the
whole affected set.

#### 5.8.3 Covenant resetting

A restructure resets covenants. Three things must stay distinguishable and are
routinely confused:

| | What it is | Evidence obligation |
|---|---|---|
| **A waiver** | Not enforcing a breach of an existing covenant, for a period, with an expiry | §5.5.5 |
| **A reset** | Replacing the covenant with a new definition version from a date | AM-06; prior tests keep the prior version |
| **A concession** | A reset granted *because the client is in financial difficulty* and would not have been granted otherwise | §5.8.5 — this is forbearance |

The third is a subset of the second, and the test that separates them is the
client's condition, not the mechanics of the change. A covenant reset on a
healthy client repricing after a good year is not forbearance. The same reset on
a client that would have breached is.

#### 5.8.4 Group implications

A concession to business A may trigger cross-default across the group (§5.5.6),
may push a connected business over a concentration limit, and may itself be part
of a group restructure across four businesses that is **one decision with four
clients**. The requirement: the flow must be able to assess a restructure at
group level, producing per-client outcomes from one search, with one authority
determined by the aggregate and one set of records that each client can be shown
without seeing the others.

#### 5.8.5 Provisioning and credit-impairment staging

This is the consequence that is most often discovered afterwards, by Finance,
from a payment pattern — and the requirement is that it be **produced by the flow
at the point of decision**.

**The forbearance test has two limbs, and both must be evidenced:**

1. **Financial difficulty** — the client is experiencing, or is about to
   experience, difficulty meeting its financial commitments. Evidenced by:
   arrears, a covenant breach, watchlist grade, a grade deterioration, a declared
   hardship, or the business's own forecast.
2. **A concession** — a modification of terms the client could not comply with,
   which the Bank would not have granted otherwise; or a refinancing of a
   troubled facility.

Both limbs, or it is not forbearance. And — the part that gets missed — **a
restructure that is not forbearance must carry the evidence for the negative
classification.** "We assessed it and it was not forbearance" is exactly what a
regulator tests, and an empty record is indistinguishable from a classification
that was never made.

**Staging consequences**, produced as an output of this phase and handed to
Provisioning rather than derived by them:

| Situation | Consequence |
|---|---|
| Concession granted, performing, no loss | Forborne flag set; **Stage 2** at minimum, for the whole probation |
| Concession granted where the client is already non-performing | Non-performing forborne; **Stage 3** |
| Concession resulting in a diminished financial obligation — a net present value loss above **1%** | Default event; **Stage 3**; reported as such |
| A second concession granted during the probation period | Re-triggers non-performing classification and restarts the clocks |
| Any exposure to the client more than 30 days past due during the performing-forborne probation | Same |

**The probation clocks**, which are lifecycle state of exactly the kind
§5.10 governs:

| Transition | Minimum period | Conditions |
|---|---|---|
| Non-performing forborne → performing forborne | **1 year** from the later of the concession date and the non-performing classification | No past-due amounts; evidence supporting full repayment |
| Performing forborne → no longer forborne | **2 years** from the date it became performing forborne | Regular payments of a more than insignificant amount over at least half the probation; no exposure to the client more than 30 days past due at the end |

Four flags exist per facility and they interact without implying each other, and
the permitted transitions between them are a declared artefact:
**`watchlist_grade`**, **`forbearance_flag`** with its probation clock, the
**non-performing / default** state with its 90-day counter, and **`staging_code`**.
A facility can be W2, not forborne, performing and Stage 2. It can be W0,
forborne under probation, performing and Stage 2. Collapsing any two of them into
one field is a defect that will be found at a reporting date, by Finance, under
time pressure.

#### 5.8.6 A concession must be reported as a concession

The single hardest governance requirement in this phase, and the one §9.5 tests:
the classification is produced at decision time, by the flow, with its evidence;
it is not derivable later; it persists through the probation clocks; it is
visible to the next review, the next amendment and the next watchlist pass; and
it raises the authority for any subsequent concession.

### 5.9 L6 — Exit and handoff

**Trigger**: an exit decision, a workout transfer, or the client leaving.

**Determines**: where the facility goes and what goes with it.

Three exits:

| Exit | Trigger | Destination |
|---|---|---|
| **Recovery** | W5, or an uncured severe breach, or default | Business Recoveries, with [project 08](08-collections-treatment.md)'s treatment machinery taking over the operational side |
| **Provisioning and impairment** | Stage 3 with no viable workout | Finance, with the facility remaining on the book |
| **Voluntary closure** | Client repays, refinances elsewhere, or the Bank declines to renew | Documentation, with a security release under AM-05's rules |

**The handoff contract** — what Recoveries receives, and it is not a summary:

1. The full decision history: origination and every review, amendment,
   restructure and covenant-driven decision, each with its own evidence.
2. The security position **with its allocation** as at the exit date, every
   valuation with its date and basis, every prior encumbrance, and the deeds and
   registrations with their status.
3. The covenant and waiver history: every test, every breach, every cure, every
   waiver with its conditions and expiry, and the cross-default state.
4. The entity structure as at the exit date, **and** the structure as at
   origination, with the delta — because the sureties Recoveries will pursue were
   given by people whose roles may have changed.
5. The group position: members, exposures, cross-guarantees and the composite
   security package.
6. The forbearance history with its probation clocks, and the staging history
   with its triggers.

**Two requirements that outlast the handoff:**

- **The history does not stop.** A workout decision in 2032 must be able to reach
  the 2026 origination record and read it, including the committee pack that
  approved it. Handoff is a change of owner, not an archive.
- **Return is possible and is not amnesia.** A facility that comes back from W4
  to W1 carries its history with it. A "clean" facility with a two-year-old
  forbearance and its probation still running is not a clean facility, and the
  next review, the next amendment and the next appetite check must all know that.

### 5.10 Statefulness across time

The first of seven cross-cutting requirements. It is the spine of the document
and everything in §5.4 to §5.9 depends on it.

> **Every re-decision must be comparable with every prior one for the same
> facility.**

#### 5.10.1 The decision history

A facility owns an ordered, append-only sequence of **decisions of record**:

| | |
|---|---|
| Identity | `decision_of_record_id`, globally unique, never reused |
| Subject | `facility_id`, and where relevant `client_id` and the group composition with its date |
| Kind | `assessment_kind_code` — which of the nine entry points produced it |
| Dates | `decision_date`, `knowledge_date`, the run date, and the effective date of the outcome |
| Predecessor | The `decision_of_record_id` it supersedes, and the `comparison_basis_code` against it |
| Outcome | The decision, the terms, the covenant set, the conditions, the authority that approved it |
| Provenance | Everything §5.10.2 requires |

Mean **11 decisions of record per facility over five years**: one origination,
five reviews, two amendments, and three covenant-driven decisions. The p99 is
**58**.

#### 5.10.2 What must be pinned

Each decision of record names, and is not valid without:

| Pinned | Why |
|---|---|
| `decision_date` and `knowledge_date` | The two time axes. §5.13 |
| Entity structure as at both | §5.13 |
| Group composition as at the decision date, with its member list | §4.1 |
| Financial periods used, with their accounting bases | O7 |
| Every table version and every cell read | [00](00-shared-credit-core-library.md) §8 |
| Scorecard and calibration versions | O5, O9 |
| `master_scale_version` | §5.10.4 |
| Sector table and benchmark versions | O7, O11 |
| Appetite grid and authority matrix versions | O11, O17 |
| The adjustment set, with every overlay applied, its effect and the order | [00](00-shared-credit-core-library.md) §6.22 |
| Covenant definition versions for every instance touched | §5.5.1 |
| Collateral valuations with their dates, bases and allocation | §5.11 |
| The flow build identifier | [09](09-governance-and-replay-harness.md) §5.15 item 21 |
| The four state flags as at the decision | §5.8.5 |

**What may change**: all of it, between decisions. That is the point. The
requirement is not stability; it is that instability be enumerated.

#### 5.10.3 The cause decomposition

Between any two consecutive decisions of record for one facility, the difference
in every material output must be apportioned across the six causes of §5.4.1, and
the apportionment must sum to the observed difference. An unapportionable
residual is a defect and is reported as one — which is the same discipline
[project 09](09-governance-and-replay-harness.md) §5.5 applies to a swap-set
across a release, applied here to one subject across a year.

#### 5.10.4 Grade migration when the scheme itself changed

This is the requirement the document is named for.

**The problem, concretely.** Marang Engineering is graded **6** at origination in
March 2026, on master scale **v2**, by scorecard `BUS-COMM-01`. Between then and
its 2031 review:

| When | What happened |
|---|---|
| Nov 2027 | `BUS-COMM-01` recalibrated. Same characteristics, new coefficients. Master scale unchanged. |
| Jun 2028 | Master scale **v3**: the 6/7 boundary moved by 0.35 percentage points of PD, to correct observed default rates in grades 5–7. |
| Feb 2030 | `BUS-COMM-02` replaces `BUS-COMM-01`: 34 characteristics instead of 29, four of them new, two of the old ones no longer collected. |
| Sep 2030 | Master scale **v4**: a 15-grade scale trialled in 2029 collapsed back to 12, with different boundaries again. |

The 2031 review grades it **7**.

A migration report showing **6 → 7** says the business deteriorated one notch.
Restating the 2026 assessment onto scale v4 — the 2026 PD of 2.1% falls in v4's
grade 7 — says it did not move at all. **Both numbers are defensible and they
contradict each other.**

**The requirement**:

1. **A grade is never comparable without its scale.** `master_scale_version` is a
   required output of O9 and a required field of every decision of record.
2. **A migration report declares its basis.** `comparison_basis_code` is
   as-graded, restated, or not comparable, and the report carries it visibly. A
   report mixing bases across rows is **invalid** and is rejected, not
   footnoted.
3. **Restatement is a published artefact, not a calculation somebody does.** A
   restatement map between two master scale versions is owned by Credit Risk
   Modelling, approved, versioned, dated, and retained for the life of any
   decision it can touch. There are 6 pairwise maps across 4 scale versions, each
   12 × 12.
4. **The original grade is never overwritten.** A restated grade is carried
   *beside* the grade as originally assigned, with the map that produced it. The
   2026 record says grade 6 on v2 and, where a map exists, grade 7 on v4 by map
   v2→v4. It never says grade 7.
5. **Not every change can be restated, and the honest answer must be
   available.** The 2030 scorecard replacement is not invertible: two
   characteristics are no longer collected, so the 2026 assessment cannot be
   re-scored under `BUS-COMM-02`. Where a restatement would require inputs that
   do not exist, the basis is **not comparable**, and the system must produce
   that word rather than a number. A design that always produces a number will
   always produce a wrong one.
6. **A recalibration must not be able to look like a portfolio improvement.**
   Portfolio-level migration statistics crossing a recalibration date are either
   computed on a restated basis or excluded, and which was done is stated on the
   report.

#### 5.10.5 What this makes possible, and what it costs

With the above, the questions in §9.1 are answerable. Without it, the Bank has
five years of grades that cannot be put in one column, and a migration matrix
that measures its own model changes. The cost is that every decision of record
carries roughly **180 provenance fields** it would not otherwise need, and that
a restatement map must be produced on every recalibration — which is work the
Model Team does not do today and which is change scenario 15 in §11.

### 5.11 Security allocation across facilities

The second cross-cutting requirement, and the one that cannot be decided
record by record.

#### 5.11.1 The problem

One property, at a 70% advance rate, secures a term facility and a revolving
facility simultaneously — because the bond is expressed to secure all present and
future liabilities and is continuing security. It is not discharged when one
facility is repaid and it is not allocated by the client. The Bank allocates it,
for its own purposes: to compute a cover ratio, to derive a `security_type`, to
key the rate card, to set the authority level and to drive the uncovered portion
that provisioning uses.

Growing one facility re-opens the security position of the other. The facilities
are assessed at different times, by different people, under different entry
points.

#### 5.11.2 A worked illustration

**Marang Engineering (Pty) Ltd**, three facilities and two items in the shared
pool.

| Facility | Type | Exposure | Notes |
|---|---|---|---|
| F1 | 50 — Business Term Facility | R4 000 000 outstanding | Amortising, 48 months remaining |
| F2 | 57 — Business Overdraft | R2 500 000 limit, R2 100 000 drawn | Annual review |
| F3 | 52 — Commercial Asset Finance | R3 200 000 | Self-secured on the financed CNC line; stands outside the shared pool except for its deficiency |

| Collateral | Class | Value | Advance rate | Adjusted |
|---|---|---|---|---|
| C1 | Industrial property, first covering bond | R6 800 000, valued 2027-04-11 | 70% | **R4 760 000** |
| C2 | Debtors book under cession, < 90 days | R4 100 000 of R5 200 000 gross | 40% | **R1 640 000** |
| | | | **Total** | **R6 400 000** |

Shared-pool exposure: F1 + F2 = **R6 500 000**. Two allocations, both legitimate:

| | F1 cover | F1 ratio | F1 `security_type` | F2 cover | F2 ratio | F2 `security_type` |
|---|---|---|---|---|---|---|
| **Allocation A** — pro rata by exposure | R3 938 000 | 0.98 | 2 partially secured | R2 462 000 | 0.98 | 2 partially secured |
| **Allocation B** — specific: C1 → F1, C2 → F2 | R4 760 000 | 1.19 | **1 fully secured** | R1 640 000 | 0.66 | 2 partially secured |

`security_type` keys the rate card. Under allocation B, F1 prices 140 basis
points lower — **R56 000 a year** — and F2 is unchanged because 0.66 and 0.98 sit
in the same band. The allocation is therefore worth R56 000 a year, and in the
absence of a declared policy it is decided by whoever happened to run the
assessment first.

**Now amend F2.** The client asks to raise the overdraft from R2 500 000 to
R4 000 000. Shared-pool exposure becomes R8 000 000 against R6 400 000 of cover.

| Option | F2 outcome | F1 consequence |
|---|---|---|
| Keep allocation B, give F2 only C2 | Ratio 0.41 → `security_type` **4 unsecured** → rate +310 bps, and the amendment probably fails appetite | None |
| Re-allocate: C1 split R3 200 000 to F1, R1 560 000 to F2 | F2 ratio 0.80 → type 2 | F1 ratio falls to 0.80 → type 2 → **F1 reprices upward by 140 bps** |
| Take additional security | Both hold | A new condition precedent, a valuation, and a registration |

The second option is the one that matters: **an amendment to F2 reprices F1** — a
facility nobody asked about, owned by a different relationship, whose client
conversation has not happened, and whose own last decision of record said
`security_type` 1.

#### 5.11.3 The requirements

1. **Allocation is an explicit, recorded decision.** An `allocation_id` per
   (collateral item, facility) pair, as at a date, with the policy that produced
   it. It is never an emergent property of the order in which facilities were
   evaluated.
2. **The allocation policy is declared.** This project's default is: specific
   allocation where a facility is contractually linked to an asset; then priority
   by ranking; then allocation maximising total adjusted cover across the set,
   tie-broken by the longest-dated facility. Alternatives must be evaluable and
   their consequences shown, because a client asking "why is my term loan priced
   as partially secured when you hold a bond worth more than it" is asking about
   an allocation policy.
3. **The over-allocation invariant.** The sum of a collateral item's allocations
   across all facilities may never exceed its adjusted value at that date.
   Testable, and tested — across the whole book, nightly.
4. **Any assessment that changes a facility's claim on shared collateral must
   re-test every other facility sharing it**, and must state whether their
   `security_type` changed and what that does to their price.
5. **The authority is that of the whole affected set.** An amendment to F2 that
   worsens F1 is not approvable at F2's authority level alone. §5.14.
6. **Assessed at different times by different people** — the invariant that makes
   this possible: every facility's decision of record pins the allocation it
   relied on, with its date and the valuation that underpinned it. Reconstructing
   the pool's position as at any past date is then a query, not an archaeology
   exercise.
7. **A revaluation is an event, not a field update.** C1 revalued on 2029-09-02
   at R5 900 000 drops adjusted cover from R4 760 000 to R4 130 000. Every
   facility relying on C1 changes its cover ratio on that date, whether or not
   anyone was assessing it. The revaluation must produce a defined consequence
   set: re-allocation, re-derived security types, an LTV covenant test where one
   exists (product 53), a signal to L3, a flag to the next review, and an
   amendment where the resulting position breaches a cover condition. Valuations
   refresh on a cadence by class — property 24 months, plant 12 months, debtors
   monthly, listed securities daily — so this fires continually.
8. **Release requires the authority of the resulting position.** Releasing C2 is
   approved at the level required by the post-release cover across all facilities
   in the pool, not by the size of C2.

### 5.12 The group cascade

The third cross-cutting requirement. A decision about business A changes the
exposure headroom of business B because they share a director; an entity's
adverse event changes the people component of every business it is attached to.

#### 5.12.1 What triggers a cascade

Nine triggers, each producing a `cascade_id`:

| Trigger | Typical daily volume |
|---|---|
| A new adverse event on any entity, classified material or worse | ≈ 1 100 |
| An entity's grade moves by ≥ 2 notches | ≈ 640 |
| An ownership change: transfer, allotment, or a restatement of beneficial ownership | ≈ 370 |
| A director, member or trustee appointment or resignation | ≈ 760 |
| A new facility, limit change or amendment anywhere in a group | ≈ 320 |
| A covenant breach classified material or severe at any group member | ≈ 190 |
| A watchlist escalation to W3 or worse at any group member | ≈ 140 |
| A business rescue, liquidation or deregistration filing on any entity | ≈ 40 |
| A screening status change on any entity | ≈ 90 |
| | **≈ 1 900/day**, touching **≈ 14 000 facilities** |

#### 5.12.2 How far it propagates, and how it is bounded

**Propagation**, in hops from the trigger:

```
trigger (entity or facility)
  → hop 1: every business the entity attaches to        (mean 1.4, p99 11)
  → hop 2: the group of each of those businesses        (mean 2.3 businesses)
  → hop 3: the facilities of every business so reached  (mean 1.3 each)
```

**Four bounds**, and all four are required. Without them a director's tax notice
re-assesses the book.

| Bound | Rule |
|---|---|
| **Depth** | Propagation stops at hop 2 — entity → business → that business's group. It does **not** continue from the group's members to *their* groups. Transitive closure over a connected-client graph reaches 40% of the book from most starting points. |
| **Materiality** | The trigger must be capable of changing something. An event classified immaterial on a peripheral entity propagates as a **flag** on the affected records, not as a re-assessment. Roughly 71% of candidate triggers stop here. |
| **Fan-out cap** | A cascade that would touch more than **250 facilities** is not run inline. It is queued as a batch, escalated to Portfolio Management, and run in the next scheduled window with a named owner. The 340 entities touching more than 100 facilities are exactly the cases where inline propagation would be both slowest and most consequential. |
| **Re-entrancy** | An entity, business or facility already processed within a `cascade_id` is not processed again, and two cascades triggered by the same underlying fact are de-duplicated. Without this, cross-holdings produce loops — project 05 §5.1 already truncates cycles inside one structure; this is the same problem between structures. |

#### 5.12.3 What is re-assessed, and what is only flagged

The distinction is the whole design, and getting it wrong in either direction is
a failure: too much re-assessment and the book churns; too little and the Bank's
exposure numbers are stale.

| | Re-assessed | Flagged only | Never |
|---|---|---|---|
| The triggering entity | Verdict, score, grade, criticality | | |
| Businesses where that entity is critical or significant | People component (O6), combined grade (O9) | | |
| Businesses where that entity is peripheral | | Flagged for the next review | |
| Group exposure aggregates | Recomputed (O10) | | |
| Appetite headroom of every business in those groups | Recomputed (O11) | | |
| Undrawn amounts and new lending | Blocked immediately where headroom is exceeded | | |
| Watchlist grade | Recomputed (L3) | | |
| The price of a **contracted** facility | | Flagged for the next review or an amendment | Changed automatically |
| A **drawn, committed** facility's limit | | Flagged; an amendment decision with its own authority and client notice is required | Reduced automatically |

The last row is the one that matters commercially: **a cascade may reduce
headroom and block new lending immediately; it may never reduce a committed
facility on its own.** Reducing a committed facility is an amendment (AM-01 or
AM-02 in reverse), with an authority, a notice period and a client conversation.

#### 5.12.4 Explaining it to the client of business B

Business B's facility headroom drops on 14 June because a judgment was registered
against a director of business A on 9 June, and A and B share that director.

This must be explainable — and the two explanations are different documents:

**The internal attribution** is complete: `cascade_id`, the trigger (event
identifier, entity, date, classification, the rule that classified it), business
A, the propagation path, every subject touched, what was recomputed for each,
what changed, and the decisions produced. Retained for seven years, available to
Audit and to the regulator.

**The communicable explanation** to business B may not disclose business A's
affairs or the director's personal credit record. It draws on the reason
registry's client-facing wording
([00](00-shared-credit-core-library.md) §6.18), owned by Compliance, and says
something in the shape of: *the Bank's aggregate exposure to a group of connected
businesses of which yours is one has reached its limit, and your available
headroom has been adjusted accordingly* — with the route by which the individual
concerned may obtain their own reasons directly, exactly as
[project 05](05-business-credit-nested-entities.md) §9.2 requires.

Which reason codes are communicable at the level of a *third* business is a new
attribute on the registry and a new question for Compliance: project 05's
registry distinguishes what may be said to a business about an individual, and
this project needs what may be said to business B about business A.

#### 5.12.5 The cascade is a first-class record

Not a log line. A cascade has an identity, a trigger, a scope, a set of
consequences and a bound that it either respected or hit. "Why did my headroom
drop on 14 June" must be answerable in 2031, and the answer is a `cascade_id`.

### 5.13 Entity structure changes over time

The fourth cross-cutting requirement, and the smallest one with the sharpest
edge.

Directors resign, shareholders change, entities are added, ownership percentages
move: **≈ 34 000 structural change events a month** across the book. Each is
dated, and every assessment must use the structure as it was at its own decision
date.

#### 5.13.1 Two dates per fact

| | |
|---|---|
| `effective_from` / `effective_to` | When the fact was true in the world |
| `known_from` | When the Bank learned it |

Median lag **47 days**; p95 **310 days**; 4% of facts arrive with an effective
date more than two years past, landing behind decisions already made.

#### 5.13.2 The two queries, and why both are needed

> A director resigned in **March 2028**. The Bank learned in **November 2028**.
> The 2028 annual review ran in **June 2028**.

| Question | Query | Answer |
|---|---|---|
| Reproduce the June 2028 review | `known_from ≤ 2028-06-30` | The director is still in the structure. The review's answer stands and replays correctly. |
| Who was a director in March 2028? | `effective_from ≤ 2028-03-31 < effective_to` | The director had resigned. |

Both are required, both are correct, and conflating them breaks something in each
direction: using the effective-date view for replay makes a correct 2028 decision
appear to have been made on a structure the Bank did not have, which fails
[project 09](09-governance-and-replay-harness.md)'s reproduction test; using the
knowledge-date view for a covenant makes an ownership-change covenant blind to a
change it exists to catch.

#### 5.13.3 The covenant case

> **A 2029 covenant test must not be evaluated against 2031's shareholder
> register.**

An ownership-change covenant tests the register **as at the test date** against a
**baseline pinned at origination**. Three requirements:

1. The baseline is stored **in the covenant instance**, not recomputed from
   history at test time — because recomputing it depends on facts learned since,
   which moves a baseline that is a contractual term.
2. The test-date register is the **effective-date** view as at the test date, on
   the knowledge available at the determination date. The three dates of §5.5.2
   apply here too.
3. A late-arriving fact that retrospectively breaches a covenant already tested
   and passed produces a **re-test**, not a silent correction. The original test
   stands as the test that was performed; the re-test is a new record with a new
   determination date, and the difference between them is the Bank's knowledge,
   which is itself a finding.

#### 5.13.4 The requirement on every phase

Every phase that reads structure declares which view it uses. A phase that does
not declare is a defect found by a check. This is the same discipline as "no
reliance on today" ([09](09-governance-and-replay-harness.md) §5.15 item 4), one
level harder, because both answers are plausible and neither errors.

### 5.14 Authority and delegation

The fifth cross-cutting requirement.

#### 5.14.1 The matrix

Four dimensions → one of seven levels. **9 amount bands × 12 grades × 4 security
types × 9 facility types = 3 888 cells**, owned by Credit Governance, approved by
the Board Credit Committee, effective-dated, diffable.

The amount dimension is **total group exposure after the proposal**, not the
facility amount. A R400 000 overdraft to a client whose group already carries
R38 000 000 is a level 5 decision, and a matrix keyed on the facility amount
would route it to level 2.

| Level | Authority | Indicative ceiling (group exposure after) | Notes |
|---|---|---|---|
| **1** | **Automated mandate** — no human | R750 000 | Grades 1–6, security type 1–2, facility types 50, 51, 57 only, W0, no live forbearance, no policy exception |
| **2** | Relationship manager with a personal credit mandate | R2 000 000 | Never alone on a new-to-bank client |
| **3** | Business credit analyst / credit manager, independent of the business | R5 000 000 | The lowest level that may grant a waiver |
| **4** | Senior credit manager | R15 000 000 | Two signatures above R10 000 000 |
| **5** | Head of Business Credit | R40 000 000 | |
| **6** | Credit Committee | R100 000 000 | Quorum-based; meets twice weekly |
| **7** | Board Credit Committee | Above R100 000 000 | Plus every mandatory referral below, regardless of amount |

**Modifiers that raise the level irrespective of the cell** — each by one level
unless stated:

watchlist W2 (+1) or W3 and above (to level 6 minimum); any live forbearance
(+1); a second concession within the probation period (to level 6); a probable
screening match (to level 6); `structure_unresolved` (to level 6); a restricted
sector (+1); an approve recommendation at grade 10–12 (to level 6); a waiver on a
covenant already waived twice (to level 6); any policy exception (+1, +2 where
the exception is to an absolute rule); group exposure above the single-name limit
(to level 7); a degraded review at basis 4 or 5 (+1); a decision affecting a
facility other than the one being assessed (§5.11: the authority is that of the
whole affected set); and connected-party lending — to a director, employee or
significant shareholder of the Bank — to level 7 at any amount.

#### 5.14.2 The pack per level

Each level needs a different pack, and the pack is **generated**, never
assembled by hand ([05](05-business-credit-nested-entities.md) §9.3). A
**322-cell** matrix of 7 levels × 46 pack sections declares which sections
appear at which level.

| Level | Pack |
|---|---|
| 1 | None. The decision record **is** the pack. |
| 2 | 2 pages: client, group, facility asked for, grade, DSCR, security and cover, conduct summary, decision and conditions |
| 3 | 6 pages: + three-period financial spread with ratio trend, the nine sector benchmark bands, the entity structure table, covenant proposal with headroom to base case, conditions |
| 4 | 12 pages: + management and ownership assessment, business model and sector commentary, downside sensitivity, primary and secondary repayment source, collateral valuation basis and dates, policy exceptions with mitigants, pricing against the rate floor |
| 5 | + group exposure and concentration analysis, the cross-facility security position (§5.11), a stress case, the five-year facility history where one exists |
| 6 | + the full committee pack of [05](05-business-credit-nested-entities.md) §9.3 — every entity, every event classification, the blend weights, the overlay decomposition, the pricing candidate record — plus portfolio impact, the watchlist and covenant history, and an independent Credit Risk opinion stated separately from the business case |
| 7 | + board risk appetite context, the exception register, the precedent set, capital and impairment impact, and the exit strategy |

#### 5.14.3 The authority moves during the assessment

This is H7 and it is the part that is routinely missed.

The authority level is determined by the **proposed** structure, and the proposed
structure changes while O13 is structuring it. Three real sequences:

| | Sequence | Authority |
|---|---|---|
| **a** | Requested R5 200 000; DSCR binds; the largest admissible amount is R4 800 000 | Level 4 → **level 3** |
| **b** | Reducing the amount improves the cover ratio from 0.94 to 1.03; `security_type` moves 2 → 1 | Level 4 → **level 3**, and the rate falls, and the DSCR improves, and the admissible amount rises again |
| **c** | The analyst adds a surety to improve the price; cover rises; security type improves; the level drops to the analyst's own | Level 4 → **level 3** — **and the person who structured it can now approve it** |

**Five requirements:**

1. The authority is **recomputed on every change to the proposed structure**, and
   the sequence of levels the proposal passed through is recorded. Sequence (b)
   is also a second source of non-monotonicity in O13's search, on top of the two
   [project 05](05-business-credit-nested-entities.md) §5.12 already names.
2. **The authority in force at approval governs**, and it must be the authority
   for the structure actually approved — not for the structure at routing.
3. **A structure change after approval that would have required a higher
   authority invalidates the approval.** Explicitly, automatically, and with the
   invalidation recorded.
4. **Self-approval is detected and controlled.** Where the person who structured
   the proposal holds the authority that would approve it, that is flagged and a
   second signature is required above R1 000 000. Sequence (c) is the reason: it
   is not misconduct, it is the natural consequence of a matrix keyed on
   something the assessor controls.
5. **The pack regenerates when the level changes**, and the pack presented must
   correspond to the structure approved. A level 6 pack describing a structure
   that was subsequently reduced to a level 4 decision is a governance failure
   that looks like a formatting problem.

### 5.15 Multi-team ownership at scale

The sixth cross-cutting requirement. Fourteen teams (§3) own one artefact that
executes 2.4 M covenant tests a year, a daily pass over 240 000 facilities,
15 000 full reviews a month and 2 900 interactive assessments a day.

#### 5.15.1 The collisions

Not hypothetical. Each has a mechanism and each needs an answer.

| # | Collision | Mechanism |
|---|---|---|
| 1 | Treasury reissues the asset finance rate card on the 1st; Policy changes the DSCR thresholds on the 1st; both land in the same monthly review batch | The batch's results cannot be attributed to either change. [Project 09](09-governance-and-replay-harness.md) §5.5's swap-set requires individually attributable changes; two policy-bearing changes in one window destroys that. |
| 2 | Legal standardises covenant wording; Covenant Operations' test schedules still reference the old templates | New instances get new definitions; old instances must not. The failure is silent and appears as a breach 18 months later. |
| 3 | Early Warning adds 14 signals; the watchlist grade boundaries were calibrated on the previous 172 | Every facility's aggregate score shifts. Thousands move a grade for no credit reason, and the mandated actions fire. |
| 4 | Credit Risk Modelling recalibrates the commercial scorecard mid-quarter | Portfolio Management's quarterly migration report now spans two scales. §5.10.4. |
| 5 | Recoveries needs a field in the handoff that Origination Engineering owns | A change to a phase for the benefit of a team that never runs it |
| 6 | Provisioning needs the forbearance classification at decision time; Policy owns the classification rules; Recoveries grants the concession | One output, three owners, and a reporting date that is not the Bank's decision date |
| 7 | Portfolio Management tightens a concentration limit while the EP-8 cascade queue is draining | Half the cascade ran on the old limit |
| 8 | Sector Analytics patches 60 sector codes after a shock; historical decisions must continue to resolve their old codes | [Project 05](05-business-credit-nested-entities.md) change scenario 7, at five years' depth |

#### 5.15.2 The deployment question

One artefact, five execution profiles (§8), fourteen owners. A release that is
safe for the interactive origination path may be unsafe mid-batch.

**Six requirements:**

1. **Independent deployability.** A team must be able to change what it owns
   without a coordinated release across fourteen teams. The alternative is a
   release train whose cadence is the slowest team's, and the observable
   consequence of that is not slowness but **workarounds** — the 546 inline
   literals and 79 passthrough steps of doc 01 §5 were all produced by people
   avoiding a release.
2. **A batch in flight pins its versions at the start.** A version change during
   an 8-hour review batch either does not apply to that batch or aborts and
   restarts it. It never applies to part of it. Same for the 3-hour early-warning
   window and the covenant test runs.
3. **Every change declares which entry points it affects.** A change affecting
   more than three requires a joint release with a named coordinator.
4. **Ownership is expressible below the level of the flow.** Every one of the
   1 900 decision points resolves to exactly one owning team. Zero owners or two
   owners is a defect found by a check, and the check runs on every build.
5. **A change calendar with declared freeze windows.** At most one
   policy-bearing change per entry point per release window. The monthly review
   batch window and the quarter-end covenant peak are freezes.
6. **Read-only owners constrain the writers.** Provisioning, Recoveries and
   Audit change nothing and can block anything, because a change that makes the
   forbearance classification unavailable at decision time is a reporting failure
   regardless of how good it is for origination. Their requirements must be
   expressible as constraints on the artefact, not as review comments.

### 5.16 Navigability and the five-year question

The seventh cross-cutting requirement, stated as a hard, measurable one.

> **"Why was this facility graded 7 at its 2027 review?"**
> Asked in 2031, by someone who was not there, when the flow has changed forty
> times since.

Four measurable requirements. Each has a number, a method and a pass mark,
because a navigability requirement without one is a preference.

**N1 — The historical decision question.** A competent credit analyst who has
never seen the facility can answer the question above, unaided, **within 30
minutes**, using only the decision record and the generated plain-language
rendering of the flow build that produced it
([09](09-governance-and-replay-harness.md) §5.7). *Tested quarterly on a random
sample of ten past decisions at least two years old, by people who did not work
on them. Pass mark 9 of 10.*

**N2 — The "where is this decided" question.** A new engineer, given the name of
an output — `watchlist_grade`, `security_type`, `covenant_test_result`,
`authority_level_code` — finds the logic that produces it in **under 5 minutes**,
and the complete list of everything that can change it in **under 15**. *Tested
on each new joiner, on five outputs drawn at random. Pass mark 5 of 5 for the
first, 4 of 5 for the second.*

**N3 — Every decision point is addressable.** All 1 900 resolve to an owner,
effective dates, a plain-language description written by the owner, the reason
codes it can raise, and the tables it reads. *Machine-checked on every build;
zero exceptions permitted.*

**N4 — The change history is navigable.** "What changed between the build that
produced the 2027 review and the build that produced the 2028 review, and which
of those changes could have moved **this** facility's grade?" must be answerable
**statically**, without running anything, and confirmable dynamically by
re-running. *Tested on twenty facility-year pairs annually; the static answer
must be a superset of the dynamic one and no more than three times its size.*

N4 is the one that fails first, and it fails quietly: forty builds is forty
diffs, and a diff of a 1 900-decision-point artefact is unreadable unless the
artefact was designed so that most changes touch a small, nameable part of it.

### 5.17 What reuse costs at scale

The eighth cross-cutting requirement, and — per §2.2 — the reason this project
exists in the form it does. Project 11 consumes roughly **1 280 of its 1 900
decision points** rather than writing them. This section states what that
actually costs and what must be true for it to be cheaper than rebuilding.

Each of the six subsections is a requirement, not an observation.

#### 5.17.1 Vocabulary mismatch

The library publishes `applicant_age_years`, `gross_monthly_income`,
`net_monthly_income`, `months_employed`, `dependants_count`,
`employment_type_code`, `existing_obligations`, `living_expenses`,
`max_affordable_instalment` and `affordability_verdict_code`
([00](00-shared-credit-core-library.md) §4). Every one of them was named for a
retail applicant.

Inside a business flow, each belongs to somebody with a role, and the role
changes what the name means:

| Library name | In this flow it is | And it means |
|---|---|---|
| `applicant_age_years` | A director's age, a surety's age, a sole proprietor's age | At E-AROD-10 it is the surety's age at the **final instalment date**, not at `decision_date` |
| `gross_monthly_income` | A sole proprietor's drawings; a surety's personal income; a director's salary from the applicant | For the sole proprietor it is **the same money** as the business's net profit, counted once |
| `existing_obligations` | A surety's personal obligations | Excluding the facility they are standing surety for, which is not their obligation until it is |
| `max_affordable_instalment` | A capacity figure for one natural person | Against a business instalment that person does not pay |
| `total_exposure` | `core.exposure`'s client-level aggregate | Against a group-level aggregate over a derived set with a composition date |
| `decision_date` | The date whose rules govern | Which here is one of four dates — decision, knowledge, test, determination |

**Count**: **31 published names** are consumed in a role-dependent sense here, of
which **12** would be materially misread if the role were dropped.

**The requirement, and the failure it prevents**: a consumer must be able to use
a published value **in a role** without renaming it at every use site. Doc 01
§5.1 records what happens otherwise — **79 identity-passthrough steps** written
because a consumer could not rename a value a shared unit produced. Here the
volume would be worse: `core.income` is called for up to 40 entities, and
rewrapping its five outputs per entity per role is 200 wrappers on one phase.

Three properties are required:

1. A value carries **whose** it is, structurally, not by naming convention. The
   same capability applied to entity 7 and entity 12 produces two answers that
   cannot be confused and do not need distinct names.
2. Role is a **property of the application of the capability**, not of the
   capability or of the value. `core.adverse_events` does not know about
   criticality classes; the criticality class parameterises its thresholds
   ([05](05-business-credit-nested-entities.md) §13 question 3).
3. **Count the passthroughs.** A build-time count of identity relabels is
   reported, and a rise in it is treated as a design finding rather than as
   noise. The number is the measurement instrument for this whole section.

#### 5.17.2 Outputs the component does not produce

The recurring shape: a consumed component gives this project **90% of what it
needs**, and the missing 10% is not a small version of the 90%.

| Component | Produces | This project needs | Gap |
|---|---|---|---|
| [05](05-business-credit-nested-entities.md) §5.10 combined grade | A business grade | A grade **migration** with a six-way cause decomposition against the previous decision of record | The predecessor is not an input to project 05's assessment at all |
| [05](05-business-credit-nested-entities.md) §5.5–5.7 entity assessment | A full assessment of all entities | **One entity re-scored** and **one event classified**, 9 400 times a day | Published as an optimisation (§8's 500 ms partial re-assessment), not as an interface |
| [05](05-business-credit-nested-entities.md) §5.8 people blend | A blend over all entities | A **partial re-blend** where one entity's PD moved and the weights did not | Same shape, different answer, and the equivalence must be provable |
| [05](05-business-credit-nested-entities.md) §5.11 collateral | Adjusted cover for **one** facility | An **allocation across a set** of facilities with an over-allocation invariant | Cross-record, which the component is not |
| [02](02-affordability-assessment.md) §5.8 | Four assessment modes | A **fifth**: periodic re-assessment of a sole proprietor on year-old evidence | The modes are declared as a closed set |
| [06](06-consolidation-and-restructure.md) §5.9 | Eight concessions, retail | **21**, across nine facility types, with a forbearance classification of record | The catalogue is a table; the classification is new logic |
| [00](00-shared-credit-core-library.md) §6.16 `core.exposure` | Client and related-party aggregate | Aggregate **as at a date**, over a group with a composition date, including contingent exposure at two conversion factors | Time and contingency are both absent |
| [00](00-shared-credit-core-library.md) §6.22 `core.adjustments` | The overlay stack in force | The **difference** between two stacks a year apart, decomposed | Requires two dates, not one |

**What the consuming project does when it gets 90%** — four options, and the
choice must be declared per gap rather than made ad hoc:

| Option | When it is right | What it costs |
|---|---|---|
| **Extend the component** — a new published output, minor release | The gap is genuinely general | The owning team's roadmap, and every other consumer's regression |
| **Compose around it** — consume it and compute the gap locally | The gap is this consumer's concern only | Duplication if a second consumer needs it; and the local part is unowned by the component's team |
| **Parameterise it** — the gap is a mode or a scope | The component already varies this way | Mode proliferation; [02](02-affordability-assessment.md) §5.8's warning that "modes must not become copies" |
| **Fork** | Never, by policy | §5.17.6 |

**The requirement**: every gap in the table above is resolved by a **declared,
recorded choice** among the first three, with the reason, the owner and the
review date. An undeclared gap is resolved by whoever hits it first, at 2am,
during a batch, and the resolution is always the fourth option.

#### 5.17.3 Semantics that differ by consumer

The same component, correctly used twice, must give different answers — and
telling that apart from a bug is the problem.

**The sole proprietor case.** [Project 02](02-affordability-assessment.md)
computes `discretionary_income` from `net_monthly_income` less `living_expenses`
less `existing_obligations`. For a sole proprietor, the business's net profit
after drawings **is** the personal income. Three things must hold and none is
automatic:

1. The money is counted **once**. Drawings deducted from EBITDA at O14 and
   counted as income at project 02 is the same rand on both sides, and the
   resulting affordability is wrong in the client's favour by exactly the
   drawings.
2. The proposed instalment is a **new obligation** in project 02's terms and a
   **debt service item** in O14's terms, and it must not appear in both.
3. Project 02's statutory verdict is a **hard fail** under the regulated regime
   and an **input** under the unregulated one
   ([05](05-business-credit-nested-entities.md) §5.12 item 7). One component, one
   answer, two consequences.

**Null semantics.** [Project 00](00-shared-credit-core-library.md) §7.4 requires
three null situations to stay distinct: not collected, collected as zero, could
not be established. In a business flow a fourth appears: **not applicable to this
role**. A surety has no `months_employed` if they are retired; a corporate
guarantor has no `applicant_age_years` at all. Collapsing "not applicable" into
"not established" puts a scoreable entity into the unscoreable bucket and drags
the 75% coverage ratio down ([05](05-business-credit-nested-entities.md) §5.8
PP-02), producing a referral for a structural reason.

**Roll-up semantics.** `core.adverse_events` classifies one event.
[Project 02](02-affordability-assessment.md) rolls its results up by worst-of;
[project 05](05-business-credit-nested-entities.md) §5.6 rolls them up by twelve
rules over counts, aggregates, recency, velocity and trend; L3 rolls them up by
weighted score with decay. **Three consumers, three roll-ups, one component** —
which is project 00 §13 question 6, and this project is its third data point.

**Materiality thresholds.** The same R80 000 judgment is material on a peripheral
shareholder, disqualifying on a controlling one
([05](05-business-credit-nested-entities.md) §5.5), and a weight-6 contributor
signal on the daily pass. Three correct, different answers about one fact, and
each must be able to point at the others (project 05 §9.6, extended from two
applications to *n* subjects).

**The requirement**: where a component's answer legitimately differs by consumer,
the difference must be **attributable to a declared parameterisation**, visible
in the record of both. "Why does this event show as material here and
disqualifying there" must be answerable by reading the two records, not by
reading the code.

#### 5.17.4 Version coupling

This project consumes components from **four** other projects, each on its own
release cadence and its own approval path.

| Source | Cadence | Approval | Coupling risk |
|---|---|---|---|
| [00](00-shared-credit-core-library.md) library | Release train; two majors may be live at once (§7.1) | Credit Systems + Compliance | Highest: 22 capabilities, one of them in every phase |
| [02](02-affordability-assessment.md) | Statutory changes on gazette; policy quarterly | Compliance | A major version lands on a statutory date nobody chose |
| [05](05-business-credit-nested-entities.md) | Policy quarterly, out of cycle after a loss event | Credit Committee | A roll-up rule change moves every entity verdict in the book |
| [06](06-consolidation-and-restructure.md) | Semi-annual | Credit Committee + Recoveries | Lower volume, higher consequence |

**Three scenarios, each a requirement:**

1. **Project 02 ships a major version.** Under
   [00](00-shared-credit-core-library.md) §7.1, two majors may be live
   simultaneously in different consumers. Here they must be live simultaneously
   **within one consumer**: EP-1 moves to the new major on release, while EP-3's
   in-flight review cohort and every replay of a past decision stay on the old
   one. The requirement is that a major version be selectable **per assessment**,
   resolved from the decision record, not per deployment.
2. **Project 05 changes a roll-up rule.** AE-R-02 becomes four-in-18 for
   peripheral entities (project 05 change scenario 4). Every entity verdict in a
   240 000-facility book potentially moves. This project must be able to say,
   before the change ships, how many facilities change watchlist grade, how many
   reviews would have graded differently, and how many covenant sets would have
   been set differently — which is
   [project 09](09-governance-and-replay-harness.md) §5.5's swap-set, run over
   five years of history rather than one population.
3. **A covenant test from 2027 replays in 2031.** It must resolve the component
   versions in force at its determination date — project 05's rule set, the
   library capabilities, the flow build — **and** the covenant definition version
   agreed in 2026, which is a different resolution rule (H2). Two resolution
   mechanisms in one replay, and the replay must use each for the right thing.

**The requirement**: a decision of record pins the **version of every consumed
component**, not just of every table, and the pinned set must be resolvable for
the full retention period. A component version that was later overwritten in
place is not a version
([09](09-governance-and-replay-harness.md) §5.15 item 5, applied to code rather
than to data).

#### 5.17.5 The performance of reuse

The components were sized for project 05's profile: 900 applications a day, a
3-second budget, an 8-hour monthly batch over 180 000 clients.

This project calls them:

| Call pattern | Volume | Designed for |
|---|---|---|
| Whole-assessment, interactive | 2 900/day | Yes — this is project 05's case |
| Whole-assessment, batch | 15 000/month in 8 hours | Yes |
| **Single event classification** | ≈ 9 400/day as singletons | No — published as an optimisation |
| **Single entity re-score and re-verdict** | ≈ 48 000/day | No |
| **Partial people re-blend** | ≈ 13 200/day | No |
| **Covenant-scoped financial spreading** | ≈ 2.4 M/year, of which the peak month is 310 000 | No — O7 spreads for an assessment, not for one ratio |
| **Cascade-scoped exposure aggregation** | ≈ 1 900/day over groups of mean 2.3 businesses | No |

**The tension, stated plainly**: the cheapest way to make a component fast in the
call pattern it was not designed for is to write a second, faster version of it
for that pattern. That second version is a fork with a performance justification,
and it diverges on the first policy change.

**The requirement**: the same implementation serves both call patterns, and the
partial call is **provably equivalent** to the whole call over the same data —
which is [project 05](05-business-credit-nested-entities.md) §13 question 8
("what is the smallest re-run that is provably equivalent to a full one, and how
is that proof obtained?") turned from a design question into an operational
obligation. The weekly full reconciliation pass of §5.6.3 is how the proof is
maintained rather than merely asserted.

#### 5.17.6 The fork pressure

Naming, explicitly, the seven places where forking would be the cheapest thing
the team could do — because an unnamed pressure is one nobody is watching.

| # | Where | Why forking is tempting | What must be true for it not to happen |
|---|---|---|---|
| 1 | **The daily pass's event classification** | Calling a component built for a whole assessment 9 400 times a day as singletons is slow, and a stripped-down copy would be fast | The component publishes a single-element interface that is the same implementation, and its cost at singleton scale is measured and acceptable |
| 2 | **The partial people re-blend** | Project 05 publishes a blend, not a re-blend. Writing the re-blend locally is fifty lines | The re-blend is published by project 05 with its equivalence proof, or composition around the published blend is cheap enough |
| 3 | **Covenant-scoped spreading** | O7 spreads 48 lines and computes 11 measures; a DSCR covenant needs two lines and one ratio | The spreading component can be invoked for a declared subset, without a second mapping table appearing |
| 4 | **Project 02 for the sole proprietor** | The regulated assessment is heavy, the fifth mode does not exist, and a local approximation would pass most cases | The fifth mode is added, by project 02, on project 02's cadence — and this project can wait for it |
| 5 | **The seven new facility types' pricing** | Project 05's search is built around two products; six new candidate spaces is a large change to somebody else's component | The search is genuinely parameterised by its candidate space, which is project 05 §13 question 10 |
| 6 | **`core.exposure` for the cascade** | Group exposure as at a date, over contingent facilities, bounded by a propagation rule, in 15 minutes, 1 900 times a day | The library's exposure capability takes a date and an exposure-conversion convention, rather than this project holding a second aggregation |
| 7 | **The library's null handling for role-inapplicable values** | Adding a fourth null situation to `core` touches six consumers; a local convention touches none | The library adds it, or the three existing situations are genuinely sufficient and the fourth is shown to be a modelling error here |

**The measurement**: the number of forks is reported, per release, with the
reason for each and a remediation owner. **Zero is the target and is not
expected.** What is required is that every fork be visible, justified and owned —
because the previous generation's failure was not that forks happened, but that
nobody could say how many there were or what had diverged
([00](00-shared-credit-core-library.md) §1).

#### 5.17.7 The question this section exists to answer

> At what point does reuse cost more than rebuilding, and what evidence would
> show that line had been crossed?

Candidate evidence, stated so that it can be collected rather than argued about:

| Indicator | Threshold that should prompt the question |
|---|---|
| Identity-passthrough relabels attributable to consumed components | More than 40 |
| Declared gaps resolved by "compose around it" | More than half of the gaps in §5.17.2 |
| Forks, at any point | More than 2 |
| Releases blocked waiting on another project's cadence | More than 4 per year |
| Decision points consumed but locally re-tested because the component's tests do not cover this use | More than 15% of the 1 280 |
| Defects whose root cause is a component behaving correctly for its owner and wrongly for this consumer | More than 6 per year |

None of these individually proves the line has been crossed. All six moving
together does, and the point of writing them down before the project starts is
that they will otherwise be argued about afterwards with no data.

---

## 6. Parameters and tables

### 6.1 The sixty-three tables

Grouped by where they come from, because §5.17's argument depends on the ratio:
**12 consumed from the library, 16 inherited from project 05, 35 written here**.

Two properties are required of all 63, without exception
([00](00-shared-credit-core-library.md) §8): **cell-level attribution** — an
outcome names the cell of the version that produced it — and **diffability** — a
new version produces a review artefact showing what moved and what the aggregate
impact is.

#### A. Consumed from the library — [project 00](00-shared-credit-core-library.md) §8

| # | Table | Dimensions | Cells | Owner | Cadence |
|---|---|---|---|---|---|
| 1 | Sector risk table | 420 sectors × 6 attributes | 2 520 | Sector Analytics | Annual + shocks |
| 2 | Sector ratio benchmarks | 420 × 9 ratios × 3 statistics | 11 340 | Sector Analytics | Annual |
| 3 | **Business rate card** (products 50, 51) | 40 amount bands × 55 terms × 12 grades × 4 security types | **105 600** | Treasury | Monthly |
| 4 | Risk grade boundaries | 8 products × 5 segments × 12 grades | 480 | Credit Risk Modelling | Semi-annual |
| 5 | Appetite grid | 12 grades × 8 products × 6 segments × 5 values | 2 880 | Credit Committee | Quarterly |
| 6 | Reason code registry | 380 base codes × 9 attributes | 3 420 | Compliance | Monthly |
| 7 | Statutory expense norms | 12 income bands × 6 dependants × 2 | 144 | Compliance | On gazette |
| 8 | Internal expense norms | as above × 8 products | 1 152 | Credit Risk Policy | Quarterly |
| 9 | Obligation treatment matrix | 45 account types × 6 attributes | 270 | Credit Risk Policy | Quarterly |
| 10 | Tax tables | 8 brackets × 4 rebate classes | ~50/year | Credit Systems | Annual |
| 11 | Statutory fee caps | 4 product classes × 3 components | 12 | Compliance | Annual |
| 12 | Adjustment register | 40–120 live overlays × 11 attributes | ~1 000 | Credit Risk Policy | Ad hoc, sometimes weekly |

#### B. Inherited from [project 05](05-business-credit-nested-entities.md) §6.2

| # | Table | Dimensions | Cells | Owner | Cadence |
|---|---|---|---|---|---|
| 13 | Entity disqualification disposition | 14 rules × 3 criticality classes | 42 | Business Credit Risk Policy | Quarterly |
| 14 | Event amount thresholds | 14 event types × 3 classes × 2 severities | 84 | Policy | Quarterly |
| 15 | Event ageing decay | 14 types × 6 age bands | 84 | Policy | Quarterly |
| 16 | Collateral advance rates | **extended** to 14 classes × 4 attributes | 56 | Credit Committee | Annual |
| 17 | Surety cover table | 12 grades × 2 values | 24 | Credit Committee | Annual |
| 18 | Appetite maximum facility | **extended** to 12 grades × 5 appetite classes × 4 security types × 9 facility types | **2 160** | Credit Committee | Quarterly |
| 19 | Group exposure caps | 12 grades | 12 | Portfolio Management | Quarterly |
| 20 | Statement line mapping | ~600 source labels → 48 standard lines | 600 | Policy | Continuous |
| 21 | Transaction classification for turnover | ~20 rules × 6 attributes | 120 | Policy | Semi-annual |
| 22 | Scorecard `BUS-PERS-01` | 38 characteristics × ~8 bins | ~300 | Credit Risk Modelling | On release |
| 23 | Scorecard `BUS-COMM-01` | 29 × ~8 | ~230 | Credit Risk Modelling | On release |
| 24 | Thin-file variants | 17 + 12 characteristics × ~6 | ~175 | Credit Risk Modelling | On release |
| 25 | Behavioural scorecard `BUS-BEH-01` | 21 × ~8 | ~170 | Credit Risk Modelling | On release |
| 26 | Fallback grades, no-hit entities | 5 roles × 6 age × 4 tenure | 120 | Policy | Annual |
| 27 | Conditions catalogue | **extended** to 61 conditions (precedent and subsequent) × 6 attributes | 366 | Legal and Policy | Semi-annual |
| 28 | Override reason list | 18 reasons | 18 | Policy | Annual |

#### C. Written by this project

| # | Table | Dimensions | Cells | Owner | Cadence |
|---|---|---|---|---|---|
| 29 | Facility type register | 9 types × 16 attributes | 144 | Product | Annual |
| 30 | **Rate card, commercial asset finance** | 30 amount bands × 8 asset classes × 8 term bands × 12 grades | **23 040** | Treasury | Monthly |
| 31 | **Rate card, commercial property finance** | 24 amount × 6 property types × 10 LTV bands × 12 grades | **17 280** | Treasury | Monthly |
| 32 | Pricing grid, invoice and debtor finance | 12 grades × 8 debtor-quality bands × 6 turnover bands | 576 | Treasury | Monthly |
| 33 | Fee table, trade and guarantee | 7 instrument types × 12 grades × 6 tenor bands | 504 | Treasury | Quarterly |
| 34 | Rate table, business overdraft | 12 grades × 15 limit bands × 4 security types | 720 | Treasury | Monthly |
| 35 | Rate and exit-test table, bridging | 12 grades × 6 exit-source classes × 5 LTV bands | 360 | Treasury | Monthly |
| 36 | Asset life and term cap | 9 asset classes × 6 attributes | 54 | Product and Policy | Annual |
| 37 | Property LTV and advance table | 6 property types × 10 LTV bands × 3 | 180 | Credit Committee | Annual |
| 38 | Debtor eligibility and borrowing base rules | 22 rules × 7 attributes | 154 | Policy | Semi-annual |
| 39 | **Covenant definition library** | 148 templates × ~22 attributes, across **≈ 1 900 live versions** | **≈ 41 800** | Legal, with Policy on levels | Continuous |
| 40 | Covenant default sets | 9 facility types × 4 grade bands × 5 amount bands | 180 | Policy | Quarterly |
| 41 | Covenant test schedule rules | 148 templates × 6 schedule attributes | 888 | Covenant Operations | On documentation |
| 42 | Breach classification matrix | 148 templates × 4 classes | 592 | Policy and Legal | Semi-annual |
| 43 | Cure period table | 16 covenant families × 4 breach classes | 64 | Legal | Annual |
| 44 | **Waiver authority matrix** | 4 breach classes × 12 grades × 9 facility types | 432 | Credit Governance | Annual |
| 45 | Cross-default thresholds | 9 facility types × 3 de minimis tiers | 27 | Legal | Annual |
| 46 | Information undertaking catalogue | 26 undertakings × 7 attributes | 182 | Covenant Operations | Annual |
| 47 | Negative pledge and permitted exceptions | 18 × 6 | 108 | Legal | Annual |
| 48 | **Early warning signal catalogue** | 186 signals × 11 attributes | **2 046** | Early Warning | Monthly |
| 49 | Signal weighting and decay | 186 × 6 age bands | 1 116 | Early Warning | Quarterly |
| 50 | Watchlist grade boundaries | 6 grades × 5 inputs | 30 | Credit Risk Policy | Semi-annual |
| 51 | Mandated action matrix | 6 watchlist grades × 9 facility types × 5 action classes | 270 | Portfolio Management | Semi-annual |
| 52 | **Review scope matrix** | 9 entry points × 28 phase-parts | 252 | Credit Governance | Semi-annual |
| 53 | **Amendment re-open matrix** | 14 amendment kinds × 28 phase-parts | 392 | Credit Governance | Semi-annual |
| 54 | **Delegated authority matrix** | 9 amount bands × 12 grades × 4 security types × 9 facility types | **3 888** | Credit Governance, Board-approved | Annual, and on mandate change |
| 55 | Authority pack content matrix | 7 levels × 46 pack sections | 322 | Credit Governance | Annual |
| 56 | Master scale registry | 4 scale versions × 12 grades × 6 attributes | 288 | Credit Risk Modelling | On recalibration |
| 57 | **Grade migration restatement maps** | 6 pairwise maps × 12 × 12 | 864 | Credit Risk Modelling with Portfolio Management | On recalibration |
| 58 | Staging and SICR trigger table | 24 triggers × 5 attributes | 120 | Provisioning | Semi-annual |
| 59 | Forbearance concession catalogue | 21 concessions × 9 attributes | 189 | Policy with Recoveries | Annual |
| 60 | Forbearance probation and cure rules | 8 rules × 5 attributes | 40 | Provisioning and Policy | On standard change |
| 61 | Exit trigger table | 18 triggers × 4 attributes | 72 | Portfolio Management | Annual |
| 62 | Collateral revaluation cadence and allocation priority | 14 classes × 9 facility types | 126 | Legal with Portfolio Management | Annual |
| 63 | Degraded review rules | 9 data-gap classes × 4 severities | 36 | Policy | Quarterly |

**The largest, named**: the business rate card at 105 600 cells; the asset
finance card at 23 040; the property card at 17 280; the sector ratio benchmarks
at 11 340; the delegated authority matrix at 3 888; the reason registry at 3 420
base codes plus roughly **260 contributed by this project** in ranges 5100–5999;
the appetite grid at 2 880; the appetite maximum facility table at 2 160 after
extension to nine facility types; the early warning catalogue at 2 046; and the
covenant definition library, which is not the largest by cell count but is the
hardest by every other measure.

### 6.2 Reason codes contributed

Roughly **260 new codes**, ranges 5600–5999, on top of project 05's 140 in
5100–5599. They cover covenant breach classes and families (48), waiver and cure
outcomes (22), watchlist grades and mandated actions (38), amendment outcomes
(26), forbearance and staging classifications (31), exit and handoff reasons
(18), degraded review bases (12), cascade consequences (24), authority and
exception reasons (26), and comparability outcomes (15).

Three of those groups are new in kind, not just in number: **cascade
consequences** need a communicability attribute for what may be said to business
B about business A (§5.12.4); **comparability outcomes** include a code meaning
*not comparable*, which is a reason for the absence of a number rather than for a
decision; and **forbearance classifications** must carry the evidence reference
for the negative case (§5.8.5).

### 6.3 The covenant definition library as a versioned artefact

This is table 39, and it deserves its own treatment because it is the only
artefact in the estate that is **contractual** rather than policy, and the
distinction changes everything about how it is versioned.

| | Policy artefact (62 of the 63) | Covenant definition (table 39) |
|---|---|---|
| Resolution rule | By `decision_date` ([00](00-shared-credit-core-library.md) §7.3) | By the version **bound to the instance** at documentation |
| Who may change it | Its owner, under approval | Nobody, for an existing instance — only a new instance or an amendment |
| When a new version applies | To every assessment from its effective date | Only to instances created from that date |
| How old versions die | Superseded | Only when the last instance bound to them closes — up to 20 years for product 53 |
| Live versions | 1 (plus history for replay) | **≈ 1 900** simultaneously |

**Structure.** 148 **templates**, each with a family (financial ratio,
information undertaking, negative pledge, ownership and control, positive
undertaking, borrowing-base, portfolio-specific). Each template has a chain of
**versions**, ordered and dated. Each version has 22 attributes (§5.5.1). Each
facility's covenant is an **instance**, which binds one version, adds its own
level, step schedule, test dates and delivery terms, and never re-binds.

**Distribution**: the median template has 3 versions; the most-revised — the DSCR
template — has **11**, of which 9 still have live instances. 960 000 live
instances across 240 000 facilities.

**Four requirements on the library itself:**

1. **A version is immutable once an instance binds to it.** Editing a version in
   place retrospectively changes the terms of live contracts, which is not a
   data-quality problem but a legal one.
2. **Every version must remain computable for as long as any instance binds it,
   plus seven years.** Which means the 48-line mapping table version, the
   accounting-basis rules and the annualisation conventions that a 2026 version
   depends on must also survive — a covenant definition has a dependency
   closure, and retiring anything in it retires the definition.
3. **The library must be diffable at the level of a version pair**, because the
   question "what changed between DSCR v4 and DSCR v7, and how many live
   instances are on each" is asked every time Legal standardises wording (change
   scenario 13).
4. **A standardisation across a live book is an amendment programme, not a
   library edit.** Moving 40 000 instances from nine DSCR versions to one is
   40 000 amendments (AM-06), each with an authority, a client consent and a new
   definition version dated at the amendment. The library must make it impossible
   to do it any other way.

### 6.4 Parameters

Roughly **190 parameters**, of which about **130** are owned by people who are
not engineers. Five ownership classes, and the boundaries between them are the
governance model.

| Class | Examples | Owner | Who may not change it |
|---|---|---|---|
| **Statutory** | Expense norms, tax brackets, fee caps, the regulated-regime thresholds, the 1% NPV materiality for a diminished obligation, the 90-day default counter and its materiality components | Compliance, transcribing the published instrument | Everyone, including Credit Risk Policy |
| **Policy** | Criticality thresholds, coverage requirement, blend weights, DSCR and gearing thresholds, event thresholds, degraded review bases, signal weights, watchlist boundaries, breach classification bands | Business Credit Risk Policy, Early Warning, Portfolio Management | Product teams, engineering |
| **Governance** | The delegated authority matrix, the review scope matrix, the amendment re-open matrix, the pack content matrix, the waiver authority matrix | Credit Governance, Board-approved | Everyone else |
| **Contractual** | Covenant levels, step schedules, test dates, delivery terms, cure rights, cross-default de minimis, security terms, the contractual margin | Fixed **per instance** at documentation | **Everyone, permanently** — including the owner of the standard it was drawn from |
| **Overlay** | The eleven-position stack of [05](05-business-credit-nested-entities.md) §5.10, with scope, order, approval and expiry | Credit Risk Policy and Credit Committee | Product teams; no overlay may loosen an absolute rule |

The **contractual** class is new to this set. No earlier project has a parameter
that nobody may change, ever, including the person who owns the standard it came
from — and the failure to represent that class is H2. A design in which
contractual parameters live in the same mechanism as policy parameters will apply
a policy resolution rule to them, and will do so silently.

Selected values, to fix magnitudes:

| Parameter | Value | Owner | Cadence |
|---|---|---|---|
| Covenant headroom convention | 20% on cover ratios; 0.25× on leverage-style | Policy | Annual |
| Equity cure limits | 2 in any 4 consecutive test periods; never consecutive; 4 over the life | Legal and Policy | Annual |
| Technical breach escalation | 3 in 12 months → class 2 | Policy | Quarterly |
| Waiver repeat limits | 2 consecutive on one covenant; 3 on a facility in 24 months | Policy | Quarterly |
| Cross-default de minimis | Greater of R250 000 and 1% of group exposure | Legal | Annual |
| Watchlist score boundaries | 15 / 30 / 50 / 75 | Credit Risk Policy | Semi-annual |
| Cascade depth bound | 2 hops | Credit Governance | Rare |
| Cascade fan-out cap | 250 facilities | Credit Governance | Rare |
| Review completion deadline | Review date + 30 days | Portfolio Management | Annual |
| Degraded review next-review dates | 12 / 12 / 6 / 3 / 2 months by basis | Policy | Quarterly |
| Forbearance probation | 1 year non-performing→performing; 2 years performing→cleared | Provisioning | On standard change |
| Diminished-obligation materiality | 1% NPV | Compliance | On regulation |
| Economic dependence benchmark | 50% of gross receipts or expenditure from one counterparty | Portfolio Management | Annual |
| Self-approval second-signature floor | R1 000 000 | Credit Governance | Annual |
| Over-allocation tolerance | Zero | Legal | Rare |

---

## 7. Outputs

### 7.1 To the caller, by entry point

Every entry point emits the common core, plus its own.

**Common core, on every assessment**: `decision_of_record_id`,
`assessment_kind_code`, `facility_id` and/or `client_id`, `decision_date`,
`knowledge_date`, the predecessor decision and `comparison_basis_code`,
`risk_grade` with `master_scale_version` and `probability_of_default`, their
unadjusted counterparts, `adjustment_set_id` and `adjustments_applied`,
`authority_level_code`, `outcome_code`, `decline_reason_codes` and
`primary_reason_code`, the four state flags (`watchlist_grade`,
`forbearance_flag`, non-performing state, `staging_code`), every table version
and cell read, every consumed component version, and the data-quality flag set.

| Entry point | Additional output |
|---|---|
| EP-1, EP-2 | The offer: amount, term, rate with `rate_cell_id`, fees, instalment, DSCR, `binding_constraint_code`; the security schedule with its allocation; the covenant set with definition versions; conditions precedent and subsequent; the pack for the required level; the per-entity summary of [05](05-business-credit-nested-entities.md) §7.1 |
| EP-3 | `review_basis_code`; the three grades with their scales; the six-way cause decomposition; the re-pricing numbers and action; the limit decision with its portfolio context; the exit triggers met; the covenant, waiver and watchlist history for the year; the next review date and the rule that set it |
| EP-4 | `covenant_test_id`, the definition version, all three dates, the inputs by standard line, the computed value, the threshold on the step schedule, headroom in ratio and percentage terms, the result, `breach_class_code`, cure availability and clock, any waiver applied, cross-default consequences, signals emitted |
| EP-5 | `watchlist_grade`, the previous grade, the `signal_set` with values, weights and decay states, the cause of any change classified as *new signal* or *decay*, the mandated actions with owners and deadlines |
| EP-6 | The amendment kind, the phases re-opened and the matrix version that said so, the amended terms, the re-tested position of every facility sharing affected collateral, the new covenant definition versions where reset, the authority sequence |
| EP-7 | The concession package across the facility set, per facility; the NPV cost; both affordability verdicts; the forbearance classification **with its evidence, including for the negative case**; the staging consequence and probation clocks; the options rejected with reasons; the before-and-after comparison ([06](06-consolidation-and-restructure.md) §5.10) |
| EP-8 | `cascade_id`, the trigger with its classification, the propagation path, every subject touched, what was recomputed and what merely flagged, the revised group composition with its date, the revised headroom per business, the facilities whose position changed, and whether a bound was hit |
| EP-9 | An indicative capacity range and price range, a confidence band, the assumptions made, the three things not done, and an explicit non-binding marker. **No decision of record.** |

### 7.2 Persisted

Everything in §5.10.2, plus: the full nested record of
[project 05](05-business-credit-nested-entities.md) §7.2 at all three levels; the
complete covenant test history; the waiver register with conditions and expiries;
the watchlist grade history with signal sets; the allocation history per
(collateral item, facility, date); the cascade records; the authority sequence
and the pack as presented; and the forbearance and staging history with clocks.

**Retention**: seven years after the facility closes, or seven years after a
decline — [project 05](05-business-credit-nested-entities.md) §7.2's rule,
unchanged. Within that, the **five years of re-decisions** are the comparability
window of §5.10 and every one of them must remain reproducible and comparable for
the whole period.

**Volume**: ≈ 2.6 M decisions of record over five years (11 mean per facility ×
240 000, net of closures), ≈ 16.4 M covenant test records, ≈ 438 M watchlist
grade evaluations at daily granularity — of which only grade *changes* and a
weekly full snapshot are retained in full, which is itself a decision that has to
be justified to Audit rather than assumed.

---

## 8. Non-functional requirements

### 8.1 One flow, five profiles

The volume asymmetry is the defining non-functional problem. The same 1 900
decision points are exercised under five profiles whose budgets differ by four
orders of magnitude, and a design optimised for any one of them fails at least
two of the others.

| | **A — interactive origination** | **B — interactive pre-assessment** | **C — review batch** | **D — covenant tests** | **E — daily early warning** |
|---|---|---|---|---|---|
| Entry points | EP-1, EP-2, EP-6, EP-7 | EP-9 | EP-3 | EP-4 | EP-5, EP-8 |
| Volume | 900 + 1 400 + 600 + 120 = **3 020/day** | **3 000/day** | **15 000/month**, 180 000/year | **2.4 M/year**; peak month 310 000 | **240 000/day** |
| Shape | One subject, full depth | One subject, shallow, partial data | 15 000 subjects, full depth | 2.4 M small evaluations | 240 000 subjects × 186 signals, with fan-out |
| Budget | p95 **4 s**, p99 8 s | p95 **2 s**, p99 3 s | **8 hours**, overnight, once a month | Nightly scheduled run inside **4 hours**; certificate-driven within 2 business days | **3 hours**, complete by 05:00 |
| Peak | 4× mean in the last two business days of the month | Flat, office hours | One cohort a month | **13× seasonal**: 310 000 in the peak month against 195 000 mean | Flat, but fan-out p99 is 31 facilities per entity |
| External calls | Permitted, outside the budget | **None** | Permitted, batched | None — the certificate is the input | Deltas only |
| Degradation | To referral, never to approval | To "insufficient data", never to a number | Cohort splits across nights; never partial reviews | Test deferred with the deferral recorded, never a default pass | Incremental pass extends; weekly full pass is the backstop |

### 8.2 Requirements

| Requirement | Value |
|---|---|
| **Determinism** | Identical inputs, versions, `decision_date` and `knowledge_date` ⇒ identical outputs, bit for bit, including the order of `decline_reason_codes`, the pricing candidate record and the signal set |
| **Ordering independence** | Outcomes invariant under shuffling of entities, events, facilities in a pool, signals and covenant instances. Tested by shuffling, as [05](05-business-credit-nested-entities.md) §8 requires — extended here to the cross-facility allocation, which is the case most likely to violate it |
| **Cross-run consistency** | The incremental early-warning pass and the weekly full pass agree on every facility's grade. Any disagreement is a defect, reported |
| **Partial equivalence** | A single-entity re-score, a partial people re-blend and a covenant-scoped spread each produce results identical to the whole-assessment path over the same data (§5.17.5) |
| **Cascade latency** | p95 **15 minutes** from the triggering event to every affected subject's headroom being revised; p99 4 hours for queued cascades above the fan-out cap |
| **Cascade boundedness** | No cascade touches more than 250 facilities inline; no cascade exceeds 2 hops; no subject processed twice within one `cascade_id` |
| **Replay** | Any decision of record within the retention period re-derives exactly, with no network and no live service ([09](09-governance-and-replay-harness.md) §5.15 item 9) |
| **Cold start** | No per-request compilation. The first assessment after a deployment is not materially slower than the thousandth, on any of the five profiles |
| **Table refresh** | Any of the 63 tables refreshes without a code deployment. A covenant **definition** version is published, never refreshed (§6.3) |
| **Batch version pinning** | A batch pins every table, component and flow version at its start and holds them to completion (§5.15.2) |
| **Availability** | The flow degrades to referral or to a restricted outcome, never to an approval, when any of the eleven external views is unavailable. EP-9 degrades to "insufficient data" |
| **Evidence emission** | Idempotent, at-least-once, deduplicated by `decision_of_record_id`; a failure to persist is counted and alerted and may never fail the decision ([09](09-governance-and-replay-harness.md) §5.15 items 17, 18) |
| **Storage growth** | ≈ 2.6 M decisions of record, ≈ 16.4 M covenant test records and ≈ 5 M watchlist change events over five years, each fully provenanced. Retrieval of any one within 5 seconds |

### 8.3 The tensions between profiles, named

1. **B against A.** The 2-second pre-assessment must reach a plausible number
   without the phases that make the number right. If it is implemented as a
   separate path, it diverges from A within two quarters and the reconciliation
   requirement of §5.1 fails. If it is implemented as A with phases switched off,
   it either misses the budget or discovers that A's phases are not
   independently switchable.
2. **E against A.** The daily pass calls components built for A, 48 000 times a
   night, as singletons (§5.17.5). Making them fast for E without forking is the
   requirement; forking them is the failure.
3. **D against everything.** 310 000 covenant tests in the peak month, each
   needing a financial spread on a definition that may be five years old and a
   mapping table version that has since been superseded eleven times.
4. **C against the release cadence.** An 8-hour window once a month is a freeze
   window for fourteen teams (§5.15.2), and the freeze is what makes the batch
   attributable.
5. **Cascades against all four.** Cascades are event-driven and arrive whenever
   they arrive, including during C's window and E's window, and they must not be
   allowed to extend either.

---

## 9. Audit, evidence and explainability

### 9.1 What must be answerable, by whom, how long after

| Question | Asked by | How long after |
|---|---|---|
| Why was this facility graded 7 at its 2027 review? | The client, the RM, Internal Audit, the regulator | Up to 7 years, by someone who was not there |
| Did this facility deteriorate between 2026 and 2031, or did the scale change? | Credit Committee, the regulator, Model Validation | At every review and every recalibration |
| Why was business B's headroom reduced on 14 June? | Business B, its RM, the Ombud | Immediately, and for 7 years |
| Was this covenant breached, on the definition agreed at origination? | The client, Legal, the regulator | For the life of the facility plus 7 years |
| Was this concession forbearance, and was it reported as forbearance? | The regulator, Internal Audit, Finance | Up to 7 years |
| Which of these 1 900 decision points could have produced this outcome, and who owns each? | Credit Risk Policy, Internal Audit, a new engineer | Continuously |
| What was the Bank's approved policy in March 2028, and did this decision follow it? | Internal Audit, the regulator | Up to 7 years |

### 9.2 The five-year audit trail of repeated decisions

Project 09 replays one decision. Here the unit of audit is a **sequence**.

**What must be producible for any facility, for any window inside the retention
period**: an ordered list of every decision of record with its kind, its dates,
its outcome, its authority and its approver; the grade at each point with its
scale; the covenant test history with results, breaches, cures and waivers; the
watchlist history; the security position at each point with its allocation; the
group composition at each point; the forbearance and staging history; and, for
every consecutive pair, the six-way cause decomposition and the
`comparison_basis_code`.

**Three properties**, and the third is the one that fails:

1. **Each decision replays individually**, on the artefacts and component
   versions in force at its own determination date.
2. **Each pair reconciles**: the decomposition sums to the observed difference.
3. **The sequence is coherent**: no gap where a review was due and no decision of
   record exists; no decision whose predecessor link is broken by an amendment or
   a restructure; no grade expressed on a scale with no registry entry. A
   coherence check over the whole book, run monthly, reporting exceptions — and a
   book with zero exceptions is a book where the check is not working.

### 9.3 Explaining a facility reduction caused by a group cascade

Two artefacts, produced together, never merged (§5.12.4):

**Internal**: `cascade_id`, the triggering event with its classification and the
rule that classified it, the triggering entity, business A, the propagation path
hop by hop, every subject touched, what was recomputed for each, the before and
after of each group aggregate and each headroom figure, the bound that applied,
and every decision of record produced.

**Communicable**: the client-facing wording from the registry, which may name the
existence of a connected group and its aggregate limit, and may not name business
A, the entity, the event or the amounts — plus the route by which the individual
concerned obtains their own reasons directly.

**The new registry attribute**: project 05's registry says what may be disclosed
to a business about an individual. This project needs what may be disclosed to
business B about **business A**, which is a third party of a different kind, and
Compliance owns the answer. Getting it backwards is a confidentiality breach in
one direction and a failure to give reasons in the other.

### 9.4 The reviewable artefact at 1 900 decision points

[Project 09](09-governance-and-replay-harness.md) §5.7 requires a generated,
plain-language rendering of what a flow does, approved quarterly, diffed against
the last approved version. At 1 900 decision points, 63 tables and 9 entry
points, a naive rendering is several hundred pages, and project 09 §5.7 names
exactly that outcome as the appearance of governance rather than governance.

**Four requirements specific to this project's size:**

1. **Layered, and the layering is generated.** A policy summary per entry point
   (4 pages each, 9 of them) stating what the entry point does and what can
   change its outcome; a full reference; and a **diff view**, which is the thing
   actually read at re-approval.
2. **Sliceable by owner.** Each of the fourteen teams must be able to render
   exactly the decision points it owns, with everything those points read and
   everything that reads them. A team that cannot review its own surface without
   reading the other thirteen's will not review it.
3. **Sliceable by entry point.** "What does the annual review actually do" must
   render as a coherent document of its own, drawn from the same source, showing
   phases run in full, in part and not at all — otherwise the review scope matrix
   is a table nobody can check.
4. **Frozen and retained on approval.** Which rendering was approved, when, by
   whom. When the regulator asks what policy was in force in March 2028, this is
   the answer — and for this project the answer must also identify which of the
   forty builds since then changed it.

### 9.5 A covenant breach dispute requiring the definition in force at the time

The client disputes a 2029 DSCR breach. What must be produced:

1. The **covenant instance** and the exact `covenant_definition_version` bound to
   it at documentation in 2026, rendered in the words it was agreed in.
2. The three dates: test date, delivery date, determination date (§5.5.2).
3. The inputs by standard line, with the source period, the accounting basis, the
   statement mapping version and the audit level — **as they were used**, not as
   they would be mapped today.
4. The computed value, the threshold on the step schedule as at the test date,
   and the headroom.
5. The classification, the cure availability, and any waiver with its scope and
   expiry.
6. A **re-derivation**, on the 2026 definition and the 2029 inputs, producing the
   same number to the cent, with no network and no live service.
7. And the counterfactual the client will actually ask for: what the result would
   have been on the Bank's **current** standard definition — labelled clearly as
   not the contractual test, because the client will ask and answering with
   silence is worse than answering with a caveat.

The failure this exists to prevent is precise: a system that resolves the
definition by date rather than by instance produces item 6 without error, using
the wrong definition, and the Bank defends a breach the client did not commit.

### 9.6 A regulator testing whether forbearance was reported as forbearance

The hardest of the five, because it tests a **negative**.

The regulator takes a sample of restructures, amendments and covenant resets
over three years and asks, for each, whether it met the forbearance definition
and whether it was flagged.

**What must be producible for every one:**

| | |
|---|---|
| The classification | Forbearance or not, produced at decision time, with its date |
| Limb 1 evidence | Financial difficulty: which indicators, with their values and sources as at the decision date |
| Limb 2 evidence | Concession: what was changed, against what the client could not comply with, and the basis for concluding the Bank would or would not have granted it otherwise |
| **The negative case** | For every restructure classified **not** forbearance, the same two limbs assessed and recorded as not met, with the evidence |
| The consequences | Staging, probation clocks, the forborne flag's lifetime, and any re-trigger |
| The pattern test | Whether a sequence of individually-innocuous concessions — three waivers, a covenant reset and a term extension across two years — constituted forbearance in aggregate (§5.5.5) |

**The two failure modes, both of which the requirement forecloses**: a
classification derived later by Finance from a payment pattern, which is not a
decision and cannot be evidenced; and an empty record on the negative cases,
which is indistinguishable from a classification that was never made.

### 9.7 Attribution, and the limit on disclosure

[Project 05](05-business-credit-nested-entities.md) §9.2 governs unchanged: an
internal attribution that is complete and names the entity and the events, and a
communicable reason that does not disclose a third party's record. This project
adds two claimants that project 05 does not have — **business B** in a cascade
(§9.3), and **the individual whose personal surety is being pursued** after L6's
handoff, who is entitled to the basis of the claim against them and not to the
business's assessment.

---

## 10. Acceptance criteria

Dominated by structural and temporal properties, because those are what fail.

**Comparability and time**

1. Every decision of record carries `master_scale_version`, its predecessor link
   and a `comparison_basis_code`. Zero exceptions across the whole book.
2. For any two consecutive decisions of record on one facility, the six-way cause
   decomposition sums to the observed difference in grade, price and limit.
   Unapportionable residuals are reported; the target is zero and the acceptance
   threshold is that every residual is investigated within 30 days.
3. A migration report mixing comparison bases is **rejected**, not footnoted.
   Demonstrated by a test that attempts one.
4. Where a restatement is impossible — a non-invertible scorecard replacement —
   the system produces *not comparable* rather than a number. Demonstrated on the
   2030 `BUS-COMM-02` case.
5. Any decision of record within the retention period replays exactly, with no
   network, on the artefacts **and component versions** in force at its
   determination date.
6. Every facility in every monthly cohort produces a decision of record by its
   review date plus 30 days. No deferrals. Degraded bases 4 and 5 are outcomes,
   not failures.

**Covenants**

7. A covenant test resolves its definition by **instance binding**, never by
   date. Demonstrated by a test in which the standard wording changed after the
   instance was created and the old definition is used.
8. The three dates are separately recorded on every test, and a test computed on
   any of them incorrectly is detected by a check.
9. A waiver names its covenant instance, its definition version, its period and
   its expiry, and tests resume automatically at expiry. No waiver without an
   expiry can be recorded.
10. Repeat-waiver limits fire: a third waiver on one covenant is structurally
    impossible without an amendment or a restructure.
11. A cured breach is recorded as a cured breach and never as a pass.

**Cross-facility and cascade**

12. The over-allocation invariant holds across the whole book, checked nightly:
    no collateral item's allocations sum above its adjusted value at that date.
13. Any assessment changing a facility's claim on shared collateral re-tests
    every other facility sharing it and reports whether its `security_type`
    moved. Demonstrated on the §5.11.2 worked case.
14. The authority for a decision affecting facilities other than the subject is
    that of the whole affected set.
15. No cascade exceeds 2 hops, touches more than 250 facilities inline, or
    processes a subject twice. Enforced, not monitored.
16. Every cascade produces an internal attribution and a communicable
    explanation, and the communicable one discloses nothing about business A.
    Tested adversarially.

**Entity structure over time**

17. Every phase that reads structure declares which temporal view it uses;
    undeclared reads are caught at build time.
18. A replay of a past decision uses the knowledge-date view and reproduces the
    original answer, on the same facility where an effective-date query returns a
    different structure. Demonstrated on the March/June/November 2028 case.
19. An ownership-change covenant tests against a baseline pinned in the instance,
    and a late-arriving fact produces a re-test with a new determination date,
    never a silent correction.

**Authority**

20. The authority recomputes on every change to the proposed structure, and the
    sequence of levels is recorded. A post-approval structure change that would
    require a higher level invalidates the approval automatically.
21. Self-approval is detected, and a second signature is enforced above the
    floor.
22. The pack presented corresponds to the structure approved, at the level that
    approved it.

**Reuse**

23. No consumed component is forked. Where a fork exists it is registered, with
    its reason, its owner and a remediation date; the count is reported every
    release.
24. Identity-passthrough relabels attributable to consumed components are counted
    at build time and reported; a rise is treated as a design finding.
25. Every gap in §5.17.2 is resolved by a declared, recorded choice among extend,
    compose or parameterise, with an owner and a review date.
26. A major version of a consumed component is selectable **per assessment**,
    resolved from the decision record — demonstrated by running EP-1 on a new
    major and an EP-3 cohort and a replay on the old one, simultaneously.
27. The partial call paths — single-entity re-score, partial re-blend,
    covenant-scoped spread — produce results identical to the whole-assessment
    path over the same data, demonstrated over at least 5 000 subjects.

**Organisation, deployment and navigability**

28. Every one of the 1 900 decision points resolves to exactly one owning team, a
    plain-language description, effective dates, the reason codes it can raise
    and the tables it reads. Machine-checked on every build; zero exceptions.
29. A team deploys a change to what it owns without a coordinated release across
    the other thirteen. Demonstrated by three teams releasing in one week.
30. A batch in flight holds its pinned versions to completion; a version change
    during a batch never applies to part of it.
31. N1 to N4 of §5.16 are met at their stated pass marks, tested at their stated
    cadences.

**Scale**

32. Each of the five profiles meets its budget at the stated volumes, on the p99
    fan-out distribution rather than the median.
33. The incremental early-warning pass and the weekly full pass agree on every
    facility's grade.
34. A policy owner changes a signal weight, a watchlist boundary, a degraded
    review rule or a breach classification band without an engineer, producing a
    reviewable diff and a re-scored sample over the live book.

---

## 11. Change scenarios

Each has been requested of a real business-lending estate. A good structure makes
them cheap; the previous generation made most of them a rewrite.

1. **A scorecard recalibration lands mid-portfolio.** `BUS-COMM-01` is
   recalibrated in November, with 62 000 reviews already completed for the year
   under the old version and 118 000 to come. The year's migration report must
   not show a portfolio improvement that is an artefact of the recalibration, and
   the restatement map must exist before the first review under the new version
   runs.
2. **A tenth facility type launches** — supply chain finance, revolving against
   approved payables, priced per invoice, with a concentration covenant and a
   confirming-buyer dependency that makes the *buyer's* grade relevant to the
   *seller's* facility. New rate table, new covenant default set, new authority
   cells, new collateral class, new entry-point behaviour for EP-4.
3. **The sector table is rebuilt**: 420 codes become 500, with 60 remapped.
   Historical decisions must continue to resolve their old codes; five years of
   sector benchmarks must remain readable; and the sector-scoped overlays in
   force must be re-scoped without lapsing.
4. **An accounting standard change alters how financials are spread.** Lease
   commitments move onto the balance sheet. Gearing, interest cover and EBITDA
   all move for every business, with no change in economics. Covenants written on
   frozen-GAAP terms must continue to test on the old basis; covenants written
   without that wording must be reset; and the review's ratios and the covenant's
   ratios now differ for the same business in the same year.
5. **The group definition widens** from control-and-ownership to include economic
   interdependence. The average group grows from 2.3 businesses to 3.8; roughly
   14 000 clients acquire a connected party they did not have; concentration
   limits bind in 900 cases that previously had headroom; and every one of those
   needs a client conversation.
6. **The behavioural weight becomes tenure-dependent.** Project 05's
   18-combination weight table becomes 72. Whether that is a table edit or a
   re-release is the test.
7. **The authority matrix is restructured** from seven levels to five, with the
   automated mandate extended to R2 000 000 and the credit committee threshold
   raised. 3 888 cells become 2 160 with different boundaries; every in-flight
   proposal must be re-routed; and every historical decision must still resolve
   the level that actually approved it.
8. **Covenant wording is standardised across a live book.** Legal reduces nine
   live DSCR definition versions to one. 40 000 instances are affected. This is
   40 000 amendments with 40 000 client consents, not a library edit (§6.3), and
   the programme runs for eighteen months during which both worlds are live.
9. **The regulator requires restated grade migration across five years.** Every
   grade on every decision of record for 240 000 facilities restated onto the
   current master scale, with the restatement basis stated per facility and *not
   comparable* where no map exists. Delivered in six weeks.
10. **An equity cure is exercised for the first time** on a covenant whose
    definition applies cures to cashflow rather than to borrowings, and the
    client wants the cure to count in the following three rolling tests as well.
11. **Forbearance probation rules change**: the performing-forborne probation
    moves from two years to three. Every live forborne exposure's clock is
    affected; clocks already expired must not restart; and the reporting for the
    prior periods must remain as reported.
12. **A sector shock.** Construction moves to restricted overnight. Every
    construction facility is re-assessed, the sector PD multiplier goes to 1.40,
    2 800 facilities move watchlist grade in one night, and the daily pass's
    mandated actions generate 2 800 site visits against a capacity of 180 a week.
13. **A new early warning source** arrives: a national supplier payment register
    with its own ageing behaviour and no satisfaction concept. 12 new signals,
    new weights, and a recalibration of the watchlist boundaries that shifts
    every facility's score.
14. **Bridging facilities are withdrawn** from new business. Product 58 goes to
    run-off: no new facilities, existing ones honoured to maturity, extensions
    permitted only at level 5. Every phase must handle a product that exists for
    servicing and not for origination.
15. **Credit Risk Modelling is asked to produce restatement maps** for the first
    time, retrospectively, for two recalibrations that happened before the
    requirement existed. One is invertible and one is not.
16. **Project 02 ships a major version** on a statutory date. EP-1 must move on
    the date; the in-flight review cohort must not; and every replay must resolve
    the version it ran on.
17. **Project 05 changes a roll-up rule** — AE-R-02 becomes four-in-18 for
    peripheral entities. The swap-set must be produced over five years of history
    before the change ships.
18. **The library adds a fourth null situation** — not applicable to this role —
    across six consumers. This project asked for it (§5.17.3); the other five did
    not.
19. **Cross-default is called for the first time in eight years.** Every system
    that records the state has never had it exercised, and the one facility where
    it was called turns out to have three sibling facilities whose recorded
    cross-default state was set by a breach that was later cured.
20. **A single collateral item is found to be allocated twice** — to a facility
    in a group that was re-defined in scenario 5, and to one that was not. The
    over-allocation check is added retrospectively and finds 340 cases across the
    book.

---

## 12. Out of scope

- **Data acquisition**: bureau calls, screening enquiries, registry retrieval,
  document collection, optical extraction from a PDF financial statement, and
  receipt and parsing of compliance certificates. The flow consumes a spread and
  a certified set of figures; the mapping of labels to standard lines is in
  scope, the extraction is not.
- **Screening itself.** Consumed as an outcome and a confidence, as in
  [project 05](05-business-credit-nested-entities.md) §12.
- **Facility documentation and execution**: drafting, negotiation, signature,
  bond and notarial bond registration, and the registry filings behind a
  condition subsequent. The flow determines what is required and tracks whether
  it happened; it does not do it.
- **Disbursement, servicing, billing and collection of instalments.**
- **Operational collections and workout execution** —
  [project 08](08-collections-treatment.md) owns treatment; this project owns the
  decision to hand over and the contents of the handoff.
- **Impairment measurement.** This project produces the staging classification,
  the forbearance flag and their triggers; the expected credit loss calculation,
  the models behind it and the reporting are Finance's.
- **Regulatory reporting itself.** The flow produces the classifications the
  returns consume; it does not produce the returns.
- **Model development.** Scorecards and calibrations are consumed, not fitted.
  Restatement maps are consumed as published artefacts.
- **Corporate and large-corporate lending** above R150 000 000 turnover, which is
  individually underwritten — [project 05](05-business-credit-nested-entities.md)
  §12, unchanged.
- **Specialised lending**: property development and project finance, and
  asset-based lending against a revolving borrowing base beyond product 54's
  mechanics.
- **Decision record storage and replay tooling** —
  [project 09](09-governance-and-replay-harness.md).
- **The case-management workflow** around referrals, watchlist actions and
  workout: queues, allocation, diaries, service levels and the analyst's screen.
  The flow says a site visit is required within 30 days; the system that books
  it, chases it and records it is not this.
- **Client communication delivery.** The flow produces the communicable reason;
  the letter, the channel and the language selection are not this.

---

## 13. Questions the implementation must answer

The measurement instrument. The sharpest are about time and about reuse.

**Time**

1. **How is a flow that re-decides the same subject for five years kept
   comparable with itself?** Forty builds, two scorecard recalibrations, a master
   scale change and 63 tables moving on six cadences. What construct carries
   comparability, and is it part of the flow or outside it?

2. **Where does the history live?** A facility's eleven decisions of record are
   an input to the twelfth. Is the decision history a parameter, an input, a
   store the flow reads, or a first-class subject the flow operates on — and what
   happens to replay if it is a store?

3. **How are two version-resolution rules expressed in one flow** without either
   being applied to the wrong artefact? Policy artefacts resolve by
   `decision_date`; contractual ones resolve by instance binding and never move.
   Both are effective dating and they are not the same mechanism.

4. **How is bi-temporality expressed** so that a phase cannot accidentally read
   the wrong view? Both views are plausible, neither errors, and the difference
   shows up years later.

5. **What is a schedule, structurally?** A covenant instance fires on its own
   calendar for five years, against a definition frozen at its creation, with a
   step schedule, cure rights and an expiry. It is not a rule, not a table and
   not a parameter. Is it a fourth thing, or a rule set with a temporal scope?

6. **How is "not comparable" represented** as a first-class answer rather than as
   a missing value? A system that always produces a number will always produce a
   wrong one.

7. **What is the unit of state?** Four flags per facility — watchlist,
   forbearance, non-performing, staging — with probation clocks, permitted
   transitions and interactions that do not imply each other. Does the flow own
   them, read them, or both?

**Cross-record structure**

8. **How is a cross-facility constraint expressed** when the facilities are
   decided separately, months apart, by different people? One collateral item,
   three facilities, an over-allocation invariant, and an allocation policy whose
   choice is worth R56 000 a year.

9. **How is a cascade bounded** so that it is both terminating and complete? Four
   bounds are declared in §5.12.2; is boundedness a property of the construct or
   a discipline applied to it, and how is it proven rather than hoped for?

10. **How does a decision about one subject produce decisions about others**, and
    what is the record of that? A cascade produces up to 250 consequences from
    one trigger, and each must be explainable to a different client.

11. **What identifies a group**, given that a group is a derived set whose
    composition changes? Is a composition-dated set a first-class thing, and if
    not, where does the date live?

**Reuse — the project's reason for existing**

12. **At what point does reuse cost more than rebuilding, and what evidence would
    show that line had been crossed?** §5.17.7 proposes six indicators. Are they
    the right six, are they collectable, and does the project actually collect
    them?

13. **How does a consumer use a published value in a role** — a director's age, a
    surety's income, a sole proprietor's drawings — without renaming it at every
    use site? Doc 01 §5.1's 79 passthrough steps are the failure mode, and here
    the multiplier is 40 entities.

14. **What does a consumer do when a component produces 90% of what it needs?**
    Extend, compose, parameterise or fork. The choice must be declared per gap;
    is there a mechanism that makes an undeclared gap visible?

15. **How does a component produce different correct answers for different
    consumers** — three roll-ups over `core.adverse_events`, four null
    situations, materiality that varies by criticality class — while the
    difference remains attributable to a declared parameterisation rather than to
    a code path?

16. **How is a major version of a consumed component selected per assessment**
    rather than per deployment, so that a new origination, an in-flight review
    cohort and a 2027 replay can run on three different versions of project 02
    simultaneously?

17. **Is a partial call the same implementation as a whole call**, and how is the
    equivalence proven rather than asserted? 9 400 singleton event
    classifications and 48 000 entity re-scores a night, against components sized
    for 900 applications a day.

18. **What is the unit of ownership** when one artefact serves origination,
    portfolio management, covenant operations, early warning, provisioning and
    recoveries? Fourteen teams, 1 900 decision points, five execution profiles,
    and three owners who read and never write but can block anything.

**Organisation and scale**

19. **How is independent deployability achieved** across fourteen teams on one
    artefact, given that a batch in flight must hold its versions and an
    interactive path must not wait for a monthly window?

20. **How does one flow express five execution profiles** without becoming five
    implementations? A 2-second pre-assessment, a 4-second assessment, an 8-hour
    batch, 2.4 M small evaluations and a 240 000-subject daily pass with fan-out.

21. **Where does lifecycle vocabulary belong?** This project declares 20 names
    locally (§4.8), fifteen of them about time or state over time, and none of
    the ten earlier specs needed any of them. Is that a gap in the library, a
    second library, or evidence that lifecycle concerns are not library-shaped?

22. **How does the amendment re-open matrix stay honest?** 392 cells declaring
    which phases an amendment re-opens. It is a governance artefact that is also
    a control-flow artefact, and the failure mode is that the code and the matrix
    drift apart silently.

23. **How is the reviewable artefact layered and sliced** at 1 900 decision
    points, so that a policy analyst reads 4 pages, a team reads its own surface,
    an entry point renders as a coherent document, and the diff at re-approval is
    the thing actually read?

24. **How does someone answer "why was this graded 7 at its 2027 review" in
    2031?** Thirty minutes, no prior knowledge, forty builds of drift, and the
    person asking cannot read code. If the answer to this question is a person
    rather than an artefact, the design has failed, and every other answer in
    this list is worth less than it appears.
