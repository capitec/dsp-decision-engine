# 10 — Retail credit, end to end

> Fictional. The Bank, its products, thresholds, table dimensions and volumetrics
> are invented for this repository. Regulatory mechanisms referred to are the
> published public ones; every number attached to them is illustrative.

---

## 1. What this is

Every retail credit decision the Bank makes enters one place, and one decision
record comes out.

A client applying for a loan in a branch. A client tapping "increase my limit" in
the app. The monthly programme that moves 4.1 million card limits. The monthly
pre-approval run over 14.2 million clients. A consolidation worked through at a
desk with the client sitting opposite. An automated re-price triggered when a
competitor moves. A quotation that commits the Bank to a price without deciding
anything at all. An analyst's what-if against a decision made eighteen months
ago. All eight arrive at the same flow, traverse an overlapping subset of the
same eighteen phases, read the same 47 tables under the same 340 parameters, and
emit one record against one contract.

That flow contains approximately **1 400 distinct decision points**. It is owned
by **12 teams**, none of which owns all of it, and three of which will change it
in any given week. It serves **6 products** across **12 client segments** using
**9 scorecards**. Its tightest budget is 50 milliseconds and its loosest is a
six-hour window.

This is not an aspiration. It is a description of the thing being replaced. The
Bank's current retail credit decisioning does the whole job in one place and does
it badly: a decision point cannot be located without asking the person who wrote
it; a change cannot ship without a release train that three teams wait on; a
four-year-old tightening is still in force and nobody can say what unwinding it
would do. None of that is incidental. It is what a system of this size becomes
when composition is not designed.

### 1.1 This project is deliberately built standalone, and project 11 is not

Projects 00 through 09 are isolated: each takes one difficulty and goes deep on
it. Projects 10 and 11 are end to end. They are also a deliberate A/B, and the
contrast between them is the experiment:

- **This project is built standalone.** It is its own project in the codebase,
  rebuilt from nothing, with its own thresholds, its own tables, its own rule
  counts, its own band edges and its own budgets. The only thing it takes from
  outside is the shared library's published capabilities (project 00 §6) and the
  canonical vocabulary (project 00 §4) — and even there, the library supplies the
  capability and this project supplies the calibration. It asks: **how do you
  structure one very large project from scratch?**
- **Project 11 is built by reuse**, consuming components that already exist, by
  reference. It asks: **how easy is it to reuse components that already exist?**

So where this document covers the same ground as project 02, 03, 06 or 07 — and
it covers a great deal of it — **the duplication is intentional, not an
oversight**. The numbers here are different from theirs on purpose. Two documents
that happen to share a threshold look like one project split in two, which
defeats the comparison. A reader who has never opened another spec in this set
must be able to build this one.

What follows is therefore self-contained, and long, because a very large project
specified honestly is long.

### 1.2 What is actually hard here

Six things, none of which is visible in a flow of six phases, and all of which
are the point:

1. **Values computed once and consumed everywhere.** `net_monthly_income` is
   established in phase 6 and read by eleven later phases. Forty other values
   have more than one consumer. One of them — the client's existing obligations —
   must exist in **two contradictory versions at the same instant**, because the
   consolidation phase needs a hypothetical figure while the origination path
   needs the actual one, and both feed a record that must be unambiguous.
2. **Ordering.** Roughly forty hard ordering constraints between phases, of which
   about a third are non-obvious and two are genuine cycles that must be broken
   by declaration rather than discovered by a defect.
3. **A feedback loop spanning four phases.** Affordability fails, consolidation
   searches, affordability re-runs against hypothetical obligations, pricing
   re-runs, the solve re-runs, offer assembly runs. It may execute four times. A
   consultant must afterwards be able to say why the third pass produced nothing.
4. **One budget, eighteen phases, six external calls.** 120 milliseconds at p99
   on the busiest entry point, with a declared policy for what may be abandoned
   on overrun and what happens when the phase that overran may not be.
5. **Eight entry points.** One flow, eight ways in, eight output shapes. Entry
   point 7 runs seven of the eighteen phases. Entry point 4 skips exactly two and
   runs the other sixteen. Expressing that without eight near-copies of the flow
   is the structural problem this project poses.
6. **Twelve teams and one artefact.** Three of them change it in the same week,
   under three approval routes, on three cadences, with three different notions
   of what an acceptable release looks like. What must be true for one of them to
   deploy without waiting for the other two?

---

## 2. Why it is in this set

**This project's purpose is to find the scale wall.** Not to demonstrate that the
framework works — the isolated projects do that, each inside its own boundary —
but to establish at what size, and along which axis, composition stops working.
Every other spec asks whether a shape can be expressed. This one asks whether
four hundred expressible shapes can live in one deployed artefact without
becoming unnavigable, unshippable or unreviewable.

| Question | How this project stresses it |
|---|---|
| **Q6 codebase organisation** ●● | **The dominant question, and the reason this document exists.** 1 400 decision points, 18 phases, 47 tables, 340 parameters, 12 owning teams, 8 entry points, 6 products, 12 segments, 4 change cadences, one deployment. Five of the eighteen phases have a single owner; the other thirteen have two or more; one has five. Three teams change the flow in a typical week. What is the unit of ownership? What is the unit of deployment? Can a team own a subtree? Can a new engineer find where a number is decided? None of those questions stings at six modules, and all of them are fatal at six hundred. |
| **Q1 reuse** ● | Reuse *inside* one project rather than between projects. The affordability determination is invoked in four evidence modes; on one entry point it runs twice in one decision and up to five times when the loop fires. Pricing is invoked as a body by two phases that do not own it, up to 340 times each. Settleability classification is written for phase 14 and consumed by phase 15's decrease path. A seam in the wrong place is survivable in a small flow and is not survivable here. |
| **Q4 custom modules across flows** ● | Eight entry points run overlapping phase sets. Doing that **without eight near-copies** is the concrete test, and it is the same test the previous generation failed when 79 identity-passthrough steps appeared because a value could not be renamed (doc 01 §5.1). A structure that makes a new entry point cost a fork will get a fork, and then the tax table will be right in six of the eight. |
| **Q7 audit** ● | One decision record now spans eighteen phases, up to four loop passes, up to 250 evaluated scenarios, six external sources and an overlay stack — for eight entry points with different phase sets, at roughly 258 million records a year. Explaining a decision is no longer explaining a rule; it is explaining a traversal. And the reviewable artefact (doc 04 §6) at 1 400 decision points is that document's top risk at its worst: a 400-page rendering approved in a 40-minute agenda item is the appearance of governance, and an adjudicator will say so. |
| **Q3 parameters** ● | 340 parameters and four cadences that must coexist — statutory on gazette, policy quarterly, product weekly, overlay ad hoc. 61 parameters are read by more than one phase and 14 by more than four. A product manager must move a threshold on Thursday without a release and without being able to reach a statutory value. |
| Q2 tables ○ | 47 tables, ~283 500 cells, five artefacts holding 90% of them, one patched within two hours of a repo announcement. The table problems themselves are well covered elsewhere in the set; what is new here is that a change to any of them must have an answerable blast radius across eighteen phases *before* it ships. |
| Q5 core component set ○ | Secondary by design. What this project adds is the question of whether a **phase** — the unit of composition — is itself a component kind, and whether it is the same kind as the things inside it. |

### 2.1 The numbers, stated once

Used consistently throughout. Where a later section gives a breakdown, the
breakdown sums to the figure here.

| | Count | Broken down in |
|---|---|---|
| Phases | **18** | §5.1 |
| Distinct decision points | **1 400** | §5.1 |
| Tables read | **47** (~283 500 cells) | §6.1 |
| Parameters | **340** | §6.2 |
| Owning teams | **12** | §3.1, §5.26 |
| Entry points | **8** | §4.1 |
| Products | **6** | §4.2 |
| Client segments | **12** | §4.3 |
| Scorecards | **9** | §6.3 |
| Cap register entries | **118** (47 applicable to product 10) | §5.10 |
| Application-fraud rules | **188 live, 31 shadow** | §5.6 |
| Campaign trees | **44** (~6 800 nodes) | §5.17 |
| Hard ordering constraints between phases | **~40** (24 tabulated) | §5.22 |
| Shared intermediates with more than one consumer | **41** | §5.21 |
| External calls, worst case, one decision | **6** | §5.24 |
| Decision records emitted per year | **~258 M** | §5.31 |

**What counts as a decision point.** A named location at which the flow selects
an outcome, narrows a value or chooses a route, and which can be cited
individually in a decision record: a gate, a policy rule, a cap register entry, a
fraud rule, a routing condition, a treatment selection, a threshold comparison, a
validation assertion, a suppression, an arbitration rule.

Two exclusions, both arguable, and §13 asks whether they are right:

- **Table cells are not decision points.** The lookup that reads a cell is one.
  Counting the Drive Finance card's 71 424 cells would make the number
  meaningless and would turn a monthly Treasury refresh into 71 424 changes.
- **Campaign tree nodes are not counted in the 1 400.** The 44 live trees carry
  ~6 800 nodes and ~2 900 leaves. They are analyst-authored, structurally
  uniform, published weekly on their own cadence, and evaluated on one entry
  point. They are counted as 44 tree evaluations plus the arbitration that
  follows. If a tree node *is* a decision point on the same footing as a cap
  rule, this flow has 8 200 of them and every statement in this document about
  navigability, blast radius and dead logic becomes an order of magnitude
  harder. That is a live question, not a settled one.

Shadow fraud rules (31) are also excluded, because they provably cannot affect
the answer. They are recorded, measured and governed; they are not decision
points.

---

## 3. Actors

### 3.1 The twelve owning teams

| # | Team | What it owns | Cadence | Approval |
|---|---|---|---|---|
| T1 | **Decision Platform Engineering** | The spine: admission and routing, acquisition orchestration, the search mechanics of the solve, record emission, determinism, the latency budget, the certification gate. Owns no credit policy. | Fortnightly | Engineering |
| T2 | **Client Identity & Data Platform** | Identity resolution, the client feature mart and its freshness contract, bureau normalisation, source-derived features, the feature band definitions. | Weekly | Data owner |
| T3 | **Regulatory Compliance** | Consent interpretation, statutory eligibility grounds, the statutory rate ceiling, fee and premium caps, the in duplum test, disclosure content and wording, the 412-code reason registry, notice requirements, retention. | On gazette; registry monthly | Compliance |
| T4 | **Credit Risk Policy** | 89 of phase 9's 196 decision points, the grade boundaries, the affordability policy values, the obligation treatment matrix, the internal expense norms, the limit assignment matrix, the settleability rules, the anti-harm thresholds, and the live overlay register. | Quarterly, faster when a portfolio metric moves | Credit Committee |
| T5 | **Model Risk & Modelling** | The 9 scorecards, their calibrations, the challenger design and its traffic share, the model-derived features. | On model release; traffic share weekly | Model Validation |
| T6 | **Treasury / Pricing** | Four rate cards, the long-term loading, the reference rate, the cost-of-funds inputs. | Monthly, plus patches within 2 h of a repo announcement | Treasury committee |
| T7 | **Unsecured Lending Product** | Products 10 and 11: permitted terms, minimum viable offer rules, deduplication tolerance, the recommendation objective, 14 cap register entries. | Weekly | Product owner |
| T8 | **Cards & Revolving Product** | Products 20 and 21: limit bands, minimum payment rates, spend-cap parameters, cooling-off windows, 11 cap register entries. | Weekly | Product owner |
| T9 | **Secured Lending Product** | Products 30 and 40: loan-to-value policy, valuation tolerance, security requirements, registration and conveyancing schedules, 9 cap register entries. | Monthly | Product owner |
| T10 | **Financial Crime** | The 188 live application-fraud rules across five families, the 31 shadow rules, the exclusion lists, the device and consortium contracts, 11 of phase 9's decision points. | Weekly; emergency path in minutes | Four-eyes within Financial Crime |
| T11 | **Campaign Analytics & Marketing** | The 44 campaign trees, the suppression registry, the contact fatigue rules, the arbitration weights, the holdout design. | Weekly | Campaign forum |
| T12 | **Collections & Restructure** | The consolidation search: scenario ordering, the evaluation budget, the objective in force, the restructure variant, the concession authority bands. | Quarterly | Credit Committee |

### 3.2 Everyone else

| Actor | What they need from it |
|---|---|
| **Credit Committee** | Approves policy changes, overlay sets and the uplift authority. Asks, every month, what the unadjusted answer would have been. |
| **Asset and Liability Committee (ALCO)** | Sets the monthly additional-limit budget (currently **R2.1 bn** of applied limit), the risk-weighted-asset and expected-loss envelopes, and the ranking objective for entry point 3. |
| **Decision Governance** | Replays, explains, diffs, certifies and packs. Blocks releases. Consumes the record; produces nothing the flow reads. The estate-wide harness is specified in project 09; this flow's obligation is to emit what it needs. |
| **Contact centre consultants** (~900) | Explain a decline, a limit, a price or a missing term in under 90 seconds, from the record, without an analyst. |
| **Branch consultants** (~2 100) | Run entry point 5 with the client at the desk. Discuss the top three scenarios. Cannot alter the search or the objective. |
| **Credit analysts** (~35) | Investigate suspected defects. Answer "why was the term capped at 48 for this client" without an engineer. Run entry point 8 constantly. |
| **Model Validation** | Must see each scorecard's own output separately from the overlays applied to it, and must run the flow with the overlay stack disabled. |
| **Internal Audit** | Samples decisions from every entry point and re-derives them independently. Tests that consent preceded every limit change and that every table version was recorded. |
| **Information Officer** | Owns retention, masking and access. Every value in the record is personal information. |
| **The regulator, the ombud, a court** | Arrive years later with one decision and no patience for "the system decided". |
| **The client** | Entitled to reasons on refusal, to an accurate quotation, to substitute their own credit life policy, and to be told why the answer changed between Tuesday and Thursday. |

---

## 4. Inputs

### 4.1 The eight entry points

The structural fact that most shapes this specification, so it is stated first and
in full. One flow, eight ways in, eight output shapes, and budgets three orders of
magnitude apart.

`entry_point_code` is set at admission, is immutable for the life of the
decision, and appears on every record.

| Code | Entry point | Mode | Volume | Budget | Phases |
|---|---|---|---|---|---|
| **1** | **New credit application** | Real time | 55 000/day; peak 9/s sustained 20 min | **p99 120 ms** excluding external calls; 2 350 ms client-observed | 16 of 18 |
| **2** | **Client-initiated limit change** | Real time | 14 000/day; peak 25/s | **p99 200 ms**, p50 60 ms | 15 of 18 |
| **3** | **Limit programme evaluation** | Batch, monthly | 4.1 M accounts | **3-hour window**; allocation within 25 min of it | 14 of 18 |
| **4** | **Campaign pre-approval** | Batch, monthly | 14.2 M clients | **6-hour window**; ~657 clients/s sustained | **16 of 18** |
| **5** | **Consolidation assessment** | Interactive, in branch | 6 000/day; 850/h peak | **2.5 s p95, 4.0 s p99**, including up to 400 internal re-evaluations | 17 of 18 |
| **6** | **Re-price / retention offer** | Event-driven | 9 000/day | **p99 600 ms** | 12 of 18 |
| **7** | **Quotation only, no decision** | Real time | 22 000/day | **sub-50 ms p99** | **7 of 18** |
| **8** | **What-if simulation** | On demand | ~1 800/month | **< 5 s p95** | The phase set of the decision it derives from |

Phase identifiers `P01`–`P18` are defined in §5.1; the full matrix is §5.20.

**Entry point 1 — new credit application.** All six products. The client names a
product and usually an amount and a term; 34% of app-channel volume names
neither. Runs P01–P13, P16, P17, P18 — sixteen phases. Skips P15 (limit
assignment). **P14 is conditional**: the consolidation search runs when
affordability fails and the client is consolidation-eligible, which happens on
6.8% of applications — and when it runs, the request leaves the 120 ms budget and
enters an 820 ms one. That is a structural fact about the budget, not an
exception to it. Output: an `outcome_code`, 0 to 8 offers with one recommended, a
ranked decline reason set, or a referral queue, plus the full pre-agreement
disclosure block.

**Entry point 2 — client-initiated limit change.** Products 20 and 21. The client
names a limit, or asks what they qualify for. Runs P01–P10, P12, P15, P16, P17,
P18 — fifteen phases. Skips P11 (the product is given), P13 (there is no
amount–rate–instalment solve on a revolving facility) and P14. P15 runs its
per-account portion only; the population-level allocation must not run, because
this path is not budget-constrained. Output: a new limit, its `change_type_code`,
the consent request, and — where a decrease is indicated — the notice class and
effective date.

**Entry point 3 — limit programme evaluation.** Products 20 and 21; 4.1 M
accounts; monthly. Runs P01 (batch admission), P02 in reduced form, P03, P04 from
snapshot with zero external calls, P06, P07, P08, P09, P10, P12, P15, P16, P17,
P18 — fourteen phases. Skips P05 (there is no event to assess; the exclusion-list
checks that matter live in P03), P11, P13 and P14. Output: 4.1 M per-account rows
carrying the proposed limit, the binding cap, the affordability verdict, the
allocation rank, the funded or unfunded outcome and the notice class —
**including for the accounts excluded at P03**, because "we did consider your
account" is itself an answer and a client will eventually ask for it.

**Entry point 4 — campaign pre-approval.** 14.2 M clients, monthly, six-hour
window. **It skips exactly two phases — P02 (identity is pre-resolved in the
mart) and P05 (fraud) — and runs the other sixteen.** That makes it the most
complete traversal of the flow in the estate, which surprises people who expect a
marketing run to be shallow. It evaluates the 44 campaign trees, the applicable
subset of the 9 scorecards, the cap waterfall per candidate product, a bulk
pre-assessment for the 8.4 M clients with a credit campaign leaf, a
**batch-identification** variant of the consolidation search for the 760 000
clients flagged as candidates (36 scenarios and 55 ms each, not 250 and 820), and
the proposed-limit logic for the 3.6 M clients holding a revolving facility.
Output: 14.2 M client rows carrying campaign assignment, offer tier, pre-approved
amount and term, the tree path taken, the suppressions that applied, the
arbitration result and the holdout flag.

**Entry point 5 — consolidation assessment.** Interactive, client present. Runs
seventeen of eighteen — everything except P15. 2.5 s p95 including up to **400
internal re-evaluations**: at most 250 candidate settlement scenarios, each
triggering one to three re-evaluations of affordability, pricing and the solve,
bounded in total at 400. Output: the chosen scenario in full, two materially
distinct runners-up, the before-and-after comparison, the execution package of
per-account settlement instructions with expiries, and every rejected scenario
with its rejection reasons — because the rejected ones are what people ask about.

**Entry point 6 — re-price / retention offer.** Event-driven: a competitor rate
move, a retention trigger, a scheduled re-price on a variable-rate facility. Runs
P01, P02, P03, P06, P07, P08, P09, P12, P16, P17, P18 — eleven phases — plus P10
**only where the re-price raises the instalment** (a shortened term, a
capitalised arrears amount), which is 21% of cases and makes twelve. Skips P04
(no fresh external retrieval; the flow runs on internal state and the bureau view
already held), P05, P11, P13, P14, P15. Output: a revised rate, the revised
instalment, the notice class and the offer's validity period.

**Entry point 7 — quotation only, no decision.** The sharpest structural case in
the document. A client, a consultant or a comparison site asks *what would this
cost*. No decision is made, no client is assessed, no bureau is touched, no
credit decision record is created — and yet every number must be
**disclosure-accurate**, because a quotation is a representation the Bank can be
held to, and a quotation that understates the initiation fee is a refundable
overcharge across every agreement written from it.

It runs **7 of 18 phases**:

| Phase | What it does on entry point 7 |
|---|---|
| P01 | Validation and routing; sets `entry_point_code` 7 and `decision_date` |
| P03 | **Reduced**: product and channel availability only. No client gates, because there is no resolved client. 9 of its 62 decision points. |
| P09 | **Reduced**: regulatory-class entries only — the statutory rate ceiling, the fee caps, the credit life cap, the in duplum test, the minimum and maximum amount and term. 12 of its 196 decision points. |
| P11 | Which product's arithmetic applies |
| P12 | Pricing |
| P17 | **Reduced**: 6 of 61 assertions — the ones that do not require a client |
| P18 | Disclosure, and emission of a quotation record |

Two consequences recur throughout this document:

- **P17 runs a different assertion set on entry point 7 than on entry point 1.**
  Fourteen assertions for product 10 on entry point 1; six on entry point 7,
  because the others test a client that does not exist here. A validation phase
  whose assertion set is a function of the entry point is either one phase with a
  declared applicability per assertion, or two phases that will diverge within a
  year. This document does not choose. §13 asks.
- **Sub-50 ms with no external calls means the whole seven-phase traversal has
  less budget than entry point 1 gives to the solve alone.** The quotation path
  cannot afford to resolve a 34 560-cell rate card lazily, cannot afford to build
  a record of the shape entry point 1 builds, and cannot afford a per-request
  resolution of the overlay stack. Whatever makes entry point 1 work must be
  absent here — and absent by construction, not by a conditional inside every
  phase.

**Entry point 8 — what-if simulation.** An analyst changes one input, one
parameter or one table cell and re-runs a recorded decision with everything else
held at its recorded values. Its phase set is not fixed: it is the phase set of
the decision being intervened on, which is one of the other seven. Under 5
seconds, because an analyst does this forty times in an afternoon and a
sixty-second turnaround means they stop doing it and start guessing.
**Non-confusability is a hard requirement**: the result carries a non-production
identifier, is structurally marked in every rendering, cannot be written to the
decision store, cannot be issued to a client and cannot satisfy a regulator
request. The failure mode is an exploratory re-run exported to a spreadsheet,
emailed, and quoted back to the Bank two years later by a complainant's attorney
as what the Bank's own system says.

### 4.1.1 The structural problem, stated plainly

Entry point 7 runs 7 phases. Entry point 4 runs 16, skipping precisely fraud and
identity. Entry point 1 runs 16, a different 16. Between them the eight entry
points exercise 18 phases in eight different combinations, with different
degradation policies, different assertion sets, different record shapes, and
budgets from 50 milliseconds to six hours.

**Expressing that without eight near-copies of the flow is the problem this
project poses.** Eight copies would work on day one. Eight copies would also
guarantee that within two years the tax table is correct in six of them, that a
cap rule added in one is missing in three, and that the entry-point agreement
criterion in §10 can never be met. The previous generation's evidence is
unambiguous about which of those outcomes occurs.

### 4.2 Products

| `product_code` | Product | Term | Amount / limit | Entry point 1 volume |
|---|---|---|---|---|
| 10 | **Flex Loan** | 6–84 months | R2 000 – R500 000 | 31 900/day (58.0%) |
| 20 | **Everyday Card** | revolving | R1 000 – R300 000 | 10 450/day (19.0%) |
| 21 | **Access Facility** | revolving | R500 – R150 000 | 6 490/day (11.8%) |
| 11 | **Flex Loan Consolidation** | 12–84 months | R10 000 – R500 000 | 3 850/day (7.0%) |
| 30 | **Drive Finance** | 12–72 months | R30 000 – R1 500 000 | 2 200/day (4.0%) |
| 40 | **Home Loan Further Advance** | 12–240 months | R50 000 – R2 000 000 | 110/day (0.2%) |

Product 40 is ~2 400 decisions a month. It exercises 14 of the 18 phases, carries
28 cap register entries, 7 final-validation assertions and its own 28 800-cell
rate card. It is the single largest source of dead-logic ambiguity in §5.29: at
that volume, a decision point that fires on one application in forty is
indistinguishable, in a firing count, from one that is dead.

Open revolving book, for entry point 3: **4.1 M accounts** — 2.35 M Everyday Card
and 1.75 M Access Facility — carrying **R79.4 bn** of granted limit against
**R28.9 bn** drawn.

### 4.3 Client segments

Twelve. Segment assignment is made in P06 and consumed by P07 (scorecard
selection), P08 (calibration and grade boundaries), P09 (31 register entries are
segment-scoped), P10 (evidence tier and buffer), P11, P12 and P16.

| `segment_code` | Segment | Defining condition | Share of entry point 1 |
|---|---|---|---|
| 1 | New to bank, thin file | No internal relationship; < 3 bureau accounts ever, or < 15 months' history | 11.4% |
| 2 | New to bank, thick file | No internal relationship; ≥ 3 accounts and ≥ 15 months | 16.6% |
| 3 | Existing, transactional only | Internal tenure ≥ 4 months; no internal credit | 14.2% |
| 4 | Existing, credit-holding, clean | Internal credit; no arrears beyond 1 month in 24 | 24.8% |
| 5 | Existing, credit-holding, past arrears | Internal credit; any 2+ month arrears in 24 | 12.1% |
| 6 | Self-employed | `employment_type_code` 3 | 7.3% |
| 7 | Pensioner | `employment_type_code` 4 | 4.9% |
| 8 | Social grant | `employment_type_code` 5 | 2.6% |
| 9 | Informal income | `employment_type_code` 6 | 2.2% |
| 10 | Non-resident | Three residency classes, collapsed | 1.3% |
| 11 | Staff | Employee or immediate family | 0.8% |
| 12 | Joint household | `is_joint_application` true | 1.8% |

Segments 8–11 together are 6.9% of entry point 1 and carry 41 of the 1 400
decision points. §5.29 returns to them.

### 4.4 Sources, freshness and external calls

| Source | Read by | Entry points | Freshness | Call budget |
|---|---|---|---|---|
| Credit bureau (three bureaux, three schemas) | P04, P06 | 1, 5; 2 above R20 000 | ≤ 40 days at `decision_date` for 10/11/20/21; ≤ 25 days for 30/40 | **600 ms** |
| Device and consortium fraud intelligence | P04, P05 | 1, 2, 5, 6 | Real time | **350 ms** |
| Bank statement aggregator | P04, P06 | 1, 5 (41% of volume) | ≤ 120 days of history, retrieved live | **1 400 ms** |
| Identity verification service | P02 | 1, 5 (new-to-bank, 28%) | Real time | **450 ms** |
| Employer confirmation | P04, P06 | 1 (7% of volume) | Real time | **1 100 ms** |
| Internal account and exposure state | P02, P06, P09 | All except 7 | Real time on 1, 2, 5, 6; snapshot on 3, 4 | **60 ms** |
| Client feature mart | P06 | 3, 4 | Closes 22:00; ≤ 24 h stale | snapshot |
| Vehicle valuation guide (38 500 rows) | P12 | 1, 5 (product 30) | Monthly | **120 ms** |
| Settlement quotation service (up to 24 providers) | P14 | 5, and 1 when the loop fires | Quotation validity 7–30 days | **400 ms** per provider, 3 concurrent |
| Property valuation and deeds | P12, P17 | 1, 5 (product 40) | ≤ 6 months | **2 400 ms**, and therefore asynchronous — see §5.12 |

Worst case on entry point 1: **6 external calls**. Median 2. Entry points 3, 4
and 7 make **none**.

### 4.5 Nullability, and the fourth null

Three null situations are distinct and must never be conflated: a value **not
collected**, a value **collected as zero**, and a value that **could not be
established**. "No bureau record" and "a bureau record showing no accounts" are
different applicants.

This project adds a fourth, which no isolated flow needs: **the phase that would
have produced this value did not run on this entry point.** A missing
`fraud_verdict_code` on entry point 4 is not an absence of evidence, a zero, or a
failure — it is a structural property of entry point 4. Conflating it with the
other three makes every cross-entry-point comparison in §10 meaningless, and
makes a degraded decision indistinguishable from a normal one. It is carried in
`record_completeness_code` (§4.6).

### 4.6 Names this project declares locally

The canonical vocabulary (project 00 §4) is used unchanged. These fifteen names
do not exist in it, because no isolated flow needs them. That this project needs
fifteen of them is itself a finding about how much of composition the shared
vocabulary does not cover.

| Name | Type | Meaning |
|---|---|---|
| `entry_point_code` | int8 | 1..8. Set at admission, immutable, on every record. |
| `phase_id` | int16 | One of the 18. Declared, stable, never positional. |
| `phase_set_id` | int32 | The identity of the exact set of phases that ran, resolvable to the list. **Not** derivable from `entry_point_code` alone, because P14 is conditional on entry point 1 and P10 is conditional on entry point 6. |
| `decision_point_id` | int32 | One of the 1 400. Content-derived or explicitly declared. Never positional; renaming or retiring one is a declared, versioned event. |
| `owning_team_code` | int8 | 1..12, per §3.1. Carried on every decision point, table and parameter. |
| `value_basis_code` | int8 | Which version of a shared concept a recorded value is: 1 actual, 2 hypothetical, 3 stressed, 4 entry-point-substituted. See §5.21. |
| `scenario_ref` | int32 | Where `value_basis_code` = 2, which evaluated scenario the value belongs to. Null otherwise. |
| `loop_pass_index` | int8 | 0..4. Which pass of the cross-phase loop produced this value. See §5.23. |
| `loop_termination_code` | int8 | 1 converged, 2 no-improvement, 3 bound-reached, 4 no-candidate, 5 abandoned-on-budget. |
| `source_degradation_codes` | list[int16] | Every source that was degraded, with its degradation kind, as at the moment it was read. |
| `degraded_mode_code` | int16 | The composite mode in force — declared in advance (§5.25), never computed after the fact. |
| `record_completeness_code` | int8 | 1 complete, 2 phase-absent-by-entry-point, 3 phase-abandoned-on-budget, 4 phase-degraded, 5 emission-partial. |
| `phase_budget_overrun_codes` | list[int16] | Phases that exceeded their declared budget on this decision. |
| `blast_radius_id` | int64 | The identity of the computed blast radius under which a change shipped. See §5.28. |
| `entry_agreement_delta` | float64 | Where one client was assessed through two entry points inside the comparison window, the measured difference. See §10. |

### 4.7 Volumes

| | Value |
|---|---|
| Interactive decisions | 106 000/day across entry points 1, 2, 5, 6, 7 |
| Batch account-decisions | 4.1 M/month (entry point 3) |
| Batch client-decisions | 14.2 M/month (entry point 4) |
| Decision records per year | **~258 M** — 38.4 M interactive, 219.6 M batch |
| Distinct clients touched per month | ~14.4 M |
| Clients assessed on two or more entry points within 30 days | **~1.9 M** — the population on which §10's agreement criterion is measured |
| Offers issued per year | ~11.6 M |
| Declines per year, all entry points | ~19.4 M |
| Referrals per year | ~1.1 M |
| Formal disputes per year | ~2 400, of which ~40 escalate to an ombud or the regulator |

---
## 5. The flow

### 5.1 The eighteen phases

A **phase** is the unit this document uses to talk about the flow. It is a
requirements-level unit, not a proposed implementation unit: it names a body of
work with a determinable output, an owner, a budget and a degradation policy.
Whether a phase becomes one component or forty is exactly the question this
project asks and deliberately does not answer.

| # | Phase | Determines | Decision points | Primary owner | Entry points |
|---|---|---|---|---|---|
| P01 | Request validation and routing | The entry point, the phase set, `decision_date`, the candidate product set | 41 | T1 | all |
| P02 | Client and identity resolution | `client_id`, and whether it is trustworthy | 33 | T2 | 1, 2, 3, 5, 6 |
| P03 | Consent and hard eligibility | Whether the Bank may proceed, and on what basis | 62 | T3, T4 | all |
| P04 | Data acquisition orchestration | What is fetched, in what order, and what arrived | 47 | T1, T2 | 1–6 |
| P05 | Fraud and financial crime | `fraud_verdict_code` and its reason set | 205 | T10 | 1, 2, 5, 6 |
| P06 | Feature derivation | ~510 derived values including income, expenses and obligations | 134 | T2, T5 | all but 7 |
| P07 | Scoring | `score`, per-characteristic contributions | 79 | T5 | all but 7 |
| P08 | Calibration, grading and adjustments | `probability_of_default`, `risk_grade`, the resolved overlay stack | 46 | T5, T4 | all but 7 |
| P09 | Policy gates and the cap waterfall | Five ceilings, their chains, and outright declines | 196 | T4 +4 | all |
| P10 | Affordability | `max_affordable_instalment`, `affordability_verdict_code` | 88 | T4, T3 | all but 7; conditional on 6 |
| P11 | Product routing | Which products may carry this request | 52 | T7, T8, T9 | 1, 4, 5, 7 |
| P12 | Pricing | Rate, fees, premium, instalment, effective rate | 83 | T6, T3 | all |
| P13 | The solve | The largest affordable amount at each permitted term | 31 | T1, T7 | 1, 4, 5 |
| P14 | Consolidation search | Which existing debts to settle, and what replaces them | 63 | T12 | 5; conditional on 1; reduced on 4 |
| P15 | Limit assignment | The proposed limit, and whether it is funded | 66 | T8, T4 | 2, 3, 4 |
| P16 | Offer assembly and cross-product arbitration | What the client is actually shown, and in what order | 74 | T7, T8, T9, T11 | all but 7 |
| P17 | Final validation | Whether the Bank is willing to be bound by it | 61 | T1, T3 | all |
| P18 | Disclosure and decision record emission | The client-facing answer and the permanent record | 39 | T3, T1 | all |
| | **Total** | | **1 400** | | |

**A worked client runs through the whole document.** **Client V**: existing
client, age 41, permanent employment, 67 months' tenure with the employer, two
dependants, channel 3 (web), segment 4, requesting **R95 000 over 60 months** on
campaign 7712. Her numbers join up across every phase below.

---

### 5.2 P01 — Request validation and routing

**Determines.** `entry_point_code`, `phase_set_id`, `decision_date`, the
candidate product set, the latency class, and the degradation posture in force at
admission.

**Runs on.** All eight entry points. **41 decision points.** **Budget 1.5 ms.**

**What it does.** Accepts a request in one of eight shapes and normalises it into
one internal request. Validates structural completeness (14 checks), field
domains (11 checks), and cross-field consistency (9 checks) — a term outside the
product's range, an amount below the product minimum, a `decision_date` in the
future, a channel that does not exist, a batch cycle identifier that does not
match an open cycle. Rejects malformed requests **before** any client is
resolved and any cost is incurred, with a structured rejection that is not a
credit decline and must never be recorded as one.

Then it routes. Routing is the assignment of `phase_set_id`, and it is a function
of `entry_point_code`, the named product set, the channel and — for entry point 1
— whether the client is consolidation-eligible, which is why the phase set is not
derivable from the entry point alone.

`decision_date` is fixed here, once, and every effective-dated artefact in every
later phase resolves against it. It is never re-read and "today" appears nowhere
after this phase. On a replay, `decision_date` is the recorded one, which is the
entire mechanism by which a 2027 decision reproduces in 2034.

**Degradation.** None available. A request that cannot be validated and routed is
rejected. This phase has no external dependency by design, precisely so that it
cannot fail for a reason outside the Bank.

**What changes because it is inside this flow.** Everything, because **this phase
exists only because there are eight entry points**. A single-purpose granting
flow does not need it; the request shape is the request shape. Here it is the
first structural consequence of composition, and it carries an obligation nothing
else does: it is the only place where the set of phases that will run is decided,
and therefore the only place from which "why did phase 14 not run for this
client" is answerable without inference.

---

### 5.3 P02 — Client and identity resolution

**Determines.** `client_id`, the identity confidence, and whether the resolution
is safe to build a credit decision on.

**Runs on.** Entry points 1, 2, 3, 5, 6. **33 decision points.** **Budget 2.5 ms
in-flow** (the external identity call is outside it).

**What it does.** Resolves the presented identity — identity number, passport,
name and date of birth, contact details, device — to exactly one `client_id`, or
to none, or ambiguously. Four resolution paths, tried in order, with a confidence
attached to each:

| Path | Basis | Confidence | Share |
|---|---|---|---|
| R1 | Authenticated session against an existing relationship | 1.00 | 61% |
| R2 | Exact identity-number match on the internal master | 0.97 | 19% |
| R3 | Identity verification service confirmation | 0.92 | 14% |
| R4 | Probabilistic match on name, date of birth and contact | 0.55–0.85 | 6% |

A match below the **0.88** confidence floor does not become a resolved client. It
becomes `outcome_code` refer, queue 2, with the candidate set attached. A
probabilistic match that resolves to two clients is a **first-class outcome**,
not an error: 0.6% of entry point 1 requests hit it, usually a shared identity
number in the master or a client who exists twice after an acquisition.

Where a client is resolved, the phase also establishes the **related-party set**
— up to 12 identifiers — because group exposure is tested in P09 and the graph
query is expensive enough that it must not be repeated.

**Degradation.** Identity verification service unavailable: R3 is unavailable,
R4 becomes the fallback, the confidence floor rises to **0.94**, and the share of
requests reaching queue 2 rises from 0.6% to an expected 4.1%. Recorded, marked
degraded, and re-assessable.

**What changes because it is inside this flow.** Two things. First, on entry
points 3 and 4 identity is pre-resolved in the overnight snapshot and this phase
runs in a **reduced form** that must be *provably equivalent* to the full form —
otherwise the entry-point agreement criterion in §10 fails at the very first
phase, before any credit logic has run. Second, the related-party set it produces
is consumed five phases later by an exposure rule that has no idea it is
expensive, which is the first of forty-one shared intermediates and the first
place where a phase pays a cost for a consumer it cannot see.

---

### 5.4 P03 — Consent and hard eligibility

**Determines.** Whether the Bank may proceed at all, and the complete set of
reasons if it may not.

**Runs on.** All eight entry points, in three different forms. **62 decision
points.** **Budget 2.5 ms.**

**What it does.** Two bodies of work that are conventionally separate and are
joined here because they share a short-circuit.

**Consent (19 decision points).** Six consent classes, each with its own
timestamp, channel and expiry: bureau enquiry, data sharing, marketing contact,
automated decisioning, credit life solicitation, and third-party disclosure. The
phase establishes which downstream actions are permitted at `decision_date`. The
awkwardness is that one consent state serves six different questions: a client
who has withdrawn marketing consent must still be assessable on entry point 1,
must be excluded from entry point 4, must still receive a statutorily required
notice on entry point 3, and must still be quotable on entry point 7.

Bureau-enquiry consent is mandatory and its absence is a hard stop, not a
degradation — an enquiry without consent is an offence, and an enquiry leaves a
footprint on the client's file that cannot be withdrawn.

**Hard eligibility (43 decision points).** Age at `decision_date` against the
product's minimum (18) and maximum-at-maturity (75 for unsecured, 70 for
secured); residency and capacity to contract; product availability on the channel
and in the client's jurisdiction; deceased and estate flags; debt review status
across five states; administration order; insolvency across three states;
sanctions and internal exclusion lists; staff restriction; existing-relationship
requirements for products 11 and 40; and an in-flight-application check that
declines a duplicate request within 48 hours.

**The short-circuit tension.** These gates *should* short-circuit — there is no
point scoring a deceased applicant, and 3.1% of entry point 1 volume fails here.
But the client is entitled to the complete reason set, and a reason set truncated
at the first failure produces the well-known complaint pattern where a client
fixes one problem, re-applies, and is declined for a second problem the Bank knew
about the first time. The requirement is therefore: **evaluate every gate, record
every verdict, short-circuit only the expensive work downstream.** A gate that
was not evaluated and a gate that was evaluated and passed are different facts
and must be recorded as different facts.

*Client V: all 43 gates evaluated, all pass, 6 consent records current, automated
decisioning consent given 2024-03-11 via channel 2.*

**Degradation.** Exclusion list stale by more than 6 hours: the flow continues,
the decision is marked, and approvals above R50 000 are held for a re-check when
the list refreshes. Sanctions list unavailable: hard stop, no degraded mode, all
entry points. Debt review registry unavailable: continue, mark, and cap approvals
at R25 000.

**What changes because it is inside this flow.** The completeness requirement
gets much worse. In a single-purpose flow, the complete reason set is the set
this phase produced. Here the reason set is assembled from eighteen phases and up
to four loop passes, and a gate failure in P03 must rank correctly against an
affordability failure in P10 and a validation failure in P17 — under a registry
of 412 codes owned by a team that owns neither phase. The ranking is P18's
problem; the obligation to produce a complete, comparable, ranked contribution is
this phase's, and it is an obligation that only exists because there are
seventeen other contributors.

---

### 5.5 P04 — Data acquisition orchestration

**Determines.** Which external retrievals are issued, in what order, with what
concurrency and what budget; and what actually arrived.

**Runs on.** Entry points 1, 2, 5 and 6 in full; entry points 3 and 4 in a
reduced form that collates a snapshot and issues no external call; entry point 7
not at all. **47 decision points.** **Budget 3.0 ms of in-flow work**; the
waiting is not in the 120 ms.

**What it does.** Decides, per request, which of the nine sources in §4.4 to call.
The decision is not static:

| Retrieval | Issued when | Share of entry point 1 |
|---|---|---|
| Bureau | Always, unless a view under 40 days is already held | 78% |
| Device and consortium | Always on channels 2, 3, 5, 6 | 84% |
| Statement aggregator | Income tier 1 or 2 evidence absent, or declared income variance > 18% | 41% |
| Identity verification | Resolution path R3 or R4 | 20% |
| Employer confirmation | Tenure < 9 months, or employer on the verification list | 7% |
| Internal state | Always | 100% |
| Vehicle valuation | Product 30 | 4% |
| Property valuation | Product 40 | 0.2% |
| Settlement quotations | P14 reachable | 6.8% |

Then it sequences them. Bureau, device, identity and internal state are issued
**concurrently** once consent is established. The statement aggregator must
follow identity resolution, because the aggregator request is keyed on a resolved
client. Employer confirmation is conditional on a value the bureau returns, so it
is a second wave. Property valuation takes 2 400 ms and therefore **cannot** sit
on a 2 350 ms client-observed path: product 40 runs an asynchronous
acquisition with a conditional offer, which is a different journey and is
declared as one.

It also owns the **arrival contract**: what a late response means, what a partial
response means, what a schema-invalid response means, and which of those are
degradations and which are failures. A bureau response that arrives after its
budget is not used, even if it arrives before the decision completes, because a
decision that sometimes uses a late response and sometimes does not is not
deterministic and cannot be replayed.

**Degradation.** This phase carries the largest share of the degradation policy
in §5.25, and it is the phase where degradation is *detected* rather than merely
suffered.

**What changes because it is inside this flow.** This phase does not exist in any
single-purpose flow, where acquisition is somebody else's job upstream. Here it
is a phase because eight entry points need different retrievals and two need
none, and because the ordering between it and the fraud phase **inverts by entry
point** (§5.22, constraint O-06). It is also the phase that makes the 120 ms
budget honest or dishonest: most of its wall-clock is waiting, and excluding
waiting from a latency target is exactly how a latency target becomes
unfalsifiable. The requirement is therefore stated twice — an in-flow budget of
3.0 ms, and a separate, measured, published external budget of 2 000 ms (§5.24).

---

### 5.6 P05 — Fraud and financial crime

**Determines.** `fraud_verdict_code` — 1 proceed, 2 refer, 3 decline, 4 could not
be established — and the reason set behind it.

**Runs on.** Entry points 1, 2, 5, 6. **205 decision points.** **Budget 5.5 ms**
of in-flow rule evaluation; the external consortium call is outside it.

**What it does.** Evaluates **188 live application-fraud rules** across five
families, plus **31 shadow rules** whose firings are recorded and which provably
cannot influence the answer.

| Family | Live | Shadow | Owner | What it looks for |
|---|---|---|---|---|
| F1 Identity manipulation | 46 | 9 | T10 | Synthetic identities, identity-number reuse, manipulated dates of birth, mismatched biographical data |
| F2 Application content inconsistency | 38 | 6 | T10 | Income inconsistent with employer, employer inconsistent with sector, address velocity, contact reuse |
| F3 Device and channel | 34 | 8 | T10 | Device reputation, emulator and automation signals, session anomalies, geolocation conflicts |
| F4 Syndicate and network | 41 | 5 | T10 | Shared devices, shared beneficiaries, application bursts against one employer, referral-chain patterns |
| F5 Known fraud and watchlist | 29 | 3 | T10 | Internal fraud register, consortium hits, first-party fraud history, mule watchlists |

**Every applicable rule is evaluated and every firing is recorded**, not just the
one that decided. On a genuinely fraudulent application it is normal for seven
rules to fire across three families. The verdict is computed by a **separate
precedence ordering** over the firing set: any F5 hit of severity 4 declines; two
or more F1 firings of severity 3 refer; a weighted score over all firings above
**0.71** refers; above **0.88** declines. The precedence is data, not code, and
it is changed by Financial Crime under four-eyes approval without a release.

Seventeen further decision points handle the verdict itself: the bypass rules (a
request under R8 000 from a client with ≥ 30 months' internal tenure and no
adverse internal history bypasses the consortium call entirely — 23% of entry
point 1), the unavailability path, and the degradation posture.

*Client V: 188 rules evaluated, 2 fire — F2-0118 (contact detail changed 11 days
before application, severity 1) and F3-0042 (new device, severity 1) — weighted
score 0.19, verdict 1 proceed.*

**Degradation.** Consortium intelligence unavailable: 29 of the 188 rules become
unevaluable. They are recorded as *not evaluated*, not as *did not fire* — a
distinction that matters enormously to the monthly dead-rule measurement in
§5.29. The weighted threshold drops to **0.62** to compensate for the missing
signal, which is a declared population adjustment, not an emergent one, and every
decision made under it carries `degraded_mode_code` 21.

**What changes because it is inside this flow.** Three things.

First, the verdict is not only a gate. Six later phases read it: P09 (three cap
entries tighten on a refer verdict), P11 (product 40 is unavailable on any fraud
refer), P14 (a refer verdict suppresses the consolidation path entirely), P16
(offer channel restriction), P17 (assertion 4) and P18 (reason ranking). A phase
whose output is consumed six times cannot be treated as a pass/fail.

Second, its **ordering against P04 inverts by entry point**. On entry points 1
and 2 for low-value requests, the fraud assessment runs *before* bureau retrieval,
because there is no sense paying for a bureau enquiry on a known fraudulent
identity, and because a bureau enquiry on a synthetic identity creates a footprint
that helps the fraudster. On entry points 5 and 6 it runs *after*, because 34 of
the 188 rules consume bureau-derived velocity and address-history features. That
is not a preference. It is a cycle in the dependency graph that must be broken by
declaration, and the declaration is per entry point.

Third, it is **skipped entirely on entry points 3 and 4** — 219.6 M of the 258 M
annual records carry no fraud verdict at all. That absence must be recorded as
`record_completeness_code` 2 and must be distinguishable from verdict 4 (could
not be established), which is a degraded state, and from a null, which is a
defect.

---

### 5.7 P06 — Feature derivation

**Determines.** Approximately **510 derived values**, including the income,
expense and obligation determinations on which eleven later phases depend.

**Runs on.** All entry points except 7. **134 decision points.** **Budget
13.0 ms** — the second-largest in the flow.

**What it does.** Four bodies of work.

**(a) Bureau normalisation (38 decision points).** Three bureaux, three schemas,
one normalised view. Header and identity block (which may return more than one
subject — 0.5% of enquiries); an account list of 0..95 entries, median 8, p95 26,
p99 47; an enquiry list of 0..140; a public record list of 0..30. Produces
normalised counts, worst-status indicators, ages of adverse items, enquiry
velocity over 30/60/90/365-day windows, `bureau_as_of_date` and
`bureau_is_stale`. An adverse item that cannot be mapped is a **defect**, not a
silent drop: the data-quality verdict has four values and an unmappable adverse
item forces the worst of them.

No-hit rate is 5.8%. Thin file — fewer than three accounts ever opened, or under
15 months of history — is a further 12.7%.

**(b) Income, deductions and expenses (34 decision points).** Consumes
`core.income`, `core.deductions` and `core.expense_norms`, calibrated as follows.

Seven evidence tiers with their own haircuts:

| Tier | Evidence | Haircut | Share (entry point 1) |
|---|---|---|---|
| 1 | Employer-confirmed, current | 0% | 7% |
| 2 | Payslip, ≤ 45 days | 2% | 29% |
| 3 | Statement-derived salary credit, ≥ 3 consecutive | 6% | 38% |
| 4 | Statement-derived, irregular | 14% | 9% |
| 5 | Internal deposit history (batch entry points) | 11% | — |
| 6 | Bureau-estimated income | 22% | 4% |
| 7 | Declared only | 35% | 13% |

Haircut modifiers: +4 percentage points where `income_variability_ratio` > 0.28;
+3 where tenure < 9 months; −2 where the client is segment 11. Tiers 6 and 7 are
below the minimum evidence tier for products 30 and 40, which is a decline, and
below the minimum for amounts above R80 000 on products 10 and 11, which is a cap.

Statutory deductions resolve from an effective-dated table of **9 brackets × 4
rebate classes**, selected by `decision_date`. Living expenses are the **higher**
of declared, statement-derived, and the norm floor from a **15 income bands × 7
dependant counts × 2 components** statutory table, with a stricter internal
variant per product (× 6 = 1 260 cells). Which basis bound is recorded.

**(c) Obligations (31 decision points).** Converts a ragged list of 0..95
existing accounts — internal and bureau, de-duplicated on a four-key match — into
`existing_obligations` plus a per-account annotation. The treatment matrix is
**52 account types × 7 attributes**: use the stated instalment, impute at 3.5% of
limit, impute at 5.0% of balance, impute at the minimum payment rate, exclude,
exclude only where a settlement is in flight, or treat as a contingent liability
at 50%.

Both shapes of the answer are required output. The scalar feeds nine phases; the
per-account annotation feeds P14, and P14 cannot function without it.

**(d) Derived risk and behavioural features (31 decision points).** The ~380
remaining features: utilisation bands, arrears profiles, payment behaviour over
1/3/6/12-month windows, exposure ratios, product holding flags, tenure bands,
channel history, campaign response history, and the **36 feature band
definitions** that turn continuous values into the codes the scorecards, the cap
register and the campaign trees all key off.

*Client V: bureau hit, 11 accounts, no adverse items, enquiry velocity 4 in 90
days. Income tier 2, payslip 19 days old, `gross_monthly_income` R42 900.00 after
a 2% haircut; statutory deductions R11 500.00; `net_monthly_income` **R31
400.00**. Declared living expenses R9 850, norm floor R11 280 for two dependants
at that income band — **the floor binds**, `living_expenses` R11 280.00,
`expense_basis_code` 2. Nine of her eleven accounts carry an obligation;
`existing_obligations` **R7 640.00**, of which R2 180.00 internal.*

**Degradation.** Bureau unavailable: (a) produces a null view with a declared
absence marker, 31 of the ~510 features become unavailable, and four of the nine
scorecards become unusable. Statement aggregator unavailable: tier 3 and 4
evidence is unreachable and income falls to the best remaining tier, usually 7,
with its 35% haircut — which is a real and correct penalty, not a workaround.
Feature mart stale beyond 36 hours on entry points 3 and 4: the cycle does not
run. There is no degraded mode for a stale mart, because a limit programme run on
month-old behaviour is worse than no limit programme.

**What changes because it is inside this flow.** This is where composition bites
hardest, and it bites in two places.

**First, the income and obligation determination has been pulled *forward* out of
affordability.** A standalone affordability assessment is one chain: income →
deductions → expenses → obligations → capacity → verdict. Here, the first four
links produce values that **eleven later phases** need, and three of those phases
— scoring, grading and the cap waterfall — run *before* the affordability verdict.
So the chain is split across two phases: P06 establishes income, deductions,
expenses and obligations; P10 establishes capacity and verdict.

That split is forced by composition and it is dangerous. The two halves must
share one tax table resolution, one norm version, one treatment matrix version
and one `decision_date`, or they are a fork with a shared name. The requirement
is stated positively: **the determination is performed once per decision and
every consumer is required to consume that result.** An implementation that
re-derives income inside P10 is wrong even when it happens to agree, and an
implementation that re-derives it and disagrees by a cent has produced an
assessment nobody can defend. §13 asks whether a phase is the right unit at all
when a capability's internal stages have different downstream consumers.

**Second, the same named feature is computed from different evidence on different
entry points.** `income_band_code` derived from a payslip on entry point 1 and
from internal deposit history on entry points 3 and 4 is the same name, the same
band table, and a different number. The §10 entry-point agreement criterion lives
or dies here, and `value_basis_code` 4 (entry-point-substituted) exists so that
the difference is visible in the record rather than inferred from the entry point
by whoever is reading it.

---

### 5.8 P07 — Scoring

**Determines.** `score` and the signed per-characteristic contributions.

**Runs on.** All entry points except 7. **79 decision points.** **Budget 5.0 ms.**

**What it does.** Selects and evaluates one of **nine scorecards** — and, for 12%
of entry point 1 traffic, a second one in parallel as a challenger.

| Id | Scorecard | Characteristics × bins | Applies to |
|---|---|---|---|
| SC-A1 | Application, new-to-bank thin file | 38 × 9 | Segment 1 |
| SC-A2 | Application, new-to-bank thick file | 52 × 9 | Segment 2 |
| SC-A3 | Application, existing client | 61 × 9 | Segments 3, 4, 5, 11, 12 |
| SC-A4 | Application, self-employed / variable | 44 × 9 | Segments 6, 9 |
| SC-A5 | Challenger to SC-A3 | 64 × 9 | 12% of SC-A3 traffic |
| SC-B1 | Behavioural, Everyday Card | 47 × 8 | Entry points 2, 3, 4 (product 20) |
| SC-B2 | Behavioural, Access Facility | 41 × 8 | Entry points 2, 3, 4 (product 21) |
| SC-S1 | Secured origination | 55 × 9 | Products 30, 40 |
| SC-P1 | Campaign propensity | 36 × 7 | Entry point 4 |

Roughly **3 780 scoring rows** in total. Segments 7, 8 and 10 have no scorecard
of their own and are scored on SC-A4 with a declared segment adjustment, which is
a policy decision recorded as such.

Selection is a function of segment × product × entry point, and it has 28 rules,
not a lookup, because the precedence between them is not uniform — a segment-12
joint application on product 30 uses SC-S1 on the primary applicant and SC-A3 on
the secondary, and combines them by the worse of the two grades.

**Per-characteristic contributions are required output, not a diagnostic.**
Adverse-action explanation depends on them: where a decline is score-driven, the
reasons communicated to the client are the largest negative contributions,
phrased in the registry's client-facing wording. Nulls are a scoring bin, not an
error, and the three null situations of §4.5 map to three different bins for 14
of SC-A3's 61 characteristics.

*Client V: segment 4 → SC-A3, 61 characteristics, 3 in the null bin, `score`
**648**. Challenger SC-A5 also ran (she is in the 12%): score 641. The challenger's
result is recorded and does not affect her decision.*

**Degradation.** Bureau unavailable: SC-A2, SC-A3, SC-A5 and SC-S1 are unusable
because more than a third of their characteristics are bureau-derived. Segments
2–5 fall back to SC-A1 with a declared **−45 point** shift, which is deliberately
punitive, and the resulting approvals are capped at R25 000 and marked for
re-assessment. Segment 1 is unaffected, which means a degraded bureau makes the
Bank's *thinnest* files its most normal ones — an inversion worth knowing about
before it happens.

**What changes because it is inside this flow.** Nine scorecards, not one, and up
to two running per decision, means the record cost is up to 125
characteristic-contribution rows per decision rather than 45. At 258 M decisions
a year that is a material part of the storage question in §5.31. It also means
the scorecard replacement scenario in §11 is not a model release: a new SC-A3
changes the features P06 must derive, the calibration in P08, the grade
boundaries, the 31 segment-scoped entries in P09 that key off grade, and the
propensity model's covariates in P16 — five phases, four owners, one model.

---

### 5.9 P08 — Calibration, grading and adjustments

**Determines.** `probability_of_default`, `risk_grade`, and the resolved overlay
stack for the whole decision.

**Runs on.** All entry points except 7. **46 decision points.** **Budget 3.0 ms.**

**What it does.** Three things, in a fixed order.

**(a) Calibration.** Maps `score` to a calibrated 12-month
`probability_of_default` per scorecard and calibration segment, with a factor, an
offset and an anchor. Six distinct calibrations serve the nine scorecards. The
relationship must be **invertible** — appetite logic asks "what score would we
need for this to be worth doing at this price?" — and the two directions must
agree to 1 × 10⁻⁶.

**(b) Adjustments.** Resolves the overlay stack in force at `decision_date` and
applies the overlays that act on the model's own answers: score shifts, scaling
changes, odds multipliers and calibration re-anchors. Every adjusted value keeps
its unadjusted counterpart — `score_unadjusted`,
`probability_of_default_unadjusted` — because "what would we have done without the
overlay?" is asked at every Credit Committee.

**(c) Grading.** Assigns `risk_grade` 1..12 from **adjusted** PD against a table
of 12 boundaries per product per segment (6 × 12 × 12 = 864 values). The
boundaries themselves may be moved by a grade-boundary overlay, which is a
different overlay kind applied at a different point.

*Client V: score 648 → PD 0.0294 unadjusted. Overlay ADJ-0061 (channel 3 odds
multiplier × 1.18, approved CC-2027-04, expires 2027-09-30) → PD 0.0347. Grade
boundaries for product 10 segment 4 put 0.0347 in **grade 6**; unadjusted 0.0294
would have been grade 5. The one-notch difference is entirely policy and is
recorded as such.*

**Degradation.** The adjustment register unreachable is a **hard stop**, not a
degradation, and this is worth being explicit about. Running without the register
does not mean "run unadjusted" — it means the flow does not know whether an
overlay applies, which is different from knowing that none does. A flow that
silently produces unadjusted answers when the register is down produces answers
that look completely normal and are systematically wrong in one direction for as
long as the outage lasts. The only safe behaviours are to use the last
successfully resolved register version, pinned and recorded, or to stop. This
flow uses the last resolved version for up to **90 minutes**, marks every
decision with `degraded_mode_code` 41, and stops after that.

**What changes because it is inside this flow.** This phase resolves the overlay
stack **once per decision, for the whole decision** — and five later phases
consume overlays that this phase does not apply: cap reductions in P09, buffer
adjustments in P10, rate add-ons in P12, matrix dials in P15 and campaign volume
dials in P16. So P08 carries an obligation it does not own and does not benefit
from. The reason it must is ordering constraint O-21: if two phases resolved the
register independently, and the register changed between them, one decision would
contain two different overlay stacks and its record would be indefensible.

---

### 5.10 P09 — Policy gates and the cap waterfall

**Determines.** Five ceilings, their complete chains, and any outright decline.

**Runs on.** All eight entry points, in three forms. **196 decision points** —
the largest phase in the flow after fraud. **Budget 7.5 ms.**

**What it does.** Narrows five quantities from their product seeds:

| Ceiling | Seed | Direction |
|---|---|---|
| `amount_cap` | Product maximum | Reduce only, except CAP-0118 |
| `term_cap` | Product maximum | Reduce only |
| `limit_cap` (products 20, 21) | Product maximum | Reduce only |
| `worst_acceptable_grade` | 12 | Tighten only |
| `instalment_cap` | Affordability, once P10 has run | Reduce only |

Composed of four bodies of work:

| Body | Decision points | What it does |
|---|---|---|
| The cap register | **118** | Ordered rules, each narrowing one ceiling, declining outright, or doing nothing |
| Policy gates | 42 | Non-narrowing policy tests that decline or refer — purpose exclusions, product-pair restrictions, campaign validity, cooling-off windows |
| Exposure and concentration | 24 | Group exposure headroom, employer concentration, sector concentration, single-name limits, in-flight application aggregation |
| Regulatory seeds and ceilings | 12 | Statutory maxima, the rate ceiling, the in duplum test, minimum amounts, the 7 disclosure-bearing limits. **These twelve are the only ones entry point 7 runs.** |

**The register.** 118 entries. 26 are product-agnostic; 92 are product-specific,
distributed 21 / 17 / 14 / 12 / 16 / 12 across products 10, 11, 20, 21, 30, 40 —
so the applicable register is 47 entries for product 10 and 38 for product 21.
**Product-specific means applicability, not ownership.** CAP-0104 applies only
to a product's grade slice and is owned by Credit Risk Policy; of the 118 entries
the product teams own just **34** — T7 fourteen, T8 eleven, T9 nine. Ownership
of the phase's 196 decision points is split five ways: Credit Risk Policy 89,
the three product teams 41, Compliance 34, Decision Platform 21, Financial Crime
11.

**The register is reordered and added to quarterly, by four of those five
owners.** Reordering is not cosmetic: a rule that reduces to a percentage of the
current value gives a different answer depending on what ran before it.

Representative entries:

| `rule_id` | Owner | Class | Acts on | Condition and effect |
|---|---|---|---|---|
| CAP-0011 | T7 | product | amount | Product maximum R500 000 |
| CAP-0029 | T8 | product | — | Channel availability; declines on a closed channel |
| CAP-0104 | T4 | appetite | amount | Amount by grade: 1 → R500 000, 6 → R220 000, 9 → R70 000, 11 → R28 000, 12 → decline |
| CAP-0122 | T4 | appetite | term | Term by grade: 1–3 → 84, 4–6 → 72, 7–8 → 60, 9–10 → 48, 11 → 24 |
| CAP-0131 | T7 | policy | amount | Employer tenure: < 24 months → R85 000; < 9 months → R30 000; < 4 months → decline |
| CAP-0158 | T7 | policy | amount | First advance to a new-to-bank client → R70 000 |
| CAP-0176 | T4 | policy | amount, grade | Arrears history: any 3+ month arrears in 12 → R35 000 and tighten to grade 9; any 2-month in 24 → R140 000 and tighten to grade 9 |
| CAP-0212 | T4 | policy | amount | Enquiry velocity: ≥ 7 enquiries in 90 days → R40 000; ≥ 12 → decline |
| CAP-0248 | T7 | policy | amount, term | Channel: broker → R180 000 and 60 months; partner → R120 000 |
| CAP-0266 | T10 | exposure | amount | Employer or sector on the concentration watchlist (3 150 identifiers) → R60 000 |
| CAP-0301 | T1 | exposure | amount | Group exposure headroom: group limit less existing internal exposure |
| CAP-0118 | Credit Committee | campaign | amount | **The only entry permitted to raise a ceiling** |

**The uplift entry.** CAP-0118 may raise `amount_cap` by up to **20%**, subject
to an absolute authority ceiling of **R200 000**, only where a valid `campaign_id`
identifies a Credit-Committee-authorised pre-approved campaign, and never above a
ceiling set by any entry of class `regulatory`. It cannot raise `term_cap` or
relax `worst_acceptable_grade`. Where the uplift is *restrained* — authorised to
reach a value it was not permitted to reach — both the authorised value and the
restraining entry are recorded.

**The attribution requirement**, which is the hard part and is a requirement, not
a nicety. For each ceiling, after the register has run, four things must be
answerable:

1. The final value.
2. **Which entry set it** — exactly one answer, unambiguous, including where two
   entries reduced to the same value, in which case the earlier is the binder and
   the later is recorded as *coincident*.
3. The full chain: every successive value, with the entry that produced it, in
   order.
4. Which entries were **evaluated and did not bind**, and which were **not
   applicable**, and why. A policy owner asking "is my rule doing anything?"
   needs both, and they are different facts.

*Client V's `amount_cap` chain:*

| Seq | Entry | Applicable | Before | After | Status |
|---|---|---|---|---|---|
| 1 | CAP-0011 | yes | — | R500 000 | seed |
| 4 | CAP-0104 | yes (grade 6) | R500 000 | R220 000 | **bound** |
| 11 | CAP-0131 | yes (67 months) | R220 000 | R220 000 | evaluated, did not bind |
| 14 | CAP-0158 | no (existing client) | R220 000 | R220 000 | not applicable |
| 19 | CAP-0176 | yes (2-month arrears, 19 months ago) | R220 000 | R140 000 | **bound** |
| 23 | CAP-0212 | yes (4 enquiries in 90 days) | R140 000 | R140 000 | evaluated, did not bind |
| 31 | CAP-0248 | yes (channel 3) | R140 000 | R140 000 | evaluated, did not bind |
| 38 | CAP-0266 | no (employer not listed) | R140 000 | R140 000 | not applicable |
| 44 | CAP-0301 | yes (group limit R260 000, exposure R148 000) | R140 000 | R112 000 | **bound** |
| 47 | CAP-0118 | yes (campaign 7712, authority CC-2027-09) | R112 000 | R134 400 | **raised** |
| — | ADJ-0087 | yes (cap overlay × 0.90, channel 3) | R134 400 | **R120 750** | **overlay** |

Final `amount_cap` **R120 750**, last moved by an overlay, rounded to the R250
grid. `term_cap` 72, bound by CAP-0122. `worst_acceptable_grade` 9, bound by
CAP-0176; grade 6 passes.

**Cap overlays enter the chain as their own rows**, attributed to the adjustment
set rather than to a register entry, because "your cap was R134 400, reduced to
R120 750 by a policy overlay approved under CC-2027-11, expiring 2027-12-31" is a
different answer to a client and to a regulator than "a rule bound it". A cap
overlay may only **reduce**, so it can never do what CAP-0118's authority-bounded
uplift does.

**Degradation.** Exposure state stale: CAP-0301 runs on the stale figure, the
decision is marked, and approvals that CAP-0301 would have bound are held.
Concentration watchlist unavailable: CAP-0266 is recorded as *not evaluated* and
a blanket R60 000 cap applies to every application, which is deliberately blunt.

**What changes because it is inside this flow.** Three things.

**It runs per candidate product.** On entry point 1 with three eligible products,
the register runs three times with three applicable subsets, producing three sets
of five ceilings, which P16 must then compare. A waterfall that produces one
answer becomes a waterfall that produces *n* answers that must be commensurable.

**`instalment_cap` creates a cycle with P10.** Affordability produces the
instalment ceiling, but two register entries reduce it further (a
self-employment haircut and a joint-application household test). So the register
runs in two passes: entries acting on amount, term and grade before P10; entries
acting on the instalment after it. That split is invisible in the register — the
entries do not know which pass they are in — and it is the second of the two
genuine cycles in §5.22.

**Reordering is now a cross-product event.** In a single-product flow, reordering
the register is one team's quarterly change. Here, moving CAP-0176 above CAP-0104
changes the answer for six products, four owners and eight entry points, and the
blast radius must be answerable before it ships (§5.28).

---

### 5.11 P10 — Affordability

**Determines.** `discretionary_income`, `max_affordable_instalment`,
`affordability_verdict_code`.

**Runs on.** All entry points except 7; conditional on 6. **88 decision points.**
**Budget 8.5 ms**, and **1.2 ms per re-evaluation** inside P14's search.

**What it does.** Takes the income, expense and obligation determinations P06
made and converts them into capacity, in four parts and four evidence modes.

**The chain.**

```
discretionary_income = net_monthly_income − living_expenses − existing_obligations
capacity             = discretionary_income − residual_floor(dependants)
ratio_ceiling        = instalment_to_income_ratio(grade, product) × net_monthly_income
pre_buffer           = min(capacity, ratio_ceiling)
max_affordable_instalment = pre_buffer × (1 − buffer(grade, product, channel))
```

Residual floor by dependants: R1 450, R1 950, R2 400, R2 850, R3 250, R3 600,
R3 900 (0..6+). Instalment-to-income ratio ceilings run from 32% at grade 1 to
14% at grade 11 for product 10, in a 12 × 6 table. The buffer grid is 12 grades ×
6 products × 4 channel classes = 288 values, running 6% to 24%.

**The verdict.** Four values with four consequences:

| Verdict | Condition | Consequence |
|---|---|---|
| 1 pass | `max_affordable_instalment` ≥ the product's minimum viable instalment (R185) | Continue |
| 2 marginal | Within 8% of the minimum, or evidence tier 6–7 | Continue, force `outcome_code` "approve with conditions", restrict to terms ≤ 36 |
| 3 fail | Below the minimum | Decline, reason 2140 — **unless the consolidation loop is reachable, in which case P14 runs** |
| 4 indeterminate | Evidence insufficient for the product's minimum tier | Refer, queue 5 |

**Four evidence modes**, selected by `assessment_mode_code`, sharing one
arithmetic:

| | Origination (1, 5) | Limit (2, 3) | Campaign (4) | Scenario (inside P14) |
|---|---|---|---|---|
| Income evidence | Tiers 1–4 | Tier 5 dominant; no fresh payslip | Tier 5 only | Inherited, computed once |
| Stale income | → indeterminate | → conditional offer, not a failure | → the offer is indicative, not binding | n/a |
| Expenses | Declared vs statement vs norm floor | As origination, often months old | Norm floor dominant | Inherited |
| Obligations | Full actual list | Full list, the facility at its current limit | Full list from the mart | **Hypothetical** — settled accounts removed, new facility added |
| Buffer | Standard grid | Standard grid | Standard grid + 3 pp | Inherited from origination |

**A mode may not change the arithmetic.** It selects evidence rules, parameter
sets and which outputs are produced. A mode requiring different arithmetic is a
different calculation and must be justified as one. The failure this guards
against is four near-identical assessments diverging over two years, with the tax
table correct in two of them.

*Client V: discretionary income R31 400.00 − R11 280.00 − R7 640.00 =
**R12 480.00**. Residual floor (2 dependants) R2 400.00 → capacity R10 080.00.
Ratio ceiling at grade 6 product 10: 22% × R31 400.00 = R6 908.00 — **binds**.
Buffer at grade 6, product 10, channel 3: 14%, raised to 18% by overlay ADJ-0114
(+4 pp on channel 3 for the quarter, approved CC-2027-07). `max_affordable_
instalment` = R6 908.00 × 0.82 = **R5 664.56**; unadjusted **R5 940.88**. Verdict
1, pass.*

**Degradation.** **There is no degraded affordability mode, and there deliberately
is not one.** A guessed affordability answer is worse than no answer: it is the
basis of a reckless-lending defence and a number the Bank will be held to. What
exists instead is *degraded evidence*, which is a different thing with a different
name and a recorded evidence tier. Where evidence falls below the product's
minimum tier the verdict is indeterminate and the request refers. It never
approves on an assumption.

**What changes because it is inside this flow.** Four things, and this is the
phase where composition costs the most.

1. **It runs up to five times in one decision.** Once product-neutrally before
   P11 (ordering constraint O-09), once per routed product after it, and up to
   three more times inside the loop of §5.23. Every run has a
   `loop_pass_index`, and the record must say which run supported the offer.
2. **It is invoked up to 400 times inside P14's search**, at 1.2 ms per
   invocation, which makes its per-call cost a design constraint for a phase it
   does not own and cannot see.
3. **The distinction between "no degraded mode" and "degraded evidence mode" must
   survive twelve teams.** They are one word apart and they are opposite
   statements, and the first person to conflate them will build a fallback that
   approves on assumed income.
4. **Its obligations input has up to 251 simultaneous live versions** — one
   actual and up to 250 hypothetical — which is §5.21's central problem.

---

### 5.12 P11 — Product routing

**Determines.** Which of the six products may carry this request, and in what
preference order.

**Runs on.** Entry points 1, 4, 5, 7. **52 decision points.** **Budget 2.0 ms.**

**What it does.** Three things.

**(a) Eligibility per product (31 decision points).** Amount and term within the
product's range; minimum evidence tier met; security available and valued for 30
and 40; existing-relationship requirement for 11 and 40; channel permits the
product; grade at or better than the product's floor (product 40 requires grade
≤ 7, product 30 grade ≤ 9); client does not already hold the maximum permitted
number of the product family.

**(b) Substitution (13 decision points).** Where the requested product is
ineligible but another can carry the need. A R40 000 Flex Loan request from a
client who fails the tenure cap may be carried as an R40 000 Access Facility
draw-down at a different price; a consolidation need arriving as a Flex Loan
request routes to product 11. Substitution is a **policy decision with a
disclosure consequence** — the client asked for one thing and is being offered
another — and every substitution is recorded with the rule that made it.

**(c) Preference (8 decision points).** Where several products qualify, the
preference order for P16. Owned jointly by the three product teams, which is
exactly as comfortable as it sounds.

*Client V: products 10 and 20 eligible, 11 ineligible (no settleable debt), 21
eligible but dominated, 30 and 40 ineligible (no security). Preference 10, 20.*

**Degradation.** Vehicle valuation guide unavailable: product 30 is withdrawn
from routing and the withdrawal is recorded — the assessment does not fail.
Property valuation unavailable: product 40 withdrawn on entry point 1, and on
entry point 5 the scenario is evaluated on a conditional basis.

**What changes because it is inside this flow.** It is the phase most entangled
in the ordering cycle of §5.22 (O-09), and the entanglement is genuine rather
than an artefact of how the phases were drawn:

> Product eligibility is keyed on the **amount**. The amount comes from the
> solve. The solve is bounded by `max_affordable_instalment`. That figure is
> computed with a buffer keyed on grade **and product**. So the product must be
> known to compute affordability, and affordability must be known to compute the
> amount, and the amount must be known to determine the product.

The declared break: P10 runs first with a **product-neutral buffer**, defined as
the most conservative buffer across the candidate product set; P11 routes on the
resulting provisional amount; P10 then re-runs with the routed product's own
buffer, which can only be the same or more generous. If the re-run changes the
routing decision, the routing stands and the discrepancy is recorded as
`routing_provisional_delta` — because a second routing pass opens an unbounded
loop and this project has enough of those. That is a declared, arguable
compromise and it is exactly the kind of thing that must be visible in the
reviewable artefact rather than buried.

---

### 5.13 P12 — Pricing

**Determines.** The rate, the fees, the premium, the instalment, the total cost
and the effective annual rate for a given amount, term and product.

**Runs on.** All eight entry points. **83 decision points.** **Budget 2.5 ms
standalone**; called as a body by P13 and P14 up to 400 times.

**What it does.** Four parts in a fixed order, because each depends on the last.

**(a) The rate.** Four rate cards, three representations, one phase:

| Card | Shape | Cells | Representation |
|---|---|---|---|
| Flex Loan (10, 11) | 72 amount bands × 40 term columns × 12 grades | **34 560** | Absolute rate |
| Drive Finance (30) | 48 amount bands × 31 terms × 12 grades × 4 LTV bands | **71 424** | Absolute rate |
| Home Loan Further Advance (40) | 30 amount bands × 16 term bands × 12 grades × 5 LTV bands | **28 800** | **Margin over the reference rate** |
| Revolving (20, 21) | 2 products × 16 limit bands × 12 grades | **384** | **Promotional rate plus reversion rate** |

Three representations in one phase is a deliberate Treasury choice, not a
technical accident, and it must remain visible as one. A margin-over-reference
card re-prices every account when the reference moves; an absolute card does not.

Flex Loan amount bands are not uniform, because pricing granularity matters most
at small amounts: R2 000–R9 999 in R500 steps (16 bands), R10 000–R59 999 in
R2 500 (20), R60 000–R199 999 in R10 000 (14), R200 000–R379 999 in R15 000 (12),
R380 000–R500 000 in R12 000 (10). Term columns cover 6 to 45 months; terms 46 to
84 are priced from the 45-month column plus a **long-term loading** held in a
separate 3 × 12 table. Rates in force run from **12.45%** (grade 1, top band, 45
months) to **26.90%** (grade 12, R2 000–R2 499, 6 months).

**The statutory ceiling.** The maximum rate for unsecured credit is a margin over
the repo rate. At a repo rate of 7.25% the ceiling is **28.25%**, so the worst
cells sit 135 basis points inside it. **A repo cut lowers the ceiling and can put
cells out of compliance without anybody touching the card**: a 50 basis point cut
moves the ceiling to 27.25% and immediately puts roughly 740 cells in breach. The
ceiling is re-evaluated against every card on every repo move and every card
version, and no offer may be priced from a cell above the ceiling in force at
`decision_date`. The flow fails loudly rather than lending at an unlawful rate.

**(b) Fees.** The initiation fee is a piecewise function of the advance:

> **R210 plus 9.50% of the advance in excess of R1 200, capped at R1 480**, both
> excluding indirect tax at 15%.

The cap binds at an advance of **R14 568** exactly. Below it the fee rises with
the amount; above it the fee is flat. That kink sits *inside* the R12 500–R14 999
rate band, so within one rate band the instalment's dependence on the amount
changes shape. The fee is **capitalised**: the client receives `offered_amount`
and finances `offered_amount + fee including tax`. Those are different numbers
and the disclosure names both.

`monthly_service_fee` is **R76.50** excluding tax, **R87.98** including,
statutorily capped. Both are effective-dated and adjusted annually.

**(c) Credit life.** Charged monthly per R1 000 of the **amount financed** — which
includes the capitalised fee, so the premium depends on the fee, which depends on
the amount. Read from a **16 age bands × 6 term bands × 5 employment types = 480
cell** table, with values from R1.70 to R4.25 per R1 000. R4.25 is the statutory
ceiling; a cell above it is a card defect, not a price. For a 41-year-old on
permanent employment: 6–12 months R1.80, 13–24 R2.15, 25–36 R2.55, 37–48 R2.80,
49–60 **R2.95**, 61–84 R3.30.

The client's statutory right to **substitute their own policy** zeroes the
premium, lowers the instalment, makes the affordability test run against the
lower figure, and attaches a condition that proof of cover precedes disbursement.
The record states that the test was run against the lower instalment.

**(d) The instalment and the effective rate.** An ordinary monthly annuity on the
amount financed at the nominal rate over the term, plus the service fee, plus the
premium, rounded to the cent. Total cost is the instalment times the term.
`effective_annual_rate` is the annualised internal rate of return on the amount
**advanced** against the full instalment stream, which is why it exceeds the
nominal rate substantially at short terms.

**(e) Rate overlays.** A rate add-on applies basis points over a declared range —
a grade range, a term range, an amount-band range, a channel — **after** the cell
lookup and **before** the annuity. It exists because reissuing a 34 560-cell card
to move one grade's pricing by 40 basis points is a Treasury release and a
validation cycle, whereas a repricing decision is frequently made in a week. Four
requirements: the statutory ceiling is re-checked **after** the add-on; the cell
value and the add-on are recorded separately, so a regulator asking whether
pricing matched the published card gets "cell 17.40% plus a 60 basis point
overlay approved under CC-2027-13" rather than an unexplained 18.00%; the add-on
participates in the solve like any other rate movement, including its effect on
band-edge behaviour; and an add-on is **never** merged into the card.

*Client V at R95 000 over 60 months, grade 6: band R90 000–R99 999, rate 15.85%;
fee capped at R1 480 → R1 702.00 with tax; financed R96 702.00; annuity
R2 343.91; credit life 96.702 × R2.95 = R285.27; service fee R87.98; **instalment
R2 717.16**; total cost of credit R163 029.60; effective annual rate 18.94%.*

**Degradation.** **No degraded mode.** A missing rate card version, a missing fee
schedule or a missing statutory ceiling stops the flow for the affected product.
There is no defensible way to price without a card, and "use last month's card"
is a contract written at a price the Bank did not publish.

**What changes because it is inside this flow.** It is **called as a body**, not
run once in sequence, by two phases that do not own it — up to 152 times by P13 and up
to 400 times by P14 — which makes its per-call cost a constraint on three
budgets, two of them owned by other teams. It also carries four cards with three
representations where a single-product flow carries one, and the three
representations mean "the rate" is not one concept: a reversion rate has no
meaning on product 10 and a margin has no meaning on product 20. That is the
clearest case in the document of a shared phase whose output *shape* varies by
product, and §13 asks what that does to the record.

---

### 5.14 P13 — The solve

**Determines.** For each permitted term, the largest amount the client can be
offered that they can afford — or the fact that no such amount exists.

**Runs on.** Entry points 1, 4, 5. **31 decision points.** **Budget 38.0 ms** —
the largest single allocation in the flow.

**The domain.** Advances round to **R250**. Candidates run from the product
minimum to `min(requested_amount, amount_cap, product maximum)`. Over the full
Flex Loan range that is 1 993 candidates; across eight permitted terms, 15 944
candidate evaluations if every one were tried. Permitted terms are **6, 12, 24,
36, 48, 60, 72, 84** — eight values, filtered by `term_cap` and by a per-segment
permitted term list.

**Why it is not division.** Five reasons, each independently sufficient:

1. **The rate is a step function of the amount.** Reducing the amount can move it
   into a worse-priced band, so the instalment goes **up** when the amount goes
   **down**. The instalment is not monotone in the amount.
2. **The initiation fee is capitalised and piecewise**, with a kink at R14 568,
   so the amount financed is not the amount advanced and the gap is itself a
   function of the amount.
3. **The credit life premium is charged on the amount financed**, inheriting both
   the fee's piecewise shape and the term band's step.
4. **Rounding to R250 interacts with band selection.** Rounding down can cross a
   band edge; rounding up can breach a ceiling.
5. **Ceilings apply to the result**, and a capped result can land in another band
   again.

**The worked failure.** Client V's twin at grade 8, 60 months, with
`max_affordable_instalment` **R1 820.00**. The card prices R60 000–R69 999 at
17.40% and the bands below R60 000 at 18.95% — Treasury rewards crossing R60 000
with 155 basis points:

| Amount | Band | Rate | Financed | Instalment | Affordable? |
|---|---|---|---|---|---|
| R60 250 | 60 000–69 999 | 17.40% | R61 952.00 | R1 823.76 | no |
| **R60 000** | 60 000–69 999 | 17.40% | R61 702.00 | **R1 816.76** | **yes** |
| R59 750 | 57 500–59 999 | 18.95% | R61 452.00 | R1 861.67 | no |
| R58 750 | 57 500–59 999 | 18.95% | R60 452.00 | R1 832.81 | no |
| R58 500 | 57 500–59 999 | 18.95% | R60 202.00 | R1 825.59 | no |
| R58 250 | 57 500–59 999 | 18.95% | R59 952.00 | R1 818.38 | yes |
| R57 500 | 57 500–59 999 | 18.95% | R59 202.00 | R1 796.73 | yes |

The feasible set is **{R2 000 … R58 250} ∪ {R60 000}**. The correct answer is
**R60 000**. A search that halves the interval on the assumption that
affordability is monotone returns R58 250 and under-lends by **R1 750**. A search
that scans downward from the requested amount and stops at the first affordable
point returns R60 000 — but only because it started above it. Neither property is
safe to assume; both must be *demonstrated*.

**Requirements on the search.** Not how it works — what must be true of it.

1. **Bounded.** A hard ceiling of **19 pricing evaluations per term**, 152 per
   application. Exceeding it is not permitted. If the ceiling is reached without
   a proven maximum, the term yields no offer, the outcome becomes refer, queue
   6, reason 2420 — an explicit, recorded, monitored outcome, not a silently
   truncated answer.
2. **Terminating** on every input, including a domain of one candidate, an empty
   domain, and a domain where nothing is affordable.
3. **Deterministic.** No randomness, no wall-clock dependence, no dependence on
   iteration order over an unordered collection, no dependence on floating-point
   accumulation order that could differ between the real-time and batch paths.
4. **Correct.** The returned amount is the **true maximum** over the R250 grid —
   the largest feasible candidate, not the largest candidate below the first
   infeasible one.
5. **Tie-broken** by a declared rule, not by evaluation order: prefer the larger
   amount; among equal amounts the lower total cost; among equal costs the
   shorter term.
6. **Attributed.** Each term records the evaluation count, the amounts evaluated,
   and the binding constraint:

   | Code | Constraint |
   |---|---|
   | BIND-AFF | The instalment ceiling |
   | BIND-CAP | `amount_cap` |
   | BIND-REQ | The requested amount |
   | BIND-MIN | The product minimum |
   | BIND-MAX | The product maximum |
   | BIND-TCR | The total cost ratio threshold (1.92) |
   | BIND-DUP | The scheduled in duplum test |
   | BIND-CEIL | No cell at or below the statutory ceiling |
   | BIND-EXH | Evaluation ceiling reached |

7. **Re-checkable.** **The offer must never fail affordability when re-checked
   from scratch.** An independent re-derivation carrying nothing from the search
   must pass. This is a mandatory acceptance test over a 180 000-application
   regression set with **zero** tolerated failures, because the failure it catches
   is silent: an offer affordable at the moment the search evaluated it and not
   affordable at the amount finally written, because a ceiling moved it into
   another band afterwards.
8. **Exhaustively verified, periodically.** For a 6 000-application sample, the
   search's answer is compared against evaluating **every** R250 candidate at
   every permitted term. Zero disagreements. This runs on **every rate card
   version**, because a new card can introduce a band-edge inversion no previous
   card had. The current Flex Loan card declares **27** such inversions.

*Client V: requested R95 000 at 60 months is below `amount_cap` R120 750 and the
instalment R2 717.16 is below `max_affordable_instalment` R5 664.56. BIND-REQ, 4
evaluations. At 72 months the maximum is also R95 000 (BIND-REQ); at 84 months the
term exceeds `term_cap` and no evaluation occurs.*

**Degradation.** None. A solve that cannot complete refers.

**What changes because it is inside this flow.** Three things, all of them
uncomfortable.

**Its budget shrank.** A standalone granting flow can give the solve 55 ms of a
120 ms budget. Here there are five phases that a standalone flow does not have —
routing, identity, orchestration, arbitration, and a second affordability pass —
and they cost 13 ms between them. The solve gets **38 ms**. The evaluation
ceiling of 19 per term is a *correctness* parameter and is not available to be
cut, so the 13 ms must come from somewhere else, and where it comes from is a
negotiation between two teams that the ownership map in §5.26 has to be able to
resolve. This is the single clearest example in the document of composition
imposing a cost on a phase that did nothing wrong.

**Its domain gained a product dimension.** On entry point 1 with three eligible
products, the solve runs three times over three cards with three ceiling sets,
and P16 must compare the results across products where the total cost is not
commensurable — a 60-month unsecured loan and a 60-month secured advance at the
same instalment are not the same offer.

**Its re-check requirement now spans the loop.** Requirement 7 says the offer must
not fail affordability when re-checked. Inside the loop of §5.23, the offer is
produced at pass 3 and re-checked at P17 against pass 4's obligations. The
re-check must therefore be against the obligations basis the offer was **priced**
on, and the record must show which. A re-check against the wrong basis passes
when it should fail, which is the worst possible failure of a safety net.

---
### 5.15 P14 — Consolidation search

**Determines.** Which of the client's existing debts to settle, what replaces
them, and whether the result leaves her better off.

**Runs on.** Entry point 5 always; entry point 1 conditionally (6.8%); entry
point 4 in a reduced batch form. **63 decision points.** **Budget 820 ms**
interactive, **55 ms** in batch.

**What it does.** A client servicing eleven accounts at a blended 24.1% for a
combined R9 340 a month, against a discretionary income of R1 120, does not have
a borrowing problem. She has a shape problem. Somewhere among the subsets of her
settleable accounts there may be a set the Bank can settle such that her
instalment falls, the new advance becomes affordable, the Bank's exposure
improves, and the total she will pay before she is debt-free does not rise so far
that she has been harmed by being helped.

**Five bodies of work.**

**(a) Settleability (14 decision points).** Per account: is it settleable at all,
by whom, at what cost, and within what window. Driven by a **240 provider × 6
attribute** matrix covering quotation availability, early settlement charges,
notice periods, whether a settlement closes the facility, and whether the limit
survives. A settled card with its limit intact is a re-accumulation waiting to
happen, so the post-settlement state, not the settlement, is what the obligation
re-derivation must see.

**(b) Settlement amount derivation (11 decision points).** Balance, accrued
interest to the settlement date, early settlement charge, notice interest, less
any unearned premium, plus a **1.8% buffer capped at R2 800** against quotation
drift. Where no quotation is obtainable, an estimate with a declared conservatism
and the outcome marked conditional.

**(c) Scenario generation (17 decision points).** For a client with 16 settleable
accounts there are 65 535 non-empty settlement sets; across four candidate
products and eight terms that is over 2 million priced combinations. **At most
250 are evaluated.** Which 250 is a business decision, expressed as an ordered
set of generation rules owned by Credit Risk Policy — settle highest-rate first,
settle highest-instalment first, settle all arrears accounts, settle everything
external, settle the client's nominated set, and eleven more — each resolving to
a **total order**, ties broken on account reference then product code then term.

Budget exhaustion is a **recorded fact, not a silent truncation**: the output
states the budget, the consumption, and whether the search ended because it ran
out of candidates or ran out of budget. A client whose search was truncated is in
a different position from one whose space was exhausted, and the contact centre
needs to know which. Because the budget is also expressed in milliseconds, a
**deterministic candidate-count bound** must exist alongside it — a time-based
cut-off alone is not reproducible.

**(d) Per-scenario evaluation (13 decision points).** Reduce the inventory,
re-derive obligations over the reduced inventory, compute the required advance
(settlements + buffer + new money + capitalised fees + product-specific costs,
which is itself a small circular solve with a declared bound of five iterations
and R1 tolerance), re-derive affordability, price it, and measure it. Each
scenario invokes P06's obligation logic, P10, P12 and P13's inner arithmetic
between seven and twelve times. At 250 scenarios that is 1 750 to 3 000
invocations inside 820 ms.

**What must not vary between scenarios**, established exactly once per
assessment: `gross_monthly_income`, the evidence tier, the haircut,
`statutory_deductions`, `net_monthly_income`, `living_expenses`,
`expense_basis_code`, `dependants_count`, and every bureau-derived characteristic
that does not depend on the settlement set. Two scenarios in one assessment that
disagree about the client's income are a defect, not a difference of opinion.

**(e) Objective and the anti-harm rule (8 decision points).** The objective is
configuration: five named objectives, blendable by weight, per channel,
effective-dated, changed by Credit Risk Policy without a release —
instalment relief, total cost, blended rate, Bank expected value, and a composite
client outcome score.

The anti-harm rule exists because lengthening a term always lowers an instalment
and almost always raises the total cost. R120 000 of card debt at 22% over 36
months costs about R4 583 a month and about R45 000 in interest; the same amount
secured against a home at 12% over 240 months costs about R1 321 a month — a 71%
reduction and an easy sale — and about R197 000 in interest, and now the house is
the security. A scenario is rejected where `total_cost_delta` exceeds **35%** of
the baseline total cost, or where the term extension exceeds **48 months** beyond
the longest settled account's remaining term, unless a documented client
instruction overrides it and the override is recorded with the consultant's
identifier.

**Degradation.** Settlement quotations unobtainable for one provider: that
provider's accounts are evaluated on an estimate and every scenario containing
them is marked conditional. Unobtainable for more than 40% of the settleable
balance: the assessment refers rather than proceeding on guesses.

**What changes because it is inside this flow.** This phase changes character
completely depending on how it was reached.

On entry point 5 it is the *purpose* of the request: the client asked for it, the
consultant is present, the budget is 820 ms and 250 scenarios. On entry point 1
it is a **consequence** — affordability failed and the flow is looking for a way
to say yes — with a budget of 60 scenarios and 180 ms, reached through the loop
in §5.23, on a client who never asked to consolidate and must be told clearly
that she is being offered something different from what she requested. On entry
point 4 it is a batch screen at 36 scenarios and 55 ms whose output is an
indicative invitation, not an offer.

**Three budgets, one implementation, and the truncation must be deterministic in
all three.** A structure in which the three budgets become three copies is a
structure in which the anti-harm rule is tightened in one of them next quarter
and not in the other two.

---

### 5.16 P15 — Limit assignment

**Determines.** The proposed limit for a revolving account, and — on entry point
3 only — whether it is funded.

**Runs on.** Entry points 2, 3, 4. **66 decision points.** **Budget 4.0 ms**
per account (outside the 120 ms path, which does not run this phase).

**What it does.** Four bodies of work.

**(a) The assignment matrix (18 decision points).** A **12 grades × 9 utilisation
bands × 5 tenure bands × 2 products = 1 080 cell** matrix, three values per cell
(increase percentage, absolute increase ceiling, minimum increase) — **3 240
values**. Authored in a spreadsheet by Credit Risk Policy, re-tuned quarterly,
and the single artefact that most determines the revolving book's shape. It must
be diffable cell by cell, validatable before it is trusted, attributable after
the fact, and capable of existing in a *candidate* state that has been simulated
but not deployed.

**(b) Caps (21 decision points).** The matrix proposes; seven caps dispose.
Income multiple (grade-keyed, 1.2× to 3.5× of `net_monthly_income`), the
affordability ceiling from P10, the product maximum, the group exposure headroom,
a single-cycle increase ceiling (R25 000), a spend-cap parameter for accounts
with volatile utilisation, and a cooling-off window (no increase within 5 months
of the last, none within 9 months of a decline).

**(c) The decrease path (17 decision points).** Opposite direction, different
triggers, different governance. Eleven risk triggers — behavioural score
deterioration beyond a threshold, arrears, adverse bureau movement, persistent
over-limit, payment-pattern break, fraud markers — each with its own target and
its own notice class. Reduction to zero requires a named authority other than on
fraud or deceased grounds. Notice periods run 0 to 30 days by change type and
jurisdiction.

**(d) Allocation (10 decision points), entry point 3 only.** The aggregate of
proposed increases lands on the balance sheet, and ALCO sets a monthly ceiling —
currently **R2.1 bn** of applied limit, with risk-weighted-asset and
expected-loss envelopes alongside it. A typical cycle identifies **664 000**
accounts that are eligible, affordable and worth increasing, totalling
**R4.6 bn**. Roughly **351 000** can be funded. Which 351 000 is a decision about
the population, not about any one account, and no amount of per-account reasoning
produces it.

**An account declined because it fell below the funding line has a decline reason
that depends on 663 999 other accounts.** Explaining that to the account holder is
§9's hardest interrogation.

**Degradation.** Behavioural score unavailable: no increase is proposed; the
decrease path still runs on non-score triggers. Exposure state stale: increases
capped at R5 000 and marked.

**What changes because it is inside this flow.** The population-level allocation
**cannot run** on entry points 2 and 4, and must not. Entry point 2 is a client
request, which is not budget-constrained: a client who asks in a month when she
ranked 412 000th gets the increase, and that is correct, not a defect. Entry
point 4 produces indicative pre-approvals, not applied limits.

So one phase produces a *funded* answer on one entry point and an *unfunded*
proposal on two others, from the same matrix, the same caps and the same
scorecard — and §10 requires that the **proposed limits** agree within tolerance
across the three while the **funded outcomes** are permitted to differ. Stating
that as a requirement inside one flow, rather than as a reconciliation between two
systems, is the part that is new here.

---

### 5.17 P16 — Offer assembly and cross-product arbitration

**Determines.** What the client is actually shown, in what order, with what
labelled as recommended.

**Runs on.** All entry points except 7 — a quotation has nothing to assemble and
nothing to arbitrate. **74 decision points.** **Budget 6.5 ms.**

**What it does.** Five bodies of work.

**(a) Candidate assembly (14 decision points).** Everything the flow produced: up
to 8 term-amount offers per eligible product from P13, up to 3 consolidation
scenarios from P14, a proposed limit from P15, a re-price from P12, and — on
entry point 4 — up to 44 campaign leaves.

**(b) Suppression (16 decision points).** Minimum viable offer thresholds (an
offer below R2 000, or with an instalment under R185, or whose effective annual
rate exceeds the 58% suppression ceiling, is not shown); contact fatigue rules
(26 rules on entry point 4); cooling-off windows; consent-derived channel
restrictions; and the do-not-target register. Every suppression is recorded with
its reason, because "why was I not offered X" is a question with a legal answer.

**(c) Deduplication (9 decision points).** Two offers whose instalments differ by
less than **2.5%** and whose amounts differ by less than R1 000 are the same
offer to a client; the shorter term survives.

**(d) Campaign trees and arbitration on entry point 4 (21 decision points).** The
**44 live campaign trees** — 3 to 10 levels deep, 15 to 260 nodes wide, ~6 800
nodes and ~2 900 leaves in total — are evaluated per client, and **the path taken
through each tree is captured as first-class output**, not as a debugging aid.
Node-level volume reporting, response-by-leaf measurement, dead-branch detection
and "why did this client get this offer, eighteen months ago" all depend on it.
Where several campaigns qualify, arbitration ranks them by a weight per campaign,
a channel capacity constraint, and a per-client offer limit of three.

**(e) Cross-product arbitration (14 decision points).** The phase that exists
only because of composition. One client can simultaneously have a Flex Loan offer
set, a card limit increase, a consolidation scenario and a retention re-price.
Something must decide what she is shown and in what order. The ranking basis is
configuration — client outcome, Bank expected value, or requested-product-first —
and it is owned jointly by three product teams and the campaign forum, which is
four owners for one number.

*Client V: two offers survive — R95 000 over 60 months at R2 717.16 (recommended)
and R95 000 over 72 months at R2 402.88. The 72-month offer is shown but not
recommended because it costs R9 977 more in total. The 48-month offer at
R3 195.22 was deduplicated against a 46-month variant. The Everyday Card limit
increase of R14 000 is shown second.*

**Degradation.** Channel capacity unavailable on entry point 4: dispatch is
deferred rather than allocated arbitrarily, because an arbitrary allocation is
irreversible once sent.

**What changes because it is inside this flow.** **No isolated flow owns this
phase.** Each of the things it arbitrates has its own notion of "best" — a
granting flow ranks by recommendation objective, a campaign flow ranks by
priority weight, a consolidation flow ranks by its objective. Composing them
requires a *fourth* notion of best, across incommensurable units, owned by four
teams. This is the phase most likely to be got wrong quietly, because every input
to it is individually correct.

---

### 5.18 P17 — Final validation

**Determines.** Whether the Bank is willing to be bound by what it is about to
say.

**Runs on.** All eight entry points, with an entry-point-conditional assertion
set. **61 decision points.** **Budget 8.0 ms.**

**What it does.** Re-derives the recommended offer **end to end from its own
amount, term and product**, carrying nothing forward from any earlier phase, and
asserts every derived value. The assertion count is **61 across six products**:
14 for product 10, 12 for 11, 9 for 20, 8 for 21, 11 for 30, 7 for 40.

Product 10's fourteen:

| # | Assertion |
|---|---|
| 1 | The rate equals the card cell for the final band, term and grade, in the version in force at `decision_date` |
| 2 | The rate, after any add-on, is at or below the statutory ceiling in force |
| 3 | The initiation fee equals the piecewise calculation and is at or below the cap |
| 4 | The service fee equals the capped value |
| 5 | The credit life premium equals the table lookup and is at or below R4.25 per R1 000 |
| 6 | The instalment recomputes **to the cent** |
| 7 | The instalment is at or below `max_affordable_instalment`, **on the obligations basis the offer was priced on** |
| 8 | The total cost satisfies the scheduled in duplum test |
| 9 | The total cost ratio is at or below 1.92 |
| 10 | The amount is at or below every ceiling that bound and at or above the product minimum |
| 11 | The term is at or below `term_cap` and in the permitted list for the segment |
| 12 | The grade is at or better than `worst_acceptable_grade` |
| 13 | The amount is a multiple of R250 |
| 14 | Every table version referenced is the one `decision_date` resolves to |

On entry point 7, six of these run — 1, 2, 3, 4, 5, 6 — because the others test a
client that does not exist.

**Any mismatch is a hard failure.** Not a warning, not a logged anomaly, not a
value quietly corrected. The offer is withdrawn, the outcome becomes refer to
queue 7, and an incident is raised. An offer presented to a client is an offer the
Bank may be held to: a cent of disagreement is a reconciliation break, a rate
above the ceiling is an unlawful agreement, and a fee above the cap is a
refundable overcharge across every account written since the defect appeared. The
failure rate of this phase is a monitored metric and should be zero; any non-zero
rate is a defect somewhere upstream.

**Degradation.** None, ever. A validation that cannot run means the offer does
not ship.

**What changes because it is inside this flow.** Two things.

**The isolation requirement is much harder.** "Carry nothing forward" is easy in a
flow with six intermediates and hard in a flow with 41 shared ones and up to 251
live versions of the obligations figure. The phase must be *unable* to read them,
not merely disciplined about it — an ordering constraint expressed as an isolation
constraint (§5.22, O-18).

**The loop makes assertion 7 subtle.** The offer may have been priced at loop pass
3 and validated after pass 4. The re-derivation must test against the obligations
basis the offer was priced on, identified by `loop_pass_index` and
`value_basis_code`, not against "the" affordability answer — of which there are
four. An assertion that tests the wrong basis passes when it should fail, which
is the worst possible behaviour for a safety net.

---

### 5.19 P18 — Disclosure and decision record emission

**Determines.** What the client is told, and what survives for seven years.

**Runs on.** All eight entry points, in eight output shapes. **39 decision
points.** **Budget 2.5 ms** synchronous; the record write is asynchronous.

**What it does.** Three things.

**(a) Reason assembly and ranking (17 decision points).** Every reason raised by
any of the eighteen phases across every loop pass, ranked by the **412-code
registry**'s severity order, with `primary_reason_code` designated. Up to **four**
are communicated; all are recorded. Where the decline was score-driven, the
communicated reasons are the largest negative characteristic contributions,
phrased in the registry's client-facing wording, in the client's language of
record (three languages).

**(b) Disclosure (13 decision points).** On approval: the advance; the initiation
fee and its tax and the statement that it is capitalised; **the amount financed,
stated explicitly as a different number from the advance**; the nominal rate and
whether it is fixed; the service fee including tax; the credit life premium, the
cover it buys, and **the client's right to substitute her own policy**; the
instalment, the number of instalments, and the first and final payment amounts and
dates; the total cost broken into capital, interest, initiation fee, service fees
and premiums — five figures that must sum to the total; the effective annual rate;
and the validity period.

**(c) Record emission (9 decision points).** One record, eight shapes. §5.31.

**Degradation.** Reason registry unreachable: a decline cannot be issued, because
a decline without a registered code cannot be explained to the person it declined.
The decision refers instead. Evidence store unavailable: **the decision still
completes** — evidence capture is on the critical path for correctness and must
never be on the critical path for availability — but the record is buffered
locally, the failure is counted and alerted, and the count of decisions without
evidence is reported monthly.

**What changes because it is inside this flow.** It emits eight output shapes and
eight record shapes from one phase; it ranks reasons contributed by seventeen
other phases under a registry owned by a team that owns none of them; and it must
emit a record for an entry point that produced no decision (7) and for a run that
is not a decision at all (8).

---

### 5.20 The entry points, phase by phase

`●` runs · `◐` runs in reduced or modified form · `○` conditional · blank skipped.

| | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 | P14 | P15 | P16 | P17 | P18 | Count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** New application | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ○ | | ● | ● | ● | **16** |
| **2** Limit change | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | | ◐ | | | ◐ | ● | ● | ● | **15** |
| **3** Limit programme | ● | ◐ | ● | ◐ | | ● | ● | ● | ● | ◐ | | ◐ | | | ● | ● | ● | ● | **14** |
| **4** Campaign pre-approval | ● | | ● | ◐ | | ● | ● | ● | ● | ◐ | ● | ● | ● | ◐ | ◐ | ● | ● | ● | **16** |
| **5** Consolidation | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | | ● | ● | ● | **17** |
| **6** Re-price | ● | ● | ● | | | ● | ● | ● | ● | ○ | | ● | | | | ● | ● | ● | **12** |
| **7** Quotation | ● | | ◐ | | | | | | ◐ | | ● | ● | | | | | ◐ | ● | **7** |
| **8** What-if | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | — |

\* Entry point 8 runs the phase set of the decision it derives from.

Three facts to take from this table.

**No two entry points run the same set.** Eight entry points, eight distinct
`phase_set_id` values before the conditionals, and eleven after them.

**Six phases run on every entry point** — P01, P03, P09, P12, P17, P18 — and
four of those six run in a reduced form on at least one of them. "Runs on all"
and "runs identically on all" are different statements, and only P01 and P18
satisfy the second.

**The reduced forms are where the risk lives.** ◐ appears twelve times. Each is a
place where a phase behaves differently for a reason that is not about the client
— and each is a place where, in the previous generation, somebody wrote a second
copy.

---

### 5.21 Shared intermediates

Forty-one values produced in one phase and read in another. Twelve have five or
more consumers; three have nine or more. This is the connective tissue that no
isolated specification can show, and it is where a large flow's dependencies
actually live.

| Value | Produced in | Consumed in | Consumers |
|---|---|---|---|
| `decision_date` | P01 | every phase | **18** |
| `net_monthly_income` | P06 | P07, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18 | **11** |
| `risk_grade` | P08 | P09, P10, P11, P12, P13, P14, P15, P16, P17 | **9** |
| `adjustment_set_id` | P08 | P09, P10, P12, P13, P14, P15, P16, P17, P18 | **9** |
| `existing_obligations` | P06 | P07, P09, P10, P13, P14, P15, P17 | 7 |
| `segment_code` | P06 | P07, P08, P09, P10, P11, P12, P16 | 7 |
| `max_affordable_instalment` | P10 | P12, P13, P14, P15, P16, P17 | 6 |
| `amount_cap`, `term_cap`, `worst_acceptable_grade` | P09 | P11, P12, P13, P14, P16, P17 | 6 |
| `fraud_verdict_code` | P05 | P09, P11, P14, P16, P17, P18 | 6 |
| `instalment` | P12 | P10, P13, P14, P16, P17, P18 | 6 |
| `probability_of_default` | P08 | P09, P12, P15, P16, P17 | 5 |
| `bureau_as_of_date`, `bureau_is_stale` | P06 | P07, P09, P10, P17, P18 | 5 |
| `nominal_annual_rate` | P12 | P13, P14, P16, P17, P18 | 5 |
| `living_expenses` | P06 | P10, P14, P17, P18 | 4 |
| `total_exposure` | P06 | P09, P11, P15, P16 | 4 |
| `worst_arrears_months` | P06 | P07, P09, P10, P15 | 4 |
| `consent_state` | P03 | P04, P14, P16, P18 | 4 |
| `income_band_code` | P06 | P07, P09, P16 | 3 |
| … 23 further values | | | 2 each |

Three consequences follow from the table, and none of them is visible at six
phases.

**A change to P06 can reach eleven phases.** Which means a feature band edit —
the kind of thing that happens on a Tuesday and looks like data — is a change to
eleven phases across six owners, and must be treated as one. §5.28.

**A phase cannot be understood by reading it.** P13's behaviour depends on six
values it does not compute, three of which are overlaid, two of which have
hypothetical twins, and one of which may have been produced on a previous loop
pass. Navigability (§5.27) is a requirement precisely because reading does not
work here.

**Producing a value is an obligation to consumers you cannot see.** P02 builds a
related-party set nothing in P02 needs. P08 resolves an overlay register whose
overlays mostly act elsewhere. P06 derives income four phases before anyone tests
it. Each of those is a cost borne by one team for the benefit of another, and the
ownership map has to make that visible or it will be optimised away by somebody
tidying up.

#### 5.21.1 The hard case — two versions of one concept, live at once

`existing_obligations` is the client's monthly cost of debt already held. Eleven
phases read something derived from it. Inside one execution of entry point 5 — and
inside entry point 1 whenever the loop fires — **it exists in up to 251
simultaneous versions**:

| Version | `value_basis_code` | `scenario_ref` | Meaning |
|---|---|---|---|
| Actual | 1 | null | What she owes today. R9 340.00. |
| Hypothetical, scenario 1 | 2 | 1 | If the Bank settles accounts {3, 7}, she would owe R6 115.00 |
| Hypothetical, scenario 2 | 2 | 2 | If it settles {3, 7, 9}, R4 880.00 |
| … | 2 | 3..250 | up to 250 of them |

Both the actual and the chosen hypothetical are **live at the same instant**, and
both must be unambiguous in the record. Nine values have this property:
`existing_obligations`, `revolving_utilisation`, `worst_arrears_months`,
`total_exposure`, `discretionary_income`, `max_affordable_instalment`,
`affordability_verdict_code`, `risk_grade` and `probability_of_default` — the last
two because some scorecard characteristics are obligation-derived, so the
hypothetical settlement changes the grade, which changes the rate, which changes
the instalment, which changes the affordability test.

Four requirements, each of which fails a naive design:

1. **Every recorded value names its basis.** `value_basis_code` and, where
   hypothetical, `scenario_ref`. A recorded `existing_obligations` with no basis
   is not evidence; it is a number.
2. **A phase may not consume a hypothetical value unless it declares that it
   does.** P09's exposure rules must see the actual figure — the Bank's real
   exposure does not fall because a scenario was evaluated. P10 inside the search
   must see the hypothetical. A phase that reads "the" obligations figure without
   saying which must be **impossible**, not merely discouraged.
3. **The decision record must state which basis supported the shipped offer**,
   and must state that the basis is *conditional*. If the client takes the
   consolidation offer and the settlements execute, the hypothetical becomes
   actual; if a quotation expires and one settlement fails, it does not, and the
   affordability defence for that offer evaporates. The record must carry the
   conditionality, not just the number.
4. **The disclosure uses the actual figure and the assessment uses the
   hypothetical**, and the two appear on the same page of the same document to
   the same client. Anything that collapses them produces a disclosure that
   contradicts its own working.

**And there is a third axis.** Every adjusted value also carries its unadjusted
counterpart (§5.9). So `risk_grade` has four live versions in a consolidation
assessment — actual-adjusted, actual-unadjusted, hypothetical-adjusted,
hypothetical-unadjusted — and five values in the flow are in that position.
`max_affordable_instalment` is one of them: R5 664.56 adjusted actual, R5 940.88
unadjusted actual, and a pair per scenario.

A naming scheme that handles two of the three axes and not the third will be
discovered in year two, by an adjudicator.

---

### 5.22 Ordering constraints

Roughly **forty** hard ordering constraints hold between the eighteen phases. The
obvious ones — pricing before the solve, scoring before grading — are not listed.
These twenty-four are the ones that are non-obvious, entry-point-dependent, or
genuinely circular, and each is a requirement with a reason rather than a
convention.

| # | Constraint | Why |
|---|---|---|
| O-01 | P08's adjustments apply **before** grading | Grading keys off **adjusted** PD. Grading first and adjusting after produces a grade that does not correspond to the PD recorded beside it, and a rate read from the wrong row. |
| O-02 | P03's consent **before** P04's acquisition | A bureau enquiry without consent is an offence, and the enquiry leaves a footprint on the client's file that cannot be withdrawn. Not reversible by apology. |
| O-03 | P02's identity resolution **before** P03's consent | Consent is held against `client_id`. Resolving the wrong client applies the wrong consents, and a probabilistic match at 0.89 confidence reading someone else's marketing preferences is a privacy incident, not a rounding error. |
| O-04 | `decision_date` fixed in P01 and never re-read | Every effective-dated artefact resolves against it. A phase that re-reads "today" replays *successfully* with the wrong answer, which is worse than failing. |
| O-05 | P08 resolves the overlay register **once**, before any phase that reads an overlaid value | Two phases resolving independently, with a register change between them, put two different overlay stacks in one decision. |
| O-06 | **P05 before P04's bureau call on entry points 1 and 2 below R40 000; P05 after it on entry points 5 and 6** | Low-value: no sense paying for a bureau enquiry on a known fraudulent identity, and the enquiry itself helps a synthetic identity build a file. High-value and re-price: 34 of the 188 fraud rules consume bureau-derived velocity and address history. **This is a cycle in the graph, broken by declaration, per entry point.** |
| O-07 | P06's income determination **before** P07 | 14 of SC-A3's 61 characteristics are income-derived. This is why the affordability chain is split across P06 and P10 at all (§5.7). |
| O-08 | P06's bureau normalisation **before** segment assignment, which is **before** scorecard selection | Segments 1 and 2 are distinguished by thin-file status, which is bureau-derived. Scorecard selection keys off segment. A wrong segment silently selects a valid scorecard and produces a plausible wrong score. |
| O-09 | **P10 before P11, and P10 again after it** | Product eligibility is amount-keyed; the amount comes from the solve; the solve is bounded by affordability; affordability's buffer is grade- and **product**-keyed. A genuine cycle, broken by running P10 product-neutrally with the most conservative buffer, routing, then re-running with the routed product's buffer. §5.12. |
| O-10 | P09's amount, term and grade entries **before** P10; P09's instalment entries **after** it | `instalment_cap` is seeded from affordability, and two register entries reduce it further. The register therefore runs in two passes and the entries do not know which pass they are in. The second of the two cycles. |
| O-11 | P02's related-party set **before** P09's CAP-0301 | Group exposure headroom needs the graph. The graph query is expensive and set-shaped; the cap check is cheap and record-shaped. Computing it late means computing it inside a 7.5 ms budget. |
| O-12 | P09's CAP-0118 uplift **last in the register**, and after every `regulatory`-class entry | It is the only entry that may raise a ceiling. Raising before a regulatory entry runs would let a campaign authority exceed a statutory maximum, which is not a policy error but an unlawful agreement. |
| O-13 | The statutory rate ceiling check **after** the rate add-on, not only at card validation | A card validated as compliant, plus a 60 basis point overlay, can be non-compliant. Checking only at validation checks the wrong artefact. |
| O-14 | Within P12: rate → fee → premium → instalment | The fee is a function of the advance; the premium is a function of the amount **financed**, which includes the capitalised fee; the instalment is a function of all three. Any other order computes a premium on the wrong base. |
| O-15 | P14 sits **inside** the P10 loop, not before or after it | It is reached because affordability failed, and its output is an input to affordability's next run. Modelling it as a phase that runs before or after affordability makes the loop inexpressible. §5.23. |
| O-16 | P15's per-account work **before** its allocation, and the allocation **after every account** | The funding line cannot be drawn until every account's proposed increase is known. A population-level stage inside a record-level flow — and the reason entry point 3 has a 25-minute allocation stage inside a 3-hour window. |
| O-17 | Every product's P11–P13 fan-out **before** P16 | With three eligible products, three routings, three waterfalls, three price sets and three solves complete before one arbitration runs. |
| O-18 | P17 must be **unable** to read any shared intermediate | "Carry nothing forward" is an isolation constraint expressed as an ordering constraint. If P17 can read `instalment` it will eventually assert `instalment == instalment`. |
| O-19 | P18's reason ranking **after** every phase that can raise a reason, **including P17** | A validation failure is a reason. Ranking before P17 produces a reason set that omits the reason the offer was withdrawn. |
| O-20 | P18's disclosure **after** P17 passes | A quotation issued from an offer that failed validation is a representation the Bank is bound by. |
| O-21 | Record emission last, **and off the critical path** | Two different statements. Last, so it captures everything; off the path, so an evidence-store outage never declines an applicant. |
| O-22 | Rounding applied at exactly one place per value | Rounding to R250 twice, or rounding an already-rounded advance after a cap, changes the answer by up to R250 and breaks the multiple-of-R250 assertion. |
| O-23 | P03's consent read once, and the **same read** used by P18 | P18 needs consent state for the marketing disclosure. Re-reading gives a fresher answer and an internally inconsistent decision. Freshness loses to consistency, and the choice is declared. |
| O-24 | In-flight application aggregation is **serialised per client** | Two simultaneous applications from one client each see the other's exposure or neither does, depending on interleaving. Order-dependence here is a concurrency constraint masquerading as an ordering one, and it is the only place in the flow where two decisions can affect each other. |

**Two of these are genuine cycles** — O-06 and O-09, with O-10 a third in
miniature — and they are the important ones. A cycle broken by declaration is
defensible, explainable and testable. A cycle broken by accident is a defect that
appears when a rule is added on the far side of it, and it will be found by a
client whose second application was declined for a reason the first one did not
mention.

---

### 5.23 Cross-phase feedback loops

Seven loops exist in the flow. One of them spans four phases, each of which would
be an entire project standing alone.

| Loop | Span | Bound | Termination guarantee | Where |
|---|---|---|---|---|
| **L1** | P10 → P14 → P10 → P12 → P13 → P16 | **4 passes** | Each pass must strictly reduce the candidate space or improve the objective by ≥ R25 of instalment relief | Entry points 1 (conditional), 5 |
| L2 | Inside P13 | 19 evaluations per term, 152 per application | Discrete finite domain | 1, 4, 5 |
| L3 | P10 → P11 → P10 | **2 passes, fixed** | Declared single re-run; routing does not re-open | 1, 5 |
| L4 | Inside P14's advance derivation | 5 iterations, R1 tolerance | Contraction on a bounded interval | 5, and 1 when L1 fires |
| L5 | P16 → P17 → P16 | **3 promotions** | Each promotion takes the next-best candidate from a finite ordered set | all |
| L6 | Inside P15's allocation | 2 passes | Over-allocation factor applied once, then measured once | 3 |
| L7 | P09 grade tightening → appetite re-read | **Zero — deliberately not a loop** | Single pass declared; register order is load-bearing instead | all |

**L7 deserves a note.** CAP-0176 tightens `worst_acceptable_grade`, and the
tightened grade re-keys the appetite grid that CAP-0104 read earlier in the same
register. The obvious fix is to re-run the register. The declared answer is that
**the register is single-pass** and earlier entries do not re-run — which means
register order is semantically load-bearing, which means a quarterly reordering
can change outcomes for reasons nobody intended. That is an arguable trade and it
must be visible in the reviewable artefact, because it is precisely the kind of
thing a policy owner will not discover by reading their own rule.

#### 5.23.1 L1 in detail

Affordability fails. The client is consolidation-eligible. The flow does not
decline; it looks for a shape in which it can say yes.

```
P10  affordability against ACTUAL obligations        → fail
P14  consolidation search, settlement set proposed   → hypothetical obligations
P10  affordability against HYPOTHETICAL obligations  → pass or fail
P12  pricing re-runs on the new advance and product
P13  the solve re-runs over the new domain
P16  offer assembly
        ↳ if no shippable offer and the bound is not reached, next pass
```

**The bound is four passes.** Not a timeout. A pass count, because a time-based
bound is not reproducible and a search whose truncation differs between two runs
of the same inputs cannot be replayed.

**The termination guarantee.** A pass runs only if it can strictly improve: the
candidate settlement space must shrink, or the objective must improve by at least
R25 of monthly instalment relief. A pass that can do neither is not run, and
`loop_termination_code` records which of the five reasons applied. Termination is
therefore a property of the input, not of the clock.

**The determinism requirement.** Same client, same inputs, same artefact versions,
same `decision_date`, same overlay stack ⇒ **same number of passes, same
scenarios in the same order, same winner**. Every value the loop produces carries
`loop_pass_index`. Two replays that differ in pass count have not reproduced the
decision even if they happen to agree on the offer.

**A worked four-pass loop**, which is what the record must be able to explain:

| Pass | What ran | Obligations | Result |
|---|---|---|---|
| 0 | P10 on actual | R9 340.00 | **Fail**, reason 2140. Loop entered. |
| 1 | P14 set A {3, 7} → P10 → P12 → P13 → P16 | R6 115.00 | Affordable. Solve returns R118 000 at 60 months, instalment R3 940. **Rejected at P16**: anti-harm, `total_cost_delta` +R41 200 = 38% of baseline, over the 35% threshold. |
| 2 | P14 set B {3, 7, 9, 11} → P10 → P12 → P13 → P16 | R4 880.00 | Affordable. Solve returns R141 000 at 72 months. **Rejected at P17**: assertion 11 — the routed product is 11, and `term_cap` for grade 8 on product 11 is 60. |
| 3 | P14 set B at 60 months → P10 → P12 → P13 → P16 | R4 880.00 | Affordable. Solve returns R128 500 at 60 months, instalment R4 180. Anti-harm passes at 29%. **Validation passes. Ships.** |

`loop_termination_code` 1, converged at pass 3. Had pass 4 also failed, the
outcome would have been refer, queue 6, with all four passes recorded.

**What the record must show about a loop that ran four times.** Nine things, and
none of them is optional:

1. The pass count and `loop_termination_code`.
2. The trigger of each pass — which phase's failure caused it.
3. The obligations basis at each pass, with its `scenario_ref`.
4. Every value that changed between passes, with its `loop_pass_index`.
5. **The three offers that did not ship, and why each did not.** These are what
   people ask about. "Why did you not give me the 72-month option?" has an answer
   and the answer is at pass 2.
6. Which pass produced the shipped offer.
7. Which affordability verdict supports the shipped offer, and that it rests on a
   **conditional** obligations basis that becomes actual only on execution.
8. The evaluation budget consumed at each pass, because passes 1 to 3 consumed
   180 of the 250 scenario budget between them and pass 4 would have had 70.
9. That the client requested a Flex Loan and is being offered a consolidation —
   a substitution, with its disclosure consequence.

**The navigability consequence.** A consultant asked "why 60 months and not 72"
must be able to answer: *at 72 months the routed product's term cap for grade 8 is
60; that was pass 2, and it was withdrawn at final validation.* That answer
requires the record to connect a term to a pass to a product to a cap entry to a
grade — five hops — and to render them as one sentence. §5.27 is about whether
that is achievable.

---
### 5.24 The latency budget

Entry point 1: **p99 under 120 ms end to end, excluding external calls**, at
55 000 applications a day and a peak of 9 per second sustained for twenty minutes
on payday Fridays.

#### 5.24.1 The internal budget

| Phase | p99 budget | Notes |
|---|---|---|
| P01 Request validation and routing | 1.5 ms | |
| P02 Client and identity resolution | 2.5 ms | Excludes the identity service call |
| P03 Consent and hard eligibility | 2.5 ms | 62 decision points, all evaluated |
| P04 Data acquisition orchestration | 3.0 ms | Dispatch and collation only; waiting is excluded |
| P05 Fraud and financial crime | 5.5 ms | 219 rule evaluations, no early exit |
| P06 Feature derivation | 13.0 ms | ~510 derived values |
| P07 Scoring | 5.0 ms | Up to two scorecards, 125 contributions |
| P08 Calibration, grading and adjustments | 3.0 ms | Includes overlay register resolution |
| P09 Policy gates and the cap waterfall | 7.5 ms | Up to three products × 47 entries |
| P10 Affordability | 8.5 ms | Two passes: product-neutral and routed |
| P11 Product routing | 2.0 ms | |
| P12 Pricing | 2.5 ms | Outside the solve |
| P13 **The solve** | **38.0 ms** | Up to 152 pricing evaluations |
| P14 Consolidation search | — | Conditional; see below |
| P15 Limit assignment | — | Not run on entry point 1 |
| P16 Offer assembly and arbitration | 6.5 ms | |
| P17 Final validation | 8.0 ms | Full independent re-derivation |
| P18 Disclosure and record emission | 2.5 ms | Synchronous portion only |
| **Serial total** | **111.5 ms** | |
| **Slack** | **8.5 ms** | 7.1% |
| **Target** | **120.0 ms** | |

**When P14 fires**, the request leaves this budget. It does not get a bigger
slice; it becomes a different latency class with a declared budget of **820 ms
p99**, of which P14 takes 180 ms and the loop's repeated P10/P12/P13 passes take
up to 460 ms. 6.8% of entry point 1 volume is in that class, and it is reported
separately rather than being allowed to inflate the p99 of the other 93.2%.
Averaging the two produces a number that describes neither.

#### 5.24.2 The external budget

Not in the 120 ms, but very much in the client's experience.

| Call | Budget | Observed p99 | Issued on | Concurrency |
|---|---|---|---|---|
| Bureau | 600 ms | 455 ms | 78% | Wave 1 |
| Device and consortium | 350 ms | 210 ms | 84% | Wave 1 |
| Identity verification | 450 ms | 320 ms | 20% | Wave 1 |
| Internal state and exposure | 60 ms | 38 ms | 100% | Wave 1 |
| Statement aggregator | 1 400 ms | 1 090 ms | 41% | Wave 2 (needs a resolved client) |
| Employer confirmation | 1 100 ms | 740 ms | 7% | Wave 2 (conditional on bureau) |

External critical path: **wave 1 max 600 ms + wave 2 max 1 400 ms = 2 000 ms.**

| Component | Budget |
|---|---|
| Internal logic | 120 ms |
| External critical path | 2 000 ms |
| Transport, serialisation, queueing | 230 ms |
| **Client-observed p99** | **2 350 ms** |

Two things must be stated because they are how latency targets become dishonest.
First, **the 120 ms is measured and published separately from the 2 350 ms**, and
neither is allowed to stand in for the other. Second, a phase's budget is a
**declared property of the phase**, measured per decision, not an average across a
day: a phase exceeding its budget on more than **0.5%** of requests in a day is a
defect with a named owner, and the count is in `phase_budget_overrun_codes` on
every affected record.

#### 5.24.3 Overrun policy

**Abandonable.** The request completes without them, with the omission recorded
and the outcome marked:

| What | Effect of abandoning |
|---|---|
| The challenger scorecard in P07 | The champion result stands; the challenger sample is short for the day |
| The shadow fraud rules in P05 | Shadow measurement is short for the day |
| P14 and the loop | The decline stands as a decline; the client is told a consolidation assessment was not completed and may request one |
| The fan-out to the second and third eligible products in P11–P13 | Only the requested product is offered; the substitution opportunity is lost and recorded as lost |
| The unadjusted counterfactual | Recorded as not computed, available on demand later |
| Terms beyond the requested one in P13 | The requested term is solved; the alternatives are not offered |

**Not abandonable**, under any circumstances:

P03 consent and eligibility · P05's verdict handling (the external call may time
out, which is degradation, not abandonment) · P09's twelve regulatory entries ·
P10 · P17 · P18's reason assembly and the obligation to emit a record.

**When a non-abandonable phase overruns**, the request **does not return a
decision**. It returns `outcome_code` refer, queue 9, reason 2495, marked
`record_completeness_code` 3, with every phase's partial evidence emitted. It
does **not** return a decline — a decline is a decision with legal consequences
and a reason the client is entitled to, and "we ran out of time" is not a reason
that appears in a 412-code registry. It does **not** return a degraded approval.
The cost of referring is a few minutes of an underwriter's time; the cost of
shipping a decision the flow did not finish making is unbounded.

---

### 5.25 Degraded operation

Twelve sources, each with a declared failure behaviour. The composite is
`degraded_mode_code`, and it is **declared in advance**, not computed after the
fact — "it degraded gracefully" is not an acceptable answer to "what did it do".

| Source | Entry points affected | Continue? | Outcome becomes | Code |
|---|---|---|---|---|
| **Bureau down** | 1, 2, 5 | Yes, narrowly | Approve only within a reduced envelope: ≤ R25 000, term ≤ 36, segments 3–5 only, internal tenure ≥ 24 months, no internal arrears in 24. Everyone else **refers**. Four scorecards unusable; segments 2–5 fall to SC-A1 with a −45 point shift. | 11 |
| **Statement aggregator down** | 1, 5 | Yes | Income falls to the best remaining tier — usually 7, at a 35% haircut. That is a real penalty, correctly applied. Amounts above R80 000 refer. | 12 |
| **Fraud service timing out** | 1, 2, 5, 6 | Yes | `fraud_verdict_code` 4. Requests ≤ R8 000 from clients with ≥ 30 months' tenure proceed on the bypass; everything else **refers**, queue 3. Never approves on an unknown fraud verdict. | 13 |
| **Consortium intelligence down** (fraud service up) | 1, 2, 5, 6 | Yes | 29 of 188 rules recorded as *not evaluated*; weighted threshold drops to 0.62. | 21 |
| **Identity verification down** | 1, 5 | Yes | Confidence floor rises to 0.94; queue 2 referrals rise from 0.6% to ~4.1%. | 14 |
| **One rate card version missing** | 1, 4, 5, 6, 7 | **No, for that product** | The product is withdrawn from routing and the withdrawal is recorded. Other products proceed. There is no defensible way to price without a card, and last month's card is a price the Bank did not publish. | 31 |
| **Statutory ceiling or fee schedule missing** | all | **No** | Hard stop. Lending at an unverified rate is unlawful, not degraded. | 32 |
| **Adjustment register unreachable** | all but 7 | Yes, for 90 minutes | The last successfully resolved version is used, pinned and recorded. After 90 minutes, hard stop. Running "unadjusted" is **not** a fallback: not knowing whether an overlay applies is different from knowing none does, and the silent version produces normal-looking answers that are systematically wrong in one direction. | 41 |
| **Reason registry unreachable** | all | Partly | Approvals proceed. **Declines cannot be issued** — a decline without a registered code cannot be explained to the person it declined — so they refer instead. | 42 |
| **Internal account state stale** (> 6 h) | 1, 2, 5, 6 | Yes | CAP-0301 runs on the stale figure; approvals it would have bound are held for re-check. Increases on entry point 2 capped at R5 000. | 51 |
| **Feature mart stale** (> 36 h) | 3, 4 | **No** | The cycle does not run. A limit programme on month-old behaviour is worse than no limit programme. | 52 |
| **Evidence store unavailable** | all | Yes | The decision completes; the record buffers locally; the failure is counted, alerted and reconciled. Evidence capture is on the critical path for correctness and must never be on the critical path for availability. | 61 |
| **Affordability tables unavailable** | all but 7 | **No** | There is no degraded affordability mode, deliberately. A guessed affordability answer is the basis of a reckless-lending defence. | 33 |

**Four sources have no degraded mode at all** — the statutory ceiling and fee
schedule, the rate cards, the affordability tables, and (after 90 minutes) the
adjustment register. That is a design position, stated once, rather than a gap.

#### 5.25.1 The three rules that make degradation safe

**1. Degrade toward refer, never toward decline.** Declining for want of a bureau
record is a decision the client can complain about and the Bank cannot defend:
the reason is the Bank's, not the client's. Every degraded path either approves
inside a narrower declared envelope or refers. No degradation path in this flow
produces a decline that would not have been produced without it, and that is a
testable property (§10, criterion 14).

**2. A degraded decision is marked, in the record and in the answer.** Every
decision carries `degraded_mode_code`, `source_degradation_codes` per source as
at the moment it was read, and `record_completeness_code` 4. A consultant
explaining the decision sees the marking. A replay asserts it. An analyst
querying outcomes can exclude or isolate them.

**3. A degraded decision is re-assessable, and the re-assessment is scheduled,
not hoped for.** Every degraded approval and every degraded referral enters a
queue that is re-run automatically within **5 business days** of the source
returning, using the recorded inputs plus the newly available source. Where the
re-run changes the answer materially — a larger amount, a different price, an
approval where there was a referral — the client is contacted. Where it changes
the answer adversely, **the original stands**: the Bank does not withdraw an
offer because a source came back. Roughly **0.9%** of annual volume passes
through this queue, which at 258 M records is 2.3 M re-assessments a year and is
therefore a capacity requirement, not a footnote.

---

### 5.26 Twelve teams, one artefact

#### 5.26.1 The ownership map

| Team | Phases owned or co-owned | Tables | Parameters | Decision points |
|---|---|---|---|---|
| T1 Decision Platform | P01, P04, P09, P13, P17, P18 | 1 | 21 | 174 |
| T2 Client Identity & Data | P02, P04, P06 | 3 | 9 | 152 |
| T3 Regulatory Compliance | P03, P09, P12, P17, P18 | 7 | 41 | 112 |
| T4 Credit Risk Policy | P03, P08, P09, P10, P15 | 12 | 79 | 282 |
| T5 Model Risk & Modelling | P06, P07, P08 | 4 | 8 | 135 |
| T6 Treasury / Pricing | P12 | 6 | 19 | 44 |
| T7 Unsecured Lending Product | P09, P11, P12, P13, P16 | 2 | 38 | 76 |
| T8 Cards & Revolving Product | P09, P11, P12, P15, P16 | 2 | 34 | 78 |
| T9 Secured Lending Product | P09, P11, P12, P16 | 2 | 24 | 39 |
| T10 Financial Crime | P05, P09 | 3 | 14 | 216 |
| T11 Campaign Analytics | P16 | 4 | 19 | 29 |
| T12 Collections & Restructure | P14 | 1 | 12 | 63 |
| ALCO and the Credit Committee — govern, own no phase | — | — | 22 | — |
| **Total** | **18** | **47** | **340** | **1 400** |

Every decision point has exactly one owner, so the last column sums exactly.
Phases do not: thirteen of the eighteen appear in more than one row, which is the
whole difficulty.

**Only five of the eighteen phases have a single owner**: P02, P05, P07, P13 (for
its mechanics) and P14. Thirteen have two or more. P09 has five. No phase has one
owner for both its logic and every value it reads.

#### 5.26.2 Three teams changing the flow in the same week

An ordinary week, not a bad one:

| Day | Team | Change | Route | SLA |
|---|---|---|---|---|
| Tue 14:10 | T6 Treasury | Repo cut announced at 11:00. Patches 3 100 Flex Loan cells and re-validates all 34 560 against the new statutory ceiling of 27.75% | Treasury committee, table refresh, no deployment | **2 hours** |
| Wed 09:30 | T4 Credit Risk Policy | Quarterly register release: adds 2 entries, retires 1, reorders 4, moves CAP-0176 above CAP-0104 | Credit Committee, quarterly | 6 weeks' notice |
| Thu 16:45 | T7 Unsecured Lending | Removes 84 months from the permitted term list for segments 1 and 2; moves the total cost ratio threshold from 1.92 to 1.88 | Product owner signature | **Same day** |

All three change what entry point 1 outputs for overlapping populations. None of
the three can test the other two's change, because on Tuesday the other two do
not exist yet. The reordering in Wednesday's release changes the answer for the
population Tuesday's patch re-priced. Thursday's threshold change interacts with
both through the solve's BIND-TCR constraint.

The questions this forces, none of which has an obvious answer:

- Does Thursday's change **invalidate the certification** of Tuesday's and
  Wednesday's? If yes, nothing ships on a Thursday. If no, what was certified?
- If Wednesday's release is rolled back on Friday, **does Tuesday's patch survive
  it?** It must, and that is a requirement on how the two are versioned, not a
  hope.
- Is there **one release train or three independent deploys**? A release train
  makes Thursday wait for the quarterly cadence, which is how inline literals get
  written. Three independent deploys mean three artefacts whose interaction was
  never tested together before it was live.
- Which of the three is **accountable** if the composed result is wrong?

#### 5.26.3 Two teams needing to change the same value

The affordability buffer. T4 owns it and varies it by grade and product. T8 wants
it to vary by **channel** for entry point 2, because app-originated limit requests
behave differently from branch ones. One value, two owners, and a scope dimension
only one of them cares about.

Three resolutions, three different consequences:

| Resolution | Consequence |
|---|---|
| **Widen the parameter's key** to grade × product × channel | The grid goes from 288 to 288 values it already had — but now T8 can change a value that also affects entry points 1 and 5, which T7 owns. Widening a key widens an ownership boundary. |
| **Split into two parameters**, one per path | The two will diverge. In three years one of them will be right about the residual floor. |
| **Overlay it** for the channel and the entry point | Correct, but an overlay has a mandatory expiry, and a permanent behavioural difference expressed as an expiring overlay is an overlay somebody renews forever without re-justifying. |

This flow chose the first, with the key widened and the ownership of the channel
dimension delegated to T8 **within declared bounds** (6% to 24%). That the choice
had to be made, and that it moves an ownership boundary rather than a value, is
the point.

#### 5.26.4 A team that owns a phase but not what it reads

T12 owns P14 — all 63 of its decision points, its budget, its scenario ordering
and its objective. P14 reads four rate cards owned by T6, the settleability matrix
owned by T4, the anti-harm thresholds owned by T3, the obligation treatment matrix
owned by T4, and the affordability values owned by T4 and T3. **T12 owns the
phase and owns almost nothing the phase depends on.**

Consequence: Tuesday's repo patch changes which scenario wins for an unknown
number of clients, and T12 finds out when the objective's realised distribution
moves. The requirement is that the blast-radius answer (§5.28) **reaches T12
before the card ships**, automatically, because T6 cannot be expected to know
that a card cell participates in a consolidation objective.

#### 5.26.5 What must be true for team A to deploy without waiting for team B

Five conditions. Each is testable and each is a constraint on structure, not on
process.

1. **A's blast radius is computable, and is provably disjoint from B's** — or the
   overlap is declared and the two changes are certified jointly. Computable in
   under 60 seconds, statically, without running anything (§5.28).
2. **A's certification set is *derived* from its blast radius, not curated.** It
   covers A's outputs and the outputs of every phase downstream that A can reach.
   A curated set is a set somebody forgot to extend when they added a consumer.
3. **The artefact A changes is versioned separately from B's and resolves by
   `decision_date`**, so a decision in flight completes on one consistent set and
   never on a half-applied mixture.
4. **A's change is separately identifiable and separately replayable.** If the
   composed release's effect cannot be attributed to A or to B individually, the
   release must be split.
5. **Rolling A back does not roll back B**, and the rollback is itself a version,
   not a restoration of a prior state.

Where any of the five fails, A waits. **The measurement is what fraction of
changes satisfy all five**: a flow in which 80% of changes are independently
deployable is a flow twelve teams can work in, and one in which 30% are is a flow
with a release train, a queue, and inline literals within a year.

---

### 5.27 Navigability

A hard requirement with measurable tests, not a documentation aspiration. **This
is where a large flow usually fails first**, and it fails silently: nothing breaks,
people simply stop being able to answer questions, and the answers get replaced by
the opinion of whoever has been there longest.

| Test | Who | Target | Measurement |
|---|---|---|---|
| **N1 Locate** | A new engineer, in their first month, with the repository and the generated artefacts and **no colleague** | Given a named value, locate **every** place it can be set, in **under 10 minutes** | 12 named values, 4 new joiners a year, pass mark 10 of 12 |
| **N2 Explain** | A credit analyst, no engineer | Answer "why was the term capped at 48 for this client" from the record and the reviewable artefact, in **under 5 minutes** | 20 sampled decisions, quarterly, pass mark 18 of 20 |
| **N3 Resolve** | Anyone, from a decision record | Every one of the 1 400 decision points cited in a record resolves to its identity, its owner, its current rendering and its version chain | 10 000-decision sample, **zero** unresolvable citations permitted |
| **N4 Invert** | An analyst | Given a decision point, list every decision in the last 90 days that it **bound** | Under 60 seconds over ~64 M decisions |
| **N5 Trace a loop** | A consultant, 90 seconds | Given a decision that traversed L1, state which pass produced the shipped offer and why the earlier passes did not | 40 sampled loop decisions, quarterly |

**N1 has a corollary that is easy to miss.** The test is not "find where the value
is set"; it is "find *every* place it can be set". For `amount_cap` that is: 118
register entries (47 applicable to product 10), 6 cap overlays, 6 product maxima,
1 regulatory maximum and 1 uplift entry — **132 places**. A human cannot enumerate
132 places correctly by reading, and will not try twice. The enumeration must be
**generated**, and the requirement is therefore on the structure, not on whoever
writes the documentation.

**N2 is the one that matters commercially.** 31 000 client calls a year turn on
it. Answering it requires connecting a term to a cap entry to a grade to a
scorecard characteristic to an overlay — five hops across four phases and three
owners — and rendering the chain as one sentence in the client's language.

**Why this is not a documentation problem**, stated explicitly because it will be
mistaken for one:

- Documentation **diverges**. It diverges within weeks, silently, and a divergence
  between what a governance forum approved and what is deployed is a regulatory
  finding independently of whether the deployed logic was correct. N1 and N2 must
  be answered from artefacts **generated from what actually runs**.
- Documentation is **not addressable**. N3 requires that a citation in a record
  resolves to a thing. A paragraph in a document is not a thing a record can cite.
- Documentation is **not queryable**. N4 is an inversion over 64 M decisions. No
  amount of writing answers it.
- Documentation does not **version-chain**. N5 requires the rendering that
  corresponds to the build that ran, not the current one.

**The failure mode, named.** In the system being replaced, N1 currently takes
between two days and never, and is answered by asking a person. That person is
the actual index. When they leave, the flow becomes unmodifiable in the regions
only they could navigate, and those regions accumulate workarounds instead of
changes. Every one of the 1 400 decision points has a chance of ending up in such
a region, and nothing in the code will say which ones have.

---

### 5.28 Change collision and blast radius

**Before any change ships, its blast radius must be answerable.** Six questions,
with time bounds:

| Question | Answered by | Bound |
|---|---|---|
| Which **outputs** can this change affect? | Static lineage over the flow graph | **60 s**, no data, no execution |
| Which **entry points**? | Static, conditioned on `phase_set_id` | 60 s |
| Which **products** and which **segments**? | Static, conditioned on applicability | 60 s |
| Which **teams** need to know? | The ownership map, from the affected decision points | 60 s |
| How many of the last 30 days' decisions **would have changed**, in which direction, by how much? | Empirical swap set over ~21 M decisions | **2 hours** |
| What is the **expected effect** the change's owner declares? | The owner, in advance, in words | At proposal |

The last two together are the governance mechanism: a declared expected effect
that the measured swap set contradicts is a change that its own owner did not
understand, and it does not ship.

**At 1 400 decision points this cannot be done by reading.** The counterexample is
mundane. T2 adjusts one feature band edge — `income_band_code` boundary 7 moves
from R28 000 to R29 500 — which looks like a data edit and takes four minutes.
That code is read by:

- **9 of the 18 phases**
- **31 cap register entries** across four products
- **4 of the 9 scorecards**, as a binned characteristic
- **11 of the 44 campaign trees**, at 23 distinct nodes
- The expense norm lookup, the buffer grid selection and two suppression rules
- An estimated **1.7 M clients** move band on the next entry point 4 cycle

No human enumerates that correctly, and the person making the change has no
reason to suspect most of it.

**Lineage must survive the loop, or it is useless.** Naive reachability over a
graph containing L1 returns "everything can affect everything", because the loop
makes P16 reachable from P10 and P10 reachable from P16. The requirement is
therefore stronger: lineage must be answerable **conditioned on entry point,
product and loop pass**, so that "can a change to the anti-harm threshold affect
a quotation on entry point 7" returns *no* rather than *unknown*. A lineage
facility that returns `unknown` for a third of queries is one nobody uses.

**Three collision classes**, each with a different remedy:

| Class | Example | Remedy |
|---|---|---|
| **Disjoint** | T6 patches the Drive Finance card; T11 retires campaign 31 | Independent deployment, no coordination |
| **Overlapping, same direction** | T4 tightens CAP-0176; T7 lowers the total cost ratio threshold — both reduce approvals | Joint certification, combined swap set, one declared expected effect |
| **Overlapping, opposing** | T4 tightens a cap while T6 reprices the same cells downward | Must not ship in the same release. The combined effect is measurable but not attributable, and an unattributable effect cannot be rolled back selectively |

`blast_radius_id` is recorded on the change and on every decision made under it,
so that "which change caused this movement" is answerable months later without
re-deriving anything.

---

### 5.29 Dead logic at scale

Of 1 400 decision points, an expected 5–9% have not fired in a year. The current
measurement says **109 — 7.8%**.

| Category | Count | What it is |
|---|---|---|
| **Correct and rare** | 47 | Real logic serving a small population |
| **Shadowed** | 31 | Subsumed by an earlier entry in an ordered set; can never fire regardless of data |
| **Dead** | 22 | The condition refers to something that no longer exists |
| **Unknown** | 9 | Insufficient evidence either way |

**The 47 correct-and-rare**, because this is the category that gets removed by
mistake: 14 serve segments 8–11 (6.9% of entry point 1, and some fire on a tenth
of that); 11 belong to product 40 at 2 400 decisions a month; 9 are regulatory
backstops that are *designed* never to fire and whose firing would indicate a
defect elsewhere; 8 handle degraded modes that did not occur this year; 5 are
joint-application paths on products that rarely see them.

**The 31 shadowed** are the interesting ones, because they are a **structural**
property answerable without running anything: an entry whose condition is
subsumed by an earlier entry in the same ordered register can never fire, and
this is detectable statically. Twelve of the 31 became shadowed when CAP-0176 was
re-thresholded eighteen months ago and started absorbing a population three later
entries were written for. Nobody noticed, because nothing broke.

**The 22 dead**: 8 reference a data source withdrawn in 2027; 6 test a segment
that was merged; 4 relate to a product variant that never launched; 4 are
overlays' residue — conditions added to support an overlay that has since expired.

#### 5.29.1 Why counting firings is not enough

Three measurements are needed, not one, and confusing them is how rare logic gets
deleted:

1. **Reached count** — how many decisions evaluated it at all.
2. **Firing count** — how many satisfied it.
3. **Binding count** — how many had their outcome changed by it.

A decision point reached 4 100 times and firing 4 100 times is a different animal
from one reached 19 M times and firing 4 100 times, and both are different from
one reached zero times. **Firing counts alone cannot distinguish a rule that is
never reached from one that is reached and never satisfied**, and the second is
usually a defect while the first is usually correct.

**Normalisation is mandatory.** The denominator is not 258 M. It is the
population that could have reached the decision point, given its entry points, its
products and its applicability. A product 40 entry measured against 258 M looks
dead at any firing rate; measured against 28 800 product 40 decisions a year it
may be perfectly healthy. Getting this wrong deletes the logic that protects the
smallest, least-defended populations first.

#### 5.29.2 Removing one safely

Removal is a change like any other and carries the full §5.28 obligation. The
additional gate: **a swap set over 90 days must show zero swaps before removal is
permitted.** That is a weaker statement than "it is dead" — 90 days of production
is not proof about a condition that fires annually — and the gap between the two
is the residual risk, which is accepted explicitly rather than assumed away.

**The standing obligation.** Every owner either actions or explicitly retains,
with a written reason, every candidate attributed to them, **annually**. At 12
teams and 109 candidates that is real work, and it must be routed by owner
automatically from the ownership map, because a single list of 109 items sent to
everyone is a list nobody actions.

---

### 5.30 Testing a flow this size

#### 5.30.1 The combinatorics

| Axis | Values |
|---|---|
| Entry points | 8 |
| Products | 6 |
| Segments | 12 |
| Degradation states | 13 declared + normal = 14 |
| Overlay stack | on / off |
| Loop passes (where L1 is reachable) | 0..4 |

8 × 6 × 12 = 576 entry-point/product/segment triples, of which **197 are
reachable** — product 30 does not occur on entry point 3, segment 10 does not
occur on product 40, entry point 7 has no segment at all. 197 × 14 degradation
states = 2 758 states; × 2 overlay modes = **5 516**; plus the loop-pass
dimension on the 34 triples where L1 is reachable adds a further 136. Call it
**5 650 distinct states** that can occur in production.

Exhaustive certification of 5 650 states on every change is not affordable, and
pretending otherwise produces a suite that is skipped.

#### 5.30.2 What the strategy must cover

| Layer | Owned by | Size | Runs |
|---|---|---|---|
| **Per-phase unit sets** | The phase's owner | 18 sets, 1 200–31 000 cases each, ~164 000 total | On every change to that phase |
| **Capability conformance** | T1, from owners' spreadsheets | Policy owners supply expected inputs and outputs; they run as tests | On every change to a policy-bearing value |
| **The golden set** | T1 + owners | **140 000 decisions**, stratified over the 197 reachable triples, **minimum 200 per stratum**, quarterly refresh | On every change |
| **Degraded-mode set** | T1 | 14 states × the entry points each affects = 74 combinations × 400 decisions | Weekly, and on any change to §5.25 |
| **The re-check regression** | T1 | 180 000 applications; the offer must never fail affordability when re-derived; **zero** failures | On every change to P09–P13 |
| **Exhaustive solve verification** | T1 | 6 000 applications × every R250 candidate × every permitted term | On **every rate card version** |
| **Entry-point agreement** | T1 + Decision Governance | 2 000 clients assessed on two entry points within 30 days | Monthly |
| **Replay** | Decision Governance | 50 000 recorded decisions re-derived exactly | Nightly |
| **Swap set** | Decision Governance | 30 days of production, ~21 M decisions | Per change |

**The golden set here is not a sample of production.** It is a *stratified*
construction: 197 strata with a minimum of 200 decisions each is 39 400
decisions before anything else, and the strata that production does not populate
— segment 10 on product 30 under a degraded bureau — must be **synthesised and
marked as synthetic**, because the alternative is a test suite that is strongest
exactly where the risk is lowest. Synthetic cases carry their own flag and their
expected outputs are signed off by the owning team rather than harvested.

#### 5.30.3 The requirement that actually matters

**A change to one phase must not require re-certifying all eighteen.**

Stated testably: the certification set for a change is **derived from its blast
radius**, and for a change whose radius is confined to *k* phases, certification
runs in time proportional to *k*.

| Change | Radius | Certification | Target |
|---|---|---|---|
| Permitted term list, product 10, segment 2 | 2 phases (P13, P16) | Golden subset, 14 strata | **< 6 minutes** |
| A rate card patch | 4 phases (P12, P13, P14, P17) | Solve verification + affected strata | < 25 minutes |
| A cap register reorder | 6 phases | Full golden set + swap set | < 90 minutes |
| A grading change | 14 phases | Full certification | < 3 hours |
| A new scorecard | 5 phases, 4 owners | Full certification + model validation | Days |

**If every change costs three hours, the weekly-cadence teams stop shipping
weekly.** They do not stop needing to change things; they start writing the
change somewhere cheaper. That is how 546 inline literals appear where parameters
should have been, and it is a structural failure being paid for as a behavioural
one.

**One negative requirement, which is easy to forget.** A phase's tests must not
be able to pass while the phase is **unreachable in the composed flow**. Sixty-six
modules behind one test file is the documented previous-generation failure; the
version of it that applies here is a phase whose unit tests are green and whose
`phase_set_id` never includes it on any live entry point. Coverage must be
measured against the composed flow, not against the phase.

---

### 5.31 One decision record

Everything from up to eighteen phases, up to four loop passes, up to 250
evaluated scenarios, six external sources and an overlay stack — in one artefact,
for eight entry points with different phase sets.

#### 5.31.1 One shape, eight fillings

The record has **one schema**, not eight. Eight schemas guarantee that the fourth
one is missing the table versions when the finding lands. The schema is
**sparse by construction**: a phase that did not run is *absent*, and the absence
carries `record_completeness_code` 2 with the `phase_set_id` that explains it.

This is why §4.5's fourth null exists. Four absences are distinguishable and must
remain so:

| Absence | Code | Meaning |
|---|---|---|
| Phase did not run on this entry point | 2 | Structural. Normal. |
| Phase was abandoned on budget | 3 | Operational. Monitored. |
| Phase ran degraded and could not produce the value | 4 | Degraded. Re-assessable. |
| Value could not be established from available evidence | (§4.5) | A credit fact about the client. |

A design that maps all four to null makes every cross-entry-point comparison
meaningless and makes a degraded decision indistinguishable from a normal one.

#### 5.31.2 What the record must contain

The non-negotiable floor, and the reason each item is on it:

1. A stable decision identifier assigned before any logic runs.
2. `entry_point_code`, `phase_set_id`, `decision_date`, `blast_radius_id` and the
   build identifier that resolves to exactly the logic that ran.
3. Inputs **as received**, before normalisation, with the three null situations
   preserved and each source's as-at date.
4. Every table, norm, cap, card, scorecard, boundary set and registry read, **to
   the cell and to the version**.
5. Every parameter in force, including every one that fell back to a default,
   marked as such. No decision may depend on a value absent from its record.
6. The overlay stack: set identifier, the overlays that applied, each one's
   effect, the composition order, and the unadjusted input each consumed.
7. Which decision points were **evaluated**, with their outcomes — not only which
   fired. Otherwise "why did CAP-0212 not bind" is unanswerable and the dead-logic
   measurement collapses into firing counts.
8. All five cap chains in full, with every entry's verdict and status.
9. Per-characteristic scorecard contributions, signed, with bins.
10. Every pricing evaluation the solve performed, in order, with the candidate,
    the cell, the fee, the premium, the instalment and the verdict — because "why
    R60 000 and not R59 750" is only answerable from the evaluations.
11. Every loop pass, its trigger, its obligations basis, its result and its
    discarded offers (§5.23.1).
12. `value_basis_code` and `scenario_ref` on every value that has more than one
    live version.
13. All applicable final-validation assertions with computed and expected values.
14. `degraded_mode_code` and per-source degradation states as at read time.
15. The complete reason set, ranked, with the registry version and the primary.
16. Per-phase timings and `phase_budget_overrun_codes`.

Emission is **idempotent and at-least-once**: a decision retried three times
produces one record, not three and not zero.

#### 5.31.3 Sizing, and the tension

| Entry point | Mean record | Annual volume | Annual |
|---|---|---|---|
| 1 New application | **58 KB** (bureau payload ~41 KB) | 20.1 M | 1 166 GB |
| 2 Limit change | 7.2 KB | 5.1 M | 37 GB |
| 3 Limit programme | 1.7 KB | 49.2 M | 84 GB |
| 4 Campaign pre-approval | 2.3 KB | 170.4 M | 392 GB |
| 5 Consolidation | **296 KB** (250 scenarios × ~58 values) | 1.5 M | 444 GB |
| 6 Re-price | 5.4 KB | 3.3 M | 18 GB |
| 7 Quotation | 1.1 KB | 8.0 M | 9 GB |
| 8 What-if | 164 KB | 21 600 | 4 GB |
| **Total** | | **~258 M** | **~2.15 TB/year** |
| Seven-year window | | | **~15.1 TB** |
| Provisioned, with indices, replicas and immutability | | | **~30 TB** |

**Two artefacts dominate.** Entry point 1's bureau payload is 71% of its record
and 54% of the whole estate's bytes. Entry point 5 is 0.6% of decisions and 21%
of bytes.

**The levers, and what each costs:**

| Lever | Saving | Cost |
|---|---|---|
| Content-addressed de-duplication of bureau views (one view serves multiple decisions within its 40-day validity) | −710 GB/yr | Retrieval indirection; a purge must not orphan a decision |
| Summarising entry point 5's rejected scenarios — full detail for the top 12, summary for the rest | −385 GB/yr | **"We did not keep it" is not available as an answer for a rejected scenario**, because those are the ones people ask about. A summary that cannot answer "why was my store card not settled" has saved the wrong 385 GB |
| Full intermediate detail only for a deterministic 0.5% sample plus 100% of declines, referrals and disputed decisions | −0.9 TB/yr against recording everything | The 0.5% is chosen before anyone knows which decision will be disputed |
| **Net** | **~0.94 TB/yr, ~6.6 TB over seven years** | |

**The tension, stated plainly.** Completeness is a legal property and storage is
an economic one, and they trade against each other at roughly a factor of three
here. The resolution is that **replay is what makes it affordable not to store
everything**: intermediate detail can be regenerated from the recorded inputs and
versions, which is precisely why the replay guarantee must be *exact* rather than
approximate. If replay is approximate, the honest answer is to store everything,
and the bill triples.

That is a decision to be taken deliberately, at design time, rather than
inherited from whichever default an implementation happens to have.

---
## 6. Parameters and tables

47 tables, ~283 500 cells. 340 parameters. Four change cadences that must coexist
inside one deployed artefact. Listing every one would be 900 rows and would
obscure the two facts that matter: **where the mass is**, and **who moves it
when**.

### 6.1 The 47 tables, by owner

| Owner | Tables | Cells | Cadence | Largest artefact |
|---|---|---|---|---|
| T6 Treasury | 6 | **135 204** | Monthly, patched within 2 h of a repo move | Drive Finance rate card, 71 424 cells |
| T11 Campaign Analytics | 4 | **83 400** | Weekly | 44 tree definitions, ~6 800 nodes and ~2 900 leaves |
| T2 Client Identity & Data | 3 | **39 900** | Weekly | Vehicle valuation guide, 38 500 rows |
| T4 Credit Risk Policy | 12 | **9 618** | Quarterly | Limit assignment matrix, 1 080 cells / 3 240 values |
| T3 Regulatory Compliance | 7 | **5 012** | On gazette; registry monthly | Reason code registry, 412 codes × 8 attributes |
| T5 Model Risk & Modelling | 4 | **4 632** | On model release | 9 scorecard definitions, ~3 780 rows |
| T10 Financial Crime | 3 | **3 954** | Weekly; emergency in minutes | Concentration watchlist, 3 150 identifiers |
| T7/T8/T9 Product | 6 | **1 302** | Weekly (T7, T8), monthly (T9) | Cap register, 118 entries × 12 attributes |
| T12 Collections & Restructure | 1 | **420** | Quarterly | Scenario ordering and objective register |
| T1 Decision Platform | 1 | **84** | Rare | Entry-point and phase-set registry |
| **Total** | **47** | **~283 500** | | |

**Five artefacts hold 90% of the cells**: the Drive Finance card (71 424), the
campaign trees (~82 000 attribute values), the Flex Loan card (34 560), the Home
Loan card (28 800) and the vehicle valuation guide (38 500). **Thirty-one of the
47 tables have fewer than 1 200 cells**, and those thirty-one are the ones that
change most often and cause the most incidents. Size and risk are uncorrelated
here, which is worth knowing before effort is allocated by size.

**Two properties are required of every table in the list**, without exception:

1. **Cell-level attribution.** Given an outcome, it must be possible to say which
   cell of which version of which table was read. Recorded, not reconstructible.
2. **Diffability.** A new version produces a review artefact naming the cells
   that changed, by how much, and the aggregate impact on the last 30 days'
   decisions re-derived under it. "Treasury sent a new spreadsheet" is not a
   reviewable change, and at 71 424 cells nobody is going to eyeball it.

**Rate card validation** runs on every version before it may go live: every cell
populated (a null rate is a defect, never "use the neighbour"); every cell at or
below the statutory ceiling in force on its effective date; rates monotone
non-increasing across grades within an amount-band/term cell, because a better
grade is never priced worse, and violations block the card; **band-edge
inversions permitted but declared**, listed in the review artefact and counted —
the current Flex Loan card has **27**, deliberately, and P13 must handle every
one, while an *undeclared* inversion blocks the card; and the mid-month patch
path produces the same artefact as a full refresh.

### 6.2 The 340 parameters, by owner and cadence

| Class | Owner | Count | Cadence | Approval | Cannot be changed by |
|---|---|---|---|---|---|
| **Statutory** | T3 Compliance, transcribing the gazette | **41** | On gazette; unpredictable | Compliance | Everyone, including Credit Risk Policy. **Unreachable from a product team, not merely forbidden to them.** |
| **Policy** | T4 Credit Risk Policy | **79** | Quarterly, faster on a portfolio move | Credit Committee | Product teams |
| **Product** | T7 (38), T8 (34), T9 (24) | **96** | **Weekly** | Product owner signature | Anyone outside the owning product |
| **Pricing** | T6 Treasury | **19** | Monthly + patches | Treasury committee | Product, Policy |
| **Platform** | T1 (21), T2 (9) | **30** | Rare | Engineering | Business owners |
| **Campaign** | T11 | **19** | Weekly | Campaign forum | Credit Risk Policy values |
| **Crime** | T10 | **14** | Weekly; emergency path | Four-eyes within T10 | Everyone else |
| **Restructure** | T12 | **12** | Quarterly | Credit Committee | Product teams |
| **Portfolio** | ALCO | **11** | Monthly | ALCO | Everyone else |
| **Overlay** | Credit Committee | **11** governing values | **Ad hoc, sometimes twice a month** | Individual approval, individual expiry | Product teams; Treasury over its own card |
| **Model** | T5 | **8** | On model release; traffic shares weekly | Model Validation | Everyone else |
| **Total** | | **340** | | | |

**Sixty-one parameters are read by more than one phase, and fourteen by more than
four.** The rounding unit (R250) is read by five phases; the statutory rate
ceiling by four; the buffer grid by three; `decision_date` resolution behaviour
by all eighteen. A parameter with fourteen readers is not a setting, it is an
interface, and changing it is a change to fourteen phases.

**The four cadences, and why they cannot be merged.** A statutory change arrives
when it arrives and cannot wait for a quarterly train. A policy change needs six
weeks of committee lead time and a simulated impact. A product change is made on
a Thursday afternoon by someone who will not file a ticket if it takes a week.
An overlay is approved individually, expires individually, and exists precisely
because the artefact it modifies moves too slowly. Forcing all four onto one
release train makes three of them late and produces workarounds for the fourth.
Allowing all four to deploy freely produces the Tuesday–Wednesday–Thursday
collision in §5.26.2 with no coordination at all.

### 6.3 The nine scorecards

| Id | Characteristics × bins | Rows | Calibration | Segments / products |
|---|---|---|---|---|
| SC-A1 | 38 × 9 | 342 | C1 | Segment 1 |
| SC-A2 | 52 × 9 | 468 | C2 | Segment 2 |
| SC-A3 | 61 × 9 | 549 | C3 | Segments 3, 4, 5, 11, 12 |
| SC-A4 | 44 × 9 | 396 | C3 | Segments 6, 9; 7, 8, 10 with a declared shift |
| SC-A5 (challenger) | 64 × 9 | 576 | C3 | 12% of SC-A3 traffic |
| SC-B1 | 47 × 8 | 376 | C4 | Product 20, entry points 2, 3, 4 |
| SC-B2 | 41 × 8 | 328 | C4 | Product 21, entry points 2, 3, 4 |
| SC-S1 | 55 × 9 | 495 | C5 | Products 30, 40 |
| SC-P1 | 36 × 7 | 252 | C6 | Entry point 4 propensity |
| **Total** | | **3 782** | 6 calibrations | |

### 6.4 The overlay register

Live count runs **34 to 90** across the six products. It is one artefact with one
version, and it belongs to none of the tables it modifies.

| Attribute | Notes |
|---|---|
| Overlay id, description, **rationale** | The rationale is a required field, not a courtesy |
| Kind | Score shift, scaling change, odds multiplier, calibration re-anchor, boundary shift, cut-off shift, cap reduction, rate add-on, buffer adjustment, matrix dial, campaign volume dial |
| Scope | Scorecard, segment, channel, product, grade range, amount range, term range, entry point, campaign |
| Magnitude | Points, ratio, percentage points, basis points |
| Position in the stack | An integer. Composition order is **declared**, not emergent |
| Owner and approval reference | A named person and a committee minute |
| Effective from, effective to | Both required |
| **Review date** | Required and **enforced**. An overlay reaching it without renewal surfaces |
| Enabled | So the stack can be run off without deleting its definition |

**Three prohibitions, stated as requirements.** An overlay is **never merged**
into the base artefact "to simplify", because merging destroys the separation
between what the model said and what policy decided, which is the only reason the
mechanism exists. An overlay may **not** be expressed as an edit to a card cell,
a grade boundary or a characteristic's points. And **every flow must be runnable
with the stack disabled, through the same implementation** — that is how the base
artefacts' own performance is monitored and how a validator separates model
output from policy decision.

The failure mode this guards against is specific: a tightening applied during one
bad quarter, still silently in force four years later, with nobody able to say
what unwinding it would do. §11 scenario 16 is that failure, arriving.

---

## 7. Outputs

### 7.1 Eight output shapes

| Entry point | To the caller |
|---|---|
| **1** | `outcome_code`; 0..8 offers, each with amount, term, rate, fees, premium, instalment, total cost, effective rate, `is_recommended` and its binding constraint; ranked `decline_reason_codes` with `primary_reason_code`; `referral_queue_code`; `risk_grade`; `max_affordable_instalment`; the full pre-agreement disclosure block; `assessment_id` |
| **2** | The new `limit`, `change_type_code`, the consent request and its reference, the notice class and effective date where a decrease, the binding cap, `assessment_id` |
| **3** | 4.1 M account rows: proposed limit, unadjusted matrix limit, binding cap, affordability verdict and assessment reference, allocation rank, `allocation_outcome_code`, notice class, exclusion set for excluded accounts, every artefact version |
| **4** | 14.2 M client rows: campaign assignment, offer tier, pre-approved amount and term, indicative instalment, the **tree path taken** per evaluated tree, suppressions applied, arbitration result, holdout flag, expiry |
| **5** | The chosen scenario in full; two materially distinct runners-up; the before-and-after comparison; the execution package of per-account settlement instructions with expiries and security actions; `offer_valid_until` as the binding minimum of quotation, valuation and offer validity; every rejected scenario with its rejection reasons and actual-versus-threshold values; the search evidence — budget, consumption, termination cause, ordering rules and objective in force |
| **6** | The revised rate, the revised instalment, the notice class, the validity period, the reason the re-price was triggered |
| **7** | A quotation: advance, initiation fee and tax, **amount financed stated as a distinct number**, nominal rate, service fee, credit life premium and the substitution right, instalment, first and final payments, total cost broken into five components that sum, effective annual rate, validity period. **No `outcome_code`.** |
| **8** | A side-by-side against the original: which outputs moved, which gates changed outcome, which decision points started or stopped binding, which tree nodes were visited instead — carrying a **non-production identifier**, structurally marked, un-writable to the decision store, un-issuable to a client |

### 7.2 Persisted

Everything in §5.31.2, for every decision on every entry point, sufficient to
re-derive the decision **with no network and no live service**. Seven years —
twenty-five months online, the remainder archived with a six-working-hour
retrieval.

Plus, on a cycle: per-decision-point evaluated/fired/bound counts, daily; the
overlay register with measured effect and days to expiry, daily; the degraded-mode
queue with re-assessment status; the entry-point agreement reconciliation,
monthly; the coverage and dead-logic report, monthly; the funding-line
reconciliation to ALCO, monthly; and every approved reviewable rendering with its
approval, frozen, for as long as any decision made under it is retained.

---

## 8. Non-functional requirements

| Requirement | Value |
|---|---|
| Entry point 1 latency | **p99 < 120 ms** excluding external calls; **2 350 ms** client-observed; 820 ms for the 6.8% that enter the loop, reported separately |
| Entry point 2 latency | p99 < 200 ms, p50 < 60 ms |
| Entry point 5 latency | 2.5 s p95, 4.0 s p99, including up to 400 internal re-evaluations |
| Entry point 7 latency | **p99 < 50 ms**, zero external calls |
| Entry point 8 latency | < 5 s p95 |
| Entry point 3 window | 4.1 M accounts in **3 hours**, allocation within 25 minutes of it |
| Entry point 4 window | 14.2 M clients in **6 hours** — 657 clients/second sustained, ~1.1 M tree evaluations/second at peak, ~94 000 pricing evaluations/second |
| Interactive throughput | 106 000 decisions/day; peak 9/s on entry point 1 sustained 20 minutes |
| **Determinism** | Identical inputs, artefact versions, overlay stack and `decision_date` ⇒ identical outputs **bit for bit**, on any host, in any order, at any concurrency, in any year within the retention window — **including the loop pass count, the scenario order and the funded set** |
| **Cross-entry-point identity** | Where two entry points run the same phase on the same evidence, they produce the same value. Verified monthly. §10 criterion 9 |
| Cold start | No per-request compilation. The first request after a deployment is not materially slower than the ten-thousandth. 135 204 rate card cells resident and indexed before the first request, not on it |
| Live artefact swap | A card, matrix, register or overlay refresh takes effect without a code deployment and without interrupting serving. A mid-month patch is applicable within **2 hours** of a repo announcement |
| Version pinning | A decision in flight when an artefact changes completes on the artefact it started with |
| Restartability | The 3-hour and 6-hour windows have no room for a full restart. A failed cycle **resumes**, and a resumed cycle produces the same result as an uninterrupted one |
| Partial-failure isolation | A campaign, a product or an entry point failing must not fail the other seven |
| Availability | 99.95% for entry points 1, 2, 6, 7 during business hours; 99.5% for entry point 5; a 24-hour recovery window for 3 and 4 |
| Evidence overhead | < 1.8 ms added at p99 to the 120 ms budget; durable within 2 s of the decision |
| Evidence durability | No record may be lost. Loss is an incident, counted and reported monthly |
| Retention | 7 years; 25 months online |

---

## 9. Audit, evidence and explainability

Four interrogations, each of which has been made of a real retail credit system,
and each of which is harder here than in any isolated flow.

### 9.1 A decision that spans eighteen phases

**Who asks, and how fast.** A contact centre consultant, on the phone, in under
**90 seconds**, 31 000 times a year. A credit analyst, in under 15 minutes. An
ombud adjudicator, in a written pack, reviewed before issue.

**Why it is harder here.** In a single-purpose flow, an explanation is a rule and
a value. Here it is a **traversal**: eighteen phases, five cap chains, up to two
scorecards, an overlay stack, six external sources and a `phase_set_id` that
explains why four phases produced nothing. The consultant does not want eighteen
phases. They want one sentence.

**The requirement.** Three renderings from one record, generated, never
hand-written:

| Rendering | Audience | Contains | Bound |
|---|---|---|---|
| Consultant | Contact centre | The outcome, the primary reason in client wording, the binding constraint, the one number that would have to change | < 3 s, one screen |
| Analyst | Credit analysts | Every phase's contribution, every chain in full, every evaluated-and-did-not-bind verdict, the overlay stack | < 15 s |
| Adjudicator | Disputes, ombud, regulator | The above plus the artefact versions, the approval references, the evidence tiers and the plain-language rendering in force on `decision_date` | < 1 working day, reviewed before issue |

**The hard part is not the content. It is the reduction.** Going from ~4 800
recorded facts to one sentence requires knowing which fact was decisive, and
"which was decisive" is exactly the attribution requirement in §5.10 — which is
why it is a requirement and not a nicety.

### 9.2 Explaining a decision that traversed a feedback loop

*"Why was I offered R128 500 over 60 months when I asked for R180 000 over 72,
and why did you settle four of my accounts and not six?"*

The answer requires the record to say: the request failed affordability on the
actual obligations at pass 0; three settlement sets were evaluated across three
passes; the first was rejected because the total cost would have risen 38%
against a 35% anti-harm threshold; the second was withdrawn at final validation
because the routed product's term cap at grade 8 is 60 months; the third shipped.
And it must say that the affordability verdict supporting the offer rests on a
**hypothetical** obligations figure that becomes actual only when the settlements
execute.

**Four properties this demands** that no single-pass flow needs:

1. Every value carries `loop_pass_index`, so "the affordability verdict" is never
   ambiguous between four of them.
2. **The discarded offers are retained with their rejection reasons.** They are
   the answer to most of the questions actually asked.
3. The termination cause is recorded, so "was there a better answer you did not
   look for?" is answerable: converged, or out of budget, are different answers
   and the client is entitled to the true one.
4. The conditionality of the shipped offer is explicit, so that if a settlement
   fails and the deal unwinds, the record already says why the affordability
   assessment no longer holds.

### 9.3 The reviewable artefact at 1 400 decision points

This is doc 04 §6's top risk, at its worst, and it is worth being precise about
why it is worse here than anywhere else in the estate.

**The requirement.** Credit Risk Policy, Compliance and the Credit Committee must
read, understand and approve logic they did not write and cannot read as code, at
quarterly re-approval and on every change requiring policy approval. It must be
**generated from what actually runs**, because a hand-maintained description
diverges within weeks, and a divergence between what a governance forum approved
and what is deployed is a regulatory finding on its own — independently of
whether the deployed logic was correct.

**Why volume defeats naive generation.** One entry per decision point, rendered
plainly, is roughly **400 pages**, plus 47 tables, 340 parameters and an overlay
register. A 400-page document approved in a 40-minute agenda item is not
governance; it is the appearance of governance, and an adjudicator will say so in
writing.

**So the artefact must be layered, and the layering must itself be generated:**

| Layer | Size | Read by | When |
|---|---|---|---|
| Per-entry-point summary | 4 pages × 8 | Credit Committee | Quarterly |
| Per-phase reference | 8–40 pages × 18 | The owning team | On change |
| Full reference | ~400 pages | Nobody, cover to cover; everybody, by lookup | On demand |
| **Diff against the last approved version** | Usually 2–6 pages | **This is what is actually read at re-approval** | Every change |

**The governance gap this exposes, which is the sharpest one in the document.**
Twelve teams approve their own slices. Each slice is reviewable. **Nobody owns the
composition.** The cap register is reviewable; the cap register *interacting with*
the affordability buffer through the solve is not in anyone's slice. The anti-harm
threshold is reviewable; the anti-harm threshold *interacting with* a rate card
patch is not. There is no forum whose remit is the composed flow, and there is no
artefact that renders the composition rather than the parts.

Creating one is not a documentation task. It requires that the composition be
*expressible* — the ordering constraints, the shared intermediates and the loop
must be first-class enough to render — which is a structural property of the
implementation and is why this sits in §9 rather than in a style guide.

**And it is people-blocked, not code-blocked.** The only way to know whether the
artefact works is to hand a real credit risk analyst a real policy document and a
generated rendering and ask them to find the three places they disagree. That
exercise should run before the format is fixed, not after.

### 9.4 Reproducing a decision made under a degraded mode

**The trap.** A decision made on 14 March 2027 while the bureau was down must
replay, in 2033, to the **degraded** answer — not to the answer the flow would
give now with the bureau available. A replay that "fixes" the degradation
produces a different decision and proves nothing.

**What this requires.**

1. `degraded_mode_code` and `source_degradation_codes` are **inputs to the
   replay**, not annotations on it. The replay environment asserts them.
2. The absent source is recorded as an **absence with a cause**, not as a null
   and not as an empty payload. "The bureau returned no accounts" and "the bureau
   was not called" must replay differently, and they do — one is a credit fact
   and the other is an operational one.
3. The **declared population adjustments** in force under that mode — the −45
   point scorecard shift, the reduced fraud threshold of 0.62, the R25 000
   envelope — are recorded with the decision, because they were parameters at
   that moment and will not be parameters later.
4. **No external calls during replay**, enforced by the replay environment rather
   than trusted. That prohibition is the only real test of whether capture was
   complete.
5. The re-assessment that followed (§5.25.1, rule 3) is **linked** to the
   original decision, so the pair reads as one story: what was decided under
   degradation, what was decided afterwards, and which one governs.

**The measurement.** A quarterly sample of 400 decisions made under each of the
thirteen degraded modes replays to an exact match. Non-reproducibility above
**0.05%** per entry point per month is an incident, not a metric.

### 9.5 Attribution across teams

*"Approval rate on product 10, channel 3, segment 4 fell 4.1 points last week.
Whose change did that?"*

Three changes shipped that week (§5.26.2). The record must support attributing the
movement to one of them, or to their interaction, without re-deriving anything
by hand. `blast_radius_id` on every decision and every change is what makes that
a query rather than an investigation. Where the effect is genuinely joint, the
answer must be "joint, and here is the decomposition attempt and why it fails" —
which is itself the argument for not shipping opposing changes together (§5.28).

---

## 10. Acceptance criteria

Dominated deliberately by **structural** properties. A flow that produces correct
answers and cannot be changed, navigated, reviewed or attributed has failed this
specification.

**Structure and ownership**

1. **No phase is duplicated per entry point.** Zero instances of a phase existing
   in two forms because two entry points need different behaviour. Reduced forms
   are declared applicability, not second implementations, and the test is that
   removing a decision point removes it from every entry point that runs it.
2. **Independent deployability.** At least **80%** of changes, measured over a
   rolling quarter, ship without waiting for another team, satisfying all five
   conditions in §5.26.5. Measured, published monthly, and owned by T1.
3. A product team changes a threshold it owns and it is live **within one
   business day** without any other team acting and without re-certifying the
   other seventeen phases.
4. A statutory value is **unreachable** from a product team's change surface —
   demonstrated by attempt, not by convention.
5. Rolling back one team's change does not roll back another's, demonstrated on a
   live release.
6. A capability used twice in one decision with different settings — the
   affordability determination in two modes, the pricing body under two products
   — produces two correctly-attributed results from **one implementation**.

**Navigability**

7. **N1–N5 in §5.27 all pass** at their stated marks, measured on the stated
   cadences, with the results published.
8. Every one of the 1 400 decision points is locatable from a decision record
   that cites it: 10 000-decision sample, **zero** unresolvable citations.

**Entry-point agreement**

9. **The same client assessed through two entry points in the same week does not
   get a materially different answer.** Stated testably, on the ~1.9 M clients a
   month who take two paths:

   > For a client whose entry point 4 pre-approval in month M proposed an amount
   > A, an entry point 1 application in month M on unchanged evidence must
   > produce a proposed amount A′ with **|A′ − A| ≤ max(R2 000, 4% of A)**, and a
   > risk grade differing by **at most one notch**.
   >
   > For an account whose entry point 3 programme run proposed a limit L, an
   > entry point 2 request in the same month on unchanged evidence must produce
   > L′ with **|L′ − L| ≤ max(R750, 3% of L)**.

   A monthly reconciliation samples **2 000 clients that took both paths** and
   reports the agreement rate, which must be **≥ 99.0%** within tolerance. Every
   breach must be attributable to a **named input difference** — fresh declared
   income against deposit-derived income, a bureau refresh between the snapshot
   and the request, a transaction moving a utilisation band — and to nothing
   else. "The batch path works differently" is a failure of this criterion, not
   an explanation of it.

   Note what this does **not** require: the *funded* outcomes need not agree,
   because entry point 3 is budget-constrained and entry point 2 is not.

**Correctness**

10. Identical inputs, versions, overlay stack and `decision_date` produce
    identical outputs bit for bit, including the loop pass count, the scenario
    order, the evaluation sequence and the funded set.
11. **The offer never fails affordability when re-derived from scratch**:
    180 000-application regression, zero failures, on the obligations basis the
    offer was priced on.
12. The solve's answer equals the exhaustive answer over every R250 candidate at
    every permitted term: 6 000-application sample, zero disagreements, re-run on
    **every rate card version**.
13. Every decision is re-derivable to the cent from its record with **no network
    and no live service**, seven years later, using the artefacts in force at its
    `decision_date`.
14. **No degradation path produces a decline that would not have been produced
    without it.** Tested per degraded mode against the same population.

**Blast radius and change**

15. Any change's static blast radius — outputs, entry points, products, segments,
    teams — is answerable in **under 60 seconds** without executing anything, and
    returns `unknown` for no more than **2%** of queries.
16. Any change's empirical swap set over 30 days of production is answerable in
    under **2 hours**.
17. A change confined to *k* phases certifies in time proportional to *k*; the
    permitted-term-list change certifies in **under 6 minutes**.
18. Every shipped change carries a declared expected effect, and a change whose
    measured swap set contradicts its declaration does not ship.

**Governance and evidence**

19. The reviewable artefact is **generated from what runs**, is layered per
    §9.3, produces a diff at every change, and a real credit risk analyst can
    verify a named policy against it without an engineer.
20. Any decision can be run with the overlay stack disabled, through the **same
    implementation**, producing the unadjusted answer beside the adjusted one.
21. Every live overlay's scope, order, approval, effect and expiry is recoverable
    for any past date, and an overlay past its review date surfaces.
22. The dead-logic measurement distinguishes **reached**, **fired** and **bound**,
    normalises by reachable population, and routes candidates to owners
    automatically.
23. A decision made under a degraded mode replays to the **degraded** answer:
    400 decisions per mode per quarter, exact match.
24. The count of decisions emitted without a record is known, reported monthly,
    and is zero.

---

## 11. Change scenarios

Twenty changes that will arrive within months of go-live. A good structure makes
them cheap; a bad one makes each of them a negotiation. These are the
maintainability test, and they are deliberately the heavyweight ones.

1. **A seventh product launches** — a fee-free starter facility, `product_code`
   12, unsecured, R1 000 to R25 000, for segments 1 and 2 only. It needs a
   seventh cap register slice, a seventh column in every product-keyed table, a
   seventh routing rule, its own rate representation (flat, no card), its own
   validation assertions, an eighth entry in P16's arbitration, and a scorecard
   it does not have. **Which of the 47 tables and 340 parameters must gain a
   dimension, and which must not?**

2. **A regulatory change touching six phases simultaneously.** New affordability
   regulations change the income evidence hierarchy (P06), the expense norm table
   and its bands (P06), the disclosure content and ordering (P18), the reason
   taxonomy classification of 31 codes (P18), the record retention period (P18),
   and the statutory fee cap formula (P12). Effective on a date, with decisions
   before that date using the old rules **forever**. Six phases, four owners, one
   effective date, no phased rollout available.

3. **A ninth entry point is added** — an embedded-finance partner API where a
   retailer requests a credit decision at checkout, with a 900 ms budget, a
   different consent basis, a subset of four products, and an output shape that
   is neither an offer set nor a quotation. **What does adding it cost, and does
   any existing phase change?**

4. **A team splits in two.** T4 Credit Risk Policy becomes Retail Policy and
   Portfolio Policy. The 89 decision points, 13 tables and 88 parameters divide
   unevenly, the limit assignment matrix goes to one and the cap register to the
   other, and the affordability buffer is wanted by both. **What exactly is being
   divided, and can it be divided without touching any logic?**

5. **Two teams merge.** T7 and T8 become Retail Lending Product. Two weekly
   cadences, two approval signatures, two parameter sets and 47 cap entries
   become one. **Does the merge require a change, or only a re-attribution?**

6. **A scorecard is replaced across four phases.** SC-A3 goes from 61
   characteristics to 73, seven of which do not exist in P06 yet. It changes the
   feature derivation, the calibration, the grade boundaries, 31 grade-keyed cap
   entries and the propensity covariates in P16. Both versions must be live for a
   **four-month parallel run** on different segments.

7. **The fraud service moves in-process.** The external device and consortium
   call disappears; 188 rules that were behind a 350 ms boundary become an in-flow
   phase with a 4 ms budget. The P04/P05 ordering inversion (O-06) becomes moot
   for one entry point and sharper for another. Ownership of a 205-decision-point
   phase moves from a contract to a codebase. The degradation policy for
   `degraded_mode_code` 13 and 21 is rewritten. The decision point count goes from
   1 400 to roughly 1 400 — the rules were already counted — but the **blast
   radius of every Financial Crime change** changes completely.

8. **A phase is removed entirely.** The front end moves to single-product
   journeys and P11 product routing has nothing to route. Twelve phases read its
   outputs. Two ordering constraints (O-09, O-17) exist only because of it.
   **Can a phase be removed, or only emptied?**

9. **The latency budget is cut by 40%** — 120 ms to 72 ms, because the front end
   added 48 ms of its own. The solve's 38 ms cannot survive proportional cutting
   without dropping the evaluation ceiling below correctness. **Where do 48 ms
   come from, who decides, and what is the recorded trade?**

10. **The batch and real-time paths are found to disagree.** The monthly
    reconciliation (§10 criterion 9) shows agreement at 96.4%, below the 99.0%
    floor, concentrated in segment 6. The differences are **not** attributable to
    a named input. Finding the cause requires diffing two traversals of the same
    18 phases on the same client with two `phase_set_id` values.

11. **The Drive Finance card gains a key dimension** — new versus used vehicle —
    taking it from 71 424 to 142 848 cells, while the other three cards are
    unchanged and product 30's solve must handle a dimension no other product
    has.

12. **The cap register is reordered across products.** CAP-0176 moves above
    CAP-0104 for products 10 and 11 only. The answer changes for a population
    nobody has enumerated, on four entry points, and the swap set must be
    produced before the Credit Committee will approve it.

13. **A prohibited-ground finding.** A regulator determines that a feature used
    in three scorecards, 12 cap entries and 9 campaign trees is a proxy for a
    prohibited ground. It must be removed within **30 days**, the models
    re-fitted without it, and every decision made using it in the last two years
    identified.

14. **ALCO halves the limit budget mid-cycle**, after 4.1 M accounts have been
    scored and ranked but before offers are despatched. The funding line moves;
    351 000 funded becomes 174 000; the decline reason for 177 000 accounts
    changes from nothing to "below the funding line".

15. **The consolidation objective changes** from instalment relief to the
    composite client outcome score, and the anti-harm threshold tightens from 35%
    to 28%. Scenarios that won last month lose this month. A client re-assessed
    in the same quarter gets a different recommendation and will ask why.

16. **A four-year-old overlay is discovered still in force.** A −18 point score
    shift on segment 2, approved for one quarter in 2027, never expired because
    the review date was not enforced at the time. The Committee needs the list of
    every live overlay with its age, approval and expiry; the population it
    currently affects; and **what unwinding it would do**, before deciding.

17. **A bureau is decommissioned and a fourth format arrives** through an
    acquisition. The normalisation in P06 must absorb it without losing an
    adverse item, and 620 000 acquired clients enter entry point 4 with a
    different history shape.

18. **Seven-year replay.** A decision made in 2027 is disputed in 2034. Nine of
    the eighteen phases have been rewritten; three of the twelve teams no longer
    exist; two products have been withdrawn; the reason registry has gained 90
    codes and reclassified 40. The decision must reproduce to the cent under the
    artefacts, parameters and overlay stack in force on its `decision_date`, and
    the plain-language rendering that corresponds to the build that ran must be
    retrievable.

19. **A deletion request** arrives for a client whose decisions are evidence in
    an open ombud matter and in a fairness study. Retention, legal hold and the
    right to erasure collide across 640 records spanning six entry points.

20. **A product is withdrawn but not closed.** Product 21 stops accepting new
    business. P11 must never route to it; P15 must still run for 1.75 M existing
    accounts on entry points 2 and 3; P12 must still price its re-prices on entry
    point 6; and its 38 applicable cap entries must remain live for the existing
    book while being unreachable for new business. **A product that is half
    alive is the state most flows handle worst.**

---

## 12. Out of scope

- **The client-facing journey.** Screens, wording flows, document upload,
  application abandonment and re-entry.
- **Data acquisition transport.** The connectors, retries and circuit breakers
  behind the six external calls. This flow declares budgets, ordering and
  degradation policy; it does not implement the plumbing.
- **Model development.** The flow consumes nine scorecards; it does not fit them.
- **Contract generation, disbursement, servicing and collections.** The decision
  ends at the offer and the record.
- **Settlement execution.** P14 produces instructions; the back office executes
  them and discovers three days later that a quotation expired.
- **The governance harness itself.** Replay, diff, swap-set, certification and
  the regulator pack are specified in project 09; this flow's obligation is to
  emit what they need, and that obligation is in scope and in §5.31.
- **Business credit.** Products 50 and 51 and everything about nested entities
  belong to project 11.
- **The physical deployment topology**, capacity planning and cost model.

---

## 13. Questions the implementation must answer

The measurement instrument. These are deliberately the sharpest in the set,
because this is the project where the answers stop being academic.

**Ownership and deployment**

1. **What is the unit of ownership at this scale?** A phase has up to five
   owners. A decision point has one. A table has one and is read by nine phases.
   A parameter has one and is read by fourteen. Is ownership a property of code,
   of data, of a namespace, or of something the framework does not currently
   have?
2. **What is the unit of deployment?** Eighteen phases, four cadences, twelve
   teams, one artefact. Is it the flow, the phase, the artefact, or the change?
   And whichever it is, how does an in-flight decision complete on one consistent
   set?
3. **Can a team own a subtree?** T12 owns P14 entirely and owns almost nothing it
   reads. Is there a structural boundary that makes that ownership real — one
   that T6 cannot cross accidentally and T12 cannot escape — or is it only a
   convention in a wiki?
4. **What must be true for team A to deploy without team B?** §5.26.5 gives five
   conditions. Can the framework make all five checkable, or does it make some of
   them a matter of discipline?
5. **How is a change's blast radius computed**, statically, in 60 seconds, over
   1 400 nodes, conditioned on entry point and product — and what fraction of
   queries return `unknown` before the answer stops being useful?

**Composition**

6. **How are eight entry points expressed without near-copies?** This is the
   central Q4 question of the project. Applicability per decision point? A phase
   set as data? Conditional phases? And whichever it is, does removing a decision
   point remove it from all eight, or does it have to be removed eight times?
7. **What is a phase, structurally** — if it is anything at all? Is the unit of
   composition a component kind, a namespace, a contract, or merely a word this
   document uses?
8. **Is a phase the right unit of ordering when a capability's internal stages
   have different downstream consumers?** The affordability chain is split across
   P06 and P10 because three phases need its first four links before its verdict
   exists. Is that split expressible without forking the calculation?
9. **How are the two genuine cycles (O-06, O-09) expressed** so that the break is
   declared, visible in the reviewable artefact, and impossible to reintroduce by
   adding a rule on the far side of it?
10. **How is a loop that spans four phases expressed**, with a pass bound, a
    termination guarantee, per-pass value identity and three discarded offers
    retained — without the loop body becoming a copy of the four phases?
11. **How does one value carry three orthogonal axes of version** — actual versus
    hypothetical-per-scenario, adjusted versus unadjusted, and
    entry-point-substituted — with 251 live versions of one concept in one
    decision, unambiguously, in the record?
12. **Is `core.*` enough?** This project consumes the library's capabilities and
    supplies its own calibration throughout. Where did that arrangement chafe,
    and which of the 22 capabilities needed a parameter the library does not
    expose?

**Scale and navigability**

13. **Is static lineage answerable in bounded time over 1 400 nodes?** And is it
    answerable *usefully* — conditioned on entry point, product and loop pass —
    or does the loop make everything reachable from everything?
14. **Can a new engineer find where a number is decided in ten minutes**, when
    `amount_cap` can be set in 132 places? What has to be generated for that to
    be true, and what has to be true of the structure for it to be generatable?
15. **Can an analyst answer "why 48" without an engineer?** Across five hops, four
    phases and three owners, rendered as one sentence.
16. **Is the reviewable artefact still reviewable at 1 400 decision points?**
    400 pages layered into 4-page summaries and a diff — does the layering
    survive contact with a committee, and **who reviews the composition**, which
    is in nobody's slice?
17. **Is a campaign tree node a decision point?** If it is, this flow has 8 200 of
    them and every answer to questions 13, 14 and 15 changes by an order of
    magnitude. If it is not, why not — and what else is not?
18. **How is dead logic distinguished from rare logic** when the denominator is
    entry-point-, product- and segment-specific, and the smallest populations are
    the ones whose protections get deleted first?

**Correctness and evidence**

19. **How is a latency budget divided and enforced across eighteen phases with
    twelve owners**, when one phase's 13 ms cost another phase 13 ms it used to
    have? Is a budget a property the framework can carry, or a spreadsheet?
20. **What happens to determinism when a phase is abandoned on budget?** An
    abandoned phase is a different `phase_set_id`, so two runs of the same inputs
    can differ. Is that acceptable, and if so, what does "deterministic" mean
    here — and how does replay reproduce an abandonment?
21. **How is a degraded decision replayed to its degraded answer**, with the
    degradation as an asserted input rather than an annotation, seven years later?
22. **How much evidence is enough**, when the honest answer costs 2.15 TB a year
    and the cheap one costs 0.94 TB, and the difference is whether a rejected
    consolidation scenario can be explained eighteen months later?
23. **Can certification cost be made proportional to blast radius?** If not, the
    weekly-cadence teams stop shipping weekly and start writing literals, and the
    framework will have caused the failure it was built to prevent.

**The scale wall itself**

24. **At what point should one flow become two?** This document asserts that all
    eight entry points belong in one artefact. Is that right? What is the boundary
    — number of decision points, number of owning teams, number of entry points,
    divergence of budgets, absence of shared intermediates?
25. **What evidence would show the boundary had been crossed?** Candidates, and
    the implementation should say which it believes: independent deployability
    falling below the 80% floor; blast-radius `unknown` rates rising above 2%;
    certification time growing superlinearly in the number of phases; the
    reviewable artefact's diff exceeding what a committee reads; N1 failing for
    new joiners; the count of reduced-form phases (◐) growing; or the first
    appearance of a second implementation of a phase for one entry point.
26. **And if the wall is found, what is on the other side?** Two flows sharing
    eleven phases is not obviously better than one flow with eight entry points —
    it may simply move the composition problem into a place where nobody owns it
    at all. The implementation should say what it would split, along which seam,
    and what the split would cost in the four things this document says matter:
    entry-point agreement, one decision record, blast-radius answerability, and
    the ability to explain a decision to the person it was made about.
