# 08 — Collections treatment assignment

> Fictional. The Bank, its products, thresholds, table dimensions and volumetrics
> are invented for this repository. Regulatory mechanisms referred to are the
> published public ones; every number attached to them is illustrative.

---

## 1. What this is

Every weekday morning, before the contact centre opens, the Bank must decide what
happens today to each of **2.3 million delinquent accounts**. Not what they are
worth, not what they will pay — what is *done* to them: a message, a call, a
visit, a notice, a hand-off to an agency, a settlement offer, a legal instruction,
or nothing at all.

The decision is not hard because the rules are clever. It is hard because of two
things that most credit decisions do not have.

The first is **time**. Almost every rule in collections is a rule about elapsed
days or a count within a window: days past due, days since last payment, days
since last contact attempt on this channel, business days since a prescribed
notice was delivered, months since the arrangement started, promises broken in
the last 90 days, times cured in the last twelve months, days until the debt
prescribes. Nothing is a snapshot. A rule that says "SMS" says it only because of
what did or did not happen on a set of previous dates.

The second is **history**. Today's treatment is not a fresh decision; it is the
next position in a sequence that began when the account first went into arrears.
An account that received an SMS yesterday and did not respond must escalate, not
repeat. An account that promised R900 on Friday must be left alone until Friday.
An account that broke that promise must be treated differently from one that was
never reached at all. The Bank's own past actions are inputs, and they must be
reconstructible years later when somebody asks why a client was telephoned eleven
times in a fortnight.

On top of that sits a third difficulty that is not about any one account: the
treatment matrix recommends far more work than exists. It will ask for roughly
**310 000 agent call attempts** on a day when **46 000** are available. Something
must rank and ration, fairly, and must be able to say of any untouched account
why it was not touched.

The flow runs daily over the whole delinquent book, re-runs intraday after
payments post, and is also called one account at a time, in under 400 ms, by an
agent on a live call who needs a revised arrangement assessed while the client
waits.

---

## 2. Why it is in this set

| Question | How this project stresses it |
|---|---|
| **Q2 tables** | ● The treatment matrix is 8 × 6 × 7 × 4 × 4 = **5 376 cells**, each carrying a treatment and three modifiers. It is owned by non-engineers, replaced monthly, split by champion/challenger, and sparse — a large minority of cells are never exercised. Validating it, diffing it, versioning it and attributing a cell to an outcome are all unsolved here. |
| **Q5 core component set** | ● This is the sharpest Q5 test in the slate. Sequence position over time — where an account is in an escalation path, what resets it, what carries across a roll — is neither a scorecard, nor a decision table, nor a decision tree. Either a fourth kind falls out, or it is written by hand 40 times. |
| **Q3 parameters** | ● The matrix changes monthly; contact intervals change when Compliance says so; capacity changes daily; champion/challenger allocations change per experiment and must remain stable per account while they run. On top of all of it sit post-model overlays (`core.adjustments`) — score shifts and matrix intensity dials, applied without reissuing the artefact they modify, under a faster approval and with a mandatory expiry. Five cadences, five owners, one flow. |
| **Q7 audit** | ○ An ombud complaint of harassment requires the full contact history *and the rule that authorised each contact*. A regulator asks whether a prescribed notice preceded legal action in every case, for every case. |
| **Q1 reuse** | ○ `core.affordability` is consumed here in a distressed mode that the granting projects never use. Same logic, different question: not "can they take on this debt" but "will this arrangement hold". |
| **Q6 codebase** | ○ Eight arrears buckets × four product families × three operating areas (early collections, late collections, recoveries and legal), each with an owner who changes things independently. |

Four things in particular are worth watching an implementer do.

**The matrix against capacity.** Per-account logic produces a recommendation;
a population-level constraint decides whether it happens. Project 07 has the same
shape at a portfolio budget level, but here the constraint is multi-dimensional
(nine capacity pools), has fairness floors, and must not correlate with the
experiment split.

**Suspensions that override everything but stay individually attributable.**
The easy implementation short-circuits on the first suspension and reports it.
That fails the second audit question in §9, which is about a client under debt
review who was also outside contact hours and had withdrawn SMS consent.

**Temporal state that must be recomputable.** If the sequence position is a
mutable field that yesterday's run overwrote, there is no replay. If it is
derived from the event log, the derivation itself is logic that must be versioned
and must honour what was *known* on the day, not what is known now.

**Overlays that may tilt commerce but never statutory protection.** Collections
is the project where the overlay mechanism of `core.adjustments` meets rules that
must not be dialled. "Escalate bucket 3 by one intensity level for six weeks" is
a legitimate overlay approved in a Monday strategy meeting. "Relax the contact
frequency cap for six weeks" is not an overlay at all, and the system must make
that impossible rather than merely discourage it.

---

## 3. Actors

| Actor | Responsibility |
|---|---|
| **Collections Strategy** | Owns the treatment matrix, the contact interval rules, the arrangement grids and the champion/challenger programme. Analysts, not engineers. Change the matrix monthly. |
| **Collections Operations** | Runs the contact centre. Owns capacity, queue definitions, agent skills and the dialler. Consumes the daily output; cannot change the matrix. |
| **Recoveries and Legal** | Owns pre-legal notices, legal handover criteria, the attorney panel, and the write-off recommendation thresholds. |
| **Agency Management** | Owns third-party agency contracts, monthly contracted volumes, commission bands and placement quality. |
| **Regulatory Compliance** | Owns the suspension rules, contact time and frequency restrictions, notice requirements, prescription handling and the script registry. Sign-off is by evidence, not code. |
| **Credit Risk Policy** | Owns arrangement minimums, settlement discount authority levels and the provisioning interaction. |
| **Model team** | Owns the collections roll/cure score, the recovery-value estimate and the cost-to-collect estimate. Does not own the overlays applied over them. |
| **Model Risk and Validation** | Approves score-level overlays, monitors the base score with the overlay stack disabled, and enforces overlay expiry. |
| **Collections Credit Forum** | Weekly. Approves matrix intensity overlays and their expiry dates. A faster path than the monthly matrix release, deliberately, and therefore the one that needs the tightest limits. |
| **Complaints and Ombud liaison** | Answers, months later, why a specific client was contacted on a specific day. |
| **Internal Audit** | Samples settlements and checks the authority under which each was granted. |
| **Finance** | Consumes write-off recommendations and settlement impairment effects. |

---

## 4. Inputs

### 4.1 Locally declared vocabulary

The library (project 00 §4) does not publish collections names. This project
declares the following and every other spec that touches collections uses them
unchanged.

| Name | Type | Meaning |
|---|---|---|
| `account_id` | int64 | One credit agreement. A `client_id` may hold up to 9. |
| `days_past_due` | int16 | Days since the oldest unpaid contractual instalment fell due. |
| `arrears_bucket_code` | int8 | 1..8, see §4.3. Derived from `days_past_due`, but not only from it. |
| `arrears_amount` | float64 | Contractual arrears, excluding unbilled interest. |
| `collections_score` | float64 | Output of the roll/cure model. |
| `collections_band_code` | int8 | 1 (most likely to cure) .. 6 (most likely to roll). |
| `contact_band_code` | int8 | 1 responsive, 2 intermittent, 3 silent, 4 unreachable. |
| `balance_band_code` | int8 | 1..7, see §4.3. |
| `product_family_code` | int8 | 1 unsecured term, 2 revolving, 3 secured asset, 4 business. |
| `treatment_code` | int8 | 0..13, see §5.4. |
| `treatment_intensity` | int8 | 1 (lightest) .. 5. |
| `treatment_instance_id` | int64 | One assignment of one treatment to one account on one date. The join key for all feedback. |
| `episode_id` | int64 | One delinquency episode: opened when `days_past_due` first exceeds 0, closed on cure. |
| `path_position` | int8 | Position within the escalation sequence for the current episode. |
| `suspension_code` | int16 | See §5.2. |
| `promise_id` | int64 | One promise to pay. |
| `arrangement_id` | int64 | One payment arrangement. |
| `cohort_code` | int16 | Champion/challenger cell membership. |
| `matrix_version` | int32 | Which treatment matrix was in force. |
| `matrix_cell_id` | int32 | Which of the 5 376 cells was read. |
| `adjustment_set_version` | int32 | Which overlay stack was in force on `decision_date`. |
| `overlay_id` | int16 | One named, approved, effective-dated overlay. |
| `collections_score_unadjusted` | float64 | The score before any overlay. Always carried beside the adjusted value. |
| `recovery_estimate` | float64 | Expected rand recovered over the next 24 months under continued normal collections. |
| `cost_to_collect` | float64 | Expected rand cost of that recovery. |
| `prescription_date` | date | The date on which the debt becomes unenforceable if nothing interrupts. |
| `non_selection_reason_code` | int16 | Why an account received no treatment today. |

`decision_date`, `client_id`, `product_code`, `channel_code`, `instalment`,
`risk_grade` and the consent vocabulary come from the library unchanged.

### 4.2 Sources

| Source | Shape | Volume | Freshness | Nullability |
|---|---|---|---|---|
| Account master | One row per account | 2.3 M daily | T-1 close, lands 04:05 | Complete; `instalment` null for 41 000 revolving accounts in permanent over-limit |
| Payment history, monthly | 12 values per account | 27.6 M | T-1 | Ragged for accounts under 12 months old (312 000 accounts) |
| Payment transactions, 90 days | 0..140 per account, p99 = 11 | ~14 M | Intraday, four payment rails, one of which lags 36 h | — |
| Bucket roll history | 24 monthly observations per account | 55.2 M | T-1 | Short for new accounts |
| Contact history, 90 days | 0..400 events per account, p50 = 6, p99 = 62 | ~180 M events | Dialler hourly; agency files T+2; field agents T+1 | Outcome null for 4.1% of dialler records |
| Promises | 0..12 per account | 1.4 M open or recently closed | Intraday | — |
| Arrangements and their instalment-level performance | 0..6 per account, 0..24 instalments each | 390 000 live | T-1 | — |
| Status and legal feed | One row per status event | ~40 000 daily | Debt review registry daily 03:40; deceased weekly and ad hoc; litigation from the attorney panel weekly | Registry occasionally short-delivers; see §5.2 |
| Consent and preferences (`core.consent`) | One view per client | 2.3 M | T-1 | 6.8% of accounts have no valid mobile number; 31% no email; 18% app-active |
| Hardship declarations | One row per declaration | 62 000 live | Intraday | — |
| Capacity feed | One row per pool per day | 9 pools | 04:00, revised 11:00 | Agent hours revised downward on 14% of days |
| Parameter and table set | See §6 | — | Effective-dated, resolved on `decision_date` | — |

Bureau data is **not** refreshed for the whole book. A monthly bureau refresh
covers accounts in buckets 5–8 and any account with an arrangement (about
640 000). Everything else carries a bureau view up to 13 months old, and the
flow must treat that as a distinct state from "no bureau view".

### 4.3 Banding

Bands are parameters, not constants, and their edges move. They are stated here
because the matrix dimensions depend on them.

| `arrears_bucket_code` | Definition | Accounts |
|---|---|---|
| 1 | 1–14 days past due | 780 000 |
| 2 | 15–29 | 410 000 |
| 3 | 30–59 | 330 000 |
| 4 | 60–89 | 205 000 |
| 5 | 90–119 | 148 000 |
| 6 | 120–179 | 171 000 |
| 7 | 180–364 | 143 000 |
| 8 | 365+, including charged-off but collectable | 113 000 |

| `balance_band_code` | Outstanding balance |
|---|---|
| 1 | < R2 500 |
| 2 | R2 500 – R9 999 |
| 3 | R10 000 – R24 999 |
| 4 | R25 000 – R49 999 |
| 5 | R50 000 – R99 999 |
| 6 | R100 000 – R249 999 |
| 7 | ≥ R250 000 |

Bucket assignment is not a pure function of `days_past_due`. An account under a
performing arrangement is held at the bucket it entered the arrangement in for
reporting, while its `days_past_due` continues to age; an account that has cured
twice in six months is floored at bucket 3 for treatment purposes. Both of these
are policy rules, both are changed by Collections Strategy, and both mean the
bucket must be computed and recorded, not read.

---

## 5. The flow

### 5.1 Account state assembly

**Precondition.** The account master, the status feed and yesterday's assignment
record have all landed.

**Determines.** A single assembled state per account, sufficient for every later
stage, with every temporal quantity resolved against `decision_date` and not
against the wall clock.

Per account the assembly must establish: `arrears_amount`, `days_past_due`,
`arrears_bucket_code`, outstanding balance, limit where revolving, `product_code`
and `product_family_code`, contractual `instalment`, last payment amount and
date, payments made in the last 30/60/90 days, the 12-month payment pattern as a
month-by-month paid/partial/missed sequence, roll history between buckets over
24 months, `times_cured_12m` and `times_cured_24m`, contact history by channel
over 90 days with each attempt's outcome, promises made with their kept, partial
and broken status, arrangements past and present with per-instalment performance,
legal status, agency placement status, hardship flags, and the last 14 days of
treatments assigned by this flow.

Contact outcomes are a closed set, and the distinction between them carries
treatment consequences: **no answer**, **engaged tone or call failure**, **wrong
number**, **right party contacted — promise**, **right party contacted —
refusal**, **right party contacted — dispute**, **right party contacted — no
commitment**, **third party answered**, **number invalid**, **payment taken on
call**. "No answer" and "wrong number" both look like failure and must not be
treated alike: three wrong-number outcomes on a channel invalidate the contact
point rather than escalating the account.

**Must be recorded.** The assembled state, or a deterministic means of rebuilding
it, with the as-of watermark of every contributing feed. A state assembled at
04:20 using a payment file that was complete to 23:50 the previous night must be
distinguishable from the same account assembled at 13:30 with two more payments
in it.

**Hard part.** Every quantity here is a window function over an event log of
180 million rows, and the windows are not the same: 30/60/90 days for payments,
90 days for contacts, 12 and 24 months for patterns, business days for notices,
rolling 7-day windows for frequency caps. Business days are calendar-dependent
and the calendar is a parameter.

### 5.2 Regulatory and status suspensions

**Precondition.** Assembled state.

**Determines.** The set of suspensions in force for this account on
`decision_date`, each with its scope, and the resulting set of treatments that
are permitted at all today.

These are evaluated **first** and they override everything downstream. A
suspended treatment is not downgraded to a cheaper one by default; the rule says
per suspension whether it blocks outright, blocks a channel, or forces a
downgrade.

| Code | Suspension | Scope | Expiry |
|---|---|---|---|
| 101 | Debt review application received | All collections contact except statutory notices | On stage change; see below |
| 102 | Debt review proposal issued | All except arrangement servicing | Event-dependent |
| 103 | Debt review court order granted | All normal collections; rearrangement servicing only | Until order set aside or clearance issued |
| 104 | Debt review rearrangement in default | Partial — releases 103 after the prescribed process | Computable from default date |
| 105 | Administration order | All normal collections | Until discharge |
| 106 | Insolvency or sequestration | All | Until rehabilitation |
| 107 | Deceased, estate under administration | All client contact; executor correspondence only | Until estate finalised |
| 108 | Active internal complaint | Contact and legal | 15 business days from logging, extendable |
| 109 | Active ombud or regulator referral | Contact and legal | Until case closed |
| 110 | Dispute raised on the debt | Legal and agency handover; contact permitted in writing | Until dispute resolved |
| 111 | Hardship arrangement in good standing | All collections while performing | On first missed arrangement instalment |
| 112 | Prescribed notice period unexpired | Legal handover only | **Computable**: delivery date + 10 business days |
| 113 | Litigation in progress | All non-legal treatments | Until judgment or withdrawal |
| 114 | Prescription reached | All collection activity; see below | Permanent unless interrupted before the date |
| 115 | Outside permitted contact hours or days | Channel-specific, time-specific | Same day, at the window boundary |
| 116 | Channel consent withdrawn | That channel | Until consent re-obtained |
| 117 | Contact frequency cap reached | Contact channels | Computable from the window |
| 118 | Written communication only, at client request | All voice and field | Until withdrawn by the client |
| 119 | Promise in force | All except the permitted promise reminder | Promise date + 2 business days |
| 120 | No valid contact point on any permitted channel | Contact channels | Until a contact point is validated |

Four requirements attach to this stage and they are the ones that fail in
practice.

**Every suspension that applied must be individually attributable.** Not the
first one found. If four suspensions were in force, the record names four, each
with: the code, the source record and its identifier (the debt review case
number, the complaint reference, the consent withdrawal timestamp, the notice
delivery proof), the moment it took effect, its scope, and whether it blocked or
downgraded. An implementation that stops at the first match cannot answer §9.2.

**Every suspension's expiry must be computable or explicitly event-dependent.**
For each suspension the flow states either the date it lifts — notice delivered
14 August plus 10 business days is 28 August — or that it lifts on an event, and
names the event. "Suspended indefinitely" is not an acceptable state; "suspended
until the debt review registry reports a stage change, reviewed daily" is.

**Prescription is not merely a suspension.** Where the debt has prescribed, it is
unenforceable, and the constraint is on what may be *said* as much as on what may
be done: no demand may be made, no acknowledgement may be solicited, the debt may
not be handed to a third party for collection, and if the client volunteers a
payment a specific disclosure is required first. The `prescription_date` must be
computed per account from the event history — the later of last payment, last
acknowledgement of the debt (including a recorded verbal acknowledgement on a
call), and service of legal process, plus the prescription period — and it must
be recomputed every day, because any qualifying event resets it. Accounts within
**60 days** of `prescription_date` raise a pre-prescription flag, which interacts
with §5.9: an account worth suing that is about to prescribe must be prioritised
for legal handover, and that priority competes for the same scarce capacity as
everything else.

**Nothing in this stage is overlayable.** The post-model overlay mechanism used
in §5.3 and §5.4 has no reach here. Suspension rules, notice periods, permitted
contact hours, frequency caps and prescription handling are Compliance-owned
parameters on Compliance's release path, and the authoring of an overlay that
names any of them fails. An overlay may make the Bank do less than the matrix
says; it may never make the Bank do more than the law allows. Where a commercial
overlay and a suspension would disagree, the suspension wins by construction
rather than by evaluation order, and the evidence must show that the suspension
was evaluated against the *overlaid* recommendation, not against the raw one.

**Must be recorded.** The full suspension set, the permitted-treatment set after
suspensions, and the parameter versions of the suspension rules — which are
versioned independently of `adjustment_set_version` and must be seen to be.

### 5.3 Risk assessment

**Precondition.** Assembled state; suspensions do not gate this stage, because
the estimates are needed for ranking even where no treatment is permitted.

**Determines.**

1. **`collections_score`** — a roll/cure model over **28 characteristics**,
   producing the probability of rolling to the next bucket within 30 days and
   the probability of curing within 30 days. Characteristics include current
   bucket, times cured in 12 months, the 12-month payment pattern encoded as
   recency-weighted ratios, ratio of last payment to contractual instalment,
   days since last payment, arrears-to-balance ratio, utilisation at default for
   revolving, contact responsiveness over 90 days, right-party-contact rate,
   promise-kept rate, broken promises in 90 days, arrangement history, age of
   the account, product family, salary-date alignment, and the count of other
   internal accounts of the same client in arrears. Roughly 30% of
   characteristics are null for some part of the book, and null is a bin.
2. **`collections_band_code`** — the score banded into 6.
3. **`recovery_estimate`** — expected rand recovered over 24 months under
   continued normal collections, from a curve keyed on bucket, band, product
   family and balance.
4. **`cost_to_collect`** — the expected cost of achieving that recovery, from the
   per-treatment cost schedule and the expected number of treatments.
5. **`contact_band_code`** — responsiveness over 90 days: right-party contacts
   achieved, attempts required per contact, and channel-level validity.

**Overlays on the score and its banding.** The collections score is revalidated
roughly every eighteen months, but performance drifts faster than that and
strategy sometimes wants to tilt for a period. The response is an overlay applied
through `core.adjustments`, never an edit to the score itself. Four kinds must be
expressible here:

| Kind | Applies to | Illustrative overlay |
|---|---|---|
| Score shift | `collections_score` | −22 points on accounts placed with an agency in the last 12 months |
| Scaling change | the score-to-odds relationship | points-to-double-the-odds from 20 to 23 for the revolving segment |
| Odds multiplier | the modelled roll probability | roll probability × 1.28 on the unsecured book for a deteriorating quarter |
| Band boundary shift | `collections_band_code` | move the band 3/4 edge by 0.03 of roll probability |

The consequences run straight into the rest of the flow, and they are the reason
this is stated as a requirement rather than left implicit:

- **A band overlay re-keys the treatment matrix.** Moving a band edge moves
  accounts between cells, which changes the treatment, which changes demand on
  capacity pools. A one-notch band shift on the unsecured book moves roughly
  74 000 accounts and adds about 9 000 agent call attempts to a pool of 46 000.
  The overlay's approval must therefore show its volume effect, not only its
  risk rationale.
- **The unadjusted value survives.** `collections_score_unadjusted` and the
  unadjusted band are carried beside the adjusted ones, on every account, every
  day. "What would we have done without the overlay" is asked at every forum that
  approves one.
- **The flow must be runnable with the overlay stack disabled**, which is how the
  base score's own performance is monitored (project 00 §7.6). See §8.
- **Overlays stack in a declared order** that is part of the overlay set's
  definition. A segment score shift and a product odds multiplier compose
  differently in each order and the order is not permitted to be emergent.

**Must be recorded.** Score adjusted and unadjusted, per-characteristic
contributions (`core.scorecard` publishes these and they are required output, not
a debugging nicety), band adjusted and unadjusted, model version,
`adjustment_set_version`, the ordered list of `overlay_id`s that applied with the
effect of each, and both estimates with the curve version that produced them.

### 5.4 The treatment matrix

**Precondition.** Bands from 5.1 and 5.3; suspensions from 5.2; cohort from
§5.12.

**Determines.** The treatment the matrix recommends for this account today,
before sequence and capacity are applied.

The matrix is keyed on five dimensions:

`arrears_bucket_code` (8) × `collections_band_code` (6) × `balance_band_code` (7)
× `contact_band_code` (4) × `product_family_code` (4) = **5 376 cells**.

Each cell carries a treatment code and three modifiers:

| Value | Range | Meaning |
|---|---|---|
| `treatment_code` | 0..13 | What to do |
| `treatment_intensity` | 1..5 | Script severity, agent skill required, call attempt profile |
| `permitted_retries` | 0..4 | How many times this treatment may be repeated before the sequence must escalate |
| `cooling_off_days` | 0..21 | Minimum days before this treatment may recur on this account |

The treatments:

| Code | Treatment | Unit cost | Capacity pool |
|---|---|---|---|
| 0 | No action | — | — |
| 1 | Automated SMS | R0.09 | SMS |
| 2 | Automated email | R0.004 | Email |
| 3 | In-app message | R0.00 | In-app |
| 4 | Automated voice message | R0.28 | IVM |
| 5 | Agent call, low intensity | R12.80 | Early agents |
| 6 | Agent call, standard | R12.80 | Early or late agents |
| 7 | Agent call, high intensity (negotiator) | R21.40 | Late agents |
| 8 | Field visit | R420 | Field |
| 9 | Third-party agency handover | 12–22% commission | Agency |
| 10 | Pre-legal prescribed notice | R38.50 | Notices |
| 11 | Legal handover | R1 250 upfront | Legal |
| 12 | Settlement offer | Treatment cost of the carrying channel | Carrying channel |
| 13 | Write-off recommendation | — | — |

**Ownership and cadence.** Collections Strategy owns the matrix and changes it
monthly, on the first business day, with mid-month patches to individual cells
after a portfolio review. A matrix version is a whole artefact: a cell change is
a new version of the matrix, not an edit in place.

**Intensity dials over the matrix.** A matrix release is monthly and carries a
full review. Collections Strategy also needs to move within the month, and the
mechanism for that is an overlay over the matrix — not a reissued matrix and not
a cell patch. Three kinds are required:

| Kind | Effect | Example |
|---|---|---|
| Intensity dial | `treatment_intensity` shifted, or `permitted_retries` and `cooling_off_days` scaled, over a declared scope | "Escalate bucket 3 by one intensity level for six weeks" |
| Treatment suppression or substitution | A treatment code is disabled over a scope, optionally falling back to a named alternative | "Suspend field visits for the quarter"; "substitute voice message for agent call in balance band 1" |
| Allocation weighting dial | The ranking weight or reserved share of a capacity pool, over a scope | "Dial agent-call allocation up 20% in balance bands 6–7" |

Each overlay carries, without exception: an `overlay_id`, a description, an
owner, an approval reference, a rationale, an effective-from date, an
effective-to date and a **mandatory review date**; a declared scope stated in the
matrix's own dimensions plus channel and cohort; and a position in the declared
stacking order. Applying an overlay outside its scope is an error, not a silent
no-op. An overlay reaching its review date without renewal must surface loudly:
the failure mode this mechanism exists to prevent is a six-week tightening
imposed after one bad quarter and still quietly in force four years later, with
nobody left who can explain the collections mix.

Overlays are approved weekly by the Collections Credit Forum; matrix versions are
approved monthly with a full review. The faster path is the point and also the
risk, so the limits on what an overlay may do are narrow: an overlay may move
intensity, retries, cooling-off, treatment availability and allocation weight. It
may not introduce a treatment into a cell that the matrix does not permit, and it
may not touch anything in §5.2.

**Overlays may never weaken a statutory protection.** This is an asymmetry and it
must be *enforced*, not documented and trusted. No overlay, at any authority
level, may relax a regulatory suspension, shorten or waive a prescribed notice
period, raise or disapply a contact frequency cap or a permitted-contact-hours
rule, or alter prescription handling. Those parameters belong to Compliance and
change only through Compliance's own release path with its own approval. An
overlay definition that names one of them is rejected at authoring time, not at
run time, and the attempted definition is itself recorded. Overlays may tighten
commercial intensity in either direction; they may only ever be *more*
conservative than a statutory rule, never less.

**Sparsity is a real problem, not a theoretical one.** Of the 5 376 cells, about
**41%** see fewer than 50 accounts in a month and about **12%** see none at all
(bucket 8 × band 1 × balance band 7 × business, for instance, is empty most
months). Validation must therefore distinguish a cell that is wrong from a cell
that is merely unused, and a new matrix version must be checkable without waiting
for traffic.

**Must be recorded.** `matrix_version`, `matrix_cell_id`, the five key values
that produced it, all four cell values **as read from the matrix and as modified
by overlays**, the ordered `overlay_id`s that applied with the effect of each,
`adjustment_set_version`, and — where a key value fell on a band edge or outside
the banded range — that fact.

**Hard part.** A 5 376-cell artefact authored in a spreadsheet by analysts,
reviewed monthly by people who need to see what changed, split by cohort so that
three versions may be simultaneously in force for different accounts, overlaid by
a separately-approved and separately-expiring stack of intensity dials, and
attributable cell-by-cell to an outcome eighteen months later.

### 5.5 The treatment path over time

**Precondition.** The matrix recommendation, the permitted-treatment set, and the
account's own treatment history.

**Determines.** What is actually appropriate today given what has already been
tried — which is frequently not what the matrix said.

This is the stage that makes collections different from every other project in
this set. The matrix answers "what kind of account is this today". It does not
answer "what have we already done, and what does that make today's action".

**The episode.** A delinquency episode opens when `days_past_due` moves from 0 to
positive and closes when the account has been at 0 for 5 consecutive days.
`path_position` is defined within an episode. An account may have several
episodes in a year; the 6.1% of accounts with four or more episodes in twelve
months are treated as chronic re-agers under a separate path.

**Minimum intervals and maximum attempts.** Per channel and per bucket, three
things are specified: the minimum days between two contacts of that type, the
maximum attempts on that channel before the sequence must escalate, and the
maximum contacts in a rolling window. Illustratively:

| Channel | Min interval (buckets 1–2) | Min interval (buckets 3–5) | Min interval (buckets 6–8) | Max attempts before escalation |
|---|---|---|---|---|
| SMS | 4 days | 3 days | 3 days | 3 |
| Email | 3 days | 2 days | 2 days | 4 |
| In-app | 2 days | 2 days | 2 days | 5 |
| Voice message | 5 days | 4 days | 3 days | 2 |
| Agent call | 2 days | 1 day | 1 day | 4 per position |
| Field visit | — | 21 days | 14 days | 2 |

Over the top of this sit absolute caps that no bucket may exceed: **3 contact
attempts of any kind per day**, **2 spoken-contact attempts per day**, **10 per
rolling 7 days**, **24 per rolling 30 days**. Caps are per client, not per
account, which matters for the 340 000 clients with more than one delinquent
account — three accounts each politely observing a per-account cap produce a
harassment complaint.

**Escalation.** Where the permitted retries for the current position are
exhausted and the outcome was non-engagement, the sequence advances. Escalation
is monotonic within an episode: **treatment intensity may not decrease** except
after a reset event. An account that has had a field visit is not sent an SMS
next; an account that has been handed to an agency does not return to the early
contact path while the placement stands.

**Resets.** Five events reset the sequence, and they reset it differently:

| Event | Effect |
|---|---|
| Qualifying payment (≥ 50% of one contractual instalment, or ≥ 30% of arrears) | `path_position` returns to the entry position of the current bucket; channel attempt counters clear; intensity ceiling drops by one |
| Promise captured | Sequence held (suspension 119) until promise date + 2 business days; position unchanged |
| Cure (`days_past_due` = 0) | Episode closes; re-entry governed below |
| Dispute raised | Sequence held; account moves to the dispute path; legal and agency blocked |
| Arrangement activated | Sequence suspended while the arrangement performs; arrangement-servicing path applies |

A *partial* payment that does not qualify does not reset anything, and this is a
deliberate, contested policy setting that Collections Strategy changes.

**Rolling mid-sequence.** When an account rolls to a worse bucket while part-way
through a sequence, the rule is asymmetric and must be stated explicitly because
both halves are needed:

- `path_position` moves to the entry position of the **new, worse** bucket.
- Channel attempt counters and the rolling-window contact counts **carry across**
  the roll. Fatigue is a property of the client, not of the bucket.
- The intensity floor carries: the new position's intensity is the greater of
  the new bucket's entry intensity and the intensity already delivered.
- Cooling-off periods in flight are honoured across the roll.

A roll to a *better* bucket, which happens through partial payment, does not
de-escalate on its own; only a qualifying payment does.

**Re-entry after curing.** An account that cures and re-defaults is not a new
account:

| Prior cure history | Re-entry |
|---|---|
| No cure in 12 months | Normal entry at position 1 of bucket 1 |
| Cured once, re-defaulted within 90 days | Enter at position 3; skip automated-only positions |
| Cured twice in 12 months | Enter at position 3, bucket floored at 3 for matrix purposes |
| Cured three or more times in 12 months | Chronic re-ager path: agent contact from day 1, arrangement offered before escalation, promises limited |

**Two requirements that the rest of this stage rests on.**

The account's treatment history **is an input to today's decision**. Not a log
written afterwards. Yesterday's assignment, its outcome, the attempt counters,
the position and the intensity ceiling are read at the start of today's run and
their absence changes the answer.

The history must be **reconstructible**. Given an account and a past date, it
must be possible to re-derive the sequence position that was in force on that
date, from the event log, using the rules that were in force then. This has a
sharp consequence for late-arriving data: an agency contact outcome that lands
two days after the contact happened means the position computed on the day was
computed without it. The flow must therefore distinguish the **as-known-on-the-day**
derivation, which is what must be reproducible for audit and replay, from the
**as-at-now** derivation, which is what analysis wants. Both are needed and they
disagree on roughly 2.4% of account-days.

**Must be recorded.** `episode_id`, `path_position` before and after, the
escalation or hold decision and its rule, every interval and cap that was tested
and whether it bound, and the reset event if any.

### 5.6 Payment arrangements

**Precondition.** Client engagement — a right-party contact, an inbound call, or
a digital self-service request.

**Determines.** Whether a proposed arrangement may be accepted, at what minimum,
for how long, and under whose authority.

Six arrangement types:

| Type | Description | Max duration |
|---|---|---|
| A1 | Reduced instalment | 6 months |
| A2 | Arrears catch-up, arrears spread on top of the contractual instalment | 12 months |
| A3 | Payment holiday, interest capitalised | 3 months |
| A4 | Term extension with re-scheduled instalment | 24 months added |
| A5 | Interest concession | 12 months |
| A6 | Hardship plan, with mandatory review | 12 months, reviewed at 6 |

**Affordability in a distressed mode.** Every arrangement is tested with
`core.affordability`, but the question is not the granting question. In granting,
the test is whether new debt can be added to a stable position. Here the position
is not stable, and three things differ:

- **Expense treatment.** Statement-derived expenses take precedence over declared
  ones where statements exist, and the statutory minimum-expense norm operates as
  a plausibility floor for recording purposes rather than as a disqualifier —
  a client whose real expenses are below the norm should not be refused an
  arrangement they can afford.
- **Income evidence.** Declared income is frequently unverifiable; a substantial
  share of the book is no longer employed where the account was opened. The
  haircut tiers are different, and "income could not be established" must not
  collapse into "income is zero".
- **The test itself.** Not "can they afford new credit" but "is this arrangement
  sustainable": residual after the arrangement instalment must be at least
  **R350** and at least **5%** of net monthly income; arrangement-instalment to
  discretionary income must not exceed **85%** (against 65% in granting).

Using the same capability for both is the point and the risk. The distressed mode
must be a parameterisation of one capability, not a fork of it, and the
evidence must state which mode applied.

**Minimums.** As a percentage of the contractual instalment and as an absolute
floor:

| Product family | Min % of contractual instalment | Absolute floor | Max concurrent |
|---|---|---|---|
| Unsecured term | 40% | R150 | 1 |
| Revolving | 30% of minimum payment | R100 | 1 |
| Secured asset | 60% | R450 | 1 |
| Business | 65% | R1 200 | 1 |

**Counts and limits.** At most **3 arrangements in 24 months**; at most **2
consecutive failed arrangements** before the path locks to escalation; a fourth
arrangement requires L3 authority; an arrangement may not be granted within 30
days of a failed one without L2 authority.

**Authority levels for concessions beyond the grid.**

| Level | Who | Concession permitted |
|---|---|---|
| L1 | Agent | Within the grid |
| L2 | Team leader | Up to 15% below the minimum instalment; duration +3 months |
| L3 | Collections manager | Up to 25% below; duration +6 months; 4th arrangement |
| L4 | Head of Collections | Up to 40% below |
| L5 | Credit Committee | Anything else, case by case |

**Must be recorded.** The arrangement terms, the affordability assessment in full
with its mode and the versions of the norms used, the minimum that applied and
the grid version, the authority level required and the authority actually
obtained with approver identity and timestamp, and the justification code.

### 5.7 Settlement and discount offers

**Precondition.** Bucket, recovery estimate, cost to collect, provision already
raised, time since default, and the prescription position.

**Determines.** Whether a discounted settlement may be offered, at what discount,
and under what authority.

A settlement discounts the balance in exchange for immediate payment or payment
over a short period. The permitted discount comes from a grid keyed on
`arrears_bucket_code` × recovery band × provision status, with authority levels
above it:

| Bucket | Standard discount (L1) | L2 | L3 | L4 |
|---|---|---|---|---|
| 4 | 5% | 10% | 15% | 20% |
| 5 | 10% | 15% | 22% | 30% |
| 6 | 15% | 22% | 30% | 40% |
| 7 | 25% | 35% | 45% | 55% |
| 8 | 40% | 50% | 62% | 75% |

Conditions attach: immediate settlement earns the full discount; settlement over
three instalments within 90 days earns the discount less 8 percentage points; a
lapsed settlement arrangement voids the discount and returns the full balance,
which must be disclosed at offer.

**The justification must be recorded, not merely the offer.** Three things go
into the record: the discount offered, the authority code under which it was
offered and by whom, and the **expected-recovery comparison** that justified it —
the present value of continued normal collections (`recovery_estimate` less
`cost_to_collect`, discounted at the policy rate of 12% per annum) against the
settlement proceeds. Internal Audit samples settlements and checks exactly this
(§9.3). A settlement granted at L1 where the grid required L3 is an audit
finding; a settlement where the comparison cannot be reproduced is a worse one.

**Prescription interacts here and the interaction is easy to get wrong.** A
settlement offer invites an acknowledgement of the debt, and an acknowledgement
interrupts prescription. Where the debt has prescribed, no offer may be made at
all. Where the account is within 60 days of `prescription_date`, offers require
L3 authority and a specific disclosure, and the fact that the discussion could
have interrupted prescription must be on the record.

### 5.8 Promise-to-pay handling

**Precondition.** A right-party contact in which the client commits.

**Determines.** Whether the promise is accepted, what it suspends, and how its
outcome reshapes the path.

A promise carries an amount, a date, the channel it was captured on, the agent or
system that captured it, and up to 3 scheduled parts. Acceptance is constrained:
the amount must be at least **R100** and at least **25%** of arrears for buckets
1–3 or **15%** for buckets 4–8; the date must be within **21 days**; a promise
falling more than 3 days after the client's known salary date requires a reason
code.

**Outcome classification**, evaluated at promise date + 2 business days:

| Outcome | Condition |
|---|---|
| Kept | ≥ 95% of the promised amount received by the grace end |
| Partial | ≥ 40% and < 95% |
| Broken | < 40% |
| Superseded | An arrangement was activated before the promise date |

**A broken promise is not the same as no contact, and the path must differ.**
Where a promise breaks: contact is attempted within 1 business day, on a spoken
channel where permitted, with a distinct script; the sequence advances one
position rather than retrying the current one; the promise credibility counter
decrements. Where there was simply no contact, the sequence follows the normal
retry-then-escalate pattern. Treating the two the same is the single most common
complaint from Collections Operations about the system they have today.

**Limits.** At most **4 promises per episode**; after **2 consecutive broken
promises**, promises are no longer accepted from that account without L2
override; a kept-promise history of 3 or more raises the permitted promise window
from 21 to 30 days. Promise history must survive across episodes for the
credibility counter even though promise counts reset per episode.

### 5.9 Operational capacity and allocation

**Precondition.** Every account has a per-account recommendation from 5.4 as
modified by 5.5, and a permitted-treatment set from 5.2.

**Determines.** Which of those recommendations actually happen today.

Treatments have finite supply and the matrix does not know it:

| Pool | Supply per day | Typical demand |
|---|---|---|
| Early agents | 1 100 hours ≈ 27 000 attempts | 186 000 |
| Late agents / negotiators | 500 hours ≈ 11 000 attempts | 94 000 |
| Legal and pre-legal agents | 180 hours ≈ 4 200 attempts | 21 000 |
| Business and commercial agents | 120 hours ≈ 3 800 attempts | 9 000 |
| SMS | 900 000 | 1 410 000 |
| Email | 2 400 000 | 780 000 |
| In-app | unconstrained, 414 000 eligible | 390 000 |
| Voice message | 350 000 | 505 000 |
| Field visits | 180, routable to at most 3 regions | 2 600 |
| Notices (print and registered post) | 12 000 | 14 500 |
| Legal handover | 2 500 matters per month, paced | 4 100 per month |
| Agency placement | Contracted monthly: 40 000 across 3 agencies at 45/35/20, paced daily ± 15% | 71 000 per month |

Total agent supply is **1 900 hours, about 46 000 call attempts**, against roughly
**310 000** recommended. Two-thirds of the recommended SMS volume fits; one third
does not.

**The allocation must therefore rank and ration**, and the following are all
requirements of it.

**Ranking basis is a parameter, not a constant.** Two bases must both be
expressible and switchable by Collections Strategy without a release: expected
recovery value — the marginal expected rand from applying this treatment to this
account today, net of its cost — or a policy priority ordering, where for example
pre-prescription legal candidates and first-time bucket-3 entrants outrank
everything regardless of value. Which basis was used is part of the record.

**Fairness and coverage constraints bind before value.**

- **No account goes untouched for more than 21 days** while in buckets 3–8, or
  30 days in buckets 1–2, unless a suspension forbids all contact. Accounts
  approaching the limit are pulled into the allocation ahead of higher-value
  accounts.
- **High-balance accounts may not be permanently crowded out.** Volume plays in
  bands 1–3 have better cost-per-rand ratios and will starve balance bands 6–7
  if value alone rules. At least **8% of agent capacity** is reserved for
  balance bands 6–7.
- **At least 60% of accounts newly entering bucket 3** receive an agent attempt
  within 5 days of entering.
- **Channel starvation is disallowed**: an account may not receive only automated
  treatments for more than 45 consecutive days while in buckets 4–8.
- **Challenger cohorts are allocated before champion overflow.** If rationing
  falls harder on challengers than on champions, the experiment measures
  rationing rather than treatment. Allocation must be proportionally neutral
  with respect to `cohort_code`, and the residual imbalance must be reported.

**Overlays change demand, and the allocation must stay feasible.** An intensity
dial is authored as a risk or strategy decision and lands as an operations
problem: escalating bucket 3 by one intensity level moves roughly 61 000 accounts
from automated channels onto agent channels that have no spare hours, and
suppressing field visits pushes 2 600 recommendations onto the late-agent pool.
Three requirements follow.

- The allocation must remain **feasible** with any approved overlay stack in
  force. An overlay cannot create capacity; it can only change who gets it. If an
  overlay's effect is to push a pool from 6× oversubscribed to 9×, the flow still
  produces a valid allocation and the overlay does not silently become a no-op
  for the accounts below the cut-off.
- The **effect of the overlay on unallocated volume must be reportable**: per
  pool, per day, the demand with the overlay stack in force against the demand
  without it, the change in cut-off rank, and the count of accounts that moved
  from actioned to unactioned as a consequence. An intensity dial that raises
  demand 20% on a saturated pool may deliver no additional contacts at all, and
  the forum that approved it is entitled to be told that.
- An **allocation weighting dial** — reserving a larger share of a pool for a
  scope — is scored against the fairness floors of this stage, not over them. A
  weighting overlay may not push any account past the untouched-days limit, and
  an overlay that would do so is rejected at authoring with the constraint it
  would breach named.

**Pacing.** Agency placements and legal handovers are monthly contracted volumes
consumed daily. The daily allocation must pace against the month-to-date
position, not simply take the best available candidates on the first of the
month and starve the rest.

**Non-selection must be explainable.** For every account that receives no
treatment today, a `non_selection_reason_code` and its supporting detail:

| Code | Meaning | Detail required |
|---|---|---|
| 200 | Matrix recommended no action | Cell id |
| 205 | Treatment suppressed by an overlay | `overlay_id`, scope, expiry, the treatment that would otherwise have applied |
| 210 | Suspended | Every suspension code that applied |
| 220 | Cooling-off not expired | Treatment, last applied date, days remaining |
| 230 | Interval or cap would be breached | Which cap, current count, window |
| 240 | Promise in force | Promise id, promise date |
| 250 | Below the capacity cut-off | Pool, rank achieved, number ranked, cut-off rank, ranking basis |
| 260 | Coverage constraint already satisfied by another account of the same client | The sibling account actioned |
| 270 | No permitted channel available | Which contact points failed and why |

Code 250 is the one that makes this stage hard. "You were 91 412th of 186 000 in
the early agent pool, and we stopped at 27 000" is a population-level fact being
recorded as a per-account explanation.

**Determinism.** Ranking ties are broken on a stable key, never on input order or
on anything derived from processing order. The same book, the same parameters and
the same capacity produce the same allocation, account for account, on re-run.

### 5.10 Output assembly

**Determines.** The instruction that leaves this flow, per account.

Per account, every day, whether actioned or not:

| Field | Note |
|---|---|
| `treatment_instance_id` | The join key for every downstream outcome |
| `treatment_code`, `treatment_intensity` | Today's action |
| `channel_code` | Where it lands |
| Timing window | Earliest and latest permitted contact times today, after suspension 115, the client's known contactable hours and the client's stated preference |
| `script_id` / template id, and language | From the script registry; compliance-approved wording only |
| Negotiation envelope | Minimum acceptable arrangement instalment, maximum duration, maximum discount at the agent's own authority, promise acceptance limits, what requires referral and to which level |
| Suspensions applied | Every one, with code, source and expiry |
| Expected next treatment | What happens if today's treatment fails, and the earliest date it would happen |
| `matrix_version`, `matrix_cell_id`, `cohort_code` | Attribution |
| `adjustment_set_version` and the applied `overlay_id`s | In stack order, with the effect of each, and the unadjusted treatment, intensity, score and band beside the adjusted ones |
| Allocation | Pool, rank, cut-off, ranking basis — or `non_selection_reason_code` |
| Evidence reference | Pointer to the full trail (project 09) |

The **expected next treatment** is not decoration. Agents use it on the call
("if we do not hear from you by Thursday, the account goes to our legal team"),
which makes it a statement to the client, which makes it something the Bank must
honour or explain.

### 5.11 Feedback

Outcomes arrive continuously and must be joinable back to the treatment that
produced them:

| Outcome | Latency | Join |
|---|---|---|
| Payments | 0–36 h, four rails | To the account and, by attribution window, to `treatment_instance_id` |
| Contact attempts and outcomes | Dialler hourly; agency T+2; field T+1 | Directly to `treatment_instance_id` |
| Promises | Intraday | To the contact that captured them |
| Disputes and complaints | Intraday | To the account and the contact |
| Arrangement events | Daily | To `arrangement_id` |
| Agency events | T+2, and some never arrive | To the placement |

**Attribution windows** are parameters: a payment within 7 days of a contact is
attributed to that contact; a payment within 14 days of a promise is attributed
to the promise; where both apply the promise wins; where two treatments overlap
the later one wins and the earlier is recorded as co-present. These rules decide
what champion/challenger measures, so they are owned by Collections Strategy and
versioned with the experiments.

Every outcome must carry the `matrix_version` and `cohort_code` in force when the
treatment was issued, not when the outcome arrived. A matrix change mid-experiment
does not retroactively relabel last week's results.

**Measurability requirement.** Per-cell and per-band roll rates, cure rates,
promise-kept rates, cost per rand collected and net recovery must be computable
by cohort, and cells with insufficient volume for a conclusion must be flagged as
such rather than reported as a small difference.

### 5.12 Cohorts, the intraday re-run, and the real-time path

**Cohorts.** Champion/challenger assignment is by a deterministic function of
`account_id` and a per-experiment salt, so that an account's cohort is the same
every day for the life of the experiment, is recomputable without storing it, and
is uncorrelated across concurrent experiments. Up to **6 experiments** run at
once; an account may be in at most 2, and the interaction is recorded. A standing
holdout of **0.5%** receives minimum-contact treatment for measurement, is
excluded above bucket 5, and has an expiry date. Cohort is an **account-level**
property and does not change when the account rolls — reassigning on roll would
select challengers on outcome.

**Cohorts and overlays are different mechanisms and must not be conflated.** A
challenger is a *comparison*: two populations, identical in expectation,
receiving different treatment, measured against each other, running for a fixed
test window under an experiment approval. An overlay is a *tilt*: one population,
scoped by bucket or product or band, treated differently from what the approved
artefact says, for a period, under a strategy approval, with a mandatory expiry.
They have different owners, different lifetimes, different approval paths and
different records. A design that expresses a challenger as "an overlay scoped to
some accounts" will produce results nobody can interpret.

Three requirements keep them separable:

- An overlay's scope is declared in the matrix's dimensions, channel and — only
  where deliberate — `cohort_code`. An overlay that is *not* cohort-scoped must
  apply equally to champion and challenger, and that equality must be
  demonstrable rather than assumed, because scope dimensions correlate with
  nothing only if the cohort split is sound.
- The overlay stack in force is recorded **per treatment instance**, so a
  challenger's results can be partitioned by the overlay conditions that held
  when each was issued.
- **Interpreting a challenger result when an overlay was live.** Where an overlay
  was in force for the whole test window and applied equally to both arms, the
  comparison stands and the result is reported as conditional on that overlay —
  the challenger beat the champion *under* a −22 point score shift, which is not
  the same claim as beating it without one. Where an overlay started, ended or
  changed scope mid-window, the window is split at the change and each segment
  reported separately, with the segments flagged if either is below the
  volume threshold. Where an overlay was cohort-scoped, the arms are no longer
  comparable and the result must be reported as uninterpretable rather than as a
  small difference. The overlay-disabled run of §8 gives the base against which
  both arms can be restated.

**Intraday re-run.** At 13:30, after the second payment posting, accounts whose
state has materially changed — roughly 180 000 — are re-decided. A treatment
already dispatched cannot be recalled; the re-run may withdraw a queued treatment
and must record that it did. An account that paid at 11:00 must not be
telephoned at 14:00 by a list built at 05:00.

**Real-time path.** An agent on a live call needs a revised arrangement or
settlement assessed against the same rules, on the same parameter versions, in
under 400 ms. The answer must be identical to what the batch would have produced
from the same state. Two implementations of the same rules, batch and real-time,
that can drift apart, is the failure mode being tested.

---

## 6. Parameters and tables

| Table | Dimensions | Cells | Owner | Cadence | Source |
|---|---|---|---|---|---|
| **Treatment matrix** | **8 buckets × 6 collections bands × 7 balance bands × 4 contact bands × 4 product families** | **5 376 cells, each a treatment code plus 3 modifiers** | Collections Strategy | Monthly, plus mid-month cell patches | Spreadsheet |
| **Collections score overlay set** | Typically 4–9 live overlays × 11 attributes, each with a declared scope and stack position | ~90 | Collections Strategy, approved by Model Risk | Ad hoc, expiring | Overlay register |
| **Treatment matrix overlay set** | Typically 6–15 live overlays × 12 attributes (kind, scope over 5 matrix dimensions plus channel and cohort, magnitude, stack position, approval, effective window, review date) | ~160 | Collections Strategy, approved weekly by the Collections Credit Forum | Weekly, expiring | Overlay register |
| Suspension rules | 20 codes × 9 attributes | 180 | Compliance and Legal | On regulatory change | Internal, legally reviewed |
| Contact interval rules | 7 channels × 8 buckets × 4 values | 224 | Collections Strategy, Compliance co-sign | Quarterly | Internal |
| Contact frequency caps | 4 channel groups × 5 windows × per-client and per-account | 40 | Compliance | Annual | Internal |
| Permitted contact hours | 7 days × 4 channel groups × 2 edges, by client preference override | 56 | Compliance | On regulatory change | Public restrictions |
| Escalation path definitions | 8 buckets × 4 product families × up to 9 positions × 5 attributes | ~5 800 | Collections Strategy | Monthly | Internal |
| Discount authority grid | 8 buckets × 6 recovery bands × 5 authority levels | 240 | Credit Risk Policy | Quarterly | Internal |
| Arrangement minimums | 8 products × 6 arrangement types × 5 values | 240 | Credit Risk Policy | Quarterly | Internal |
| Arrangement authority matrix | 6 types × 5 levels × 4 concession dimensions | 120 | Credit Risk Policy | Quarterly | Internal |
| Notice period requirements | 8 products × 4 notice types × 6 attributes | 192 | Compliance | On regulatory change | Statute |
| Capacity by channel | 12 pools × 8 attributes | 96 | Collections Operations | **Daily**, revised intraday | Workforce management feed |
| Agency contracts | 3 agencies × 8 buckets × 6 attributes | 144 | Agency Management | Monthly | Contract |
| Script registry | ~340 scripts × 3 languages × 7 attributes | ~7 100 | Compliance | Monthly | Internal, legally approved |
| Collections scorecard | 28 characteristics × ~9 bins | ~250 | Model team | Annual, monitored monthly | Model documentation |
| Recovery curves | 8 buckets × 6 bands × 4 families × 24 months | 4 608 | Model team | Semi-annual | Model |
| Cost-to-collect schedule | 14 treatments × 4 families × 3 values | 168 | Finance | Annual | Cost model |
| Champion/challenger registry | 6 live experiments × 12 attributes, plus salts | 72 | Collections Strategy | As launched | Internal |
| Attribution window rules | 6 event types × 4 values | 24 | Collections Strategy | With each experiment | Internal |
| Business day calendar | 10 years × public holidays | ~120 | Compliance | Annual | Gazette |
| Banding definitions | 4 banded dimensions × edges | ~30 | Collections Strategy | Semi-annual | Internal |

As in project 00 §8, two properties are required of every one of these: **cell-level
attribution** and **diffability**. A new treatment matrix must produce a review
artefact showing which cells changed, from what to what, how many accounts sat in
each changed cell last month, and what the expected volume impact is per capacity
pool — because a matrix change that adds 40 000 agent calls to a book with 46 000
available is a change to every other account's chances of being contacted, and the
person approving it must see that before it goes live.

The two overlay sets have three further properties, following project 00 §7.6.
They are **never merged into the artefacts they modify**: a matrix overlay that
has run for six months is not folded into next month's matrix release to tidy
things up, because doing so destroys the distinction between what Collections
Strategy approved monthly with review and what the forum approved on a Monday for
six weeks. They are **independently versioned and independently expiring**, so
`matrix_version` 214 with overlay set 87 and `matrix_version` 214 with overlay
set 91 are different configurations and must be recorded as such. And the set of
overlays **past their review date** must be a standing report with an owner, not
a query somebody thinks to run.

---

## 7. Outputs

**Returned per account, daily, for all 2.3 M:** the assignment record of §5.10,
including the non-selection record where no treatment was assigned.

**Persisted:**

1. The assignment record itself, retained for 7 years.
2. The suspension evaluation, in full, for every account, every day — including
   accounts with no suspensions, because "none applied" is the answer to an ombud
   question and must be evidenced rather than inferred from absence.
3. The assembled state, or the watermarks and derivation sufficient to rebuild it
   exactly.
4. The sequence position before and after, the episode, and the reset event.
5. The matrix version, cell id, cohort and ranking basis.
6. The overlay stack in force — `adjustment_set_version`, the ordered overlays
   that applied, each one's scope, approval reference and expiry — together with
   the unadjusted score, band, treatment and intensity, so that "what would we
   have done without the overlay" is answerable per account and not only in
   aggregate.
7. The capacity position: per pool, the demand, the supply, the cut-off rank and
   the binding constraints, with and without the overlay stack.
7. Arrangement and settlement records with their authority trails.
8. Promise records with their outcome classification and the date it was
   evaluated.

**Downstream consumers:** the dialler (list, by pool, by timing window), the
messaging platforms, the field agent application, the agency placement files, the
notice production run, the attorney panel instruction file, Finance (write-off
recommendations and settlement impairment), and project 09 (evidence).

---

## 8. Non-functional requirements

| Requirement | Value |
|---|---|
| Daily batch | 2.3 M accounts assembled, decided and allocated within a **90-minute window**, starting when the payment file lands at 04:15 and publishing by 05:45 |
| Dialler list availability | 06:30, ahead of an 08:00 contact centre open |
| Intraday re-run | ~180 000 changed accounts re-decided within 20 minutes of the 13:30 posting |
| Real-time path | Arrangement or settlement re-assessment in **under 400 ms at p99**, ~55 000 requests per day, peak 40 per second between 10:00 and 11:30 |
| Determinism | Identical inputs, versions and `decision_date` ⇒ identical assignments **and identical allocation ranks**, bit for bit |
| Re-runnability | A failed run may be restarted from the beginning and must produce the same result; a partial run must not have partially dispatched |
| Batch/real-time agreement | The real-time path and the batch must produce identical answers from identical state; a nightly reconciliation over a 5 000-account sample must find zero material differences |
| Recovery | A missed daily run must be recoverable by running it for the missed `decision_date`, not by running today's rules against yesterday's book |
| Late data | Outcomes arriving up to 14 days late must be absorbable without altering historically recorded decisions |
| Retention | 7 years, queryable within 2 working days for an ombud request |

---

## 9. Audit, evidence and explainability

Four things will be asked, and the implementation is judged on whether they can
be answered without a data-science project.

**9.1 "Your client says he was harassed."** An ombud requires, for a named client
over a named period: every contact attempt across every account and channel with
its date, time, outcome and the script used; **and the rule that authorised each
one** — the matrix cell, the sequence position, the interval and cap checks that
passed, and the parameter versions of each. Per contact. Including contacts made
by an agency on the Bank's behalf.

**9.2 "Your client was under debt review and you contacted her anyway."** The
answer requires the suspension evaluation for that account on that day: whether
the debt review status was known, when it arrived in the registry feed, what
stage it was at, what the evaluation concluded, and — if the contact was
permitted — under which exception. If the status arrived after the run, that must
be demonstrable from the feed watermark rather than asserted.

**9.3 Internal Audit samples settlements.** For each sampled settlement: the
discount granted, the grid cell and version that set the standard discount, the
authority level required, the authority actually obtained with approver identity
and timestamp, and the expected-recovery comparison reproduced from the estimates
in force on the day.

**9.4 The regulator tests notice compliance exhaustively.** For **every** account
handed to legal in a period, evidence that the prescribed notice was delivered,
the delivery method and proof, and that the required business days elapsed before
handover. Not a sample. Any exception must be identifiable in advance rather than
discovered by the regulator.

**Two more that follow from the design.**

**9.5 "Why was this account never contacted for 40 days?"** The non-selection
record for each of those 40 days, including the capacity rank and cut-off where
that was the cause, and the coverage constraint that should have caught it.

**9.6 "Did the challenger actually beat the champion?"** The cohort assignment
and its stability over the experiment, the matrix versions in force, the
attribution rules applied, the per-cell volumes, and evidence that rationing did
not fall differently on the two arms.

---

## 10. Acceptance criteria

1. Collections Strategy changes the treatment matrix and sees a reviewable diff —
   changed cells, prior and new values, accounts affected last month, expected
   demand impact per capacity pool — before it goes live, without an engineer.
2. Every suspension that applied to an account on a day is individually named in
   the record, with its source, scope and expiry, and the record distinguishes
   "no suspension applied" from "suspensions not evaluated".
3. Given an account and any past date within retention, the sequence position in
   force on that date is re-derivable from the event log using the rules in force
   then, and the derivation matches what was recorded.
4. The as-known-on-the-day and as-at-now derivations are both available and
   distinguishable, and the account-days on which they disagree are countable.
5. A full daily run over 2.3 M accounts completes within the 90-minute window,
   and a re-run of the same `decision_date` produces identical assignments and
   identical allocation ranks.
6. For every unactioned account, a non-selection reason with the detail required
   by §5.9, including pool, rank and cut-off where capacity was the cause.
7. The same affordability capability serves granting and the distressed
   arrangement test, with the mode as a parameter, no forked implementation, and
   the mode named in the evidence.
8. An arrangement or settlement assessed on the real-time path returns in under
   400 ms at p99 and agrees with the batch on a 5 000-account nightly sample.
9. Champion/challenger cohorts are stable per account for the life of an
   experiment, recomputable without storage, and demonstrably uncorrelated with
   capacity rationing.
10. Every dispatched contact carries the matrix version, cell id, sequence
    position, cohort and script version, and these are joinable to the outcome
    that followed.
11. A matrix cell that has seen no traffic in six months is identifiable as
    unexercised, and a new matrix version is validatable without waiting for
    traffic.
12. Legal handover cannot occur without the notice evidence of §9.4 being
    present, and the attempt to hand over without it is itself recorded.

---

## 11. Change scenarios

Each of these will arrive within a year of go-live.

1. **Collections Strategy adds a ninth arrears bucket**, splitting 180–364 into
   180–269 and 270–364, taking the matrix from 5 376 to 6 048 cells. The new
   cells must be populated, reviewed and versioned, and history recorded under
   the old banding must remain interpretable.
2. **A new treatment is introduced** — a WhatsApp-style rich message with a
   payment link — with its own cost, capacity, consent basis, interval rules and
   script family, and it must take a position in existing escalation paths.
3. **Compliance tightens contact frequency** from 10 per 7 days to 6, per client
   rather than per account, effective on a date, with the old rule governing
   everything before it.
4. **The regulator shortens the prescribed notice period** for one product
   family, changing the computable expiry of suspension 112 for that family only.
5. **Champion/challenger goes from 1 challenger to 4 concurrent experiments**,
   with an account permitted in at most 2 and interactions recorded.
6. **Capacity is cut 30% for six weeks** during a contact centre relocation, and
   the coverage constraints must be relaxed deliberately, with the relaxation
   recorded, rather than silently violated.
7. **A new arrangement type** — an income-linked variable instalment — that is
   not a fixed amount, and so does not fit the minimum-percentage grid.
8. **Settlement authority is decentralised**: agents get L2 authority for
   balances under R15 000, which changes the envelope on the call and the audit
   sample basis.
9. **Two agencies become five**, with per-agency bucket appetite and
   performance-based volume reallocation monthly.
10. **A court decision changes prescription handling**, requiring a proactive
    written notification to clients whose debt has prescribed — a new treatment
    triggered by a suspension rather than blocked by it.
11. **The model team replaces the collections score** with a 34-characteristic
    version, needing both live in different cohorts for a four-month parallel
    run, with the band definitions re-cut.
12. **Finance requires write-off recommendations to respect a monthly quantum
    cap**, making a per-account recommendation subject to a second
    population-level constraint alongside capacity.
13. **The intraday re-run becomes hourly** between 08:00 and 17:00.

---

## 12. Out of scope

- Payment collection mechanics: debit order submission, payment rails,
  reconciliation and allocation of receipts to accounts.
- The dialler itself: pacing algorithms, agent state, call recording.
- Contact centre workforce scheduling. Capacity is an input here, not an output.
- Legal process after handover: summons, judgment, execution, attorney billing.
- Provisioning and impairment calculation. This flow feeds it.
- Agency commercial management and commission settlement.
- Origination-side credit decisions (projects 02, 03, 06, 07).
- Storage and presentation of the evidence record (project 09).
- Model development. The flow consumes the collections score and the recovery
  curves; it does not fit them.

---

## 13. Questions the implementation must answer

1. **What expresses sequence position over time?** An account's place in an
   escalation path, the events that advance it, the events that reset it, the
   counters that carry across a roll and the ceiling that cannot go down — is
   this a fourth core component kind alongside scorecard, decision table and
   decision tree, or is it composed from the three? If composed, how many times
   is the composition written before somebody copies it?
2. **Where does temporal state live between runs?** If yesterday's position is
   stored, replay depends on a mutable field. If it is derived from the event
   log, the derivation is itself versioned logic over 180 million rows in a
   90-minute window. Which, and what does the other cost?
3. **How are as-known-on-the-day and as-at-now expressed** in one implementation,
   given that audit needs the first and analysis needs the second, and that they
   disagree on 2.4% of account-days?
4. **How is a 5 376-cell matrix authored, validated, versioned and diffed** by
   analysts, when 41% of cells are sparse and 12% are empty, and when three
   versions may be in force simultaneously for different cohorts?
5. **How does a population-level constraint compose with per-account logic?**
   Capacity allocation is not a rule about an account; it is a rule about all of
   them at once, and its outcome must be recorded as a per-account explanation.
   Is this the same shape as project 07's portfolio budget, or a different one?
6. **How is non-selection made explainable** without materialising a rank and a
   cut-off for every one of 2.3 M accounts every day?
7. **How do suspensions override everything while remaining individually
   attributable?** Short-circuiting is the natural implementation and it is
   wrong. What structure gives complete evaluation with the performance of a
   short-circuit?
8. **How is a computable expiry expressed** for a suspension, uniformly, when
   some expiries are date arithmetic over business days, some are event-driven,
   and some depend on a feed that may be late?
9. **How does one capability serve both granting and distressed affordability**
   without a fork, and how is the mode made visible in the evidence rather than
   implied?
10. **How are time windows expressed** so that 30/60/90-day, 12/24-month,
    rolling-7-day, business-day and until-a-date windows are one concept rather
    than forty hand-written filters?
11. **How is a champion/challenger split kept stable per account** across daily
    runs, matrix versions, bucket rolls and concurrent experiments, and how is
    its neutrality with respect to capacity rationing demonstrated rather than
    assumed?
12. **How do batch and real-time share one implementation** when one processes
    2.3 M accounts in 90 minutes and the other one account in 400 ms?
13. **What is the unit that feedback joins to?** A treatment instance, a sequence
    position, a matrix cell or an episode — and can the same join answer "did the
    challenger win", "why was this client called" and "which cell is unexercised"?
14. **How much of this is collections-specific?** Sequences, capacity rationing,
    suspensions with expiries and time-windowed aggregates are all recognisable
    in projects 01, 04 and 07. If they are implemented here as collections logic,
    the reuse seam has been missed; if they are generalised, what are they called?
