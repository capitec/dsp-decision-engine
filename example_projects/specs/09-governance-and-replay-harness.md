# 09 — Decision governance and replay harness

> Fictional. The Bank, its products, thresholds, table dimensions and volumetrics
> are invented for this repository. Regulatory mechanisms referred to are the
> published public ones; every number attached to them is illustrative.

---

## 1. What this is

Eight decision flows are live. Between them they produce **22 million decisions a
year** — who gets a loan, at what price, whose card limit moves, who gets a
collections call, whose payment is stopped mid-flight. Each of those decisions
may have to be defended: to the client on the phone the same afternoon, to an
analyst investigating a suspected defect next week, to the Credit Committee next
month, to an ombud adjudicator next year, and to a regulator or an external
auditor in 2033.

This project is what makes that possible. It replays decisions, explains them,
diffs the logic that produced them, measures what changed between two versions,
certifies releases before they ship, watches the inputs and the outcomes for
drift, and assembles the packs that go out to regulators and ombuds. It is the
Bank's answer to the question *"why did this happen, and can you prove it?"*

**This project is unusual and the difference matters.** It is not a decision
flow. It has no applicants, no scorecards, no rate cards and no outcome of its
own. It operates *on* the other eight. Almost everything it needs is something
one of those eight has to have emitted at the moment the decision was made, and
none of it can be reconstructed afterwards. That inverts the usual reading of a
specification: most of the requirements below are not requirements on this
project at all, they are **constraints on the other eight**, and they are only
achievable if those eight were built to satisfy them from the first release.
§5.15 states them as a numbered contract, and it is the most useful part of this
document.

The corollary is worth saying once, plainly, because every governance programme
in every bank learns it the expensive way: **none of this can be retrofitted.**
If a flow did not record which version of the rate card it read, no amount of
cleverness in 2031 recovers it. If a challenger split came from a random number
generator rather than from a hash of the application identifier, the assignment
is gone. The evidence either exists or it does not.

---

## 2. Why it is in this set

| Question | How this project stresses it |
|---|---|
| **Q7 audit and debugging** | This *is* Q7, at full scale and with real obligations attached. Every other spec says "must be answerable years later"; this one says by whom, in what format, within how many working days, and what happens when it is not. It is the hardest test of whether the framework's audit story survives contact with a regulator. |
| **Q6 codebase organisation** | Governance is a cross-cutting property of eight codebases owned by six teams on different release trains. A harness that requires each team to opt in, per flow, by hand, will be 60% covered forever. What has to be true of the way the code is organised for governance to be a property of the whole estate rather than a per-team chore? |
| **Q3 parameters** | Replay and diff both turn on parameter values being a first-class, versioned, externally-stored, individually-attributable artefact — not a code default, not an inherited fallback, not a literal. "What was the affordability buffer on 3 March 2027, who set it, and under whose approval" must answer from one artefact, not from a git archaeology exercise. |
| Q1 reuse | Secondary. One harness serving eight flows with different shapes — real-time single record, monthly batch, daily batch, nested entities — is itself a reuse test. |
| Q2 tables | Secondary, but sharp: cell-level attribution and diffability (00 §8) are *this* project's requirements, stated in the library. |
| Q5 core kinds | Secondary. Coverage and dead-logic measurement must work for rules, tree nodes, matrix cells and scorecard characteristics alike, which is a question about whether those have anything in common. |

---

## 3. Actors

| Actor | Responsibility | Cadence |
|---|---|---|
| **Decision Governance** (owns this project) | Runs the harness. Assembles regulator packs. Publishes the monthly monitoring pack. Blocks releases that fail certification. | Continuous |
| **Contact centre consultants** (~900) | Explain a decline or a limit to a client, on the phone, in under 90 seconds. ~31 000 such calls a year. | Continuous |
| **Credit analysts** (~35) | Investigate suspected defects, disputes and anomalies. Run what-if interventions constantly. | Daily |
| **Disputes and complaints** | Handle ~2 400 formal disputes a year, of which ~40 escalate to an ombud or the regulator. | Daily |
| **Credit Risk Policy** | Approves the plain-language rendering of every flow. Owns thresholds and table contents. Not engineers. | Quarterly |
| **Regulatory Compliance** | Owns the reason taxonomy, the statement-of-reasons wording, and the response to information requests. | Continuous |
| **Credit Committee** | Approves policy changes. Receives the monthly outcome pack. Signs off releases with a material expected effect. | Monthly |
| **Model Validation** (independent of model development) | Validates the nine scorecards and the flows that use them. Must be able to replicate a model's output independently. | Annual per model |
| **Internal Audit** | Tests, without warning, that a sampled decision reproduces and that the deployed logic matches the approved rendering. | Two reviews a year |
| **Information Officer / privacy** | Owns retention, masking and access to decision evidence, which is personal information throughout. | Continuous |
| **Product engineering teams** (×6) | Emit the evidence. Consume the certification gate. Cannot ship without it. | Per release |
| **Credit Systems** | Owns the shared library (project 00) and the version records for its twenty-one capabilities. | Per release |
| **External auditors and the regulator** | Request evidence. Do not accept "the system says so". | ~40 requests a year |

---

## 4. Inputs

This project's inputs are other projects' outputs. Volumes are annual unless
stated.

### 4.1 Decision evidence, from the eight flows

| Flow | Decisions / year | Shape | Retention |
|---|---|---|---|
| 01 fraud and scam interdiction | 9 400 000 | Real-time, single event, sub-second | 5 years |
| 02 affordability assessment | 2 100 000 | Single record, called by four flows | 7 years |
| 03 unsecured granting and pricing | 1 750 000 | Single record, iterative solve inside | 7 years |
| 04 campaign targeting | 4 800 000 | Monthly batch, 60 trees | 5 years (marketing) |
| 05 business credit | 48 000 | Single record, two-level nesting | 7 years |
| 06 consolidation and restructure | 210 000 | Single record, bounded search | 7 years |
| 07 credit limit management | 2 900 000 | Monthly batch, portfolio-constrained | 7 years |
| 08 collections treatment | 790 000 | Daily batch, temporal state | 7 years |
| **Total** | **21 998 000** | | |

Per decision, the evidence comprises: the inputs as received; the reference data
read; the parameter set in force; the identity of the logic that ran; the
outcome and its reasons; and the internal detail needed for explanation (gates
evaluated, rules fired, nodes visited, cap chains, score contributions, cells
read). Sizes in §8.3.

**Freshness.** Real-time flows must have evidence durable within 2 seconds of
the decision. Batch flows within 30 minutes of batch completion. A decision whose
evidence has not landed within those windows is an incident, counted and
reported, not a silent gap.

### 4.2 Reference-data version records

Every versioned artefact named in 00 §8 — 15 table families, including a 63 360-cell
Flex Loan rate card refreshed monthly and a 219 600-cell Drive Finance card — with
its version identifier, effective dates, content hash, author, approver and
approval date. Roughly 340 new versions a year across all families.

### 4.3 Logic version records

For each of the eight flows and each of the twenty-one shared capabilities: a
build identifier resolving to exactly the logic that ran, the approved
plain-language rendering that corresponds to it, the approval record, and the
certification result that let it ship. ~260 flow releases a year across eight
teams.

### 4.4 Approval and change records

Who changed what, when, under whose authority: ~1 900 parameter changes a year,
~360 new rules in project 01 alone, ~130 table version promotions, ~260 flow
releases. Source systems are the change-management tool, the artefact store and
the Credit Committee minutes. These must be joinable to the logic and reference
versions by identifier, not by date proximity.

### 4.5 Production volumes for monitoring

Full decision volume, not a sample, for outcome and coverage monitoring; a 5%
deterministic sample is acceptable for input-distribution monitoring on the two
highest-volume flows, provided the sampling is reproducible.

### 4.6 The input inventory

A declared list of every input feature consumed by any flow: name, type,
nullability, source system, expected range or category set, PII classification,
and whether it is or could proxy for a prohibited ground. Currently **1 240
distinct features** across the eight flows, ~180 of them shared. Drift monitoring
iterates over this inventory; a feature absent from it is unmonitored, so absence
must be detectable rather than invisible.

---

## 5. The flow

Fourteen capabilities, then the contract they impose on everything else. They are
not sequential — each is invoked on its own trigger — but they are ordered by how
much the ones after depend on the ones before.

### 5.1 Exact replay

**Trigger.** A decision identifier, from any of the eight flows, from any date
inside the retention window.

**Preconditions.** Everything in §5.15. Replay is the capability that fails first
and hardest when a flow was built without it.

**What it determines.** Whether the recorded outcome is re-derivable, and if so,
re-derives it.

To re-derive a decision, **all seven of the following must have been captured at
the time**:

1. **The inputs as received.** Before normalisation, before defaulting, before
   any cleansing. The three null situations of 00 §7.4 — not collected, collected
   as zero, could not be established — must survive capture distinctly, because
   the flow's behaviour differs across them and a replay that flattens them
   produces a different answer for a defensible-looking reason.
2. **The reference data as at the decision date.** The bureau view as returned,
   not re-queried; internal account state as read; the version identifier of every
   table, norm, cap, rate card, scorecard, grade boundary set, appetite grid and
   reason registry consulted, together with the specific cells read.
3. **Every parameter value in force.** Explicit and complete — every value the
   run actually used, including values that fell back to a code default, named as
   such. A record that says "the defaults applied" is not a record.
4. **The structure of the logic that ran.** Not a description of it: an
   identifier that resolves, unambiguously and for the full retention period, to
   the exact logic. A version tag that was later moved is not an identifier.
5. **The versions of every shared capability used.** All twenty-one are
   independently versioned (00 §7.1) and two majors may be simultaneously live,
   so "the library version" is not sufficient; it must be per capability.
6. **Every random or hashed assignment.** Holdout groups, challenger splits,
   champion/challenger allocation, sampling flags, queue assignment. Each must be
   derivable from a stable identifier and a named seed, and the derived value must
   also be recorded, so that a change to the derivation is detectable rather than
   silently absorbed.
7. **The `decision_date` itself**, distinct from the timestamp at which the
   decision was computed, and distinct from today. 00 §7.3 states this for the
   library; it holds for every flow.

**Acceptance standard.** Bit-identical outputs. Every declared output of the
flow, compared byte for byte against what was recorded, including reason code
lists and their order.

Where bit-identical is genuinely impossible, the replay must say so explicitly
and bound it:

- Floating-point results that depend on hardware, compiler or library version:
  permitted to differ by a relative tolerance of 1e-12 on intermediates, and
  **not at all** on any monetary figure after rounding (00 §6.20), because a cent
  of drift in an instalment is a reconciliation failure and a client-facing
  discrepancy.
- Any value that was read live and not captured: **not permitted.** A replay
  that calls an external service is not a replay. If a flow's evidence is
  insufficient to replay without a live call, the replay fails and the flow is
  non-compliant.
- Genuinely non-reproducible decisions must be **fewer than 0.05%** of decisions
  in any flow in any month (that is, under 11 000 of 22 million), each one
  individually explainable, and the count reported to the Credit Committee.

**What it emits.** A replay verdict — reproduced / reproduced within declared
tolerance / not reproduced — the re-derived outputs, a field-by-field comparison
against the recorded outputs, and for a failure, the first point of divergence.

**What must be recorded.** Every replay: who ran it, when, against which
decision, with what verdict. Replays are themselves evidence, particularly when
they were run during a dispute, and are retained for 24 months.

**Retention and format stability.** 7 years for credit decisions, 5 for
marketing. Seven years is longer than the expected life of any storage format,
serialisation library or schema currently in use. The evidence format must
therefore be readable by something that does not exist yet: self-describing,
versioned, with a written specification, and with a standing obligation that any
format change ships with a migration for everything already stored. A yearly
exercise must replay a random sample of 1 000 decisions from each of the seven
preceding years, and failure of that exercise is an audit finding.

### 5.2 Single-decision explanation

**Trigger.** A decision identifier plus an audience.

**Preconditions.** The flow recorded not just what fired but what was
*evaluated*; §5.15 items 14–16.

**What it determines.** A human-readable account of one decision, comprising:

- every gate evaluated, in order, with its outcome and the values it tested;
- every rule that fired, with its identifier, description, owner and effective
  date — and, for the analyst rendering, every rule that was evaluated and did
  not fire, with why;
- every node visited in every tree, by stable node identity, with the condition
  at each and the value that satisfied it;
- for every successively constrained quantity, the **complete ordered chain of
  values and their causes**: requested R180 000 → capped to R145 000 by the
  affordability limit → capped to R120 000 by the grade-6 appetite ceiling →
  rounded to R120 000 (00 §6.20) → final. Not the final value with a note; the
  chain. This is what an analyst reads first and what an adjudicator needs to
  understand that a lower offer was the product of four separate policies and not
  a single arbitrary number;
- score contributions per characteristic, signed, ranked, with the bin each value
  fell in — including the null bin, which is a bin and not an error (00 §6.10);
- every table cell read, with its coordinates, its value and its table version;
- the reasons: the full `decline_reason_codes` list in severity order, the
  `primary_reason_code`, and the wording **in force at the decision date**, not
  today's wording.

**Three audiences, three renderings.** They are not the same document at
different lengths; they differ in what they may contain.

| | Contact-centre consultant | Credit analyst | Ombud adjudicator |
|---|---|---|---|
| **Reads it** | On a call, under 90 seconds | At a desk, for 20 minutes | Over days, alongside a complaint file |
| **Knows** | The product; no credit theory | Everything | Consumer law; no technical background |
| **Needs** | The dominant reason, in plain language, and what the client can do about it | Every evaluation, every value, every version, and the ability to intervene | A defensible narrative: what the policy was, who approved it, that it was applied correctly and consistently |
| **Length** | ≤ 5 sentences, one screen | Unbounded; typically 4–10 pages | 6–12 pages plus appendices |
| **Must not contain** | Thresholds, scores, internal codes, bureau detail the client has not seen | Nothing excluded | Unexplained internal identifiers, code, or anything a lay reader must take on faith |
| **Language** | Three languages, from the reason registry | English | English, with every identifier glossed |
| **Latency** | Under 3 seconds, from a call-centre desktop | Under 15 seconds | Not time-critical; correctness absolute |

Two tensions are permanent and must be resolved by policy rather than by the
harness quietly picking a side:

- **The dominant reason versus all the reasons.** A consumer's statutory right is
  to be told the *dominant* reason for a refusal or for a lower limit than
  applied for. The Bank's data-protection obligation for a decision taken by
  automated means is to provide sufficient information about the underlying logic
  for the person to make representations about it. One asks for a single reason;
  the other asks for enough of the mechanism to argue with. Both renderings must
  be producible from the same evidence, and which is issued is Compliance's call.
- **Explanation versus gaming.** Telling a declined applicant that the cut-off
  was a score of 612 tells the next applicant what to aim at, and tells a
  fraudster where the fence is — acutely so for project 01. The consultant and
  client-facing renderings therefore state reasons, not thresholds. The analyst
  and adjudicator renderings state both. This is a per-field classification on
  the evidence, not an editorial habit.

**What must be recorded.** Every explanation issued to a client: which decision,
which rendering, which wording version, who issued it, when. When a client
disputes what they were told, the Bank must be able to show what it actually told
them.

### 5.3 What-if intervention

**Trigger.** An analyst investigating a decision, or debugging a rule.

**What it determines.** What would have happened if one input or one parameter
had been different, without deploying anything, and without the result being
confusable with the real decision.

**Requirements.**

- Change one input value, or one parameter value, or one table cell, and re-run
  the decision as it stood, with everything else held at its recorded values.
- Change several, when investigating an interaction, with each change listed.
- Run in under 5 seconds for a single record, because an analyst will do this
  forty times in an afternoon and a 60-second turnaround means they stop doing
  it and start guessing instead.
- Produce a side-by-side against the original: which outputs moved, which gates
  changed outcome, which rules started or stopped firing, which nodes were
  visited instead.
- **Numerically faithful.** A what-if that runs different arithmetic from
  production proves nothing. If the harness can run the logic in more than one
  way, the ways must be demonstrably equivalent, and the divergence must be
  attributable to a layer rather than merely observed.

**Non-confusability is a hard requirement, not a nicety.** Every what-if result
carries a distinct, non-production identifier; is visually and structurally
marked in every rendering; cannot be written into the decision store; cannot be
issued to a client; and cannot satisfy a regulator request. The failure mode this
prevents is real: an analyst's exploratory re-run, exported to a spreadsheet,
emailed, and six weeks later quoted back to the Bank by a complainant's attorney
as what the Bank's own system says.

**What must be recorded.** Every what-if run: who, when, against which decision,
what was changed, what the result was. Retained 24 months. Approximately 1 800
runs a month. These are evidence of investigation and are routinely requested
during disputes.

### 5.4 Version diff

**Trigger.** Any change to any flow, or to any artefact a flow reads.

**What it determines.** A readable change record between two versions of the same
thing.

**Applies to**, at minimum: a whole flow; a rule set (project 01, 610 live rules
growing by ~30 a month); a decision tree (project 04, 60 trees); a treatment
matrix (projects 07 and 08); a rate card (project 03 and 00 §8, up to 219 600
cells); a scorecard (nine of them); an appetite grid; the expense norm tables;
the reason registry (380 codes).

**The change record must state**, for every difference:

- **what** changed — for a rate card, the cells, their before and after values and
  the percentage move; for a rule set, rules added, removed, reworded, reordered
  or repriced, and changes to the priority resolution between them; for a tree,
  nodes added, removed, re-thresholded or re-parented, and the sub-populations
  affected; for a threshold, the old and new values and the validated bounds of
  each;
- **who** changed it, by name, from the authoring system — not "the release";
- **when** it was authored, approved and promoted, as three distinct timestamps;
- **under whose approval**, resolving to a specific approval record with a
  specific approver and forum;
- **the estimated impact**, from a run over a recent representative population
  (§5.5).

**Requirements.** The diff must be **semantic, not textual.** Reordering the rows
of a rate card spreadsheet, renaming an internal identifier, or reformatting an
authoring file must produce an empty or near-empty change record; moving one
cell's value by 0.25% must produce a one-line one. A diff that reports 63 360
changed cells because the export ordering changed is worse than no diff, because
it trains reviewers to approve without reading.

Aggregation is required, not optional: "1 840 of 63 360 cells changed; all in
grades 7–9; mean move +0.31 percentage points; maximum +0.95; no cell moved down;
affects an estimated 14.2% of Flex Loan volume" is a reviewable statement. A list
of 1 840 rows is not.

**What must be recorded.** The change record is retained with the version it
describes, for as long as any decision made under that version is retained.

### 5.5 Swap-set analysis

**Trigger.** Any proposed change with an expected behavioural effect, and any
change at all where the effect is claimed to be nil.

**What it determines.** Who would have got a different outcome, and why.

**Method.** Run version A and version B over the same population — normally the
most recent complete month, 2 M records for the largest flows — with everything
else held identical, and compare record by record.

**The report must contain**, at minimum:

| Measure | Example figure |
|---|---|
| Approvals lost (approved under A, declined under B) — count, rate, and their distribution by grade, product, channel and segment | 1 840 (0.09%) |
| Approvals gained | 2 310 (0.12%) |
| Net approval rate movement | +0.02 pp |
| Amount increased / decreased — counts, mean and median move, total rand exposure move | 41 200 up, 38 900 down, net +R11.4 m |
| Referral volume movement, against operational capacity | +310/month against a queue sized for 4 000 |
| Reason codes appearing and disappearing, by code, with counts | code 214 +1 210; code 318 −940 |
| Expected credit quality of the swap groups — modelled bad rate of approvals gained versus the portfolio | 7.9% vs 4.2% |
| Records whose outcome did not change but whose reasons did | 3 400 |
| Records that could not be compared, and why | 112 |

**The hard part, stated plainly.** Attribution. A swap-set tells you 1 840
approvals were lost. It does not tell you *which change* lost them, and a
release almost never contains one change. A normal monthly release for project 03
contains a rate card refresh, two threshold moves, four new rules and a scorecard
recalibration. Five changes, one swap-set, and a Credit Committee question —
"which one of these cost us the 1 840?" — that the analysis as usually performed
cannot answer.

The requirement is therefore stated as an obligation on the release, not on the
analysis: **every change in a release must be individually attributable, or the
release must be split.** Concretely, a release of *n* changes must support *n+1*
runs over the same population — the base, then each change applied cumulatively
in a declared order — so that each increment's swap set is measured separately.
The ordering is declared because interactions exist and the increments are not
commutative; where an interaction is suspected, the complementary order is run
too, and a difference between the two attributions is itself the finding.

The cost is real and must be budgeted: five increments over 2 M records at 30
minutes each is 2½ hours of compute per release, which is acceptable monthly and
unacceptable daily. A team that ships daily must either ship one change at a time
or accept unattributed swap sets, and which of those they have chosen must be
visible to the Credit Committee rather than discovered during an incident.

**What must be recorded.** The population used (by identifier, reproducibly), the
versions compared, the increments and their order, the full report, and the
approval that followed it.

### 5.6 Release certification

**Trigger.** Every change to every flow, without exception, before promotion.

**What it determines.** Whether the change does what it says it does and nothing
else.

**The regression suite.**

- **Golden set.** 50 000 historical records per flow, 400 000 in total, with
  expected outputs. Selected for coverage rather than randomly: stratified across
  products, grades, channels, segments and outcome types, deliberately loaded
  with boundary cases (band edges, rounding boundaries, exactly-at-threshold
  values, maximum-cardinality collections, every null pattern seen in
  production), and refreshed quarterly with newly-seen edge cases while retaining
  every record that has ever caught a defect.
- **Tolerance rules**, distinguishing an intended change from a regression:

  | Output class | Tolerance |
  |---|---|
  | `outcome_code`, `decline_reason_codes`, `primary_reason_code`, tree node path | Exact. Zero tolerance. |
  | Monetary outputs after rounding | Exact to the cent. |
  | Rates | Exact to four decimal places (00 §6.20). |
  | Scores, probabilities | 1e-6 absolute. |
  | Intermediate unrounded values | 1e-12 relative. |

- **Declared expected effect.** Every release declares, before the suite runs,
  which golden records it expects to move, in which direction, and why. The suite
  compares the actual movement against the declaration. Three failure modes are
  reported separately and all three block: records that moved and were not
  expected to; records that were expected to move and did not; records that moved
  in the wrong direction or by an unexpected magnitude.
- **The no-effect rule.** A change declared to have no expected effect — a
  refactor, a rename, a reorganisation, a library upgrade — that moves **any**
  golden output blocks the release. There is no tolerance band and no override at
  team level; the only remedies are to fix the change or to re-declare it as a
  behavioural change and take it through impact review. This single rule catches
  more real defects than the rest of the suite combined, and it is the rule teams
  under delivery pressure most want relaxed.
- **Coverage measurement.** Which rules were exercised, which tree nodes were
  reached, which table cells were read, which reason codes were raised, which
  gates were evaluated in both directions. Thresholds: ≥ 98% of rules exercised,
  ≥ 95% of tree nodes reached, 100% of reason codes reachable by *some* golden
  record, and every code path a release touches exercised by at least one record
  that distinguishes it from the previous version. Table cell coverage is
  measured but not thresholded — a 219 600-cell rate card cannot be covered by
  50 000 records, and pretending otherwise produces a meaningless number.
- **Sign-off artefact.** A single document, generated, naming: the versions
  before and after, the change record (§5.4), the declared expected effect, the
  suite result, the coverage figures, the swap-set summary (§5.5), the approvals
  obtained, and the name of the person accountable. This artefact is what Internal
  Audit samples, and its existence — not the passing of the tests — is what makes
  the release defensible.

**Non-functional.** The full suite across all eight flows must complete within a
**40-minute** continuous-integration budget, because a certification gate that
takes two hours becomes a gate that is bypassed. 400 000 records, eight flows,
parallelisable across flows but not within a golden set comparison. A team must
also be able to run its own flow's 50 000 records in under 6 minutes locally,
before raising the change.

### 5.7 The reviewable artefact

**Trigger.** Quarterly re-approval of every flow, and every change requiring
policy approval.

**What it determines.** Whether Credit Risk Policy, Compliance and the Credit
Committee can read, understand and approve logic they did not write and cannot
read as code.

**What they need.** An ordered, plain-language rendering of what a flow does:

- the stages in the order they execute, each with a plain-language description of
  its purpose, written by whoever owns it;
- every gate and every rule, in evaluation order, with its identifier, its
  condition in words, its effect, its owner, its effective dates, and the reason
  codes it can raise;
- every threshold, with its current value, the bounds within which it may be
  set, who may set it, and when it was last changed;
- every table consulted, with its version, its effective date, its dimensions,
  its owner and its approval record;
- every tree, rendered as a readable structure with the population reaching each
  node, not as an indented list of 300 conditions;
- every quantity that can be constrained, with the complete list of things that
  can constrain it, in the order they apply;
- the full set of possible outcomes and the reason codes associated with each;
- the shared capabilities consumed, with their versions and their own renderings
  available one level down;
- **and a diff of all of the above against the version last approved.**

**It must be generated from what actually runs.** A hand-maintained policy
document describing a flow is not acceptable, for a reason that has nothing to do
with effort: it will diverge. It diverges within weeks, it diverges silently, and
a divergence between the documentation a governance forum approved and the logic
actually deployed is a regulatory finding on its own — independently of whether
the deployed logic was correct. The generated artefact and the deployed logic
must be incapable of disagreeing, because they are the same thing rendered twice.

**This is the requirement most likely to be underestimated**, and it is worth
being specific about why, because it looks like a documentation task and is not:

1. **Volume defeats naive generation.** Project 01's rendering, generated
   one-rule-per-entry from 610 rules, is roughly 180 pages. A 180-page document
   that a committee approves in a 40-minute agenda item is not governance; it is
   the appearance of governance, and an adjudicator will say so. The artefact
   must therefore be layered — a 6-page policy summary stating what the flow does
   and what can change an outcome, a full reference, and a diff view that is the
   thing actually read at re-approval — and the layering must itself be generated.
2. **Plain language cannot be generated from structure alone.** A rendering can
   state that a value is compared against a threshold. It cannot state *why* the
   Bank compares them without a human having written that down, attached to the
   thing itself, and kept it current. Whatever carries that description must be
   impossible to omit and must be reviewed when the logic it describes changes.
3. **The audience cannot tell you what they need until they see a wrong one.**
   The only way to know whether the artefact works is to hand a real credit risk
   analyst a real policy document and a generated rendering and ask them to find
   the three places they disagree. That exercise is people-blocked, not
   code-blocked, and should run before the format is fixed rather than after.
4. **If it cannot be met, several other things stop making sense.** The
   parameter/structure separation, the discipline of describing each stage, and
   the recording of cap chains are all partly justified by this artefact. If a
   reviewer cannot read the output, those justifications weaken and the
   governance story has no ending.

**What must be recorded.** Every rendering that was approved, with its approval,
frozen and retained. When the regulator asks what policy was in force in March
2027, this is the answer.

### 5.8 Coverage and dead logic

**Trigger.** Monthly, over production volume; and on demand before any release
that removes logic.

**What it determines.** Which logic is actually doing something.

**Measures**, per flow, over rolling 30- and 90-day windows of full production
volume:

- **Rule hit rate.** Evaluations, firings, firing rate, and the rate among
  records where the rule's preconditions were met (a rule that fires 100% of the
  time it is reached, but is reached 0.2% of the time, is a different animal from
  one that is always reached and never fires).
- **Dead rules.** Zero firings in 90 days. Project 01 adds ~30 rules a month and
  removes none without this measurement; over the eighteen months since go-live
  that is 540 additions, and experience in every fraud rule estate says a third
  of them stopped mattering within a quarter of being written.
- **Rules that are not rules.** Firing on more than 40% of traffic. A rule that
  fires on 40% of records is not a rule, it is a policy that someone wrote as a
  rule, and it should be recognised as such, named, and reviewed at the level a
  policy is reviewed at — not left in a list of 610 where it is invisible.
- **Unreachable tree branches.** Nodes with zero records in 90 days, and the
  condition that makes them unreachable — often another node upstream that was
  re-thresholded and now absorbs everything. Across 60 trees in project 04 with
  several hundred nodes each, this is not findable by reading.
- **Never-read table cells.** Of the 63 360 Flex Loan rate card cells, production
  typically reads about 11% in a year; the other 89% are either legitimately
  unused combinations or errors nobody will ever notice. The measurement does not
  say which, but it makes the question askable, and it makes a new card's changes
  to never-read regions correctly uninteresting.
- **Unreachable reason codes.** Of 380 registered codes, which have never been
  raised by any flow. A code that cannot be raised is either dead or a gap.
- **Shadowed logic.** Rules or nodes whose conditions are subsumed by an earlier
  one in priority order, so they can never fire regardless of data. This is a
  structural property and should be answerable without running anything.

**What it emits.** A monthly coverage report per flow, and a standing removal
candidate list that each flow's owner must either action or explicitly retain
with a reason, annually.

### 5.9 Input drift monitoring

**Trigger.** Daily, per flow, over the input inventory (§4.6).

**What it determines.** Whether the data arriving today resembles the data the
logic was built and calibrated on.

**Measures**, per feature, against a declared baseline window:

- Distribution shift: population stability index against baseline, banded at
  < 0.10 stable, 0.10–0.25 watch, > 0.25 alert; plus the deciles themselves,
  because the index alone hides which part of the distribution moved.
- Null rate, absolute and change. A feature whose null rate moves from 2% to 19%
  overnight is a broken upstream join, not a change in applicants.
- Category set: new codes appearing, known codes disappearing, cardinality
  change. Project 04's trees and project 08's matrices key on codes; an
  unrecognised code is a silent fall-through to a default arm.
- Volume and timing: record counts per source per day against expectation.

**The specific case that matters most.** An upstream source silently changing
**units, encoding or null behaviour**. Income arriving in cents rather than
rands. A date changing from ISO to a local format that parses successfully but
wrongly. A field whose "not applicable" changes from null to zero, or to −1, or
to the empty string. An account-type code set renumbered by a source system
upgrade.

These are the dangerous class because they **do not fail**. Nothing errors. The
flow runs, every record completes, latency is normal, the dashboards are green,
and every answer is wrong in a plausible direction — most often systematically
generous, because a hundredfold income makes everyone affordable. Detection must
therefore be distributional and relational, not exceptional:

- ratio tests against baseline central tendency that catch order-of-magnitude
  shifts (a median ratio outside [0.5, 2.0] alerts immediately, regardless of the
  stability index);
- internal consistency checks that must hold regardless of distribution — net
  income below gross, instalments below balances, dates ordered, sums of parts
  matching totals;
- cross-source agreement where two sources describe the same quantity;
- and a hard, declared expected range per feature from the input inventory, with
  out-of-range values counted rather than silently clipped.

**What it emits.** Daily alerts routed to the owning team and to Decision
Governance; a monthly drift summary; and, for any alert on a feature used by a
live flow, a required disposition within 5 working days. Expect ~40 alerts a
month across 1 240 features, most benign; the cost of the false ones is the price
of catching the units change.

### 5.10 Outcome monitoring

**Trigger.** Daily collection, monthly reporting.

**What it determines.** Whether the decisions themselves are moving, whether or
not the inputs are.

**Measures**, per flow, per product, per channel, per segment, per risk grade:

- approval, decline, refer and approve-with-conditions rates;
- decline reason distribution across the 380 registered codes, with month-on-month
  movement and a flag on any code whose share moves by more than 3 percentage
  points or which appears or disappears entirely;
- score distribution — mean, standard deviation, deciles — per scorecard, with
  the stability index against the calibration population;
- risk grade distribution, and migration against the prior month;
- offered amount and instalment distributions, and the gap between requested and
  offered;
- take-up rate by segment, which is where a pricing change shows up before it
  shows up anywhere else;
- referral volumes against operational capacity, and queue ageing;
- for project 07, portfolio budget consumption; for project 08, treatment mix and
  arrangement rates; for project 01, interdiction rate and the false-positive
  proxy.

**Reporting.** A monthly pack to the Credit Committee, generated, with
month-on-month and year-on-year movement, exceptions called out rather than left
to be found, and every material movement cross-referenced to the release that
caused it (§5.5) or explicitly marked as unattributed. An unattributed material
movement is an open item carried forward until it is explained.

### 5.11 The regulator and ombud pack

**Trigger.** An information request. About **40 a year**: roughly 25 from ombud
schemes on individual complaints, 10 from the regulator, 5 from external audit or
Internal Audit.

**Turnaround: 10 working days.** That is the binding constraint, and it is what
makes this a systems problem rather than an analyst's problem. Ten working days
is comfortable if the pack assembles in an afternoon and impossible if it
requires three teams to go looking.

**Contents**, for a single-decision request:

1. **The decision** — identifier, date, product, channel, outcome, reasons, and
   the amounts and terms offered or refused.
2. **The inputs** — as received, with their sources, their dates, and for bureau
   data the bureau, the reference and the as-at date.
3. **The explanation** — the adjudicator rendering (§5.2), which is the part that
   must stand on its own to a lay reader.
4. **The policy in force** — the approved plain-language rendering of the flow at
   that decision's date (§5.7), extracted to the parts that bore on this
   decision, with the full rendering available.
5. **The approval record for that policy** — who approved it, in which forum, on
   which date, with which supporting impact analysis. This is the part most often
   missing, and its absence converts a defensible decision into a governance
   finding.
6. **Evidence of consistent application** — that comparable applicants were
   treated comparably. Concretely: a cohort of the nearest comparable applicants
   — same product, same risk grade, same decision month, affordability and
   requested amount within a declared band, a target cohort size of 200 — with
   the outcome distribution across that cohort and the complainant's position
   within it. Where the complainant's outcome differs from the cohort's mode, the
   pack must state which input or rule accounts for the difference. This is what
   answers the question an adjudicator is actually asking, which is not "was the
   rule followed" but "was this person treated the way you treat people like
   them".
7. **Provenance** — a statement of how each element was produced, including the
   replay verdict (§5.1), so that the pack is itself auditable.

For a thematic request — "all declines on ground X between March 2027 and April
2028" — the same contents, aggregated, plus the per-decision detail for a sample
the requester specifies, plus the population definition used and the query that
produced it, reproducibly.

**What must be recorded.** Every pack issued, in full, immutably, with the
request it answered. The Bank must be able to show, three years later, exactly
what it sent.

### 5.12 Fairness and prohibited-basis testing

**Trigger.** Quarterly, per flow, and before any release that changes who is
approved.

**What it determines.** Evidence that no flow discriminates on a prohibited
ground, direct or indirect. The prohibition is statutory: a credit provider may
not unfairly discriminate on the grounds set out in the Constitution and the
equality legislation, while retaining the right to refuse on reasonable
commercial grounds consistent with its customary risk management practices. Both
halves of that sentence have to be evidenced.

**Direct use.** Every flow declares the inputs it consumes (§4.6), and the
declaration is checked automatically against the list of prohibited
characteristics. The check must be **structural, not sampled** — a feature cannot
be concluded unused because it did not appear in a sample. Where a prohibited
ground is used lawfully — age, which bears on capacity to contract and on credit
life premium rating (00 §6.8); marital status, which bears on joint applications
and matrimonial property regime — the use must be declared, justified in writing,
approved, and re-approved annually. Undeclared use of a declared-prohibited input
blocks the release.

**Proxy and outcome testing.** Direct exclusion is necessary and nowhere near
sufficient. Outcome rates by protected group, with statistical significance,
across approval rate, mean offered amount, mean rate and decline reason mix; and
proxy detection — the degree to which any input or combination of inputs predicts
a protected characteristic, particularly geography, which in South Africa proxies
for race with uncomfortable strength.

**The tension, stated rather than hidden.** The Bank does not collect several of
these attributes for retail credit, deliberately and for good reasons, which
means it cannot test on them. It holds date of birth and gender; it holds marital
status where a joint application was made; it holds residential geography. It
does not hold race, and collecting it in order to test for bias against it is a
decision with its own legal, reputational and data-protection consequences that
is not this project's to make. Three honest positions follow, and the Bank must
pick one per attribute rather than drifting:

1. test on what is held, and say explicitly what is not tested;
2. test on a proxy — geography, surname-based inference — and state the error
   rate of the proxy, accepting that a bad proxy can manufacture a finding as
   easily as it can hide one;
3. obtain the attribute under a lawful basis for the sole purpose of monitoring,
   ring-fenced from every decision flow, with technical enforcement of that
   ring-fence and evidence that the enforcement holds.

The harness's requirement is not to choose. It is to make the choice visible, to
implement whichever is chosen, and to state on every fairness report exactly
which attributes were tested, which were not, and why — so that the report is
never read as broader assurance than it is.

### 5.13 Cross-cutting: personal information

Every artefact in this project contains applicant financial data. A decision
record holds income, expenses, employment, bureau history, account balances,
arrears and a score. An explanation holds all of that plus the reasoning. A
golden set holds 50 000 real applicants per flow. A swap-set population holds 2 M.
A drift report holds distributions derived from all of it.

**Requirements.**

1. **A decision record is production data.** So is a trace, an explanation, a
   what-if result and a golden set. None of them is a debugging artefact that
   happens to contain data; each is personal information about an identified
   person, and inherits the classification of its inputs.
2. **Access is named, purposed and logged.** Access to unmasked evidence is
   limited to named individuals — currently about 40 across disputes, credit
   analysis, governance and audit — each access records the decision accessed, the
   person, the time and the stated purpose, and the access log is reviewed
   monthly. Bulk extraction requires two-person authorisation.
3. **Non-production use is masked by default.** Golden sets, developer
   environments, demonstration data and shared analyses use masked or synthetic
   derivatives. Masking must preserve the properties the logic depends on —
   distributions, correlations, boundary values, null patterns, collection
   cardinalities — or the golden set stops testing anything. Where a real record
   is required to reproduce a defect, its use is time-boxed, logged and approved.
4. **Retention is bounded by purpose.** 7 years for credit decisions, 5 for
   marketing, 24 months for what-if runs and replay records, 13 months for
   monitoring detail beyond which only aggregates are kept. Retention is enforced
   by deletion, and the deletion is evidenced. Keeping evidence longer than the
   obligation requires is not caution; it is a separate compliance problem.
5. **Deletion requests collide with retention obligations**, and the collision
   must be resolved in advance rather than per request. A statutory obligation to
   retain records of credit applications and of the affordability assessments
   behind them generally prevails over an erasure request — but the Bank must be
   able to state which records are retained under which obligation, delete
   everything not covered, and produce that reasoning within the statutory
   response window.
6. **Every recorded field carries a classification**, applied at emission by the
   flow, so that masking, redaction for a client-facing rendering, and export
   controls are automatic rather than a matter of somebody remembering.

**The conflict, named.** Everything above pushes decision evidence towards being
scarce, controlled and hard to reach. Everything in §5.2, §5.3 and §5.8 pushes it
towards being abundant, searchable and on every engineer's desk, because an
analyst who cannot see a trace cannot diagnose a defect and will instead guess,
or add logging, which is worse. These do not reconcile by policy statement. They
reconcile — partially — by making the masked derivative genuinely good enough for
most debugging, by making unmasked access fast to obtain and fully logged rather
than slow and therefore circumvented, and by accepting that a small number of
people will hold broad access and that the control on them is the log, not the
gate. An organisation that claims to have solved this has usually just made the
gate slow enough that people stopped asking and started copying.

### 5.14 Adjustment governance

**Trigger.** Continuous. Plus every Credit Committee meeting, every model
validation cycle, and every overlay's review date.

Six of the eight flows carry **adjustments** — named, approved, effective-dated
overlays layered over an already-validated artefact rather than edited into it
(00 §6.22). A score shift, a points-to-double-the-odds change, a probability
multiplier, a calibration re-anchor, a grade-boundary move, a cut-off shift, a
rate add-on in basis points, a cap reduction, an affordability buffer increase, a
matrix intensity dial. Between 40 and 120 are live across the estate at any time.

They are governed here rather than in the flows because an overlay changes
answers **with no code change, no structure change and no table change** — which
means every other mechanism in this project sees nothing happen while outcomes
move. An estate with overlays and without this capability is an estate where the
most frequently changed thing is the least governed.

#### 5.14.1 The live register

One view across all eight flows: every overlay with its id, kind, scope,
magnitude, position in its stack, owning flow, owner, approval reference,
rationale, effective window, review date and enabled state. It must be
answerable for any past date, not only today — "which overlays were in force on
14 September 2026, over what scope, in what order" is the question a replay and a
regulator both ask.

Scope is what makes this hard to assemble rather than merely tedious: an overlay
scoped to grades 9–12 on channel 4 for product 10 is not comparable, cell for
cell, with one scoped to a sector code in project 05. The register must
nevertheless support the two aggregate questions the committee asks — *how much
of the estate is currently overlaid* and *where do overlays concentrate*.

#### 5.14.2 Ageing and expiry

Every overlay carries a mandatory review date, and this capability is what makes
the word mandatory mean something. Reported monthly:

| Report | Why |
|---|---|
| Overlays past review date, by age | The dominant failure mode: a tightening applied in one bad quarter, still in force four years later |
| Overlays renewed more than twice without re-justification | Renewal as a formality rather than a decision |
| Overlays whose approving committee member or owning team no longer exists | Nobody left who can say why |
| Overlays never observed to fire | Scoped to a population that no longer occurs — dead logic, in the §5.8 sense |
| Overlays firing on more than 40% of a flow's volume | No longer an adjustment; it is the policy, and belongs in the base artefact under that artefact's approval |
| Estimated effect of unwinding each expired overlay | Because "we cannot remove it, we do not know what would happen" is the state this exists to prevent |

The last row is the one with teeth. An expiry report that cannot say what
unwinding would cost produces indefinite renewal, which is indistinguishable from
having no expiry at all.

#### 5.14.3 Running with the stack disabled

Every flow must be runnable with its overlays off, through the same
implementation. Three consumers need it and none of them is optional:

- **Model validation.** The validation opinion covers the model, not the policy
  overlay laid over it. Comparing predicted against realised default rates on
  adjusted values measures the overlay and reports it as model accuracy. Model
  Risk therefore needs the base model's own predictions on the production
  population, including where an overlay caused a decline.
- **The committee's before-and-after.** An overlay is approved on an estimate of
  its effect and reviewed on the realised one. Both are the difference between
  the stack on and the stack off over a real population.
- **Attribution.** §5.5's swap-set cannot separate an overlay change from a logic
  change unless it can hold one of them fixed.

The requirement is explicitly **one implementation**. A separate "unadjusted"
calculation is the same defect as a separate simulation implementation in project
07 and a separate backtest implementation in project 01: it will agree at first
and diverge silently, and the divergence will be discovered during the exercise
that most needed it to be right.

Cost constrains the shape. Running every flow both ways for every decision is not
affordable at 22 M decisions a year — project 03 alone is 55 000 applications a
day through a search that dominates its cost. So: the *inputs* to the overlay
(`score_unadjusted`, `probability_of_default_unadjusted`, the unadjusted grade,
the unadjusted cap chain, the card cell before the add-on) are recorded on every
decision, being cheap; the *counterfactual outcome* — what the flow would have
decided — is produced on demand, over samples and over full populations for
impact estimation, by the same implementation.

#### 5.14.4 Attribution of a swap to an overlay

When outcomes move, the estate has four candidate causes: the client population,
the input data, the logic, and the overlays. Overlays are the hardest of the four
because they leave no trace in any version-controlled artefact.

This capability must therefore contribute an increment to §5.5's ordered
attribution — the overlay set is one of the increments, applied at a declared
position — and must be able to answer the committee's standing question: *of this
quarter's decline-rate movement, how much was the population, how much was the
model, and how much was what we ourselves approved?*

#### 5.14.5 What replay and explanation require

- **Replay (§5.1)** pins the overlay stack as tightly as it pins a rate card
  version. A decision replayed without its overlays reproduces a plausible,
  internally consistent, wrong answer — which is worse than a failure, because
  nothing flags it.
- **Explanation (§5.2)** shows the adjusted and unadjusted values side by side
  for all three audiences. For the ombud adjudicator this is not a technical
  nicety: whether a decline came from the applicant's own circumstances or from
  the Bank's deliberate conservatism is a different question with a different
  answer, and the adjudicator is entitled to know which they are looking at.
- **Certification (§5.6)** runs the golden sets against base logic with overlays
  held fixed, because an overlay change and a logic defect are otherwise
  indistinguishable in a regression result.
- **The reviewable artefact (§5.7)** renders the base artefact and its overlays
  as two visibly separate things. A rendering that shows a single effective grade
  boundary, with no indication that policy moved it, misrepresents what was
  approved.

#### 5.14.6 The asymmetry

Most overlays in this estate may only tighten. Project 02's may not increase
affordability capacity; project 03's may not raise a ceiling or price below the
card; project 08's may not weaken a regulatory suspension, a notice period or a
contact frequency cap. This capability holds the register of which asymmetry
applies to which overlay kind in which flow, and verifies that each flow enforces
it at definition time rather than at run time. An overlay that can be made to
loosen by supplying a negative magnitude has no asymmetry at all, and finding
that out belongs here rather than in an incident.

**What must be recorded.** Every change to the register, with before and after,
author and approval; every expiry report and what was done about it; every
stack-disabled run, its population and its result; and the overlay increment of
every swap-set attribution.

### 5.15 The eight-flow contract

Everything above is achievable only if the eight flows satisfy the following.
These are not recommendations. Each of them, absent, disables at least one
capability above permanently and retrospectively.

1. **Stable decision identifier.** Globally unique, assigned before any logic
   runs, present on every record and every artefact derived from it, never reused,
   and resolvable to the flow, the flow version and the decision date.
2. **Stable identity for every element of logic.** Rules, gates, tree nodes,
   matrix cells, scorecard characteristics and outcomes carry identifiers that are
   content-derived or explicitly declared — never positional, never
   auto-generated, never dependent on ordering. An identifier that changes when a
   rule is inserted above it makes every historical record uncomparable, and this
   is the single most common way an audit trail is destroyed by a routine
   refactor. Renaming or retiring an identifier is a declared, versioned event.
3. **Deterministic assignment.** Every holdout group, challenger split,
   champion/challenger allocation, queue assignment and sampling flag is derived
   from a stable identifier and a named, recorded seed. No call to a random
   number generator, ever, anywhere in a flow. The derived value is recorded as
   well as the derivation, so that a change to the derivation is detectable.
4. **No reliance on "today".** Every date-sensitive selection resolves against
   `decision_date` (00 §7.3). A flow that reads the current date anywhere cannot
   be replayed, and — worse — will replay *successfully* with the wrong answer.
5. **Recorded reference-data versions, to the cell.** Every table, norm, cap,
   rate card, scorecard, grade boundary set, appetite grid and reason registry
   read records which version and which cell (00 §8). A version identifier that
   was later overwritten in place is not a version identifier.
6. **Inputs captured as received.** Before normalisation or defaulting, with the
   three null situations preserved distinctly (00 §7.4), and with the source and
   as-at date of each.
7. **The overlay stack recorded on every decision.** The adjustment set
   identifier, the overlays that applied, each one's effect, and the composition
   order — plus the unadjusted inputs each overlay consumed. An overlay is the
   only thing in the estate that changes an answer without changing an artefact,
   so a flow that does not record it has decisions whose cause is unrecoverable.
   A flow must also be runnable with its stack disabled through the same
   implementation (§5.14.3).
8. **Mutable state snapshotted or addressable.** Any read of a live store —
   account balances, delinquency state, arrangement history, existing exposure,
   pending applications — is either captured in full or captured as an address
   into a store that retains that state as at that instant. Project 07 reads the
   revolving book monthly and project 08 reads delinquency state daily; neither
   store today retains history, and that is a replay defect in those flows, not
   in this one.
9. **No external calls during replay.** Evidence must be sufficient to re-derive
   the decision with no network and no live service. This is the test of whether
   capture was complete, and it must be enforced by the replay environment rather
   than trusted.
10. **Explicit, complete parameter sets.** The recorded set names every value that
   was in force, including every value that fell back to a built-in default,
   marked as such. No decision may depend on a value that is not in its record.
11. **Declared reason codes.** Every non-trivial outcome carries codes drawn from
    the registry, with the registry version, in severity order, plus the
    designated primary. A flow that produces a decline without a registered code
    cannot be explained to the person it declined.
12. **Declared expected effect on every change.** Each release states which
    population it expects to move, in which direction, by roughly how much, or
    declares that it expects no behavioural effect at all (§5.6).
13. **Individually attributable changes.** Each change within a release is
    separately identifiable and separately replayable, or the release is split
    (§5.5).
14. **Evaluation recorded, not only firing.** Which gates and rules were
    *evaluated*, with their outcomes — not only which fired. Without this,
    "why did rule 212 not fire" is unanswerable and dead-rule measurement
    degenerates into counting firings, which cannot distinguish a rule that is
    never reached from one that is reached and never satisfied.
15. **Cap chains recorded.** For every quantity that can be successively
    constrained, the ordered list of values it took and what caused each
    transition. The final value alone is not an explanation.
16. **Score contributions as output.** Per-characteristic, signed, with the bin
    each value fell into. Adverse-action explanation depends on them, so they are
    a required output and not a diagnostic (00 §6.10).
17. **Idempotent, at-least-once evidence emission.** Evidence survives a retry
    and is deduplicated by decision identifier. A decision retried three times
    produces one evidence record, not three, and not zero.
18. **Evidence emission cannot fail the decision, and a failure to persist must
    be visible.** A real-time flow must not decline an applicant because an
    evidence store was unavailable. It must also not silently lose the record:
    failures are counted, alerted and reconciled, and the count of decisions
    without evidence is reported monthly. Today that number is unknown for at
    least three of the eight flows, which is itself the finding.
19. **A declared input inventory.** Every input feature named, typed, ranged,
    classified for PII and for prohibited-ground proximity. A feature not in the
    inventory is unmonitored, and its absence must be detectable by comparing the
    inventory against what the flow actually reads.
20. **Runnable outside production.** A flow can be executed against supplied
    evidence with no scheduler, no queue, no live service and no ambient
    environment. If a flow can only run inside its production deployment, it can
    never be replayed, certified, swap-set or what-iffed.
21. **Versioned, resolvable logic identity.** A build identifier that resolves,
    for the full retention period, to exactly the logic that ran — and to the
    approved plain-language rendering that corresponds to it.
22. **Declared prohibited-ground usage.** Each flow declares which inputs it
    uses and asserts that none is a prohibited characteristic except where
    declared, justified and approved. The assertion is machine-checkable against
    the inventory (§5.12).
23. **Field-level PII classification at emission.** Applied by the flow that
    produces the value, not inferred later by the harness, so that masking and
    redaction are automatic.

---

## 6. Parameters and tables

This project owns no credit policy. Its tunables are the settings that decide
what gets caught, what gets escalated and what gets kept.

| Parameter / table | Size | Owner | Source | Cadence |
|---|---|---|---|---|
| Retention periods, by decision class | 6 classes | Information Officer | Statutory obligation + internal policy | Annual review |
| Replay tolerance bands, by output class | 5 classes × 8 flows | Decision Governance | Internal | Annual |
| Non-reproducible decision ceiling | 0.05% per flow per month | Credit Committee | Internal | Annual |
| Golden set size and stratification | 50 000 per flow × 8 flows, ~14 strata each | Flow owner + Decision Governance | Internal | Quarterly refresh |
| Certification coverage thresholds | 4 thresholds × 8 flows | Decision Governance | Internal | Annual |
| CI time budget | 40 minutes total; 6 minutes per flow locally | Engineering leadership | Internal | Annual |
| Drift thresholds per feature | 1 240 features × 4 thresholds | Owning team, approved by Decision Governance | Baseline study | Per baseline refresh, ~annual |
| Drift baseline windows | 1 per feature | Owning team | Production history | Annual, or on material change |
| Outcome monitoring alert thresholds | ~90 measures × 8 flows | Credit Risk Policy | Internal | Semi-annual |
| Dead-rule and dominant-rule thresholds | 0 firings / 90 days; 40% of traffic | Decision Governance | Internal | Annual |
| Comparability cohort definition | Per product: keys, bands, target size 200 | Compliance | Internal | Annual |
| PII field classifications | ~1 240 features + ~600 derived values | Information Officer | Internal | Continuous |
| Unmasked-access named list | ~40 people | Information Officer | Internal | Quarterly re-attestation |
| Masking rules by classification | 7 classes | Information Officer | Internal | Annual |
| Prohibited-ground list and permitted-use exceptions | 17 grounds, 4 declared exceptions | Compliance | Constitution + equality legislation | On legislative change |
| Adjustment register, all flows | 40–120 live overlays × 13 attributes | Credit Committee per flow; consolidated view by Decision Governance | Flow owners | Ad hoc, sometimes weekly |
| Overlay ageing thresholds | 4 thresholds | Decision Governance | Internal | Annual |
| Overlay dominance threshold | 40% of a flow's volume | Credit Committee | Internal | Annual |
| Overlay asymmetry register | 10 kinds × 8 flows | Compliance | Internal | On new overlay kind |
| Swap-set population definitions | 8 flows × 1 standing definition | Decision Governance | Production | Monthly |
| Regulator pack templates | 4 request types | Compliance | Internal + requester format | Annual |

Two properties are required of every one of these, and they are the same two the
library requires of its tables (00 §8): the value in force at any past date must
be recoverable, and a change to any of them must produce a reviewable record. A
governance harness whose own settings are unversioned is not a governance
harness.

---

## 7. Outputs

**Produced on demand:**

| Output | Consumer | Turnaround |
|---|---|---|
| Replay verdict and re-derived outputs | Analyst, audit, disputes | < 5 s hot, < 6 h from cold storage |
| Explanation — consultant rendering | Contact centre | < 3 s |
| Explanation — analyst rendering | Credit analyst | < 15 s |
| Explanation — adjudicator rendering | Disputes, Compliance | < 1 working day, reviewed before issue |
| What-if result | Credit analyst | < 5 s |
| Version change record | Approver, Credit Committee | On demand, generated at change |
| Swap-set report | Credit Committee, flow owner | < 30 min per 2 M-record comparison |
| Certification sign-off artefact | Release approver, Internal Audit | At release, blocking |
| Reviewable rendering and its diff | Credit Risk Policy, Compliance, Credit Committee | At change; quarterly re-approval |
| Regulator / ombud pack | Regulator, ombud, external audit | ≤ 10 working days |

**Produced on a cycle:** daily drift alerts; monthly coverage report per flow;
monthly outcome pack to the Credit Committee; quarterly fairness report;
quarterly access-list attestation; annual seven-year replay sample.

**Persisted, immutably:** every decision's evidence for its retention period;
every replay verdict and what-if run for 24 months; every approved rendering and
its approval for as long as any decision made under it is retained; every
certification artefact for 7 years; every issued pack and every client-facing
explanation for 7 years; every change record for the life of the version it
describes; every access to unmasked evidence for 3 years.

---

## 8. Non-functional requirements

### 8.1 Latency and throughput

| Requirement | Value |
|---|---|
| Single-decision replay, evidence in the hot tier | < 5 s at p95, including evidence retrieval |
| Single-decision replay, cold tier (older than 24 months) | Retrieval within 6 working hours, then < 5 s |
| What-if re-run | < 5 s at p95 |
| Consultant explanation render | < 3 s at p95, from a call-centre desktop |
| Swap-set over 2 M records | < 30 minutes, single comparison |
| Full-release attributed swap set, 5 increments | < 2.5 hours |
| Certification suite, all 8 flows, 400 000 records | < 40 minutes in CI |
| Certification suite, one flow, 50 000 records, locally | < 6 minutes |
| Coverage aggregation over a month of full volume | < 2 hours |
| Drift computation, 1 240 features, daily | < 45 minutes |
| Evidence write overhead, real-time flows | < 1.5 ms added at p99 to a 40 ms budget |
| Evidence durability, real-time flows | Within 2 s of the decision |

### 8.2 Availability and failure behaviour

Evidence capture is on the critical path for correctness and must never be on the
critical path for availability. A flow must continue to decide when the evidence
store is degraded, buffering locally and reconciling afterwards, and must count
and alert on every record it could not persist. The harness itself is not
real-time: an outage of the replay or reporting surfaces is an inconvenience, not
an incident, with the single exception of the consultant explanation, which sits
in front of 31 000 client calls a year and requires 99.5% availability during
business hours.

### 8.3 Storage

Sizing at 22 M decisions a year, illustrative but not conservative:

| Component | Mean size | Annual |
|---|---|---|
| Inputs as received, credit flows (bureau payload dominates) | 46 KB compressed | 790 GB |
| — after content-addressed de-duplication of shared bureau views | | ~515 GB |
| Decision summary, versions, parameter set reference, reasons | 3.1 KB | 68 GB |
| Marketing decisions (04) | 1.1 KB | 5 GB |
| Reference-data version store, change records, renderings, artefacts | | ~50 GB |
| **Annual total** | | **~640 GB** |
| **Seven-year window** | | **~4.5 TB** |
| Provisioned, with indices, replicas and immutability overhead | | **~9 TB** |

Against that, the naive alternative: recording every intermediate value for every
decision averages ~140 KB, which is **3.1 TB a year and 22 TB over the window** —
roughly five times the total cost of everything else, for evidence that is almost
never read. The question of what constitutes *sufficient* evidence is therefore
an economic one as well as a legal one, and it has to be answered deliberately
rather than by whichever default the implementation happens to have. The current
position: full evidence for everything in the §5.15 contract, always; full
intermediate detail only for a deterministic 0.5% sample plus 100% of declines,
referrals and any decision subsequently disputed. Declines and referrals are
~18% of volume, so the sample costs ~570 GB a year rather than 3.1 TB.

The uncomfortable part is that the 0.5% sample is chosen before anyone knows
which decision will be disputed. Retrospective enrichment is impossible; the
mitigation is that a replay (§5.1) regenerates the intermediate detail on demand,
which is precisely why the replay guarantee has to be exact rather than
approximate. **Replay is what makes it affordable not to store everything**, and
if replay is not exact, the storage bill is 22 TB.

### 8.4 Determinism

Identical inputs, identical versions, identical `decision_date` ⇒ identical
outputs, bit for bit, on any machine, in any year within the retention window
(00 §9). Every capability in §5 depends on it. A flow that is deterministic in
practice but not by construction will eventually not be.

---

## 9. Audit, evidence and explainability

This entire document is §9 for the other eight projects. What this section adds
is what must be true of the *harness itself*.

1. **The harness is in scope for audit.** Internal Audit tests the harness, not
   merely through it: sampling decisions, running replays independently,
   confirming that a certification failure actually blocked a release, and
   confirming that an approved rendering matches deployed logic.
2. **The harness's own changes are governed** under the same regime it imposes:
   versioned, diffable, certified, approved. A change to a tolerance band is a
   governance change and is approved as one.
3. **The harness must not be able to alter the evidence.** Decision evidence is
   written once by the flow and never modified — not corrected, not enriched, not
   re-keyed. Corrections are new records that reference the original. Write-once
   storage with independently verifiable integrity, so that "the evidence was
   edited" is a question with an answer.
4. **Gaps must be visible.** The count of decisions without evidence, without a
   resolvable logic version, without a recorded table version, or that fail
   replay, is reported monthly per flow. A flow reporting zero for all of these
   because it does not measure them is the worst case, so the measurement itself
   is what is audited.
5. **Model risk management.** The nine scorecards, the flows that consume them
   and the material judgemental overlays sit in a model inventory subject to
   independent validation, in line with the supervisory expectation that models
   be soundly developed, independently validated and governed by policy. The
   harness supplies validation with: the ability to replicate any scored decision
   independently; population and score stability over time; the swap-set and
   coverage evidence for every change; and the approval chain for every override
   of a model's output. Validation forms the opinion; the harness supplies the
   facts, and an opinion that cannot be supported by reproducible facts is a
   finding against the Bank.

---

## 10. Acceptance criteria

1. Any decision from any of the eight flows, from any date in its retention
   window, replays to a bit-identical outcome, or to a documented and bounded
   difference, in under 5 seconds from the hot tier.
2. Fewer than 0.05% of decisions per flow per month are non-reproducible, and
   each one is individually explained.
3. A contact-centre consultant, with no credit training, can explain a decline
   accurately to a client using only the consultant rendering, within 90 seconds,
   without opening anything else.
4. A credit analyst can change one input and re-run a historical decision in
   under 5 seconds, and the result cannot be mistaken for, exported as, or issued
   as a real decision.
5. Every change to every flow produces a change record naming what, who, when,
   under whose approval, and with what estimated impact — generated, not written.
6. Every release passes certification before promotion, and a change declared to
   have no expected effect that moves any golden output is blocked.
7. Every flow's approved plain-language rendering is generated from what runs,
   and a divergence between approved rendering and deployed logic is structurally
   impossible rather than merely unlikely.
8. Credit Risk Policy approves a flow's rendering without reading code, and can
   locate a specific policy rule within it in under two minutes.
9. Every release of more than one change reports a swap set attributed per
   change, or is split.
10. A dead rule, an unreachable tree branch and a never-read table cell are each
    visible in the monthly coverage report within 90 days of becoming so.
11. An upstream source changing income from rands to cents is detected within one
    day, by distribution rather than by error.
12. A regulator or ombud pack is assembled and issued within 10 working days,
    including the consistency evidence, for all 40 requests a year.
13. Every fairness report states which protected attributes were tested and which
    were not, and why.
14. No non-production environment contains unmasked applicant data, and every
    access to unmasked evidence is attributable to a named person and a stated
    purpose.
15. The annual seven-year replay sample — 1 000 decisions from each of the seven
    preceding years — passes.
16. A ninth flow can be brought under the harness by satisfying the §5.15
    contract, with no change to the harness.
17. The overlay stack in force on any past date is recoverable for every flow,
    with each overlay's scope, position, approval and expiry.
18. Every flow can be run with its overlay stack disabled through the same
    implementation, and a full-population run both ways is available for impact
    estimation.
19. No live overlay is past its review date without a recorded renewal decision,
    and the estimated effect of unwinding each expired overlay is reportable.
20. A swap-set attributes outcome movement between population, data, logic and
    overlays, with the overlay increment applied at a declared position.
21. Every decision carries the unadjusted inputs each overlay consumed, whatever
    the outcome, including declines.
22. Each flow's overlay asymmetry — which kinds may only tighten — is enforced at
    definition time, verified independently by this project, and not bypassable
    by a negative magnitude.

---

## 11. Change scenarios

1. **The regulator requests every decline on a specified ground over a 14-month
   period** — approximately 38 000 decisions — with the policy in force for each
   and evidence of consistent treatment, in 10 working days.
2. **A scorecard is retired and replaced.** Decisions made under the old one must
   remain replayable for 7 years after its retirement, which means the retired
   scorecard, its calibration and its bin definitions must outlive the system
   that used it by most of a decade.
3. **Project 04 refactors its trees** and renumbers several hundred nodes.
   Historical path records now point at nodes that mean something else.
   Recovering from this is either a mapping exercise or a permanent hole, and the
   only real answer was constraint 2 of §5.15, eighteen months earlier.
4. **A bureau restates 140 000 records retrospectively** after correcting a
   feed defect. Every affected decision now replays differently if the bureau is
   re-queried — which is why it must not be. The captured view is the record; the
   restatement is a separate fact about the world.
5. **The Credit Committee asks for an attributed swap set on every release**,
   including the weekly ones, not only the material ones. Compute cost goes from
   ~30 hours a year to ~260, and the constraint moves from acceptable to binding.
6. **An ombud determination requires the Bank to demonstrate**, for one
   complainant, how 200 comparable applicants were treated — including the ones
   who were approved, with the inputs that distinguished them.
7. **Project 08 moves from daily to intra-day**, ×6 volume for that flow and an
   evidence write rate that no longer fits the batch reconciliation window.
8. **Compliance adds 40 reason codes and reclassifies 12** (00 §11.4). Every
   historical explanation must continue to render with the wording and
   classification in force at its decision date, not today's.
9. **A data subject requests erasure** of everything the Bank holds about them,
   including three declined applications from 2029. The retention obligation and
   the erasure right collide, and the Bank has 30 days to respond with a
   defensible position and to actually delete whatever is not covered.
10. **An acquisition adds a ninth flow**, built by a team that has never heard of
    the §5.15 contract, already live, with two years of decisions behind it and
    no evidence for any of them.
11. **An engineer discovers that one flow has been reading the current date**
    rather than `decision_date` for a single threshold since go-live. Every
    decision it made is now non-reproducible in a specific, bounded way, and the
    Bank must determine how many, quantify the effect, and decide what to tell
    whom.
12. **Storage costs are challenged in a cost review** and the harness is asked to
    halve them without reducing what can be answered.
13. **Model Risk rejects a validation** because the monitoring pack compared
    realised defaults against *adjusted* predictions, measuring the overlay
    rather than the model. Every monitoring pack for four scorecards across three
    flows must be reproduced on base predictions, retrospectively.
14. **A decline-rate movement of 4 percentage points** is put to the Credit
    Committee, which asks how much was the client population, how much the model
    and how much the overlay set it approved itself two quarters earlier.
15. **An overlay approved for one quarter in 2023 is found still live**, its
    author gone and its approving committee reconstituted. The Bank must say what
    it has cost, what unwinding it would do, and how many other overlays are in
    the same position.
16. **An overlay is found firing on 62% of a flow's volume.** It is no longer an
    adjustment; it is the policy, under the wrong approval, outside the artefact
    that governs it — and moving it into the base artefact must be doable without
    breaking the comparability of two years of decisions.
17. **A flow is found to have merged an overlay into its base table** during a
    refactor "to simplify", destroying the separation between what the model said
    and what policy decided for every decision since.

---

## 12. Out of scope

- **Model development and validation itself.** The harness supplies validation
  with reproducible facts; the validation opinion is Model Validation's.
- **Upstream data lineage** inside the warehouse and source systems, before a
  value reaches a flow. The harness records what arrived, not how it got there.
- **Complaint and dispute case management.** The workflow, correspondence and
  case files live elsewhere; the harness supplies the evidence that goes into
  them.
- **The wording of client-facing statements of reasons.** Compliance owns the
  words; the registry holds them; the harness selects the correct version and
  renders it.
- **Contract, disbursement and servicing records.** A different retention regime
  and a different system of record.
- **Financial-crime investigation.** Project 01 supplies interdiction decisions
  and the harness explains them; the investigation itself is not here.
- **General application and infrastructure observability** — logs, metrics,
  uptime, error rates. Necessary, and a different discipline; per-decision
  evidence is not a logging concern and must not become one.
- **Authoring surfaces.** How a rule, a tree or a rate card gets written is
  projects 01, 04 and 00's problem. How the result is diffed, certified and
  reviewed is this one's.

---

## 13. Questions the implementation must answer

1. **What, exactly, is the minimum recorded artefact from which a decision can be
   re-derived bit-identically** — and is it one artefact or an assembly of
   several produced by different parts of a run? If it is an assembly, what
   guarantees it is complete, and how is incompleteness detected at the time
   rather than at replay seven years later?
2. **How is the identity of "the logic that ran" expressed** such that it
   resolves unambiguously for seven years, survives every refactor, and cannot be
   satisfied by a tag that was later moved?
3. **What makes an identifier stable** for a rule, a tree node, a matrix cell or
   a scorecard characteristic — content-derived, declared, or registered — and
   what happens at the moment somebody legitimately needs to change one?
4. **Can evidence emission be a property of the framework rather than of each
   flow?** If a team has to remember to record something, some team will not, and
   the gap will be found by an auditor. What can be made automatic, and what
   irreducibly requires a human to declare an intent?
5. **What is the cost of always-on evidence**, and where is the line between what
   is always recorded, what is sampled, and what is regenerated on demand by
   replay? The answer is a five-times storage difference, so it is a design
   decision and not a default.
6. **Is replay a re-execution of the same logic, or a separate interpreter?** A
   re-execution risks not being available in seven years; an interpreter risks not
   agreeing with production. What makes either trustworthy, and how is the
   agreement demonstrated rather than assumed?
7. **How are two versions run side by side over the same population** at 2 M
   records in 30 minutes, when the two versions may have different inputs,
   different parameter shapes and different outputs?
8. **How is a change decomposed into individually attributable increments**, and
   what does the implementation have to look like for that decomposition to be
   mechanical rather than a manual re-staging of the release?
9. **What does the reviewable artefact actually look like** for a 610-rule flow
   and a 60-tree flow, such that a non-technical approver reads it and finds a
   discrepancy against a policy document? This is the open risk the framework docs
   rank highest, it is people-blocked rather than code-blocked, and it should be
   tested against a real reviewer before the format is fixed.
10. **How is the description that makes a rendering readable kept honest?**
    Somebody must write why a comparison exists; what makes that impossible to
    omit, and what makes it get updated when the logic it describes changes?
11. **What does a diff of a flow mean** when the change is structural — a rule
    inserted mid-priority, a tree re-parented, a stage reordered? Value diffs are
    easy; structural diffs are where reviewers actually need help.
12. **How is "which rules were evaluated but did not fire" recorded** at 22 M
    decisions a year without recording everything, and is it derivable by replay
    instead of stored?
13. **Can coverage be measured uniformly** across rules, tree nodes, table cells,
    scorecard characteristics and reason codes, or does each shape need its own
    measurement — and if the latter, what does that say about whether they have
    anything in common (Q5)?
14. **How does the what-if result stay structurally incapable** of being mistaken
    for a real decision, once it has been exported to a spreadsheet and emailed?
15. **How does a flow declare its complete input set** so that prohibited-ground
    checking and drift monitoring are structural rather than sampled, and so that
    an undeclared input is an error rather than an omission?
16. **What is the masked derivative of a golden set**, such that it preserves
    boundary values, null patterns, correlations and collection cardinalities well
    enough to still catch defects, while containing no real person?
17. **How does the evidence format survive seven years** of library, language and
    storage change, and what obligation attaches to a format change with 4.5 TB of
    history behind it?
18. **What happens to everything above when a flow does not comply with §5.15?**
    Is there a partial mode, or is non-compliance simply a flow that cannot be
    governed — and if the latter, who is empowered to stop it shipping?
19. **How is an overlay governed when it is the one thing that changes an answer
    without changing an artefact?** Every other mechanism in this project —
    version diff, certification, coverage, the reviewable artefact — is triggered
    by something changing in a versioned thing. An overlay changes outcomes while
    every one of those reports nothing.
20. **How does a flow run with its overlay stack disabled**, through one
    implementation, when running both ways for every decision is unaffordable and
    running a second implementation is the defect this project exists to catch?
21. **What is the unit of comparison for overlay scope?** A grade-and-channel
    overlay in project 03 and a sector overlay in project 05 must both appear in
    one register and answer the same two aggregate questions, without flattening
    them into something that means nothing.
22. **How is an expiry made to have teeth?** A review date nobody can act on —
    because the effect of unwinding is unknown — produces indefinite renewal,
    which is no expiry at all. What must be true for the unwinding estimate to be
    cheap enough to produce monthly?
