# 04 — Retail campaign targeting trees

> Fictional. The Bank, its products, thresholds, table dimensions and volumetrics
> are invented for this repository. Regulatory mechanisms referred to are the
> published public ones; every number attached to them is illustrative.

---

## 1. What this is

Once a month the Bank decides, for each of 14.2 million clients, which of its
live credit campaigns that client should be targeted for, at what offer tier,
for what amount, through which channel, and in what order of preference. Between
monthly runs it repeats the exercise daily for the 150 000 – 400 000 clients
whose circumstances have changed. The output drives pre-approved offers in the
app, SMS campaigns, outbound call lists and branch prompts, and it is the single
largest source of new unsecured lending the Bank writes.

The targeting policies are decision trees. That is not a modelling convenience —
it is how the business works. Campaign analysts build trees in a modelling tool
against a historical sample, argue about them in a forum, and publish them.
There are 30 to 60 live at any time, one per campaign, each 4 to 12 levels deep
and 20 to 400 nodes wide. Nobody in the business thinks about targeting in any
other shape, and any attempt to make them think differently has failed twice.

The requirement that distinguishes this project from every other in the set is
that **the path each client took through each tree must be captured, stored, and
queried afterwards**. The analytics team's entire measurement approach is built
on it: node-level volume reporting, response by leaf, dead-branch detection,
diagnosis of volume movements, and the ability to answer "why did this client get
this offer" two years later without re-running anything. A targeting system that
produces correct answers and no paths is, for this business, a failed system.

---

## 2. Why it is in this set

| Question | How this project stresses it |
|---|---|
| **Q5 core component set** | `decider2` names decision trees as a core kind. This project asks what that kind must actually do: 4–12 levels, nodes testing several features at once rather than one per level, comparisons between two features, set membership over categorical codes, 60 of them live, and — the hard part — an obligation to emit the traversal itself as first-class output rather than as a debugging aid. Whether a tree kind that satisfies §5.4 is recognisably the same construct as a tree kind that does not is the question. |
| **Q7 audit and debugging** | Three different interrogators with three different timescales: a client disputing an offer within days, an analyst diagnosing a volume drop within hours, a regulator asking within weeks whether targeting discriminates on a prohibited basis across every live tree. All three are answered from stored evidence, not from a re-run. |
| **Q3 parameters** | Tree structure arrives from outside engineering, authored by analysts in a modelling tool and exported. Thresholds, band definitions, priority weights, capacity limits, fatigue caps and control percentages are all tunable by different owners on different cadences. The structure is the parameter, which is a harder version of Q3 than any other project in this set poses. On top of all of it sit adjustments — approved, expiring overlays that change a published tree's answer without editing the tree — arriving on a different approval path and a different clock from the artefact they modify. |
| **Q6 codebase organisation** | 60 campaigns, each with a tree, a candidate population definition, offer tiers, channel rules and a sign-off history, changing weekly, owned by eleven campaign owners. The artefacts must stay navigable when campaign 41 is retired and campaign 62 is added on a Thursday. |
| Q1 reuse | Consumes `core.consent`, `core.eligibility`, `core.scorecard` and `core.appetite` unchanged, and consumes project 03 in batch for offer amounts. Suppression logic and fatigue logic are candidates for reuse by project 07 and project 08. |
| Q2 tables | Feature band definitions, channel capacity, fatigue rules, campaign registry, tier definitions, suppression registry. Individually small; collectively the thing that changes most often. |
| Q4 custom units | Suppression evaluation, path recording, hash-based group assignment and arbitration are used by every campaign and must exist once. |

---

## 3. Actors

| Actor | Responsibility |
|---|---|
| **Campaign Analytics** | Builds and owns the trees. Builds them in a modelling tool against a 24-month sample, exports them, and expects them in production without an engineer. Owns the measurement approach that path capture exists to serve. |
| **Campaign owners** (×11) | Own individual campaigns: the offer, the tiers, the priority weight, the channel preference, the expiry. Change thresholds weekly. |
| **Credit Risk Policy** | Owns the risk floors every credit campaign's tree must respect, the appetite settings consumed via `core.appetite`, the pre-assessment tolerance, and every risk overlay (`§5.3.4`). Co-signs every tree change on a credit campaign. |
| **Regulatory Compliance** | Owns the do-not-target list, the prohibited feature register, marketing consent interpretation, and the proxy-variable attestation. Can suspend a campaign unilaterally. |
| **Channel Operations** | Owns channel capacity, cut-off times, blackout windows and the dispatch contract. Tells the Bank in March that outbound call capacity is being cut in April. |
| **Contact Centre** | Consumes outbound call lists. Needs, per client, enough evidence to hold a conversation and to answer "why am I getting this call". |
| **Decision Platform engineering** | Runs the monthly and daily cycles. Owns determinism, re-runnability and the six-hour window. Does not own any tree. |
| **Data Engineering** | Owns the client feature mart and its freshness contract. |
| **Internal Audit** | Asks, 18 months later, which clients were excluded from campaign 17 and on what basis. |

---

## 4. Inputs

### 4.1 The client feature mart

| Property | Value |
|---|---|
| Rows | 14 200 000 clients, one row per `client_id` |
| Columns | ~600 features |
| Freshness | Monthly cycle: as at the last calendar day, closed 22:00 on the last business day. Daily delta: as at 20:00 the previous day |
| Nullability | ~38% of features nullable; three null situations are distinguished per `00 §7.4` |
| Volume on disk | ~68 GB per monthly snapshot, retained 25 months |

Feature families:

| Family | Count | Examples | Typical nullability |
|---|---|---|---|
| Demographics and relationship | 42 | age, tenure, province code, employment type, income band code, segment code | Low |
| Product holdings and balances | 118 | per-product held flag, balance, limit, utilisation, months on book, term remaining, settlement ratio | Structural nulls where product not held |
| Behavioural aggregates, 3/6/12-month windows | 186 | salary credit count and regularity, average balance, turnover, debit order returns, missed payments, arrears months, spend by merchant class | High for clients under 12 months' tenure |
| Bureau summary | 74 | enquiry counts over 3/6/12 months, external unsecured balance and growth, worst external status, adverse event counts and ages, thin-file indicator | ~4% no bureau record |
| Internal risk scores | 26 | behaviour score, application score, propensity scores, `risk_grade`, `probability_of_default` | Low; nulls are a scoring bin |
| Channel engagement | 61 | app logins over 30/90 days, SMS response rate, email open rate, call answer rate, branch visits, preferred channel code | Zero and null are distinct |
| Prior campaign exposure and response | 79 | contacts in last 30/90/365 days by channel, offers presented, offers declined, take-ups, opt-out events, last contact date per campaign family | Structural nulls for never-contacted |
| Derived bands and flags | 14 | banded forms of continuous features, per `§6.1` | None |

**Type stability is a contract.** A feature's type and permitted value set may not
change without notice to every tree that references it. Eleven trees referencing
a feature that Data Engineering has silently widened from `int8` to `int16` is a
failure mode the design must prevent, not detect afterwards.

### 4.2 Reference inputs

| Input | Shape | Freshness |
|---|---|---|
| Campaign registry | 60 campaigns × 20 attributes | Weekly; mid-cycle additions permitted |
| Tree definitions | 60 trees, 20–400 nodes each, ~9 200 nodes in total | Weekly; effective-dated |
| Suppression registry | ~34 suppressions × 11 attributes | Monthly, plus emergency additions |
| Feature band definitions | ~340 bands across 95 features | Quarterly |
| Channel capacity | 4 channels × 14 attributes | Monthly |
| Contact fatigue rules | ~26 rules | Quarterly |
| Offer tier definitions | 7 tiers × 12 attributes | Quarterly |
| Compliance do-not-target list | 3 100 clients, list of `client_id` | Continuously; treated as current at cycle start |
| Prohibited feature register | ~40 named features and ~90 declared proxies | On Compliance ruling |
| Prior-cycle assignments | ~9.1 M rows | Previous cycle |
| Response and outcome feed | ~14 M rows rolling 12 months | Daily |

### 4.3 Consumed capabilities

| Capability | Used for |
|---|---|
| `core.consent` | Marketing consent and per-channel permission at `decision_date`. A suppression input, not an advisory one. |
| `core.eligibility` | Hard gates: deceased and estate, debt review, administration, sanctioned status, capacity to contract, product availability. |
| `core.scorecard` | Behaviour and propensity scores where a tree references a score rather than a stored feature. |
| `core.appetite` | Maximum amount, maximum term and minimum price per `risk_grade` × `product_code` × segment, bounding what any campaign may advertise. |
| `core.adjustments` | The overlay stack in force at `cycle_date` — cut-off shifts, volume dials, score shifts and amount cap reductions — applied over published trees and pre-assessed amounts without editing either (`§5.3.4`, `§5.5`). |
| Project 03, batch mode | Bulk pre-assessment producing the amount the Bank would actually grant (`§5.5`). |

### 4.4 Locally declared vocabulary

Names this project needs which `credit-core` does not publish. Declared once
here and used unchanged throughout:

| Name | Type | Meaning |
|---|---|---|
| `cycle_id` | int32 | One monthly or daily run. Monotonic, never reused. |
| `cycle_date` | date | The `decision_date` for every assessment in the cycle. |
| `campaign_id` | int16 | One campaign. Stable for life; never reused after retirement. |
| `tree_id` | int16 | The tree belonging to a campaign. |
| `tree_version` | int16 | A published, effective-dated version of a tree. |
| `node_key` | see `§5.4.3` | The identity of a node, stable across tree versions under the conditions in `§5.4.3`. |
| `leaf_outcome_code` | int8 | Target / do not target / target-as-control. |
| `offer_tier_code` | int8 | One of seven tiers. |
| `priority_weight` | float64 | 0..1, the campaign's own statement of how much it wants this client. |
| `reason_label` | int16 | The leaf's business-readable reason, from a registry of ~220. |
| `assignment_id` | int64 | One (client, campaign, cycle) outcome. The join key for the response feed. |
| `suppression_code` | int16 | Which suppression removed a client. |
| `unadjusted_leaf` | int16 | The leaf the client would have reached with the overlay stack disabled. Distinct from `leaf` whenever an overlay bit. |
| `overlay_stack_id` | int32 | The ordered set of adjustments in force for a (campaign, cycle), as resolved at `cycle_date`. |

---

## 5. The flow

### 5.1 Stage 1 — Population assembly

**Preconditions.** The feature mart is closed for the cycle and its freshness
contract is met. A cycle with a mart older than its contract does not start; it
raises.

**Determines.** The population under consideration for the cycle, and the value
of every feature for every client as at `cycle_date`.

**Emits.** A frozen snapshot: 14.2 M × ~600, immutable for the life of the cycle
and retained 25 months.

**Must be recorded.** The mart version and its as-at timestamp; the row count;
the per-feature null rate and its deviation from the trailing three-cycle mean;
the list of features whose type or permitted value set differs from the previous
cycle.

**Requirements.**

1. Every downstream stage in the cycle reads the same frozen snapshot. A feature
   that changes value during the run must not be observable as having done so.
2. The snapshot must be sufficient, on its own and with the tree definitions, to
   re-derive every decision the cycle made, without access to any live system.
3. A daily delta cycle operates on the changed subset but must use the same
   feature definitions as the monthly cycle it sits between. A client processed
   in a delta and again in the next monthly run must not be treated under
   different definitions.

### 5.2 Stage 2 — Global suppressions

**Preconditions.** Stage 1 complete. `core.consent` and `core.eligibility`
resolved at `cycle_date`.

**Determines.** For each client, which campaigns and which channels that client
may not be contacted for, and why.

**Emits.** A suppression record per (client, suppression, scope) that applied.

**Must be recorded.** Every suppression that applied, not merely the first.
Attribution must answer: *which suppression removed which client from which
campaign, on which channel, in which cycle.* A client suppressed by four
different rules must show four.

**The registry.** ~34 suppressions. Representative entries with September cycle
volumes:

| Code | Suppression | Scope | Clients removed |
|---|---|---|---|
| S01 | Deceased or estate in progress | All campaigns | 41 200 |
| S02 | Debt review or debt counselling | All campaigns | 318 400 |
| S03 | Administration order | All campaigns | 12 700 |
| S04 | Insolvency or sequestration | All campaigns | 8 900 |
| S05 | Legal action in progress | All campaigns | 64 300 |
| S06 | Staff, director or related party | All campaigns | 27 600 |
| S07 | Marketing opt-out | All campaigns | 1 042 000 |
| S08 | Per-channel consent absent | Per channel | SMS 384 000 / call 706 000 / email 1 180 000 / in-app 51 000 |
| S09 | Active complaint or dispute | All campaigns | 22 400 |
| S11 | Fraud marker on client or device | All campaigns | 18 300 |
| S14 | Compliance do-not-target list | All campaigns | 3 100 |
| S15 | Invalid or duplicate contact detail | Per channel | 620 000 |
| S17 | Non-resident or address unverified | All campaigns | 96 500 |
| S21 | Recent decline, within cooling-off | Per campaign, by product | 205 000 |
| S22 | In-flight application for the same product | Per campaign | 71 800 |
| S23 | Product already held at or above the offered tier | Per campaign | 2 410 000 |
| S24 | Post-sale quiet period, 30 days from take-up | Per campaign | 188 000 |
| S30 | Contact fatigue, all-channel cap | Per client | 890 000 |
| S31 | Contact fatigue, per-channel sub-cap | Per channel | SMS 1 340 000 / call 410 000 |

**Cooling-off** is a table, not a constant: 8 products × 3 prior outcomes
(declined, referred and lapsed, offer presented and not taken), values from 30 to
180 days. A Home Loan Further Advance decline suppresses for 180 days; an
Everyday Card offer not taken suppresses for 45.

**Contact fatigue caps.** No more than 4 contacts in any rolling 30 days across
all campaigns and all channels, with sub-caps of 2 SMS, 1 outbound call, 4 emails
and 6 in-app slots per 30 days. The cycle cap is 2 contacts (`§5.6`). Fatigue is
evaluated against contacts already made, including those issued by daily deltas
since the last monthly run — which means fatigue state is carried between cycles
and is not derivable from the current cycle alone.

**Requirements.**

1. Suppressions have scope. Some remove a client from everything, some from one
   campaign, some from one channel of one campaign. The evidence must preserve
   the distinction.
2. Suppression must not be a silent filter. The count removed by each suppression,
   per campaign, is a reported figure that campaign owners reconcile monthly, and
   a 20% movement in any suppression's volume raises before dispatch.
3. Some suppressions are **absolute** (S01, S11, S14) and some are
   **measurement-relevant** (S07, S23, S30). For measurement-relevant
   suppressions, the suppressed client must still be evaluated through the tree
   (`§5.3`) and the path recorded, because the analytics team reports the
   population that *would* have been targeted. For absolute suppressions no
   evaluation is required and none should occur. Which class a suppression falls
   into is an attribute of the registry, changeable by Compliance.

### 5.3 Stage 3 — The campaign trees

**Preconditions.** Stage 1 complete; Stage 2 has determined absolute
suppressions; each campaign's tree version in force at `cycle_date` is resolved;
each campaign's candidate population definition is resolved; the overlay stack in
force at `cycle_date` is resolved per `§5.3.4`.

**Determines.** For each (client, campaign) pair in the campaign's candidate
population, the leaf reached and the outcome it carries.

**Emits.** A leaf outcome per evaluation, and the path (`§5.4`).

**Must be recorded.** The tree version used, the variant assigned (`§5.7`), the
`overlay_stack_id` in force, the leaf reached, every value the leaf carries, and
the path.

#### 5.3.1 What a tree is

| Property | Range |
|---|---|
| Depth | 4–12 levels |
| Nodes | 20–400; ~9 200 across all live trees |
| Mean path length | 7.4 nodes |
| Conditions per node | 1–6; mean 2.3. **Not** one feature per level, and **not** one feature per node |
| Features referenced per tree | 9–74 |
| Leaves per tree | 12–190 |

A node tests a condition built from one or more of:

| Kind | Example |
|---|---|
| Numeric comparison | `discretionary_income >= 3500` |
| Band membership | `income_band_code in {5,6,7,8,9}` |
| Set membership over categorical codes | `employment_type_code in {1,2,4}` |
| Boolean flag | `has_active_flex_loan` |
| Comparison between two features | `pre_assessed_amount >= 1.5 * current_flex_balance` |
| Null test | `bureau_as_of_date is not established` |

Conditions within a node combine with `and` and `or`. A node has exactly two
outgoing directions — the condition held or it did not. Nodes are not restricted
to testing the same feature as their siblings, and a tree is not required to be
balanced: campaign 23's shortest path is 2 nodes and its longest is 6.

#### 5.3.2 What a leaf carries

| Value | Meaning |
|---|---|
| `leaf_outcome_code` | Target / do not target |
| `offer_tier_code` | One of seven tiers |
| Amount or limit rule | A fixed amount, a cap, or an expression over the pre-assessed amount |
| Channel preference | Ordered list of permitted channels |
| `priority_weight` | 0..1, the campaign's own bid for the client |
| `reason_label` | Business-readable, from a registry of ~220 |

A "do not target" leaf is a decision, not an absence, and carries a
`reason_label` like any other. "The client fell out of the tree" is not a
permitted outcome; every path terminates in a leaf that says something.

#### 5.3.3 A worked tree — campaign 23, Flex Loan pre-approved top-up

Six levels, ten internal nodes, seven leaves. Shortened for illustration; the
real campaign 23 tree has 74 nodes.

| Node | Level | Condition | If true | If false |
|---|---|---|---|---|
| 1 | 1 | `has_active_flex_loan` **and** `months_on_book_flex >= 6` **and** `flex_term_remaining_months >= 4` | 2 | leaf 901 |
| 2 | 2 | `settlement_ratio <= 0.65` **and** `payments_missed_12m = 0` | 3 | 4 |
| 3 | 3 | `behaviour_score >= 620` **and** `worst_arrears_months_12m = 0` **and** `bureau_enquiries_3m <= 2` **and** `external_unsecured_growth_6m <= 0.20` | 5 | 6 |
| 4 | 3 | `behaviour_score >= 580` **and** `months_since_last_arrears >= 9` **and** `worst_arrears_months_12m <= 1` | 6 | leaf 902 |
| 5 | 4 | `discretionary_income >= 3500` **and** `estimated_instalment_to_income <= 0.28` **and** `income_band_code in {5,6,7,8,9}` | 7 | 8 |
| 6 | 4 | `discretionary_income >= 2200` **and** `estimated_instalment_to_income <= 0.33` **and** `employment_type_code in {1,2,4}` | 8 | leaf 903 |
| 7 | 5 | `pre_assessed_amount >= 40000` **and** `pre_assessed_amount >= 1.5 * current_flex_balance` | 9 | 10 |
| 8 | 5 | `pre_assessed_amount >= 15000` | 10 | leaf 904 |
| 9 | 6 | `app_logins_90d >= 3` **or** `channel_preference_code = 2` | leaf 910 | leaf 911 |
| 10 | 6 | `sms_response_rate_12m >= 0.04` **and** `prior_offer_declines_6m <= 1` | leaf 912 | leaf 913 |

| Leaf | Outcome | Tier | Amount | Channel | Priority | Reason label |
|---|---|---|---|---|---|---|
| 901 | Do not target | — | — | — | — | No qualifying facility |
| 902 | Do not target | — | — | — | — | Risk below campaign floor |
| 903 | Do not target | — | — | — | — | Insufficient instalment headroom |
| 904 | Do not target | — | — | — | — | Pre-assessed amount below campaign minimum |
| 910 | Target | A | min(pre-assessed, R250 000) | In-app, then email | 0.86 | Prime top-up, digitally engaged |
| 911 | Target | A | min(pre-assessed, R250 000) | Outbound call, then SMS | 0.81 | Prime top-up, assisted channel |
| 912 | Target | B | min(pre-assessed, R120 000) | SMS, then in-app | 0.58 | Standard top-up, SMS responsive |
| 913 | Target | C | min(pre-assessed, R60 000) | In-app | 0.34 | Standard top-up, low channel response |

Note what the example demonstrates and what an implementation must therefore
support: nodes 3 and 4 sit at the same level and test overlapping but different
feature sets; node 6 is reachable from both node 3 and node 4, so the tree is not
strictly a tree in the graph sense unless node 6's two arrivals are distinct
nodes — and whether they are is a decision with direct consequences for `§5.4`;
node 7 compares two features; node 9 uses `or`; node 10 chooses a channel rather
than a credit outcome, which is the common pattern in the last one or two levels
of most campaign trees.

**Requirements.**

1. Evaluation must be exhaustive over the candidate population and total over the
   tree: every client reaches exactly one leaf.
2. A client's evaluation in one campaign must not be affected by the outcome of
   any other campaign at this stage. Interaction between campaigns happens in
   `§5.6` and nowhere earlier.
3. Between 30 and 60 campaigns are live. Candidate populations average 6.7 M
   clients and range from 240 000 to 13.1 M, giving approximately **400 million
   evaluations per monthly cycle**.

#### 5.3.4 Overlays over a published tree

A tree is fitted against 24 months of history, validated, signed off and then
expected to hold for two or three quarters. It does not. Early-arrears
performance deteriorates on one channel; a funding constraint arrives in
October; the campaign forum commits to a contact volume the tree as published
will miss by 90 000. The business's answer to all three is the same, and it is
not to re-author the tree: policy layers an **adjustment** over it, per
`core.adjustments` — a named, approved, effective-dated overlay that changes the
answer while leaving the artefact exactly as validated.

Roughly 40 overlays are live across the programme at any time. The kinds used
here:

| Kind | Applies to | Worked example |
|---|---|---|
| Cut-off shift | A campaign's minimum acceptable risk grade | Tighten campaign 23 by one notch — minimum grade 6 becomes minimum grade 5 — for the October and November cycles, after early-arrears deterioration on the August cohort |
| Threshold shift (**volume dial**) | One named numeric threshold inside a published tree | Move node 6's `discretionary_income` test from 2 200 to 2 600 to bring campaign 23's targeted volume from 412 000 to the 350 000 the contact centre has committed to staff |
| Score shift | A score consumed by node conditions | −15 points on the new-to-bank segment across all credit campaigns |
| Odds multiplier | The propensity or PD used in expected value at arbitration | PD × 1.20 on partner-channel-originated clients |
| Cap reduction | The amount a tier may advertise (`§5.5`) | Tier A maximum reduced from R250 000 to R200 000 for one cycle |

**Requirements.**

1. **An overlay is not an edit.** The published tree is untouched. In particular,
   applying, changing or removing an overlay must **not** change any `node_key`.
   An overlay that renumbers the tree destroys the node-level history the whole
   of `§5.4` exists to produce. The overlay is recorded beside the path, never
   folded into it.
2. **The unadjusted answer survives.** For every client whose outcome differs
   under the overlay stack, both the leaf reached and `unadjusted_leaf` must be
   recorded, together with both paths where they diverge. Separately, and per
   `00 §7.6`, the cycle must be runnable with the overlay stack disabled: a
   5% shadow evaluation with overlays off runs every monthly cycle, which is how
   the tree's own performance — as opposed to the tree-plus-policy composite — is
   monitored.
3. **Overlays stack, and the order is declared.** A segment score shift and a
   campaign cut-off shift may both apply to one evaluation, and the composition
   order changes the answer. The order is part of the overlay definitions, not an
   emergent property of how the cycle happened to run, and it is recorded in
   `overlay_stack_id`.
4. **Identity and justification.** Each overlay carries an id, a description, an
   owner, a rationale, an approval reference, an effective-from date and an
   effective-to date. A risk overlay — cut-off shift, score shift, odds
   multiplier — requires Credit Risk approval. A volume dial requires the
   campaign owner and the campaign forum. Neither may approve the other's kind.
5. **Expiry is mandatory.** Every overlay declares a review date — default
   maximum 3 cycles for a volume dial, 6 months for a risk overlay. An overlay
   reaching it without renewal must surface before the cycle that would run under
   it, and the cycle must refuse to apply it. The failure mode is a tightening
   approved for one bad quarter, still suppressing 200 000 clients a month three
   years later, which nobody can now explain.
6. **Scope is declared and enforced.** An overlay names the campaigns, segments,
   channels, products, grade ranges or tiers it applies to. Applying an overlay
   outside its declared scope is an **error that fails the cycle**, not a silent
   no-op — and so is an overlay whose scope matches nothing, which is how a
   tightening scoped to a campaign retired last month is caught rather than
   assumed to be working.
7. **The stack is resolved by `cycle_date`, never by today.** A re-run of the
   March cycle in 2028 resolves March's overlays, in March's order, including the
   ones that have since expired.

**Two interactions are requirements in their own right, not consequences.**

**(a) Reporting must separate population movement from policy.** An overlay
changes targeted volume by design, which means month-on-month node-level volume
reporting (`§5.4.2`) must be able to say whether a node's volume moved because
the client population moved, because the tree version changed, or because an
overlay was applied, changed or expired. A cycle-on-cycle comparison in which
`overlay_stack_id` differs must be flagged as such and decomposed into those
three causes before it is published to the campaign forum. Node volumes reported
without the stack they were produced under are not comparable figures and must
not be presentable as though they were.

**(b) Measurement windows constrain when an overlay may start.** Control and
holdout groups (`§5.7`) are only comparable if treated and control populations
experienced the same policy. Therefore: control clients are evaluated under the
same overlay stack as treated clients — a control evaluated with overlays off is
measuring the overlay, not the campaign — and an overlay affecting a campaign
with an in-flight measurement window may only take effect at a measurement
boundary, unless the campaign owner and Campaign Analytics jointly accept the
break and the acceptance is recorded against the measurement. The system must be
able to list, for any measurement window, every overlay that started, changed or
expired inside it.

### 5.4 Stage 4 — Path capture

This stage records no new decision. It exists because the record of how the
decision was reached is a deliverable of equal standing to the decision itself,
and stating it as a property of Stage 3 has, historically, resulted in its being
dropped when the run got tight.

**Determines.** Nothing. **Emits.** For every evaluation performed in Stage 3 —
including suppressed-but-measurement-relevant clients and including control
clients — the ordered sequence of nodes visited, terminating in the leaf.

#### 5.4.1 The four properties

**(a) Compact.** Approximately 400 M evaluations per monthly cycle at a mean path
length of 7.4 nodes. Held at one row per evaluation that is 400 M rows; held
exploded at one row per node visited it is roughly 3.0 billion. The spec does not
say which, and does not say what the path looks like on disk — but it does
constrain the result: the complete path artefact for a monthly cycle must occupy
no more than **120 GB** in the warehouse, and must be produced within the six-hour
window alongside everything else. Retention is 25 months online and 7 years
archived, so the cycle figure multiplies by 25.

**(b) Joinable.** Paths are analysed in the warehouse by people who will never
run the engine. The stored path must join to node metadata using ordinary
database operations, with no decoding logic that exists only inside the decision
system. Node metadata must be published per (campaign, tree version) with at
least: node key, level, parent lineage, the condition in readable form, the
features it references, the direction taken for each outcome, and — for leaves —
every value the leaf carries.

**(c) Stable across versions.** The analytics team compares node volumes month on
month. "Node 17 in March" and "node 17 in April" must be comparable **exactly
when they are genuinely the same test**, and must be visibly not comparable
otherwise. Concretely:

- If a node's condition is unchanged and its position in the tree's logic is
  unchanged, its identity is unchanged, and March and April volumes may be
  compared directly.
- If a node's condition changes in any way — a threshold moved from 2 200 to
  2 600, a code added to a set, a fourth condition added — its identity must
  change, and any comparison of the old and new must be flagged rather than
  silently produced.
- If a node elsewhere in the tree is added, removed or changed, an unaffected
  node's identity must **not** change. Re-authoring a tree in the modelling tool
  must not renumber the world.
- Publication of a new tree version must produce a **node identity map** against
  the previous version, classifying every node as carried forward, changed, added
  or removed, with the changed ones showing what changed. This map is a review
  artefact seen by the campaign owner and Credit Risk before sign-off, and it is
  retained.

This is the single hardest requirement in the project, because the modelling tool
that exports the trees has its own notion of node ordering and does not guarantee
it across exports of a re-fitted tree.

**(d) Human-readable on demand.** Given a stored path, the node metadata for the
tree version in force, and the stage 1 feature snapshot, the system must render
the path as an ordered list of conditions, each showing the client's actual
values, the condition as written, whether it held, and where it led. This must be
producible without re-running the tree and without the original run environment.

Worked example. Client 8 412 907, cycle 2026-09, campaign 23, tree version 11:

```
path: 1 → 4 → 6 → 8 → 10 → leaf 912
```

rendered:

```
node 1  has_active_flex_loan = true
        months_on_book_flex = 31 (>= 6)
        flex_term_remaining_months = 22 (>= 4)                       HELD    → 2
node 2  settlement_ratio = 0.71 (<= 0.65 FAILS)
        payments_missed_12m = 0                                      NOT HELD → 4
node 4  behaviour_score = 604 (>= 580)
        months_since_last_arrears = 14 (>= 9)
        worst_arrears_months_12m = 1 (<= 1)                          HELD    → 6
node 6  discretionary_income = R3 118 (>= 2200)
        estimated_instalment_to_income = 0.26 (<= 0.33)
        employment_type_code = 1 (in {1,2,4})                        HELD    → 8
node 8  pre_assessed_amount = R84 000 (>= 15000)                     HELD    → 10
node 10 sms_response_rate_12m = 0.061 (>= 0.04)
        prior_offer_declines_6m = 0 (<= 1)                           HELD    → leaf 912
leaf 912  TARGET, tier B, R84 000, SMS then in-app, priority 0.58,
          "Standard top-up, SMS responsive"
```

Note that node 2 appears in the rendering although it is not in the stored path
as a *visited-and-passed* node — it was visited and its condition did not hold.
Whether the stored path contains node 2, or whether node 4's presence implies it,
is exactly the kind of decision this requirement is designed to force into the
open. The rendering must be correct either way.

#### 5.4.2 What the paths are for

These are the actual analytics use cases, and they are the justification for
every constraint above.

1. **Node-level volume and response reporting.** For each campaign, each tree
   version and each cycle: how many clients entered each node, how many left by
   each direction, how many reached each leaf, and — joined to the response feed
   — the contact, response and take-up rate of every leaf and every node. This is
   produced for all 60 campaigns every cycle and reviewed in a monthly forum.
2. **Dead branch identification.** Nodes that received zero traffic for three
   consecutive cycles, and leaves that were never reached. In the September
   review, 1 340 of 9 200 live nodes had received no traffic in 90 days. Some are
   genuinely unreachable and should have been caught at publication (`§5.9`);
   most reflect population drift and are a signal to the campaign owner.
3. **Diagnosing a volume movement to a node.** Campaign 23's targeted volume fell
   from 412 000 to 198 000 between August and September. The required answer is
   not "the population changed" but "node 6's true rate fell from 0.71 to 0.38
   because `discretionary_income` fell below 2 200 for 214 000 clients following
   the change to the expense norm table on 1 September". Reaching that answer
   requires per-node volumes on both cycles, node identity stable enough to
   compare them, and the feature values at the node. Population movement, a new tree
   version and an overlay (`§5.3.4`) produce the identical volume figure, and the
   answer must say which of the three it was.
4. **Re-deriving a client's leaf without re-running.** Given the stored path and
   the stored features, an analyst must be able to reproduce the leaf and every
   condition evaluation on a laptop. This is the mechanism behind the dispute
   answer in `§9.1` and it must not require the decision system to be available.
5. **Uplift by node and by leaf.** Control clients (`§5.7`) are evaluated fully,
   so response rates can be compared treated-versus-control per leaf, and the
   incremental effect of a campaign can be attributed to the parts of the tree
   that produced it. A leaf with a positive raw take-up rate and zero uplift is
   the finding this measurement exists to surface, and it cannot be found without
   control paths.
6. **Feature drift at the node.** The distribution of each tested feature, at the
   node where it is tested, per cycle — so that a shifting feature is detected
   where it matters rather than in aggregate.

#### 5.4.3 Node identity, stated as a requirement

`node_key` identifies a node. The requirement is behavioural, not structural:

1. Two nodes in two tree versions share a `node_key` **only if** their conditions
   are identical in every respect — same features, same operators, same
   thresholds, same sets, same combination — and their position in the tree's
   decision logic is the same.
2. A `node_key` must not depend on the export ordering of the modelling tool, on
   the number of nodes in the tree, or on anything about a different part of the
   tree.
3. Given two tree versions, the system must be able to state, for every node,
   whether it is the same node.
4. A `node_key` must be usable as a database key: fixed width, no whitespace, no
   semantics that require the decision system to interpret.
5. Node keys must survive a campaign being paused for four months and resumed.
6. Node keys are unaffected by overlays (`§5.3.4`). An overlay that moves a
   threshold changes what the node decides; it does not create a different node,
   and the comparison it breaks is handled by reporting the stack, not by
   re-identifying the tree.

### 5.5 Stage 5 — Eligibility and amount determination

**Preconditions.** Stage 3's candidate populations known. Campaigns flagged as
requiring a pre-assessment identified — currently 38 of 60. Amount overlays in
force resolved per `§5.3.4`.

**Determines.** For each client in a credit campaign's candidate population, the
amount or limit the Bank would actually grant today, and whether the client is
eligible at all.

**Emits.** `pre_assessed_amount`, `pre_assessed_term`, the binding constraint
(appetite, affordability, exposure, product maximum), the `risk_grade` used, and
a validity window.

**Requirements.**

1. **Nothing is advertised that the Bank would not grant.** Every amount a tree
   puts in front of a client must be backed by a pre-assessment produced by
   project 03 running in batch over the same feature snapshot, using the same
   appetite grid, affordability rules and rate cards the branch would use on
   application. A campaign may advertise less than the pre-assessed amount; it may
   never advertise more.
2. **The tolerance is measured.** Where a client accepts and the granted amount at
   application is less than the amount advertised, that is a **breach**. Required:
   breach rate ≤ **3%** of take-ups per campaign per cycle; where breached, the
   shortfall ≤ **10%** of the advertised amount in at least 90% of breaches. A
   campaign exceeding 5% in a cycle is suspended pending review. The breach rate
   is reported per campaign, per cycle, and per leaf — because a breach
   concentrated in one leaf is a tree defect, not a tolerance problem.
3. **Pre-assessments expire.** Valid 35 days from `cycle_date` for monthly
   campaigns, 10 days for daily delta campaigns. An offer whose pre-assessment has
   expired must not be presentable, and the app must be able to tell.
4. **Bulk volume is a real constraint.** Approximately 7.9 M clients require a
   pre-assessment per monthly cycle, inside the same six-hour window.
5. Amounts are rounded per `core.rounding` — advances to the nearest R100 —
   before they reach a tree, so that the tree's comparisons and the client's
   communication see the same number.
6. **Amount overlays may only reduce.** Policy layers cap reductions over
   pre-assessed amounts — 20% off a sector while its performance is reviewed,
   tier A's ceiling cut for a cycle while funding is constrained — as overlays,
   never by editing project 03's output. The pre-assessed amount and the
   advertised amount are both retained on the assignment, and the difference is
   attributable to the named overlay. An overlay that would raise an advertised
   amount above the pre-assessed amount is prohibited by construction: the whole
   point of requirement 1 is that nothing is advertised the Bank would not grant,
   and an uplift dial is a direct attack on it.
7. The tolerance in requirement 2 is measured against the **advertised**
   post-overlay amount, because that is what the client saw, and reported against
   the pre-assessed amount as well — a breach rate that is only acceptable
   because an overlay is suppressing amounts returns the day it expires.

### 5.6 Stage 6 — Arbitration

**Preconditions.** All campaign trees evaluated; suppressions applied; amounts
determined; channel capacity and fatigue state known.

**Determines.** Which of a client's qualifying campaigns actually result in a
contact this cycle, through which channel, in what order.

**Emits.** Per (client, campaign): contacted or not, and if not, why not.

**The problem.** A contactable client qualifies for a mean of **5.3** campaigns
and, at the 99th percentile, **9**. The cycle cap is **2 contacts per client**.
Roughly 11.6 M clients qualify for at least one campaign, generating unconstrained
demand of ~61 M qualifications, from which at most ~19 M contacts may be issued
and, after channel limits, substantially fewer.

**The inputs to the decision.**

| Input | Detail |
|---|---|
| `priority_weight` | From the leaf; the campaign's own bid |
| Expected value | Response propensity × expected margin per take-up, per campaign per tier |
| Channel capacity | Population-level; see below |
| Recency | Days since the client's last contact, overall and by campaign family |
| Fairness | A rule preventing the same campaign always winning |

**Channel capacity is a population-level constraint.**

| Channel | Monthly capacity | Daily cap | Per-client cap / 30 days | Cost per contact |
|---|---|---|---|---|
| SMS | 4 000 000 | 260 000 | 2 | R0.19 |
| Outbound call | 260 000 | 13 000 | 1 | R14.80 |
| Email | 6 500 000 | 400 000 | 4 | R0.04 |
| In-app | Unlimited | — | 6 live slots | R0.00 |

Unconstrained arbitration produces SMS demand of ~7.3 M against 4 M capacity and
outbound call demand of ~640 000 against 260 000. **This cannot be decided one
client at a time.** Whether client 8 412 907 gets an SMS depends on how many other
clients want one, which is not knowable while evaluating that client. The spec
states the requirement and deliberately does not prescribe the mechanism:
thresholding, ranking-and-cutting, repeated passes and optimisation are all
permitted. What is required of whatever is used:

1. It is deterministic. The same inputs produce the same assignment, exactly.
2. It is re-runnable. A cycle re-run produces the identical contact list.
3. It records, for every qualifying (client, campaign) pair that did **not**
   result in a contact, which of the following applied: lost on rank to a named
   competing campaign; channel capacity exhausted; fatigue cap reached; control
   group; campaign suspended. "Not selected" without a reason is not acceptable
   output, because the campaign owner's first question every month is "how much
   of my population did I lose and to whom".
4. Capacity utilisation is reported per channel: allocated, used, unused, and the
   value of the demand that was refused.

**The fairness rule.** No single campaign may account for more than **35%** of a
given client's contacts over a rolling 6 cycles. Additionally, every live campaign
must receive at least **60%** of its rank-one demand; a campaign falling below
that raises to the campaign forum rather than quietly starving. Without this,
one high-margin campaign wins every arbitration for the same 1.2 M clients every
month and nine other campaigns never learn anything about them.

### 5.7 Stage 7 — Holdout, control and challenger

**Preconditions.** Campaign registry resolved. Control and variant assignment
depend on no output of any other stage, which is deliberate — assignment must be
computable from identifiers alone. Variant assignment must be established before
Stage 3, because it selects which tree version is evaluated.

**Determines.** For each (client, campaign): control or treated, and which tree
variant applies.

**Requirements.**

1. **Deterministic assignment.** Control membership is a function of a stable hash
   of (`client_id`, `campaign_id`, `holdout_design_version`) alone. No randomness
   at run time, no stored assignment table that can drift, no dependence on the
   order clients were processed.
2. **Stable across cycles.** The same client lands in the same group in March,
   April and May, and only a change to `holdout_design_version` moves anyone. When
   it does change, the set of clients who moved must be enumerable.
3. **5% control per campaign** by default; 12 campaigns run 10% while their trees
   are new. Additionally, a **universal holdout of 1%** of the whole client base
   is excluded from all campaigns, to measure the total effect of the programme
   rather than of any one campaign.
4. **Control clients are evaluated in full.** They pass through the tree, reach a
   leaf, and their path is recorded, exactly as a treated client's is. The only
   difference is that no contact is issued. A control group whose members were
   never evaluated tells you nothing, because you cannot compare like with like —
   and this is the requirement most often lost when a run is optimised for the
   contacts it actually produces.
5. **Champion / challenger.** A campaign may run 2 or 3 tree variants on a
   percentage split — 90/10 by default, and 70/20/10 where three are live.
   Assignment is a stable hash of (`client_id`, `campaign_id`,
   `variant_design_version`). The variant is recorded on the assignment and on the
   path, because a node volume compared across variants without knowing the split
   is a misleading number.
6. Control and variant membership must be re-derivable from stored identifiers
   alone, with no access to the cycle that assigned them.
7. **Overlays and measurement.** Control clients are evaluated under exactly the
   overlay stack the treated population experienced (`§5.3.4(b)`). Where an
   overlay starts, changes or expires inside a measurement window, the window is
   marked as broken by that overlay, and any uplift reported across the break
   states it. A measurement whose treated population was tightened halfway
   through and whose control was not is not a measurement.

### 5.8 Stage 8 — Output and dispatch

**Preconditions.** Arbitration complete. Dispatch cut-off 06:00 on the first
business day of the month.

**Emits.**

| Output | Grain | Rows per monthly cycle |
|---|---|---|
| Assignments | client × campaign × cycle | ~61 M qualifications, of which ~9.1 M contacted |
| Paths | see `§5.4.1` | ~400 M evaluations |
| Suppression attributions | client × suppression × scope × cycle | ~11 M |
| Arbitration ledger | client × campaign × cycle, for non-contacts | ~52 M |
| Control and variant register | client × campaign × design version | ~61 M |
| Channel dispatch files | one per channel per send window | 4 channels, 22 windows |

**An assignment record carries**: `assignment_id`, `client_id`, `campaign_id`,
`cycle_id`, `tree_id`, `tree_version`, variant, control flag, leaf, `node_key`
path reference, `offer_tier_code`, offer amount or limit, term where applicable,
indicative instalment and rate where the campaign advertises them, channel,
contact sequence within the cycle, offer valid-from and valid-to dates,
`reason_label`, and the pre-assessment reference.

**The contact centre and the app need more than the answer.** An agent taking an
inbound call from a client who received an outbound offer must be able to see, in
their own system: what was offered, on what terms, when it expires, what the
client's indicative instalment would be, and a plain-language statement of why
this client was selected — derived from the `reason_label` and the path, not
written by hand per campaign.

**The reverse feed.** Responses, take-ups, declines, opt-outs and complaints are
joined back to `assignment_id`. Required within 48 hours of the event.
Approximately 14 M response events per rolling 12 months. An event that cannot be
joined to an assignment is an exception and is counted; an unjoinable rate above
2% raises, because unjoinable responses silently deflate every measured response
rate in the programme.

### 5.9 Stage 9 — Authoring, validation and publication

This stage does not run in the cycle. It is how a tree gets to production.

**The path a tree takes.** An analyst fits or edits a tree in the modelling tool
against a 24-month sample, exports it, and submits it. A tree must reach
production **without an engineer**. Typical volume: 15–40 tree changes a month,
of which roughly 30 are threshold moves and 5 are structural.

**Validation, before publication.** A submitted tree is rejected unless:

1. Every path terminates in a leaf. No path falls off the end.
2. There are no unreachable nodes — no node whose conditions cannot be satisfied
   given the conditions on every path leading to it.
3. There are no contradictory conditions on any path (`age >= 65` under
   `age < 60`).
4. Every referenced feature exists in the mart, with a compatible type and, where
   banded, a band definition that covers the values tested.
5. Every leaf outcome is drawn from the permitted set: a known
   `offer_tier_code`, a channel the campaign is permitted to use, a
   `reason_label` in the registry, a `priority_weight` in 0..1, and an amount rule
   that cannot exceed the `core.appetite` maximum for the client's grade.
6. No feature on the prohibited register appears, directly or as a declared
   proxy.
7. The estimated population impact, computed against the most recent snapshot, is
   within the bound the campaign declared — default ±25% of the current targeted
   volume. A tree that would triple a campaign's volume does not publish silently.
8. Every overlay currently in force over the campaign (`§5.3.4`) is still
   applicable to the submitted tree. An overlay naming a threshold on a node the
   new version deletes, or scoped to a tier the new version no longer produces,
   must be re-approved or withdrawn as part of the same change — never left
   pointing at nothing, and never silently reinterpreted against a different node.
   The estimated population impact in item 7 is reported both with and without
   the overlay stack, because publishing a tree that is only within bound because
   a volume dial is holding it there is the way a campaign doubles overnight
   three weeks later when the dial expires.

**Sign-off.** A tree change requires the campaign owner and, for any campaign
offering credit, Credit Risk. Both sign the same artefact: the tree, the node
identity map against the prior version, the estimated population impact with and
without overlays, the disposition of every overlay in force over the campaign,
and the validation report. An overlay is approved separately and on its own
schedule, which is the point of it — but a tree change and an overlay change
landing in the same cycle must be reviewable as one combined effect, because that
is what the client experiences. Trees are versioned and effective-dated; a cycle resolves the
version in force at its `cycle_date` and never "the latest".

---

## 6. Parameters and tables

### 6.1 Tables

| Table | Dimensions | Cells | Owner | Cadence | Source |
|---|---|---|---|---|---|
| Campaign registry | 60 campaigns × 20 attributes | 1 200 | Campaign owners | Weekly | Internal |
| Tree definitions | ~9 200 nodes × ~9 attributes, plus ~4 000 leaves × 7 | ~110 000 | Campaign Analytics | Weekly | Modelling tool export |
| Feature band definitions | ~340 bands across 95 features | ~1 000 | Campaign Analytics | Quarterly | Internal |
| Suppression registry | 34 suppressions × 11 attributes | 374 | Compliance / Campaign Analytics | Monthly | Internal |
| Cooling-off periods | 8 products × 3 prior outcomes | 24 | Credit Risk Policy | Quarterly | Internal |
| Contact fatigue rules | 26 rules × 8 attributes | 208 | Channel Operations | Quarterly | Internal |
| Channel capacity | 4 channels × 14 attributes | 56 | Channel Operations | Monthly | Internal |
| Offer tier definitions | 7 tiers × 12 attributes | 84 | Campaign owners | Quarterly | Internal |
| Reason label registry | ~220 labels × 6 attributes | 1 320 | Campaign owners / Compliance | Monthly | Internal |
| Prohibited feature register | ~40 features, ~90 declared proxies × 5 attributes | 650 | Compliance | On ruling | Compliance |
| Holdout design register | 60 campaigns × 6 attributes | 360 | Campaign Analytics | On design change | Internal |
| Variant design register | ~18 campaigns × 5 attributes | 90 | Campaign Analytics | Weekly | Internal |
| Arbitration weights | 60 campaigns × 7 attributes | 420 | Campaign forum | Monthly | Internal |
| Campaign adjustment register | ~40 live overlays × 12 attributes, plus expired history | ~480 live | Credit Risk Policy (risk kinds) / campaign owners (volume dials) | Weekly | Internal |
| Overlay stacking order | ~6 overlay kinds × precedence, plus per-campaign exceptions | ~90 | Credit Risk Policy | On change | Internal |
| Node identity map | Per published tree version pair | ~9 200 per publication | Generated | On publication | Generated |

**Feature band definitions are shared across trees and that is the point.**
`income_band_code` means the same thing in all 38 trees that reference it. A band
definition change is therefore a change to every tree that uses it, and
publication must say which trees are affected and what the estimated impact is on
each — a band edit that quietly moves 400 000 clients across node 5 of campaign 23
is the same event as a threshold edit and must be reviewed as one.

### 6.2 Parameter ownership

Following `00 §7.2`:

| Class | Example | Owner | Cadence | Approval |
|---|---|---|---|---|
| Library-global | Appetite grid, expense norms, rate cards | Credit Systems | Per library release | Compliance |
| Programme-wide | Fatigue caps, channel capacity, universal holdout, prohibited features | Channel Operations / Compliance | Monthly to quarterly | Compliance |
| Campaign-local | Thresholds within a tree, priority weight, offer tiers, control percentage, expiry days | Campaign owner | Weekly | Campaign owner + Credit Risk for credit campaigns |
| Overlay | Cut-off shift, volume dial, score shift, amount cap reduction | Credit Risk Policy for risk kinds; campaign owner for volume dials | Weekly, always with an expiry | Credit Risk for risk kinds; campaign forum for volume dials |

A campaign owner must be able to change their own campaign's thresholds without
being able to change the fatigue caps, the prohibited feature register, or another
campaign's anything.

**An overlay is not a campaign-local parameter**, even when it is scoped to one
campaign and even when it moves the same number a campaign owner could have moved
by re-authoring. It changes the answer of an artefact that has already been
validated and signed off, under a separate approval and with a mandatory expiry,
and the record must keep it visibly separate from the artefact for exactly as long
as the artefact is kept. Collapsing the two — writing the dial into the tree "to
keep things simple" — is prohibited, because it destroys the distinction between
what the tree decided and what policy decided, which is the distinction the
mechanism exists to preserve.

---

## 7. Outputs

**To the channels.** Dispatch files per channel per send window, containing only
what the channel needs: contact details, campaign, offer values, expiry, and the
message parameters. No risk scores, no path, no feature values.

**To the app.** The client's live offers with tier, amount, term, indicative
instalment, expiry and the reason label, refreshed daily from the delta cycle.

**To the contact centre.** Call lists with the offer, the terms, the expiry, the
evidence needed for the conversation, and the plain-language reason.

**Persisted.** Everything in `§5.8`, plus the stage 1 feature snapshot, the tree
versions in force, the node metadata per version, the node identity maps, the
overlay stack in force with each overlay's id, scope, order, approval and expiry,
the `unadjusted_leaf` and overlays-disabled shadow results, the suppression
registry as at `cycle_date`, and the validation reports of every tree used. Retention: 25 months online, 7 years archived. The persisted set must be
sufficient to reconstruct any cycle's decisions with nothing else.

---

## 8. Non-functional requirements

| Requirement | Value |
|---|---|
| Monthly cycle wall-clock | 14.2 M clients × up to 60 trees, ~400 M evaluations, complete within **6 hours**, starting after the mart closes at 22:00 and delivering dispatch files by 06:00 |
| Sustained evaluation rate | ~18 500 evaluations/second, ~137 000 node condition tests/second |
| Daily delta | 150 000 – 400 000 changed clients, all live campaigns, within **30 minutes**, five times a week |
| Bulk pre-assessment | 7.9 M clients through project 03 in batch, inside the same 6-hour window |
| Path artefact | ≤ 120 GB per monthly cycle; node-level aggregation for one campaign-cycle readable in ≤ 90 seconds |
| Determinism | Identical snapshot, identical tree versions, identical overlay stack, identical registry state, identical `cycle_date` ⇒ identical output, row for row, including arbitration |
| Overlays-disabled shadow | A 5% overlays-off evaluation of every campaign inside the same 6-hour window, plus full dual evaluation for every client whose leaf the overlay stack changed |
| Re-runnability | A single campaign may be re-run without re-running the other 59, producing exactly what the full run produced for it — except where arbitration is involved, in which case the dependency must be explicit and stated, not discovered |
| Preparation cost | Preparing 60 trees for a run must cost under 4 minutes in total and must not scale with population size |
| Partial failure | A cycle that fails at 70% must resume without re-evaluating the completed 70% and without producing different answers for it |
| Freshness of app offers | Never more than 24 hours stale |
| Dispatch integrity | A campaign suspended after file generation but before send must be removable from the files without regenerating the cycle |

---

## 9. Audit, evidence and explainability

### 9.1 A client asks why

*"Why was I offered R84 000 on a top-up?"* or, more often, *"Why did my
neighbour get an offer and I did not?"* Answerable within **5 business days**,
for any cycle in the last 25 months, from stored evidence only. The answer
comprises: the campaigns the client was in the candidate population for; the
suppressions that applied and which removed them from what; for each campaign
evaluated, the path rendered as in `§5.4.1(d)`; the leaf and its reason label;
whether the client was in a control group; and, where the client qualified but
was not contacted, the arbitration reason. The staff member producing this answer
is not an engineer and does not run the decision system.

### 9.2 A regulator asks whether targeting discriminates

The question arrives as: *does the Bank's marketing of credit differentiate on a
prohibited basis?* The response, required within **10 business days**, must
contain:

1. The complete feature list of every live tree, per campaign, per version in
   force during the period under review — currently 60 trees referencing 287
   distinct features.
2. Evidence that no feature on the prohibited register appears in any tree,
   directly.
3. Evidence that no feature acts as a **proxy** for a prohibited basis. This is
   the hard half. It requires, per feature per tree, a documented association
   measure against each prohibited attribute over the current population, a
   declared threshold, and a Compliance attestation with a date. A feature
   crossing the threshold is either removed or carries a documented business
   justification and a monitoring commitment.
4. Outcome distributions by leaf against the prohibited attributes, to show that
   the tree's *effects* are examined and not only its inputs.
5. The sign-off record for every tree version in the period.
6. Every overlay in force during the period, with its scope, rationale and
   approval. An overlay is a targeting rule like any other — a cut-off shift
   scoped to a segment can differentiate exactly as effectively as a node can —
   and a pack that discloses the trees but not the overlays over them is
   incomplete.

The prohibited feature register is maintained by Compliance and grows: a
postcode-derived affluence index was ruled a proxy in March and had to be removed
from 11 trees, which is `§11` scenario 5.

### 9.3 The analytics team asks why a volume moved

*"Campaign 23 targeted 412 000 clients in August and 198 000 in September. Why?"*

Required: an answer located to a node, within the working day, without a re-run.
The mechanism is `§5.4.2(3)`: node volumes for both cycles, compared on stable
identity, showing that node 6's true rate fell from 0.71 to 0.38 while every other
node was stable, and that the feature driving node 6 shifted. An answer that
stops at "fewer clients qualified" is not an answer.

### 9.4 Internal audit asks about exclusion

*"List every client excluded from campaign 17 in the March cycle, with the
reason."* Answerable from the suppression attributions and arbitration ledger,
18 months after the fact, without the cycle being re-run.

### 9.5 Which answer was the tree's, and which was policy's

Asked at every Credit Committee and by every model validator: *what would we have
done without the overlays?* Answerable per cycle, per campaign and in aggregate,
from the `unadjusted_leaf` and the overlays-disabled shadow evaluation: the
targeted volume with and without the stack, the clients whose outcome each
overlay changed, the amounts each cap reduction suppressed, and the response and
take-up of the two populations where both exist. Also answerable in the other
direction: for a given date, which overlays were in force, over what scope, in
what order, approved by whom, expiring when — and which were within 30 days of an
expiry that nobody had renewed.

The campaign forum sees, every cycle: overlays in force, started, expired, and
approaching review without a renewal decision, with the volume attributable to
each. The expensive failure here is not a wrong overlay; it is a forgotten one.

### 9.6 Evidence retention

| Artefact | Online | Archive |
|---|---|---|
| Assignments | 25 months | 7 years |
| Paths | 25 months | 7 years |
| Feature snapshots | 25 months | 7 years |
| Tree versions, node metadata, identity maps, validation reports, sign-offs | Indefinite | Indefinite |
| Adjustment register, including expired overlays, scopes, orders and approvals | Indefinite | Indefinite |
| Suppression attributions | 25 months | 7 years |

---

## 10. Acceptance criteria

1. A campaign analyst publishes a tree change — threshold or structure — and it is
   live in the next cycle without an engineer touching anything, having passed
   every validation in `§5.9` and both sign-offs.
2. A submitted tree with an unreachable node, a contradictory path, a missing
   feature, an out-of-set leaf outcome or a prohibited feature is rejected with a
   message naming the node and the problem, before publication.
3. A full monthly cycle completes in under 6 hours and a delta in under 30
   minutes, at stated volumes.
4. Re-running a completed cycle from its persisted snapshot reproduces every
   assignment, every path and every arbitration outcome exactly.
5. For any (client, campaign, cycle) in the last 25 months, the path renders as an
   ordered list of conditions with values, produced from stored data, with the
   decision system unavailable.
6. Node-level volume and response reporting is produced for all live campaigns
   every cycle, and node volumes are comparable month on month wherever, and only
   wherever, nodes are genuinely the same.
7. Publishing a tree version produces a node identity map classifying every node
   as carried forward, changed, added or removed.
8. Control assignment is reproducible from `client_id`, `campaign_id` and
   `holdout_design_version` alone, and control clients have complete paths.
9. Every suppressed client can be shown the suppressions that applied, per
   campaign, per channel.
10. Every qualifying-but-not-contacted pair carries an arbitration reason.
11. Channel capacity is never exceeded in any cycle, and utilisation and refused
    demand are reported per channel.
12. Advertised-versus-granted breach rates are measured per campaign per cycle and
    reported per leaf, and a campaign breaching 5% is suspendable within the cycle.
13. The complete feature list of every live tree is producible on demand, with
    prohibited-feature and proxy attestations.
14. The path artefact for a monthly cycle fits within 120 GB and a single
    campaign-cycle aggregates in under 90 seconds.
15. An overlay can be approved, scoped, ordered and applied over a published tree
    without the tree being edited, without any `node_key` changing, and with the
    unadjusted outcome recorded alongside the adjusted one.
16. A cycle runs with the overlay stack disabled, and the difference between the
    two runs is attributable overlay by overlay.
17. An overlay past its review date without renewal, applied outside its declared
    scope, or whose scope matches nothing, fails the cycle with a message naming
    it. In none of those cases does the cycle continue quietly.
18. Node-level volume reporting states the stack each figure was produced under,
    flags comparisons across a differing stack, and decomposes a movement into
    population, tree version and overlay.

---

## 11. Change scenarios

1. **A campaign owner moves a threshold on Tuesday** — node 6's
   `discretionary_income` from 2 200 to 2 600 — and wants it live in Wednesday's
   delta. The March-versus-April comparison of node 6 must become visibly
   incomparable rather than silently wrong.
2. **An analyst restructures a tree**, inserting two levels and splitting one node
   into three, after a re-fit. Fourteen nodes are unchanged and must keep their
   identity; the modelling tool has renumbered all of them in the export.
3. **A new campaign is added mid-cycle** — campaign 61, a Drive Finance
   settlement offer — with its own tree, tiers, capacity claim and control design,
   between monthly runs.
4. **A feature is retired from the mart.** Eleven trees reference it. The impact
   must be known before the retirement, not after the cycle fails.
5. **Compliance rules a feature a proxy.** A postcode-derived affluence index is
   prohibited with 30 days' notice; 11 trees must be re-authored, re-validated and
   re-attested, and the Bank must be able to state which cycles used it.
6. **Control percentages change** from 5% to 3% on 12 campaigns. The clients who
   move between groups must be enumerable, and the measurement team needs to know
   which historical comparisons are now broken.
7. **Outbound call capacity falls** from 260 000 to 150 000 a month for three
   months while a contact centre site is relocated. Arbitration must reallocate
   without any campaign being changed, and the value of the refused demand must be
   reported to the campaign forum.
8. **The all-channel fatigue cap tightens** from 4 per 30 days to 3, programme
   wide, at two weeks' notice.
9. **Six campaigns move from monthly to daily.** Their trees do not change; the
   cycle they run in does, and fatigue and capacity now interact across cycles
   within a month in a way they did not before.
10. **A campaign is suspended after dispatch files are generated** and 40 minutes
    before the SMS send window, following a complaint.
11. **The analytics team wants node-level reporting split by variant and by
    control**, where today it is produced in aggregate — without re-running any
    historical cycle.
12. **The modelling tool is replaced.** The same 60 trees arrive in a different
    export format with a different notion of node ordering. Node identity must
    survive the migration or two years of node-level history is lost.
13. **Card campaigns must take their amounts from project 07's limit management
    run** rather than from a project 03 pre-assessment, while loan campaigns keep
    the current source.
14. **A second country entity adopts the platform** with 2.1 M clients, 40 of its
    own features, 18 campaigns and a different prohibited feature register, and
    must not be able to see or affect the first entity's campaigns.
15. **The contact centre cuts its committed volume** and the forum asks campaign
    23 to land at 350 000 rather than 412 000 — a volume dial on node 6's
    threshold, approved Thursday, expiring after three cycles, with the unadjusted
    volume still reported. Next month it is renewed or lapses, and the node 6
    comparison across the change must not read as a population shift.
16. **Early-arrears performance deteriorates** and Credit Risk tightens nine
    campaigns by one grade notch for two cycles while the model team investigates.
    No tree changes, node history stays continuous, and the Credit Committee wants
    volume and expected loss with and without the tightening.
17. **An overlay is found still in force** eleven months after the quarter it was
    approved for, its review date having passed during a migration. Which cycles
    ran under it, how many clients it suppressed, and what would have been
    targeted without it must all be answerable.
18. **A volume dial and a tree re-fit land in the same cycle**, and the analyst
    has moved the node the dial points at. The dial must not silently reattach to
    a different node, and the combined effect must be reviewable as one number.

---

## 12. Out of scope

- Creative content: message copy, imagery, rendering and localisation.
- Actual delivery: SMS gateways, dialler integration, email infrastructure, push
  notification.
- Fitting the trees. Campaign Analytics builds them; this project consumes and
  runs them.
- Propensity and margin model development. The scores are inputs.
- Consent capture. `core.consent` reports consent state; it is collected
  elsewhere.
- The application journey a targeted client enters. Once the client applies, it is
  project 03's decision, not this one's.
- Attribution modelling beyond joining the response to the assignment.
- Inbound and real-time next-best-action at the point of service. Related, and
  deliberately separated.

---

## 13. Questions the implementation must answer

1. **What is a node?** A tree here has nodes testing several features with `and`
   and `or`, comparisons between two features, set membership over codes and null
   tests. Is a node one thing with a compound condition, or a small structure of
   its own? What does the answer cost when 9 200 of them must be prepared for a
   run in under four minutes?
2. **Is the path a by-product or an output?** If traversal is recorded only when
   something asks for it, what happens in the run where nothing asked and the
   analytics team needed it anyway? If it is always recorded, what does it cost at
   400 M evaluations?
3. **What is the grain of a stored path** — one row per evaluation holding the
   whole traversal, or one row per node visited? 400 M rows against 3 bn rows,
   with opposite consequences for storage and for the joins the analytics team
   actually writes. Can both be served from one recording?
4. **How is node identity computed** so that it depends on the node's condition
   and its logical position and on nothing else — not on export order, not on tree
   size, not on a sibling three levels away?
5. **What exactly makes two nodes "the same test"?** Is `x >= 2200` in March the
   same node as `x >= 2200 and y in {1,2,4}` in April with the second condition
   always true for the population that reaches it? Who decides, and is the
   decision reviewable?
6. **Where does a node's metadata live** such that the warehouse can join to it
   without the decision system in the loop, and such that it cannot drift from the
   logic that actually ran?
7. **How does a tree authored outside engineering arrive**, and what is the
   boundary between what the modelling tool exports and what the Bank's validation
   adds? Is the published artefact the export, or something derived from it — and
   if derived, is the derivation reviewable by the analyst who authored it?
8. **How is a tree validated before publication** — reachability, contradiction,
   feature existence and type, permitted outcomes, population impact — and is that
   validation expressible over the same representation the run uses, or does it
   need its own?
9. **Where does a population-level constraint sit** relative to per-client
   decisions? Arbitration cannot be decided one client at a time, yet everything
   before it is per client. What is the shape of the thing that sits on top, and
   how does it stay deterministic and re-runnable?
10. **Can one campaign be re-run alone?** Arbitration says no; operations say it
    must be possible. What is the honest answer, and how is the dependency made
    explicit rather than discovered at 03:00?
11. **How are 60 campaigns' artefacts kept navigable** — tree, candidate
    population, tiers, channel rules, holdout design, sign-off history, per
    campaign, changing weekly, owned by eleven people? What does adding campaign 61
    cost, and what does retiring campaign 41 leave behind?
12. **How are shared band definitions owned** when 38 trees depend on them and a
    band edit is functionally a simultaneous edit to all 38?
13. **How is a suppression expressed** such that it is attributable per client per
    campaign per channel, and such that "evaluate anyway, but do not contact"
    (control, and measurement-relevant suppressions) is a first-class outcome
    rather than a special case bolted on?
14. **What does effective dating mean for a structure?** `core.dates` selects a
    table version by `decision_date`. Selecting a *tree* version is the same
    problem with a much larger artefact and a node identity map attached. Is it the
    same mechanism?
15. **How is 25 months of path data queried** by people who are not engineers,
    against trees that changed 40 times in that period?
16. **What is an overlay, structurally?** It is a parameter change (a number
    moves without new logic), a piece of real logic (the stacking order changes
    the answer), and an audit artefact (the stack in force on a date is part of
    the decision record) at the same time. Is it one thing in the implementation
    or three? Where does it attach — to the tree, to the campaign, to the cycle,
    or beside all of them — such that the tree is provably unedited, the
    unadjusted answer is still computable, scope violations fail loudly, and
    expiry is impossible to forget?
17. **Does the decision-tree core kind survive this project unchanged?** If path
    capture, stable node identity, multi-condition nodes and 400 M evaluations per
    cycle each require something the kind does not have, is the answer a richer
    tree kind, or is the tree the wrong unit and the path the right one?
