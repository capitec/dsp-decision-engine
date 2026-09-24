# 01 — Transaction fraud and scam interdiction

> Fictional. The Bank, its products, thresholds, rule identifiers, table
> dimensions and volumetrics are invented for this repository. Regulatory
> mechanisms referred to are the published public ones; every number attached to
> them is illustrative.

---

## 1. What this is

The Bank interdicts fraud while it is happening. Every card authorisation, every
instant payment, every beneficiary added, every device registered and every SIM
swap notification is assessed before it completes, and the assessment lets it
through, watches it, challenges the client, holds it for a human, declines it,
closes a channel, or freezes the account. There is no overnight window and no
second attempt: an instant payment that clears is gone within seconds, and a card
authorisation not answered inside the scheme's timeout is answered by the
scheme's stand-in rules instead of by the Bank.

The assessment is made by a **flat rule set** — 521 live rules, 114 shadow rules
and 1 360 retired ones — authored and tuned by fraud analysts rather than
engineers, and modified at runtime by **approved overlays** that tighten or
loosen it without editing it. Rules overlap deliberately: on a genuinely
fraudulent instant payment it is normal for nine rules to fire at once across
three families with three owners, and all nine must be recorded, not the one that
won. Which rules fired feeds the review agent's screen, the analyst's tuning, the
dispute pack, the regulatory inventory and the monthly false-positive review. It
is required output at 3 500 events per second, not a facility switched on when
something looks wrong.

The same rule definitions must also run in **backtest mode** over 90 days of
stored events, so a proposed rule, threshold or overlay can be costed before
deployment — hit rate, incremental catch, false positive rate, overlap with rules
already live, and rand value that would have been blocked. The backtest must
produce exactly what production would have produced; if the two can disagree it
is worthless, because its purpose is to be believed by a committee that will not
re-derive it.

---

## 2. Why it is in this set

| Question | How this project stresses it |
|---|---|
| **Q5 core component set** | The sharpest test of whether a decision table is enough. A decision table selects **one** row and returns its result. A fraud rule set evaluates **every** rule, keeps **every** firing, attaches owner/severity/family/status/effective-dates to each rule as first-class data, computes the action by a **separate precedence ordering** over the firing set, carries a shadow population whose firings are recorded but provably cannot influence the answer, and has its thresholds modified in flight by an overlay stack the rules know nothing about. Rules are individually effective-dated and retired; a decision table versions as a whole. Membership varies per record by segment, so the applicable set is not fixed. The answer is a set with an argmax over it, not a lookup. If "flat rule set" is not a core component kind, this is where the gap shows. |
| **Q7 audit** | A cardholder dispute can arrive 540 days later. Answering it needs the rule set as it stood that minute, the thresholds then in force **and whether they were base or overlaid**, the feature values as computed at the time rather than as recomputable now, the full firing set, the precedence applied and the wording shown to the client. Separately, regulators ask for a rule inventory, the overlay register, and evidence that a rule claimed retired on a date actually stopped affecting outcomes then. |
| **Q2 tables** | Merchant category risk (1 024 codes), country risk (249), BIN/issuer (≈50 000 rows), merchant reputation (3.2 M identifiers), mule watchlists (up to 2 000 000 entries refreshed hourly), sanctions lists refreshed every 15 minutes, plus segment, challenge and velocity-window definitions. Some are small and analyst-edited; one is large, hot and replaces itself 24 times a day; all must be replayable as at a past instant, because "was this beneficiary on the mule list last Tuesday?" is asked in writing. |
| **Q3 parameters** | ≈1 900 tunable numbers across the live set, owned by five families with five owners, changed by non-engineers under four-eyes approval, normal path in minutes and emergency path in single-digit minutes. On top sits an overlay stack with a *faster* path and a *different* approval — which is exactly why every overlay carries an enforced expiry. The split between "an analyst may change this without a release", "this needs an overlay" and "this needs engineering" is the governance model, and getting it wrong either way is fatal: too tight and analysts route around the system, too loose and one person takes the card estate offline at 19:40 on a Friday. |
| Q1 reuse *(secondary)* | Consumes `core.reason_codes`, `core.dates`, `core.consent`, `core.rounding` and `core.adjustments`, and needs almost nothing else — which tests whether the library's seams suit a non-credit consumer. |
| Q4 custom modules *(secondary)* | Velocity interpretation, watchlist membership-as-at, merchant reputation banding and overlay composition are shared between the real-time and backtest paths and must not be written twice. |
| Q6 codebase *(secondary)* | Five rule families, five owners, one deployment, no downtime window. |

---

## 3. Actors

| Actor | Responsibility |
|---|---|
| **Fraud Strategy** (14 analysts, 5 family owners) | Author, tune and retire rules. Propose overlays. Own thresholds, merchant and country risk tables, device bands, segment definitions. Not engineers. |
| **Head of Fraud / 24-7 duty officer** | Approves overlays, critical rules and the action precedence table. The duty officer is the out-of-hours authority during a live attack. |
| **Fraud Operations** (24/7, ~180 agents over three shifts) | Work the review queues, contact clients, release or confirm holds. Their throughput is a hard constraint on how many holds the rule set may generate. |
| **Financial Crime Compliance** | Owns sanctions obligations, the suspicious-activity reporting triggers, and the AML-adjacent family. |
| **Fraud Engineering** | Owns the runtime, enrichment feeds and deployment path. Does **not** own rule content and must not be in the path of a threshold or overlay change. |
| **Model Risk** | Owns the external model and its score. Requires that absence behaviour is specified rather than emergent, and that any overlay on the score cut-off stays visible as an overlay. |
| **Disputes and Chargebacks** | Answers disputes and scheme representments up to 540 days later. The heaviest consumer of the decision record. |
| **Client Experience / Contact Centre** | Must tell a client in plain language why a payment stopped, without reading a rule. |
| **Internal Audit** | Asks, years later, which rules and overlays were live on a date, who approved them, and whether things claimed retired or expired actually were. |
| **Regulator and card schemes** | Ask for rule inventories, fraud-rate reporting and evidence of interdiction controls. |

Two actor facts shape the design more than anything in §5. The people who change
the logic are not the people who deploy it. And Fraud Operations is fixed
capacity: a change that increases holds by 30% does not produce 30% more reviews,
it produces a queue that breaches its service level and a manual instruction to
release everything above a certain age unexamined.

---

## 4. Inputs

### 4.1 Events in scope

One engine serves all of these. They do not share a field set.

| Event type | Code | Volume / month | Fields | Latency ceiling | Notes |
|---|---|---|---|---|---|
| Card authorisation, card-present | 110 | 48.0 M | ~62 | 25 ms | Scheme timeout is the binding constraint. |
| Card authorisation, card-not-present | 111 | 31.0 M | ~71 | 25 ms | Carries 3-D Secure context, merchant descriptor, device fingerprint where available. |
| Instant payment (real-time credit transfer) | 210 | 22.0 M | ~38 | 25 ms | Irrevocable once cleared. The scam-loss concentration. |
| EFT credit transfer | 211 | 14.0 M | ~34 | 400 ms | Originated in batches, decisioned individually. |
| Debit order dispute lodgement | 220 | 1.4 M | ~26 | 400 ms | First-party fraud surface. |
| Login / session establishment | 310 | 19.0 M | ~29 | 40 ms | Not a payment; precedes account takeover. |
| Device registration | 311 | 0.9 M | ~33 | 40 ms | |
| SIM or mobile number change | 312 | 0.4 M | ~18 | 40 ms | Arrives from the network minutes late, sometimes hours. |
| Beneficiary addition | 313 | 2.6 M | ~24 | 40 ms | The highest-yield scam interdiction point. |
| Contact detail change | 314 | 0.7 M | ~21 | 40 ms | |
| Limit change request | 315 | 0.5 M | ~19 | 40 ms | |
| Card re-issue / replacement | 316 | 0.3 M | ~22 | 400 ms | |
| **Total** | | **140.8 M** | | | |

The union across all twelve types is ≈210 distinct fields, of which 41 appear on
every type (event identity, timestamp, client, channel, device where known,
session). A card authorisation carries 30 fields a login does not have and could
not have. A rule referencing a field its event type does not carry is a
definition error that must be caught before deployment, not at 02:00 on a Sunday.
Peak load is 12 000 events/second — last Friday of the month, 16:00 to 20:00, and
two days after a public-sector pay run. Sustained load is 3 500/second.

### 4.2 Enrichment sources

| Source | Shape | Freshness | Missing rate | On failure |
|---|---|---|---|---|
| Client profile snapshot | ~70 attributes: tenure, segment, holdings, vulnerability flag, travel notice, dispute counts, prior confirmed-fraud count | Nightly; hot fields (frozen, card status) read live | 0.02% | §5.6 |
| Device and session signals | Device id, reputation band, age on file, root/emulator flags, IP, IP-to-account geo distance, session age, behavioural-biometrics band | Live | 11% of card-present events legitimately have none | Absent is a value, not a failure |
| Counterparty / beneficiary | Beneficiary age on file, first-payment flag, prior payment count and sum, mule-list membership, beneficiary bank, confirmation-of-payee name-match band | Live for membership; daily for aggregates | 0.4% | §5.6 |
| Merchant and acquirer | MCC, acquirer country, merchant identifier, reputation band, chargeback-rate band, terminal type | Authorisation message plus merchant table | 2.1% unrecognised merchant | Unknown band |
| Velocity aggregates | 168 precomputed values (§4.3) | Median lag 180 ms, p99 900 ms | 0.3% steady state, up to 9% in streaming incidents | §5.4 |
| External fraud model score | Float 0–1000, model id, version, three contributing-factor codes | Per event, over a network call | 0.4% steady state, up to 6% in serving incidents | §5.5 |
| Sanctions and blocked-party list | ≈240 000 entries | Every 15 minutes | — | Fail closed |
| Mule watchlist | Up to 2 000 000 identifiers | Hourly, membership time-stamped | — | Fail closed on last good copy |

### 4.3 Velocity aggregates

Aggregates arrive precomputed from a streaming store. They are not computed here;
correctness depends on knowing how old they are.

| Dimension | Values |
|---|---|
| Grouping key | client, card, device, beneficiary, merchant, IP, client×event-type (7) |
| Window | 1 min, 10 min, 1 h, 24 h, 7 d, 30 d (6) |
| Statistic | count, sum of amount, distinct counterparties, distinct devices (4) |
| Total per event | 168 values |

Each aggregate carries its own watermark. **Staleness tolerance is per window**: a
1-minute aggregate is stale at 2 seconds, a 24-hour aggregate at 60 seconds, a
30-day aggregate at 15 minutes. Stale is not missing and missing is not zero. A
client who made no payments in the last hour and a client whose hourly count could
not be retrieved must be distinguishable by any rule reading the value, and the
distinction must survive into the decision record.

### 4.4 Vocabulary

From `core`, verbatim: `client_id`, `channel_code`, `product_code`,
`decision_date`, `decline_reason_codes`, `primary_reason_code`. Declared locally,
because the library does not publish them:

| Name | Type | Meaning |
|---|---|---|
| `event_id` | uint64 | One assessable event. Globally unique, monotonic per partition. |
| `event_type_code` | int16 | §4.1. |
| `event_timestamp` | timestamp(µs, UTC) | When the event occurred, not when it was assessed. |
| `assessment_timestamp` | timestamp(µs, UTC) | When this engine answered. |
| `rule_id` | string(8) | Stable, e.g. `CF-0412`. Never reused after retirement. |
| `rule_version` | int16 | Increments on any change to the rule, including a threshold. |
| `rule_set_version` | int32 | The complete set in force at an instant. |
| `fired_rule_ids` | list[string] | Every live rule that evaluated true. |
| `fired_on_overlay_ids` | list[string] | The subset that fired only because an overlay moved a threshold. |
| `shadow_fired_rule_ids` | list[string] | Every shadow rule that evaluated true. |
| `adjustment_stack_version` | int32 | The overlay stack in force at `event_timestamp`. |
| `applied_adjustment_ids` | list[string] | Overlays in scope for this event, in composition order. |
| `action_code` | int8 | §5.12. |
| `action_source_rule_id` | string(8) | The rule whose action was taken. |
| `enrichment_degradation_code` | int16 | Bitset of sources absent or stale. |

---

## 5. The flow

Stages 5.1–5.15 run on every event. 5.16 runs days to weeks later. 5.17 is the
batch mode and shares 5.1–5.15's definitions exactly.

### 5.1 Event admission and normalisation

**Precondition.** A raw event has arrived on one of six inbound channels.

**Determines.** `event_type_code`; that the mandatory field set for the type is
present; that `event_timestamp` is within tolerance of now (events over 5 minutes
old on a real-time channel are assessed but flagged — a SIM change notification
routinely arrives late, a card authorisation never legitimately does); the client
and account; the amount in the reporting currency, converted at the rate carried
in the message.

**Emits.** A normalised event: the 41 common fields populated, type-specific
fields present or explicitly absent.

**Recorded.** The raw event byte-identical, keyed by `event_id`; the normalised
event; any field that failed validation, with the reason; the conversion rate and
source. A malformed event is never silently dropped — it is assessed under a
reduced rule population and recorded as such, because a malformed message is
itself a weak fraud signal and a dropped event is the one the dispute is about.

### 5.2 Client and account context

**Precondition.** 5.1 resolved a `client_id` and an account.

**Determines.** The client profile as at `event_timestamp`: tenure, segment
membership, vulnerability flag, active travel notice, product holdings, prior
confirmed-fraud count, prior disputes lodged, prior false-positive holds in 90
days, account status (open, frozen, restricted, in estate), card status per card.
Also which `core.consent` permissions are in force — SMS, in-app push, voice —
and whether a contact-detail change in the last 24 hours makes any of those
untrustworthy for a challenge.

**Emits.** The context attributes rules may reference.

**Recorded.** Every attribute value used, and the snapshot it came from. Not a
pointer to the profile store — the value. The store is mutable and will not hold
this value in 540 days.

### 5.3 Counterparty, device and merchant enrichment

**Precondition.** 5.1 complete. Which parts apply depends on event type.

**Determines — counterparty** (payments, beneficiary additions, debit order
disputes). Beneficiary age on file in hours; first-payment flag; count and sum of
prior payments to this beneficiary over each velocity window; beneficiary bank;
confirmation-of-payee name-match band (exact, close, no-match, not-available);
mule watchlist membership **as at `event_timestamp`** with listing source and the
instant it was added; whether the beneficiary was added within the last 60
minutes and by which channel.

**Determines — device and session.** Device reputation band; hours since this
device was first seen on this account; hours since **any** device change; hours
since the last SIM or number change; hours since the last contact-detail change;
emulator, rooting and automation indicators; behavioural-biometrics anomaly band
where supplied; session-IP-to-locus distance in bands. These cross-event deltas
are the account-takeover family's core features and are why non-payment events
are in scope at all — they exist only because logins, device registrations and
number changes were themselves assessed and recorded.

**Determines — merchant and acquirer** (card authorisations). MCC and risk band;
acquirer country and risk band; merchant identifier, reputation band and
chargeback-rate band; terminal type; merchant-initiated flag; BIN-derived issuer
attributes where the card is not the Bank's; whether the merchant is on the
client's own prior-use list.

**Emits.** The three attribute groups, with explicit unknown bands where a lookup
did not resolve (2.1% of card-not-present events have an unrecognised merchant).

**Recorded.** Every table version and the cell read from each. A watchlist match
must be reproducible as a statement: "on the list at 14:22 on 3 March, added
09:11 on 2 March by the consortium feed, removed 11:00 on 18 April."

### 5.4 Velocity aggregate attachment

**Precondition.** 5.2 resolved client and card; 5.3 resolved the beneficiary
where applicable.

**Determines.** The 168 aggregate values and, per aggregate, one of **fresh**,
**stale** (older than its window's tolerance) or **absent**; then
`velocity_completeness_band` over the set — complete, partially degraded (any
stale), materially degraded (more than 10 absent, or any absent aggregate in the
1-minute or 10-minute windows).

**Emits.** The aggregates and their states.

**Recorded.** Every value used by any rule, with watermark and state. A rule that
fired on `count_1min > 4` must be reproducible, and it cannot be if the recorded
artefact is "the aggregate store said so at the time".

Rules referencing a stale or absent aggregate do not silently read zero. Each
rule declares its behaviour per referenced feature: evaluate as false, use the
last known good value, or suppress the rule and record it as unevaluable. The
third is the honest option and the one that makes an aggregate outage visible in
the outcome statistics rather than as a quiet dip in catch rate.

### 5.5 External model score

**Precondition.** 5.2–5.4 produced the model's feature set.

**Determines.** `model_score` (0–1000), `model_id`, `model_version` and up to
three contributing-factor codes — or the explicit absence of all of them. The
score is fetched over a network call with a hard **8 ms deadline**, issued
concurrently with the enrichment lookups so it does not extend the critical path.

Behaviour when it does not arrive is a stated rule, not a timeout artefact: the
score is marked absent, `enrichment_degradation_code` records it, rules
referencing it follow their declared absence behaviour, and a designated set of
compensating rules referencing no model score becomes applicable. Two events with
identical inputs, one scored and one not, may receive different actions; what is
not allowed is the same event replayed producing a different action because the
replay happened to get a score. A replay uses the recorded score, or the recorded
absence.

**Recorded.** Score and factors, or absence with its reason (timeout, error,
circuit open, not requested for this type); model id and version; the deadline;
and the score **cut-off** applied, both base and overlaid where an overlay moved
it (§5.9).

### 5.6 Enrichment completeness verdict

**Precondition.** 5.2–5.5 have each reported success, degradation or absence.

**Determines.** `enrichment_degradation_code`, a bitset over the eight sources;
and `degraded_mode_code` — **normal**, **reduced** (one non-critical source
degraded), **restricted** (two or more degraded, or velocity materially
degraded), **fail-closed** (sanctions or mule list unavailable and the last good
copy older than tolerance). In restricted mode a declared subset of rules is
suspended and a declared subset of compensating rules activated. In fail-closed
mode, payments to never-before-paid beneficiaries above a threshold are held
rather than allowed.

**Emits.** The mode and the applicability adjustment it implies.

**Recorded.** The mode, the bitset, and every rule suspended or activated by it.
When the post-mortem asks why losses spiked on the afternoon the streaming store
fell over, the answer must be in that afternoon's decision records, not inferred.

### 5.7 Hard blocks

**Precondition.** Enrichment complete or explicitly degraded.

**Determines.** Whether any of the following holds:

| Gate | Action forced | Reason family |
|---|---|---|
| Counterparty on the sanctions list | `block_channel` + mandatory report | Sanctions |
| Account frozen | `decline` | Account state |
| Beneficiary on the confirmed-mule watchlist | `decline` + mandatory report | Mule |
| Device on the blocked-device list | `block_channel` | Device |
| Card reported lost, stolen or compromised | `decline` | Card state |
| Court-ordered or investigation hold | `hold_for_review` | Legal |

These short-circuit the **action** — no rule and no overlay can soften them — but
not the **evidence**. The full live and shadow set is still evaluated and the
complete firing set still recorded, because the analyst tuning the mule family
next quarter needs to know which of her rules would have caught this
independently, and because a sanctions block later found to be a false name match
must be explainable in terms of everything else that was true. Where two gates
hold, both are recorded and the more severe action applies.

**Emits.** `hard_block_code` and the forced action, or none.

**Recorded.** Every gate that held, the list version and matching entry, and the
fact that rule evaluation continued regardless.

### 5.8 Segment and applicability resolution

**Precondition.** Client context and event type known.

**Determines.** Which of the 521 live and 114 shadow rules apply. A rule applies
when its event-type set contains this type; its status is live or shadow;
`event_timestamp` falls within its effective-from and effective-to; and its
segment set intersects the client's segments (or is "all"). Applicability is then
adjusted by 5.6's degraded-mode suspensions and activations and by any
scope-restriction overlay from 5.9. Segments are not disjoint — a client is
typically in 3–6 of the 46 defined segments (new-to-bank, high-value,
business-linked, youth, pensioner, offshore traveller, prior-fraud-victim,
vulnerable, and so on).

**Emits.** The applicable live and shadow populations, as explicit sets.

**Recorded.** `rule_set_version`; counts applicable by family; and — the awkward
one — enough to reconstruct the applicable set exactly, without storing 635 rule
identifiers on every one of 140 million monthly records.

### 5.9 Adjustment overlay resolution

**Precondition.** Segment, channel, event type and family scopes are known.

An **adjustment** is a named, approved, effective-dated overlay applied on top of
the rule set rather than an edit to it. When an attack is running, when a
family's precision drifts, or when the business wants to tighten one channel for
a quarter, policy does not rewrite 40 rules — it layers an overlay. The base rule
set stays exactly as authored and approved; this follows `core.adjustments` and
the library's prohibition on merging an overlay into its base (00 §7.6).

**Determines.** Which overlays are in force at `event_timestamp` and in scope for
this event; their composition order; and the resulting **effective** value of
every threshold, cut-off, severity and action the applicable rules will read.
Kinds are in §6.5. For every modified value it determines both the base and the
effective value, so that 5.10 can answer "would this rule have fired without the
overlay?" for every firing.

**Emits.** `adjustment_stack_version`, `applied_adjustment_ids` in composition
order, and the effective parameter set for this event.

**Recorded.** The stack version; every overlay that applied, with its effect;
every overlay in force but **out of scope** for this event, which is how an
analyst confirms scoping does what she intended; and, per modified value, base
and effective side by side. An overlay applied outside its declared scope is an
error, not a silent no-op, and must fail loudly.

Overlays never change a rule's shape, never change which action a rule asks for
absent an explicit action-escalation overlay, and never create or delete a rule.
Anything needing that is a rule change and takes the rule change path.

### 5.10 Live rule evaluation

**Precondition.** 5.8 produced the applicable live population; 5.9 produced the
effective parameter set.

**Determines.** For every applicable live rule: true, false, or **unevaluable** (a
referenced input was stale or absent and the rule's declared behaviour is
suppression). Each rule is evaluated against its **effective** thresholds and,
where an overlay moved them, separately against its **base** thresholds.

**There is no early exit.** Every applicable rule is evaluated even after a
`decline`-action rule has fired and even after a hard block in 5.7, because the
complete firing set is required output. Evaluation order must not affect the
result: rules are independent and may not observe one another's outcomes.

| Rule shape, across the live set | Value |
|---|---|
| Mean predicates per rule | 4.2 |
| Maximum observed | 17 |
| Pure conjunctions | 78% |
| One disjunctive group | 19% |
| Rule-local derived quantity (a ratio, a difference of two features) | 3% |
| Referencing the model score | 11% |
| Referencing at least one velocity aggregate | 64% |
| Referencing a table lookup band | 71% |

> `MS-0208` — first payment to a beneficiary added under 2 hours ago, amount
> above R8 000, beneficiary bank in the high-mule-rate band, and a device change
> within 72 hours. Family: mule/scam. Severity 4. Action: `hold_for_review`,
> queue `SCAM-1`. Segments: all except business-linked. Critical: no.

Its tunables are R8 000, 2 hours and 72 hours. Its shape — four predicates over
four features joined by `and` — is not a tunable. Under a 1.4× sensitivity
overlay on the mule/scam family the amount threshold becomes R5 715 for the
overlay's lifetime; the rule's definition still reads R8 000.

**Emits.** `fired_rule_ids`, and per fired rule: `rule_id`, `rule_version`, the
action asked for, severity, priority, family, reason code, the feature values
tested, and whether it fired on the base threshold or only under the overlay —
the latter subset being `fired_on_overlay_ids`.

**Recorded.** All of the above plus the unevaluable set. Not recorded: the ~599
rules that evaluated false — but the applicable population must be
reconstructable, so "did `CF-0311` apply and not fire, or not apply?" has an
answer.

### 5.11 Shadow rule evaluation

**Precondition.** 5.8 produced the applicable shadow population.

**Determines.** For every applicable shadow rule, true or false, by exactly the
same evaluation as 5.10 over the same feature values and the same effective
parameter set.

Shadow rules are candidates under test on live traffic. They are evaluated, their
firings recorded, and they **cannot** influence `action_code`, `fired_rule_ids`,
the reason set, queue routing, or anything a client or agent sees. A shadow rule
that changed an outcome is a defect of the severest class, and the system must
make it structurally impossible rather than merely discouraged.

Promotion to live is how a rule is normally born. At promotion the rule keeps its
`rule_id`, and its shadow firing history must remain joinable to its live
history, so an analyst can say "11 400 firings in shadow over six weeks, 340
later confirmed fraud" and compare that to its first six weeks live.

**Emits.** `shadow_fired_rule_ids` with the same per-rule detail as 5.10.

**Recorded.** Separately from the live firing set, in a way a downstream consumer
cannot accidentally union with it.

### 5.12 Action resolution

**Precondition.** 5.7's hard block verdict and 5.10's live firing set.

**Determines.** One `action_code`, one `action_source_rule_id`, and a complete
reason set. Action precedence, most severe first:

| Rank | `action_code` | Action | Meaning |
|---|---|---|---|
| 1 | 70 | `freeze_account` | All movement stops pending investigation. |
| 2 | 60 | `block_channel` | The originating channel is closed for this client. |
| 3 | 50 | `decline` | This event does not complete. |
| 4 | 40 | `hold_for_review` | Held pending a human decision, within an SLA. |
| 5 | 30 | `step_up` | Challenge the client; on success the event proceeds. |
| 6 | 20 | `monitor` | Proceeds; flagged for downstream analysis. |
| 7 | 10 | `allow` | Proceeds. |

1. A hard block from 5.7 sets a floor. Nothing may produce a less severe outcome.
2. Any fired rule marked **critical** is binding: its action is taken even where a
   less severe action would otherwise have won — and, in the allow-listing
   direction, a critical rule with action `allow` suppresses all non-critical
   firings for the families it names. (A corporate client's pre-notified bulk
   payment run, where velocity rules would otherwise stop every payment after the
   fourth.)
3. If two or more critical rules fire with different actions, the more severe wins
   and the collision is recorded as a **governance exception** routed to both
   owners. Critical rules are meant to be rare and non-overlapping; when they
   overlap, two owners have made incompatible assumptions.
4. Otherwise the most severe action among the fired rules wins.
5. Ties — two or more rules asking for the same action — resolve by severity
   descending, then priority ascending, then family precedence, then `rule_id`
   ascending. The last guarantees determinism; the three before it guarantee it is
   not arbitrary.
6. The complete reason set is the ordered reason codes of **all** fired rules,
   ranked through `core.reason_codes`, not only the winner's.
   `primary_reason_code` is the winner's.

**Emits.** `action_code`, `action_source_rule_id`, `decline_reason_codes`,
`primary_reason_code`, the governance exception flag, and — where any contributing
firing was overlay-induced — the **counterfactual action** the base rule set alone
would have produced.

**Recorded.** The precedence table version, every resolution point that
discriminated, the runners-up (second and third most severe firings), and the
counterfactual. When an analyst asks why her rule did not take effect although it
fired, and when the Committee asks what an overlay actually bought, the answers
are in the record.

### 5.13 Action dispatch and downstream obligations

**Precondition.** `action_code` determined.

**Determines.**

- **Review queue and SLA tier**, where the action is `hold_for_review`. Eleven
  queues by family and severity; four SLA tiers (15 minutes, 1 hour, 4 hours,
  next business day). Instant payments held for review occupy the 15-minute tier
  because the client is standing at a till. Queue choice comes from the winning
  rule, subject to any queue-reroute overlay and to capacity-aware demotion: when
  a queue's depth exceeds its threshold a declared overflow policy governs, and
  the overflow decision is part of the record.
- **Step-up challenge type**, where the action is `step_up`. Selected from the
  challenge matrix by segment, channel and available contact methods, constrained
  by 5.2's consent and trust findings: a client whose mobile number changed 40
  minutes ago is not challenged by SMS one-time password. Where no permitted
  challenge exists the action escalates to `hold_for_review`, recorded as an
  escalation rather than silently.
- **Client notification.** Every `decline`, `hold_for_review`, `block_channel` and
  `freeze_account` raises a notification obligation with a wording key and a
  channel, subject to `core.consent`.
- **Suspicious-activity reporting trigger.** Certain firings — the AML-adjacent
  family, confirmed-mule matches, sanctions blocks — raise a reporting obligation
  with a statutory clock. The public mechanism is a report to the financial
  intelligence authority no later than 15 business days after the Bank becomes
  aware. The engine does not file the report; it starts the clock, names the
  trigger, and is the evidence of when awareness arose.
- **Downstream instruction**, where the action is `freeze_account` or
  `block_channel`: which systems must act, and whether the instruction is
  idempotent on retry.

**Emits.** The dispatch instruction set.

**Recorded.** Every obligation raised, its clock start, its addressee, and whether
dispatch was acknowledged. The service runs 24 hours a day, 365 days a year; no
obligation may be deferred to "the next run", because there is no next run.

### 5.14 Client-facing reason wording

**Precondition.** The action is client-visible.

**Determines.** The wording shown to or spoken to the client, resolved from
`core.reason_codes` in the client's language at the correct disclosure level.

Fraud wording is deliberately less specific than credit decline wording — telling
a client which velocity threshold they crossed tells the fraudster holding their
phone the same thing. The record must hold **both** the internal reason set and
the external wording used, and the mapping between them: a complaint six months
later is about what the client was told, a scheme representment about what was
true.

**Emits.** Wording key, rendered text, language, disclosure level.

**Recorded.** All of the above plus the reason registry version. Wording changes;
what the client was told on the day does not.

### 5.15 Decision record emission

**Precondition.** 5.1–5.14 complete.

**Determines.** The durable record. Written asynchronously — it must not sit on
the client's critical path — but not optionally: a record that fails to persist is
an incident, not a dropped metric.

Contents: `event_id`, both timestamps, the normalised event, every enrichment
value used, every velocity value with watermark and state, the model score or its
absence, `rule_set_version`, `adjustment_stack_version` and
`applied_adjustment_ids` with per-overlay effect, the applicable population,
`fired_rule_ids` with per-rule feature values and base-versus-overlay basis,
`shadow_fired_rule_ids`, `action_code`, `action_source_rule_id`, the
counterfactual action, the resolution trace, every table version and cell read,
the degradation bitset and mode, dispatch instructions, obligations raised, and
the client wording.

Volume: 4–9 KB per event before compression; ~140 M records a month; retained 7
years. Queryable by `event_id`, `client_id`, `rule_id` (all events a given rule
fired on over a date range — the analyst's daily question), adjustment id, and
time range.

**Recorded.** Itself, plus a per-minute aggregate of firing counts by rule and by
overlay, which is what makes rule-level circuit breakers, overlay impact
monitoring and dead-rule detection possible without scanning the record store.

### 5.16 Outcome feedback

**Precondition.** Days to weeks have passed. An outcome is now known.

**Determines.** For an event: confirmed fraud (with loss amount and type),
confirmed scam (with typology), confirmed false positive (the client complained,
or the agent released it and nothing followed), chargeback lodged, chargeback won
or lost, or no outcome.

| Source | Typical lag | Volume |
|---|---|---|
| Review agent disposition | 15 min – 1 day | 1.1 M/month |
| Client-reported fraud | 1 – 30 days | 42 000/month |
| Scheme chargeback notification | 5 – 120 days, up to 540 | 31 000/month |
| Confirmed-mule listing (retrospective) | 7 – 60 days | 9 000/month |
| Recovery / repatriation outcome | 30 – 180 days | 6 000/month |

**Determines also.** The join back. An outcome attaches to the exact `event_id`,
`rule_set_version`, `adjustment_stack_version` and firing set — not to "the rule
as it is now". A rule whose threshold moved three times since must still be
measurable against the events it fired on at each threshold, and separately
against those it fired on only under an overlay.

**Emits.** Labelled events feeding rule performance statistics: per rule, per
version, per segment and per overlay state — fire count, true and false positive
counts, precision, incremental catch over the rest of the live set, value blocked.

**Recorded.** The outcome, its source, its lag, and any revision. Outcomes get
revised — a confirmed fraud is sometimes reclassified as first-party abuse and a
rule's measured precision changes retroactively. Both must survive.

### 5.17 Batch backtest mode

**Precondition.** A proposed change exists — a rule, a threshold move, a
retirement, an overlay, or a whole candidate set — and 90 days of stored events
with their recorded enrichment values are available (≈420 M events).

**Determines.** What the change would have done:

| Metric | Definition |
|---|---|
| Hit rate | Events on which the proposal fires, over applicable events. |
| Precision | Confirmed fraud or scam among firings, where an outcome is known. |
| Incremental catch | Confirmed fraud caught by the proposal that **no currently live rule** caught. The number the committee actually cares about. |
| False positive rate | Firings on events with a confirmed-good outcome. |
| Overlap | Per existing live rule, the proportion of the proposal's firings that rule also caught. |
| Value blocked | Rand value of events whose action would change to a blocking action. |
| Operational load | Additional holds per day by queue, against that queue's capacity. |
| Action churn | Events whose resolved action changes, in each direction. |

**With and without the overlay stack.** Every backtest is runnable in two modes:
with the adjustment stack applied as it was (or as proposed) and with it disabled
entirely. The second is how the base rule set's own performance stays measurable,
which is the only defence against an overlay stack that has quietly become the
real policy while the rule set it modifies has stopped being maintained. A result
that does not state which mode produced it is not admissible at the Committee.

**The equivalence requirement.** Running the live rule set and live overlay stack
in backtest mode over historical events must reproduce, for every event, the exact
`action_code`, `fired_rule_ids`, `fired_on_overlay_ids`, `shadow_fired_rule_ids`
and reason set production produced at the time. This is an acceptance criterion
(§10.6), checked continuously: a daily job replays a sample of the previous day's
events and asserts equality.

Two things make it hard. The backtest must use **recorded** feature values, never
recomputed ones — a velocity aggregate recomputed today will not equal what the
streaming store had at 14:22:03.417 on the day, and a backtest that silently
recomputes produces conclusions about a system that never existed. And the
backtest runs 90 days against one proposed rule set version while production ran
against however many versions and overlay stacks were live across those 90 days;
the comparison must state which baseline it used and support either — the set as
it was at each event, or the set as it is today.

**Emits.** A decision pack: the metrics, the overlap matrix, a sample of 100
firing events with full firing detail, and the recommendation.

**Recorded.** The run: inputs, event window, baseline choice, overlay mode,
proposed change, results, who ran it and when. Anything deployed on the strength
of a backtest must be traceable to it.

### 5.18 Where this goes wrong in practice

- **Rule set bloat.** The live set grows monotonically because adding a rule is an
  incident response and removing one is a project. Of 521 live rules, 60–90 have
  not fired in 90 days and another 40 fire only on events a higher-severity rule
  already catches. Dead rules cost latency, review time and credibility.
- **Overlapping rules that all fire.** Normal, not pathological, and the reason the
  firing set rather than the winner is the output. It means rule performance is not
  additive: "this rule caught 300 frauds" is meaningless without "of which 280 were
  also caught by four other rules".
- **Priority collisions across families.** Card fraud severity 4 and AML-adjacent
  severity 4 asking for different actions, owned by two teams who have never
  spoken. The tie-break resolves it deterministically; it does not resolve it
  *correctly*, which is what the governance exception is for.
- **Feature staleness changing the answer.** The commonest cause of "why did this
  not fire in the backtest when it fired in production". A rule reading a 1-minute
  velocity count is reading a value that was 400 ms old when production saw it.
- **Overlays outliving their reason.** A tightening applied during one bad month
  and still in force four years later, which nobody can explain, silently absorbed
  into the baseline against which every subsequent change was measured. This is why
  expiry is enforced and not advisory.
- **Shadow-to-live promotion.** The rule that performed beautifully in shadow and
  degrades on promotion, because in shadow it never changed an outcome and so never
  suppressed the behaviour it was measuring.
- **"Which rules fired" at full volume.** 140 million records a month, each with a
  variable-length firing list, per-rule feature values and a base-versus-overlay
  basis. Not a debug facility with a sampling rate. The product.

---

## 6. Parameters and tables

### 6.1 Rule definitions

The rule set is the largest parameterised artefact. Every rule carries:

| Attribute | Type | Changed by | Approval |
|---|---|---|---|
| `rule_id` | string(8), immutable, never reused | — | — |
| `rule_version` | int16 | System, on any change | — |
| Description, internal | text | Author | Four-eyes |
| Description, client-safe | text | Author + Client Experience | Four-eyes |
| Owner | team + named individual | Family owner | Family owner |
| Family | `CF` / `AT` / `MS` / `FP` / `AA` | Fixed at creation | Head of Fraud |
| Severity | 1–5 | Family owner | Four-eyes |
| Action | one of seven | Family owner | Four-eyes |
| Priority | 0–999 within action | Family owner | Four-eyes |
| Status | live / shadow / retired | Family owner | Four-eyes |
| Effective from / to | timestamp | Family owner | Four-eyes |
| Event types | set of `event_type_code` | Author | Four-eyes |
| Segments | set of segment ids, or all | Author | Four-eyes |
| Overlay-exempt | bool — thresholds may not be overlaid | Head of Fraud | Two named approvers |
| Critical | bool | Head of Fraud only | Two named approvers |
| Suppressible | bool | Family owner | Four-eyes |
| Reason code | int16 into `core.reason_codes` | Compliance | Compliance |
| Queue + SLA tier | where action is `hold_for_review` | Family owner + Fraud Ops | Both |
| Challenge type | where action is `step_up` | Family owner | Four-eyes |
| Stale/absent behaviour | per referenced feature | Author | Four-eyes |
| Max fire rate | % of applicable events in 5 min | Family owner | Four-eyes |
| Predicates | the rule's shape and its thresholds | §6.2 | §6.2 |

| Family | Live | Shadow | Retired (replayable) |
|---|---|---|---|
| `CF` card fraud | 214 | 31 | 690 |
| `AT` account takeover | 112 | 22 | 300 |
| `MS` mule and scam | 96 | 40 | 180 |
| `FP` first-party fraud | 41 | 9 | 70 |
| `AA` AML-adjacent | 58 | 12 | 120 |
| **Total** | **521** | **114** | **1 360** |

A rule firing on more than its **max fire rate** in a five-minute window
self-demotes to shadow and pages its owner. This is the protection against a
threshold typo taking the card estate offline: `amount > 800` where
`amount > 8000` was meant trips within minutes rather than within a news cycle.
The demotion is outcome-affecting and is recorded on every event assessed after
it. A rule tripping **while an overlay is in force on it** must attribute the trip
correctly — the overlay is the likelier culprit and demoting the rule may be the
wrong remedy.

### 6.2 Thresholds

Approximately **1 900 tunable numbers** across the live set — amounts, counts,
hour deltas, band cut-offs, score cut-offs. Roughly 85% of rule changes are
threshold-only and change no shape; the remaining 15% add or remove a predicate,
change a conjunction to a disjunction, or reference a feature the rule never used.

The distinction carries different approvals, tests and latencies. A threshold move
is backtested, four-eyes approved and live in under 10 minutes. A shape change
additionally requires that every referenced feature exists for every event type in
the rule's scope. A **threshold overlay** (§6.5) changes the value a rule reads
without changing the rule at all, is live in under 2 minutes, and expires.

### 6.3 Global and family parameters

| Parameter | Range | Owner | Cadence |
|---|---|---|---|
| Per-family severity floors | 1–5 | Family owner | Quarterly |
| Action precedence table | 7 rows | Head of Fraud | Annual; changing it is a major event |
| Queue capacity thresholds and overflow policy | 11 queues × 4 tiers | Fraud Ops | Weekly |
| Velocity staleness tolerances | 6 windows | Fraud Engineering + Fraud Strategy | Rarely |
| Model score deadline | ms | Fraud Engineering | Rarely |
| Degraded-mode suspension and activation sets | ~40 and ~15 rules | Head of Fraud | Quarterly |
| Adjustment stack | §6.5 | Head of Fraud / duty officer | Continuous |

### 6.4 Tables

| Table | Rows / cells | Owner | Cadence | Source | Replay requirement |
|---|---|---|---|---|---|
| Merchant category risk | 1 024 MCCs × 6 attributes | Fraud Strategy | Weekly | Internal | Version per event |
| Country risk | 249 countries × 5 attributes | Financial Crime Compliance | Monthly; emergency same-day | Internal + sanctions feeds | Version per event |
| BIN / issuer | ≈50 000 rows × 9 attributes | Cards Operations | Monthly | Scheme file | Version per event |
| Merchant reputation | 3.2 M identifiers × 4 attributes | Fraud Strategy | Daily | Internal + consortium | Version per event |
| Device reputation bands | 12 bands × 7 attributes | Fraud Strategy | Quarterly | Internal | Version per event |
| Beneficiary age buckets | 9 buckets | Fraud Strategy | Rarely | Internal | Version per event |
| Mule watchlist | up to 2 000 000 entries × 5 attributes | Financial Crime Compliance | **Hourly** | Internal + consortium + law enforcement | **Membership as at an instant** |
| Sanctions / blocked party | ≈240 000 entries | Financial Crime Compliance | Every 15 min | External | Membership as at an instant |
| Blocked devices | ≈180 000 | Fraud Strategy | Continuous | Internal | Membership as at an instant |
| Compromised card register | ≈90 000 open | Cards Operations | Continuous | Internal + scheme alerts | Membership as at an instant |
| Velocity window definitions | 7 keys × 6 windows × 4 statistics = 168 | Fraud Strategy + Engineering | Rarely | Internal | Version per event |
| Segment definitions | 46 segments | Fraud Strategy | Monthly | Internal | Version per event |
| Challenge matrix | 7 challenge types × 46 segments × 6 channels | Fraud Strategy + Client Experience | Quarterly | Internal | Version per event |
| Queue and SLA matrix | 11 queues × 4 tiers | Fraud Ops | Weekly | Internal | Version per event |
| Reason code registry, fraud subset | ≈120 of `core.reason_codes`' 380 | Compliance | Monthly | `core.reason_codes` | Version per event |
| Adjustment register | 12–40 live overlays × 11 attributes | Head of Fraud | Continuous | Internal | Stack version per event |

Every table requires cell-level attribution (which cell of which version produced
this band) and diffability (a new merchant category risk table produces a
reviewable change list, not a replaced file). The large, fast tables add a third:
**membership as at an instant.** A 2 M-row watchlist that replaces itself 24 times
a day cannot be replayed by keeping 24 full copies a day for seven years. "Was
account X on the list at 14:22:03 on 3 March 2026?" must be answerable cheaply and
exactly, and the answer must be the same in 2031.

### 6.5 Adjustments — the overlay stack

Overlays are how the system is tuned between rule releases, consumed through
`core.adjustments`. The kinds below are this project's instances of it.

| Kind | Applies to | Example |
|---|---|---|
| Sensitivity dial | every tunable in a family, by a declared multiplier | run the card-fraud family at 1.4× sensitivity for 72 hours during an attack |
| Threshold multiplier | one named threshold class | amount thresholds on mule/scam rules × 0.7 for the festive period |
| Model cut-off shift | the score cut-off read by the 11% of rules that use it | cut-off 620 → 540 while the model under-predicts on a channel |
| Severity shift | a family's or a named rule's severity | `MS` severity +1 for a campaign period, changing tie-breaks and queue routing |
| Action escalation | the action a family's rules ask for | `monitor` → `step_up` for account-takeover rules on the web channel only |
| Scope restriction | applicability | suspend three named rules for the business-linked segment while a corporate client migrates |
| Queue reroute | dispatch | send `SCAM-1` holds to `SCAM-2` while `SCAM-1` is over capacity |

Each overlay carries `adjustment_id`, description, rationale, owner, approval
reference, effective-from, effective-to, mandatory review date, declared scope
(family, rule ids, segments, channels, event types, client segments), composition
position, and the backtest that supported it. Six properties, each load-bearing:

1. **Overlay, not edit.** The rule set is never rewritten. A model validator and a
   regulator must see the authored rule and the policy overlay separately, and
   merging an overlay into the base "to keep things simple" is prohibited
   (00 §7.6).
2. **The unadjusted answer survives.** Every event records the counterfactual:
   which rules would have fired and what action would have resulted without the
   stack. "What would we have done without the overlay?" is asked at every monthly
   fraud forum.
3. **Overlays stack in a declared order.** A family sensitivity dial and a
   channel-scoped threshold multiplier can both apply to one event; composition
   order changes the answer and is part of the definition, not emergent from load
   order or insertion time.
4. **Identity and justification.** No anonymous overlays. The rationale must be
   readable by someone who was not in the room during the attack.
5. **Expiry is enforced, not advisory.** Maximum initial duration 90 days. An
   overlay reaching its effective-to lapses automatically, and the lapse is an
   outcome-affecting event recorded on every event assessed after it. Renewal is a
   fresh approval requiring a fresh backtest of the overlay's *current* effect, not
   a re-signature. A standing register lists every overlay, its age, its measured
   effect and its days to expiry. Overlays surviving three renewals are escalated
   for conversion into rule changes.
6. **Scope is declared.** An overlay states exactly which populations it touches.
   Applying it outside that scope is an error, not a silent no-op.

The approval asymmetry is why expiry matters. An overlay is live in under 2
minutes on the duty officer's authority plus one other named person; a rule change
takes 10 minutes and four-eyes from the family owner. The faster path exists
because attacks do not wait, and the enforced expiry exists because the faster
path is exactly the one that accumulates unexplained residue. Rules marked
**overlay-exempt** cannot have their thresholds modified by any overlay;
sanctions-adjacent and regulatory rules are exempt by default, because a
tightening dial applied in a hurry must not be able to loosen a control the Bank
is required to operate.

---

## 7. Outputs

**Synchronous response**, within the event type's latency ceiling: `action_code`;
`primary_reason_code`; client wording key and rendered text where the action is
client-visible; challenge type where `step_up`; hold reference and SLA deadline
where `hold_for_review`; `enrichment_degradation_code` and `degraded_mode_code`;
`rule_set_version` and `adjustment_stack_version`; and `event_id`, the join key
for everything afterwards. `fired_rule_ids` is **not** returned to the authorising
system — it goes to the decision record, the review agent's screen and the
analyst's tooling. A merchant acquirer does not learn which of the Bank's rules
stopped a card.

**Persisted**: the decision record of §5.15; per-minute firing aggregates by rule
and by overlay; obligations raised with their clocks; dispatch acknowledgements;
and the outcomes of §5.16 joined by `event_id`.

**Operational and analytical**: review queue items with the firing set rendered
for a human who has 90 seconds to decide; daily rule performance per rule per
version, with and without the overlay stack; the overlay register with measured
effect and days to expiry; dead-rule and overlap reports; and, on demand, a
regulatory inventory of every rule and overlay live between two dates with owner,
approver, effective dates and firing volume.

---

## 8. Non-functional requirements

| Requirement | Value |
|---|---|
| Latency, card authorisation | p99 ≤ 25 ms, p999 ≤ 45 ms, from receipt to response |
| Latency, channel events | p99 ≤ 40 ms |
| Latency, EFT and re-issue | p99 ≤ 400 ms |
| Sustained throughput | 3 500 events/second |
| Peak throughput | 12 000 events/second, sustained 4 hours |
| Monthly volume | ≈140.8 M events |
| Availability | 24/7/365. No maintenance window, no batch cut-over. |
| Rule change to live | ≤ 10 minutes, normal path, including approval |
| Emergency rule change to live | ≤ 4 minutes from approval, during a live attack |
| Overlay to live, and overlay withdrawal | ≤ 2 minutes from approval |
| Determinism | Identical event, enrichment values, rule set version and adjustment stack version ⇒ identical action, identical firing set, identical ordering |
| Determinism under concurrency | The answer must not depend on how many events are in flight, which worker handled it, or what that worker had just done |
| Cold start | The first event after a rule set or overlay swap is not materially slower than the ten-thousandth |
| Live swap | Rule set and overlay stack swap in place, no dropped events, no event assessed against a half-applied set or stack |
| Backtest throughput | 420 M events (90 days) against a candidate set within 90 minutes, in either overlay mode |
| Record durability | No decision record may be lost; loss is an incident |

Latency budget, card authorisation, p99: admission and normalisation 1.5 ms;
context, counterparty, device and merchant enrichment 8.0 ms, concurrent with the
model call; velocity attachment 2.0 ms; completeness verdict and hard blocks
0.5 ms; applicability and overlay resolution 1.0 ms; live and shadow evaluation
6.0 ms; resolution, dispatch preparation and wording 1.0 ms. Serial total 20.0 ms,
headroom 5.0 ms. Record emission is asynchronous and off the budget.

The 6 ms of rule evaluation covers ≈635 evaluations including shadow, with no
early exit — around 9 µs per rule at p99 — and the budget does not grow when the
rule set does. A set reaching 900 live rules must still fit, and resolving an
overlay stack of 40 entries must not be a per-rule cost.

**Degraded-mode behaviour** is a requirement, not a fallback. Each source has a
declared failure behaviour (§5.6), the composite mode is declared, the population
adjustment is declared in advance, and every declaration is recorded on every
event assessed while it was in force. "It degraded gracefully" is not an
acceptable answer to "what did it do".

---

## 9. Audit, evidence and explainability

### 9.1 What must be answerable, and to whom

| Question | Asked by | How long after |
|---|---|---|
| Why was my card declined? | Client, via the contact centre | Minutes to days |
| Why was this payment held, and by which rule? | Fraud Ops agent | Seconds |
| What was the client told? | Complaints, Ombud | Up to 3 years |
| Reproduce this decision exactly: rule set, overlay stack, thresholds, feature values, firings, action | Disputes, scheme representment | **Up to 540 days** |
| Which rules and overlays were live on 14 March, who owned and approved them? | Internal Audit, regulator | Up to 7 years |
| Prove rule `MS-0117` stopped affecting outcomes when you say it did | Internal Audit | Any time |
| Prove overlay `ADJ-0042` stopped applying on its expiry date | Internal Audit | Any time |
| What would we have decided without the overlay stack? | Fraud forum, Committee | Monthly |
| Show a rule's firing history across all its versions | Fraud Strategy | Daily |
| Was this beneficiary on the mule list at the time? | Disputes, law enforcement | Up to 7 years |
| When did the Bank become aware, for reporting-clock purposes? | Financial Crime Compliance | Up to 5 years |

### 9.2 Replay

Exact replay of one event requires, and must therefore have recorded: the raw
event; every enrichment value **as used**, including velocity values with their
watermarks and the model score or its recorded absence; `rule_set_version` and the
full definition of every rule in it, including retired ones;
`adjustment_stack_version` and the full definition of every overlay in it,
including expired ones, with composition order; every table version and cell read;
the degraded mode; and the precedence table version. A replay that reaches for any
current value has failed.

Replay must work for a rule that no longer exists and an overlay that has expired.
Retiring a rule or lapsing an overlay removes it from effect going forward; it must
not remove the ability to say what it did. 1 360 retired rules and the full overlay
history are retained for this reason, and both only grow.

### 9.3 Governance evidence

- **Four-eyes approval** on every rule create, change, promotion and retirement, and
  on every overlay create, renewal and withdrawal, with author, approver, timestamp,
  before-and-after definition, and the backtest that supported it. An emergency
  change or an overlay still requires two people; the fast path drops the waiting,
  not the second pair of eyes. Both are reviewed within 24 hours, with the review
  recorded against them.
- **Effective dating.** A rule's or overlay's effective-from is the authority for
  when it began affecting outcomes; the deployment record is the evidence that it
  did. A discrepancy — effective from 09:00, actually deployed 09:40 — must be
  detectable rather than assumed away.
- **Retirement and expiry evidence.** "Rule `AA-0093` was retired on 30 June" and
  "overlay `ADJ-0042` expired on 12 August" are provable by their absence from every
  event after that instant, demonstrable without scanning 140 million records.
- **The unadjusted counterfactual.** Because the overlay stack changes the answer of
  an approved artefact under a faster approval, the record of what the base rule set
  alone would have done is not an analytical nicety. It is the evidence separating
  what the authored policy said from what an overlay decided, and the first thing a
  regulator asks for once overlays are known to exist.
- **Shadow isolation.** Evidence that no shadow rule has ever changed an outcome. A
  structural claim, tested continuously (§10.4).

---

## 10. Acceptance criteria

1. A fraud analyst creates a rule, backtests it over 90 days, obtains four-eyes
   approval and sees it live, without an engineer and without a code deployment, in
   under 10 minutes.
2. An emergency threshold change reaches production in under 4 minutes from
   approval at peak load with no dropped events; an overlay reaches production, and
   is withdrawn again, in under 2 minutes.
3. Every assessed event has a decision record containing the complete live firing
   set with per-rule feature values and base-versus-overlay basis, at 3 500
   events/second sustained and 12 000 at peak, with no sampling.
4. No shadow rule has ever changed an `action_code`, reason set, queue routing or
   client notification. Demonstrated continuously, not asserted.
5. Any event from the last 540 days replays to the identical `action_code`,
   identical `fired_rule_ids` in identical order, identical `fired_on_overlay_ids`
   and identical reason set, using only recorded artefacts.
6. Running the live rule set and live overlay stack in backtest mode over a sampled
   day reproduces production's action and firing set for 100% of events. Checked
   daily; any mismatch is an incident.
7. The same backtest runs with the overlay stack disabled and reports the base rule
   set's own performance, with the difference attributable per overlay.
8. An overlay reaching its effective-to lapses automatically within one assessment,
   and every event thereafter records that it no longer applied. No overlay can be
   in force without a review date, and none can exceed 90 days without an explicit
   renewal carrying a fresh backtest.
9. A rule retired on a date can be shown to have had no effect on any event after
   that instant, and can still be replayed against events before it.
10. Two rules with the same action and severity resolve identically on every replay
    of the same event, on every worker, at every load level.
11. A regulator receives, for any past date range, every rule and every overlay live
    in the period, with owner, approver, effective dates, rationale, firing volume
    and confirmed-fraud catch.
12. When the velocity store is unavailable, behaviour matches the declared
    degraded-mode specification and each affected event records which rules were
    suspended and which compensating rules activated.
13. When the model score does not arrive, the action follows the declared behaviour
    and the replay reproduces it exactly, including the absence.
14. A rule exceeding its declared maximum fire rate self-demotes within one 5-minute
    window, with correct attribution where an overlay was in force on it.
15. The latency budget holds when the live set grows from 521 to 900 rules and the
    overlay stack from 12 to 40 entries.
16. A Fraud Strategy analyst reads the generated description of a rule, and of the
    overlay stack in force, and confirms both match what was authorised, without
    reading code.

---

## 11. Change scenarios

1. **A live attack at 19:40 on a Friday.** A card-testing pattern against one
   acquirer, 400 attempts a minute. A new rule authored, sanity-checked, approved by
   two people and live in under 4 minutes — and removable just as fast.
2. **A sensitivity dial during the same attack.** Rather than author rules, the duty
   officer runs the card-fraud family at 1.4× sensitivity for 72 hours, scoped to
   card-not-present events on that acquirer's country. Live in 2 minutes, expires on
   its own, and every event under it records what would have happened without it.
3. **An overlay that should have expired.** Audit finds a mule/scam threshold
   multiplier applied 14 months ago during a scam wave, renewed twice by
   re-signature without a fresh backtest, now absorbed into the baseline against
   which four subsequent rule changes were measured. Those four must be re-costed
   against the base rule set.
4. **A bulk threshold move.** After a repricing, the amount thresholds on 40
   card-fraud rules move at once, in one approval, with one backtest covering the
   aggregate effect rather than 40 separate ones.
5. **A new event type.** Merchant QR payments launch: new message shape, 31 new
   fields, 9 of them referenced by rules that must apply to the new type and the
   existing instant-payment type at once.
6. **A scheme field addition.** The scheme adds a merchant-initiated transaction
   indicator. 30 rules want it; the field is absent on every historical event, so
   every backtest involving it must handle a feature that did not exist before a
   date.
7. **The model becomes two models.** A card model and a scam model, each with its own
   score, deadline and cut-off, each absent independently. Rules must reference
   either or both, degraded mode must handle four combinations instead of two, and
   the existing cut-off overlay must be split or retired.
8. **Mandatory scam reimbursement.** A regulator introduces a reimbursement duty for
   victims of authorised push payment scams, with a defined liability split and
   standard of client caution. A new outcome category appears in §5.16 and the
   evidence requirement is applied to events already assessed under the old regime.
9. **The watchlist grows 5×.** A consortium adds 9 M entries, taking the mule list to
   11 M, and refresh moves from hourly to every 5 minutes. Membership-as-at-an-instant
   must still be answerable for seven years.
10. **A window that does not exist.** An analyst wants "distinct counterparties in the
    last 6 hours", not one of the six defined windows. Adding it means 28 more
    aggregates on every event and a backfill decision for history.
11. **Retire 180 dead rules.** Each proved dead, retired with evidence, removed from
    the applicable population, and still fully replayable against events it fired on
    years ago.
12. **A priority collision.** A card-fraud severity-4 rule and an AML-adjacent
    severity-4 rule fire together on 900 events a day asking for `decline` and
    `hold_for_review`. The owners disagree. The resolution must be expressible
    without either owner editing the other's rules.
13. **A challenge type is withdrawn.** SMS one-time passwords are deprecated for
    high-value step-up. 60 rules must be repointed, and the challenge matrix must
    express "not permitted" distinctly from "not configured".
14. **An 18-month backtest.** Annual model validation runs over 18 months rather than
    90 days, across four rule set generations and eleven overlay stacks, stating
    which baseline and which overlay mode were used.
15. **A segment redefinition.** The youth segment's upper age moves from 24 to 26,
    changing which rules apply to 1.9 M clients overnight. Events before the change
    replay against the old definition.
16. **A country is sanctioned overnight.** The country risk table changes outside its
    monthly cadence with same-day effect, and four rules' behaviour changes without
    any rule being edited.
17. **A court order.** A named client's transactions must all be held for 90 days
    regardless of the rule set, then automatically revert.
18. **A cross-domain feature.** A first-party-fraud rule must reference whether the
    client is in debt review — credit-side state published elsewhere in the Bank,
    refreshed daily, not in this engine's enrichment set.

---

## 12. Out of scope

- Computation of velocity aggregates. They arrive precomputed; this specification
  governs their consumption, staleness semantics and recording.
- Development, training and monitoring of the external fraud model.
- The review queue application — screen design, agent workflow, case management.
  Routing, SLA tier and the evidence the screen is fed are in scope.
- Client communications delivery: SMS gateways, push infrastructure, voice.
  Obligations are raised here; delivery is elsewhere.
- Chargeback and dispute processing. The decision record is an input to it.
- Filing of suspicious activity reports. The trigger and clock start are here.
- Sanctions list curation and name-matching algorithms. Membership is consumed.
- Card and account lifecycle actions — the actual freezing, blocking and re-issue.
- Credit decisioning of any kind. See projects 02, 03 and 07.

---

## 13. Questions the implementation must answer

1. **Is a flat rule set a core component kind?** It is not a decision table: it
   returns a set rather than a row, every member is evaluated rather than the first
   match, each member carries governance metadata that is not part of the answer,
   members are individually versioned, dated and retired, and the answer is computed
   by a separate precedence over the firing set. If it is a core component, what is
   its interface? If it is not, what is it assembled from, and does that assembly
   survive 900 rules?
2. **How is "every rule that fired" produced at 3 500 events per second**, with
   per-rule feature values and base-versus-overlay basis, without sampling and
   without a separate slow path? If the answer is "a debug mode", the answer is
   wrong, because this is the product's primary output.
3. **How is the applicable population per event expressed and recorded** when it
   varies by event type, segment, effective dates, degraded mode and overlay scope,
   and when storing 635 rule identifiers on every one of 140 million monthly records
   is not viable?
4. **What is the unit of change** — a rule, a threshold, an overlay, a rule set
   version? What can an analyst change without engineering, what needs an overlay,
   and how is that boundary enforced rather than documented? Where does "tuning a
   threshold" end and "changing a rule's shape" begin, and is that distinction
   visible in the artefact or only in a review process?
5. **What, structurally, is an adjustment?** It is a parameter change (values move
   without new logic), a structural concern (composition order is real logic) and an
   audit artefact (the stack in force on a date is part of the decision record) at
   once. Is it the same mechanism as a parameter, a second mechanism layered over
   it, or something the rule set must be aware of? How does a rule read an overlaid
   threshold and its base value in the same evaluation without doubling the cost,
   and how is a stack of 40 overlays resolved once per event rather than once per
   rule?
6. **How does a rule set version swap happen live**, under load, with no event
   assessed against a partially applied set or a partially applied overlay stack,
   and with both versions recorded on every event either side of the swap?
7. **How is action precedence expressed** — critical override, allow-listing
   suppression, hard-block floors, overlay-driven escalation and a four-level
   deterministic tie-break — as something an analyst can read and a regulator can be
   shown, rather than as ordering logic?
8. **How does a rule declare its behaviour when an input is stale or absent**, and
   how is that declaration kept honest when its author is thinking about fraud and
   not about watermarks?
9. **What guarantees shadow isolation structurally?** Not by convention or review
   discipline — by construction, such that a shadow rule influencing an outcome is
   impossible rather than caught.
10. **How is the backtest proved equivalent to production**, given that the two run
    at different scales, on different shapes of work, over recorded rather than live
    enrichment? What exactly is the shared artefact, and what is the test that they
    have not drifted — in both overlay modes?
11. **How is a 2 M-row watchlist refreshed 24 times a day made replayable for seven
    years**, so that membership at a microsecond in the past is answerable cheaply
    and exactly?
12. **How are the sixteen tables versioned and attributed** when they have five
    owners, cadences from quarterly to every 15 minutes, and two are hot enough that
    a version pointer per event is itself a cost?
13. **How does a rule reference a feature that did not exist before a date**, so that
    a backtest over 18 months across four rule set generations is honest about what
    was computable when?
14. **What does the generated description look like** — of a rule, and of the overlay
    stack in force — such that its author recognises it, an operations agent can act
    on it in 90 seconds, and a regulator can accept it as an inventory entry, from
    one definition?
15. **What is the cost of adding the 522nd rule**, and is it the same as the cost of
    the 900th? If evaluation, recording or deployment cost grows worse than linearly
    in rule count, the rule set will be pruned for the wrong reasons.
16. **Where does this project's vocabulary belong?** Fifteen names are declared
    locally in §4.4 because `core` does not publish them. Is `rule_id` a fraud
    concept or a framework concept, and does the answer change when project 08
    arrives with its own flat rule set and its own overlays?
