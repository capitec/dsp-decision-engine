# Fraud-detection and AML engines vs decider2

Evidence grading used throughout: **[FP]** = first-party verified (fetched from the vendor's
own domain and read directly), **[SEC]** = secondary (third-party host, a vendor doc
republished by a licensee, or a search-engine snippet that could not be opened directly),
**[UNREACH]** = attempted and failed, with the exact failure mode stated. Grades are attached
inline or in the closing bracket of a paragraph; where a whole sub-section shares one grade it
is stated once in the sub-section's opening sentence. decider2 file:line citations were
spot-checked against the current working tree (2026-09-21); the repo has uncommitted changes
on `feature/decider-v2` touching `compile/driver.py`, `params.py`, `runtime/{invoke,modes,serve}.py`,
`tables/{codegen,schema}.py`, `trees/{codegen,schema}.py` — every citation below was re-read
against that live tree, not against the last commit.

---

## 0. decider2's fraud baseline — what it actually is today

Three layers have to be kept apart or every comparison below is dishonest.

**Implemented code** (`decider2/src/decider2/`, ~9 600 lines): the polars↔numba boundary
(`boundary/`), content-addressed codegen and compile (`compile/`), the three execution
modes (`runtime/modes.py:1-33` — `interpreted`/`stepped`/`fused`), the single-record and
batch entry points (`runtime/invoke.py`), a params surface that is a thin adapter over
pydantic `Field` (`params.py:1-33`), a serving handle (`runtime/serve.py`) and an HTTP
surface of exactly eight routes (`serving/dispatch.py:137-146`: `/ping`, `/invocations`,
`GET|POST /params`, `/params/schema`, `/params/preview`, `/rollback`, `/health`).
Decision **tables** and **trees** are ported from decider 1 (`tables/schema.py:551-590`,
`trees/schema.py`); the table's hit policy is *first match wins with a `default` row*
(`tables/schema.py:560-565`).

**Specified but unbuilt.** `runtime/serve.py:12-19` states it plainly: "decider2 today has
no document-driven way to change a pipeline's *structure* (doc 00-BUILD.md Layer 4,
`interiors/`, is 'BLOCKED' — nothing like `ruleset()`/`decision_table()` exists yet). So
every `.stage(doc)` this module can ever actually perform is a **values** change … a params
document can retune a threshold, never add or remove a rule." `docs/00-BUILD.md:246` lists
O15 as the blocker: "`ruleset.py` is unwritable". Layer 3 (`graph/`) is
"SPECIFIED BUT UNVALIDATED … **has no empirical support at all**"
(`docs/00-BUILD.md:160-163`).

**Designed on paper.** `example_projects/01-transaction-fraud-interdiction.md` is a 1 179-line
fraud spec — 521 live rules, 114 shadow, 1 360 retired, 3 500 events/s sustained and 12 000
peak, p99 ≤ 25 ms on card authorisation (`:884-900`). It is explicitly fictional (`:3-7`).
`example_projects/examples/01-transaction-fraud-interdiction/` is a *mock implementation* of it
written by an independent agent for the cold-read study; the symbols it imports
(`decider2.ruleset`, `stable_blocks`, `decider2.governance`, `temporal_table`, `fan`) do not
exist in the package. Its `FRAMEWORK-DEMANDS.md` (42 numbered demands, grouped A–G) is the
best existing statement of the gap.

Four numbers from `docs/EXPERIMENTS.md` frame every latency comparison below:

- **Kernel: ~1.38 µs** for 400-in/633-out, njit call with a pooled output buffer
  (`EXPERIMENTS.md:713-717`).
- **Framework overhead, as the docs' own conventions imply it: 971.4 µs p50 = 4.86% of a
  20 ms budget**, of which 92% is two per-field Python loops (marshal 220 µs, readback 673 µs);
  rewritten bulk it is **145.2 µs = 0.73%** (`EXPERIMENTS.md:678-770`).
- **Tail, single thread, n=30 000: p99 1 114.2 µs (5.57%), p99.9 1 723.6 µs (8.62%), max
  3 840.5 µs (19.20%)** (`EXPERIMENTS.md:1010-1018`). Root cause of the tail is OS scheduler
  preemption, not GC (`EXPERIMENTS.md:1164`, `docs/00-BUILD.md:254-259`: `ru_nivcsw` nonzero in ~1%
  of calls but 100% of the top-0.1% slowest; **pinning to one core makes it worse**, p99.9 +63%).
- **Concurrency: `nogil=True` holds p99 at ~883 µs (4.4%) flat from 1 to 16 threads;
  `nogil=False` reaches p99 254 052 µs = 1270% of the 20 ms budget at 16 threads**, a GIL
  convoy (`EXPERIMENTS.md:1063-1070`). `nogil` is authored per step and **off by default**
  (`docs/00-BUILD.md:103-117`), so serving surfaces which kernels hold the GIL
  (`runtime/serve.py` `gil_report()`) rather than deciding.

And two rule-count numbers, which are the ones that decide whether decider2 can host a
521-rule fraud set at all (`EXPERIMENTS.md:381-390`):

| rules | compile `first_match` | compile `all` | exec @100k rows, `first_match` / `all` |
|---|---|---|---|
| 10 | 1.37 s | 1.43 s | 1.8 ms / 10.1 ms |
| 40 | 10.86 s | 9.44 s | 1.8 ms / 52.5 ms |
| 100 | 56.56 s | 48.08 s | 1.8 ms / 126.7 ms |

Compile is **∝ lines^1.4**, local exponent 1.96 between 60 and 100 rules; the lever is
splitting into compilation units, not shrinking. `all` — the policy a fraud rule set needs —
costs **1.27 µs/row at 100 rules**, i.e. roughly linear in rule count: extrapolated to 635
rules that is ~8 µs/row, still trivial against a 25 ms per-event budget but ~5× the whole
rest of the kernel. And a *disabled* rule costs full compile time, so `enabled` must mean
"not emitted" (`EXPERIMENTS.md:417-425`).

---

## 1. Per-engine blocks

### 1.1 Feedzai — Railgun, ARMS, RIFF (deepest evidence in the set)

**Feature computation.** Railgun is a distributed streaming engine whose entire purpose is exact
per-event sliding-window aggregation. The paper is *Railgun: managing large streaming windows
under MAD requirements*, Gomes, Oliveirinha, Cardoso, Bizarro (all Feedzai), PVLDB 14(12):
3069–3082, 2021, CC BY-NC-ND
([PDF](https://research.feedzai.com/oshuzuxa/2022/08/Gomes_Railgun_VLDB2021.pdf)) [FP]. MAD =
"Msec-level latencies at high percentiles (<250ms @ 99.9%); Accurate sliding window
aggregations event-by-event; Distributed, scalable and fault-tolerant" (§1).

The aggregation language is published in full (§3.4, Figure 4):

```
SELECT AggExpression FROM streamName WHERE filterExpression
GROUP BY fields OVER WindowExpression
Aggregation      ::= count | sum | avg | stdDev | max | min | last | prev | countDistinct
WindowExpression ::= TimeWindowExpr | TimeWindowExpr delayed by offset
TimeWindowExpr   ::= sliding windowSize | tumbling windowSize | infinite
```

So "count of transactions on this card in the last 10 minutes" is
`SELECT COUNT(*) FROM payments GROUP BY cardId OVER sliding 10 minutes`. Filters are Apache
Commons JEXL. Hopping windows are refused on principle — "we see them as an approximation of
our sliding windows"; the engineering blog's title is *Are Streaming Engines lying to you
too?* ("Streaming engines claim they support sliding windows. In truth what they really have
are **stepping windows**"). No stream joins: "we implement joins … prior to the streaming
engine, in an enrichment stage."

The mechanism is the *event reservoir* (§4.1.1): per task processor, a tiny in-memory head and
tail per window plus compressed, immutable, append-only chunk files on local disk, with an
in-memory timestamp→file index for random reads. Two iterators per window, shared between
aligned windows. The consequence, verbatim: "**windows of years are equivalent to windows of
seconds – in performance, accuracy, and memory consumption**." Aggregation state is RocksDB;
`max`/`min` keep a deque, `stdDev` uses Welford, `countDistinct` uses a second column family.
Out-of-order events are accepted only until chunk closure, then "either discarded, or have
their timestamp rewritten to the first timestamp of the chunk" — and critically, "we never
delay the answer and computation of event metrics, as opposed to systems such as Spark
Streaming or Flink." Exactly-once = Kafka at-least-once plus in-memory-chunk dedup by event id.

**Rule authoring.** Product docs are private (`docs.feedzai.com` 301s to an Atlassian
Confluence login) [UNREACH], so the rule semantics come from Feedzai's own papers. ARMS
([PDF](https://research.feedzai.com/oshuzuxa/2022/08/Aparicio_ARMS_KDD2020.pdf),
arXiv:2002.06075) [FP] states the architecture: "a machine learning (ML) model followed by a
rule-based system. The model scores the transaction. The rule-based system uses the score and
triggers of manually defined rules to decide an action (i.e., accept, alert, or decline)."
A rule is `(condition, action, priority, active)`; **priority = −1 is deactivation** and the
priority value itself maps to the action class; the hit policy is "**the action of the highest
priority rule triggered for the transaction that is active**". Lists are rules: `Bu` blacklist
*updater* rules and `Bc` blacklist *checker* rules, with membership stored as time intervals
and aged by closing an interval.

The two real production rule sets ARMS reports are the best public data on fraud rule-set
shape: client D1 has **198 rules** (30 accept / 89 alert / 79 decline) across 9 priority
levels, default action accept; client D2 grew **77 → 90 rules** over seven months, 3 blacklist
checkers and all 36 alert&decline rules are blacklist updaters. RIFF
([PDF](https://research.feedzai.com/oshuzuxa/2025/09/riff_inducing_rules_for_fraud_detection_from_decision_trees.pdf)) [FP]
adds the shape: "typically composed of **tens to hundreds of simple rules** … usually a
**conjunction of a small set of logical conditions** … evolved with time by tuning specific
thresholds", under "very low FPR values, typically under 2%". ARMS itself is a rule-set
optimiser that replays a historical trigger matrix and can hold recall/FPR with "≈ 50% in one
case, and ≈ 20% in the other" of the original rules. First-party governance claim: "Use
**challenger testing and backtesting to validate changes in real time**"
([RiskOps](https://www.feedzai.com/riskops/)) [FP].

**ML integration and explainability.** [feedzai-openml](https://github.com/feedzai/feedzai-openml)
(Java ServiceLoader API) [FP] is the documented third-party-model seam; `feedzai-openml-python`
exists; none of Feedzai's 76 public repos contains a rules engine or Railgun. Whitebox
Explanations™ gives "detailed information on the suspicious activity and why a given risk score
was computed" ([2019 release](https://www.feedzai.com/pressrelease/feedzai-unveils-expanded-aml-capabilities-underpinned-by-advanced-machine-learning-and-explainability/)) [FP];
`timeshap` (203★) is the open-source RNN explainer behind it.

**Latency.** 250 ms @ p99.9, contractual, **measured externally by the injector**,
coordinated-omission-corrected (§5) — and restated in 2026 ("250 ms at the 99.9th",
[probabilistic thinning paper](https://research.feedzai.com/oshuzuxa/2026/06/2606.16981v1.pdf)) [FP].
Throughput: 25k ev/s comfortably per node; 750k ev/s at 35 nodes is where degradation starts;
"1 million ev/sec … 50 nodes … 20 thousand ev/sec" each (§5.3). Against Flink at 500 ev/s on
one node: "with hops of 10s or less, Flink is unable to keep with a 500 ev/s throughput", and
meeting 250 ms @ p99.9 in Flink "need hops of at least 1 minute, which would severely
compromise accuracy". Named bottleneck: GC — "at 25 thousand ev/sec, we are creating objects
at a rate of about 5GB/sec". **No fail-open, backpressure, load-shedding or request timeout is
described anywhere in the paper** — grepped for it; the only shedding mention is attributed
to other systems. The 2026 paper does discuss backpressure, and its design decouples the two:
"every event is scored, but durable state updates are selectively triggered", up to "90% of
events … excluded from the persistence path". Maturity caveat, their words: "We benchmarked an
initial **prototype** of Railgun."

**vs decider2.** This is the engine decider2's fraud spec quietly depends on. The 168
precomputed aggregates at
`example_projects/01-transaction-fraud-interdiction.md:126-134` (7 keys × 6 windows × 4
statistics) are exactly Railgun's output, and `:1097-1099` puts computing them out of scope.
Three things transfer directly. (1) **Railgun's grammar is the schema decider2's velocity input
contract is missing.** decider2 records a value, a watermark and fresh/stale/absent
(`:246-260`); Railgun would additionally pin the *definition* — statistic, key, window kind
(sliding vs tumbling vs infinite), delay offset. Without that, `:1066-1069`'s "an analyst wants
distinct counterparties in the last 6 hours" has no way to state what it is asking for.
(2) **decider2's staleness model is better than Railgun's and should be marketed as such**:
Railgun rewrites or discards a late event's timestamp and never tells the consumer; decider2
carries the watermark into the decision record and lets each rule declare `evaluate_false` /
`last_good` / `suppress` per feature (`:264-266`). (3) **ARMS's `(condition, action, priority,
active)` with highest-priority-wins is a strictly simpler resolver than decider2's**, which has
a hard-block floor, a critical override, allow-list suppression and a four-level tie-break
(`:463-500`). decider2's is richer because a bank needs it, but ARMS is evidence that
priority-encodes-action is enough for two real card portfolios — worth knowing before building
the five-mechanism version. On latency, Feedzai's 250 ms @ p99.9 is *ten times* decider2's own
25 ms p99 target (`:888`), because Railgun's number is end-to-end across Kafka and decider2's
is in-process; decider2's measured p99.9 of 1 723.6 µs (`docs/EXPERIMENTS.md:1010-1018`) is not
comparable and should never be quoted against it. Finally, Bizarro's methodology note is
directly applicable to decider2's own docs: "**systems cannot reliably measure their own
latency** … External latency measurements are essential"
([blog](https://www.feedzai.com/blog/latency-in-machine-learning-what-fraud-prevention-leaders-need-to-know/)) [FP]
— every N-series figure in `EXPERIMENTS.md` is self-measured in-process.

### 1.2 Featurespace ARIC (Visa)

**Feature computation.** Not windowed aggregates at all. US12118559B2 (*Training a machine
learning system for transaction data processing*, Featurespace Ltd, Wong/Sutton/Perez/
Barns-Graham, granted 2024-10-15,
[patent](https://patents.google.com/patent/US12118559B2/en)) [FP] discloses a **per-entity
fixed-size learned state vector**: "the state vector has a **fixed size** … in the range of
**4 to 128 elements**"; "the state data used by the recurrent neural network architecture 620
is **entity dependent**"; it "must **summarise the past behaviour of any entity up to any point
in time within the form of this fixed size vector**", with a "forget gate using time-decay
functions" and the state "cached in memory or otherwise stored until a next transaction". So
the behavioural profile is O(1) per entity and its "window" is an implicit learned decay —
the opposite trade from Railgun's exact-but-O(events) reservoir.

**Rule authoring.** AMDL ("ARIC Model Definition Language") is confirmed to exist and to be
"at its core"
([ADBN page](https://www.featurespace.com/automated-deep-behavioral-networks)) [FP] — and **no
syntax, keyword list, example or reference is published anywhere**. The two pages that once
described it were deleted or 301'd in the post-Visa site rebuild, as was the ARIC Risk Hub
brochure PDF (now returns a Next.js shell) [UNREACH]. None of the 41 patent records under assignee
Featurespace describes AMDL. The only first-party statement about rules is in EP4610915A1: "a
**rules system may apply decisioning logic upon the risk indicator**. For example, the rules
system may trigger a decline and alert if a risk score exceeds a predetermined operational
threshold" — score-into-rule, same as everyone else. **There is no first-party support for the
idea that ARIC auto-generates rules from the model**; "Adaptive Rules" is unverified.

**Latency.** Only from the patent, and only as averages: "a transaction processing pipeline
typically needs to be completed within **one or two hundred milliseconds**"; "the time allotted
for the machine learning system … such as **10 ms**"; "may process requests within **7-12 ms**
and be able to manage **1000-2000 requests a second**". Scale claims: "500m consumers
protected", "50.4bn events annually" ([platform](https://www.featurespace.com/aric-risk-hub)) [FP]
versus Visa's "over 100 billion payment events each year"
([Visa, 19 Dec 2024](https://investor.visa.com/news/news-details/2024/Visa-Completes-Acquisition-of-Featurespace/default.aspx),
terms not disclosed) [FP] — the two conflict.

**vs decider2.** Featurespace is the one engine in this set that is *not* an alternative to
decider2 — it is an alternative to the external model at
`example_projects/01-transaction-fraud-interdiction.md:270-289`. That matters for two design
decisions. First, the spec's model contract (score 0–1000, model id, version, three
contributing-factor codes, 8 ms deadline, declared absence behaviour) is a good fit for a
7–12 ms recurrent scorer, and change scenario 7 (`:1055-1058`, "the model becomes two models")
is exactly what a card model plus a scam model looks like. Second, Featurespace is the
cautionary case for decider2's own documentation posture: a proprietary DSL with no published
grammar is unreviewable from outside, and `docs/04-observability-and-governance.md:277-333`
(the reviewable-artefact risk, tested and failed) is decider2's version of the same problem.
decider2's `docs/08-configuration-and-lifecycle.md:265-315` decision — no expression strings in
config, derived features are registered typed steps — is the right answer precisely because it
refuses to create an AMDL.

### 1.3 NICE Actimize — Policy Manager, AIS profiles, Platform Lists

**Rule authoring, well evidenced from NICE's own material.** Two rule species, and NICE
teaches the distinction as a course objective ("Comparing scoring rules with Policy Manager
rules"): **scoring rules** authored in Visual Modeler's Analytics Authoring Environment feed a
score inside AIS; **Policy Manager rules** authored in ActOne decide the action. The Policy
Manager object model, from the public course catalogue
([2023 catalog PDF](https://www.niceactimize.com/Lists/Brochures/2023_actimize_university_course_catalog.pdf),
"Policy Manager for Advanced Analysts and Strategy Managers") [FP]: create rule → use the
**expression builder** → **set an action** → **activate/deactivate** → **set/remove priority**
→ **activate a draft** → **deploy** → **monitor activated policies**. Structurally
`(expression, action, priority, active, draft/deployed)` — the same tuple as ARMS, except NICE
does not publish its hit-policy semantics.

The distinctive concept is the **classifier**: "anything one can use to create a
lower-cardinality description of underlying transactions for the purposes of aggregation",
authored **as SQL**, because "the cardinality of amounts is high"
([Policy Manager blog](https://www.niceactimize.com/blog/fraud-prevention-starters-guide-to-mitigate-fraud-using-policy-manager)) [FP].
Windowed behaviour lives elsewhere, in **AIS profiles** — created, read, browsed and
"using profile information in scoring rules" is a 4-day course topic requiring "previous
programming or application development experience". **No profile syntax, window semantics,
window length or freshness guarantee is published.**

**Rule discovery and backtesting are manual.** The blog documents finding rules with Oracle's
`CUBE()` function as a "'budget' HDT", pruning by "complexity" (count of non-null classifiers)
and dominance, then reading the hit/fraud counts straight off the query — and its author's own
verdict: "**If all this strikes you as incredibly manual, that's because it is** … Leaning on
manual processes will be a liability." **Platform List ageing is likewise not a product
feature**: "To let items age out, modify your HAVING clause as desired." The nearest published
simulation mechanism is an AIS server run-mode switch ("Validation, Loading, Tuning, Full
Mode", IFM-X course). Business users get a deliberately narrower surface: "Managing the
configuration of business logic: thresholds and score scales" and "Managing rule related lists,
such as inclusion/exclusion lists" (ActOne for Business Users). **No latency or TPS figure is
published anywhere on niceactimize.com** — all three course catalogues were grepped [UNREACH].

**vs decider2.** Three lessons. (1) **The two-species split is decider2's split.** Actimize's
scoring-rules-in-the-model-tier versus policy-rules-in-the-action-tier maps onto decider2's
record tier of `@step` functions versus the flat rule set that decides the action — and
Actimize's evidence is that the *governance* split (engineer tool vs strategy-manager tool)
follows the technical one. decider2 asserts the same boundary at
`docs/08-configuration-and-lifecycle.md:104-128` (three change classes) but has not built the
strategy-manager side. (2) **The classifier is a better primitive than decider2 has.** Band a
high-cardinality field into a low-cardinality categorical, then let rules and reporting both
use the band. decider2's fraud spec does this everywhere — merchant category risk bands, device
reputation bands, beneficiary age buckets, IP-distance bands (`:774-800`) — but as sixteen
separate tables rather than one named concept. A `classifier` kind in `tables/` would collapse
them. (3) **Actimize's manual-SQL rule discovery is the thing decider2's backtest mode
(`:621-669`) already beats on paper** — and the gap is that decider2's version does not exist
in code either. Both are currently "a data team with a warehouse".

### 1.4 FICO Falcon — the only engine that gives rule authors their own windowed state

Everything on fico.com is unreachable (bot interstitial, JS shells, `investors.fico.com`
timeout) [UNREACH], so the language evidence is FICO product documentation **republished by a licensee**:
the Falcon Fraud Manager Expert help set at `https://fraud.sia.eu/FalconRmaHelp/` (SIA/Nexi) —
verbatim FICO docs, but not on a FICO domain, so [SEC].

Architecture split: the **Falcon Scoring Server** produces the score; **Falcon Expert**, a
"Rule Maintenance Application (RMA)", applies rules to the already-scored transaction. The
language is **SRL (Structured Rule Language)**, case-sensitive. An actual rule:

```
if (CRTRAN25.authPostFlag = "A" and ffmFrdCard.Score > 800 and
    CRTRAN25.transactionAmount > 150 and (CRTRAN25.mcc = ("5310" or
     "5311" or "5331" or "5732" or "5733")))
then SendAuthAdvice(DECLINE);
```

Note what this shows that no SaaS engine does: `if … then …;` with real `then{}`/`else{}`
blocks and nesting; `=` as equality; field references namespaced by data-feed object
(`CRTRAN25.`, `ffmFrdCard.`); set-membership sugar with `or` in the value position; and an
action that is a **function call carrying a typed decision**, `SendAuthAdvice(DECLINE)` with
codes `approve | approveWithId | decline | pickUpCard | refer`, plus `TriggerCase()` and
`ForceCase()`. Rules sit in **rulesets** with a full check-out / edit / submit-for-approval /
approve-or-reject / validate / audit-log lifecycle. "**Rules are executed in the order in which
they are listed in a ruleset**", each executing its actions when matched — ordered, with no
documented short-circuit.

**The capability nothing else has:** community posts describe **UDVs** (user-defined variables,
"track events over periods of time") and **UDPs** (user-defined profiles, "store values and
track behavior over time") authored inside Falcon Expert [SEC, community forum]. That is a rule
author declaring and maintaining her own time-windowed aggregate. Underneath, FICO's published
profile design is recursive rather than retrospective: "Each profile is a continuous learning
cognitive 'mini-model'"; profiles "summarize each customer's banking transaction history into
behavioral analytic features using **recursive analytic algorithms**, making ultra-low latency,
real-time analytics possible"; plus **B-LISTS** (behaviour-sorted lists) giving "a real-time
ranking of features associated with each customer's most frequent behaviors" [FP, fico.com blog
snippets]. Score scale 1–999. Consortium: "More than 10,000 financial institutions" (9,000 on
another FICO page — the figure has drifted), "more than 2.6 billion payment cards". **No
FICO-published millisecond or TPS figure exists**; the strongest is "millisecond response
times" with no number [UNREACH]. Patent titles confirm the architecture (US 11,636,485
"Efficient parallelized computation of global behavior profiles in real-time transaction
scoring systems"; US 11,481,777 / 11,023,894 / 11,875,355 "Fast access vectors in real-time
behavioral profiling…"); `patents.google.com` returned 503 on every attempt and the USPTO PDFs
are scanned images with no text layer [UNREACH].

**vs decider2.** Falcon is the closest thing in the field to what decider2 *is*: a
compiled-ish, statement-oriented rule language over a bank's own message schema, with an
approval workflow around the ruleset. Three specific transfers. (1) **Ordered execution with no
short-circuit is decider2's fraud spec exactly** — "There is no early exit. Every applicable
rule is evaluated even after a `decline`-action rule has fired"
(`:398-401`) — which is evidence the design is conventional rather than exotic, and worth citing
in the spec. (2) **Namespaced field references beat decider2's flat name space** for 210 fields
across 12 event types. decider2 wires steps by matching parameter names in one flat scope
(`docs/02-architecture.md:575-580`, "names are distinct within a scope"); Falcon's
`CRTRAN25.transactionAmount` is how you keep a card-authorisation field and a login field of
the same name apart. FRAMEWORK-DEMANDS #4 asks for exactly this and calls it a variant schema.
(3) **An action as a typed function call, not a bare code.** decider2's spec has `action_code`
plus a queue, an SLA tier and a challenge type scattered as separate rule attributes
(`:704-728`); `SendAuthAdvice(DECLINE)` / `TriggerCase(caseLevel)` bundles the action with its
parameters, which is the shape that makes validation ("`hold_for_review` carries a queue and an
SLA tier", `ruleset/__init__.py` `validate_documents`) a type check rather than a lint rule.
The one place decider2 should *not* follow Falcon is UDVs: letting a rule author declare a
windowed aggregate inside the rule is exactly the coupling
`docs/08-configuration-and-lifecycle.md:265-315` removed when it deleted `_ComputedFeature`.

### 1.5 DataVisor — Rules/Features/Workflows, dEdge, Vera (thin evidence — real docs are login-walled)

**Evidentiary posture, stated up front.** DataVisor's actual product documentation lives on
Document360 behind a login: both `d360_llms-full.txt` and `d360_llms.txt` fetched for this
survey resolved to the Document360 sign-in page's HTML, not content — the same situation this
report already records for NICE Actimize (`docs.feedzai.com`/course-catalog-only) and Unit21
(`docs.unit21.ai` login-walled). What follows is therefore built from `www.datavisor.com`
marketing/product pages, one case study, and a sitemap — no rule-syntax grammar, no hit-policy
statement, and no API reference were reachable. Everything below is graded accordingly, and the
gaps are named rather than papered over.

**(a) Feature computation / velocity / profiles.** The platform's own framing is a single
"strategy layer": "DataVisor gives teams a common environment for the logic that detects risk
and the actions that follow. Strategies can draw on real-time features, model scores, graph
intelligence, and device signals without managing separate control planes" **[FP]**
([Rules, Features & Workflow Automations](https://www.datavisor.com/platform/rules-features-workflows)).
Feature reuse is named as a pillar: "Create features once and put them to work across
strategies... so teams can apply consistent risk logic without rebuilding the same intelligence
in multiple systems" **[FP]**, and the automatable-policy categories are listed verbatim:
"Velocity Controls · Transaction Monitoring · Account Takeover · Payment Fraud · Fraud Rings ·
Investigation Routing · Evolving Policies" **[FP]** — confirming velocity is a first-class
category, but with zero published window syntax, statistic vocabulary, or freshness/staleness
semantics (contrast Feedzai's fully published Railgun grammar in §1.1, or Actimize's AIS
profiles which are at least *named* as a distinct authored object in §1.3). The one mechanism
described in any concrete terms is device/behavioral: dEdge computes "100+ device signals
across Android, iOS, and web" **[FP]**
([dEdge](https://www.datavisor.com/intelligence-detection/device-behavioral-intelligence---dedge)),
spanning device/browser attributes (user agent, screen/CPU/GPU fingerprint), integrity
indicators (incognito, WebDriver, ad-blocking, "suspicious browser/OS changes"), behavioral
signals ("typing speed, paste, autofill, mouse movement, click timing, page duration,
navigation, and session-switching behavior"), and automation detection ("bot patterns, browser
automation, WebDriver, emulator/cloud-phone, app-cloning, hooking, tampering") — all **[FP]**,
quoted exactly. Persistent device IDs are stated to "support linkage across sessions" **[FP]**,
but no latency or refresh-cadence figure accompanies any of it.

**(b) Rule authoring / hit policy / lists.** "Let authorized users create rules, features,
thresholds, actions, and workflows without turning every change into an engineering project"
**[FP]**. The distinctive claim is Vera, an LLM copilot: "Turn natural-language intent into risk
logic... reducing the distance between an analyst's idea and an executable strategy while
keeping human review in the loop," and the FAQ is explicit about scope: "Vera can help create
and refine rules and features from natural-language input, explain existing rules, and support
quick testing — with human review built into every step before changes go live" **[FP]**. The
testing/lifecycle list is the fullest published anywhere in this survey: "backtesting, unit
testing, shadow mode, A/B or champion/challenger testing... [f]ull version control for rules,
features, and workflows... [v]ersioned audit history of every strategy change... [o]ne-click
rollback to a prior published version" **[FP]**. Set against that breadth, three things are
conspicuously absent from the visible text of the platform's own rules page: the word "list"
does not appear once (grepped directly), nor does "priority," "hit policy," "first match," or
"conflict" — so unlike AWS's `FIRST_MATCHED`/`ALL_MATCHED`, Stripe's priority order, or even
Actimize's unpublished-but-named `(expression, action, priority, active)` tuple, DataVisor
publishes no conflict-resolution semantics at all when multiple rules fire on one event. Given
"Fraud Rings," "AML," and "rulesets" are all product-page nouns (`dv_nonblog.txt`'s
`/aml-platform`, sitemap entries), lists plainly exist as a product surface, but their ageing,
membership-as-at-instant, and refresh mechanics are unreachable **[UNREACH — Document360 login]**.

**(c) ML integration / explainability.** "Combine supervised and unsupervised ML, rules,
features, device intelligence, graph signals, identity context, and third-party data within one
decision strategy," and per-decision output is stated as "Decision explainability — reasons,
rule execution, model scores, and features returned per decision" **[FP]**
([Real-Time Decisioning](https://www.datavisor.com/platform/real-time-decisioning)). Unsupervised
ML is named as one of six connected capabilities alongside supervised ML, "Fraud Pattern
Intelligence," dEdge, "Knowledge Graph," and case management **[FP]**, but no model architecture,
scoring range, or explainability mechanism (no SHAP/reason-code equivalent named the way
Feedzai names Whitebox/`timeshap` in §1.1) is documented. One case study gives real, if
customer-anonymized, numbers: "a leading U.S. accounting and financial-operations platform
supporting more than 7 million SMBs" deploying "Unsupervised Machine Learning (UML)" reports
"70% detection rate of coordinated fraud rings," "60% reduction in fraud losses," and a "10x"
operational-efficiency improvement **[FP, customer not named]**
([case study](https://www.datavisor.com/intelligence-center/case-studies/global-financial-management-platform-detects-70-percent-of-coordinated-fraud-rings)).
No methodology, baseline period, or measurement definition accompanies any of the three numbers.

**(d) Latency / throughput / failure mode.** The one hard technical spec sheet found: "Decisions
in 30 ms or less, at peak throughput above 15,000 QPS," restated as two headline stats, "30ms
Decision latency" and "15,000+ Peak QPS" **[FP]**
([Real-Time Decisioning](https://www.datavisor.com/platform/real-time-decisioning)). This is
worth flagging against this report's own cross-engine latency table (`_alternatives.md:51`),
which currently attributes DataVisor "under 100ms" — the correct, directly first-party figure
is **30 ms**, over 3× tighter, and the citation above should replace or supplement that entry.
No p99/p99.9 or tail figure is published (only a flat "30 ms or less"), no methodology
(in-process vs. network-inclusive) is stated, and no fail-open/fail-closed policy, timeout value,
or degraded-mode behavior appears anywhere in the pages fetched — the same gap this report
records for every vendor in `_unverified.md`'s "Every vendor" row.

**vs decider2.** Three points transfer. (1) **Vera is the Featurespace-AMDL risk in reverse
polarity.** Where Featurespace's AMDL is a proprietary DSL with *no* published grammar (§1.2),
Vera is an LLM front-end whose *output* representation is unpublished — is a Vera-authored rule
still data (JSON/table row), or does it write toward some richer structure DataVisor doesn't
expose? decider2 made the opposite bet explicitly: `docs/08-configuration-and-lifecycle.md`'s
§3.2 removed `_ComputedFeature` (an expression string parsed via `simpleeval`) precisely because
"there is no expression language to specify, secure, version or explain," and a derived value is
instead "a real node in the graph." If decider2 ever adds NL rule authoring per `_inventory.md`'s
existing row (A.2, "Natural-language / LLM rule generation... maybe — cheap once the rule
document schema exists"), Vera's own FAQ answer — "with human review built into every step
before changes go live" — is the right minimum bar, and decider2's closed, typed rule-document
schema is a *better* foundation for it than an expression-string DSL would be, because the LLM's
output can be validated against the same JSON Schema doc 08 §3.3 already exports. (2) **DataVisor's
testing-lifecycle list is further evidence for `_recs.md` items 12 and 13, not a new gap.**
"Shadow mode," "A/B or champion/challenger," "[f]ull version control," and "[o]ne-click rollback"
are exactly the vocabulary of decider2's unbuilt shadow-isolation-by-lineage
(`example_projects/01-transaction-fraud-interdiction.md` §5.11, `:434-457`, "structurally
impossible rather than merely discouraged") and backtest harness (`_recs.md` item 13) — DataVisor
publishes the checklist without publishing the enforcement mechanism, so it does not settle
whether isolation is structural or conventional there either. (3) **Rollback is the one row where
decider2's package, not just its spec, already matches the vendor claim.** decider2's
`runtime/serve.py:305-315` implements exactly DataVisor's "one-click rollback to a prior
published version" — "the previous generation never stopped being valid, so this is just popping
the history stack" — exposed over HTTP at `serving/dispatch.py:144` (`POST /rollback`). It is
scoped to *values* only (params/tables), matching doc 08 §3.3's three-surface split, not
structure — but on this one specific row decider2 is not behind the field, it is shipping. On
latency, decider2's in-process p99 of 1 114 µs (`docs/EXPERIMENTS.md:1010-1018`) is roughly 27×
inside DataVisor's headline 30 ms, but the comparison is the same apples-to-oranges problem as
Feedzai's: DataVisor's number is presumably network-inclusive end-to-end, decider2's is a single
in-process call, and neither publishes the breakdown that would make the two comparable.

### 1.6 Sardine — no-code Rule Editor, DIBB, and the ACH/card indemnification guarantee

**Evidentiary posture.** `sardine_full.txt` is real first-party content fetched directly from
`docs.sardine.ai`'s public pages (each section carries its own `Source:` URL, quoted below), but
the very first page in it says outright: "We provide a combination of both public and protected
documentation... Our protected documentation includes: step-by-step integration guides, API
references, SDKs, sample apps" **[FP]**
([Get Full Documentation Access](https://docs.sardine.ai/guides/public/getting-started/apiaccess)).
So the rule-syntax internals, the actual `/customers` API schema beyond field names visible on
the billing page, and any SDK detail are not in what's available — what is available is
conceptual/product-overview material, which is nonetheless more first-party detail than
DataVisor's marketing pages give, because it is Sardine's own docs platform rather than its
marketing site.

**(a) Feature computation / velocity / profiles.** "Our proprietary Risk SDK identifies risky
devices and intrinsic behavior... available for both web and native apps" **[FP]**
([What Powers Sardine](https://docs.sardine.ai/guides/public/getting-started/what-powers-sardine)).
ML is explicitly split into two regimes: "Sardine uses a **consortium approach** to building
supervised ML models around payments and onboarding fraud. By training models on rich datasets
across our network, the models 'see' a diverse group of fraud patterns... if there is a *modus
operandi* that targeted one company, the model will help prevent it the next time it comes
around to your company," plus "a few methods for anomaly detection" on the unsupervised side
**[FP]**. Transaction monitoring is profile-based, not window-based in the published description:
"creating behavioral profiles of customers based on their historical transaction... history and
interactions with an online platform, and comparing that profile to known money laundering
typologies (for example, placement, layering, integration)... an assessment of whether a certain
transaction was anomalous with respect to the historical behavioral baseline of the customer,
customer cohort or an online business," with the illustrative case "a customer, whose historical
average transaction amount is \$100 per week, suddenly starts depositing \$10,000 per week"
**[FP]** ([Transaction Monitoring](https://docs.sardine.ai/guides/public/risk/transaction-monitoring/transaction-monitoring)).
For payment-authorization fraud specifically, Sardine names its behavioral layer: "Sardine's
**Device Intelligence and Behavioral Biometrics (DIBB)** layer captures how a user interacts
with the app during a payment: typing cadence, hesitation, copy-paste behavior, and
remote-access tool signatures. Phone-coached sessions produce distinct behavioral patterns.
Sardine scores those patterns in real time, before the payment leaves the sending firm" **[FP]**
([Payment Fraud](https://docs.sardine.ai/guides/public/risk/funding-risk/funding-risk)). As with
DataVisor, no window length, aggregation statistic, or staleness/freshness guarantee is
published anywhere in the public docs — the profile mechanism is described entirely in prose,
never as a grammar.

**(b) Rule authoring / hit policy / lists.** "Sardine provides hundreds of out-of-the-box rules
to protect your business from fraud and compliance risks from day 1. You will have access to
hundreds of data points and features, including custom data points in our **no-code Rule
Editor**. Rules are developed in real-time... New rules can be launched in **shadow-mode**,
which allows you to assess their efficacy and performance before making the rule live" **[FP]**.
The dashboard bundles rules with lists as co-equal objects: "access to device and session data,
ID verification results, checkpoints, and rules, the Rule Editor, reporting, anomaly detection,
**block, and allow list**, queues for alert review and remediation, and full user management and
access control" **[FP]** — confirming allow/block lists as a first-class dashboard object, with
no published ageing, TTL, or membership-as-at-instant semantics (the same gap `_inventory.md`
records against every vendor except decider2's *spec*, which is itself unbuilt). No hit-policy
statement (first-match vs. all-match), no priority field, and no version-control/rollback
language is published for Sardine — a real asymmetry against DataVisor's explicit "[f]ull
version control... [o]ne-click rollback." One genuinely new detail not elsewhere in this
report: the billing page exposes a **checkpoint** concept as the actual API-level scoping
mechanism — "Sanctions (SDN, OFAC), PEP & Adverse Media, Screening & Monitoring... when the
checkpoint includes 'AML' and relevant customer data points... are provided. The Sanctions
feature must also be enabled by Sardine" **[FP]**
([How Sardine Bills](https://docs.sardine.ai/guides/public/getting-started/how-sardine-bills)).
A `checkpoint` value (e.g. `"AML"`) passed on the `/customers` call is what selects which rule/model
families evaluate *and* what gets billed — a request-scoped applicability flag, not a per-rule
attribute. The rule engine also reaches beyond transactions: "Rules can be created around
business entities through the rule engine" for KYB **[FP]**
([Business Verification](https://docs.sardine.ai/guides/public/risk/account-risk/kyb)).

**(c) ML integration / explainability.** The issuer-risk product states the pattern plainly:
"an 'issuing risk'-specific machine learning model that is trained on historical card
transaction & fraud data to assess the riskiness of a given card purchase... combined with
Sardine's proprietary, no-code rules engine to pick up on granular and complex patterns and
tuned over time to increase its detection accuracy to provide an optimal trade-off between
stopping bad transactions while allowing good transactions to go through" **[FP]**
([Issued Card Fraud](https://docs.sardine.ai/guides/public/risk/card-spending-risk/card-spending)).
That is score-feeds-rules, identical in shape to every other vendor in this survey. What is
**not** published anywhere in the public docs is any explainability mechanism for the ML side —
no reason-code taxonomy, no SHAP/feature-attribution claim, nothing resembling Feedzai's
Whitebox or AWS's `ruleResults` — the only stated transparency mechanism is the no-code rule
editor's visibility into rule logic itself, not model logic. **[gap, not found — may exist in
the login-walled docs]**.

**(d) Latency / throughput / failure mode.** Grepping all 668 lines of `sardine_full.txt` for
"latency," "ms," "TPS," or a number attached to a time unit returns nothing — no latency or
throughput figure of any kind is published in Sardine's public docs. The "~25ms" figure this
report's own `_alternatives.md:51` attributes to Sardine could not be traced to a first-party
page: a direct fetch of `sardine.ai`'s marketing homepage found only qualitative language —
"Real-time transaction monitoring," "Real-time, behavior-based fraud detection," "Turn insight
into real-time action" — and explicitly **no millisecond specification, TPS/QPS number, or
response-time benchmark anywhere on the page** **[UNREACH — checked docs.sardine.ai public pages
and sardine.ai homepage directly; the figure is not on either]**. It should be treated as
originating from a secondary source (an analyst comparison, a conference statement, or a review
site) until someone locates the primary page, and `_alternatives.md`'s citation should be
downgraded from implied first-party to **[SEC, unconfirmed]** pending that. No fail-open/
fail-closed policy or timeout value is published either.

**The distinctive item: financial guarantees, not detection.** Sardine sells two products that
put its own balance sheet behind its risk score, described with more mechanical precision than
almost anything else in its public docs. ACH: "Sardine can guarantee you against **fraud
liabilities** on ACH debits. We can indemnify for only unauthorized returns (R05, R07, R10, R11
& R29)" **[FP]**, gated by a stated prerequisite: "ACH fraud indemnification is an add-on to ACH
funding risk and we usually require about **3 months worth of ACH transaction data** before it
can be considered and offered" **[FP]**
([ACH Indemnification](https://docs.sardine.ai/guides/public/risk/funding-risk/ach-indemnification)).
Card: "Sardine can guarantee businesses against fraud liabilities on card transactions... offer
liability protection against fraudulent card charges to businesses and help businesses smooth
out the variability in cost associated with card chargebacks" **[FP]**
([Card Chargeback Guarantee](https://docs.sardine.ai/guides/public/risk/funding-risk/card-indemnification)).
The stated mechanism for why Sardine can underwrite this where a bank-verification vendor
cannot: "Existing bank verification products (Plaid, MX, Yodlee, Finicity, etc.) only verify
authenticity of the bank credentials but do not detect if the person connecting the bank
credentials is the true owner of the bank account. Sardine figures this out by building a full
360 profile of a customer by connecting their bank account to all other forms of identity
(email, social media, phone number) and further by utilizing our cutting edge Device
Intelligence" **[FP]**. Notably, **no dollar cap, percentage limit, or pricing figure for either
guarantee is published anywhere in the public docs** — grepped directly for "cap," "limit," and
"up to \$" and found none; "limit" appears only in an unrelated sentence about card-network
reversal-rate caps. So while the underlying idea (a vendor indemnifying losses up to a
negotiated limit) is real and confirmed, the specific limit is **[UNREACH — not published, likely
contract-specific]**.

**vs decider2.** Three points, and one explicit scoping call. (1) **The financial-guarantee
product is out of scope for a rule-engine comparison, and should be named as such rather than
tabulated.** Sardine's ACH/card indemnification is a balance-sheet product layered on top of its
detection output — it changes who bears a false-negative's cost, not how a decision is computed.
decider2's action taxonomy tops out at `freeze_account` / `decline` / `hold_for_review`
(`example_projects/01-transaction-fraud-interdiction.md` §5.12, action ranks 70–10); nothing in
decider2's spec or `_inventory.md`'s feature rows contemplates the engine's operator underwriting
the bank's residual risk, and it shouldn't — that is a lending/insurance decision a bank makes
about *a vendor*, not a capability an in-process rules kernel could ever provide. This is worth
one sentence in §6 (business-model difference), not a row in §2 (adopt: no — not an engine
capability). (2) **The `checkpoint` concept is a coarser, request-scoped cousin of decider2's
per-rule `Event types` field, and decider2's is already the more expressive design.** Sardine's
`checkpoint` (e.g. `"AML"`) gates an entire family of rules/models for a whole API call and
doubles as the billing unit; decider2's governed rule attribute "Event types | set of
`event_type_code`" (`example_projects/01-transaction-fraud-interdiction.md` §6.1) is scoped per
*rule*, not per *call*, so two rules with different event-type sets can coexist against the same
event without a coarse flag partitioning them. Nothing to adopt here — this is confirmation
decider2's existing design choice is sound, not a gap. (3) **Sardine's shadow-mode claim has the
same unproven-isolation problem as every other vendor's, which sharpens rather than weakens the
case for `_recs.md` item 12.** "New rules can be launched in shadow-mode, which allows you to
assess their efficacy and performance before making the rule live" states the *feature* but, like
DataVisor and Unit21, publishes no enforcement mechanism — no lineage assertion, no "cannot
reach a terminal" guarantee. decider2's spec already asks for more than any vendor in this
survey publishes ("structurally impossible rather than merely discouraged,"
`:434-457`), and FRAMEWORK-DEMANDS #9's "provable by lineage, at import time, with no data" would
put decider2 ahead of the entire vendor set on this one property if `_recs.md` item 12 is ever
built. On latency, no comparison is possible in either direction: Sardine publishes nothing, and
decider2's own `docs/EXPERIMENTS.md` figures measure a single in-process call, not an API
round-trip, so there is no like-for-like number to set against decider2's p99 1 114 µs
(`docs/EXPERIMENTS.md:1010-1018`) even if one existed.

### 1.7 Stripe Radar

All facts below are **[FP]** — fetched raw (`curl https://docs.stripe.com/<path>.md`, HTTP
200) and read as literal doc text, not paraphrase — unless marked otherwise. Pages: `/radar/rules.md`,
`/radar/rules/reference.md`, `/radar/rules/supported-attributes.md` (130 537 bytes, 868 lines),
`/radar/lists.md`, `/radar/risk-settings.md`, `/radar/how-radar-works.md`,
`/radar/transaction-reviews.md`, `/radar/transaction-risk-prevention.md`,
`/radar/reviews/risk-insights.md`, `/radar/testing.md`, `/radar/bot-abuse.md`,
`/radar/multiprocessor.md`, `/radar/supported-payment-methods.md`, `/connect/radar.md`,
`/api/charges/object.md`, `/api/radar/reviews/object.md`, `/api/radar/value_lists/object.md`,
`/api/radar/early_fraud_warnings/object.md`, all under `docs.stripe.com`. `/radar/backtesting.md`
and `/radar/rules/backtesting.md` both 404 ("Page not found | Stripe Documentation") — Stripe's
backtest mechanism is documented only inline in `/radar/rules.md` and `/radar/testing.md`, not on
a dedicated page; there is no separate URL to cite for it.

**Rule grammar and evaluation priority.** A rule is `{action} if {condition}` — Stripe's own term
for the pair is "the *predicate*". `Condition = [attribute] [operator] [value]`, e.g.
`Block if :amount_in_usd: > 1000.00`. Four condition shapes are documented verbatim —
string/country/numeric-attribute-vs-value, plus bare boolean attributes — and attribute-to-attribute
comparison is allowed: `Block if :card_country: != :ip_country:`. Sigils carry the type: `:attribute:`
for built-ins, `::key::` for payment/charge metadata (case-sensitive as a string; the *only* type
that supports both `<`/`>`/`<=`/`>=` **and** `INCLUDES`/`LIKE`, because it can act as string or
number), `::customer:key::`/`::destination:key::`/`::account:key::` for metadata on related objects,
and `@list_alias` for value-list membership ("All list aliases referenced in rules must start with
`@`"). Boolean composition is `AND`/`OR`/`NOT` (`&&`/`||`/`!`) with "standard operator *precedence*":
`{X} OR NOT {Y} AND {Z}` parses as `{X} OR ((NOT {Y}) AND {Z})`.

The action list, **in Radar's own evaluation priority order**, is itself the hit policy:
1. **Request 3D Secure** — non-terminating: "Whether or not there are matches on this rule, we
   evaluate rules for allow, block, and review afterward."
2. **Allow** — "not subject to further Radar rules evaluation," and gated behind a support request.
3. **Block** — "Stripe rejects it and it's not subject to further rules evaluation."
4. **Review** — the charge still completes, then enters the review queue.

Object applicability is itself a small compatibility matrix: Charge supports Allow/Block/Review but
not 3DS; SetupIntent supports 3DS/Allow/Block but not Review; only PaymentIntent supports all four.
Within a class Stripe states plainly: **"Rules of the same action type aren't ordered."** And across
classes: **"If a payment matches the criteria for a rule, Radar takes the appropriate action and
discontinues evaluation."** So the composite hit policy is: **priority-by-action-class (3DS →
Allow → Block → Review, with 3DS alone non-terminating), then first-match-wins within the winning
class, with order unspecified among same-class matches** (outcome-identical, since same-class rules
share an action). Stripe's own worked example: "A high risk payment over 1,000 USD is blocked
because it doesn't meet the criteria of either allow rule, then triggers both block rules,
regardless of the order of evaluation." Two documented refinements complicate this further: rules
using post-authorization attributes (`:cvc_check:`, `:address_zip_check:`, `:address_line1_check:`)
execute *after* rules that don't — "This won't affect whether a charge is blocked or not, but might
impact which rule blocks the charge" — and on Connect, platform and connected-account rules merge
into **one** rule set with platform rules evaluated first, so "a platform allow rule overrides a
connected account block rule, and a connected account allow rule overrides a platform block rule."

Operators (reproduced from the doc's own table): `=`, `!=`, `IN` work on every type; `<`/`>`/`<=`/`>=`
work on metadata and numeric only; `INCLUDES`/`LIKE` work on string/metadata/country/state only
(`%` wildcard for `LIKE`). Documented invalid conditions include `:risk_level: < 'highest'` (strings
take only `=`/`!=`) and `:is_anonymous_ip: = 'true'` (booleans take no operator/value at all).
**Exactly one function is documented: `is_missing(...)`.** Its three-valued-logic rule is stated
verbatim: "any comparison (for example, `=`, `!=`, `>`, `<`) of a missing feature against another
static value or feature (missing or present) **always returns false**. Usage of the `NOT` operator
with any comparison containing a missing feature always returns false." Booleans are exempt from
this: "`:is_new_card_on_customer:` is false instead of missing for ACH and SEPA transactions." No
arithmetic, no string functions, no user-defined functions — this is a strictly closed grammar.
Published limits: **"a maximum of 200 transaction rules and 100 account rules"**; lists cap at
**50,000 items**; rule-change visibility is **"the past 180 days"** only. There is no published
limit on expression size, nesting depth, or condition count — grepped and absent.

**Velocity and windowed attributes — the rare published aggregate catalogue.** This is the sharpest
technical find in Stripe's docs and worth reproducing exactly. From `/radar/rules/reference.md` §
"Velocity rules," verbatim:

> "Stripe calculates attributes using bucket increments. The increment length varies based on the
> attribute interval. This means the velocity for any attribute might include data that occurred
> within the interval plus one bucket... `hourly` is up to 3900 seconds (5 minute buckets); `daily`
> is up to 90000 seconds (1 hour buckets); `weekly` is up to 608400 seconds (1 hour buckets);
> `yearly` is up to 31622400 seconds (1 day buckets); `all_time` includes 5 years of data with
> velocity up to 31622400 seconds (1 day buckets)."

These are **tumbling-bucket approximations, not sliding windows**: `hourly` can include up to 65
minutes of data, `daily` up to 25 hours, `weekly` up to 7d 1h, `yearly`/`all_time` up to 366 days of
slack — and this is undocumented as a caveat anywhere except this one paragraph. A sixth suffix,
`monthly`, exists only on Connect account-level and Treasury attributes and is undefined against
the bucket table. Counting semantics exclude the current event: "the count excludes the payment
that you're currently processing... for the first charge attempt in a given hour for a customer,
`total_charges_per_customer_hourly` has a value of `0`." Recency attributes behave differently again:
a genuinely-new email makes `seconds_since_email_first_seen` **missing**, not zero.

A distinct numeric subtype, ***Bounded numeric (less than or equal to 25)***, covers every
cardinality/dispute/EFW/refund/name counter, and its mechanics are a saturating ring buffer of
recent timestamps, stated with a worked example:

> "After authorizing the 6th charge within the hour from `jenny.rosen@example.com`, the counter
> stops incrementing and stays at `5`... If an attempt to increment the counter above the cap
> occurs, we exclude older values from consideration and replace them with newer values. For
> example, consider a counter with a cap of `3`... filled up with `[10, 20, 30]`. When a charge
> arrives at time `50`, the counter now looks like `[20, 30, 50]`."

The practical consequence is a fourth value-state beyond "present/stale/absent": **saturated** —
`> 25` conditions written against these attributes can literally never fire, because the counter
stops counting rather than continuing to grow.

Mechanically counted from the raw attribute tables: **513 attribute rows, 485 distinct names**
(duplicates are platform attributes listed under both transaction- and account-rule sections).
Window-suffix census: `hourly` 76, `daily` 105, `weekly` 111, `monthly` 29, `yearly` 5, `all_time`
54. Families: (a) unbounded payment-outcome counters (`{authorized|blocked|declined|total}
_charges_per_{billing_address|card_number|customer|email|ip_address|shipping_address}
_{all_time|weekly|daily|hourly}`, plus a `_transactions_per_` variant with no `all_time`); (b)
bounded cardinality counters (`card_count_for_*`, `email_count_for_*`, `name_count_for_card_*`);
(c) bounded bad-event counters (`dispute_count_on_*`, `efw_count_on_*`, `refund_count_on_*`); (d)
bounded, ≤72h-fresh Customer-object counters (`total_customers_for_email_*`, `..._with_prior_fraud
_activity_...`); (e) monetary aggregates that exist **only at `all_time`** — there is no hourly,
daily, or weekly money aggregate anywhere in the catalogue; (f) recency/time-since numerics; (g)
Connect/Treasury platform aggregates and rate types. One naming inconsistency worth flagging for
anyone porting rules: the IP key is spelled `ip_address` in the card family
(`card_count_for_ip_address_hourly`) but `ip` in the email/dispute/EFW families
(`email_count_for_ip_hourly`, `dispute_count_on_ip_hourly`) — and joiners split between `_for_`/`_on_`
(cardinality/bad-event) and `_per_` (outcome counters). Keying is limited to **card number/card
fingerprint, cross-payment-method fingerprint, IP address, email, billing address, shipping
address, Customer ID, cardholder name, and connected account, and only those** — there is no keying
on device ID, session, BIN, phone number, or arbitrary metadata. **A user cannot define a new
windowed aggregate.** The attribute list is closed and enumerated (typing `:` in the rule editor
opens it); the only extensible inputs are per-transaction metadata scalars and list membership,
neither of which aggregates. `/api/radar/*` exposes only `value_lists`, `value_list_items`,
`reviews`, `early_fraud_warnings` — no aggregate-definition endpoint exists anywhere.

**Scores, ML integration, and explanation surfaces.** Five scores, all in `/radar/risk-settings.md`
§ "Radar scores": `fraudulent_dispute_score`, `early_fraud_warning_score`, and
`fraudulent_payment_score` (each 0–99, "Recommended"), the legacy `risk_score` (documented
inconsistently as 0–99 in one place and 0–100 in two others — a genuine Stripe documentation defect
worth flagging if quoted), and `bot_score` (0–99, Private preview, Radar Pro). `risk_level` ∈
`normal`/`elevated`/`highest`/`not_assessed`, with **default thresholds stated exactly**: "a risk
score of 65 or above corresponds to a risk level of `elevated`, while a risk score of 75 or above
corresponds to a risk level of `highest`," both adjustable. A set of ML-driven controls run
alongside rules, explicitly not as rules ("won't override the custom rules you created"): Fraudulent
card payments, Fraudulent dispute, Early fraud warning, Adaptive 3DS, and **Dynamic risk
thresholds** — "automatically blocks additional elevated and high-risk payments when your account is
under fraud pressure... When we no longer detect a threat, the block threshold automatically
readjusts" (card only, Pro only).

The blocked-payment explanation surface is entirely on the Charge's `outcome` object:
`outcome.type` (`authorized|manual_review|issuer_declined|blocked|invalid`); `outcome.reason`, whose
value is `rule` specifically for "Charges authorized, blocked, or placed in review by custom
rules" (versus `highest_risk_level`/`elevated_risk_level`/`low_probability_of_authorization` for
Stripe's own defaults); and — the single machine-readable "which rule fired" field in the whole
product — **`outcome.rule`: "The ID of the Radar rule that matched the payment, if applicable."**
Stripe also publishes an explicit honesty caveat: **"For a small subset of payments, Stripe modifies
the reported risk score so we can measure the performance of our models"** — deliberate score
perturbation for holdout measurement, admitted in the docs rather than left implicit. On the ML
itself, Stripe's own blog is more forthcoming than most vendors: the account-similarity model uses
"gradient-boosted decision trees (GBDTs)," specifically **XGBoost**, over a candidate-generation →
score-edges → threshold → connected-components pipeline
([similarity-clustering](https://stripe.com/blog/similarity-clustering)); Radar 2.0 shipped "Nightly
model training" and "Hundreds of new signals" and claimed "reduce fraud by over 25% compared to
previous models, without increasing the false positive rate"
([radar-2018](https://stripe.com/blog/radar-2018)); and the guide states "Radar uses hundreds of
features and most of them are aggregates computed from across the Stripe network" and "we need to
be able to compute the value of every feature for every new payment in real time"
([radar/guide](https://stripe.com/radar/guide)). Headline marketing figures — "AI trained on more
than 70 trillion data points," "32% average reduction in fraud" — are on `stripe.com/radar` itself,
so first-party, but not accompanied by methodology.

**Backtesting, rule analytics, and audit log.** The mechanism, from `/radar/testing.md`: **"before
you add or update a rule, we'll search for historical live mode payments that match the rule
criteria"**, summarised as Disputes-and-EFWs / Refunded payments / Blocked-and-failed payments /
Succeeded payments, plus **Overrides** for allow rules specifically ("payments that Radar blocks...
but now will be allowed by your proposed rule"). Stripe's rule-testing page states the backtest
window directly in the rules doc: **"a simulation on the last 6 months of charges."** The same page
gives an explicit decision procedure per action type — reproduced because it is the clearest
published statement anywhere in this survey of *how to read a backtest result*: implement a Block
rule if it "matches payments that were disputed, received an EFW, or refunded as fraud at the cost
of an acceptable amount of legitimate payments"; implement a Review rule on the same disputed/EFW/
refunded-as-fraud signal, "to closely evaluate potential fraudulent transactions"; implement Request
3DS the same way, with the caveat that "3DS doesn't always guarantee that your user will receive a
challenge"; and for Allow rules specifically, "there's no way of knowing which previously-blocked
charges would, if allowed, have turned out to be fraudulent" — Stripe states its own backtest's blind
spot in the doc rather than leaving a user to discover it.

Live per-rule metrics are the deepest published rule-analytics surface in this comparison. Every
rule gets a performance chart with edits marked as "triangular symbols... for before/after
comparison." Block rules specifically get **"Est. false positive rate"**, defined verbatim: "The
estimated percentage of non-fraudulent payments that were blocked for both your block rules as a set
and by individual rules. **These estimates are made using the estimated false positive rates of the
corresponding AI risk scores, which we calculate with experiments across the Stripe network**," plus
"Est. fraudulent payments prevented." This is a genuine solution to a hard problem every
suppression-action rule engine has: a blocked payment produces no observed outcome (it never
happened), so precision/FPR cannot be computed from labelled ground truth the way it can for allowed
payments. Stripe's answer is to substitute a **calibrated score-FPR curve**, measured by
network-wide score-perturbation experiments (the same mechanism as the score-modification honesty
caveat above), for the missing ground truth. Review rules get "Refund rate" and "Disputes from
approved reviews"; allow rules get "Disputes from overrides." Row-level detail — "view and download a
filtered list of every payment that a rule has been applied to" — and Sigma/Data Pipeline querying of
"rule decisions and attributes for each individual payment" round out the surface. The **Rule
activity** audit log records "the complete rule predicate before and after the update," who changed
it, retained for **180 days**, restricted to "the account owner, administrators, and developers."

**Lists, reviews, and Radar Assistant.** Default, non-deletable lists exist per payment method (Card
BIN, Card country, Card fingerprint, Email, Email domain, Client IP, etc.); custom list item types
number ten in the Dashboard and twelve in the API enum (`radar.value_list.item_type` adds
`crypto_fingerprint` and `account`, undocumented in the UI). Refunding with
`reason: 'fraudulent'` auto-populates the default block lists with the card fingerprint and email —
the only documented write-path into lists that isn't a manual edit or the API. The review queue
supports Approve/Capture/Refund/"Refund and report fraud" (the last feeds the auto-population
above); ACH and SEPA Direct Debit **cannot be manually reviewed**; a dispute on an in-review payment
auto-closes the review. **Radar Assistant** — an LLM rule generator built into the rule editor —
"constructs a Radar transaction rule from a natural language prompt" ("Block Discover card payments
of more than $1000"), with an explicit training-data consent notice ("Stripe can log and use your
chat entries to train and improve the Radar Assistant capabilities") and no published model name,
latency, or accuracy figure. All of the above — risk scores, custom rules, backtesting, manual
review, Radar Assistant — gate behind the Plus/Pro plan tiers; Dynamic risk thresholds and bot/abuse
signal families are Pro-only.

**Latency — unpublished.** Checked the entire `/radar/*` corpus, `stripe.com/radar`,
`stripe.com/radar/guide`, and the Radar blog. **No latency percentile, millisecond figure, or SLA
for Radar evaluation is published anywhere reachable.** The guide says only that features must be
computed "in real time," with no number attached. A search-engine summary claiming Radar "produces
a risk score in milliseconds" could not be located on any first-party page and should be treated as
**[SEC]**/unverified, not cited as fact.

**vs decider2.** Four transfers, one of them a genuinely new hit-policy shape worth its own line in
§4. (1) **Stripe's "ordered action classes, first-match-within-class" is a third hit-policy shape**,
distinct from both `first_match` and the `all_match` already proposed for `tables/schema.py`'s
`DecisionTable` (`_recs.md` item 1). decider2's own action resolution
(`example_projects/01-transaction-fraud-interdiction.md:463-500`) already does something similar by
hand — a seven-rank action-severity table (`freeze_account` > `block_channel` > `decline` >
`hold_for_review` > `step_up` > `monitor` > `allow`) with a critical-rule override and a four-level
deterministic tie-break (severity, then priority, then family, then `rule_id`) — but this logic lives
entirely in prose (§5.12), not as a named, reusable hit-policy kind the table/ruleset machinery
understands. Stripe is evidence that "partition into ordered classes, resolve within a class" is a
common-enough shape to name and generalise rather than leave as one project's bespoke resolver.
(2) **decider2's velocity contract needs a fourth state.** decider2 already carries **fresh / stale /
absent** per aggregate with a per-window staleness tolerance
(`example_projects/01-transaction-fraud-interdiction.md:134-139`) — strictly better than Stripe's
undocumented tumbling-bucket slack, which silently admits up to 65 minutes of extra data into an
"hourly" attribute with no flag for it. But Stripe's bounded counters expose a state decider2's
current three-way model has no room for: **saturated** — a value that has stopped growing and is now
a floor, not a fact, so that `> N` conditions silently never fire. If decider2's velocity aggregates
are ever produced by a bounded upstream store (a very plausible failure mode at 3 500–12 000
events/second, §4.1), the same silent-never-fires bug is live unless the input contract has a name
for it. (3) **decider2's per-rule declared absence policy is strictly richer than Stripe's single
fixed rule.** Stripe's three-valued logic is uniform and non-configurable ("any comparison... always
returns false"); decider2 lets each rule declare its own behaviour per referenced feature —
`evaluate_false` / `last_good` / `suppress`, the third recorded as "unevaluable"
(`example_projects/01-transaction-fraud-interdiction.md:261-266`, `:393-401`) — which is exactly the
distinction a bank needs between "the client made zero payments" and "we couldn't tell." (4) **The
early-exit philosophies are opposite by design, not by accident.** Stripe explicitly stops
evaluating once Allow or Block matches ("discontinues evaluation") because its job is to answer one
question fast; decider2 explicitly forbids early exit — **"There is no early exit. Every applicable
rule is evaluated even after a `decline`-action rule has fired"**
(`example_projects/01-transaction-fraud-interdiction.md:398-401`) — because the complete firing set,
not the fastest true predicate, is the product a bank's dispute process needs. Worth stating in the
doc as a deliberate trade-off rather than a design gap: decider2 pays for every rule on every event
so that "why wasn't this blocked" always has a complete answer. Finally, Stripe's own **published
rule-count ceiling (200 transaction rules)** is a useful scale marker: decider2's fraud spec runs
**521 live + 114 shadow + 1 360 retired** rules (`:731-738`) — nearly three times what Radar's own
product allows per account — which is a concrete argument for why a Tier-1 bank cannot simply buy a
SaaS rules engine off the shelf and is evidence for the framework's own reason to exist.

---

### 1.8 AWS Fraud Detector

All facts below are **[FP]**, fetched directly from `docs.aws.amazon.com/frauddetector/latest/` (User
Guide, path prefix `ug/`; API Reference, path prefix `api/`).

**Deprecation, first.** Every User Guide page carries: **"Amazon Fraud Detector is no longer open to
new customers as of November 7, 2025. For capabilities similar to Amazon Fraud Detector, explore
Amazon SageMaker, AutoGluon, and AWS WAF."** (The API Reference banner is a variant with a
grammatical error — "is no longer be open" — and names "Amazon SageMaker AI" instead.) This is a
**closed-to-new-customers** notice, not a shutdown date; no end-of-support date is published
anywhere in the pages read, and the API version (`frauddetector-2019-11-15`) is unchanged. Treat
"when does this actually go away" as **unverified** — worth a §6 line, since anyone citing AWS Fraud
Detector as a live reference architecture should know new accounts cannot be provisioned at all.

**The rule language: DETECTORPL.** The single most important design fact, quoted exactly from
`.../ug/create-a-rule.html`: **"Each rule must contain a single expression that captures your
business logic. All expressions must evaluate to a Boolean value (true or false) and be less than
4,000 characters in length. If-else type conditions are not supported."** (The `CreateRule` API
constraint disagrees by 96 characters — "Maximum length of 4096" — a minor but real documentation
inconsistency.) So a rule is **exactly one boolean expression** plus an attached array of outcome
names: no statements, no assignment, no local variables, no loops, no user-defined functions, and no
rule-internal control flow of any kind — every bit of control flow in the product lives in the
detector's execution mode (below), not the rule.

Variables are `$`-prefixed account-level resources, not declared in the expression:
`$example_variable < 100`; a model score is referenced as an ordinary variable via the convention
`$<model_name>_insightscore` (e.g. `$sample_fraud_detection_model_insightscore > 900`). Lists use
`$example_list_variable in @list_name` — and crucially, **"A rule associated with your list
automatically incorporates newly added or removed data"** without touching the rule at all, a
deliberate separation of policy from data. A list "can contain up to 100,000 unique entries and each
entry can be up to 320 characters long," and **"you can use up to 3 lists in a rule."** The full
operator set is `>`, `>=`, `<`, `<=`, `!=`, `==`, `in`, `not in`, `and`, `or`, `!`, plus arithmetic
`+ - * / %`; inline array literals work directly in `in`: `$variable in [5, 10, 25, 100]`. Boolean
short-circuit is documented precisely: **"Amazon Fraud Detector stops in an `OR` expression when a
single true value is found, and it stops in an `AND` when a single false value is found."**
Operator precedence between mixed `and`/`or` is **not documented anywhere** — a real risk for anyone
porting a compound rule.

Regex support is exactly one function, `regex_match(pattern, subject)`, "based on `.matches()` in
java (using the RE2J Regular Expression library)" — which means whole-string anchoring, so every
"contains" pattern needs a leading `.*` (`regex_match(".*mystring", $variable)`); the doc's own
"starts with" example (`regex_match("^mystring", $variable)`) is arguably wrong under that
semantics. Null handling is a first-class literal usable only with `==`/`!=`
(`$variable == null`), and whether a variable can ever be null at evaluation time is governed
entirely by its **mandatory default value** (§ below), not by the expression. Datetime support is
four functions — `getcurrentdatetime()`, `isbefore()`, `isafter()`, `getepochmilliseconds()` — and no
date arithmetic, diff, or timezone conversion beyond that; **any "within the last N days" logic must
be hand-rolled** as an epoch-millisecond subtraction. Two documented bugs are worth flagging for
anyone quoting these functions: `isbefore`/`isafter` are typed Boolean but every example compares the
result to the *string* `"True"`/`"False"`, and AWS's own worked example,
`getepochmilliseconds("2019-11-30T01:01:01Z") == 1575032461`, returns epoch **seconds**, contradicting
the function's own name. The entire string library is two functions, `uppercase()` and
`lowercase()` — no `contains`, `length`, `substring`, `split`, `trim`, or concatenation exist.
**There is no coercion or casting section in the documentation at all**: five data types (String,
Integer, Boolean, DateTime, Float) each have a mandatory default, AWS's own example silently mixes
Integer and Float (`$variable_1 * 100.0 > $variable_3`), and the wire format
(`GetEventPrediction.eventVariables`) is a **plain string-to-string map** — every value arrives as a
string and is coerced to its declared type by the service, implicitly and undocumented.

**Execution modes and outcomes.** From `.../ug/create-a-detector-version.html`, verbatim: **"If the
rule execution mode is `FIRST_MATCHED`, Amazon Fraud Detector evaluates rules sequentially, first to
last, stopping at the first matched rule... If the rule execution mode is `ALL_MATCHED`, then all
rules in an evaluation are executed in parallel, regardless of their order... and returns the
defined outcomes for every matched rule."** Ordering is simply array order in `CreateDetectorVersion`
and matters only under `FIRST_MATCHED` — there is no priority field, no rule groups, no phases.
Under `ALL_MATCHED` the union of outcomes across every matched rule is returned with **no ordering,
no de-duplication, and no conflict-resolution semantics documented at all**. Detector versions carry
a `DRAFT`/`ACTIVE`/`INACTIVE` status, with exactly one `ACTIVE` version used by default. Outcomes
themselves are **account-level named resources with no payload** — no score, no reason string, no
structured data — "Each rule that is used in a fraud prediction must specify one or more outcomes,"
and in the response they come back as bare strings (`ruleResults: [{outcomes: [...], ruleId}]`). All
decision richness must therefore be encoded purely in the *choice* of outcome name.

**Shadow mode, simulation, and rule-performance analytics — CONFIRMED ABSENT.** This was checked on
two independent grounds and both confirm the negative. First, the complete `Actions` list (74
operations at `.../api/API_Operations.html`, read in full) contains **no simulate, test,
evaluate-draft, replay, shadow, champion/challenger, or rule-statistics operation** — the full
enumerated list runs from `BatchCreateVariable` through `UpdateVariable` with nothing in between that
resembles it. Second, the only adjacent capabilities are explicitly *not* substitutes: **batch
predictions can run against a `DRAFT` detector version** ("You can choose a detector version in any
status"), which is an offline what-if against an S3 CSV file, not a live-traffic split; and
`ListEventPredictions`/`GetEventPredictionMetadata` give post-hoc per-event audit, not aggregate
rule hit-rate analytics. **There is no rule-level metric emitted to CloudWatch documented anywhere
in the User Guide.** Of everything in this survey's larger vendor set, AWS — the most exhaustively
API-documented engine of the two in this section — is the one with the flattest simulation story:
no backtest harness, no shadow population, no per-rule live analytics of any kind.

**Statefulness: a stateless rule layer over a stateful model layer.** This is the sharpest finding on
the feature-computation axis and deserves to be read carefully. **Transaction Fraud Insights (TFI)**,
verbatim from `.../ug/transaction-fraud-insights.html`: **"the Transaction Fraud model's feature
engineering engine calculates values for each unique entity within your training dataset... For
example, during the training process, Amazon Fraud Detector computes and stores the last time an
entity made a purchase and dynamically updates this value each time you call the
`GetEventPrediction` or `SendEvent` API."** **`GetEventPrediction` is therefore not a pure function
— it mutates stored per-entity state as a side effect of scoring.** **Account Takeover Insights
(ATI)** does the same for login behaviour: "continuously compute aggregated variables that describe
the past user behavior... for example, the model might calculate the number of times a user has
logged in using the same IP address." **Online Fraud Insights (OFI)**, by contrast, keeps no
aggregates at all and is the only model type trainable from external S3 data. The critical
limitation for a rules comparison: **these aggregates are not addressable from DETECTORPL at all** —
there is no `@count(...)`, no window function, no velocity operator; the rule layer sees only the
supplied payload variables, list membership, and the model's scalar `_insightscore`. So the split is
exact: **rule engine — purely stateless over the supplied payload; model layer — stateful per
entity for TFI/ATI, advanced by the very call that reads it.** Missing-value semantics compound this:
an *explicit* null and an *absent* key are handled differently, and differently again for built-in
models ("model will replace the null value or the missing variable... with calculated default
mean/medians for numeric variables and with special values for categorical variables") versus
imported SageMaker models ("the model and rules will use `'null'` as the value" for an explicit
null, but the variable's declared default for an absent key). Retention of stored events is
**undocumented** on every page that could plausibly state it (`event-data-storage.html`,
`storing-event-data-afd.html`, the quotas page) — deletion is manual only (`DeleteEvent`,
`DeleteEventsByEventType`).

**Explainability: `GetEventPrediction` vs `GetEventPredictionMetadata` — the closest thing in this
survey to a "replay a past decision" API.** The synchronous scoring call
(`.../api/API_GetEventPrediction.html`) returns only `externalModelOutputs`, `modelScores`, and bare
`ruleResults` — no timing, no evaluated-expression text, no per-rule matched flag, no explanation.
Score scale is stated precisely: **"Amazon Fraud Detector generates model scores between 0 and 1000,
where 0 is low fraud risk and 1000 is high fraud risk. Model scores are directly related to the
false positive rate (FPR). For example, a score of 600 corresponds to an estimated 10% false
positive rate whereas a score of 900 corresponds to an estimated 2% false positive rate."** (A
separate page states "a risk score between 1 and 1000" — a one-off-by-one documentation
inconsistency.) All of the explanation richness is deferred to a **separate, post-hoc** call,
`GetEventPredictionMetadata`, which requires all five of `detectorId`, `detectorVersionId`,
`eventId`, `eventTypeName`, and `predictionTimestamp` — **you cannot fetch an explanation by event ID
alone**, and AWS itself recommends calling `ListEventPredictions` first to get the exact
`predictionTimestamp`. Its response is the richest artefact in the entire product:

```json
"rules": [ { "evaluated": boolean, "expression": "string",
             "expressionWithValues": "string", "matched": boolean,
             "outcomes": [ "string" ], "ruleId": "string", "ruleVersion": "string" } ]
```

`expressionWithValues` — **the rule's own expression text with the actual runtime values substituted
in** — is the single best explainability primitive found anywhere in this survey: a human-readable,
per-event trace of exactly what each rule saw, requiring no separate join between a feature table and
a rule definition. `evaluated` vs `matched` separately distinguishes "was reached" from "was true,"
which is exactly the pair needed to audit `FIRST_MATCHED` short-circuiting, and `eventVariables[]
.source` records whether each value was supplied, defaulted, or enriched. Model-level explanation
rides alongside: prediction explanations use **SHAP** ("Amazon Fraud Detector uses SHAP (SHapeley
Additive exPlanations) to explain individual event predictions"), producing a raw `logOddsImpact`
("usually between -10 to +10... from -infinity to +infinity") mapped to a 0–5 star `relativeImpact`
rating with direction, available only for "models trained on or after June 30, 2021." No retention
period is stated for `GetEventPredictionMetadata` data either.

**Quotas, lists, and external models.** The full published quota table (`.../ug/limits.html`) is
worth reproducing because it bounds exactly the axes decider2 cares about: **5,000 variables, 5,000
rules, 5,000 outcomes, and 100 detectors per account; 3 lists per rule, 30 lists per detector; 200
draft versions per detector; 10 models per detector version; 100 event types, 100 entity types, 100
labels per account.** Lists themselves cap at 100,000 entries × 320 characters. The only published
API-level figure is throughput, not latency: **"GetEventPrediction API calls per second: 200 TPS
(default, adjustable)"** — a rate limit, not a capability claim, and **there is no published latency
figure for `GetEventPrediction` anywhere in the docs read.** External models plug in via
`PutExternalModel`, whose mechanism is a template-substitution adapter: `ModelInputConfiguration`
carries a `jsonInputTemplate`/`csvInputTemplate` (max 2000 chars) whose `{{variable}}` placeholders
are substituted from the event at call time, and `ModelOutputConfiguration` maps the SageMaker
response's JSON keys or CSV column indices **back into first-class Fraud Detector variables** — which
is what lets an external model's output be referenced in DETECTORPL exactly like any built-in
`$variable`. Batch predictions run "in the same way as the `GetEventPrediction` operation," one job
at a time, max input size **1 GB**, against a detector version "in any status" — the only offline
what-if mechanism the product has.

**vs decider2.** Four transfers, and the most important one is a validation rather than a gap.
(1) **AWS independently confirms decider2's own architectural boundary: rules are a pure function,
state lives elsewhere.** decider2's inventory already records this as "decider2 is stateless per
record by construction" — the AWS evidence above is a hyperscaler's fraud product landing on
*exactly* the same split (stateless DETECTORPL over a stateful TFI/ATI model layer, aggregates
computed outside the expression language) independently. That is worth citing in the report as
corroboration, not as something to copy: decider2's velocity aggregates arrive as declared inputs
with a watermark (`example_projects/01-transaction-fraud-interdiction.md:126-140`), which is a
better contract than AWS's — AWS's aggregates are invisible even in the audit trail beyond the
model's SHAP attributions, whereas decider2 records every aggregate's value and freshness state
(`:257-260`). (2) **`FIRST_MATCHED`/`ALL_MATCHED` as an explicit, named, per-run mode is exactly
`_recs.md` item 1's motivation** for adding a `hit_policy` field to `tables/schema.py`'s
`DecisionTable`, which today is hard-wired to first-match-with-default (`tables/schema.py:561-564`);
that recommendation stands as written and this section adds no new argument for it beyond confirming
AWS's own `ALL_MATCHED` returns an outcome union with **no documented conflict resolution** — a
concrete cautionary data point for how *not* to leave `all_match` semantics
underspecified when it is eventually built. (3) **decider2's no-early-exit design makes the
`evaluated`-vs-`matched` distinction unnecessary — for free.** AWS needs two separate booleans
because `FIRST_MATCHED` means most rules in a detector are never reached at all; decider2's spec
states **"There is no early exit. Every applicable rule is evaluated"**
(`example_projects/01-transaction-fraud-interdiction.md:398-401`), so "applicable" and "evaluated"
are the same set by construction and decider2's audit trail never needs to answer "was this rule
skipped or did it just not fire" — one fewer bit of state to track and explain to a reviewer, a
direct dividend of §5.10's design choice. (4) **AWS having *no* shadow mode or backtest harness at
all is the strongest validation yet for `_recs.md` items 12 and 13** (structural shadow isolation,
`backtest()` entry point): a fully-documented, quota-complete, API-first hyperscaler product ships
zero rule-simulation capability, so decider2's spec requiring shadow isolation "structurally
impossible rather than merely discouraged" (`:434-457`, §5.11) is not catching up to table stakes —
it is building a capability the best-documented public reference in this whole comparison simply
does not have. Finally, on the explainability axis, `expressionWithValues` is a concrete rendering
technique decider2 does not yet have anywhere in its own tooling: decider2's emitted-intermediates
mechanism (`docs/04-observability-and-governance.md:155-170`) records column *values*, and its
reviewer test (`docs/04-observability-and-governance.md:277-333`) found that a reviewer shown values
in a separate table still could not connect them to the predicate that used them — a single string
per fired rule, built by substituting each leaf's runtime value directly into the predicate's own
text, is a narrowly-scoped, cheap thing to build from pieces decider2 already has (the rule's
structure plus its emitted feature values) and is a more direct answer to that specific failure mode
than anything currently proposed.

---
### 1.9 BioCatch

**A naming correction, first — [FP].** The datasheet at
[DS_PolicyManager_New_3.pdf](https://www.biocatch.com/hubfs/Data%20Sheets/DS_PolicyManager_New_3.pdf)
resolves, but the document inside is titled **"BioCatch Rule Manager"**, footer "© BioCatch
2020" — not "Policy Manager". BioCatch's current resources index confirms the product name as
[Rule Manager](https://www.biocatch.com/resources/data-sheet/rule-manager). "Policy" turns out
to be one of four *outcome types* the tool can trigger, not the product name — cite the tool as
Rule Manager, and reserve "policy" for the outcome kind below.

**Rule builder model — [FP], verbatim.** "Using the BioCatch Rule Manger, fraud teams can build
rules using BioCatch insights such as risk score and leading indicators in order to trigger a
desired outcome with exceptional flexibility and ease." Four documented input classes: **"Risk
scores: a score that specifies the degree of risk associated with a session or activity, ranging
from 0-1000"**; **"Risk factors: risky behavioral patterns such as app toggling and excessive
deleting of personal information"**; **"Genuine factors: genuine behavioral patterns such as
fluid typing and high data familiarity"**; **"Threat Indicators: presence of threats such as
Remote Access Tools (RATs), malware, Bots, and social engineering voice scams."** Plus external
inputs ("third-party solutions such as external risk scores") and, distinctively, **"Aggregate
scores from multiple systems to be considered in a rule."** "Genuine factors" is the one input
class nothing else in this survey publishes — evidence *against* fraud as first-class rule input,
not just evidence for it.

**Outcomes/actions — [FP].** "rules built within the BioCatch Rule Manager can trigger **policy,
score, alert, and custom action** outcomes at both the activity and session level." Policy =
"allowing, declining, authenticating, or reviewing activity according to risk level" (four
terminal actions). **Score = "customizing risk score output"** — a rule can *rewrite the score
itself*, not just emit a decision. Alert = automated email/API/case-generation. This is the only
vendor in the whole survey where a rule's output can be a mutated score rather than only a
decision or a list-membership change.

**Segregation of duties — [FP], the headline quote, read carefully.** "Implement segregation of
duties with **administrator and user level access and edit controls**." That sentence is the
entirety of what is published. It is role-based access control — two named roles — not a
maker-checker workflow: there is no stated second-person approval step, no versioning, no
rollback, and no policy simulation/backtesting anywhere in the datasheet. The distinction matters
for how this report cites BioCatch elsewhere: "segregation of duties" is real and first-party,
but it is weaker evidence for *four-eyes-on-a-specific-change* than the phrase suggests on its
own — BioCatch never says a second person reviews rule X before it goes live, only that two
*role tiers* exist.

**Latency — [UNREACH].** The datasheet says only "mitigate risk in real time" / "Initiate
immediate action" — no ms figure, no percentile. `biocatch.com/platform` names Rule Manager in
navigation with no spec. BioCatch has no public API/developer portal at all, so there is no
secondary source to fall back on either.

**vs decider2.**
1. BioCatch's **Score outcome** — a rule that overwrites the score rather than only the decision
   — has no analogue in decider2's seven-action taxonomy (`example_projects/01-transaction-fraud-
   interdiction.md:465-473`, `freeze_account` … `allow`) or in its overlay mechanism, which
   changes what a threshold *reads* (§6.5) but never what a downstream score *is*. Worth an
   explicit scope call: if any consumer downstream of decider2 uses the model score for its own
   purposes (a partner bank, a case-management priority queue), a rule that can correct that
   score is a real gap, not a nice-to-have.
2. BioCatch is the strongest public "segregation of duties" phrase in the survey, and it is
   still only RBAC. That is directly relevant to decider2's own **O17 — Approval granularity**
   (`docs/06-open-questions-and-experiments.md:204-207`: "Does activating a staged generation need
   per-rule approval, or is document-level enough?", referenced from `:99-113`) and to the
   "Approval: Four-eyes" column that already appears on every row of the rule attribute table
   (`example_projects/01-transaction-fraud-interdiction.md:704-729`). decider2's spec already
   assumes something stronger than BioCatch's best public evidence demonstrates — that is a
   point in decider2's favour, not a citation to lean on for validation.
3. "Genuine factors" as a first-class, negatively-weighted input type has no counterpart in
   decider2's rule model, which reads risk-additive features only. Not concrete enough for a
   recommendation on its own, but worth naming in §6 as a design idea BioCatch validates: a
   governed input type whose presence *lowers* a resolved severity, symmetric to how a critical
   `allow` rule already suppresses firings (`:475-481`).

### 1.10 Ravelin

**Timeout + fail-open — [FP], the sharpest published number in the survey.** From
[error-handling](https://developer.ravelin.com/merchant/guides/payment-fraud-integration/error-handling/):
> "In the scenario Ravelin returns a non-2xx HTTP status code, we recommend **accepting the
> order**." … "We recommend configuring a **750ms timeout**." … "Persistent responses at this
> latency suggest degraded performance of our services."

That 750 ms figure already appears correctly in `_recs.md` item 7 — verified against the digest,
no correction needed. What the recs list has *not* yet captured is a second, contradictory data
point from [rate-limits](https://developer.ravelin.com/merchant/api/rate-limits/) — [FP]: a
per-customer breach (**"50 events per minute per customer ID"**) does not fail open. Instead
Ravelin "**return[s] a 200 response with the action set to PREVENT and the source set to
RATE_LIMIT**. We do not retain the data from the request in this case." So Ravelin's own
published policy is not one constant — it fails *open* on an infrastructure-side non-2xx/timeout,
and fails *closed* on a self-protective rate-limit trip. [FP] also:
[load-testing](https://developer.ravelin.com/merchant/api/load-testing/) — "Ravelin does not
support external load testing of our API."

**Decision/recommendation response — [FP].** `{"data":{"action":"ALLOW","score":12,
"source":"RAVELIN","customerId":"...","scoreId":"..."}}`. Action enum: `ALLOW`, `REVIEW`,
`3DS_AUTHENTICATE`, `MANUAL_REVIEW`, `PREVENT`. Two explicit cautions worth quoting exactly:
**"`score`… Debugging purposes only"**, and **"Do not implement any logic based upon the `source`
field… as we may add new source values in the future."** There is no reason-code field in the
merchant response at all — the nearest things are `source`, `warnings[]`, and a free-text
reviewer `comment`.

**Checkpoints + rule-category scoping — [FP].** "A checkpoint is a key event within a customer's
journey where you can request a recommendation":
[checkpoints](https://developer.ravelin.com/merchant/guides/other-guides/checkpoints/) lists
`accountRegistration`, `login`, `paymentMethodRegistration`, `paymentMethodSelection`,
`checkoutPreAuth`, `checkoutPostAuth`, `refundRequest`. Orthogonal to that, rule **categories**
(`payment`, `ato`, `refundAbuse`, `supplier`) scope which rules run at a given checkpoint via
`&category=` — **"if the category parameter is not provided, all rules at the checkpoint will be
executed regardless of their category."** Two independent scoping axes, journey-stage and
abuse-category, deliberately kept separate.

**Rules vs ML, hit policy — [FP] for the override, [SEC] for the rest.** Manual review is an
absolute ceiling: "If you review a customer as genuine then Ravelin will always return `ALLOW`
for that customer… however strongly connected to fraud networks. **Any rules that you configure
will not change this.**" The symmetric fraud mark is permanent too: "**No rules you set will
change this** (Ravelin personnel are able to install rules that do change this)." Also [FP]:
network propagation via **"hops to fraud"**. [SEC] (`ravelin.com/blog/ai-fraud-prevention-rules`):
**"a rule can contain up to sixteen conditions"**; an AI rule generator (June 2024) that turns
natural language into a rule; **"Estimate rule impact"** backtesting **"if it had been active in
the past week"** — a one-week backtest window, versus Sift's 30 days and Stripe's six months.

**Latency — [UNREACH].** `merchant/api/guarantees/` states only content-type/versioning
guarantees, no SLA. The only quantitative signal Ravelin publishes is the 750 ms client-side
timeout above.

**Datalists — [UNREACH].** Not present anywhere on `developer.ravelin.com` (confirmed against
its 34 KB sitemap.xml — no rules/datalists path). The rules and Datalists UI live in Ravelin's
Help Centre, which is Notion-hosted and JS-rendered (`support.ravelin.com` returns
`<title>Notion</title>`) — not machine-readable.

**vs decider2.**
1. Ravelin's split of fail-open-on-infra-failure vs fail-closed-on-self-protection is a materially
   different, and better-specified, design than a single flag. `serving/dispatch.py` has **no**
   deadline, timeout, or degraded-response concept at all today — `invocations()`
   (`dispatch.py:61-71`) calls `self.handle.score(record)` unconditionally, and `ROUTES`
   (`dispatch.py:137-146`) has nothing for a slow-kernel case. `_recs.md` item 7 already proposes
   a `deadline_ms`; Ravelin's evidence says that page needs **two** documented defaults, not
   one: what happens on a deadline breach caused by something upstream (fail open or fail closed,
   per project), and what happens when decider2's own fire-rate circuit breaker trips
   (`example_projects/01-transaction-fraud-interdiction.md:740-748`) — which should default to
   fail-closed and carry a distinguishing marker, exactly as Ravelin's `source: RATE_LIMIT`
   differs from `source: RAVELIN`.
2. Checkpoint × category is a two-dimensional scoping matrix decider2 does not have. The rule
   attribute table has `Event types` and `Segments` (`:704-729` — `example_projects/01-
   transaction-fraud-interdiction.md`), which is close to "category" but has no journey-stage
   axis: nothing distinguishes a rule meant to run at authorisation time from one meant to run at
   payout time within the same event type. See candidate addition below.
3. Ravelin's permanent per-customer manual override ("any rules you configure will not change
   this") is entity-state, not rule-state — a different mechanism from decider2's critical-rule
   `allow` suppression (`:475-481`), which suppresses *families*, not a specific client
   permanently. decider2's case management is explicitly out of scope (`:1100-1103`), so this is
   a boundary question to write down rather than build: does "permanently allow-list this one
   client" belong to decider2 or to the system that calls it?

### 1.11 Forter

**Decision response shape — [FP]** (`docs.forter.com/reference/overview`,
`docs.forter.com/reference`). `forterDecision` ∈ `APPROVE`, `DECLINE`, `NOT_REVIEWED`,
`VERIFICATION_REQUIRED`. `recommendation` carries named strings functioning as reason codes:
`MONITOR_POTENTIAL_LIMITED_ITEM_ABUSE`, `MONITOR_POTENTIAL_COUPON_ABUSE`,
`VERIFICATION_REQUIRED_3DS_CHALLENGE`, `REQUEST_SCA_EXEMPTION_TRA`, `EMAIL_VERIFICATION`,
`SMS_VERIFICATION`, and others. `verificationMethod.status` ∈ `frictionless`, `attempted`,
`authenticated`, `not authenticated`. A distinct `reasonCode` field does **not** appear on the
core decision — it only shows up in the separate abuse-policy response (below). Schema
convention, verbatim: fields marked "conditional" are "dependent/required or part of a one-of
group, and excluding conditional fields will not result in an error response" — a materially more
permissive validation posture than decider2's, which returns 400 on any pydantic `ValidationError`
(`serving/dispatch.py`'s `_as_bad_request`, lines 39-42).

**Timeout & retry — [FP].** "Set a client side timeout of `2 seconds` on the HTTP request."
Retry recommended on client timeout, `5xx`, `429`. Latency guidance recommends TLS session reuse
and connection pooling "to avoid unnecessary latency from repeated TLS handshake processes."
**No fail-open / default-decision policy is stated anywhere reachable** — this is a confirmed
absence, not a gap in research: unlike Ravelin, Forter never says what a caller should do if
Forter is silent.

**Can customers author rules? Yes — but only for abuse, not fraud — [FP].**
`docs.forter.com/extensions/abuse-prevention`: "once policy builder has been enabled in your
forter portal, you can create custom policies and rules." Execution timing is a per-policy
choice: **"forter's abuse policies can be executed pre auth (prior to calling your payment
processor) or post auth (after the payment has been authorized)."** A policy decline response
does carry a reason code: `{"status":"success","action":"decline",
"reasoncode":"merchantpolicylimiteditemabuse","message":"<portal link>"}`, mapped to six named
abuse policy types (reshipper, checkout INR, returns, coupon abuse, limited item, reseller). The
structural point: **Forter's core fraud decision (`forterDecision`) is a vendor-owned ML output
with no published merchant predicate grammar at all**; merchant-authored logic exists only in
this separate, parallel abuse-policy layer with its own action/reasonCode vocabulary. There is no
single unified rule surface the way decider2, Ravelin or Sift have one.

**Latency claim — [FP], and it is 400 ms, not the marketed "one second."**
[forter.com/platform](https://www.forter.com/platform/) states verbatim: **"99% of decisions in
under 400 milliseconds."** The looser "less than a second" phrasing recurs across Forter blog
posts [SEC] but the pinned, sourceable number is 400 ms at p99. `tech.forter.com` (Forter's
engineering blog, which reportedly discusses low-latency decisioning design patterns in more
depth) is **[UNREACH]** — DNS failure (`getaddrinfo ENOTFOUND tech.forter.com`), and the
`web.archive.org` fallback is blocked for this tool.

**vs decider2.**
1. Forter draws a hard line between ML-owned fraud decisioning and customer-owned abuse policy,
   with separate response shapes, separate reason-code vocabularies, and a separate pre-/post-auth
   timing choice for the latter. decider2 instead governs all five rule families — `CF`, `AT`,
   `MS`, `FP`, `AA` — under one rule schema and one approval regime
   (`example_projects/01-transaction-fraud-interdiction.md:704-729`, `:713`). Forter's split is
   evidence worth weighing, not copying: is treating first-party-fraud rules (`FP`) identically to
   card-fraud rules (`CF`) for authorship rights the right call, or does first-party abuse deserve
   the same "customer builds it, vendor doesn't touch the model" separation Forter enforces?
   Flag as an open design question rather than a rule change.
2. Forter's pre-auth/post-auth execution-timing toggle is additional, independent evidence for the
   same gap Ravelin's checkpoints expose: decider2's rule attributes have no execution-stage field
   distinct from `Event types` (`:704-729`). Two vendors independently arriving at a second scoping
   axis is stronger evidence than either alone — see the combined candidate addition below.
3. Forter is the most heavily first-party-documented vendor in this whole survey and it still
   leaves fail-open policy completely unstated. That is useful negative evidence for decider2's
   own posture: even the vendor with the deepest public API reference does not consider a
   fail-open answer optional-but-skippable at this scale, they simply have not published one —
   which strengthens rather than weakens the case for `_recs.md` item 7 doing so explicitly in
   `docs/02-architecture.md §3.6` and `serving/dispatch.py`.

### 1.12 Sift

**Workflow model — [FP], `engineering.sift.com/how-workflows-work/`, Micah Wylde, Jan 2017.**
"We model the workflow as an **acyclic directed graph**." Each node has "an **ordered set of
edges**, each of which has an associated **criteria**… we evaluate the criteria for each edge in
order and traverse the first edge that evaluates to true. Each node additionally has a **default
edge**." Well-formedness is enforced structurally: "They must be acyclic"; "All nodes must be
reachable from the root node"; "All terminal nodes must be decisions (and decisions must all be
terminal nodes)." Three stated design goals, ranked: Consistency, Availability, and
**"Low-Latency — Using the synchronous workflow API should add < 50ms latency at the 99th
percentile over calling the score API directly."** That is an added-overhead SLA stated as a
relative percentile over a baseline call, not an absolute number.

**Decisions API — [FP], `developers.sift.com/docs/curl/decisions-api/overview`.** A Decision is
customer-defined in the console; the API only applies existing ones. `category` ∈ `BLOCK`,
`WATCH`, `ACCEPT`. **Immutability, verbatim: "once a Decision is made, it can't be deleted or
modified. However, you can represent updates by applying a subsequent Decision to that entity."**
So an entity's current state is a derived read over an append-only Decision log, never an
in-place mutation. `source` on apply ∈ `MANUAL_REVIEW`, `AUTOMATED_RULE`, `CHARGEBACK`.

**Score API / synchronous evaluation — [FP].** Score scale is stated inconsistently by Sift
itself: docs prose says "a score between 0 and 100" but the JSON actually returns 0–1
(`"score": 0.898391231245`); the console UI uses 0–100. `workflow_statuses[]` carries `route.name`
— per-run route attribution — and error code **`-3` "Server-side timeout processing request.
Please try again later."** — Sift's own engine-side timeout signal, distinct from Ravelin's
client-side 750 ms recommendation. Rate limits (verbatim table): Events API 500 req/s; **Events
API with synchronous scoring 27 req/s**; Score API 9 req/s; Decisions API 40 req/s.

**List management — [FP], Workflow List API.** Up to **10,000 terms per list**; a term "cannot
start with whitespace," max 100 characters for newly created terms (up to 700 for grandfathered
ones). Concurrency is handled explicitly: **`ETag` on create/update/get + `If-Match` on
update — "the server will reject the update"** on a stale write. Draft management is not exposed
through the public API — only published keyword sets.

**Backtesting — [FP].** Runs "in isolation" over **"up to the last 30 days"** of real Workflow
run data, available "roughly 15 minutes" after processing. Limits: up to 30 unique fields, up to
2,000 Workflow List terms, **1 active backtest per user, 5 per Sift account**, results cached 24
hours and "only visible to the user that ran the backtest." Runtime "roughly 1 minute." Cannot
test multi-abuse-type workflows.

**Sift gaps — [UNREACH]/[SEC].** Per-route live analytics: not in developer docs, only
Help-Center article titles referenced in search results, not opened. No published maximum route
count per workflow. Route-ordering mechanics beyond the engineering blog rest on a blog summary
[SEC], though the underlying mechanism (ordered edges, first-true-wins, default edge) is
independently [FP] from the 2017 design doc.

**vs decider2.**
1. Sift's append-only Decision log — "represent updates by applying a subsequent Decision" — is
   the same shape decider2 already has at the event level: `5.15 Decision record emission`
   (`example_projects/01-transaction-fraud-interdiction.md:563-580`) never mutates a past record.
   What decider2 has *not* stated is an entity-level current-state read analogous to Sift's
   "latest Decision" — relevant to the same open boundary question Ravelin's permanent-override
   evidence raises above (§1.10 lesson 3): is "what is this client's status right now" decider2's
   job, or its caller's?
2. Sift's `ETag`/`If-Match` optimistic concurrency on Workflow List edits has no counterpart in
   decider2's own list/table work. `docs/03-authoring-api.md:720-747` marks the keyed-lookup
   sketch "provisional… lowest-confidence part of this document" and notes the real
   implementation is "BEING BUILT" in `decider2.tables` — concurrent-editor conflict detection is
   not mentioned anywhere in that note. Cheap to add before an editing UI exists; expensive to
   retrofit after two analysts have silently clobbered each other's table edit.
3. Sift's three published DAG well-formedness constraints — acyclic, all nodes reachable from
   root, terminal nodes are exactly the decisions — are a complete, cheap, load-bearing checklist
   for a graph-shaped authoring surface. decider2's own `graph/` layer is, in its own docs'
   words, "SPECIFIED BUT UNVALIDATED … has no empirical support at all" (`docs/00-BUILD.md:160-
   163`). Sift is proof this class of validation is standard practice, not a research problem —
   decider2 should specify its equivalent set (every rule reachable, no path from a `shadow_*`
   population to a terminal, exactly one resolved action per event) as a named, tested pass.
4. Sift's "<50ms added at p99 over calling the score API directly" is a *relative*, added-overhead
   SLA — the same framing decider2's own `EXPERIMENTS.md` already uses internally ("971.4 µs p50
   = 4.86% of a 20 ms budget," `EXPERIMENTS.md:678-770`). Sift is evidence that framing overhead
   as a percentage of a caller's own budget, rather than an absolute number decoupled from any
   budget, is a convention worth publishing externally, not just keeping in the experiments log.

---

### 1.13 Hawk AI

**Two docs.hawk.ai surfaces, two access levels — worth stating up front because it
shapes everything below.** The **API Reference** section (`docs.hawk.ai/apidocs/...`) is
public and returns full OpenAPI JSON with no authentication. The **Guides** section —
rule authoring, entity risk detection, the investigative agent, screening, regulatory
reporting, program management, customer lists, user management — is not: every one of
nine guide pages fetched (`rule-manager`, `entity-risk-detection`,
`investigative-agent`, `screening`, `regulatory-reporting-1`,
`program-management-dashboards`, `customer-cases-3`, `customer-list-1`,
`user-management`) came back as a Document360-hosted page with `<title>Login</title>`
instead of content, and the guides' own API-overview stub returned the literal string
"Not found". **[UNREACH]** This means Hawk's actual rule-authoring surface — the thing
this comparison most wants — is not independently verifiable from outside a customer
login, despite the API reference living on the identical domain. Everything below that
touches rule mechanics is therefore built from (a) the public API's response schemas,
which describe rule *outputs* but not rule *syntax*, and (b) marketing pages, graded
accordingly.

**API surface — decisioning-relevant fields, [FP], read directly from
`docs.hawk.ai/apidocs/*`.** A `POST /v1/transaction-checks` returns synchronously
(`TransactionResponse`): `caseId`, `caseStatus`, a boolean `proceed` ("Should the
transaction proceed (true) or be blocked (false)"), `tenantTransactionId`, and `took`
(an elapsed-time field returned on every call — a small thing, but it means Hawk's API
self-reports its own latency per request, which none of the marketing pages below do).
If a real-time decision can't complete inside Hawk's own timeout the API returns **HTTP
202** and finishes asynchronously, "usually within a few seconds," delivered via
callback (`docs.hawk.ai/apidocs/transaction-checks`, FAQ). Cases can reopen "even weeks
later," and the audit trail is the documented mechanism for reconciling a case whose
outcome changed after the fact.

The richer object is the **case**, fetched with `GET /v1/transaction-checks/{caseId}`
(`get-transaction-case`, `docs.hawk.ai`). `RuleExecutionResults` buckets fired rules
into `openRules` / `notOpenRules` / `autoClosedRules`; each `RuleInstanceResult` carries
`instanceId`, `instanceName`, a `blocking: boolean`, a numeric `value` ("score or value
produced by the rule instance"), `ruleCategories`, and two free-form bags — `details`
and `configuration` — explicitly documented as "rule-specific… fields vary by rule."
That last part matters: Hawk does not publish a closed rule-instance schema, it
publishes an envelope and defers the interior to whatever the (unreachable) rule manager
built. `Decision` carries `transactionStatus` (`proceeded|blocked|unblocked|none`) and
`decisionType` (`complete|incomplete|immediate`) — i.e. Hawk distinguishes a final
decision from a provisional one at the type level. Screening hits are two-tier: a
deduplicated `screeningResults` list (type only, e.g. `SANCTIONED_PERSON`) kept "for
backward compatibility," and richer `screeningHits` with `matchScore`, `listSource`,
`screeningAreas`, and a `sanctionsOwnership: boolean` that "distinguishes SAN from SCO
for sanctions hits" — direct-list match versus ownership/control match, a real
sanctions-ops distinction most vendors in this set don't expose as a typed field.

The customer-side check (`perform-check-on-customer-v2`, `get-customer-case`) returns
`riskLevel` (an open string, not an enum — "should be treated as open sets," per
`customer-checks.md`'s own FAQ), `totalPoints` (int32 — a points-based scoring model),
`riskFactors` (array of factor-name strings), and `flaggingResults` (a `flag` plus
`countriesMatched`). A **206** response is documented for "case found but some data
sources are unavailable" — Hawk returns partial data rather than failing the call
outright, a graceful-degradation choice worth noting for anyone designing decider2's own
behaviour when an upstream table is stale (`docs/04-observability-and-governance.md`
doesn't yet specify an equivalent partial-response contract). No `velocity`,
`aggregate`, or `window` keyword appears anywhere in any of the fourteen OpenAPI specs
fetched — consistent with velocity logic living entirely inside the login-walled rule
engine and never being expressed in the wire schema.

**Marketing claims, graded separately — [FP], hawk.ai marketing domain, marketing
register.** Architecture: "multi-tenant SaaS" on "Kubernetes, Containerd, Spring
Boot/Cloud," with "reactive communication utilizing Apache Kafka," "processing billions
of transactions in real time" (`hawk.ai/technology/tech-stack`). Two separate,
inconsistent-scope latency/throughput numbers: "**True real-time response (150 ms
average)**" on the FRAML page (`hawk.ai/solutions/framl`), and "**Real-time, high
throughput model performance (30,000 TPS)**" specifically on the **AML AI Overlay**
product page (`hawk.ai/solutions/aml-ai-overlay`) — the overlay is a bolt-on scoring
layer over an *existing* AML system, not necessarily the same code path as core
transaction monitoring, so the two numbers should not be quoted as one system's figure.
The overlay page also claims "up to a 70% reduction in false positives and 3-5x increase
in detection precision," a named tier-1 bank case study (+88% prediction accuracy, −85%
false positives, +2× threat detection in 3 months), and "model pipeline for
collaborative development and model retraining in **<1 day**."

**Governance language — the "4-eyes" claim, precisely sourced.** There are *two*
different 4-eyes claims on hawk.ai and they should not be conflated. The
**development-process** one is about code review: "Backed by processes: 4-eyes principle
built into development cycle through code review as a mandatory step"
(`hawk.ai/technology/technology-behind-hawk`). The **operational-governance** one — the
one relevant to a bank's segregation-of-duties requirement, and the one this report's
earlier citation of "Hawk's 4-eyes reviews" refers to — is on the FRAML page: "Program
controls & quality assurance: Ensure analysts conduct all necessary tasks and make
accurate, quality decisions with **automated workflows, 4-eyes reviews, and alert
sampling processes**" (`hawk.ai/solutions/framl`). Neither page documents what a 4-eyes
review actually requires (which change classes, who the second reviewer must be, whether
it's enforced or advisory) — it is a governance *claim*, not a governance *mechanism*
description, unlike the two-eyes/UDV mechanism FICO documents for Falcon elsewhere in
this report.

**Federated learning — real claim, but framed as a roadmap rung, not a shipped
feature.** "Decentralized Learning (also known as Federated Learning) - Refinement of
model performance based on decentralized data supplied by different external
institutions" appears as the *third of four* "distinct levels of information sharing,"
introduced with "we see four distinct levels… each improving the detection accuracy over
the last" (`hawk.ai` AI/Data Science page, branded "HAWK:AI" — an older, unrefreshed
page). The prose never states which level is live for which customer; treat "Hawk does
federated learning" as aspirational-vendor language, not a verified capability, pending
a page that says a specific bank is on it.

**Shadow-mode / sandbox — the best-evidenced operational claim on the site.** The
Production Sandbox page describes a **production-parity** test environment: "duplicate
settings" from a live tenant, "testing with live production data," and explicit
**"above-the-line (ATL) and below-the-line (BTL) testing"** — the same ATL/BTL
vocabulary Unit21 uses independently below (`hawk.ai/platform/production-sandbox`). The
five-step flow (clone tenant → adjust rules → select a sample/timeframe/alert-status
subset → run → compare sandbox vs. production) is described in enough procedural detail
that this is graded **[FP]**, not aspirational — it reads like a documented product
feature, not a slogan.

**vs decider2.** Three transfers. (1) **Hawk's `blocking: boolean` + free-form
`details`/ `configuration` per rule instance is a weaker contract than decider2's, and
that's useful to know**: decider2's rule document carries eighteen governed attributes —
severity, action, priority, status, effective-from/to, event types, segments,
overlay-exempt, critical, suppressible, reason code, queue/SLA tier, challenge type,
stale/absent behaviour, max fire rate — each with its own approval class, mostly
**Four-eyes** (`example_projects/01-transaction-fraud-interdiction.md:702-728`). Hawk's
public schema proves only that a rule instance is a name plus a blocking flag plus an
opaque bag; decider2 already publishes the richer, closed version of the same idea, and
the four-eyes-per-field governance model (not just "4-eyes reviews" as an unspecified
process claim) is exactly the sharper machine decider2's O17 open question is trying to
settle (`docs/06-open-questions-and-experiments.md:204-207`). (2) **decider2's
shadow-rule isolation is a stronger, structural claim than either of Hawk's testing
surfaces.** Hawk's sandbox is production-parity but runs in a *separate tenant*;
decider2's spec requires shadow rules to run *in the same live pipeline* and be
structurally incapable of changing `action_code`, `fired_rule_ids`, or the reason set
(`:434-457`, "must make it structurally impossible rather than merely discouraged") — a
stronger property than "a copy of production you can break safely," and worth stating in
decider2's own docs as a differentiator once §5.11 is actually built (it isn't yet;
recommendation #12 in `_recs.md`). (3) **The `took` field is a small, adoptable idea
decider2 doesn't have**: Hawk returns per-call elapsed time in its own response payload.
decider2 measures its own tail exhaustively in `EXPERIMENTS.md` but does not expose a
per-decision timing field in the audit record spec at
`docs/04-observability-and-governance.md:239-253` — trivial to add, and it turns "was
this call slow" from a question requiring log correlation into a field on the record
itself.

### 1.14 Unit21

**docs.unit21.ai is confirmed login-walled — exact mechanism.** Fetching
`docs.unit21.ai/llms.txt` and `/llms-full.txt` both return a Unit21-branded magic-link
sign-in page (`<title>Sign in | Unit21</title>`, "Use your email to request access… We
sent you a sign-in link"). **[UNREACH]** Separately, `www.unit21.ai/llms-full.txt` 404s
to a generic Webflow "Page Not Found" template — that URL was never a real page. Both
failure modes are clean and unambiguous; nothing here should be treated as "docs exist
but I read them wrong." Everything below is therefore from `www.unit21.ai` marketing
pages and two engineering- register **blog posts**, which is a materially better source
than pure marketing copy because both blogs explain mechanics in some depth, not just
outcomes.

**Rule engine — three rule types, [FP], from the velocity-rules blog
(`unit21.ai/blog/.../featuring-velocity-rules`, published 2 Sept 2025, byline Gal
Perelman, Product Marketing Lead).** **Scenario rules**: guided templates for named
typologies (layering, structuring, dormant activity) — "fill in the blanks," least
customizable, easiest to deploy. **Dynamic rules**: "build logic using variables and
trigger conditions," explicitly not limited to transaction data — "logins, PII updates,
password changes, or any other system event" can feed a rule. **Real-time rules**:
evaluated inline through "Unit21's real-time API endpoint," optimized for low latency,
used for blocking or step-up decisions. This is a materially more specific rule taxonomy
than most of this report's vendors publish sight-unseen of documentation.

**Rule modes — three, and the vocabulary matters.** **Live** (active, alerting).
**Validating** ("run against historical data… fine-tune thresholds… without impacting
production alerts" — explicitly usable for "A/B testing (above-the-line/below-the-line
testing)," the same ATL/BTL term Hawk uses independently). **Shadow** ("active but do
not generate alerts… silently evaluate live data on a schedule" — for "performance
benchmarking, A/B testing, and rule refinement"). Unit21's "shadow" is therefore
live-traffic-but-silent, and its "validating" is historical-replay — the two concepts
decider2 keeps separate as backtest (§5.17) versus shadow rules (§5.11) are given the
same two names independently by a second vendor, which is some evidence the split is the
industry's actual mental model, not decider2 inventing structure.

**Velocity rules — mechanics quoted precisely, [FP].** "A variable could represent the
total amount sent in the last three days or the number of transactions in the past week.
Trigger conditions then compare these variables against historical baselines." The
worked example: "Create a variable for 'Sum of Transactions Over Last 3 Days.' Create
another for 'Sum of Transactions Over Last 30 Days.' Set a trigger condition: alert if
the 3-day total exceeds the 30-day average by 100 percent." This is genuinely a
windowed-aggregate authoring surface exposed to a rule author — the same category FICO's
UDVs/UDPs occupy elsewhere in this report, though Unit21's is documented at the
marketing-blog level of detail (named windows and a ratio trigger), not a published
grammar or function list. Five worked use cases are given with concrete thresholds
(structuring: ">5 deposits in 48h totalling >$9,500"; ATO/fast-cashout: "new device/IP
then >3 transactions above $2,000 within 30 minutes"; dormant-to-hyperactive: "no
transactions for 90 days, then >$10,000 withdrawn within one day"). The blog also
documents chaining a scheduled/retrospective velocity rule into a real-time rule's
trigger condition ("the entity triggered a velocity rule alert in the last 72 hours") as
an explicit pattern, not just a hypothetical.

**Graph-Based Rules (GBRs) — mechanics found, extending this report's existing
`_unverified.md` / `_inventory.md` entries, [FP],
`unit21.ai/blog/the-network-strikes-back...`, published 20 June 2025.** GBRs are "part
of the built-in rule engine, no need to purchase additional components," have "no code
interface," and support **shadow mode** for testing before deployment (the same
three-mode vocabulary as above). Mechanically: entities that share device/IP, bank
account, or phone-number attributes are linked; once an entity is flagged, it can be
"labeled with tags that allow rules to be automatically applied" to everything connected
to it — i.e. GBRs are tag-propagation over a link graph, triggered from existing
entity-resolution signals, not a general graph-query language. Three worked scenarios
are given (marketplace ban-evasion via device/IP overlap; lending-fraud rings linked by
shared bank accounts and phone numbers; ACH layering across "multiple newly created
accounts"). This is enough to say the *feature* is real and roughly how it works, but
there is still no published syntax for writing a GBR — the "no-code interface" is
described, never shown.

**Real-time latency claim, [FP], no methodology — flag exactly as this report already
flags Forter's "under 1 second."** "Our real-time monitoring evaluates transactions in
under 250 milliseconds, meeting the speed demands of FedNow, RTP, Zelle, and instant
payments" (`unit21.ai/products/real-time-monitoring`). No percentile, no measurement
methodology, no distinction between p50 and worst case — same evidentiary weight as
Feedzai's marketing "3 milliseconds" claim already flagged in `_unverified.md`, i.e.
treat as a target, not an SLA.

**Other decisioning-relevant surfaces, [FP], briefly.** **Device Intelligence**: a
"transparent 0–100 Device Risk Score" with "visible signal composition" — explicitly
"glass box, not black box," fed into the same rule builder as transaction rules.
**Customer Risk Rating**: segment-based (e.g. Enterprise vs. Retail), weighted risk
categories, "Integrate CRR scores into rules to automate alert generation" — CRR as an
ordinary rule input, the same pattern as a model score elsewhere in this report. **Fraud
Consortium**: "shared, cross-institution intelligence covering 80M+ U.S. consumers"
(rtfraud page) versus "a network of 100+ institutions" (fintech page) — two different
units (consumers vs. institutions), not necessarily inconsistent, but not directly
reconcilable either. **AI investigation agent**: customer-facing numbers ("1.5M+ alert
reviews run by AI Agents to date," "up to 93% fewer false positives," "up to 80% faster
investigation handle times") with no stated baseline or measurement window — read as
vendor-supplied outcome claims, not audited figures. **Security**: SOC 2 Type I and Type
II (auditor: Armanino), GDPR/CCPA compliance, Auth0-backed authentication, mandatory MFA
on production access, 256-bit AES at rest / TLS in transit — standard-but-verifiable
claims, all [FP].

**vs decider2.** (1) **Unit21's three rule modes independently corroborate decider2's
shadow- rule design**, but decider2's spec is stricter: Unit21's shadow rules run
silently on live traffic (an operational convention), while decider2 requires shadow
isolation to be a *provable* property — "no shadow rule has ever changed an
`action_code`… demonstrated continuously, not asserted" (spec criterion 4, cf.
`_recs.md` #12's "lineage assertion in `graph/`"), which no vendor evidence gathered
anywhere in this report, Unit21 included, claims to do. (2) **Velocity rules are exactly
the capability `_inventory.md`'s "windowed aggregates as declared inputs" row calls
decider2's differentiator** — Unit21 lets an analyst *name* a window and a comparison
ratio in a UI, but the underlying value still arrives with no documented watermark,
staleness state, or fresh/stale/absent contract; decider2's spec already requires
exactly that per window (`:126-140` — a real capability gap in decider2's own package,
not the spec). (3) **Graph-Based Rules is real evidence for `_inventory.md`'s existing
"Graph / link features" row** ("Unit21 ('Graph-Based Rules'), DataVisor," currently
marked "no"/"no" for decider2 and adoption) — worth deepening that row with
"tag-propagation over an entity link graph, triggered from device/IP/bank-account/phone
overlap" rather than leaving it as a bare product-name citation, while keeping the adopt
decision at "no": decider2's frame tier (`docs/02-architecture.md:20-25`) explicitly
defers `window` and has never claimed graph traversal, and nothing in the fraud spec's
own scope note contradicts that boundary.

### 1.15 Mastercard Brighterion / Decision Intelligence

**This is the weakest-evidenced vendor in the set, and it's an access-control artefact,
not an absence of documentation.** Every developer-facing surface failed:
`b2b.mastercard.com` (the DI/Brighterion product page and the "Decision Intelligence
Beyond Mastercard" playbook PDF) is a **DNS failure**; `brighterion.com` 301-redirects
into that same dead host — meaning Brighterion no longer has an independently reachable
site at all; `developer.mastercard.com`'s three fraud/DI documentation and API-reference
paths are **JS shells** that return only the literal string "Mastercard Developers";
`www.mastercard.com/.../risk-decisioning.html` is **HTTP 403**; and the
Adjaoute/Brighterion patent family (US2015/0046332A1, US11348110B2, US2015/0032589A1) is
unreachable on every patent host tried (`patents.google.com` 503 on all attempts,
`patentscope.wipo.int` 403, `freepatentsonline.com` connection reset). **[UNREACH]**,
all of it, with those exact failure modes.

**What did work is Mastercard's own newsroom, [FP], and it's enough for four hard
numbers.** The Decision Intelligence Pro launch release (`newsroom.mastercard.com`, 1
Feb 2024): DI Pro scores a transaction "in **under 50 milliseconds**," scanning "**1
trillion data points**," improving fraud detection "**20% on average**, as high as
**300%** in some instances," and reducing false positives "**by 85%**," against a base
of "**143 billion** transactions annually." The architecture description is deliberately
vague — "assessing the relationships between multiple entities surrounding a
transaction" — with no mention of graph, transformer, or recurrent methods. A second,
separate release (22 May 2024, Cyber Secure / compromised-card detection — not DI Pro)
reports a doubled detection rate and up to 200% fewer false positives for a *different*
product; a third (18 July 2024, AI Garage) confirms "gen-AI and graph technology"
language but adds no new numbers. **These three releases must not be conflated** — only
the February release carries the 50 ms / 1 trillion / 20–300% / 85% figures.

**Brighterion's "smart agents" architecture is documented only in patent search
snippets, not retrievable full text.** The one substantive passage obtainable:
"smart-agents update their profiles by adjusting normal/abnormal thresholds or creating
exceptions," describing a self-spawning, per-entity stateful actor with adjustable
thresholds and per-entity exceptions — the closest thing in this entire comparison to
"one stateful actor per entity" as an architecture, but unverifiable beyond that one
sentence because every patent full-text host failed. **[SEC, snippet-only]**

**Bottom line, and it's short on purpose.** DI/DI Pro is a **network-side scoring
service** injected into the authorization flow — public search-snippet evidence
(unverified on any Mastercard page directly) describes it as adding a risk score to the
ISO 8583 authorization message, not a customer-authorable rule engine. There is no
public rule language, no policy DSL, and no reachable API reference for a bank
integrator to read. If decider2's own docs ever cite Mastercard DI as a comparison
point, the only defensible citation is the four February 2024 numbers above — everything
else about Brighterion's mechanics is inference from an unreadable patent family.

**vs decider2.** The contrast worth making is about *transparency of the decision
itself*, not architecture. Mastercard's public description of DI Pro's output is a
single score with an unspecified basis ("assessing relationships between multiple
entities"); decider2's spec requires every decision to carry a complete, ranked reason
set from *all* fired rules, not just the winner
(`example_projects/01-transaction-fraud-interdiction.md:458-473`,
`docs/04-observability-and-governance.md:175-180` — "reason codes need no new
machinery," and the explicit warning that a ported legacy system's decline-reason
taxonomy was once dropped in translation). A bank plugging a network-side score like DI
Pro into decider2 would do exactly what the fraud spec's model contract already
anticipates — treat it as one more scored input with a declared absence behaviour
(`:270-289`) — which is a stronger, more auditable position than anything publicly
documented about consuming DI Pro's score today.

### 1.16 ACI Worldwide

**Evidence is thin and mostly a mix of first-party datasheets plus one analyst report
ACI itself hosts and distributes — shorter treatment here reflects that, not any attempt
to pad it.** No developer portal, API reference, or rule-syntax page exists publicly for
ACI; this is the sharpest contrast with, e.g., AWS Fraud Detector's fully public grammar
and quota tables in this same report. **[UNREACH by absence]** — there was nothing to
fetch, not a page that failed.

**What ACI does publish, [FP], is a set of PDF datasheets and product pages making
incremental-learning and outcome claims with no latency or TPS figures.** "ACI advanced
AI machine learning uses incremental learning algorithms… allows machine learning models
to adjust to new behaviors, without the need to relearn everything they already know"
(merchant-fraud datasheet, doc code AFL2104 03-25, copyright 2025); "outperformed
traditional models by more than 10%" within three months (repeated across the merchant
and billers datasheets, both footnoted "Source: ACI customer data"); "More than 18
vertical-focused models"; **8,000+ AI features** (2025 merchant datasheet) versus
**9,000+ AI features** (2024 billers datasheet) — a datasheet-to-datasheet drift worth
flagging if either number is quoted. Web pages add ">85% Detection rates," "3:1 False
positive rate," and "patented incremental learning capabilities" deployed on Microsoft
Azure — but, again, **no datasheet or product page states a latency or TPS number.**

**The only latency/TPS numbers anywhere in ACI's material come from an analyst-authored
PDF that ACI hosts and distributes, and the report says so on its own cover page**:
"Beyond Point Solutions: Orchestrating the Future of Fraud Prevention," Datos Insights,
May 2025, authors Jim Mortensen and Gabrielle Inhofe, page 1 reading "This report
provided compliments of:" followed by the ACI logo — hosted at
`aciworldwide.com/wp-content/.../Datos-Insights-....pdf` behind a `?gated=` parameter.
**[FP fetch, secondary authorship — grade the content as vendor-supplied-to-analyst, not
vendor-published.]** The two figures: "response times under 300 milliseconds" (generic
claim), and a named case — "a large financial institution in India with processing
volume rising to over **6,000 transactions per second with sub-100 millisecond
latency**" — sitting in the report's "Datos Insights' Take" section, i.e. analyst
commentary, not the vendor specification table. The two numbers are in tension (generic
<300ms vs. one reference site at sub-100ms) and neither should be quoted as ACI's
general SLA.

**The one named rule-authoring surface, from the same report:** "**ACI Model
Generator**: A citizen data-science workbench enabling business users to rapidly create
and adapt predictive AI and ML models **along with business rules** without requiring
data-science expertise." No syntax, screenshot, or example is published anywhere. The
same report documents a shadow/simulation capability — "The platform allows users to run
AI models in parallel and to test models in simulation mode before deployment" — which
is ACI's answer to shadow mode, but again only analyst-relayed, not a first-party ACI
page.

**vs decider2.** Two contrasts. (1) **ACI bundles the rule surface and the
model-training surface into one workbench ("business rules" and "predictive AI and ML
models" in the same sentence, same tool); decider2 deliberately keeps them apart** —
derived features are registered, typed steps rather than expressions a rule author
writes inline (`docs/08-configuration-and-lifecycle.md:265-315`, removing
`_ComputedFeature`), and a model score enters the rule tier only as an ordinary input
column, never authored logic. Given ACI publishes no syntax for its combined tool,
there's no way to know whether "business rules" in the Model Generator means anything
richer than threshold tuning — decider2's separation is the more defensible position
purely because it's the one with a published boundary. (2) **ACI's "simulation mode" is
a single-sentence claim with no metric set; decider2's batch backtest mode already has
one, on paper** — hit rate, precision, incremental catch, false positive rate, per-rule
overlap, value blocked, operational load, and action churn, run both with and without
the overlay stack applied
(`example_projects/01-transaction-fraud-interdiction.md:621-640`). Neither ACI's nor any
other vendor's shadow/simulation claim gathered in this report comes with a comparably
explicit metric list — which makes §5.17, once built, a genuine differentiator rather
than parity with what vendors already ship.

---

## 2. Feature inventory

One row per capability the fraud engine set demonstrates. "decider2 status" is graded against the
**package** first, then the spec/mock — a capability that exists only in `example_projects/` is
marked *no (spec only)*, because the mock's imports (`decider2.ruleset`, `temporal_table`, `fan`,
`stable_blocks`, `decider2.governance`) are not in the package. New rows contributed by DataVisor,
Sardine, Stripe Radar, AWS Fraud Detector, BioCatch, Ravelin, Forter, Sift, Hawk AI, Unit21,
Mastercard Brighterion and ACI Worldwide are folded into the original four subsections rather than
appended separately, since each belongs with a specific axis.

### 2.1 Feature computation

| Feature | Which engine(s) | What it does | decider2 status | Adopt? |
|---|---|---|---|---|
| Windowed velocity aggregates computed **inside** the engine | Feedzai Railgun, Hawk, Sift | count/sum/distinct per key over a sliding window, maintained as engine state | **no** — explicitly out of scope (`example_projects/01-transaction-fraud-interdiction.md:1097-1099`); frame tier has join/aggregate/filter only, `window` deferred (`docs/02-architecture.md:20-25`) | **no** — keep it upstream, but say so as a product boundary, not a scope note |
| Windowed aggregates as **declared inputs** with a watermark and a per-window staleness tolerance | none found publishing this | value + freshness state travels with the feature | partial (spec only) — `:126-140` defines fresh/stale/absent per window; nothing in `boundary/` carries a watermark | **yes** — this is decider2's differentiator over every vendor in this survey |
| **Closed, published catalogue of velocity attributes** — enumerated names, no user-defined aggregate | Stripe Radar (513 attribute rows, 485 distinct names, keying limited to 9 named dimensions) | rule authors pick from a fixed list; nothing new can be defined without a platform release | decider2's spec is the opposite: an analyst asking for an undefined window is scenario 10 (`:1066-1069`), explicitly acknowledged as costly | **no** — decider2's open, declared-input model is already the better trade; cite Stripe as the industry's dominant (worse) pattern |
| **Tumbling-bucket velocity approximation** with undocumented slack (`hourly` up to 65 min, `daily` up to 25 h, `yearly`/`all_time` up to 366 days) | Stripe Radar | O(1) bucket storage traded for window imprecision, with the slack undocumented outside one paragraph | decider2's exact, watermarked windows with per-window staleness tolerance (`:126-140`) are already strictly better | **no, but document the contrast** — a real, marketable advantage over the industry's dominant pattern |
| **Saturating/bounded counters** — a fourth value-state ("saturated") where a counter stops growing at a cap and `> cap` can never fire again | Stripe Radar (`card_count_for_*`, `dispute_count_on_*`, capped at 25) | bounds memory/compute on high-cardinality counters at the cost of silently-wrong comparisons above the cap | **no** — decider2's velocity contract has only fresh/stale/absent (`:249-266`); no saturated state is named | **yes** — cheap addition to the input-contract vocabulary; protects against a real bug class if any upstream aggregate store is ever bounded |
| Behavioural profile / per-entity state carried between events | FICO Falcon, Featurespace ARIC | long-lived per-card/per-account state updated each event | **no** — decider2 is stateless per record by construction | **no** — belongs in the streaming/profile tier |
| **A stateless rule layer over a stateful, side-effecting model layer** (the model call itself mutates per-entity state; the rule language can never read those aggregates directly) | AWS Fraud Detector (TFI/ATI: `GetEventPrediction` "dynamically updates this value each time you call" it) | confirms, independently of decider2, that this exact split — pure rules, stateful model — is the industry answer | **already asserted** — decider2's "stateless per record by construction" boundary is independently corroborated by a hyperscaler product landing on the same split | already have; cite as corroboration, not a gap |
| Graph / link features — **tag-propagation over an entity-link graph** | Unit21 ("Graph-Based Rules": entities linked by shared device/IP, bank account or phone number; once one entity in a cluster is flagged, tags propagate and auto-apply rules to the rest), DataVisor ("Knowledge Graph") | shipped inside the base rule engine, no-code interface, testable in shadow mode, but no published query/authoring syntax | **no** — out of scope per `docs/02-architecture.md:20-25` (graph/window explicitly deferred) | no — out of scope; a data-platform capability, not a rule-engine one |
| Named windowed-comparison velocity rules ("name two time-bound sums and a trigger ratio," e.g. "3-day total exceeds the 30-day average by 100%") | Unit21 (dynamic + velocity rules, marketing-blog depth, no published grammar) | an authoring-level "name a window, name a comparison" primitive for a non-engineer | partial (spec only) — `:126-140` defines the fresh/stale/absent contract per window but offers no such authoring-level primitive | **no as framework machinery** — decider2's removal of `_ComputedFeature` (`docs/08-configuration-and-lifecycle.md:265-315`) argues against exactly this; but useful confirmation the underlying watermark/window contract is worth building well, since a UI would eventually need to express it |
| Device/behavioural-biometric signal bundle (100+ device signals, keystroke/mouse/session biometrics) | DataVisor (dEdge), Sardine (DIBB), BioCatch (risk/genuine factors, threat indicators) | a pre-computed risk signal ingested like any other feature | **n/a** — decider2 is a library; an upstream system would supply this as an ordinary input column | no — not an engine-kernel capability |
| Negatively-weighted "genuine behaviour" input class (evidence *against* fraud, not just for it) | BioCatch ("Genuine factors": fluid typing, high data familiarity) | lowers a resolved severity symmetrically to how risk-additive features raise it | **no** — decider2's rule model reads risk-additive features only; nearest analogue is a critical `allow` rule suppressing firings (`:475-481`), which is rule-based, not feature-based | maybe — worth naming as a design idea, not concrete enough yet for a recommendation |
| A **feature catalogue** that says which field exists on which event variant, from when | none publishing it | validates a rule against the event types it claims to cover | no (spec only) — FRAMEWORK-DEMANDS #4/#5; package has one flat input schema | **yes** — 12 event types × 210 fields cannot share one schema |
| Derived feature as a registered, typed, testable step (not an expression string) | none — vendors all use expression strings or UI builders (DataVisor's Vera is an LLM front end onto an unpublished representation) | a rule leaf refers to a feature by id | **yes** — `docs/08-configuration-and-lifecycle.md:265-315` settles this; `params.py` harvests the signature | already have; keep |

### 2.2 Rule authoring

| Feature | Which engine(s) | What it does | decider2 status | Adopt? |
|---|---|---|---|---|
| Flat rule set evaluated **all-match**, firing set as output | every bank-grade engine; AWS `ALL_MATCHED` | returns the set of rules that fired, not one row | **no** in package — `DecisionTable` is first-match-with-default (`tables/schema.py:560-565`); `all` exists only as a benchmark mode in `EXPERIMENTS.md:381-390` and in decider 1 (`decider/modules/rules/flat_rules/module.py:168-170`) | **yes, must-have** |
| First-match / priority hit policy | AWS `FIRST_MATCHED`, Sift (first true edge), DMN | ordered evaluation, stop at first hit | **yes** (first-match) — `tables/schema.py:560-565` | already have; rename to the field's vocabulary |
| **Ordered action-class hit policy** — partition into ranked classes, first-match-wins *within* the winning class, order-unspecified within it | Stripe Radar (3DS → Allow → Block → Review, "rules of the same action type aren't ordered") | a third, distinct hit-policy shape from `first_match`/`all_match` | **partial (spec only, hand-written)** — decider2's own action resolution already does this by hand: a seven-rank severity table with a critical-rule override and a four-level tie-break (`example_projects/01-transaction-fraud-interdiction.md:463-500`, §5.12) — but it is prose, not a named, reusable hit-policy kind | **yes, must-have** — Stripe is evidence this shape is common enough to generalise, not bespoke |
| AWS `ALL_MATCHED`'s outcome union with **no documented conflict resolution** | AWS Fraud Detector | when `all_match` returns multiple outcomes, nothing de-duplicates or ranks them | cautionary data point for decider2's own `all_match` design (rec #1) | n/a — a warning, not a row to adopt |
| Rule **priority** and **severity** as first-class rule attributes | Stripe (action priority), Actimize, Sardine | orders the firing set without changing predicates | no (spec only) — `:704-729` (18 governed attributes per rule) | **yes, must-have** |
| Action / outcome taxonomy incl. step-up and review | Stripe (Request 3DS / Allow / Block / Review), BioCatch (allow/decline/authenticate/review), Sardine, Actimize | the engine's own outcome enum | no (spec only) — 7 actions at `:465-473` | **yes** — but as a project vocabulary, not framework (spec Q16) |
| **Two-tier screening-hit detail** — a deduplicated hit-type list plus an enriched per-hit record distinguishing a direct sanctions-list match from an ownership/control match | Hawk AI (`screeningResults` type-only list "for backward compatibility" vs. `screeningHits` with `matchScore`, `listSource`, `screeningAreas`, and a `sanctionsOwnership` boolean distinguishing SAN from SCO) | lets a compliance team route a direct hit differently from an ownership/control hit | **no** — the fraud spec's sanctions gate (`:317`) treats a sanctions hit as one hard-block outcome with no SAN/SCO distinction | **maybe** — a real compliance distinction, worth a field on the sanctions gate's output if compliance needs to route them differently |
| **Score-mutation rule outcome** — a rule rewrites the score itself, not only the decision | BioCatch (uniquely in this survey: "Score = customizing risk score output") | lets a rule correct a value consumed downstream of the decision | **no** — decider2's 7-action taxonomy is decision-only (`:465-473`); the overlay mechanism changes what a rule *reads*, never what score is *emitted* | **maybe** — real if any downstream consumer uses the score independent of `action_code`, otherwise scope creep |
| Rule **effective-from / effective-to** dating, enforced | none publishing enforced expiry on rules; SEON/Stripe have versions | a rule stops affecting outcomes at an instant, provably | **no** | **yes, must-have** for a bank |
| Rule **enabled** flag that costs nothing when off | implied by every UI | turning a rule off is free | **no** — measured: a disabled rule costs full compile (`EXPERIMENTS.md:417-425`) | **yes** — `enabled` must mean "not emitted" |
| Rule sets / rule grouping by family with owners | Actimize, DataVisor (rulesets), Sardine | governance and precedence unit | no (spec only) | yes |
| **Rule-level execution-stage / checkpoint scoping**, orthogonal to event type or abuse category | Ravelin (checkpoint × category matrix: `accountRegistration`, `login`, `checkoutPreAuth`, `checkoutPostAuth`, `refundRequest` × `payment`/`ato`/`refundAbuse`/`supplier`), Forter (pre-auth/post-auth execution timing for abuse policies) | binds a rule to a named point in the customer journey, independent of the event's own type | **no** — rule attributes have `Event types` and `Segments` but no journey-stage axis (`:704-729`) | **yes** — two vendors independently needed a second scoping axis beyond event/category |
| **Named request-scoped evaluation gate** that selects which rule/model families run and doubles as the billing unit | Sardine (`checkpoint="AML"` on the `/customers` call) | a coarse, per-call flag rather than a per-rule attribute | **n/a — already more expressive** — decider2's `Event types` is scoped per-*rule*, not per-*call* | **no** — adopting a coarser flag would be a regression |
| Allow/deny **lists** and watchlists as a first-class engine object | AWS `@list` (100,000 entries × 320 chars, ≤3 lists/rule, auto-incorporates edits with no rule change), Stripe `@alias` value lists (50,000-item cap), Ravelin Datalists, Actimize Platform Lists, DataVisor/Sardine (block/allow lists, mechanics undocumented behind login walls), SEON | membership test inside a rule; list edited without touching rules | **no** — `docs/03-authoring-api.md:720-747` is "provisional… lowest-confidence part"; the real `tables/` is a decision table, not a keyed set | **yes, must-have** |
| **Membership as at an instant** over a 2 M-row list refreshed hourly, for 7 years | none found (every vendor list is present-tense only) | "was X on the list at 14:22:03 on 3 March?" | no (spec only) — `temporal_table`/`IntervalStore`, FRAMEWORK-DEMANDS #17 | **yes, must-have** for disputes |
| Overlay / sensitivity dial that tunes a family without editing rules | Actimize (policy tuning), FICO (strategy), none with enforced expiry | per-record effective threshold | **no** — `docs/02-architecture.md:643-644`: "params don't vary by record"; FRAMEWORK-DEMANDS #10 calls it "the single largest new surface" | **yes, must-have** (or state it out of scope) |
| Natural-language / LLM rule generation | Sardine (Vera), Ravelin (AI rule generator, Jun 2024), Forter (policy-writing agent), DataVisor, Stripe (Radar Assistant), SEON | analyst describes intent, engine writes the rule | **no** | maybe — cheap once the rule document schema exists; DataVisor's Vera FAQ ("human review built into every step before changes go live") is the right minimum bar, and a closed typed schema is a *better* target for LLM output validation than an expression-string DSL |
| Rule scheduling (only active in a window of day/week) | — | | **no** | no |
| Expression string DSL in the rule | AWS DETECTORPL, Stripe Radar DSL, Sardine, SEON | one boolean expression per rule | **deliberately no** — `docs/08-configuration-and-lifecycle.md:265-315` removes `_ComputedFeature`/`simpleeval` | no — decider2's closed algebra is the better trade |
| **One-click rollback to a prior published version** | DataVisor ("full version control... one-click rollback to a prior published version") | restores the previous live configuration without a new deploy | **already have** — `runtime/serve.py:305-315` (`ServeHandle.rollback`, "popping the history stack"), exposed at `serving/dispatch.py:144` (`POST /rollback`); scoped to values, not structure | already have — one of the only rows in this survey where decider2's *package*, not just its spec, matches the vendor claim |

### 2.3 Rule lifecycle, analytics, operations

| Feature | Which engine(s) | What it does | decider2 status | Adopt? |
|---|---|---|---|---|
| **Backtest** a proposed rule on stored traffic | Stripe (6 months of charges), Sift (30 days, ~15 min lag, 1 min runtime), Ravelin ("if it had been active in the past week"), DataVisor, Sardine, SEON, Unit21, Actimize, Feedzai | hit rate / precision before deployment | partial — the *engine* runs batch (`runtime/invoke.py`), but no backtest harness, metric set or decision pack exists | **yes, must-have** |
| Backtest ≡ production **proved** | none | the same artefact answers both, asserted | **yes, uniquely** — `runtime/modes.py:1-33` equivalence ladder, `testing/equivalence.py` | already have; make it the headline |
| **Shadow mode** with structural isolation | Sardine (published feature, no enforcement mechanism), DataVisor (published feature, no enforcement mechanism), Unit21, Hawk | candidate rules fire but cannot change the outcome | **no** — spec §5.11 (`:434-457`) requires it be "structurally impossible rather than merely discouraged"; nothing enforces it, and no vendor in this survey publishes an enforcement mechanism either | **yes, must-have** — decider2's bar is already ahead of the entire vendor set on paper; building it would put decider2 ahead of the field on this property, not merely catching up |
| Champion / challenger, A/B | Sardine, DataVisor, FICO, SAS, Pega | route traffic between strategies | **no** | nice-to-have |
| Per-rule **hit rate / precision / FPR** analytics | Stripe (rule performance chart, "Est. false positive rate," per-rule and per-set), Sardine, DataVisor, Actimize | is this rule earning its place | **no** — `:590-620` §5.16 specifies the join; no rollup primitive exists | **yes, must-have** |
| **Calibrated counterfactual FPR/precision estimator for suppressing actions** (decline/block), substituting a score-calibration curve for the ground-truth label a suppressed event will never produce | Stripe Radar ("Est. false positive rate," calibrated via network-wide score-perturbation experiments) | solves the structural problem that a blocked/declined event produces no observed outcome | **no** — §5.16/5.17's backtest metrics (`:590-640`) implicitly assume an eventual outcome label; decider2's suppressing actions (`freeze_account`, `block_channel`, `decline`) have exactly this blind spot | **yes, must-have** — a currently-undocumented gap in the backtest metric design, not just a Stripe nicety |
| Rule **changelog** with before/after and user | Stripe ("Rule activity," 180 days, predicate before/after + user), Sardine, SEON (restore/compare versions), Hawk (4-eyes) | who changed what, when | partial — `docs/04-observability-and-governance.md:239-253`: "A config diff *is* an audit record," and `ServeHandle` keeps generations (`runtime/serve.py`) — but no identity, no approval, no persistence | **yes, must-have** |
| Four-eyes / maker-checker approval on a rule change | Hawk ("4-eyes reviews") | two people, recorded | **no** — O17 (`docs/06-open-questions-and-experiments.md:204-207`) is open — and worth noting BioCatch's widely-quotable "segregation of duties" phrase is, on inspection, RBAC (two role tiers) rather than a maker-checker workflow: no vendor in this survey publishes a true two-person-approval mechanism | **yes, must-have**, but as a field in the rule document, not framework machinery |
| Rule-level **circuit breaker** on fire rate | none found publishing this | a mistyped threshold self-demotes in minutes | no (spec only) — `:740-748`; FRAMEWORK-DEMANDS #33/#34 | **yes** — cheapest protection against the R800-vs-R8000 typo |
| **Case management** / review queues and SLA tiers | Feedzai Case Manager, Actimize ActOne, Unit21, DataVisor, Sift, Hawk (Unified Case Manager) | a human works the hold | out of scope by decision (`:1100-1103`) — routing and evidence in scope | no — but the *evidence projection* for the queue screen is in scope and unbuilt |
| **Analyst feedback loop / label capture** | all of them | disposition feeds model + rule metrics | out of scope (`:590-620` consumes it) | no |
| Replay of a single past decision from recorded artefacts | AWS `GetEventPredictionMetadata` (closest public shape, but requires 5 keys incl. `predictionTimestamp` — cannot fetch by event id alone) | reproduce exactly, using only recorded values | partial — `docs/04-observability-and-governance.md:239-253` specifies the audit record; nothing writes one | **yes, must-have** |
| **Optimistic concurrency on list/table edits** (version token or ETag, reject a stale write) | Sift Workflow List API (`ETag`/`If-Match`, "the server will reject the update") | prevents two analysts silently clobbering each other's edit | **no** — `docs/03-authoring-api.md:720-747`'s keyed-lookup work ("BEING BUILT" in `decider2.tables`) has no concurrency story | **yes** — cheap now, a real bug once an editing UI exists |
| **Entity-level persistent manual determination** outranking future automated firings | Ravelin (permanent manual-review ALLOW/fraud mark: "any rules you configure will not change this"), Sift (immutable, append-only Decision log — "apply a subsequent Decision" rather than mutate) | a human's determination becomes a standing fact the rule/ML layer cannot silently override | **no, by design** — case management and entity state are explicitly out of scope (`:1100-1103`); the nearest analogue, a critical `allow` rule (`:475-481`), is rule-based, not entity-based | **maybe** — document as a deliberate boundary between decider2 and its caller, not a build item |
| Explicit **DAG/graph well-formedness validation** (acyclic; every node reachable from a declared root; terminal nodes are exactly the decision nodes) | Sift workflow engine (three published, structurally-enforced constraints) | cheap, load-bearing checks run before a graph is trusted | **no** — `graph/` (Layer 3) is "SPECIFIED BUT UNVALIDATED … has no empirical support at all" (`docs/00-BUILD.md:160-163`) | **yes** — Sift is proof this is a three-item checklist, not a research problem |
| **Financial guarantee / loss-indemnification** layered on the risk score (vendor underwrites residual fraud loss for a fee) | Sardine (ACH indemnification for unauthorized returns R05/R07/R10/R11/R29; card chargeback guarantee; ~3 months of data required first, no published cap) | changes who bears a false-negative's cost, not how a decision is computed | **n/a — not an engine capability** | **no** — a business-model difference between a vendor and a framework, not a row to adopt; worth one sentence, not a build item |
| A per-decision explainability bundle returned as one object (reasons, rule execution, model scores, features) | DataVisor ("Decision explainability — reasons, rule execution, model scores, and features returned per decision") | one response object carries the full trace | partial — folds into the firing-set/bitset work already proposed (rec #6); DataVisor's phrasing is a cleaner target response shape | **yes** — same recommendation, sharper spec |

### 2.4 ML integration, latency, failure

| Feature | Which engine(s) | What it does | decider2 status | Adopt? |
|---|---|---|---|---|
| Model score as a **rule attribute / variable** | AWS (`$model_insightscore`, 0–1000), Stripe (`:risk_score:`, documented inconsistently as 0–99/0–100 — a genuine Stripe doc defect), Sift Score (0–1 in the API, 0–100 in prose — also inconsistent), BioCatch 0–1000, Sardine | score enters rules as an ordinary feature | **yes** — a score is just an input column; no machinery needed | already have; document it |
| Bring-your-own model plugged into the engine | AWS `PutExternalModel` (SageMaker, template-substitution I/O adapter), Feedzai openml (Java SPI), FICO Analytics Workbench, ACI Model Generator | vendor hosts a customer model | **n/a** — decider2 is a library; the caller does the network call | no |
| **Template-based external-model I/O adapter** — declared placeholder substitution in, declared output-to-variable remapping out | AWS `PutExternalModel`/`ModelInputConfiguration`/`ModelOutputConfiguration` | generalises "call one external model" into a reusable, declarative N-model contract | **no** — one ad hoc external-model contract in the spec only (score/id/version/3 factor codes, 8 ms deadline, `:270-289`) | **maybe** — worth building only if/when decider2 needs to host more than one external scorer per event type |
| PMML / ONNX import | **none of the fraud set** claims either | | **no** | no — nobody in this set does it |
| Reason codes for customer communication | Stripe (`outcome.reason`, `outcome.rule` — the only machine-readable "which rule fired" field in Stripe's product), Google (`riskReasons`), AWS (bare outcome strings only, no reason payload), FICO XAI, Forter (`reasonCode` on abuse-policy declines only, not core fraud) | why, in a code a CX system can render | **yes** — `docs/04-observability-and-governance.md:175-180`: "Reason codes need no new machinery" | already have; the warning there (a ported system dropped its whole taxonomy) is the thing to heed |
| **Per-fired-rule value-substituted predicate text** (`expressionWithValues`: the rule's structure with each leaf's runtime value substituted inline) | AWS Fraud Detector (`GetEventPredictionMetadata.rules[].expressionWithValues`) | renders "what the rule actually saw" as one human-legible string, no join required between structure and values | partial — emitted intermediates give values as columns (`docs/04-observability-and-governance.md:155-170`); decider2's own reviewer test found this insufficient — a reviewer could not connect a value to the predicate that used it (`:277-333`) | **yes, must-have** — narrow, cheap, targets a failure mode decider2 has already documented in itself |
| Model-absence behaviour declared, not emergent | AWS (built-in models get imputed defaults; SageMaker models get literal `'null'`, differently) | a timed-out or absent score is a stated rule, replayable | no (spec only) — `:270-289` | **yes** |
| **Two-mode fail-open/fail-closed default, keyed to the cause of silence** | Ravelin (non-2xx/timeout → "accept the order," 750 ms recommended timeout; but its own rate-limit breach → fails *closed*, `action:PREVENT, source:RATE_LIMIT`) | the safe default depends on *why* the engine went quiet, not on one constant | **no** — `serving/dispatch.py` has no timeout, deadline or degraded response at all; `invocations()` calls `handle.score(record)` unconditionally | **yes, must-have** — reshapes the fail-open recommendation into two explicit defaults rather than one |
| Published **fail-open** policy for the calling system | Ravelin (750 ms, fail-open on non-2xx), Forter (2 s client timeout, **no fail-open policy published at all** — a confirmed absence even from the most heavily-documented vendor in this survey), SEON (min 1,500 ms) | what the caller does when the engine is silent | **no** — `serving/dispatch.py` has no timeout, deadline or degraded response | **yes, must-have** — one page in doc 02 §3.6 |
| **Partial/degraded response contract** — the call succeeds (HTTP 206) but returns available fields when some upstream data source is unavailable, rather than failing outright | Hawk AI (`GET /v1/transaction-checks/{caseId}`: "case found but some data sources are unavailable") | graceful degradation distinct from a hard timeout/fail-open decision | **no** — the freshness model (`:246-266`) governs individual *feature* staleness (`evaluate_false`/`last_good`/`suppress`) but nothing analogous exists for the *response envelope* when an upstream table is degraded | **maybe** — small, orthogonal to the fail-open/timeout page; worth folding into the same design pass |
| **Per-decision self-reported latency field** returned on every response | Hawk AI (`took`) | turns "was this call slow" into a queryable field instead of a log-correlation exercise | **no** — the audit record spec (`docs/04-observability-and-governance.md:239-253`) lists inputs/outputs/emitted intermediates but no per-decision timing field | **yes** — trivial (a wall-clock delta already available inside `runtime/invoke.py`) |
| Concurrency-safe tail latency | vendors hide it; none publish a tail figure | p99 under concurrent load | **yes, measured** — `nogil=True` flat to 16 threads; `nogil=False` at 1270% of budget (`EXPERIMENTS.md:1063-1070`) | already have; make the serving default explicit |
| Per-decision trace of every rule that fired | AWS `ruleResults` (bare outcome strings, no timing/explanation — richness deferred to a second, post-hoc call), GoRules per-node trace, Actimize, Feedzai Whitebox | the firing set, as data | partial — emitted columns at +0.11 ns/row and a branch `_path` (`docs/04-observability-and-governance.md:155-170`); no bitset-of-521-rules primitive | **yes, must-have** — FRAMEWORK-DEMANDS #7 |
| **Relative, added-overhead latency SLA** (published as a percentage over a caller's own baseline call, not an absolute number) | Sift ("< 50ms latency at the 99th percentile **over calling the score API directly**") | frames engine overhead against the caller's own budget | **already have the framing internally** — `EXPERIMENTS.md:678-770` states 971.4 µs p50 as "4.86% of a 20 ms budget" — Sift is evidence this convention is worth publishing externally, not just keeping in the experiments log | already have; publish it |

---

## 3. Improvements for decider2 — ranked, de-duplicated

The 29 items in §4 collapse into six priority tiers. Ranking logic: foundational data
primitives first (nothing else is buildable without them), then the rule-evaluation semantics
that decide whether decider2 can host a 521-rule set at all, then zero-dependency production
safety, then governance/audit, then lifecycle analytics (which depend on the bitset primitive),
then the one big open design decision. Rec numbers refer to §4. "Bank must-have" reflects
whether a Tier-1 bank running card/EFT fraud interdiction could ship without it — not whether
any vendor happens to have it.

| Tier | Theme | Recs | Demonstrated by | Effort | Bank must-have? |
|---|---|---|---|---|---|
| **P0 — Foundational data primitives** | Lists, temporal (as-at-instant) tables, per-event-type field catalogue | #4, #5, #17 | AWS `@list`, Stripe `@alias` lists, Ravelin Datalists (list mechanics); no vendor publishes membership-as-at-an-instant — this is a bank requirement, not a vendor feature; FICO's namespaced `CRTRAN25.field` (event-type schema) | M, L, M | **Yes** — §4.3's rule-shape data shows 71% of live rules reference a table-lookup band; nothing is authorable without a list primitive |
| **P1 — Rule evaluation semantics** | Generic kernel for `ruleset` (unblocks free `enabled`), `all_match`, `class_first_match` hit policies, vocabulary alignment | #2, #1, #3, #19, #16 | Every vendor deploys a rule change in seconds because nothing compiles; AWS `FIRST_MATCHED`/`ALL_MATCHED`; Stripe's ordered-action-class hit policy, already present in decider2's own spec as hand-written prose (§5.12) | L, S, S, S, S | **Yes** — #2 is the single biggest lever; without it the spec's "≤10 minutes including approval" criterion is unreachable for a 900-rule set |
| **P2 — Production safety, zero dependency** | Fail-open/fail-closed with two defaults, `nogil` default for `serve()`, partial/degraded response | #7, #8, #26 | Ravelin's two-mode fail-open/fail-closed split; Forter publishes a timeout with *no* fail-open policy at all despite being the most thoroughly documented vendor in this survey; Hawk's HTTP 206 partial-data pattern | S, S, S | **Yes** (#7, #8); nice-to-have (#26) — these are S-effort and currently entirely unaddressed in `serving/` |
| **P3 — Governance & audit trail** | Effective dating, author/approver/change-class, the audit record itself, decision-latency field | #9, #10, #11, #27 | Stripe's 180-day rule-activity log; Hawk's "4-eyes reviews" (a claim, not a mechanism — decider2 building #10 would be a *mechanism*); AWS `GetEventPredictionMetadata` (thinner than what #11 proposes, and unfetchable by event id alone); Hawk's self-reported `took` field | M, S, M, S | **Yes** (#9, #10, #11) for a 540-day dispute window; nice-to-have (#27) |
| **P4 — Rule lifecycle analytics** | Firing-set bitset, per-rule rollup, backtest() harness, shadow-population lineage assertion, graph well-formedness, counterfactual-FPR metric, rendered-predicate diagnostic, optimistic concurrency, stage/checkpoint scoping | #6, #14, #13, #12, #25, #21, #22, #24, #23 | Stripe's per-rule "Est. false positive rate" (a calibrated answer to the exact blind spot #21 targets); AWS confirmed-absent shadow mode (zero rule-simulation capability across a 74-operation API — decider2 building #12 clears the entire field, not just catches up); Sift's three published DAG well-formedness constraints; Sift's `ETag`/`If-Match`; Ravelin + Forter's independent convergence on a stage/checkpoint scoping axis | M, S*, M, M, M, M, S–M, S, S | **Yes** for #6, #12, #13, #21, #22, #23; nice-to-have for #14 (*S once #6 exists), #24, #25 leans must-have given `graph/`'s "no empirical support at all" status |
| **P5 — Big open design decision** | Overlays: build the gain-vector mechanism or declare out of scope | #15 | No vendor publishes an overlay-with-enforced-expiry mechanism at all — this is either decider2's differentiator or its scope boundary | L | **Yes, as a decision** (L as a build) — leaving it undecided is the only wrong answer |
| **Opportunistic / nice-to-have** | Concurrent-enrichment combinator, saturated velocity state, published latency spec sheet, SAN/SCO sanctions field | #18, #20, #28, #29 | Every vendor issues enrichment concurrently and publishes one end-to-end number; Stripe's saturating counters; DataVisor's "30ms/15,000 QPS" headline framing; Hawk's SAN/SCO distinction | M, S, S, S | No — each is real but narrow; do opportunistically alongside its parent tier |

**The single highest-leverage item is #2** (generic kernel for `ruleset`): it is the only item
on the list that changes decider2 from "cannot compile a 521-rule set in a bounded time" to
"can", and every governance/analytics item downstream (P3, P4) is worthless if a rule change
still takes tens of seconds to land. **The single cheapest, most-neglected item is #7/#8**
(fail-open policy and `nogil` default): both are S-effort, both are measured and specified
today, and `serving/dispatch.py` addresses neither. **The one item that would make decider2
provably ahead of the entire vendor set, not merely at parity, is #12** (shadow isolation as a
lineage assertion) — no vendor in this survey, including a fully API-documented hyperscaler
product (AWS Fraud Detector), publishes an enforcement mechanism for shadow-mode isolation, only
the feature name.

---

## 4. Actionable recommendations

Each is a change someone could start on Monday. "Motivated by" names the engine feature that
makes the case. Items 1–18 are unchanged from the working draft; items 19–29 are new, surfaced
by the DataVisor, Sardine, Stripe Radar, AWS Fraud Detector, BioCatch, Ravelin, Forter, Sift,
Hawk AI and Unit21 research passes.

1. **Add an `all_match` hit policy to `tables/schema.py`'s `DecisionTable` and a
   `hit_policy` field on the document** (`first_match` | `all_match` | `priority`), so a
   document can ask for the firing *set* instead of the first row. Today the semantics are
   hard-wired: "First match wins, and a row that matches nothing takes `default`"
   (`tables/schema.py:560-565`). `tables/codegen.py` already emits one condition per row;
   `all_match` means writing a `bool[n_rows]` output column instead of breaking. **Effort S.**
   Motivated by: AWS `ALL_MATCHED`/`FIRST_MATCHED` as an explicit per-detector setting, and
   `EXPERIMENTS.md:381-390`, which already benchmarked both and found `all` compiles *faster*.
   **Must-have.**

2. **Add `ruleset.py` under a new `decider2/interiors/` as a generic kernel, not codegen.**
   `docs/08-configuration-and-lifecycle.md:328-345` currently assigns `ruleset` to **codegen**
   with a staged compile. Against a 521-rule set that is the wrong side of the trade: compile
   is ∝ lines^1.4, 56.56 s for 100 rules in one unit (`EXPERIMENTS.md:381-390`), and
   `docs/REVIEW.md:355-386` already identifies the way out — `experimentation/jittree/test.py`
   Approach B, "one generic njit walker that compiles exactly once, ever — new configs are
   just new arrays, no recompilation." Build the array encoding (per-rule: feature index,
   operator code, threshold index, group id, on-absent policy code) and one walker over it.
   **Effort L.** Motivated by: every vendor deploys a rule change in seconds because nothing
   compiles — Sardine, DataVisor, SEON, Stripe, AWS. **Must-have**; without it the spec's
   "≤10 minutes including approval" (`:986-988`, criterion 1) is not reachable for a 900-rule set.
   *If* codegen is kept instead, the mock's `stable_blocks(by="rule_id", count=16)` is the
   right mitigation and should be the framework's, not the project's.

3. **Give a rule document an `enabled` field whose semantics are "not emitted".**
   `EXPERIMENTS.md:417-425` measured a disabled rule costing full compile time (60 rules:
   21.12 s dead vs 21.28 s live). **Effort S.** Motivated by: every vendor UI has an on/off
   toggle and none of them charges for it. **Must-have** if #2 goes the codegen route;
   moot if it goes generic-kernel.

4. **Add a `keyed_set` table kind to `decider2/tables/` — `key -> bool` membership, with
   `contains(key)` emitted inside the kernel.** `docs/03-authoring-api.md:720-747` admits the
   keyed-lookup sketch is "provisional… lowest-confidence part of this document" and the real
   `tables/` is a decision table, not a set. Lists are the single most universal object in the
   fraud set: AWS `@list` (100,000 entries × 320 chars, documented size quotas), Stripe `@alias`
   value lists (50,000-item cap), Ravelin Datalists, Actimize Platform Lists, SEON lists.
   **Effort M.** **Must-have.**

5. **Add a second table kind `temporal_table` — `(key, instant) -> row`, stored as intervals.**
   FRAMEWORK-DEMANDS #17: a 2 M-row mule list refreshing 24×/day, retained 7 years, cannot be
   61 320 snapshots, and "was this beneficiary on the list at 14:22:03 on 3 March?" is asked in
   writing (`:799-801`). No vendor publishes this; it is a bank requirement, not a vendor
   feature, which is precisely why building it is decider2's job. **Effort L.** **Must-have**
   for disputes and sanctions evidence.

6. **Add a `firing_set` output convention: a fixed-width bitset column plus a frame-tier
   `bitset_explode` primitive.** `docs/04-observability-and-governance.md:155-170` gives
   emitted columns at +0.11 ns/row and a branch `_path`, which covers "which arm" but not
   "which 9 of 635 rules". A `uint64[10]` per record plus one declarative explode is what makes
   "every rule that fired, at 3 500 events/s, no sampling" (`:992-996`, criterion 3) a column
   rather than a debug mode. **Effort M.** Motivated by: AWS returns `ruleResults` as a list on
   every prediction; Actimize, Feedzai and GoRules all return the fired-rule trace as data;
   DataVisor's stated response shape ("reasons, rule execution, model scores, and features
   returned per decision") is a clean target spec for the response object this builds toward.
   **Must-have.**

7. **Write the fail-open/timeout page for `docs/02-architecture.md §3.6` and add a
   `deadline_ms` to `serving/dispatch.py`'s `/invocations`, with two separate defaults rather
   than one.** Today there is no timeout, no deadline and no degraded response anywhere in
   `serving/` — `invocations()` calls `self.handle.score(record)` unconditionally. The design
   needs to distinguish *why* the engine went quiet: an infrastructure-side deadline breach
   resolves to a declared safe action per project (fail open or fail closed), while decider2's
   own self-protective circuit breaker (`:740-748`, FRAMEWORK-DEMANDS #33) always resolves to
   fail-**closed** and carries a distinguishing marker — mirroring Ravelin's own published split
   (non-2xx/timeout → "accept the order," 750 ms recommended timeout; but its own per-customer
   rate-limit breach → `action:PREVENT, source:RATE_LIMIT`, fails *closed*). Forter, the most
   heavily first-party-documented vendor in this survey, publishes a 2 s client timeout and
   **no fail-open policy at all** — further evidence that even mature vendors leave this
   unstated, not that it is safe to leave unstated. **Effort S.** **Must-have.**

8. **Make `nogil=True` the default for kernels built through `pipeline.serve()`, or fail the
   handle's construction with a loud warning.** `EXPERIMENTS.md:1063-1070`: `nogil=False`
   reaches p99 = 1270% of a 20 ms budget at 16 threads. `runtime/serve.py` currently
   only *reports* it (`gil_report()`), per the owner decision at `docs/00-BUILD.md:103-117`.
   Reporting is the right default for a library; for `serve()` specifically, a GIL-holding kernel
   is a production defect, so `health()` should mark it `"status": "degraded"` rather than `"ok"`.
   **Effort S.** Motivated by: no vendor publishes a tail that bad because none of them serves
   from CPython. **Must-have.**

9. **Add `effective_from` / `effective_to` to the params document schema and to
   `ServeHandle._validate`, so a value change carries its own dating.** FRAMEWORK-DEMANDS #26:
   "Params must be effective-dated per entry, not one flat bundle per generation." Today
   `runtime/serve.py`'s validation is a flat document with no time dimension. **Effort M.**
   Motivated by: Stripe's rule activity log (180 days, predicate before/after + user), SEON's
   version restore/compare, Actimize's rule versioning — all of which answer "what was in force
   then". **Must-have** for a 540-day dispute window.

10. **Add an `author`/`approver`/`approved_at`/`change_class` block to the params document and
    record it in the audit record spec at `docs/04-observability-and-governance.md:239-253`.**
    O17 — Approval granularity (`docs/06-open-questions-and-experiments.md:204-207`) is open and
    is blocking nothing technical — it is a schema decision. Also add a distinguishing marker for
    a value written by an automated control rather than a person (FRAMEWORK-DEMANDS #33: a
    fire-rate circuit breaker writes a params field with no human review, and it is
    outcome-affecting). **Effort S.** Motivated by: Hawk's "4-eyes reviews" (a governance
    *claim*, precisely sourced to its FRAML page but with no published mechanism — this
    recommendation is what would make decider2's version an actual mechanism rather than an
    equally unspecified claim), BioCatch's "segregation of duties" (verified to be RBAC only, not
    maker-checker — a lower bar than what this item proposes), Stripe's role-restricted rule
    creation. **Must-have.**

11. **Write `observe/audit.py` so that something actually emits the record
    `docs/04-observability-and-governance.md:239-253` specifies.** The fields are already
    enumerated (structure fingerprint, compiled artefact id, resolved params, framework
    versions, inputs, outputs, emitted intermediates) and
    `example_projects/examples/01-transaction-fraud-interdiction/fraud_interdiction/artefacts/decision-record.example.json`
    is a worked shape including the toolchain block (numba/llvmlite/cpu_target). Nothing in the
    package writes one. **Effort M.** Motivated by: AWS's `GetEventPredictionMetadata` is the
    only public engine-side replay artefact in the set, and it is thinner than this — it cannot
    even be fetched by event id alone (it requires `detectorId`, `detectorVersionId`, `eventId`,
    `eventTypeName` and `predictionTimestamp` together). **Must-have.**

12. **Add a `population` argument to the ruleset declaration with `live` / `shadow`, and make
    shadow isolation a *lineage assertion* in `graph/` rather than a convention.** The spec's
    criterion 4 (`:997-998`) is "No shadow rule has ever changed an `action_code` … Demonstrated
    continuously, not asserted", and FRAMEWORK-DEMANDS #9 wants it provable "by lineage, at
    import time, with no data". decider2 already has static lineage
    (`docs/04-observability-and-governance.md:121-150`), so the assertion is "no path from
    `shadow_*` to any terminal" — a graph query, not new machinery. **Effort M.** Motivated by:
    Sardine, Unit21, DataVisor and Hawk all publish a shadow-mode *feature*, and not one of them
    publishes an *enforcement mechanism* — AWS Fraud Detector has no shadow mode at all,
    confirmed by grepping its complete 74-operation API surface. Building this would put decider2
    ahead of the entire vendor set on this specific property, not merely catching up.
    **Must-have.**

13. **Add a `backtest()` entry point in `runtime/` that takes `baseline=` and a metric set, and
    emits a decision pack.** The spec's metric list is already written (`:628-640`: hit rate,
    precision, incremental catch, FPR, overlap matrix, value blocked, operational load, action
    churn) and is a superset of what Stripe and Sardine publish. The engine half exists
    (`runtime/invoke.py` batch apply); the harness does not. **Effort M.** Motivated by: Stripe
    backtests on 6 months of charges before a rule is enabled; Sift backtests 30 days with a
    documented "1 active backtest per user, 5 per account" concurrency limit; Ravelin's is a
    one-week window; DataVisor, SEON and Unit21 all ship it too. **Must-have.**

14. **Add a per-rule rollup primitive — `firing_counts_by(rule_id, minute)` over the emitted
    bitset — and a `dead_rule` report.** `:585-589` wants "a per-minute aggregate of firing
    counts by rule and by overlay, which is what makes rule-level circuit breakers, overlay
    impact monitoring and dead-rule detection possible without scanning the record store";
    FRAMEWORK-DEMANDS #41 wants retirement provable by absence from that cheap aggregate.
    This is a polars group-by over #6's exploded bitset, i.e. frame-tier work decider2 already
    has. **Effort S** once #6 exists. Motivated by: Stripe's per-rule performance chart with
    "Est. false positive rate"; DataVisor's rule performance monitoring. **Must-have.**

15. **Decide and document decider2's answer on overlays — either build the gain-vector
    mechanism or declare it out of scope.** `docs/02-architecture.md:643-644` says params do
    not vary by record; FRAMEWORK-DEMANDS #10 calls the per-record overlay "probably the single
    largest new surface this project asks the framework to own", and the mock's answer is a
    5 × 8 `float64` gain vector (320 bytes) built once per event and multiplied into a tunable
    at the point a predicate reads it, with `overlay_exempt` rules emitting the same line
    *without* the gain term so no runtime path exists. That is a good design and it needs a
    framework home or an explicit refusal — the current silence means every project invents
    it. **Effort L.** Motivated by: no vendor publishes an overlay-with-enforced-expiry
    mechanism, so this is either decider2's differentiator or its scope boundary; leaving it
    undecided is the only wrong answer. **Must-have (as a decision); L as a build.**

16. **Rename the table's hit policy in the field's own vocabulary and note the mapping in
    `tables/schema.py`'s docstring**: `first_match` ≡ DMN FIRST / AWS `FIRST_MATCHED` / JDM
    `first`; `all_match` ≡ DMN COLLECT / AWS `ALL_MATCHED`. `docs/research/decision-engine-landscape.md:268`
    already makes this point. **Effort S.** Nice-to-have, and it makes import from any of them
    a mapping rather than a translation.

17. **Add a `variant` dimension to the input schema — one field catalogue keyed by event type
    with per-field effective dates.** FRAMEWORK-DEMANDS #4/#5: one schema cannot express twelve
    event types differing by 30 fields, nor a field that did not exist before a date (needed for
    an 18-month backtest across a scheme field addition, `:1051-1054`). `docs/00-BUILD.md`'s
    Layer 1 already had to hand-author `nullable` because polars cannot supply it, so the schema
    is already a governance artefact — this extends it rather than inventing it. **Effort M.**
    Motivated by: AWS Fraud Detector's event types with per-type variable sets; FICO Falcon's
    namespaced `CRTRAN25.field`/`ffmFrdCard.field` references for the same reason. **Nice-to-have
    for credit, must-have for fraud.**

18. **Add a `fan()` combinator (or document why not) for concurrent independent enrichment.**
    FRAMEWORK-DEMANDS #12: the model score's 8 ms deadline must overlap the other lookups or the
    25 ms budget fails, and `|` is sequence by definition (`docs/03-authoring-api.md:1161`).
    The mock added a fifth execution combinator and flagged it as a taxonomy problem for the
    framework. **Effort M.** Motivated by: every vendor issues enrichment concurrently and
    publishes a single end-to-end number; decider2 currently has no way to express it.
    **Nice-to-have** — a caller can do this outside the pipeline, and should be told so in
    doc 02 §3.6 if the answer is no.

19. **Add a third `hit_policy` variant to `tables/schema.py`'s `DecisionTable`:
    `class_first_match`** — ordered action classes, first-match-within-class, order-unspecified
    within a class, parameterised by a caller-supplied class ordering. This is distinct from the
    `first_match`/`all_match` pair item 1 proposes: it is a third, named hit-policy shape.
    Motivated by Stripe Radar's evaluation-priority-by-action-class ("Rules of the same action
    type aren't ordered," `docs.stripe.com/radar/rules/reference`) and by decider2's own
    hand-rolled action resolution (`example_projects/01-transaction-fraud-interdiction.md:463-500`,
    §5.12: a seven-rank action-severity table, critical-rule override, four-level deterministic
    tie-break), which already implements exactly this shape in prose with no reusable name or
    generic implementation. **Effort S** once item 1's discriminated-union groundwork exists.
    **Must-have** — the fraud spec needs this hit policy today and gets it only via bespoke
    Python.

20. **Add a `saturated` state to the velocity/aggregate value-state vocabulary, alongside
    `fresh`/`stale`/`absent`.** Motivated by Stripe's bounded counters — a ring-buffer counter
    that stops incrementing at a cap (25 for every cardinality/dispute/EFW/refund counter in
    Stripe's catalogue), after which `> cap` conditions can never fire again, silently.
    **Effort S** — a fourth enum member plus a documented rule-authoring warning. **Nice-to-have**
    unless decider2's upstream streaming store is known to use bounded counters, in which case
    **must-have**.

21. **Add a counterfactual-FPR/precision metric to the §5.17 backtest metric set for suppressing
    actions**, modelled on Stripe's calibrated-score approach — computed from the external
    model score's own calibration curve rather than waiting on an outcome label a declined event
    will structurally never produce (a blocked/declined transaction never completes, so it is
    rarely disputed and its true-fraud status frequently never confirmed). **Effort M** — needs a
    calibration curve maintained per model version, new machinery, not a rollup over existing
    columns. **Must-have** — without it, the backtest metric list's own precision/FPR numbers are
    silently biased for every rule whose primary action is `decline`/`freeze_account`/
    `block_channel`, a large fraction of the 521-rule live set per §4.3's family split.

22. **Add a `rendered_predicate` diagnostic** — the fired rule's structure with each leaf's
    runtime value substituted inline as text — as a named tier-1 emitted artefact, distinct from
    emitted value columns. Motivated directly by AWS's `expressionWithValues`
    (`GetEventPredictionMetadata.rules[].expressionWithValues`) and, more importantly, by
    decider2's own reviewer test (`docs/04-observability-and-governance.md:277-333`), which found
    that showing values in a table next to a predicate's definition was *not* sufficient for a
    reviewer to connect the two. **Effort S–M** — a rendering function over data decider2 already
    has (rule structure + emitted feature values), not new instrumentation. **Must-have** for the
    reviewable artefact given it is the design's own top-ranked risk.

23. **Add a `stage`/`checkpoint` field to the rule/policy document**, independent of
    `event_type_code`, enumerated per project (e.g. `login`, `checkout`, `payout` for fraud;
    `app_open`, `statement_request` for ATO), validated against a declared per-project checkpoint
    list the same way `Segments` is validated today. **Effort S.** Motivated by Ravelin's
    checkpoint × category matrix (`accountRegistration`/`login`/`paymentMethodRegistration`/
    `paymentMethodSelection`/`checkoutPreAuth`/`checkoutPostAuth`/`refundRequest` ×
    `payment`/`ato`/`refundAbuse`/`supplier`) and Forter's pre-auth/post-auth execution-timing
    toggle for abuse policies — two vendors independently found event-type scoping alone
    insufficient. **Must-have.**

24. **Add optimistic concurrency to the params/table authoring surface** — a version token
    returned by `GET /params` and required on `POST /params` (`serving/dispatch.py`'s
    `post_params`), rejecting a stale write. Motivated by Sift's `ETag`/`If-Match` on Workflow
    Lists ("the server will reject the update" on a stale write). **Effort S.** Without it, two
    analysts editing the same table or params document within the same minute silently clobber
    one another the moment `tables/` ships an editing UI. **Must-have** (cheap now, expensive to
    retrofit after the first incident).

25. **Write the `graph/` well-formedness checks as a named, tested validation pass**: acyclic;
    every step reachable from a declared root; every terminal node is exactly a
    rule-resolution/action node; no path exists from a `shadow_*` population to a terminal (this
    last clause is item 12's lineage assertion, expressed as one of four checks in the same
    pass). Closes the "no empirical support at all" gap at `docs/00-BUILD.md:160-163`.
    **Effort M.** Motivated by Sift's own published workflow-engine design doc, which enforces
    exactly three of these four constraints structurally ("must be acyclic," "all nodes must be
    reachable from the root," "all terminal nodes must be decisions") — proof this is a known,
    cheap checklist, not a research problem. **Must-have.**

26. **Add a `partial`/`degraded` response state to `serving/dispatch.py`'s `/invocations`
    contract**, distinct from item 7's call-level timeout/deadline handling — the case where the
    call succeeds but an upstream table used mid-pipeline was degraded, so the caller needs to
    know the response is provisional even though it received a 200. Motivated by Hawk AI's HTTP
    206 "some data sources unavailable" pattern on `GET /v1/transaction-checks/{caseId}`.
    **Effort S**, same doc 02 §3.6 page as item 7 — write together, not as a separate design pass.
    **Nice-to-have.**

27. **Add a `decision_latency_ms` field to the audit record spec at
    `docs/04-observability-and-governance.md:239-253` and populate it in `runtime/invoke.py`.**
    Motivated by Hawk AI's `took` field, returned on every API response. decider2 already
    measures its own latency exhaustively in `EXPERIMENTS.md`, but nothing in the *production*
    record captures per-decision elapsed time — a bank operating decider2 cannot answer "was
    decision X slow" without external APM correlation, which the audit-record philosophy itself
    argues against ("what were the parameters on 3 March answers from one artefact").
    **Effort S.** **Nice-to-have**, trivial once item 11 exists.

28. **Publish a one-page decider2 latency/throughput spec sheet in vendor-comparable units**, the
    way DataVisor states "30ms Decision latency / 15,000+ Peak QPS" as two headline numbers.
    Today decider2's real numbers are scattered across `EXPERIMENTS.md`'s N-series tables
    (kernel ~1.38 µs; p99 1 114 µs single-thread; `nogil`-safe to 16 threads); a bank evaluator
    comparing vendors wants the same two-number presentation, with the caveat those vendors
    omit stated explicitly — that decider2's number is in-process only, not network-inclusive —
    so it is not misread as directly comparable to a vendor's end-to-end figure. **Effort S.**
    **Nice-to-have** (documentation/positioning, not a mechanism change).

29. **Add a SAN/SCO-equivalent distinction to the sanctions gate's output** — a direct-list hit
    versus an ownership/control hit as separate, typed fields rather than one hard-block outcome.
    Motivated by Hawk AI's `screeningHits[].sanctionsOwnership` boolean, which distinguishes a
    directly sanctioned entity from one merely owned/controlled by a sanctioned party. **Effort
    S.** **Nice-to-have** unless a bank's compliance team already routes these two cases
    differently, in which case must-have.

---

## 5. Genuine alternatives — where not to build this in decider2

### A1. Windowed features belong in a stream processor, not in decider2. Not an alternative — a prerequisite.

decider2's own fraud spec already concedes this: "Computation of velocity aggregates. They
arrive precomputed; this specification governs their consumption, staleness semantics and
recording" (`example_projects/01-transaction-fraud-interdiction.md:1097-1099`), and
`:124-125`: "Aggregates arrive precomputed from a streaming store. They are not computed
here". The 168 aggregates (7 keys × 6 windows × 4 statistics, `:126-134`) are an input
contract, with a per-window staleness tolerance and a watermark per value.

That is the right call and it should be stated as a *product boundary*, not left as a
scope note. Apache Flink's SQL windowing TVFs (`TUMBLE`/`HOP`/`CUMULATE`/`SESSION`) give
exactly this shape declaratively
([Flink 2.0 window aggregation](https://nightlies.apache.org/flink/flink-docs-release-2.0/docs/dev/table/sql/queries/window-agg/)),
and Flink's own fraud-detection case study is the canonical "stream processor + thin rule
layer" design: rules are JSON with `groupingKeyNames`, `aggregateFieldName`,
`aggregatorFunctionType`, `limitOperatorType`, `limit`, `windowMinutes`, `ruleState`, and are
pushed into the running job over a Kafka control topic via broadcast state, so a rule change
needs no redeploy
([vol.1](https://flink.apache.org/2020/01/15/advanced-flink-application-patterns-vol.1-case-study-of-a-fraud-detection-system/),
[vol.2](https://flink.apache.org/2020/03/24/advanced-flink-application-patterns-vol.2-dynamic-updates-of-application-logic/)).
Stripe Radar and AWS Fraud Detector confirm the pattern from the other direction: Stripe's 513
velocity attributes are a closed, tumbling-bucket-approximated catalogue with **no
user-definable aggregate at all**, and AWS's stateful aggregates (TFI/ATI) are computed and
stored *inside the model layer*, invisible to the rule language (DETECTORPL) entirely — two
independent hyperscaler-grade products landing on "rules never compute their own windows" as
the answer.

**Where that beats decider2 outright.** If the bank's rule population is mostly of the
shape "SUM(amount) per (payer, beneficiary) over 10 080 minutes > limit", Flink expresses
each rule as seven JSON fields and needs no compiler at all. decider2's answer to the same
rule is a numba kernel whose compile is **∝ lines^1.4** — 56.56 s for 100 rules in one unit
(`docs/EXPERIMENTS.md:381-390`) — plus a whole separate upstream system to maintain the 168
aggregates the rule reads. For a greenfield card-fraud deployment with no existing
per-record credit logic, the honest recommendation is Flink (or a vendor) for the velocity
tier and something thin on top; decider2's value appears only when the per-record logic is
large, numerically fussy, and shared with credit decisioning.

**Where decider2 still wins.** Flink's rule shape is one aggregate compared to one limit.
It cannot express `example_projects/01-transaction-fraud-interdiction.md:402-412`'s real
distribution — mean 4.2 predicates per rule, max 17, 3% with a rule-local derived quantity,
71% with a table-lookup band — without dropping into the Java CEP/ProcessFunction API, at
which point rule authoring is code again and the governance model the spec is built on
(§6.1's 18 governed attributes per rule, four-eyes, effective dating) has to be built from
scratch anyway.

### A2. Buy the fraud engine; keep decider2 for credit. The case for it, in decider2's own numbers.

For card/EFT interdiction specifically, a vendor arrives with four things decider2 does not
have and has not designed: a case-management application, an analyst feedback loop that
labels outcomes, a consortium model, and a rule editor non-engineers actually use. The
decider2 spec puts all four out of scope or on someone else (`:1100-1112`, §12) while
depending on all four (§5.16 outcome feedback, §5.13 queues and SLA tiers).

The published latency envelope is not the deciding factor either way, but it needs correcting
against the primary sources gathered for this report, not the earlier secondary figures.
Verified, first-party numbers: DataVisor **"30 ms or less, at peak throughput above 15,000
QPS"** (`datavisor.com/platform/real-time-decisioning` — supersedes an earlier, unsourced "under
100ms" figure), Hawk **"150 ms average"** (`hawk.ai/solutions/framl`) against a separate,
narrower **"30,000 TPS"** claim scoped only to its AML AI Overlay bolt-on product (not
necessarily the same code path — do not combine the two into one system's figure), Sift
**"<150 ms"** scoring plus a *relative* **"<50 ms added at p99 over calling the score API
directly"** workflow-overhead SLA, Unit21 **"under 250 milliseconds"** (no percentile or
methodology stated), Mastercard Decision Intelligence Pro **"under 50 milliseconds"** (first-party
newsroom, Feb 2024), and ACI **"under 300 milliseconds"** generic plus a named reference site
at "sub-100 ms" (both figures from an analyst report ACI itself hosts and distributes, not an ACI
product page). Two figures repeated in earlier drafts of this comparison do **not** hold up under
direct verification and should be treated as unconfirmed: Sardine's widely-quoted **"~25ms"**
could not be traced to any first-party page after grepping the entirety of its public docs and
its marketing homepage directly — downgrade to **[SEC, unconfirmed]**; Stripe Radar and Forter's
core ML decisioning publish **no latency figure at all** (Forter's citable number, 400 ms at
p99, is for its abuse-policy layer specifically, not "under 1 second" as its own marketing
repeats elsewhere).

Against that corrected envelope, the scheme-timeout-derived **p99 ≤ 25 ms** on card
authorisation (`example_projects/01-transaction-fraud-interdiction.md:888`) and decider2's
in-process figures — p99 1 114 µs single-thread (5.57% of a 20 ms budget), max 3 840 µs
(`docs/EXPERIMENTS.md:1010-1018`) — remain an order of magnitude inside every vendor number
above. So *if* the aggregates are already computed, decider2 is the faster place to evaluate
rules — the question is never latency, it is the other four things (case management, feedback
loop, consortium data, non-engineer rule editing).

**A vendor-specific reason to be cautious about "just buy it," not just a reason to build:**
AWS Fraud Detector — the single best-documented rule language in this whole survey
(DETECTORPL, full quota tables, `GetEventPredictionMetadata`) — has been **closed to new
customers since 7 November 2025**. It remains the cleanest public design reference precisely
because it is winding down, not because it is a live buying option; anyone using it as a
reference architecture should build against its *documented mechanics*, not its
*availability*.

### A3. The narrow, defensible decider2 position

decider2 is the only engine in this survey that runs **the same artefact** in single-record
serving and a 420 M-event backtest with an automated three-way equivalence assertion
(`runtime/modes.py:1-33`; acceptance criterion 6 at
`example_projects/01-transaction-fraud-interdiction.md:1000-1002`). Every vendor's backtester
is a second implementation of its evaluator; Stripe (6 months of charges), Sift (30 days,
"1 active backtest per user, 5 per account"), Ravelin (1 week, "Estimate rule impact"),
DataVisor, Sardine, SEON and Unit21 all ship backtesting, and none of them publishes an
equivalence proof between backtest and production. AWS Fraud Detector, the most exhaustively
API-documented product in the set, ships **no** backtest/simulation/shadow capability of any
kind — confirmed by reading its complete 74-operation API surface and finding nothing resembling
one. That is the one axis on which decider2 is not merely competitive but structurally
better, and it is the axis the spec's own §5.17 says is worthless if broken ("if the two can
disagree it is worthless").

### A4. A vendor's own rule-count ceiling is evidence a bank cannot simply buy this off the shelf

Stripe Radar publishes a hard limit — **"a maximum of 200 transaction rules and 100 account
rules"** — below which the product presumably will not scale its own rule-evaluation engine.
decider2's fraud spec runs **521 live + 114 shadow + 1 360 retired** rules
(`example_projects/01-transaction-fraud-interdiction.md:731-738`), nearly three times what
Radar's own product allows per account. This is not evidence against buying a PSP-grade fraud
SaaS for a PSP-scale rule set; it is evidence that the merchant/PSP tier of this market (Stripe
Radar, Sift, Ravelin, Forter) is architecturally sized for hundreds of rules, not the thousand-plus
a Tier-1 bank's card-and-EFT interdiction program accumulates over a decade, and the
bank-grade tier (Feedzai, Featurespace, Actimize, FICO Falcon, Hawk) is exactly where the
custom-build-vs-buy question actually gets asked.

---

## 6. Unverified / unreachable

Recorded so a later reader knows which comparisons rest on primary sources and which do not.
"Unreachable" means a fetch was attempted and failed in the stated way. Rows carried over from
the original pass are kept; rows the DataVisor/Sardine/Stripe/AWS/BioCatch/Ravelin/Forter/Sift/
Hawk/Unit21 research passes corrected, deepened or newly found are marked accordingly.

| Vendor | What could not be verified | Failure mode |
|---|---|---|
| **Feedzai** | the rule/"risk strategy" authoring syntax, hit policy, list management, champion-challenger, case-manager fields | `docs.feedzai.com` redirects to a Confluence login |
| **Featurespace** | AMDL (ARIC Model Definition Language) syntax; the "Adaptive Rules Engine" and "Analytical Workflow Manager" mechanics; any latency or throughput figure | product pages render as "Loading…" JS shells; some PDF hosts DNS-fail |
| **NICE Actimize** | Policy Manager's actual rule grammar, classifier definition, list-ageing semantics; any latency/throughput figure | datasheet PDFs return binary; `/fraud-prevention/ifm-x` 404s |
| **DataVisor** *(new)* | the real product documentation entirely — rule syntax, hit-policy semantics, list ageing, membership-as-at-instant mechanics, any API reference | `docs.datavisor.com` (Document360) resolves to a sign-in page for both `llms.txt` and `llms-full.txt`, not content |
| **DataVisor** *(correction)* | this report's earlier latency citation ("under 100ms") is superseded | the correct, directly first-party figure is **"30 ms or less, at peak throughput above 15,000 QPS"** (`datavisor.com/platform/real-time-decisioning`); no p99/tail figure or methodology accompanies it |
| **DataVisor** *(new)* | the named case study's customer identity and measurement methodology ("70% detection... 60% reduction... 10x efficiency") | customer described only as "a leading U.S. accounting and financial-operations platform," no baseline period stated |
| **Sardine** *(new)* | the widely-repeated **"~25ms"** latency figure | grepped all 668 lines of public `docs.sardine.ai` content (zero latency/TPS numbers of any kind) and fetched `sardine.ai`'s marketing homepage directly (explicitly qualitative "real-time" language only, no number found on either) — downgrade to **[SEC, unconfirmed]** until a primary source is located |
| **Sardine** *(new)* | the dollar/percentage cap on its ACH indemnification and card chargeback guarantees | confirmed first-party that the guarantees exist and require ~3 months of ACH data first, but grepped directly for "cap"/"limit"/"up to $" and found nothing — presumably negotiated per-contract |
| **Unit21** | rule syntax (beyond marketing-blog depth — see §1.14), the basis for "sub-250ms" | *(mechanism now confirmed exactly)* `docs.unit21.ai/llms.txt` and `/llms-full.txt` both return a magic-link sign-in page (`<title>Sign in \| Unit21</title>`); separately `www.unit21.ai/llms-full.txt` 404s to a generic Webflow template (never a real page — a different failure mode from a login gate) |
| **Unit21** *(partially resolved)* | Graph-Based Rules mechanics | now documented at marketing-blog depth (tag-propagation over a device/IP/bank-account/phone-number link graph, shadow-mode testable, no additional purchase) — still no published query/authoring syntax |
| **Unit21** *(new)* | "under 250 milliseconds" real-time monitoring claim | confirmed as a bare marketing-page claim (`unit21.ai/products/real-time-monitoring`) with zero percentile or methodology — same evidentiary class as Feedzai's "3 milliseconds" and Forter's "under 1 second" |
| **Unit21** *(new)* | Fraud Consortium size — "80M+ U.S. consumers" (one page) vs. "a network of 100+ institutions" (another page) | different units, not directly reconcilable from the pages fetched |
| **Mastercard Brighterion / Decision Intelligence** | the rule-writing/policy-authoring surface (if any); the DI score in ISO 8583 DE 48.75.1 claim; smart-agent architecture beyond one patent-snippet sentence | every host failed: `b2b.mastercard.com` DNS, `brighterion.com` redirects into the same dead host, `developer.mastercard.com` JS shells, `mastercard.com`/`.us` 403, patent hosts 503/403/reset on every attempt — re-attempted for this report and still unverifiable on any Mastercard-owned page |
| **Mastercard** *(new, resolved)* | four hard numbers for DI Pro specifically | **verified [FP]** from Mastercard's own newsroom (1 Feb 2024): "under 50 milliseconds," "1 trillion data points," "20% average / up to 300%" detection improvement, "85%" false-positive reduction — but two *other*, separate newsroom releases (22 May 2024 Cyber Secure; 18 July 2024 AI Garage) report different numbers for different products and must not be conflated with these four |
| **FICO Falcon** | Falcon rule-writing syntax; the customer-profile/behavioural-state mechanism; any millisecond figure ("Delivers millisecond response times" carries no number) | Falcon product pages and `investors.fico.com` return JS shells or time out; `community.fico.com` returns an error shell |
| **ACI Worldwide** | the only numbers ("under 300 milliseconds"; an Indian FI at "over 6,000 transactions per second with sub-100 millisecond latency") appear in an **analyst-authored, ACI-hosted** PDF, not in ACI's own product documentation | `developer.aciworldwide.com` DNS failure; `investor.aciworldwide.com` timeout; no developer portal or rule-syntax page exists publicly at all (absence, not a failed fetch) |
| **ACI** *(new)* | its own AI-feature-count claim drifts between datasheets | "8,000+ AI features" (2025 merchant datasheet, doc code AFL2104 03-25) vs. "9,000+ AI features" (2024 billers datasheet) — flag if either is quoted |
| **ACI** *(new)* | the incremental-learning patent number | not located in any datasheet or press page fetched |
| **Fiserv / FIS** | API shapes and rule-authoring surfaces | `developer.fiserv.com` is JS-only; an FIS product sheet failed SSL |
| **Forter** | whether a customer can author a rule *for core fraud decisioning* (confirmed: only for abuse policies, via Policy Builder — see §1.11) | `docs.forter.com` itself is reachable and thorough; the gap is scope, not access |
| **Forter** *(new)* | its engineering blog, reportedly discussing latency design in more depth (a "300/500ms" figure is sometimes attributed to it) | `tech.forter.com` — DNS failure (`getaddrinfo ENOTFOUND`); do not cite any figure attributed to it beyond the platform page's verified 400 ms at p99 |
| **Ravelin** *(new)* | Datalists as a documented, reviewable feature | lives only in a Notion-hosted, JS-rendered Help Centre (`support.ravelin.com` renders `<title>Notion</title>`), entirely absent from `developer.ravelin.com`'s own sitemap |
| **Ravelin** *(new)* | the 16-conditions-per-rule limit and the one-week "Estimate rule impact" backtest window | **[SEC only]** — sourced from `ravelin.com/blog/ai-fraud-prevention-rules`, a marketing blog page, with no corroborating reference on `developer.ravelin.com` |
| **BioCatch** *(new)* | any approval workflow beyond RBAC (maker-checker, versioning, rollback, simulation/backtesting) | not stated anywhere in the one public datasheet; BioCatch publishes no developer/API documentation at all, so no secondary corroboration is possible either — its "segregation of duties" phrase is RBAC (two role tiers), not a maker-checker mechanism |
| **BioCatch** | latency, throughput, or any SLA figure | the datasheet says only "real time"/"immediate action," no number, no percentile |
| **Sift** *(new)* | per-route live analytics dashboard; any published maximum route count per workflow | visible only as Help Center article titles in search results, not opened; the public docs bundle covers neither |
| **Sift** | the "<50 ms p99" workflow-overhead figure is a *design target* stated on the engineering blog (2017), not a measured production SLA | — |
| **Hawk AI** *(new)* | the entire Guides documentation section (rule manager, entity risk detection, investigative agent, screening, regulatory reporting, program dashboards, customer cases/list, user management) | every fetched guide page returns a Document360-hosted `<title>Login</title>` page on the *same domain* as Hawk's fully public API Reference section — a same-domain, section-specific auth wall, not a blanket JS-shell problem; Hawk's actual rule-authoring syntax remains unverified as a result |
| **Hawk AI** *(new)* | the federated-learning ("Decentralized Learning") claim | appears as the third of four "levels of information sharing" on an older, unrefreshed page, with no statement of which level is live or for whom — reachable but ambiguous, not unreachable |
| **Hawk AI** *(new)* | whether its "150 ms average" (FRAML page) and "30,000 TPS" (AML AI Overlay page) describe the same code path | the two figures are on different product pages (core transaction monitoring vs. a bolt-on scoring layer over an existing AML system) and should not be combined into one system's figure |
| **AWS Fraud Detector** *(new)* | its actual end-of-support/shutdown date | only a "closed to new customers as of November 7, 2025" notice is published on every UG/API page; no shutdown date appears anywhere — do not conflate "closed to new customers" with "will stop working" |
| **AWS Fraud Detector** *(new)* | `GetEventPrediction` latency beyond the 200 TPS default quota | no latency figure of any kind is published — 200 TPS is a rate limit, not a capability or SLA claim |
| **AWS Fraud Detector** *(new)* | operator precedence for mixed, unparenthesised `and`/`or` in DETECTORPL | AWS's own worked example mixes them with no precedence table published anywhere |
| **AWS / Stripe** *(new)* | internal documentation consistency | AWS states its model score range as "0 to 1000" on two pages and "1 to 1000" on a third; Stripe states `risk_score` as 0–99 in one place and "between 0 and 100" in another; AWS's `isbefore`/`isafter` are typed Boolean but every worked example compares the result to the string `"True"`/`"False"`; AWS's own `getepochmilliseconds` worked example returns epoch **seconds**, contradicting the function's name — none of these are resolvable from documentation alone and should be flagged if quoted |
| **Stripe Radar** *(new)* | any latency/SLA figure | checked the entire `/radar/*` doc corpus, `stripe.com/radar` and `stripe.com/radar/guide` — no percentile, millisecond figure or SLA published anywhere; a search-result claim ("produces a risk score in milliseconds") could not be traced to a first-party page |
| **Every vendor** | no vendor in this set publishes a p99.9, a tail distribution, a fail-open policy for its own internal model call, or a rule-count-versus-latency curve. decider2's `EXPERIMENTS.md` N-series publishes all four for itself, which makes every latency comparison directional, not like-for-like. Neither DataVisor nor Sardine's public material documents a fail-open/fail-closed policy or timeout value for its own model/list-lookup calls, and Forter — despite the deepest published API reference in the survey — states a client timeout with **no fail-open policy at all**. | — |
| **South Africa** | no vendor names a South African bank as a *fraud* customer on its own site. DataVisor says "one of the largest banks in South Africa" without naming it; FICO names Network International across UAE/KSA/Egypt/South Africa for 40+ FIs; Hawk shows an Ecobank logo. | — |

**Method note.** Page text was read through a mix of raw first-party fetches (`curl` on
`docs.stripe.com/*.md`, `docs.aws.amazon.com/*`, `docs.sardine.ai/*`, `docs.hawk.ai/apidocs/*`)
and a summarising fetch tool; quoted strings from raw fetches are literal doc text, quoted
strings from the summarising tool are as that tool returned them. Where a summariser produced a
number not present in the page text, the claim was discarded rather than cited — the same
discipline `docs/research/decision-engine-landscape.md:449` records for its own AWS fetches.

---

## 7. Sources

Grouped by vendor. Fetched across two research passes (2026-09-20 and 2026-09-21); failed
fetches are listed at the end of each group with their failure mode. This list supplements,
rather than replaces, the broader landscape survey's own source list at
`decider2/docs/research/decision-engine-landscape.md:431-449`, which every fraud-vendor URL in
this report's §0/§1 was cross-checked against.

**Feedzai** — https://research.feedzai.com/oshuzuxa/2022/08/Gomes_Railgun_VLDB2021.pdf (Railgun,
PVLDB 14(12), 2021) · https://research.feedzai.com/oshuzuxa/2022/08/Aparicio_ARMS_KDD2020.pdf
(ARMS, arXiv:2002.06075) ·
https://research.feedzai.com/oshuzuxa/2025/09/riff_inducing_rules_for_fraud_detection_from_decision_trees.pdf
(RIFF) · https://research.feedzai.com/oshuzuxa/2026/06/2606.16981v1.pdf (probabilistic
thinning, 2026) · https://github.com/feedzai/feedzai-openml ·
https://www.feedzai.com/riskops/ ·
https://www.feedzai.com/pressrelease/feedzai-unveils-expanded-aml-capabilities-underpinned-by-advanced-machine-learning-and-explainability/
·
https://www.feedzai.com/blog/latency-in-machine-learning-what-fraud-prevention-leaders-need-to-know/.
Failed: `docs.feedzai.com` (Confluence login).

**Featurespace** — https://patents.google.com/patent/US12118559B2/en (state-vector patent) ·
https://www.featurespace.com/aric-risk-hub · https://www.featurespace.com/automated-deep-behavioral-networks
· https://investor.visa.com/news/news-details/2024/Visa-Completes-Acquisition-of-Featurespace/default.aspx.
Failed: ARIC Risk Hub brochure PDF and AMDL-specific pages (301'd/deleted post-Visa rebuild);
EP4610915A1 read via patent search snippet only.

**NICE Actimize** —
https://www.niceactimize.com/Lists/Brochures/2023_actimize_university_course_catalog.pdf ·
https://www.niceactimize.com/blog/fraud-prevention-starters-guide-to-mitigate-fraud-using-policy-manager.
Failed: all three course-catalogue PDFs grepped for a latency/TPS figure and none exists;
niceactimize.com product datasheets return binary.

**FICO Falcon** — https://fraud.sia.eu/FalconRmaHelp/ (licensee-republished FICO docs, secondary)
· community.fico.com UDV/UDP forum posts (secondary) · US patents 11,636,485 / 11,481,777 /
11,023,894 / 11,875,355 (titles only — full text unreachable). Failed: all fico.com Falcon
product pages, `investors.fico.com`, `patents.google.com` (503 on every attempt), USPTO scanned
PDFs (no text layer).

**DataVisor** — https://www.datavisor.com/platform/rules-features-workflows ·
https://www.datavisor.com/platform/real-time-decisioning ·
https://www.datavisor.com/intelligence-detection/device-behavioral-intelligence---dedge ·
https://www.datavisor.com/intelligence-center/case-studies/global-financial-management-platform-detects-70-percent-of-coordinated-fraud-rings.
Failed: `docs.datavisor.com` (Document360) — both `llms.txt` and `llms-full.txt` resolve to a
sign-in page.

**Sardine** — https://docs.sardine.ai/guides/public/getting-started/apiaccess ·
.../getting-started/what-powers-sardine · .../getting-started/how-sardine-bills ·
.../risk/account-risk/account-risk · .../risk/account-risk/account-takeover ·
.../risk/account-risk/kyb · .../risk/business-risk/about-business-risk ·
.../risk/card-spending-risk/card-spending · .../risk/funding-risk/ach-indemnification ·
.../risk/funding-risk/card-indemnification · .../risk/funding-risk/funding-risk ·
.../risk/transaction-monitoring/transaction-monitoring (all under `docs.sardine.ai`) ·
`sardine.ai` marketing homepage (fetched directly for a latency figure; none found). Failed: no
first-party latency/TPS figure located on either domain; protected integration
guides/API-reference/SDK docs are login-walled by design.

**Stripe Radar** — https://docs.stripe.com/radar/rules.md ·
.../radar/rules/reference.md · .../radar/rules/supported-attributes.md · .../radar/lists.md ·
.../radar/risk-settings.md · .../radar/how-radar-works.md · .../radar/transaction-reviews.md ·
.../radar/transaction-risk-prevention.md · .../radar/reviews/risk-insights.md ·
.../radar/testing.md · .../radar/bot-abuse.md · .../radar/multiprocessor.md ·
.../radar/supported-payment-methods.md · .../connect/radar.md · .../api/charges/object.md ·
.../api/radar/reviews/object.md · .../api/radar/value_lists/object.md ·
.../api/radar/early_fraud_warnings/object.md (all fetched raw via `curl .../<path>.md`, HTTP
200) · https://stripe.com/blog/similarity-clustering · https://stripe.com/blog/radar-2018 ·
https://stripe.com/radar/guide. Failed: `/radar/backtesting.md` and `/radar/rules/backtesting.md`
both 404 — no dedicated backtesting URL exists; the mechanism is documented inline in
`/radar/rules.md` and `/radar/testing.md` only.

**AWS Fraud Detector** — https://docs.aws.amazon.com/frauddetector/latest/ug/create-a-rule.html ·
.../ug/rule-language-reference.html · .../ug/create-a-detector-version.html ·
.../ug/transaction-fraud-insights.html · .../ug/limits.html · .../ug/batch-predictions.html ·
.../api/API_GetEventPrediction.html · .../api/API_Operations.html (74 operations, read in full)
· .../api/API_PutExternalModel.html · .../api/API_CreateRule.html. All fetched directly from
`docs.aws.amazon.com`; no failures in this group.

**BioCatch** — https://www.biocatch.com/hubfs/Data%20Sheets/DS_PolicyManager_New_3.pdf (titled
"BioCatch Rule Manager" internally, despite the filename) ·
https://www.biocatch.com/resources/data-sheet/rule-manager. Failed: no developer/API portal
exists publicly at all (absence, not a failed fetch); no latency figure anywhere on
`biocatch.com`.

**Ravelin** — https://developer.ravelin.com/merchant/guides/payment-fraud-integration/error-handling/
· .../merchant/api/rate-limits/ · .../merchant/guides/other-guides/checkpoints/ ·
.../merchant/api/load-testing/ · ravelin.com/blog/ai-fraud-prevention-rules (secondary,
marketing register). Failed: `support.ravelin.com` (Notion-hosted, JS-rendered — Datalists
documentation lives here and is unreachable); no SLA/latency page found on `developer.ravelin.com`.

**Forter** — docs.forter.com/reference/overview · docs.forter.com/reference ·
docs.forter.com/extensions/abuse-prevention · https://www.forter.com/platform/. Failed:
`tech.forter.com` (DNS failure, `getaddrinfo ENOTFOUND`) — do not cite any latency figure
attributed to it beyond the platform page's verified "400 milliseconds" at p99.

**Sift** — https://engineering.sift.com/how-workflows-work/ (Micah Wylde, Jan 2017) ·
developers.sift.com/docs/curl/decisions-api/overview · .../curl/score-api/synchronous-scores ·
.../curl/workflows-api/overview · .../curl/workflows-api/status · .../curl/labels-api/overview ·
.../curl/verification-api/overview · developers.sift.com/tutorials/workflows. Failed: per-route
live-analytics dashboard and maximum-route-count figures found only as Help Center article
titles in search results, not opened.

**Hawk AI** — docs.hawk.ai/apidocs/transaction-checks · docs.hawk.ai/apidocs/customer-checks ·
docs.hawk.ai/apidocs/perform-check-on-customer-v2 · docs.hawk.ai/apidocs/monitor-bank-transfers ·
docs.hawk.ai/apidocs/monitor-card-payments-acquiring ·
docs.hawk.ai/apidocs/monitor-card-payments-issuing · docs.hawk.ai/apidocs/internal-transactions ·
docs.hawk.ai/apidocs/deposit-checks-2 (all public OpenAPI JSON, no auth) ·
hawk.ai/technology/tech-stack · hawk.ai/solutions/framl · hawk.ai/solutions/aml-ai-overlay ·
hawk.ai/technology/technology-behind-hawk · hawk.ai/platform/production-sandbox. Failed: every
`docs.hawk.ai` Guides page (rule-manager, entity-risk-detection, investigative-agent, screening,
regulatory-reporting-1, program-management-dashboards, customer-cases-3, customer-list-1,
user-management) returns a Document360 `<title>Login</title>` page on the same domain as the
public API reference.

**Unit21** — unit21.ai/blog (velocity-rules post, "...featuring-velocity-rules", 2 Sep 2025,
byline Gal Perelman) · unit21.ai/blog (Graph-Based Rules post, "the-network-strikes-back...", 20
Jun 2025) · unit21.ai/products/real-time-monitoring · www.unit21.ai marketing pages (device
intelligence, customer risk rating, fraud consortium, AI investigation agent, security/SOC 2).
Failed: `docs.unit21.ai/llms.txt` and `/llms-full.txt` (magic-link sign-in gate); separately
`www.unit21.ai/llms-full.txt` (generic Webflow 404, not a gate — never a real page).

**Mastercard Brighterion / Decision Intelligence** — newsroom.mastercard.com (Decision
Intelligence Pro launch release, 1 Feb 2024; Cyber Secure release, 22 May 2024; AI Garage
release, 18 Jul 2024 — three separate releases, numbers must not be conflated). Failed:
`b2b.mastercard.com` (DNS), `brighterion.com` (redirects into the same dead host),
`developer.mastercard.com` (JS shell, returns only "Mastercard Developers"),
`www.mastercard.com/.../risk-decisioning.html` (403), patent family US2015/0046332A1 /
US11348110B2 / US2015/0032589A1 unreachable on `patents.google.com` (503), `patentscope.wipo.int`
(403), `freepatentsonline.com` (connection reset).

**ACI Worldwide** — merchant-fraud and billers datasheets (doc code AFL2104 03-25 and a 2024
equivalent) · "Beyond Point Solutions: Orchestrating the Future of Fraud Prevention," Datos
Insights, May 2025 (Jim Mortensen, Gabrielle Inhofe), hosted at
`aciworldwide.com/wp-content/.../Datos-Insights-....pdf?gated=` — analyst-authored,
vendor-distributed; grade its numbers as vendor-supplied-to-analyst, not vendor-published.
Failed: `developer.aciworldwide.com` (DNS), `investor.aciworldwide.com` (timeout); no developer
portal or rule-syntax page exists publicly (absence).

**Stream-processor alternative** —
https://nightlies.apache.org/flink/flink-docs-release-2.0/docs/dev/table/sql/queries/window-agg/
·
https://flink.apache.org/2020/01/15/advanced-flink-application-patterns-vol.1-case-study-of-a-fraud-detection-system/
·
https://flink.apache.org/2020/03/24/advanced-flink-application-patterns-vol.2-dynamic-updates-of-application-logic/.

**decider2 primary sources** (repo at
`/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2`, read against the
working tree on `feature/decider-v2`, 2026-09-21) — `docs/00-BUILD.md`,
`docs/02-architecture.md`, `docs/03-authoring-api.md`, `docs/04-observability-and-governance.md`,
`docs/06-open-questions-and-experiments.md`, `docs/08-configuration-and-lifecycle.md`,
`docs/EXPERIMENTS.md`, `docs/REVIEW.md`, `docs/research/decision-engine-landscape.md`,
`example_projects/01-transaction-fraud-interdiction.md`,
`example_projects/09-governance-and-replay-harness.md`,
`example_projects/examples/01-transaction-fraud-interdiction/FRAMEWORK-DEMANDS.md`,
`src/decider2/tables/schema.py`, `src/decider2/params.py`, `src/decider2/runtime/{invoke,modes,serve}.py`,
`src/decider2/serving/dispatch.py`, `decider/modules/rules/flat_rules/module.py`,
`experimentation/jittree/test.py`.

**Primary-source verification dumps** produced by the earlier research pass, quoted throughout
§1 and graded FP/SEC/UNREACH inline —
`.../tool-results/toolu_01JdskBP89p36FaLkc23Z95z.txt` (Feedzai/Featurespace/NICE Actimize),
`.../tool-results/toolu_01RkizraQHmzvhb2zedrXqZG.txt` (Stripe Radar/Sift/Ravelin/Forter/BioCatch),
`.../tool-results/toolu_01S66KLE91uZg5vUJ3w9xa6W.txt` (AWS Fraud Detector/FICO Falcon/ACI
Worldwide/Mastercard Brighterion), under
`/home/sholto/.claude/projects/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/1835b6ef-3ce0-40a6-a48b-1a24de526ae1/tool-results/`.
