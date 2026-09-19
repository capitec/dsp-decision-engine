# 01 — Transaction fraud and scam interdiction · design sketch

An ideal-world sketch of what this project would look like if the authoring
surface could be anything. Nothing here runs. The deliverable is a **shape**.

Read alongside [`FRAMEWORK-DEMANDS.md`](FRAMEWORK-DEMANDS.md), which is the more
useful half: it is the numbered list of things this project needs from
`decider2`, each traced to the spec section that forced it and marked *satisfied
by doc 03*, *needs extension*, or *doc 03 would make this ugly*.

The spec is `../../01-transaction-fraud-interdiction.md`. Section references
below (§5.10, §6.5, Q9) are to it unless they say "doc".

---

## 1. Five decisions, and everything else follows

If you read nothing else:

**1. The 635 rules are documents. The ~40 other steps are Python. Both live in
one repository, and the split is the governance model.**
Doc 03 §1.1 argues that one rule should cost one artefact, and it is right —
for a waterfall of thirty rules written by data scientists. Applied to 635 rules
tuned by fraud analysts under four-eyes approval with a ten-minute deadline it
produces 635 Python files nobody with the authority to change them is allowed to
commit. So rules are a validated document (`ruleset/rules/*.rules.yaml`) with a
closed vocabulary, and doc 03's step-and-module surface is used for exactly the
things it is good at: enrichment, resolution, dispatch.

**2. A rule's *shape* is structure. Everything else about it is a value.**
`evaluate`, `applies_to.event_types` and `on_absent` compile. Thresholds,
action, severity, priority, queue, status and effective dates do not — they are
read by the resolver and by applicability from an attribute table. So 85% of
rule changes (§6.2) and 100% of overlays are a params-bundle swap costing
microseconds, and every rule carries **two** version counters — `version` and
`shape_version` — which makes the change class visible *in the artefact* rather
than only in a review process. That is a direct answer to Q4.

**3. Shadow isolation is a position in the pipeline expression.**
`ShadowRules` sits **after** `Resolve`. Nothing produced after a module can be
read by it, so a shadow rule influencing an outcome is not discouraged, not
tested for — it is unreachable. The test is a static lineage query over
`action_code` that runs at import time with no data (§Q9,
`tests/test_shadow_isolation.py`).

**4. There is one feature vector, and it is simultaneously the kernels' input
schema, the decision record's schema and the backtest's input schema.**
`Enrichment` produces it; `Decision` consumes nothing else. So the backtest does
not "use recorded values where possible" — it cannot do anything else, because
`Decision`'s leaf inputs *are* the recorded columns and every step in it is
pure. §5.17's equivalence requirement stops being a promise and becomes a type.

**5. The firing set is a fixed-width bitset, and per-rule detail is derived.**
Ten `uint64` columns, 80 bytes, for all 635 rules. Per-rule feature values are
not stored per rule — they are the feature vector, stored once, joined against
the pinned rule definitions at read time. On a genuinely fraudulent instant
payment where nine rules fire across three families, the alternative stores the
same numbers up to nine times, on 140 million records a month, for seven years.

---

## 2. The layout

```
fraud_interdiction/
  events/
    catalogue.py              # which of 210 fields exist on which of 12 event types, from when
    field_catalogue.csv       # ...the data behind it
    normalise.py              # §5.1 admission
  enrichment/                 # §5.2–5.6 — produces THE FEATURE VECTOR and nothing else
    __init__.py               # ClientContext | fan(Counterparty, Device, Merchant, Velocity, Model) | Completeness
    client_context.py  counterparty.py  device_session.py
    merchant.py        velocity.py      model_score.py     completeness.py
  features/
    derived.py                # the 3% of rules that need a ratio — registered steps, referenced by id
  ruleset/
    __init__.py               # THREE ruleset declarations: live, shadow, retired
    rules/
      mule_scam.rules.yaml    # 96 live MS rules — analysts edit this through a UI
      shadow.rules.yaml       # 114 candidates
      retired/2024-h2.rules.yaml   # 1 360 rules that must still replay
  decision/                   # §5.7–5.14 — consumes the feature vector, nothing else
    hard_blocks.py  applicability.py  overlays.py  resolve.py  dispatch.py
  tables/
    definitions.py            # 16 tables: versioned_table vs temporal_table
    data/                     # country_risk.csv, queue_sla.yaml, challenge_matrix.csv
  overlays/
    ADJ-0042.yaml             # one overlay, complete
    register.yaml             # the standing register — age, measured effect, days to expiry
  pipelines/
    interdict.py              # real time
    backtest.py               # 90 days — imports the SAME Decision object
  config/interdict/
    production.params.json    # ~1 900 thresholds, effective-dated, each with its own approval
    precedence.json           # action precedence as a document, not as ordering logic
    degraded_modes.json
  contracts/
    feature_vector.json       # frozen. The most load-bearing file here.
  artefacts/
    rule-sheet-MS-0208.md     # generated. What a fraud analyst signs.
    decision-record.example.json
  tests/
    test_shadow_isolation.py  test_replay_equivalence.py
```

Five placements are deliberate and the rest is ordinary.

**`enrichment/` and `decision/` are separated by a frozen contract, not by
taste.** That boundary is the entire backtest-equivalence story. Doc 07 §1's
layout has `modules/` and `pipelines/` and no notion of a cut line inside the
flow; here there is one, it is named, and CI fails if either side crosses it.

**`ruleset/` holds one Python file and three YAML files, and the ratio is the
point.** The Python declares interfaces and changes twice a year. The YAML
changes several times a day and is edited through a UI by people who do not have
commit rights. Doc 08 §3's `ruleset` kind is exactly the right idea; what it
lacks is everything in the second half of §4 below.

**`overlays/` is a sibling of `ruleset/`, never a subdirectory.** §6.5.1 and core
§7.6 prohibit merging an overlay into its base. Putting the overlay register
inside the rule set directory is the first step toward someone doing it "to keep
things simple".

**`contracts/feature_vector.json` is not documentation.** It is a frozen
interface (doc 03 §5.1's `contract=`) and the schema of a seven-year archive.

**`config/interdict/production.params.json` holds dated entries, not values.**
See §9.

---

## 3. One instant payment, event to action

R12 400 to a beneficiary added 84 minutes ago, from a phone the client first
used a day and a half ago. Here is the whole flow, and the only place it is
written down in this order is `pipelines/interdict.py`:

```python
Interdiction = (
    Normalise
    | Enrichment                  # -> the feature vector, and nothing else
    | HardBlocks                  # computed before evaluation, consumed after it
    | Applicability
    | OverlayStack
    | LiveRules                   # -> live_fire_bits, live_base_fire_bits
    | OverlayAttribution          # -> fired_on_overlay_ids
    | Resolve                     # -> action_code, reasons, trace
    | ResolveBase                 # -> counterfactual_*, the SAME module relabelled
    | Dispatch
    | Wording
    | ShadowRules                 # AFTER Resolve. Cannot influence it.
)
```

**Normalise** resolves the type (210), converts the amount at the rate carried in
the message and records that rate, and finds the message well-formed. A
malformed one would have gone through anyway, under a reduced population, with
the defect recorded — a dropped event is the one the dispute is about.

**Enrichment** runs `ClientContext`, then five independent sources under
`fan(...)`, then the completeness verdict. `fan` is not in doc 03 and has to be:
`|` means sequence, and the model score's 8 ms deadline must overlap the lookups
or the 25 ms budget fails. The card model times out. That is not an exception —
it sets `card_model_score.state = ABSENT` with `absence_reason = TIMEOUT`, and
`model_presence_code = 2`. The scam model answers 812.

The output is the feature vector: ~380 columns, 3 184 bytes, every velocity
aggregate carrying a watermark and one of `FRESH / STALE / ABSENT /
NOT_APPLICABLE`. `client_count_1min` is 3 and **stale** — 517 ms old against a
2-second tolerance, so actually fresh; `client_count_10min` is absent. The mule
watchlist is unreachable and its last good copy is 340 seconds old, inside
tolerance, so the lookup is `ABSENT` rather than fail-closed.

**HardBlocks** computes six gates into a bitset and a forced-action floor. None
hold. Note what it does *not* do: it does not short-circuit. Doc 03 §8.2 would
model this as a `Branch`, and a branch compiles to a real branch where only the
taken arm executes — which is normally the point and here is precisely wrong.
§5.7 requires that the full live and shadow set is still evaluated on a
sanctions hit, because the analyst tuning the mule family next quarter needs to
know which of her rules would have caught it independently.

**Applicability** produces a 635-bit set from six recorded scalars: event type,
segment bits, timestamp, degraded mode, rule set version, and adjustment stack
version. 94 MS rules apply, 31 CF, 88 AT, 38 FP, 51 AA. It also produces an
8-byte digest, which is the difference between a derivation and an assumption.

**OverlayStack** resolves the register at `event_timestamp` into a 40-float gain
array — five families × eight units. Two overlays are in scope (`ADJ-0031`
tightening MS amounts ×0.70, `ADJ-0038` shifting a model cut-off that is inert
here because the card model is absent) and one is in force but out of scope
(`ADJ-0042`). All three are recorded, including the one that did nothing,
because a mis-scoped overlay's only symptom is an absence.

**LiveRules** evaluates every applicable rule with no early exit, twice for the
rules an overlay touches: once at effective thresholds, once at base. Four fire:
`MS-0208`, `MS-0311`, `AT-0117`, `CF-0402`. One is unevaluable — `MS-0498`
references a velocity aggregate that is absent and its declared behaviour is
suppression, so it is recorded as unevaluable and not as "did not fire". That
distinction is what makes an aggregate outage visible in the outcome statistics
rather than as a quiet dip in catch rate.

**OverlayAttribution** is ten `andnot` instructions:
`live_fire_bits & ~live_base_fire_bits` → `MS-0208` fired only under the
overlay.

**Resolve** applies the precedence document. `MS-0208` (severity 4, hold) beats
`MS-0311` (severity 3, step-up) at the first tie-break key, so priority and
family precedence never run — and *that* is what the resolution trace records.
Action 40, source `MS-0208`, reason 4412, runners-up `MS-0311` and `AT-0117`.

**ResolveBase** is the same module, relabelled onto `live_base_fire_bits`. It
answers "what would we have decided without the overlay stack" — here, the same
action from a different rule, so the overlay bought nothing on this event. That
finding exists only because the base resolution ran, and it is what the monthly
fraud forum is for.

**Dispatch** wants queue `SCAM-1` at a 15-minute SLA. `SCAM-1` is over its depth
threshold, so the declared overflow policy reroutes to `SCAM-2` at T60, and the
reroute is recorded — so the SLA breach report can distinguish "the rule asked
for 15 minutes" from "the floor could not give it".

**Wording** renders `FRAUD_HOLD_GENERIC_V3` in en-ZA at disclosure level 2, and
the record holds both the internal reason set and the external text. A complaint
six months later is about what the client was told; a scheme representment is
about what was true.

**ShadowRules** runs last. `MS-0512` fires. It goes into a different column with
a different name, and the action was decided four modules ago.

Serial p99 ≈ 18 ms against a 25 ms ceiling. The interesting thing about the
budget is that the 6 ms allowed for 635 evaluations is wildly over-provisioned —
2 700 predicate comparisons over a 3 KB vector that fits in L1 is a few
microseconds of compiled scalar code — and the 8 ms of enrichment is the whole
problem. **The record tier makes the rules a rounding error.** If this design
fails on latency it will fail on I/O, and no amount of rule-engine cleverness
will help.

---

## 4. The hard parts

### 4.1 Six hundred rules, authored by non-engineers, live in ten minutes

The rule document is the whole answer, and three fields in it do the work.

`shape_version` versus `version` puts the change-class boundary in the artefact.
`unit:` on every tunable gives overlays something to grip (§4.4). `on_absent:`
per referenced feature forces the author to answer the staleness question while
she is thinking about fraud (§4.8).

The mechanical path for the 85% case:

| step | who | elapsed |
|---|---|---|
| analyst moves three numbers in the editor | N. Dlamini | — |
| validation: bounds, feature availability, closure | framework, in the browser | 40 ms |
| backtest over 90 days, projected to 12 columns | framework | ~3 min |
| second pair of eyes reads the generated diff | R. Pillay | ~2 min |
| write a dated params entry; `rt.params.swap()` | framework | **11 ms** |

Six minutes fourteen seconds, of which eleven milliseconds is machine time. The
15% shape case adds a staged compile of **one block** — see §4.11 — and lands
inside four minutes, which is the emergency path.

The validation that runs in the analyst's browser is the *same function* that
runs in CI and again at `stage()`. A rule that passes in the UI cannot fail at
deployment. That is what makes ten minutes a schedule rather than a hope.

What an analyst **cannot** do: reference a field that does not exist, reference
a field her event types do not carry, write arithmetic, exceed a declared bound,
create a critical rule, touch an overlay-exempt rule's exemption, or omit an
approver. Each of those is a validator, not a review note.

### 4.2 Every rule that fired, at 3 500/s, as required output

Bitsets, and derivation instead of duplication.

| | stored per event |
|---|---|
| firing set (635 rules) | 80 B — `live_fire_bits`, 10 × uint64 |
| base-threshold firing set | 80 B — only rules an overlay touched differ |
| unevaluable set | 80 B |
| shadow firing set | 80 B, separate name |
| applicable population | 28 B of inputs + 8 B digest |
| per-rule feature values | **0 B** — derived from the feature vector + pinned catalogue |

Per-rule detail is a read-time join, and it is exact because the rule → feature
map is static data in the pinned generation. The rendered form in
`artefacts/decision-record.example.json` shows what that looks like on a screen:
`$rendered_for_humans` is produced by the reader, never stored.

`fired_on_overlay_ids` is `live_fire_bits & ~live_base_fire_bits` — the whole of
"did this fire on the base threshold or only under the overlay", in ten
instructions. The counterfactual action is `Resolve` run a second time over the
base bitset.

Fixed width also pays for itself in analysis. "Incremental catch — confirmed
fraud caught by the proposal that no currently live rule caught" is an AND-NOT
and a count. Against a variable-length list of identifiers it is a join.

### 4.3 Action priority across families with different owners

`config/interdict/precedence.json` — six stages, five tie-break keys, two
override kinds, as a document. It is a lexicographic argmax over fixed integer
columns, which passes doc 08 §3.4's generic-kernel test, so **changing it costs
no compile at all**. §6.3 calls changing it "a major event": it is, in approval.
It is not a deployment.

Two things the document has to express that ordering logic hides:

- **critical-allow suppression.** `MS-0455` (a pre-notified corporate bulk run)
  is `critical: true, action: allow` and names `critical_allow_families: [CF, MS]`.
  It silences those families' non-critical firings for this event — *in the
  resolution, not in the evidence*. The suppressed rules stay in
  `live_fire_bits`.
- **critical collision.** Two critical rules asking for different actions:
  more severe wins, and `governance_exception` is emitted to both owners. The
  tie-break resolves it deterministically; it does not resolve it *correctly*,
  and §11.12's card-fraud-versus-AML standoff is resolved by the
  `family_precedence` list in the params document — a value, owned by the Head
  of Fraud, changed **without either owner editing the other's rules**.

### 4.4 Overlays, and the one place doc 03 breaks outright

Doc 02 §4 says params do not vary by record. An overlay's scope is declared over
segments, channels and event types, so the effective value of a threshold **does
vary by record**. That is the sharpest collision in this sketch.

The resolution is that overlays do not produce a per-record parameter set. They
produce a **gain vector**: 5 families × 8 units = 40 floats, built once per event
by iterating the ≤40 overlays in the register. A rule's emitted predicate reads

```
amount_zar_cents > tunables.min_amount * gain[MS][amount_zar_cents]
```

— one multiply. Cost is O(overlays), never O(overlays × params), which satisfies
"resolving an overlay stack of 40 entries must not be a per-rule cost" by
construction rather than by optimisation.

This is why `unit:` is a required field on every tunable. An overlay names a
*class* of threshold — "every tunable in family MS with unit
`amount_zar_cents`" — and without the unit a sensitivity dial has nothing to
join on. It is also how a dial is stopped from moving a *band* cut-off, which
would silently reinterpret a versioned table.

`overlay_exempt` rules emit the same line **with the gain term absent**, decided
at codegen. An exempt rule therefore has no runtime path by which a dial could
reach it, which is stronger than checking a flag and is what §6.5's "a
tightening dial applied in a hurry must not be able to loosen a control the Bank
is required to operate" actually requires.

Expiry is four validators on the register, not a process: `effective_to`
required, duration ≤ 90 days, `review_date` required and ≤ `effective_to`, two
distinct named approvers and a backtest reference. Lapse needs no job — the
window is compared against `event_timestamp` on every event, so an overlay
lapses within one assessment and the lapse is recorded on everything after it.
`overlays/register.yaml` carries `ADJ-0011` at 431 days and two renewals,
flagged `ESCALATED` with the four rule changes that must be re-costed: §11.3,
found by the register rather than hidden by it.

The genuinely awkward part is rule-scoped overlays — scope restriction and
severity shift can name individual rules, which needs a 635-wide vector that
cannot be rebuilt per event at 12 000/s. It is memoised per **scope class**
(event type × channel × overlay-relevant segment bits) by
`keyed_materialisation`. It is a pure function of the key so a hit and a miss
cannot differ, but it is the one piece of this design whose correctness argument
is a cache-invalidation argument, and it is the first thing I would load-test.

### 4.5 Shadow rules

Three mechanisms, and only the first two are worth having.

1. `ShadowRules` is **after** `Resolve` in the pipeline expression. Nothing
   produced after a module is readable by it.
2. `Resolve.reads` is a Python list in `decision/resolve.py` that does not
   contain `shadow_fire_bits`; a CI lineage assertion over `action_code` fails
   if it ever does. No data, no execution, runs at import.
3. The two populations write disjoint column names, so a downstream consumer
   that unions them gets a schema error rather than a wrong answer.

Promotion moves a rule's document from `shadow.rules.yaml` to its family file.
That is a shape change — it recompiles one live block and one shadow block, ~4
seconds — and it is the **correct** cost, because promotion is exactly the change
that starts affecting outcomes. The rule keeps its `rule_id`, so shadow history
and live history join on one key.

`tests/test_shadow_isolation.py` keeps a skipped fourth test: replay a day of
traffic with the shadow population emptied. It is recorded there as the thing a
system without static lineage has to do, and it is worse in every way — a day of
compute, a property of one day's traffic rather than of the definition, and a
shadow rule that matters on 0.001% of events passes it.

### 4.6 Twelve event types whose fields differ

`events/field_catalogue.csv` declares, per field: dtype, unit, nullability, the
event types that carry it, and `available_from` / `available_to`. Rule
validation then intersects a rule's declared event types with each referenced
field's availability, over the whole of the rule's effective window, and
produces:

```
MS-0208 references 'three_d_secure_outcome'. That field is carried by event
types {111} only; the rule declares {210, 211, 313}. Either narrow the rule's
event types to {111} or reference a field common to all four.
```

Doc 07 §5's `build --schema` takes one input schema. One schema cannot express
twelve variants differing by 30 fields, and it certainly cannot express a field
that did not exist before 2026-04-01. §11.6's merchant-initiated indicator is in
the CSV with `available_from: 2026-04-01`, and §11.5's QR event type 317 with
`available_from: 2027-01-01` — both so that an 18-month backtest is honest about
what was computable when, instead of quietly treating a non-existent field as
absent and understating a rule's hit rate by the fraction of the window that
predates it.

### 4.7 Real time and backtest, identical by construction

```python
Interdiction = Normalise | Enrichment  | Decision
Backtest     =            ReadRecords  | Decision
```

`Decision` is the same object — the same pydantic instance, compiled from the
same source, reading the same columns. There is no path by which the two could
disagree because there is only one of them.

The remaining difficulty is that production ran against however many rule set
versions and overlay stacks were live across 90 days, while a backtest runs one
candidate. That is `PartitionByEffective`: the frame tier groups the 420 M
events by which generation was in force and invokes the kernel once per
partition — ~300 invocations of ~1.4 M events. Params stay per-invocation (doc
02 §4's fixed-type guarantee survives) and results are per-record correct.
`baseline="as_at_today"` collapses it to one partition, which is the other
admissible answer, and the pack must state which one it used.

Throughput: the kernel is ~14 core-minutes for 420 M events. The binding
constraint is reading 420 M × 380 columns ≈ 1.7 TB. So the backtest projects to
the columns the candidate can actually read, which comes from
`Decision.lineage(...)` with nothing executed. **Static lineage, argued for in
doc 02 §5 as a governance property, turns out to be this project's principal
performance mechanism.** That is the strongest argument in the doc set for
declarative frame operations and nobody in the doc set makes it.

One consequence worth stating: `fastmath` is banned project-wide. Acceptance
criterion 6 wants exact equality between two entry points that take different
codegen paths, and EXPERIMENTS.md §I measured `fastmath` making 46–73% of rows
differ by up to 17 ULP for a 1.09× gain. A threshold comparison at 1 ULP is a
different decision for the client standing at the till.

### 4.8 Missing, stale, and structurally absent

`Observed[T]` = `(value, state, watermark)`, four states:

| state | means |
|---|---|
| `FRESH` | within this window's tolerance |
| `STALE` | older than tolerance — **not** missing |
| `ABSENT` | could not be retrieved — **not** zero |
| `NOT_APPLICABLE` | structurally absent for this event type |

Doc 03 §1's three null tiers are all one bit wide, and doc 03 explicitly rejects
a `.value/.valid` wrapper for exposing an unchecked accessor. I am reintroducing
a wrapper and the justification is that the domain needs four states and a
watermark, not one bit — and that only the ~30 hand-written enrichment steps
ever see it. **No rule document mentions `Observed` at all**; a rule declares
`on_absent:` per referenced feature and the generated code does the rest.

The fourth state earns its place. A card-present event legitimately has no
device fingerprint 11% of the time, and a login has no beneficiary velocity
ever. Collapsing those into `ABSENT` makes an aggregate outage and a login
indistinguishable in the outcome statistics — which is the exact failure §5.4 is
written to prevent.

Two domain "unknowns" are deliberately **not** `Observed` states: confirmation-
of-payee's `not-available` band and merchant reputation's `UNKNOWN_BAND`. Those
are real, meaningful values that rules key on directly. Modelling them as
absence would push those rules into their `on_absent` behaviour and silently
switch them off on exactly the population they exist for.

### 4.9 Retired rules that replay for 540 days

Retirement is an `effective_to` and nothing else. It takes effect at the
instant, on every event after it, with no compile — and it is provable by the
per-minute firing aggregate going to zero at exactly that minute, which is
§9.3's retirement evidence without scanning 140 million records.

The rule stays in the live kernel, date-gated, until the next housekeeping
generation drops it. Replay of an earlier event pins that earlier generation,
whose kernel still contains it. So the live kernel does not accumulate 1 360
dead rules, and `AA-0093` retired in June 2024 is still replayable in 2031.

The thing this depends on, and it is not free: a generation must be **rebuildable
from pinned source**, and the record must name the toolchain, because a rebuild
in 2031 on a different numba is not obviously the same computation. Doc 08 §8
records a compiled artefact id and not the toolchain that produced it; over 540
days that is not enough. See FRAMEWORK-DEMANDS #24.

### 4.10 Degraded behaviour

Four modes, declared in `config/interdict/degraded_modes.json` as a decision
table with a generic kernel — so the Head of Fraud's quarterly change to which
rules a mode suspends is a value change. Each mode names its suspension set (38
rules) and its activation set (15) **in advance**, and both are recorded on
every event assessed while the mode was in force.

Two details the spec forces and a naive design loses:

- The compensating set is validated to reference no model score and no velocity
  aggregate. A compensating rule that reads the thing that just broke is not a
  compensating rule, and that is checkable statically from the rule → feature
  map.
- §11.7 splits the model in two, giving four presence combinations. They are
  **enumerated** in the document, not derived. Deriving them is how a serving
  incident on the scam model quietly switches off the mule family.

`MS-0498` in `mule_scam.rules.yaml` is a compensating rule: blunter than
`MS-0208`, deliberately worse on false positives, `activated_in_modes:
[restricted, fail_closed]` so it is never applicable in normal running.

### 4.11 Big tables, small tables, and the cost of the 522nd rule

`tables/definitions.py` has two kinds. `versioned_table` is `key → row` resolved
at `event_timestamp` — doc 03's `Table` sketch, and adequate. `temporal_table`
is `(key, instant) → row`, stored as **intervals** rather than snapshots. An
hourly refresh changing 0.3% of a 2 M-row watchlist appends ~6 000 rows; seven
years is ~370 M interval rows, a partitioned parquet dataset, not 61 320 copies
of a table. §11.9's growth to 11 M entries refreshed every five minutes is ~2×
the append volume, not 66×.

Backing store follows from the declaration — dense array at 1 024 rows, sorted
at 50 000, hash index at 3.2 M — and the step's expression is identical in all
three. `diffable` degrades honestly to `"summary_only"` for merchant reputation,
because a 3.2 M-row daily diff is not a reviewable change list and declaring
that is better than discovering it.

**The 522nd rule** (Q15): rules are assigned to one of 16 compile blocks by a
*stable hash of `rule_id`*, never by position. Adding a rule recompiles exactly
one block of ~33 rules — about two seconds — and the other fifteen come from the
numba cache byte-identical. Position-based blocking would shift every rule after
the insertion point and recompile all sixteen. The 900th rule costs the same as
the 522nd.

---

## 5. What a non-engineer sees

Three generated artefacts, and none of them is a diagram. Nobody reviews 635
diagrams.

**The rule sheet** (`artefacts/rule-sheet-MS-0208.md`). One page, generated from
the definition plus the params generation plus the overlay stack. Its most
important table has five columns: the condition, the value the author wrote, the
value **in force right now**, and why they differ. `MS-0208`'s amount reads
"R8 000 authored / R4 550 in force — threshold change to R6 500 on 2026-09-11,
then ×0.70 by overlay ADJ-0031, which expires in 47 days". An analyst confirming
"this matches what I authorised" and a regulator reading an inventory entry are
looking at the same sheet.

**The overlay register** (`overlays/register.yaml`). Every overlay in force, its
age, its renewal count, its measured incremental catch and its days to expiry —
with `ADJ-0011` sitting there at 431 days flagged `ESCALATED` and naming the four
rule changes that must be re-costed before it can be withdrawn.

**The params diff.** Because every threshold is a dated entry with its own
approval block, the approver's screen is the diff of two entries, and the
approval routing keys on `change_class`: `threshold` takes the six-minute path,
`shape` takes the ten-minute one, `status` on a critical rule takes two named
approvers.

What a non-engineer never sees: `Observed`, a bitset, a kernel, a block
assignment, `fan`, or the word numba.

---

## 6. Explaining one decision

`artefacts/decision-record.example.json` is the whole answer and is worth
reading end to end. Four properties:

- **Nothing in it is a pointer into a mutable store.** Every enrichment value is
  inline, because the profile store will not hold this value in 540 days.
- **Every version is named**: rule set, adjustment stack, precedence, params
  generation, params origin, structure fingerprint, compiled artefact, and the
  numba/LLVM/CPU triple.
- **The resolution trace records what discriminated**, not just what won —
  "severity, at the first key; priority and family precedence never ran". That
  is the answer to "why did my rule not take effect although it fired".
- **The counterfactual is there unconditionally**, not on request.

Three of the spec's §9.1 questions, traced through it:

| question | answered from |
|---|---|
| "Why was this payment held, and by which rule?" (Ops, seconds) | `action_source_rule_id` + the rule sheet, rendered for the queue screen |
| "Reproduce this decision exactly" (Disputes, 540 days) | the generation block + the inline feature vector → `Decision.score()`, checked against the applicable digest |
| "Was this beneficiary on the mule list at the time?" (7 years) | `tables_read[].as_at` → the interval store, which answers "added 09:11 on 2 March by the consortium feed, removed 11:00 on 18 April" |

---

## 7. What moves when a value moves, and when structure moves

| change | artefact touched | machine cost | who | elapsed |
|---|---|---|---|---|
| a threshold | params document, new dated entry | bundle swap, ~11 ms | analyst + one approver | ~6 min |
| a rule's action, severity, priority, queue | params document | bundle swap | family owner + one | ~6 min |
| retire a rule | params document — `effective_to` | bundle swap | family owner + one | ~6 min |
| an overlay, applied or withdrawn | overlay register | bundle swap | duty officer + one named | **< 2 min** |
| degraded-mode suspension sets | interior document, generic kernel | none | Head of Fraud | quarterly |
| action precedence, family precedence | precedence document, generic kernel | none | Head of Fraud | annual |
| a rule's predicates | rule document | **one block compiles, ~2 s** | family owner, four eyes | ~10 min |
| promote shadow → live | rule document moves file | **two blocks, ~4 s** | family owner, four eyes | ~10 min |
| a new derived feature (`feat:*`) | Python | rebuild + redeploy | engineer | days |
| a new enrichment source | Python + the frozen contract | rebuild, contract major version | engineer | weeks |

The line that matters is between rows 7 and 9: an analyst can compose registered
features into new rules all day and can never write arithmetic. That is doc 08
§3.2 working as designed, and it costs what doc 08 says it costs — about sixteen
rules a year need an engineer. Two things make that survivable and neither is in
doc 08: the derived-feature catalogue is **pre-stocked** from the same
declaration that generates the 168 velocity aggregates, so the analyst's ratio
usually already exists; and the validator's error **names the closest registered
feature and prints the `@feature` stub to send to engineering**, so a dead end
becomes a pull request. Without those, an analyst under attack at 19:40 on a
Friday inlines a literal into an adjacent rule, and that is the 546-`pl.lit()`
failure happening again.

---

## 8. Two clocks — an ambiguity in the spec

§5.9 says the overlay stack in force is resolved at `event_timestamp`. But a
rule's thresholds are whatever the deployed params generation holds at
`assessment_timestamp`. For a SIM change that arrives three hours late those are
different instants, and the spec does not say which wins.

The sketch resolves it by recording **both instants and both versions** and
making replay read the recorded generation rather than re-deriving one. That
keeps determinism (§8's requirement is about identical *recorded* inputs
reproducing the answer) and makes the discrepancy visible rather than assumed
away — which is what §9.3 asks for in the deployment-versus-effective-date case
anyway. It should be an explicit decision in the spec rather than a decision I
made in a sketch.

---

## 9. What I could not make elegant

Four things. They are in `FRAMEWORK-DEMANDS.md` with numbers; here is the honest
summary.

**The gain vector is a good answer to the wrong half of the problem.** Family ×
unit dials are clean. Rule-scoped overlays are not, and the memoised
scope-class materialisation is a cache in the middle of a determinism guarantee.
It is correct — pure function of the key — but "correct because the cache key is
complete" is a weaker sentence than everything else in this design.

**Two version counters on a rule is one more than anyone wants.** `version` and
`shape_version` make the change class visible, which is worth it, but every rule
now carries a number whose only job is to say which of two deployment paths a
change takes. A framework that derived the change class from the diff would be
better, and I could not see how to make that legible in a YAML review.

**The rule document is close to a programming language and I am not sure where
the line is.** `all`, `any`, 13 operators, `between`, `is_in`, `is_member`,
param references, registered feature references. Doc 08 §3's closed vocabulary
holds — there is no expression string and every node has an emitter — but it is
a *large* closed vocabulary, and each operator is an emitter, a validator, a
renderer and a row in the reviewable artefact. The 17-predicate rule in §5.10 is
going to look like code on the sheet no matter what I do with the template.

**`fan()` is a performance concept wearing a semantics costume.** It declares
independence and permits concurrency, which is honest, but it is the only
combinator here whose reason for existing is latency. `|` with an optimiser
would be worse (doc 02 §1.2's argument against inferred fusion applies), so an
explicit combinator is right — but three combinators plus `fuse` plus
`parallel` plus `fan` is five ways to say something about execution, and a
project using all five is a project with a taxonomy problem.

---

## 10. Against the spec's sixteen questions

| Q | Where |
|---|---|
| 1. Is a flat rule set a core component kind? | Yes — `ruleset` with a `population`, compiled in stable blocks, returning bitsets. §1, §4.11 |
| 2. "Every rule that fired" at 3 500/s | Bitsets + derivation. §4.2 |
| 3. Applicable population per event | Six recorded scalars + an 8-byte digest. §3, `decision/applicability.py` |
| 4. What is the unit of change | Shape vs value, visible as two counters. §1, §7 |
| 5. What structurally is an adjustment | A gain vector over (family × unit), plus a memoised rule-scoped patch. §4.4 |
| 6. Live rule set swap | doc 08 §4 unchanged — stage / activate / rollback, generation pointer read once per invocation |
| 7. Action precedence as something readable | A document with a generic kernel. §4.3 |
| 8. Stale/absent declaration kept honest | `on_absent:` is a required field per referenced feature. §4.8 |
| 9. Shadow isolation structurally | Topological position + a static lineage assertion. §4.5 |
| 10. Backtest proved equivalent | One `Decision` object, one feature vector. §4.7 |
| 11. 2 M-row watchlist replayable for 7 years | Intervals, not snapshots. §4.11 |
| 12. Sixteen tables, five owners, two hot | Two table kinds; backing follows the declaration. §4.11 |
| 13. A feature that did not exist before a date | `available_from` in the catalogue. §4.6 |
| 14. The generated description | Rule sheet + overlay register + params diff. §5 |
| 15. Cost of the 522nd rule | One block of 16. §4.11 |
| 16. Where does the vocabulary belong | `rule_id`, `fired_rule_ids`, `action_code` are **this project's**; `population`, `Observed`, `temporal_table`, `precedence` are the framework's. FRAMEWORK-DEMANDS #37 |
