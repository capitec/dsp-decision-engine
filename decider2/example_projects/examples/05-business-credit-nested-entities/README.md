# 05 — Business credit granting with nested entities: an ideal-world sketch

An answer to
[`05-business-credit-nested-entities.md`](../../05-business-credit-nested-entities.md),
written as the shape it would take in a codebase if the authoring surface
could be anything. Nothing here runs. Bodies are `pass` with a one-line
comment; the value is in the signatures, the composition expressions, the
grain declarations and the table files.

Doc 03 is departed from in twenty-two places, and every departure is numbered
in [`FRAMEWORK-DEMANDS.md`](FRAMEWORK-DEMANDS.md) with the spec section that
forced it. Read that file second and this one first. The short version of
what this sketch is *for*: doc 06 records O5, "ragged per-record
collections," as deprioritised and unresolved. This project is the workload
that makes that unresolvable — 1..40 entities, each with 0..60 events, rolled
up by rules that are neither worst-of nor average, attributed back to one
entity and one event, replayed identically years later. Everything else in
the spec (pricing, overlays, financials) is ordinary `decider2` sitting on
top of whatever answer nesting gets. The two inventions that answer it —
`grain` and `Gather` — are the sketch's central contribution, and every other
file in the tree is either an instance of them or a consequence of having
them.

---

## 1. The shape, in one page

```
05-business-credit-nested-entities/
  grains.py                     Read first. The four grains this project computes
                                at: Application, Entity, Event, Candidate — and the
                                severity/criticality/disposition total orders.
  vocabulary.py                 Grain-pinning on top of doc 03's project vocabulary
                                map: which library capability is consumed at which
                                grain, and the de-duplication identity (entity_key).

  business/                     Stages 5.2-5.3. Application grain. Owner: Policy,
    regime.py                   quarterly, plus Compliance on legislation change.
    disqualification.py         Fourteen absolute rules, collect="all", a policy-
                                class mapping table separate from the rule count.

  entities/                     Stage 5.4. Entity grain. Owner: Policy, quarterly.
    criticality.py               Five causes of one class, each a tap.
    disqualification.py          12 rules x 3 classes, by_class(), emit_into.
    adverse/                     Stages 5.5-5.6. Owner: Policy, quarterly and after
      classify.py                 loss events. THE innermost work: one event, no
      flags.py                    collection, no index, no loop, anywhere in here.
      gather.py                   THE central file. Event -> Entity, two ways.
      verdict.py                  A verdict() over the gathered scalars.
    scoring/                     Stage 5.7. Owner: Model Team, on release.
      families.py                 Two scorecard families, one branched collection.

  people/                       Stage 5.8. Entity -> Application, four times.
    weights.py                    PP-01..04: inclusion, coverage, two-pass weights.
    blend.py                      PP-05, PP-11: the log-odds blend, computed x3.
    caps.py                       PP-06, PP-07: worst-of, dominating the blend.
    surety.py                     PP-08, PP-09, 5.11: sureties, guarantors, ceilings.

  financial/                    Stage 5.9. A fifth grain, Period, 1..3 unequal.
    measures.py                   Owner: Business Credit Risk Policy.

  grade/                        Stages 5.10-5.11. Application grain.
    combine.py                    The blend, the override, the authority table.
    overlay_stack.py              The declared 11-position, 4-grain stack, and
                                  the unadjusted spine as a Shadow.

  pricing/                      Stage 5.12. A derived grain, Candidate, up to 2200.
    candidates.py                 Enumerate, not search.
    price_one.py                  Price ONE candidate. parallel(...) in the pipeline.
    select.py                     The one fold that chooses the offer.

  structure/                    Stage 5.1. Frame tier, not record tier.
    resolve.py                    Graph -> bounded tree: a frame-tier Loop, then a
                                  Gather that de-duplicates by identity.

  evidence/                     The three explainability inventions, grouped
    attribution.py                because each one's docstring says doc 03 has no
    counterfactual.py             story for it: a cross-grain join, a re-run under
    disclosure.py                 perturbation, and a disclosure boundary that runs
                                  along a grain boundary by coincidence.

  validation/                   Stage 5.13. Assertions about the flow, not the
    consistency.py                 business — a different kind of rule (FV-01..08),
                                  plus the three-valued outcome gate and conditions.

  pipelines/
    business_facility.py          The whole flow, thirteen stages, nine grain
                                  shifts. Read this second.
    monthly_reassessment.py       The 6.7M-event batch; partial re-runs; fan-out.

  tables/                        *.csv, effective-dated. entity_disposition.csv,
                                event_thresholds.csv, event_ageing.csv,
                                rate_card.sample.csv shown; ~16 more implied.

  README.md                     This file.
  FRAMEWORK-DEMANDS.md           Twenty-two numbered demands on decider2.
```

### Why this layout

Doc 07 §1 gives `modules/`, `pipelines/`, `config/`, `contracts/`, `schemas/`,
`tests/`, with every rule directory nested under one `modules/`. This project
does not have a `modules/` directory at all, and that is deliberate rather
than an oversight: doc 07 §1's point is that "a module directory is the unit
of reuse and audit," and grouping business/, entities/, people/, financial/,
grade/, pricing/, structure/, evidence/ and validation/ under one more path
segment buys nothing when nothing else competes with `modules/` for space at
the top level — nine stage-groups become nine top-level directories instead
of nine directories one level deeper, and the spec's own stage numbers
(§5.1–§5.13) are legible directly in `git log --stat` output without the
extra segment.

**The stage groups are split by grain and by owning team together, and where
those two axes disagree, the split follows the team.** `entities/` holds two
subdirectories, `adverse/` and `scoring/`, both at the Entity grain but
changing on different cadences — Policy retunes the adverse thresholds
quarterly and after a loss event; Model Team releases a new scorecard on its
own schedule. `people/` is a *sibling* of `entities/`, not nested under it,
even though every file in it consumes the Entity collection, because it rolls
that collection **up** to the Application grain rather than computing **at**
the Entity grain, and it is owned by the team that writes PP-01 through
PP-11 as one connected policy, not by whoever owns entity-level rules.

**`structure/` is separate because it is frame tier, not record tier** — doc
02 §1 draws that line at "is this about one record, or about many rows," and
collapsing a graph to a tree is inherently the second thing. Nothing else in
the project is frame tier except the two-line `CrossApplicationView` at the
bottom of `evidence/attribution.py`.

**`evidence/` groups three files that would otherwise be scattered by
subject** (attribution reads like part of the roll-up, disclosure reads like
part of the outcome gate, counterfactuals read like part of pricing) because
each one's own docstring says the same thing in different words: "doc 03 has
no story for this." Attribution is a cross-grain join; `Shadow` is a
re-evaluation under a declared perturbation; disclosure is a grain-level
default. Putting them together makes the directory itself a signal — these
are the explainability inventions, not business logic that happens to touch
audit fields.

**`validation/` is its own directory because §5.13 says its rules are "a
different kind of thing"**: FV-01 through FV-08 each recompute something the
flow already produced and require agreement, which makes them assertions
about the *flow*, not about the business — a defect if they fail, not a
decline.

---

## 2. Walkthrough: one application, structure to sanctioned facility

Follow one application through `pipelines/business_facility.py`, which
composes the thirteen spec stages into seven named blocks:

```python
business_facility = (
    _structure | _business | _nested | _people | _grade | _pricing | _close
).with_vocabulary(vocabulary).with_overlay_stack(STACK)
```

**`_structure`** (`structure/resolve.py`). The disclosed structure arrives as
a graph: a company held by a trust and a holding company, the holding company
held by the same two people who are trustees of the trust. `ExpandStructure`
is a bounded `FrameLoop` — three rounds of join → antijoin (cycle
truncation) → filter (the 5.0% materiality floor) — and `DeduplicateEntities`
is a `Gather("path", into=Entity, on="entity_key", ...)` that collapses every
path reaching the same natural person into one Entity row: ownership summed,
role taken as the most senior, control disjoined, **every path retained** in
`paths = collect_of("path_string", max=8)`. Say this produces forty entities
for our application — the p99 case — including entity 7, Ms M. Dlamini, who
reaches the applicant by two paths and is required to provide a personal
surety.

**`_business`** (`business/regime.py`, `business/disqualification.py`). The
applicant is a private company with turnover above R1m, so
`regulatory_regime_code` resolves to unregulated. `BusinessDisqualification`
evaluates all fourteen rules with `collect="all"` — none fire here — and the
flow continues carrying the complete (empty) fired set rather than a single
"passed" boolean, because a business that *had* failed four rules needs to be
told all four.

**`_nested`** — the five grain shifts that are the reason this document
exists:

```python
_nested = (
      Each(Entity, Criticality | EntityDisqualification)
    | SyntheticEvents
    | Each(Event, fuse(ClassifyEvent | EventFlags))
    | EntityAdverseFacts                           # Gather(Event -> Entity)
    | Each(Entity, EntityAdverseVerdict | EntityScoring)
)
```

Entity 7's criticality resolves `CRITICAL`, with `criticality_cause_code`
ranked to `critical_by_surety` even though `critical_by_ownership` also holds
at 32.0% effective ownership — both booleans are tapped, so the pack can show
she would still be critical on either ground alone. Her forty-one events (the
worst case for one entity in this run) each go through `ClassifyEvent` —
one function call per event, no loop — and event 41, a R184 000 civil
judgment dated 2025-03-14, unsatisfied, classifies `DISQUALIFYING` under
`AE-C-03` against a threshold that is itself overlaid: sector 412's judgment
materiality threshold was halved by `ADJ-05-014`, approved by Credit
Committee, so the R50 000 base becomes the R25 000 threshold event 41's
R184 000 exceeds. `EntityAdverseFacts` (a `Gather`) rolls her forty-one
events into scalars — `disqualifying_count = 1`, and eleven others — and
`EntityAdverseVerdict`, a `verdict(collect="all")`, resolves `AE-R-01` (any
event disqualifying) as the binding rule, `DISQUALIFYING`, attributed to the
witness `{event 41}`.

**`_people`** — four `Gather`s over the same Entity collection
(`people/weights.py`, `people/blend.py`, `people/caps.py`,
`people/surety.py`), described in full in §3.2 below. `PeopleCaps`'s PP-06
row 1 fires: entity 7 is included in the blend (`in_blend`, she owns above
5%) and carries a disqualifying adverse verdict, so the rule's witness names
her (`names_entity="binding_disqualifying_entity"`) and the business outcome
becomes `business_decline`, reason 5402 — regardless of what the financial
and behavioural components would otherwise have said, because PP-06 dominates
the blend rather than averaging into it.

**`_grade`, `_pricing`** still run. `BusinessGrade`, `UnadjustedSpine`,
`EnumerateCandidates | Each(Candidate, parallel(PriceCandidate)) |
SelectOffer | SelectedOffer` all execute and are all recorded — a declined
application still has a complete pricing ledger, because §5.12's "Records"
clause does not carry an exception for declines and a committee reviewing a
disqualifying decline still asks what the facility would have looked like.

**`_close`**. `FinalValidation.outcome_code` reads `business_arod_verdict`
(clear), `people_caps_verdict` (decline), and seven other inputs, and
resolves `decline`. `DisputeShadow` runs regardless — event 41 is not
disputed, so its `outcome_code_without_disputes` agrees, and
`dispute_would_change_outcome` is `False`. `AttributionSpine` then does not
*compute* the decline reason; it *joins* five levels of attribution each
grain already recorded (§5 below). `ReasonSets` produces the two payloads:
the business is told reason 5402, "an individual associated with the business
has adverse credit information," with the route by which Ms Dlamini can
obtain her own reasons; the internal record names her and event 41 directly.

---

## 3. How each hard part is expressed

### 3.1 Two levels of ragged nesting — the grain

`grains.py` declares four: `Application` (the caller's own grain),
`Entity` (`key=("application_id", "entity_id")`, `parent=Application`,
`identity="entity_key"`, `capacity=Capacity(40, on_exceed="suspend",
flag="structure_unresolved")`), `Event` (`parent=Entity`,
`parent_identity="entity_key"` — not `entity_id`, because "an event follows
the person, not the path" — `capacity=Capacity(60, on_exceed="suspend")`),
and `Candidate` (`derived=True`, `capacity=Capacity(2200, on_exceed=
"truncate")`). A grain is a key, a parent, a capacity, an identity and a
canonical order, and `entities/adverse/classify.py` states the payoff
directly: "every function here takes one event and returns one value... The
collection exists in `pipelines/business_facility.py` as `Each(Event, ...)`
and nowhere else." A 4-entity application and a 40-entity application compile
to the *same kernel*; nothing recompiles when fan-out changes. See
FRAMEWORK-DEMANDS D01.

### 3.2 A roll-up that is neither worst-of nor average — Gather, then a rule set

`entities/adverse/gather.py`'s `EntityAdverseFacts` turns forty-one raw event
rows into eleven named scalars — `count(where="is_minor_recent")`,
`sum_of("unsatisfied_amount", where="is_unsatisfied",
nulls="exclude_and_count")`, `best_of("event_severity_code",
tie_break=(desc("unsatisfied_amount"), asc("event_id")), lift=[...])` — and
`entities/adverse/verdict.py`'s `EntityAdverseVerdict` is an ordinary
`verdict(collect="all")` reading those scalars as if they were any other
inputs. The roll-up is not a new component kind; it is a fold (`Gather`, the
one genuinely new thing) composed with a rule set (`verdict`, which the
project needs anyway for §5.3 and §5.4). AE-R-10 ("trailing count exceeds
preceding by 3") *escalates* the otherwise-computed verdict by one class
rather than giving its own, which is why `verdict` needs a declared
`modifier_order=("gives", "escalates", "floors", "caps")` — cap-then-escalate
and escalate-then-cap disagree and the spec does not say which, so the
project decides once, in one place (FRAMEWORK-DEMANDS D11).

The same shape repeats at the Application grain, four separate times over
one Entity collection, because four consumers roll it up four different
ways: `people/weights.py`'s `WeightTotals` (raw weight sums for PP-03/04's
two-pass renormalisation), `people/blend.py`'s `PeopleBlendFacts` (the
log-odds sums for PP-05/PP-11), `people/caps.py`'s `PeopleCapFacts` (PP-06's
worst-of, which deliberately does **not** filter on `in_blend`, because
PP-07 says exclusion is from the average, not from the rules), and
`people/surety.py`'s `SuretyCoverFacts`/`GuaranteeCoverFacts` (PP-08/PP-09,
sum-and-cap rather than blend at all). Four `Gather`s, not one `Gather` with
four consumers arguing about its output.

### 3.3 Attribution surviving two levels — a free witness, joined by identity

Every fold in `Gather` carries a witness — the child rows that produced it —
as a bitset over ordinal position, resolved to real identities
(`Event.identity`, i.e. `event_id`) only at materialisation. Every `verdict`
carries a binding rule id and a fired-rule mask at its own grain. Nothing
*assembles* a cross-grain attribution: `evidence/attribution.py` declares

```python
SPINE = attribution_spine(
    name="business_decline_attribution",
    levels=[
        (Application, "people_caps",              names=Entity),
        (Application, "business_disqualification", names=None),
        (Entity,      "entity_adverse_verdict",    names=Event),
        (Entity,      "entity_disqualification",   names=None),
        (Event,       "classify_event",            names=None),
    ],
    identity={Entity: "entity_key", Event: "event_id"},
    primary="primary_reason_code",
)
```

a declared join of what each grain already recorded about itself. No level
knows about the level below it, and a third level (change scenario 13) adds
a row, not a rewrite. See §5 for what the rendered chain looks like, and
FRAMEWORK-DEMANDS D02, D05 and D13.

### 3.4 A graph collapsed to a bounded tree, with de-duplication

`structure/resolve.py` is frame tier because collapsing a graph is set-shaped
work: `ExpandStructure = FrameLoop(ExpandOneLevel, carries=["frontier",
"resolved", "path_so_far"], max_iterations=3, on_exhausted=
"flag:depth_limit_reached")`, where each round is a declared
join → antijoin (cycle truncation, flagging `cycle_truncated`) → filter (the
materiality floor). `DeduplicateEntities` is then a `Gather("path",
into=Entity, on="entity_key", ...)` over the *pre-resolution* path rows —
the same construct as every other `Gather` in the project, which is the
point: de-duplication gets the same order-independence guarantee and the
same free witness (`paths = collect_of("path_string", max=8)`) as the
adverse roll-up does, for nothing extra. `Entity.materialise()` then applies
`Entity.capacity` and `Entity.order`; a breach raises `structure_unresolved`
and **suspends** — never a decline, never an approval. See
FRAMEWORK-DEMANDS D04 and D09.

### 3.5 Two scorecard families over one heterogeneous collection

`entities/scoring/families.py` answers spec §13 Q9 with nothing new: a
`Branch(is_natural_person_entity, ScorePersonal, ScoreCommercial,
modifies=[...])` nested inside `Each(Entity, ...)`, where each arm is itself
`Situation | Branch(scoring_situation_code, [Scorecard, ScorecardThin,
FallbackGrade, Unscoreable], modifies=[...])` — a second, four-way branch on
whether the entity is scored, thin-file, no-hit, or unscoreable, because
those four "must not collapse." Eight arms in total, only one of which
executes per entity. The residual cost this doesn't erase: the Entity frame
must carry all 38 `BUS-PERS-01` characteristics and all 29 `BUS-COMM-01`
ones as columns, one side null for every row, because the frame is columnar
and a row is one entity regardless of which arm it takes — stated honestly
as FRAMEWORK-DEMANDS D22, with no fix offered.

### 3.6 The non-monotone pricing negotiation over 105 600 cells

`pricing/candidates.py` declares the up-to-2 200 `(amount, term)` pairs as
`Candidate`, a *derived* grain enumerated by `Enumerate(Candidate,
from_=cross(RATE_CARD_BANDS, PERMITTED_TERMS), where=[...])`, rather than
searched for with a `Loop`. `pricing/price_one.py`'s `PriceCandidate`
computes the security type, the 105 600-cell rate lookup, the instalment and
the DSCR for one candidate — the same authoring shape as
`entities/adverse/classify.py`, wrapped in `parallel(...)` because it is the
one uniform body in the flow. `pricing/select.py`'s `SelectOffer` is one
`best_of("candidate_amount", where="is_admissible", tie_break=(asc(
"term_months"), asc("total_cost_of_credit"), asc("candidate_id")))` — largest
admissible amount, shortest term, lowest cost, a stable identity as the last
tie-break so the order is total. Because the fold sees every candidate,
"bisection on the amount is invalid" (spec §5.12) is never a risk anyone has
to remember: nothing traverses the space in an order that could depend on
monotonicity. See FRAMEWORK-DEMANDS D14.

### 3.7 Partial re-runs

`pipelines/monthly_reassessment.py`'s `refresh_one_entity` calls
`rt.reapply(monthly_reassessment, prior=prior, changed=Entity.identity(
entity_keys), explain=True)`. The grain DAG makes the invalidation set
derivable without running anything: `Each(Event, ClassifyEvent)` recomputes
only the refreshed entity's event rows; `Gather(Event -> Entity)` recomputes
only that entity's row; `Each(Entity, Verdict | Scoring)` likewise; the
Application-grain `Gather`s read every child row but only one changed, so
the *fold* reruns while the other thirty-nine rows are read, not
recomputed; everything at Application grain reruns. The proof is the same
property that gives ordering independence — every fold is
commutative-associative and every step is pure — so it costs nothing extra
to state. See FRAMEWORK-DEMANDS D06.

### 3.8 Performance under variable fan-out

The batch plan (`BATCH_PLAN = monthly_reassessment.plan(chunk_by=Application,
chunk_rows=20_000, parallel_grains=[Entity, Event])`) and the arithmetic in
the same file make the NFR concrete: a 40-entity, 600-event application
contributes roughly 1 240 kernel rows against a 1-entity case's ~205 — about
6x, not 40x, because the Candidate grain (200–600 rows through one `prange`
kernel) dominates the per-application cost and does not vary with entity
fan-out at all. This falls directly out of grains-as-frames: in a
per-application loop a 40-entity application costs 40x the nested work
*serially, inside one request*; as frames, those forty rows are amortised
across a batch frame that already has thousands of rows in it.

### 3.9 Overlays inside the nesting

`grade/overlay_stack.py` declares all eleven positions across four grains in
one place — `overlay.position(1, Event, "event_amount_thresholds",
scopes=(...))` through `overlay.position(11, Candidate, "nominal_annual_rate",
scopes=(...))` — with `on_publish` checks that run once, when Credit
Committee publishes a set, over the *whole* stack: no scope overlap within a
position, no cross-position double-count (`pairs=[(3, 6)]`, the group-company
collision PP-11 names explicitly), an expiry on every entry, and every scope
reachable. `UnadjustedSpine` is a `Shadow` of the same nested modules with
`overlays_off(positions=[1..8])`, `share_prefix=True` — re-evaluated, not
re-implemented, so `lineage("people_pd_unadjusted")` answers and the
unadjusted values cannot drift from the adjusted ones. `people/blend.py`
computes the log-odds blend three times over the *same declared weights* —
`people_pd`, `people_pd_unadjusted`, and the difference decomposed per
entity — because "why did this business grade move?" has exactly three
possible answers (the entity's data, the model, an overlay) and only
identical weights make the third one isolable. See FRAMEWORK-DEMANDS D15,
D16, D17, D18.

---

## 4. What a credit committee sees

Spec §9.3 defines an eight-part pack, "generated, not assembled by hand."
Every part maps onto something the tree already emits as a tap or a fold
output, not onto new report-writing code:

| Pack section | Source |
|---|---|
| 1. Decision, offer, binding constraint | `outcome_code`, `SelectOffer.chosen`, `binding_constraint_code` |
| 2. Resolved structure, every path | `DeduplicateEntities.paths`, `effective_ownership_pct`, `criticality_class` |
| 3. Per-entity events, verdict, score | `classify_event`'s taps, `entity_adverse_verdict`'s witness, `families.py`'s per-characteristic contributions |
| 4. The people blend, weights, caps | `blend_weight`, `blend_weight_rule_applied`, `binding_rule_id` on `people_caps` |
| 5. Financial spread, benchmarks, haircuts | `financial/measures.py`'s taps: `financial_confidence_code`, `ebitda_after_haircuts`, `turnover_trend` |
| 6. Grade composition, override | `grade/combine.py`'s taps: component weights, `override_within_authority` |
| 7. Pricing ledger, rate cells | `pricing/select.py`'s candidate record, `rate_cell_id`, `rate_card_version` |
| 8. Conditions and covenants | `validation/consistency.py`'s `ConditionsAndCovenants`, `resolve="union"` |

Nothing on this list is bespoke to the pack. It is the project's own tapped
and declared values, re-projected — which is the same claim doc 04 §6 makes
about the reviewable artefact in general, applied here to eight sections
instead of one rule.

---

## 5. How a decline is traced to one event on one entity

`evidence/attribution.py`'s docstring gives the rendered chain in full, and
it is worth reproducing because every fragment of it is a recorded value and
none of it is a format string written by this project:

```
entity 7 (Ms M. Dlamini, entity_key ...4821, 32.0% effective ownership
across two paths, required surety, criticality CRITICAL because
critical_by_surety and critical_by_ownership)
  <- entity_adverse_verdict DISQUALIFYING, binding rule AE-R-01,
     witness {event 41}
    <- event 41, civil judgment R184 000 dated 2025-03-14, unsatisfied,
       classified DISQUALIFYING by AE-C-03 against a disqualifying
       threshold of R25 000
      <- threshold base R50 000, halved by overlay ADJ-05-014
         (sector 412, criticality CRITICAL, approved by Credit Committee
         2026-02-11, expires 2026-08-31)
  -> people_caps PP-06 row 1, business decline, reason 5402
```

Read bottom-up, this is five joins by grain key, each one already established
in §3.3: `classify_event`'s `material_threshold_base` and
`material_threshold_overlay_id` (generated by `overlay.position(1, ...)`, not
written by hand — D16) resolve to the overlay record; `event_severity_code`
and `classifying_rule_id` name AE-C-03; `EntityAdverseFacts`'s witness of
`disqualifying_count` resolves to `{event 41}`; `EntityAdverseVerdict`'s
`binding_rule_id` names AE-R-01; `criticality_cause_code` names the ranked
cause; `DeduplicateEntities.paths` supplies "across two paths"; and
`PeopleCaps`'s `names_entity="binding_disqualifying_entity"` is the clause
that stops the chain from having to guess which entity a business-level
decline belongs to. Delete any one entity from the application and re-run:
the same five joins run over thirty-nine rows instead of forty, and if event
41 had never existed the chain simply would not exist — there is no
intermediate summary that could go stale, because there is no intermediate
summary.

---

## 6. What changes when a value moves, versus when structure moves

**A value moves** — Business Credit Risk Policy retunes
`entities/adverse/verdict.py`'s `ae_r_02_fires`'s `minor_recent_n` from 3 to
4, or Treasury patches three cells of `business_rate_card.csv` — and nothing
recompiles. Doc 04 §2's governance boundary (params/structure) holds
unchanged at any grain: `minor_recent_n` is `param(3, ge=1, le=20)`, so the
new value is a config edit, bounded by the validator, reviewed as a diff, and
re-scored against a sample before it goes live (acceptance criterion 13). A
rate card patch is a new effective-dated version of `tables/rate_card.sample.csv`,
resolved by `decision_date` at read time — a live application decided
yesterday keeps reading yesterday's cell.

**Structure moves** — change scenario 4 ("three-minor-in-12 becomes
four-in-18, for peripheral entities only") needs `count_window_months` keyed
on `criticality_class`, which changes `is_minor_recent`'s *signature*, not
just its default. That is FRAMEWORK-DEMANDS D19's "sharpest ergonomic
failure": promoting a `param()` to a one-key table is, today, a code change,
a recompile, and an engineer, even though at the call site the two read
almost identically. Change scenario 13 (a third grain of nesting) adds a row
to `attribution_spine`'s `levels=[...]` and a new `Each`/`Gather` pair to the
pipeline expression — a structure change by any definition, but a contained
one, because nothing about `verdict`, `Gather` or the spine itself is
specific to two levels.

**An override is neither** — `grade/combine.py`'s §5.10 paragraph is explicit
that an override (one analyst's judgement about one application, recorded
against it, bounded ±1/±2 by authority) and an overlay (an approved policy
instrument over a declared population, expiring, composed in a stack
position) are "adjacent enough to be confused and must not be." The sketch
keeps them structurally apart: `requested_override_notches` is an ordinary
per-application input, validated and applied by ordinary steps
(`override_within_authority`, `override_attempts_reversal`); an overlay is a
declared stack position with a scope and an expiry. Both can move the same
grade, and the record shows which did what, because they were never the same
mechanism to begin with.

---

## 7. What this sketch does not solve

Seven of the twenty-two demands in FRAMEWORK-DEMANDS.md are marked [UGLY] —
doc 03's existing answer would produce something not fit to review, and the
sketch's own answer is stated rather than hidden: the grain and `Gather`
themselves (D01, D02, because doc 03 has no nesting primitive at all beyond
an accumulator `Loop`); `emit_into` (D10, "the one that doc 03 has no story
for at all"); the `verdict` rule-set kind (D11); the Candidate-as-grain
pricing search (D14, "the second-biggest departure... after the grain
itself"); the param-vs-table arity gap (D19, named in the code as this
sketch's sharpest ergonomic failure); and the per-arm frame cost of two
scorecard families sharing one Entity collection (D22, where no fix is
offered because none avoids reintroducing heterogeneous per-row schemas).

**One outright inconsistency was found and fixed, not just noted.**
`pipelines/business_facility.py`'s `_pricing` block composed
`pricing.select.SelectOffer` — the `Gather` that lifts the chosen candidate's
raw fields — but never composed `pricing.select.SelectedOffer`, the module
that turns those lifted fields into `offered_amount`, `offer_outcome_code`
and `search_truncated`: the exact names `validation/consistency.py`'s
`FinalValidation` and the §7.1 outputs read. Wiring in `decider2` is by name
(doc 03 §2), so every downstream reference to those three names would have
silently become a demand for a leaf input column that no upstream module
produces. The fix is two lines: `SelectedOffer` is now imported alongside
`SelectOffer`, and `_pricing` composes it last.

---

## 8. Index of inventions

Each appears at least twice in the tree; the count is what makes its
ergonomics visible rather than its mere existence.

| Invention | Where | Uses | Demand |
|---|---|---|---|
| `grain(key=, parent=, identity=, capacity=, order=)` | `grains.py` | 4 | D01 |
| `Capacity(n, on_exceed=, flag=)` | `grains.py`, `financial/measures.py` | 5 | D04 |
| `Gather(child, into=parent, **folds)` | 9 files | 12+ | D02, D05 |
| `best_of(..., tie_break=..., lift=[...])` | 8 files | 10+ | D02 |
| `witness(...)` / `union(...)` | `verdict.py`, `caps.py`, `attribution.py` | 6 | D02, D13 |
| `verdict(collect="all", resolve=, rules=[fires(...)])` | 5 files | 5 | D11 |
| `by_class(table, rule_id)` / `class_invariant=` | `entities/disqualification.py` | 12 | D12 |
| `emit_into(child, from_grain=parent, when=, fields=)` | `entities/disqualification.py` | 1 | D10 |
| `attribution_spine(levels=[...], identity=, primary=)` | `evidence/attribution.py` | 1 | D13 |
| `Shadow(pipeline, perturb=, suffix=, share_prefix=/on_demand=)` | 2 files, 4 instances | 4 | D18 |
| `overlay.position(n, grain, target, scopes=)` | 6 files | 11 | D16, D17 |
| `overlay.stack(..., on_publish=[...])` | `grade/overlay_stack.py` | 1 | D15 |
| `Enumerate(grain, from_=cross(...), where=[...])` | `pricing/candidates.py` | 1 | D14 |
| `disclosure.defaults({...})` / `.overrides({...})` | `evidence/disclosure.py` | 1 | D20 |
| `FrameLoop(body, carries=[...], max_iterations=)` | `structure/resolve.py` | 1 | D09 |
| `Vocabulary(..., grains={...})` | `vocabulary.py` | 1 | D07 |

---

## 9. Coordinator note — two constructs that did not parse

Added during review, after the sketch was written. Every `.py` file in all eleven
sketches was run through `ast.parse`. Two files here were the only failures in
286, and both were in newly invented syntax rather than in ordinary code:

| file | what was written | why it cannot parse |
|---|---|---|
| `evidence/attribution.py` | `levels=[(Application, "people_caps", names=Entity), …]` | a tuple cannot contain a keyword argument |
| `grade/overlay_stack.py` | `overlay.stack(name=…, overlay.position(1, …), …, on_publish=[…])` | a positional argument cannot follow a keyword argument |

Both were repaired in the smallest way that preserves the author's intent —
`level(...)` per row in the first, `positions=[...]` in the second — and both
files now parse.

**This is a finding, not a typo.** A sketch exists to propose an authoring
surface, so a proposed surface that Python cannot express is a design result. In
both cases the intent was clearly readable and the shape was a good one; it was
the *syntax* that was unavailable. The second is the more interesting of the
two: the author wanted `name=` first for readability and then a varargs run of
positions, which Python forbids. Making it legal cost the exact ergonomics the
ordering was chosen for.

The general lesson for `decider2`: **an authoring surface has to be checked
against the host language while it is being designed, not after.** D02's
`Gather`, D13's `attribution_spine` and D15's `overlay.stack` are all
declaration-shaped constructs of the kind where this bites — a collection of
heterogeneous rows, each wanting both positional clarity and named modifiers.
Python gives one legal arrangement of those, and it is not always the readable
one. Worth a deliberate pass over every proposed construct in doc 03 before it
is committed to.
